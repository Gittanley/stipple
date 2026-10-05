// SPDX-License-Identifier: GPL-3.0-or-later
// rd_progress.cpp -- the implementation.  See rd_progress.h for the why.
#include "rd_progress.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace rd {
namespace {

double NowMs() {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
                 .count()) /
         1000.0;
}

// Is stderr a console?  GetConsoleMode fails for a file or a pipe, which is
// exactly the distinction wanted, and it needs nothing beyond windows.h --
// the alternative (_isatty) pulls in <io.h>, and this file is compiled as C++
// alongside a CUDA translation unit, so one less platform header is one less
// thing to go wrong on a build that must work on machines nobody here can test.
//
// RD_PROGRESS_TTY=0/1 overrides the answer.  This is not decoration: the
// in-place-overwrite branch is the one users see almost every run, and it is
// the branch that can leave debris on a screen if the erase arithmetic is wrong
// -- and it is unreachable in a build harness, where stderr is always a pipe.
// Without an override the most visible code path in the program is the one
// that can never be tested.  Same idiom as RD_TRACE and RD_YUV444_OUT.
bool StderrIsConsole() {
  const char* force = std::getenv("RD_PROGRESS_TTY");
  if (force != nullptr && force[0] != '\0') {
    return force[0] == '1';
  }
  HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
  if (h == nullptr || h == INVALID_HANDLE_VALUE) return false;
  DWORD mode = 0;
  return GetConsoleMode(h, &mode) != FALSE;
}

// Visible width, used to pad the erase.  dwSize.X is the *buffer* width, which
// on a resized console is often far wider than the window, so srWindow is what
// is actually visible.  Clamped: a 40-column minimum means a narrow console
// still erases; the 400 cap is a paranoia guard against a bogus huge value
// turning the pad into a megabyte of spaces.
int ConsoleWidth() {
  HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if (h == nullptr || h == INVALID_HANDLE_VALUE) return 0;
  if (!GetConsoleScreenBufferInfo(h, &csbi)) return 0;
  const int w = static_cast<int>(csbi.srWindow.Right - csbi.srWindow.Left) + 1;
  if (w < 40) return 40;
  if (w > 400) return 400;
  return w;
}

// Live count of bars currently drawing in place.  Atomic because the
// suppression check in the engine's per-batch path can be reached from a worker
// thread while the bar's owner is elsewhere.
std::atomic<int> g_bars{0};

}  // namespace

bool ProgressHoldsLine() { return g_bars.load(std::memory_order_relaxed) > 0; }

std::string Progress::FormatDuration(double seconds) {
  if (!(seconds > 0.0) || std::isfinite(seconds) == false) return "-";
  if (seconds < 10.0) {
    char b[16];
    std::snprintf(b, sizeof(b), "%.1fs", seconds);
    return b;
  }
  long long s = static_cast<long long>(seconds + 0.5);
  const long long h = s / 3600;
  const long long m = (s % 3600) / 60;
  const long long sec = s % 60;
  char b[32];
  if (h > 0) {
    std::snprintf(b, sizeof(b), "%lldh%02lldm", h, m);
  } else if (m > 0) {
    std::snprintf(b, sizeof(b), "%lldm%02llds", m, sec);
  } else {
    std::snprintf(b, sizeof(b), "%llds", sec);
  }
  return b;
}

Progress::Progress(const char* label, std::int64_t total, bool enabled)
    : label_(label),
      total_(total),
      enabled_(enabled),
      console_(enabled && StderrIsConsole()),
      // Four updates a second is fast enough to look alive and slow enough to
      // be readable if someone is watching a screen share.  A redirected
      // stderr gets one line per 10 s: enough to prove liveness to someone
      // watching a CI log, not enough to bury the actual diagnostics in it.
      interval_(console_ ? 0.25 : 10.0),
      start_ms_(NowMs()),
      last_draw_ms_(-1e18),
      width_(console_ ? ConsoleWidth() : 0),
      last_len_(0),
      started_(false),
      finished_(false),
      done_(0),
      aux_(0),
      aux_label_(nullptr) {
  // Claim the line, if this bar is going to hold one.  Counted rather than
  // flagged because two stages can overlap: VideoBuildPalette's bar is finished
  // before VideoProcess's is constructed, but relying on that ordering across
  // two functions in one file is exactly the kind of assumption that breaks
  // silently the day someone moves a call.
  if (console_) g_bars.fetch_add(1, std::memory_order_relaxed);
}

Progress::~Progress() { Finish(); }

void Progress::Update(std::int64_t done, const char* aux_label,
                      std::int64_t aux) {
  if (!enabled_) return;
  std::lock_guard<std::mutex> lock(mu_);
  done_ = done;
  aux_ = aux;
  aux_label_ = aux_label;
  const double now = NowMs();
  // The first update always draws.  Without that a short run (< interval)
  // prints nothing at all, which is the exact "looks hung" failure this class
  // exists to remove.
  if (started_ && now - last_draw_ms_ < interval_ * 1000.0) return;
  last_draw_ms_ = now;
  DrawLocked();
}

void Progress::DrawLocked() {
  const double now = NowMs();
  const double elapsed = (now - start_ms_) / 1000.0;
  const double done = static_cast<double>(done_);

  std::string line = "[";
  line += label_;
  line += "] ";
  line += std::to_string(done_);
  if (total_ > 0) {
    line += "/";
    line += std::to_string(total_);
    char pct[16];
    std::snprintf(pct, sizeof(pct), " %3.0f%%", 100.0 * done / static_cast<double>(total_));
    line += pct;
  }

  // Rate and ETA.  The ETA is withheld until 8 units and 0.5 s have passed: the
  // first unit includes one-off setup (a CUDA context, the first palette lookup,
  // the encoder's first keyframe), and dividing by that produces a number big
  // enough to be memorable and wrong.
  //
  // The rate is labelled with the window it divides by, and that label is not
  // decoration.  Every figure on this line is a ratio of `done` to `elapsed`,
  // and `elapsed` is measured from THIS bar's own construction -- so the rate
  // describes only the span during which this bar existed.  The end-of-run
  // summary in rd_cli.cpp divides by a total that includes work done before the
  // first bar was ever constructed (the video palette stage, which is a serial
  // prefix: VideoBuildPalette returns before VideoProcess -- and therefore before
  // this bar -- is entered).  Those two rates are the same measurement over two
  // different windows, and on the published 18001-frame run they read 55.6/s and
  // 51.1/s: about 8% apart, both correct, neither saying which was which.
  //
  // Two properties are being asserted here and both are checkable by reading.  (1)
  // `elapsed` is unchanged -- the rate is still done/elapsed, not
  // done/(elapsed+something): no arithmetic was altered, only a word was added,
  // so no existing number moved.  (2) The word names the window rather than
  // claiming the number is better than the summary's: "since bar start" is a
  // statement about this line's own denominator, and the summary's rate is
  // labelled "palette included" over there.  Nothing here reconciles the two by
  // fiat; the reader is told what each one divides by and can subtract.
  //
  // A bar is per-stage, so the qualifier is true for EVERY caller rather than only
  // for the video one: there are two constructions in the tree (rd_video.cpp:1689
  // "palette" and :3497 "video"), and each covers only its own stage.
  //
  // It costs 16 columns, and the truncation a few lines below makes that worth
  // measuring rather than guessing.  For the line
  //   "[video] 12345/18001 69%  55.6/s since bar start  eta 1m30s  5m23s elapsed  read 12400"
  // (85 chars) the qualifier survives at console widths 60 and above, and is cut at 40
  // -- where the ellipsis lands mid-phrase and reads as "55.6/s since ~".  So on a
  // narrow console the label is absent, and the bar then reverts to exactly the
  // ambiguity it was added to remove.  That is a real limit of this fix and it is not
  // papered over: the summary line in rd_cli.cpp is not width-truncated at all, so
  // the interval is always named there.  A shorter qualifier ("stage") would survive
  // 40 columns and was rejected because "stage" names the caller rather than the
  // denominator, and the denominator is the thing in question.
  //
  // The alternative -- seeding start_ms_ with the prior stage's cost -- would need a
  // Progress API change (start_ms_ is private and set once in the constructor, and
  // include/rd_progress.h is not this file's) and would be wrong besides: this bar
  // does not exist during the palette, so charging it for that time would report
  // work it never watched.  The ETA is deliberately left alone for the same reason --
  // an ETA that included a fixed cost already spent would understate the time
  // remaining by exactly that cost, which is a third inconsistent number in the one
  // place where being wrong is immediately visible as the bar running late.
  // See PrintVideoSummary in rd_cli.cpp for the matching label.
  if (done > 0.0) {
    const double rate = done / std::max(1e-9, elapsed);
    // 384, not 48.  `%.1f` of a double is 311 characters at DBL_MAX (measured,
    // not estimated), and the floor on the divisor is 1e-9 while done_ is an
    // int64_t, so `done_ / elapsed` has no small upper bound in principle: the
    // longest value reachable from the int64 numerator is 30 characters, but
    // nothing stops a caller reporting a done_ that has been through a lossy
    // conversion, and 311 + 2 + 17 + NUL = 331 fits 384 with room either way.
    // snprintf truncates rather than overflowing, so 48 was never a memory bug --
    // it was a wrong-output bug: a truncated field reads as "  1.7.../s si" and the
    // console-width trim further along cannot repair it, because by then the
    // damage is inside the number.
    char r[384];
    std::snprintf(r, sizeof(r), "  %.1f/s since bar start", rate);
    line += r;
    if (total_ > 0 && done_ >= 8 && elapsed >= 0.5) {
      line += "  eta ";
      line += FormatDuration(elapsed * (static_cast<double>(total_) - done) / done);
    } else if (total_ > 0) {
      line += "  eta --";
    }
  }
  line += "  ";
  line += FormatDuration(elapsed);
  line += " elapsed";

  if (aux_label_ != nullptr) {
    line += "  ";
    line += aux_label_;
    line += " ";
    line += std::to_string(aux_);
  }

  // A console line that exceeds the window width wraps, and the wrapped remnant
  // is not erased by a following \r, so a long line leaves debris on screen.
  // Truncating with an ellipsis is better than a corrupted display; the log-file
  // case is untouched, because a file has no width.
  if (width_ > 0 && static_cast<int>(line.size()) > width_ - 1) {
    line.resize(static_cast<std::size_t>(width_ - 2));
    line += "~";
  }

  if (console_) {
    // No trailing newline: the next drawing overwrites this one, and Finish()
    // erases it.  A newline here would make every update scroll the console.
    std::fprintf(stderr, "\r%s", line.c_str());
    std::fflush(stderr);
  } else {
    // Newline-terminated, because a file has no notion of "the same line
    // again" -- and this is the form that survives being read by eye, by grep,
    // and by a CI log collector.
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
  }
  last_len_ = static_cast<int>(line.size());
  started_ = true;
}

void Progress::Finish() {
  if (!enabled_) return;
  std::lock_guard<std::mutex> lock(mu_);
  if (finished_) return;
  finished_ = true;
  if (!console_) return;
  // Erase by overwriting with spaces, then return the cursor to column 0.  A
  // bare \r would leave the tail of a longer previous line visible; the space
  // count is exact because it is the length that was actually drawn rather
  // than an estimate from the console width.  Skipped when nothing was ever
  // drawn, since there is then nothing on the screen to erase.
  if (started_) {
    std::fprintf(stderr, "\r%s\r", std::string(last_len_, ' ').c_str());
    std::fflush(stderr);
  }
  // Released here, unconditionally, and only after the erase.  Two things
  // depend on that ordering: releasing first would let the next batch's timing
  // line print onto the line still holding the bar's text, and the erase would
  // then wipe part of *that* instead.  Making it unconditional is the other
  // half -- a bar that is finished without ever drawing (the palette sampler
  // bails out on zero samples, for instance) must still give the line back, or
  // the counter leaks and every later per-batch diagnostic is suppressed for
  // the rest of the process.
  g_bars.fetch_sub(1, std::memory_order_relaxed);
}

}  // namespace rd
