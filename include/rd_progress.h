// SPDX-License-Identifier: GPL-3.0-or-later
// rd_progress.h -- one-line progress with an ETA.
#ifndef RD_PROGRESS_H_
#define RD_PROGRESS_H_

#include <cstdint>
#include <mutex>
#include <string>

namespace rd {

// A self-erasing single-line progress indicator, rate-limited by wall time.
//
// Why this exists as a class rather than as more fprintf calls: the program had
// two hand-rolled progress lines (the palette sampler) and none at all on the
// part that actually takes minutes (the video pipeline), because the second one
// was an ad-hoc `if (sampled % N == 0)` counter with three problems worth not
// repeating.  It ignored --quiet; it wrote \r unconditionally, so redirecting
// stderr to a file produced one unreadable line per update; and its interval was
// a function of the total, so a 18001-frame clip and a 300-frame one reported on
// different schedules and neither matched a human's sense of "about a second".
//
// Behaviour, chosen per destination:
//
//   console      Overwrite in place, about 4 times a second.  This is the case
//                the indicator exists for: a 5-minute render that prints nothing
//                is indistinguishable from a hang.
//   redirected   One newline-terminated line every 10 s.  A log file with 150
//                carriage returns in it is not a log file.  A build that emits
//                nothing at all is still a hang, so it says something -- just
//                not often, and without the \r.
//   disabled     Nothing.  This is --quiet.
//
// The ETA is computed from elapsed/done, not from a recent-window rate: a window
// rate swings wildly on a bounded queue whose stages alternate, and the user
// watches it change by a minute.  It is withheld until there is enough signal to
// mean anything, and it is suppressed entirely when the total is unknown (a
// stream with no frame count in the container), because an ETA for "the rest of
// the file" is a guess with no denominator.
//
// Thread safety: the count that drives an ETA is typically owned by a pipeline
// thread while the line is drawn by the main thread, so Update() is safe to call
// from any thread and from several at once.
class Progress {
 public:
  // `total` <= 0 means "length not known"; progress still updates, but no
  // percentage and no ETA are shown, because neither has a denominator.
  Progress(const char* label, std::int64_t total, bool enabled);
  ~Progress();

  Progress(const Progress&) = delete;
  Progress& operator=(const Progress&) = delete;

  // Report `done` units of work.  Returns without drawing if the rate limit has
  // not elapsed.  `aux`/aux_label add a second, secondary counter -- for the
  // video pipeline that is frames durably written, which trails the frames read
  // by the depth of the in-flight queue.  Pass aux_label = nullptr to omit it.
  void Update(std::int64_t done, const char* aux_label = nullptr,
              std::int64_t aux = 0);

  // Erase the line (console) or leave the last line in place (redirected), and
  // stop drawing.  Safe to call more than once; the destructor does it.
  void Finish();

  // "4m12s", "1h03m", "38s", "0.4s".  Exposed because the duration in a log
  // line should be as readable as the one on the console.
  static std::string FormatDuration(double seconds);

 private:
  void DrawLocked();

  std::mutex mu_;
  const char* label_;
  const std::int64_t total_;
  const bool enabled_;
  const bool console_;   // stderr is a console: overwrite in place
  const double interval_;  // seconds between draws

  double start_ms_;
  double last_draw_ms_;
  int width_;            // console width, for clearing; 0 when not a console
  int last_len_;         // length of the last line drawn, to erase exactly
  bool started_;         // has anything been drawn
  bool finished_;
  std::int64_t done_;
  std::int64_t aux_;
  const char* aux_label_;
};

// True while some Progress object is drawing an in-place line on a console.
//
// This exists for output that would collide with the bar.  An in-place progress
// line has no trailing newline -- adding one would scroll the console on every
// update -- so any other message printed to the same stream lands *on the same
// physical line*, and the next redraw erases half of it.  A per-batch timing
// line is the worst offender: one every 16 frames, glued to the bar, then
// half-deleted by it.
//
// The predicate is deliberately about "is a bar drawing right now", not about
// "is stderr a console".  Console-ness is the wrong test: a single image with
// --engine blocks also writes to a console, prints only a handful of batch
// lines, and genuinely benefits from them.
bool ProgressHoldsLine();

}  // namespace rd

#endif  // RD_PROGRESS_H_
