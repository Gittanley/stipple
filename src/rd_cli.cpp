// SPDX-License-Identifier: GPL-3.0-or-later
// rd_cli.cpp -- rdither, the Phase 1 command line front end.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <windows.h>
#include <vector>

#include "rd_checkpoint.h"
#include "rd_im.h"
#include "rd_opencl.h"
#include "rd_plugin.h"
#include "rd_riemersma.h"
#include "rd_source.h"
#include "rd_video.h"

namespace {

// Largest --max-ram-mb / --max-vram-mb value whose * 1024 * 1024 does not wrap a
// 64-bit std::size_t to zero.  2^44 - 1, about 16 exabytes.  One constant for both
// flags so the two parse-site checks cannot drift apart from each other.
constexpr std::size_t kMaxBudgetMb = (std::size_t{1} << 44) - 1;

struct Options {
  // Every flag name the user actually typed, in order.  Recorded because "was this
  // passed?" cannot be recovered from a field's value: --palette-budget-ms defaults
  // to 60000 and --palette-tile to 128, both non-zero, so a check that asks
  // `if (opt.video_opt.palette_budget_ms > 0)` can only ever fire when the user
  // happened to set a value equal to the default.  That is the defect the existing
  // `bypassed` diagnostic has: it names 5 knobs and can detect 2.
  std::vector<std::string> seen_flags;
  std::string input;
  std::string output;
  std::string format;
  int colors = 16;
  rd::Engine engine = rd::Engine::kCpu;
  double diffusion = 1.0;
  std::size_t max_ram_mb = 512;
  std::size_t max_vram_mb = 0;
  int frames = 1;
  rd::ApproxOptions approx;
  rd::BlockOptions blocks;
  bool video = false;
  int dump_palette_limit = 0;
  rd::VideoOptions video_opt;
  bool verify = false;
  bool no_cache = false;
  bool self_test = false;
  bool dump_curve = false;
  bool dump_palette = false;
  bool quiet = false;
  // Which dither to run.  "riemersma" is the built-in and the default; anything else
  // must be a registered plugin, or the CLI fails loudly rather than quietly falling
  // back -- a typo'd --dither that silently ran Riemersma would look like the plugin
  // was broken.
  std::string dither = "riemersma";
  bool list_dithers = false;
  // --palette-import: adopt a palette from a file and do not derive one.  Kept
  // at the top level rather than on video_opt, because it applies to images too:
  // --palette-from and --palette-export were both video-only for a long time, and
  // an image is exactly the case where you want to fix a palette and re-use it.
  std::string palette_import;
};

void PrintUsage() {
  // The lone argument below is for the single bare `%d` in this format string, which
  // states the real --colors ceiling.  It was written without one, so `%d` read past
  // the end of the argument list -- undefined behaviour -- and `--help` printed
  // "the parser requires 2..0" in the very sentence explaining that --colors 0 does
  // not exist.  MSVC caught it as C4473; it went unread because the build's other
  // warnings are narrowing conversions, and C4473 is the only one that means the
  // program is already wrong.  The bound comes from rd::kMaxColormapSize, the same
  // value the refusal at the --colors parse prints, so the two cannot disagree.
  std::printf(
      "rdither -- ImageMagick 7.1.2-31 Q16-HDRI Riemersma dithering\n"
      "\n"
      "Usage:\n"
      "  rdither [options] <input> <output>\n"
      "\n"
      "Options:\n"
      "  --colors N        palette size handed to ImageMagick (default 16)\n"
      "  --engine E        cpu | cuda | blocks | approx | opencl  (default cpu)\n"
      "                     blocks  block-parallel walk on CUDA.\n"
      "                     opencl  the same partition and the same arithmetic as\n"
      "                             blocks, reached through OpenCL, so AMD and\n"
      "                             Intel GPUs get a GPU engine.  Output is\n"
      "                             identical to blocks on the same frame.\n"
      "                             With --video it takes the same input modes as\n"
      "                             blocks (yuv444 is the default) and is within\n"
      "                             about 1.2-1.3x of it; rgba64le is still there\n"
      "                             and still bit-identical.  4:2:0 and the\n"
      "                             yuv444-prepass are not ported.\n"
      "  --blocks N        curve positions per walk block, --engine blocks (default\n"
      "                     512).  A block restarts its error queue, so this trades\n"
      "                     speed against fidelity to ImageMagick.  32 is 11%% faster\n"
      "                     on the dither but deviates 3.4x more from IM's output\n"
      "                     (6.1%% of pixels vs 1.7%% on a 200x150 gradient).  Use the\n"
      "                     default unless you want the speed.\n"
      "  --diffusion X     IM's dither:diffusion-amount artifact (default 1.0)\n"
      "  --dither NAME     dither algorithm (default riemersma).  Registered by\n"
      "                    plugins; see include/rd_plugin.h and examples/.\n"
      "  --list-dithers    list the registered dither algorithms and exit\n"
      "  --max-ram-mb N    RAM budget before spilling pixels to disk (512)\n"
      "  --max-vram-mb N   VRAM budget for the CUDA engine (0 = all free)\n"
      "  --frames N        concurrent walks on the GPU (default 1)\n"
      "  --format FMT      force the output coder, e.g. PNG, TIFF, MIFF\n"
      "  --verify          diff against ImageMagick's own Riemersma output\n"
      "  --no-cache        drop IM's palette memo table (diagnostic; not exact)\n"
      "  --approx-iters N  sweeps for --engine approx (default 20)\n"
      "  --approx-taps N   inverse-filter length for --engine approx (default 96)\n"
      "  --approx-fp64     run the approximate convolution in double\n"
      "  --approx-linear   use a plain nearest-colour scan instead of the octree\n"
      "  --dump-curve      print the curve level, visit count and first visits\n"
      "  --dump-palette    print the harvested palette\n"
      "\n"
      "Video (--video; decode, dither and encode run concurrently):\n"
      "  --video                ffmpeg pipeline instead of a single image\n"
      "  --palette-frames N     frames sampled for the palette.  Default 0, meaning\n"
      "                         'as many as --palette-budget-ms allows' -- a few\n"
      "                         hundred on a 3-hour clip, because 30 evenly\n"
      "                         spaced points can all land in the dull stretches\n"
      "                         and miss the colour entirely.  Give N for exactly\n"
      "                         N samples; 30 reproduces the reference pipeline.\n"
      "  --palette-budget-ms MS  time budget for that sampling (default 60000).\n"
      "                         Measured 0.12 s per sample, 6 at a time, so a\n"
      "                         slow disk takes fewer samples rather than more\n"
      "                         time.  Look at the printed swatches to judge the\n"
      "                         palette, not the sample count and not the mean\n"
      "                         saturation: both report confidently on a montage\n"
      "                         that was never read correctly.\n"
      "  --palette-tile N       lattice-sample tile edge per frame (default 128)\n"
      "  --palette-mode M       montage (default) or pool (measured worse)\n"
      "  --palette-only         build and report the palette, then stop\n"
      "  --palette-export FILE  write the palette used, as PNG (8-bit strip) or\n"
      "                         .txt (Q16, lossless).  Works for images too.\n"
      "  --palette-import FILE  use FILE's palette instead of deriving one, for\n"
      "                         images and video alike.  PNG (8-bit) or .txt (Q16,\n"
      "                         lossless).  --colors must match the file's count;\n"
      "                         a mismatch is refused rather than silently\n"
      "                         resolved.  (This used to also advertise --colors 0\n"
      "                         to adopt the file's count as-is.  No such value\n"
      "                         exists: the parser requires 2..%d and has always\n"
      "                         done so, so the documented route to a mismatched\n"
      "                         count never worked.)\n"
      "                         Sampling options (--palette-frames, --im-palette,\n"
      "                         --palette-tile, ...) are reported as ignored.\n"
      "  --im-palette           generate the palette exactly as ImageMagick does\n"
      "                         (8-bit, no alpha, +append, -unique-colors), then\n"
      "                         GPU-dither.  The one-command equivalent of running\n"
      "                         `magick f1..f30 +append -colors N -unique-colors`\n"
      "                         and then --palette-from.\n"
      "\n"      "  --batch-frames N       frames per GPU launch (default 16, saturates ~16)\n"
      "  --queue-depth N        batch slots in flight (default 3)\n"
      "  --input-mode M         yuv420 (1.5 B/px, the decoder's own format: no\n"
      "                         swscale at all, chroma reconstructed 2x bilinear on\n"
      "                         the device), yuv444 (3 B/px, swscale's YCbCr->RGB\n"
      "                         matrix on the device), yuv444-prepass (same\n"
      "                         result via a coalesced pass, +2%% where it fits),\n"
      "                         or rgba64 (8 B/px, the reference path; identical\n"
      "                         output for a 4:4:4 source, 34 dB apart for 4:2:0)\n"
      "  --no-hwaccel           software h264 decode instead of the GPU's NVDEC\n"
      "  --decode-threads N     cap ffmpeg's decoder threads (0 = automatic)\n"
      "  --encode-threads N     cap ffmpeg's encoder threads (0 = automatic)\n"
      "  --reader-threads N     threads for the reader's widening (0 = auto)\n"
      "  --cpu-threads N        -1 auto, 0 = GPU only, >0 exact host worker count\n"
      "  --host-grace-ms N      host worker wait for the GPU first (0 = 60)\n"
      "  --no-gpu               host workers only\n"
      "  --gpu-workers N        independent CUDA states, so two dithers are in\n"
      "                         flight at once (default 1; 2 measured no faster,\n"
      "                         since the dither stage is saturated not stalled)\n"
      "  --gpu-float-out        GPU returns float4, host converts (slower; A/B\n"
      "                         check for the default uint16 device output)\n"
      "  --mem-fraction F       share of physical RAM the queue may use (1/3)\n"
      "  --video-codec C        default libx264\n"
      "  --video-pix-fmt F      default yuv444p -- yuv420p destroys the palette\n"
      "  --video-lossless       ffv1 yuv444p: the palette survives exactly, at\n"
      "                         roughly 28x the bitrate of yuv420p H.264\n"
      "  --crf N / --preset P   encoder quality\n"
      "  --video-preserve-vfr   keep a variable-rate source's per-frame timing rather\n"
      "                         than flattening to its average.  Duration and A/V\n"
      "                         sync are already correct without it; this restores\n"
      "                         motion cadence too.  NOT verified end to end.\n"
      "\n"
      "Other:\n"
      "  --quiet           suppress progress output\n"
      "  -h, --help        this text\n"
      "\n"
      "Examples:\n"
      "  rdither --colors 16 --verify in.png out.png\n"
      "  rdither --engine blocks --colors 16 in.png out.png\n"
      "  rdither --engine cuda --max-vram-mb 2048 in.png out.png\n"
      "  rdither --engine approx --approx-iters 32 --verify in.png out.png\n"
      "  rdither --max-ram-mb 64 --colors 16 huge.tif out.png\n"
      "  rdither --video --colors 16 --cpu-threads 0 in.mp4 out.mp4\n"
      "  rdither --video --colors 16 --video-lossless in.mp4 out.mkv\n"
      "\n"
      "Diagnostics.  All are environment variables, all are off unless set, and none\n"
      "changes what the program computes -- they print, or they select a path that is\n"
      "already bit-identical, or both:\n"
      "  RD_KERNEL_TIMING=1   per-kernel gather/walk/scatter milliseconds\n"
      "  RD_CC_STATS=1        octree search cost: nodes and descent steps per pixel.\n"
      "                         REQUIRES RD_KERNEL_TIMING=1 as well -- the print sits\n"
      "                         inside the timing block, so setting this alone produces\n"
      "                         no output and looks like a broken build.  The build must\n"
      "                         also be configured with -DRD_CC_STATS=ON, because the\n"
      "                         counters are compile-time gated: counting per node visit\n"
      "                         cost a measured 2.5x on the dither stage, so a runtime\n"
      "                         flag is not an option.\n"
      "  RD_FLAT_SEARCH=0     force the recursive palette search, for A/B-ing it\n"
      "                         against the flat one.  Same picture either way; below\n"
      "                         16 colours the flat search wins.\n"
      "  RD_OCL_IO=1          per-transfer byte counts for the OpenCL engine\n"
      "  RD_OCL_DUMP=1        dump device buffers after a batch (large)\n"
      "  RD_TRACE=1           per-visit trace (very large)\n",
      rd::kMaxColormapSize);
}

bool ParseSize(const char* text, std::size_t* out) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *out = static_cast<std::size_t>(value);
  return true;
}

bool NeedsValue(int argc, int i, const char* flag) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "error: %s requires a value\n", flag);
    return false;
  }
  return true;
}

// Parses an integer option value, rejecting anything that is not entirely numeric.
//
// atoi() is the wrong tool here, and not because of overflow.  It returns 0 for
// anything it cannot read, and several of these flags treat 0 as a MEANINGFUL
// setting rather than as "unset":
//
//   --cpu-threads 0   means "GPU only"
//   --queue-depth 0   a depth of one
//
// It also stops at the first non-digit, so a typo is accepted as the prefix.
// Measured, before this existed:
//
//   --colors 3x        -> 3,   exit 0, silent
//   --cpu-threads auto -> 0,   exit 0, silent, i.e. "GPU only" as a side effect
//   --batch-frames 16x -> 16,  exit 0, silent
//
// So a mistyped number quietly becomes a different program.  `end` must land on the
// NUL terminator, which rejects both "auto" and "3x" in one check, and the reported
// text is the value the user actually typed rather than a guess.
bool ParseIntArg(const char* raw, const char* flag, long* out) {
  char* end = nullptr;
  const long v = std::strtol(raw, &end, 10);
  if (end == raw || end == nullptr || *end != '\0') {
    std::fprintf(stderr, "error: %s wants an integer, got '%s'\n", flag, raw);
    return false;
  }
  *out = v;
  return true;
}

// The same rule for the two floating-point options.  atof("abc") is 0.0, and 0.0 is
// not a neutral default for either of these: --mem-fraction 0 means "use no RAM" and
// --diffusion 0 means "no diffusion at all", so a typo would produce a program that
// runs and quietly discards the effect the flag was for.
bool ParseDoubleArg(const char* raw, const char* flag, double* out) {
  char* end = nullptr;
  const double v = std::strtod(raw, &end);
  if (end == raw || end == nullptr || *end != '\0') {
    std::fprintf(stderr, "error: %s wants a number, got '%s'\n", flag, raw);
    return false;
  }
  *out = v;
  return true;
}

double ElapsedMs(std::chrono::steady_clock::time_point start) {
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

// ---------------------------------------------------------------------------
// Video mode (Phase 2)
// ---------------------------------------------------------------------------
namespace {

int ReportInterrupted(const std::string& out_path, bool interrupted = true) {
  (void)interrupted;
  std::fprintf(stderr,
               "interrupted: the encoder was closed cleanly, so '%s' holds the\n"
               "             frames completed so far.  Re-run the same command\n"
               "             with --resume to continue where this stopped.\n",
               out_path.c_str());
  return 130;  // conventional 128 + SIGINT
}

// Two rates, over two intervals, both named.
//
// `result.total_ms` is the wall clock INCLUDING the palette -- the palette is a serial
// prefix, VideoBuildPalette has returned before VideoProcess is entered, and a
// throughput figure that discounts a stage the user sat and watched is not throughput.
// The progress bar's rate is a different number: its clock starts in the Progress
// constructor inside VideoProcess, so it EXCLUDES the palette.  On the published
// 18001-frame run (palette 28677 ms, wall 352507 ms) that is 51.1/s here and 55.6/s on
// the bar -- two figures ~8% apart, both unlabelled, and the bar's ETA built from the
// second of them.
//
// So this line names the interval it divides by, and the bar is labelled to match
// (rd_progress.cpp, `rate_window_`).  Neither number changed: what changed is that each
// says which window it is a rate over, so the two can be compared instead of guessed
// at.  A reader who wants the bar's figure has it on the bar, marked; a reader who
// wants end-to-end has it here.
void PrintVideoSummary(const rd::VideoResult& result) {
  const double total_s = result.total_ms / 1000.0;
  const double pipeline_ms =
      result.total_ms > result.palette_ms ? result.total_ms - result.palette_ms : 0.0;
  // `result.frames` is what VideoProcess calls frames DITHERED (rd_video.cpp:3791
  // assigns frames_dithered), and the progress bar counts frames WRITTEN
  // (segment_written, fed to Update at :3738).  On an unsegmented run the writer
  // drains every dithered batch before it exits -- its only non-segmented stop is
  // `eof && frames_dithered >= frames_read` (:3746, :3755) -- so the two agree and
  // the word is not needed.  On a segmented or resumed run they need not, which is
  // why the segmented path in RunVideo now accumulates `want` rather than
  // seg_result.frames.  Said out loud rather than left to be inferred, because the
  // alternative is a summary whose numerator and the bar's numerator are the same
  // word meaning two different things.
  std::printf("frames     : %lld in %.2f s (%.1f fps, palette included)\n",
              static_cast<long long>(result.frames), total_s,
              result.total_ms > 0.0 ? result.frames * 1000.0 / result.total_ms
                                    : 0.0);
  // The bar's number, computed here so the two are visibly the same measurement over
  // two windows rather than two unrelated figures.  Printed only when the palette was a
  // real cost, because otherwise it is the same number twice.
  //
  // "the window the progress bar covers" is exact to within the two statements between
  // VideoProcess's `t_start` (rd_video.cpp:3465) and the Progress constructor
  // (:3497) -- microseconds -- and deliberately not claimed to be identical.  The bar's
  // own denominator is its private start_ms_, which nothing outside that class can
  // read, so this line recomputes the same quantity from the fields that are visible
  // rather than pretending to a precision it does not have.
  if (result.palette_ms > 0.0 && pipeline_ms > 0.0) {
    std::printf("             %.1f fps over the %.2f s window the progress bar covers "
                "(excludes the %.2f s palette)\n",
                result.frames * 1000.0 / pipeline_ms, pipeline_ms / 1000.0,
                result.palette_ms / 1000.0);
  }
  // The middle three are SUMMED WORKER TIME, not a partition of the wall clock, and
  // saying so in the label is the point: `dither` in particular is accumulated once per
  // dither worker (rd_video.cpp:3446), so with N workers it counts N threads' elapsed
  // time and can exceed the wall clock.  Presenting them in a row separated by pipes
  // under a single word invited exactly the naive reading that the numbers do not
  // support.  convert_ms and pipe_ms ARE a partition of encode_ms, which the second
  // line now says.
  std::printf("busy time  : wall %.0f ms (palette %.0f of it)\n"
              "             summed worker time: decode %.0f | dither %.0f | "
              "encode %.0f\n",
              result.total_ms, result.palette_ms, result.decode_ms,
              result.dither_ms, result.encode_ms);
  // The writer's cost split, because the two halves scale differently and only
  // one of them is a candidate for moving to the GPU.  These two DO add up to the
  // encode figure above; the three beside it do not add up to the wall.
  std::printf("             writer %.0f ms of that encode figure: %.0f ms "
              "float->uint16 on the host, %.0f ms pushing the pipe\n",
              result.convert_ms + result.pipe_ms, result.convert_ms,
              result.pipe_ms);
}

// Concatenates the recorded segments into the requested output.  Every segment
// came from the same encoder settings, so the concat demuxer can stream copy
// without re-encoding -- no quality loss and no extra time.
bool JoinSegments(const std::string& out_path, const rd::Checkpoint& cp,
                  std::string* error) {
  if (cp.segments.empty()) {
    *error = "no segments to join";
    return false;
  }
  if (cp.segments.size() == 1) {
    // Nothing to join; move the single part into place.
    std::error_code ec;
    (void)std::filesystem::rename(
        std::filesystem::u8path(cp.segments[0].path),
        std::filesystem::u8path(out_path), ec);
    if (ec) {
      *error = "cannot move the single segment into place: " + ec.message();
      return false;
    }
    return true;
  }
  std::string ffmpeg;
  if (!rd::VideoFindTools(&ffmpeg, nullptr, error)) return false;

  // The list file must be written durably too, but it is rebuilt from the
  // checkpoint every time, so a plain write is enough.
  const std::string list = out_path + ".concat.txt";
  {
    std::ofstream out(list.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) {
      *error = "cannot write " + list;
      return false;
    }
    for (const rd::SegmentRecord& s : cp.segments) {
      // Absolute paths, because the concat demuxer resolves entries relative to
      // the *list file's* directory.  A relative entry would become
      // "outdir/outdir/segment.mkv" and fail to open.
      char full[MAX_PATH * 4];
      const DWORD n = GetFullPathNameA(s.path.c_str(), sizeof(full), full, nullptr);
      const std::string p = (n > 0 && n < sizeof(full)) ? std::string(full, n)
                                                        : s.path;
      // concat demuxer quoting: single quotes, with embedded quotes doubled.
      std::string esc;
      for (char ch : p) {
        if (ch == '\'') esc += "''";
        else esc += ch;
      }
      out << "file '" << esc << "'\n";
    }
  }
  const std::string args = "-v error -y -f concat -safe 0 -i " +
                           rd::VideoQuotePath(list) + " -c copy " +
                           rd::VideoQuotePath(out_path);
  int code = -1;
  const bool joined = rd::VideoRunTool(ffmpeg, args, 900000, &code, error);
  (void)DeleteFileA(list.c_str());
  if (!joined) return false;
  if (rd::FileSize(out_path) == 0) {
    *error = "the joined output is empty";
    return false;
  }
  return true;
}

int RunVideo(const std::string& in_path, const std::string& out_path,
             rd::VideoOptions& opt) {
  std::string error;
  rd::VideoInfo info;
  if (!rd::VideoProbe(in_path, &info, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 1;
  }
  std::printf("source     : %s\n", in_path.c_str());
  std::printf("video      : %dx%d, %s, %s, %lld frames @ %.3f fps\n", info.width,
              info.height, info.codec.c_str(), info.pix_fmt.c_str(),
              static_cast<long long>(info.frames), info.fps);

  // ROTATION.  ffmpeg's rawvideo output auto-rotates, so a stream carrying a 90 or
  // 270 degree display matrix arrives at the pipeline already turned -- at the
  // DISPLAYED dimensions, which are the coded ones swapped.  The pixel count is
  // identical, so nothing downstream notices the mismatch: the frame is simply read
  // back at the wrong row width and the picture comes out sheared.  Measured on a
  // phone capture tagged -90, the output was the 1080x1920 portrait image written
  // into a 1920x1080 buffer.
  //
  // So the geometry is corrected here, once, and everything downstream -- buffer
  // sizes, the raw pipe's -s, the encoder, the dither engines -- uses the displayed
  // frame.  The alternative, `-noautorotate` on the decoder, would keep the pixels as
  // stored but then needs the display matrix copied to the output, and ffmpeg 9
  // exposes -display_rotation as an INPUT option only.  Rotating the pixels in is
  // therefore the only route that works in every container.
  if (info.rotation != 0) {
    const int a = ((info.rotation % 360) + 360) % 360;
    if (a == 90 || a == 270) {
      std::swap(info.width, info.height);
    }
    std::fprintf(stderr,
                 "[video] source carries a %d degree display matrix; the decoder"
                 " rotates, so the frames are processed and written as %dx%d"
                 " upright with no rotation tag on the output.\n",
                 info.rotation, info.width, info.height);
  }

  // Palette is 2 MiB (MaxColormapSize == 65536) and the default thread stack is
  // 1 MiB, so both of these go on the heap.
  std::unique_ptr<rd::Palette> palette(new rd::Palette());
  std::unique_ptr<rd::ColorTree> tree(new rd::ColorTree());

  // The palette is built on its own thread so the reader can start immediately.
  // Everything that needs the palette -- the report below, and the dither workers --
  // goes through the gate, and the join happens before any of them reads it.
  //
  // RD_PALETTE_OVERLAP=0 restores the original serial ordering.  It is not a debug
  // convenience: it is the BASELINE ARM of tools\probe-palette-overlap.ps1, and the
  // probe asserts the two arms differ in timing before comparing their pixels --
  // otherwise a flag that silently did nothing would make the A/B compare one binary
  // with itself and report a perfect pass.
  // RD_PALETTE_OVERLAP=1 turns the overlap ON.  It is OFF by default because it was
  // MEASURED SLOWER: 600 frames of 1080p, interleaved, 39.0 fps with the overlap against
  // 50.8 fps without -- 24% slower, and the palette stage itself gets slower (3679 -> 4047
  // ms) because the ImageMagick quantise competes with a reader pipeline that already
  // saturates every core.  See the spec's section 6 and docs\KNOWN-ISSUES.md.
  //
  // The reader is the pipeline's floor.  Running a CPU-heavy palette build beside it does
  // not fill idle time, because there is no idle time: the two overlap by contending.
  //
  // The code stays because the arithmetic can hold on a machine that is not already
  // saturated -- a GPU with idle cores, or a reader bound on I/O rather than CPU.  But it is
  // opt-IN, because shipping a measured 24% regression as the default is not defensible.
  const char* overlap_env = std::getenv("RD_PALETTE_OVERLAP");
  const bool overlap = (overlap_env != nullptr && overlap_env[0] == '1');
  // Whether the palette actually went onto its own thread.  Reported, not gated on timing;
  // tools\probe-palette-overlap.ps1 reads this line to prove RD_PALETTE_OVERLAP reached the
  // code.  See the comment at the spawn site.
  bool overlapped = false;
  rd::PaletteGate gate;
  std::thread palette_thread;
  rd::VideoResult result;
  if (!opt.palette_from.empty()) {
    // A .txt here used to fail with "improper image header", because this branch
    // went through ImLoad -- i.e. ImageMagick -- and a rdither Q16 palette text file
    // is not an image.  The message named a corrupt file, and the file was fine.
    //
    // The fix is not a second reader, it is THE reader: ImReadPalette is what the
    // image path uses, and routing video through it means one implementation for
    // both paths, so the two formats and the two validation behaviours cannot drift
    // apart again.  It also brings three things video previously lacked:
    //
    //   * the .txt Q16 form, which --help documents for --palette-import and which
    //     the image path has always accepted;
    //   * the colour-count check, so a 14-colour file asked for as 16 is REFUSED
    //     with both numbers named, instead of quietly dithering to 14;
    //   * the RgbaF precision reconciliation, which is import-specific -- a palette
    //     read from a file holds doubles while the tree is built from floats, and
    //     without it every later "colormap entry N differs" check fails on the last
    //     digit.  A palette derived from the video's own pixels never hits this,
    //     which is why only the import path needs it and why only the import path
    //     had it.
    const bool is_text =
        (opt.palette_from.size() >= 4 &&
         opt.palette_from.compare(opt.palette_from.size() - 4, 4, ".txt") == 0);
    if (is_text) {
      if (!rd::ImReadPalette(opt.palette_from, opt.colors, palette.get(),
                             tree.get(), &error)) {
        std::fprintf(stderr, "error: --palette-from %s\n", error.c_str());
        return 1;
      }
    } else {
      // An image colormap: ImageMagick built the palette, we only dither.  Its
      // colormap is taken verbatim, because re-quantizing it is precisely what would
      // change it.
      rd::LoadedImage loaded;
      if (!rd::ImLoad(opt.palette_from, &loaded, &error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
      }
      if (!rd::ImAdoptPaletteFromColormap(loaded, opt.colors, palette.get(),
                                          tree.get(), &error)) {
        std::fprintf(stderr, "error: --palette-from %s: %s\n",
                     opt.palette_from.c_str(), error.c_str());
        rd::ImFree(&loaded);
        return 1;
      }
      rd::ImFree(&loaded);
    }
    result.palette_colors = palette->count;
    result.palette_sampled = 0;
    result.palette_mosaic_w = 0;
    result.palette_mosaic_h = 0;
    result.palette_neutral =
        rd::CountNeutralPaletteEntries(*palette, 0.2);
    // PUBLISH, or this branch deadlocks.
    //
    // Every other arm of this if/else chain ends in `gate.Publish` (rd_cli.cpp:589 for
    // the overlap thread, :596 for the ordinary serial build), because VideoProcess calls
    // `gate.Wait()` (rd_video.cpp:3819) before it dithers anything and Publish/Fail are
    // the only things that release that wait. This branch -- the `--palette-from` /
    // `--palette-import` arm -- adopted a perfectly good palette, printed it, and then fell
    // out of the chain without publishing, so the gate was never released and the run
    // blocked forever:
    //
    //     rdither --video --engine cpu --colors 14 --palette-import p.txt in.mkv out.mkv
    //     palette : 14 colours adopted from p.txt, mean saturation 83.0%, 2 near-neutral
    //     [video] 320x180  batch=16  queue=3 ...
    //     <nothing, ever>
    //
    // Reproduced: still running after 90 s on a 30-frame 320x180 clip, 1725 s of CPU
    // burned, no output file. Two were left running and had to be killed by hand.
    //
    // The specific cruelty is that it PRINTS the adopted palette first. The user is told
    // the flag worked, the numbers look right, and then the program wedges -- so the
    // evidence points at the flag rather than at the flag's plumbing.
    //
    // The image path never hit this because it never constructs a gate: RunImage dithers
    // directly and the gate exists only for VideoProcess.
    gate.Publish(std::move(*palette), std::move(*tree));
    std::printf("palette    : %d colours adopted from %s, mean saturation %.1f%%, "
                "%d near-neutral\n",
                palette->count, opt.palette_from.c_str(),
                100.0 * rd::MeanPaletteSaturation(*palette),
                result.palette_neutral);
    for (int i = 0; i < palette->count && i < opt.dump_palette_limit; ++i) {
      const rd::PaletteEntry& e = palette->entries[i];
      std::printf("  %2d: %9.1f %9.1f %9.1f\n", i, e.r, e.g, e.b);
    }
  } else if (overlap && !opt.palette_only) {
    // Printed so that "the flag reached the code" is a DETERMINISTIC fact rather than an
    // inference from wall time.  That inference is unusable on a busy machine: run-to-run
    // drift here is 10-25% and this feature's effect is 22-34%, so the two overlap and a
    // single A/B pair cannot separate them.  A line of output can.
    overlapped = true;
    // Overlapping: the palette builds here while the caller goes on to start the
    // reader.  `result` is written by this thread and read by the main thread only
    // after the join, so its palette_* fields need no synchronisation of their own.
    palette_thread = std::thread([&] {
      std::string pal_err;
      if (!rd::VideoBuildPalette(in_path, opt, info, palette.get(), tree.get(),
                                  &result, &pal_err)) {
        gate.Fail(pal_err);
        return;
      }
      gate.Publish(std::move(*palette), std::move(*tree));
    });
  } else if (!rd::VideoBuildPalette(in_path, opt, info, palette.get(), tree.get(),
                                    &result, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 1;
  } else {
    gate.Publish(std::move(*palette), std::move(*tree));
  }
  // The report reads the gate rather than the moved-from unique_ptr: Publish took
  // ownership with std::move, so *palette is empty from here on.
  auto report_palette = [&] {
    // The overlap marker.  Present iff the palette was built on its own thread, which is
    // what makes the flag's reachability checkable without timing anything.
    if (overlapped) {
      std::printf("palette    : built concurrently with the reader (RD_PALETTE_OVERLAP)\n");
    }
      const std::size_t tile = static_cast<std::size_t>(result.palette_tile);
      // Mean saturation is KEPT, but it is no longer the thing to look at, and --help
      // no longer says it is.
      //
      // It was the only signal, because there was nothing else: sampling many scenes
      // into one montage does dilute the palette toward desaturated mid-tones, and a
      // number was the only handle on it.  But a mean has no baseline -- 49.0% is fine
      // for a forest and terrible for a face -- and worse, it reported confidently on
      // garbage.  It read "mean saturation 49.0%" on a montage every cell of which was
      // an RGBA byte stream misread as RGB (rd_video.cpp, the by-seek arm's `-pix_fmt`
      // disagreeing with its buffer's channel count), and the real palette that
      // produced those cells was 7 of 16 entries PURE GREEN.  A summary statistic with
      // no baseline cannot tell you your input was corrupt, which is exactly the
      // failure it was relied on to catch.
      //
      // So the swatches are printed too, as 8-bit hex in the same order and format
      // `--palette-export FILE.txt` writes -- so this line can be diffed against that
      // file directly, and so 16 colours can be judged in one glance without a
      // reference value.  The saturation figure stays because it is the only handle on
      // the DILUTION effect specifically, which the swatches show but do not measure.
      if (!opt.palette_from.empty()) {
        // Already reported above, from the adopted colormap.
      } else if (result.palette_mosaic_w > 0) {
        std::printf("palette    : %d colours from %d per-frame palettes, "
                    "%dx%d swatch mosaic, %.1f MiB, %.1f ms "
                    "(stage 1 %.0f ms), mean saturation %.1f%%, %d near-neutral\n",
                    result.palette_colors, result.palette_sampled,
                    result.palette_mosaic_w, result.palette_mosaic_h,
                    static_cast<double>(result.palette_pixels * sizeof(rd::RgbaF)) /
                        (1024.0 * 1024.0),
                    result.palette_ms, result.palette_secondary_ms,
                    100.0 * rd::MeanPaletteSaturation(gate.palette()),
                    result.palette_neutral);
      } else {
        std::printf("palette    : %d colours from %d sampled frame(s), %zux%zu montage, "
                    "%.1f MiB, %.1f ms, mean saturation %.1f%%, %d near-neutral\n",
                    result.palette_colors, result.palette_sampled, tile, tile,
                    static_cast<double>(result.palette_pixels * sizeof(rd::RgbaF)) /
                        (1024.0 * 1024.0),
                    result.palette_ms,
                    100.0 * rd::MeanPaletteSaturation(gate.palette()),
                    result.palette_neutral);
      }
      for (int i = 0; i < palette->count && i < opt.dump_palette_limit; ++i) {
        const rd::PaletteEntry& e = palette->entries[i];
        std::printf("  %2d: %9.1f %9.1f %9.1f\n", i, e.r, e.g, e.b);
      }

      // The swatches, always (unlike the Q16 dump above, which is behind
      // --dump-palette-limit) and 8 per line so a 16-colour palette is two lines.
      //
      // Deliberately the SAME hex, in the SAME order, as the `hex` column of
      // --palette-export FILE.txt, so this and that file can be diffed or eyeballed
      // against each other without a conversion step.  Rounded the same way too --
      // e.r / 257.0 + 0.5 -- rather than invented here, so two runs that disagree
      // disagree about the palette and not about the printing.
      for (int i = 0; i < palette->count; i += 8) {
        std::printf("swatches   :");
        for (int k = i; k < palette->count && k < i + 8; ++k) {
          const rd::PaletteEntry& e = palette->entries[k];
          std::printf(" %02X%02X%02X",
                      static_cast<unsigned>(e.r / 257.0 + 0.5),
                      static_cast<unsigned>(e.g / 257.0 + 0.5),
                      static_cast<unsigned>(e.b / 257.0 + 0.5));
        }
        std::printf("\n");
      }

      if (!opt.palette_export.empty()) {
        if (!rd::ImWritePalette(gate.palette(), opt.palette_export, &error)) {
          std::fprintf(stderr, "error: --palette-export %s: %s\n",
                       opt.palette_export.c_str(), error.c_str());
          return 1;
        }
        std::printf("palette    : written to %s (%d colours; read it back with "
                    "--palette-from)\n",
                    opt.palette_export.c_str(), gate.palette().count);
      }
  };
  // Joining is what makes every read below safe, and it must happen on EVERY exit
  // path: a detached thread outliving the interrupt handler, or writing into a
  // result that has already been printed, is the failure this line exists to prevent.
  auto join_palette = [&] {
    if (palette_thread.joinable()) palette_thread.join();
  };

  if (opt.palette_only) {
    join_palette();
    report_palette();
    std::printf("palette-only: stopping before the dither, as asked\n");
    return 0;
  }

  // ---- crash-safe segment loop --------------------------------------------
  // A power cut cannot be caught, so the only defence is to make progress durable
  // and to bound what a cut costs.  The output is therefore written as a series of
  // independently-finalised segments, and after each one a checkpoint is flushed
  // to stable storage before the next begins.  A cut costs at most one segment,
  // and a torn checkpoint is impossible because the record is renamed into place
  // durably rather than updated in place.
  //
  // Without --resume this reduces to a single segment covering the whole clip, so
  // the ordinary path is unchanged.
  if (opt.segment_frames <= 0) {
    // No segmentation: run once, exactly as before.
    rd::InstallInterruptHandler();
    const bool ok = rd::VideoProcess(in_path, out_path, opt, info, gate,
                                     &result, &error);
    // The reader has been running for the whole palette build.  Join before reading any
    // palette_* field, then report, then summarise.
    join_palette();
    const bool interrupted = rd::Interrupted();
    rd::RemoveInterruptHandler();
    if (interrupted) return ReportInterrupted(out_path);
    if (!ok) {
      std::fprintf(stderr, "error: %s\n", error.c_str());
      return 1;
    }
    report_palette();
    PrintVideoSummary(result);
    return 0;
  }

  rd::Checkpoint cp;
  cp.input = in_path;
  rd::FileIdentity(in_path, &cp.input_bytes, &cp.input_mtime);
  cp.width = info.width;
  cp.height = info.height;
  cp.total_frames = info.frames;
  cp.colors = opt.colors;
  cp.block = opt.block;
  cp.segment_frames = opt.segment_frames;
  cp.palette_path = opt.palette_export;

  const std::string ckpt_path = rd::CheckpointPathFor(out_path);
  if (opt.resume) {
    rd::Checkpoint loaded;
    std::string why;
    if (rd::CheckpointLoad(ckpt_path, &loaded, &why)) {
      std::vector<int> dropped;
      std::string mismatch;
      if (rd::CheckpointValidate(&loaded, in_path, info.frames, opt.colors,
                                 opt.block, opt.segment_frames, &dropped,
                                 &mismatch)) {
        cp.segments = loaded.segments;
        std::int64_t done = 0;
        for (const rd::SegmentRecord& s : cp.segments) done += s.frames;
        std::printf("resume     : %s -- %lld of %lld frame(s) already done in "
                    "%zu segment(s)\n",
                    ckpt_path.c_str(), static_cast<long long>(done),
                    static_cast<long long>(info.frames), cp.segments.size());
        if (!dropped.empty()) {
          std::fprintf(stderr,
                       "resume     : %zu recorded segment(s) failed verification "
                       "and will be redone:",
                       dropped.size());
          for (int d : dropped) std::fprintf(stderr, " %d", d);
          std::fprintf(stderr, "\n");
        }
        if (!cp.palette_path.empty() && rd::FileSize(cp.palette_path) == 0) {
          std::fprintf(stderr,
                       "resume     : the saved palette %s is gone; rebuilding it\n",
                       cp.palette_path.c_str());
          cp.palette_path.clear();
        }
      } else {
        std::fprintf(stderr,
                     "resume     : ignoring %s -- %s\n", ckpt_path.c_str(),
                     mismatch.c_str());
      }
    } else {
      std::fprintf(stderr, "resume     : starting fresh (%s)\n", why.c_str());
    }
  } else {
    (void)DeleteFileA(ckpt_path.c_str());
  }

  const std::int64_t seg_frames = opt.segment_frames;
  const int seg_count = info.frames > 0
                            ? static_cast<int>((info.frames + seg_frames - 1) /
                                               seg_frames)
                            : 1;
  // Sized by seg_count, not by the number of recorded segments: on a fresh run
  // there are no records at all, and indexing by seg would run off the end.
  std::vector<bool> done(static_cast<std::size_t>(seg_count), false);
  for (const rd::SegmentRecord& s : cp.segments) {
    if (s.index >= 0 && s.index < seg_count) {
      done[static_cast<std::size_t>(s.index)] = true;
    }
  }

  rd::InstallInterruptHandler();
  bool failed = false;
  // The gate published ABOVE is reused for every segment.  Publishing again here is what
  // broke this path, and it is worth recording why, because the bug looked like a
  // missing join rather than an extra publish:
  //
  //   * Every arm of the palette if/else chain publishes into `gate` (rd_cli.cpp:588 for
  //     --palette-from, :614 on the overlap thread, :621 for the serial build).  So by the
  //     time control reaches the segment loop the palette is ALREADY published.
  //   * This code published a SECOND time, from std::move(*palette) and std::move(*tree),
  //     into a fresh `seg_gate`.  Palette is an aggregate holding a fixed 65536-entry
  //     array, so moving from it copies and leaves the contents readable -- which is why
  //     the dither half kept working and hid this.  ColorTree holds a std::vector<QNode>,
  //     so the move actually EMPTIED it.  The segment loop therefore ran against an empty
  //     tree and VideoProcess refused it: "segment 1: palette/tree upload failed".
  //   * With RD_PALETTE_OVERLAP=1 the palette thread was still running and writing *tree
  //     while the line above moved from it -- a data race on a vector mid-reallocation,
  //     which is the access violation (0xC0000005) rather than the tidy error.
  //   * And this path never called join_palette() at all, so with the overlap on, a
  //     joinable std::thread reached its destructor at scope exit.
  //
  // `gate` is const-correct for this: VideoProcess takes it by const reference and calls
  // gate.Wait(), which is idempotent, so one publish serves any number of segments.  The
  // palette is identical for every segment anyway -- it is a serial prefix, spent once.
  //
  // The join is still required, and it belongs HERE rather than at the publish: the
  // overlap thread is the only writer of `*palette`/`*tree`/`result`, and everything below
  // reads them.
  join_palette();
  report_palette();
  for (int seg = 0; seg < seg_count; ++seg) {
    if (done[static_cast<std::size_t>(seg)]) continue;
    const std::int64_t first = static_cast<std::int64_t>(seg) * seg_frames;
    const std::int64_t want =
        std::min<std::int64_t>(seg_frames, info.frames - first);
    const std::string seg_path = rd::SegmentPathFor(out_path, seg);
    // Discard any partial file from a previous attempt: its presence would be
    // mistaken for progress by a careless resume, and it is being rewritten.
    (void)DeleteFileA(seg_path.c_str());

    opt.start_frame = first;
    opt.frames_in_segment = want;
    rd::VideoResult seg_result;
    std::printf("segment %d/%d: frames %lld..%lld -> %s\n", seg + 1, seg_count,
                static_cast<long long>(first),
                static_cast<long long>(first + want - 1), seg_path.c_str());
    // Same gate, published once by the caller above and reused for every segment.
    if (!rd::VideoProcess(in_path, seg_path, opt, info, gate,
                          &seg_result, &error)) {
      std::fprintf(stderr, "error: segment %d: %s\n", seg + 1, error.c_str());
      failed = true;
      break;
    }
    if (rd::Interrupted()) {
      const bool intr = rd::Interrupted();
      rd::RemoveInterruptHandler();
      return ReportInterrupted(out_path, intr);
    }
    const std::uint64_t bytes = rd::FileSize(seg_path);
    if (bytes == 0) {
      std::fprintf(stderr, "error: segment %d produced an empty file\n", seg + 1);
      failed = true;
      break;
    }
    rd::SegmentRecord rec;
    rec.index = seg;
    rec.first_frame = first;
    rec.frames = want;
    rec.path = seg_path;
    rec.bytes = bytes;
    cp.segments.push_back(rec);
    if (!rd::CheckpointSave(ckpt_path, cp, &error)) {
      std::fprintf(stderr,
                   "error: cannot record progress (%s); the render is done but a\n"
                   "       resume would restart it.  Keeping the segment files.\n",
                   error.c_str());
      failed = true;
      break;
    }
    std::printf("segment %d/%d: done, %llu bytes, checkpointed (%zu recorded)\n",
                seg + 1, seg_count, static_cast<unsigned long long>(bytes),
                cp.segments.size());
    // Accumulate rather than overwrite.  VideoProcess started its own clock for this
    // segment and the palette was built once, before the loop, so seg_result.total_ms
    // covers one segment only; taking it as the whole job's time reported N times too
    // few seconds and a per-segment palette cost of zero.  The palette is carried
    // across from the pre-loop result and counted exactly once.
    //
    // `seg_result.frames` is frames DITHERED, which is not the same as frames in the
    // output file, and on this path the two genuinely differ.  The writer clips a
    // batch that straddles the segment boundary (rd_video.cpp:3595-3599:
    // `frames_to_write` is clamped to `frames_in_segment - segment_written`), while
    // the dither worker has already counted the whole batch at :3447
    // (`frames_dithered += b.frames`).  So every segment reports up to
    // batch_frames - 1 frames that were dithered and then not written, and the
    // overcount is repeated once per segment.  Measured shape, not a guess: at
    // --segment-frames 1000 --batch-frames 16 over 18001 frames that is 19 segments
    // and up to 19*15 = 285 phantom frames, which at 51 fps is 5.6 s of fictitious
    // time and inflates the reported frame count by 1.6%.  It is also a numerator
    // that disagrees with the file: ffprobe on the joined output is the authority and
    // this is not.
    //
    // `want` is the number of frames this segment was asked to cover, which is what
    // the segment file holds once the encoder has accepted them -- and it is bounded
    // by the same clip the writer applies, because both derive from
    // opt.frames_in_segment.  On the LAST segment `want` is
    // min(seg_frames, info.frames - first), so a container that over-reports its own
    // length cannot inflate the count either.
    result.frames += static_cast<std::int64_t>(want);
    result.total_ms += seg_result.total_ms;
    result.decode_ms += seg_result.decode_ms;
    result.dither_ms += seg_result.dither_ms;
    result.encode_ms += seg_result.encode_ms;
    result.convert_ms += seg_result.convert_ms;
    result.pipe_ms += seg_result.pipe_ms;
    result.cpu_frames += seg_result.cpu_frames;
    result.gpu_frames += seg_result.gpu_frames;
  }
  const bool interrupted = rd::Interrupted();
  rd::RemoveInterruptHandler();
  if (interrupted) return ReportInterrupted(out_path, true);
  if (failed) return 1;

  // ---- join the segments ---------------------------------------------------
  if (!JoinSegments(out_path, cp, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    std::fprintf(stderr,
                 "       the segments are intact; re-run with the same command to "
                 "resume, or join them by hand.\n");
    return 1;
  }
  (void)DeleteFileA(ckpt_path.c_str());
  std::printf("output     : %s (%zu segment(s) joined)\n", out_path.c_str(),
              cp.segments.size());

  // The throughput report.  It used to be missing here entirely: the non-segment
  // path returns through PrintVideoSummary and the segment path returned 0
  // directly, so a segmented run -- which is the *crash-safe* path, and therefore
  // the one long renders use -- reported no fps, no stage breakdown and no wall
  // time at all.  Measured on a 1000-frame clip in 4 segments: the join and the
  // output line appeared, and nothing else.
  //
  // The palette is added here and only here.  It is spent once, before the loop,
  // so none of the per-segment VideoProcess calls can have counted it -- each
  // started its own clock with palette_ms = 0.  Without this the fps denominator
  // excludes the palette, which is precisely the bug that was fixed on the
  // non-segment path and would otherwise have been reintroduced here.
  if (result.palette_ms > 0.0) {
    result.total_ms += result.palette_ms;
  }
  PrintVideoSummary(result);
  return 0;
}

}  // namespace

namespace {

// ImageMagick needs the path of its own magick.exe to locate policy.xml and the
// coder modules.  RD_MAGICK_EXE wins, then MAGICK_HOME, then the build default.
std::string RditherMagickExe() {
  if (const char* explicit_path = std::getenv("RD_MAGICK_EXE")) {
    return explicit_path;
  }
  if (const char* home = std::getenv("MAGICK_HOME")) {
    return std::string(home) + "\\magick.exe";
  }
  return RD_DEFAULT_MAGICK_EXE;
}

}  // namespace

int main(int argc, char** argv) {
  // Unbuffered stdout.  It carries only a few dozen lines, so the cost is
  // irrelevant, and it removes a genuinely confusing failure: redirected to a file
  // or a pipe, stdout is block-buffered, so the palette line could already have
  // been printed and still be sitting in the buffer -- which reads exactly like
  // "it never got to the palette stage".  Progress goes to stderr, which is
  // unbuffered by default, so it appears either way.
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    // Record the flag before dispatch, so `--crf 20 in.png out.png` can be told it
    // did nothing rather than being silently accepted.  Input and output positionals
    // are not flags.
    if (arg.size() > 1 && arg[0] == '-' && arg[1] == '-') opt.seen_flags.push_back(arg);
    if ((arg == "-h") || (arg == "--help")) {
      PrintUsage();
      return 0;
    } else if (arg == "--colors") {
      if (!NeedsValue(argc, i, "--colors")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--colors", &parsed_v)) return 2;
        ++i;
        opt.colors = static_cast<decltype(opt.colors)>(parsed_v);
      }
      if (opt.colors < 2 || opt.colors > rd::kMaxColormapSize) {
        std::fprintf(stderr, "error: --colors must be 2..%d\n",
                     rd::kMaxColormapSize);
        return 2;
      }
    } else if (arg == "--engine") {
      if (!NeedsValue(argc, i, "--engine")) return 2;
      const std::string engine = argv[++i];
      if (engine == "cpu") {
        opt.engine = rd::Engine::kCpu;
      } else if ((engine == "cuda") || (engine == "gpu")) {
        opt.engine = rd::Engine::kCuda;
      } else if ((engine == "approx") || (engine == "fast")) {
        opt.engine = rd::Engine::kApprox;
      } else if (engine == "blocks") {
        opt.engine = rd::Engine::kBlocks;
      } else if (engine == "opencl") {
        opt.engine = rd::Engine::kOpenCL;
      } else {
        std::fprintf(stderr,
                     "error: --engine must be cpu, cuda, approx, blocks or "
                     "opencl\n");
        return 2;
      }
    } else if (arg == "--blocks") {
      if (!NeedsValue(argc, i, "--blocks")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--blocks", &parsed_v)) return 2;
        ++i;
        opt.blocks.block = static_cast<decltype(opt.blocks.block)>(parsed_v);
      }
      // The floor is 16, and it is enforced HERE because the CLI uses this number in
      // its own arithmetic and neither engine protects it there.
      //
      //   * `--engine opencl --blocks 0` was a hard crash.  rd_opencl.cpp:1138 clamps
      //     internally -- `std::max(kErrorQueueLength, options.block)` -- and returns
      //     success, so nothing between here and the "blocks :" line below rejects 0;
      //     the line then evaluates (width*height + block - 1) / block with block == 0,
      //     which on x64 is 0xC0000094 and kills the process.  It happens AFTER the
      //     dither and BEFORE the file is written, so the whole run is lost to a flag
      //     nobody was told was wrong.
      //   * `--engine opencl --blocks 1` .. 15 did not crash, but printed a block count
      //     computed from the user's number while the engine walked 16-position blocks.
      //   * `--engine blocks` was already safe, and by accident: rd_blocks_cuda.cu:929
      //     refuses block < 16 with a message, so the CLI never reached the division.
      //
      // Clamping rather than refusing follows the --frames precedent below: the value
      // is adjusted to something that works and the adjustment is announced, so no
      // command line that runs today starts failing.  It also makes the number
      // self-consistent for the resume path, which records opt.block in the checkpoint
      // -- recording 4 while every engine walked 16 describes a render nobody performed.
      if (opt.blocks.block < 16) {
        std::fprintf(stderr,
                     "[dither] --blocks %d is below the error queue's depth of 16; "
                     "using 16.\n",
                     opt.blocks.block);
        opt.blocks.block = 16;
      }
    } else if (arg == "--approx-iters") {
      if (!NeedsValue(argc, i, "--approx-iters")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--approx-iters", &parsed_v)) return 2;
        ++i;
        opt.approx.iterations = static_cast<decltype(opt.approx.iterations)>(parsed_v);
      }
      if (opt.approx.iterations < 1) opt.approx.iterations = 1;
    } else if (arg == "--approx-taps") {
      if (!NeedsValue(argc, i, "--approx-taps")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--approx-taps", &parsed_v)) return 2;
        ++i;
        opt.approx.taps = static_cast<decltype(opt.approx.taps)>(parsed_v);
      }
    } else if (arg == "--approx-fp64") {
      opt.approx.fp64 = true;
    } else if (arg == "--approx-linear") {
      // Diagnostic: replaces IM's octree-subtree candidate set with a plain
      // nearest-colour scan so the report can separate the two effects.
      opt.approx.use_tree = false;
    } else if (arg == "--diffusion") {
      if (!NeedsValue(argc, i, "--diffusion")) return 2;
      {
        double parsed_d = 0.0;
        if (!ParseDoubleArg(argv[i + 1], "--diffusion", &parsed_d)) return 2;
        ++i;
        opt.diffusion = parsed_d;
      }
    } else if (arg == "--max-ram-mb") {
      if (!NeedsValue(argc, i, "--max-ram-mb")) return 2;
      std::size_t mb = 0;
      if (!ParseSize(argv[++i], &mb)) {
        std::fprintf(stderr, "error: --max-ram-mb expects a number\n");
        return 2;
      }
      // RANGE CHECK, and it is not defensive padding.  Both budget flags are stored in
      // megabytes and multiplied by 1024*1024 at the point of use (rd_cli.cpp:1785 for
      // RAM, :2099 for VRAM), in std::size_t, which is 64-bit unsigned.  So any value at
      // or above 2^44 MB wraps that multiply to zero: 2^44 * 2^20 == 2^64 == 0 mod 2^64.
      //
      // Measured before this existed, at exactly 2^44:
      //
      //     rdither --colors 16 --max-ram-mb 17592186044416 in.png out.png
      //     pixels     : disk (memory-mapped spill file) (0.6 MiB)
      //     ...
      //     exit 0
      //
      // A ZERO budget, which is the exact inverse of the request: the largest number the
      // flag accepts produced the smallest behaviour it has, and nothing said so.  The
      // output image is correct, so there is no visible symptom either -- it just quietly
      // spilled a whole render to disk because somebody typed a big number.
      //
      // 2^44 - 1 is the largest value the multiply cannot wrap, and it is about 16
      // exabytes, so nothing legitimate is refused by this bound.
      if (mb > kMaxBudgetMb) {
        std::fprintf(stderr,
                     "error: --max-ram-mb must be 0..%llu (that is 2^44-1; a larger "
                     "value overflows to zero and inverts the request)\n",
                     static_cast<unsigned long long>(kMaxBudgetMb));
        return 2;
      }
      opt.max_ram_mb = mb;
    } else if (arg == "--max-vram-mb") {
      if (!NeedsValue(argc, i, "--max-vram-mb")) return 2;
      std::size_t mb = 0;
      if (!ParseSize(argv[++i], &mb)) {
        std::fprintf(stderr, "error: --max-vram-mb expects a number\n");
        return 2;
      }
      // Same overflow, same bound, same reason -- see --max-ram-mb above.  Verified by
      // reading rather than by running: this one only bites on a GPU build, and the
      // multiply at :2099 has the identical shape.
      if (mb > kMaxBudgetMb) {
        std::fprintf(stderr,
                     "error: --max-vram-mb must be 0..%llu (that is 2^44-1; a larger "
                     "value overflows to zero and inverts the request)\n",
                     static_cast<unsigned long long>(kMaxBudgetMb));
        return 2;
      }
      opt.max_vram_mb = mb;
    } else if (arg == "--frames") {
      if (!NeedsValue(argc, i, "--frames")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--frames", &parsed_v)) return 2;
        ++i;
        opt.frames = static_cast<decltype(opt.frames)>(parsed_v);
      }
      if (opt.frames < 1) opt.frames = 1;
    } else if (arg == "--format") {
      if (!NeedsValue(argc, i, "--format")) return 2;
      opt.format = argv[++i];
    } else if (arg == "--verify") {
      opt.verify = true;
    } else if (arg == "--video") {
      opt.video = true;
    } else if (arg == "--dither") {
      if (!NeedsValue(argc, i, "--dither")) return 2;
      opt.dither = argv[++i];
    } else if (arg == "--list-dithers") {
      opt.list_dithers = true;
    } else if (arg == "--palette-budget-ms") {
      if (!NeedsValue(argc, i, "--palette-budget-ms")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--palette-budget-ms", &parsed_v)) return 2;
        ++i;
        opt.video_opt.palette_budget_ms = static_cast<decltype(opt.video_opt.palette_budget_ms)>(parsed_v);
      }
      if (opt.video_opt.palette_budget_ms <= 0) {
        std::fprintf(stderr, "[video] --palette-budget-ms must be positive.\n");
        return 2;
      }
    } else if (arg == "--palette-frames") {
      if (!NeedsValue(argc, i, "--palette-frames")) return 2;
      // "all" is accepted explicitly rather than relying on atoi() returning 0.
      const std::string value = argv[++i];
      if ((value == "all") || (value == "ALL")) {
        opt.video_opt.palette_frames = 0;
      } else {
        // atoi() here was a site the numeric-validation sweep missed, and it was the
        // worst instance of the class: 0 is not a neutral value for this option, it is
        // "all", so `--palette-frames abc` did not compute something useless -- it
        // silently asked for every frame, which for a long clip means sampling the
        // whole thing instead of a handful.
        long parsed_v = 0;
        if (!ParseIntArg(value.c_str(), "--palette-frames", &parsed_v)) return 2;
        opt.video_opt.palette_frames = static_cast<decltype(opt.video_opt.palette_frames)>(parsed_v);
      }
    } else if (arg == "--palette-tile") {
      if (!NeedsValue(argc, i, "--palette-tile")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--palette-tile", &parsed_v)) return 2;
        ++i;
        opt.video_opt.palette_tile = static_cast<decltype(opt.video_opt.palette_tile)>(parsed_v);
      }
    } else if (arg == "--batch-frames") {
      if (!NeedsValue(argc, i, "--batch-frames")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--batch-frames", &parsed_v)) return 2;
        ++i;
        opt.video_opt.batch_frames = static_cast<decltype(opt.video_opt.batch_frames)>(parsed_v);
      }
    } else if (arg == "--video-codec") {
      if (!NeedsValue(argc, i, "--video-codec")) return 2;
      opt.video_opt.codec = argv[++i];
    } else if (arg == "--crf") {
      if (!NeedsValue(argc, i, "--crf")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--crf", &parsed_v)) return 2;
        ++i;
        opt.video_opt.crf = static_cast<decltype(opt.video_opt.crf)>(parsed_v);
      }
    } else if (arg == "--preset") {
      if (!NeedsValue(argc, i, "--preset")) return 2;
      opt.video_opt.preset = argv[++i];
    } else if (arg == "--cpu-threads") {
      if (!NeedsValue(argc, i, "--cpu-threads")) return 2;
      const char* raw = argv[++i];
      // Parsed strictly rather than with atoi, because atoi returns 0 for anything
      // non-numeric and 0 is a MEANINGFUL value here: "GPU only".  So
      // `--cpu-threads auto` -- which reads like the obvious way to ask for the
      // default -- became "GPU only", and on a machine with no GPU that reported a
      // GPU-only error for a request the user never made.  The documented way to get
      // the default is to omit the flag, and now a typo says so.
      char* end = nullptr;
      const long parsed = std::strtol(raw, &end, 10);
      const bool numeric = end != nullptr && *end == '\0' && end != raw;
      if (!numeric) {
        std::fprintf(stderr,
                     "error: --cpu-threads wants a number: -1 for auto, 0 for "
                     "GPU only, or a positive worker count.  Got '%s'.\n"
                     "       Omit the flag entirely for auto.\n",
                     raw);
        return 2;
      }
      opt.video_opt.cpu_threads = static_cast<int>(parsed);
    } else if (arg == "--no-gpu") {
      opt.video_opt.use_gpu = false;
    } else if (arg == "--no-hwaccel") {
      opt.video_opt.hwaccel = 0;
    } else if (arg == "--input-mode") {
      if (!NeedsValue(argc, i, "--input-mode")) return 2;
      const std::string m = argv[++i];
      if (m != "yuv420" && m != "yuv444" && m != "yuv444-prepass" && m != "rgba64") {
        std::fprintf(stderr,
                     "[video] --input-mode %s: expected yuv420, yuv444, "
                     "yuv444-prepass or rgba64.\n", m.c_str());
        return 2;
      }
      opt.video_opt.input_mode = m;
    } else if (arg == "--no-audio") {
      // Audio is copied by default; this turns it off.  The flag is --no-audio rather
      // than --audio so that adding the default later does not change any existing
      // command line, and so the common case needs no flag at all.
      opt.video_opt.audio = false;
    } else if (arg == "--reader-threads") {
      if (!NeedsValue(argc, i, "--reader-threads")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--reader-threads", &parsed_v)) return 2;
        ++i;
        opt.video_opt.reader_threads = static_cast<decltype(opt.video_opt.reader_threads)>(parsed_v);
      }
    } else if (arg == "--decode-threads") {
      if (!NeedsValue(argc, i, "--decode-threads")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--decode-threads", &parsed_v)) return 2;
        ++i;
        opt.video_opt.decode_threads = static_cast<decltype(opt.video_opt.decode_threads)>(parsed_v);
      }
    } else if (arg == "--encode-threads") {
      if (!NeedsValue(argc, i, "--encode-threads")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--encode-threads", &parsed_v)) return 2;
        ++i;
        opt.video_opt.encode_threads = static_cast<decltype(opt.video_opt.encode_threads)>(parsed_v);
      }
    } else if (arg == "--queue-depth") {
      if (!NeedsValue(argc, i, "--queue-depth")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--queue-depth", &parsed_v)) return 2;
        ++i;
        opt.video_opt.queue_depth = static_cast<decltype(opt.video_opt.queue_depth)>(parsed_v);
      }
    } else if (arg == "--gpu-workers") {
      if (!NeedsValue(argc, i, "--gpu-workers")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--gpu-workers", &parsed_v)) return 2;
        ++i;
        opt.video_opt.gpu_workers = static_cast<decltype(opt.video_opt.gpu_workers)>(parsed_v);
      }
    } else if (arg == "--gpu-float-out") {
      opt.video_opt.gpu_float_out = true;
    } else if (arg == "--video-preserve-vfr") {
      // Off by default and not verified end to end; see VideoOptions::preserve_vfr.
      opt.video_opt.preserve_vfr = true;
    } else if (arg == "--video-lossless") {
      opt.video_opt.lossless = true;
    } else if (arg == "--palette-mode") {
      if (!NeedsValue(argc, i, "--palette-mode")) return 2;
      const std::string m = argv[++i];
      if (m == "auto" || m == "0") {
        opt.video_opt.palette_mode = 1;
      } else if (m == "pool" || m == "1") {
        opt.video_opt.palette_mode = 1;
      } else if (m == "montage" || m == "2") {
        opt.video_opt.palette_mode = 2;
      } else {
        std::fprintf(stderr, "error: --palette-mode must be pool or montage\n");
        return 2;
      }
    } else if (arg == "--segment-frames") {
      if (!NeedsValue(argc, i, "--segment-frames")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--segment-frames", &parsed_v)) return 2;
        ++i;
        opt.video_opt.segment_frames = static_cast<decltype(opt.video_opt.segment_frames)>(parsed_v);
      }
    } else if (arg == "--resume") {
      opt.video_opt.resume = true;
    } else if (arg == "--palette-export") {
      if (!NeedsValue(argc, i, "--palette-export")) return 2;
      opt.video_opt.palette_export = argv[++i];
    } else if (arg == "--palette-import") {
      if (!NeedsValue(argc, i, "--palette-import")) return 2;
      opt.palette_import = argv[++i];
      // The video path keeps its own copy so VideoBuildPalette can short-circuit
      // without knowing about the CLI.  Assigning here rather than in the
      // per-path blocks means one flag drives both.
      opt.video_opt.palette_from = opt.palette_import;
    } else if (arg == "--im-palette") {
      opt.video_opt.im_palette = true;
    } else if (arg == "--palette-from") {
      if (!NeedsValue(argc, i, "--palette-from")) return 2;
      opt.video_opt.palette_from = argv[++i];
    } else if (arg == "--palette-stage1-colors") {
      if (!NeedsValue(argc, i, "--palette-stage1-colors")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--palette-stage1-colors", &parsed_v)) return 2;
        ++i;
        opt.video_opt.palette_stage1_colors = static_cast<decltype(opt.video_opt.palette_stage1_colors)>(parsed_v);
      }
    } else if (arg == "--palette-max-samples") {
      if (!NeedsValue(argc, i, "--palette-max-samples")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--palette-max-samples", &parsed_v)) return 2;
        ++i;
        opt.video_opt.palette_max_samples = static_cast<decltype(opt.video_opt.palette_max_samples)>(parsed_v);
      }
    } else if (arg == "--palette-dedup") {
      if (!NeedsValue(argc, i, "--palette-dedup")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--palette-dedup", &parsed_v)) return 2;
        ++i;
        opt.video_opt.palette_dedup = static_cast<decltype(opt.video_opt.palette_dedup)>(parsed_v);
      }
    } else if (arg == "--palette-only") {
      opt.video_opt.palette_only = true;
    } else if (arg == "--host-grace-ms") {
      if (!NeedsValue(argc, i, "--host-grace-ms")) return 2;
      {
        long parsed_v = 0;
        if (!ParseIntArg(argv[i + 1], "--host-grace-ms", &parsed_v)) return 2;
        ++i;
        opt.video_opt.host_grace_ms = static_cast<decltype(opt.video_opt.host_grace_ms)>(parsed_v);
      }
    } else if (arg == "--mem-fraction") {
      if (!NeedsValue(argc, i, "--mem-fraction")) return 2;
      {
        double parsed_d = 0.0;
        if (!ParseDoubleArg(argv[i + 1], "--mem-fraction", &parsed_d)) return 2;
        ++i;
        opt.video_opt.mem_fraction = parsed_d;
      }
      // The assignment to rd::g_video_memory_fraction that used to be here is gone. It
      // was the only reference to that global outside its own declaration and
      // definition, so --mem-fraction wrote a value nothing ever read. The flag still
      // works: rd_video.cpp reads opt.mem_fraction directly.
    } else if (arg == "--video-pix-fmt") {
      if (!NeedsValue(argc, i, "--video-pix-fmt")) return 2;
      opt.video_opt.pix_fmt = argv[++i];
    } else if (arg == "--dump-palette") {
      opt.dump_palette = true;
    } else if (arg == "--self-test") {
      opt.self_test = true;
    } else if (arg == "--no-cache") {
      opt.no_cache = true;
    } else if (arg == "--dump-curve") {
      opt.dump_curve = true;
    } else if (arg == "--dump-palette") {
      opt.dump_palette = true;
    } else if (arg == "--quiet") {
      opt.quiet = true;
    } else if (!arg.empty() && (arg[0] == '-')) {
      std::fprintf(stderr, "error: unknown option '%s'\n", arg.c_str());
      return 2;
    } else if (opt.input.empty()) {
      opt.input = arg;
    } else if (opt.output.empty()) {
      opt.output = arg;
    } else {
      std::fprintf(stderr, "error: unexpected argument '%s'\n", arg.c_str());
      return 2;
    }
  }
  if (opt.list_dithers) {
    // Standalone: no input, no output, no palette.  So it is handled before the
    // "need an input" check rather than inside it.
    std::printf("registered dither algorithms:\n%s\n", rd::DitherListText().c_str());
    return 0;
  }
  // Validate the name before doing any work.  Silently running Riemersma because
  // someone typo'd a plugin name is the failure mode this check exists to prevent:
  // it would look exactly like "the plugin did nothing".
  if (opt.dither != "riemersma" && rd::FindDither(opt.dither.c_str()) == nullptr) {
    std::fprintf(stderr, "error: unknown --dither %s.\nRegistered:\n%s\n",
                 opt.dither.c_str(), rd::DitherListText().c_str());
    return 2;
  }
  if (opt.input.empty() || opt.output.empty()) {
    if (opt.self_test) {
      // The closed-form Hilbert point is what makes the CUDA engines' setup
      // cheap; prove it against the recursion it replaced rather than assume.
      // Both are bijections onto the grid, so the fix for a disagreement is a
      // dihedral transform of d2xy -- score all eight and report the best.
      int best_transform = -1;
      std::size_t best = static_cast<std::size_t>(-1);
      for (int t = 0; t < 8; ++t) {
        const std::size_t m = rd::CurveTransformMismatches(6, t);
        std::printf("  level 6, transform %d: %zu mismatches%s\n", t, m,
                    m == 0 ? "  <-- exact" : "");
        if (m < best) { best = m; best_transform = t; }
      }
      std::printf("best transform: %d (%zu mismatches at level 6)\n\n",
                  best_transform, best);
      int bad = 0;
      for (int level = 1; level <= 8; ++level) {
        const std::size_t mismatches =
            rd::CurveTransformMismatches(level, best_transform);
        std::printf("level %d: %zu of %zu positions %s\n", level, mismatches,
                    rd::CurveEntryCount(level),
                    mismatches == 0 ? "match" : "DIFFER");
        if (mismatches != 0 && level <= 3) {
          // Small enough to dump.  The last entry is the trailing
          // ForgetGravity visit, which is the origin, not a d2xy cell.
          const std::size_t side = static_cast<std::size_t>(1) << level;
          std::vector<rd::CurveStepEntry> curve(rd::CurveEntryCount(level));
          rd::BuildCurveSequence(level, side, side, curve.data());
          const std::size_t leaves = curve.empty() ? 0 : curve.size() - 1;
          for (std::size_t n = 0; n < leaves; ++n) {
            int x = 0, y = 0;
            rd::HilbertPoint(static_cast<std::int64_t>(n), level, &x, &y);
            if (x != curve[n].x || y != curve[n].y) {
              std::printf("    n=%-4zu recursion=(%d,%d) d2xy=(%d,%d)\n", n,
                          curve[n].x, curve[n].y, x, y);
            }
          }
          if (leaves > 0 && (curve[leaves].x != 0 || curve[leaves].y != 0)) {
            std::printf("    trailing visit=(%d,%d), expected the origin\n",
                        curve[leaves].x, curve[leaves].y);
          }
        }
        bad += (mismatches != 0) ? 1 : 0;
      }
            // PaletteGate: the in-flight palette handle the video overlap uses.  Exercised here
      // rather than only through the end-to-end probe because every later task depends on
      // this type's blocking and publication semantics, and a probe that renders video
      // would take minutes to tell us a Wait() never woke.
      //
      // Written to stderr, not stdout: stdout is block-buffered when redirected, so a crash
      // anywhere in this block loses every preceding printf and tells us nothing about where
      // it died.  stderr is unbuffered, so the last line printed is the one that crashed --
      // which is how the two stack overflows below were localised rather than guessed at.
      {
        std::fprintf(stderr, "palette gate:\n");
        int gbad = 0;
        auto check = [&](const char* what, bool held) {
          std::fprintf(stderr, "  %-34s %s\n", what, held ? "ok" : "FAIL");
          std::fflush(stderr);
          if (!held) ++gbad;
        };

        // A published gate wakes Wait, reports ok, and hands back the palette.
        {
          rd::PaletteGate gate;
          auto p = std::make_unique<rd::Palette>();  // heap: sizeof(Palette) is ~2 MB
          p->count = 3;
          rd::ColorTree t;
          gate.Publish(std::move(*p), std::move(t));
          gate.Wait();
          check("publish wakes Wait", gate.ok());
          check("published palette has 3 colours", gate.palette().count == 3);
        }

        // A failed gate wakes Wait, reports not-ok, and carries the error string.
        {
          rd::PaletteGate gate;
          gate.Fail("no ffmpeg on PATH");
          gate.Wait();
          check("fail wakes Wait", !gate.ok());
          check("fail carries its message", gate.error() == "no ffmpeg on PATH");
        }

        // Fail after Publish must NOT clobber a palette that already landed: the reader may
        // already be waiting on it and the palette is still usable.
        //
        // The assertion that matters is that the ERROR is not recorded.  Asserting ok() and
        // the colour count here would be tautological -- Fail() never touches palette_ or
        // ok_, so those hold with or without the guard.  The guard's whole effect is on
        // error_, and deleting it (checked by mutation) leaves every other assertion
        // passing, which is how that was found.
        {
          rd::PaletteGate gate;
          auto p = std::make_unique<rd::Palette>();
          p->count = 7;
          rd::ColorTree t;
          gate.Publish(std::move(*p), std::move(t));
          gate.Fail("late failure");
          gate.Wait();
          check("late Fail does not clobber", gate.ok());
          check("late Fail keeps 7 colours", gate.palette().count == 7);
          check("late Fail records no error", gate.error().empty());
        }

        // Wait() on a gate nothing has touched yet must block until a publisher arrives.
        // This is the property the whole overlap rests on, and the one a static check
        // cannot see.
        {
          rd::PaletteGate gate;
          std::atomic<bool> finished{false};
          std::thread waiter([&] {
            gate.Wait();
            finished.store(true);
          });
          std::this_thread::sleep_for(std::chrono::milliseconds(40));
          const bool woke_early = finished.load();
          auto p = std::make_unique<rd::Palette>();
          p->count = 1;
          rd::ColorTree t;
          gate.Publish(std::move(*p), std::move(t));
          waiter.join();
          check("Wait blocks until publish", !woke_early);
          check("Wait returns after publish", finished.load());
        }

        bad += (gbad != 0) ? 1 : 0;
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
      }      std::printf("%s\n", bad == 0 ? "self-test PASSED" : "self-test FAILED");
      return bad == 0 ? 0 : 5;
    }
    PrintUsage();
    return 2;
  }
  if (opt.self_test) {
    std::fprintf(stderr, "error: --self-test takes no input/output\n");
    return 2;
  }
  if (opt.video) {
    // The mirror image of the video-only refusal further down, and it has the same
    // justification: a flag that parses, is stored, and is then read by nothing on
    // this path is a setting the user believes took effect.
    //
    // Only two are refused, and both are refused because they make the user believe
    // something FALSE ABOUT THE OUTPUT rather than merely about a knob:
    //
    //   --verify    RunVideo never reads it.  Nothing in src/rd_video.cpp compares
    //               anything against ImageMagick, so `--verify --video` rendered the
    //               clip, printed no verify line, and exited 0 -- and that is the one
    //               flag whose entire contract is "tell me if this is wrong".
    //   --dither N  A plugin dither is a host function pointer and the video pipeline
    //               is built around the block partition, so there is no way to call it.
    //               The name is validated above, so a correctly spelled plugin name is
    //               accepted and then Riemersma runs instead: exactly the failure that
    //               validation was added to prevent, reached by adding --video.
    //
    // Deliberately NOT refused, because each is a no-op that cannot mislead -- and
    // this is a judgement about which no-ops mislead, so it is worth stating the
    // test: a flag is refused when believing it took effect would make the reader
    // wrong about the OUTPUT FILE, and left alone when it would only make them
    // wrong about a knob they can see is untouched.
    //   --no-cache  The video path is cache-free by construction (RiemersmaBlocksCpu
    //                and the two GPU blocks engines never build a memo table), so the
    //                request is satisfied by accident rather than ignored.
    //   --format    ffmpeg picks the container from the extension; there is no
    //                ImageMagick coder on this path at all.
    //   --frames    The video equivalent is --batch-frames, whose default is already
    //                16, so the request is met.
    //   --dump-curve, --dump-palette, --max-ram-mb  A diagnostic that prints nothing,
    //                and a limit the pipeline expresses as --mem-fraction.
    // A stricter line would refuse all of them, and the cost of that is a flag the
    // user can put in a script for both paths; the benefit is a message about a
    // setting that had no effect on a file that is correct.  UNVERIFIED: whether any
    // existing caller in this repository passes one of these with --video -- there is
    // no such call site in src/ or tools/, but scripts outside the tree are not
    // visible from here.
    if (opt.verify) {
      std::fprintf(stderr,
                   "error: --verify cannot apply to --video: there is no reference "
                   "Riemersma pass on that path, so it would exit 0 without comparing "
                   "anything.  Drop it.\n");
      return 2;
    }
    if (opt.dither != "riemersma") {
      std::fprintf(stderr,
                   "error: --dither %s has no --video path: a plugin is a host "
                   "function pointer and the video pipeline is built around the block "
                   "partition, so Riemersma would run instead.  Drop the flag, or "
                   "dither the frames as images.\n", opt.dither.c_str());
      return 2;
    }
    // Carry the shared colour/dither settings into the video options.
    opt.video_opt.colors = opt.colors;
    opt.video_opt.block = opt.blocks.block;
    opt.video_opt.diffusion = opt.diffusion;
    opt.video_opt.quiet = opt.quiet;
    // --max-ram-mb reaches the video queue only when the user actually typed it.  The
    // default is 512 while the video path's own default is derived from physical RAM,
    // so forwarding the value unconditionally would shrink every default run.  See the
    // `max_ram_mb_set` note in include/rd_video.h.
    opt.video_opt.max_ram_mb = opt.max_ram_mb;
    opt.video_opt.max_ram_mb_set = false;
    for (const std::string& s : opt.seen_flags) {
      if (s == "--max-ram-mb") { opt.video_opt.max_ram_mb_set = true; break; }
    }
    opt.video_opt.dump_palette_limit = opt.dump_palette ? 16 : 0;
    // --engine opencl selects the OpenCL video engine.  Everything else keeps
    // the historical behaviour, which is that --video always used the block-
    // parallel engine: the pipeline is built around the block partition
    // (independent error queues per block, one per work item), and the default
    // --engine is `cpu`, so rejecting that would break a plain `--video in out`.
    // `approx` is the one that genuinely has no video path -- it is a single-frame
    // iterative solver with no batched form -- so it is refused by name rather
    // than quietly swapped for a different dither.
    if (opt.engine == rd::Engine::kOpenCL) {
      opt.video_opt.gpu_engine = rd::VideoOptions::GpuEngine::kOpenCL;
      // The OpenCL video path now carries planar 4:4:4 in both directions, which is
      // what CUDA uses by default.  Checked HERE, at parse time, rather than leaving
      // the engine to refuse after the palette stage has already run and a partial
      // output file exists: an error the user waits a second for, and pays for, to be
      // told something the command line already said.
      //
      // What is still refused is 4:2:0 and the yuv444-prepass.  rgba64le remains
      // legal and remains bit-identical to CUDA, so nothing that worked before is
      // taken away -- but it is no longer required, and that is the difference
      // between 8 bytes per pixel and 3.
      const std::string& mode = opt.video_opt.input_mode;
      if (mode != "rgba64" && mode != "yuv444") {
        std::fprintf(stderr,
                     "error: --engine opencl with --video supports --input-mode "
                     "rgba64 and yuv444; got '%s'.  Planar 4:2:0 and the "
                     "yuv444-prepass are not ported to OpenCL.  Use --engine blocks "
                     "for that input.\n", mode.c_str());
        return 2;
      }
    } else if (opt.engine == rd::Engine::kApprox) {
      std::fprintf(stderr,
                   "error: --engine approx is a single-frame solver with no video "
                   "path; --video needs a block-parallel engine (blocks or opencl).\n");
      return 2;
    } else {
      opt.video_opt.gpu_engine = rd::VideoOptions::GpuEngine::kBlocks;
      // `cpu` and `blocks` both land here, because --video has always used the
      // block-parallel engine.  Only the second one is a request for a GPU, and
      // VideoProcess refuses rather than falling back to the host when there is no
      // CUDA device -- so the distinction has to be carried across, or the refusal
      // meant for `--engine blocks` also breaks the default `--video`.
      opt.video_opt.gpu_engine_named = (opt.engine == rd::Engine::kBlocks);
    }
    if (opt.input.empty() || opt.output.empty()) {
      std::fprintf(stderr, "error: --video needs <input> <output>\n");
      return 2;
    }
    std::string error;
    if (!rd::ImStartup(RditherMagickExe(), &error)) {
      std::fprintf(stderr, "error: %s\n", error.c_str());
      return 1;
    }
    const int status = RunVideo(opt.input, opt.output, opt.video_opt);
    rd::ImShutdown();
    return status;
  }

  // Flags that only mean something to the video pipeline.  They parse, they are
  // stored, and on this path nothing reads them -- so `rdither --crf 20 in.png
  // out.png` used to render an image and exit 0, having done nothing with --crf.
  // Silent no-op is worse than a refusal: the user believes a setting took effect.
  //
  // Palette flags are split, and the split is by what the image path actually reads
  // rather than by whether the word "palette" is in the name:
  //
  //   honoured here  --palette-export, --palette-import.  Both write or read the
  //                  palette this run uses, and an image is exactly the case where you
  //                  want to fix a palette once and re-use it (see the blocks below).
  //   listed below   --palette-from.  This one was previously described as honoured for
  //                  images, which is not true: the image path reads opt.palette_import
  //                  and nothing reads opt.video_opt.palette_from, so
  //                      rdither --palette-from p.png in.png out.png
  //                  dithered with a palette derived from the image and said nothing.
  //                  That is the worst shape of this defect -- not a knob that does
  //                  nothing but a PALETTE that does not, so the picture is not the one
  //                  that was asked for.  It is listed here rather than honoured,
  //                  because honouring it means a second reader in the image path, and
  //                  --palette-import already is that reader with the colour-count
  //                  check attached.
  //   reported, not refused  --palette-frames, --palette-tile, --palette-budget-ms,
  //                  --palette-max-samples, --palette-dedup, --palette-stage1-colors,
  //                  --palette-mode, --im-palette.  These describe how a VIDEO palette
  //                  is sampled from a decoded stream; the image path derives its
  //                  palette from the image with ImBuildPalette and never reads them.
  //                  They cannot change the picture -- the palette comes from the
  //                  image either way -- so the answer is a named note rather than an
  //                  exit code, which is also why they are exempt from the exit-2
  //                  treatment the rest of this list gets.  See the `bypassed` block
  //                  below, which now names them whenever they were TYPED.
  //
  // No flag in this list is meaningful on an image, so nothing here is caught by
  // mistake.  That was checked field by field rather than by eye: each entry sets one
  // field on opt.video_opt (or the global g_video_memory_fraction), and each of those
  // fields is read only inside RunVideo or VideoProcess.  The one that needed a second
  // look was --mem-fraction, which also assigns rd::g_video_memory_fraction at parse
  // time -- and that global is read NOWHERE in the tree: grepping it returns the
  // declaration, the definition at rd_riemersma_cpu.cpp:18, and this assignment, and
  // no reader.  The video path uses opt.mem_fraction directly (rd_video.cpp:2140), so
  // the global is dead code and the flag is video-only in fact as well as in name.
  {
    // The four at the end of the second group are the ones this list was still
    // missing, and all four are read by nothing on the image path -- verified by
    // reading every use of the fields they set, not by the absence of a mention:
    //   --cpu-threads N   opt.video_opt.cpu_threads.  The image path allocates its
    //                    worker count from hardware_concurrency inside the engine, or
    //                    not at all for the sequential walk; there is no reader here.
    //   --no-gpu          opt.video_opt.use_gpu.  Refused inside VideoProcess on a
    //                    machine with no CUDA device, and irrelevant to an image.
    //   --input-mode M    opt.video_opt.input_mode.  A decoder-output format.  The
    //                    image path's input format is whatever the coder produced.
    //   --palette-only    opt.video_opt.palette_only.  It could plausibly mean
    //                    something here -- the image path does build and print a
    //                    palette before the dither -- but nothing implements it, so
    //                    `rdither --palette-only in.png out.png` renders and dithers
    //                    the image and stops, which is the opposite of what the flag
    //                    says.  Refused rather than implemented, because implementing
    //                    it means skipping the write and that is a behaviour change to
    //                    a flag that has always been accepted.
    static const char* const kVideoOnly[] = {
        "--crf", "--preset", "--video-codec", "--video-lossless", "--video-pix-fmt",
        "--video-preserve-vfr", "--no-audio", "--segment-frames", "--resume",
        "--reader-threads", "--decode-threads", "--encode-threads", "--no-hwaccel",
        "--host-grace-ms", "--queue-depth", "--batch-frames", "--gpu-workers",
        "--gpu-float-out", "--mem-fraction", "--palette-from",
        "--cpu-threads", "--no-gpu", "--input-mode", "--palette-only",
    };
    std::vector<std::string> ignored;
    for (const char* f : kVideoOnly) {
      for (const std::string& s : opt.seen_flags) {
        if (s == f) { ignored.push_back(s); break; }
      }
    }
    if (!ignored.empty()) {
      std::string list;
      for (std::size_t k = 0; k < ignored.size(); ++k) {
        if (k != 0) list += ", ";
        list += ignored[k];
      }
      std::fprintf(stderr,
                   "error: %s %s video-only and cannot apply to an image; "
                   "drop %s, or pass --video.\n",
                   list.c_str(),
                   ignored.size() == 1 ? "is" : "are",
                   ignored.size() == 1 ? "it" : "them");
      return 2;
    }
  }

  std::string error;
  // ImageMagick needs the path of its own magick.exe to locate policy.xml and
  // the coder modules.  RD_MAGICK_EXE wins, then MAGICK_HOME, then whatever the
  // build recorded.
  if (!rd::ImStartup(RditherMagickExe(), &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 1;
  }

  rd::LoadedImage image;
  if (!rd::ImLoad(opt.input, &image, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    rd::ImShutdown();
    return 1;
  }
  if (!opt.quiet) {
    std::printf("input      : %s\n", opt.input.c_str());
    std::printf("geometry   : %zux%zu\n", image.width, image.height);
    std::printf("alpha      : %s\n", image.has_alpha ? "yes" : "no");
  }

  const std::size_t max_ram_bytes = opt.max_ram_mb * 1024ull * 1024ull;
  rd::PixelStore* store =
      rd::PixelStore::Create(image.width, image.height, max_ram_bytes, &error);
  if (store == nullptr) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    rd::ImFree(&image);
    rd::ImShutdown();
    return 1;
  }
  if (!opt.quiet) {
    std::printf("pixels     : %s (%.1f MiB)\n", store->backing(),
                static_cast<double>(store->byte_size()) / (1024.0 * 1024.0));
  }
  if (!rd::ImExtract(image, store, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    delete store;
    rd::ImFree(&image);
    rd::ImShutdown();
    return 1;
  }
  store->Prefetch();

  // ---- palette, straight out of ImageMagick's own quantizer ---------------
  // Palette is 2 MiB (MaxColormapSize == 65536 for Q16), so it goes on the heap.
  std::unique_ptr<rd::Palette> palette(new rd::Palette());
  std::unique_ptr<rd::ColorTree> palette_tree(new rd::ColorTree());
  if (!opt.palette_import.empty()) {
    // An imported palette REPLACES the quantizer rather than seeding it.  The
    // tree is built from the palette, so the imported colours are authoritative;
    // had the tree been built from the image's pixels, the octree would still
    // restrict the search to the image's own colour distribution and the import
    // would be a suggestion rather than a command.
    if (!rd::ImReadPalette(opt.palette_import, opt.colors, palette.get(),
                           palette_tree.get(), &error)) {
      std::fprintf(stderr, "error: --palette-import %s\n", error.c_str());
      delete store;
      rd::ImFree(&image);
      rd::ImShutdown();
      return 1;
    }
    if (!opt.quiet) {
      std::printf("palette    : %d colours imported from %s (associate_alpha=%s)\n",
                  palette->count, opt.palette_import.c_str(),
                  palette->associate_alpha ? "yes" : "no");
      std::printf("             mean saturation %.1f%%, %d near-neutral\n",
                  100.0 * rd::MeanPaletteSaturation(*palette),
                  rd::CountNeutralPaletteEntries(*palette, 0.2));
    }
  } else if (!rd::ImBuildPalette(image, opt.colors, palette.get(), &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    delete store;
    rd::ImFree(&image);
    rd::ImShutdown();
    return 1;
  }
  // The sampler knobs are all about HOW a palette is derived, and on this path they
  // derive nothing: ImBuildPalette quantises the image, and none of these are read
  // again.  An imported palette makes that doubly true -- there is no derivation to
  // influence at all.
  //
  // Driven off `seen_flags` rather than off the option VALUES, which is the whole
  // reason that list exists (see the comment on Options::seen_flags).  The value test
  // this replaces could only fire when the user happened to pass something other than
  // the default: `--palette-tile 128` is the default and is exactly as inert as
  // `--palette-tile 256`, but only the second was ever named.  So the diagnostic
  // under-reported precisely in the case where the user had made a deliberate choice
  // and been silently ignored -- the case the note exists for.
  //
  // The condition is also no longer gated on an import being present, because the
  // gate was the other half of the hole: with no --palette-import, `--palette-frames
  // 400 in.png out.png` was a flag that did nothing and said nothing.
  {
    static const char* const kSamplerKnobs[] = {
        "--im-palette", "--palette-frames", "--palette-tile", "--palette-budget-ms",
        "--palette-max-samples", "--palette-dedup", "--palette-stage1-colors",
        "--palette-mode",
    };
    std::vector<std::string> bypassed;
    for (const char* k : kSamplerKnobs) {
      for (const std::string& s : opt.seen_flags) {
        if (s == k) { bypassed.push_back(s); break; }
      }
    }
    if (!bypassed.empty() && !opt.quiet) {
      std::string list;
      for (std::size_t i = 0; i < bypassed.size(); ++i) {
        if (i != 0) list += ", ";
        list += bypassed[i];
      }
      std::printf("palette    : ignoring %s -- %s\n", list.c_str(),
                  opt.palette_import.empty()
                      ? "this path derives the palette from the image with "
                        "ImageMagick's own quantizer, not by sampling"
                      : "an imported palette is not sampled");
    }
  }
  // The flag lives on video_opt because that is where the parsing already put it,
  // but it is honoured here too: an image is the case where you most want to fix
  // a palette once and re-use it.  Written for a DERIVED palette as well as an
  // imported one -- an earlier version of this required a non-empty
  // --palette-import, which made --palette-export silently do nothing for every
  // ordinary run, and a flag that does nothing is worse than a missing one.
  if (!opt.video_opt.palette_export.empty() &&
      !rd::ImWritePalette(*palette, opt.video_opt.palette_export, &error)) {
    std::fprintf(stderr, "error: --palette-export %s: %s\n",
                 opt.video_opt.palette_export.c_str(), error.c_str());
    delete store;
    rd::ImFree(&image);
    rd::ImShutdown();
    return 1;
  }
  if (!opt.video_opt.palette_export.empty() && !opt.quiet) {
    std::printf("palette    : written to %s (%d colours; read it back with "
                "--palette-import)\n",
                opt.video_opt.palette_export.c_str(), palette->count);
  }
  // Suppressed for an imported palette, which already reported itself above.
  // Leaving it in place printed two contradictory lines -- "16 colours imported
  // from p.txt" immediately followed by "16 colours from IM" -- and the second
  // one was the false one.
  if (!opt.quiet && opt.palette_import.empty()) {
    std::printf("palette    : %d colours from IM (associate_alpha=%s)\n",
                palette->count, palette->associate_alpha ? "yes" : "no");
    std::printf("curve      : level %d, %zu visits\n",
                rd::ComputeCurveLevel(image.width, image.height),
                rd::CurveEntryCount(
                    rd::ComputeCurveLevel(image.width, image.height)));
  }
  if (opt.dump_curve) {
    const int level = rd::ComputeCurveLevel(image.width, image.height);
    std::vector<rd::CurveStepEntry> curve(rd::CurveEntryCount(level));
    if (!curve.empty()) {
      rd::BuildCurveSequence(level, image.width, image.height, curve.data());
      std::printf("first visits:");
      for (std::size_t i = 0; i < curve.size() && i < 12; ++i) {
        std::printf(" (%d,%d)", curve[i].x, curve[i].y);
      }
      std::printf("\n");
    }
  }
  if (opt.dump_palette) {
    for (int i = 0; i < palette->count; ++i) {
      const rd::PaletteEntry& e = palette->entries[i];
      std::printf("palette %2d: %10.4f %10.4f %10.4f %10.4f\n", i, e.r, e.g, e.b,
                  e.a);
    }
  }

  // ---- colour tree -------------------------------------------------------
  // ClosestColor() searches a subtree of this tree, so it is part of the
  // observable behaviour.  Its colormap must agree with ImageMagick's; that
  // equality is a strong check that the whole tree was reproduced.
  bool grayscale = true;
  for (std::size_t i = 0; i < store->pixel_count() && grayscale; ++i) {
    const rd::RgbaF& p = store->data()[i];
    grayscale = (p.r == p.g) && (p.g == p.b);
  }
  // ImageMagick's SetAssociatedAlpha clears associate_alpha when the image has at most
  // two colours AND is in a grey colorspace, even when it carries alpha.  That decision
  // is made ONCE, in ImBuildPalette (src/rd_im.cpp), because the palette and the tree
  // must agree: with alpha counted as a channel the quantizer needs more colours to
  // separate grey levels, so a palette built under one rule and a tree built under the
  // other disagree about how many colours there are.
  //
  // It is NOT recomputed here.  An earlier version passed `image.has_alpha`
  // unconditionally, which was the bug; a second version derived the rule from
  // `r == g == b` over the decoded pixels, which fixed the tree and left the palette
  // still using IM's half-rule.  Reading it off the palette makes the two agree by
  // construction instead of by two implementations being kept in step.
  const bool associate_alpha = palette->associate_alpha;
  // The diagnostic that used to stand here is UNREACHABLE, and deliberately not deleted
  // silently. It was:
  //
  //   if (image.has_alpha && !associate_alpha) { ... "associate_alpha cleared" ... }
  //
  // It fired only when the palette declined an alpha the image had. It no longer can.
  // ImBuildPalette now tests qi->colorspace rather than source->colorspace, and it sets
  // qi->colorspace = UndefinedColorspace -- upstream's GetQuantizeInfo default, and what
  // `magick -colors N` passes -- so `grey_colorspace` is always false and the rule
  // reduces to `alpha_trait != UndefinedPixelTrait`, which is exactly `image.has_alpha`.
  // The two agree by construction rather than by two implementations being kept in step,
  // which was the whole point of reading the flag off the palette.
  //
  // WHY IT MATTERS THAT THIS IS UPSTREAM-SHAPED: SetAssociatedAlpha's grey-colorspace
  // clause is dead code upstream too, for the same reason. Setting qi->colorspace to a
  // grey value is the only thing that would revive it -- and this branch with it.
  // Re-enable it here rather than assuming the note is no longer wanted.
  (void)associate_alpha;
  std::unique_ptr<rd::ColorTree> tree(new rd::ColorTree());
  if (!opt.palette_import.empty()) {
    // Reuse the tree ImReadPalette already built FROM the palette.  Rebuilding it
    // from the image's pixels here -- which is what the line below does -- would
    // quietly undo the import: the octree would then restrict every lookup to the
    // image's own colour distribution, and the imported entries would only
    // survive where the image happens to land near them.  That is a *seed*, not
    // an import, and the two produce different pixels.
    tree = std::move(palette_tree);
  } else {
    tree->Build(store->data(), image.width, image.height, opt.colors,
                associate_alpha, grayscale);
  }
  if (!opt.quiet) {
    std::printf("tree       : depth %d, %zu nodes, %d colours%s\n", tree->depth(),
                tree->node_count(), tree->color_count(),
                grayscale ? " (greyscale)" : "");
  }
  if (tree->color_count() != palette->count) {
    std::fprintf(stderr,
                 "error: the ported tree yields %d colours but ImageMagick "
                 "produced %d\n",
                 tree->color_count(), palette->count);
    delete store;
    rd::ImFree(&image);
    rd::ImShutdown();
    return 4;
  }
  for (int i = 0; i < palette->count; ++i) {
    const rd::PaletteEntry& a = palette->entries[i];
    const rd::PaletteEntry& b = tree->colormap()[i];
    if ((a.r != b.r) || (a.g != b.g) || (a.b != b.b) || (a.a != b.a)) {
      std::fprintf(stderr,
                   "error: colormap entry %d differs (IM %.4f %.4f %.4f %.4f / "
                   "tree %.4f %.4f %.4f %.4f)\n",
                   i, a.r, a.g, a.b, a.a, b.r, b.g, b.b, b.a);
      delete store;
      rd::ImFree(&image);
      rd::ImShutdown();
      return 4;
    }
  }
  if (!opt.quiet) std::printf("colormap   : tree matches ImageMagick exactly\n");

  // ---- dither ------------------------------------------------------------
  rd::DitherParams params;
  params.colors = opt.colors;
  params.diffusion = opt.diffusion;
  params.use_cache = !opt.no_cache;
  if (opt.no_cache) {
    // Say which engines honour it, because they do not all do it the same way and the
    // flag is a diagnostic: it exists to isolate IM's memo table from the error the
    // walk introduces, so an answer that silently ignored it makes the whole
    // comparison meaningless rather than merely different.
    //
    //   cpu     RiemersmaWalkCpu allocates NO table and recomputes every lookup
    //           (rd_riemersma_cpu.cpp:334-339, :72, :86).
    //   cuda    RiemersmaWalkKernel skips the table the same way; the two now agree
    //           on what the flag means, which they did not until this was wired up.
    //   blocks  Cache-free by construction: there is no single visit order to memoise,
    //           since every block is an independent walk.  The CLI already forces
    //           params.use_cache = false for it.
    //   opencl  Same partition and the same arithmetic as blocks.
    //   approx  Cache-free by construction; the CLI already forces it false.
    if (opt.engine == rd::Engine::kCpu || opt.engine == rd::Engine::kCuda) {
      std::fprintf(stderr,
                   "[dither] --no-cache: the %s walk recomputes every palette lookup. "
                   "This is NOT ImageMagick's output and --verify will report a "
                   "mismatch; the flag isolates the memo table's contribution to the "
                   "answer.\n",
                   opt.engine == rd::Engine::kCpu ? "cpu" : "cuda");
    }
  }

  const auto t0 = std::chrono::steady_clock::now();
  if (opt.engine == rd::Engine::kCuda || opt.engine == rd::Engine::kApprox ||
      opt.engine == rd::Engine::kBlocks || opt.engine == rd::Engine::kOpenCL) {
    const bool is_approx = opt.engine == rd::Engine::kApprox;
    const bool is_blocks = opt.engine == rd::Engine::kBlocks;
    // opencl is NOT gated on CUDA.  Gating it here would be the exact bug the
    // engine exists to avoid: a machine with a working AMD or Intel GPU and no
    // CUDA would be refused for the absence of CUDA rather than for the absence
    // of a device.
    if (opt.engine == rd::Engine::kOpenCL) {
      std::string why;
      if (!rd::OpenCLAvailable(nullptr, &why)) {
        std::fprintf(stderr, "error: --engine opencl requested but %s\n",
                     why.c_str());
        const std::string& log = rd::OpenCLBuildLog();
        if (!log.empty()) {
          std::fprintf(stderr, "--- program build log ---\n%s\n", log.c_str());
        }
        delete store;
        rd::ImFree(&image);
        rd::ImShutdown();
        return 1;
      }
    } else if (!rd::CudaAvailable()) {
      std::fprintf(stderr, "error: --engine %s requested but no CUDA device\n",
                   is_approx ? "approx" : (is_blocks ? "blocks" : "cuda"));
      delete store;
      rd::ImFree(&image);
      rd::ImShutdown();
      return 1;
    }
    std::string device;
    std::string cuda_error;
    // --max-vram-mb consults CudaMaxFrames, which has no palette and no use_cache to
    // consult either (rd_riemersma_cuda.cu, subtlety 3 in that function's comment).
    // Asking it about the opencl engine is a category error: the number describes
    // CUDA's allocation set, and OpenCL's is a different one.  It is refused rather
    // than ignored, for the same reason the video-only list is -- a budget that is
    // checked against the wrong device's arithmetic is not a budget that was honoured.
    if (opt.max_vram_mb != 0 && opt.engine == rd::Engine::kOpenCL) {
      std::fprintf(stderr,
                   "error: --max-vram-mb cannot apply to --engine opencl: the budget "
                   "is checked against CudaMaxFrames, which sizes the CUDA engine's "
                   "allocations and knows nothing about the OpenCL context.  Drop it, "
                   "or use --engine blocks.\n");
      delete store;
      rd::ImFree(&image);
      rd::ImShutdown();
      return 2;
    }
    if (opt.max_vram_mb != 0) {
      const int fits = rd::CudaMaxFrames(image.width, image.height,
                                         opt.max_vram_mb * 1024ull * 1024ull);
      if (fits < 1) {
        std::fprintf(stderr,
                     "error: the image needs more than --max-vram-mb of VRAM\n");
        delete store;
        rd::ImFree(&image);
        rd::ImShutdown();
        return 1;
      }
    }
    if (is_approx) {
      // The approximate engine is cache-free by construction; flag it so the
      // AE report is not mistaken for a bit-exactness regression.
      params.use_cache = false;
      cuda_error = rd::RiemersmaApproxCuda(*palette, params, *tree, image.width,
                                           image.height, store->data(),
                                           opt.approx, &device);
      if (!opt.quiet) {
        std::printf("approx     : %d sweeps, %d taps, %s, lookup=%s\n",
                    opt.approx.iterations, opt.approx.taps,
                    opt.approx.fp64 ? "fp64" : "fp32",
                    opt.approx.use_tree ? "octree" : "linear");
      }
    } else if (is_blocks || opt.engine == rd::Engine::kOpenCL) {
      // Block-parallel: no memo table (no single visit order to memoise) and
      // every block starts from a zeroed queue.
      //
      // opencl is the same partition and the same arithmetic as blocks, reached
      // through OpenCL instead of CUDA.  That equivalence is the point: it means
      // a frame dithered on either engine is the same frame, so a machine with
      // an AMD or Intel GPU gets the same pixels the CUDA path would have given
      // it, and the two are comparable in a test rather than merely similar.
      params.use_cache = false;
      // The store holds exactly ONE frame -- rd_source.cpp sizes it
      // width*height*sizeof(RgbaF) and rd_cli.cpp allocated it from image.width/height --
      // but both engines size every host transfer by `frames`:
      //   rd_blocks_cuda.cu  bytes = width*height*frames*sizeof(float4), memcpy at :1209,
      //                       H2D at :1212, D2H at :1341
      //   rd_opencl.cpp      pix_bytes = npix*frames*16, COPY_HOST_PTR at :1113,
      //                       clEnqueueReadBuffer at :1523
      // So --frames 4 reads and writes 3 frames' worth past the allocation on each side
      // (~100 MB each way at 1080p).  The comment above this block excused the GPU
      // engines on the grounds that they "run several frames concurrently from their
      // own device buffers and never touch this constraint" -- true of the device
      // buffers, false of the host transfers, which are exactly where the overread is.
      // The plugin path a few lines below already refuses --frames and says so; this
      // does the same rather than silently handing over a pointer to 4x its memory.
      if (opt.frames != 1) {
        std::fprintf(stderr,
                     "[dither] --frames %d does not apply to --engine %s on a single "
                     "image; the store holds one frame. Running 1 frame.\n",
                     opt.frames, is_blocks ? "blocks" : "opencl");
        opt.blocks.frames = 1;
      } else {
        opt.blocks.frames = opt.frames;
      }
      if (is_blocks) {
        // The store is page-locked whenever CUDA is present, so the engine can
        // DMA straight from it and skip its own staging memcpy.
        opt.blocks.batch_pinned = store->pinned();
        cuda_error = rd::RiemersmaBlocksCuda(*palette, params, *tree, image.width,
                                             image.height, store->data(),
                                             opt.blocks, &device);
      } else {
        cuda_error = rd::RiemersmaBlocksOpencl(*palette, params, *tree, image.width,
                                              image.height, store->data(),
                                              opt.blocks, &device);
      }
      if (!opt.quiet) {
        // The divisor is `opt.blocks.block`, which the parse now floors at 16, so
        // this can no longer divide by zero.  The guard is kept anyway and is not
        // paranoia: it is a load-bearing invariant of a printf that runs AFTER the
        // dither, so a zero here costs the whole render rather than a message.  If
        // the floor is ever moved or removed, this is the line that fails.
        const int bsize = opt.blocks.block > 0 ? opt.blocks.block : 1;
        const long long walk_blocks =
            (static_cast<long long>(image.width) * image.height + bsize - 1) / bsize;
        // %lld, not %d: the expression is a long long (a 1920x1080 frame at
        // block 32 is 64800, which happens to fit, but 8K would not) and a
        // mismatched varargs type is undefined behaviour, not a warning.
        //
        // `opt.frames` here, not `opt.blocks.frames`: the two are set from each
        // other above and are equal on every path that reaches this line, and using
        // the field the engine actually received is the one that cannot drift.
        std::printf("blocks     : %d positions/block, %d frame%s, %lld block%s/walk\n",
                    bsize, opt.blocks.frames, opt.blocks.frames == 1 ? "" : "s",
                    walk_blocks, walk_blocks == 1 ? "" : "s");
      }
    } else {
      // Same reasoning as the blocks branch above, and stronger: the engine takes ONE
      // host frame, uploads one frame and downloads one frame, so N distinct frames are
      // not even representable in this API.  The extra walks are not merely discarded,
      // they are provably IDENTICAL -- every input is shared and read-only (srcbuf,
      // palette, nodes, weights, cache size) and the only per-frame state is reset the
      // same way each launch.  So at --frames 16 this wastes 15 walks, 506.25 MiB of
      // `dst`, and up to 1 GiB of cache, for byte-identical output.
      if (opt.frames != 1) {
        std::fprintf(stderr,
                     "[dither] --frames %d does not apply to --engine cuda on a single "
                     "image; the engine uploads one frame and downloads one frame, so "
                     "the walks would be identical. Running 1 frame.\n",
                     opt.frames);
        opt.frames = 1;
      }
      cuda_error = rd::RiemersmaWalkCuda(*palette, params, *tree, image.width,
                                         image.height, store->data(), opt.frames,
                                         &device);
    }
    if (!cuda_error.empty()) {
      std::fprintf(stderr, "error: %s\n", cuda_error.c_str());
      delete store;
      rd::ImFree(&image);
      rd::ImShutdown();
      return 1;
    }
    if (!opt.quiet) std::printf("device     : %s\n", device.c_str());
  } else if (const rd::DitherAlgorithm* algo = rd::FindDither(opt.dither.c_str())) {
    // A plugin runs on the host, whatever --engine says.  That is not a limitation
    // being hidden: the GPU engines are kernel-launch paths, not function pointers,
    // because the fast ones keep the error queue in registers across a thread block.
    // A function-pointer interface cannot carry that, and copying the state to and
    // from device memory per call would give back the reason they are fast.  A
    // plugin is for new *algorithms*; see include/rd_plugin.h.
    rd::DitherJob job;
    job.palette = palette.get();
    job.width = image.width;
    job.height = image.height;
    job.diffusion = opt.diffusion;
    // The store holds ONE image (the single-image path never allocates more), so
    // --frames has to be forced to 1 for a plugin.  The GPU engines run several
    // frames concurrently from their own device buffers and never touch this
    // constraint; a plugin is handed a plain host pointer to a plain host buffer.
    // Silently handing a plugin frames=1 when the user asked for --frames 4 would
    // be a lie about the work done, so it is a reported adjustment instead.
    if (opt.frames != 1) {
      std::fprintf(stderr,
                   "[dither] --frames %d does not apply to plugin '%s'; running 1 "
                   "frame.\n",
                   opt.frames, algo->name);
    }
    job.frames = 1;
    if (store->pixel_count() < image.width * image.height) {
      std::fprintf(stderr,
                   "error: internal --dither dispatch: store holds %zu pixels, "
                   "image is %zux%zu\n",
                   store->pixel_count(), image.width, image.height);
      delete store;
      rd::ImFree(&image);
      rd::ImShutdown();
      return 1;
    }
    const rd::DitherResult r = algo->run(job, store->data());
    if (!r.error.empty()) {
      std::fprintf(stderr, "error: %s\n", r.error.c_str());
      delete store;
      rd::ImFree(&image);
      rd::ImShutdown();
      return 1;
    }
    if (!opt.quiet) {
      std::printf("dither     : %s (%s)\n", algo->name, algo->description);
    }
  } else {
    rd::RiemersmaWalkCpu(*palette, params, *tree, image.width, image.height,
                         store->data(), nullptr);
  }
  const double dither_ms = ElapsedMs(t0);
  if (!opt.quiet) {
    // Walks ACTUALLY performed, which is not `opt.frames`.
    //
    // Every branch above either honours --frames or declines it, and they record the
    // outcome in different fields: the cuda branch clamps `opt.frames` itself, the
    // blocks/opencl branch clamps `opt.blocks.frames` and leaves `opt.frames` at the
    // user's number, and the plugin and cpu branches ignore the flag entirely.  This
    // line used `opt.frames`, so:
    //
    //   --engine blocks --frames 4 in.png out.png
    //       printed "4 walks" and 4x the Mpixel/s, having run ONE.  The store holds a
    //       single frame (rd_source.cpp sizes it width*height*sizeof(RgbaF)) and the
    //       branch above says so on stderr, but the throughput line then reported the
    //       number the flag asked for rather than the work done -- so a user comparing
    //       two runs could make one look 4x faster from a flag that changed nothing.
    //   --engine cuda --frames 4 was already right, because that branch assigns
    //       opt.frames = 1.
    //   --engine cpu --frames 4 printed "4 walks" and 4x Mpixel/s for a strictly
    //       sequential walk.  This is the worst of the three: the cpu engine takes no
    //       frame count at all, so nothing warned.
    //
    // One variable, assigned where the answer is known, so the reported count cannot
    // disagree with the call that was made.
    const int walks = opt.engine == rd::Engine::kBlocks ||
                              opt.engine == rd::Engine::kOpenCL
                          ? opt.blocks.frames
                          : ((opt.engine == rd::Engine::kCpu ||
                              opt.engine == rd::Engine::kApprox)
                                 ? 1
                                 : opt.frames);
    // Aggregate throughput: an engine may run several independent walks
    // concurrently, so reporting only one frame's pixel count would hide the
    // scaling that frame batching actually buys.  Divide by the same number the walk
    // count reports, or the two figures on this one line contradict each other.
    const double walked =
        static_cast<double>(image.width) * image.height * walks;
    // walked pixels / (ms -> s) / 1e6 == Mpixel/s
    const char* engine_name = opt.engine == rd::Engine::kCuda    ? "cuda"
                              : opt.engine == rd::Engine::kApprox ? "approx"
                              : opt.engine == rd::Engine::kBlocks ? "blocks"
                              : opt.engine == rd::Engine::kOpenCL ? "opencl"
                                                                  : "cpu";
    std::printf("dither     : %.2f ms (%d walk%s, %.2f Mpixel/s aggregate, %s engine)\n",
                dither_ms, walks, walks == 1 ? "" : "s",
                walked / (dither_ms * 1000.0), engine_name);
  }

  // ---- write -------------------------------------------------------------
  // Checked here rather than at parse time because the answer comes from ImageMagick,
  // which is not started until the image is loaded.  Checked at all because an unknown
  // coder name is not an error ImageMagick reports: SetImageInfo falls back to the
  // output path's extension, so `--format BOGUS` used to exit 0 having quietly written
  // a PNG.  A file that exists, is the wrong format, and reports success.
  if (!rd::ImFormatKnown(opt.format, &error)) {
    std::fprintf(stderr, "error: --format %s\n", error.c_str());
    delete store;
    rd::ImFree(&image);
    rd::ImShutdown();
    return 2;
  }
  if (!rd::ImStore(image, *store, opt.output, opt.format, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    delete store;
    rd::ImFree(&image);
    rd::ImShutdown();
    return 1;
  }

  // ---- verify ------------------------------------------------------------
  int status = 0;
  if (opt.verify) {
    rd::PixelStore* reference = rd::PixelStore::Create(
        image.width, image.height, max_ram_bytes, &error);
    if (reference == nullptr ||
        !rd::ImReferenceDither(image, opt.colors, reference, &error)) {
      std::fprintf(stderr, "error: verification failed: %s\n", error.c_str());
      delete reference;
      status = 1;
    } else {
      const rd::DiffResult diff =
          rd::CompareStores(*store, *reference, image.has_alpha);
      std::printf(
          "verify     : AE=%zu/%zu pixels, max channel delta=%d, RMSE=%.8f -> %s\n",
          diff.differing_pixels, diff.total_pixels, diff.max_channel_delta,
          diff.rmse, !diff.comparable ? "INCOMPARABLE (geometry mismatch)"
                                      : (diff.differing_pixels == 0 ? "BIT-EXACT"
                                                                    : "MISMATCH"));
      if (!diff.comparable || diff.differing_pixels != 0) status = 3;
    }

    // The comparison above proves the STORE matches ImageMagick.  It says nothing
    // about the file on disk, and that gap hid a real defect for a long time: the
    // writer inherited the input's coder, so dithering a JPEG to a path called
    // .png wrote JPEG bytes (FF D8 FF) and the 16-colour store came back as 93,377
    // colours after lossy re-compression -- with --verify still reporting AE=0.
    //
    // So read the written file back and compare THAT against ImageMagick.  A store
    // that is right and a file that is wrong is precisely the case this catches,
    // and it is the claim --verify actually makes to the user.
    if (status == 0) {
      rd::LoadedImage written;
      rd::PixelStore* roundtrip = nullptr;
      std::string rt_error;
      if (rd::ImLoad(opt.output, &written, &rt_error)) {
        roundtrip = rd::PixelStore::Create(written.width, written.height,
                                           max_ram_bytes, &rt_error);
        if (roundtrip != nullptr && !rd::ImExtract(written, roundtrip, &rt_error)) {
          delete roundtrip;
          roundtrip = nullptr;
        }
      } else {
        std::printf("verify     : could not re-read the written file '%s': %s\n",
                    opt.output.c_str(), rt_error.c_str());
        status = 1;
      }
      if (roundtrip != nullptr) {
        const rd::DiffResult fd = rd::CompareStores(
            *roundtrip, *reference, written.has_alpha);
        // Compared at 8-bit precision, which is what any PNG or JPEG codec
        // actually guarantees, rather than exactly as the store comparison does.
        //
        // Exact comparison here reports a mismatch on a correct file: writing Q16
        // and reading it back can differ by 1 in 65535 from the codec's rounding,
        // and that is not a defect. Measured on the 64x64 smoke image: max channel
        // delta 1, RMSE 0.00000547, 1016 of 4096 pixels. Treating that as failure
        // would fail every run and train people to ignore the check.
        //
        // One 8-bit step in Q16 code values is 257, so a delta under that is
        // invisible in any real output and cannot be an encoder picking the wrong
        // coder -- that mistake shows up as thousands of differing pixels, which is
        // exactly what the JPEG-as-PNG bug produced.
        constexpr int kOne8BitStep = 257;
        // An incomparable pair is not a pass.  `roundtrip` is sized from what ImLoad
        // made of the written file and `reference` from the image, so a coder or file
        // that decodes to a different size used to yield max_channel_delta 0 -- printed
        // as "AE=0/0 pixels ... BIT-EXACT (8-bit)" and exit 0, for a file of entirely
        // the wrong dimensions.  That is the class of defect this second comparison was
        // added to catch, so it must not be one of the ways it passes.
        const bool file_ok = fd.comparable && fd.max_channel_delta < kOne8BitStep;
        std::printf(
            "verify file: AE=%zu/%zu pixels, max channel delta=%d, RMSE=%.8f -> %s\n",
            fd.differing_pixels, fd.total_pixels, fd.max_channel_delta, fd.rmse,
            !fd.comparable ? "INCOMPARABLE (geometry mismatch)"
                            : (file_ok ? "BIT-EXACT (8-bit)" : "MISMATCH"));
        if (!file_ok) {
          std::printf(
              "verify     : the store was correct but the written file is not.\n"
              "              That is an encoder problem, not a dither problem --\n"
              "              check that the output path's extension matches the coder.\n");
          status = 3;
        }
        rd::ImFree(&written);
        delete roundtrip;
      }
    }
    delete reference;
  }

  if (!opt.quiet) std::printf("output     : %s\n", opt.output.c_str());

  delete store;
  rd::ImFree(&image);
  rd::ImShutdown();
  return status;
}

