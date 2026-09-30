// SPDX-License-Identifier: GPL-3.0-or-later
// rd_video.cpp -- ffmpeg-driven decode/encode around the GPU block engine.
#include "rd_video.h"

#include "rd_opencl.h"
#include "rd_progress.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// No CUDA API is called from this file -- rd_video.cpp asks CudaAvailable() whether
// to request NVDEC and CudaAllocPinned() for staging, and both live in the .cu files,
// with no-CUDA definitions in rd_cuda_stub.cpp.  So this header was never needed here,
// and including it unconditionally meant `build.cmd --no-cuda` failed to COMPILE with
// C1083, before it could even reach the link stage.
#ifdef RD_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rd {
namespace {

// ---------------------------------------------------------------------------
// Child process with a redirected pipe.  ReadFile/WriteFile are used directly
// rather than FILE* because the payload is binary and the CRT would mangle it.
// ---------------------------------------------------------------------------
class Child {
 public:
  ~Child() { Close(); }

  // Starts `exe` with `args`.  If `capture_stdout` the child's stdout is
  // readable via Read(); if `feed_stdin` the child's stdin is writable via
  // Write().  Either, both, or neither.
  //
  // `role` is a short label used only by RD_TRACE.  It exists because the trace used
  // to live at the call sites, which meant it covered the palette decoder and
  // nothing else -- so the decode and encode command lines were invisible, and a
  // question about them (are the -thread caps even reaching ffmpeg?) could only be
  // answered by reimplementing the pipeline by hand.  Tracing here means a new
  // spawn site cannot be added without being traced.
  bool Start(const std::string& exe, const std::string& args, bool capture_stdout,
             bool feed_stdin, std::string* error, const char* role = "child") {
    if (std::getenv("RD_TRACE") != nullptr) {
      std::fprintf(stderr, "[spawn:%s] %s %s\n", role, exe.c_str(), args.c_str());
    }
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = nullptr;
    sa.bInheritHandle = TRUE;

    // CreatePipe(&readEnd, &writeEnd).  The child needs the *write* end for
    // stdout (we read it) and the *read* end for stdin (we write it), so the
    // argument order differs between the two directions.
    HANDLE parent_read = INVALID_HANDLE_VALUE;   // we read the child's stdout
    HANDLE child_write = INVALID_HANDLE_VALUE;   // child's stdout
    HANDLE child_read = INVALID_HANDLE_VALUE;    // child's stdin
    HANDLE parent_write = INVALID_HANDLE_VALUE;  // we write the child's stdin

    // 16 MiB, not 1.  The writer thread hands 33 MB frames to the encoder, and
    // with a 1 MiB buffer it blocks on backpressure every frame, so the measured
    // "pipe" stage was the context switching rather than the encoder.  The buffer
    // is the handoff window between our three stages, so it is sized like one.
    // 16 MiB, not 1.  The writer hands 33 MB frames to the encoder, and with a 1 MiB
    // buffer it blocks on backpressure every frame, so the measured "pipe" stage was
    // the context switching rather than the encoder.  The buffer is the handoff
    // window between our three stages, so it is sized like one.
    //
    // Raising it to 64 MiB was measured and made no difference (33.9 vs 33.8 fps), so
    // the reader's 11.8 s is not transport-bound -- it is ffmpeg converting and
    // writing 20 GB of rgba64le, which no buffer size can help.
    constexpr DWORD kPipeBuffer = 16u << 20;
    if (capture_stdout &&
        !CreatePipe(&parent_read, &child_write, &sa, kPipeBuffer)) {
      *error = "CreatePipe failed";
      return false;
    }
    if (feed_stdin && !CreatePipe(&child_read, &parent_write, &sa, kPipeBuffer)) {
      *error = "CreatePipe failed";
      return false;
    }

    // NUL device for whichever direction we do not drive, so ffmpeg never blocks
    // waiting on a terminal.  It must stay inheritable: the child needs a valid
    // stdio handle for it.
    HANDLE nul = CreateFileA("NUL", GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING,
                             0, nullptr);
    if (nul == INVALID_HANDLE_VALUE) {
      *error = "cannot open NUL";
      return false;
    }
    // Only the *parent's* ends must be non-inheritable; if the parent still holds
    // the write end of stdout the pipe never reports EOF.
    if (parent_read != INVALID_HANDLE_VALUE) {
      SetHandleInformation(parent_read, HANDLE_FLAG_INHERIT, 0);
    }
    if (parent_write != INVALID_HANDLE_VALUE) {
      SetHandleInformation(parent_write, HANDLE_FLAG_INHERIT, 0);
    }

    STARTUPINFOA si;
    std::memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = feed_stdin ? child_read : nul;
    si.hStdOutput = capture_stdout ? child_write : nul;
    si.hStdError = nul;

    std::string command = "\"" + exe + "\" " + args;
    std::vector<char> mutable_cmd(command.begin(), command.end());
    mutable_cmd.push_back('\0');

    PROCESS_INFORMATION pi;
    std::memset(&pi, 0, sizeof(pi));
    if (!CreateProcessA(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE, 0,
                        nullptr, nullptr, &si, &pi)) {
      *error = "CreateProcess failed for " + exe;
      return false;
    }
    process_ = pi.hProcess;
    thread_handle_ = pi.hThread;
    CloseHandle(pi.hThread);
    thread_handle_ = nullptr;

    // The parent must drop the child's ends or the pipe never reports EOF.
    if (capture_stdout) {
      CloseHandle(child_write);
      read_handle_ = parent_read;
    }
    if (feed_stdin) {
      CloseHandle(child_read);
      write_handle_ = parent_write;
    }
    CloseHandle(nul);
    return true;
  }

  // Reads exactly `bytes` unless the stream ends.  Returns bytes read.
  std::size_t Read(void* dst, std::size_t bytes) {
    std::size_t done = 0;
    auto* out = static_cast<char*>(dst);
    while (done < bytes) {
      DWORD got = 0;
      const DWORD want = static_cast<DWORD>(
          (bytes - done) > 0x1000000u ? 0x1000000u : (bytes - done));
      if (!ReadFile(read_handle_, out + done, want, &got, nullptr)) {
        if (getenv("RD_TRACE") != nullptr) {
          std::fprintf(stderr, "[video] ReadFile failed after %zu bytes, err=%lu\n",
                       done, static_cast<unsigned long>(GetLastError()));
        }
        break;
      }
      if (got == 0) break;  // EOF
      done += got;
    }
    return done;
  }

  bool Write(const void* src, std::size_t bytes) {
    std::size_t done = 0;
    const auto* in = static_cast<const char*>(src);
    while (done < bytes) {
      DWORD put = 0;
      const DWORD want = static_cast<DWORD>(
          (bytes - done) > 0x1000000u ? 0x1000000u : (bytes - done));
      if (!WriteFile(write_handle_, in + done, want, &put, nullptr)) return false;
      if (put == 0) return false;
      done += put;
    }
    return true;
  }

  void CloseStdin() {
    if (write_handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(write_handle_);
      write_handle_ = INVALID_HANDLE_VALUE;
    }
  }

  // Waits for exit; returns the exit code (or -1).
  int Wait(unsigned timeout_ms = 120000) {
    if (process_ == INVALID_HANDLE_VALUE) return -1;
    if (WaitForSingleObject(process_, timeout_ms) != WAIT_OBJECT_0) return -1;
    DWORD code = 0;
    if (!GetExitCodeProcess(process_, &code)) return -1;
    return static_cast<int>(code);
  }

  // Terminates the child if it is still running, then releases the handles.
  //
  // This is what makes the destructor safe: closing handles alone leaves the
  // process alive and orphaned, so an error return or a Ctrl-C on a three-hour
  // job used to leak ffmpeg processes still holding the output file.  A child
  // that has already exited and been waited on is left alone, so the normal path
  // still gets ffmpeg's real exit code and its chance to finalise the container.
  void Close() {
    if (process_ != nullptr) {
      if (WaitForSingleObject(process_, 0) == WAIT_TIMEOUT) {
        TerminateProcess(process_, 1);
        (void)WaitForSingleObject(process_, 2000);
      }
      CloseHandle(process_);
      process_ = nullptr;
    }
    if (thread_handle_ != nullptr) {
      CloseHandle(thread_handle_);
      thread_handle_ = nullptr;
    }
    if (write_handle_ != INVALID_HANDLE_VALUE) CloseHandle(write_handle_);
    if (read_handle_ != INVALID_HANDLE_VALUE) CloseHandle(read_handle_);
    write_handle_ = INVALID_HANDLE_VALUE;
    read_handle_ = INVALID_HANDLE_VALUE;
  }

 private:
  HANDLE process_ = nullptr;
  HANDLE thread_handle_ = nullptr;
  HANDLE read_handle_ = INVALID_HANDLE_VALUE;
  HANDLE write_handle_ = INVALID_HANDLE_VALUE;
};

double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool ToolOnPath(const char* name, std::string* full) {
  char buf[MAX_PATH + 1] = {0};
  const DWORD n = SearchPathA(nullptr, name, ".exe", MAX_PATH, buf, nullptr);
  if (n == 0 || n > MAX_PATH) return false;
  *full = buf;
  return true;
}

// rgba64le (or rgb48le) bytes -> RgbaF, with an optional box filter when stride > 1.
// `channels` is 3 for rgb48le, which the decoder is asked for when the source has no
// alpha; the same 65535 is synthesised that the device's gather kernel synthesises, so
// the host and device paths agree.
void RawToFloats(const std::uint16_t* raw, RgbaF* dst, std::size_t pixels,
                 int channels) {
  for (std::size_t i = 0; i < pixels; ++i) {
    const std::uint16_t* p = raw + i * static_cast<std::size_t>(channels);
    dst[i].r = static_cast<float>(p[0]);
    dst[i].g = static_cast<float>(p[1]);
    dst[i].b = static_cast<float>(p[2]);
    dst[i].a = channels == 4 ? static_cast<float>(p[3])
                             : static_cast<float>(65535.0);
  }
}

// The same conversion split across cores, by frame.
//
// This is the reader's dominant cost and it was entirely single-threaded.  Measured
// on this machine: ffmpeg needs ~5.0 s to decode 605 1080p frames, but the reader's
// stage was taking ~14.7 s, so roughly 10 s of it was this loop -- two billion
// uint16->float conversions plus 20 GB of pipe traffic, on one thread.  That is why
// the GPU worker sat idle 47% of its stage time: it was not waiting on the GPU or on
// queue depth (raising the depth from 3 to 8 changed nothing), it was waiting for
// batches this loop had not produced yet.
//
// Same arithmetic per pixel, so the bytes handed to the dither are unchanged.
void RawToFloatsParallel(const std::uint16_t* raw, RgbaF* dst, std::size_t pixels,
                         int frames, int threads, int channels) {
  if (frames <= 1 || threads <= 1) {
    RawToFloats(raw, dst, static_cast<std::size_t>(frames) * pixels, channels);
    return;
  }
  const int workers = std::max(1, std::min(threads, frames));
  if (workers == 1) {
    RawToFloats(raw, dst, static_cast<std::size_t>(frames) * pixels, channels);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(workers) - 1);
  const int chunk = (frames + workers - 1) / workers;
  for (int w = 1; w < workers; ++w) {
    const int f0 = w * chunk;
    if (f0 >= frames) break;
    const int f1 = std::min(frames, f0 + chunk);
    pool.emplace_back([=]() {
      RawToFloats(raw + static_cast<std::size_t>(f0) * pixels *
                                 static_cast<std::size_t>(channels),
                  dst + static_cast<std::size_t>(f0) * pixels,
                  static_cast<std::size_t>(f1 - f0) * pixels, channels);
    });
  }
  const int f1 = std::min(frames, chunk);
  RawToFloats(raw, dst, static_cast<std::size_t>(f1) * pixels, channels);
  for (std::thread& t : pool) t.join();
}

void FloatsToRaw(const RgbaF* src, std::uint16_t* dst, std::size_t pixels) {
  for (std::size_t i = 0; i < pixels; ++i) {
    dst[4 * i + 0] = static_cast<std::uint16_t>(src[i].r);
    dst[4 * i + 1] = static_cast<std::uint16_t>(src[i].g);
    dst[4 * i + 2] = static_cast<std::uint16_t>(src[i].b);
    dst[4 * i + 3] = static_cast<std::uint16_t>(src[i].a);
  }
}

// float4 RGBA -> planar 8-bit 4:4:4, 3 bytes per pixel: Y plane, then U, then V.
//
// This is the host half of a conversion the GPU engines already do on the device, and
// its absence is a shipped bug rather than a missing feature.  `--no-gpu` used to call
// FloatsToRaw, which produces rgba64le at 8 bytes per pixel, and the writer was
// measuring the frame at 3 bytes per pixel because `out_yuv444` defaults to true and
// spawning the encoder with `-pix_fmt yuv444p`.  So the host pushed 8 bytes into a
// pipe declared as 3: the encoder read rgba64le bytes as planar YUV, the stream
// desynchronised, a red band came out green, and two runs of the same command
// differed in 48.7% of all bytes -- at one worker as well, so not a race.  Guarded by
// tools\\probe-video-determinism.ps1, which is where the numbers are.
//
// The arithmetic is a transliteration of `d_rgb_to_yuv444` in rd_blocks_cuda.cu (and
// of `scatter_yuv444` in rd_opencl.cpp, which is a transliteration of that), NOT a
// fresh derivation.  It has to match them bit for bit, because the whole claim the
// project rests on is that the three engines agree; a host conversion that merely
// looked right would put them 0.5% of pixels apart, which is the class of difference
// nobody can see and everybody is annoyed by.
//
// The cast order matters and matches the device: truncate the float to int FIRST,
// then shift by 8.  Shifting the float and converting afterwards is not the same
// operation, and would round differently on the values that matter.
void FloatsToYuv444(const RgbaF* src, unsigned char* dst, std::size_t pixels) {
  for (std::size_t i = 0; i < pixels; ++i) {
    const int r8 = static_cast<int>(src[i].r) >> 8;
    const int g8 = static_cast<int>(src[i].g) >> 8;
    const int b8 = static_cast<int>(src[i].b) >> 8;
    const int yv = ((66 * r8 + 129 * g8 + 25 * b8 + 128) >> 8) + 16;
    const int uv = ((-38 * r8 - 74 * g8 + 112 * b8 + 128) >> 8) + 128;
    const int vv = ((112 * r8 - 94 * g8 - 18 * b8 + 128) >> 8) + 128;
    dst[i] = static_cast<unsigned char>(yv < 0 ? 0 : (yv > 255 ? 255 : yv));
    dst[pixels + i] = static_cast<unsigned char>(uv < 0 ? 0 : (uv > 255 ? 255 : uv));
    dst[2 * pixels + i] =
        static_cast<unsigned char>(vv < 0 ? 0 : (vv > 255 ? 255 : vv));
  }
}

void FloatsToYuv444Parallel(const RgbaF* src, unsigned char* dst,
                            std::size_t pixels, int frames, int threads) {
  if (frames <= 1 || threads <= 1) {
    FloatsToYuv444(src, dst, pixels * static_cast<std::size_t>(frames));
    return;
  }
  const int workers = std::max(1, std::min(threads, frames));
  if (workers == 1) {
    FloatsToYuv444(src, dst, pixels * static_cast<std::size_t>(frames));
    return;
  }
  // Split by FRAME, not by byte.  A frame is three contiguous planes, so a
  // frame-aligned partition is what gives each thread whole planes; splitting the
  // byte range instead would hand one thread part of the Y plane and another part of
  // the U plane, which is the same class of mistake as the one this function exists to
  // fix.
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(workers) - 1);
  const int chunk = (frames + workers - 1) / workers;
  const int first = std::min(frames, chunk);
  for (int w = 1; w < workers; ++w) {
    const int f0 = w * chunk;
    if (f0 >= frames) break;
    const int f1 = std::min(frames, f0 + chunk);
    const std::size_t count = static_cast<std::size_t>(f1 - f0) * pixels;
    pool.emplace_back([=]() {
      FloatsToYuv444(src + static_cast<std::size_t>(f0) * pixels,
                     dst + static_cast<std::size_t>(f0) * pixels * 3, count);
    });
  }
  FloatsToYuv444(src, dst, pixels * static_cast<std::size_t>(first));
  for (std::thread& t : pool) t.join();
}

std::string Quote(const std::string& s) { return "\"" + s + "\""; }

}  // namespace

// Mean HSV saturation of the palette, as a fraction.  A montage of many scenes
// drives this down: the octree's N centroids then have to span the whole gamut,
// so they land on desaturated mid-tones and the result reads as dull and grey.
// This makes that measurable instead of a matter of opinion.  Public because
// --palette-only exists to tune it.
double MeanPaletteSaturation(const Palette& palette) {
  if (palette.count <= 0) return 0.0;
  double total = 0.0;
  for (int i = 0; i < palette.count; ++i) {
    // Palette channels are Q16 (0..65535); saturation is scale-free, so the
    // normalisation cancels.
    const double r = palette.entries[i].r;
    const double g = palette.entries[i].g;
    const double b = palette.entries[i].b;
    const double hi = std::max(r, std::max(g, b));
    if (hi <= 0.0) continue;
    const double lo = std::min(r, std::min(g, b));
    total += (hi - lo) / hi;
  }
  return total / palette.count;
}

// How many palette entries are effectively neutral.  Mean saturation cannot see
// the failure this exists to catch: a palette of three greys and thirteen vivid
// colours still averages high saturation, but three of sixteen slots are gone.
int CountNeutralPaletteEntries(const Palette& palette, double threshold) {
  int n = 0;
  for (int i = 0; i < palette.count; ++i) {
    const double r = palette.entries[i].r;
    const double g = palette.entries[i].g;
    const double b = palette.entries[i].b;
    const double hi = std::max(r, std::max(g, b));
    const double lo = std::min(r, std::min(g, b));
    if (hi <= 0.0 || (hi - lo) / hi < threshold) ++n;
  }
  return n;
}

// ---- Interruption ---------------------------------------------------------
//
// A three-hour render is very likely to be interrupted, and the two things that
// must not happen are leaking the ffmpeg children and leaving a half-written
// output file presented as finished.  The handler itself does nothing but set a
// flag: doing real work in a console control handler is unsafe, so the pipeline
// threads notice the flag at their next checkpoint and unwind through the normal
// shutdown path, which closes the encoder's stdin and lets ffmpeg finalise.
std::atomic<bool> g_interrupted{false};

extern "C" BOOL WINAPI ConsoleHandler(DWORD type) {
  if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT ||
      type == CTRL_CLOSE_EVENT || type == CTRL_LOGOFF_EVENT ||
      type == CTRL_SHUTDOWN_EVENT) {
    g_interrupted.store(true);
    return TRUE;  // handled: do not fall through to the default handler
  }
  return FALSE;
}

void InstallInterruptHandler() {
  SetConsoleCtrlHandler(ConsoleHandler, TRUE);
}

void RemoveInterruptHandler() {
  SetConsoleCtrlHandler(ConsoleHandler, FALSE);
}

bool Interrupted() { return g_interrupted.load(); }

// Quotes a path for a command line.  Exposed because the segment-joining step in
// the CLI needs the same escaping the pipeline uses, and getting it subtly
// different is how a path with a space in it becomes a wrong path.
std::string VideoQuotePath(const std::string& s) { return "\"" + s + "\""; }

// Runs an ffmpeg invocation to completion with no pipes, for post-processing such
// as joining segments.  Returns false with ffmpeg's exit code in `error`.
bool VideoRunTool(const std::string& tool, const std::string& args,
                  int timeout_ms, int* exit_code, std::string* error) {
  Child child;
  if (!child.Start(tool, args, false, false, error, "ffprobe-version")) return false;
  const int code = child.Wait(static_cast<unsigned>(timeout_ms));
  if (code == -1) {
    *error = "timed out waiting for " + tool;
    return false;
  }
  if (exit_code != nullptr) *exit_code = code;
  if (code != 0) {
    *error = tool + " exited with " + std::to_string(code);
    return false;
  }
  return true;
}

// float4 -> rgba64le.  This is pure host work proportional to pixels, and it was
// measured at 9.9 ms/frame -- about a quarter of the whole encode stage, and all
// of it on the single writer thread, which is also the thread that pushes the
// encoder's pipe.  The batch's frames are independent, so it is split across
// cores.  Same arithmetic per pixel as the serial version (still float -> uint16
// truncation via C-style cast), so the output is bit-identical; only the thread it
// happens on changes.
void FloatsToRawParallel(const RgbaF* src, std::uint16_t* dst, std::size_t pixels,
                         int frames, int threads) {
  if (frames <= 1 || threads <= 1) {
    FloatsToRaw(src, dst, pixels);
    return;
  }
  const int workers = std::max(1, std::min(threads, frames));
  if (workers == 1) {
    FloatsToRaw(src, dst, pixels);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(workers) - 1);
  const int chunk = (frames + workers - 1) / workers;
  for (int w = 1; w < workers; ++w) {
    const int f0 = w * chunk;
    if (f0 >= frames) break;
    const int f1 = std::min(frames, f0 + chunk);
    pool.emplace_back([=]() {
      FloatsToRaw(src + static_cast<std::size_t>(f0) * pixels,
                  dst + static_cast<std::size_t>(f0) * pixels * 4,
                  static_cast<std::size_t>(f1 - f0) * pixels);
    });
  }
  // The calling thread takes the first chunk rather than idling.
  const int f1 = std::min(frames, chunk);
  FloatsToRaw(src, dst, static_cast<std::size_t>(f1) * pixels);
  for (std::thread& t : pool) t.join();
}

bool VideoFindTools(std::string* ffmpeg, std::string* ffprobe,
                    std::string* error) {
  // Either pointer may be null, meaning "I already have this one".
  if (ffmpeg != nullptr) {
    if (const char* e = std::getenv("FFMPEG")) {
      *ffmpeg = e;
    } else if (!ToolOnPath("ffmpeg", ffmpeg)) {
      *error = "ffmpeg not found on PATH (or set FFMPEG)";
      return false;
    }
  }
  if (ffprobe != nullptr) {
    if (const char* e = std::getenv("FFPROBE")) {
      *ffprobe = e;
    } else if (!ToolOnPath("ffprobe", ffprobe)) {
      *error = "ffprobe not found on PATH (or set FFPROBE)";
      return false;
    }
  }
  return true;
}

// ---- Frame-rate mode -------------------------------------------------------
//
// Turning the frame-rate conversion off is REQUIRED, not cosmetic, and the reason
// is at each of the three call sites below: a `select` filter leaves timestamp
// gaps, the default conversion re-fills them, and `-frames:v N` then yields the
// first N frames in time order rather than the N selected ones.  On grey-heavy
// footage that silently produced an all-grey palette.
//
// It used to be spelled `-vsync 0`, which ffmpeg accepted as a *global* option,
// so it could sit in front of `-i` where all three of these commands had it.
// ffmpeg 9 removed the option outright -- not deprecated, removed: the command
// dies during option splitting with "Unrecognized option 'vsync'" and the pipe
// gets zero bytes, which surfaced as the palette stage reporting that ffmpeg
// produced no frames.
//
// The replacement is NOT a textual substitution, which is the part worth being
// careful about.  `-fps_mode passthrough` sets the identical VSYNC_PASSTHROUGH
// value (ffmpeg_opt.c, parse_and_set_vsync) but is declared
// OPT_OUTPUT|OPT_PERSTREAM, so putting it where -vsync 0 used to sit makes ffmpeg
// reject the whole command with "cannot be applied to input url".  The spelling
// and the position both have to change together.
//
// Hence two fragments rather than one, and a probe to choose between them.
// Output bytes are unaffected: with 10 frames of tests/clip1920.mp4 as rgba64le,
// `-fps_mode passthrough` gives md5 1FD40474DE85816DECB2CDCBFCEF58F5, the same
// value the old `-vsync 0` spelling was recorded at.
//
// The probe is `-h long` and not `-h`, because fps_mode is OPT_EXPERT and the
// short listing hides expert options entirely.
struct FpsModeArgs {
  const char* input;   // goes before -i; empty on ffmpeg >= 5.1
  const char* output;  // goes after -i; empty before ffmpeg 5.1
};

FpsModeArgs ProbeFpsMode(const std::string& ffmpeg) {
  FpsModeArgs result{"", ""};
  Child child;
  std::string ignored;
  if (child.Start(ffmpeg, "-hide_banner -h long", /*capture_stdout=*/true,
                  /*feed_stdin=*/false, &ignored, "ffmpeg-caps")) {
    std::string text;
    char buffer[4096];
    for (;;) {
      const std::size_t got = child.Read(buffer, sizeof(buffer));
      if (got == 0) break;
      text.append(buffer, got);
    }
    child.Wait();
    if (text.find("-fps_mode") != std::string::npos) {
      result.output = " -fps_mode passthrough";
    } else if (text.find("-vsync") != std::string::npos) {
      result.input = " -vsync 0";
    } else {
      // Falling back the wrong way is a *silent* wrong-output bug rather than a
      // loud failure, which is the worst kind: with no frame-rate control the
      // sampled frames come back in the wrong order and the palette degrades
      // quietly.  So say so rather than guess.
      std::fprintf(stderr,
                   "[video] warning: this ffmpeg advertises neither -fps_mode nor "
                   "-vsync, so frame timing cannot be passed through; sampled "
                   "frames may come back in the wrong order.  Palette results will "
                   "not be trustworthy.\n");
      result.output = " -fps_mode passthrough";
    }
  } else {
    std::fprintf(stderr,
                 "[video] warning: could not query ffmpeg's option list (%s); "
                 "assuming -fps_mode passthrough.\n", ignored.c_str());
    result.output = " -fps_mode passthrough";
  }
  return result;
}

// Cached for the process: all three decoders ask, the answer cannot change while
// rdither runs, and the probe costs a process spawn.
const FpsModeArgs& FpsMode(const std::string& ffmpeg) {
  static FpsModeArgs cached;
  static std::once_flag once;
  std::call_once(once, [&] { cached = ProbeFpsMode(ffmpeg); });
  return cached;
}

// ---------------------------------------------------------------------------
// Variable frame rate: measuring it, and carrying it through
// ---------------------------------------------------------------------------
//
// THE BUG THIS FIXES.  The encoder is fed `-r <info.fps>`, and info.fps comes from
// r_frame_rate.  For a constant-frame-rate source that is the rate, and everything
// is fine.  For a variable one, r_frame_rate is the *maximum instantaneous* rate --
// it is a container hint, not a summary -- so every frame in the output is given
// that duration.  Measured on a VFR fixture (three concatenated segments at 30, 12
// and 60 fps, 102 frames, 3.0333 s of source):
//
//     source    102 frames   3.0333 s   avg 3060/91
//     rdither   102 frames   1.6320 s   avg 60/1
//
// 1.86x too short, and since the audio is stream-copied at its own timing, 1.86x of
// A/V desync.  The frames and their order are correct -- the decoder already passes
// timing through with -fps_mode passthrough -- so the picture looks right and only
// the clock is wrong, which is the worst shape a timing bug can have.
//
// WHY NOT "RE-RENDER AT A CONSTANT RATE", the usual advice.  That fixes the
// duration, by converting the source to CFR, and in doing so discards the thing
// that made the source variable.  Screen recordings and phone video are VFR
// because the capture only spends frames on change; flattening that resamples the
// motion and resamples it again on playback.
//
// WHAT ACTUALLY WORKS.  A rawvideo pipe carries no timestamps -- that is the whole
// constraint -- so the timing has to be re-applied at the encoder.  It can be, with
// a piecewise setpts expression built from the source's own frame durations:
//
//     settb=expr=1/<timescale>,setpts='<piecewise PTS as a function of N>'/TB
//
// and `-fps_mode passthrough -video_track_timescale <t>` on the output.  Proven
// before it was written into the pipeline: the same 102 frames come out with PTS
// running 0 -> 3.017 s, against a 3.0333 s source timeline.
//
// THREE THINGS THAT HAD TO BE RIGHT, each of which silently produces a wrong
// answer rather than an error:
//
//   * The TIMEBASE.  setpts works in the input's timebase, which for a rawvideo
//     input is 1/declared-rate.  At 1/60 s, frames 65 microseconds apart collapse
//     onto the same tick and ffmpeg drops them: 102 frames in, 100 out, and the
//     timeline silently short.  settb must raise the timebase first, to at least
//     the reciprocal of the shortest frame.
//
//   * The expression must be QUOTED.  -vf splits on commas as filter separators, so
//     `if(lt(N,29),...)` is read as a filter called "29)".  The inner single quotes
//     are load-bearing.
//
//   * MEASURE PTS, NOT DURATIONS.  Every frame's `duration` field stays at the
//     input rate no matter what setpts does, so summing packet durations reports a
//     1.6 s timeline for a 50.5 s one.  This cost an hour and produced two
//     entirely wrong conclusions before the PTS were read directly.
//
// WHY CFR INPUTS PAY NOTHING.  A constant-rate source produces one run, and the
// expression for one run is just a linear term -- at which point there is no timing
// to restore and the whole mechanism is skipped.  The overwhelming majority of
// inputs take the existing path unchanged, which is the only acceptable outcome
// for a fix to a rare input format.
struct FrameTiming {
  bool ok = false;          // the probe succeeded
  bool variable = false;    // durations are not all equal
  int frames = 0;
  double total = 0.0;       // sum of frame durations, seconds
  double avg_fps = 0.0;     // total / frames -- the honest average, unlike r_frame_rate
  double min_duration = 0.0;
  int runs = 0;             // distinct constant-rate segments
  int timescale = 0;        // 1/timescale, fine enough for min_duration
  std::string setpts;       // the expression, empty when `variable` is false
  std::string note;         // set when something was approximated; printed, not hidden
};

// Plain decimal, never scientific notation.  PowerShell is not involved here, but
// the lesson stands: ffmpeg's expression parser does not accept "6.5E-05" in the
// form some formatters produce, and it fails quietly.
std::string TimingDecimal(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9f", v);
  std::string s(buf);
  // Trim trailing zeros so the expression stays readable, but keep at least one
  // digit so "0" never becomes "".
  std::size_t last = s.find_last_not_of('0');
  if (last != std::string::npos && s[last] == '.') ++last;
  s.erase(last + 1);
  return s.empty() ? std::string("0") : s;
}

// ffprobe for the per-frame durations.  This is a DEMUX pass, not a decode: it
// reads the container's packet index where there is one, so it is orders of
// magnitude cheaper than the palette stage it runs alongside.
bool ProbeFrameTiming(const std::string& path, const std::string& ffmpeg_bin,
                      const std::string& ffprobe, FrameTiming* out,
                      std::string* error) {
  Child child;
  const std::string args =
      "-v error -select_streams v:0 -show_entries packet=duration_time "
      "-of csv=p=0 " + Quote(path);
  if (!child.Start(ffprobe, args, /*capture_stdout=*/true, /*feed_stdin=*/false,
                   error, "probe-timing")) {
    return false;
  }
  std::string text;
  char buffer[8192];
  for (;;) {
    const std::size_t got = child.Read(buffer, sizeof(buffer));
    if (got == 0) break;
    text.append(buffer, got);
  }
  child.Wait();

  // Runs of equal duration, with each run's start offset.  Built in one forward
  // pass: doing this backwards while accumulating the offset was a bug here, and
  // it produced a plausible expression with every offset wrong.
  struct Run { int start; int count; double dur; double offset; };
  std::vector<Run> runs;
  double total = 0.0;
  double min_dur = 0.0;
  int index = 0;
  std::size_t pos = 0;
  while (pos <= text.size()) {
    const std::size_t nl = text.find_first_of("\r\n", pos);
    const std::string line =
        text.substr(pos, (nl == std::string::npos ? text.size() : nl) - pos);
    pos = (nl == std::string::npos) ? text.size() + 1 : nl + 1;
    if (line.empty()) continue;
    // A duration of "N/A" is legal for the last packet of some containers; treat
    // it as unknown rather than as 0, which would be a zero-length frame.
    if (line.find("N/A") != std::string::npos) continue;
    const double dur = std::atof(line.c_str());
    if (!(dur > 0.0)) continue;
    if (min_dur == 0.0 || dur < min_dur) min_dur = dur;
    if (!runs.empty() && std::fabs(runs.back().dur - dur) < 1e-9) {
      ++runs.back().count;
    } else {
      Run r;
      r.start = index;
      r.count = 1;
      r.dur = dur;
      r.offset = total;
      runs.push_back(r);
    }
    total += dur;
    ++index;
  }
  if (index == 0) {
    *error = "no frame durations in " + path;
    return false;
  }
  out->ok = true;
  out->frames = index;
  out->total = total;
  out->min_duration = min_dur;
  out->runs = static_cast<int>(runs.size());
  out->avg_fps = static_cast<double>(index) / total;  // frames per SECOND, not seconds per frame
  out->variable = runs.size() > 1;
  if (!out->variable) return true;

  // A timebase fine enough for the shortest frame, clamped to the 90000 that
  // Matroska and the MPEG family both top out at.  Beyond that a frame is shorter
  // than the format can express and the honest thing is to say so.
  double want = 1.0 / min_dur;
  out->timescale = static_cast<int>(want);
  if (out->timescale < 15360) out->timescale = 15360;
  if (out->timescale > 90000) {
    out->timescale = 90000;
    out->note = "the source's shortest frame is " + TimingDecimal(min_dur) +
                " s, which needs a 1/" + TimingDecimal(want) +
                " timebase; clamped to 1/90000, so the shortest frames are quantised";
  }

  // The piecewise expression, built back-to-front so each branch wraps the rest.
  // For one run this is a plain linear term, which is the CFR case that never gets
  // here.
  std::string expr;
  for (int k = out->runs - 1; k >= 0; --k) {
    const Run& r = runs[static_cast<std::size_t>(k)];
    const std::string term = "(" + TimingDecimal(r.offset) + "+(N-" +
                             std::to_string(r.start) + ")*" +
                             TimingDecimal(r.dur) + ")";
    if (k == out->runs - 1) {
      expr = term;
    } else {
      expr = "if(lt(N," + std::to_string(runs[static_cast<std::size_t>(k) + 1].start) +
             ")," + term + "," + expr + ")";
    }
  }
  out->setpts = expr;
  return true;
}

// The audio stream's duration in seconds, or a negative value when there is no
// audio or ffprobe will not say.
//
// Needed because -shortest is only safe in ONE direction, and getting that wrong
// deletes real video.  See the call site.
double ProbeAudioDuration(const std::string& path, const std::string& ffprobe) {
  Child child;
  std::string ignored;
  const std::string args =
      "-v error -show_entries stream=codec_type,duration -of csv=p=0 " + Quote(path);
  if (!child.Start(ffprobe, args, /*capture_stdout=*/true, /*feed_stdin=*/false,
                   &ignored, "probe-audio")) {
    return -1.0;
  }
  std::string text;
  char buffer[4096];
  for (;;) {
    const std::size_t got = child.Read(buffer, sizeof(buffer));
    if (got == 0) break;
    text.append(buffer, got);
  }
  child.Wait();
  std::size_t pos = 0;
  while (pos <= text.size()) {
    const std::size_t nl = text.find('\n', pos);
    const std::string line =
        text.substr(pos, (nl == std::string::npos ? text.size() : nl) - pos);
    pos = (nl == std::string::npos) ? text.size() + 1 : nl + 1;
    if (line.rfind("audio,", 0) != 0) continue;
    const std::size_t comma = line.find(',', 6);
    if (comma == std::string::npos) continue;
    std::string value = line.substr(comma + 1);
    while (!value.empty() && (value.back() == '\r' || value.back() == ' ')) {
      value.pop_back();
    }
    if (value == "N/A" || value.empty()) {
      // Matroska routinely reports N/A for per-stream duration -- it is not stored
      // that way -- and Matroska is a plausible INPUT, not just an output.  Without a
      // fallback the caller sees "unknown" and omits -shortest, which is the safe
      // direction, but it silently gives up the one case -shortest exists for.
      // Summing the audio packets is exact for a stream copy and costs one extra
      // query, and only on the inputs that need it.
      child.Wait();
      Child packets;
      std::string ignored;
      const std::string pargs =
          "-v error -select_streams a:0 -show_entries packet=duration_time "
          "-of csv=p=0 " + Quote(path);
      if (packets.Start(ffprobe, pargs, /*capture_stdout=*/true,
                        /*feed_stdin=*/false, &ignored, "probe-audio-packets")) {
        std::string ptext;
        char pbuf[8192];
        for (;;) {
          const std::size_t got = packets.Read(pbuf, sizeof(pbuf));
          if (got == 0) break;
          ptext.append(pbuf, got);
        }
        packets.Wait();
        double sum = 0.0;
        int count = 0;
        std::size_t p = 0;
        while (p <= ptext.size()) {
          const std::size_t nl = ptext.find_first_of("\r\n", p);
          const std::string line =
              ptext.substr(p, (nl == std::string::npos ? ptext.size() : nl) - p);
          p = (nl == std::string::npos) ? ptext.size() + 1 : nl + 1;
          if (line.empty() || line.find("N/A") != std::string::npos) continue;
          const double dur = std::atof(line.c_str());
          if (dur > 0.0) { sum += dur; ++count; }
        }
        if (count > 0) return sum;
      }
      continue;
    }
    return std::atof(value.c_str());
  }
  return -1.0;
}

bool VideoProbe(const std::string& path, VideoInfo* out, std::string* error) {
  std::string ffprobe;
  if (!VideoFindTools(nullptr, &ffprobe, error)) return false;
  // The display matrix lives in `stream_side_data`, and combining it with the stream
  // fields needs a COLON between the two sections.  Two wrong forms were tried first
  // and both are worse than useless: a space-separated `side_data=rotation` is
  // accepted, matches nothing, and returns the stream fields with rotation silently
  // absent -- so every rotated input is then processed at the wrong geometry with no
  // error anywhere; and `stream_side_data=rotation` without the colon re-emits the
  // whole stream block instead of the side data.  A probe that exits 0 and quietly
  // omits the field it exists to fetch is the worst failure mode available.
  const std::string args =
      "-v error -select_streams v:0 -show_entries "
      "stream=width,height,nb_frames,r_frame_rate,pix_fmt,codec_name"
      ":stream_side_data=rotation "
      "-of default=nw=1 " + Quote(path);
  Child child;
  if (!child.Start(ffprobe, args, /*capture_stdout=*/true, /*feed_stdin=*/false,
                   error, "probe")) {
    return false;
  }
  std::string text;
  char buffer[4096];
  for (;;) {
    const std::size_t got = child.Read(buffer, sizeof(buffer));
    if (got == 0) break;
    text.append(buffer, got);
  }
  if (child.Wait() != 0) {
    *error = "ffprobe failed on " + path;
    return false;
  }
  std::vector<std::string> lines;
  {
    std::size_t start = 0;
    while (start < text.size()) {
      const std::size_t nl = text.find('\n', start);
      if (nl == std::string::npos) {
        lines.push_back(text.substr(start));
        break;
      }
      lines.push_back(text.substr(start, nl - start));
      start = nl + 1;
    }
  }
  for (const std::string& line : lines) {
    const std::string::size_type eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(0, eq);
    std::string value = line.substr(eq + 1);
    // ffprobe emits CRLF on Windows; strip the CR or every value keeps it.
    while (!key.empty() && (key.back() == '\r' || key.back() == ' ')) key.pop_back();
    while (!value.empty() && (value.back() == '\r' || value.back() == ' ')) {
      value.pop_back();
    }
    if (key == "width") {
      out->width = std::atoi(value.c_str());
    } else if (key == "height") {
      out->height = std::atoi(value.c_str());
    } else if (key == "rotation") {
      out->rotation = std::atoi(value.c_str());
    } else if (key == "nb_frames") {
      out->frames = std::atoll(value.c_str());
    } else if (key == "r_frame_rate") {
      const std::string::size_type slash = value.find('/');
      if (slash != std::string::npos) {
        const double num = std::atof(value.substr(0, slash).c_str());
        const double den = std::atof(value.substr(slash + 1).c_str());
        if (den != 0.0) out->fps = num / den;
      }
    } else if (key == "pix_fmt") {
      out->pix_fmt = value;
    } else if (key == "codec_name") {
      out->codec = value;
    }
  }
  if (out->width <= 0 || out->height <= 0) {
    if (getenv("RD_TRACE") != nullptr) {
      std::fprintf(stderr, "[video] ffprobe produced %zu bytes: <<%s>>\n",
                   text.size(), text.c_str());
    }
    *error = "could not read video dimensions from " + path;
    return false;
  }
  if (out->fps <= 0.0) out->fps = 25.0;
  return true;
}

// Does the input carry an audio stream?
//
// A separate, tiny probe rather than an extra entry on the one above, because the
// video probe is on the critical path of every single-image-adjacent call and
// -select_streams v:0 already scopes it to video.  Asking this question by *not*
// selecting a stream type is the point: `-select_streams a` returns one line per
// audio stream, and "at least one line came back" is the answer.
//
// Cached per path.  The encoder needs this once per render but the palette stage and
// the main pipeline both reach the same conclusion, and ffprobe costs a process spawn
// each time -- on a 3-hour clip that is worth avoiding.
bool VideoHasAudio(const std::string& path, const std::string& ffmpeg,
                   std::string* error) {
  static std::mutex mtx;
  static std::map<std::string, bool> cache;
  {
    std::lock_guard<std::mutex> lock(mtx);
    const auto it = cache.find(path);
    if (it != cache.end()) return it->second;
  }

  std::string ffprobe;
  if (!VideoFindTools(nullptr, &ffprobe, error)) return false;
  // -show_entries stream=index with no select is deliberately not used: it would
  // list video too.  The stream type selector is what makes "any audio" the question.
  const std::string args =
      "-v error -select_streams a -show_entries stream=index -of csv=p=0 " + Quote(path);
  Child child;
  if (!child.Start(ffprobe, args, /*capture_stdout=*/true, /*feed_stdin=*/false,
                   error, "probe-audio")) {
    return false;
  }
  std::string text;
  char buffer[1024];
  for (;;) {
    const std::size_t got = child.Read(buffer, sizeof(buffer));
    if (got == 0) break;
    text.append(buffer, got);
  }
  // ffprobe exits 0 with empty output when there is no such stream, which is the
  // normal case for a silent source and must not be treated as a failure.
  const bool found = !text.empty() && text.find_first_not_of(" \t\r\n") != std::string::npos;

  std::lock_guard<std::mutex> lock(mtx);
  cache[path] = found;
  return found;
}

bool VideoBuildPalette(const std::string& path, const VideoOptions& opt,
                       const VideoInfo& info, Palette* palette, ColorTree* tree,
                       VideoResult* result, std::string* error) {
  std::string ffmpeg;
  if (!VideoFindTools(&ffmpeg, nullptr, error)) return false;

  const std::size_t pixels =
      static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height);
  std::int64_t want = opt.palette_frames > 0 ? opt.palette_frames : info.frames;
  if (info.frames > 0 && want > info.frames) want = info.frames;
  if (want < 1) want = 1;

  // 'all' means "as many as the time budget allows", bounded by --palette-max-samples.
  // This is the default's answer to a real failure mode: 30 evenly spaced samples of a
  // 3-hour clip are 30 points, and a clip that spends its first four minutes on a
  // black leader and its last two on an end card can put every one of them on a grey
  // or black frame while the colourful two minutes in the middle are never looked at.
  // More samples strictly reduces that risk -- coverage is the only defence, since
  // nothing else in the pipeline can tell a bad sample from a good one.
  //
  // The cost is one ffmpeg spawn plus one seek per sample, run 6 at a time, so the
  // budget is spent on samples rather than on anything that needs a whole frame of
  // wall clock.  Measured on the 18001-frame clip: 30 samples 3.6 s, 64 samples 7.6 s,
  // 16 samples 2.1 s -- close to linear in the count, which is what makes a time
  // budget the right way to express the default.
  const bool want_all = opt.palette_frames <= 0;
  if (want_all) {
    const int cap = opt.palette_max_samples > 0 ? opt.palette_max_samples : 64;
    if (want > cap) want = cap;
  }

  // ---- pooled palette (default) -------------------------------------------
  //
  // The failure this exists to fix: the octree hands out palette slots in
  // proportion to *pixel population*, so footage that is mostly grey -- fog,
  // overcast, snow, a hyperlapse -- captures most of the N slots.  Sampling more
  // frames of that footage makes it worse, not better, because it adds more grey
  // pixels.  Changing the frame count or the tile size cannot fix it, and
  // measured mean saturation barely moves, because a palette of three greys and
  // thirteen vivid colours still averages high.
  //
  // So the weighting is removed instead of tuned.  Stage 1 quantizes every
  // sampled frame, at full resolution, to `colors` colours using ImageMagick's own
  // algorithm.  Stage 2 quantizes the pool of those per-frame palettes, where
  // every frame contributes exactly `colors` samples however many pixels it has.
  // A flat grey frame's swatches form one tight cluster and cost one slot; a
  // colourful frame's cost up to `colors`.  Slots now go to frames, not pixels.
  //
  // Only one frame is resident at a time, so memory is O(1) in the sample count,
  // which is what makes full-resolution sampling of many frames affordable -- the
  // montage path was memory-capped at 64 Mpixel precisely because it held them
  // all at once.
  // Pool is opt-in, not the default.  Implemented and measured as the user
  // proposed -- quantize each frame on its own, pool those palettes, quantize
  // again -- and it came out *worse* on every clip tested (80%-grey footage: 4.3%
  // mean saturation versus 48.8% for the montage; all-vivid footage: 63.6% versus
  // 83.7%).  Pre-reducing a frame to N colours makes those N cluster centroids,
  // and a second pass averages centroids again, so each stage desaturates what
  // the previous one produced.  Kept behind --palette-mode pool because the
  // mechanism is sound for a different failure -- one huge grey frame against one
  // colourful frame -- just not for grey-heavy footage.
  const int mode = opt.palette_mode == 1 ? 1 : 2;
  if (mode == 1) {
    // The pooled path -- mode 1 -- does a full-resolution QuantizeImage *per frame*,
    // ~1.0 s each at 1080p, so the 60 s seek budget is not the binding constraint
    // here: 60 samples would already be a minute and 500 would be nine hours.  The
    // spawn-cost argument that sets the default above simply does not apply, so the
    // cap stays tight and explicit.
    //
    // (Measured and rejected: raising this cap is the tempting fix for a clip whose
    // palette lands on grey, and it makes things *worse* here.  Stage 1 reduces each
    // frame on its own and merges the results, so a frame of solid grey contributes a
    // near-neutral entry to the merge whether or not its neighbours are colourful, and
    // adding more grey frames adds more near-neutral entries.  The montage path does
    // not have this failure because it quantises the frames *together*, so grey pixels
    // are outvoted by coloured ones in one histogram.  44.5% saturation versus 48.8%
    // for the montage on an 80%-grey clip, and the gap widens with the grey.)
    const int cap = opt.palette_max_samples > 0 ? opt.palette_max_samples : 64;
    if (opt.palette_frames <= 0 && want > cap) {
      std::fprintf(stderr,
                   "[video] palette: 'all' reduced to %d sampled frame(s); the "
                   "pooled path quantises each frame separately (~1.0 s at 1080p) "
                   "and merges, so a solid-grey frame contributes a near-neutral "
                   "entry however many you add. Use --palette-mode montage, which "
                   "quantises them together. Raise --palette-max-samples to spend "
                   "more time anyway.\n",
                   cap);
      want = cap;
    }
    const int colors = std::max(1, opt.colors);
    // Stage 1 deliberately reduces further than the final count: its job is to
    // strip the pixel-population weighting, not to choose the answer.
    const int s1 = std::max(colors, opt.palette_stage1_colors);

    // Evenly spaced sampling over the whole clip.  `select` runs before the raw
    // output, so only the sampled frames ever cross the pipe.
    char pfilter[128];
    if (want >= info.frames && info.frames > 0) {
      std::snprintf(pfilter, sizeof(pfilter), "null");
    } else {
      const long long step = (info.frames > 0) ? (info.frames / want) : 1;
      std::snprintf(pfilter, sizeof(pfilter), "select=not(mod(n\\,%lld))",
                    step < 1 ? 1 : step);
    }
    // Deliberately NOT hardware-decoded, unlike the main reader.  NVDEC has to create
    // a CUDA context per process, which costs a few hundred milliseconds, and this
    // decoder only produces ~30 frames.  Measured: the palette stage went 1057 ->
    // 1650 ms with `-hwaccel cuda` here, a clear loss.  The main reader decodes 605
    // frames and is a different story.
    const FpsModeArgs& fps = FpsMode(ffmpeg);
    const std::string pargs = "-v error" + std::string(fps.input) + " -i " + Quote(path) +
                              " -vf " + pfilter + fps.output +
                              " -frames:v " + std::to_string(want) +
                              " -f rawvideo -pix_fmt rgba64le -";
    Child pchild;
    if (!pchild.Start(ffmpeg, pargs, true, false, error, "palette-decode")) return false;

    // Full resolution: one frame at a time.  Palette and tree are 2 MiB and a
    // node array, so they live on the heap, never as locals.
    std::vector<RgbaF> frame(pixels);
    std::vector<std::uint16_t> raw(pixels * 4);
    std::unique_ptr<Palette> sub_palette(new Palette());
    std::unique_ptr<ColorTree> sub_tree(new ColorTree());
    std::vector<PaletteEntry> pool_colors;
    pool_colors.reserve(static_cast<std::size_t>(want) *
                          static_cast<std::size_t>(s1));

    const double pt0 = NowMs();
    double pdecode = 0.0, pquant = 0.0;
    int sampled = 0;
    for (; sampled < static_cast<int>(want); ++sampled) {
      const double d0 = NowMs();
      const std::size_t got =
          pchild.Read(raw.data(), raw.size() * sizeof(std::uint16_t));
      pdecode += NowMs() - d0;
      if (got != raw.size() * sizeof(std::uint16_t)) break;  // short stream
      RawToFloats(raw.data(), frame.data(), pixels, 4);

      const double q0 = NowMs();
      if (!ImBuildPaletteFromPixels(frame.data(), info.width, info.height, s1,
                                    sub_palette.get(), sub_tree.get(), error)) {
        pchild.Close();
        return false;
      }
      pquant += NowMs() - q0;

      // Collect this frame's colours rather than painting straight into the mosaic:
      // the mosaic cannot be sized until the swatches are known, and stage 1's
      // output has to be deduplicated before stage 2 (see below).
      for (int i = 0; i < sub_palette->count; ++i) {
        pool_colors.push_back(sub_palette->entries[i]);
      }
    }
    pchild.Close();
    if (sampled == 0) {
      *error = "ffmpeg produced no frames for the palette sample";
      return false;
    }

    // Deduplicate the pool, which is the step that makes two-stage quantizing
    // work at all on grey-heavy footage.
    //
    // Per-frame weighting alone is not enough, and measured it is much worse than
    // the montage: on an 80%-grey clip it drove mean saturation from 44.5% down to
    // 4.3% with 10-13 of 16 slots near-neutral, because equalising per frame still
    // leaves 80% of the swatches grey when 80% of the frames are grey.
    //
    // What is actually wrong is that a hyperlapse contributes the *same* palette
    // over and over, voting with its pixel count rather than with its distinctness.
    // Merging swatches that are within `tolerance` of one already kept turns 48
    // near-identical greys into a handful, so the palette ends up covering the
    // distinct looks in the footage instead of being dominated by whichever look
    // occupies the most pixels or the most frames.
    const int tolerance = std::max(0, opt.palette_dedup);
    std::vector<PaletteEntry> unique_colors;
    unique_colors.reserve(pool_colors.size());
    for (const PaletteEntry& c : pool_colors) {
      bool dup = false;
      if (tolerance > 0) {
        const double t2 = static_cast<double>(tolerance) * tolerance;
        for (const PaletteEntry& k : unique_colors) {
          const double dr = c.r - k.r, dg = c.g - k.g, db = c.b - k.b;
          if (dr * dr + dg * dg + db * db < t2) {
            dup = true;
            break;
          }
        }
      }
      if (!dup) unique_colors.push_back(c);
    }

    const std::size_t unique_n = unique_colors.size();
    const int cell = std::max(1, static_cast<int>(std::ceil(
                                     std::sqrt(static_cast<double>(unique_n)))));
    const int ucols = cell;
    const int urows = (static_cast<int>(unique_n) + ucols - 1) / ucols;
    // Give the octree enough pixels to prune against: the final tree is built from
    // this mosaic, and a few thousand pixels make for a degenerate one.  Scaling
    // every swatch equally does not disturb the weighting.
    int sw = 1;
    while (sw < 32 && unique_n * static_cast<std::size_t>(sw) * sw < (1u << 16)) {
      ++sw;
    }
    const std::size_t mosaic_w =
        static_cast<std::size_t>(ucols) * static_cast<std::size_t>(sw);
    const std::size_t mosaic_h =
        static_cast<std::size_t>(urows) * static_cast<std::size_t>(sw);
    std::vector<RgbaF> mosaic(mosaic_w * mosaic_h);
    for (std::size_t i = 0; i < unique_n; ++i) {
      const PaletteEntry& e = unique_colors[i];
      const std::size_t gx = (i % static_cast<std::size_t>(ucols)) *
                             static_cast<std::size_t>(sw);
      const std::size_t gy = (i / static_cast<std::size_t>(ucols)) *
                             static_cast<std::size_t>(sw);
      for (std::size_t y = 0; y < static_cast<std::size_t>(sw); ++y) {
        RgbaF* row = mosaic.data() + (gy + y) * mosaic_w + gx;
        for (std::size_t x = 0; x < static_cast<std::size_t>(sw); ++x) {
          row[x].r = static_cast<float>(e.r);
          row[x].g = static_cast<float>(e.g);
          row[x].b = static_cast<float>(e.b);
          row[x].a = static_cast<float>(e.a);
        }
      }
    }

    // Stage 2.  The tree the dither searches is built from this same mosaic, so
    // palette and tree stay consistent.
    const double q2 = NowMs();
    if (!ImBuildPaletteFromPixels(mosaic.data(), mosaic_w, mosaic_h, colors, palette,
                                  tree, error)) {
      return false;
    }
    const double ptotal = NowMs() - pt0;
    if (result != nullptr) {
      result->palette_sampled = sampled;
      result->palette_colors = palette->count;
      result->palette_tile = 0;
      result->palette_mosaic_w = static_cast<int>(mosaic_w);
      result->palette_mosaic_h = static_cast<int>(mosaic_h);
      result->palette_pixels = mosaic_w * mosaic_h;
      result->palette_neutral = CountNeutralPaletteEntries(*palette, 0.2);
      result->palette_ms = ptotal;
      result->decode_ms = pdecode;
      if (result->palette_secondary_ms <= 0.0) {
        result->palette_secondary_ms = pquant;
      }
    }
    if (!opt.quiet) {
      std::fprintf(stderr,
                   "[video] palette pool: %d frame(s) x %d colours at %dx%d, "
                   "%dx%d mosaic, stage 1 %.0f ms (decode %.0f ms), stage 2 %.0f ms\n",
                   sampled, s1, info.width, info.height,
                   static_cast<int>(mosaic_w), static_cast<int>(mosaic_h), pquant,
                   pdecode, NowMs() - q2);
    }
    return true;
  }

  // Hard cap on montage area, so a long clip cannot build a 19 GiB image.  Full
  // resolution is the default cell, which for 1080p means 30 frames fits in the
  // 64 Mpixel budget -- exactly the shape of `magick f1..f30 +append -colors 16`.
  // A tile makes the budget go much further, which is the only reason to use one.
  constexpr std::int64_t kMaxMontagePixels = 64ll << 20;  // 64 Mpixel
  const int tile = opt.palette_tile;  // <= 0 means "full frame"
  const int full_edge = std::min(info.width, info.height);
  // tile <= 0 means "full frame"; resolved to the frame's own dimensions for the
  // area budget below, then used to decide the cell size.
  const std::int64_t cell_edge_budget = opt.palette_tile > 0
                                            ? std::min(opt.palette_tile, full_edge)
                                            : full_edge;
  // Exact division, not repeated halving.  Halving overshoots badly: for 1080p,
  // 30 full frames are 62 Mpixel and fit the 64 Mpixel budget, but halving walks
  // 605 -> 302 -> 151 -> 75 -> 37 -> 18 and settles on 18, throwing away 12 of the
  // 30 frames a 605-frame clip should contribute.  Fewer samples on grey-heavy
  // footage measurably flattens the palette.
  const std::int64_t cell_px_budget =
      opt.palette_tile > 0 ? cell_edge_budget * cell_edge_budget
                           : static_cast<std::int64_t>(info.width) * info.height;
  if (want > 1 && cell_px_budget > 0) {
    const std::int64_t affordable = kMaxMontagePixels / cell_px_budget;
    if (affordable >= 1 && want > affordable) {
      std::fprintf(stderr,
                   "[video] palette: reduced to %lld sampled frame(s); a %lldx%lld "
                   "cell does not fit %lld frames in the %lld Mpixel montage cap\n",
                   static_cast<long long>(affordable),
                   static_cast<long long>(cell_px_budget > 0 && cell_edge_budget > 0
                                              ? cell_edge_budget
                                              : info.width),
                   static_cast<long long>(cell_px_budget > 0 && cell_edge_budget > 0
                                              ? cell_edge_budget
                                              : info.height),
                   static_cast<long long>(want),
                   static_cast<long long>(kMaxMontagePixels >> 20));
      want = affordable;
    }
  }
  if (want < 1) want = 1;
  if (opt.palette_frames <= 0 && want < info.frames) {
    // Only mention the montage cap when it is what actually bound the count, or the
    // line is misleading: palette_max_samples and the area cap are separate limits
    // and either may be the tighter one.  It was set to 64 by palette_max_samples on
    // this machine, not by the area arithmetic, and the old wording claimed the
    // latter -- which points a user at a knob that is not the one to turn.
    std::fprintf(stderr,
                 "[video] palette: 'all' reduced to %lld sampled frame(s) "
                 "(limits: --palette-max-samples %d, montage area %lld Mpixel)\n",
                 static_cast<long long>(want), opt.palette_max_samples,
                 kMaxMontagePixels >> 20);
  }
  if (want > 256) {
    // Progress, not a warning.  At ~0.12 s per sample six at a time, several hundred
    // samples is a minute of a serial prefix in which nothing else is happening, and
    // an unexplained pause reads as a hang.  The user asked for this and is getting
    // exactly this, so say what is happening and how long it should take.
    //
    // (The older note here argued against large counts, on the evidence that mean
    // saturation was 84.7% at 30 samples and 84.9% at 4096 on one clip.  That
    // evidence is real and it is still true -- and it is also true that 30 samples on
    // *that* clip was already saturated, so it measured the ceiling of a clip that was
    // never in danger.  It says nothing about a clip where the 30 land on grey, which
    // is the case this default exists for.  The two are not the same claim.)
    const double est = static_cast<double>(want) * 0.12;
    std::fprintf(stderr,
                 "[video] palette: sampling %lld frames, ~%.0f s at 6 seeks at a "
                 "time.\n",
                 static_cast<long long>(want), est);
  }

  // Evenly spaced sampling.  `select` runs before the raw output, so only the
  // sampled frames ever cross the pipe.
  char filter[128];
  if (want >= info.frames && info.frames > 0) {
    std::snprintf(filter, sizeof(filter), "null");
  } else {
    const long long step = (info.frames > 0) ? (info.frames / want) : 1;
    std::snprintf(filter, sizeof(filter), "select=not(mod(n\\,%lld))",
                  step < 1 ? 1 : step);
  }

  // Passthrough frame timing is REQUIRED, not cosmetic: the select filter leaves
  // timestamp gaps, and the default frame-rate conversion re-fills them, so
  // -frames:v N then yields the first N frames in time order rather than the N
  // selected ones.  That silently dropped every sampled frame after the first
  // contiguous run, which on grey-heavy footage produced an all-grey palette.
  // Software on purpose -- see the note on the palette decoder above: NVDEC's
  // per-process context cost dominates when only a handful of frames are wanted.
  // Spelled and placed through FpsMode because ffmpeg 9 deleted -vsync and
  // -fps_mode is an output option; see the note on FpsModeArgs.
  const FpsModeArgs& fps = FpsMode(ffmpeg);
  const std::string args = "-v error" + std::string(fps.input) + " -i " + Quote(path) +
                           " -vf " + filter + fps.output +
                           " -frames:v " + std::to_string(want) + " -f rawvideo" +
                           (opt.palette_depth == 16 ? " -pix_fmt rgba64le"
                                                    : " -pix_fmt rgba") +
                           " -";
  Child child;
  if (std::getenv("RD_TRACE") != nullptr) {
    std::fprintf(stderr, "[video] palette sample plan: want=%lld frames=%lld\n",
                 static_cast<long long>(want),
                 static_cast<long long>(info.frames));
  }
  // Sample by seeking rather than by decoding.
  //
  // `select=not(mod(n,step))` looks like it avoids the frames it drops, and it
  // does not: the filter runs *after* h264 decoding, so ffmpeg decodes the whole
  // clip and throws away 99.8% of the result.  Measured on an 18001-frame 1080p
  // clip, the palette stage reported 26627 ms and a bare `-f null -` decode of the
  // same file measured 26.2 s -- the stage *was* a full decode.  That is a fifth of
  // the whole render, spent decoding frames nobody looks at.
  //
  // Seeking to each sample instead decodes 30 frames rather than 18001.  The cost is
  // one process spawn plus one seek per sample, so it only wins once the clip is
  // long enough that those spawns cost less than the decode they avoid -- a few
  // hundred frames here.  Below that the sequential read stays, because for a
  // 605-frame clip the spawns alone would cost more than the entire decode.
  //
  // The frames are identical either way: H.264 reconstruction is normative, so a
  // seek and a full decode produce the same pixels for the same presentation
  // timestamp.  The seek target is the middle of the wanted frame rather than its
  // timestamp, so float rounding cannot land on its predecessor.
  // The crossover is where the clip's decode costs more than the spawns do.  On this
  // machine a 18001-frame decode is 26.2 s against 7.0 s for 30 spawns, while a
  // 3000-frame decode is 4.3 s against 6.8 s -- so the spawns only win somewhere
  // around 5000 frames, and 1200 (the first guess) picked the slower path for
  // everything between.  Measured, not assumed; the two data points bracket it.
  const char* seek_env = std::getenv("RD_PALETTE_SEEK");
  const bool sample_by_seek =
      want > 1 && info.frames > 5000 &&
      (seek_env == nullptr || seek_env[0] != '0');
  const long long sample_step =
      (want < info.frames && info.frames > 0)
          ? static_cast<long long>(info.frames / want)
          : 1;

  // The 'all' budget: as many samples as the per-sample cost says will fit, measured
  // rather than tabulated.
  //
  // Cost per sample is one ffmpeg spawn plus one seek, run 6 at a time, and it was
  // measured at 0.11-0.12 s per sample on the 18001-frame clip (16 -> 2.1 s, 30 ->
  // 3.6 s, 64 -> 7.6 s: linear, as a spawn-bound stage should be).  What this budget
  // buys with its 60 s is therefore ~500 samples, not 30.
  //
  // It is a *budget* rather than a count on purpose.  A fixed 30 was chosen to match a
  // reference pipeline that quantises 30 named files, which is a meaningful constraint
  // there and a meaningless one here: the reason to sample widely is that 30 points
  // can all land in the dull parts of a 3-hour clip, and the fix is coverage, which
  // grows with the clip's length.  A count cannot express that; a time budget can, and
  // it also degrades gracefully -- a slow disk takes fewer samples instead of taking
  // the same time it always took.
  if (want_all) {
    const double budget_s = opt.palette_budget_ms > 0 ? opt.palette_budget_ms / 1000.0 : 60.0;
    // Measured, and deliberately pessimistic: 0.12 s per sample at six workers.  The
    // 0.02 s allowance on top is for the montage write and the quantize itself, which
    // are not per-sample but would be silently unbudgeted otherwise.
    const int affordable = static_cast<int>((budget_s - 0.02) / 0.12);
    const int target = std::max(1, std::min(affordable, static_cast<int>(want)));
    if (target > 0 && info.frames > 0) {
      // Never ask for more samples than the clip has distinct frames, and never
      // fewer than the cap the user set.
      // Every term is int64_t from here.  Mixing int and int64_t in std::min is
      // ambiguous, and the ambiguity is a compile error rather than a silent
      // conversion -- which is the good kind of mistake to hit.
      std::int64_t chosen = std::min<std::int64_t>(target, want);
      if (info.frames > 0) chosen = std::min(chosen, info.frames);
      if (opt.palette_max_samples > 0) {
        chosen = std::min<std::int64_t>(chosen, opt.palette_max_samples);
      }
      want = std::max<std::int64_t>(chosen, 1);
    }
    std::fprintf(stderr,
                 "[video] palette: sampling %lld frame(s) of %lld within a %.0f s "
                 "budget\n",
                 static_cast<long long>(want), static_cast<long long>(info.frames),
                 budget_s);
  }

  if (sample_by_seek) {
    if (std::getenv("RD_TRACE") != nullptr) {
      std::fprintf(stderr, "[video] palette: sampling %lld frames by seek (step %lld)\n",
                   static_cast<long long>(want), sample_step);
    }
  } else if (!child.Start(ffmpeg, args, true, false, error, "palette-frames")) {
    return false;
  }

  // One progress indicator for the whole sampling stage, covering both the
  // by-seek and the sequential paths.  It is declared here because `want` is
  // final at this point -- the budget clamp above can still reduce it -- and
  // because a Progress's denominator must be the count that will actually be
  // attempted, not the one that was requested.
  Progress progress("palette", want, !opt.quiet);

  // Cell size.  tile == 0 (the default) means full resolution: each sampled frame
  // contributes every one of its pixels, which is what makes a single
  // quantization of the strip behave like `magick f1 f2 ... +append -colors N`.
  // A lattice tile is a sparse, aliased peek at each frame and measurably worse;
  // it is kept only for callers who want the montage to fit in less RAM.
  const int cell_w = tile > 0 ? tile : info.width;
  const int cell_h = tile > 0 ? tile : info.height;

  // Grid geometry.  The reference builds one long horizontal strip with `+append`,
  // and the octree is *spatial*, so a grid is not merely a different packing of the
  // same data: the same pixels in 6x5 rather than 1x30 subdivide differently and
  // yield a different colormap.  With --im-palette the strip is reproduced exactly.
  const int cols = opt.im_palette
                       ? static_cast<int>(want)
                       : static_cast<int>(std::ceil(std::sqrt(static_cast<double>(want))));
  const int rows = (want + cols - 1) / cols;
  const std::size_t montage_w = static_cast<std::size_t>(cols) * cell_w;
  const std::size_t montage_h = static_cast<std::size_t>(rows) * cell_h;
  const std::size_t montage_pixels = montage_w * montage_h;

  std::vector<RgbaF> montage(montage_pixels);
  // Palette source depth.  8 bits, not 16, and that is not a detail: feeding the
  // quantizer 16-bit RGBA gives a measurably flatter palette than the same pixels
  // at 8 bits.  On a 30-frame 1080p sample of the same montage:
  //     16-bit RGBA -> 16 colours, 41.2% mean saturation
  //     8-bit  RGB  -> 15 colours, 57.5%
  // IM's ClassifyImageColors prunes against colour error at Quantum precision, so
  // at 16 bits it keeps splitting until the error is negligible and settles on
  // desaturated centroids.  The reference video pipeline goes through 8-bit PPM
  // and gets the punchier palette for free, so the default follows it.
  const bool palette_8bit = opt.palette_depth != 16;
  const std::size_t raw_px = pixels * 4;
  std::vector<std::uint16_t> raw16(raw_px);
  std::vector<std::uint8_t> raw8(raw_px);
  std::vector<RgbaF> frame(pixels);

  // Places one decoded sample into its montage cell.  Factored out because the
  // parallel seek path below needs the same placement without the shared `frame`.
  auto place_frame = [&](int s, const RgbaF* src) {
    const int cx = s % cols;
    const int cy = s / cols;
    if (tile <= 0) {
      for (int y = 0; y < cell_h; ++y) {
        const RgbaF* row = src + static_cast<std::size_t>(y) * info.width;
        RgbaF* dst = montage.data() +
                     (static_cast<std::size_t>(cy) * cell_h + y) * montage_w +
                     static_cast<std::size_t>(cx) * cell_w;
        std::copy(row, row + info.width, dst);
      }
      return;
    }
    for (int ty = 0; ty < cell_h; ++ty) {
      const int sy =
          static_cast<int>(static_cast<std::int64_t>(ty) * info.height / cell_h);
      const RgbaF* row = src + static_cast<std::size_t>(sy) * info.width;
      const std::size_t fy =
          static_cast<std::size_t>(cy) * cell_h + static_cast<std::size_t>(ty);
      RgbaF* dst = montage.data() + fy * montage_w +
                   static_cast<std::size_t>(cx) * cell_w;
      for (int tx = 0; tx < cell_w; ++tx) {
        const int sx =
            static_cast<int>(static_cast<std::int64_t>(tx) * info.width / cell_w);
        dst[tx] = row[sx];
      }
    }
  };

  const double t0 = NowMs();
  int sampled = 0;
  double decode_ms = 0.0;

  if (sample_by_seek) {
    // 30 independent seeks, run concurrently.
    //
    // Each sample is a separate short-lived ffmpeg against a different timestamp,
    // and each owns exactly one montage cell, so there is nothing to share and
    // nothing to lock -- the only shared writes are to disjoint cells.  Sequentially
    // this was 30 spawns at ~230 ms, i.e. 7 s of a 400 s render spent entirely
    // waiting on process startup and seek latency, with the rest of the machine
    // idle because the palette is a serial prefix that gates the whole pipeline.
    //
    // Concurrency is capped rather than one-thread-per-sample: on a spinning disk
    // thirty simultaneous seeks thrash the head, and past about six the returns
    // flatten anyway.
    const int workers =
        std::max(1, std::min<int>(want, 6));
    std::vector<std::uint8_t> placed(static_cast<std::size_t>(want), 0);
    std::atomic<int> next_sample{0};
    // Completion count, as opposed to `next_sample` which counts *attempts*.
    // A failed seek returns early via `continue` without reaching the update, so
    // counting attempts would let the indicator run to 100% and stop, while the
    // remaining work was still queued -- the progress bar would be lying in
    // exactly the situation where a user most needs it to be true.  This is only
    // consulted for display; `sampled` below is still derived from `placed`.
    std::atomic<int> seeks_done{0};
    const double p0 = NowMs();
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers));
    for (int wi = 0; wi < workers; ++wi) {
      pool.emplace_back([&]() {
        std::vector<std::uint8_t> l8(static_cast<std::size_t>(pixels) * 4);
        std::vector<std::uint16_t> l16(static_cast<std::size_t>(pixels) * 4);
        std::vector<RgbaF> lf(static_cast<std::size_t>(pixels));
        for (;;) {
          const int s = next_sample.fetch_add(1);
          if (s >= want) return;
          const long long idx = static_cast<long long>(s) * sample_step;
          const double when =
              info.fps > 0.0
                  ? (static_cast<double>(idx) - 0.25) / info.fps
                  : 0.0;
          char sargs[256];
          std::snprintf(sargs, sizeof(sargs),
                        "-v error -ss %.6f -i %s -fps_mode passthrough -frames:v 1 "
                        "-f rawvideo -pix_fmt %s -",
                        when, Quote(path).c_str(),
                        palette_8bit ? "rgba" : "rgba64le");
          Child one;
          std::string serr;
          if (!one.Start(ffmpeg, sargs, true, false, &serr, "palette-seek")) continue;
          bool full = false;
          if (palette_8bit) {
            full = one.Read(l8.data(), l8.size()) == l8.size();
            if (full) {
              for (std::size_t i = 0; i < pixels; ++i) {
                lf[i].r = static_cast<float>(l8[4 * i + 0]) * 257.0f;
                lf[i].g = static_cast<float>(l8[4 * i + 1]) * 257.0f;
                lf[i].b = static_cast<float>(l8[4 * i + 2]) * 257.0f;
                lf[i].a = static_cast<float>(l8[4 * i + 3]) * 257.0f;
              }
            }
          } else {
            full = one.Read(l16.data(), l16.size() * sizeof(std::uint16_t)) ==
                   l16.size() * sizeof(std::uint16_t);
            if (full) RawToFloats(l16.data(), lf.data(), pixels, 4);
          }
          one.Close();
          if (!full) continue;
          place_frame(s, lf.data());
          placed[static_cast<std::size_t>(s)] = 1;
          // Progress from whichever worker happens to finish, which is the
          // point: with six concurrent seeks the completions arrive out of
          // order, and a bar driven by the sample *index* would jump around.
          // This path was entirely silent before, and it is the default path
          // for any clip over 5000 frames -- a 28-second stretch of a 5-minute
          // render with no output, on the stage that gates the whole pipeline.
          progress.Update(seeks_done.fetch_add(1) + 1);
        }
      });
    }
    for (std::thread& th : pool) th.join();
    decode_ms = NowMs() - p0;
    // Land the bar on the true total.  The workers each reported a successful
    // seek, so this is normally already there; it matters when some seeks
    // failed, and then the correct number to end on is the number of cells
    // actually filled, not the number of attempts.
    progress.Update(static_cast<std::int64_t>(std::count(
                        placed.begin(), placed.end(), static_cast<std::uint8_t>(1))));
    // `sampled` must be the count of leading filled cells, because the caller pads
    // from there and a gap would leave a cell black.  In practice every seek
    // succeeds; if one does not, stopping at the first gap is the honest report.
    for (; sampled < want; ++sampled) {
      if (!placed[static_cast<std::size_t>(sampled)]) break;
    }
    if (std::getenv("RD_TRACE") != nullptr) {
      std::fprintf(stderr, "\r[palette] %d/%d samples by seek, %.1f ms/frame elapsed %.1f s   ",
                   sampled, want, decode_ms / std::max(1, sampled),
                   decode_ms / 1000.0);
      std::fflush(stderr);
    }
  }

  for (; !sample_by_seek && sampled < want; ++sampled) {
    const double d0 = NowMs();
    std::size_t got = 0;
    if (palette_8bit) {
      got = child.Read(raw8.data(), raw8.size());
      if (got == raw8.size()) {
        for (std::size_t i = 0; i < pixels; ++i) {
          frame[i].r = static_cast<float>(raw8[4 * i + 0]) * 257.0f;
          frame[i].g = static_cast<float>(raw8[4 * i + 1]) * 257.0f;
          frame[i].b = static_cast<float>(raw8[4 * i + 2]) * 257.0f;
          frame[i].a = static_cast<float>(raw8[4 * i + 3]) * 257.0f;
        }
      }
    } else {
      got = child.Read(raw16.data(), raw16.size() * sizeof(std::uint16_t));
      if (got == raw16.size() * sizeof(std::uint16_t)) {
        RawToFloats(raw16.data(), frame.data(), pixels, 4);
      }
    }
    decode_ms += NowMs() - d0;
    if (got != (palette_8bit ? raw8.size() : raw16.size() * sizeof(std::uint16_t))) {
      break;  // short stream
    }

    // Progress, flushed.  The palette stage is the longest silent stretch in the
    // program -- it is dominated by waiting on the decoder and by per-frame
    // ImageMagick call overhead on a tiny tile -- and with stdout block-buffered
    // (any redirect to a file or pipe) an unflushed printf is invisible, so a
    // multi-minute build looks exactly like a hang.
    //
    // This used to hand-roll the interval as `sampled % ProgressEvery(want)`,
    // which is what rd_progress.h replaced: the count-based interval meant a
    // 256-sample run updated ~50 times while a 12-sample run updated twice, and
    // neither tracked a human sense of time.  It also ignored --quiet.
    progress.Update(sampled + 1);

    // Full resolution is a straight copy; a tile is a lattice sample.
    //
    // The tile used to be a box average, with a comment claiming that "keeps the
    // palette representative".  That is backwards: averaging distinct hues pulls
    // each sample toward the local mean, i.e. toward grey, so the filter was
    // desaturating the palette before the quantizer ever saw it.  Replacing it
    // with a lattice sample stopped the averaging, but a lattice is still a sparse
    // aliased view of a 2 Mpixel frame.  Copying the frame whole is both cheaper
    // to reason about and measurably better: 16-colour mean saturation on an
    // 80%-grey clip went 48.8% (tile 256 lattice) -> 52.0% (tile 128) -> 74.6%
    // (full resolution).
    place_frame(sampled, frame.data());
  }
  child.Close();
  if (sampled == 0) {
    *error = "ffmpeg produced no frames for the palette sample";
    return false;
  }

  // The grid is sized from the *requested* count, so it can have cells the
  // sampler never filled -- 151 frames go into 13x12 = 156 cells, leaving 5
  // cells as zero-initialised black.  Black is a real colour to the octree, so
  // those cells spend palette entries on nothing.  Repeat the last tile into
  // them instead; better a slightly over-weighted final frame than dead black.
  for (int s = sampled; s < cols * rows; ++s) {
    const int cx = s % cols;
    const int cy = s / cols;
    const int px = (sampled - 1) % cols;
    const int py = (sampled - 1) / cols;
    for (int ty = 0; ty < cell_h; ++ty) {
      const std::size_t src =
          (static_cast<std::size_t>(py) * cell_h + static_cast<std::size_t>(ty)) *
              montage_w +
          static_cast<std::size_t>(px) * cell_w;
      const std::size_t dst_off =
          (static_cast<std::size_t>(cy) * cell_h + static_cast<std::size_t>(ty)) *
              montage_w +
          static_cast<std::size_t>(cx) * cell_w;
      std::copy(montage.begin() + static_cast<std::ptrdiff_t>(src),
                montage.begin() +
                    static_cast<std::ptrdiff_t>(src + cell_w),
                montage.begin() + static_cast<std::ptrdiff_t>(dst_off));
    }
  }

  const std::size_t mw = montage_w;
  const std::size_t mh = montage_h;
  // Diagnostic: dump the exact montage the quantizer is about to see, so it can
  // be diffed against the equivalent `magick ... +append -colors N` strip.  Off by
  // default; the montage can be gigabytes.
  if (const char* dump = std::getenv("RD_DUMP_MONTAGE"); dump != nullptr &&
                          *dump != '\0') {
    ImWriteRgbaF(montage.data(), mw, mh, dump);
  }
  if (!opt.im_palette) {
    if (!ImBuildPaletteFromPixels(montage.data(), mw, mh, opt.colors, palette,
                                  tree, error)) {
      return false;
    }
  } else {
    // The reference pipeline's palette step, in-process: 8-bit, no alpha,
    // `+append` layout, IM's own quantizer, then the colormap deduplication that
    // `-unique-colors` performs.  See ImBuildPaletteAppend8 for why each of
    // those details changes the result.
    if (!ImBuildPaletteAppend8(montage.data(), mw, mh, opt.colors, palette, tree,
                               error)) {
      return false;
    }
  }
  const double total = NowMs() - t0;
  if (result != nullptr) {
    result->palette_sampled = sampled;
    result->palette_colors = palette->count;
    result->palette_tile = tile;
    result->palette_pixels = mw * mh;
    result->palette_neutral = CountNeutralPaletteEntries(*palette, 0.2);
    result->palette_ms = total;
    result->decode_ms = decode_ms;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Pipelined dither
// ---------------------------------------------------------------------------
// The serial version measured decode 518 s / dither 1645 s / encode 353 s on an
// 18001-frame 1080p clip, so every stage spent most of its life waiting on the
// previous one.  Three changes:
//
//   1. A reader thread fills a bounded queue of decoded batches, so decode runs
//      concurrently with dither instead of before it.
//   2. Dither workers are a pool: N host threads plus the GPU, all pulling from
//      one queue.  The GPU's walk is ~3x faster per frame than a core's, but a
//      core does 318 ms/frame against the GPU's 85 ms, so on a 12-core box the
//      host pool has ~3.4x the aggregate throughput.  Whichever engine is free
//      takes the next batch, so the mix adapts to frame cost automatically.
//   3. The queue depth is derived from available RAM rather than fixed, so a 4K
//      sequence on a small machine throttles itself instead of swapping.
//
// The pool is safe because the CPU and GPU walks implement the *same* block
// partition with the same arithmetic, so a frame's output does not depend on
// which engine dithered it.
struct Batch {
  std::vector<RgbaF> pixels;
  // rgba64le in (reader -> device) and out (device -> writer), kept apart because
  // the device reads one while it writes the other.  Before the uint16 upload these
  // were a single `raw` buffer that the reader filled and the device then overwrote
  // with the dithered result, which only works because the reader's copy was dead by
  // then; separate buffers make the flow obvious and cost nothing extra, since the
  // total is the same.
  //
  // The GPU path uses page-locked buffers instead of these vectors, so the reader can
  // deposit frames straight into memory the copy engine will DMA from.  Pageable
  // memory would mean a memcpy into staging first, per frame, for nothing.
  std::vector<std::uint16_t> in16;
  std::vector<std::uint16_t> out16;
  std::uint16_t* in16_pin = nullptr;
  std::uint16_t* out16_pin = nullptr;
  ~Batch() {
    if (in16_pin != nullptr) CudaFreePinned(in16_pin);
    if (out16_pin != nullptr) CudaFreePinned(out16_pin);
  }
  // Owns raw pinned pointers, so copying would double-free them and the implicitly
  // generated move (which a declared destructor suppresses) would do the same.  Both
  // are written out rather than defaulted for exactly that reason.
  Batch() = default;
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;
  Batch(Batch&& o) noexcept { *this = std::move(o); }
  Batch& operator=(Batch&& o) noexcept {
    pixels = std::move(o.pixels);
    in16 = std::move(o.in16);
    out16 = std::move(o.out16);
    in16_pin = o.in16_pin;
    out16_pin = o.out16_pin;
    o.in16_pin = nullptr;
    o.out16_pin = nullptr;
    frames = o.frames;
    eof = o.eof;
    raw_ready = o.raw_ready;
    return *this;
  }
  std::uint16_t* in() { return in16_pin ? in16_pin : in16.data(); }
  std::uint16_t* out() { return out16_pin ? out16_pin : out16.data(); }
  int frames = 0;
  bool eof = false;
  // Set by the GPU worker when the device wrote the output buffer itself, so the
  // writer must not convert the float buffer over the top of it.  The CPU worker
  // leaves this clear because its output is still float4.
  bool raw_ready = false;
};

class Pipeline {
 public:
  Pipeline(const VideoOptions& opt, const VideoInfo& info, const Palette& pal,
           const ColorTree& tree)
      : opt_(opt), info_(info), palette_(pal), tree_(tree) {
    const std::size_t pixels =
        static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height);
    const std::size_t frame_bytes = pixels * 4 * sizeof(std::uint16_t);
    batch_ = std::max(1, opt.batch_frames);

    // A batch slot holds up to three buffers: the float4 the host dither path works
    // on, the rgba64le going to the device, and the rgba64le coming back.  On the pure
    // GPU path the float one is never touched -- the device widens on upload and emits
    // uint16 on download -- so it is not allocated, which is 16 bytes per pixel saved
    // per slot (531 MiB at 1080p and 16 frames).  The budget below has to count
    // exactly what is allocated, or the queue will be sized against a number the
    // process does not honour, which is the failure mode this whole budget exists to
    // prevent.
    //
    // It is needed whenever *any* host path can run, not just when the GPU is off:
    // host workers dither from float4, so `--cpu-threads 1` alongside a GPU still
    // dereferences this buffer.  Omitting that case leaves it empty and hands the
    // worker a null pointer.
    float_path_ = !(opt.use_gpu && !opt.gpu_float_out) || opt.cpu_threads > 0;

    // ---- RAM budget --------------------------------------------------------
    // The queue is the only allocation this budget governs, and it is not the only
    // memory the process uses.  Three things sit outside it, and the first two
    // cannot be paged back out if the machine runs short:
    //
    //   * the CUDA pinned staging (cudaHostAlloc, one float4 buffer per GPU
    //     worker), which by definition is not swappable;
    //   * ffmpeg's own decoder, which grows with its thread count and with the
    //     frame size;
    //   * Windows and everything else on the machine.
    //
    // All three are subtracted below.  Previously the share was of *total* RAM,
    // which over-commits as soon as anything else is running: on a 16 GiB machine
    // with 3 GiB free the budget still claimed 5.3 GiB, so the queue would be paged
    // out mid-render or refused outright.  It is now the lesser of that share and
    // what is actually available, and it never aims to fill the machine.
    double fraction = opt.mem_fraction > 0.0 ? opt.mem_fraction : 0.33;
    if (fraction > 1.0) {
      std::fprintf(stderr,
                   "warning: --mem-fraction %.3f exceeds 1; clamping to 1\n",
                   fraction);
      fraction = 1.0;
    }
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    std::size_t budget = 0;
    std::size_t avail = 0;
    const std::size_t pinned =
        static_cast<std::size_t>(pixels) * static_cast<std::size_t>(batch_) *
        sizeof(RgbaF) *
        static_cast<std::size_t>(opt.gpu_workers > 0 ? opt.gpu_workers : 1);
    constexpr std::size_t kDecoderHeadroom = 768ull << 20;
    constexpr std::size_t kKeepFree = 1536ull << 20;
    const std::size_t reserve = pinned + kDecoderHeadroom;
    if (GlobalMemoryStatusEx(&status)) {
      const std::size_t share = static_cast<std::size_t>(
          static_cast<double>(status.ullTotalPhys) * fraction);
      avail = static_cast<std::size_t>(status.ullAvailPhys);
      budget = share < avail ? share : avail;
      // Never aim to leave less than kKeepFree for the OS.  A queue that fills all
      // of RAM is what turns a slow render into a crashed machine.
      const std::size_t usable = avail > kKeepFree ? avail - kKeepFree : 0;
      if (budget > usable) budget = usable;
      budget = budget > reserve ? budget - reserve : 0;
    } else {
      budget = 2ull << 30;
    }
    // in16 + out16 always; pixels only on the host dither path.  See float_path_.
    const std::size_t per_batch_bytes =
        (float_path_ ? sizeof(RgbaF) : 0) + 2 * sizeof(std::uint16_t) * 4;
    const std::size_t per_batch = pixels * batch_ * per_batch_bytes;
    std::size_t ram_batches = per_batch ? budget / per_batch : 2;
    // The floor of two is a liveness requirement -- one slot in the dither and one
    // in the writer -- not a throughput choice.  It is allowed to exceed the budget,
    // because a pipeline that cannot run is worse than one that pages, but it is
    // reported when it happens.
    if (ram_batches < 2) {
      if (per_batch * 2 > budget) {
        std::fprintf(stderr,
                     "warning: two batches need %.0f MiB but the RAM budget is "
                     "%.0f MiB; the pipeline will page or fail\n",
                     static_cast<double>(per_batch * 2) / (1024.0 * 1024.0),
                     static_cast<double>(budget) / (1024.0 * 1024.0));
      }
      ram_batches = 2;
    }
    ram_budget_ = budget;
    per_batch_ = per_batch;
    reserved_ = reserve;
    // Depth is the lesser of what RAM allows and what the workers can keep busy.
    // RAM alone is the wrong limit: at 4K and batch 16 a 33% budget is many
    // gigabytes, and holding a second of video in flight just adds latency
    // without adding throughput.  VideoProcess trims this against the worker
    // count once it knows it.
    depth_ = static_cast<int>(std::min<std::size_t>(ram_batches, 64));
    avail_phys_ = avail;
  }

  int batch() const { return batch_; }
  int depth() const { return depth_; }
  std::size_t ram_budget() const { return ram_budget_; }
  std::size_t reserved() const { return reserved_; }
  std::size_t avail_phys() const { return avail_phys_; }
  std::size_t per_batch() const { return per_batch_; }
  // Bytes per pixel a slot occupies, EXCLUDING the decoder's input, so the depth is
  // derived from the same number the allocations use.  The input is excluded because
  // it is not per-pixel: planar 4:4:4 is 3 bytes per pixel, and 4:2:0 is 1.5, so it
  // has to be added as a whole-frame quantity.  `in_channels` is the decoder's output
  // width, which is 3 (rgb48le) unless the source carries alpha.
  std::size_t per_batch_bytes() const {
    return (float_path_ ? sizeof(RgbaF) : 0) + 4 * sizeof(std::uint16_t);
  }
  bool float_path() const { return float_path_; }
  // Reports the depth actually allocated.  VideoProcess trims the constructor's
  // RAM-derived value against the worker count, and reading the untrimmed number
  // is how "queue 2" could be printed next to "5316 MiB in flight".
  void set_depth(int depth) { depth_ = depth; }
  std::size_t in_flight_bytes() const { return per_batch_ * depth_; }

  std::vector<Batch>* NewBatch() {
    const std::size_t pixels = static_cast<std::size_t>(info_.width) *
                               static_cast<std::size_t>(info_.height);
    Batch b;
    b.frames = 0;
    // Reserving up front keeps the workers from racing the allocator.
    if (float_path_) b.pixels.resize(static_cast<std::size_t>(batch_) * pixels);
    b.in16.resize(static_cast<std::size_t>(batch_) * pixels * 4);
    b.out16.resize(static_cast<std::size_t>(batch_) * pixels * 4);
    // Not vector<Batch>(1, std::move(b)): Batch owns pinned pointers and so is
    // move-only, and the count/value constructor reaches for copy.
    std::vector<Batch>* v = new std::vector<Batch>();
    v->resize(1);
    (*v)[0] = std::move(b);
    return v;
  }

 private:
  const VideoOptions& opt_;
  const VideoInfo& info_;
  const Palette& palette_;
  const ColorTree& tree_;
  int batch_ = 16;
  int depth_ = 4;
  std::size_t ram_budget_ = 0;
  bool float_path_ = true;
  std::size_t reserved_ = 0;
  std::size_t avail_phys_ = 0;
  std::size_t per_batch_ = 0;
};

bool VideoProcess(const std::string& in, const std::string& out,
                  const VideoOptions& opt, const VideoInfo& info,
                  const Palette& palette, const ColorTree& tree,
                  VideoResult* result, std::string* error) {
  std::string ffmpeg;
  if (!VideoFindTools(&ffmpeg, nullptr, error)) return false;

  const std::size_t pixels =
      static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height);

  // Decoder output.  "rgba64" is the default and is exact: ffmpeg's yuv420p ->
  // rgba64le and yuv420p -> rgb48le conversions do **not** produce the same RGB
  // (4525709 of 6220800 samples differ on the bench clip), so the narrower
  // interleaved format is not interchangeable and stays rejected.
  //
  // "yuv444" is a different trade and is opt-in: 3 bytes per pixel instead of 8, with
  // the YCbCr->RGB conversion done in the gather kernel.  Same reasoning against
  // rgb48le does not apply, because the difference there was a rounding difference in
  // what is otherwise the same algorithm.  Here the algorithm is genuinely different
  // -- 4:2:0 carries half-resolution chroma that swscale interpolates, 4:4:4 carries
  // the real thing -- so the output is *sharper* at edges and near-identical in flat
  // areas.  Measured end to end; see the README.
  const bool use_yuv444 = opt.input_mode == "yuv444" || opt.input_mode == "yuv444-prepass";
  // 4:2:0 needs no chroma reconstruction by ffmpeg at all: the decoder emits the
  // source format and the device does the interpolation.  That is the smallest input
  // the pipeline can carry (1.5 bytes per pixel) and the only mode where swscale's
  // colour conversion leaves the CPU completely.
  const bool use_yuv420 = opt.input_mode == "yuv420";
  // Refuse rather than guess.  The 4:2:0 gather reads Y at full size and Cb/Cr at a
  // quarter of it, indexed by x>>1 / y>>1, so it is only correct for a source that
  // really is 4:2:0 and really is even in both axes.  Handed anything else -- 4:2:2,
  // 4:4:4, or an odd width -- it reads the wrong bytes and produces plausible-looking
  // garbage, which is precisely the failure this codebase has already been bitten by
  // twice (the 4:4:4 plane stride, and the input/output size collision).  A hard
  // error costs the user one line of text; silent corruption costs a render nobody
  // notices until the clip is finished.
  if (use_yuv420) {
    if (info.pix_fmt != "yuv420p") {
      *error = "[video] --input-mode yuv420 needs a 4:2:0 source; this one is " +
               info.pix_fmt + ". Use yuv444 instead.\n";
      return false;
    }
    if ((info.width & 1) || (info.height & 1)) {
      *error = "[video] --input-mode yuv420 needs even width and height, because\n"
               "        4:2:0 chroma is subsampled 2x; this one is " +
               std::to_string(info.width) + "x" + std::to_string(info.height) +
               ". Use yuv444 instead.\n";
      return false;
    }
  }
  const int dec_channels = use_yuv444 ? 3 : 4;
  const char* dec_pix_fmt = use_yuv420 ? "yuv420p"
                                       : use_yuv444 ? "yuv444p" : "rgba64le";
  // Input and output frame sizes are NOT the same and must not share a variable.
  // Reading is 3 bytes per pixel in planar 4:4:4, but writing is always 4 uint16 per
  // pixel, because that is what the scatter emits and what the encoder is told to
  // expect.  Sharing one name meant the writer handed the encoder 37.5% of each
  // frame's bytes, it desynchronised, and a 605-frame render came out as 226 -- with
  // an "encode" time that had collapsed to 2.5 s and a throughput number that looked
  // like a win.  Both symptoms, one wrong constant.
  const std::size_t in_frame_bytes =
      use_yuv420 ? (pixels * 3) / 2
                 : use_yuv444 ? pixels * 3
                              : pixels * static_cast<std::size_t>(dec_channels) *
                                    sizeof(std::uint16_t);
  // What the encoder is fed.  Planar 8-bit 4:4:4 by default: the device does the
  // RGB->YUV, which is 3 bytes per pixel against rgba64le's 8, and ffmpeg then has
  // nothing to convert.  Both halves matter -- the encode stage was measured blocked
  // at 597 MB/s, under the pipe's own 0.86 GB/s, so the host conversion rather than
  // the transport was the limit, and shrinking the pipe alone would not have helped.
  // RD_YUV444_OUT=0 goes back to rgba64le for comparison.
  const char* yuv_out_env = std::getenv("RD_YUV444_OUT");
  const bool out_yuv444 = yuv_out_env == nullptr || yuv_out_env[0] != '0';
  // Reading is 3 bytes per pixel in planar 4:4:4, writing is 4 uint16 per pixel for
  // rgba64le or 3 bytes for planar 4:4:4.  Input and output frame sizes are NOT the
  // same and must not share a variable: sharing one meant a 605-frame render came out
  // as 226, with an "encode" time that had collapsed to 2.5 s and a throughput number
  // that looked like a win.  Both symptoms, one wrong constant.
  const std::size_t out_frame_bytes =
      out_yuv444 ? pixels * 3 : pixels * 4 * sizeof(std::uint16_t);

  Child decoder;
  // Segment start.  Deliberately *not* an input seek: `-ss` before -i snaps to a
  // keyframe, and an H.264 keyframe can be hundreds of frames early, which would
  // make consecutive segments overlap and silently duplicate frames.  So decode
  // from the beginning and let the filter drop everything before the segment --
  // exact by construction.  The cost is one decode of the skipped prefix, which is
  // ~25 ms/frame against ~72 ms/frame of dithering, so a resume costs a fraction
  // of a full render rather than a wrong file.
  // Thread budget.  Both ffmpeg processes used to be started with `-threads 0`,
  // which on this machine means 12 threads each on 6 physical cores, alongside
  // rdither's reader, writer and dither threads.  That is oversubscribed by roughly
  // three times over, and it is why no stage ever looks saturated while the wall
  // time stays well above the slowest of them: the threads are all waiting on cores.
  // 0 means "decide automatically", which is what `-threads 0` does; a positive
  // value caps it.  Split the 12 logical processors between decode and encode
  // rather than letting each take all of them.
  auto threads_arg = [](int n) {
    return n > 0 ? "-threads " + std::to_string(n) + " " : std::string();
  };
  // Passthrough frame timing is not optional here, and the two palette decoders
  // learned that the hard way: with a `select` filter and ffmpeg's default
  // frame-rate conversion, the output is the *first* N frames rather than the
  // selected ones, so a segment filter silently yields the wrong frames (or
  // none).  It is a no-op when no filter is in play, and the segment path is the
  // only place one is, which is why it went unnoticed there.
  //
  // Spelled and placed through FpsMode: ffmpeg 9 deleted -vsync, and this command
  // carried it in the input position, so after the upgrade the reader would have
  // emitted zero bytes too -- just later, behind the palette stage's error.
  // Hardware decode, when it is available and wanted.  ffmpeg still does the colour
  // conversion on the CPU, so the bytes the dither receives are unchanged -- H.264
  // reconstruction is normative, so the YUV is identical either way.  See the note on
  // VideoOptions::hwaccel for the verification and the corrupt-stream caveat.
  //
  // Deliberately not gated on the dither engine: NVDEC is an independent block, and it
  // helps just as much on the host dither path.
  const bool want_nvdec = opt.hwaccel != 0 && CudaAvailable();
  const std::string hwaccel_arg = want_nvdec ? "-hwaccel cuda " : "";
  const FpsModeArgs& fps = FpsMode(ffmpeg);
  std::string dec_args = "-v error" + std::string(fps.input) + " " +
                         threads_arg(opt.decode_threads) + hwaccel_arg + "-i " + Quote(in);
  if (opt.start_frame > 0) {
    char filt[96];
    // No shell-style quotes: they reach ffmpeg as literal characters and the
    // filter fails to parse, so the decoder emits nothing.
    std::snprintf(filt, sizeof(filt), " -vf select=gte(n\\,%lld)",
                  static_cast<long long>(opt.start_frame));
    dec_args += filt;
  }
  // -avioflags direct: write each frame straight to the pipe instead of staging it
  // through ffmpeg's own AVIO buffer first.  On a pipe that buffer is pure overhead --
  // an extra copy into a buffer that the OS then copies again -- and the reader is the
  // pipeline's floor.  It is a transport flag: it does not touch the conversion, so not
  // a single pixel value changes.
  dec_args += fps.output;
  dec_args += std::string(" -f rawvideo -pix_fmt ") + dec_pix_fmt +
              " -avioflags direct -";
  if (!decoder.Start(ffmpeg, dec_args, true, false, error, "video-decode")) return false;

  Child encoder;
  char rate[64];

  // Measure the source's real frame timing before choosing the rate.  This is a
  // demux pass over the container's packet index, not a decode, and it runs on
  // every input including constant-rate ones -- where it establishes that the rate
  // is constant, which is the cheapest possible answer to "is this VFR".
  FrameTiming timing;
  double audio_duration = -1.0;
  {
    std::string ffmpeg_bin;
    std::string ffprobe;
    std::string ignored;
    if (VideoFindTools(nullptr, &ffprobe, &ignored)) {
      std::string terr;
      if (!ProbeFrameTiming(in, ffmpeg_bin, ffprobe, &timing, &terr)) {
        // A timing probe that fails must not fail the render.  Fall back to the
        // previous behaviour and say so, because "silently produced a file of the
        // wrong length" is precisely the bug being fixed.
        std::fprintf(stderr,
                     "[video] warning: could not read frame timing (%s); falling back"
                     " to the container's declared rate, which makes a variable-frame"
                     " -rate output the wrong length.\n",
                     terr.c_str());
      }
      // The audio length decides whether -shortest is safe, and it is a second
      // cheap query on a file already being statted.
      audio_duration = ProbeAudioDuration(in, ffprobe);
    }
  }

  // The rate the dithered frames are FED at is the source's true average, not
  // r_frame_rate.  For a constant-rate source the two are identical and this
  // changes nothing; for a variable one, r_frame_rate is the maximum instantaneous
  // rate, and using it is what made the output 1.86x too short.
  if (timing.ok && timing.avg_fps > 0.0) {
    std::snprintf(rate, sizeof(rate), "%.6f", timing.avg_fps);
  } else {
    std::snprintf(rate, sizeof(rate), "%.6f", info.fps);
  }

  if (timing.ok && timing.variable) {
    std::fprintf(stderr,
                 "[video] variable frame rate: %d frames in %d constant-rate runs,"
                 " %.4f s total, %.3f fps average (the container declares %s).\n",
                 timing.frames, timing.runs, timing.total, timing.avg_fps,
                 TimingDecimal(info.fps).c_str());
    // Saying which of the two behaviours is in force matters more than the numbers:
    // a VFR source rendered at CFR looks identical frame-for-frame, so nothing about
    // the output tells the user which one they got.
    if (!opt.preserve_vfr) {
      std::fprintf(stderr,
                   "[video]   re-emitted at the true average, %.3f fps.  Duration and"
                   " A/V sync match the source; per-frame motion cadence is flattened."
                   " --video-preserve-vfr asks for the per-frame timing to be kept"
                   " (not verified end to end).\n", timing.avg_fps);
    } else if (!timing.setpts.empty()) {
      std::fprintf(stderr,
                   "[video]   restoring per-frame timing at the encoder, timebase"
                   " 1/%d.\n", timing.timescale);
    }
    if (!timing.note.empty()) {
      std::fprintf(stderr, "[video]   note: %s\n", timing.note.c_str());
    }
  }

  // Resolve --video-lossless, then say plainly when the chosen output cannot
  // carry a 16-colour palette.  This is the single most common way a correct
  // dither ends up looking "dull and grey": the palette is fine and the encoder
  // throws it away.  Measured on 1080p, 16 colours: ffv1 yuv444p -> 56 unique
  // colours, libx264 yuv444p crf 12 -> 15191, libx264 yuv444p -> 30445,
  // libx264 yuv420p -> 36031.
  const std::string codec = opt.lossless ? "ffv1" : opt.codec;
  const std::string pix_fmt = opt.lossless ? "yuv444p" : opt.pix_fmt;
  const bool chroma_subsampled =
      pix_fmt.find("420") != std::string::npos ||
      pix_fmt.find("422") != std::string::npos || pix_fmt.find("411") != std::string::npos;
  const bool lossless_codec = codec.find("ffv1") != std::string::npos ||
                              codec.find("ffvhuff") != std::string::npos ||
                              codec.find("utvideo") != std::string::npos ||
                              codec.find("prores") != std::string::npos ||
                              codec.find("v210") != std::string::npos;
  if (chroma_subsampled || !lossless_codec) {
    // Two distinct failures with very different numbers, so do not quote one for
    // the other: subsampling destroys the per-pixel alternation outright, while
    // lossy compression merely invents colours around the transitions.
    std::fprintf(stderr,
                 "[video] warning: %s / %s will not deliver a %d-colour palette.\n",
                 codec.c_str(), pix_fmt.c_str(), opt.colors);
    if (chroma_subsampled) {
      std::fprintf(stderr,
                   "[video]          chroma is subsampled, which averages colour"
                   " over 2x2 blocks and so averages away the dither itself"
                   " (measured 36356 unique colours from 16). Use yuv444p.\n");
    } else {
      std::fprintf(stderr,
                   "[video]          lossy compression, which invents colours"
                   " around every transition (measured 30691 unique colours from"
                   " 16 at yuv444p crf 18). Expect a flatter look than the dither.\n");
    }
    std::fprintf(stderr,
                 "[video]          --video-lossless (ffv1 yuv444p) is the only"
                 " setting that keeps it exactly, at ~28x the bitrate.\n");
  }

  // Audio.  The dither only touches the video stream, so the original's audio is
  // carried across untouched rather than dropped on the floor.
  //
  // It has to be a *second input* to the encoder, not a filter: the encoder reads
  // raw frames on stdin and the audio lives in the source container, so ffmpeg needs
  // the container named as input in order to reach it.  That is why the -i for the
  // audio is the source path again, placed after the stdin input, and why
  // -map 0:v is not used -- stream 0 of the *encoder's* inputs is the raw stdin
  // video, and the audio is stream 1.
  //
  // -c:a copy rather than a re-encode: transcoding would add a second lossy stage
  // for no reason, and would cost more CPU on a pipeline where the encoder is
  // already ~89% of the total (see the README, round twenty-two).  If the source
  // carries no audio, the second input is a normal ffmpeg error, so its presence is
  // probed first -- silently skipping it would mean never knowing whether a track
  // was lost.
  const bool has_audio = opt.audio && VideoHasAudio(in, ffmpeg, error);
  if (opt.audio && !has_audio) {
    std::fprintf(stderr, "[video] no audio stream found in the input; writing video only.\n");
  }
  if (has_audio) {
    std::fprintf(stderr, "[video] copying audio from the source (stream copy, no re-encode).\n");
  }

  std::string enc_args =
      "-v error -y " + threads_arg(opt.encode_threads) + "-f rawvideo -pix_fmt " +
      (out_yuv444 ? "yuv444p" : "rgba64le") + " -s " +
      std::to_string(info.width) + "x" + std::to_string(info.height) + " -r " +
      rate + " -i -";
  if (has_audio) enc_args += " -i " + Quote(in);
  // Variable frame rate: restore the source's own timing to the dithered frames.
  //
  // The pipe carries pixels and nothing else, so the timing has to be put back here.
  // For a constant-rate source `timing.variable` is false and NOTHING below runs --
  // no extra ffmpeg pass beyond the probe, no filter, no change to the command line,
  // which is the only acceptable outcome for a fix aimed at a rare input format.
  if (opt.preserve_vfr && timing.ok && timing.variable && !timing.setpts.empty()) {
    // settb FIRST: setpts works in its input's timebase, and for a rawvideo input
    // that is 1/rate.  Frames closer together than that collapse onto one tick and
    // ffmpeg drops them, which shortens the timeline without any error.  Raise the
    // timebase to something the shortest frame can be expressed in, then assign
    // timestamps.
    enc_args += " -vf settb=expr=1/" + std::to_string(timing.timescale) +
                ",setpts='" + timing.setpts + "'/TB";
    // passthrough, not vfr: vfr lets the muxer re-derive durations and it will
    // re-derive them from the input rate, discarding what setpts just computed.
    enc_args += " -fps_mode passthrough";
    enc_args += " -video_track_timescale " + std::to_string(timing.timescale);
  }

  // -map 0:v pins the dithered video to output 0 and, crucially, stops ffmpeg's
  // default stream selection from guessing: without it ffmpeg would pick one video
  // and one audio by type ranking, which is how a source with a cover-art or
  // data track ends up as the video.  -map 1:a? then carries every audio track.
  enc_args += " -map 0:v";
  if (has_audio) enc_args += " -map 1:a?";
  enc_args += " -c:v " + codec;
  // ffv1 takes neither -crf nor -preset; passing them is at best ignored.
  if (!lossless_codec) {
    enc_args += " -crf " + std::to_string(opt.crf) + " -preset " + opt.preset;
  }
  // -shortest stops output at the shorter of the two streams.  That is the right
  // tool for ONE direction and actively harmful in the other, and the difference is
  // measured here rather than assumed:
  //
  //   * Audio LONGER than the video: the trailing audio is what should go, and
  //     -shortest removes it.  This is what the flag was originally added for.
  //
  //   * Video longer than the audio: -shortest DELETES VIDEO FRAMES to make the
  //     tracks match.  Measured on a 30 s excerpt of a phone capture, where the
  //     source has 895 video frames against 30.0128 s of audio: the output came out
  //     with 892.  Three frames deleted, every run, and the loss scales with the
  //     clip -- on the 172 s original that is a visible cut at the tail and the
  //     audio ending early.
  //
  // The asymmetry is not a subtlety.  Trailing video with no audio is handled by
  // every player and costs nothing; deleted frames are frames the user asked to have
  // dithered, gone.  So -shortest is used only where it trims audio.
  if (has_audio) {
    const double audio_s = audio_duration;
    const double video_s = timing.ok ? timing.total : 0.0;
    const bool audio_overshoots = (audio_s > 0.0 && video_s > 0.0 && audio_s > video_s);
    if (audio_overshoots) {
      enc_args += " -c:a copy -shortest";
    } else {
      enc_args += " -c:a copy";
      if (audio_s > 0.0 && video_s > 0.0 && video_s > audio_s) {
        const double gap = video_s - audio_s;
        const double frames = gap * (timing.avg_fps > 0.0 ? timing.avg_fps : info.fps);
        std::fprintf(stderr,
                     "[video] source video is %.4f s but its audio is only %.4f s;"
                     " not using -shortest, which would delete %.1f video frame%s"
                     " to match. The output keeps every frame and ends %.0f ms after"
                     " the audio.\n",
                     video_s, audio_s, frames, frames >= 1.5 ? "s" : "",
                     gap * 1000.0);
      }
    }
  }
  enc_args += " -pix_fmt " + pix_fmt + " " + Quote(out);
  if (!encoder.Start(ffmpeg, enc_args, false, true, error, "video-encode")) return false;

  Pipeline pipe(opt, info, palette, tree);
  const int batch = pipe.batch();

  // Worker count first: the queue depth depends on it, and the slot array has to
  // be sized after the depth is known.
  int cpu_threads = opt.cpu_threads;
  const int hw = static_cast<int>(std::thread::hardware_concurrency());
  // Threads for the reader's uint16 -> float widening.  This loop is the reader's
  // dominant cost and it feeds the GPU, so it is worth more than the writer's
  // equivalent (which only runs on the CPU dither path, off by default).  Capped at
  // the physical cores rather than the logical ones: the extra six are SMT siblings
  // and mostly help ffmpeg, which is competing for the same silicon.
  const int reader_convert_threads =
      opt.reader_threads > 0
          ? opt.reader_threads
          : std::max(1, std::min(6, hw > 2 ? hw / 2 : hw));
  // OpenCL is a first-class GPU engine here, not a fallback: it is the same
  // partition and the same arithmetic as the CUDA blocks engine, so a frame
  // dithered on either is the same frame.  Availability is asked of whichever
  // engine was selected, and the failure is reported rather than papered over by
  // quietly falling back to CUDA -- a silent fallback would make `--engine opencl`
  // appear to work while measuring the wrong thing.
  const bool want_opencl = opt.gpu_engine == VideoOptions::GpuEngine::kOpenCL;
  const bool use_gpu = opt.use_gpu &&
                       (want_opencl ? OpenCLAvailable(nullptr, nullptr) : CudaAvailable());
  if (opt.use_gpu && !use_gpu) {
    if (want_opencl) {
      std::string why;
      OpenCLAvailable(nullptr, &why);
      std::fprintf(stderr, "error: --engine opencl requested for video but %s\n",
                   why.c_str());
      return false;
    }
  }
  // Independent GPU workers, each with its own device state.  Measured as no
  // faster than one (see the note on the option), so one is the default; the pool
  // stays because the negative result should be re-checkable on other hardware.
  int gpu_workers = opt.gpu_workers;
  if (gpu_workers < 1) gpu_workers = 1;
  // Clamp to whichever engine's slot count, not the CUDA one: asking OpenCL for a
  // third worker would hand the third worker slot 1, which works but silently
  // serialises it against the second.  Clamping says so.
  const int gpu_state_slots =
      want_opencl ? rd::OpenCLStateCount() : kGpuStateSlots;
  if (gpu_workers > gpu_state_slots) gpu_workers = gpu_state_slots;
  if (!use_gpu) gpu_workers = 0;
  if (cpu_threads < 0) {
    // Auto.  With a GPU present the GPU is the engine and the host pool is off:
    // the host block walk costs ~937 ms/frame at 1080p against the GPU's 63.5, and
    // measured on a deep queue the pool still costs more than it adds (115 ms/frame
    // for the GPU alone versus 219-244 ms/frame with ten host workers), because the
    // workers saturate the memory bandwidth the GPU's data path needs.  So "auto"
    // means GPU-only when there is a GPU, and every core but two when there is not.
    cpu_threads = use_gpu ? 0 : std::max(0, hw - 2);
  } else {
    cpu_threads = std::min(cpu_threads, std::max(0, hw));
  }

  // A worker with no slot to work on is worth nothing, so a slot is the unit of
  // currency: 1080p at batch 16 is 759 MiB per slot, and at a 1/3-of-16 GB budget
  // only seven slots exist.  Asking for ten host workers plus the GPU therefore
  // cannot work -- the extra workers just block on the free-slot queue while the
  // GPU starves for want of a batch.
  //
  // So the queue is sized first, and the worker count is trimmed to fit it.  A
  // deep queue with fewer workers beats a shallow queue with more, and this way
  // the two can never disagree.
  // The slot's *input* buffer is dec_channels wide, not always 4: the decoder is asked
  // for rgb48le when the source has no alpha, which is 6 bytes per pixel instead of 8.
  // Budgeting it at 4 would over-reserve by a third and quietly cost a slot.
  const std::size_t per_batch =
      pixels * static_cast<std::size_t>(batch) * pipe.per_batch_bytes() +
      static_cast<std::size_t>(batch) * in_frame_bytes;
  int depth;
  {
    std::size_t affordable = per_batch ? pipe.ram_budget() / per_batch : 2;
    if (affordable < 2) affordable = 2;
    if (affordable > 64) affordable = 64;
    depth = static_cast<int>(affordable);

    // One batch per worker, so no worker ever waits on the reader.  This is the
    // fix for the reported stall: converting "frames of demand" into batches (as
    // ceil(workers*2/batch) does) is wrong, because a slot already *is* a batch.
    // The floor is 3 rather than 2 because a 2-slot pipeline also couples the
    // reader to the writer: whenever the writer holds both slots the GPU goes
    // hungry, which costs about 25% at 1080p.
    const int want_workers = cpu_threads + (use_gpu ? gpu_workers : 0);
    if (depth > want_workers) depth = std::max(3, want_workers);

    // An explicit depth replaces that heuristic entirely, because "one batch per
    // worker" is a throughput rule and says nothing about how much buffering the
    // pipeline needs.  Measured: at the derived depth of 3 the GPU worker sat idle
    // 47% of its stage time waiting for the reader, which no amount of worker
    // count can fix -- the slack has to live in slots.  Still capped by what the RAM
    // budget affords, so this cannot overcommit.
    if (opt.queue_depth >= 2) {
      depth = std::min(opt.queue_depth, static_cast<int>(affordable));
    }

    // Trim the host pool to the slots that actually exist, keeping one per GPU
    // worker when it is enabled.
    const int keep_for_gpu = use_gpu ? gpu_workers : 0;
    const int max_host = std::max(0, depth - keep_for_gpu);
    if (cpu_threads > max_host) cpu_threads = max_host;
    if (depth < 2) depth = 2;
  }

  // Free pool of batch slots, sized after the depth is known.
  pipe.set_depth(depth);
  std::vector<Batch> slots(static_cast<std::size_t>(depth));
  for (Batch& b : slots) {
    // The float4 buffer is only allocated when a host path will use it; see
    // Pipeline::float_path_ and the note there on why the budget has to agree.
    if (pipe.float_path()) b.pixels.resize(static_cast<std::size_t>(batch) * pixels);
    if (use_gpu && !opt.gpu_float_out) {
      // Page-locked, so the reader writes straight into memory the copy engine DMAs
      // from.  These bytes cannot be paged out, and per_batch_bytes already counts
      // them, so the budget and the allocations still agree.
      //
      // The input is dec_channels wide, or 3 bytes per pixel in planar 4:4:4, or 1.5
      // in planar 4:2:0 -- hence in_frame_bytes as a whole-frame figure rather than a
      // per-pixel one.  The output is always 4 uint16, because that is what the
      // encoder consumes and what the scatter writes.
      const std::size_t in_bytes = static_cast<std::size_t>(batch) * in_frame_bytes;
      const std::size_t out_bytes =
          static_cast<std::size_t>(batch) * out_frame_bytes;
      b.in16_pin = static_cast<std::uint16_t*>(CudaAllocPinned(in_bytes));
      b.out16_pin = static_cast<std::uint16_t*>(CudaAllocPinned(out_bytes));
    } else {
      b.in16.resize(static_cast<std::size_t>(batch) * pixels *
                    static_cast<std::size_t>(dec_channels));
      b.out16.resize(static_cast<std::size_t>(batch) * pixels * 4);
    }
  }
  std::deque<int> free_slots;
  for (int i = 0; i < depth; ++i) free_slots.push_back(i);
  std::deque<int> ready;
  std::deque<int> done;

  std::mutex mu;
  std::condition_variable cv_free, cv_ready, cv_done;
  bool eof = false;
  std::int64_t frames_read = 0, frames_dithered = 0;
  // Frames this segment has read and written, as opposed to dithered.  The boundary
  // has to be judged on what was *written*: the writer used to test
  // `frames_dithered >= segment_size`, which is true the moment a batch lands, so it
  // broke out of its loop and threw away the very batch it had just waited for.  Every
  // segment came out as a bare container header.
  int segment_read = 0;
  std::int64_t segment_written = 0;
  double decode_ms = 0.0, dither_ms = 0.0, encode_ms = 0.0;
  double convert_ms = 0.0, pipe_ms = 0.0;
  // Leave the reader a core; the writer is the only other busy host thread here.
  const int writer_convert_threads =
      std::max(1, std::min(3, static_cast<int>(std::thread::hardware_concurrency()) - 2));
  std::int64_t cpu_frames = 0, gpu_frames = 0;
  std::string gpu_error;
  bool failed = false;

  DitherParams params;
  params.colors = opt.colors;
  params.diffusion = opt.diffusion;
  params.use_cache = false;

  if (!opt.quiet) {
    std::fprintf(stderr,
                 "[video] %dx%d  batch=%d  queue=%d (%.0f MiB of %.0f MiB RAM)  "
                 "cpu_workers=%d  gpu=%s x%d\n",
                 info.width, info.height, batch, depth,
                 static_cast<double>(pipe.in_flight_bytes()) / (1024.0 * 1024.0),
                 static_cast<double>(pipe.ram_budget()) / (1024.0 * 1024.0),
                 cpu_threads, use_gpu ? "yes" : "no", gpu_workers);
    // Print the ceiling, not just the plan.  The queue is the part the budget
    // governs; the reserve is pinned CUDA staging plus ffmpeg's decoder, and the
    // sum is the real peak.  This is the number to check before raising
    // --mem-fraction or --queue-depth on a machine that is already busy.
    {
      const double queue_mb =
          static_cast<double>(pipe.in_flight_bytes()) / (1024.0 * 1024.0);
      const double reserve_mb =
          static_cast<double>(pipe.reserved()) / (1024.0 * 1024.0);
      const double avail_mb =
          static_cast<double>(pipe.avail_phys()) / (1024.0 * 1024.0);
      const double ceiling = queue_mb + reserve_mb;
      std::fprintf(stderr,
                   "[ram]    %.0f MiB of %.0f MiB free at start; queue %.0f + "
                   "reserve %.0f = ~%.0f MiB peak%s\n",
                   avail_mb, avail_mb, queue_mb, reserve_mb, ceiling,
                   avail_mb > 0.0 && ceiling > avail_mb ? "  ** OVER FREE **"
                                                       : "");
    }
    if (cpu_threads > 0) {
      std::fprintf(stderr,
                   "[video] note: the %d host worker(s) use the host block walk, "
                   "which is NOT bit-identical to the verified CUDA walk -- "
                   "measured ~1.6%% of pixels differ (visually indistinguishable). "
                   "For bit-exact output use --cpu-threads 0.\n",
                   cpu_threads);
    }
  }

  // ---- reader -------------------------------------------------------------
  std::thread reader([&]() {
    try {
      for (;;) {
        int slot = -1;
        {
          std::unique_lock<std::mutex> lock(mu);
          // eof belongs in the predicate: the writer can stop on a segment's
          // frame limit while the reader is blocked here with every slot held by a
          // worker whose result will now never be consumed.  Without it,
          // reader.join() waits forever.
          cv_free.wait(lock, [&] {
            return !free_slots.empty() || failed || eof;
          });
          if (failed || eof) return;
          if (Interrupted()) return;
          slot = free_slots.front();
          free_slots.pop_front();
        }
        Batch& b = slots[static_cast<std::size_t>(slot)];
        int filled = 0;
        // A segment stops at its own frame count, and the reader is the only place
        // that can enforce it.  Without this the reader decodes to end-of-file: a
        // 15-frame segment pulled 30 frames, which both wasted the decode and left
        // the decoder to be torn down mid-stream, so ffmpeg exited with a crash code
        // and the run reported failure.
        const int want =
            opt.frames_in_segment > 0
                ? std::min<int>(batch,
                                static_cast<int>(opt.frames_in_segment -
                                                 segment_read))
                : batch;
        for (; filled < want; ++filled) {
          const double d0 = NowMs();
          const std::size_t got = decoder.Read(
              reinterpret_cast<std::uint8_t*>(b.in()) +
                  static_cast<std::size_t>(filled) * in_frame_bytes,
              in_frame_bytes);
          const double dt = NowMs() - d0;
          { std::lock_guard<std::mutex> lock(mu); decode_ms += dt; }
          if (got != in_frame_bytes) break;
        }
        segment_read += filled;
        // The dither engine works on RgbaF, but the pipe carries rgba64le.  On the
        // GPU path this widening happens on the device now, so the reader stops
        // here -- which is the whole point: the loop was the reader's dominant cost
        // and it was bandwidth-bound, not compute-bound.
        //
        // The two host paths still need it.  Both are exact and lossless: Q16
        // Quantum is a float, and the decoder's uint16 maps onto it exactly.  Skipping
        // it where it *is* required leaves b.pixels at its zero-initialised value and
        // the whole clip comes out black, so the condition is explicit.
        if (!use_gpu || opt.gpu_float_out) {
          RawToFloatsParallel(b.in(), b.pixels.data(), pixels, filled,
                              reader_convert_threads, dec_channels);
        }
        b.frames = filled;
        {
          std::lock_guard<std::mutex> lock(mu);
          frames_read += filled;
          // End of file for this *segment* as well as for the clip: once the segment's
          // frame count is read there is nothing more to pull, and saying so stops the
          // decoder being killed mid-stream at teardown.
          const bool segment_done =
              opt.frames_in_segment > 0 && segment_read >= opt.frames_in_segment;
          if (filled == 0 || segment_done) {
            eof = true;
            free_slots.push_back(slot);
            if (filled > 0) {
              // The batch we just filled still has to be dithered and written.
              ready.push_back(slot);
            }
          } else {
            ready.push_back(slot);
          }
        }
        // The writer sleeps on cv_done, so EOF has to wake it too -- signalling
        // only cv_ready leaves it blocked forever once the last batch is drained.
        cv_ready.notify_one();
        cv_done.notify_all();
        if (filled == 0) return;
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mu);
      failed = true;
      cv_ready.notify_all();
      cv_free.notify_all();
      cv_done.notify_all();
    }
  });

  // ---- dither workers -----------------------------------------------------
  // The GPU is the primary engine: it is ~109 ms/frame here, and letting the host
  // pool grab work first measurably *hurts* (155 ms/frame with 10 host workers
  // plus the GPU, versus 109 with the GPU alone) because the host walk is
  // roughly 8x slower per frame and starves the GPU of memory bandwidth.
  //
  // So host workers wait a grace period before taking a batch: if the GPU has
  // not drained the queue in that time, the GPU is genuinely behind and the host
  // pool absorbs the slack.  This is the "dynamically" part -- the mix adapts to
  // whichever engine is actually the constraint, with no tuning knob.
  std::atomic<std::int64_t> gpu_taken{0};
  std::atomic<int64_t> last_gpu_ms{0};
  const int host_grace_ms = opt.host_grace_ms > 0 ? opt.host_grace_ms : 60;

  // `gpu_index` is the device-state slot this worker owns; host workers pass -1 and
  // never reach the CUDA call.
  auto worker = [&](bool is_gpu, int gpu_index) {
    for (;;) {
      int slot = -1;
      {
        std::unique_lock<std::mutex> lock(mu);
        if (is_gpu) {
          cv_ready.wait(lock, [&] { return !ready.empty() || eof || failed; });
        } else {
          // Grace period: only take work the GPU has left sitting in the queue.
          const int64_t budget = static_cast<int64_t>(host_grace_ms);
          int64_t waited = 0;
          for (;;) {
            if (!ready.empty()) {
              const int64_t since = NowMs() - static_cast<double>(last_gpu_ms.load());
              // The GPU moved recently, so it is keeping up: let it go first.
              if (waited >= budget || since > budget) break;
            } else if (eof || failed) {
              break;
            }
            cv_ready.wait_for(lock, std::chrono::milliseconds(2));
            waited += 2;
            cv_ready.notify_all();  // re-broadcast: cv_ready is signalled, not waited
          }
        }
        if (failed) return;
          if (Interrupted()) return;
        if (ready.empty()) {
          if (eof) return;
          continue;
        }
        slot = ready.front();
        ready.pop_front();
      }
      if (is_gpu) {
        last_gpu_ms.store(static_cast<int64_t>(NowMs()));
        gpu_taken.fetch_add(1);
      }
      Batch& b = slots[static_cast<std::size_t>(slot)];
      const double t0 = NowMs();
      // Whether the device wrote `b.raw` itself.  Only true when the uint16 output
      // was actually requested: if the GPU returns float4 (--gpu-float-out, or any
      // future path that does not convert on the device) the host still has to do
      // it, and claiming otherwise hands the encoder an uninitialised buffer.
      bool raw_written = false;
      if (is_gpu) {
        BlockOptions blocks;
        blocks.block = std::max(16, opt.block);
        blocks.frames = b.frames;
        // The GPU writes the quantised colour straight out as packed rgba64le,
        // which is exactly the byte layout the encoder pipe consumes.  That halves
        // the download *and* removes the host float->uint16 conversion, which was
        // measured at 3649 ms per 605 frames.  It is the same arithmetic
        // FloatsToRaw performs -- double, then float, then a truncating cast to
        // uint16 -- just done on the device.
        blocks.emit_u16 = !opt.gpu_float_out;
        raw_written = blocks.emit_u16;
        // The matching half of the output change: hand the device the rgba64le the
        // reader already has, and let the gather kernel widen it.  This deletes the
        // reader's uint16 -> float loop -- the reader's dominant cost, ~48 GB of
        // memory traffic over 605 frames, and bandwidth-bound rather than
        // compute-bound, which is why parallelising it bought only 0.8 fps -- and
        // halves the H2D.  Exact: a uint16 always fits a float's mantissa, and the
        // host did nothing but static_cast<float>.
        blocks.upload_u16 = !opt.gpu_float_out;
        blocks.in_channels = dec_channels;
        blocks.in_mode =
            opt.input_mode == "yuv444-prepass"
                ? BlockOptions::InMode::PlanarYuv444Prepass
                : use_yuv420 ? BlockOptions::InMode::PlanarYuv420
                             : use_yuv444 ? BlockOptions::InMode::PlanarYuv444
                                          : BlockOptions::InMode::Interleaved16;
        // Page-locked buffers mean the engine can DMA straight from what the reader
        // wrote, with no staging memcpy in between.
        blocks.in_u16_pinned = b.in16_pin != nullptr;
        blocks.emit_yuv444 = out_yuv444;
        blocks.out_u16_pinned = b.out16_pin != nullptr;
        std::string device;
        std::string err;
        if (want_opencl) {
          // The u16 options are set identically to the CUDA branch below, and the
          // engine honours each one only when the matching pointer is also
          // supplied -- so this is the same three-way contract, not a new one.
          err = rd::RiemersmaBlocksOpencl(
              palette, params, tree, info.width, info.height, b.pixels.data(),
              blocks, &device, raw_written ? b.out() : nullptr, gpu_index,
              blocks.upload_u16 ? b.in() : nullptr);
        } else {
          err = rd::RiemersmaBlocksCuda(
              palette, params, tree, info.width, info.height, b.pixels.data(),
              blocks, &device, raw_written ? b.out() : nullptr, gpu_index,
              blocks.upload_u16 ? b.in() : nullptr);
        }
        if (!err.empty()) {
          std::lock_guard<std::mutex> lock(mu);
          if (gpu_error.empty()) gpu_error = err;
          failed = true;
          cv_ready.notify_all();
          cv_free.notify_all();
          cv_done.notify_all();
          return;
        }
      } else {
        RiemersmaBlocksCpu(palette, params, tree, info.width, info.height,
                           b.pixels.data(), b.frames, std::max(16, opt.block));
      }
      const double dt = NowMs() - t0;
      {
        std::lock_guard<std::mutex> lock(mu);
        dither_ms += dt;
        frames_dithered += b.frames;
        b.raw_ready = raw_written;
        if (is_gpu) {
          gpu_frames += b.frames;
        } else {
          cpu_frames += b.frames;
        }
        done.push_back(slot);
      }
      cv_done.notify_one();
    }
  };

  std::vector<std::thread> workers;
  for (int i = 0; i < cpu_threads; ++i) workers.emplace_back(worker, false, -1);
  for (int i = 0; i < gpu_workers; ++i) workers.emplace_back(worker, true, i);

  // ---- writer -------------------------------------------------------------
  const double t_start = NowMs();
  bool write_failed = false;
  // Progress for the whole pipeline, driven from the writer.  The writer is the
  // right thread and not merely the convenient one: `segment_written` counts
  // frames that are in the output file, so the bar cannot outrun the result, and
  // the ETA is an estimate of when the file will be complete.  Driving it from
  // the reader would have been the opposite -- frames decoded, which on a
  // bounded queue runs up to `queue * batch` frames ahead of anything the user
  // could play back.
  //
  // The secondary counter is the decode lead for the same reason it is useful:
  // if `read` is climbing while the bar is stuck, the bottleneck is downstream,
  // and that is a different problem from a stalled decoder.
  //
  // Denominator: a segment knows its own length; a whole clip takes it from the
  // container.  When the container does not say (info.frames <= 0 -- a stream
  // piped in, or a format ffprobe cannot count) there is no total, and the bar
  // degrades to a rate and an elapsed time rather than inventing a denominator.
  //
  // The numerator below is deliberately NOT offset by opt.start_frame.  It was,
  // and the combination is a bug this comment exists to prevent: `run_total` is
  // the *segment's* length when segmenting, so adding the segment's start offset
  // to the count made a 250-frame segment at offset 2000 report 2000/250 and
  // climb to 306% by the last segment.  Measured, not hypothesised.  Both sides
  // of the fraction are "frames in this run", which is also the only thing the
  // reader of the line can act on.
  const std::int64_t run_total =
      opt.frames_in_segment > 0
          ? opt.frames_in_segment
          : (info.frames > 0
                 ? std::max<std::int64_t>(0, info.frames - opt.start_frame)
                 : 0);
  Progress progress("video", run_total, !opt.quiet);
  while (true) {
    int slot = -1;
    {
      std::unique_lock<std::mutex> lock(mu);
      // wait_for, not wait: a 18000-frame job must not be able to hang silently
      // if a stage wedges.  Ten minutes without a single frame is a fault.
      cv_done.wait_for(lock, std::chrono::seconds(600), [&] {
        return !done.empty() || eof || failed;
      });
      if (done.empty() && !eof && !failed) {
        *error =
            "pipeline stalled: no batch completed within 600 s "
            "(check the decoder and encoder processes)";
        std::lock_guard<std::mutex> l2(mu);
        failed = true;
        cv_ready.notify_all();
        cv_free.notify_all();
        break;
      }
      if (failed) break;
      if (Interrupted()) {
        // Unwind through the normal shutdown path so the encoder's stdin is
        // closed and ffmpeg can finalise what it has.
        std::lock_guard<std::mutex> l2(mu);
        failed = true;
        cv_ready.notify_all();
        cv_free.notify_all();
        break;
      }
      // A segment stops at its own frame count, not at end of file.  Stopping
      // here -- rather than letting the decoder run dry -- is what finalises this
      // segment as a complete, independently-valid file, which is the whole basis
      // of crash-safe resuming.
      //
      // Judged on frames *written*, and only after the batch has been written.  The
      // old test was `frames_dithered >= frames_in_segment`, evaluated before taking
      // the batch off the done queue: it fired the instant a batch completed, so the
      // writer discarded the batch it was waiting for and every segment came out as a
      // 585-byte container header.  It is also the wrong variable, because a batch
      // straddling the boundary is written in part.
      if (opt.frames_in_segment > 0 && segment_written >= opt.frames_in_segment) {
        break;
      }
      if (done.empty()) {
        if (eof && frames_dithered >= frames_read) break;
        continue;
      }
      slot = done.front();
      done.pop_front();
    }
    Batch& b = slots[static_cast<std::size_t>(slot)];
    const double e0 = NowMs();
    // Only the host paths still need this.  On the GPU path the scatter kernel wrote
    // `out16` itself, so this would be pure waste -- and it would overwrite correct
    // bytes with a float4 buffer that no longer holds the result.
    if (!b.raw_ready) {
      if (out_yuv444 && !use_gpu) {
        // The host has to do what the device does on the GPU path.  Without this the
        // writer would call FloatsToRaw, hand the encoder rgba64le, and the encoder
        // would read it as planar 4:4:4 -- see FloatsToYuv444 for what that cost.
        FloatsToYuv444Parallel(b.pixels.data(),
                               reinterpret_cast<unsigned char*>(b.out()), pixels,
                               b.frames, writer_convert_threads);
      } else {
        FloatsToRawParallel(b.pixels.data(), b.out(), pixels, b.frames,
                            writer_convert_threads);
      }
    }
    const double e1 = NowMs();
    // A batch that straddles the segment boundary is written in part: the frames past
    // the boundary belong to the next segment, which will decode them again from its
    // own start.  Writing the whole batch would duplicate them.
    int frames_to_write = b.frames;
    if (opt.frames_in_segment > 0) {
      const std::int64_t left = opt.frames_in_segment - segment_written;
      if (left < frames_to_write) frames_to_write = static_cast<int>(std::max<std::int64_t>(0, left));
    }
    const std::size_t bytes =
        static_cast<std::size_t>(frames_to_write) * out_frame_bytes;
    const bool wrote = encoder.Write(b.out(), bytes);
    segment_written += frames_to_write;
    const double dt = NowMs() - e0;
    std::int64_t frames_read_now = 0;
    {
      std::lock_guard<std::mutex> lock(mu);
      encode_ms += dt;
      // Split the writer's cost, because the two halves scale differently: the
      // float->uint16 conversion is host work proportional to pixels, while the pipe
      // write is bounded by ffmpeg.  Knowing which dominates decides whether to
      // move the conversion onto the GPU or to parallelise the encoder.
      convert_ms += (e1 - e0);
      pipe_ms += (NowMs() - e1);
      frames_read_now = frames_read;
      free_slots.push_back(slot);
    }
    cv_free.notify_one();
    if (!wrote) {
      write_failed = true;
      // Tell the reader and the workers to stop, or reader.join() waits for a
      // queue that nobody will ever drain.
      std::lock_guard<std::mutex> lock(mu);
      failed = true;
      cv_ready.notify_all();
      cv_free.notify_all();
      break;
    }
    // After the write succeeded, never before: a bar that counts a batch the
    // encoder then rejected would have told the user the file was further along
    // than it is, in the one case where they most need to know it is not.
    // Both counters are frames within this run, matching run_total above.
    progress.Update(segment_written, "read", frames_read_now);
  }
  progress.Finish();

  {
    std::lock_guard<std::mutex> lock(mu);
    eof = true;
    if (write_failed) failed = true;
  }
  cv_ready.notify_all();
  cv_free.notify_all();
  cv_done.notify_all();
  reader.join();
  for (std::thread& t : workers) t.join();

  encoder.CloseStdin();
  decoder.Close();
  const int encode_code = encoder.Wait();

  if (!gpu_error.empty()) {
    *error = gpu_error;
    return false;
  }
  if (write_failed || encode_code != 0) {
    *error = "ffmpeg encode failed";
    return false;
  }

  if (result != nullptr) {
    result->frames = frames_dithered;
    result->dither_ms = dither_ms;
    result->encode_ms = encode_ms;
  result->convert_ms = convert_ms;
  result->pipe_ms = pipe_ms;
    result->decode_ms = decode_ms;
    // t_start is the pipeline's own start.  Adding palette_ms makes this the time the
  // user actually waited for, which is what "55 fps" is supposed to mean.  The
  // palette is a serial prefix: VideoBuildPalette has returned before VideoProcess is
  // entered, so no dithering overlapped it and none of it can be discounted.
  result->total_ms = (NowMs() - t_start) + result->palette_ms;
    result->cpu_frames = cpu_frames;
    result->gpu_frames = gpu_frames;
    result->in_flight_mb = static_cast<double>(pipe.in_flight_bytes()) / (1024.0 * 1024.0);
    result->ram_budget_mb = static_cast<double>(pipe.ram_budget()) / (1024.0 * 1024.0);
    result->queue_depth = depth;
    result->batch_frames = batch;
    result->cpu_workers = cpu_threads;
  }
  return true;
}

}  // namespace rd
