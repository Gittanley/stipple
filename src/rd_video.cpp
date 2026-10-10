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

// SwsClip16 and SwsYuvToRgb16 now live in include/rd_types.h, which rd_riemersma.h
// already pulls in, so the host dither engine can do this widening per visit without
// copying the matrix a fifth time. See the comment there for why it is ported rather
// than derived.

// Planar 8-bit 4:4:4 -> RgbaF, for the host paths.  This is the conversion the host
// never had.
//
// A frame here is three planes of `pixels` bytes -- Y, then Cb, then Cr -- so the frame
// stride is 3 * pixels BYTES.  It was being handed to RawToFloats, which takes a
// `const std::uint16_t*` and strides by pixels * channels uint16: twice the distance,
// in the wrong unit.  Frame f was written at byte f*pixels*3 and read from
// f*pixels*6, so every frame after the first was read from the wrong place and the tail
// was uninitialised heap.
//
// Nothing noticed, which is the part worth recording.  The palette came out
// byte-identical across every batch size, the frame count was right, and
// RiemersmaBlocksCpu zeroes its error queue per frame per block -- so the dither could
// not average the error away, and no check compared the host path against a known-good
// one.  A run at --batch-frames 1 produced 0 of 30 frames that appear in the
// single-batch output: wrong content, well-formed file.
//
// The device has always had this (BlkGatherYuv444Kernel, gated on upload_u16).  Only
// the host lacked it, which is why it survived: probe-video-exact compares two DEVICE
// engines, and the host is only ever checked for determinism -- a wrong-but-stable
// result passes every check there is.
void RawYuv444ToFloats(const unsigned char* raw, RgbaF* dst, std::size_t pixels,
                       std::size_t frames) {
  const std::size_t plane = pixels;  // bytes per plane, per frame
  for (std::size_t f = 0; f < frames; ++f) {
    const unsigned char* p = raw + f * 3 * plane;
    RgbaF* d = dst + f * pixels;
    for (std::size_t i = 0; i < pixels; ++i) {
      std::uint16_t rgb[3];
      SwsYuvToRgb16(p[i], p[plane + i], p[2 * plane + i], rgb);
      d[i].r = static_cast<float>(rgb[0]);
      d[i].g = static_cast<float>(rgb[1]);
      d[i].b = static_cast<float>(rgb[2]);
      d[i].a = static_cast<float>(65535.0);
    }
  }
}

// The same conversion split across cores by frame -- the shape RawToFloatsParallel
// already has, with the byte stride the planar layout actually has.
void RawYuv444ToFloatsParallel(const unsigned char* raw, RgbaF* dst, std::size_t pixels,
                               int frames, int threads) {
  if (frames <= 1 || threads <= 1 || threads >= frames) {
    RawYuv444ToFloats(raw, dst, pixels, static_cast<std::size_t>(frames));
    return;
  }
  const int workers = std::max(1, std::min(threads, frames));
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(workers) - 1);
  const int chunk = (frames + workers - 1) / workers;
  for (int w = 1; w < workers; ++w) {
    const int f0 = w * chunk;
    if (f0 >= frames) break;
    const int f1 = std::min(frames, f0 + chunk);
    pool.emplace_back([=]() {
      RawYuv444ToFloats(raw + static_cast<std::size_t>(f0) * pixels * 3,
                        dst + static_cast<std::size_t>(f0) * pixels, pixels,
                        static_cast<std::size_t>(f1 - f0));
    });
  }
  const int f1 = std::min(frames, chunk);
  RawYuv444ToFloats(raw, dst, pixels, static_cast<std::size_t>(f1));
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
  // Every path below converts ONE FRAME at a time, and that is the whole point.
  //
  // FloatsToYuv444 uses its `pixels` argument as the PLANE STRIDE as well as the pixel
  // count, so it is correct only for a single frame.  Handed `count` pixels spanning
  // several frames it writes every frame's Y, then every frame's U, then every frame's
  // V -- batch-planar -- where the encoder is told to expect each frame's Y, U and V
  // together.  The two layouts disagree about where frame f's chroma lives, and the
  // result fails silently: right frame count, right palette, plausible picture, wrong
  // pixels, drifting brighter the further into the clip you get.
  //
  // This function did exactly that.  The parallel path stepped dst by f0 * pixels * 3
  // (per-frame layout) while passing `count` as the helper's plane stride (batch-planar
  // layout), and those agree only while a chunk is a single frame.  The two serial
  // early-returns were wrong for ANY frames > 1.  Because writer_convert_threads is
  // capped at 3, every --batch-frames above 3 took a multi-frame chunk -- which is the
  // default of 16, so the host path was wrong by default and right only by accident at
  // 1, 2 and 3.
  //
  // Measured on 320x180 against the CUDA engine as reference: host output was
  // byte-identical at --batch-frames 1, 2 and 3 and wrong at 4, 5, 6, 7, 8, 9, 10, 11,
  // 15, 16, 17 and 30.  With RD_YUV444_OUT=0 the interleaved writer is used instead
  // and was batch-invariant throughout, which is what localised it here: the dithered
  // floats were identical at every batch size, so nothing upstream of this call was
  // involved.  Nothing in the suite could see it -- probe-video-exact compares two
  // DEVICE engines, and the host path is only ever checked for determinism, which this
  // failure is.
  const auto convert_frames = [=](int f0, int f1) {
    for (int f = f0; f < f1; ++f) {
      FloatsToYuv444(src + static_cast<std::size_t>(f) * pixels,
                     dst + static_cast<std::size_t>(f) * pixels * 3, pixels);
    }
  };
  if (frames <= 1 || threads <= 1) {
    convert_frames(0, frames);
    return;
  }
  const int workers = std::max(1, std::min(threads, frames));
  if (workers == 1) {
    convert_frames(0, frames);
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
    pool.emplace_back([=]() { convert_frames(f0, f1); });
  }
  convert_frames(0, first);
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

// Set when a HOST dither worker refuses a batch -- the >256-colour check, or a
// degenerate geometry.  Distinct from an interrupt, because an interrupt is a user
// action and exits on whatever grounds the user chose, while this is a defect and must
// make the process fail.  Written at most once per process (guarded by its own CAS at
// the write site) because every host worker would otherwise race to report it.
std::mutex g_host_dither_error_mu;
std::string g_host_dither_error;

void SetHostDitherError(const std::string& what) {
  std::lock_guard<std::mutex> lock(g_host_dither_error_mu);
  if (g_host_dither_error.empty()) g_host_dither_error = what;
}

const std::string& HostDitherError() {
  std::lock_guard<std::mutex> lock(g_host_dither_error_mu);
  return g_host_dither_error;
}

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
  // BOTH serial early-outs below convert the WHOLE batch, so they get frames * pixels.
  // They used to pass `pixels`, which converts frame 0 only and leaves frames 1..N-1
  // holding whatever was already in the output buffer -- zero on the first pass, and
  // on every pass after, the PREVIOUS batch's dithered pixels.  Reachable whenever
  // writer_convert_threads == 1, which is `max(1, min(3, hardware_concurrency() - 2))`
  // on any machine reporting 3 or fewer logical CPUs (0 is permitted by the standard
  // and does occur, which also yields 1) -- and the probes that set RD_YUV444_OUT=0
  // all run on many-core boxes, so the suite could not see it.  Symptom: a correct
  // frame count, the right palette, exit 0, and frames that jump backwards by a whole
  // batch and then repeat.  The three sibling converters got this right; only this one
  // dropped the factor.
  if (frames <= 1 || threads <= 1) {
    FloatsToRaw(src, dst, static_cast<std::size_t>(frames) * pixels);
    return;
  }
  const int workers = std::max(1, std::min(threads, frames));
  if (workers == 1) {
    FloatsToRaw(src, dst, static_cast<std::size_t>(frames) * pixels);
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
//
// `ffmpeg_bin` used to be a parameter and was never read: the query is an ffprobe
// one.  Dropped rather than left as a second tool path nobody maintains.
bool ProbeFrameTiming(const std::string& path, const std::string& ffprobe,
                      FrameTiming* out, std::string* error) {
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

// ---------------------------------------------------------------------------
// Per-path memo for the two probes above
// ---------------------------------------------------------------------------
//
// WHY, IN SPAWNS.  VideoProcess is called once per SEGMENT by the crash-safe loop in
// rd_cli.cpp, always with the SAME input path, and it ran ProbeFrameTiming and
// ProbeAudioDuration every time.  So an N-segment render spawned 2N ffprobe processes
// where 2 suffice, plus one more per segment whenever the container reports "N/A" for
// per-stream duration -- which Matroska routinely does, and which takes the
// packet-summing fallback inside ProbeAudioDuration.  That is 3N on exactly the
// container this repository's own test fixtures are in.
//
// It is latency, not throughput: both queries sit between the decoder spawn and the
// reader thread starting, with nothing to overlap them, so on the first segment they
// are pure serial prefix.
//
// WHY A PATH KEY AND NOT A static / call_once.  A bare static would be wrong for any
// caller driving two inputs in one process: the second would silently be given the
// first one's frame timing and audio duration, and both feed the OUTPUT -- the encoder
// rate and the -shortest decision.  So this is keyed on the path, exactly like
// VideoHasAudio's cache below, which is the same question asked of the same file.  It
// is deliberately NOT shaped like FpsMode's call_once, which is correct there precisely
// because its answer cannot vary.
//
// Only SUCCESSES are cached.  A failed probe must stay retryable: caching the failure
// would turn one transient ffprobe hiccup into a permanent fallback for the whole
// render, and the fallback (the container's declared rate) silently changes the output
// length.  So a miss re-probes every time and only a definite answer is remembered.
//
// Thread safety: the same static mutex + map shape VideoHasAudio uses, for the same
// reason -- VideoProcess is called from one thread at a time today, but the cache
// outlives any single call and the cost of being wrong here is a wrong file.
struct SourceProbeCache {
  std::mutex mtx;
  std::map<std::string, FrameTiming> timing;
  std::map<std::string, double> audio;
};

SourceProbeCache& ProbeCache() {
  static SourceProbeCache cache;
  return cache;
}

bool ProbeFrameTimingCached(const std::string& path, const std::string& ffprobe,
                            FrameTiming* out, std::string* error) {
  SourceProbeCache& cache = ProbeCache();
  {
    std::lock_guard<std::mutex> lock(cache.mtx);
    const auto it = cache.timing.find(path);
    if (it != cache.timing.end()) {
      *out = it->second;
      return true;
    }
  }
  if (!ProbeFrameTiming(path, ffprobe, out, error)) return false;
  std::lock_guard<std::mutex> lock(cache.mtx);
  cache.timing[path] = *out;
  return true;
}

double ProbeAudioDurationCached(const std::string& path,
                                const std::string& ffprobe) {
  SourceProbeCache& cache = ProbeCache();
  {
    std::lock_guard<std::mutex> lock(cache.mtx);
    const auto it = cache.audio.find(path);
    if (it != cache.audio.end()) return it->second;
  }
  const double seconds = ProbeAudioDuration(path, ffprobe);
  // A negative answer means "no audio, or ffprobe would not say", which is not a fact
  // about the file worth remembering -- see the note on caching failures only.
  if (seconds >= 0.0) {
    std::lock_guard<std::mutex> lock(cache.mtx);
    cache.audio[path] = seconds;
  }
  return seconds;
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
  // `nb_frames` is a DECLARED count, and a container is under no obligation to declare it.
  // mp4/h264 do; mkv/ffv1 do not, and ffprobe prints `N/A` for them.  `std::atoll("N/A")`
  // is 0 -- not an error, not a flag, just 0 -- so the declared field silently became "this
  // clip has no frames", and every downstream clamp treated that as authoritative.
  //
  // The consequence was measured on this repository's own fixture, tests\clip1920_audio.mkv
  // (an mkv, which is the shape the project uses for its own clips):
  //
  //     ffprobe nb_frames        N/A      <- what was read
  //     ffprobe -count_frames    60       <- reality
  //     rdither prints           1920x1080, h264, yuv420p, 0 frames
  //     [video] palette: sampling 1 frame(s) of 0
  //     palette: 16 colours from 1 sampled frame(s)
  //
  // One frame, one palette, exit 0. `want` clamps to its floor of 1 and the montage is a
  // single cell. See docs\KNOWN-ISSUES.md item 2.
  //
  // So when the declared count is absent, COUNT it. `-count_frames` decodes the stream, so
  // this is not free -- it is a second pass over the file -- and it is paid exactly once,
  // only when the cheap answer was unavailable. That ordering is the whole design: trust
  // the declared count when there is one, and only spend a decode when there is not.
  //
  // Cached in `out->frames`, so every caller sees one number and the cost is never paid
  // twice. A source that declares 0 frames AND cannot be counted keeps 0, which is the
  // pre-existing behaviour rather than a new failure.
  if (out->frames <= 0) {
    const std::string cargs =
        "-v error -select_streams v:0 -count_frames "
        "-show_entries stream=nb_read_frames -of default=nw=1:nk=1 " + Quote(path);
    Child counter;
    if (counter.Start(ffprobe, cargs, /*capture_stdout=*/true,
                      /*feed_stdin=*/false, error, "probe-count")) {
      std::string ctext;
      char cbuf[256];
      for (;;) {
        const std::size_t got = counter.Read(cbuf, sizeof(cbuf));
        if (got == 0) break;
        ctext.append(cbuf, got);
      }
      if (counter.Wait() == 0) {
        const long long counted = std::atoll(ctext.c_str());
        if (counted > 0) {
          out->frames = counted;
          if (getenv("RD_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[video] container declared no frame count; counted %lld\n",
                         counted);
          }
        }
      }
    }
    // Deliberately does NOT fail the probe if the count cannot be obtained. A caller that
    // wants the frame count for the PALETTE has a correct-but-degraded answer without it
    // (one sample), and turning a cosmetic gap into a hard error would refuse renders that
    // are otherwise fine. The value is a hint, and 0 stays the documented "unknown".
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

// Does this source pixel format carry an alpha channel?
//
// The palette decoder emits 8-bit RGB rather than RGBA when the answer is no, which
// removes 25% of the pipe traffic (2.12 GB -> 1.59 GB per 600-frame 1080p render, the
// palette prefix being 126x amplification: full frames emitted to fill a 16.8 MB montage
// of 128x128 cells).  The alpha the montage needs is synthesised host-side instead -- the
// same 65535 that RawToFloats already synthesises for a 3-channel rgb48le read, so the
// host and device agree.
//
// This guard exists because the alternative would be to silently flatten real transparency
// into an opaque constant.  ffprobe names the format; the alpha-bearing families are the
// planar/packed ones with a trailing or leading 'a'.  Unknown formats are assumed to
// carry alpha, so a new format degrades to today's behaviour rather than to a wrong one.
bool SourcePixFmtHasAlpha(const std::string& pix_fmt) {
  // Deny by default.  An allowlist of the opaque planar/packed formats rdither's video
  // reader can be handed means an unrecognised or new format keeps its alpha channel and
  // so keeps today's behaviour, rather than silently losing transparency.
  static const char* kOpaque[] = {
      "yuv410p", "yuv411p", "yuv420p", "yuv422p", "yuv440p", "yuv444p",
      "yuvj420p", "yuvj422p", "yuvj444p", "yuvj440p",
      "nv12", "nv16", "nv21", "nv24", "nv42",
      "gbrp", "gray", "yuyv422", "uyvy422", "y210", "p010", "p016",
      "rgb24", "bgr24"};
  for (const char* name : kOpaque) {
    if (pix_fmt == name) return false;
  }
  // Everything else -- rgba, bgra, yuva420p, gbrap, pal8, and anything ffmpeg adds later --
  // is treated as carrying alpha and keeps the 4-channel path.
  return true;
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
  // 8-bit AND no source alpha -> ask ffmpeg for 3 channels and synthesise the 4th
  // host-side.  Verified bit-identical before being relied on: 256 frames of 1080p,
  // 1,592,524,800 RGB samples, 0 differing, max delta 0.  Round Eight (DESIGN.md:1836)
  // measured the 16-bit equivalent as DIFFERING -- rgba64le and rgb48le round
  // differently -- so this is checked per depth, never assumed.  The montage still
  // receives alpha == 65535, so image->alpha_trait stays BlendPixelTrait and
  // quantize.c's `alpha_trait != Undefined -> depth--` does not shift the tree.
  const bool palette_8bit = opt.palette_depth != 16;
  const int pal_ch = (palette_8bit && !SourcePixFmtHasAlpha(info.pix_fmt)) ? 3 : 4;
  // ONE format string for BOTH samplers.  The buffer a sample is read into is sized
  // `pixels * pal_ch` on both paths (rd_video.cpp:1819 for the sequential read, and one
  // per worker at the by-seek arm), so the `-pix_fmt` handed to ffmpeg and `pal_ch` are
  // the same fact stated twice, and a run that disagrees between them reads a truncated
  // stream as if it were whole: `Child::Read` returns exactly the bytes asked for, the
  // `full` guard passes because the pipe still has data, and every sample is three wrong
  // bytes.
  //
  // That is not hypothetical.  The by-seek arm asked for `rgba` while its buffer was
  // `pixels * 3`, so on any clip long enough to take that path (info.frames > 5000) every
  // montage cell was an RGBA byte stream read as RGB triples -- and the quantiser built a
  // green-grey palette from it, 7 of 16 entries pure green, reporting "mean saturation
  // 49.0%" the whole time.  Verified bit-exact: replaying the truncated read in isolation
  // reproduced the dumped montage cell with `np.array_equal` True.
  //
  // It reached a render because nothing in the gate exercises the by-seek arm at all (the
  // 165 bit-exact cases are short clips, and `RD_PALETTE_SEEK` can only DISABLE the path,
  // never force it).  tools/probe-palette-deadline.ps1's 5100-frame fixture can.
  const char* const pal_fmt =
      palette_8bit ? (pal_ch == 3 ? "rgb24" : "rgba") : "rgba64le";
  const std::string args = "-v error" + std::string(fps.input) + " -i " + Quote(path) +
                           " -vf " + filter + fps.output +
                           " -frames:v " + std::to_string(want) + " -f rawvideo" +
                           " -pix_fmt " + pal_fmt +
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
  // A count cannot express coverage either. The budget does two separate things, and
  // they were confused here until 2026-10-09: the arithmetic above picks a COUNT from an
  // assumed 0.12 s per sample, which is a coverage decision; the DEADLINE below actually
  // stops the stage, which is the time bound. Before the deadline existed the flag did
  // only the first, so on a slow machine it took the same number of samples for LONGER
  // -- which is the exact opposite of the "degrades gracefully" claim this replaced.
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


  // The DEADLINE, which is what the flag's name and help text actually promise.
  // The arithmetic above picks a SAMPLE COUNT and then, before this existed, nothing
  // bounded the stage's wall time at all: the count was chosen up front and the run
  // then took as long as it took. On a loaded machine that overran the budget, which is
  // the opposite of what the comment above this block claimed.
  //
  // RD_PALETTE_DEADLINE_MS overrides the deadline WITHOUT touching the count, and that
  // is the only reason the deadline is testable: `--palette-budget-ms 1` collapses the
  // arithmetic to one sample and so cannot distinguish 'count reduced' from 'deadline
  // fired'. Naming follows RD_PALETTE_OVERLAP / RD_PALETTE_SEEK / RD_TRACE.
  const char* dl_env = std::getenv("RD_PALETTE_DEADLINE_MS");
  // Progressive sample order. The by-seek workers take samples by ATTEMPT number, and
  // that number is mapped through a bit-reversal before it becomes a sample index.
  //
  // Why: a deadline truncates by ATTEMPT, and if attempt n mapped to index n then any
  // truncated prefix would be the head of the clip -- the exact coverage loss this path
  // exists to avoid. Bit-reversal is the standard fix: the first k of the reversed order
  // are already a stratified subset of the whole range, so stopping early keeps a
  // SMALL SPREAD ACROSS THE ENTIRE CLIP rather than the first k/256 of it. It costs no
  // extra fetches, which a post-hoc re-spread would, and that matters precisely because
  // the situation it fixes is running out of time.
  //
  // Indices that bit-reversal maps at or past `want` are skipped and the attempt retried,
  // so the count stays exact for a `want` that is not a power of two.
  int dl_order_bits = 1;
  while ((1LL << dl_order_bits) < static_cast<long long>(want)) ++dl_order_bits;
  const auto progressive_index = [&](int attempt) {
    int r = 0;
    for (int b = 0; b < dl_order_bits; ++b) {
      r |= ((attempt >> b) & 1) << (dl_order_bits - 1 - b);
    }
    return r;
  };

  const double deadline_s =
      (dl_env != nullptr && dl_env[0] != '\0')
          ? std::atof(dl_env) / 1000.0
          : (opt.palette_budget_ms > 0 ? opt.palette_budget_ms / 1000.0 : 60.0);
  std::atomic<bool> deadline_hit{false};
  // Highest SOURCE frame the sampler actually reached, so a deadline run can report its
  // coverage rather than just its count. With the progressive order this should land near
  // the end of the clip even when few samples were placed; with the old index order it
  // landed proportionally short, which is what probe-palette-deadline.ps1 now asserts on.
  std::atomic<long long> max_src_frame{-1};

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

  // Cell size.  The default is a 128px lattice tile (VideoOptions::palette_tile = 128,
  // which is what --help documents); tile <= 0 means FULL RESOLUTION, each sampled frame
  // contributing every one of its pixels, which is what makes a single quantization of
  // the strip behave like `magick f1 f2 ... +append -colors N`.
  //
  // (This comment used to say "tile == 0 (the default)", which stopped being true when the
  // default moved to 128.  A reader following it would believe a default render builds a
  // full-resolution montage; a real 600-frame run reports "128x128 montage, 64.0 MiB".)
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
  // `want` is int64_t and is only capped when --palette-frames is absent (:1284), so
  // this narrowing cannot be waved off with "it is small".  It is safe because `rows`
  // is about sqrt(want), not want: `cols` is ceil(sqrt(want)) on the line above, so
  // rows overflows int only once want exceeds (2^31)^2 = 4.6e18, and any want that
  // large fails allocating the montage several lines earlier.  Stated rather than
  // left implicit, so the next reader is not left guessing which of the two it is.
  const int rows = static_cast<int>((want + cols - 1) / cols);
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
  const std::size_t raw_px = pixels * static_cast<std::size_t>(pal_ch);
  // Only the buffer the requested depth actually decodes into.  Both were
  // allocated unconditionally, so the default 8-bit run carried a 16,589,440 B
  // uint16 buffer it never read -- and on the by-seek path one PER WORKER, six of
  // them, 99.5 MB, for the same reason.  `palette_8bit` is fixed above, before
  // either is reached, so this cannot change which of the two a sample lands in.
  std::vector<std::uint16_t> raw16;
  if (!palette_8bit) raw16.resize(raw_px);
  std::vector<std::uint8_t> raw8(raw_px);

  // One decoded sample pixel -> RgbaF.  The identical arithmetic to the two
  // full-frame widening loops this replaces, kept in one place because the tile
  // path now calls it per PLACED pixel and the full-frame path per pixel.
  auto widen = [&](const std::uint8_t* p8, const std::uint16_t* p16, RgbaF* dst) {
    if (p8 != nullptr) {
      dst->r = static_cast<float>(p8[0]) * 257.0f;
      dst->g = static_cast<float>(p8[1]) * 257.0f;
      dst->b = static_cast<float>(p8[2]) * 257.0f;
      // 3-channel read: the same constant RawToFloats synthesises for rgb48le, so
      // host and device agree on opaque alpha.
      dst->a = pal_ch == 4 ? static_cast<float>(p8[3]) * 257.0f : 65535.0f;
      return;
    }
    dst->r = static_cast<float>(p16[0]);
    dst->g = static_cast<float>(p16[1]);
    dst->b = static_cast<float>(p16[2]);
    dst->a = static_cast<float>(p16[3]);
  };

  // Places one decoded sample into its montage cell, WIDENING AS IT PLACES.
  //
  // This was two steps: widen the whole 2,073,600-pixel frame into an RgbaF
  // buffer (33,177,600 B, 16 B/px), then copy the cell out of it.  At the default
  // 128 px tile that is the wrong shape by a factor of 127 -- the cell is 16,384
  // pixels, so 99.21% of what the widening wrote was never read.  And this is a
  // SERIAL stage that gates the entire render, before the main reader has decoded
  // a frame, so the traffic is latency added in front of the job rather than
  // throughput lost inside it.
  //
  // Per sample at 1920x1080, counting bytes read plus bytes written:
  //
  //   tile 128 (the default), 256 samples, cell 16,384 px = 262,144 B of RgbaF:
  //     before  raw8 8,294,400 + frame 33,177,600 + frame 262,144
  //             + montage 262,144                               = 41,996,288 B
  //     after   raw8    65,536 + montage 262,144                =    327,680 B
  //     saving  41,668,608 B per sample; 10,667,163,648 B over 256 samples,
  //             10,173.00 MiB
  //
  //   --palette-tile 0 (full resolution), 32 samples -- what the 64 Mpixel cap
  //   binds at 1080p, cell 2,073,600 px:
  //     before  raw8 8,294,400 + frame 33,177,600 + frame 33,177,600
  //             + montage 33,177,600                            = 107,827,200 B
  //     after   raw8 8,294,400 + montage 33,177,600             =  41,472,000 B
  //     saving  66,355,200 B per sample; 2,123,366,400 B over 32 samples,
  //             2,025.00 MiB
  //
  // Two caveats on those numbers, both stated rather than rounded away.  First,
  // they are byte counts, and the tile path's 16,384 reads are STRIDED, so in
  // cache lines they cost up to 16,384 x 64 B rather than 16,384 B on the source
  // side; that puts the saving on the default geometry nearer 9,933.00 MiB than
  // 10,173.00 MiB, and it moves in the direction of the change rather than against
  // it (the same lines were being pulled into the 33 MB frame before).  Second,
  // the scattered reads land in an 8,294,400 B buffer instead of a 33,177,600 B
  // one, so the working set per worker shrinks by 75%.
  //
  // OUTPUT-PRESERVING, and by arithmetic rather than by argument: every value
  // written is `float(raw) * 257.0f` at 8-bit depth or `float(raw)` at 16 -- one
  // float multiply or one float cast either way, exactly as the loops it replaces
  // did -- from the same source byte, at the same (sx, sy) the lattice picks, into
  // the same cell offset.  Nothing else read `frame`: the quantizer is handed
  // `montage.data()`, and the diagnostic dump at RD_DUMP_MONTAGE dumps the montage.
  auto place_raw = [&](int s, const std::uint8_t* src8,
                       const std::uint16_t* src16) {
    const int cx = s % cols;
    const int cy = s / cols;
    if (tile <= 0) {
      // Full resolution is a straight row copy; it stays one, and the widening
      // simply happens on the way through instead of beforehand.
      for (int y = 0; y < cell_h; ++y) {
        const std::size_t base = static_cast<std::size_t>(y) * info.width;
        RgbaF* dst = montage.data() +
                     (static_cast<std::size_t>(cy) * cell_h + y) * montage_w +
                     static_cast<std::size_t>(cx) * cell_w;
        if (src8 != nullptr) {
          const std::uint8_t* row = src8 + static_cast<std::size_t>(pal_ch) * base;
          for (int x = 0; x < cell_w; ++x) {
            widen(row + static_cast<std::size_t>(pal_ch) * static_cast<std::size_t>(x), nullptr, dst + x);
          }
          continue;
        }
        const std::uint16_t* row = src16 + static_cast<std::size_t>(pal_ch) * base;
        for (int x = 0; x < cell_w; ++x) {
          widen(nullptr, row + static_cast<std::size_t>(pal_ch) * static_cast<std::size_t>(x), dst + x);
        }
      }
      return;
    }
    for (int ty = 0; ty < cell_h; ++ty) {
      const int sy =
          static_cast<int>(static_cast<std::int64_t>(ty) * info.height / cell_h);
      const std::size_t fy =
          static_cast<std::size_t>(cy) * cell_h + static_cast<std::size_t>(ty);
      const std::size_t row_base = static_cast<std::size_t>(sy) * info.width;
      RgbaF* dst = montage.data() + fy * montage_w +
                   static_cast<std::size_t>(cx) * cell_w;
      for (int tx = 0; tx < cell_w; ++tx) {
        const int sx =
            static_cast<int>(static_cast<std::int64_t>(tx) * info.width / cell_w);
        const std::size_t o = row_base + static_cast<std::size_t>(sx);
        if (src8 != nullptr) {
          widen(src8 + static_cast<std::size_t>(pal_ch) * o, nullptr, dst + tx);
        } else {
          widen(nullptr, src16 + static_cast<std::size_t>(pal_ch) * o, dst + tx);
        }
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
    // Clamp in 64-bit and narrow afterwards, rather than letting std::min<int> do the
    // narrowing: that instantiation is what produced the least readable warning in the
    // build (MSVC prints the target as `const _Ty`, `_Ty=int`, because _Ty is this
    // call's template parameter and not a type anyone can grep for).  Narrowing after
    // the clamp is provably safe rather than incidentally safe -- the value is <= 6.
    const int workers =
        static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(want, 6)));
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
        // The pipe buffers, and nothing else: the widening that used to fill an
        // `lf` here now happens straight into the montage cell (place_raw), so
        // six workers no longer each carry a 33,177,600 B RgbaF frame.
        std::vector<std::uint8_t> l8(static_cast<std::size_t>(pixels) * pal_ch);
        std::vector<std::uint16_t> l16;
        if (!palette_8bit) l16.resize(static_cast<std::size_t>(pixels) * pal_ch);
        for (;;) {
      const int attempt = next_sample.fetch_add(1);
      if (attempt >= want) return;
      // Bit-reversed, skipping the out-of-range half of the padded range.
      const int s = progressive_index(attempt);
      if (s >= want) continue;
      // Deadline: a worker that has already started a seek finishes it, because the
      // time is spent either way; only NEW samples are refused. The cells it would
      // have filled are padded from the last real sample further down, so refusing is
      // safe and costs palette coverage rather than producing black cells.
      // Only once at least one sample has LANDED. seeks_done is incremented after `placed[s] = 1`,
      // so it counts successful placements, not attempts. Without the guard a
      // 1 ms deadline refuses every seek before any cell is filled, `sampled`
      //  stays 0, and the "ffmpeg produced no frames" guard below turns a
      //  working deadline into a FAILED render. Found by probe-palette-deadline.ps1.
      if (seeks_done.load() > 0 && NowMs() - p0 > deadline_s * 1000.0) {
        deadline_hit.store(true);
        return;
      }
          const long long idx = static_cast<long long>(s) * sample_step;
          const double when =
              info.fps > 0.0
                  ? (static_cast<double>(idx) - 0.25) / info.fps
                  : 0.0;
          char sargs[256];
          std::snprintf(sargs, sizeof(sargs),
                        "-v error -ss %.6f -i %s -fps_mode passthrough -frames:v 1 "
                        "-f rawvideo -pix_fmt %s -",
                        when, Quote(path).c_str(), pal_fmt);
          Child one;
          std::string serr;
          if (!one.Start(ffmpeg, sargs, true, false, &serr, "palette-seek")) continue;
          bool full = false;
          if (palette_8bit) {
            full = one.Read(l8.data(), l8.size()) == l8.size();
          } else {
            full = one.Read(l16.data(), l16.size() * sizeof(std::uint16_t)) ==
                   l16.size() * sizeof(std::uint16_t);
          }
          one.Close();
          if (!full) continue;
          place_raw(s, palette_8bit ? l8.data() : nullptr,
                    palette_8bit ? nullptr : l16.data());
          placed[static_cast<std::size_t>(s)] = 1;
      { long long prev = max_src_frame.load();
        while (idx > prev &&
               !max_src_frame.compare_exchange_weak(prev, idx)) {} }
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
      std::fprintf(stderr,
                   "\r[palette] %d/%lld samples by seek, %.1f ms/frame elapsed %.1f s   ",
                   sampled, static_cast<long long>(want),
                   decode_ms / std::max(1, sampled), decode_ms / 1000.0);
      std::fflush(stderr);
    }
  }

  for (; !sample_by_seek && sampled < want; ++sampled) {
    // `sampled > 0` for the same reason as the by-seek arm: a deadline that fires
    // before the first sample leaves `sampled` at 0 and the guard below
    // reports a hard failure rather than a short palette.
    if (sampled > 0 && NowMs() - t0 > deadline_s * 1000.0) {
      deadline_hit.store(true);
      break;
    }
    const double d0 = NowMs();
    std::size_t got = 0;
    if (palette_8bit) {
      got = child.Read(raw8.data(), raw8.size());
    } else {
      got = child.Read(raw16.data(), raw16.size() * sizeof(std::uint16_t));
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

    // Full resolution is a straight copy; a tile is a lattice sample.  Both widen
    // from the decoder's own bytes as they place, so the montage never sees an
    // intermediate full-resolution frame -- see place_raw.
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
    place_raw(sampled, palette_8bit ? raw8.data() : nullptr,
              palette_8bit ? nullptr : raw16.data());
  }
  child.Close();
  // Land the bar on what was actually placed when a deadline cut the loop short. Without
  // this a truncated run ended the bar at, say, 78% and looked interrupted rather than
  // finished. The by-seek arm already did this at its own progress.Update.
  progress.Update(sampled);

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
  if (deadline_hit.load()) {
    std::fprintf(stderr,
                 "[video] palette: deadline of %.0f ms reached; %d of %lld samples "
                 "placed, source frames up to %lld of %lld\n",
                 deadline_s * 1000.0, sampled, static_cast<long long>(want),
                 max_src_frame.load(), static_cast<long long>(info.frames));
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
  // Where this batch starts in the source, relative to this segment.  Set by the
  // reader, which is the only thing that knows it.
  //
  // It exists because the writer used to emit batches in COMPLETION order: `done` is a
  // deque of slot indices and nothing carried a position, so the writer took
  // `done.front()` and wrote whatever had finished first.  With one batch that is
  // correct by accident.  With several it is a silent permutation of the output -- and a
  // silent one, because the frame count was always right and every frame was always
  // present, just not in order.  Measured on a 30-frame clip: two runs produced the same
  // 30 frames as a multiset with only 6 of 30 in identical positions.
  int first_frame = 0;
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
    // first_frame belongs here.  It was missing, and it is the one field the whole
    // source-ordering scheme rests on -- HasTurn() decides which batch may be written
    // next purely by comparing this against next_write.  A move that drops it does not
    // corrupt anything by itself; it makes the moved-to batch unmatchable, so the
    // writer waits for a turn that never comes.  No live symptom today only because
    // nothing moves a populated Batch: the slot pool is sized in place
    // (`std::vector<Batch> slots(depth)` in VideoProcess) and every slot is filled by
    // the reader and drained by the writer in the same object, so this operator has no
    // caller at all.  It stays because a move-only type with a declared destructor
    // needs one, and because removing it would make adding a field to Batch a
    // double-free the first time somebody did use it.  Add the next field to this
    // operator and this comment gets shorter.
    first_frame = o.first_frame;
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
  // The palette and the tree are NOT parameters.  They were, and nothing here has
  // read them since the dither moved out of this class and into the worker lambda in
  // VideoProcess -- which captures them by reference from its own arguments.  Three
  // unused reference members is three dangling-reference hazards for a reader who
  // assumes they matter, and removing them is what let the constructor shrink to the
  // one option it actually reads.
  explicit Pipeline(const VideoOptions& opt) {
    batch_ = std::max(1, opt.batch_frames);

    // A batch slot holds up to three buffers: the float4 `b.pixels`, the frame data
    // going to the engine (`in16`, or the pinned pointer beside it), and the frame data
    // coming back (`out16`, or its pinned pointer).  The budget below sizes the queue,
    // so it has to count what is allocated rather than a second guess at it: the
    // budget is the number a user reads to decide whether to lower --queue-depth, and
    // a number that disagrees with the process is worse than no number at all.
    //
    // `float_path_` is UNCONDITIONALLY true, so `b.pixels` is allocated on every video
    // run at sizeof(RgbaF) = 16 bytes per pixel per slot -- 530,841,600 B, which is
    // 506.25 MiB (not the 531 MiB an earlier comment here claimed), at 1080p with
    // --batch-frames 16 -- including the default GPU runs where nothing reads it.
    //
    // The unconditional form is deliberate and is not a simplification.  This used to
    // be a condition over `opt.use_gpu`, `opt.gpu_float_out` and a host-worker count,
    // and reading what the user REQUESTED rather than what the hardware turned out to
    // be is what made `--cpu-threads 0` access-violate on a machine with no device:
    // 0 is documented as "GPU only", the request says use_gpu and gpu_float_out is
    // false, so the whole expression evaluated false and the float4 buffer was never
    // allocated -- while RunVideo, having found no device, launched a host worker that
    // dereferenced exactly that buffer.  0xC0000005, after a 577-byte container that
    // ffprobe calls malformed.
    //
    // What the field's being constant costs, and what would have to be proved before
    // dropping the allocation, is written out at the `b.pixels.resize` in
    // VideoProcess.
    float_path_ = true;

    // ---- RAM budget --------------------------------------------------------
    // The queue is the only allocation this budget governs, and it is not the only
    // memory the process uses.  Two things sit outside it, and the first cannot be
    // paged back out if the machine runs short:
    //
    //   * ffmpeg's own decoder, which grows with its thread count and with the
    //     frame size;
    //   * Windows and everything else on the machine.
    //
    // Both are subtracted below.  Previously the share was of *total* RAM,
    // which over-commits as soon as anything else is running: on a 16 GiB machine
    // with 3 GiB free the budget still claimed 5.3 GiB, so the queue would be paged
    // out mid-render or refused outright.  It is now the lesser of that share and
    // what is actually available, and it never aims to fill the machine.
    //
    // The CUDA pinned staging used to be the third item on that list, subtracted as
    // `pixels * batch * sizeof(RgbaF) * gpu_workers`.  That was wrong in both
    // directions and it could not be right by construction: the pinned buffers are
    // `batch * (in_frame_bytes + out_frame_bytes)` PER SLOT, so they scale with the
    // queue DEPTH and not with the worker count, and a depth is not known until after
    // this budget has been divided -- so the term was subtracted once, whatever the
    // depth turned out to be, for an amount that matches no allocation.
    //
    // It is now inside the per-slot figure, which is where it belongs: every byte of
    // it is a queue slot's bytes.  The part that cannot be paged out is reported
    // separately by in_flight_pinned_bytes(), which VideoProcess fills in once the
    // depth is known.  Adding it to reserved() as well would count it twice.
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
    constexpr std::size_t kDecoderHeadroom = 768ull << 20;
    constexpr std::size_t kKeepFree = 1536ull << 20;
    const std::size_t reserve = kDecoderHeadroom;
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
    // --max-ram-mb, when the user actually passed it, caps this and can only lower it.
    // Physical RAM stays the real ceiling: a request for more memory than the machine
    // has is answered with the memory the machine has, not with an error, because the
    // alternative is a flag that refuses ordinary numbers on a small box.
    //
    // Capping rather than overriding is the whole decision.  Overriding would let a
    // user promise the pipeline bytes that do not exist, which is how you get an
    // out-of-memory crash instead of a queue that is merely smaller than asked for.
    if (opt.max_ram_mb_set) {
      const std::size_t cap = opt.max_ram_mb * 1024ull * 1024ull;
      if (budget > cap) budget = cap;
    }
    ram_budget_ = budget;
    reserved_ = reserve;
    avail_phys_ = avail;
    // No depth, and no per-slot byte count, is derived here.  Both are made in
    // VideoProcess, which is the only place the flags that decide them exist:
    // `use_gpu` is resolved there, `in_frame_bytes` and `out_frame_bytes` are
    // computed there, and the depth is trimmed against the worker count there.
    // Deriving them here meant a second guess at the allocations, and there were
    // three of them disagreeing by 32, 24 and 32-plus-the-input.
    //
    // The floor of two -- one slot in the dither and one in the writer -- is applied
    // in VideoProcess, where the figure it applies to is real, and so is its warning.
  }

  int batch() const { return batch_; }
  std::size_t ram_budget() const { return ram_budget_; }
  // The reserve that is NOT the queue: ffmpeg's decoder.  Everything else the process
  // allocates is a queue slot, including the page-locked staging, so this is the only
  // thing that must be added to in_flight_bytes() to reach the peak.
  std::size_t reserved() const { return reserved_; }
  std::size_t avail_phys() const { return avail_phys_; }
  bool float_path() const { return float_path_; }

  // Bytes ONE queue slot occupies, and the part of that which cannot be paged out.
  // A setter rather than a constructor constant because neither number can be computed
  // here: the input and output frame sizes are VideoProcess's locals and the pinned
  // staging depends on the depth, which is only known after this budget is divided.
  void set_slot_bytes(std::size_t bytes, std::size_t pinned) {
    slot_bytes_ = bytes;
    slot_pinned_ = pinned;
  }
  // Reports the depth actually allocated.  VideoProcess trims the RAM-derived value
  // against the worker count, and reading the untrimmed number is how "queue 2" could
  // be printed next to "5316 MiB in flight".
  void set_depth(int depth) { depth_ = depth; }
  std::size_t in_flight_bytes() const { return slot_bytes_ * depth_; }
  // The page-locked part of in_flight_bytes().  Already inside that figure, so it is
  // reported as a component and never ADDED to reserved(): doing that would count the
  // same bytes twice in the peak -- which is what the ** OVER FREE ** flag is there to
  // catch, so a double count in the number that drives the flag defeats its purpose.
  std::size_t in_flight_pinned_bytes() const { return slot_pinned_ * depth_; }

 private:
  int batch_ = 16;
  int depth_ = 4;
  std::size_t ram_budget_ = 0;
  bool float_path_ = true;
  std::size_t reserved_ = 0;
  std::size_t avail_phys_ = 0;
  std::size_t slot_bytes_ = 0;
  std::size_t slot_pinned_ = 0;
};

// PaletteGate methods.  Deliberately thin: the class exists so the palette can be built
// while the reader runs, and every piece of policy about WHEN that is worth doing lives in
// the callers, not here.
void PaletteGate::Publish(Palette&& palette, ColorTree&& tree) {
  // Allocated BEFORE the lock: a 2 MB allocation under a mutex the reader may already be
  // blocked on is a stall nobody is measuring.
  auto held_palette = std::make_unique<Palette>(std::move(palette));
  auto held_tree = std::make_unique<ColorTree>(std::move(tree));
  {
    std::lock_guard<std::mutex> lock(mu_);
    palette_ = std::move(held_palette);
    tree_ = std::move(held_tree);
    ok_ = true;
    done_ = true;
  }
  cv_.notify_all();
}

void PaletteGate::Fail(std::string error) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    // A Publish that already landed wins.  The build thread can fail after publishing (a
    // cleanup path that reports failure), and clobbering a good palette there would turn a
    // completed render into an error for no reason.
    if (done_) return;
    error_ = std::move(error);
    done_ = true;
  }
  cv_.notify_all();
}

void PaletteGate::Wait() const {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this] { return done_; });
}

bool PaletteGate::ok() const {
  std::lock_guard<std::mutex> lock(mu_);
  return ok_;
}

const std::string& PaletteGate::error() const { return error_; }

// palette() and tree() take no lock.  See the header: they are reachable only after Wait()
// has returned, and locking them would suggest a guarantee the class deliberately does not
// make (that they are safe to call concurrently with a late Fail).
const Palette& PaletteGate::palette() const { return *palette_; }

const ColorTree& PaletteGate::tree() const { return *tree_; }

bool VideoProcess(const std::string& in, const std::string& out,
                  const VideoOptions& opt, const VideoInfo& info,
                  const PaletteGate& gate,
                  VideoResult* result, std::string* error) {
  // Both tools, once.  The second VideoFindTools further down used to ask for ffprobe
  // only, having already been handed ffmpeg -- two walks of the same PATH search for
  // two answers from the same two environment variables.  It is one lookup now, and it
  // is a PATH search either way: ToolOnPath is SearchPathA, so this saves no process
  // spawns.  The spawns it looked like it saved are item 3's, below.
  std::string ffmpeg;
  std::string ffprobe;
  if (!VideoFindTools(&ffmpeg, &ffprobe, error)) return false;

  const std::size_t pixels =
      static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height);

  // Decoder output.  "yuv444" is the DEFAULT (VideoOptions::input_mode), and it is
  // exact: 3 bytes per pixel instead of 8, with the YCbCr->RGB conversion done in the
  // gather kernel.  The reasoning that once made it opt-in does not apply to it.
  // "rgba64" is the reference it is compared against, and it stays available by name:
  // 3 bytes per pixel of pipe against 8 is 2.7x less traffic across the pipe and the
  // H2D, and for a 4:4:4 source the two are bit-identical end to end (605 frames),
  // because the colour matrix is swscale's own, ported exactly (see d_sws_yuv_to_rgb16).
  //
  // On a 4:2:0 source they are NOT identical: letting ffmpeg reconstruct 4:4:4 and then
  // converting applies a different chroma reconstruction than swscale's fused 4:2:0 ->
  // RGBA path, which moves 0.67% of output components (34.3 dB) once the dither has
  // amplified the difference through its own error feedback.  --input-mode rgba64 is
  // that reference and remains the thing to compare against.  See the README.
  //
  // (The "4525709 of 6220800 samples differ" figure in an earlier version of this
  // comment was a comparison of ffmpeg's yuv420p->rgba64le against yuv420p->rgb48le.
  // Both of those are swscale conversions of the same input, and neither is a mode
  // this pipeline can select: dec_pix_fmt below is one of yuv420p, yuv444p or
  // rgba64le.  It is kept here because it is the measurement that justifies treating
  // the interleaved formats as distinct, not because it describes a code path.)
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
  // Reading is 3 bytes per pixel in planar 4:4:4 (1.5 in 4:2:0, 8 in rgba64le); writing
  // is 3 bytes per pixel for the default planar yuv444p output and 8 for rgba64le.
  // Sharing one name meant the writer handed the encoder 37.5% of each frame's bytes,
  // it desynchronised, and a 605-frame render came out as 226 -- with an "encode" time
  // that had collapsed to 2.5 s and a throughput number that looked like a win.  Both
  // symptoms, one wrong constant.
  const std::size_t in_frame_bytes =
      use_yuv420 ? (pixels * 3) / 2
                 : use_yuv444 ? pixels * 3
                              : pixels * static_cast<std::size_t>(dec_channels) *
                                    sizeof(std::uint16_t);
  // The pageable `in16` fallback was sized `batch * pixels * dec_channels`, treating
  // dec_channels as if it were a byte count for every mode.  It is a uint16 CHANNEL
  // count, correct only for rgba64le.  In planar 4:4:4 -- the default -- that is 6
  // bytes per pixel where the frame is 3, so the vector held twice what it needed:
  // 759.4 MiB across a 16-frame batch at queue depth 8.  Size it from the byte
  // figure the reader actually strides by, and round UP, because pixels*3/2 is not
  // integral for an odd pixel count and truncating would leave the last frame's
  // final byte outside the allocation.
  //
  // NOTE there is no rgb48le path in the VIDEO pipeline, and a note above this used to
  // claim one.  dec_pix_fmt above is exactly one of yuv420p, yuv444p or rgba64le, so
  // `dec_channels` is 3 only for planar yuv444 -- and 3 there means three BYTES per
  // pixel, not three uint16 channels.  For rgba64le it is 4, so that branch is 8 bytes
  // per pixel.  Both are already in in_frame_bytes above, which is why the pageable
  // size must come from that and not from dec_channels.
  //
  // rgb48le itself is a real format in the codebase (rd_riemersma.h's BlockOptions
  // documents in_channels = 3 as rgb48le, and rd_opencl.cpp names it) -- it is the
  // IMAGE path that asks for it, via ImLoad.  The comment that was here was reading
  // that as a video decoder option and budgeting 6 bytes per pixel for a format the
  // decoder is never asked to produce.
  const auto in16_elems_for = [&](int frames) {
    return (static_cast<std::size_t>(frames) * in_frame_bytes + 1) / 2;
  };
  // What the encoder is fed.  Planar 8-bit 4:4:4 by default: the device does the
  // RGB->YUV, which is 3 bytes per pixel against rgba64le's 8, and ffmpeg then has
  // nothing to convert.  Both halves matter -- the encode stage was measured blocked
  // at 597 MB/s, under the pipe's own 0.86 GB/s, so the host conversion rather than
  // the transport was the limit, and shrinking the pipe alone would not have helped.
  // RD_YUV444_OUT=0 goes back to rgba64le for comparison.
  const char* yuv_out_env = std::getenv("RD_YUV444_OUT");
  const bool out_yuv444 = yuv_out_env == nullptr || yuv_out_env[0] != '0';
  // Reading is 3 bytes per pixel in planar 4:4:4, writing is 3 bytes per pixel for
  // planar yuv444p and 4 uint16 (8 bytes) for rgba64le.  Input and output frame sizes
  // are NOT the same and must not share a variable: sharing one meant a 605-frame
  // render came out as 226, with an "encode" time that had collapsed to 2.5 s and a
  // throughput number that looked like a win.  Both symptoms, one wrong constant.
  const std::size_t out_frame_bytes =
      out_yuv444 ? pixels * 3 : pixels * 4 * sizeof(std::uint16_t);
  // The same rounding-up as in16_elems_for, for the same reason and the same reason
  // it is needed at all here: `pixels * 3 / 2` is not integral for an odd pixel
  // count, and truncating would leave the last frame's final byte outside the
  // allocation.  Round UP, so the buffer is never short by one byte.
  const auto out16_elems_for = [&](int frames) {
    return (static_cast<std::size_t>(frames) * out_frame_bytes + 1) / 2;
  };

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
  //
  // Both queries below are memoised per PATH, because VideoProcess is called once per
  // SEGMENT by the crash-safe loop in rd_cli.cpp with the SAME input path, and each
  // call re-ran both.  That is 2N ffprobe spawns where 2 suffice -- 3N on a Matroska,
  // which routinely reports "N/A" for per-stream duration and so takes
  // ProbeAudioDuration's packet-summing fallback.  All of it is SERIAL-PREFIX latency:
  // it sits after the decoder has been spawned and before the reader thread starts,
  // with nothing to overlap it.
  //
  // Keyed on the path, deliberately, rather than a bare static or std::call_once.  A
  // bare static is wrong for any caller driving two inputs in one process, and this
  // file already holds both precedents: FpsMode caches an answer that cannot change
  // (ffmpeg's own option list) under call_once, and VideoHasAudio keys its cache on
  // the path.  These two are per-path facts about a FILE, so they take the second
  // shape.
  FrameTiming timing;
  double audio_duration = -1.0;
  {
    std::string terr;
    if (!ProbeFrameTimingCached(in, ffprobe, &timing, &terr)) {
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
    audio_duration = ProbeAudioDurationCached(in, ffprobe);
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

  Pipeline pipe(opt);
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
  // 4:2:0 on the host has no chroma reconstruction, and the alternative is worse than
  // useless: it was being handed to the rgba64le converter, which strides by
  // pixels * 4 uint16 over a buffer holding pixels * 1.5 bytes, so it read past the
  // frame into the next one and off the end of the allocation.  The device has a real
  // bilinear reconstruction (BlkGatherYuv420Kernel); porting it is a second piece of
  // work and not the same shape as the 4:4:4 fix above, so until it exists this
  // refuses rather than emitting plausible garbage.  The rule the codebase already
  // follows twice over, for the 4:4:4 plane stride and the input/output size collision.
  //
  // `cpu_threads > 0` is the THIRD disjunct, and it is here because the consumer of
  // b.pixels is not only the GPU.  The reader's widening condition below tests
  // `!use_gpu || opt.gpu_float_out || cpu_threads > 0` -- host workers dither b.pixels
  // whenever cpu_threads > 0, which is exactly what --cpu-threads N asks for on a GPU
  // machine.  This refusal enumerated only the first two, so fixing that condition (it
  // used to omit the host-worker case, and every documented use of --cpu-threads on a
  // GPU machine was dithering an unwritten buffer) made this combination reachable:
  // RawToFloats then reads pixels*4 uint16 from a pixels*1.5 byte buffer, which at
  // 1920x1080 with --batch-frames 16 is ~206 MiB past a 49.8 MB allocation.  Both
  // conditions must name the same three things or one of them is a hole.
  if (use_yuv420 && (!use_gpu || opt.gpu_float_out || cpu_threads > 0)) {
    *error =
        "[video] --input-mode yuv420 needs a GPU engine to reconstruct chroma; this\n"
        "        run has the host doing the conversion, which does not implement it.\n"
        "        Use --input-mode yuv444 (the default) or rgba64, or build with CUDA.\n";
    return false;
  }
  if (opt.use_gpu && !use_gpu) {
    if (want_opencl) {
      std::string why;
      OpenCLAvailable(nullptr, &why);
      std::fprintf(stderr, "error: --engine opencl requested for video but %s\n",
                   why.c_str());
      return false;
    }
    // The same refusal the IMAGE path already makes at rd_cli.cpp ("--engine %s
    // requested but no CUDA device"), which the video path was simply missing.
    //
    // Measured on a -DRD_WITH_CUDA=OFF build: `--engine blocks` on video ran the whole
    // job on the host with no diagnostic at all -- 30 frames, a valid file, exit 0 --
    // so a probe case named `cuda` was quietly measuring the host while printing as a
    // cuda result.  That is what turned a CI run red, and the awkward part was that
    // the suite had ALREADY decided how to report the non-determinism of that host
    // path as a KNOWN DEFECT: the same bug was counted once as a failure and once as
    // excused, from the same code path, because one route into it was named `cpu` and
    // the other was named `cuda`.
    //
    // A silent fallback is the thing being refused, not the absence of a device: it
    // makes `--engine X` measure something other than X, which is the same reason the
    // OpenCL branch above refuses rather than falling back to CUDA.
    //
    // gpu_engine_named is what keeps this off the default path.  `--engine cpu` and an
    // unspecified engine both mean the host here -- --video has always used the
    // block-parallel engine -- so refusing those would break a plain `--video in out`
    // on any build without CUDA.  The first version of this guard did exactly that,
    // which is why the field exists.
    if (opt.gpu_engine_named) {
      std::fprintf(stderr,
                   "error: --engine blocks requested for video but no CUDA device\n"
                   "       refusing rather than running on the host: a silent fallback\n"
                   "       makes --engine X measure something other than X.  Use\n"
                   "       --engine cpu, or --no-gpu, for the host path.\n");
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

  // `--cpu-threads 0` means "GPU only" (see --help), and on a machine with no GPU
  // that is an unsatisfiable request: nothing would dither anything.  It used to
  // crash instead of saying so, and the fix was in Pipeline's float4 allocation
  // rather than here -- see the note on float_path_.  Refusing it outright is the
  // honest response, because silently running a host worker would give "GPU only"
  // an answer that quietly is not bit-identical to what a GPU produces, which is the
  // one thing that flag exists to guarantee.
  if (cpu_threads == 0 && !use_gpu) {
    *error =
        "--cpu-threads 0 means \"GPU only\", but this build found no usable GPU "
        "device.  Use --cpu-threads 1 (or leave it on auto) for a host dither; note "
        "that the host walk is not bit-identical to the CUDA walk.";
    return false;
  }

  // A worker with no slot to work on is worth nothing, so a slot is the unit of
  // currency.  Asking for ten host workers plus the GPU therefore cannot work -- the
  // extra workers just block on the free-slot queue while the GPU starves for want of
  // a batch.
  //
  // So the queue is sized first, and the worker count is trimmed to fit it.  A
  // deep queue with fewer workers beats a shallow queue with more, and this way
  // the two can never disagree.
  //
  // ---- ONE bytes-per-pixel constant, from the allocations below ----------------
  //
  // Four sites used to disagree on this number.  One of them fed the [ram] peak and one
  // fed the depth divisor, so the queue was sized against a figure the process did not
  // honour and the peak was reported against a different one again.
  //
  // THE PER-SLOT TRUTH, straight off the resize/alloc calls in the loop below, at
  // 1920x1080 and --batch-frames 16 (33,177,600 batch-pixels per slot):
  //
  //     buffer   bytes/pixel    per slot (B)      condition
  //     pixels        16       530,841,600      always -- float_path_ is the constant true
  //     input          3        99,532,800      pinned branch, yuv444p
  //                    8       265,420,800      pinned branch, rgba64le
  //                  1.5        49,766,400      pinned branch, yuv420p
  //     output         3        99,532,800      pinned branch, out yuv444p (the default)
  //                    8       265,420,800      pinned branch, out rgba64le
  //                    8       265,420,800      pageable branch, either output
  //
  // So, per pixel and per slot:
  //
  //              input  output   total B/px   per slot (B)     per slot (MiB)
  //     GPU yuv444    3       3            22      729,907,200        696.09
  //     GPU rgba64    8       3            27      895,795,200        854.30
  //     host yuv444   3       8            27      895,795,200        854.30
  //     host rgba64   8       8            32    1,061,683,200      1,012.50
  //
  // And here is where the four old sites went wrong, each in a DIFFERENT direction --
  // which is the whole reason none of them looked broken on its own:
  //
  //   * The [ram] peak's per_batch_ used a FLAT 32 for all four, so it OVER-stated
  //     every GPU configuration: 1,012.50 MiB reported against 696.09 MiB held on the
  //     default, 45.5% too high, and 18.5% too high on GPU rgba64.  A constant 32 is
  //     what made the *default* configuration look like the most memory-hungry one
  //     when it is the cheapest of the four.
  //   * The depth divisor used `pixels * batch * 24 + batch * in_frame_bytes`: the 24
  //     is pixels 16 plus a flat 8 for the output, and in_frame_bytes was added but
  //     out_frame_bytes WAS NOT.  On the default that is 16 + 8 + 3 = 27 against a
  //     real 22 -- 22.7% too high, so `affordable` is 22.7% too LOW.  Note the
  //     asymmetry with the line above: the peak over-counted and the divisor
  //     over-counted TOO, but by different amounts, which is why they disagreed.
  //   * in_flight_bytes() multiplied that same flat-32 per_batch_ by depth, so it
  //     reported the same 1,012.50 MiB per slot that [ram] printed -- 45.5% above the
  //     696.09 MiB the default path actually holds.  Worth being precise about what
  //     this one got wrong, because they are easy to conflate: it omitted NOTHING,
//     it multiplied a wrong constant.  The site that added in_frame_bytes while
  //     omitting out_frame_bytes was the depth divisor above; the site that used a flat
  //     32 for every configuration was this one and the [ram] print.  Both numbers fed
//     the ** OVER FREE ** flag, and neither was a figure the process held.
  //
  //   * And the ctor's own `pinned` reserve, subtracted from the budget before any of
  //     that, was a fifth number: pixels * batch * sizeof(RgbaF) * gpu_workers, i.e.
  //     float4 bytes per GPU *worker*.  No allocation has that shape.  The pinned
  //     buffers are per SLOT (the loop below does one CudaAllocPinned pair per slot)
  //     and are the in and out FRAME buffers, not a float4 batch -- so it was both the
  //     wrong quantity and, being subtracted once regardless of depth, the wrong
  //     amount at every depth except 1.
  //
  // THE DIRECTION MATTERS, and it is the reassuring way round: fixing the divisor
  // RECOVERS depth.  The queue was throttled by an over-estimate, not permitted to
  // spend memory it did not have, so the corrected figure is shallower-per-slot and
  // therefore DEEPER for the same budget.  The [ram] peak moves the other way, down,
  // by the 45.5% above -- which is why the two have to be fixed together: fixing the
  // peak alone would look like the process had started leaking.
  //
  // The 4:2:0 input is 1.5 bytes per pixel, which is why the input and output terms
  // stay whole-frame byte counts (in_frame_bytes / out_frame_bytes, already computed
  // above from dec_pix_fmt) rather than being folded into an integer B/px constant
  // that would have to round.  Same reason the reader's comment round up.
  //
  // The pinned term is only real when the pinned allocation is attempted at all,
  // which is `use_gpu && !opt.gpu_float_out` -- the same condition as the branch in
  // the loop below, and for the same reason: on the float-out path the device neither
  // DMAs from the reader's buffer nor writes a pinned output, so the pageable vectors
  // are the only buffers that exist.  This is also why the figure cannot live in
  // Pipeline's constructor: `use_gpu` is resolved further DOWN, in this function, and
  // the depth is not known until after the budget has been divided.
  const bool pinned_slots = use_gpu && !opt.gpu_float_out;
  // ONE definition of "will anything read b.pixels", used by the budget here AND by
  // the allocation further down.  They must agree: if the budget counted the float
  // buffer while the allocation skipped it, the reported queue would be the depth the
  // machine could not actually run, which is the over-commit direction.  It is
  // under-claiming today rather than over-claiming, so this is a correctness fix and
  // not a safety one -- but it is the whole reason the 4050 MiB reaches the queue.
  //
  // Names the RESOLVED engine.  use_gpu is settled at :2720 and cpu_threads at :2702,
  // both above here.  c5efb68 was a crash because a neighbouring condition named the
  // REQUESTED branch instead.  (The line numbers were 2531 and 2609 until edits above
  // this one moved them; a citation that has drifted is worse than none, because it
  // reads as if it were checked.)
  //
  // WAS: !use_gpu || opt.gpu_float_out || cpu_threads > 0, which is now WRONG in the
  // other direction -- it under-claims, so the float buffer was still allocated for
  // every configuration. With the gather fused, the host worker widens the decoder's
  // own bytes per visit, so `b.pixels` has no consumer in any configuration except
  // one: a GPU asked to RETURN float4. That is the whole condition.
  //
  // The three cases, and why each is what it is:
  //   GPU, not gpu_float_out -- gathers on the DEVICE from b.in() (upload_u16 and
  //     emit_u16, established by experiment, not by inspection).
  //   host worker              -- gathers on the HOST from b.in(), fused.
  //   writer's convert         -- skipped, because both set raw_ready, so it never
  //     sweeps b.pixels.
  // The one that still needs float4 is gpu_float_out, where the device hands floats
  // back. Hence the condition is `&&`, not `||`.
  const bool host_touches_pixels = use_gpu && opt.gpu_float_out;
  const std::size_t per_batch =
      static_cast<std::size_t>(batch) *
      ((host_touches_pixels ? pixels * sizeof(RgbaF) : 0) + in_frame_bytes +
       (pinned_slots ? out_frame_bytes
                     : pixels * 4 * sizeof(std::uint16_t)));
  // The page-locked part of that, for the reserve report.  Zero when the allocation is
  // not attempted.  Reported rather than subtracted from the budget, because it IS the
  // queue: subtracting it here as well as counting it per slot is the double count that
  // made the old figure (a flat `pinned` sized in float4 bytes per GPU *worker*, not
  // per slot) both wrong and untraceable.
  const std::size_t per_batch_pinned =
      pinned_slots ? static_cast<std::size_t>(batch) *
                        (in_frame_bytes + out_frame_bytes)
                   : 0;
  pipe.set_slot_bytes(per_batch, per_batch_pinned);
  int depth;
  {
    std::size_t affordable = per_batch ? pipe.ram_budget() / per_batch : 2;
    // The floor of two is a LIVENESS requirement -- one slot in the dither and one in
    // the writer -- not a throughput choice.  It is allowed to exceed the budget,
    // because a pipeline that cannot run is worse than one that pages, but it is
    // reported when it happens, and it is what makes the ** OVER FREE ** flag on the
    // [ram] line reachable.
    if (affordable < 2) {
      if (per_batch * 2 > pipe.ram_budget()) {
        std::fprintf(stderr,
                     "warning: two batches need %.0f MiB but the RAM budget is "
                     "%.0f MiB; the pipeline will page or fail\n",
                     static_cast<double>(per_batch * 2) / (1024.0 * 1024.0),
                     static_cast<double>(pipe.ram_budget()) / (1024.0 * 1024.0));
      }
      affordable = 2;
    }
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
    //
    // CORRECTION 2026-10-09: the conclusion above is wrong, and the measurement behind
    // it was too narrow.  The GPU worker does idle waiting for the reader -- but the
    // reason is the reader's SUPPLY RATE, not the number of slots, so slots cannot fix
    // it.  Measured on the 600-frame 1080p clip: reader 8227 ms of VideoProcess's
    // 8860 ms (92.9%), dither 6203 ms, encode 4373 ms.  Per 16-frame batch the reader
    // supplies in 219 ms while dither consumes in 163 ms, so the worker drains faster
    // than frames arrive; queue 3 and queue 553 buffer the same starved stream.  And the
    // reader is ONE ffmpeg process (a single std::thread at :3400 calling one
    // decoder.Read() per frame at :3433), so no amount of buffering or worker count makes
    // it faster.
    //
    // Both the palette/reader overlap and this dial were expected to win and both lost,
    // for this one reason: 93% of the runtime is one serial decoder, and neither change
    // touches it.  See the design spec's section 9.
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
    // RESEARCH ONLY -- DO NOT MAKE THIS CONDITIONAL YET.  Not edited below.
    //
    // `float_path_` is the constant true, so this runs on EVERY video run and costs
    // sizeof(RgbaF) = 16 B/px per slot: 530,841,600 B = 506.25 MiB per slot at
    // 1920x1080 with --batch-frames 16, and 4,050 MiB at a queue depth of 8, which is
    // 72.7% of that configuration's whole queue (5,568.7 MiB at the real 22 B/px).
    // On the default video path nothing below reads it:
    //
    //   * the reader's widening at :3210 is gated on
    //     `!use_gpu || opt.gpu_float_out || cpu_threads > 0`, which is false for a
    //     default run (GPU present, float-out off, cpu_threads auto-resolved to 0);
    //   * the writer's convert at :3517 is gated on `!raw_ready`, and the GPU sets
    //     raw_ready whenever upload_u16 && emit_u16, which is the default;
    //   * the CUDA blocks engine allocates no device pixel buffer when upload_u16 (so
    //     d_pixels_buf is not even cudaMalloc'd -- rd_blocks_cuda.cu:1001) and passes
    //     d_pixels_out = nullptr to the scatter when emit_u16 (:1180), so the host
    //     buffer is neither uploaded nor downloaded on that path.
    //
    // WHAT WOULD HAVE TO HOLD for the allocation to be droppable.  Every consumer must
    // either be unreachable under the same condition the allocation would be gated on,
    // or tolerate a null/empty b.pixels.  The set of consumers in these two files is:
    //
    //   1. RawYuv444ToFloatsParallel / RawToFloatsParallel -- the reader's widening,
    //      rd_video.cpp:3216 and :3219, both writing b.pixels.data() as a
    //      non-null destination.  Already gated on `!use_gpu || gpu_float_out ||
    //      cpu_threads > 0`.  DROPPABLE only if that gate and the allocation gate are
    //      the SAME predicate -- the exact lesson of the 0xC0000005 recorded on
    //      float_path_ and of the --input-mode yuv420 refusal above.
    //   2. FloatsToYuv444Parallel / FloatsToRawParallel -- the writer's convert,
    //      rd_video.cpp:3534 and :3538, reading b.pixels.data().  Already gated on
    //      `!b.raw_ready`.
    //   3. RiemersmaBlocksCpu -- the host worker, rd_video.cpp:3392, writing
    //      b.pixels.data().  Reached whenever cpu_threads > 0, i.e. --cpu-threads N
    //      on a GPU machine.  MUST keep the buffer whenever any host worker can run.
    //   4. RiemersmaBlocksCuda / RiemersmaBlocksOpencl -- rd_video.cpp:3377 and
    //      :3372, passing b.pixels.data() as the `batch` argument.  THE ONE THAT IS
    //      NOT SETTLED.  Passing nullptr is safe only if the engine never dereferences
    //      it, and the two engines demonstrably differ: CUDA does not touch it under
    //      upload_u16, but OpenCL's check_u16 path reads AND WRITES batch[i]
    //      (rd_opencl.cpp:1517-1531, 1631, 1639) and its float-upload path does
    //      MakeBuffer(..., batch, ...) at :1305.
    //
    // WHAT WOULD SETTLE THE OPENCL QUESTION.  Not inspection of the flags; a run.
    // `RD_OCL_CHECK_U16=1` exists precisely to run the float and uint16 paths against
    // each other on one input (rd_opencl.cpp:1483-1511) and it dereferences `batch` on
    // the float pass, so it is the instrument already in the tree.  What is needed is
    // one `--engine opencl --video` run with b.pixels allocated (today's behaviour)
    // and the same run with it not allocated, over the same fixture, comparing the
    // output files byte-for-byte -- plus one run with RD_OCL_CHECK_U16=1 under each.
    // If they match, the allocation is droppable for OpenCL; if the CHECK_U16 run
    // differs, it is not, and that is a NULL DEREFERENCE risk rather than a slow path,
    // which is why this stays a note.
    // `b.pixels` is 16 B/px -- 506.25 MiB per slot at 1080p and batch 16, so 4050 MiB
    // across a depth-8 queue.  It is only needed when something will actually READ it,
    // and on the default GPU video path (GPU on, gpu_float_out off, no host workers)
    // nothing does:
    //
    //   - the reader's widening into it is gated on `host_touches_pixels` (:2882)
    //   - the writer's convert is gated on !raw_ready, and the GPU writes raw itself
    //   - rd_blocks_cuda.cu:1001 allocates no device pixel buffer under
    //     upload_u16 && emit_u16, so the pointer it is handed goes unread
    //
    // ESTABLISHED BY EXPERIMENT, not by reading the code.  Gating the allocation off
    // and byte-comparing at 320x180: blocks and opencl produced IDENTICAL decoded md5
    // both with and without it, including under RD_OCL_CHECK_U16=1, whose OpenCL path
    // reads AND writes batch[i] and was the reason this was not simply assumed.  The
    // negative control is what makes that meaningful -- the same gate on
    // --gpu-float-out and on --engine cpu gives 0xC0000005, an access violation, so the
    // experiment does detect a missing buffer.  tools/probe-host-oracle.ps1 and
    // artifacts/pixels_probe.ps1 are that experiment.
    //
    // So the predicate names the RESOLVED engine, not the request.  That distinction
    // has bitten this file before: c5efb68 was a crash because the condition named the
    // requested branch.  use_gpu is resolved at :2720 and cpu_threads at :2702, both
    // above this point, so all three inputs are known here.
    // host_touches_pixels is declared once above, where the RAM budget uses it, so the
    // figure reported and the memory actually taken cannot drift apart.
    if (pipe.float_path() && host_touches_pixels) {
      b.pixels.resize(static_cast<std::size_t>(batch) * pixels);
    }
    if (pinned_slots) {
      // Page-locked, so the reader writes straight into memory the copy engine DMAs
      // from.  These bytes cannot be paged out.  `per_batch` above already counts them,
      // which is why nothing is subtracted for them here: the old code did both, and
      // the [ram] peak counted the same bytes twice.
      //
      // The input is 3 bytes per pixel in planar 4:4:4, 1.5 in planar 4:2:0 and 8 in
      // rgba64le -- hence in_frame_bytes as a whole-frame figure rather than a per-pixel
      // one, since 1.5 is not an integer and truncating it desynchronises the reader
      // by a byte a frame.  The output is out_frame_bytes: 3 per pixel for the default
      // planar yuv444p, 8 for rgba64le.
      //
      // The PAGEABLE fallback is now sized from out_frame_bytes as well, which closes
      // the gap this note used to leave open.  It was `batch * pixels * 4 uint16`
      // -- 8 B/px -- regardless of the output format, so on the default planar
      // yuv444p output it held 265,420,800 B per slot where the pinned buffer it
      // stands in for holds 99,532,800 B: 165,888,000 B and 158.22 MiB per slot
      // of pure over-allocation, on the path that runs precisely when memory is
      // already short.  The budget counted the true 3 B/px, so the reported queue
      // was never wrong -- the process just held more than it said.
      //
      // It was left alone out of fear of the silent zero-frame bug below, which is
      // a fair reason to be careful here and not a reason to keep 2.67x.  What makes
      // this safe is that `batch * out_frame_bytes` is not a new bound: it is the
      // size the PINNED buffer has always been allocated at, in production, on
      // every run.  And every consumer of `out()` is inside that bound, which is
      // what was actually needed to be shown rather than assumed:
      //
      //   * both engines, and RiemersmaBlocksCpu, write one out_frame_bytes per
      //     frame -- `emit_yuv444` is set from `out_yuv444` at the call site;
      //   * FloatsToYuv444Parallel writes pixels*3, which IS out_frame_bytes when
      //     out_yuv444, and FloatsToRawParallel writes pixels*8, which IS
      //     out_frame_bytes when !out_yuv444 -- so the two writers agree with the
      //     one constant either way;
      //   * the writer pipes frames * out_frame_bytes.
      //
      // So nothing in this file reads or writes a byte of b.out16 past
      // batch * out_frame_bytes, in any configuration.
      const std::size_t in_bytes = static_cast<std::size_t>(batch) * in_frame_bytes;
      const std::size_t out_bytes =
          static_cast<std::size_t>(batch) * out_frame_bytes;
      b.in16_pin = static_cast<std::uint16_t*>(CudaAllocPinned(in_bytes));
      b.out16_pin = static_cast<std::uint16_t*>(CudaAllocPinned(out_bytes));
      // CudaAllocPinned is a CUDA function.  In a --no-cuda build it returns nullptr,
      // and the OpenCL engine is perfectly usable there -- so this branch is taken and
      // then yields nothing at all.
      //
      // in() and out() already fall back to the pageable vectors when the pinned
      // pointers are null, which is the right design and was clearly meant for exactly
      // this case.  But those vectors were only ever SIZED in the else branch below, so
      // the fallback handed the engine an EMPTY vector: nothing to read from, nowhere
      // to write to.  The engine then did nothing, reported success, and the run
      // finished with 0 frames, exit 0, and a 572-byte container.  That is the whole
      // OpenCL-on-video fault in a --no-cuda build, and it survived because the failure
      // is silent rather than loud.
      //
      // So size them whenever the pinned allocation did not happen.  The condition is
      // on whether the allocation SUCCEEDED, not on which engine was requested -- which
      // is the same mistake as the float_path_ crash in c5efb68, one layer down: there
      // the buffer was allocated on the requested engine rather than the resolved one,
      // here it is sized on the requested branch rather than the one that was taken.
      if (b.in16_pin == nullptr) {
        b.in16.resize(in16_elems_for(batch));
      }
      if (b.out16_pin == nullptr) {
        b.out16.resize(out16_elems_for(batch));
      }
    } else {
      b.in16.resize(in16_elems_for(batch));
      b.out16.resize(out16_elems_for(batch));
    }
  }
  std::deque<int> free_slots;
  for (int i = 0; i < depth; ++i) free_slots.push_back(i);
  std::deque<int> ready;
  std::deque<int> done;
  // The next source frame the writer is allowed to emit.  Batches land on `done` in
  // whatever order their workers finish, and the encoder is a stream, so this is what
  // turns completion order back into source order.
  int next_write = 0;
  // True when the batch holding `next_write` has finished.  Called with `mu` held.
  const auto HasTurn = [&](int want) {
    for (int s : done) {
      if (slots[static_cast<std::size_t>(s)].first_frame == want) return true;
    }
    return false;
  };

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
  // THE THREE STAGE TIMERS ARE NOT CPU TIME AND NOT A PARTITION OF THE WALL CLOCK.
  // They are summed worker-thread intervals, and that has three consequences the
  // summary line in rd_cli.cpp currently hides:
  //
  //   * dither_ms is accumulated by EVERY dither worker (see the `dither_ms += dt`
  //     below, inside the per-worker loop), so with N workers it counts N threads'
  //     elapsed time and can exceed the wall clock by nearly N.  It is a measure of
  //     work done, not of how long anything took.
  //   * decode_ms is accumulated inside the reader's per-frame loop, so it is a sum
  //     over frames of time blocked in Read -- it measures the decoder's supply rate
  //     and is not the reader's cost either.
  //   * encode_ms is a sum over batches on the single writer thread, and is split into
  //     convert_ms and pipe_ms, which DO partition it (convert_ms + pipe_ms ==
  //     encode_ms to within a NowMs() call).
  //
  // So the summary's "palette | decode | dither | encode | wall" line adds four
  // incommensurable quantities: palette_ms is real wall clock (VideoBuildPalette is
  // single-threaded and measures NowMs() - t0), total_ms is real wall clock, and the
  // three middle ones are per-thread sums that need not add to anything.  The file's
  // own design notes say this twice -- "stage times are not CPU" and "the naive
  // reading is misleading" -- and then the printed line invites exactly the naive
  // reading, by putting them in a row under a single label.
  //
  // The producers are correct and unchanged; they are measuring what they claim to.
  // What is wrong is the LABEL, and it lives in PrintVideoSummary at rd_cli.cpp:271,
  // which this file does not own.  See the request at the end of this file's changes.
  // What the consumer needs, precisely: the line should not present the three as
  // components of `wall`.  Either sum only the quantities that partition the wall
  // (palette + a genuine wall-clock pipeline figure), or label the middle three as
  // summed worker time and say so in the label itself rather than in a comment nobody
  // reading the output will see.
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
  // `use_cache` is deliberately left at DitherParams' own default.  It was explicitly
  // false here, which read as a decision and was not one.  The only reader of the field
  // is RiemersmaWalkCpu -- the memo table is allocated at rd_riemersma_cpu.cpp:334-336
  // and consulted at :71 and :85 inside Visit, and Visit is called only from
  // RiemersmaWalkCpu (:346, :356, :364).
  //
  // (CORRECTION: "the only reader" was FALSE as written.  rd_riemersma_cuda.cu:583 also
  // reads `params.use_cache`, into the kernel's `use_cache`.  The CONCLUSION below still
  // holds -- the video path calls RiemersmaBlocksCpu, whose inner loop walks the tree
  // inline and is cache-free on both the host and the device precisely so the two agree
  // ("Cache-free on both paths, so the two agree", rd_riemersma_cpu.cpp:612) -- but the
  // premise it rested on was wrong, and a comment that reaches for "only" should be
  // checked against every reader rather than the one that was in mind.)
  //
  // This file never calls it: the video path
  //
  // So the assignment set a flag no code in the video path could observe, and the
  // default already said the same thing.  Zero behavioural difference: `false` and
  // `DitherParams::use_cache`'s `true` are only distinguishable through the memo table,
  // and RiemersmaBlocksCpu never builds one.
  //
  // The day this WOULD matter is a future engine that grows a memo table inside
  // RiemersmaBlocksCuda or RiemersmaBlocksOpencl -- both take `params` and both already
  // hand params.diffusion down to the kernel.  Then it should be set explicitly, and
  // deliberately rather than by inheriting whichever default.

  if (!opt.quiet) {
    std::fprintf(stderr,
                 "[video] %dx%d  batch=%d  queue=%d (%.0f MiB of %.0f MiB RAM)  "
                 "cpu_workers=%d  gpu=%s x%d\n",
                 info.width, info.height, batch, depth,
                 static_cast<double>(pipe.in_flight_bytes()) / (1024.0 * 1024.0),
                 static_cast<double>(pipe.ram_budget()) / (1024.0 * 1024.0),
                 cpu_threads, use_gpu ? "yes" : "no", gpu_workers);
    // Print the ceiling, not just the plan.  The queue is the part the budget
    // governs; the reserve is what the queue does not cover, and the sum is the real
    // peak.  This is the number to check before raising --mem-fraction or
    // --queue-depth on a machine that is already busy -- so it has to be the number
    // the allocations add up to.
    //
    // The reserve used to also carry a flat `pinned` term, sized as
    // pixels * batch * sizeof(RgbaF) * gpu_workers: float4 bytes per GPU *worker*.
    // The real page-locked buffers are per SLOT and are the input and output frame
    // buffers, not a float4 batch, so that term matched no allocation -- and since
    // the per-slot figure already counted the same memory, adding it here as well
    // counted it twice.  It is now reported separately, as the page-locked COMPONENT
    // of the queue, because that is the fact worth knowing (it cannot be paged out)
    // and it is already inside queue_mb.
    {
      const double queue_mb =
          static_cast<double>(pipe.in_flight_bytes()) / (1024.0 * 1024.0);
      const double pinned_mb =
          static_cast<double>(pipe.in_flight_pinned_bytes()) / (1024.0 * 1024.0);
      const double reserve_mb =
          static_cast<double>(pipe.reserved()) / (1024.0 * 1024.0);
      const double avail_mb =
          static_cast<double>(pipe.avail_phys()) / (1024.0 * 1024.0);
      const double ceiling = queue_mb + reserve_mb;
      std::fprintf(stderr,
                   "[ram]    %.0f MiB of %.0f MiB free at start; queue %.0f "
                   "(%.0f of it page-locked) + reserve %.0f = ~%.0f MiB peak%s\n",
                   avail_mb, avail_mb, queue_mb, pinned_mb, reserve_mb, ceiling,
                   avail_mb > 0.0 && ceiling > avail_mb ? "  ** OVER FREE **"
                                                       : "");
    }
    if (cpu_threads > 0) {
      // Only offer --cpu-threads 0 when there is a GPU to honour it.  Saying it
      // unconditionally told users on a GPU-less build to pass a flag that means
      // "GPU only" there, which is the request that used to crash.
      std::fprintf(stderr,
                   "[video] note: the %d host worker(s) use the host block walk, "
                   "which is NOT bit-identical to the verified CUDA walk -- "
                   "measured ~1.6%% of pixels differ (visually indistinguishable). "
                   "%s\n",
                   cpu_threads,
                   use_gpu ? "For bit-exact output use --cpu-threads 0."
                           : "There is no GPU in this build, so --cpu-threads 0 is "
                             "not available: it means \"GPU only\".");
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
        // `|| cpu_threads > 0` is the fix, and it is the same mistake as the two above:
        // the condition named the GPU, but the CONSUMER of `b.pixels` is not the GPU.
        // Host workers dither `b.pixels`, and they exist whenever cpu_threads > 0 --
        // which on a GPU machine is exactly what `--cpu-threads N` asks for.  With a
        // GPU present, gpu_float_out off, and `--cpu-threads 4`, none of the original
        // three disjuncts held, so the reader never widened into `b.pixels` and the
        // host workers dithered a zero-initialised buffer: measured mean R 56.37
        // against 120.74 for the same clip with the default `--cpu-threads 0`, and
        // byte-identical for N=1 and N=4.  Silently, deterministically, exit 0.
        //
        // It now reads `host_touches_pixels` rather than re-spelling the three
        // disjuncts, because it WAS re-spelling them.  That predicate is documented
        // as "declared once ... so the figure reported and the memory actually taken
        // cannot drift apart", and this second copy is exactly the drift it exists to
        // prevent -- two sites to edit, one of which the note above never mentions,
        // and a mismatch here is not a slow path but a black or garbage frame.  The
        // two conditions are equal today; the duplication is the defect, not the
        // value.
        if (host_touches_pixels) {
          if (use_yuv444) {
            // Planar 8-bit, three BYTES per pixel.  RawToFloats would read this as
            // uint16 and land on frame f-1's chroma planes, which is right for frame 0
            // and nonsense for every frame after it.
            RawYuv444ToFloatsParallel(
                reinterpret_cast<const unsigned char*>(b.in()), b.pixels.data(),
                pixels, filled, reader_convert_threads);
          } else {
            RawToFloatsParallel(b.in(), b.pixels.data(), pixels, filled,
                                reader_convert_threads, dec_channels);
          }
        }
        b.frames = filled;
        {
          std::lock_guard<std::mutex> lock(mu);
          // Taken before the increment below, so this is the batch's FIRST source frame
          // and not its last-plus-one.  frames_read is reader-only up to this point.
          b.first_frame = static_cast<int>(frames_read);
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
  // Must start at NOW, not at 0.  `since = NowMs() - last_gpu_ms` measures the gap
  // since the GPU last took a batch, and `NowMs()` is an ABSOLUTE steady_clock count
  // -- large, not elapsed.  Initialised to 0, the first evaluation computed a `since`
  // of roughly 2^61 ms, `since > budget` was trivially true, and every host worker
  // broke out of its grace loop on its first turn and took a batch.  After that the
  // flag would have worked -- except the host had usually already drained `ready`, so
  // the GPU never got a look in.  Which is why `--cpu-threads N>0` on a GPU machine
  // ran the HOST WALK for essentially the whole clip.
  //
  // Verified with the flag's own knob: `--host-grace-ms 100000` produced a
  // BYTE-IDENTICAL file to the default 60 ms, which is what a setting that changes
  // nothing looks like.  With this initialisation the two differ.
  std::atomic<int64_t> last_gpu_ms{static_cast<int64_t>(NowMs())};
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
              const int64_t since = static_cast<int64_t>(
                  NowMs() - static_cast<double>(last_gpu_ms.load()));
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
        // The palette is built concurrently with this whole function, so a worker can reach
        // here with a decoded batch and no palette yet.  Wait HERE rather than at the top of
        // VideoProcess: the reader thread starts ~180 lines earlier, so a Wait() at function
        // entry would block BEFORE the reader was ever spawned and the two stages would not
        // overlap at all.  That version compiles, gates green, and saves nothing.
        gate.Wait();
        if (!gate.ok()) {
          // The palette build failed while this worker was waiting.  Record it and unwind
          // rather than dither against an absent palette; VideoProcess reports the error.
          std::lock_guard<std::mutex> gate_lock(mu);
          failed = true;
          cv_done.notify_all();
          return;
        }
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
              gate.palette(), params, gate.tree(), info.width, info.height, b.pixels.data(),
              blocks, &device, raw_written ? b.out() : nullptr, gpu_index,
              blocks.upload_u16 ? b.in() : nullptr);
        } else {
          err = rd::RiemersmaBlocksCuda(
              gate.palette(), params, gate.tree(), info.width, info.height, b.pixels.data(),
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
        // Fused output: the scatter writes the encoder-ready bytes itself, so the
        // writer's convert pass is deleted rather than run over a buffer nobody reads.
        // raw_ready is what makes the writer skip it -- set it HERE, where the bytes
        // were produced, not at the writer.  That ordering is the whole hazard: if the
        // writer ran its convert over a buffer the scatter had already filled with
        // planar YUV bytes, it would reinterpret 3 B/px as 16 B/px and produce garbage
        // with no error, which is the failure mode the note above the writer's convert
        // describes for the --gpu-float-out case.
        //
        // raw_written is set ONLY on success.  It used to be set unconditionally, and
        // that was a silent wrong encode: RiemersmaBlocksCpu returns void, so a refusal
        // (the >256-colour check, or frames < 1) returned without writing b.out(), and
        // raw_ready then told the writer to skip its convert -- so the encoder received
        // whatever the PREVIOUS batch left in that buffer.  Right frame count, right
        // palette, plausible picture, wrong pixels.
        std::string cpu_err;
        // Fused gather: hand the engine the decoder's bytes and the stride between
        // frames, and let it widen per visit. `b.pixels.data()` is passed but will be
        // nullptr on most configurations now -- `host_touches_pixels` no longer
        // allocates it -- and the engine does not read `batch` when gather_layout != 0.
        //
        // Layout 3 is planar 8-bit 4:4:4, which is what `use_yuv444` decodes to;
        // anything else here is the interleaved uint16 path. `use_yuv420` never reaches
        // this call: the host refusal above it returns first, because 4:2:0 chroma has
        // to be reconstructed and this engine has no upsampler for it.
        const int gather_layout = use_yuv444 ? 3 : 4;
        // `b.in()` is uint16*; the planar layout is the SAME memory viewed as bytes,
        // which is how the reader already hands it to RawYuv444ToFloatsParallel. No
        // separate 8-bit buffer exists -- the cast is the whole difference.
        RiemersmaBlocksCpu(gate.palette(), params, gate.tree(), info.width, info.height,
                           b.pixels.data(), b.frames, std::max(16, opt.block),
                           reinterpret_cast<unsigned char*>(b.out()), pixels,
                           out_yuv444, &cpu_err,
                           reinterpret_cast<const unsigned char*>(b.in()),
                           in_frame_bytes, gather_layout, dec_channels);
        if (!cpu_err.empty()) {
          // Hard stop, not a note.  Reported once per process because this runs on
          // every host worker on every batch.
          //
          // NOTE: no `return` here.  An early return skips the worker's normal exit
          // path -- the slot is never returned to free_slots and cv_done is never
          // signalled -- so the writer waits on a condition variable that will never be
          // met.  That showed up as 0xC0000409, a fail-fast, which is louder than the
          // bug but is still the wrong shape of failure.  Falling through lets the
          // worker clean up after itself; the pipeline then unwinds through
          // g_interrupted, and VideoProcess returns non-zero because of the recorded
          // error below.
          static std::atomic<bool> cpu_err_reported{false};
          bool expected = false;
          if (cpu_err_reported.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr, "error: host dither failed: %s\n", cpu_err.c_str());
          }
          SetHostDitherError(cpu_err);
          g_interrupted.store(true);
          // raw_written deliberately stays FALSE: this batch has no valid fused output,
          // so the writer must not be told the bytes are there.
        } else {
          raw_written = true;
        }
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
        return HasTurn(next_write) || eof || failed;
      });
      // Waiting for the TURN rather than for any batch.  A batch that has finished out of
      // turn has to stay on `done` until its predecessor is written, which is the whole
      // fix: the encoder is a stream, so the order frames reach it is the order they
      // appear in the file.  Taking whatever arrived first is what permuted the output.
      if (!HasTurn(next_write) && !eof && !failed) {
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
      const auto it = std::find_if(done.begin(), done.end(), [&](int s) {
        return slots[static_cast<std::size_t>(s)].first_frame == next_write;
      });
      if (it == done.end()) {
        // Everything on `done` belongs to a later part of the source.  Leave it there;
        // the predicate above will wake us when the missing batch lands.
        if (eof && frames_dithered >= frames_read) break;
        continue;
      }
      slot = *it;
      done.erase(it);
    }
    Batch& b = slots[static_cast<std::size_t>(slot)];
    const double e0 = NowMs();
    // Only the host paths still need this.  On the GPU path the scatter kernel wrote
    // `out16` itself, so this would be pure waste -- and it would overwrite correct
    // bytes with a float4 buffer that no longer holds the result.
    if (!b.raw_ready) {
      if (out_yuv444) {
        // The host has to do what the device does on the GPU path.  Without this the
        // writer would call FloatsToRaw, hand the encoder rgba64le, and the encoder
        // would read it as planar 4:4:4 -- see FloatsToYuv444 for what that cost.
        //
        // The `&& !use_gpu` this used to carry was wrong, and wrong in the direction
        // that produces garbage rather than an error.  We are already inside
        // `if (!b.raw_ready)`, which has answered the only question that matters: DID
        // THE DEVICE WRITE THESE BYTES?  If it did not, the host must convert, and
        // whether a GPU exists is irrelevant.  With `--gpu-float-out` a GPU is present
        // and `raw_ready` is false -- the float path returns pixels instead of writing
        // the scatter -- so the old condition took the else branch and pushed
        // rgba64le at 8 bytes per pixel into a pipe the encoder was told is yuv444p at
        // 3.  Measured: 97.1% of samples differed from the same render without the
        // flag.  With RD_YUV444_OUT=0 the two runs were byte-identical, which is what
        // isolated it to this branch rather than to the flag.
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
    // Advanced by what was WRITTEN, not by b.frames: a batch straddling the segment
    // boundary is clipped above, and advancing by the unclamped size would skip past a
    // frame the encoder never saw and stall the predicate forever.
    next_write += frames_to_write;
    cv_done.notify_all();
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
    // Name the codec that was asked for.  "ffmpeg encode failed" on its own sends the
    // reader to ffmpeg, when the thing worth knowing is which of their own arguments
    // ffmpeg rejected -- and an unknown encoder name is the usual reason, because
    // ffmpeg reports it as a generic failure rather than as a bad argument.
    *error = "ffmpeg encode failed (codec: " +
             std::string(opt.lossless ? "ffv1" : opt.codec) +
             ").  `ffmpeg -encoders` lists what this build supports; an unknown name "
             "is reported by ffmpeg as a plain encode failure, not as a bad argument.";
    return false;
  }

  // Zero frames dithered is not a success, and nothing above noticed.
  //
  // The encoder is opened before the first frame reaches it, so a run that
  // dithered nothing still closes it cleanly and ffmpeg still exits 0.  The
  // result is a container with no frames in it: measured at 572 bytes, which
  // ffprobe rejects as malformed ("Duplicate element", "invalid as first byte of
  // an EBML number", "End of file").  So this returned true, rdither printed
  // "frames     : 0 in ..." and exited 0 -- a file on disk, no diagnostic, and
  // nothing playable in it.  gpu_error is empty and encode_code is 0 on that
  // path, which is why neither of the two checks above caught it.
  //
  // An input that genuinely has no frames is a different thing and gets its own
  // message, because "the clip is empty" is the user's situation to fix and
  // "the dither produced nothing from a clip that has frames" is ours.
  if (frames_dithered == 0) {
    if (info.frames > 0) {
      *error = "no frames were dithered, but the input has " +
               std::to_string(static_cast<long long>(info.frames)) +
               " -- the encoder was closed cleanly, so the output holds no "
               "frames.  This is a fault, not an empty clip.";
    } else {
      // NOT "the input has no decodable frames".  That was measured to be wrong:
      // ffprobe reports 0 for a Matroska whose frame count it cannot determine,
      // and in the case that reached here the palette stage had already sampled a
      // frame from the very same input.  So the count being 0 says nothing about
      // the input, and blaming it sends the reader looking in the wrong place.
      *error = "no frames were dithered, and ffprobe reported 0 frames for the "
               "input, so this is NOT an empty clip -- the palette stage read "
               "frames from it.  Something selected but never ran; check the "
               "engine line above.";
    }
    return false;
  }

  if (!HostDitherError().empty()) {
    // A host worker refused a batch.  The pipeline has already been told to unwind, and
    // the frames it did produce are not a complete render, so the caller must be able
    // to tell that apart from success.  Without this the run reported a plausible
    // summary and exit 0 over a partial encode.
    std::fprintf(stderr, "error: %s\n", HostDitherError().c_str());
    if (result != nullptr) result->frames = 0;
    return result;
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
    //
    // WHAT THIS INTERVAL IS NOT, because the summary prints a rate against it and the
    // progress bar prints a DIFFERENT one against a different interval.  The bar is
    // constructed at :3448, after VideoBuildPalette has returned, so its rate EXCLUDES
  // the palette; this figure INCLUDES it.  On the published 18001-frame run -- palette
  // 28677 ms, wall 352507 ms -- the bar would print 18001 / (352507 - 28677) =
  // 55.6 frames/s and the summary would print 18001 / 352507 = 51.1 frames/s, on
  // screen at the same time, ~8% apart, and neither labelled.
  //
  // The summary's interval is the correct one and should stay inclusive: the palette
  // is serial, the user waited through it, and a throughput figure that excludes a
  // stage the user sat and watched is not throughput.  What has to change is the
  // reader's ability to tell which interval a number covers.  Both the fix and the
  // label are in rd_cli.cpp (PrintVideoSummary) and rd_progress.cpp, neither of which
  // this file owns -- the requests are in the note above PrintVideoSummary there.
  // The bar's own start clock is Progress' start_ms_, set in its constructor at :3448
  // and not settable afterwards, so seeding it with the palette's cost needs a
  // Progress API change and not a call-site one.
    result->total_ms = (NowMs() - t_start) + result->palette_ms;
    // in_flight_mb is the QUEUE, from the same per-slot figure the depth was derived
    // from, so the number printed next to "queue=N" is N times what one slot actually
    // holds.  It used to be 32 bytes per pixel per slot for every configuration --
    // 1,012.50 MiB per slot at 1080p/batch 16 -- while the real figure ranges from
    // 696.09 MiB (GPU yuv444, the default) to 1,012.50 MiB (host rgba64).  See the
    // table above set_slot_bytes.
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

// ---------------------------------------------------------------------------
// Requests for files this one does not own
// ---------------------------------------------------------------------------
//
// Recorded here rather than left in a review, because each one is a number or a label
// a user reads and each is wrong now.
//
// 1. src/rd_cli.cpp:271, PrintVideoSummary.  The line reads
//
//        busy time  : palette %.0f | decode %.0f | dither %.0f | encode %.0f | wall %.0f
//
//    and presents four quantities as if they decomposed `wall`.  They do not:
//    palette_ms and total_ms are wall clock, while decode_ms, dither_ms and
//    encode_ms are summed worker-thread intervals (dither_ms in particular is summed
//    once per dither worker, so it scales with the worker count and can exceed the
//    wall clock).  Only convert_ms and pipe_ms partition anything -- they split
//    encode_ms.  Please either drop the three from the wall arithmetic or rename the
//    label so it says "summed worker time", and if the second line about the writer
//    is kept, say that those two ARE a partition of the encode figure.  The producers
//    here are correct and are not asking to change.
//
// 2. src/rd_cli.cpp:266, PrintVideoSummary, and the Progress bar in
//    src/rd_progress.cpp.  Two throughput numbers are printed on screen at the same
//    time and they cover different intervals.  The summary's fps divides by
//    total_ms, which includes the palette (result->total_ms above adds palette_ms to
//    the pipeline's own elapsed time).  The bar divides by its own start_ms_, set in
//    the Progress constructor at src/rd_video.cpp:3448, which runs after
//    VideoBuildPalette has returned -- so it excludes the palette entirely.  On the
//    published 18001-frame run (palette 28677 ms, wall 352507 ms) that is 55.6/s on
//    the bar against 51.1/s in the summary, ~8% apart, neither labelled.  Two fixes,
//    either acceptable: give Progress a way to start its clock earlier (a constructor
//    parameter or a SetStartOffset, since start_ms_ is private and set once), or label
//    the bar's rate as excluding the palette.  The summary's interval should stay
//    inclusive -- the palette is serial and the user waited through it.
//
// 3. src/rd_video.cpp is the only file that has needed a bytes-per-pixel figure for a
//    queue slot, and it is now derived once from the allocations (see the table above
//    set_slot_bytes).  If another path ever needs the same number -- an image batch, a
//    plugin -- it should come from that one place rather than a fifth guess.
//
// 4. Pipeline::reserved() is now ffmpeg's decoder headroom alone, because every other
//    byte the process allocates is a queue slot.  Anything that used to read
//    reserved() expecting a figure that included page-locked staging will see a smaller
//    number by exactly the pinned term.  In this file the only reader is the [ram]
//    line, and it adds the figure to in_flight_bytes(), which counts the pinned bytes
//    already -- so the peak is unchanged and correct.  There is no other reader in the
//    tree, but the semantics changed, so a grep for reserved() is worth doing before
//    anything else reaches for it.

}  // namespace rd
