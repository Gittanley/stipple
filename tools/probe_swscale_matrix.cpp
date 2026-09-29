// probe_swscale_matrix.cpp -- determine the exact structure of swscale's 8-bit
// YUV -> 16-bit RGB conversion, empirically.
//
// Why
// ---
// Round fourteen established that the mapping is none of the four classic
// integer forms, which rules out a formula but says nothing about what to build.
// To port it we need its *shape*:
//
//   * separable, R = f(Y) + g(V)?   (two 256-entry tables, trivially portable)
//   * or a genuine 2-D function of (Y,V)?
//   * is R independent of U, as BT.601 requires?
//   * what is the output quantisation?  (a table path gives coarse steps)
//
// ffmpeg is the oracle for all of them.  yuv444p throughout, so no chroma *filter*
// is in play: this isolates the colour matrix, which is what has to be reproduced
// on the device.
//
// The equivalent PowerShell version of this probe (tools/probe-swscale-matrix.ps1)
// takes minutes; this takes under a second, which is what makes iterating on a
// candidate model practical.
//
// Build:  cl /O2 /EHsc probe_swscale_matrix.cpp
// Usage:  probe_swscale_matrix.exe <ffmpeg.exe> <workdir>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace {

// Limited-range BT.601: Y in [16,235], U/V in [16,240].
constexpr int kY0 = 16, kY1 = 235, kNY = kY1 - kY0 + 1;   // 220
constexpr int kV0 = 16, kV1 = 240, kNV = kV1 - kV0 + 1;   // 225

std::string g_ffmpeg;
std::string g_work;

// Resolves a bare tool name through PATH by hand.  std::system goes via cmd.exe,
// and a quoted bare name is not reliably looked up there, so the probe resolves
// the executable itself rather than depending on the caller's shell.
std::string ResolveTool(const std::string& name) {
  if (name.find('\\') != std::string::npos || name.find('/') != std::string::npos)
    return name;
  const char* path = std::getenv("PATH");
  if (path == nullptr) return name;
  const std::string p(path);
  std::size_t start = 0;
  while (start <= p.size()) {
    std::size_t end = p.find(';', start);
    if (end == std::string::npos) end = p.size();
    std::string dir = p.substr(start, end - start);
    if (!dir.empty()) {
      if (dir.back() != '\\') dir += '\\';
      const std::string cand = dir + name + ".exe";
      FILE* f = std::fopen(cand.c_str(), "rb");
      if (f != nullptr) { std::fclose(f); return cand; }
    }
    start = end + 1;
  }
  return name;
}

std::wstring Widen(const std::string& s) {
  if (s.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(static_cast<std::size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), &w[0], n);
  return w;
}

// Runs `exe` with `args`, sending both stdout and stderr to `log_path`.
//
// CreateProcess rather than std::system: the UCRT wraps a command in quotes and
// hands it to cmd.exe, which then mis-parses a command that itself begins with a
// quoted absolute path.  The identical string works from a shell and fails with
// ERROR_FILE_NOT_FOUND from system().  Passing the executable as lpApplicationName
// sidesteps cmd entirely, which is also what rdither's own Child class does.
int RunProcess(const std::string& exe, const std::string& args,
               const std::string& log_path) {
  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = nullptr;
  sa.bInheritHandle = TRUE;
  HANDLE log = CreateFileW(Widen(log_path).c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                           &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (log == INVALID_HANDLE_VALUE) return -1000;
  HANDLE nul = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ, &sa, OPEN_EXISTING,
                           0, nullptr);
  STARTUPINFOW si;
  ZeroMemory(&si, sizeof(si));
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = nul;
  si.hStdOutput = log;
  si.hStdError = log;
  PROCESS_INFORMATION pi;
  ZeroMemory(&pi, sizeof(pi));
  // lpApplicationName carries the executable, so only the arguments go on the
  // command line proper and no quoting of the path is needed.
  std::wstring cmdline = L"\"" + Widen(exe) + L"\" " + Widen(args);
  if (!CreateProcessW(Widen(exe).c_str(), &cmdline[0], nullptr, nullptr, TRUE,
                      CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
    CloseHandle(log);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    return -1001;
  }
  WaitForSingleObject(pi.hProcess, 180000);
  DWORD code = 0;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  CloseHandle(log);
  if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
  return static_cast<int>(code);
}

bool ReadFile(const std::string& path, std::vector<std::uint8_t>* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  out->resize(static_cast<std::size_t>(n));
  const std::size_t got = n > 0 ? std::fread(out->data(), 1, out->size(), f) : 0;
  std::fclose(f);
  return got == out->size();
}

bool WriteFile(const std::string& path, const std::vector<std::uint8_t>& b) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) return false;
  const std::size_t put = std::fwrite(b.data(), 1, b.size(), f);
  std::fclose(f);
  return put == b.size();
}

// Runs one planar yuv444p frame through ffmpeg, returns the rgba64le bytes.
bool Convert(const std::vector<std::uint8_t>& yuv, int w, int h, const char* tag,
             std::vector<std::uint8_t>* out) {
  const std::string in = g_work + "\\in_" + tag + ".yuv";
  const std::string dst = g_work + "\\out_" + tag + ".rgba";
  if (!WriteFile(in, yuv)) return false;
  char cmd[2048];
  const std::string errlog = g_work + "\\err_" + tag + ".txt";
  std::snprintf(cmd, sizeof(cmd),
                "-v error -f rawvideo -pix_fmt yuv444p -s %dx%d -i \"%s\" "
                "-f rawvideo -pix_fmt rgba64le -y \"%s\"",
                w, h, in.c_str(), dst.c_str());
  const int rc = RunProcess(g_ffmpeg, cmd, errlog);
  if (rc != 0) {
    // Never let an ffmpeg failure be silent.  A swallowed "Unrecognized option"
    // is exactly how the -vsync removal hid for a whole round: the probe would
    // otherwise go on to "discover" a formula from an empty input.
    std::vector<std::uint8_t> e;
    std::fprintf(stderr, "ffmpeg failed on %s (rc=%d); command was:\n  %s %s\n", tag, rc,
                 g_ffmpeg.c_str(), cmd);
    if (ReadFile(errlog, &e) && !e.empty()) {
      std::fprintf(stderr, "ffmpeg said:\n  %.*s\n", static_cast<int>(e.size()),
                   reinterpret_cast<const char*>(e.data()));
    } else {
      std::fprintf(stderr, "ffmpeg wrote nothing to its error log\n");
    }
    return false;
  }
  return ReadFile(dst, out);
}

inline std::uint16_t U16(const std::vector<std::uint8_t>& v, std::size_t byte_off) {
  return static_cast<std::uint16_t>(v[byte_off] | (v[byte_off + 1] << 8));
}

int Gcd(int a, int b) {
  while (b != 0) { const int t = a % b; a = b; b = t; }
  return a < 0 ? -a : a;
}

// The candidate conversion, transcribed from libswscale and evaluated with the
// *same* integer types the original uses, because the wrapping is load-bearing.
//
//   yuv2rgb.c:786-791   the six coefficients, for contrast = saturation = 65536
//                       and brightness = 0 (ffmpeg's defaults), limited range:
//                         cy  = ((1<<16) * 255) / 219   = 76309
//                         oy  = 16 << 16
//                         crv, cbu, cgu, cgv = the BT.601 inv_table entries
//                       then  coeff = (int16_t)roundToInt16(value * (1 << 13))
//   output.c:1385-1428  the arithmetic itself
//
// The 1:1 (yuv444p) filter collapses to one tap, so the 15-bit intermediate is
// (byte - 128) * 512; the -0x40000000 / -(128<<23) seeds and the >>14 are the
// filter's fixed-point bias and are reproduced rather than simplified away.
struct Coeffs {
  int y_offset;   // roundToInt16(oy  * (1 <<  9))
  int y_coeff;    // roundToInt16(cy  * (1 << 13))
  int v2r;        // roundToInt16(crv * (1 << 13))
  int v2g;        // roundToInt16(cgv * (1 << 13))
  int u2g;        // roundToInt16(cgu * (1 << 13))
  int u2b;        // roundToInt16(cbu * (1 << 13))
};

inline int RoundToInt16(std::int64_t f) {
  const std::int64_t r = (f + (1 << 15)) >> 16;
  if (r < -0x7FFF) return static_cast<std::int16_t>(0x8000);
  if (r >  0x7FFF) return static_cast<std::int16_t>(0x7FFF);
  return static_cast<int>(static_cast<std::int16_t>(r));
}

const Coeffs& SwsCoeffs() {
  static const Coeffs k = [] {
    Coeffs c;
    const std::int64_t cy  = ((1LL << 16) * 255) / 219;   // 76309
    const std::int64_t oy  = 16LL << 16;                 // 1048576
    const std::int64_t crv = 104597, cbu = 132201, cgu = -25675, cgv = -53279;
    c.y_coeff = RoundToInt16(cy  * (1 << 13));
    c.y_offset = RoundToInt16(oy * (1 <<  9));
    c.v2r = RoundToInt16(crv * (1 << 13));
    c.v2g = RoundToInt16(cgv * (1 << 13));
    c.u2g = RoundToInt16(cgu * (1 << 13));
    c.u2b = RoundToInt16(cbu * (1 << 13));
    return c;
  }();
  return k;
}

inline std::uint16_t Clip16(std::int64_t v) {
  if (v < 0) return 0;
  if (v > 65535) return 65535;
  return static_cast<std::uint16_t>(v);
}

// Exactly output.c:1385-1428, with the unsigned wraparound made explicit: the
// original writes `(unsigned)V * coeff` and `(int)(R + (unsigned)Y)`, both of
// which are mod-2^32 before the arithmetic shift.
void Model(int yin, int uin, int vin, int* r, int* g, int* b) {
  const Coeffs& c = SwsCoeffs();
  int Y = static_cast<int>((yin - 128) * 512);
  const int U = (uin - 128) * 512;
  const int V = (vin - 128) * 512;

  Y += 0x10000;
  Y -= c.y_offset;
  Y *= c.y_coeff;
  Y += (1 << 13) - (1 << 29);

  const std::uint32_t Ri = static_cast<std::uint32_t>(V) * static_cast<std::uint32_t>(c.v2r);
  const std::uint32_t Gi = static_cast<std::uint32_t>(V) * static_cast<std::uint32_t>(c.v2g) +
                           static_cast<std::uint32_t>(U) * static_cast<std::uint32_t>(c.u2g);
  const std::uint32_t Bi = static_cast<std::uint32_t>(U) * static_cast<std::uint32_t>(c.u2b);
  const std::uint32_t Yu = static_cast<std::uint32_t>(Y);

  *r = Clip16((static_cast<std::int32_t>(Ri + Yu) >> 14) + (1 << 15));
  *g = Clip16((static_cast<std::int32_t>(Gi + Yu) >> 14) + (1 << 15));
  *b = Clip16((static_cast<std::int32_t>(Bi + Yu) >> 14) + (1 << 15));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 5 && std::string(argv[1]) == "-cmp") {
    // Which frames differ between two raw dumps.  Needed because every single
    // frame compared equal while the whole clip did not, and "the first frame
    // matches" is not an explanation.
    const char* pa = argv[2];
    const char* pb = argv[3];
    const long fsz = std::atol(argv[4]);
    FILE* fa = std::fopen(pa, "rb");
    FILE* fb = std::fopen(pb, "rb");
    if (fa == nullptr || fb == nullptr) { std::printf("cannot open\n"); return 1; }
    std::vector<std::uint8_t> ba(static_cast<std::size_t>(fsz)), bb(static_cast<std::size_t>(fsz));
    long idx = 0, diffFrames = 0, firstDiff = -1;
    std::vector<long> shown;
    for (;;) {
      const std::size_t ra = std::fread(ba.data(), 1, ba.size(), fa);
      const std::size_t rb = std::fread(bb.data(), 1, bb.size(), fb);
      if (ra != ba.size() || rb != bb.size()) break;
      if (ra != rb || std::memcmp(ba.data(), bb.data(), ba.size()) != 0) {
        ++diffFrames;
        if (firstDiff < 0) firstDiff = idx;
        if (shown.size() < 12) shown.push_back(idx);
      }
      ++idx;
    }
    std::printf("\nframe-by-frame comparison: %ld frames, %ld differ\n", idx, diffFrames);
    if (firstDiff >= 0) {
      std::printf("    first differing frame: %ld\n", firstDiff);
      std::printf("    first few           :");
      for (long s : shown) std::printf(" %ld", s);
      std::printf("\n");
      // How much does one differing frame actually differ?
      std::fseek(fa, firstDiff * fsz, SEEK_SET);
      std::fseek(fb, firstDiff * fsz, SEEK_SET);
      if (std::fread(ba.data(), 1, ba.size(), fa) == ba.size() &&
          std::fread(bb.data(), 1, bb.size(), fb) == bb.size()) {
        long comp = 0, worst = 0;
        double se = 0.0;
        for (std::size_t i = 0; i < ba.size() / 2; ++i) {
          const int a = ba[2 * i] | (ba[2 * i + 1] << 8);
          const int b = bb[2 * i] | (bb[2 * i + 1] << 8);
          if (a != b) ++comp;
          const long t = a > b ? a - b : b - a;
          if (t > worst) worst = t;
          se += static_cast<double>(t) * t;
        }
        const double n = ba.size() / 2.0;
        std::printf("    in that frame        : %ld / %.0f components differ, worst %ld\n",
                    comp, n, worst);
        std::printf("    PSNR                 : %.2f dB\n",
                    se > 0 ? 10.0 * std::log10(65535.0 * 65535.0 * n / se) : 99.99);
      }
    } else {
      std::printf("    every frame is identical\n");
    }
    std::fclose(fa);
    std::fclose(fb);
    std::printf("\n");
    return 0;
  }

  if (argc >= 4 && std::string(argv[1]) == "-video") {
    // What the change actually costs: compare ffmpeg's fused yuv420p -> rgba64le
    // against "ffmpeg upsample the chroma to 8-bit 4:4:4, then apply the exact
    // matrix".  The second is what the device kernel does.
    const std::string ffmpeg = ResolveTool(argv[2]);
    const std::string clip = argv[3];
    g_ffmpeg = ffmpeg;
    g_work = g_work.empty() ? std::string(".") : g_work;
    g_work = clip.substr(0, clip.find_last_of("/\\"));
    const int W = 1920, H = 1080;

    auto run = [&](const std::string& args, const std::string& out) {
      return RunProcess(g_ffmpeg, args, g_work + "\\vid_err.txt") == 0 &&
             [&] { std::vector<std::uint8_t> t; return ReadFile(out, &t); }();
    };
    std::vector<std::uint8_t> rgba, y444;
    const std::string a1 = "-v error -i \"" + clip + "\" -frames:v 1 -f rawvideo -pix_fmt rgba64le -y \"" + g_work + "\\v_rgba.raw\"";
    const std::string a2 = "-v error -i \"" + clip + "\" -frames:v 1 -f rawvideo -pix_fmt yuv444p -y \"" + g_work + "\\v_444.raw\"";
    if (RunProcess(g_ffmpeg, a1, g_work + "\\vid_err.txt") != 0 ||
        RunProcess(g_ffmpeg, a2, g_work + "\\vid_err.txt") != 0 ||
        !ReadFile(g_work + "\\v_rgba.raw", &rgba) ||
        !ReadFile(g_work + "\\v_444.raw", &y444)) {
      std::printf("ffmpeg failed; see vid_err.txt\n");
      return 1;
    }
    const std::size_t px = static_cast<std::size_t>(W) * H;
    if (rgba.size() != px * 8 || y444.size() != px * 3) {
      std::printf("unexpected sizes: rgba=%zu y444=%zu (want %zu / %zu)\n",
                  rgba.size(), y444.size(), px * 8, px * 3);
      return 1;
    }
    long diff = 0, worst = 0;
    double se = 0.0;
    for (std::size_t i = 0; i < px; ++i) {
      int mr, mg, mb;
      Model(y444[i], y444[px + i], y444[2 * px + i], &mr, &mg, &mb);
      const int d[3] = {mr - U16(rgba, i * 8), mg - U16(rgba, i * 8 + 2),
                        mb - U16(rgba, i * 8 + 4)};
      for (int k = 0; k < 3; ++k) {
        if (d[k] != 0) ++diff;
        const long a = d[k] < 0 ? -d[k] : d[k];
        if (a > worst) worst = a;
        se += static_cast<double>(d[k]) * d[k];
      }
    }
    std::printf("\nExact matrix vs ffmpeg's fused conversion, one %dx%d frame of %s\n",
                W, H, clip.c_str());
    std::printf("    differing components : %ld / %zu (%.4f%%)\n", diff, px * 3,
                100.0 * diff / (px * 3.0));
    std::printf("    largest difference   : %ld\n", worst);
    std::printf("    PSNR                 : %.2f dB\n",
                se > 0 ? 10.0 * std::log10(65535.0 * 65535.0 * px * 3.0 / se) : 99.99);

    // Is the 8-bit 4:4:4 intermediate itself faithful?  Ask ffmpeg to convert the
    // yuv444p we just produced, and compare that against the direct conversion.
    // If *this* is already far off, the plan of letting ffmpeg do the chroma
    // filter and doing only the matrix on the device cannot work at any
    // precision, and the FIR has to be ported rather than approximated.
    const std::string a3 = "-v error -f rawvideo -pix_fmt yuv444p -s " +
                           std::to_string(W) + "x" + std::to_string(H) + " -i \"" +
                           g_work + "\\v_444.raw\" -f rawvideo -pix_fmt rgba64le -y \"" +
                           g_work + "\\v_444rgba.raw\"";
    std::vector<std::uint8_t> via444;
    if (RunProcess(g_ffmpeg, a3, g_work + "\\vid_err.txt") != 0 ||
        !ReadFile(g_work + "\\v_444rgba.raw", &via444) || via444.size() != px * 8) {
      std::printf("    (could not convert the 4:4:4 intermediate back)\n");
    } else {
      long d2 = 0, w2 = 0;
      double se2 = 0.0;
      for (std::size_t i = 0; i < px * 3; ++i) {
        const int a = U16(rgba, i * 2), b = U16(via444, i * 2);
        if (a != b) ++d2;
        const long t = a > b ? a - b : b - a;
        if (t > w2) w2 = t;
        se2 += static_cast<double>(t) * t;
      }
      std::printf("\nffmpeg's own 4:4:4 round trip, same frame\n");
      std::printf("    differing components : %ld / %zu (%.4f%%)\n", d2, px * 3,
                  100.0 * d2 / (px * 3.0));
      std::printf("    largest difference   : %ld\n", w2);
      std::printf("    PSNR                 : %.2f dB\n",
                  se2 > 0 ? 10.0 * std::log10(65535.0 * 65535.0 * px * 3.0 / se2) : 99.99);
    }
    // A few samples, because a number this large is structural and needs looking
    // at rather than averaging.  Y/U/V from the intermediate against the RGB both
    // routes produced.
    std::printf("\n  sample   Y     U     V  | direct R G B          | via 4:4:4 R G B\n");
    for (int k = 0; k < 8; ++k) {
      // Scattered, not strided: a fixed stride lands on the same offset within
      // each strip and the frame has large flat regions, so every sample came
      // back identical and looked like a bug in the conversion.
      const std::size_t i = (static_cast<std::size_t>(k) * 2654435761ull) % px;
      std::printf("  %6zu %4d %5d %5d  | %5u %5u %5u          | %5u %5u %5u\n", i,
                  y444[i], y444[px + i], y444[2 * px + i], U16(rgba, i * 8),
                  U16(rgba, i * 8 + 2), U16(rgba, i * 8 + 4), U16(via444, i * 8),
                  U16(via444, i * 8 + 2), U16(via444, i * 8 + 4));
    }
    std::printf("\n");
    return 0;
  }

  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <ffmpeg.exe> <workdir>\n", argv[0]);
    return 2;
  }
  g_ffmpeg = ResolveTool(argv[1]);
  g_work = argv[2];

  // 256x256 covers 220*225 = 49500 (Y,V) pairs.
  constexpr int kW = 256, kH = 256, kPS = kW * kH;
  std::vector<std::uint8_t> grid(static_cast<std::size_t>(kPS) * 3);
  std::vector<std::uint8_t> grid_u(static_cast<std::size_t>(kPS) * 3);
  for (int row = 0; row < kH; ++row) {
    const int s = kV0 + (row % kNV);
    for (int col = 0; col < kW; ++col) {
      const int y = kY0 + (col % kNY);
      const int i = row * kW + col;
      grid[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(y);
      grid[static_cast<std::size_t>(kPS + i)] = 128;
      grid[static_cast<std::size_t>(2 * kPS + i)] = static_cast<std::uint8_t>(s);
      grid_u[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(y);
      grid_u[static_cast<std::size_t>(kPS + i)] = static_cast<std::uint8_t>(s);
      grid_u[static_cast<std::size_t>(2 * kPS + i)] = 128;
    }
  }

  std::vector<std::uint8_t> ov, ou;
  if (!Convert(grid, kW, kH, "grid", &ov)) return 1;
  if (!Convert(grid_u, kW, kH, "grid_u", &ou)) return 1;
  if (ov.size() != static_cast<std::size_t>(kPS) * 8 ||
      ou.size() != static_cast<std::size_t>(kPS) * 8) {
    std::fprintf(stderr, "unexpected output size\n");
    return 1;
  }

  // R,G indexed by (y,v); B indexed by (y,u).
  std::vector<std::uint16_t> R(static_cast<std::size_t>(kNY) * kNV);
  std::vector<std::uint16_t> G(static_cast<std::size_t>(kNY) * kNV);
  std::vector<std::uint16_t> B(static_cast<std::size_t>(kNY) * kNV);
  for (int row = 0; row < kH; ++row) {
    const int vi = row % kNV;
    for (int col = 0; col < kW; ++col) {
      const int yi = col % kNY;
      const std::size_t i = static_cast<std::size_t>(row) * kW + col;
      R[static_cast<std::size_t>(yi) * kNV + vi] = U16(ov, i * 8);
      G[static_cast<std::size_t>(yi) * kNV + vi] = U16(ov, i * 8 + 2);
    }
  }
  const int uMid = 128 - kV0, vMid = 128 - kV0;
  for (int row = 0; row < kH; ++row) {
    const int ui = row % kNV;
    for (int col = 0; col < kW; ++col) {
      B[static_cast<std::size_t>(col % kNY) * kNV + ui] =
          U16(ou, (static_cast<std::size_t>(row) * kW + col) * 8 + 4);
    }
  }

  std::printf("\nQ1  separability: is R(Y,V) - R(Y,Vmid) independent of Y?\n");
  int sepR = 0, sepG = 0, sepB = 0, tot = 0;
  for (int yi = 0; yi < kNY; ++yi) {
    for (int vi = 0; vi < kNV; ++vi) {
      ++tot;
      if (R[static_cast<std::size_t>(yi) * kNV + vi] - R[static_cast<std::size_t>(yi) * kNV + vMid] ==
          R[static_cast<std::size_t>(vi)] - R[static_cast<std::size_t>(vMid)]) ++sepR;
      if (G[static_cast<std::size_t>(yi) * kNV + vi] - G[static_cast<std::size_t>(yi) * kNV + vMid] ==
          G[static_cast<std::size_t>(vi)] - G[static_cast<std::size_t>(vMid)]) ++sepG;
      if (B[static_cast<std::size_t>(yi) * kNV + vi] - B[static_cast<std::size_t>(yi) * kNV + uMid] ==
          B[static_cast<std::size_t>(vi)] - B[static_cast<std::size_t>(uMid)]) ++sepB;
    }
  }
  std::printf("    R separable : %6d / %6d\n", sepR, tot);
  std::printf("    G separable : %6d / %6d\n", sepG, tot);
  std::printf("    B separable : %6d / %6d\n", sepB, tot);

  std::printf("\nQ2  quantisation\n");
  int gAll = 0, gDiffY = 0, gDiffV = 0;
  for (int yi = 0; yi < kNY; ++yi) {
    for (int vi = 0; vi < kNV; ++vi) {
      gAll = Gcd(gAll, R[static_cast<std::size_t>(yi) * kNV + vi]);
      if (yi > 0)
        gDiffY = Gcd(gDiffY, R[static_cast<std::size_t>(yi) * kNV + vi] -
                                  R[static_cast<std::size_t>(yi - 1) * kNV + vi]);
      if (vi > 0)
        gDiffV = Gcd(gDiffV, R[static_cast<std::size_t>(yi) * kNV + vi] -
                                  R[static_cast<std::size_t>(yi) * kNV + vi - 1]);
    }
  }
  std::printf("    gcd(all R)          = %d\n", gAll);
  std::printf("    gcd(dR/dY adjacent) = %d\n", gDiffY);
  std::printf("    gcd(dR/dV adjacent) = %d\n", gDiffV);

  std::printf("\nQ3  cross-checks\n");
  int dup = 0, dupTot = 0;
  for (int yi = 0; yi < kNY; ++yi) {
    ++dupTot;
    const std::size_t i = static_cast<std::size_t>(uMid) * kW + yi;
    if (R[static_cast<std::size_t>(yi) * kNV + vMid] == U16(ou, i * 8)) ++dup;
  }
  std::printf("    R identical at U=128 across both grids : %d / %d\n", dup, dupTot);

  std::printf("\nQ4  the separable tables, so the port can be checked by eye\n");
  std::printf("    g(V) = R(Y0,V) - R(Y0,128), V=%d..:\n    ", kV0);
  for (int vi = 0; vi < 14; ++vi)
    std::printf("%7d", R[static_cast<std::size_t>(vi)] - R[static_cast<std::size_t>(vMid)]);
  std::printf("\n    f(Y) = R(Y,128) - R(Y0,128), Y=%d..:\n    ", kY0);
  for (int yi = 0; yi < 14; ++yi)
    std::printf("%7d", R[static_cast<std::size_t>(yi) * kNV + vMid] - R[static_cast<std::size_t>(vMid)]);
  std::printf("\n    h(U) = B(Y0,U) - B(Y0,128), U=%d..:\n    ", kV0);
  for (int ui = 0; ui < 14; ++ui)
    std::printf("%7d", B[static_cast<std::size_t>(ui)] - B[static_cast<std::size_t>(uMid)]);
  std::printf("\n    clip: R at Y=16,V=16 is %u, at Y=235,V=240 is %u\n",
              R[0], R[static_cast<std::size_t>(kNY - 1) * kNV + kNV - 1]);

  // ---------------------------------------------------------------------------
  // Q5  The candidate model, ported from swscale and checked over the *whole*
  //      grid rather than a few hand-picked points.
  //
  //      Derived by reading the source, not fitted:
  //        yuv2rgb.c  ff_yuv2rgb_c_init_tables   builds the six coefficients
  //        output.c   yuv2rgba64_full_X_c_template  consumes them
  //
  //      With contrast = saturation = 65536 and brightness = 0 (ffmpeg's
  //      defaults) the coefficients reduce to constants, and the unscaled
  //      1:1 filter makes the 15-bit intermediate (byte - 128) * 512.
  // ---------------------------------------------------------------------------
  std::printf("\nQ5  the model from swscale's source, over the full grid\n");
  const Coeffs& c = SwsCoeffs();
  std::printf("    coefficients: y_coeff=%d y_offset=%d v2r=%d v2g=%d u2g=%d u2b=%d\n",
              c.y_coeff, c.y_offset, c.v2r, c.v2g, c.u2g, c.u2b);
  int badR = 0, badG = 0, badB = 0, n = 0;
  int firstBad = 0;
  for (int yi = 0; yi < kNY; ++yi) {
    for (int vi = 0; vi < kNV; ++vi) {
      const int y = kY0 + yi, v = kV0 + vi;
      int mr = 0, mg = 0, mb = 0;
      Model(y, 128, v, &mr, &mg, &mb);
      const int gr = R[static_cast<std::size_t>(yi) * kNV + vi];
      const int gg = G[static_cast<std::size_t>(yi) * kNV + vi];
      ++n;
      if (mr != gr) { ++badR; if (!firstBad) firstBad = n; }
      if (mg != gg) ++badG;
      if (mb != B[static_cast<std::size_t>(yi) * kNV + (128 - kV0)]) ++badB;
    }
  }
  std::printf("    R mismatches : %d / %d\n", badR, n);
  std::printf("    G mismatches : %d / %d\n", badG, n);
  std::printf("    B mismatches : %d / %d\n", badB, n);

  // Out-of-range samples: Y/U/V below 16 or above 235 are legal input bytes even
  // though they are outside the nominal range, and they are where a clipping
  // mistake would hide.
  int oob = 0, oobBad = 0;
  for (int y = 0; y < 256; ++y) {
    for (int c = 0; c < 256; c += 7) {
      int mr, mg, mb;
      Model(y, c, (c * 3) & 255, &mr, &mg, &mb);
      ++oob;
      if (mr < 0 || mr > 65535 || mg < 0 || mg > 65535 || mb < 0 || mb > 65535) ++oobBad;
    }
  }
  std::printf("    out-of-range samples in [0,65535] : %d / %d\n", oob - oobBad, oob);

  // A real frame, if one was supplied.  This is the test that matters: the grid
  // above is synthetic and may miss whatever the filter does at the edges.
  if (argc >= 6) {
    const int rw = std::atoi(argv[3]);
    const int rh = std::atoi(argv[4]);
    std::vector<std::uint8_t> src;
    if (!ReadFile(argv[5], &src) || src.size() != static_cast<std::size_t>(rw) * rh * 3) {
      std::printf("    (could not read the supplied yuv444p frame)\n");
    } else {
      std::vector<std::uint8_t> got;
      if (!Convert(src, rw, rh, "real", &got)) {
        std::printf("    (ffmpeg failed on the supplied frame)\n");
      } else {
        const std::size_t px = static_cast<std::size_t>(rw) * rh;
        long bad = 0;
        for (std::size_t i = 0; i < px; ++i) {
          int mr, mg, mb;
          Model(src[i], src[px + i], src[2 * px + i], &mr, &mg, &mb);
          if (mr != U16(got, i * 8) || mg != U16(got, i * 8 + 2) || mb != U16(got, i * 8 + 4))
            ++bad;
        }
        std::printf("    real frame %dx%d: %ld / %zu pixels differ\n", rw, rh, bad, px);
      }
    }
  }

  if (badR == 0 && badG == 0 && badB == 0) {
    std::printf("    -> EXACT on every sample of the (Y,V) grid.\n");
  } else {
    std::printf("    -> NOT exact; first R mismatch at sample %d\n", firstBad);
  }
  std::printf("\n");
  return 0;
}
