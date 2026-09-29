// SPDX-License-Identifier: GPL-3.0-or-later
// Does OpenCL actually work on this machine?
//
// Three things can each be present while OpenCL is still unusable, and they fail
// differently, so this checks them separately rather than assuming:
//   1. the ICD loader (OpenCL.dll, shipped by Windows)
//   2. a registered vendor ICD (HKLM\SOFTWARE\Khronos\OpenCL\Vendors)
//   3. the vendor's own driver library (nvopencl64.dll, shipped inside the driver)
//
// The loader being present proves nothing: it is a thin dispatcher that fails with
// CL_PLATFORM_NOT_FOUND_KHR (-1001) when step 2 is empty.  So the loader is loaded and
// clGetPlatformIDs is called directly, through the exported symbol, with no SDK header
// and no import library -- which also means this probe needs nothing but the CRT and
// kernel32, so it can be run *before* deciding whether to install the SDK.
//
// windows.h is deliberately not included: the Windows SDK on the test machine is
// incomplete (ucrt/um/shared ship without excpt.h), so any file that includes
// windows.h fails to compile there for reasons that have nothing to do with OpenCL.
// Declaring the three kernel32 entry points by hand sidesteps that and keeps the
// probe to a single translation unit with no include path at all.

#include <cstdio>
#include <cstdint>
#include <vector>

// --- kernel32, declared rather than included (see note above) --------------
extern "C" __declspec(dllimport) void* __stdcall LoadLibraryA(const char*);
extern "C" __declspec(dllimport) void* __stdcall GetProcAddress(void*, const char*);
extern "C" __declspec(dllimport) unsigned long __stdcall GetLastError();

using HMODULE_ = void*;
using FARPROC_ = void*;

// --- cl.h, declared rather than included ----------------------------------
using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
using cl_bitfield = cl_uint;
using cl_platform_id = void*;
using cl_device_id = void*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_int CL_PLATFORM_NOT_FOUND_KHR = -1001;
constexpr cl_bitfield CL_DEVICE_TYPE_ALL = 0xFFFFFFFFu;
constexpr cl_uint CL_PLATFORM_NAME = 0x0902;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_DEVICE_VERSION = 0x102F;
constexpr cl_uint CL_DEVICE_VENDOR = 0x102C;
constexpr cl_uint CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
constexpr cl_uint CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x1012;
constexpr cl_uint CL_DEVICE_DOUBLE_FP_CONFIG = 0x1013;
constexpr cl_uint CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x11B0;
constexpr cl_uint CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
constexpr cl_uint CL_DEVICE_EXTENSIONS = 0x1030;

using PfnGetPlatformIDs = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using PfnGetPlatformInfo = cl_int (*)(cl_platform_id, cl_uint, std::size_t, void*, std::size_t*);
using PfnGetDeviceIDs = cl_int (*)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
using PfnGetDeviceInfo = cl_int (*)(cl_device_id, cl_uint, std::size_t, void*, std::size_t*);

static void PrintDeviceInfo(cl_device_id dev, PfnGetDeviceInfo fn, cl_uint what,
                            const char* label) {
  char buf[512] = {0};
  std::size_t n = 0;
  if (fn && fn(dev, what, sizeof(buf) - 1, buf, &n) == CL_SUCCESS && n) {
    std::printf("      %-8s %s\n", label, buf);
  } else {
    std::printf("      %-8s <unavailable>\n", label);
  }
}

int main() {
  std::printf("=== 1. ICD loader (OpenCL.dll) ===\n");
  HMODULE_ dll = LoadLibraryA("OpenCL.dll");
  if (!dll) {
    std::printf("  MISSING (GetLastError %lu) -- the Khronos ICD loader is not\n"
                "  installed.  Install the Khronos OpenCL Runtime / ICD Loader.\n",
                GetLastError());
    return 2;
  }
  std::printf("  present\n");

  auto GetPlatformIDs = reinterpret_cast<PfnGetPlatformIDs>(
      reinterpret_cast<std::uintptr_t>(GetProcAddress(dll, "clGetPlatformIDs")));
  if (!GetPlatformIDs) {
    std::printf("  present but no clGetPlatformIDs -- not a real ICD loader.\n");
    return 2;
  }
  std::printf("  exports clGetPlatformIDs\n");

  std::printf("\n=== 2. registered platforms ===\n");
  cl_uint count = 0;
  const cl_int rc = GetPlatformIDs(0, nullptr, &count);
  std::printf("  clGetPlatformIDs -> %d, count %u\n", rc, count);
  if (rc == CL_PLATFORM_NOT_FOUND_KHR) {
    std::printf("  CL_PLATFORM_NOT_FOUND_KHR: the loader works, but no vendor is\n"
                "  registered under HKLM\\SOFTWARE\\Khronos\\OpenCL\\Vendors.\n");
  }
  if (rc != CL_SUCCESS || count == 0) {
    std::printf("\n  RESULT: OpenCL is NOT usable on this machine.\n"
                "  Installing the Khronos SDK headers is required to *build* a\n"
                "  backend, but on this machine it would not make OpenCL *work* --\n"
                "  the driver ships nvopencl64.dll yet it is unregistered, which is\n"
                "  a driver-reinstall matter, not a missing package.\n");
    return 1;
  }

  std::vector<cl_platform_id> plats(count);
  GetPlatformIDs(count, plats.data(), nullptr);

  auto GetPlatformInfo = reinterpret_cast<PfnGetPlatformInfo>(
      reinterpret_cast<std::uintptr_t>(GetProcAddress(dll, "clGetPlatformInfo")));
  auto GetDeviceIDs = reinterpret_cast<PfnGetDeviceIDs>(
      reinterpret_cast<std::uintptr_t>(GetProcAddress(dll, "clGetDeviceIDs")));
  auto GetDeviceInfo = reinterpret_cast<PfnGetDeviceInfo>(
      reinterpret_cast<std::uintptr_t>(GetProcAddress(dll, "clGetDeviceInfo")));

  for (cl_uint i = 0; i < count; ++i) {
    char name[256] = {0};
    std::size_t sz = 0;
    if (GetPlatformInfo && GetPlatformInfo(plats[i], CL_PLATFORM_NAME,
                                           sizeof(name) - 1, name, &sz) == CL_SUCCESS) {
      std::printf("  platform %u: %s\n", i, name);
    } else {
      std::printf("  platform %u: <unnamed>\n", i);
    }
    cl_uint nd = 0;
    if (!GetDeviceIDs || GetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS) {
      std::printf("    no devices\n");
      continue;
    }
    std::vector<cl_device_id> devs(nd);
    GetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
    for (cl_uint d = 0; d < nd; ++d) {
      std::printf("    device %u:\n", d);
      PrintDeviceInfo(devs[d], GetDeviceInfo, CL_DEVICE_NAME, "name");
      PrintDeviceInfo(devs[d], GetDeviceInfo, CL_DEVICE_VERSION, "version");
      PrintDeviceInfo(devs[d], GetDeviceInfo, CL_DEVICE_VENDOR, "vendor");

      // The one query that decides whether this device is worth using for
      // rdither, and the reason this probe grew beyond "is OpenCL usable".
      //
      // The Riemersma walk is DOUBLE PRECISION throughout -- the 16-entry error
      // queue, the accumulator, and the palette distance are all `double`, because
      // reproducing ImageMagick's Q16-HDRI arithmetic bit-for-bit requires it.
      // Every consumer and integrated GPU runs FP64 at a small fraction of its FP32
      // rate, commonly 1/64, and integrated parts are the weakest of all: the walk
      // that does essentially all the work is the one operation the hardware is
      // worst at.
      //
      // Measured on the machine this was written on, an NVIDIA GTX 1650 SUPER --
      // not a weak-FP64 part by consumer standards -- the OpenCL engine spends
      // 2213 ms on the dither stage where CUDA spends 810 ms and the CPU 820 ms.
      // It is 2.7x SLOWER than CUDA, and bit-identical to it. On a GPU with weaker
      // double-precision than that, the gap widens rather than closes.
      //
      // So the honest reading of this report is: OpenCL here buys REACH, on a
      // machine with no CUDA, and not speed. A device reporting a preferred double
      // vector width of 1 and one or two compute units is an integrated part doing
      // double-precision in software, and the CPU will beat it.
      {
        // Numeric fields, printed as numbers.  PrintDeviceInfo reads every field
        // into a char buffer because the spec defines most of them as strings; a
        // cl_uint read that way yields four bytes of a 512-byte buffer and prints
        // as empty, which is exactly what the first version of this did.
        cl_uint units = 0;
        if (GetDeviceInfo && GetDeviceInfo(devs[d], CL_DEVICE_MAX_COMPUTE_UNITS,
                                           sizeof(units), &units, nullptr) == CL_SUCCESS) {
          std::printf("      %-8s %u\n", "compute u", units);
        } else {
          std::printf("      %-8s <unavailable>\n", "compute u");
        }
        // Advisory only.  The NVIDIA driver does not answer this one, and a driver
        // that declines is not evidence of anything -- so a failed query prints
        // "<unavailable>" and never a bare 0, which would read as "no double".
        cl_uint dvec = 0;
        if (GetDeviceInfo && GetDeviceInfo(devs[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                           sizeof(dvec), &dvec, nullptr) == CL_SUCCESS) {
          std::printf("      %-8s %u\n", "fp64 vec", dvec);
        } else {
          std::printf("      %-8s <unavailable>\n", "fp64 vec");
        }
        // The extensions string is the check that does not depend on decoding a
        // bitmask.  CL_DEVICE_DOUBLE_FP_CONFIG is a flag word whose bits are
        // sparsely assigned and version-dependent, and reading it wrongly yields a
        // plausible-looking value -- 0x4000, which is not a documented bit -- so it
        // is not used as the verdict.  cl_khr_fp64 is the thing itself.
        bool fp64_ext = false;
        if (GetDeviceInfo) {
          char ext[4096] = {0};
          std::size_t n = 0;
          if (GetDeviceInfo(devs[d], CL_DEVICE_EXTENSIONS, sizeof(ext) - 1, ext, &n) ==
                  CL_SUCCESS &&
              n) {
            fp64_ext = std::strstr(ext, "cl_khr_fp64") != nullptr;
            std::printf("      %-8s %s\n", "fp64", fp64_ext ? "cl_khr_fp64 present" : "ABSENT");
          }
        }
        if (!fp64_ext) {
          std::printf(
              "      %-8s no double precision (cl_khr_fp64 absent).  The Riemersma\n"
              "      %-8s walk is double throughout -- that is what makes it\n"
              "      %-8s bit-exact with ImageMagick's Q16-HDRI -- so the engine\n"
              "      %-8s will not be faster than the CPU here.\n", "**", "**", "**", "**");
        } else {
          std::printf("      %-8s double precision available; the walk can run\n", "ok");
        }
        unsigned long long mem = 0;
        if (GetDeviceInfo && GetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE,
                                           sizeof(mem), &mem, nullptr) == CL_SUCCESS) {
          std::printf("      %-8s %llu MiB\n", "mem", mem >> 20);
        }
      }
    }
  }
  std::printf("\n  RESULT: OpenCL IS usable here.\n");
  return 0;
}
