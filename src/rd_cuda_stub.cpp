// SPDX-License-Identifier: GPL-3.0-or-later
// rd_cuda_stub.cpp -- the four CUDA entry points, for a build without CUDA.
//
// `build.cmd --no-cuda` is a documented configuration ("CPU only.  Same program,
// much faster to compile") and it did not compile at all until this file existed.
// Two separate faults, and the second only became visible once the first was fixed:
//
//   1. src/rd_video.cpp included <cuda_runtime.h> unconditionally, with no #ifdef
//      anywhere in the file, so the header was required even though rd_video.cpp
//      calls no CUDA API directly.  -> error C1083.
//
//   2. All four definitions below live in .cu files, and CMakeLists.txt only adds
//      those `if(RD_WITH_CUDA)`.  So the symbols were declared in rd_riemersma.h,
//      called from rd_video.cpp and rd_cli.cpp, and defined nowhere -- which is
//      LNK2019, the same shape of failure as the missing OpenCLStateCount().
//
// This file defines SEVEN of them, and the number matters: it is more than the
// compile error named and more than the first link error named, because the linker
// reports them one at a time.  Fixing only what it complained about would have shipped
// as a build that still did not link.  Enumerated from include/rd_riemersma.h instead --
// every function declared there, checked for a definition in a .cu, and checked for a
// call from a file that is always compiled.  That found CudaMaxFrames, which is
// reached only under --max-vram-mb and so is invisible until then, and it also found
// BuildCurveIndex, which looks CUDA-only and is not (it is defined in
// rd_riemersma_cpu.cpp, so stubbing it would have been a duplicate symbol).
//
// Four are questions about the device -- CudaAvailable, CudaMaxFrames, CudaAllocPinned,
// CudaFreePinned -- and three are the engine itself: RiemersmaWalkCuda,
// RiemersmaApproxCuda and RiemersmaBlocksCuda.
//
// These have to EXIST rather than be optional, because the call sites are not
// conditional: rd_video.cpp asks CudaAvailable() to decide whether to ask ffmpeg for
// NVDEC, and asks CudaAllocPinned() for staging, in every build.  The engine
// selection is conditional; the questions are not.
//
// Returning false and nullptr hides nothing.  The pinned-memory path is already
// written to work without pinned memory -- Pipeline::in() returns
// in16_pin ? in16_pin : in16.data(), and the *_pinned block options are set only
// when the allocation succeeded (rd_video.cpp, the in()/out() accessors and the
// option assignment) -- so a nullptr here takes exactly the path a failed
// allocation takes on a machine that does have CUDA.  And CudaAvailable() returning
// false is what makes `--engine blocks` refuse with "no CUDA device" rather than
// pretending to have an accelerator.
#ifndef RD_WITH_CUDA

#include "rd_riemersma.h"

namespace rd {

bool CudaAvailable() { return false; }

// 0, not 1: "no frames fit in a device that is not there".  Unreachable in practice,
// because rd_cli.cpp refuses a CUDA engine before it consults this -- but 0 is the
// honest answer, and the caller already handles it by erroring when fits < 1.
int CudaMaxFrames(std::size_t, std::size_t, std::size_t) { return 0; }

void* CudaAllocPinned(std::size_t) { return nullptr; }
void CudaFreePinned(void*) {}

// The three engine entry points.  THESE MUST NOT RETURN AN EMPTY STRING.
//
// All three use the success-is-empty convention: an empty std::string means the dither
// ran and `pixels` now holds the result, anything else is a diagnostic the caller
// prints.  A stub that returned "" would therefore report SUCCESS while having
// written nothing at all, and the caller would go on to write the undithered input to
// the output file.  That is a silent wrong answer, which is the one failure mode this
// project treats as worst.  So each returns a message naming the missing engine.
//
// The call sites are unconditional in rd_cli.cpp -- the engine is chosen by an if/else
// at the call, not by whether the symbol exists -- so these have to link whether or not
// the build has CUDA.  A caller reaching one of these has already been told the engine
// is unavailable by CudaAvailable() returning false, so in a correct run they are
// unreachable; they exist so that a build without CUDA links and so that a mistake
// surfaces as this message rather than as an image nobody dithered.
std::string RiemersmaWalkCuda(const Palette&, const DitherParams&, const ColorTree&,
                              std::size_t, std::size_t, RgbaF*, int,
                              std::string* device_name) {
  if (device_name != nullptr) *device_name = std::string();
  return "cuda: this build has no CUDA engine (configure with -DRD_WITH_CUDA=ON)";
}

std::string RiemersmaApproxCuda(const Palette&, const DitherParams&, const ColorTree&,
                                std::size_t, std::size_t, RgbaF*, const ApproxOptions&,
                                std::string* device_name) {
  if (device_name != nullptr) *device_name = std::string();
  return "approx: this build has no CUDA engine (configure with -DRD_WITH_CUDA=ON)";
}

std::string RiemersmaBlocksCuda(const Palette&, const DitherParams&, const ColorTree&,
                                std::size_t, std::size_t, RgbaF*, const BlockOptions&,
                                std::string* device_name, std::uint16_t*,
                                int, const std::uint16_t*) {
  if (device_name != nullptr) *device_name = std::string();
  return "blocks: this build has no CUDA engine (configure with -DRD_WITH_CUDA=ON)";
}

}  // namespace rd

#endif  // !RD_WITH_CUDA
