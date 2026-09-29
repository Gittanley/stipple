// SPDX-License-Identifier: GPL-3.0-or-later
// rd_opencl.h -- an OpenCL engine for the block walk.
//
// WHY THIS EXISTS.  The CUDA engine is the fast path on NVIDIA hardware.  On an
// AMD or Intel GPU there is no CUDA, and until this file the only GPU-shaped
// option was `--engine blocks` falling back to `RiemersmaBlocksCpu`, i.e. the
// host pool.  The point of the port is *reach*, not speed: a machine with a
// perfectly good GPU and no CUDA SDK should not be told it has no GPU engine.
//
// WHAT "SAME ENGINE" MEANS HERE.  This is deliberately the *blocks* engine, not
// the sequential walk, and that choice is what makes the port verifiable.
// rd_riemersma.h states that `RiemersmaBlocksCpu` has "the same partition and
// the same arithmetic as RiemersmaBlocksCuda, on the host", so there is a
// host reference that is known to agree with the GPU.  Bit-exactness is then
// checked against the CPU blocks engine, which needs no second GPU and no
// CUDA -- a comparison this project can actually run.  The sequential-walk
// engine's bit-exactness rests on ImageMagick itself as the reference, which
// is a slower and weaker test (see docs/OPENCL.md for why that matters).
//
// VERIFIED SCOPE.  Bit-identical to RiemersmaBlocksCuda on every cell of
// tools\probe-opencl-exact.ps1 -- 54 cells across six images, three palette
// sizes and three block lengths -- including the two that carry an alpha
// channel.  Alpha is worth naming explicitly: for a while this engine refused
// alpha input over a nondeterminism that turned out to be unfilled channels in
// the image writer, not anything here.  The guard is gone and those cells are
// compared for real.  docs/OPENCL.md has the account, including the measurement
// mistake that kept the real fault hidden for so long.
//
// Video is bit-identical too: 60 frames of 1920x1080 through the whole pipeline,
// lossless, both engines, same palette and same 16-frame batches.  Only the
// rgba64le data path is ported, so `--engine opencl --video` needs
// `--input-mode rgba64` and `RD_YUV444_OUT=0`; the CLI checks that at parse
// time and this engine refuses the YUV modes as a backstop.
//
// ARITHMETIC.  CUDA spells its correctly-rounded double operations
// __dadd_rn / __dmul_rn so that nvcc cannot contract a*b+c into an FMA.  OpenCL
// C's default rounding for double is also correctly rounded and the default
// build does not contract, so plain + and * are the equivalents -- but only
// while `-cl-fast-relaxed-math` and `-cl-mad-enable` stay off.  BuildOpenClProgram
// below passes neither, and says so in the build log; that is the whole reason
// the log line exists.
#ifndef RD_OPENCL_H_
#define RD_OPENCL_H_

#include <cstdint>
#include <string>

#include "rd_riemersma.h"
#include "rd_types.h"

namespace rd {

// Is there a usable OpenCL device?  Fills `device_name` with a human-readable
// "vendor device (OpenCL n.m)" when it succeeds.  Safe to call when OpenCL was
// compiled out; it then returns false and leaves `device_name` alone.
//
// Never throws and never aborts: OpenCL's loader is a DLL that may be absent,
// and its failure modes include a driver that reports a device it cannot then
// initialise.  A missing engine is a feature that is off, not a crash.
bool OpenCLAvailable(std::string* device_name, std::string* detail);

// How many independent device states this engine keeps -- its equivalent of the
// CUDA engine's state slots.  Each is a separate command queue with its own
// kernel objects, so two callers using different slots run concurrently instead
// of serialising behind one in-order queue.  The single-image path only ever
// uses slot 0, so it never pays for the second program build.
//
// A caller passing a slot >= this count is clamped to the last one: slow, but
// correct.  Scheduling is not this function's problem to reject.
int OpenCLStateCount();

// The build log of the most recent successful program build, or empty.  Aimed at
// a developer staring at a driver that accepted the source and then produced
// different pixels: without the log, "clBuildProgram failed" and "clBuildProgram
// succeeded and got the arithmetic wrong" look identical from the outside.
const std::string& OpenCLBuildLog();

// Dithers `batch` (options.frames consecutive RGBA-float frames) in place using
// the same block partition as RiemersmaBlocksCpu, on an OpenCL device.  Returns
// an empty string on success and a diagnostic otherwise -- the same
// success-is-empty convention RiemersmaBlocksCuda uses, so the two can be
// swapped behind one call site.
//
// The uint16 arguments mirror RiemersmaBlocksCuda exactly, and are honoured
// only when the matching BlockOptions flag is ALSO set.  Setting the flag
// without the pointer falls back to the float path rather than dereferencing
// null, which is the safer of two behaviours that are both wrong at the call
// site.
//
// NOT yet ported, and refused with a message rather than silently substituted:
// the planar-YUV input modes (options.in_mode other than Interleaved16) and
// planar 4:4:4 output (options.emit_yuv444).  Each is a bit-exact
// transliteration of a specific libswscale routine on the CUDA side, and
// reading rgba64le instead would yield a plausible picture from a different
// decoder.
//
// `state_slot` selects an independent device state, matching the CUDA engine's
// arrangement: two GPU workers each take one so their launches do not serialise.
// Reusing a slot from two threads is safe but slower, and a slot past
// OpenCLStateCount() is clamped to the last.
std::string RiemersmaBlocksOpencl(const Palette& palette,
                                  const DitherParams& params,
                                  const ColorTree& tree, std::size_t width,
                                  std::size_t height, RgbaF* batch,
                                  const BlockOptions& options,
                                  std::string* device_name,
                                  std::uint16_t* out_u16 = nullptr,
                                  int state_slot = 0,
                                  const std::uint16_t* in_u16 = nullptr);

}  // namespace rd

#endif  // RD_OPENCL_H_
