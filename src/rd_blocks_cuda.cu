// SPDX-License-Identifier: GPL-3.0-or-later
// rd_blocks_cuda.cu -- single-pass block-parallel Riemersma, for when the goal
// is the *look* rather than bit-exactness.
//
// ---------------------------------------------------------------------------
// Why this exists
// ---------------------------------------------------------------------------
// quantize.c's error queue has exactly ErrorQueueLength == 16 entries, and each
// visit consumes the front of it and appends at the back:
//
//     for (i = 0; i < 15; i++) error[i] = error[i+1];
//     error[15] = v - colormap[index];
//
// So a walk is fully described by its 16-entry state.  Once a block has run 16
// steps, its output no longer depends on the incoming state *at all*: the state
// it started from has been shifted out.  A block of B >= 16 positions is
// therefore bit-identical to the sequential walk from position 16 onward, and
// only its first 16 positions can be affected by a wrong boundary state.
//
// That is a much stronger statement than the error-decay bound, and it means the
// seams are a *bounded, localised* defect rather than a global drift.  Starting
// every block from a zeroed queue therefore perturbs at most 16 positions per
// block, and how much they move is bounded by the loop gain:
//
//     |dv| <= (G/(1-G)) * Delta/2 = 0.541 * Delta/2 ~ 0.27 Delta
//
// i.e. under a third of a quantisation step.  Most of those pixels will not even
// change palette index, and where they do the change is a neighbouring palette
// entry -- visually a slightly different speckle, not banding.
//
// With B = 512 and N = 480000 that is 937 independent blocks and < 0.7% of
// positions even *eligible* to change.  At 1 thread per block the device is far
// from saturated, so B can be raised until the seams are negligible: this is the
// one axis where the GPU genuinely wins, because the work is embarrassingly
// parallel at block granularity and needs no iteration, no scan and no prefix.
//
// ---------------------------------------------------------------------------
// Deliberate differences from ImageMagick
// ---------------------------------------------------------------------------
// * No memo table.  quantize.c memoises the palette lookup behind a 6-bit
//   key, which is a visit-order effect; blocks have no single order, so the
//   lookup is performed exactly every time.  This is the *more* accurate
//   nearest-colour rule (IM quantises the search key to 6 bits per channel).
// * Blocks start from a zeroed queue, not the state left by the previous block.
//   --blocks-rounds N optionally propagates real boundary states for N rounds.
// ---------------------------------------------------------------------------
#include "rd_riemersma.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "rd_progress.h"

#include "rd_cuda_common.cuh"
#include "rd_octree.h"

namespace rd {
namespace {

using cuda_common::DevNode;
using cuda_common::DevSearch;
using cuda_common::Rgba;

// One thread == one (frame, block) pair: a block of `block_size` consecutive
// curve positions within one frame.  The 16-entry error queue lives in a
// per-thread slice of global scratch: at 16*4 doubles = 512 B it is exactly the
// sm_75 local-memory ceiling, and local arrays there overflow the stack, whereas
// a private global slice is L2-resident and effectively free.
//
// The chosen palette indices are written out and scattered by a second pass, so
// the image buffer is written exactly once per pixel in a deterministic order.

// Occupancy target for the walk.
//
// Measured with Nsight Compute (which needs an elevated shell on this machine):
// 255 registers per thread -- the architectural ceiling -- and 19.5% achieved
// occupancy.  The 64-double register queue is 128 of those registers on its own, so
// the kernel is register-starved, spilling, and has only ~8 warps per SM to hide the
// 16-deep serial error-diffusion chain behind.  That is the 11,800 cycles per pixel.
//
// Capping registers with __launch_bounds__ to buy occupancy was tried and is much
// worse, because the queue does not survive the spill.  Swept, walk ms per launch:
//   unconstrained 163.8 | bounds(128,1) 172.0 | (128,2) 167.2 | (128,3) 302.9
//   (128,4) 367.8   | (128,6) 475.7
// At 3 blocks the cap falls below the 128 registers the queue needs, it spills to
// local memory, and the 64x traffic returns.  So the register queue is at a genuine
// local optimum: it needs 128 of the 255 available and cannot be shrunk, because
// the only smaller version (dropping the 16 unused alpha entries, 64 -> 48 doubles)
// measured 54% worse -- a stride of 3 rather than 4 stops nvcc proving the array
// register-resident, so it spills anyway.
//
// The walk is therefore not improvable by occupancy tuning, and the remaining 19.5%
// occupancy is a consequence of the algorithm's state, not an oversight.
__global__ void BlkIndexWalkKernel(const float* __restrict__ cx,
                                   const double* __restrict__ palette, int pcount,
                                   const double* __restrict__ weights,
                                   double diffusion, int assoc,
                                   const DevNode* __restrict__ nodes, int n,
                                   int nblocks, int frame_pitch, int block_size,
                                    unsigned char* __restrict__ out_index) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= nblocks * frame_pitch) return;

  // Which frame this work item belongs to, and which block within it.  Each
  // frame is an independent walk, so each keeps its own error state.
  const int frame = slot / nblocks;
  const int local = slot - frame * nblocks;
  const float* frame_cx = cx + static_cast<std::size_t>(frame) * n * 4;

  const int begin = local * block_size;
  const int end = (begin + block_size < n) ? (begin + block_size) : n;

  // The error queue lives in REGISTERS, not global memory.
  //
  // It is 16 entries x 4 channels = 64 doubles, and every position shifts all 64.
  // With the queue in global memory each thread touched 32 * 512 B * 2 = 32 KB per
  // block of B=32 to process 512 B of pixel data -- 64x more traffic than useful
  // work, and that, not bandwidth, was the bottleneck (measured: ~2.7 GB of
  // device traffic per launch at an effective 2.6 GB/s, far below any modern GPU).
  // As a local array of constant size with constant indices it stays in registers,
  // which also frees 531 MB of VRAM at batch 16.
  //
  // The arithmetic is untouched: same __dadd_rn / __dmul_rn, same order, so the
  // output is bit-identical.
  double e[4 * kErrorQueueLength];
#pragma unroll
  for (int k = 0; k < kErrorQueueLength; ++k) {
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) e[4 * k + ch] = 0.0;
  }

  // The error weights are loop-invariant: they depend only on the position in the
  // queue and on `diffusion`, both fixed for the whole launch.  They used to be
  // recomputed inside the pixel loop, twice per entry -- 32 double multiplies for
  // every pixel, to produce 16 values that never change.  On a GeForce card FP64 runs
  // at 1/64 rate, so those were among the most expensive instructions in the kernel,
  // and hoisting them is bit-exact: same inputs, same operations, same results.
  //
  // Tried and reverted: holding these in shared memory instead.  It removes a real
  // spill (cuobjdump reports the stack frame dropping 160 -> 88 bytes, so the
  // compiler was already evicting this array to local memory) but it does NOT free
  // registers -- the kernel stays at 255 -- because 128 of those belong to the
  // 64-double error queue, which is IM's ErrorQueueLength and cannot shrink without
  // breaking bit-exactness.  Occupancy is therefore stuck at 8 warps per SM no matter
  // what is done to the other 127.  Measured neutral (L605 189.0 -> 182.6 ms, this
  // clip's content 287.1 -> 290.9 ms), so it is not worth the __syncthreads().
  double w[kErrorQueueLength];
#pragma unroll
  for (int k = 0; k < kErrorQueueLength; ++k) {
    w[k] = __dmul_rn(__dmul_rn(cuda_common::kErb, diffusion), weights[k]);
  }

  for (int i = begin; i < end; ++i) {
    const int c = i * 4;
    double pr = frame_cx[c + 0], pg = frame_cx[c + 1];
    double pb = frame_cx[c + 2], pa = frame_cx[c + 3];
    if (assoc && pa != cuda_common::kQRange) {
      const double alpha = cuda_common::kQScale * pa;
      pr = alpha * pr; pg = alpha * pg; pb = alpha * pb;
    }
#pragma unroll
    for (int k = 0; k < kErrorQueueLength; ++k) {
      pr = __dadd_rn(pr, __dmul_rn(w[k], e[4 * k + 0]));
      pg = __dadd_rn(pg, __dmul_rn(w[k], e[4 * k + 1]));
      pb = __dadd_rn(pb, __dmul_rn(w[k], e[4 * k + 2]));
      if (assoc) pa = __dadd_rn(pa, __dmul_rn(w[k], e[4 * k + 3]));
    }
    pr = static_cast<double>(cuda_common::d_clamp_pixel(pr));
    pg = static_cast<double>(cuda_common::d_clamp_pixel(pg));
    pb = static_cast<double>(cuda_common::d_clamp_pixel(pb));
    if (assoc) pa = static_cast<double>(cuda_common::d_clamp_pixel(pa));

    Rgba target;
    cuda_common::d_associate_from_double(assoc, pr, pg, pb, pa, &target);
    const int index = cuda_common::d_select_index(nodes, palette, assoc, nullptr,
                                                  target);
    out_index[static_cast<std::size_t>(frame) * n + i] =
        static_cast<unsigned char>(index);

    const double* p = palette + 4 * index;
    // Shifting 64 doubles by one slot, in registers.  Unrolled so every index is
    // a compile-time constant and the array is provably register-resident.
#pragma unroll
    for (int k = 0; k < kErrorQueueLength - 1; ++k) {
#pragma unroll
      for (int ch = 0; ch < 4; ++ch) e[4 * k + ch] = e[4 * (k + 1) + ch];
    }
    double cr = p[0], cg = p[1], cb = p[2], ca = p[3];
    if (assoc && ca != cuda_common::kQRange) {
      const double alpha = cuda_common::kQScale * ca;
      cr = alpha * cr; cg = alpha * cg; cb = alpha * cb;
    }
    const int last = 4 * (kErrorQueueLength - 1);
    e[last + 0] = __dadd_rn(pr, -cr);
    e[last + 1] = __dadd_rn(pg, -cg);
    e[last + 2] = __dadd_rn(pb, -cb);
    if (assoc) e[last + 3] = __dadd_rn(pa, -ca);
  }
}

// Writes the quantised colour as rgba64le when `u16` is non-null, and always as
// float4 otherwise.  The uint16 path exists so the video pipe gets exactly the
// bytes it is about to hand to ffmpeg: four uint16 per pixel against float4's
// sixteen, so it halves the download and removes the host-side conversion
// entirely.
//
// The two casts must match FloatsToRaw() on the host exactly -- double, then
// float, then a C-style truncating conversion to uint16 -- because the encoder
// consumes these bytes directly.  They are spelled out rather than folded so the
// sequence is visible at the point where it is load-bearing.
__global__ void BlkScatterKernel(float4* __restrict__ pixels,
                                const int* __restrict__ curve,
                                const int* __restrict__ owner,
                                const unsigned char* __restrict__ index,
                                const double* __restrict__ palette, int n,
                                int frame_pitch, int pixel_pitch, int assoc,
                                std::uint16_t* __restrict__ u16) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= n * frame_pitch) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const int pixel = curve[i];
  // Skip positions that a later visit supersedes; without this the scatter's
  // last-writer would be whichever thread happened to run last.
  if (owner[pixel] != i) return;
  const double* p = palette + 4 * static_cast<int>(index[slot]);
  const float fr = static_cast<float>(p[0]);
  const float fg = static_cast<float>(p[1]);
  const float fb = static_cast<float>(p[2]);
  const std::size_t off =
      static_cast<std::size_t>(frame) * pixel_pitch + pixel;
  if (pixels != nullptr) {
    float4 out;
    out.x = fr;
    out.y = fg;
    out.z = fb;
    out.w = assoc ? static_cast<float>(p[3]) : static_cast<float>(65535.0);
    pixels[off] = out;
  }
  if (u16 != nullptr) {
    // Four uint16 per pixel, which is exactly rgba64le.  Note the alpha: the
    // float path leaves it at 65535 unless alpha is associated, and the encoder
    // consumes these bytes directly, so it has to be written rather than assumed.
    u16[off * 4 + 0] = static_cast<std::uint16_t>(fr);
    u16[off * 4 + 1] = static_cast<std::uint16_t>(fg);
    u16[off * 4 + 2] = static_cast<std::uint16_t>(fb);
    u16[off * 4 + 3] =
        static_cast<std::uint16_t>(assoc ? static_cast<float>(p[3])
                                         : static_cast<float>(65535.0));
  }
}


// The same scatter, writing planar 8-bit 4:4:4 instead of rgba64le.
//
// This exists to take the RGB->YUV conversion off the CPU.  The encoder is fed raw
// frames and converts them itself, which for 18001 frames of 1920x1080 is 298 GB of
// rgba64le read and converted on the host -- measured as the encode stage becoming
// 94% of the wall, with the writer blocked at 597 MB/s, *below* the pipe's own
// 0.86 GB/s.  So the limit was never the pipe; shrinking the pipe without moving the
// conversion would just move the same CPU work onto our writer thread and win
// nothing.  Both halves have to move together, and they do: 3 bytes per pixel against 8.
//
// The fidelity cost is zero, and that is worth being explicit about.  The round trip
// already happens today -- ffmpeg converts our dithered RGB to yuv444p before writing
// the file, so the palette entries in the file are already RGB->YUV->RGB and already
// not exactly the entries we chose.  This only decides *who* rounds, so the difference
// is a last-unit disagreement, not a change of pipeline.
__device__ __forceinline__ void d_rgb_to_yuv444(unsigned r, unsigned g, unsigned b,
                                                unsigned char* y, unsigned char* u,
                                                unsigned char* v) {
  const int r8 = static_cast<int>(r >> 8);
  const int g8 = static_cast<int>(g >> 8);
  const int b8 = static_cast<int>(b >> 8);
  const int yv = ((66 * r8 + 129 * g8 + 25 * b8 + 128) >> 8) + 16;
  const int uv = ((-38 * r8 - 74 * g8 + 112 * b8 + 128) >> 8) + 128;
  const int vv = ((112 * r8 - 94 * g8 - 18 * b8 + 128) >> 8) + 128;
  *y = static_cast<unsigned char>(yv < 0 ? 0 : (yv > 255 ? 255 : yv));
  *u = static_cast<unsigned char>(uv < 0 ? 0 : (uv > 255 ? 255 : uv));
  *v = static_cast<unsigned char>(vv < 0 ? 0 : (vv > 255 ? 255 : vv));
}

__global__ void BlkScatterYuv444Kernel(const int* __restrict__ curve,
                                       const int* __restrict__ owner,
                                       const unsigned char* __restrict__ index,
                                       const double* __restrict__ palette, int n,
                                       int frame_pitch, int pixel_pitch, int assoc,
                                       unsigned char* __restrict__ yuv) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= n * frame_pitch) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const int pixel = curve[i];
  if (owner[pixel] != i) return;
  const double* p = palette + 4 * static_cast<int>(index[slot]);
  // Three planes per frame, so frame f's luma starts at f * 3 * pixel_pitch -- the
  // same stride rule the 4:4:4 gather got wrong once.
  const std::size_t base =
      static_cast<std::size_t>(frame) * 3 * static_cast<std::size_t>(pixel_pitch) +
      static_cast<std::size_t>(pixel);
  d_rgb_to_yuv444(static_cast<unsigned>(p[0]), static_cast<unsigned>(p[1]),
                  static_cast<unsigned>(p[2]), yuv + base,
                  yuv + base + static_cast<std::size_t>(pixel_pitch),
                  yuv + base + 2 * static_cast<std::size_t>(pixel_pitch));
}

// One thread per frame: write the single pixel the walk never visits, taking it from
// the source rather than from the palette.  This is the whole of the uint16 fix, and
// it is cheap precisely because the unvisited set is at most one pixel per frame.
//
// Why the source is still intact at that pixel: the scatter only writes where
// `owner[pixel] == i`, and this pixel has no owner, so neither BlkScatterKernel nor
// BlkScatterYuv444Kernel has touched it.  `src16` (the rgba64le upload) and `srcf`
// (the float4 upload) still hold the original frame at that offset, so the correct
// value is simply still there.  ImageMagick leaves that same pixel at its source
// value -- the recursion genuinely does not visit it -- so this makes the uint16 and
// yuv444 paths agree with ImageMagick and with the float path, rather than merely
// agreeing with each other.
//
// Without it the pixel is whatever cudaMalloc returned: not wrong-looking, just
// unreproducible, because it is uninitialised device memory, and it would differ
// between runs, between drivers, and between CUDA and OpenCL.
__global__ void BlkFillUnvisitedKernel(const float4* __restrict__ srcf,
                                       const std::uint16_t* __restrict__ src16,
                                       const unsigned char* __restrict__ srcyuv,
                                       int pixel, int frames, int pixel_pitch,
                                       int assoc, bool as_yuv444,
                                       std::uint16_t* __restrict__ u16) {
  const int frame = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame >= frames) return;
  const std::size_t off =
      static_cast<std::size_t>(frame) * pixel_pitch + static_cast<std::size_t>(pixel);
  if (as_yuv444) {
    unsigned char* planes = reinterpret_cast<unsigned char*>(u16);
    const std::size_t base = static_cast<std::size_t>(frame) * 3 *
                                 static_cast<std::size_t>(pixel_pitch) +
                             static_cast<std::size_t>(pixel);
    if (srcyuv != nullptr) {
      // The production case: a planar 4:4:4 source into a planar 4:4:4 output.  The
      // layouts are identical -- Y, then U, then V, each pixel_pitch bytes -- so the
      // faithful value is the source's own three bytes, with no conversion at all.
      // Converting would be *less* accurate, not more: it would put the source
      // through YCbCr->RGB and back, and ImageMagick never dithers this pixel at all,
      // so it never round-trips it either.
      planes[base] = srcyuv[base];
      planes[base + pixel_pitch] = srcyuv[base + pixel_pitch];
      planes[base + 2 * static_cast<std::size_t>(pixel_pitch)] =
          srcyuv[base + 2 * static_cast<std::size_t>(pixel_pitch)];
    } else {
      // No usable source plane -- a 4:2:0 source, whose chroma is subsampled and so
      // has no single correct value for one pixel.  Write a defined neutral rather
      // than nothing: reproducible, and black is the honest answer for a pixel the
      // walk never looked at.
      planes[base] = 0;
      planes[base + pixel_pitch] = 128;
      planes[base + 2 * static_cast<std::size_t>(pixel_pitch)] = 128;
    }
    return;
  }
  if (src16 != nullptr) {
    const std::uint16_t* p = src16 + off * 4;
    u16[off * 4 + 0] = p[0];
    u16[off * 4 + 1] = p[1];
    u16[off * 4 + 2] = p[2];
    u16[off * 4 + 3] = p[3];
  } else if (srcf != nullptr) {
    const float4 f = srcf[off];
    u16[off * 4 + 0] = static_cast<std::uint16_t>(f.x);
    u16[off * 4 + 1] = static_cast<std::uint16_t>(f.y);
    u16[off * 4 + 2] = static_cast<std::uint16_t>(f.z);
    u16[off * 4 + 3] =
        static_cast<std::uint16_t>(assoc ? f.w : 65535.0f);
  } else {
    u16[off * 4 + 0] = 0; u16[off * 4 + 1] = 0; u16[off * 4 + 2] = 0;
    u16[off * 4 + 3] = 65535;
  }
}


template <typename T>
__global__ void BlkGatherKernel(const float4* __restrict__ pixels,
                                const int* __restrict__ curve, T* __restrict__ cx,
                                int n, int frame_pitch, int pixel_pitch) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= n * frame_pitch) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const float4 p =
      pixels[static_cast<std::size_t>(frame) * pixel_pitch + curve[i]];
  T* dst = cx + static_cast<std::size_t>(frame) * n * 4 + 4 * i;
  dst[0] = static_cast<T>(p.x);
  dst[1] = static_cast<T>(p.y);
  dst[2] = static_cast<T>(p.z);
  dst[3] = static_cast<T>(p.w);
}

// The same gather, reading rgba64le and widening on the device.
//
// This exists to delete the reader's uint16 -> float loop.  That loop was the
// reader's dominant cost and it is memory-bandwidth-bound: 605 1080p frames is two
// billion conversions, each reading 8 bytes and writing 16, so roughly 48 GB of host
// memory traffic.  Six threads bought only 0.8 fps on it, because the bus is the
// constraint, not the cores.  Uploading the uint16 bytes as they arrive moves that
// traffic off the host's critical path and halves the H2D as well.
//
// The widening is exact: every uint16 is representable in a float's 24-bit mantissa,
// and the host did nothing but `static_cast<float>`.  So the dither sees bit-identical
// inputs either way, which is what makes this safe to do at all.
template <typename T>
__global__ void BlkGatherU16Kernel(const std::uint16_t* __restrict__ pixels,
                                   const int* __restrict__ curve,
                                   T* __restrict__ cx, int n, int frame_pitch,
                                   int pixel_pitch, int channels) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= n * frame_pitch) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const std::uint16_t* p =
      pixels + (static_cast<std::size_t>(frame) * pixel_pitch + curve[i]) * channels;
  T* dst = cx + static_cast<std::size_t>(frame) * n * 4 + 4 * i;
  dst[0] = static_cast<T>(static_cast<float>(p[0]));
  dst[1] = static_cast<T>(static_cast<float>(p[1]));
  dst[2] = static_cast<T>(static_cast<float>(p[2]));
  // rgb48le carries no alpha.  Synthesise the same 65535 that rgba64le would have
  // delivered, so the dither sees identical input either way -- and it is the value
  // the scatter writes back, so the encoder's stream is unchanged.
  dst[3] = static_cast<T>(channels == 4 ? static_cast<float>(p[3])
                                         : static_cast<float>(65535.0));
}

// av_clip_uintp2(x, 16): the shared tail of every swscale RGB output.
__device__ __forceinline__ std::uint16_t d_sws_clip16(int x) {
  const int v = x + (1 << 15);
  if (v < 0) return 0;
  if (v > 65535) return 65535;
  return static_cast<std::uint16_t>(v);
}

// swscale's 8-bit YUV -> 16-bit RGB, bit-exact.
//
// PORTED, NOT DERIVED.  Two places in libswscale produce it:
//   yuv2rgb.c   ff_yuv2rgb_c_init_tables()   builds six coefficients
//   output.c    yuv2rgba64_full_X_c_template() consumes them
// and with a 1:1 plane ratio (yuv444p) the scaling filter collapses to a single
// tap, so what remains is pure per-pixel arithmetic.
//
// The six constants below are that function's output for limited range with
// contrast = saturation = 65536 and brightness = 0, which are ffmpeg's defaults
// and which rdither's decoder commands leave in place:
//
//   cy  = ((1 << 16) * 255) / 219 = 76309      crv, cgu, cgv, cbu = the BT.601
//   oy  = 16 << 16 = 1048576                   inv_table entries
//   coeff = (int16_t)roundToInt16(v * (1 << 13)), y_offset uses (1 << 9)
//
// This is NOT one of the four classic integer forms, which is why guessing failed
// in round fourteen (0/376 exact each).  The intermediate is 15-bit
// ((byte - 128) * 512) and there is a single >>14 at the end; the 8-bit forms
// cannot express either.
//
// Verified by tools/probe_swscale_matrix.cpp against ffmpeg's own rgba64le output:
// 0 mismatches over the whole 220x225 (Y,V) grid, and 0 over a real 1920x1080
// frame.  The constants are printed by that tool rather than transcribed here by
// hand -- two of the six were got wrong by hand and would have produced a
// near-miss that a casual eyeball would have accepted.
//
// The original accumulates through `unsigned` and shifts an `int`, so the
// arithmetic is mod 2^32.  Every intermediate here fits in int32 (worst case is
// bi + Y at 1,704,927,232 against a limit of 2,147,483,647), so the plain int32
// form below is bit-identical while being far easier to read.
__device__ __forceinline__ void d_sws_yuv_to_rgb16(
    unsigned y, unsigned u, unsigned v, std::uint16_t* out) {
  int yy = static_cast<int>((y - 128u) * 512u);
  const int uu = static_cast<int>((u - 128u) * 512u);
  const int vv = static_cast<int>((v - 128u) * 512u);
  yy += 0x10000;
  yy -= 8192;                              // yuv2rgb_y_offset
  yy *= 9539;                              // yuv2rgb_y_coeff
  yy += (1 << 13) - (1 << 29);

  const int ri = vv * 13075;               // yuv2rgb_v2r_coeff
  const int gi = vv * -6660 + uu * -3209;  // v2g, u2g
  const int bi = uu * 16525;               // yuv2rgb_u2b_coeff

  out[0] = d_sws_clip16((ri + yy) >> 14);
  out[1] = d_sws_clip16((gi + yy) >> 14);
  out[2] = d_sws_clip16((bi + yy) >> 14);
}

// The same gather, reading planar 8-bit 4:4:4 and converting YCbCr -> RGB on the
// device.
//
// The win is bytes, not arithmetic: 4:4:4 is 3 bytes per pixel where rgba64le is 8,
// so the pipe and the H2D both move 2.7x less.  ffmpeg's own timings for producing
// each format scale almost exactly with output size -- 9.35 GB in 5055 ms, 3.51 GB in
// 2200 ms -- and the reader is the pipeline's floor, so that is where the time was.
// The matrix is swscale's own, bit-exactly; see d_sws_yuv_to_rgb16 above.  Getting
// this wrong is a hue shift, not noise, and the earlier float approximation of it
// was measurably *not* the same picture -- see the note on the walk kernel below.
template <typename T>
__global__ void BlkGatherYuv444Kernel(const unsigned char* __restrict__ planes,
                                      const int* __restrict__ curve,
                                      T* __restrict__ cx, int n, int frame_pitch,
                                      int pixel_pitch) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= n * frame_pitch) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const int pix = curve[i];
  // Planar 4:4:4 is three planes per frame, so frame f's luma starts at
  // f * 3 * pixel_pitch, not f * pixel_pitch.  The latter reads frame f-1's U and
  // V planes as if they were frame f's luma, which is correct for frame 0 and
  // nonsense for every frame after it -- and it was invisible because frame 0 is
  // all anyone looked at.  It cost the 4:4:4 input mode its entire reputation:
  // garbage pixels make the octree search in the walk kernel take a different,
  // much deeper path, which is why the walk measured 2x slower (348 ms against
  // 167 ms) for what looked like a precision effect.
  const std::size_t base =
      static_cast<std::size_t>(frame) * 3 * static_cast<std::size_t>(pixel_pitch) +
      static_cast<std::size_t>(pix);
  std::uint16_t rgb[3];
  d_sws_yuv_to_rgb16(planes[base], planes[base + pixel_pitch],
                     planes[base + 2 * pixel_pitch], rgb);
  T* dst = cx + static_cast<std::size_t>(frame) * n * 4 + 4 * i;
  dst[0] = static_cast<T>(static_cast<float>(rgb[0]));
  dst[1] = static_cast<T>(static_cast<float>(rgb[1]));
  dst[2] = static_cast<T>(static_cast<float>(rgb[2]));
  dst[3] = static_cast<T>(static_cast<float>(65535.0));
}

// Planar 4:2:0 gather.  The decoder hands over the source format untouched, so swscale
// converts nothing and this reconstructs chroma itself.
//
// Chroma is half resolution in both axes, so each output sample sits at input
// (x/2, y/2).  Even coordinates are an exact copy; odd ones interpolate the four
// surrounding samples.  Bilinear rather than nearest because it tracks swscale's
// filter far more closely, and because the alternative -- a block replicate -- is
// visibly wrong at hard chroma edges for no saving: the filter is 3 loads and a handful
// of integer ops against a walk that costs 180 ms.
//
// Luma is used as-is, so it is bit-identical to every other path.  The 15-bit matrix
// downstream is swscale's own, unchanged.
__device__ __forceinline__ unsigned d_chroma_bilinear(const unsigned char* plane,
                                                     int stride, int cx, int cy,
                                                     int ox, int oy, int w, int h) {
  const int x0 = cx, y0 = cy;
  const int x1 = (cx + 1 < w) ? cx + 1 : cx;
  const int y1 = (cy + 1 < h) ? cy + 1 : cy;
  const int a = plane[static_cast<std::size_t>(y0) * stride + x0];
  const int b = plane[static_cast<std::size_t>(y0) * stride + x1];
  const int c = plane[static_cast<std::size_t>(y1) * stride + x0];
  const int d = plane[static_cast<std::size_t>(y1) * stride + x1];
  // (2-ox)(2-oy)a + ox(2-oy)b + (2-ox)oy c + ox*oy d, over 4.  Integer throughout so
  // the result is a byte, and +2 rounds rather than truncating.
  const int top = a * (2 - ox) + b * ox;
  const int bot = c * (2 - ox) + d * ox;
  return static_cast<unsigned>((top * (2 - oy) + bot * oy + 2) >> 2);
}

template <typename T>
__global__ void BlkGatherYuv420Kernel(const unsigned char* __restrict__ planes,
                                      const int* __restrict__ curve,
                                      T* __restrict__ cx, int n, int frame_pitch,
                                      int pixel_pitch, int w, int h) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= n * frame_pitch) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const int pix = curve[i];
  const int x = pix % w;
  const int y = pix / w;
  // Planar 4:2:0 within one frame: a full-size Y plane (pixel_pitch bytes), then Cb
  // and Cr at a quarter of it each.  Total 1.5 * pixel_pitch per frame, so frame f's
  // luma starts at f * 3 * pixel_pitch / 2 -- the same "planes per frame" stride rule
  // that the 4:4:4 gather got wrong once.
  //
  // The frame stride was right; the index WITHIN each plane was not.  This used to
  // compute one `base = frame_base + pix` and use it for all three planes, but `pix`
  // is a FULL-RESOLUTION pixel index and the chroma planes hold a quarter as many
  // samples.  So each chroma read was displaced by up to a whole luma plane, and for
  // the last rows of the last frame it walked past the end of the allocation --
  // measured at +2,073,598 bytes at 1920x1080 with --batch-frames 16.  The signature
  // was three mutually different outputs at --batch-frames 1, 4 and 16, which only an
  // allocation-relative read produces: a correct kernel is batch-invariant.
  //
  // Luma is indexed by `pix`; each chroma plane is indexed by its own coordinates,
  // which is exactly what d_chroma_bilinear already receives as (x >> 1, y >> 1).
  const std::size_t frame_base =
      static_cast<std::size_t>(frame) * 3 * static_cast<std::size_t>(pixel_pitch) / 2;
  const int w2 = w >> 1, h2 = h >> 1;
  const std::size_t y_size = static_cast<std::size_t>(pixel_pitch);
  const std::size_t c_size = y_size / 4;
  const std::size_t cb_plane = frame_base + y_size;
  const std::size_t cr_plane = cb_plane + c_size;
  const unsigned yy = planes[frame_base + static_cast<std::size_t>(pix)];
  const unsigned cb = d_chroma_bilinear(planes + cb_plane, w2, x >> 1, y >> 1,
                                        x & 1, y & 1, w2, h2);
  const unsigned cr = d_chroma_bilinear(planes + cr_plane, w2, x >> 1,
                                        y >> 1, x & 1, y & 1, w2, h2);
  std::uint16_t rgb[3];
  d_sws_yuv_to_rgb16(yy, cb, cr, rgb);
  T* dst = cx + static_cast<std::size_t>(frame) * n * 4 + 4 * i;
  dst[0] = static_cast<T>(static_cast<float>(rgb[0]));
  dst[1] = static_cast<T>(static_cast<float>(rgb[1]));
  dst[2] = static_cast<T>(static_cast<float>(rgb[2]));
  dst[3] = static_cast<T>(static_cast<float>(65535.0));
}
// Whole-frame YCbCr -> RGB, coalesced.
//
// The point is that this thread maps one-to-one onto pixels, so every warp reads 32
// consecutive bytes from each plane and writes 32 consecutive pixels.  That is the
// opposite of the curve gather's access pattern, and it is why splitting the work in
// two beats one fused kernel that has to do both at once.
//
// Writes 3 uint16 per pixel (8-bit RGB widened into the dither's 0..65535 range), so
// the intermediate is 6 bytes per pixel against the input's 3 -- still far less than
// the 8 the interleaved path carries, and it never crosses the bus at all, being
// device-local.
__global__ void BlkYuvToRgb16Kernel(const unsigned char* __restrict__ planes,
                                    std::uint16_t* __restrict__ rgb,
                                    int pixel_pitch, int frames) {
  // A 2-D grid rather than a flat one, because the plane stride has to be applied
  // per frame: see the note on the same bug in BlkGatherYuv444Kernel.
  const int within = blockIdx.x * blockDim.x + threadIdx.x;
  if (within >= pixel_pitch || blockIdx.y >= static_cast<unsigned>(frames)) return;
  const std::size_t base =
      static_cast<std::size_t>(blockIdx.y) * 3 * static_cast<std::size_t>(pixel_pitch) +
      static_cast<std::size_t>(within);
  d_sws_yuv_to_rgb16(planes[base], planes[base + pixel_pitch],
                     planes[base + 2 * pixel_pitch], rgb + (static_cast<std::size_t>(blockIdx.y) *
                                                            static_cast<std::size_t>(pixel_pitch) +
                                                            static_cast<std::size_t>(within)) * 3);
}

}  // namespace

// ---------------------------------------------------------------------------
// Persistent device state
// ---------------------------------------------------------------------------
//
// Every launch used to rebuild everything from scratch, and at batch 16 / 1080p
// that cost per launch ~60 ms of host work, 8 cudaMalloc + 8 cudaFree (one of them
// 531 MB), and five uploads.  Measured on a 605-frame clip that was ~20% of the
// dither stage and almost all of it waste: the curve, the palette, the tree and
// the error weights depend only on the geometry and the palette, both constant for
// a whole clip.  Only the pixel buffer changes between launches.
//
// So they are built once and kept.  The key covers geometry, batch size, block
// size and a content hash of the palette and tree, so a caller that reuses a
// palette object with new contents rebuilds instead of silently reusing stale
// device state.
//
// The pixel upload goes through pinned staging on a dedicated stream.  Pageable
// H2D transfers are staged through a small internal bounce buffer, which caps them
// at a few GB/s; pinned memory plus cudaMemcpyAsync removes that ceiling.
namespace {

std::uint64_t HashPaletteAndTree(const Palette& palette, const ColorTree& tree) {
  std::uint64_t h = 1469598103934665603ull;  // FNV-1a offset basis
  auto mix = [&h](std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      h ^= (v >> (i * 8)) & 0xffu;
      h *= 1099511628211ull;
    }
  };
  mix(static_cast<std::uint64_t>(palette.count));
  mix(static_cast<std::uint64_t>(palette.associate_alpha ? 1 : 0));
  for (int i = 0; i < palette.count; ++i) {
    // Quantised to 1/256 of the Q16 range: enough to notice a changed palette,
    // negligible beside the copy it guards.
    mix(static_cast<std::uint64_t>(palette.entries[i].r * 256.0 + 0.5));
    mix(static_cast<std::uint64_t>(palette.entries[i].g * 256.0 + 0.5));
    mix(static_cast<std::uint64_t>(palette.entries[i].b * 256.0 + 0.5));
  }
  mix(static_cast<std::uint64_t>(tree.node_count()));
  mix(static_cast<std::uint64_t>(tree.color_count()));
  return h;
}

// Per-kernel timing, gated on RD_KERNEL_TIMING=1.
//
// Added because both profilers are unavailable on this machine: ncu fails with
// ERR_NVGPUCTRPERM (the driver reserves performance counters for administrators) and
// nsys injects but records no CUDA data at all.  cudaEvent pairs around the three
// kernels answer the same question -- which kernel is expensive -- with no privileges
// and no replay.  Off unless asked for, since the event queries add a sync.
struct KernelTiming {
  cudaEvent_t beg[3];
  cudaEvent_t end[3];
  bool on = false;
  KernelTiming() {
    if (std::getenv("RD_KERNEL_TIMING") == nullptr) return;
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
      if (cudaEventCreate(&beg[i]) != cudaSuccess) ok = false;
      if (cudaEventCreate(&end[i]) != cudaSuccess) ok = false;
    }
    on = ok;
  }
  ~KernelTiming() {
    if (!on) return;
    for (int i = 0; i < 3; ++i) { cudaEventDestroy(beg[i]); cudaEventDestroy(end[i]); }
  }
};


// Streams for the per-batch buffers.
//
// This is 1, and the double-buffered version it replaced is worth recording as a
// measured negative result.  Overlapping batch N+1's upload with batch N's kernels
// needs two launches in flight at once, and the caller is *synchronous* -- it
// cudaStreamSynchronizes before returning -- so the next launch cannot even be
// issued until the previous download has landed.  With two buffers and one thread
// it measured 20668 ms of dither against 20334 ms for one buffer: no gain, at a
// cost of ~1 GB of VRAM and ~1 GB of pinned host memory.
//
// Real overlap needs two GPU worker threads with independent device states, so
// that one call is downloading while the other is computing.  That is the actual
// fix and it is not implemented; see the README.
constexpr int kStreams = 1;

struct DeviceState {
  bool valid = false;
  int level = -1;
  std::size_t width = 0, height = 0;
  int frames = 0, block = 0;
  std::uint64_t phash = 0;
  bool want_u16 = false;
  bool want_in_u16 = false;
  int in_channels = 4;
  int in_mode = 0;
  int n = 0, nblocks_total = 0;
  // Only the per-clip, per-palette buffers remain single: the curve, the owner
  // table, the palette, the tree and the error weights are the same for every
  // launch of a clip.  Everything that differs per batch is double buffered.
  std::size_t pixel_total = 0, curve_total = 0;
  int* d_curve = nullptr;
  int* d_owner = nullptr;
  double* d_palette = nullptr;
  double* d_weights = nullptr;
  DevNode* d_nodes = nullptr;
  // Flat candidate index; null when the palette is too large for the nibble packing,
  // in which case the recursive ClosestColor is used instead.
  DevSearch* d_search = nullptr;
  // One buffer set per stream, and kStreams is 1: overlapping a transfer with
  // compute needs two *calls* in flight, which is what the state pool above is
  // for, not two buffers on one thread.  The array form is kept because it makes
  // the per-stream buffers explicit and costs nothing when the count is 1.
  float4* d_pixels_buf[kStreams] = {nullptr};
  // Pixels the curve never reaches, and the first of them.  Part of the state
  // because it is a property of the geometry: every batch of this shape has the
  // same unvisited pixel, and the uint16/yuv444 output has to write it every time.
  std::size_t unvisited = 0;
  int unvisited_pixel = -1;
  // The note about it is per-process, not per-batch.  It fires from the setup path,
  // which runs once per geometry but is re-entered for every batch, and a
  // 18001-frame job would otherwise print the same line a thousand times.
  bool warned_unvisited = false;
  float* d_cx_buf[kStreams] = {nullptr};
  unsigned char* d_index_buf[kStreams] = {nullptr};
  std::uint16_t* d_u16_buf[kStreams] = {nullptr};
  float4* h_pinned_buf[kStreams] = {nullptr};
  // Pinned staging plus a device landing zone for the uint16 upload path, same
  // lifetime as the float one.  Only allocated when the caller asks for it, so the
  // single-image path is unaffected.  Together these are half the bytes of the
  // float4 pair beside them.
  std::uint16_t* h_pinned16_buf[kStreams] = {nullptr};
  std::uint16_t* d_in16_buf[kStreams] = {nullptr};
  // Planar 4:4:4 upload buffer: 3 bytes per pixel, so a third of the interleaved one.
  unsigned char* d_in_yuv_buf[kStreams] = {nullptr};
  // Device-local RGB intermediate for the two-kernel YCbCr path.  6 bytes per pixel.
  std::uint16_t* d_rgb16_buf[kStreams] = {nullptr};
  cudaStream_t stream_buf[kStreams] = {nullptr};
  int stream_turn = 0;

  void Release() {
    for (int i = 0; i < kStreams; ++i) {
      if (d_index_buf[i]) cudaFree(d_index_buf[i]);
      if (d_cx_buf[i]) cudaFree(d_cx_buf[i]);
      if (d_pixels_buf[i]) cudaFree(d_pixels_buf[i]);
      if (d_u16_buf[i]) cudaFree(d_u16_buf[i]);
      if (h_pinned_buf[i]) cudaFreeHost(h_pinned_buf[i]);
      if (h_pinned16_buf[i]) cudaFreeHost(h_pinned16_buf[i]);
      if (d_in16_buf[i]) cudaFree(d_in16_buf[i]);
      if (d_in_yuv_buf[i]) cudaFree(d_in_yuv_buf[i]);
      if (d_rgb16_buf[i]) cudaFree(d_rgb16_buf[i]);
      if (stream_buf[i]) cudaStreamDestroy(stream_buf[i]);
      d_index_buf[i] = nullptr;
      d_cx_buf[i] = nullptr;
      d_pixels_buf[i] = nullptr;
      d_u16_buf[i] = nullptr;
      h_pinned_buf[i] = nullptr;
      h_pinned16_buf[i] = nullptr;
      d_in16_buf[i] = nullptr;
      d_in_yuv_buf[i] = nullptr;
      d_rgb16_buf[i] = nullptr;
      stream_buf[i] = nullptr;
    }
    if (d_nodes) cudaFree(d_nodes);
    if (d_search) cudaFree(d_search);
    if (d_weights) cudaFree(d_weights);
    if (d_palette) cudaFree(d_palette);
    if (d_owner) cudaFree(d_owner);
    if (d_curve) cudaFree(d_curve);
    d_nodes = nullptr;
    d_search = nullptr;
    d_weights = nullptr;
    d_palette = nullptr;
    d_owner = nullptr;
    d_curve = nullptr;
    valid = false;
  }
};

// A pool of independent device states, one per GPU worker.
//
// The single-state design made every launch serialise behind one mutex, so the
// transfer chain of the next batch could not start until the previous download
// had landed -- which is exactly the overlap the round-two double-buffering
// experiment failed to get on its own.  One state per worker fixes that properly:
// two calls are then genuinely in flight, one downloading while the other
// computes.  Each slot carries its own mutex, so workers on different slots never
// contend; a slot is only ever touched by the worker it belongs to, and the mutex
// is there to make that true rather than to serialise them.
//
// The cost is memory: a full second copy of the per-batch buffers (about 1.9 GiB
// of VRAM plus 0.5 GiB pinned at 1080p and 16 frames per batch).  `--gpu-workers 1`
// falls back to the single-state behaviour, and halving --batch-frames halves the
// per-copy cost without giving up the overlap.
constexpr int kStateSlots = 2;

struct StateSlot {
  DeviceState st;
  std::mutex mu;
};

StateSlot& Slot(int i) {
  static StateSlot slots[kStateSlots];
  i %= kStateSlots;
  if (i < 0) i += kStateSlots;
  return slots[i];
}

}  // namespace

std::string RiemersmaBlocksCuda(const Palette& palette, const DitherParams& params,
                                const ColorTree& tree, std::size_t width,
                                std::size_t height, RgbaF* batch,
                                const BlockOptions& options,
                                std::string* device_name,
                                std::uint16_t* out_u16, int state_slot,
                                const std::uint16_t* in_u16) {
  if (!CudaAvailable()) return "no CUDA device available";
  if (options.block < kErrorQueueLength) {
    return "--blocks must be at least 16 (the error queue is 16 entries deep)";
  }
  if (options.frames < 1) return "--frames must be >= 1";

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return "cudaGetDevice failed";
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess && device_name) {
    *device_name = std::string(prop.name) + " (sm_" +
                    std::to_string(prop.major) + std::to_string(prop.minor) + ")";
  }

  const int level = ComputeCurveLevel(width, height);
  StateSlot& slot = Slot(state_slot);
  std::lock_guard<std::mutex> lock(slot.mu);
  DeviceState& st = slot.st;

  const int frames = options.frames;
  const std::size_t pixel_pitch = width * height;
  // Two different pitches, and conflating them overflows by `frames` slots:
  //   curve_pitch = n       -- number of visit positions (pixels + 1, because the
  //                            trailing ForgetGravity visit revisits the origin)
  //   pixel_pitch = pixels  -- the frame buffer the caller owns
  const std::uint64_t phash = HashPaletteAndTree(palette, tree);
  const bool want_u16 = options.emit_u16 && out_u16 != nullptr;
  const bool want_in_u16 = options.upload_u16 && in_u16 != nullptr;
  const int in_channels = want_in_u16 ? (options.in_channels == 3 ? 3 : 4) : 4;
  const bool in_yuv = want_in_u16 &&
                      options.in_mode == BlockOptions::InMode::PlanarYuv444;
  const bool in_yuv_pre = want_in_u16 &&
      options.in_mode == BlockOptions::InMode::PlanarYuv444Prepass;
  const bool in_yuv420 = want_in_u16 &&
      options.in_mode == BlockOptions::InMode::PlanarYuv420;
  const bool in_yuv_any = in_yuv || in_yuv_pre || in_yuv420;
  // Bytes of input for a given number of pixels.  4:2:0 is 1.5 bytes per pixel and
  // 1.5 is not an integer, so a per-pixel figure truncates and the reader
  // desynchronises by a byte a frame -- the same class of bug as the input/output
  // size collision this codebase has already paid for once.  4:2:0 also requires
  // even dimensions, which the decoder enforces for us, so the division is exact.
  const auto in_bytes_for = [&](std::size_t total_px) -> std::size_t {
    if (in_yuv420) return (total_px * 3) / 2;
    if (in_yuv_any) return total_px * 3;
    return total_px * in_channels * sizeof(std::uint16_t);
  };
  // Reuse whenever this call fits in the buffers already allocated.  A clip whose
  // frame count is not a multiple of the batch ends on a short batch, and keying
  // on the exact frame count would rebuild every device buffer -- ~440 ms -- for
  // that one launch.  Keying on capacity lets the tail reuse the full-size ones.
  // Reuse whenever this call fits in the buffers already allocated *and* the
  // output format asked for is the one they were built for.  A clip that switches
  // between the float and uint16 paths rebuilds rather than reading a buffer of
  // the wrong kind.
  const bool reuse = st.valid && st.level == level && st.width == width &&
                     st.height == height && st.block == options.block &&
                     st.phash == phash && frames <= st.frames &&
                     st.want_u16 == want_u16 &&
                     st.want_in_u16 == want_in_u16 &&
                     st.in_channels == in_channels &&
                     st.in_mode == (in_yuv ? 1 : in_yuv_pre ? 2 : in_yuv420 ? 3 : 0);
  const auto t_setup0 = std::chrono::steady_clock::now();
  double curve_ms = 0.0;
  std::size_t unvisited = 0;
  int unvisited_pixel = -1;
  int n = 0, nblocks_total = 0;
  // Capacity (allocated) versus this call's extent (used).
  std::size_t pixel_capacity = 0, curve_capacity = 0;
  std::size_t pixel_used = 0, curve_used = 0;

  if (reuse) {
    n = st.n;
    nblocks_total = st.nblocks_total;
    pixel_capacity = st.pixel_total;
    curve_capacity = st.curve_total;
    // The unvisited pixel belongs to the geometry, not to the batch, so it has to
    // come back with the reused state.  Without this the fill would run for the
    // first batch and silently stop for every batch after it -- which is the kind of
    // bug that only shows up as a slowly drifting output nobody can reproduce.
    unvisited = st.unvisited;
    unvisited_pixel = st.unvisited_pixel;
  } else {
    st.Release();
    std::vector<int> owner;
    std::vector<int> curve;
    const auto t_curve0 = std::chrono::steady_clock::now();
    BuildCurveIndex(level, width, height, &curve, &owner);
    curve_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t_curve0)
                   .count();
    n = static_cast<int>(curve.size());
    if (n < static_cast<int>(width * height)) {
      return "compacted curve is shorter than the image";
    }
    // Which pixels the curve never reaches, and how many.  This is not always
    // zero, and it is not a simple function of the shape: BuildCurveIndex
    // enumerates the closed-form Hilbert path over [0, 4^level - 1) and keeps only
    // the cells inside the image, and which in-bounds cell that leaves unvisited
    // was measured across ten geometries rather than derived --
    //
    //     1920x1080  1280x720  768x1024  720x1280  640x360  854x480   ->  0
    //     1024x768   1024x1024                                        ->  1 at (w-1, 0)
    //     2048x2048                                                    ->  1 at (w-1, 0)
    //     33x17                                                         ->  1 at (31, 0)
    //
    // So FHD is clean and 1024x768 is not, which is why this went unnoticed: the
    // five-minute benchmark clip has none.  One cell is excluded, so the unvisited
    // set is at most one pixel, but it is counted rather than assumed, and the index
    // is kept so the uint16 and yuv444 outputs can write it from the source.
    for (std::size_t p = 0; p < owner.size(); ++p) {
      if (owner[p] < 0) {
        if (unvisited == 0) unvisited_pixel = static_cast<int>(p);
        ++unvisited;
      }
    }

    const std::size_t curve_pitch = static_cast<std::size_t>(n);
    st.unvisited = unvisited;
    st.unvisited_pixel = unvisited_pixel;
    // Allocate for `frames`, which is the capacity the cache is keyed on.
    st.frames = frames;
    pixel_capacity = pixel_pitch * static_cast<std::size_t>(frames);
    curve_capacity = curve_pitch * static_cast<std::size_t>(frames);
    nblocks_total = (n + options.block - 1) / options.block;

    double host_weights[kErrorQueueLength];
    build_error_weights(host_weights);

    for (int i = 0; i < kStreams; ++i) {
      if (cudaMalloc(&st.d_cx_buf[i], curve_capacity * 4 * sizeof(float)) !=
          cudaSuccess) {
        st.Release();
        return "cudaMalloc curve field failed";
      }
      if (cudaMalloc(&st.d_index_buf[i], curve_capacity) != cudaSuccess) {
        st.Release();
        return "cudaMalloc index failed";
      }
      // The float4 pixel buffer is the gather's *input*, so it exists only when the
      // input is float4.  With a uint16 input the gather reads the uint16 buffer
      // directly and never touches it -- which at batch 16 / 1080p is 531 MB of VRAM
      // that simply is not allocated.
      if (!want_in_u16) {
        if (cudaMalloc(&st.d_pixels_buf[i], pixel_capacity * sizeof(float4)) !=
            cudaSuccess) {
          st.Release();
          return "cudaMalloc pixels failed";
        }
        // Pinned staging for the pixel buffer.  Falls back to a direct copy from the
        // caller's pageable memory if the pin fails (e.g. no more pageable RAM),
        // which costs bandwidth but never correctness.
        if (cudaHostAlloc(&st.h_pinned_buf[i], pixel_capacity * sizeof(float4),
                          cudaHostAllocPortable) != cudaSuccess) {
          st.h_pinned_buf[i] = nullptr;
        }
      } else {
        // Four uint16 per pixel: half the bytes of the float4 buffer, and half the
        // H2D.  Planar 4:4:4 is 3 bytes per pixel, 2.7x less again.
        const std::size_t need = in_yuv_any
                                     ? in_bytes_for(pixel_capacity)
                                     : pixel_capacity * in_channels *
                                           sizeof(std::uint16_t);
        if (in_yuv_any) {
          if (cudaMalloc(&st.d_in_yuv_buf[i], need) != cudaSuccess) {
            st.Release();
            return "cudaMalloc yuv input failed";
          }
          if (in_yuv_pre) {
            // The coalesced pass's output: 3 uint16 per pixel, device-local.
            if (cudaMalloc(&st.d_rgb16_buf[i],
                           pixel_capacity * 3 * sizeof(std::uint16_t)) !=
                cudaSuccess) {
              st.Release();
              return "cudaMalloc rgb16 intermediate failed";
            }
          }
        } else if (cudaMalloc(&st.d_in16_buf[i], need) != cudaSuccess) {
          st.Release();
          return "cudaMalloc uint16 input failed";
        }
        // Staging only for callers whose buffer is *not* already page-locked.  The
        // video pipeline reads frames straight into pinned memory, so it needs none,
        // and 265 MiB of pinned host memory is not worth reserving for a memcpy that
        // no longer happens.
        if (!options.in_u16_pinned &&
            cudaHostAlloc(&st.h_pinned16_buf[i],
                          pixel_capacity * in_channels * sizeof(std::uint16_t),
                          cudaHostAllocPortable) != cudaSuccess) {
          st.h_pinned16_buf[i] = nullptr;
        }
      }
      if (cudaStreamCreate(&st.stream_buf[i]) != cudaSuccess) {
        st.stream_buf[i] = nullptr;
      }
      // Four uint16 per pixel, so half the bytes of the float4 buffer beside it.
      // Only allocated when asked for, so the single-image path pays nothing.
      if (want_u16) {
        if (cudaMalloc(&st.d_u16_buf[i],
                       pixel_capacity * 4 * sizeof(std::uint16_t)) != cudaSuccess) {
          st.Release();
          return "cudaMalloc uint16 output failed";
        }
      }
    }
    if (cudaMalloc(&st.d_curve, curve.size() * sizeof(int)) != cudaSuccess) {
      st.Release();
      return "cudaMalloc curve failed";
    }
    if (cudaMalloc(&st.d_owner, owner.size() * sizeof(int)) != cudaSuccess) {
      st.Release();
      return "cudaMalloc owner failed";
    }
    if (cudaMalloc(&st.d_weights, sizeof(host_weights)) != cudaSuccess) {
      st.Release();
      return "cudaMalloc weights failed";
    }

    cuda_common::UploadPalette(palette, &st.d_palette);
    cuda_common::UploadTree(tree, &st.d_nodes);
    if (st.d_palette == nullptr || st.d_nodes == nullptr) {
      st.Release();
      return "palette/tree upload failed";
    }
    // Flat candidate index.  Optional: if the palette does not fit the packing the
    // recursive ClosestColor remains the reference path, and nothing else changes.
    st.d_search = nullptr;
    DevSearch* built = nullptr;
    const int search_assoc = palette.associate_alpha ? 1 : 0;
    // RD_FLAT_SEARCH=0 forces the recursive ClosestColor, which is how the two are
    // A/B'd on real video: same render, decoded output compared byte for byte.
    const char* fs = std::getenv("RD_FLAT_SEARCH");
    const bool want_flat = fs == nullptr || fs[0] != '0';
    if (want_flat && cuda_common::BuildFlatSearch(tree, search_assoc, &built) &&
        built != nullptr) {
      st.d_search = built;
      cudaMemcpyToSymbol(cuda_common::g_flat_search, &st.d_search, sizeof(st.d_search));
    } else {
      const DevSearch* none = nullptr;
      cudaMemcpyToSymbol(cuda_common::g_flat_search, &none, sizeof(none));
    }
    if (cudaMemcpy(st.d_weights, host_weights, sizeof(host_weights),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      st.Release();
      return "weights upload failed";
    }
    if (cudaMemcpy(st.d_curve, curve.data(), curve.size() * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      st.Release();
      return "curve upload failed";
    }
    if (cudaMemcpy(st.d_owner, owner.data(), owner.size() * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      st.Release();
      return "owner upload failed";
    }
    st.level = level;
    st.width = width;
    st.height = height;
    st.block = options.block;
    st.phash = phash;
    st.want_u16 = want_u16;
    st.want_in_u16 = want_in_u16;
    st.in_channels = in_channels;
    st.in_mode = in_yuv ? 1 : in_yuv_pre ? 2 : in_yuv420 ? 3 : 0;
    st.n = n;
    st.nblocks_total = nblocks_total;
    st.pixel_total = pixel_capacity;
    st.curve_total = curve_capacity;
    st.valid = true;
  }

  // This call's extent, which may be smaller than the allocation for a short batch.
  pixel_used = pixel_pitch * static_cast<std::size_t>(frames);
  curve_used = static_cast<std::size_t>(n) * static_cast<std::size_t>(frames);
  (void)curve_used;

  if (unvisited != 0 && !st.warned_unvisited) {
    st.warned_unvisited = true;
    // Reported, not fatal.  The float path is correct as it stands -- `batch` arrives
    // pre-filled with the source frame and the scatter only overwrites owned pixels,
    // so an unvisited pixel keeps its source value, which is what ImageMagick's own
    // recursion leaves there too.  The uint16 and yuv444 paths needed the explicit
    // fill, because their buffer has no pre-fill; that is handled above, so this is a
    // note rather than a warning.  The pixel is named because the condition is not
    // something the code can currently predict, and a number nobody can act on is
    // only half an answer.
    std::fprintf(stderr,
                 "[blocks] curve reaches %zu of %zu pixels; the %zu it does not "
                 "reach keep their source value, which is what ImageMagick's own "
                 "recursion leaves there.  First at index %d (%zu,%zu).  The "
                 "uint16/yuv444 output path writes these explicitly.\n",
                 width * height - unvisited, width * height, unvisited,
                 unvisited_pixel, static_cast<std::size_t>(unvisited_pixel) % width,
                 static_cast<std::size_t>(unvisited_pixel) / width);
  }

  const int assoc = palette.associate_alpha ? 1 : 0;
  const std::size_t bytes = pixel_used * sizeof(float4);
  // Round-robin the buffers.  Crucially, the sync at the end waits only on *this*
  // stream, so the other stream's copies keep moving while this batch is in
  // flight -- that is where the overlap comes from.
  st.stream_turn = (st.stream_turn + 1) % kStreams;
  const int si = st.stream_turn;
  float4* const d_pixels = st.d_pixels_buf[si];
  float* const d_cx = st.d_cx_buf[si];
  unsigned char* const d_index = st.d_index_buf[si];
  std::uint16_t* const d_u16 = want_u16 ? st.d_u16_buf[si] : nullptr;
  // When the caller wants uint16 output the float4 buffer is only needed as the
  // *input* to the gather, so it is not written back -- that is the other half of
  // the saving, on top of downloading half the bytes.
  // The float4 buffer is written by the scatter only when the caller wants the
  // float result.  When it wants uint16 the buffer is purely the gather's input,
  // so writing it back is the other half of the saving -- on top of downloading
  // half the bytes.  RD_CHECK_U16 needs both, so it turns this back on.
  const bool check_u16 = std::getenv("RD_CHECK_U16") != nullptr;
  float4* const d_pixels_out = (want_u16 && !check_u16) ? nullptr : d_pixels;
  float4* const h_pinned = st.h_pinned_buf[si];
  cudaStream_t const stream = st.stream_buf[si];
  const double setup_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t_setup0)
                             .count();

  // Host -> device.  Through pinned staging when available, so the transfer runs
  // at full PCIe speed and asynchronously.
  //
  // The uint16 path copies half the bytes, which is the point of it: the host no
  // longer widens rgba64le into float4, so neither the pinned memcpy nor the H2D
  // carries 16 bytes per pixel to carry 8.
  std::uint16_t* const h_pinned16 = st.h_pinned16_buf[si];
  std::uint16_t* const d_in16 =
      want_in_u16 && !in_yuv_any ? st.d_in16_buf[si] : nullptr;
  unsigned char* const d_in_yuv = in_yuv_any ? st.d_in_yuv_buf[si] : nullptr;
  if (want_in_u16) {
    const std::size_t bytes_in = in_bytes_for(pixel_used);
    if (in_yuv_any) {
      cudaMemcpyAsync(d_in_yuv, in_u16, bytes_in, cudaMemcpyHostToDevice, stream);
      // Checked here rather than left to the next launch check: a failed copy used to
      // surface as "invalid argument" from the gather, hundreds of lines away from
      // its cause.
      { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return std::string("yuv H2D failed: ") + cudaGetErrorString(e); }
    } else if (options.in_u16_pinned) {
      // The caller deposited the frames straight into page-locked memory, so there is
      // nothing to stage.  This is the path the video pipeline uses; without it every
      // frame is memcpy'd into staging first, which is 9.35 GB of host copy over a
      // 605-frame 1080p clip for no benefit.
      cudaMemcpyAsync(d_in16, in_u16, bytes_in, cudaMemcpyHostToDevice, stream);
    } else if (h_pinned16 != nullptr) {
      std::memcpy(h_pinned16, in_u16, bytes_in);
      cudaMemcpyAsync(d_in16, h_pinned16, bytes_in, cudaMemcpyHostToDevice, stream);
    } else {
      cudaMemcpyAsync(d_in16, in_u16, bytes_in, cudaMemcpyHostToDevice, stream);
    }
  } else if (options.batch_pinned) {
    // `batch` is page-locked, so this is a direct DMA and there is nothing to
    // stage.  Same bytes, same kernel, same output as the staging path.
    cudaMemcpyAsync(d_pixels, batch, bytes, cudaMemcpyHostToDevice, stream);
  } else if (h_pinned != nullptr) {
    std::memcpy(h_pinned, batch, bytes);
    cudaMemcpyAsync(d_pixels, h_pinned, bytes, cudaMemcpyHostToDevice, stream);
  } else {
    cudaMemcpyAsync(d_pixels, batch, bytes, cudaMemcpyHostToDevice, stream);
  }

  const int threads = 128;
  // Work items are (frame, block) pairs: each frame is an independent walk with
  // its own error state, so one launch spans the whole batch.
  const std::int64_t slots = static_cast<std::int64_t>(nblocks_total) * frames;
  const auto t_kernel0 = std::chrono::steady_clock::now();

  // The walk is one thread per (frame, block); gather and scatter are one thread
  // per (frame, position), i.e. block_size times more.  Sharing the walk's grid
  // here silently leaves most pixels unwritten.
  const int grid_walk = static_cast<int>((slots + threads - 1) / threads);
  const int grid_pixel = static_cast<int>(
      (static_cast<std::int64_t>(n) * frames + threads - 1) / threads);

  KernelTiming kt;
  if (kt.on) cudaEventRecord(kt.beg[0], stream);
  if (in_yuv_pre) {
    // Two kernels.  The first is a straight pixel-to-pixel pass, so every memory
    // access is coalesced; the second is the ordinary curve gather, now reading 3
    // contiguous uint16 instead of three separate bytes.  Same arithmetic, same
    // result as the fused kernel -- the point is purely the access pattern.
    std::uint16_t* const d_rgb16 = st.d_rgb16_buf[si];
    const std::int64_t total_px =
        static_cast<std::int64_t>(pixel_used);
    // One block-row per frame; see BlkYuvToRgb16Kernel for why the plane stride
    // cannot be folded into a flat index.
    dim3 pre(static_cast<unsigned>((pixel_pitch + threads - 1) / threads),
             static_cast<unsigned>(frames));
    BlkYuvToRgb16Kernel<<<pre, threads, 0, stream>>>(
        d_in_yuv, d_rgb16, static_cast<int>(pixel_pitch), frames);
    { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return std::string("yuv prepass launch failed: ") + cudaGetErrorString(e); }
    BlkGatherU16Kernel<float><<<grid_pixel, threads, 0, stream>>>(
        d_rgb16, st.d_curve, d_cx, n, frames, static_cast<int>(pixel_pitch), 3);
    { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return std::string("rgb16 gather launch failed: ") + cudaGetErrorString(e); }
  } else if (in_yuv420) {
    BlkGatherYuv420Kernel<float><<<grid_pixel, threads, 0, stream>>>(
        d_in_yuv, st.d_curve, d_cx, n, frames, static_cast<int>(pixel_pitch),
        static_cast<int>(width), static_cast<int>(height));
    { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return std::string("yuv420 gather launch failed: ") + cudaGetErrorString(e); }
  } else if (in_yuv) {
    BlkGatherYuv444Kernel<float><<<grid_pixel, threads, 0, stream>>>(
        d_in_yuv, st.d_curve, d_cx, n, frames, static_cast<int>(pixel_pitch));
    { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return std::string("yuv gather launch failed: ") + cudaGetErrorString(e); }
  } else if (want_in_u16) {
    BlkGatherU16Kernel<float><<<grid_pixel, threads, 0, stream>>>(
        d_in16, st.d_curve, d_cx, n, frames, static_cast<int>(pixel_pitch),
        in_channels);
  } else {
    BlkGatherKernel<float><<<grid_pixel, threads, 0, stream>>>(
        d_pixels, st.d_curve, d_cx, n, frames, static_cast<int>(pixel_pitch));
  }
  { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return std::string("gather launch failed: ") + cudaGetErrorString(e); }
  if (kt.on) cudaEventRecord(kt.end[0], stream);
  if (kt.on) cudaEventRecord(kt.beg[1], stream);
  BlkIndexWalkKernel<<<grid_walk, threads, 0, stream>>>(
      d_cx, st.d_palette, palette.count, st.d_weights, params.diffusion, assoc,
      st.d_nodes, n, nblocks_total, frames, options.block, d_index);
  { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return std::string("block walk launch failed: ") + cudaGetErrorString(e); }
  if (kt.on) cudaEventRecord(kt.end[1], stream);
  if (kt.on) cudaEventRecord(kt.beg[2], stream);
  if (options.emit_yuv444 && d_u16 != nullptr) {
    // Planar 4:4:4, 3 bytes per pixel.  The same buffer as the uint16 path, just a
    // third of the width, and the download below is sized to match.
    BlkScatterYuv444Kernel<<<grid_pixel, threads, 0, stream>>>(
        st.d_curve, st.d_owner, d_index, st.d_palette, n, frames,
        static_cast<int>(pixel_pitch), assoc,
        reinterpret_cast<unsigned char*>(d_u16));
  } else {
    BlkScatterKernel<<<grid_pixel, threads, 0, stream>>>(
        d_pixels_out, st.d_curve, st.d_owner, d_index, st.d_palette, n, frames,
        static_cast<int>(pixel_pitch), assoc, d_u16);
  }
  if (cudaGetLastError() != cudaSuccess) return "scatter launch failed";
  // The uint16 and yuv444 outputs are a bare cudaMalloc with no COPY_HOST_PTR, so the
  // one pixel the walk never visits has to be written from the source or it encodes
  // uninitialised device memory.  One thread per frame; the scatter left that pixel
  // alone, so the upload buffers still hold the original value there.  This is what
  // OpenCL does with a whole-buffer copy, done here for a single pixel because there
  // is only ever one -- and unlike OpenCL's fill-with-zeros branch, it lands on the
  // value ImageMagick itself would leave, so all three paths agree.
  if (d_u16 != nullptr && unvisited != 0) {
    const int fill_threads = 64;
    const int fill_blocks = (frames + fill_threads - 1) / fill_threads;
    // Exactly one of these is the live source, and which one depends on the input
    // mode -- `d_pixels_buf` is not even allocated on the uint16 path, so passing it
    // unconditionally is a null dereference, not a harmless extra argument.  The
    // planar source only serves the yuv444 output, where the layouts already match;
    // for rgba64le output out of a planar source the kernel writes a defined
    // neutral instead, because a 4:4:4 plane has no single source value per pixel
    // once it is being written as interleaved RGBA.
    const float4* fill_srcf = want_in_u16 ? nullptr : d_pixels;
    const std::uint16_t* fill_src16 = (want_in_u16 && !in_yuv_any) ? d_in16 : nullptr;
    const unsigned char* fill_srcyuv =
        (options.emit_yuv444 && in_yuv_any && !in_yuv420) ? d_in_yuv : nullptr;
    BlkFillUnvisitedKernel<<<fill_blocks, fill_threads, 0, stream>>>(
        fill_srcf, fill_src16, fill_srcyuv, unvisited_pixel, frames,
        static_cast<int>(pixel_pitch), assoc, options.emit_yuv444, d_u16);
    if (cudaGetLastError() != cudaSuccess) return "unvisited-pixel fill launch failed";
  }
  if (kt.on) cudaEventRecord(kt.end[2], stream);
  // Download the uint16 result when asked for: half the bytes, and the host then
  // has nothing to convert.
  if (d_u16 != nullptr) {
    // Four uint16 per pixel, not the float4 buffer's sixteen bytes per pixel.
    // Using `bytes` here reads four times past the end of the buffer; the error
    // surfaces at the *next* batch's launch check, which is where it cost me an
    // afternoon to trace.
    // Planar 4:4:4 is three bytes per pixel instead, so the width follows the
    // output mode rather than being hard-coded to the rgba64le layout.
    const std::size_t bytes_u16 = options.emit_yuv444
                                      ? pixel_used * 3
                                      : pixel_used * 4 * sizeof(std::uint16_t);
    if (stream != nullptr) {
      cudaMemcpyAsync(out_u16, d_u16, bytes_u16, cudaMemcpyDeviceToHost, stream);
      // Debug only: fetch the float result too, so the check below compares like
      // with like.  In the shipping path this copy does not happen.
      if (d_pixels_out != nullptr) {
        cudaMemcpyAsync(batch, d_pixels, bytes, cudaMemcpyDeviceToHost, stream);
      }
    } else {
      cudaMemcpy(out_u16, d_u16, bytes_u16, cudaMemcpyDeviceToHost);
      if (d_pixels_out != nullptr) {
        cudaMemcpy(batch, d_pixels, bytes, cudaMemcpyDeviceToHost);
      }
    }
  } else {
    if (stream != nullptr) {
      cudaMemcpyAsync(batch, d_pixels, bytes, cudaMemcpyDeviceToHost, stream);
    } else {
      cudaMemcpy(batch, d_pixels, bytes, cudaMemcpyDeviceToHost);
    }
  }
  // One sync for the whole chain instead of two mid-pipeline ones, and only on the
  // stream this launch used.
  // Report the driver's own words, not "dither failed".  A sync failure is almost
  // always a sticky error from an earlier launch -- an illegal access in some kernel
  // three launches back -- and a bare "dither failed" sends you looking at the wrong
  // place.  cudaGetErrorString on the value the sync actually returned names it.
  if (const cudaError_t sync_err = cudaStreamSynchronize(stream)) {
    return std::string("dither failed: ") + cudaGetErrorString(sync_err);
  }
  if (kt.on) {
    // Read after the sync, so these are the kernels' own durations and not the copy
    // time folded in.
    float ms[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) cudaEventElapsedTime(&ms[i], kt.beg[i], kt.end[i]);
    const char* names[3] = {"gather", "walk", "scatter"};
    std::fprintf(stderr, "[kern] gather %7.2f  walk %7.2f  scatter %7.2f  (sum %7.2f ms)\n",
                 ms[0], ms[1], ms[2], ms[0] + ms[1] + ms[2]);
    // Search cost, per pixel.  Compile-time gated; see the note on RD_CC_COUNT in
    // rd_cuda_common.cuh for why it cannot be a runtime flag.
#if defined(RD_CC_STATS)
    if (std::getenv("RD_CC_STATS") != nullptr) {
      unsigned long long e = 0, p = 0, d = 0;
      cudaMemcpyFromSymbol(&e, cuda_common::g_cc_entries, sizeof(e));
      cudaMemcpyFromSymbol(&p, cuda_common::g_cc_pixels, sizeof(p));
      cudaMemcpyFromSymbol(&d, cuda_common::g_cc_descent, sizeof(d));
      cudaMemcpyToSymbol(cuda_common::g_cc_entries, (unsigned long long)0, sizeof(e));
      cudaMemcpyToSymbol(cuda_common::g_cc_pixels, (unsigned long long)0, sizeof(p));
      cudaMemcpyToSymbol(cuda_common::g_cc_descent, (unsigned long long)0, sizeof(d));
      if (p > 0) {
        std::fprintf(stderr, "[search] nodes/pixel %6.2f  descent/pixel %5.2f  (n=%llu)\n",
                     static_cast<double>(e) / static_cast<double>(p),
                     static_cast<double>(d) / static_cast<double>(p), p);
      }
    }
#endif
    (void)names;
  }
  // The device's uint16 must equal what FloatsToRaw computes from the float4 it
  // also produced, sample for sample, because the encoder consumes these bytes
  // directly and there is no second chance to convert them.  Checking it costs a
  // full host pass over the batch, so it is off unless asked for:
  //   set RD_CHECK_U16=1
  // The byte-for-byte A/B against the host path is `--gpu-float-out`; that one
  // needs no debug download and covers the whole pipeline.
  if (want_u16 && d_pixels_out != nullptr && std::getenv("RD_CHECK_U16") != nullptr) {
    // Temporary: with both buffers downloaded, the uint16 the device produced must
    // equal what FloatsToRaw would compute from the float4 it also produced.
    std::size_t bad = 0;
    int shown = 0;
    for (std::size_t i = 0; i < pixel_used; ++i) {
      for (int c = 0; c < 4; ++c) {
        const RgbaF& px = batch[i];
        const float f = c == 0 ? px.r : c == 1 ? px.g : c == 2 ? px.b : px.a;
        const std::uint16_t want = static_cast<std::uint16_t>(f);
        const std::uint16_t got = out_u16[4 * i + c];
        if (want != got) {
          ++bad;
          if (shown < 5) {
            std::fprintf(stderr,
                         "[u16check] pixel %llu ch %d: float %.1f -> want %u, got %u\n",
                         static_cast<unsigned long long>(i), c, static_cast<double>(f),
                         want, got);
            ++shown;
          }
        }
      }
    }
    std::fprintf(stderr, "[u16check] %llu of %llu samples differ\n",
                 static_cast<unsigned long long>(bad),
                 static_cast<unsigned long long>(pixel_used * 4));
  }
  const double kernel_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t_kernel0)
                               .count();
  // Suppressed while a progress bar owns the console line.  The bar is drawn
  // without a trailing newline (a newline per update would scroll the screen),
  // so this line would otherwise be appended to the bar's own line and then
  // half-erased by the bar's next update -- and there is one of these every
  // `frames` (16) frames, so on a long render the console ends up a mess.
  //
  // Not suppressed when stderr is a file: there the two interleave harmlessly
  // and this timing is exactly what you want in the log.  Not suppressed
  // without a bar either, which is the single-image case where there are only
  // a handful of these and they are the only diagnostic output there is.
  // RD_TRACE overrides, so the per-batch numbers are always one env var away.
  if (!rd::ProgressHoldsLine() || std::getenv("RD_TRACE") != nullptr) {
    std::fprintf(stderr,
                 "[blocks] %.0fx%.0f x%d frames  setup=%.1f ms%s  dither+IO=%.1f ms  "
                 "(%.1f ms/frame, n=%d, B=%d, %lld work items)\n",
                 static_cast<double>(width), static_cast<double>(height), frames,
                 setup_ms, reuse ? " (cached)" : "", kernel_ms,
                 kernel_ms / static_cast<double>(frames), n, options.block,
                 static_cast<long long>(slots));
  }
  return std::string();
}

void* CudaAllocPinned(std::size_t bytes) {
  void* p = nullptr;
  if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) != cudaSuccess) return nullptr;
  return p;
}

void CudaFreePinned(void* p) {
  if (p != nullptr) cudaFreeHost(p);
}

}  // namespace rd



