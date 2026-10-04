// SPDX-License-Identifier: GPL-3.0-or-later
// rd_opencl.cpp -- OpenCL engine for the block walk.  See rd_opencl.h for scope
// and the reason the blocks partition is the one being ported.
#include "rd_opencl.h"

#if defined(RD_WITH_OPENCL)

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
// cl.h does not pull this in on every SDK, and the CL_PLATFORM_* codes below
// are the difference between "no OpenCL installed" and "clGetPlatformIDs
// returned 37 for a reason nobody can name".
#include <CL/cl_platform.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "rd_octree.h"

namespace rd {
namespace {

// ---------------------------------------------------------------------------
// The kernels.
//
// A transliteration of BlkGatherKernel / BlkIndexWalkKernel /
// BlkScatterKernel in src/rd_blocks_cuda.cu and of the helpers in
// include/rd_cuda_common.cuh.  A transliteration rather than a shared header
// because those are CUDA C++ (__device__, float4, __ffs, atomicAdd) and cannot
// be included by an ordinary C++ compiler; the two files are kept in step by
// hand, and every place the dialect forced a difference is called out below.
//
// THE THREE PLACES THE DIALECT MATTER:
//
//   1. __dadd_rn / __dmul_rn  ->  + and *.
//      OpenCL C's default rounding for double is correctly rounded and the
//      default build does not contract, so these are the same operations.  That
//      is true only while -cl-fast-relaxed-math and -cl-mad-enable stay off; one
//      of those would let the compiler reassociate or fuse, the image would
//      still look plausible, and every pixel would be wrong.  BuildOptions below
//      passes neither and says so in the source, because that is the only place
//      a future editor will look.
//
//   2. __ffs(m) - 1  ->  cl_ctz() below.
//      OpenCL C has no __ffs.  Written as a scan over the mask rather than a
//      vendor builtin so nothing depends on how wide a clz the driver has; a
//      16-bit mask means at most 16 iterations and the loop is never the cost.
//
//   3. The scatter's owner test.
//      Not a dialect difference, but the single easiest thing in this port to
//      get wrong, so it is called out: the Hilbert curve is NOT a permutation of
//      a 1920x1080 frame.  Some pixels are visited twice.  Without
//      `owner[p] != i -> skip`, the first visit would win where ImageMagick's
//      loop -- and RiemersmaBlocksCpu's -- take the last.  The output would be
//      almost right, which is the worst kind of wrong.
//
// The register-resident error queue ports unchanged: `double e[4 * 16]` with
// constant indices is private memory in OpenCL too, and the compiler may keep it
// in registers.  That array is the reason the kernel is fast, and it is IM's
// ErrorQueueLength, so it cannot shrink.
// ---------------------------------------------------------------------------
const char* kSource = R"CLC(
#define kErrorQueueLength 16
#define kQRange 65535.0
#define kQScale (1.0 / 65535.0)
#define kErb (1.0 / 16.0)
#define kMaxTreeDepth 8

typedef struct {
  int child[16];
  int parent;
  double total_color[4];
  double quantize_error;
  ulong number_unique;
  uint color_number;
  uint id;
  uint level;
} DevNode;

typedef struct {
  uint mask;
  ulong packed;
} DevSearch;

// Raster order -> curve order.  Pure data movement, no arithmetic.
__kernel void gather(__global const float* pixels, __global const int* curve,
                     int n, int frames, int pixels_per_frame,
                     __global float* cx) {
  const int slot = get_global_id(0);
  if (slot >= n * frames) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const int p = curve[i];
  const __global float* src = pixels + (ulong)frame * pixels_per_frame * 4;
  __global float* dst = cx + (ulong)frame * n * 4;
  dst[i * 4 + 0] = src[(ulong)p * 4 + 0];
  dst[i * 4 + 1] = src[(ulong)p * 4 + 1];
  dst[i * 4 + 2] = src[(ulong)p * 4 + 2];
  dst[i * 4 + 3] = src[(ulong)p * 4 + 3];
}

// Raster order -> curve order, reading rgba64le and widening on the device.
//
// The CUDA original is BlkGatherU16Kernel, and this is a transliteration of it.
// The point is to delete the host's uint16 -> float loop, which is the reader's
// dominant cost and memory-bandwidth-bound: 605 1080p frames is two billion
// conversions reading 8 bytes and writing 16, about 48 GB of host traffic.
//
// The widening is EXACT -- every uint16 is representable in a float's 24-bit
// mantissa, and the host did nothing but static_cast<float> -- so the dither
// sees bit-identical input either way.  That exactness is the whole reason this
// is safe to move off the host at all; a lossy widening here would be
// indistinguishable from a dither bug later on.
__kernel void gather_u16(__global const ushort* pixels, __global const int* curve,
                         int n, int frames, int pixels_per_frame, int channels,
                         __global float* cx) {
  const int slot = get_global_id(0);
  if (slot >= n * frames) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const int p = curve[i];
  const __global ushort* src = pixels + (ulong)frame * pixels_per_frame * channels;
  __global float* dst = cx + (ulong)frame * n * 4;
  dst[i * 4 + 0] = (float)src[(ulong)p * channels + 0];
  dst[i * 4 + 1] = (float)src[(ulong)p * channels + 1];
  dst[i * 4 + 2] = (float)src[(ulong)p * channels + 2];
  // rgb48le carries no alpha.  Synthesise the same 65535 that rgba64le would have
  // delivered, so the dither sees identical input either way -- and it is the
  // value the scatter writes back, so the encoder's stream is unchanged.
  dst[i * 4 + 3] = (channels == 4) ? (float)src[(ulong)p * channels + 3] : 65535.0f;
}
__kernel void scatter(__global const int* curve, __global const int* owner,
                      __global const uchar* index, __global const double* palette,
                      int n, int frames, int pixels_per_frame, int assoc,
                      __global float* pixels) {
  const int slot = get_global_id(0);
  if (slot >= n * frames) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const int p = curve[i];
  // Bounds check.  A curve entry outside the frame would make the gather read
  // past `pixels` and, far worse, make the WRITE below land past the end of the
  // destination buffer -- corrupting whatever the driver allocated next, which is
  // a textbook source of "a few scattered pixels, intermittently, in a run that
  // is otherwise identical".  The CUDA scatter has no such check, which is one
  // reason this engine could get away with something the other could not.
  if (p < 0 || p >= pixels_per_frame) return;
  if (owner[p] != i) return;
  // The address-space qualifier is part of the pointer's type in OpenCL C, so a
  // helper that takes a buffer pointer must say __global on the parameter.  This
  // is the first thing a CUDA-trained eye misses: in CUDA a `const T*` parameter
  // just works, and here it does not compile.
  const __global double* e = palette + 4 * (int)index[(ulong)frame * n + i];
  __global float* dst = pixels + (ulong)frame * pixels_per_frame * 4;
  dst[(ulong)p * 4 + 0] = (float)e[0];
  dst[(ulong)p * 4 + 1] = (float)e[1];
  dst[(ulong)p * 4 + 2] = (float)e[2];
  // The alpha channel is written UNCONDITIONALLY, and forced opaque when alpha is
  // not associated.  This mirrors BlkScatterKernel in src/rd_blocks_cuda.cu
  // exactly:
  //
  //     out.w = assoc ? (float)p[3] : (float)65535.0;
  //
  // The earlier version of this kernel wrote alpha only when `assoc`, leaving
  // whatever the destination buffer already held.  That is not a no-op: the
  // destination came from the caller's frame, so on an image that HAS an alpha
  // channel but does not associate it, the output kept the *source* alpha while
  // the palette colour went into RGB -- a half-dithered pixel that is neither the
  // input nor the quantised colour.
  dst[(ulong)p * 4 + 3] = assoc ? (float)e[3] : (float)65535.0;
}

// Curve order -> raster order, writing packed rgba64le: four uint16 per pixel,
// little-endian, which is exactly what the video pipe hands to ffmpeg.
//
// The point is that this is the byte layout the encoder consumes, so the host
// does no float -> uint16 conversion at all -- and half the download goes with
// it, since four uint16 is 8 bytes against float4's 16.
//
// THE CAST IS THE WHOLE RISK, and it must be the two-step one.  CUDA spells it
// `static_cast<std::uint16_t>(static_cast<float>(p[i]))` and so must this:
// double -> float -> truncating uint16.  Writing (ushort)e[0] in one step is
// *not* the same operation -- the double->float step rounds, and the truncation
// then happens on a different value -- and it would differ from CUDA on exactly
// the samples where that rounding crosses an integer boundary.  A one-character
// simplification here is a silent, scattered, off-by-one-in-the-last-bit
// divergence from the engine this whole port is defined against, and it would
// only ever show up in the encoded video.
__kernel void scatter_u16(__global const int* curve, __global const int* owner,
                          __global const uchar* index, __global const double* palette,
                          int n, int frames, int pixels_per_frame, int assoc,
                          __global ushort* pixels) {
  const int slot = get_global_id(0);
  if (slot >= n * frames) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const int p = curve[i];
  if (p < 0 || p >= pixels_per_frame) return;
  if (owner[p] != i) return;
  const __global double* e = palette + 4 * (int)index[(ulong)frame * n + i];
  __global ushort* dst = pixels + (ulong)frame * pixels_per_frame * 4;
  dst[(ulong)p * 4 + 0] = (ushort)(float)e[0];
  dst[(ulong)p * 4 + 1] = (ushort)(float)e[1];
  dst[(ulong)p * 4 + 2] = (ushort)(float)e[2];
  // Alpha is written, not assumed, for the same reason as in the float scatter:
  // the encoder consumes these bytes directly and cannot know what the float
  // path would have put there.
  dst[(ulong)p * 4 + 3] = (ushort)(assoc ? (float)e[3] : (float)65535.0);
}

float cl_clamp_pixel(double v) {
  if (v < 0.0) return 0.0f;
  if (v >= kQRange) return (float)kQRange;
  return (float)v;
}

uchar cl_scale_char(float q) {
  if (isnan(q) || q <= 0.0f) return 0;
  const float scaled = q / 257.0f;
  if (scaled >= 255.0f) return 255;
  return (uchar)(scaled + 0.5f);
}

void cl_associate_from_double(int assoc, double r, double g, double b, double a,
                              double* out) {
  const double ca = (double)cl_clamp_pixel(a);
  out[3] = ca;
  if (!assoc || a == kQRange) {
    out[0] = (double)cl_clamp_pixel(r);
    out[1] = (double)cl_clamp_pixel(g);
    out[2] = (double)cl_clamp_pixel(b);
    return;
  }
  const double alpha = kQScale * ca;
  // The CUDA original writes __dadd_rn(..., 0.0) rather than the bare product.
  // That is not a no-op: it forces the value through a rounding step, and
  // dropping it would let the multiply's unrounded intermediate through.
  // Kept, deliberately, for that reason.
  out[0] = alpha * (double)cl_clamp_pixel(r) + 0.0;
  out[1] = alpha * (double)cl_clamp_pixel(g) + 0.0;
  out[2] = alpha * (double)cl_clamp_pixel(b) + 0.0;
}

int cl_node_id_packed(uint c0, uint c1, uint c2, uint c3, int assoc, int index) {
  int id = (int)((c0 >> index) & 0x01);
  id |= (int)((c1 >> index) & 0x01) << 1;
  id |= (int)((c2 >> index) & 0x01) << 2;
  if (assoc) id |= (int)((c3 >> index) & 0x01) << 3;
  return id;
}

int cl_ctz(uint m) {
  if (m == 0u) return -1;
  int n = 0;
  while ((m & 1u) == 0u) { m >>= 1; ++n; }
  return n;
}

// quantize.c:ClosestColor() over the subtree at `node`, with the traversal
// flattened into the ordered scan the flat index encodes.  Same candidate order,
// same running distance, same <= tie rule, same per-channel early exits -- the
// index *is* the traversal.
void cl_closest_color_flat(const __global const DevSearch* search,
                           const __global const double* palette, int assoc,
                           int node, const double* target, double* distance,
                           int* color_number) {
  const DevSearch s = search[node];
  uint m = s.mask;
  ulong pk = s.packed;
  while (m != 0u) {
    const int j = cl_ctz(m);
    m &= m - 1u;
    const __global const double* p =
        palette + 4 * (int)((pk >> (4 * j)) & 0xFull);
    double alpha = 1.0;
    double beta = 1.0;
    if (assoc) {
      alpha = kQScale * p[3];
      beta = kQScale * target[3];
    }
    double pixel = alpha * p[0] - beta * target[0];
    double d = pixel * pixel;
    if (d <= *distance) {
      pixel = alpha * p[1] - beta * target[1];
      d = d + pixel * pixel;
      if (d <= *distance) {
        pixel = alpha * p[2] - beta * target[2];
        d = d + pixel * pixel;
        if (d <= *distance) {
          if (assoc) {
            pixel = p[3] - target[3];
            d = d + pixel * pixel;
          }
          if (d <= *distance) {
            *distance = d;
            *color_number = (int)((pk >> (4 * j)) & 0xFull);
          }
        }
      }
    }
  }
}

int cl_select_index(const __global const DevNode* nodes,
                    const __global const DevSearch* search,
                    const __global const double* palette, int assoc,
                    const double* target) {
  const uint c0 = cl_scale_char(cl_clamp_pixel(target[0]));
  const uint c1 = cl_scale_char(cl_clamp_pixel(target[1]));
  const uint c2 = cl_scale_char(cl_clamp_pixel(target[2]));
  const uint c3 = cl_scale_char(cl_clamp_pixel(target[3]));
  int node = 0;
  for (int i = kMaxTreeDepth - 1; i > 0; --i) {
    const int id = cl_node_id_packed(c0, c1, c2, c3, assoc, i);
    const int child = nodes[node].child[id];
    if (child < 0) break;
    node = child;
  }
  double distance = 4.0 * (kQRange + 1.0) * (kQRange + 1.0) + 1.0;
  int index = 0;
  cl_closest_color_flat(search, palette, assoc, nodes[node].parent, target,
                        &distance, &index);
  return index;
}

__kernel void walk(__global const float* cx, __global const double* palette,
                   __global const double* weights, double diffusion, int assoc,
                   __global const DevNode* nodes, __global const DevSearch* search,
                   int n, int nblocks, int frames, int block_size,
                   __global uchar* out_index) {
  const int slot = get_global_id(0);
  if (slot >= nblocks * frames) return;

  const int frame = slot / nblocks;
  // NOT named `local`: that is a reserved address-space keyword in OpenCL C, so
  // `const int local = ...` is a parse error.  Renamed to blk (block index
  // within the frame), which is what the CUDA original calls it.
  const int blk = slot - frame * nblocks;
  // Same indices as the CUDA original: cx is four floats per pixel and the frame
  // offset is n*4 floats.  Written with float4 it reads better and is also
  // correct, but keeping the arithmetic identical makes the two files diffable
  // line by line -- which is the only practical defence against one of them
  // being edited and the other silently not.
  __global const float* frame_cx = cx + (ulong)frame * n * 4;

  const int begin = blk * block_size;
  const int end = (begin + block_size < n) ? (begin + block_size) : n;

  double e[4 * kErrorQueueLength];
  for (int k = 0; k < kErrorQueueLength; ++k) {
    for (int ch = 0; ch < 4; ++ch) e[4 * k + ch] = 0.0;
  }

  // Loop-invariant, hoisted with the same bit-exactness argument: same inputs,
  // same operations, same results.  On a consumer GeForce part FP64 runs at
  // 1/64 rate, so recomputing 16 of these per pixel is not free.
  double w[kErrorQueueLength];
  for (int k = 0; k < kErrorQueueLength; ++k) {
    w[k] = (kErb * diffusion) * weights[k];
  }

  for (int i = begin; i < end; ++i) {
    const int c = i * 4;
    double pr = (double)frame_cx[c + 0], pg = (double)frame_cx[c + 1];
    double pb = (double)frame_cx[c + 2], pa = (double)frame_cx[c + 3];
    if (assoc && pa != kQRange) {
      const double alpha = kQScale * pa;
      pr = alpha * pr; pg = alpha * pg; pb = alpha * pb;
    }
    for (int k = 0; k < kErrorQueueLength; ++k) {
      pr = pr + w[k] * e[4 * k + 0];
      pg = pg + w[k] * e[4 * k + 1];
      pb = pb + w[k] * e[4 * k + 2];
      if (assoc) pa = pa + w[k] * e[4 * k + 3];
    }
    pr = (double)cl_clamp_pixel(pr);
    pg = (double)cl_clamp_pixel(pg);
    pb = (double)cl_clamp_pixel(pb);
    if (assoc) pa = (double)cl_clamp_pixel(pa);

    double target[4];
    cl_associate_from_double(assoc, pr, pg, pb, pa, target);
    const int index = cl_select_index(nodes, search, palette, assoc, target);
    out_index[(ulong)frame * n + i] = (uchar)index;

    const __global const double* p = palette + 4 * index;
    for (int k = 0; k < kErrorQueueLength - 1; ++k) {
      for (int ch = 0; ch < 4; ++ch) e[4 * k + ch] = e[4 * (k + 1) + ch];
    }
    double cr = p[0], cg = p[1], cb = p[2], ca = p[3];
    if (assoc && ca != kQRange) {
      const double alpha = kQScale * ca;
      cr = alpha * cr; cg = alpha * cg; cb = alpha * cb;
    }
    const int last = 4 * (kErrorQueueLength - 1);
    e[last + 0] = pr - cr;
    e[last + 1] = pg - cg;
    e[last + 2] = pb - cb;
    if (assoc) e[last + 3] = pa - ca;
  }
}
)CLC";

// A SECOND source string, passed to clCreateProgramWithSource as a separate
// argument rather than concatenated onto kSource.
//
// That is not stylistic.  MSVC rejects a string literal over 16380 bytes, and
// kSource is already 14 KB of OpenCL C; the first attempt to add the two kernels
// below to it failed to compile with "string too big, closing truncated" partway
// through cl_select_index.  Concatenating adjacent literals would have worked and
// been more fragile, since every future kernel added to this file would have to
// respect an invisible character budget.  Two arguments is what the API is for, and
// the program text is identical either way.
//
// Leading newline so the two sources do not run together at the join.
const char* kSourceYuv = R"CLC(

// swscale's 8-bit YUV -> 16-bit RGB, bit-exact.
//
// PORTED, NOT DERIVED, and a transliteration of d_sws_yuv_to_rgb16 and
// d_sws_clip16 in rd_blocks_cuda.cu -- read the long note there before changing
// any of these numbers.  They come from yuv2rgba64_full_X_c_template with a 1:1
// plane ratio, and two of the six were got wrong by hand before they were printed
// by tools/probe_swscale_matrix.cpp and checked against ffmpeg's own rgba64le
// output: 0 mismatches over the whole 220x225 (Y,V) grid, 0 over a real 1920x1080
// frame.  The CUDA kernels are the reference these must match bit for bit -- if
// the two ever disagree, the disagreement presents as a hue shift and gets blamed
// on the dither.
//
// The one apparent difference from the CUDA source is signedness: there the first
// line is (y - 128u) * 512u in unsigned arithmetic, cast to int after.  That is
// mod 2^32, and since |y - 128| <= 128 the wrapped value cast back to int is
// exactly (y - 128) * 512, so the plain int form below is bit-identical.
inline int sws_clip16(int x) {
  const int v = x + (1 << 15);
  if (v < 0) return 0;
  if (v > 65535) return 65535;
  return v;
}

inline void sws_yuv_to_rgb16(int y, int u, int v, __private int* rgb) {
  int yy = (y - 128) * 512;
  const int uu = (u - 128) * 512;
  const int vv = (v - 128) * 512;
  yy += 0x10000;
  yy -= 8192;    // yuv2rgb_y_offset
  yy *= 9539;    // yuv2rgb_y_coeff
  yy += (1 << 13) - (1 << 29);

  const int ri = vv * 13075;               // yuv2rgb_v2r_coeff
  const int gi = vv * -6660 + uu * -3209;  // v2g, u2g
  const int bi = uu * 16525;               // yuv2rgb_u2b_coeff

  rgb[0] = sws_clip16((ri + yy) >> 14);
  rgb[1] = sws_clip16((gi + yy) >> 14);
  rgb[2] = sws_clip16((bi + yy) >> 14);
}

// Raster order -> curve order, reading planar 8-bit 4:4:4 and converting
// YCbCr -> RGB on the device.  Transliteration of BlkGatherYuv444Kernel.
//
// The win is bytes, not arithmetic: 4:4:4 is 3 bytes per pixel where rgba64le is 8,
// so the pipe and the H2D both move 2.67x less.  The reader is the pipeline's
// floor, so that is where the time was.  For OpenCL this is also the whole of the
// remaining video gap: without it the engine is confined to the rgba64le path
// while CUDA defaults to this one, and the engine itself is within 8% of CUDA's.
//
// Three planes per frame, so frame f's luma starts at f * 3 * pixels_per_frame
// and NOT f * pixels_per_frame.  The CUDA original's comment records that the
// latter reads frame f-1's U and V as if they were frame f's luma: correct for
// frame 0, nonsense for every frame after, and invisible because frame 0 is all
// anyone looked at.
__kernel void gather_yuv444(__global const uchar* planes, __global const int* curve,
                            int n, int frames, int pixels_per_frame,
                            __global float* cx) {
  const int slot = get_global_id(0);
  if (slot >= n * frames) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const int p = curve[i];
  const size_t base = (ulong)frame * 3 * pixels_per_frame + p;
  __private int rgb[3];
  sws_yuv_to_rgb16((int)planes[base], (int)planes[base + pixels_per_frame],
                   (int)planes[base + 2 * (ulong)pixels_per_frame], rgb);
  __global float* dst = cx + (ulong)frame * n * 4;
  dst[i * 4 + 0] = (float)rgb[0];
  dst[i * 4 + 1] = (float)rgb[1];
  dst[i * 4 + 2] = (float)rgb[2];
  dst[i * 4 + 3] = 65535.0f;
}

// The scatter, writing planar 8-bit 4:4:4.  Transliteration of
// BlkScatterYuv444Kernel plus d_rgb_to_yuv444.
//
// This exists for the same reason on the other side of the pipe: the encoder
// consumes raw frames, so 3 bytes per pixel against rgba64le's 8 takes the
// RGB->YUV conversion off the CPU entirely.  Measured on the CUDA side that
// conversion was 298 GB of host reads for an 18001-frame 1080p clip, after
// which the encode stage became 94% of the wall with the writer blocked at
// 597 MB/s -- below the pipe's own 0.86 GB/s.  Shrinking the pipe without moving
// the conversion would only have moved the same CPU work onto the writer thread.
//
// The rounding is BT.601 with 8-bit inputs, matching d_rgb_to_yuv444.  It is a
// different rounding from swscale's, and deliberately so: this decides WHO rounds,
// not what the picture is.  ffmpeg round-trips these frames through YUV on the way
// into the file either way.
__kernel void scatter_yuv444(__global const int* curve, __global const int* owner,
                             __global const uchar* index, __global const double* palette,
                             int n, int frames, int pixels_per_frame, int assoc,
                             __global uchar* yuv) {
  const int slot = get_global_id(0);
  if (slot >= n * frames) return;
  const int frame = slot / n;
  const int i = slot - frame * n;
  const int pixel = curve[i];
  if (owner[pixel] != i) return;
  const __global double* p = palette + 4 * (int)index[slot];
  // Truncate to int before the shift, as the CUDA original does.  p[] is a double
  // holding an exact 16-bit value so the truncation is exact; shifting in double
  // and converting afterwards would not be, and would round differently.
  const int r8 = ((int)p[0]) >> 8;
  const int g8 = ((int)p[1]) >> 8;
  const int b8 = ((int)p[2]) >> 8;
  const int yv = ((66 * r8 + 129 * g8 + 25 * b8 + 128) >> 8) + 16;
  const int uv = ((-38 * r8 - 74 * g8 + 112 * b8 + 128) >> 8) + 128;
  const int vv = ((112 * r8 - 94 * g8 - 18 * b8 + 128) >> 8) + 128;
  const size_t base = (ulong)frame * 3 * pixels_per_frame + pixel;
  yuv[base] = (uchar)(yv < 0 ? 0 : (yv > 255 ? 255 : yv));
  yuv[base + pixels_per_frame] = (uchar)(uv < 0 ? 0 : (uv > 255 ? 255 : uv));
  yuv[base + 2 * (ulong)pixels_per_frame] =
      (uchar)(vv < 0 ? 0 : (vv > 255 ? 255 : vv));
}
)CLC";

// ---------------------------------------------------------------------------
// Host mirrors of the kernel's structs.
//
// Memcpy'd to the device whole rather than field by field, so a layout
// disagreement would be silent memory corruption rather than a wrong answer.
//
// The meaningful assertion is against QNode, not a literal: this struct must be
// the same size as the tree node it mirrors, because that is what the upload
// assumes.  A hardcoded 128 is what the CUDA header's comment claims and it is
// WRONG -- int parent leaves 4 bytes of padding before the doubles, so the real
// size is 136 on an 8-byte-aligned compiler.  The comment was right about the
// useful payload and wrong about the struct, and only sizeof settles it.
// ---------------------------------------------------------------------------
struct ClNode {
  int child[16];
  int parent;
  double total_color[4];
  double quantize_error;
  std::uint64_t number_unique;
  std::uint32_t color_number;
  std::uint32_t id;
  std::uint32_t level;
};

struct ClSearch {
  std::uint32_t mask;
  std::uint64_t packed;
};

static_assert(sizeof(ClNode) == sizeof(QNode),
              "the OpenCL node mirror must be the same size as QNode; the upload "
              "copies raw bytes and a size mismatch is silent corruption");
static_assert(offsetof(ClNode, quantize_error) == offsetof(QNode, quantize_error) &&
                  offsetof(ClNode, color_number) == offsetof(QNode, color_number) &&
                  offsetof(ClNode, level) == offsetof(QNode, level),
              "QNode and its OpenCL mirror must agree field by field, not merely "
              "in total size");
static_assert(sizeof(ClSearch) == 16, "search entry must match the kernel's layout");

// BuildFlatSearch from include/rd_cuda_common.cuh, minus the cudaMalloc: it
// fills a vector and the caller uploads it.  The traversal is copied verbatim,
// including the shift-not-OR fold, because the ORDER of the candidate list is
// the answer (a leaf wins ties) and an order bug yields a search of the right
// shape and the wrong result -- which no image comparison would catch, because
// the result is still a valid 16-colour dithering.
bool BuildFlatSearchHost(const ColorTree& tree, int assoc,
                         std::vector<ClSearch>* out) {
  out->clear();
  const int colours = tree.color_count();
  if (colours <= 0 || colours > 16) return false;
  const std::vector<QNode>& nodes = tree.nodes();
  const int n = static_cast<int>(nodes.size());
  if (n <= 0) return false;
  const int children = assoc ? kRgbaChildren : kRgbChildren;
  std::vector<ClSearch> host(static_cast<std::size_t>(n));
  std::vector<std::pair<int, int>> stack;
  stack.reserve(64);
  stack.emplace_back(0, 0);
  bool ok = true;
  while (!stack.empty() && ok) {
    const int node = stack.back().first;
    const int next = stack.back().second;
    if (next < children) {
      stack.back().second = next + 1;
      const int child = nodes[static_cast<std::size_t>(node)].child[next];
      if (child >= 0) stack.emplace_back(child, 0);
      continue;
    }
    stack.pop_back();
    const QNode& q = nodes[static_cast<std::size_t>(node)];
    if (q.number_unique != 0) {
      unsigned slot = 0;
      while (host[static_cast<std::size_t>(node)].mask & (1u << slot)) ++slot;
      if (slot >= 16 || q.color_number > 15) { ok = false; break; }
      host[static_cast<std::size_t>(node)].mask |= 1u << slot;
      host[static_cast<std::size_t>(node)].packed |=
          static_cast<std::uint64_t>(q.color_number & 0xFu) << (4 * slot);
    }
    if (!stack.empty()) {
      const std::size_t p = static_cast<std::size_t>(stack.back().first);
      unsigned base = 0;
      for (unsigned t = host[p].mask; t != 0u; t >>= 1) base += t & 1u;
      unsigned child = host[static_cast<std::size_t>(node)].mask;
      unsigned len = 0;
      for (unsigned t = child; t != 0u; t >>= 1) len += t & 1u;
      if (base + len > 16) { ok = false; break; }
      host[p].mask |= child << base;
      host[p].packed |= host[static_cast<std::size_t>(node)].packed << (4 * base);
    }
  }
  if (!ok) return false;
  *out = std::move(host);
  return true;
}

std::string ClErrorName(cl_int e) {
  switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    default: break;
  }
  char b[32];
  std::snprintf(b, sizeof(b), "CL error %d", static_cast<int>(e));
  return b;
}

std::string g_build_log;
std::mutex g_log_mu;

// ---------------------------------------------------------------------------
// Device state, and why it is an array
// ---------------------------------------------------------------------------
//
// One slot per concurrent caller.  The single-image path only ever uses slot 0,
// so it pays for one queue and one program build; the video path has a worker
// pool, and two workers sharing one in-order queue would serialise on it, which
// is the whole reason the CUDA engine has `state_slot` in the first place.
//
// The split is: ONE context and ONE device, shared, and per-slot queues,
// programs and kernels.  The context is shared because creating several is
// wasteful and queues are context-scoped anyway; the program and kernels are
// per-slot because `clSetKernelArg` MUTATES the kernel object, so two threads
// setting arguments on one kernel is a data race, and no amount of queue
// separation fixes that.  Sharing the program would be the natural next saving
// and is exactly the wrong one.
//
// Every slot is built lazily but under g_mu, so context and queue creation never
// races with a launch.  Once built, a slot is used with no lock held at all: the
// calls left on the hot path are clCreateBuffer, clEnqueue*, clSetKernelArg,
// clFinish and clReleaseMemObject, all of which are thread-safe in OpenCL 1.2.
// The one API call that is not safe to race, clCreateCommandQueue, only ever
// happens under the mutex.
constexpr int kSlots = 2;

struct Ctx {
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel k_gather = nullptr;
  cl_kernel k_gather_u16 = nullptr;
  // Planar 4:4:4, the CUDA default's data path.  Null on a device whose compiler
  // rejects the source, which the engine reports rather than silently falling back
  // to rgba64le -- a silent fallback would produce a plausible picture from a
  // different decoder, and the difference would be blamed on the dither.
  cl_kernel k_gather_yuv444 = nullptr;
  cl_kernel k_scatter_yuv444 = nullptr;
  cl_kernel k_walk = nullptr;
  cl_kernel k_scatter = nullptr;
  cl_kernel k_scatter_u16 = nullptr;
  std::string detail;
  bool built = false;
};

std::mutex g_mu;
cl_context g_context = nullptr;
cl_device_id g_device = nullptr;
std::string g_device_name;
std::string g_detail;
bool g_probed = false;
Ctx g_slots[kSlots];

// Deliberately does not name the CL_PLATFORM_* codes.  They are OpenCL 1.0/1.1
// values that a 3.0-targeted header hides, and the SDK's cl_platform.h here
// confirms it: naming them would not compile.  Two of the few that survive are
// worth spelling out, because they distinguish "you have no OpenCL at all" from
// "something is wrong" -- which is the first thing anyone needs to know.
const char* PlatformErrorName(cl_int e) {
  switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    default: break;
  }
  return "CL_PLATFORM error (legacy code; see cl.h for the numeric value)";
}

// Picks the first device of the first platform that actually initialises.
// Deliberately not a vendor preference list: an ICD stack reports every vendor's
// device, and "first that works" is the only rule that cannot select a device the
// driver then refuses to compile for.
bool Probe() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_probed) return g_context != nullptr;
  g_probed = true;

  cl_uint np = 0;
  const cl_int pe = clGetPlatformIDs(0, nullptr, &np);
  if (pe != CL_SUCCESS || np == 0) {
    g_detail = std::string("no OpenCL platform (") + PlatformErrorName(pe) + ")";
    return false;
  }
  std::vector<cl_platform_id> plats(np);
  if (clGetPlatformIDs(np, plats.data(), nullptr) != CL_SUCCESS) {
    g_detail = "clGetPlatformIDs failed";
    return false;
  }
  for (cl_platform_id plat : plats) {
    cl_uint nd = 0;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS ||
        nd == 0) {
      continue;
    }
    std::vector<cl_device_id> devs(nd);
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr) !=
        CL_SUCCESS) {
      continue;
    }
    for (cl_device_id d : devs) {
      cl_int e = CL_SUCCESS;
      cl_context ctx = clCreateContext(nullptr, 1, &d, nullptr, nullptr, &e);
      if (e != CL_SUCCESS || ctx == nullptr) continue;
      char name[256] = {0};
      clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
      char vendor[256] = {0};
      clGetDeviceInfo(d, CL_DEVICE_VENDOR, sizeof(vendor) - 1, vendor, nullptr);
      char ver[64] = {0};
      clGetDeviceInfo(d, CL_DEVICE_VERSION, sizeof(ver) - 1, ver, nullptr);
      // The device is kept alive for the process by the context that owns it, so
      // there is deliberately no clReleaseDeviceID here: releasing it would leave
      // g_device dangling for every later slot build.
      g_context = ctx;
      g_device = d;
      g_device_name = std::string(vendor) + " " + name + " (OpenCL " + ver + ")";
      return true;
    }
  }
  g_detail = "OpenCL platforms present but no device could be initialised";
  return false;
}

bool BuildSlotLocked(int n) {
  Ctx& s = g_slots[n];
  if (s.built) return true;
  if (g_context == nullptr) return false;

  cl_int e = CL_SUCCESS;
  s.queue = clCreateCommandQueue(g_context, g_device, 0, &e);
  if (e != CL_SUCCESS || s.queue == nullptr) {
    g_detail = std::string("clCreateCommandQueue: ") + ClErrorName(e);
    s.detail = g_detail;
    return false;
  }

  // Two sources, not one.  kSourceYuv holds the planar 4:4:4 gather and scatter,
  // kept separate because MSVC refuses a string literal over 16380 bytes and
  // kSource is already most of that.  The driver concatenates them exactly as given,
  // so the program text is identical to a single literal -- and unlike splicing the
  // literal, this does not impose a character budget on every kernel added later.
  const char* srcs[2] = {kSource, kSourceYuv};
  // size_t, not cl_int: the 1.2 headers take `const size_t*` for the lengths even
  // though the 1.0 prototype said cl_int, and a cl_int array here does not convert.
  const std::size_t lens[2] = {std::strlen(kSource), std::strlen(kSourceYuv)};
  cl_program prog = clCreateProgramWithSource(g_context, 2, srcs, lens, &e);
  if (e != CL_SUCCESS || prog == nullptr) {
    g_detail = std::string("clCreateProgramWithSource: ") + ClErrorName(e);
    s.detail = g_detail;
    return false;
  }
  // BIT-EXACTNESS lives or dies on this line.
  //
  // -cl-fast-relaxed-math is the one to keep away: it permits reassociation and
  // contraction, and the walk would still produce a plausible 16-colour image
  // that is wrong.  -cl-std pins the dialect so the default cannot drift under
  // us.
  //
  // -cl-mad-disable is the more explicit guard against a*b+c fusion, and it was
  // tried here -- and is deliberately NOT used, because this driver rejects it
  // ("Don't understand command line argument"), which would abort every build
  // rather than protect one.  So contraction is left to the default.  That is a
  // real residual risk, stated rather than hidden: a driver that contracts
  // double arithmetic by default would differ from CUDA on the near-tie pixels
  // and nowhere else.  It is the first thing to try on a machine that reports a
  // small scattered difference, and the build log is captured precisely so that
  // a driver's opinion on this is visible.
  //
  // The fix for a driver that does contract is not to re-enable fast math but to
  // force the two-rounding form explicitly, the way CUDA does with
  // __dadd_rn(__dmul_rn(x, y)).
  const char* opts = "-cl-std=CL1.2";
  e = clBuildProgram(prog, 1, &g_device, opts, nullptr, nullptr);

  // The build log is captured unconditionally.  A driver that rejects the source
  // and a driver that miscompiles it are indistinguishable from outside without
  // it, and "it built and produced different pixels" is the failure this whole
  // port exists to be able to rule out.
  std::size_t log_len = 0;
  clGetProgramBuildInfo(prog, g_device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                        &log_len);
  if (log_len > 1) {
    std::string log(log_len, '\0');
    clGetProgramBuildInfo(prog, g_device, CL_PROGRAM_BUILD_LOG, log_len,
                          log.data(), nullptr);
    while (!log.empty() && (log.back() == '\0' || log.back() == '\n')) log.pop_back();
    std::lock_guard<std::mutex> l2(g_log_mu);
    g_build_log = log;
  }
  if (e != CL_SUCCESS) {
    g_detail = std::string("clBuildProgram: ") + ClErrorName(e);
    s.detail = g_detail;
    clReleaseProgram(prog);
    return false;
  }

  struct { const char* name; cl_kernel* out; } kernels[7] = {
      {"gather", &s.k_gather}, {"gather_u16", &s.k_gather_u16},
      {"walk", &s.k_walk},
      {"scatter", &s.k_scatter}, {"scatter_u16", &s.k_scatter_u16},
      // Planar 4:4:4, 3 bytes per pixel both ways.  These are what let the engine
      // use CUDA's default data path instead of being confined to rgba64le.
      {"gather_yuv444", &s.k_gather_yuv444},
      {"scatter_yuv444", &s.k_scatter_yuv444},
  };
  for (auto& kn : kernels) {
    cl_kernel k = clCreateKernel(prog, kn.name, &e);
    if (e != CL_SUCCESS || k == nullptr) {
      g_detail = std::string("clCreateKernel(") + kn.name + "): " + ClErrorName(e);
      s.detail = g_detail;
      clReleaseProgram(prog);
      return false;
    }
    *kn.out = k;
  }
  s.program = prog;
  s.built = true;
  return true;
}

// Returns the slot to use, building it on first use.  Clamps rather than rejects:
// a caller asking for a slot past the end is a scheduling detail, and running it
// on the last slot is slow but correct.  Returning nullptr means the device is
// unusable, and the caller should report `g_detail`.
Ctx* SlotFor(int state_slot) {
  if (!Probe()) return nullptr;
  std::lock_guard<std::mutex> lock(g_mu);
  int n = state_slot;
  if (n < 0) n = 0;
  if (n >= kSlots) n = kSlots - 1;
  if (!BuildSlotLocked(n)) return nullptr;
  return &g_slots[n];
}

}  // namespace

bool OpenCLAvailable(std::string* device_name, std::string* detail) {
  const bool ok = SlotFor(0) != nullptr;
  std::lock_guard<std::mutex> lock(g_mu);
  if (!ok) {
    if (detail != nullptr) *detail = g_detail;
    return false;
  }
  if (device_name != nullptr) *device_name = g_device_name;
  if (detail != nullptr) *detail = std::string();
  return true;
}

int OpenCLStateCount() { return kSlots; }

const std::string& OpenCLBuildLog() {
  std::lock_guard<std::mutex> lock(g_log_mu);
  return g_build_log;
}

std::string RiemersmaBlocksOpencl(const Palette& palette,
                                  const DitherParams& params,
                                  const ColorTree& tree, std::size_t width,
                                  std::size_t height, RgbaF* batch,
                                  const BlockOptions& options,
                                  std::string* device_name,
                                  std::uint16_t* out_u16,
                                  int state_slot,
                                  const std::uint16_t* in_u16) {
  // Take the slot up front and hold no lock for the rest of the function.  That
  // is the entire point of the slot array: two video workers on different slots
  // run concurrently, and nothing below is shared mutable state except the
  // context, which is read-only once Probe() has returned.
  Ctx* ctx = SlotFor(state_slot);
  if (ctx == nullptr) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (device_name != nullptr) *device_name = std::string();
    return "opencl: " + g_detail;
  }
  if (device_name != nullptr) {
    std::lock_guard<std::mutex> lock(g_mu);
    *device_name = g_device_name;
  }
  const int frames = std::max(1, options.frames);
  const std::size_t npix = width * height;
  if (npix == 0) return "opencl: empty frame";
  const int block_size = std::max(kErrorQueueLength, options.block);
  const int assoc = palette.associate_alpha ? 1 : 0;

  // Planar 4:4:4 is now ported, on both sides.  What is still refused is planar
  // 4:2:0 and the yuv444-prepass, and the refusal is for the same reason it
  // originally applied to all of them: each is a bit-exact transliteration of a
  // specific libswscale routine, and quietly reading rgba64le instead would
  // produce a perfectly plausible picture that came from a different decoder --
  // with the difference attributed to the dither.  Refusing names the difference.
  //
  // 4:4:4 is ported because it is CUDA's DEFAULT path, and being confined to
  // rgba64le is what made this engine look 2.2x slower than CUDA: 8 bytes per
  // pixel against 3, for a data-path difference rather than an engine one.  On the
  // same path the two engines are within 8%.  See docs/OPENCL.md.
  if (options.upload_u16 && options.in_mode != BlockOptions::InMode::Interleaved16 &&
      options.in_mode != BlockOptions::InMode::PlanarYuv444) {
    return "opencl: planar 4:2:0 input and the yuv444-prepass are not ported; this "
           "engine supports --input-mode rgba64le and yuv444.  Use --engine blocks "
           "or --engine cpu for 4:2:0.";
  }
  if (options.in_channels != 3 && options.in_channels != 4) {
    return "opencl: in_channels must be 3 (rgb48le) or 4 (rgba64le), not " +
           std::to_string(options.in_channels);
  }

  // ALPHA INPUT IS ACCEPTED, and there is deliberately no guard on it.  A guard
  // WAS here, refusing alpha images on the stated ground that this engine's
  // output on them was nondeterministic.  That ground was wrong, and this note
  // is here so the guard is not reinstated on the strength of the old reasoning.
  //
  // The measurements that overturned it came from dumping the device and host
  // buffers directly (RD_OCL_DUMP) instead of inspecting the rendered file, over
  // 16 runs of a 96x64 greyscale+alpha image:
  //
  //     buffer                                    distinct hashes / 16 runs
  //     cx     gather output                                1
  //     idx    the walk's chosen palette indices            1
  //     owner  the scatter's owner table                    1
  //     pal    the palette as the device sees it            1
  //     pix    device pixels after the scatter              1
  //     host   the readback into `batch` -- what the writer
  //            actually receives                           1  (byte-identical to pix)
  //     png    the file, decoded back to RGBA               3
  //
  // The engine is deterministic, and the host buffer the image writer receives
  // is byte-identical on every run.  The variation was always downstream of the
  // engine, in the PNG write; it was visible only because the instrument decoded
  // the PNG, so the instrument was the thing that varied.  Everything upstream of
  // that was exonerated the hard way: the poison fill (RD_OCL_POISON, which
  // writes 0xA5 over the two buffers created without initial contents) left
  // opaque output bit-identical and did not reduce the alpha run-to-run spread
  // at all, so nothing was being read before it was written -- the wrong values
  // are computed, not remembered from uninitialised memory.
  //
  // The engine is also CORRECT on alpha, not merely stable: against the CUDA
  // blocks engine on that image, 0 of 24576 decoded bytes differ.
  //
  // The defect that did exist downstream of here -- the image writer leaving
  // IM's fifth pixel channel uninitialised, which changed the PNG *encoding* from
  // stale heap memory -- is fixed, in ZeroQueuedRow() in src/rd_im.cpp, and is
  // guarded by tools/probe-determinism.ps1.  Nothing about it was ever this
  // engine's, and nothing about it made the dither wrong.

  // The curve and its owner map, from the CPU engine's cache, so the walk visits
  // in the identical order and the scatter agrees on who owns a pixel.
  const std::vector<int>* curve = BlockCurveOrder(width, height);
  const std::vector<int>* owner = BlockCurveOwner(width, height);
  if (curve == nullptr || owner == nullptr) return "opencl: no curve for geometry";
  const int n = static_cast<int>(curve->size());
  if (n <= 0) return "opencl: empty curve";
  const int nblocks = (n + block_size - 1) / block_size;

  std::vector<ClSearch> hsearch;
  if (!BuildFlatSearchHost(tree, assoc, &hsearch)) {
    // Not a crash but a refusal, and deliberately loud.  On the CUDA side a
    // palette too large for the 4-bit packing falls back to the recursive
    // ClosestColor, which this port does not carry; silently doing something
    // else would redefine what a given --engine means.
    return "opencl: this engine implements the flat search only, so --colors must "
           "be 16 or fewer";
  }
  std::vector<ClNode> hnodes(tree.nodes().size());
  for (std::size_t i = 0; i < tree.nodes().size(); ++i) {
    const QNode& q = tree.nodes()[i];
    std::memcpy(hnodes[i].child, q.child, sizeof(q.child));
    hnodes[i].parent = q.parent;
    for (int k = 0; k < 4; ++k) hnodes[i].total_color[k] = q.total_color[k];
    hnodes[i].quantize_error = q.quantize_error;
    hnodes[i].number_unique = q.number_unique;
    hnodes[i].color_number = q.color_number;
    hnodes[i].id = q.id;
    hnodes[i].level = q.level;
  }
  std::vector<double> hpal(static_cast<std::size_t>(palette.count) * 4);
  for (int i = 0; i < palette.count; ++i) {
    hpal[4 * i + 0] = palette.entries[i].r;
    hpal[4 * i + 1] = palette.entries[i].g;
    hpal[4 * i + 2] = palette.entries[i].b;
    hpal[4 * i + 3] = palette.entries[i].a;
  }
  double hweights[kErrorQueueLength];
  // The project's own builder, from rd_types.h -- deliberately NOT a local
  // reimplementation.  This port started with one that computed
  // 16^(-i), which is the obvious reading of "exponential decay" and is wrong:
  // ImageMagick's GetQCubeInfo() builds the weights by repeated multiplication
  // from an exp/log constant, giving 16^(-i/15).  So the queue decays over 15
  // steps, not one per step, and weights[1] is 0.911 rather than 0.0625.
  //
  // It was caught only by the bit-exactness comparison against the CUDA engine,
  // and it is worth recording *why* that is not a silly thing to have got wrong:
  // the result was still a plausible, still-16-colour, still-dithered image that
  // differed from the reference on 10% of pixels.  Nothing about looking at it
  // would have found it.
  build_error_weights(hweights);

  // No lock is held from here to the end of the function.  That is deliberate
  // and is the reason the slot array exists: everything below -- buffer creation,
  // argument setting, enqueue, finish, readback -- is thread-safe in OpenCL 1.2,
  // so two workers on different slots run genuinely concurrently.  The calls that
  // are NOT safe to race all happen in SlotFor(), under the mutex.
  cl_int e = CL_SUCCESS;
  const std::size_t pix_bytes = npix * static_cast<std::size_t>(frames) * 4 * sizeof(float);
  const std::size_t cx_bytes = static_cast<std::size_t>(n) * frames * 4 * sizeof(float);
  const std::size_t idx_bytes = static_cast<std::size_t>(n) * frames;
  // rgba64le is 8 bytes per pixel against float4's 16, so the batch and the
  // result both cross the bus at half the width.
  const std::size_t u16_bytes = npix * static_cast<std::size_t>(frames) * 4 * sizeof(std::uint16_t);
  // Planar 4:4:4 is 3 bytes per pixel -- a third of rgba64le's 8 -- and that byte
  // count is the entire reason the engine was 2.2x slower on video: not the walk,
  // the pipe.  Three planes per frame, so a batch of F frames is 3 * npix * F
  // bytes, with the SAME stride for all three planes.  That stride is not a detail;
  // using npix*F for the chroma planes reads frame f-1's chroma as frame f's luma.
  const std::size_t yuv_bytes = npix * static_cast<std::size_t>(frames) * 3;

  // The video pipe asks for the uint16 data path in both directions at once, and
  // the single-image path asks for neither.  Each is honoured only if the caller
  // actually supplied the buffer, so a caller that sets the flag without the
  // pointer gets the float path rather than a null dereference.
  const bool want_in_u16 = options.upload_u16 && in_u16 != nullptr;
  const bool want_u16 = options.emit_u16 && out_u16 != nullptr;
  // Planar 4:4:4 in and out.  Both are 3 bytes per pixel, and both use the same
  // layout, so the scatter's output can be pre-filled from the input with a plain
  // device-to-device copy -- which is how the pixel the curve never visits keeps
  // its source value on this path, exactly as the float path gets it for free.
  const bool in_yuv444 =
      want_in_u16 && options.in_mode == BlockOptions::InMode::PlanarYuv444;
  const bool out_yuv444 = want_u16 && options.emit_yuv444;
  if (in_yuv444 && ctx->k_gather_yuv444 == nullptr) {
    return "opencl: this device did not build the planar 4:4:4 gather kernel";
  }
  if (out_yuv444 && ctx->k_scatter_yuv444 == nullptr) {
    return "opencl: this device did not build the planar 4:4:4 scatter kernel";
  }

  cl_mem b_pix = nullptr;    // float4, gather source and/or scatter destination
  cl_mem b_in16 = nullptr;   // rgba64le in
  cl_mem b_out16 = nullptr;  // rgba64le out
  cl_mem b_in_yuv = nullptr;  // planar 4:4:4 in
  cl_mem b_out_yuv = nullptr;  // planar 4:4:4 out

  if (in_yuv444) {
    b_in_yuv = clCreateBuffer(g_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              yuv_bytes, const_cast<std::uint16_t*>(in_u16), &e);
    if (e != CL_SUCCESS || b_in_yuv == nullptr) return "clCreateBuffer(in yuv444)";
  } else if (want_in_u16) {
    // READ_WRITE, not READ_ONLY: the source is copied into the output buffer
    // below, so that the pixel the curve never visits keeps its SOURCE value
    // rather than uninitialised memory.  See the fill below.
    b_in16 = clCreateBuffer(g_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            u16_bytes, const_cast<std::uint16_t*>(in_u16), &e);
  } else {
    b_pix = clCreateBuffer(g_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           pix_bytes, batch, &e);
  }
  if (out_yuv444) {
    b_out_yuv = clCreateBuffer(g_context, CL_MEM_READ_WRITE, yuv_bytes, nullptr, &e);
    if (e != CL_SUCCESS || b_out_yuv == nullptr) return "clCreateBuffer(out yuv444)";
  } else if (want_u16) {
    b_out16 = clCreateBuffer(g_context, CL_MEM_READ_WRITE, u16_bytes, nullptr, &e);
  } else if (b_pix == nullptr) {
    // Nothing to scatter into except the float buffer, so one is needed either
    // way: as the gather's source when uploading uint16, as the scatter's
    // destination otherwise.
    b_pix = clCreateBuffer(g_context, CL_MEM_READ_WRITE, pix_bytes, nullptr, &e);
  }

  // b_cx and b_idx are created with no initial contents on purpose: both are
  // written in full by the gather and the walk before anything reads them.  That
  // was long asserted by inspection; it is now measured, via RD_OCL_DUMP.  A former
  // RD_OCL_POISON facility filled them with 0xA5 to prove it at run time.  It
  // answered -- nothing is read before it is written, so the wrong values an
  // earlier investigation chased were computed, not remembered -- and it has been
  // removed rather than left as a live branch on buffer creation, re-asking a
  // settled question on every launch.
  cl_mem b_cx = clCreateBuffer(g_context, CL_MEM_READ_WRITE, cx_bytes, nullptr, &e);
  cl_mem b_idx = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, idx_bytes, nullptr, &e);
  cl_mem b_curve = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  curve->size() * sizeof(int), const_cast<int*>(curve->data()), &e);
  cl_mem b_owner = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  owner->size() * sizeof(int), const_cast<int*>(owner->data()), &e);
  cl_mem b_nodes = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  hnodes.size() * sizeof(ClNode), hnodes.data(), &e);
  cl_mem b_search = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   hsearch.size() * sizeof(ClSearch), hsearch.data(), &e);
  cl_mem b_pal = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                hpal.size() * sizeof(double), hpal.data(), &e);
  cl_mem b_w = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(hweights), hweights, &e);

  cl_mem all[13] = {b_pix, b_in16, b_out16, b_in_yuv, b_out_yuv,
                    b_cx, b_idx, b_curve, b_owner,
                    b_nodes, b_search, b_pal, b_w};
  // Releases every buffer, then says what went wrong.  Split from the reporting so
  // it can run on the success path too, and so the pipeline lambda below can report
  // a launch failure without releasing anything out from under its caller.
  auto release_all = [&]() {
    for (cl_mem b : all) {
      if (b != nullptr) clReleaseMemObject(b);
    }
  };
  auto fail = [&](const char* what) {
    release_all();
    return std::string("opencl: ") + what + " failed (" + ClErrorName(e) + ")";
  };
  // b_in16 and b_out16 are legitimately absent -- they exist only on the paths
  // that asked for them -- so only the buffers that were supposed to be created
  // are checked for null.
  if (b_pix == nullptr && !want_in_u16 && !in_yuv444) return fail("clCreateBuffer");
  if (b_in16 == nullptr && want_in_u16 && !in_yuv444) return fail("clCreateBuffer(in16)");
  if (b_out16 == nullptr && want_u16 && !out_yuv444) return fail("clCreateBuffer(out16)");
  for (cl_mem b : {b_cx, b_idx, b_curve, b_owner, b_nodes, b_search, b_pal, b_w}) {
    if (b == nullptr) return fail("clCreateBuffer");
  }

  const std::size_t local = 64;
  auto global_for = [&](std::size_t items) {
    return ((items + local - 1) / local) * local;
  };

  const int np = static_cast<int>(npix);
  // 3 for rgb48le, 4 for rgba64le.  Only 3 when the source genuinely has no
  // alpha, because the gather then synthesises the 65535 that rgba64le would
  // have delivered -- so the dither's input is the same either way.
  const int in_channels = (options.in_channels == 3) ? 3 : 4;

  // One complete pass -- gather, walk, scatter -- with an explicit clFinish between
  // each stage.
  //
  // The barriers are deliberate and blunt.  An in-order command queue is the
  // OpenCL default and this code was originally written assuming that, which is how
  // a real bug got in: output was nondeterministic on alpha images, 42 of 6144
  // pixels differing between runs with all four channels changing together -- the
  // signature of a different palette entry, which is what a launch overtaking its
  // own producer produces.  Three barriers cost nothing measurable next to a frame
  // and remove the dependence on a driver default.
  //
  // (That investigation eventually cleared the device entirely; the fault was
  // uninitialised channels in the image writer.  The barriers stayed because they
  // are correct and because the argument they rest on -- never rely on a driver
  // default for something bit-exactness depends on -- is worth keeping.)
  //
  // A lambda rather than straight-line code because RD_OCL_CHECK_U16 runs the whole
  // pipeline a second time through the uint16 kernels and diffs the two results.
  // Duplicating three launches to allow that would guarantee the copies drift, which
  // is the same mistake the CUDA engine's paired gather kernels exist to avoid.
  auto run_pipeline = [&](cl_kernel k_gather, cl_mem gather_src, bool src_is_u16,
                          cl_kernel k_scatter, cl_mem scatter_dst) -> std::string {
    const std::size_t g = global_for(idx_bytes);

    clSetKernelArg(k_gather, 0, sizeof(cl_mem), &gather_src);
    clSetKernelArg(k_gather, 1, sizeof(cl_mem), &b_curve);
    clSetKernelArg(k_gather, 2, sizeof(int), &n);
    clSetKernelArg(k_gather, 3, sizeof(int), &frames);
    clSetKernelArg(k_gather, 4, sizeof(int), &np);
    if (k_gather == ctx->k_gather_yuv444) {
      // gather_yuv444 has no in_channels: planar 4:4:4 is always three planes and
      // alpha never exists in it, so the 65535 is written unconditionally below.
      clSetKernelArg(k_gather, 5, sizeof(cl_mem), &b_cx);
    } else if (src_is_u16) {
      clSetKernelArg(k_gather, 5, sizeof(int), &in_channels);
      clSetKernelArg(k_gather, 6, sizeof(cl_mem), &b_cx);
    } else {
      clSetKernelArg(k_gather, 5, sizeof(cl_mem), &b_cx);
    }
    e = clEnqueueNDRangeKernel(ctx->queue, k_gather, 1, nullptr, &g, &local,
                               0, nullptr, nullptr);
    if (e != CL_SUCCESS) return "gather launch";
    e = clFinish(ctx->queue);
    if (e != CL_SUCCESS) return "clFinish after gather";

    double diffusion = params.diffusion;
    clSetKernelArg(ctx->k_walk, 0, sizeof(cl_mem), &b_cx);
    clSetKernelArg(ctx->k_walk, 1, sizeof(cl_mem), &b_pal);
    clSetKernelArg(ctx->k_walk, 2, sizeof(cl_mem), &b_w);
    clSetKernelArg(ctx->k_walk, 3, sizeof(double), &diffusion);
    clSetKernelArg(ctx->k_walk, 4, sizeof(int), &assoc);
    clSetKernelArg(ctx->k_walk, 5, sizeof(cl_mem), &b_nodes);
    clSetKernelArg(ctx->k_walk, 6, sizeof(cl_mem), &b_search);
    clSetKernelArg(ctx->k_walk, 7, sizeof(int), &n);
    clSetKernelArg(ctx->k_walk, 8, sizeof(int), &nblocks);
    clSetKernelArg(ctx->k_walk, 9, sizeof(int), &frames);
    clSetKernelArg(ctx->k_walk, 10, sizeof(int), &block_size);
    clSetKernelArg(ctx->k_walk, 11, sizeof(cl_mem), &b_idx);
    const std::size_t gw = global_for(static_cast<std::size_t>(nblocks) * frames);
    e = clEnqueueNDRangeKernel(ctx->queue, ctx->k_walk, 1, nullptr, &gw, &local,
                               0, nullptr, nullptr);
    if (e != CL_SUCCESS) return "walk launch";
    e = clFinish(ctx->queue);
    if (e != CL_SUCCESS) return "clFinish after walk";

    clSetKernelArg(k_scatter, 0, sizeof(cl_mem), &b_curve);
    clSetKernelArg(k_scatter, 1, sizeof(cl_mem), &b_owner);
    clSetKernelArg(k_scatter, 2, sizeof(cl_mem), &b_idx);
    clSetKernelArg(k_scatter, 3, sizeof(cl_mem), &b_pal);
    clSetKernelArg(k_scatter, 4, sizeof(int), &n);
    clSetKernelArg(k_scatter, 5, sizeof(int), &frames);
    clSetKernelArg(k_scatter, 6, sizeof(int), &np);
    clSetKernelArg(k_scatter, 7, sizeof(int), &assoc);
    clSetKernelArg(k_scatter, 8, sizeof(cl_mem), &scatter_dst);
    e = clEnqueueNDRangeKernel(ctx->queue, k_scatter, 1, nullptr, &g, &local,
                               0, nullptr, nullptr);
    if (e != CL_SUCCESS) return "scatter launch";
    e = clFinish(ctx->queue);
    if (e != CL_SUCCESS) return "clFinish after scatter";
    return std::string();
  };

  // RD_OCL_CHECK_U16=1 verifies the uint16 data path against the float path on
  // every run, by running the pipeline twice over the same input and diffing.
  //
  // Why this exists rather than a test: the uint16 path is what the video pipe
  // uses, and video's acceptance bar is "almost frame exact", which is far too
  // loose to catch an off-by-one-in-the-last-bit conversion.  A one-character
  // mistake in scatter_u16 -- writing (ushort)e[0] instead of (ushort)(float)e[0]
  // -- would produce a picture indistinguishable from correct and would only
  // ever show up as compression noise in an encoded file.
  //
  // The float path is the reference because it is already verified bit-identical
  // to the CUDA engine on all 54 cells of tools/probe-opencl-exact.ps1.  So this
  // needs no new fixture and no new reference: the strong test is already green,
  // and this just checks the new kernels against it.
  //
  // THE INPUT IS QUANTISED FIRST, and that is the whole subtlety.  An image
  // loaded from PNG is HDRI float, and its samples are generally NOT integers:
  // 8-bit sRGB 128 arrives as 32896.06, not 32896.  rgba64le cannot represent
  // that, so feeding the image's own pixels to the uint16 gather and diffing
  // against the float result compares a lossy path against a lossless one and
  // reports thousands of differences that mean nothing.  Two earlier versions of
  // this check did exactly that, and both looked like a broken kernel.
  //
  // So: round-trip the input through rgba64le ONCE, write the widened values
  // back into `batch`, and let both passes see that.  The video pipe's input is
  // rgba64le to begin with, so this is not a contrivance -- it is the only input
  // the uint16 path ever sees in production.
  const bool check_u16 = !want_in_u16 && !want_u16 &&
                         std::getenv("RD_OCL_CHECK_U16") != nullptr;
  std::vector<std::uint16_t> check_src16;
  if (check_u16) {
    const std::size_t pixels = npix * static_cast<std::size_t>(frames);
    check_src16.resize(u16_bytes / sizeof(std::uint16_t));
    for (std::size_t i = 0; i < pixels; ++i) {
      const RgbaF& px = batch[i];
      const std::uint16_t r = static_cast<std::uint16_t>(px.r);
      const std::uint16_t g = static_cast<std::uint16_t>(px.g);
      const std::uint16_t b = static_cast<std::uint16_t>(px.b);
      const std::uint16_t a = static_cast<std::uint16_t>(px.a);
      check_src16[4 * i + 0] = r;
      check_src16[4 * i + 1] = g;
      check_src16[4 * i + 2] = b;
      check_src16[4 * i + 3] = a;
      // Widen back, exactly as gather_u16 does, so the float pass sees the same
      // numbers the uint16 pass will.
      batch[i].r = static_cast<float>(r);
      batch[i].g = static_cast<float>(g);
      batch[i].b = static_cast<float>(b);
      batch[i].a = static_cast<float>(a);
    }
  }

  if (out_yuv444) {
    // Same reason as below, and easier here: a planar 4:4:4 source and a planar
    // 4:4:4 output have the SAME layout -- Y, then U, then V, each npix bytes per
    // frame -- so the unvisited pixel can keep the source's own three bytes with a
    // straight device-to-device copy, no conversion and no rounding.  That is also
    // what the CUDA fill does on this path, so the two engines agree on it.
    if (in_yuv444) {
      e = clEnqueueCopyBuffer(ctx->queue, b_in_yuv, b_out_yuv, 0, 0, yuv_bytes, 0,
                              nullptr, nullptr);
    } else {
      // rgba64le in, planar 4:4:4 out: there is no planar source to copy, so fill
      // with neutral black (Y=0, and 128 for the chroma centre) rather than leave
      // the buffer undefined.
      //
      // clEnqueueFillBuffer requires pattern_size to be 1, 2, 4, 8, ...  Passing 3 --
      // one byte per plane -- is CL_INVALID_VALUE on any conformant runtime, so this
      // call ALWAYS failed.  `--engine opencl --input-mode rgba64` with the default
      // planar output exited 1 and left a 565-byte container ffprobe calls malformed.
      // docs\OPENCL.md records that this combination needs RD_YUV444_OUT=0, but a
      // documented caveat is not a check, and what the user saw was an internal OpenCL
      // error string rather than anything they could act on.
      //
      // Three fills of one byte each instead.  The layout is Y, then U, then V, each
      // yuv_bytes / 3, so the intent -- luma 0, chroma at the 128 centre -- is kept
      // exactly and no byte-pattern trick is needed.
      const std::size_t plane3 = yuv_bytes / 3;
      const cl_uchar luma_fill = 0;
      const cl_uchar chroma_fill = 128;
      e = clEnqueueFillBuffer(ctx->queue, b_out_yuv, &luma_fill, 1, 0, plane3, 0,
                              nullptr, nullptr);
      if (e == CL_SUCCESS)
        e = clEnqueueFillBuffer(ctx->queue, b_out_yuv, &chroma_fill, 1, plane3,
                                plane3, 0, nullptr, nullptr);
      if (e == CL_SUCCESS)
        e = clEnqueueFillBuffer(ctx->queue, b_out_yuv, &chroma_fill, 1, 2 * plane3,
                                plane3, 0, nullptr, nullptr);
    }
    if (e != CL_SUCCESS) return fail("fill(yuv444 out)");
  } else if (want_u16) {
    // Riemersma() visits 4^L - 1 cells and the trailing ForgetGravity visit lands
    // on the first one again, so the grid's LAST cell is never dithered at all --
    // which is why a 1920x1080 frame at level 11 leaves exactly one pixel at its
    // source value, and why ImageMagick itself does.  The float path gets this for
    // free: b_pix is COPY_HOST_PTR from the caller's frame, so the unvisited pixel
    // still holds its source value.  A freshly allocated uint16 buffer holds
    // whatever the driver left there, and that value goes straight into the
    // encoder.
    //
    // It is one pixel in 2,073,600, so it is invisible -- which is exactly why it
    // survived.  What it is not is harmless: it is uninitialised memory, so the
    // output is not reproducible, and the determinism test added for the writer
    // bug would flag it the moment it was pointed at video.
    //
    // NOTE: the CUDA engine had the same exposure -- d_u16_buf is a bare
    // cudaMalloc and its scatter writes only owned pixels.  Fixed there since, by
    // BlkFillUnvisitedKernel in rd_blocks_cuda.cu, which writes the same pixel from
    // the source with one thread per frame.  It copies the source Y/U/V verbatim in
    // the planar-4:4:4 case rather than round-tripping them through RGB, so it
    // matches what this pre-fill produces on the same input.
    if (want_in_u16) {
      // The production combination: the source is the caller's rgba64le, so copy
      // it and let the scatter overwrite the visited pixels.  This makes the
      // uint16 path's unvisited pixel agree with the float path's exactly.
      e = clEnqueueCopyBuffer(ctx->queue, b_in16, b_out16, 0, 0, u16_bytes, 0,
                              nullptr, nullptr);
    } else {
      // uint16 out from a float4 source: there is no rgba64le source to copy, so
      // define the unvisited pixel as opaque black rather than as garbage.  Not
      // identical to the float path here -- the float path would keep the source
      // value -- but deterministic, which is the property that matters, and this
      // combination is not what the video pipe uses.
      const cl_uint zero = 0;
      e = clEnqueueFillBuffer(ctx->queue, b_out16, &zero, sizeof(zero), 0,
                              u16_bytes, 0, nullptr, nullptr);
    }
    if (e != CL_SUCCESS) return fail("fill(u16 out)");
  }

  // Kernel and buffer selection.  The planar-4:4:4 pair is the CUDA default's path
  // and is independent in each direction: planar in with rgba64le out, and vice
  // versa, are both legal, because the video pipe chooses them separately.
  const cl_kernel k_g = in_yuv444 ? ctx->k_gather_yuv444
                                  : (want_in_u16 ? ctx->k_gather_u16 : ctx->k_gather);
  const cl_mem b_g = in_yuv444 ? b_in_yuv : (want_in_u16 ? b_in16 : b_pix);
  const cl_kernel k_s = out_yuv444 ? ctx->k_scatter_yuv444
                                    : (want_u16 ? ctx->k_scatter_u16 : ctx->k_scatter);
  const cl_mem b_s = out_yuv444 ? b_out_yuv : (want_u16 ? b_out16 : b_pix);

  std::string msg = run_pipeline(k_g, b_g, want_in_u16 && !in_yuv444, k_s, b_s);
  if (!msg.empty()) return fail(msg.c_str());

  if (check_u16) {
    // Read pass one's result back FIRST.  Without this, `batch` still holds the
    // quantised input -- the float scatter wrote into the device buffer b_pix,
    // and the readback below has not run yet -- so the comparison would be
    // pass two's output against the input, which differs on every pixel the dither
    // actually changed.  That is exactly what it did for a while, and it reported
    // 18432 differences on a case whose kernels are correct.
    e = clEnqueueReadBuffer(ctx->queue, b_pix, CL_TRUE, 0, pix_bytes, batch, 0,
                            nullptr, nullptr);
    if (e != CL_SUCCESS) return fail("readback(check float)");

    std::vector<std::uint16_t> got16(check_src16.size());
    std::vector<std::uint16_t> want16(check_src16.size());
    const std::size_t pixels = npix * static_cast<std::size_t>(frames);
    // Pass one's result, in the form pass two's output is in.
    for (std::size_t i = 0; i < pixels; ++i) {
      const RgbaF& px = batch[i];
      want16[4 * i + 0] = static_cast<std::uint16_t>(px.r);
      want16[4 * i + 1] = static_cast<std::uint16_t>(px.g);
      want16[4 * i + 2] = static_cast<std::uint16_t>(px.b);
      want16[4 * i + 3] = static_cast<std::uint16_t>(px.a);
    }
    // Upload the quantised input and run the uint16 gather and scatter for real.
    // b_cx and b_idx are reused, which is safe precisely because the first pass
    // has finished and clFinish has returned.
    cl_mem chk_in = clCreateBuffer(g_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   u16_bytes, check_src16.data(), &e);
    if (e != CL_SUCCESS || chk_in == nullptr) {
      return fail("clCreateBuffer(check in16)");
    }
    cl_mem chk_out = clCreateBuffer(g_context, CL_MEM_READ_WRITE, u16_bytes, nullptr, &e);
    if (e != CL_SUCCESS || chk_out == nullptr) {
      clReleaseMemObject(chk_in);
      return fail("clCreateBuffer(check out16)");
    }
    // Same pre-fill the production path does, so the check validates the shipped
    // behaviour rather than a stricter one.  Without this the single never-visited
    // pixel would report as a difference and mask a real one.
    e = clEnqueueCopyBuffer(ctx->queue, chk_in, chk_out, 0, 0, u16_bytes, 0,
                            nullptr, nullptr);
    if (e != CL_SUCCESS) {
      clReleaseMemObject(chk_in);
      clReleaseMemObject(chk_out);
      return fail("copy(check u16)");
    }
    msg = run_pipeline(ctx->k_gather_u16, chk_in, true, ctx->k_scatter_u16, chk_out);
    if (msg.empty()) {
      e = clEnqueueReadBuffer(ctx->queue, chk_out, CL_TRUE, 0, u16_bytes,
                              got16.data(), 0, nullptr, nullptr);
      if (e != CL_SUCCESS) msg = "readback(check u16)";
    }
    clReleaseMemObject(chk_in);
    clReleaseMemObject(chk_out);
    if (!msg.empty()) return fail(msg.c_str());

    std::size_t diff = 0;
    for (std::size_t i = 0; i < want16.size(); ++i) {
      if (got16[i] != want16[i]) {
        if (diff < 4) {
          std::fprintf(stderr, "[u16check] sample %llu: want %u, got %u\n",
                       static_cast<unsigned long long>(i),
                       static_cast<unsigned>(want16[i]),
                       static_cast<unsigned>(got16[i]));
        }
        ++diff;
      }
    }
    std::fprintf(stderr, "[u16check] %llu of %llu samples differ\n",
                 static_cast<unsigned long long>(diff),
                 static_cast<unsigned long long>(want16.size()));
    // `batch` holds the quantised-input float result, not the original image's
    // result.  That is a real difference from a normal run, and it is why this is
    // a diagnostic switch and not something the test suite turns on.
  }

  if (out_yuv444) {
    // 3 bytes per pixel, not 8.  The caller's buffer is the encoder's input and is
    // sized for this format, so reading u16_bytes here would run four times past
    // the end of it.
    e = clEnqueueReadBuffer(ctx->queue, b_out_yuv, CL_TRUE, 0, yuv_bytes,
                            out_u16, 0, nullptr, nullptr);
    if (e != CL_SUCCESS) return fail("readback(yuv444)");
  } else if (want_u16) {
    // The uint16 result has to come back to the host: the encoder pipe consumes
    // it directly, so unlike the float path there is no COPY_HOST_PTR allocation
    // the scatter could have written into.
    e = clEnqueueReadBuffer(ctx->queue, b_out16, CL_TRUE, 0, u16_bytes, out_u16, 0,
                            nullptr, nullptr);
    if (e != CL_SUCCESS) return fail("readback(u16)");
  } else if (!want_in_u16 && !in_yuv444) {
    // The pixel buffer was created with COPY_HOST_PTR and is read-write, so the
    // three kernels have already left the dithered frames in `batch`.  No
    // download is needed: the same allocation the reader filled is the one the
    // scatter wrote.  That is the one thing this port does better than CUDA,
    // where the uint16 path needs an explicit copy out of device memory.
    e = clEnqueueReadBuffer(ctx->queue, b_pix, CL_TRUE, 0, pix_bytes, batch, 0,
                            nullptr, nullptr);
    if (e != CL_SUCCESS) return fail("readback");
  }

  if (msg.empty() && !want_in_u16 && !want_u16) {
    // The pixel buffer was created with COPY_HOST_PTR and is read-write, so the
    // three kernels have already left the dithered frames in `batch`.  No
    // download is needed: the same allocation the reader filled is the one the
    // scatter wrote.  That is the one thing this port does better than CUDA,
    // where the uint16 path needs an explicit copy out of device memory.
    e = clEnqueueReadBuffer(ctx->queue, b_pix, CL_TRUE, 0, pix_bytes, batch, 0,
                            nullptr, nullptr);
    if (e != CL_SUCCESS) msg = fail("readback");
  }

  // RD_OCL_DUMP=<prefix> writes the gather's output and the walk's index buffer
  // straight back to the host.  This is the bisect: it separates "the walk
  // computed different indices" from "the scatter did the wrong thing with the
  // same indices", which no amount of staring at the final image can tell apart.
  // Whichever buffer's hash is unstable is the one carrying the fault, and
  // everything downstream of it is exonerated.
  if (msg.empty()) {
    if (const char* prefix = std::getenv("RD_OCL_DUMP")) {
      std::vector<unsigned char> tmp_cx(cx_bytes);
      std::vector<unsigned char> tmp_idx(idx_bytes);
      clEnqueueReadBuffer(ctx->queue, b_cx, CL_TRUE, 0, cx_bytes, tmp_cx.data(),
                          0, nullptr, nullptr);
      clEnqueueReadBuffer(ctx->queue, b_idx, CL_TRUE, 0, idx_bytes, tmp_idx.data(),
                          0, nullptr, nullptr);
      const std::string cx_path = std::string(prefix) + ".cx.bin";
      const std::string idx_path = std::string(prefix) + ".idx.bin";
      FILE* f = std::fopen(cx_path.c_str(), "wb");
      if (f != nullptr) { std::fwrite(tmp_cx.data(), 1, tmp_cx.size(), f); std::fclose(f); }
      f = std::fopen(idx_path.c_str(), "wb");
      if (f != nullptr) { std::fwrite(tmp_idx.data(), 1, tmp_idx.size(), f); std::fclose(f); }
      // The scatter's other inputs and its output, read back as the DEVICE sees
      // them.  cx and idx came back identical across 14 runs, so the fault is in
      // the scatter or in what it reads; this names which.
      {
        const std::size_t owner_bytes = owner->size() * sizeof(int);
        const std::size_t pal_bytes = hpal.size() * sizeof(double);
        std::vector<unsigned char> tmp_owner(owner_bytes);
        std::vector<unsigned char> tmp_pal(pal_bytes);
        std::vector<unsigned char> tmp_pix(pix_bytes);
        clEnqueueReadBuffer(ctx->queue, b_owner, CL_TRUE, 0, owner_bytes,
                            tmp_owner.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(ctx->queue, b_pal, CL_TRUE, 0, pal_bytes,
                            tmp_pal.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(ctx->queue, b_pix, CL_TRUE, 0, pix_bytes,
                            tmp_pix.data(), 0, nullptr, nullptr);
        f = std::fopen((std::string(prefix) + ".owner.bin").c_str(), "wb");
        if (f != nullptr) { std::fwrite(tmp_owner.data(), 1, tmp_owner.size(), f); std::fclose(f); }
        f = std::fopen((std::string(prefix) + ".pal.bin").c_str(), "wb");
        if (f != nullptr) { std::fwrite(tmp_pal.data(), 1, tmp_pal.size(), f); std::fclose(f); }
        f = std::fopen((std::string(prefix) + ".pix.bin").c_str(), "wb");
        if (f != nullptr) { std::fwrite(tmp_pix.data(), 1, tmp_pix.size(), f); std::fclose(f); }
        // ...and the same bytes AFTER the readback into `batch`, which is what the
        // image writer will actually see.  `pix` stable while `host` varies puts
        // the fault in the readback; both stable while the PNG varies puts it in
        // the writer.  That is the last split.
        f = std::fopen((std::string(prefix) + ".host.bin").c_str(), "wb");
        if (f != nullptr) { std::fwrite(batch, 1, pix_bytes, f); std::fclose(f); }
      }
    }
  }

  release_all();
  return msg;
}

}  // namespace rd

#else  // !RD_WITH_OPENCL

// Compiled without OpenCL.  These must exist and must be honest: the CLI probes
// for the engine, and an engine that reports "not available" is a feature that
// is off, whereas one that claims to be available and then fails to initialise
// is a bug report nobody can act on.
namespace rd {

bool OpenCLAvailable(std::string* device_name, std::string* detail) {
  (void)device_name;
  if (detail != nullptr) {
    *detail = "this build has no OpenCL (configure with -DRD_WITH_OPENCL=ON)";
  }
  return false;
}

// MISSING HERE UNTIL 2026-09-30, and not a cosmetic gap.
//
// rd_video.cpp calls rd::OpenCLStateCount() unconditionally, and this function lived
// only inside the #if defined(RD_WITH_OPENCL) half of this file.  So every build
// without the OpenCL SDK failed at LINK time with
//
//   LNK2019: unresolved external symbol "int __cdecl rd::OpenCLStateCount(void)"
//
// The SDK directory is gitignored, so it is absent from any fresh clone, and CMake
// advertises the no-SDK build as "complete and usable".  The build only ever worked
// in a working tree that happened to have the SDK unpacked beside it, which is why an
// incremental build here stayed green for the entire life of the OpenCL engine.  It
// was found by cloning the repository clean and building it, which is the only test
// that distinguishes "my tree compiles" from "this compiles".
//
// The value is a property of the engine's design rather than of how this file was
// compiled: the real build keeps two (kSlots), and so does the CUDA engine
// (kGpuStateSlots).  Returning the same number keeps the header's promise that a slot
// past the end is clamped rather than rejected, true in both builds.
int OpenCLStateCount() { return 2; }

const std::string& OpenCLBuildLog() {
  static const std::string empty;
  return empty;
}

std::string RiemersmaBlocksOpencl(const Palette& palette,
                                  const DitherParams& params,
                                  const ColorTree& tree, std::size_t width,
                                  std::size_t height, RgbaF* batch,
                                  const BlockOptions& options,
                                  std::string* device_name,
                                  std::uint16_t* out_u16,
                                  int state_slot,
                                  const std::uint16_t* in_u16) {
  (void)palette; (void)params; (void)tree; (void)width; (void)height;
  (void)batch; (void)options; (void)device_name; (void)state_slot;
  (void)out_u16; (void)in_u16;
  return "opencl: this build has no OpenCL engine";
}

}  // namespace rd

#endif  // RD_WITH_OPENCL
