// SPDX-License-Identifier: GPL-3.0-or-later
// rd_approx_cuda.cu -- GPU approximation of the Riemersma walk.
//
// ---------------------------------------------------------------------------
// The formula
// ---------------------------------------------------------------------------
// IM's walk along the Hilbert index n is a 16-tap FIR error-feedback modulator
// (a sigma-delta modulator) whose residual satisfies a relation that is
// *independent of the quantiser*:
//
//     x - q = (I - H) e        =>        e = (I - H)^-1 (x - q)
//
// Substituting e into v = x + H e and using H (I-H)^-1 = (I-H)^-1 - I collapses
// the whole recursion to
//
//     t = q + F (x - q),      F = (I - H)^-1,      q <- Q(t)
//
// which is a fixed-point iteration in which *every* operation is a convolution
// along the curve index.  No scan, no blocks, no carried state -- so it maps
// directly onto a CUDA kernel with no sequential phase at all.
//
// The sequential result is a fixed point of this map, not merely a limit, so the
// iteration is an exact solver whose only question is convergence.  Linearising
// about the fixed point gives the iteration matrix
//
//     M = I - F,    |M(0)| = |1 - 1/(1-G)| = G/(1-G) = 0.541,  G = sum h_k
//
// so the residual contracts by ~0.54 per sweep.  Truncating F to D taps leaves
// a fixed-point offset of order rho^D, with rho = 16^(-1/15) = 0.831:
// D=64 -> 7e-6, D=96 -> 2e-8, D=128 -> 5e-11.
//
// ---------------------------------------------------------------------------
// What this cannot reproduce
// ---------------------------------------------------------------------------
// quantize.c memoises the palette lookup behind a 6-bit-per-channel key, so the
// first target to touch a key fixes the answer for every later target sharing
// it.  That is a *visit-order* effect and no order-free iteration can match it.
// Measured on this machine, deleting the cache alone moves 6% of the pixels of
// an 800x600 plasma image (RMSE 0.036) -- so that, not the parallelisation, is
// the floor on the AE reported for this engine.
// ---------------------------------------------------------------------------
#include "rd_riemersma.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "rd_cuda_common.cuh"
#include "rd_octree.h"

namespace rd {
namespace {

using cuda_common::DevNode;
using cuda_common::Rgba;
constexpr double kQRange_d = 65535.0;

// Impulse response of F = (I - H)^-1.
//
// With h_k = (1/16) rho^(k-1) the feedback filter is a finite geometric taper, so
//
//     H(z)   = (z^-1 - rho^16 z^-17) / (16 (1 - rho z^-1))
//     F(z)   = (1 - rho z^-1) / (1 - 0.89375 z^-1 + 0.003247 z^-17)
//
// and g is obtained by running the first-order recurrence
// g_d = sum_k h_k g_{d-k}.  Computed on the host so both engines and the tests
// see identical coefficients.
std::vector<double> BuildInverseFilter(int taps, double diffusion) {
  double weights[kErrorQueueLength];
  build_error_weights(weights);
  std::vector<double> g(static_cast<std::size_t>(taps) + 1, 0.0);
  g[0] = 1.0;
  for (int d = 1; d <= taps; ++d) {
    double acc = 0.0;
    const int kmax = (d < kErrorQueueLength) ? d : kErrorQueueLength;
    for (int k = 1; k <= kmax; ++k) {
      acc += (diffusion / 16.0) * weights[k - 1] *
             g[static_cast<std::size_t>(d - k)];
    }
    g[static_cast<std::size_t>(d)] = acc;
  }
  return g;
}

__global__ void ScatterKernel(float4* __restrict__ pixels,
                              const int* __restrict__ curve,
                              const int* __restrict__ owner,
                              const unsigned char* __restrict__ q,
                              const double* __restrict__ palette, int n,
                              int assoc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int pixel = curve[i];
  // Skip positions that a later visit supersedes; without this the scatter's
  // last-writer would be whichever thread happened to run last.
  if (owner[pixel] != i) return;
  const double* p = palette + 4 * static_cast<int>(q[i]);
  float4 out;
  out.x = static_cast<float>(p[0]);
  out.y = static_cast<float>(p[1]);
  out.z = static_cast<float>(p[2]);
  out.w = assoc ? static_cast<float>(p[3]) : static_cast<float>(65535.0);
  pixels[pixel] = out;
}

template <typename T>
__global__ void GatherKernelT(const float4* __restrict__ pixels,
                              const int* __restrict__ curve, T* __restrict__ cx,
                              int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float4 p = pixels[curve[i]];
  cx[4 * i + 0] = static_cast<T>(p.x);
  cx[4 * i + 1] = static_cast<T>(p.y);
  cx[4 * i + 2] = static_cast<T>(p.z);
  cx[4 * i + 3] = static_cast<T>(p.w);
}

// Seeds q with the nearest palette entry for the raw pixel (no error diffusion).
__global__ void InitIndexKernel(const float* __restrict__ cx,
                                const double* __restrict__ palette, int count,
                                int assoc, const DevNode* __restrict__ nodes,
                                int use_tree, unsigned char* __restrict__ q,
                                int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 v;
  v.x = cx[4 * i + 0];
  v.y = cx[4 * i + 1];
  v.z = cx[4 * i + 2];
  v.w = cx[4 * i + 3];
  Rgba target;
  cuda_common::d_associate_alpha_pixel(assoc, v, &target);
  const int index = (use_tree != 0)
                        ? cuda_common::d_select_index(nodes, palette, assoc, nullptr,
                                                      target)
                        : cuda_common::d_nearest_linear(palette, count, assoc,
                                                        target);
  q[i] = static_cast<unsigned char>(index);
}

// One Jacobi sweep.  Each thread owns one curve position; the D+1 sample window
// it needs is staged in shared memory so global traffic is (TILE+D)/TILE per
// sample rather than D per sample -- without that tiling the sweep is 30x
// bandwidth bound and slower than the sequential CPU walk.
template <typename T>
__global__ void JacobiStepKernel(const T* __restrict__ cx,
                                 const unsigned char* __restrict__ q_in,
                                 unsigned char* __restrict__ q_out,
                                 const T* __restrict__ g, int taps,
                                 const double* __restrict__ palette, int count,
                                 const DevNode* __restrict__ nodes, int use_tree,
                                 int assoc, int n) {
  constexpr int kTile = 256;
  const int tile_start = blockIdx.x * kTile;
  const int thread = threadIdx.x;
  const int base = tile_start - taps;  // window origin, may be negative

  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* sx = reinterpret_cast<T*>(smem_raw);
  unsigned char* sq = smem_raw + sizeof(T) * static_cast<std::size_t>(kTile + taps) * 4;

  const int window = kTile + taps;
  for (int k = thread; k < window; k += blockDim.x) {
    const int m = base + k;
    if ((m >= 0) && (m < n)) {
      sx[4 * k + 0] = cx[4 * m + 0];
      sx[4 * k + 1] = cx[4 * m + 1];
      sx[4 * k + 2] = cx[4 * m + 2];
      sx[4 * k + 3] = cx[4 * m + 3];
      sq[k] = q_in[m];
    } else {
      sx[4 * k + 0] = sx[4 * k + 1] = sx[4 * k + 2] = sx[4 * k + 3] = T(0);
      sq[k] = 0;
    }
  }
  __syncthreads();

  const int i = tile_start + thread;
  if (i >= n) return;

  // r_d = x[n-d] - pal[q[n-d]];  t = pal[q[n]] + sum_d g[d] r_d
  const int here = thread + taps;  // this thread's slot in the window
  T acc[4] = {T(0), T(0), T(0), T(0)};
  const int dmax = (taps < i) ? taps : i;
  for (int d = 0; d <= dmax; ++d) {
    int k = here - d;
    if (k < 0) k = 0;
    const T* xs = sx + 4 * k;
    const double* p = palette + 4 * static_cast<int>(sq[k]);
    const T gk = g[d];
    for (int c = 0; c < 4; ++c) {
      acc[c] += static_cast<T>(gk) * (xs[c] - static_cast<T>(p[c]));
    }
  }

  const double* p0 = palette + 4 * static_cast<int>(sq[here]);
  // The sequential walk applies ClampPixel() to v before the palette lookup.
  // The inverse-filter form folds the feedback into one expression and would
  // otherwise drop that saturation entirely; keeping it costs two instructions
  // and is what lets the fixed point track the real walk.
  const double vr = static_cast<double>(p0[0]) + static_cast<double>(acc[0]);
  const double vg = static_cast<double>(p0[1]) + static_cast<double>(acc[1]);
  const double vb = static_cast<double>(p0[2]) + static_cast<double>(acc[2]);
  const double va = assoc ? (static_cast<double>(p0[3]) + static_cast<double>(acc[3]))
                          : kQRange_d;

  Rgba target;
  cuda_common::d_associate_from_double(assoc, vr, vg, vb, va, &target);
  const int index = (use_tree != 0)
                        ? cuda_common::d_select_index(nodes, palette, assoc, nullptr,
                                                      target)
                        : cuda_common::d_nearest_linear(palette, count, assoc,
                                                        target);
  q_out[i] = static_cast<unsigned char>(index);
}

}  // namespace

std::string RiemersmaApproxCuda(const Palette& palette, const DitherParams& params,
                                const ColorTree& tree, std::size_t width,
                                std::size_t height, RgbaF* pixels,
                                const ApproxOptions& options,
                                std::string* device_name) {
  if (!CudaAvailable()) return "no CUDA device available";
  if (palette.count < 1) return "empty palette";
  // `q` is one unsigned char per curve position -- InitIndexKernel writes
  // `static_cast<unsigned char>(index)` at :144 and the Jacobi sweep reads it back
  // at :102 -- so indices 256..count-1 fold modulo 256 and alias onto the first 256.
  // d_select_index returns an IN-RANGE index every time, so no lookup fails and the
  // image simply comes out in the wrong colours with no error.  At 257 colours one
  // entry aliases, at 300 forty-four, at 4096 3840 (93.75%).  Refuse rather than
  // truncate.  Matches the blocks engine's wording exactly, so all three engines
  // fail the same way.
  if (palette.count > 256) {
    return "the approx engine carries one palette index per byte, so " +
           std::to_string(palette.count) +
           " colours would alias onto the first 256 and the image would come out "
           "in the wrong colours with no error; use --engine cuda, which is "
           "bit-exact, carries no index buffer, and has no 256-colour limit, or "
           "--colors 256 or fewer.";
  }
  if (options.taps < 1 || options.taps > 4096) return "--approx-taps out of range";

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return "cudaGetDevice failed";
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess && device_name) {
    *device_name = std::string(prop.name) + " (sm_" +
                    std::to_string(prop.major) + std::to_string(prop.minor) + ")";
  }

  const int level = ComputeCurveLevel(width, height);
  // BuildCurveIndex walks the same recursion but never materialises the full
  // 4^level+1 entry list, which at 1920x1080 (level 11) is 67 MB of pure setup.
  std::vector<int> owner;
  std::vector<int> curve;
  BuildCurveIndex(level, width, height, &curve, &owner);
  const int n = static_cast<int>(curve.size());
  if (n < static_cast<int>(width * height)) {
    return "the compacted curve is shorter than the image; the Hilbert "
           "transcription is wrong";
  }

  const int assoc = palette.associate_alpha ? 1 : 0;
  const std::vector<double> g = BuildInverseFilter(options.taps, params.diffusion);

  float4* d_pixels = nullptr;
  void* d_cx = nullptr;
  int* d_curve = nullptr;
  int* d_owner = nullptr;
  unsigned char* d_q0 = nullptr;
  unsigned char* d_q1 = nullptr;
  double* d_palette = nullptr;
  DevNode* d_nodes = nullptr;
  float* d_g_f = nullptr;
  double* d_g_d = nullptr;

  const bool fp64 = options.fp64;
  const std::size_t elem = fp64 ? sizeof(double) : sizeof(float);

  auto cleanup = [&]() {
    if (d_g_d) cudaFree(d_g_d);
    if (d_g_f) cudaFree(d_g_f);
    if (d_nodes) cudaFree(d_nodes);
    if (d_palette) cudaFree(d_palette);
    if (d_q1) cudaFree(d_q1);
    if (d_q0) cudaFree(d_q0);
    if (d_curve) cudaFree(d_curve);
    if (d_owner) cudaFree(d_owner);
    if (d_cx) cudaFree(d_cx);
    if (d_pixels) cudaFree(d_pixels);
  };

  if (cudaMalloc(&d_pixels, width * height * sizeof(float4)) != cudaSuccess) return "cudaMalloc pixels failed";
  if (cudaMalloc(&d_cx, static_cast<std::size_t>(n) * 4 * elem) != cudaSuccess) { cleanup(); return "cudaMalloc curve field failed"; }
  if (cudaMalloc(&d_curve, curve.size() * sizeof(int)) != cudaSuccess) { cleanup(); return "cudaMalloc curve failed"; }
  if (cudaMalloc(&d_q0, curve.size()) != cudaSuccess) { cleanup(); return "cudaMalloc q failed"; }
  if (cudaMalloc(&d_q1, curve.size()) != cudaSuccess) { cleanup(); return "cudaMalloc q2 failed"; }
  cuda_common::UploadPalette(palette, &d_palette);
  if (d_palette == nullptr) { cleanup(); return "palette upload failed"; }
  cuda_common::UploadTree(tree, &d_nodes);
  if (d_nodes == nullptr) { cleanup(); return "tree upload failed"; }

  if (fp64) {
    if (cudaMalloc(&d_g_d, g.size() * sizeof(double)) != cudaSuccess) { cleanup(); return "cudaMalloc g failed"; }
    if (cudaMemcpy(d_g_d, g.data(), g.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) { cleanup(); return "g upload failed"; }
  } else {
    std::vector<float> gf(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) gf[i] = static_cast<float>(g[i]);
    if (cudaMalloc(&d_g_f, g.size() * sizeof(float)) != cudaSuccess) { cleanup(); return "cudaMalloc g failed"; }
    if (cudaMemcpy(d_g_f, gf.data(), g.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { cleanup(); return "g upload failed"; }
  }
  if (cudaMemcpy(d_pixels, pixels, width * height * sizeof(float4), cudaMemcpyHostToDevice) != cudaSuccess) { cleanup(); return "upload failed"; }
  if (cudaMemcpy(d_curve, curve.data(), curve.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) { cleanup(); return "curve upload failed"; }
  if (cudaMalloc(&d_owner, owner.size() * sizeof(int)) != cudaSuccess) { cleanup(); return "cudaMalloc owner failed"; }
  if (cudaMemcpy(d_owner, owner.data(), owner.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) { cleanup(); return "owner upload failed"; }

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  if (fp64) {
    GatherKernelT<double><<<blocks, threads>>>(d_pixels, d_curve,
                                               static_cast<double*>(d_cx), n);
  } else {
    GatherKernelT<float><<<blocks, threads>>>(d_pixels, d_curve,
                                              static_cast<float*>(d_cx), n);
  }
  InitIndexKernel<<<blocks, threads>>>(static_cast<const float*>(d_cx), d_palette,
                                       palette.count, assoc, d_nodes,
                                       options.use_tree, d_q0, n);

  const int tile = 256;
  const int jacobi_blocks = (n + tile - 1) / tile;
  const std::size_t shmem =
      static_cast<std::size_t>(tile + options.taps) * 4 * elem +
      (tile + options.taps) + 32;

  for (int j = 0; j < options.iterations; ++j) {
    if (fp64) {
      JacobiStepKernel<double><<<jacobi_blocks, tile, shmem>>>(
          static_cast<const double*>(d_cx), d_q0, d_q1, d_g_d, options.taps,
          d_palette, palette.count, d_nodes, options.use_tree ? 1 : 0, assoc, n);
    } else {
      JacobiStepKernel<float><<<jacobi_blocks, tile, shmem>>>(
          static_cast<const float*>(d_cx), d_q0, d_q1, d_g_f, options.taps,
          d_palette, palette.count, d_nodes, options.use_tree ? 1 : 0, assoc, n);
    }
    if (cudaGetLastError() != cudaSuccess) { cleanup(); return "jacobi launch failed"; }
    std::swap(d_q0, d_q1);
  }

  ScatterKernel<<<blocks, threads>>>(d_pixels, d_curve, d_owner, d_q0, d_palette,
                                    n, assoc);
  const cudaError_t sync = cudaDeviceSynchronize();
  if (sync != cudaSuccess) {
    const std::string detail = cudaGetErrorString(sync);
    cleanup();
    return "kernel execution failed: " + detail;
  }
  if (cudaMemcpy(pixels, d_pixels, width * height * sizeof(float4), cudaMemcpyDeviceToHost) != cudaSuccess) {
    cleanup();
    return "download failed";
  }
  cleanup();
  return std::string();
}

std::vector<double> ApproxInverseFilter(int taps, double diffusion) {
  return BuildInverseFilter(taps, diffusion);
}

}  // namespace rd
