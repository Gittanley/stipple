// rd_riemersma_cuda.cu -- CUDA engine for the bit-exact Riemersma walk.
//
// Bit-exactness notes (these are the whole ballgame here):
//   * nvcc contracts a*b+c into FMA by default, which perturbs the last bit of
//     every accumulation.  The error queue and the distance metric therefore go
//     through __dadd_rn / __dmul_rn, which are guaranteed not to be fused, so
//     the device performs one IEEE-754 round-to-nearest per operation exactly
//     like the host.  Do not add -use_fast_math to this file.
//   * The curve level is computed on the host and passed in.  Recomputing
//     log2() on the device risks a 1-ULP disagreement that would shift the
//     level by one and walk a different curve.
//   * The 16-entry error queue and the memo cache are per walk.  Each CUDA
//     thread drives one independent walk, so batching frames is bit-exact per
//     frame -- there is no cross-frame state to invalidate.
//
// Parallelism reality check: Riemersma diffusion is a sequential dependency
// chain (one error queue, one cursor, each visit consuming the previous
// residual), so a single image exposes exactly one thread of useful work.  The
// GPU earns its keep by (a) keeping the entire walk resident in device memory,
// eliminating host/device round trips, and (b) running many frames at once, each
// on its own thread, which is what Phase 2 video does.
#include "rd_riemersma.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace rd {
namespace {

// The dither uploads pixels as float4; RgbaF must be layout-identical.
static_assert(sizeof(RgbaF) == sizeof(float4),
              "RgbaF must be layout compatible with CUDA's float4");
static_assert(offsetof(RgbaF, r) == 0 && offsetof(RgbaF, g) == 4 &&
                  offsetof(RgbaF, b) == 8 && offsetof(RgbaF, a) == 12,
              "RgbaF channel order must match float4");

constexpr double kQRange = 65535.0;
constexpr double kQScale = 1.0 / 65535.0;
constexpr double kErb = 1.0 / 16.0;

struct Rgba {
  double r, g, b, a;
};

__device__ __forceinline__ float d_clamp_pixel(double v) {
  if (v < 0.0) return 0.0f;
  if (v >= kQRange) return static_cast<float>(kQRange);
  return static_cast<float>(v);
}

__device__ __forceinline__ unsigned char d_scale_char(float q) {
  if (isnan(q) || q <= 0.0f) return 0;
  const float scaled = q / 257.0f;
  if (scaled >= 255.0f) return 255;
  return static_cast<unsigned char>(scaled + 0.5f);
}

__device__ __forceinline__ int d_cache_offset(int assoc, double r, double g,
                                              double b, double a) {
  const int rr = d_scale_char(d_clamp_pixel(r)) >> 2;
  const int gg = d_scale_char(d_clamp_pixel(g)) >> 2;
  const int bb = d_scale_char(d_clamp_pixel(b)) >> 2;
  int offset = rr | (gg << 6) | (bb << 12);
  if (assoc) offset |= (d_scale_char(d_clamp_pixel(a)) >> 2) << 18;
  return offset;
}

__device__ __forceinline__ void d_associate_alpha_pixel(int assoc, const float4& px,
                                                        Rgba* out) {
  out->a = static_cast<double>(px.w);
  if (!assoc || px.w == static_cast<float>(kQRange)) {
    out->r = static_cast<double>(px.x);
    out->g = static_cast<double>(px.y);
    out->b = static_cast<double>(px.z);
    return;
  }
  const double alpha = kQScale * static_cast<double>(px.w);
  out->r = __dmul_rn(alpha, static_cast<double>(px.x));
  out->g = __dmul_rn(alpha, static_cast<double>(px.y));
  out->b = __dmul_rn(alpha, static_cast<double>(px.z));
}

__device__ __forceinline__ void d_associate_alpha_info(int assoc, const double* e,
                                                       Rgba* out) {
  out->a = e[3];
  if (!assoc || e[3] == kQRange) {
    out->r = e[0];
    out->g = e[1];
    out->b = e[2];
    return;
  }
  const double alpha = kQScale * e[3];
  out->r = __dmul_rn(alpha, e[0]);
  out->g = __dmul_rn(alpha, e[1]);
  out->b = __dmul_rn(alpha, e[2]);
}

// Device mirror of rd::QNode.  Laid out so it can be memcpy'd straight from the
// host-side vector.
struct DevNode {
  int child[16];
  int parent;
  double total_color[4];
  double quantize_error;
  unsigned long long number_unique;
  unsigned int color_number;
  unsigned int id;
  unsigned int level;
};

// quantize.c:ColorToQNodeId().
__device__ __forceinline__ int d_node_id(int assoc, double r, double g, double b,
                                         double a, int index) {
  int id = (d_scale_char(d_clamp_pixel(r)) >> index) & 0x01;
  id |= ((d_scale_char(d_clamp_pixel(g)) >> index) & 0x01) << 1;
  id |= ((d_scale_char(d_clamp_pixel(b)) >> index) & 0x01) << 2;
  if (assoc) id |= ((d_scale_char(d_clamp_pixel(a)) >> index) & 0x01) << 3;
  return id;
}

// quantize.c:ClosestColor() over the subtree rooted at `node`, with IM's
// post-order traversal and per-channel early exits.  Recursion depth is bounded
// by the tree depth (<= 8), so device recursion is safe here.
__device__ void d_closest_color(const DevNode* __restrict__ nodes,
                                const double* __restrict__ palette, int assoc,
                                int node, const Rgba& target, double* distance,
                                int* color_number) {
  const int children = assoc ? 16 : 8;
  for (int i = 0; i < children; ++i) {
    const int child = nodes[node].child[i];
    if (child >= 0) {
      d_closest_color(nodes, palette, assoc, child, target, distance,
                      color_number);
    }
  }
  if (nodes[node].number_unique == 0) return;
  const double* p = palette + 4 * static_cast<int>(nodes[node].color_number);
  double alpha = 1.0;
  double beta = 1.0;
  if (assoc) {
    alpha = __dmul_rn(kQScale, p[3]);
    beta = __dmul_rn(kQScale, target.a);
  }
  double pixel = __dadd_rn(__dmul_rn(alpha, p[0]), __dmul_rn(-beta, target.r));
  double d = __dmul_rn(pixel, pixel);
  if (d <= *distance) {
    pixel = __dadd_rn(__dmul_rn(alpha, p[1]), __dmul_rn(-beta, target.g));
    d = __dadd_rn(d, __dmul_rn(pixel, pixel));
    if (d <= *distance) {
      pixel = __dadd_rn(__dmul_rn(alpha, p[2]), __dmul_rn(-beta, target.b));
      d = __dadd_rn(d, __dmul_rn(pixel, pixel));
      if (d <= *distance) {
        if (assoc) {
          pixel = __dadd_rn(p[3], -target.a);
          d = __dadd_rn(d, __dmul_rn(pixel, pixel));
        }
        if (d <= *distance) {
          *distance = d;
          *color_number = static_cast<int>(nodes[node].color_number);
        }
      }
    }
  }
}

struct Frame {
  int level;
  int dir;
  int stage;
};

// One thread == one independent walk.  `level` comes from the host so the curve
// cannot drift; see the file header.
//
// The 16-entry error queue (512 B) and the explicit recursion stack (768 B) are
// held in per-thread *global* scratch rather than local arrays: compute
// capability 7.x caps local memory at 512 B per thread, and keeping them local
// overflows the stack and kills the context.
__global__ void RiemersmaWalkKernel(float4* __restrict__ pixels, int width,
                                    int height, int level,
                                    const DevNode* __restrict__ nodes,
                                    const double* __restrict__ palette,
                                    int palette_count,
                                    const double* __restrict__ weights,
                                    double diffusion, int assoc,
                                    int* __restrict__ cache, int cache_entries,
                                    double* __restrict__ error_state,
                                    int* __restrict__ frame_state) {
  const int frame_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (level <= 0) return;

  // Per-thread scratch, carved out of the batch allocations.
  Rgba* error = reinterpret_cast<Rgba*>(error_state) + frame_id * 16;
  Frame* stack = reinterpret_cast<Frame*>(frame_state) + frame_id * 64;
  int* my_cache = cache + static_cast<size_t>(frame_id) * cache_entries;
  int x = 0, y = 0;

  for (int i = 0; i < cache_entries; ++i) my_cache[i] = -1;
  for (int i = 0; i < 16; ++i) {
    error[i].r = 0.0;
    error[i].g = 0.0;
    error[i].b = 0.0;
    error[i].a = 0.0;
  }

  // Mirrors rd_riemersma.h:CurveStep().  0=W 1=E 2=N 3=S; >=0 is a dither
  // visit, <0 is a descent into (-e-1).
  const signed char leaf[4][3] = {{+1, +3, +0}, {+0, +2, +1}, {+3, +1, +2},
                                  {+2, +0, +3}};
  const signed char node[4][7] = {{-3, +1, -1, +3, -1, +0, -4},
                                  {-4, +0, -2, +2, -2, +1, -3},
                                  {-1, +3, -3, +1, -3, +2, -2},
                                  {-2, +2, -4, +0, -4, +3, -1}};

  // The trailing ForgetGravity visit is folded in as one more loop iteration by
  // running the stack walk and then repeating the body once without advancing.
  int sp = 0;
  stack[0] = Frame{level, 2 /* North */, 0};

  while (sp >= 0) {
    Frame& f = stack[sp];
    const int stages = (f.level == 1) ? 3 : 7;
    if (f.stage >= stages) {
      --sp;
      continue;
    }
    const signed char* table = (f.level == 1) ? leaf[f.dir] : node[f.dir];
    const signed char e = table[f.stage];
    ++f.stage;
    const int next = (e < 0) ? (-e - 1) : e;

    if (e >= 0) {
      if (x >= 0 && y >= 0 && x < width && y < height) {
        const size_t idx = static_cast<size_t>(y) * width + x;
        float4* px = pixels + idx;
        const float4 src = *px;
        Rgba pixel;
        d_associate_alpha_pixel(assoc, src, &pixel);

        for (int i = 0; i < 16; ++i) {
          const double w = __dmul_rn(__dmul_rn(kErb, diffusion), weights[i]);
          pixel.r = __dadd_rn(pixel.r, __dmul_rn(w, error[i].r));
          pixel.g = __dadd_rn(pixel.g, __dmul_rn(w, error[i].g));
          pixel.b = __dadd_rn(pixel.b, __dmul_rn(w, error[i].b));
          if (assoc) pixel.a = __dadd_rn(pixel.a, __dmul_rn(w, error[i].a));
        }
        pixel.r = static_cast<double>(d_clamp_pixel(pixel.r));
        pixel.g = static_cast<double>(d_clamp_pixel(pixel.g));
        pixel.b = static_cast<double>(d_clamp_pixel(pixel.b));
        if (assoc) pixel.a = static_cast<double>(d_clamp_pixel(pixel.a));

        const int key = d_cache_offset(assoc, pixel.r, pixel.g, pixel.b, pixel.a);
        int index = my_cache[key];
        if (index < 0) {
          int node = 0;  // the root is always node 0
          for (int i = kMaxTreeDepth - 1; i > 0; --i) {
            const int id =
                d_node_id(assoc, pixel.r, pixel.g, pixel.b, pixel.a, i);
            const int child = nodes[node].child[id];
            if (child < 0) break;
            node = child;
          }
          double distance = __dadd_rn(
              __dmul_rn(4.0, __dmul_rn(kQRange + 1.0, kQRange + 1.0)), 1.0);
          index = 0;
          d_closest_color(nodes, palette, assoc, nodes[node].parent, pixel,
                          &distance, &index);
          my_cache[key] = index;
        }

        const double* ce = palette + 4 * index;
        px->x = static_cast<float>(ce[0]);
        px->y = static_cast<float>(ce[1]);
        px->z = static_cast<float>(ce[2]);
        if (assoc) px->w = static_cast<float>(ce[3]);

        for (int i = 0; i < 15; ++i) error[i] = error[i + 1];
        Rgba color;
        d_associate_alpha_info(assoc, ce, &color);
        error[15].r = __dadd_rn(pixel.r, -color.r);
        error[15].g = __dadd_rn(pixel.g, -color.g);
        error[15].b = __dadd_rn(pixel.b, -color.b);
        if (assoc) error[15].a = __dadd_rn(pixel.a, -color.a);
      }
      switch (next) {  // Advance()
        case 0: --x; break;
        case 1: ++x; break;
        case 2: --y; break;
        case 3: ++y; break;
      }
    } else {
      const int child_level = f.level - 1;
      ++sp;
      stack[sp] = Frame{child_level, next, 0};
    }
  }

  // Trailing RiemersmaDither(ForgetGravity): visit the resting cursor, no move.
  if (x >= 0 && y >= 0 && x < width && y < height) {
    const size_t idx = static_cast<size_t>(y) * width + x;
    float4* px = pixels + idx;
    const float4 src = *px;
    Rgba pixel;
    d_associate_alpha_pixel(assoc, src, &pixel);
    for (int i = 0; i < 16; ++i) {
      const double w = __dmul_rn(__dmul_rn(kErb, diffusion), weights[i]);
      pixel.r = __dadd_rn(pixel.r, __dmul_rn(w, error[i].r));
      pixel.g = __dadd_rn(pixel.g, __dmul_rn(w, error[i].g));
      pixel.b = __dadd_rn(pixel.b, __dmul_rn(w, error[i].b));
      if (assoc) pixel.a = __dadd_rn(pixel.a, __dmul_rn(w, error[i].a));
    }
    pixel.r = static_cast<double>(d_clamp_pixel(pixel.r));
    pixel.g = static_cast<double>(d_clamp_pixel(pixel.g));
    pixel.b = static_cast<double>(d_clamp_pixel(pixel.b));
    if (assoc) pixel.a = static_cast<double>(d_clamp_pixel(pixel.a));
    const int key = d_cache_offset(assoc, pixel.r, pixel.g, pixel.b, pixel.a);
    int index = my_cache[key];
    if (index < 0) {
      int node = 0;
      for (int i = kMaxTreeDepth - 1; i > 0; --i) {
        const int id = d_node_id(assoc, pixel.r, pixel.g, pixel.b, pixel.a, i);
        const int child = nodes[node].child[id];
        if (child < 0) break;
        node = child;
      }
      double distance = __dadd_rn(
          __dmul_rn(4.0, __dmul_rn(kQRange + 1.0, kQRange + 1.0)), 1.0);
      index = 0;
      d_closest_color(nodes, palette, assoc, nodes[node].parent, pixel,
                      &distance, &index);
      my_cache[key] = index;
    }
    const double* ce = palette + 4 * index;
    px->x = static_cast<float>(ce[0]);
    px->y = static_cast<float>(ce[1]);
    px->z = static_cast<float>(ce[2]);
    if (assoc) px->w = static_cast<float>(ce[3]);
  }
}

struct DeviceScratch {
  float4* pixels = nullptr;
  DevNode* nodes = nullptr;
  double* palette = nullptr;
  double* weights = nullptr;
  int* cache = nullptr;
  double* error_state = nullptr;
  int* frame_state = nullptr;

  void Release() {
    if (frame_state) cudaFree(frame_state);
    if (error_state) cudaFree(error_state);
    if (cache) cudaFree(cache);
    if (weights) cudaFree(weights);
    if (palette) cudaFree(palette);
    if (nodes) cudaFree(nodes);
    if (pixels) cudaFree(pixels);
    *this = DeviceScratch{};
  }
};

}  // namespace

bool CudaAvailable() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
  return count > 0;
}

int CudaMaxFrames(std::size_t width, std::size_t height, std::size_t vram_budget) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 0;
  std::size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return 1;
  if (vram_budget != 0 && vram_budget < free_bytes) free_bytes = vram_budget;
  const std::size_t per_frame = width * height * sizeof(float4);
  if (per_frame == 0) return 0;
  std::size_t frames = free_bytes / per_frame;
  if (frames < 1) return 0;
  if (frames > 4096) frames = 4096;
  return static_cast<int>(frames);
}

std::string RiemersmaWalkCuda(const Palette& palette, const DitherParams& params,
                              const ColorTree& tree, std::size_t width,
                              std::size_t height, RgbaF* pixels, int frames,
                              std::string* device_name) {
  if (!CudaAvailable()) return "no CUDA device available";
  if (frames < 1) frames = 1;

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return "cudaGetDevice failed";
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess && device_name) {
    *device_name = std::string(prop.name) + " (sm_" +
                    std::to_string(prop.major) + std::to_string(prop.minor) + ")";
  }

  const int level = ComputeCurveLevel(width, height);
  const int cache_entries = palette.associate_alpha ? kCacheEntries : (1 << 18);
  const std::size_t pixel_count = width * height;
  const std::size_t frames_sz = static_cast<std::size_t>(frames);

  DeviceScratch scratch;
  if (cudaMalloc(&scratch.pixels, pixel_count * frames_sz * sizeof(float4)) !=
      cudaSuccess) {
    return "cudaMalloc for pixels failed (the image does not fit in VRAM)";
  }
  if (cudaMalloc(&scratch.palette, sizeof(double) * 4 * kMaxColormapSize) !=
          cudaSuccess ||
      cudaMalloc(&scratch.weights, sizeof(double) * kErrorQueueLength) !=
          cudaSuccess) {
    scratch.Release();
    return "cudaMalloc for the palette failed";
  }
  if (cudaMalloc(&scratch.nodes, tree.nodes().size() * sizeof(DevNode)) !=
      cudaSuccess) {
    scratch.Release();
    return "cudaMalloc for the colour tree failed";
  }
  {
    // Translate the host tree into the device layout.
    std::vector<DevNode> host_nodes(tree.nodes().size());
    for (std::size_t i = 0; i < tree.nodes().size(); ++i) {
      const QNode& src = tree.nodes()[i];
      DevNode& dst = host_nodes[i];
      for (int c = 0; c < kRgbaChildren; ++c) dst.child[c] = src.child[c];
      dst.parent = src.parent;
      for (int c = 0; c < 4; ++c) dst.total_color[c] = src.total_color[c];
      dst.quantize_error = src.quantize_error;
      dst.number_unique = src.number_unique;
      dst.color_number = src.color_number;
      dst.id = src.id;
      dst.level = src.level;
    }
    if (cudaMemcpy(scratch.nodes, host_nodes.data(),
                   host_nodes.size() * sizeof(DevNode),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      scratch.Release();
      return "upload of the colour tree failed";
    }
  }
  if (cudaMalloc(&scratch.cache,
                 static_cast<std::size_t>(cache_entries) * frames_sz *
                     sizeof(int)) != cudaSuccess) {
    scratch.Release();
    return "cudaMalloc for the memo cache failed (try fewer frames)";
  }
  // Per-thread walk state: 16 DoublePixelPacket-equivalents plus a 64-entry
  // recursion stack.  See the kernel comment on why this is not local memory.
  if (cudaMalloc(&scratch.error_state, frames_sz * 16 * sizeof(Rgba)) !=
          cudaSuccess ||
      cudaMalloc(&scratch.frame_state, frames_sz * 64 * sizeof(Frame)) !=
          cudaSuccess) {
    scratch.Release();
    return "cudaMalloc for the per-thread walk state failed";
  }

  double host_weights[kErrorQueueLength];
  build_error_weights(host_weights);
  std::vector<double> host_palette(4 * kMaxColormapSize, 0.0);
  for (int i = 0; i < palette.count; ++i) {
    host_palette[4 * i + 0] = palette.entries[i].r;
    host_palette[4 * i + 1] = palette.entries[i].g;
    host_palette[4 * i + 2] = palette.entries[i].b;
    host_palette[4 * i + 3] = palette.entries[i].a;
  }
  if (cudaMemcpy(scratch.palette, host_palette.data(),
                 sizeof(double) * 4 * kMaxColormapSize, cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemcpy(scratch.weights, host_weights, sizeof(host_weights),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    scratch.Release();
    return "upload of the palette failed";
  }

  // Upload frame 0, then replicate it for the remaining batch slots so the
  // kernel has `frames` independent, identical starting states.
  if (cudaMemcpy(scratch.pixels, pixels, pixel_count * sizeof(float4),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    scratch.Release();
    return "upload of the source frame failed";
  }
  for (std::size_t f = 1; f < frames_sz; ++f) {
    void* dst = scratch.pixels + f * pixel_count;
    if (cudaMemcpy(dst, scratch.pixels, pixel_count * sizeof(float4),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
      scratch.Release();
      return "frame replication failed";
    }
  }

  const int threads = frames < 256 ? frames : 256;
  const int blocks = static_cast<int>((frames + threads - 1) / threads);
  RiemersmaWalkKernel<<<blocks, threads, 0, nullptr>>>(
      scratch.pixels, static_cast<int>(width), static_cast<int>(height), level,
      scratch.nodes, scratch.palette, palette.count, scratch.weights,
      params.diffusion, palette.associate_alpha ? 1 : 0, scratch.cache,
      cache_entries, scratch.error_state, scratch.frame_state);

  const cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    scratch.Release();
    return std::string("kernel launch failed: ") + cudaGetErrorString(launch);
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    const std::string detail = cudaGetErrorString(cudaGetLastError());
    scratch.Release();
    return "kernel execution failed: " + detail;
  }
  if (cudaMemcpy(pixels, scratch.pixels, pixel_count * sizeof(float4),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    scratch.Release();
    return "download of the dithered frame failed";
  }
  scratch.Release();
  return std::string();
}

}  // namespace rd
