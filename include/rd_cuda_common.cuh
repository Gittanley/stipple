// rd_cuda_common.cuh -- device-side helpers shared by the exact and the
// approximate CUDA engines.
//
// Bit-exactness rules that apply to everything in here:
//   * Never let nvcc contract a*b+c into an FMA.  Use __dadd_rn / __dmul_rn so
//     each operation rounds exactly once, like the host does under /fp:precise.
//   * Never recompute log2() on the device; a 1-ULP disagreement would shift the
//     Hilbert level and walk a different curve.  The level arrives as an
//     argument.
//   * -use_fast_math breaks both of the above.  Do not add it.
#pragma once

#include <cuda_runtime.h>

#include "rd_octree.h"
#include "rd_types.h"

namespace rd {
namespace cuda_common {

constexpr double kQRange = 65535.0;
constexpr double kQScale = 1.0 / 65535.0;
constexpr double kErb = 1.0 / 16.0;

// Double-precision 4-vector, mirroring DoublePixelPacket.
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

// quantize.c:AssociateAlphaPixel() applied to values already held in double,
// which is what the approximate engine needs: IM's sequential walk clamps to
// float and then *widens back to double* before the palette lookup, so the
// target entering ClosestColor() is a double that has been through a float.
__device__ __forceinline__ void d_associate_from_double(int assoc, double r,
                                                        double g, double b,
                                                        double a, Rgba* out) {
  const double ca = static_cast<double>(d_clamp_pixel(a));
  out->a = ca;
  if (!assoc || a == kQRange) {
    out->r = static_cast<double>(d_clamp_pixel(r));
    out->g = static_cast<double>(d_clamp_pixel(g));
    out->b = static_cast<double>(d_clamp_pixel(b));
    return;
  }
  const double alpha = __dmul_rn(kQScale, ca);
  out->r = __dadd_rn(__dmul_rn(alpha, static_cast<double>(d_clamp_pixel(r))), 0.0);
  out->g = __dadd_rn(__dmul_rn(alpha, static_cast<double>(d_clamp_pixel(g))), 0.0);
  out->b = __dadd_rn(__dmul_rn(alpha, static_cast<double>(d_clamp_pixel(b))), 0.0);
}

// quantize.c:ColorToQNodeId().
//
// The packed form takes the ScaleQuantumToChar values that are *already
// computed* and only shifts them.  This matters: RiemersmaDither() calls
// ColorToQNodeId once per descent level with a different `index`, and each call
// otherwise recomputes four ScaleQuantumToChar values -- each of which contains
// a float division.  That is 32 divisions per pixel per block where 4 suffice,
// since the clamp+scale is level-independent and only the shift changes.
__device__ __forceinline__ int d_node_id_packed(unsigned int c0, unsigned int c1,
                                               unsigned int c2,
                                               unsigned int c3, int assoc,
                                               int index) {
  int id = static_cast<int>((c0 >> index) & 0x01);
  id |= static_cast<int>((c1 >> index) & 0x01) << 1;
  id |= static_cast<int>((c2 >> index) & 0x01) << 2;
  if (assoc) id |= static_cast<int>((c3 >> index) & 0x01) << 3;
  return id;
}

// The four ScaleQuantumToChar(ClampPixel(.)) values, computed once.
__device__ __forceinline__ void d_packed_channels(double r, double g, double b,
                                                  double a,
                                                  unsigned int* c0,
                                                  unsigned int* c1,
                                                  unsigned int* c2,
                                                  unsigned int* c3) {
  *c0 = d_scale_char(d_clamp_pixel(r));
  *c1 = d_scale_char(d_clamp_pixel(g));
  *c2 = d_scale_char(d_clamp_pixel(b));
  *c3 = d_scale_char(d_clamp_pixel(a));
}

// quantize.c:ColorToQNodeId().
__device__ __forceinline__ int d_node_id(int assoc, double r, double g, double b,
                                         double a, int index) {
  int id = (d_scale_char(d_clamp_pixel(r)) >> index) & 0x01;
  id |= ((d_scale_char(d_clamp_pixel(g)) >> index) & 0x01) << 1;
  id |= ((d_scale_char(d_clamp_pixel(b)) >> index) & 0x01) << 2;
  if (assoc) id |= ((d_scale_char(d_clamp_pixel(a)) >> index) & 0x01) << 3;
  return id;
}

// quantize.c:AssociateAlphaPixel().
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

// Device mirror of rd::QNode, laid out so a host-side std::vector<QNode> can be
// field-copied into it.
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

// quantize.c:ClosestColor() over the subtree rooted at `node`, with IM's
// post-order traversal and per-channel early exits.  Recursion depth is bounded
// by the octree depth (<= 8).  Declared `inline` because this header is included
// by several translation units and __device__ functions otherwise get external
// linkage, producing duplicate symbols at link time.
// Flat search index: one entry per tree node, holding that node's colour-bearing
// descendants as an ORDERED candidate list.
//
// quantize.c:ClosestColor() recurses over a node's children in index order and
// evaluates leaves post-order, and a leaf wins ties (`d <= distance`), so the answer
// is the LAST leaf in that order that matches.  Reproducing the answer means
// reproducing the order, not merely the set -- which is why this stores positions,
// not a plain membership set.
//
// The traversal itself was the cost, not the arithmetic.  It is a chain of dependent
// loads through a 128-byte node, and only ~3.7 nodes are visited per pixel, so almost
// none of the time is arithmetic: what dominates is load latency and the fact that
// adjacent pixels take different paths, so a warp pays for the union of its lanes'
// traversals.  Measured on two clips of identical geometry, 2.63 versus 3.71
// nodes/pixel produced 10.4 versus 24.2 ms/frame -- a negative intercept when fitted
// linearly, which is the signature of that amplification rather than of work.
//
// A 16-colour palette has at most 16 leaves in any subtree, so the whole ordered list
// is a presence mask plus a nibble-packed array of palette indices: 12 bytes,
// register-resident, no chasing.  Same visits, same order, same distances.
//
// Only valid when the palette has at most 16 entries, which is what the nibble
// packing holds; BuildFlatSearch reports that, and the recursive path stays for
// anything larger.
struct DevSearch {
  std::uint32_t mask;    // bit j: post-order position j holds a colour
  std::uint64_t packed;  // nibble j: palette index at post-order position j
};

// Diagnostics: ClosestColor node entries, pixels searched, descent steps.
//
// COMPILE-TIME gated, and that is not a style choice.  These are atomics on a
// kernel that runs once per pixel: leaving them in cost a measured 2.5x on the
// dither stage (12.6 s -> 29.4 s for 605 1080p frames) while a runtime env check
// appeared to work perfectly, because the check only gated the readback.  There is
// no cheap runtime test here -- a branch per pixel to protect a diagnostic is still
// per-pixel work, and the counter is the expensive part regardless.
//
// Enable with -DRD_CC_STATS=ON; RD_CC_STATS=1 in the environment then prints
// nodes/pixel, which is the number that explains why one clip's walk costs more
// than another's: the search is data-dependent, and the data is the tree.
#if defined(RD_CC_STATS)
__device__ unsigned long long g_cc_entries = 0;
__device__ unsigned long long g_cc_pixels = 0;
__device__ unsigned long long g_cc_descent = 0;
#define RD_CC_COUNT(sym) atomicAdd(&(sym), 1ull)
#else
#define RD_CC_COUNT(sym) ((void)0)
#endif

// The flat candidate index, as a device global so that neither the walk's signature
// nor its register budget has to grow by a pointer.  Set once per clip alongside
// d_nodes, and only when the palette fits the nibble packing.
__device__ const DevSearch* g_flat_search = nullptr;

__device__ inline void d_closest_color(const DevNode* __restrict__ nodes,
                                const double* __restrict__ palette, int assoc,
                                int node, const Rgba& target, double* distance,
                                int* color_number) {
  const int children = assoc ? 16 : 8;
  RD_CC_COUNT(g_cc_entries);
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

// Builds the flat search index by *running* the traversal on the host, once.
//
// The recursive device version visits children in index order and then the node
// itself, so a node's candidate list is: every child's list, in child order,
// concatenated, then its own colour if it has one.  That is exactly what this
// assembles, which is why the flat scan is the same visit sequence rather than an
// approximation of it.
//
// Returns false when the palette does not fit the 4-bit packing, or when a subtree
// somehow carries more leaves than that; the caller then leaves g_flat_search null
// and the recursive path stays in charge.
inline bool BuildFlatSearch(const ColorTree& tree, int assoc, DevSearch** out) {
  *out = nullptr;
  const int colours = tree.color_count();
  if (colours <= 0 || colours > 16) return false;
  const std::vector<QNode>& nodes = tree.nodes();
  const int n = static_cast<int>(nodes.size());
  if (n <= 0) return false;
  const int children = assoc ? kRgbaChildren : kRgbChildren;
  std::vector<DevSearch> host(static_cast<std::size_t>(n));
  // Iterative post-order: an explicit stack of (node, next child) pairs, so a deep
  // or wide tree cannot overflow the host stack the way a recursive helper could.
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
    // `node` is finished, so this is where the recursion would have evaluated it.
    stack.pop_back();
    const QNode& q = nodes[static_cast<std::size_t>(node)];
    if (q.number_unique != 0) {
      // Post-order: the node comes after all of its children, so it takes the
      // first free position, which is exactly where the recursion would reach it.
      unsigned slot = 0;
      while (host[static_cast<std::size_t>(node)].mask & (1u << slot)) ++slot;
      if (slot >= 16 || q.color_number > 15) { ok = false; break; }
      host[static_cast<std::size_t>(node)].mask |= 1u << slot;
      host[static_cast<std::size_t>(node)].packed |=
          static_cast<std::uint64_t>(q.color_number & 0xFu) << (4 * slot);
    }
    // Fold into the parent only now that this node's list is complete, and SHIFT it
    // up past whatever the parent already holds.  Two things go wrong if this is
    // folded with a plain OR: the child's positions are relative to its own list, so
    // a first child occupying slots 0..1 makes a second child's slots 0..1 collide
    // with it; and the packed nibbles OR together instead of concatenating, so the
    // second child's colour overwrites the first's.  Both leave a search with the
    // right shape and the wrong order -- and the wrong order is the answer, because
    // a leaf wins ties.  Only a decoded-raw A/B catches it: the render looks fine and
    // the palette is unchanged.
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
  if (cudaMalloc(out, host.size() * sizeof(DevSearch)) != cudaSuccess) {
    *out = nullptr;
    return false;
  }
  if (cudaMemcpy(*out, host.data(), host.size() * sizeof(DevSearch),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(*out);
    *out = nullptr;
    return false;
  }
  return true;
}

// quantize.c:ClosestColor() over the subtree rooted at `node`, with the traversal
// flattened into an ordered scan.  Bit-for-bit the same result as the recursive
// version -- same candidate order, same running distance, same `<=` tie rule, same
// per-channel early exits -- because the index *is* the traversal, precomputed.
__device__ __forceinline__ void d_closest_color_flat(const DevSearch* __restrict__ search,
    const double* __restrict__ palette, int assoc, int node, const Rgba& target,
    double* distance, int* color_number) {
  const DevSearch s = search[node];
  unsigned m = s.mask;
  unsigned long long pk = s.packed;
  while (m != 0u) {
    const int j = __ffs(m) - 1;
    m &= m - 1u;
    const double* p = palette + 4 * static_cast<int>((pk >> (4 * j)) & 0xFull);
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
            *color_number = static_cast<int>((pk >> (4 * j)) & 0xFull);
          }
        }
      }
    }
  }
}

// quantize.c:RiemersmaDither()'s candidate selection, memo table included.
// `cache` may be null, in which case the lookup is performed every time -- that
// is the cache-free reference the approximate engine converges to.
__device__ __forceinline__ int d_select_index(const DevNode* __restrict__ nodes,
                                              const double* __restrict__ palette,
                                              int assoc, int* __restrict__ cache,
                                              const Rgba& target) {
  const int key = d_cache_offset(assoc, target.r, target.g, target.b, target.a);
  if (cache != nullptr) {
    const int hit = cache[key];
    if (hit >= 0) return hit;
  }
  int node = 0;  // the root is always node 0
  // The clamp+scale is independent of the level, so do it once rather than
  // 8 times inside the descent.
  unsigned int c0, c1, c2, c3;
  d_packed_channels(target.r, target.g, target.b, target.a, &c0, &c1, &c2, &c3);
  for (int i = kMaxTreeDepth - 1; i > 0; --i) {
    const int id = d_node_id_packed(c0, c1, c2, c3, assoc, i);
    const int child = nodes[node].child[id];
    if (child < 0) break;
    node = child;
    RD_CC_COUNT(g_cc_descent);
  }
  RD_CC_COUNT(g_cc_pixels);
  double distance = __dadd_rn(__dmul_rn(4.0, __dmul_rn(kQRange + 1.0, kQRange + 1.0)),
                             1.0);
  int index = 0;
  // Uniform across the whole launch -- g_flat_search is set once per clip -- so the
  // branch costs no divergence.  The recursive path is not dead code: it is what
  // serves a palette too large for the nibble packing (--colors above 16), and it
  // stays the reference the flat index is checked against.
  if (g_flat_search != nullptr) {
    d_closest_color_flat(g_flat_search, palette, assoc, nodes[node].parent, target,
                         &distance, &index);
  } else {
    d_closest_color(nodes, palette, assoc, nodes[node].parent, target, &distance,
                    &index);
  }
  if (cache != nullptr) cache[key] = index;
  return index;
}

// Plain ascending nearest-colour scan.  This is *not* what ImageMagick does --
// IM restricts the candidate set to an octree subtree -- so it is offered only
// to quantify how much of an observed difference is the octree restriction.
__device__ __forceinline__ int d_nearest_linear(const double* __restrict__ palette,
                                                int count, int assoc,
                                                const Rgba& target) {
  double distance = __dadd_rn(__dmul_rn(4.0, __dmul_rn(kQRange + 1.0, kQRange + 1.0)),
                             1.0);
  int best = 0;
  const double beta = __dmul_rn(kQScale, target.a);
  for (int i = 0; i < count; ++i) {
    const double* e = palette + 4 * i;
    const double alpha = assoc ? __dmul_rn(kQScale, e[3]) : 1.0;
    const double dr = __dadd_rn(__dmul_rn(alpha, e[0]), __dmul_rn(-beta, target.r));
    const double dg = __dadd_rn(__dmul_rn(alpha, e[1]), __dmul_rn(-beta, target.g));
    const double db = __dadd_rn(__dmul_rn(alpha, e[2]), __dmul_rn(-beta, target.b));
    double d = __dadd_rn(__dmul_rn(dr, dr),
                         __dadd_rn(__dmul_rn(dg, dg), __dmul_rn(db, db)));
    if (assoc) {
      const double da = __dadd_rn(e[3], -target.a);
      d = __dadd_rn(d, __dmul_rn(da, da));
    }
    if (d <= distance) {  // last candidate wins an exact tie, as in IM
      distance = d;
      best = i;
    }
  }
  return best;
}

// Copies a host-side colour tree into the device layout.
inline void UploadTree(const ColorTree& tree, DevNode** out) {
  std::vector<DevNode> host(tree.nodes().size());
  for (std::size_t i = 0; i < tree.nodes().size(); ++i) {
    const QNode& src = tree.nodes()[i];
    DevNode& dst = host[i];
    for (int c = 0; c < kRgbaChildren; ++c) dst.child[c] = src.child[c];
    dst.parent = src.parent;
    for (int c = 0; c < 4; ++c) dst.total_color[c] = src.total_color[c];
    dst.quantize_error = src.quantize_error;
    dst.number_unique = src.number_unique;
    dst.color_number = src.color_number;
    dst.id = src.id;
    dst.level = src.level;
  }
  *out = nullptr;
  if (cudaMalloc(out, host.size() * sizeof(DevNode)) != cudaSuccess) return;
  if (cudaMemcpy(*out, host.data(), host.size() * sizeof(DevNode),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(*out);
    *out = nullptr;
  }
}

// Flattens a palette into the 4-doubles-per-entry device layout.  Only
// `count` entries are allocated: sizing for MaxColormapSize (65536) meant a
// 2 MiB host vector and a 2 MiB upload for a 16-colour palette, on every call.
inline void UploadPalette(const Palette& palette, double** out) {
  const std::size_t entries =
      static_cast<std::size_t>(palette.count > 0 ? palette.count : 1);
  std::vector<double> host(entries * 4, 0.0);
  for (int i = 0; i < palette.count; ++i) {
    host[4 * i + 0] = palette.entries[i].r;
    host[4 * i + 1] = palette.entries[i].g;
    host[4 * i + 2] = palette.entries[i].b;
    host[4 * i + 3] = palette.entries[i].a;
  }
  *out = nullptr;
  if (cudaMalloc(out, host.size() * sizeof(double)) != cudaSuccess) return;
  if (cudaMemcpy(*out, host.data(), host.size() * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(*out);
    *out = nullptr;
  }
}

}  // namespace cuda_common
}  // namespace rd
