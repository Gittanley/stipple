// SPDX-License-Identifier: GPL-3.0-or-later
// rd_riemersma_cpu.cpp -- bit-exact CPU implementation and the shared curve
// builder.  This file is the correctness oracle for the CUDA engine.
#include "rd_riemersma.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "rd_octree.h"

namespace rd {

double g_video_memory_fraction = 0.0;

namespace {

// Applies one Gravity to the cursor, matching the switch at the tail of
// quantize.c:RiemersmaDither().
inline void Advance(Gravity dir, int* x, int* y) {
  switch (dir) {
    case Gravity::kWest:  --*x; break;
    case Gravity::kEast:  ++*x; break;
    case Gravity::kNorth: --*y; break;
    case Gravity::kSouth: ++*y; break;
  }
}

// quantize.c:RiemersmaDither() body, minus the cursor advance.  Returns nothing;
// out-of-bounds visits are skipped exactly as IM skips them.
//
// `cache` is quantize.c's memo table (-1 == not computed yet).  It is part of
// the observable behaviour, not an optimisation -- see rd_types.h.
inline void Visit(const Palette& palette, const DitherParams& params,
                  const double (&weights)[kErrorQueueLength],
                  RgbaD (&error)[kErrorQueueLength], int* cache,
                  const ColorTree& tree, std::size_t width, std::size_t height,
                  RgbaF* pixels, int x, int y) {
  if (x < 0 || y < 0 || static_cast<std::size_t>(x) >= width ||
      static_cast<std::size_t>(y) >= height) {
    return;
  }

  RgbaF* px = pixels + static_cast<std::size_t>(y) * width + x;
  RgbaD pixel;
  associate_alpha_pixel(palette.associate_alpha, *px, &pixel);

  // Distribute error.  Both the accumulation order and the parenthesisation of
  // the product are IM's, and both matter for the final bit.
  const double erb = kErrorRelativeWeight;
  for (int i = 0; i < kErrorQueueLength; ++i) {
    const double w = erb * params.diffusion * weights[i];
    pixel.r += w * error[i].r;
    pixel.g += w * error[i].g;
    pixel.b += w * error[i].b;
    if (palette.associate_alpha) pixel.a += w * error[i].a;
  }

  // ClampPixel() narrows through float in a Q16-HDRI build.
  pixel.r = static_cast<double>(clamp_pixel(pixel.r));
  pixel.g = static_cast<double>(clamp_pixel(pixel.g));
  pixel.b = static_cast<double>(clamp_pixel(pixel.b));
  if (palette.associate_alpha) {
    pixel.a = static_cast<double>(clamp_pixel(pixel.a));
  }

  const int key = cache_offset(palette.associate_alpha, pixel);
  int index = params.use_cache ? cache[key] : -1;
  if (index < 0) {
    // quantize.c: descend from the root, then search node_info->parent's
    // SUBTREE.  The subtree restriction is observable, so the tree is required.
    std::int32_t node = tree.root();
    for (int i = kMaxTreeDepth - 1; i > 0; --i) {
      const std::size_t id = tree.NodeId(pixel, i);
      const std::int32_t child = tree.Child(node, id);
      if (child < 0) break;
      node = child;
    }
    double distance = 4.0 * ((kQuantumRange + 1.0) * (kQuantumRange + 1.0)) + 1.0;
    index = 0;
    tree.ClosestColor(tree.Parent(node), pixel, &distance, &index);
    if (params.use_cache) cache[key] = index;
  }

  // Assign the pixel to the closest colormap entry.  ClampToQuantum() is a bare
  // (float) narrowing cast in an HDRI build.
  px->r = clamp_to_quantum(palette.entries[index].r);
  px->g = clamp_to_quantum(palette.entries[index].g);
  px->b = clamp_to_quantum(palette.entries[index].b);
  if (palette.associate_alpha) px->a = clamp_to_quantum(palette.entries[index].a);

  // Shift the error queue and append this step's residual.
  for (int i = 0; i < kErrorQueueLength - 1; ++i) error[i] = error[i + 1];
  RgbaD color;
  associate_alpha_info(palette.associate_alpha, palette.entries[index], &color);
  error[kErrorQueueLength - 1].r = pixel.r - color.r;
  error[kErrorQueueLength - 1].g = pixel.g - color.g;
  error[kErrorQueueLength - 1].b = pixel.b - color.b;
  if (palette.associate_alpha) {
    error[kErrorQueueLength - 1].a = pixel.a - color.a;
  }
}

// One activation record of the explicit stack that replaces the recursion of
// quantize.c:Riemersma().
struct Frame {
  int level = 0;
  Gravity dir = Gravity::kNorth;
  int stage = 0;
};

// The explicit-stack form of Riemersma().  `leaf(x, y, dir)` is invoked once per
// dither visit, in IM's exact order; the cursor advance is done here so `leaf`
// only ever sees the position that is about to be visited.
template <typename LeafFn>
void WalkCurve(int level, LeafFn&& leaf) {
  if (level <= 0) return;
  constexpr int kMaxStack = 64;  // level <= 32 for any addressable image
  Frame stack[kMaxStack];
  int sp = 0;
  stack[0] = Frame{level, Gravity::kNorth, 0};
  int x = 0, y = 0;
  while (sp >= 0) {
    Frame& f = stack[sp];
    const int stages = (f.level == 1) ? 3 : 7;
    if (f.stage >= stages) {
      --sp;
      continue;
    }
    Gravity next = Gravity::kNorth;
    const bool is_leaf = CurveStep(f.level, f.dir, f.stage, &next);
    ++f.stage;
    if (is_leaf) {
      leaf(x, y, next);
      Advance(next, &x, &y);
    } else {
      const int child_level = f.level - 1;
      ++sp;
      stack[sp] = Frame{child_level, next, 0};
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Closed-form Hilbert point, matching the recursion exactly
// ---------------------------------------------------------------------------
// WalkCurve costs 4^level leaf steps -- 4.19M of them for a 1920x1080 frame at
// level 11 -- and that dominated the engine's wall clock (267 ms of a 368 ms
// single-shot run).  The same curve has a closed form: the standard d2xy
// iteration, untransposed.  CurveTransformMismatches() confirms the mapping
// point for point against the recursion, which is what settled the orientation
// (a transpose also fits the first quadrant, and only the full check exposes
// that it does not).
void HilbertPoint(std::int64_t n, int level, int* out_x, int* out_y) {
  int x = 0, y = 0;
  std::int64_t side = 1;
  const std::int64_t span = static_cast<std::int64_t>(1) << level;
  for (; side < span; side <<= 1) {
    const int rx = static_cast<int>(1 & (n / 2));
    const int ry = static_cast<int>(1 & (n ^ rx));
    if (ry == 0) {
      if (rx == 1) {
        x = static_cast<int>(side - 1 - x);
        y = static_cast<int>(side - 1 - y);
      }
      const int tmp = x;
      x = y;
      y = tmp;
    }
    x += static_cast<int>(side * rx);
    y += static_cast<int>(side * ry);
    n /= 4;
  }
  *out_x = x;
  *out_y = y;
}

// Confirms the closed form reproduces the recursion exactly (0 == identical).
//
// The two curves are both bijections onto the 2^level grid, so if they disagree
// the fix is a dihedral transform of d2xy.  Rather than guess, all eight are
// scored and the best is reported; that turned up the fact that the transpose
// alone is *not* sufficient for this recursion.
std::size_t CurveTransformMismatches(int level, int transform) {
  const std::size_t total = CurveEntryCount(level);
  if (total == 0) return 0;
  std::vector<CurveStepEntry> curve(total);
  BuildCurveSequence(level, 1u << level, 1u << level, curve.data());
  const int side = 1 << level;
  // CurveEntryCount() is 4^level: 4^level-1 recursive leaves plus the trailing
  // ForgetGravity visit.  d2xy models only the leaves, so compare those, then
  // check the trailing entry separately.
  const std::size_t leaves = total > 0 ? total - 1 : 0;
  std::size_t mismatches = 0;
  for (std::size_t n = 0; n < leaves; ++n) {
    int x = 0, y = 0;
    HilbertPoint(static_cast<std::int64_t>(n), level, &x, &y);
    int tx = x, ty = y;
    switch (transform) {
      case 0: break;
      case 1: std::swap(tx, ty); break;
      case 2: tx = side - 1 - x; ty = y; break;
      case 3: tx = x; ty = side - 1 - y; break;
      case 4: tx = side - 1 - x; ty = side - 1 - y; break;
      case 5: std::swap(tx, ty); tx = side - 1 - tx; break;
      case 6: std::swap(tx, ty); ty = side - 1 - ty; break;
      default: std::swap(tx, ty); tx = side - 1 - tx; ty = side - 1 - ty; break;
    }
    if (tx != curve[n].x || ty != curve[n].y) ++mismatches;
  }
  // The trailing visit is the origin: the Hilbert path closes on itself.
  if (curve[leaves].x != 0 || curve[leaves].y != 0) ++mismatches;
  return mismatches;
}

std::size_t SelfTestCurveMatchesRecursion(int level) {
  return CurveTransformMismatches(level, 0);
}

void BuildCurveSequence(int level, std::size_t /*width*/, std::size_t /*height*/,
                        CurveStepEntry* out) {
  const std::size_t count = CurveEntryCount(level);
  if (count == 0 || out == nullptr) return;
  std::size_t n = 0;
  int x = 0, y = 0;
  WalkCurve(level, [&](int vx, int vy, Gravity dir) {
    out[n].x = vx;
    out[n].y = vy;
    out[n].dir = static_cast<int>(dir);
    ++n;
  });
  // quantize.c:DitherImage() finishes with RiemersmaDither(ForgetGravity):
  // one more visit at the resting cursor, and no advance.
  out[n].x = x;
  out[n].y = y;
  out[n].dir = kNoMove;
}

std::size_t BuildCurveIndex(int level, std::size_t width,
                                           std::size_t height,
                                           std::vector<int>* out,
                                           std::vector<int>* owner) {
  out->clear();
  owner->assign(width * height, -1);
  if (level <= 0) return 0;

  const std::int64_t cells = static_cast<std::int64_t>(1) << (2 * level);

  // Split [0, 4^level-1) into chunks, evaluate the closed form in parallel,
  // then concatenate the in-bounds positions.  No 4^level-sized buffer is ever
  // materialised, so the cost is O(4^level * level / threads) with a small
  // constant instead of a 4.19M-step serial recursion.
  //
  // The leaves run over [0, 4^level - 1): the grid's last cell is the one the
  // recursion never visits.
  unsigned int threads = std::thread::hardware_concurrency();
  if (threads == 0) threads = 4;
  threads = std::min<unsigned int>(threads, 64);
  if (cells < 4096) threads = 1;

  const int chunks = static_cast<int>(threads);
  const std::int64_t leaves = cells - 1;
  const std::int64_t per_chunk = (leaves + chunks - 1) / chunks;

  std::vector<std::vector<int>> local(chunks);
  std::vector<int> counts(chunks, 0);
  std::vector<std::thread> pool;
  pool.reserve(chunks);
  for (int c = 0; c < chunks; ++c) {
    pool.emplace_back([&, c]() {
      const std::int64_t begin = std::min<std::int64_t>(leaves, c * per_chunk);
      const std::int64_t end = std::min<std::int64_t>(leaves, begin + per_chunk);
      local[c].reserve(static_cast<std::size_t>(end - begin));
      for (std::int64_t n = begin; n < end; ++n) {
        int x = 0, y = 0;
        HilbertPoint(n, level, &x, &y);
        if (x < 0 || y < 0 || static_cast<std::size_t>(x) >= width ||
            static_cast<std::size_t>(y) >= height) {
          continue;
        }
        local[c].push_back(y * static_cast<int>(width) + x);
      }
      counts[c] = static_cast<int>(local[c].size());
    });
  }
  for (std::thread& t : pool) t.join();

  // Concatenate in chunk order, which is ascending n, i.e. ImageMagick's order.
  std::size_t total_in = 0;
  for (int c = 0; c < chunks; ++c) total_in += static_cast<std::size_t>(counts[c]);
  out->reserve(total_in + 1);
  for (int c = 0; c < chunks; ++c) {
    out->insert(out->end(), local[c].begin(), local[c].end());
  }

  // The trailing RiemersmaDither(ForgetGravity) runs wherever the cursor came to
  // rest -- and the Hilbert path closes on itself, so that is (0,0), the cell
  // visited first.  ImageMagick therefore genuinely re-dithers the origin, and
  // the sequence is one longer than width*height.  (Established by comparing
  // against BuildCurveSequence: every one of the 4^level cells matches the
  // closed form exactly, and only this final entry needs stating explicitly.)
  out->push_back(0);

  // Last writer wins, matching the sequential final write.
  for (std::size_t i = 0; i < out->size(); ++i) {
    (*owner)[(*out)[i]] = static_cast<int>(i);
  }
  return out->size();
}


void RiemersmaWalkCpu(const Palette& palette, const DitherParams& params,
                      const ColorTree& tree, std::size_t width,
                      std::size_t height, RgbaF* pixels,
                      const CurveStepEntry* curve) {
  double weights[kErrorQueueLength];
  build_error_weights(weights);

  // DitherImage() zeroes the whole queue before the first visit.
  RgbaD error[kErrorQueueLength];
  std::memset(error, 0, sizeof(error));

  // GetQCubeInfo() memsets the memo table to -1.  Only 18 bits of the key are
  // reachable without an alpha channel, so 2^18 entries suffice there; with
  // alpha the key spans 24 bits and IM's own 2^24 allocation is reproduced.
  // When the cache is disabled the table is never consulted, so it is not
  // allocated at all.
  const int cache_entries =
      (palette.associate_alpha && params.use_cache) ? kCacheEntries : (1 << 18);
  std::vector<int> cache(params.use_cache ? static_cast<std::size_t>(cache_entries)
                                           : 0,
                         -1);
  int* const cache_data = cache.empty() ? nullptr : cache.data();

  // A precomputed sequence carries absolute positions, so the cursor advance is
  // already baked in and each entry is simply visited where it lies.
  if (curve != nullptr) {
    const int level = ComputeCurveLevel(width, height);
    const std::size_t count = CurveEntryCount(level);
    for (std::size_t i = 0; i < count; ++i) {
      Visit(palette, params, weights, error, cache_data, tree, width, height,
            pixels, curve[i].x, curve[i].y);
    }
    return;
  }

  const int level = ComputeCurveLevel(width, height);
  int x = 0, y = 0;
  if (level > 0) {
    WalkCurve(level, [&](int vx, int vy, Gravity dir) {
      Visit(palette, params, weights, error, cache_data, tree, width, height,
            pixels, vx, vy);
      x = vx;
      y = vy;
      Advance(dir, &x, &y);
    });
  }
  // Trailing RiemersmaDither(ForgetGravity): visit the resting cursor, no move.
  Visit(palette, params, weights, error, cache_data, tree, width, height, pixels,
        x, y);
}

// The visit order depends only on the geometry, and every engine rebuilds it on
// every call.  That is free for the GPU (once per launch, amortised over a
// batch) but ruinous for the host workers: at 1920x1080 a rebuild is ~60 ms, so a
// 18001-frame job with a batch of 4 would spend ~270 s rebuilding the same curve.
//
// Each worker thread keeps its own copy, keyed by geometry.  The key is stable
// for a whole clip, so this is built once per thread and then costs nothing --
// and being thread_local it needs no locking on the hot path.
struct CachedCurve {
  int level = -1;
  std::size_t width = 0;
  std::size_t height = 0;
  std::vector<int> index;
  std::vector<int> owner;
  // Pixels the curve never visits -- `owner[p] < 0`.  A count AND a list, not an
  // index, deliberately: the list is derived by scanning owner after the build, so it
  // stays correct for however many gaps a future change to ComputeCurveLevel produces,
  // including none.  Carrying a single index would silently under-fill the moment there
  // were two.
  //
  // At the moment there is exactly one, at (2^level - 1, 0), and it lands inside the
  // image only when width is a power of two AND width >= height -- so 1024x768,
  // 1024x1024, 2048x2048 and 4096x2160 have one, and 1920x1080, 1280x720, 33x17 and
  // the rest have none.  That is why a probe must assert this list rather than assume
  // it is empty or has one element.
  std::vector<std::int32_t> unvisited;
};

const CachedCurve& CachedCurveFor(int level, std::size_t width,
                                 std::size_t height) {
  static thread_local CachedCurve cache;
  if (cache.level != level || cache.width != width || cache.height != height) {
    BuildCurveIndex(level, width, height, &cache.index, &cache.owner);
    cache.unvisited.clear();
    for (std::size_t p = 0; p < cache.owner.size(); ++p) {
      if (cache.owner[p] < 0) cache.unvisited.push_back(static_cast<std::int32_t>(p));
    }
    cache.level = level;
    cache.width = width;
    cache.height = height;
  }
  return cache;
}

void RiemersmaBlocksCpu(const Palette& palette, const DitherParams& params,
                        const ColorTree& tree, std::size_t width,
                        std::size_t height, RgbaF* batch, int frames,
                        int block, unsigned char* raw_out, std::size_t out_pixels,
                        bool out_yuv444, std::string* error_out) {
  // `raw_out != nullptr && out_pixels > 0` selects the fused output.  Everything below
  // is written so the two paths produce identical bytes: the fused path reproduces the
  // convert pass's arithmetic EXACTLY rather than approximately, including the places
  // where that arithmetic looks like it could be simplified.
  const bool fused = (raw_out != nullptr) && (out_pixels > 0);
  // `chosen` below is a std::vector<unsigned char>, so the palette index this
  // function hands to the scatter is truncated to 8 bits.  `--colors` accepts
  // up to kMaxColormapSize (65536, rd_cli.cpp), and the index is the winning
  // leaf's color_number, so it runs to palette.count - 1.  Above 256 entries
  // the truncation aliases: index i lands on i & 0xFF and the scatter then
  // reads palette.entries[i & 0xFF] -- a valid entry, the wrong one.  Nothing
  // fails, no allocation is out of range, and the picture is quietly wrong:
  //   count 257 ->   1 entry aliases (0.4%)
  //   count 300 ->  44 entries alias (14.7%)
  //   count 4096 -> 3840 entries alias (93.8%, i.e. the whole palette)
  //
  // So this refuses instead.  Widening `chosen` to uint16 is the real fix and
  // is NOT local: rd_blocks_cuda.cu:185 and rd_opencl.cpp:390 truncate the same
  // index the same way, so all three would have to move together and the
  // `.idx.bin` on-disk format (1 byte per index) would change with them.
  //
  // NOTE ON THE FALLBACK: returning leaves `batch` holding the decoded frames,
  // i.e. UNDITHERED pixels.  That is still a wrong picture -- this function
  // returns void and there is no error channel to report through -- so the
  // stderr line below is the only thing standing between this and a silent bad
  // ERROR CHANNEL.  This function returns void, so a refusal had nowhere to go and the
  // caller set `raw_ready` unconditionally -- which made both early returns produce a
  // SILENT WRONG ENCODE: the writer skipped its convert and encoded whatever the
  // previous batch happened to leave in `b.out()`.  The stderr line was the only
  // defence, and on a quiet run nobody reads stderr.
  //
  // The refusal now travels back through the `std::string* error_out` the caller
  // supplies, using the same success-is-empty convention the two GPU engines already
  // use.  Written on failure, never on success, and a null pointer is tolerated so the
  // declaration's other callers keep working.
  //
  // This is what gates the gather fusion.  Once the reader's widening is fused into the
  // gather there is no float buffer left to fall back on, so the same silent path stops
  // being wrong output and becomes reading uninitialised memory.
  if (palette.count > 256) {
    const char* kRefusal =
        "the blocks engine carries one palette index per byte, so the palette would "
        "alias onto the first 256 and the image would come out in the wrong colours "
        "with no error; use --engine cuda, which is bit-exact, carries no index "
        "buffer, and has no 256-colour limit, or --colors 256 or fewer.";
    if (error_out != nullptr) {
      *error_out = std::string(kRefusal) + " (" + std::to_string(palette.count) +
                   " colours requested)";
    }
    static std::atomic<bool> reported{false};
    bool expected = false;
    if (reported.compare_exchange_strong(expected, true)) {
      // The sentence below is character-for-character the one rd_blocks_cuda.cu
      // returns for the same refusal, because the same refusal on two engines
      // that a single --engine flag chooses between should read as one fault.
      // Only the "[dither] error: " prefix differs now, and only because the
      // stderr line is a courtesy -- the caller is what acts on it.
      std::fprintf(stderr, "[dither] error: %s (%d colours requested)\n", kRefusal,
                   palette.count);
    }
    return;
  }
  if (frames < 1 || block < kErrorQueueLength) {
    if (error_out != nullptr) {
      *error_out = "invalid dither batch: frames must be >= 1 and block must be >= " +
                   std::to_string(kErrorQueueLength) + " (got frames=" +
                   std::to_string(frames) + ", block=" + std::to_string(block) + ")";
    }
    return;
  }
  const std::size_t pixels = width * height;
  double weights[kErrorQueueLength];
  build_error_weights(weights);

  const int level = ComputeCurveLevel(width, height);
  const CachedCurve& cached = CachedCurveFor(level, width, height);
  const std::vector<int>& curve = cached.index;
  const std::vector<int>& owner = cached.owner;
  const int n = static_cast<int>(curve.size());
  const int nblocks = (n + block - 1) / block;
  // Report the gaps in the same words the CUDA blocks engine uses, so one probe can
  // assert both engines and the wording cannot drift between them.  The host has no
  // BlkFillUnvisitedKernel equivalent: nothing here writes those pixels at all, they
  // simply keep whatever the reader put in the frame -- which is the correct answer,
  // and is why this is a note and not a warning.  Named because a count nobody can
  // act on is only half an answer.
  if (!cached.unvisited.empty()) {
    std::fprintf(stderr,
                 "[cpu] curve reaches %zu of %zu pixels; the %zu it does not "
                 "reach keep their source value, which is what ImageMagick's own "
                 "recursion leaves there.  First at index %d (%zu,%zu).  The "
                 "uint16/yuv444 output path writes these explicitly.\n",
                 width * height - cached.unvisited.size(), width * height,
                 cached.unvisited.size(), cached.unvisited.front(),
                 static_cast<std::size_t>(cached.unvisited.front()) % width,
                 static_cast<std::size_t>(cached.unvisited.front()) / width);
  }
  // Chosen index per visit position, exactly as the kernel's d_index buffer.
  // Storing it and scattering from it halves the octree work and is what makes
  // the host output match the device output.  thread_local so ten workers do not
  // each allocate 2 MB per batch.
  static thread_local std::vector<unsigned char> chosen;
  if (chosen.size() < static_cast<std::size_t>(n)) {
    chosen.resize(static_cast<std::size_t>(n));
  }

  for (int f = 0; f < frames; ++f) {
    RgbaF* frame = batch + static_cast<std::size_t>(f) * pixels;
    for (int b = 0; b < nblocks; ++b) {
      // Each block starts from a zeroed queue, exactly as the kernel does, so the
      // host and device produce identical output.
      RgbaD error[kErrorQueueLength];
      std::memset(error, 0, sizeof(error));
      const int begin = b * block;
      const int end = (begin + block < n) ? (begin + block) : n;
      for (int i = begin; i < end; ++i) {
        RgbaD pixel;
        associate_alpha_pixel(palette.associate_alpha, frame[curve[i]], &pixel);
        for (int k = 0; k < kErrorQueueLength; ++k) {
          const double w = kErrorRelativeWeight * params.diffusion * weights[k];
          pixel.r += w * error[k].r;
          pixel.g += w * error[k].g;
          pixel.b += w * error[k].b;
          if (palette.associate_alpha) pixel.a += w * error[k].a;
        }
        pixel.r = static_cast<double>(clamp_pixel(pixel.r));
        pixel.g = static_cast<double>(clamp_pixel(pixel.g));
        pixel.b = static_cast<double>(clamp_pixel(pixel.b));
        if (palette.associate_alpha) {
          pixel.a = static_cast<double>(clamp_pixel(pixel.a));
        }
        // Cache-free on both paths, so the two agree; see rd_blocks_cuda.cu.
        std::int32_t node = tree.root();
        for (int i2 = kMaxTreeDepth - 1; i2 > 0; --i2) {
          const std::size_t id = tree.NodeId(pixel, i2);
          const std::int32_t child = tree.Child(node, id);
          if (child < 0) break;
          node = child;
        }
        double distance =
            4.0 * ((kQuantumRange + 1.0) * (kQuantumRange + 1.0)) + 1.0;
        int index = 0;
        tree.ClosestColor(tree.Parent(node), pixel, &distance, &index);
        chosen[i] = static_cast<unsigned char>(index);
        for (int k = 0; k < kErrorQueueLength - 1; ++k) {
          error[k] = error[k + 1];
        }
        RgbaD color;
        associate_alpha_info(palette.associate_alpha, palette.entries[index],
                             &color);
        error[kErrorQueueLength - 1].r = pixel.r - color.r;
        error[kErrorQueueLength - 1].g = pixel.g - color.g;
        error[kErrorQueueLength - 1].b = pixel.b - color.b;
        if (palette.associate_alpha) {
          error[kErrorQueueLength - 1].a = pixel.a - color.a;
        }
      }
    }
    // Scatter: last writer wins, matching the kernel's ownership map.
    for (int i = 0; i < n; ++i) {
      const int pixel_index = curve[i];
      if (owner[pixel_index] != i) continue;
      const PaletteEntry& e = palette.entries[chosen[i]];
      if (!fused) {
        RgbaF& out = frame[pixel_index];
        out.r = clamp_to_quantum(e.r);
        out.g = clamp_to_quantum(e.g);
        out.b = clamp_to_quantum(e.b);
        if (palette.associate_alpha) out.a = clamp_to_quantum(e.a);
        continue;
      }
      // Fused: FloatsToYuv444's arithmetic, on the same float the old path would have
      // stored first.  `clamp_to_quantum` is a bare narrowing cast and the convert pass
      // then did `static_cast<int>(that float) >> 8`, so this is bit-for-bit the same
      // two operations in the same order -- not a shortcut around them.
      const float fr = clamp_to_quantum(e.r);
      const float fg = clamp_to_quantum(e.g);
      const float fb = clamp_to_quantum(e.b);
      unsigned char* o = raw_out + static_cast<std::size_t>(f) * out_pixels * 3;
      if (out_yuv444) {
        const int r8 = static_cast<int>(fr) >> 8;
        const int g8 = static_cast<int>(fg) >> 8;
        const int b8 = static_cast<int>(fb) >> 8;
        const int yv = ((66 * r8 + 129 * g8 + 25 * b8 + 128) >> 8) + 16;
        const int uv = ((-38 * r8 - 74 * g8 + 112 * b8 + 128) >> 8) + 128;
        const int vv = ((112 * r8 - 94 * g8 - 18 * b8 + 128) >> 8) + 128;
        o[pixel_index] = static_cast<unsigned char>(yv < 0 ? 0 : (yv > 255 ? 255 : yv));
        o[out_pixels + pixel_index] =
            static_cast<unsigned char>(uv < 0 ? 0 : (uv > 255 ? 255 : uv));
        o[2 * out_pixels + pixel_index] =
            static_cast<unsigned char>(vv < 0 ? 0 : (vv > 255 ? 255 : vv));
      } else {
        // FloatsToRaw's four uint16.  Alpha is NOT taken from the palette entry here:
        // that pass read `src[i].a` out of the frame buffer, which for a visited pixel
        // still holds whatever the reader put there.  With associate_alpha off the
        // scatter never wrote .a, so the encoder has always received the SOURCE alpha
        // here.  Reproducing that is deliberate -- changing it would change output, and
        // the YUV444 writer ignores alpha entirely so the two writers cannot both be
        // "fixed" in one change.
        std::uint16_t* u = reinterpret_cast<std::uint16_t*>(raw_out) +
                           static_cast<std::size_t>(f) * out_pixels * 4;
        const RgbaF& src = frame[pixel_index];
        u[4 * pixel_index + 0] = static_cast<std::uint16_t>(fr);
        u[4 * pixel_index + 1] = static_cast<std::uint16_t>(fg);
        u[4 * pixel_index + 2] = static_cast<std::uint16_t>(fb);
        u[4 * pixel_index + 3] = palette.associate_alpha
                                     ? static_cast<std::uint16_t>(clamp_to_quantum(e.a))
                                     : static_cast<std::uint16_t>(src.a);
      }
    }

    // The pixels the curve never reaches.  The scatter above is FILTERED by
    // `owner[pixel_index] == i`, so it does not write them -- and the convert pass this
    // fusion deletes used to cover them only because it swept the whole buffer.  So
    // they must be written here, from the source value the reader left in `frame`,
    // which is what ImageMagick's own recursion leaves there and what
    // probe-unvisited-pixel.ps1 asserts.  Skipping this is the failure the fusion
    // invites: stale heap on the first batch and the PREVIOUS batch's pixels after,
    // which is the same shape as the bug at rd_blocks_cuda.cu:612.
    //
    // Driven off `cached.unvisited`, a list rather than an index, so a future
    // ComputeCurveLevel that produces more than one gap cannot silently leave the rest
    // unwritten.  Four measured geometries have exactly one (1024x768, 1024x1024,
    // 2048x2048, 4096x2160); the other six have none.
    if (fused) {
      for (std::int32_t p : cached.unvisited) {
        const std::size_t pixel_index = static_cast<std::size_t>(p);
        const RgbaF& s = frame[pixel_index];
        unsigned char* o = raw_out + static_cast<std::size_t>(f) * out_pixels * 3;
        if (out_yuv444) {
          const int r8 = static_cast<int>(clamp_to_quantum(s.r)) >> 8;
          const int g8 = static_cast<int>(clamp_to_quantum(s.g)) >> 8;
          const int b8 = static_cast<int>(clamp_to_quantum(s.b)) >> 8;
          const int yv = ((66 * r8 + 129 * g8 + 25 * b8 + 128) >> 8) + 16;
          const int uv = ((-38 * r8 - 74 * g8 + 112 * b8 + 128) >> 8) + 128;
          const int vv = ((112 * r8 - 94 * g8 - 18 * b8 + 128) >> 8) + 128;
          o[pixel_index] = static_cast<unsigned char>(yv < 0 ? 0 : (yv > 255 ? 255 : yv));
          o[out_pixels + pixel_index] =
              static_cast<unsigned char>(uv < 0 ? 0 : (uv > 255 ? 255 : uv));
          o[2 * out_pixels + pixel_index] =
              static_cast<unsigned char>(vv < 0 ? 0 : (vv > 255 ? 255 : vv));
        } else {
          std::uint16_t* u = reinterpret_cast<std::uint16_t*>(raw_out) +
                             static_cast<std::size_t>(f) * out_pixels * 4;
          u[4 * pixel_index + 0] = static_cast<std::uint16_t>(clamp_to_quantum(s.r));
          u[4 * pixel_index + 1] = static_cast<std::uint16_t>(clamp_to_quantum(s.g));
          u[4 * pixel_index + 2] = static_cast<std::uint16_t>(clamp_to_quantum(s.b));
          u[4 * pixel_index + 3] = static_cast<std::uint16_t>(s.a);
        }
      }
    }
  }
}

// The curve for one frame geometry, for a port of this engine to another
// accelerator.  The OpenCL engine needs the visit order and the ownership map
// verbatim: the order is the walk, and the owner map is what makes the scatter
// correct when the curve is not a permutation (a 1920x1080 frame is not a power
// of two, so the curve reaches some pixels twice and the last visit wins).
//
// Exposed as two pointers into the same thread-local single-entry cache, for the
// same reason that cache is a single entry: it is a large allocation rebuilt from
// a closed form, and the CUDA engine wants the identical data.
const std::vector<int>* BlockCurveOrder(std::size_t width, std::size_t height) {
  const CachedCurve& c =
      CachedCurveFor(ComputeCurveLevel(width, height), width, height);
  return &c.index;
}

const std::vector<int>* BlockCurveOwner(std::size_t width, std::size_t height) {
  const CachedCurve& c =
      CachedCurveFor(ComputeCurveLevel(width, height), width, height);
  return &c.owner;
}

}  // namespace rd
