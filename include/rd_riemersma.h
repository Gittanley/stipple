// SPDX-License-Identifier: GPL-3.0-or-later
// rd_riemersma.h -- bit-exact reimplementation of ImageMagick 7.1.2-31
// Riemersma dithering for Q16-HDRI, plus the engine entry points.
//
// ---------------------------------------------------------------------------
// Why this can be bit-exact without reimplementing IM's colour octree
// ---------------------------------------------------------------------------
// quantize.c does the palette lookup in ClosestColor(), reached as
//
//     ClosestColor(image,p,node_info->parent)
//
// so the candidate set is the leaves carrying data *under one particular node*,
// not all N palette entries.  A plain nearest-colour scan over the palette is
// therefore NOT equivalent: it agrees on greyscale (where the cube collapses)
// and disagrees on colour.  rd_octree.h carries the transliterated tree so the
// subtree and the post-order traversal are reproduced, and the palette that
// DefineImageColormap() derives from it is checked against ImageMagick's own
// colormap on every run.
//
// The error-queue diffusion itself is then an exact transliteration of
// Riemersma() + RiemersmaDither().
//
// ---------------------------------------------------------------------------
// Parallelism
// ---------------------------------------------------------------------------
// Riemersma diffusion is a strictly sequential walk: a single 16-entry error
// queue, a single (x,y) cursor, each step consuming the previous step's
// residual.  No intra-image parallelisation can preserve bit-exactness.  The
// only exact scaling axes are (a) building the colour tree, which is a
// data-parallel histogram-style pass, and (b) running many independent walks at
// once -- i.e. frames of a video, each with its own error queue and its own memo
// table, which is bit-exact per frame.  RiemersmaWalkCuda is already
// parameterised by frame count for that reason.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rd_octree.h"
#include "rd_source.h"
#include "rd_types.h"

namespace rd {

// Fraction of *physical* RAM the video pipeline may spend on in-flight frame
// buffers.  Set from --mem-fraction; 0 means auto.
extern double g_video_memory_fraction;

// Gravity directions, matching the subset ImageMagick uses for the curve.
enum class Gravity : int { kWest = 0, kEast = 1, kNorth = 2, kSouth = 3 };

// The exact output palette: ImageMagick's own colormap after
// QuantizeImage(RiemersmaDitherMethod, number_colors).
struct Palette {
  PaletteEntry entries[kMaxColormapSize] = {};
  int count = 0;
  bool associate_alpha = false;
};

struct DitherParams {
  int colors = 16;
  double diffusion = 1.0;   // artifact "dither:diffusion-amount", IM default 1.0
  // quantize.c memoises the palette lookup behind a 6-bit-per-channel key, so
  // the first target to touch a key fixes the answer for every later target
  // sharing it.  Disabling it isolates that order dependence from the error
  // introduced by parallelising the walk; it is *not* a mode you would ship,
  // because it no longer matches ImageMagick.
  bool use_cache = true;
};

// ---------------------------------------------------------------------------
// Hilbert visit sequence
// ---------------------------------------------------------------------------
// A direct transliteration of quantize.c:Riemersma().  The recursion is turned
// into an explicit stack so the same order can be produced either
// sequentially (CPU) or by a parallel kernel (CUDA).
//
// `steps` must hold 4^level entries; `level` is computed exactly as IM does.
inline int ComputeCurveLevel(std::size_t width, std::size_t height) {
  const std::size_t extent = width > height ? width : height;
  if (extent == 0) return 0;
  int level = static_cast<int>(std::log2(static_cast<double>(extent)));
  if ((static_cast<std::size_t>(1) << level) < extent) ++level;
  return level;
}

// Returns true when step `stage` of (level, dir) is a dither visit, writing the
// direction that the visit moves in on the way out.  When it returns false the
// step is a recursive descent into (level-1, out_dir).
inline bool CurveStep(int level, Gravity dir, int stage,
                      Gravity* out_dir) noexcept {
  // dir encoding: 0=W 1=E 2=N 3=S.  Value >= 0 marks a dither visit,
  // value < 0 marks a recursive descent into -value-1.
  static constexpr signed char kLeaf[4][3] = {
      {+1, +3, +0},  // West : E, S, W
      {+0, +2, +1},  // East : W, N, E
      {+3, +1, +2},  // North: S, E, N
      {+2, +0, +3},  // South: N, W, S
  };
  static constexpr signed char kNode[4][7] = {
      {-3, +1, -1, +3, -1, +0, -4},  // West : R(N) D(E) R(W) D(S) R(W) D(W) R(S)
      {-4, +0, -2, +2, -2, +1, -3},  // East : R(S) D(W) R(E) D(N) R(E) D(E) R(N)
      {-1, +3, -3, +1, -3, +2, -2},  // North: R(W) D(S) R(N) D(E) R(N) D(N) R(E)
      {-2, +2, -4, +0, -4, +3, -1},  // South: R(E) D(N) R(S) D(W) R(S) D(S) R(W)
  };
  const signed char* table = (level == 1) ? kLeaf[static_cast<int>(dir)]
                                         : kNode[static_cast<int>(dir)];
  const int count = (level == 1) ? 3 : 7;
  if (stage >= count) return false;
  const signed char e = table[stage];
  *out_dir = static_cast<Gravity>(e < 0 ? (-e - 1) : e);
  return e >= 0;
}

// One entry of the flattened visit sequence.
struct CurveStepEntry {
  int x = 0;
  int y = 0;
  int dir = 0;   // Gravity applied *after* the visit
  int dir_next = -1;  // direction of the following visit (informational)
};

// quantize.c:DitherImage() performs the Riemersma() recursion and then one
// final RiemersmaDither(ForgetGravity) visit at wherever the cursor landed.
// ForgetGravity advances nothing, so the trailing entry's `dir` is kNoMove.
inline constexpr int kNoMove = -1;

// Number of entries produced for a given level.
//
// The recursion is V(1) = 3 dither visits and V(L) = 4*V(L-1) + 3, i.e.
// V(L) = 4^L - 1 -- *not* 4^L.  Adding the trailing ForgetGravity visit gives
// 4^L entries in total, which is also the cell count of the 2^level grid.  The
// leaves cover every cell except the grid's *last* one; ForgetGravity then
// visits the origin, which was already the first leaf.  So the grid's last cell
// is never dithered at all -- which is why a 1920x1080 frame at level 11 leaves
// exactly one pixel at its source value, and why ImageMagick itself does.
inline std::size_t CurveEntryCount(int level) {
  return (level <= 0) ? 0 : (static_cast<std::size_t>(1) << (2 * level));
}

// Fills `out` with exactly CurveEntryCount(level) entries in ImageMagick's
// visit order.  `out` must have room for CurveEntryCount(level) elements.
void BuildCurveSequence(int level, std::size_t width, std::size_t height,
                        CurveStepEntry* out);

// ---------------------------------------------------------------------------
// Compacted visit order, built without materialising the 4^level entry list
// ---------------------------------------------------------------------------
// The full traversal is 4^level entries, but only width*height of them are
// in bounds.  Materialising the full list costs 4^level * 16 bytes -- 67 MB for
// a 1920x1080 frame at level 11 -- and dominated the engine's wall clock (254 ms
// of a 374 ms single-shot run).  BuildCurveIndex walks the same recursion but
// emits only the in-bounds positions straight into a pre-sized vector.
//
// `owner[pixel]` receives the curve position that writes that pixel last, which
// is what makes a parallel scatter deterministic (DitherImage's trailing
// ForgetGravity visit can land on an already-visited cell).
//
// Returns the number of in-bounds positions.
std::size_t BuildCurveIndex(int level, std::size_t width, std::size_t height,
                            std::vector<int>* out, std::vector<int>* owner);

// Closed-form Hilbert point (the transpose of the textbook d2xy, which is what
// quantize.c's East-first recursion generates).
void HilbertPoint(std::int64_t n, int level, int* out_x, int* out_y);

// Number of positions where the closed form and the recursion disagree at this
// level.  Zero means they are identical; --self-test reports it for levels 1..8.
std::size_t SelfTestCurveMatchesRecursion(int level);

// Same, after applying dihedral transform `transform` (0..7) to d2xy's output.
std::size_t CurveTransformMismatches(int level, int transform);

// ---------------------------------------------------------------------------
// Engines
// ---------------------------------------------------------------------------
enum class Engine { kCpu, kCuda, kApprox, kBlocks, kOpenCL };

// Dithers `pixels` in place, exactly as
// `magick in.png -dither Riemersma -colors N out.png` would.
//
// `tree` supplies the octree that ClosestColor() walks; its colormap must equal
// `palette` (the CLI asserts this).  `curve` is optional: when non-null it must
// hold CurveEntryCount(level) entries and is consumed instead of the stack walk.
// `width`/`height` must match the store.
void RiemersmaWalkCpu(const Palette& palette, const DitherParams& params,
                      const ColorTree& tree, std::size_t width,
                      std::size_t height, RgbaF* pixels,
                      const CurveStepEntry* curve);

// Same semantics, executed on the GPU.  `frames` copies of the same geometry
// are dithered concurrently (Phase 2 uses this for video, where each frame
// keeps an independent error queue and stays bit-exact).  For Phase 1 pass 1.
//
// Returns an empty string on success, otherwise a human readable error.
std::string RiemersmaWalkCuda(const Palette& palette, const DitherParams& params,
                              const ColorTree& tree, std::size_t width,
                              std::size_t height, RgbaF* pixels, int frames,
                              std::string* device_name);

// Same partition and the same arithmetic as RiemersmaBlocksCuda, on the host.
// This is what makes dynamic load balancing between CPU workers and the GPU
// invisible: a frame dithered on either path comes out identical, so frames may
// be handed to whichever engine is free without introducing a visible seam.
// `batch` holds `frames` consecutive frames of width*height RGBA-float pixels.
void RiemersmaBlocksCpu(const Palette& palette, const DitherParams& params,
                        const ColorTree& tree, std::size_t width,
                        std::size_t height, RgbaF* batch, int frames,
                        int block);

// The Hilbert visit order for a frame, and the map from a pixel back to the visit
// that owns it.  Exposed so a port of this engine to another accelerator (see
// rd_opencl.h) uses the identical order rather than a recomputed one: the order
// *is* the walk, and the owner map is what makes a scatter correct when the curve
// is not a permutation of the frame -- which at 1920x1080 it is not.
//
// Both point into the same thread-local single-entry cache, so they must be taken
// together and used before anything else asks for a different geometry.
const std::vector<int>* BlockCurveOrder(std::size_t width, std::size_t height);
const std::vector<int>* BlockCurveOwner(std::size_t width, std::size_t height);

// Fraction of physical RAM the pipeline may use for in-flight frame buffers.
// 0 means "decide automatically" (about a third of physical RAM).
// True when a usable CUDA device is present.
bool CudaAvailable();

// Number of whole frames that fit the device given the current geometry.
int CudaMaxFrames(std::size_t width, std::size_t height, std::size_t vram_budget);

// ---------------------------------------------------------------------------
// Approximate engine (not bit-exact by construction -- see src/rd_approx_cuda.cu)
// ---------------------------------------------------------------------------
// The sequential result is a fixed point of an order-free iteration, so this is
// an exact solver with a convergence question rather than a heuristic.  It
// converges to the *cache-free* fixed point, which is why its AE against
// ImageMagick has a non-zero floor that no amount of extra sweeps will remove.
struct ApproxOptions {
  int iterations = 20;
  int taps = 96;         // length of the truncated inverse filter (I-H)^-1
  bool fp64 = false;     // double the convolution as well as the state
  bool use_tree = true;  // octree-restricted lookup (matches IM) vs plain scan
};

std::string RiemersmaApproxCuda(const Palette& palette, const DitherParams& params,
                                const ColorTree& tree, std::size_t width,
                                std::size_t height, RgbaF* pixels,
                                const ApproxOptions& options,
                                std::string* device_name);

// Exposed so the coefficient schedule can be inspected and unit-tested.
std::vector<double> ApproxInverseFilter(int taps, double diffusion);

// ---------------------------------------------------------------------------
// Block-parallel engine -- single pass, no iteration, for the visual goal
// ---------------------------------------------------------------------------
// The error queue is exactly 16 deep, so a block of B >= 16 positions is
// bit-identical to the sequential walk from its 16th position onward.  Only the
// first 16 positions of each block can be perturbed by a wrong incoming state,
// and the perturbation is bounded by (G/(1-G)) * Delta/2 ~ 0.27 Delta.  Running
// every block from a zeroed queue is therefore a bounded, *localised* defect --
// a different speckle at the seams -- not banding and not global drift.
//
// This is the engine to use when the goal is the Riemersma look rather than
// byte-equality: one pass, no scan, no iteration, and B can be raised until the
// seams are statistically invisible because the device is nowhere near saturated.
struct BlockOptions {
  // Positions per block; must be >= 16.
  //
  // FASTER IS NOT THE SAME AS BETTER, and 32 is the counter-example.  Measured on
  // 1000 frames of 1080p, interleaved, dither stage only (rep-to-rep spread 20-40 ms,
  // so these gaps are not noise):
  //
  //     32 -> 10373 ms      256 -> 11224 ms
  //    128 -> 11652 ms     512 -> 11656 ms
  //
  // So 32 is 11% faster than 512.  And against ImageMagick's own output, on a
  // 200x150 gradient, `magick -colors 16 -dither Riemersma`:
  //
  //     B=32   1819 of 30001 pixels differ (6.1%), RMSE 0.0239
  //     B=512   501 of 30001 pixels differ (1.7%), RMSE 0.0107
  //
  // 512 is 2.2x closer to ImageMagick.  The mechanism is in the name: a block is a
  // walk that restarts its error queue from zero, so a smaller block means more restarts
  // and less error carried between distant pixels.  Bigger blocks carry error further
  // and land nearer the reference; the reference itself is one uncut walk.
  //
  // So 512 is the default because it is the faithful one, and `--blocks 32` is a
  // documented opt-in for anyone who would rather have 11% on the dither than 3.4x
  // the deviation.  On a 3-hour clip the dither is ~48% of the wall, so 11% of it is
  // about 5% end to end.
  //
  // Note the 135-case bit-exactness suite does NOT cover this: it tests the default
  // *image* engine (cpu/cuda), which is the uncut walk and is bit-exact at any block
  // size.  The blocks engine is approximate by construction and no test guards it.
  int block = 512;
  int frames = 1;    // distinct frames in the batch, each an independent walk
  // When set, the caller also wants the result as packed rgba64le in `out_u16`
  // (4 uint16 per pixel, little-endian order, which is exactly what the video
  // pipe consumes).  The device then writes the quantised colour straight into a
  // uint16 buffer, which halves the download and lets the host skip the
  // float->uint16 conversion entirely -- together 3649 ms of 605 frames plus a
  // third of the per-batch transfer.  Left clear, the float4 buffer is written as
  // before, which is what the single-image path and verify.ps1 use.
  bool emit_u16 = false;
  // When set, the device writes planar 8-bit 4:4:4 (3 bytes per pixel) instead of
  // rgba64le, doing the RGB->YUV itself, and `out_u16` is that byte buffer rather
  // than a uint16 one.  ffmpeg would otherwise convert the frames on the CPU, and
  // that conversion -- not the pipe -- is what the encoder stage ends up waiting on.
  // Requires emit_u16, and changes the caller's out_frame_bytes to pixels * 3.
  bool emit_yuv444 = false;
  // When set, the batch arrives as packed rgba64le in `in_u16` rather than as
  // float4 in `batch`, and the gather kernel widens it on the device.  This deletes
  // the caller's uint16 -> float loop, which on a 605-frame 1080p clip is two billion
  // conversions and ~48 GB of host memory traffic, and it halves the H2D.  The
  // widening is exact -- a uint16 always fits a float's mantissa -- so the dither sees
  // bit-identical input either way.
  bool upload_u16 = false;
  // Channels per pixel in `in_u16`: 3 for rgb48le, 4 for rgba64le.  The caller
  // chooses 3 only when the source carries no alpha, because the kernel then
  // synthesises the 65535 that rgba64le would have delivered.  Must be 3 or 4.
  int in_channels = 4;
  // How `in_u16`'s buffer is to be interpreted when `upload_u16` is set.
  enum class InMode {
    // Interleaved uint16, 3 or 4 per pixel: rgba64le / rgb48le straight from the
    // decoder.  Widened with a cast, which is exact.  The default, and what every
    // round-to-date has used.
    Interleaved16 = 0,
    // Planar 8-bit 4:4:4: a Y plane, then Cb, then Cr, three bytes per pixel.  The
    // gather converts YCbCr to RGB itself, which is where the speed comes from --
    // 3 bytes per pixel instead of 8, so 2.7x less crosses the pipe and through
    // the H2D.
    //
    // This is NOT the same picture as the interleaved path, and the reason is
    // specific: 4:2:0 carries chroma at half resolution, so swscale's yuv420p ->
    // RGB conversion reconstructs chroma by *interpolating* it, while 4:4:4 carries
    // the real per-pixel chroma.  In flat areas the two agree closely; at edges and
    // fine colour detail they diverge substantially, so this trades a little accuracy
    // for sharpness.  It is also 8-bit RGB, so the values are multiples of 257 where
    // the interleaved path is not.
    PlanarYuv444 = 1,
    // Same input, but the conversion is split into two kernels: a coalesced
    // whole-frame YCbCr->RGB pass writing a small intermediate, then the ordinary
    // curve gather reading that.  Differs from PlanarYuv444 only by <=1 per sample
    // (65 dB), because the fused kernel stores r*257 as a float while this stores
    // it truncated to uint16.
    //
    // It was built to test a specific hypothesis -- that the fused kernel's 71%
    // overhead (399 vs 233 ms per launch, despite moving 2.7x *less* data) came from
    // byte-granular scattered reads, since the gather walks pixels in Riemersma curve
    // order.  **The hypothesis was wrong.**  Splitting the work so the bulk transform
    // is fully coalesced measured 398.5 ms against the fused 399.3 ms: no change at
    // all, and still 165 ms worse than the interleaved path while doing strictly more
    // work.  So the overhead is not the access pattern, and the real cause is still
    // unknown.  Kept because the negative result is worth being able to re-check, and
    // because it is harmless -- but the yuv444 line does not pay for itself either
    // way, since the reader's saving is given back here.
    PlanarYuv444Prepass = 2,
    // Planar 8-bit 4:2:0, 1.5 bytes per pixel: Y at full resolution, then half-width
    // Cb and Cr.  The decoder emits the source format untouched, so swscale does no
    // conversion at all and the device reconstructs chroma itself.
    //
    // This is the smallest input the pipeline can carry, and unlike the modes above
    // it is the only one where ffmpeg's colour conversion disappears from the CPU
    // entirely rather than merely shrinking.  Measured on the 18001-frame clip:
    // decoding to yuv444p takes 77.0 s, to yuv420p 33.0 s, and the pipe carries 52 GB
    // instead of 104.
    //
    // The *palette* is unaffected: the palette samplers run their own ffmpeg with
    // their own `-pix_fmt rgba`, and never consult this field, so the palette stays
    // byte-identical to the yuv444 path's.  What changes is the dither's input chroma:
    // this interpolates with a 2x bilinear filter where swscale used a wide FIR, so
    // the dithered picture moves by a small amount at hard chroma edges.
    PlanarYuv420 = 3,
  };
  InMode in_mode = InMode::Interleaved16;
  // `in_u16` is already page-locked (see CudaAllocPinned), so the engine copies it
  // straight to the device and skips its own staging memcpy.
  bool in_u16_pinned = false;
  // Likewise `out_u16`: a pinned destination makes the download a true async DMA
  // instead of one through the driver's internal bounce buffer.
  bool out_u16_pinned = false;
  // `batch` is already page-locked; skip staging memcpy.
  bool batch_pinned = false;
};

// `batch` holds `options.frames` consecutive frames of width*height RGBA-float
// pixels, and is dithered in place.  Every frame gets its own error state, so a
// batch is bit-exact per frame relative to a single-frame run.  The work items
// are (frame, block) pairs, which is what gives the batch its parallelism:
// independent walks, not shared work.
std::string RiemersmaBlocksCuda(const Palette& palette, const DitherParams& params,
                                const ColorTree& tree, std::size_t width,
                                std::size_t height, RgbaF* batch,
                                const BlockOptions& options,
                                std::string* device_name,
                                std::uint16_t* out_u16 = nullptr,
                                // Which independent device state to use.  Each GPU
                                // worker takes its own, so two launches can be in
                                // flight at once; using the same value from two
                                // threads is safe but serialises them.
                                int state_slot = 0,
                                // The batch as packed rgba64le, when
                                // BlockOptions::upload_u16 is set.  `batch` is then
                                // unused and may be null.
                                const std::uint16_t* in_u16 = nullptr);

// How many independent device states the CUDA module keeps.  A worker pool larger
// than this would have workers sharing a state and serialising behind it.
constexpr int kGpuStateSlots = 2;

// Page-locked (pinned) host memory for the video pipeline's batch buffers.
//
// Pinned is what makes an async DMA worthwhile, and it is also what lets the reader
// deposit decoded frames *directly* into the buffer the copy engine will read: without
// it, every frame is read into pageable memory, memcpy'd into staging, and only then
// uploaded -- 20 GB of host-to-host copy over a 605-frame 1080p clip, sitting on the
// GPU worker's critical path and achieving nothing.
//
// Pinned memory cannot be paged out, so the RAM budget has to count it; see
// Pipeline's budget and the [ram] line it prints.
void* CudaAllocPinned(std::size_t bytes);
void CudaFreePinned(void* p);

}  // namespace rd
