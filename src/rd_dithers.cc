// rd_dithers.cc -- the error-diffusion dither algorithms.
//
// This file, plus examples\bayer_dither.cc and the void-and-cluster generator in
// rd_dither_voidcluster.cc, is the set of algorithms reachable through --dither
// other than the built-in Riemersma walk.  They are plugins (see rd_plugin.h), not
// engine variants: they compose with the palette builder, the video pipeline and
// crash resume, which is the entire point of the seam.
//
// WHY THESE AND NOT OTHERS.  The built-in reproduces ImageMagick's Riemersma
// exactly, and that is a narrow, valuable thing.  Everything around it -- palette
// from ImageMagick, the traversal curve, the block partitioner, the concurrent
// decode/dither/encode pipeline, the checkpoints -- is independent of which kernel
// you feed it, and none of that work should be locked behind one algorithm.  So
// these are the ones people actually reach for when they want a different look
// rather than a different implementation of the same one.
//
// NOTHING HERE IS BIT-EXACT WITH ANYTHING, and none of it is trying to be.  The
// correct reference for these is a perceptually sensible result, not a matching
// file.  That is a different contract from the Riemersma engine's and it is worth
// stating, because "it differs from ImageMagick" is a bug report for the built-in
// and not for these.
//
// THE THREE RULES, from rd_plugin.h, restated because they are the ones that fail:
//
//   * per-frame error state.  One buffer per frame, reset at the frame's start.
//     Sharing one across a batch makes frame f depend on frame f-1, which breaks
//     the parallel block partition and makes --resume produce a different file.
//   * never read outside the frame you were given.  Frames are contiguous, so
//     overrun reads the next frame and corrupt it silently.
//   * deterministic.  No randomness, or randomness from a fixed seed.  The video
//     path re-runs from a checkpoint and frame N has to come out the same twice.
//
// The nearest-colour search, the error-plane layout and the trailing quantise live
// in rd_dither_common.h, so all of these agree on what "the closest palette entry"
// means.  One copy on purpose: two copies that disagreed would make two algorithms
// that should agree on flat colour pick different entries.

#include "rd_dither_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace rd {
namespace {

using dither_common::Accumulate;
using dither_common::ErrorPlane;
using dither_common::NearestEntry;
using dither_common::Prepare;
using dither_common::WriteChosen;

// ---------------------------------------------------------------------------
// Floyd-Steinberg
// ---------------------------------------------------------------------------
//
// The 1966 kernel, and the one every other diffusion dither is described relative
// to.  The error goes to four neighbours:
//
//        *  7/16
//     3/16 5/16 1/16
//
// All the weights sum to 1, which is the point: the diffusion neither creates nor
// destroys error, it only moves it.  A kernel whose weights do not sum to 1 (or to
// some other deliberate constant) changes the image's overall level as it goes, and
// the classic symptom is a slow darkening towards the bottom of the frame.
//
// Serpentine scanning is NOT used.  It genuinely reduces the artefacts, and it is
// also why Floyd-Steinberg output depends on scan direction in a way that is
// awkward to reason about; the raster order here matches the rest of the pipeline,
// and matches what a plugin author would write without thinking about it.
DitherResult FloydSteinberg(DitherJob& job, RgbaF* px) {
  DitherResult res;
  const Palette* pal_ptr = nullptr;
  if (!Prepare(res, "floyd-steinberg", job, pal_ptr)) return res;
  const Palette& pal = *pal_ptr;  // by reference: see rd_dither_common.h
  const std::size_t w = job.width;
  const std::size_t n = w * job.height;

  ErrorPlane e;
  for (int f = 0; f < job.frames; ++f) {
    e.Reset(n);
    RgbaF* frame = px + static_cast<std::size_t>(f) * n;
    for (std::size_t y = 0; y < job.height; ++y) {
      const std::size_t row = y * w;
      for (std::size_t x = 0; x < w; ++x) {
        const std::size_t i = row + x;
        // Read the accumulated error BEFORE choosing, which is the whole contract
        // of a diffusion kernel.  Reading it after the neighbour writes land is the
        // single most common way to get these wrong.
        const float r = frame[i].r + e.r[i] * static_cast<float>(job.diffusion);
        const float g = frame[i].g + e.g[i] * static_cast<float>(job.diffusion);
        const float b = frame[i].b + e.b[i] * static_cast<float>(job.diffusion);
        const int best = NearestEntry(pal, r, g, b);
        const PaletteEntry& c = pal.entries[best];
        const float er = r - static_cast<float>(c.r);
        const float eg = g - static_cast<float>(c.g);
        const float eb = b - static_cast<float>(c.b);
        Accumulate(e, n, i + 1, 7.0f / 16.0f, 7.0f / 16.0f, 7.0f / 16.0f, er, eg, eb);
        Accumulate(e, n, i + w - (x > 0 ? 1 : 0), 3.0f / 16.0f, 3.0f / 16.0f, 3.0f / 16.0f, er, eg, eb);
        Accumulate(e, n, i + w + 1, 5.0f / 16.0f, 5.0f / 16.0f, 5.0f / 16.0f, er, eg, eb);
        if (x + 2 < w) {
          Accumulate(e, n, i + w + 2, 1.0f / 16.0f, 1.0f / 16.0f, 1.0f / 16.0f, er, eg, eb);
        }
        WriteChosen(frame[i], c);
      }
    }
  }
  return res;
}

// ---------------------------------------------------------------------------
// Atkinson
// ---------------------------------------------------------------------------
//
// Michael Atkinson's 1986 simplification: six neighbours, every weight 1/8.
//
//        *  1/8  1/8
//     1/8  1/8  1/8  1/8
//
// The weights sum to 6/8 = 0.75, not 1, and that is deliberate rather than sloppy:
// Atkinson discards a quarter of the error on the floor, which lightens the texture
// considerably at the cost of some accuracy.  It was designed for the Apple II, with
// one-byte-per-channel arithmetic in mind, and it is still the usual answer to "my
// dithering looks too contrasty".
//
// The discarded quarter is why this is kept as its own algorithm rather than being
// presented as a Floyd-Steinberg variant with different constants: renormalising the
// six weights to sum to 1 is a one-character change that produces a visibly
// different, and wrong, Atkinson.
DitherResult Atkinson(DitherJob& job, RgbaF* px) {
  DitherResult res;
  const Palette* pal_ptr = nullptr;
  if (!Prepare(res, "atkinson", job, pal_ptr)) return res;
  const Palette& pal = *pal_ptr;
  const std::size_t w = job.width;
  const std::size_t n = w * job.height;
  const float k = 1.0f / 8.0f;

  ErrorPlane e;
  for (int f = 0; f < job.frames; ++f) {
    e.Reset(n);
    RgbaF* frame = px + static_cast<std::size_t>(f) * n;
    for (std::size_t y = 0; y < job.height; ++y) {
      const std::size_t row = y * w;
      for (std::size_t x = 0; x < w; ++x) {
        const std::size_t i = row + x;
        const float d = static_cast<float>(job.diffusion);
        const float r = frame[i].r + e.r[i] * d;
        const float g = frame[i].g + e.g[i] * d;
        const float b = frame[i].b + e.b[i] * d;
        const int best = NearestEntry(pal, r, g, b);
        const PaletteEntry& c = pal.entries[best];
        const float er = r - static_cast<float>(c.r);
        const float eg = g - static_cast<float>(c.g);
        const float eb = b - static_cast<float>(c.b);
        // Right, then the three below, then down-left and down-right.  No
        // same-row-left and no two-rows-down: those are exactly the two Atkinson
        // drops relative to Floyd-Steinberg.
        Accumulate(e, n, i + 1, k, k, k, er, eg, eb);
        if (x > 0) Accumulate(e, n, i + w - 1, k, k, k, er, eg, eb);
        Accumulate(e, n, i + w, k, k, k, er, eg, eb);
        if (x + 1 < w) Accumulate(e, n, i + w + 1, k, k, k, er, eg, eb);
        if (y + 1 < job.height) {
          Accumulate(e, n, i + 2 * w, k, k, k, er, eg, eb);
          if (x + 1 < w) {
            Accumulate(e, n, i + 2 * w + 1, k, k, k, er, eg, eb);
          }
        }
        WriteChosen(frame[i], c);
      }
    }
  }
  return res;
}

// ---------------------------------------------------------------------------
// Jarvis, Judice and Ninke
// ---------------------------------------------------------------------------
//
// Twelve neighbours in four rows, with the weights 7/48, 5/48, 3/48, 1/48 falling
// off with distance.  They sum to 1.
//
//    1     3     5     1              the top row is 48 pixels back
//    7     5     7     5              two rows back
//    3     1     3     1              one row back
//        7     5     7                 same row
//
// The name is three people, not a typo, and it is the most widely used of the
// "many-neighbour" kernels.  It gives markedly smoother output than
// Floyd-Steinberg at the cost of blurring fine detail, and it is the right choice
// for gradients and photographic content rather than for line art.
//
// It is the kernel that makes the error plane's size matter: it reaches two rows
// ahead, so a single-row buffer would be an out-of-bounds read on the last two
// rows of every frame.
DitherResult Jarvis(DitherJob& job, RgbaF* px) {
  DitherResult res;
  const Palette* pal_ptr = nullptr;
  if (!Prepare(res, "jarvis", job, pal_ptr)) return res;
  const Palette& pal = *pal_ptr;
  const std::size_t w = job.width;
  const std::size_t h = job.height;
  const std::size_t n = w * h;

  ErrorPlane e;
  for (int f = 0; f < job.frames; ++f) {
    e.Reset(n);
    RgbaF* frame = px + static_cast<std::size_t>(f) * n;
    for (std::size_t y = 0; y < h; ++y) {
      const std::size_t row = y * w;
      for (std::size_t x = 0; x < w; ++x) {
        const std::size_t i = row + x;
        const float d = static_cast<float>(job.diffusion);
        const float r = frame[i].r + e.r[i] * d;
        const float g = frame[i].g + e.g[i] * d;
        const float b = frame[i].b + e.b[i] * d;
        const int best = NearestEntry(pal, r, g, b);
        const PaletteEntry& c = pal.entries[best];
        const float er = r - static_cast<float>(c.r);
        const float eg = g - static_cast<float>(c.g);
        const float eb = b - static_cast<float>(c.b);
        // (dy, dx, weight/48), rows relative to the current pixel.
        struct Tap { int dy, dx; float w; };
        static const Tap kTaps[12] = {
            {2, -2, 1.0f / 48.0f}, {2, -1, 3.0f / 48.0f}, {2, 0, 5.0f / 48.0f}, {2, 1, 3.0f / 48.0f},
            {1, -2, 7.0f / 48.0f}, {1, -1, 5.0f / 48.0f}, {1, 0, 7.0f / 48.0f}, {1, 1, 5.0f / 48.0f},
            {0, -2, 3.0f / 48.0f}, {0, -1, 1.0f / 48.0f}, {0, 1, 3.0f / 48.0f}, {0, 2, 1.0f / 48.0f},
        };
        for (const Tap& t : kTaps) {
          // Clamp rather than skip: an edge tap that goes off-frame contributes to
          // the nearest pixel instead of being lost, which is what ImageMagick's own
          // kernels do and what keeps the total error roughly conserved at the
          // borders.  The alternative -- dropping it -- darkens the last column and
          // row.
          std::size_t tx = x + t.dx;
          std::size_t ty = y + t.dy;
          if (t.dx < 0 && x == 0) tx = 0;
          if (t.dx > 0 && x + 1 >= w) tx = w - 1;
          if (t.dy < 0 && y == 0) ty = 0;
          if (t.dy > 0 && y + 1 >= h) ty = h - 1;
          Accumulate(e, n, ty * w + tx, t.w, t.w, t.w, er, eg, eb);
        }
        WriteChosen(frame[i], c);
      }
    }
  }
  return res;
}

// ---------------------------------------------------------------------------
// Clustered dot
// ---------------------------------------------------------------------------
//
// A halftone screen, not a diffusion kernel: a fixed 8x8 pattern of on/off dots
// that is tiled across the frame and thresholded against the pixel's brightness.
// There is no error queue at all, which is what makes it a useful contrast with the
// three above -- it cannot smear, so it cannot preserve gradients, and it will show
// the 8x8 period as a texture wherever the image is smooth.
//
// The matrix is the classic "clustered dot" arrangement, not a Bayer matrix.  The
// difference is visible: a Bayer matrix is a recursive bit-reversal ordering, so its
// threshold rises and falls in a recognisable cross-hatch, whereas clustered dot
// packs the dots as tightly as an 8x8 grid allows, so flat tones read as texture
// rather than as a pattern.
//
// The threshold is compared against the pixel's own luminance, scaled to the
// palette's range.  Comparing against a raw 0..65535 fraction would make every
// threshold fire for any palette darker than full white, so the range comes from
// the palette and adapts to whatever ImageMagick built.
DitherResult ClusteredDot(DitherJob& job, RgbaF* px) {
  DitherResult res;
  const Palette* pal_ptr = nullptr;
  if (!Prepare(res, "clustered-dot", job, pal_ptr)) return res;
  const Palette& pal = *pal_ptr;
  const std::size_t n = job.width * job.height;

  // 8x8 clustered dot: 32 of 64 on, packed towards the centre.  Row-major, 0 = off.
  static const unsigned char kDot[64] = {
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 1, 1, 1, 1, 0, 0,
      0, 1, 1, 1, 1, 1, 1, 0,
      0, 1, 1, 1, 1, 1, 1, 0,
      0, 0, 1, 1, 1, 1, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
  };

  // The palette's own range, so the screen is calibrated to the colours in play
  // rather than assuming a full 0..65535 ramp.  A 16-colour palette built from a
  // dark photograph has a much narrower range, and a screen tuned to 65535 would
  // fire every threshold and come out solid.
  double lo = 0.0, hi = 0.0;
  bool first = true;
  for (int i = 0; i < pal.count; ++i) {
    const PaletteEntry& c = pal.entries[i];
    // Rec. 601 luma, the same weighting ImageMagick uses to decide a pixel is gray.
    const double l = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    if (first || l < lo) lo = l;
    if (first || l > hi) hi = l;
    first = false;
  }
  const double span = (hi > lo) ? (hi - lo) : 1.0;

  for (int f = 0; f < job.frames; ++f) {
    RgbaF* frame = px + static_cast<std::size_t>(f) * n;
    for (std::size_t y = 0; y < job.height; ++y) {
      for (std::size_t x = 0; x < job.width; ++x) {
        const std::size_t i = y * job.width + x;
        const double lum = 0.299 * frame[i].r + 0.587 * frame[i].g + 0.114 * frame[i].b;
        // (lum - lo) / span in [0,1] against the cell's threshold, offset by
        // diffusion so --diffusion 0 is plain thresholding.
        const double t = (lum - lo) / span;
        const double cell = static_cast<double>(kDot[(y & 7) * 8 + (x & 7)]);
        // Above the cell's threshold means "this dot is on", so the pixel wants a
        // DARK colour.  Inverted here so `on` picks the palette's dark entry.
        const double level = 1.0 - t;
        double dark = cell / 32.0;  // the 32 lit cells span [1/32, 1]
        if (cell == 0) dark = 0.0;
        dark = 0.5 + (dark - 0.5) * job.diffusion;
        // Two candidates only: the palette's darkest and lightest entries, which is
        // what a halftone screen actually resolves to.  Searching all 16 would let a
        // mid-tone entry win and turn the screen back into a diffusion dither.
        int lo_i = 0, hi_i = 0;
        double lo_l = 0.0, hi_l = 0.0;
        bool seen = false;
        for (int k = 0; k < pal.count; ++k) {
          const double l = 0.299 * pal.entries[k].r + 0.587 * pal.entries[k].g +
                           0.114 * pal.entries[k].b;
          if (!seen || l < lo_l) { lo_l = l; lo_i = k; }
          if (!seen || l > hi_l) { hi_l = l; hi_i = k; }
          seen = true;
        }
        const int best = (level >= dark) ? lo_i : hi_i;
        WriteChosen(frame[i], pal.entries[best]);
      }
    }
  }
  return res;
}

const bool kRegisteredDiffusion = [] {
  RegisterDither({"floyd-steinberg",
                  "error diffusion, 4 neighbours (7/16 3/16 5/16 1/16)", FloydSteinberg});
  RegisterDither({"atkinson",
                  "error diffusion, 6 neighbours at 1/8; discards 1/4 of the error",
                  Atkinson});
  RegisterDither({"jarvis",
                  "Jarvis/Judice/Ninke, 12 neighbours, 48-denominator weights", Jarvis});
  RegisterDither({"clustered-dot",
                  "8x8 halftone screen, no diffusion; shows the 8x8 period", ClusteredDot});
  return true;
}();

}  // namespace
}  // namespace rd
