// SPDX-License-Identifier: GPL-3.0-or-later
// bayer_dither.cc -- a complete, working second dither algorithm.
//
// This file exists to prove the plugin seam works end to end.  It is not there to be
// useful: Bayer ordered dithering produces visible 8x8 texture rather than Riemersma's
// error diffusion, and nobody would choose it over the built-in for that reason.  Copy
// it, delete the body, write your own.
//
// What to copy is the *shape*, not the arithmetic:
//
//   * the signature and the per-frame loop (see the comment on frames in
//     rd_plugin.h -- this is the one rule that is easy to get wrong),
//   * the nearest-colour search, which is a plain loop over 16 entries and is
//     deliberately not clever: at 16 colours the whole table fits in registers and a
//     clever search costs more time than it saves,
//   * the trailing quantise, which writes the chosen colour *exactly* rather than
//     blending towards it.  That is what makes the output reproduce ImageMagick's
//     palette instead of merely resembling it.
//
// Then read "Writing your own" in include/rd_plugin.h for the three steps to make it
// yours.

#include "rd_plugin.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace rd {
namespace {

// Index of the palette entry closest to (r, g, b).
//
// Squared Euclidean distance, no sqrt: the square root is monotonic, so it cannot
// change which entry wins, and skipping it is a measurable saving when this runs per
// pixel per frame.  This is the same rule ImageMagick uses to build the palette in
// the first place, which is why quantising to a palette built by IM reproduces IM's
// own choice of entry.
int NearestEntry(const Palette& pal, float r, float g, float b) {
  int best = 0;
  float best_d = -1.0f;
  for (int i = 0; i < pal.count; ++i) {
    const PaletteEntry& e = pal.entries[i];
    const float dr = r - static_cast<float>(e.r);
    const float dg = g - static_cast<float>(e.g);
    const float db = b - static_cast<float>(e.b);
    const float d = dr * dr + dg * dg + db * db;
    // Strictly less-than, so the lowest index wins a tie.  Determinism matters: the
    // video path re-runs from a checkpoint, and a tie broken differently on a second
    // pass would make --resume produce a different file from the original run.
    if (best_d < 0.0f || d < best_d) {
      best_d = d;
      best = i;
    }
  }
  return best;
}

// The 8x8 Bayer threshold matrix, normalised to [0, 1).
//
// The classic recursive construction: each 2x2 cell is the previous 4x4 matrix
// scaled by 4, with 2 added, and the four cells offset by 0, 2, 3, 1.  The
// disorganised ordering is what makes the thresholds spread evenly rather than
// clumping.
const float kBayer8[64] = {
    0.0f / 64,  32.0f / 64,  8.0f / 64,  40.0f / 64,  2.0f / 64, 34.0f / 64, 10.0f / 64, 42.0f / 64,
   48.0f / 64,  16.0f / 64, 56.0f / 64, 24.0f / 64, 50.0f / 64, 18.0f / 64, 58.0f / 64, 26.0f / 64,
   12.0f / 64,  44.0f / 64,  4.0f / 64, 36.0f / 64, 14.0f / 64, 46.0f / 64,  6.0f / 64, 38.0f / 64,
   60.0f / 64,  28.0f / 64,  52.0f / 64, 20.0f / 64, 62.0f / 64, 30.0f / 64, 54.0f / 64, 22.0f / 64,
    3.0f / 64,  35.0f / 64, 11.0f / 64, 43.0f / 64,  1.0f / 64, 33.0f / 64,  9.0f / 64, 41.0f / 64,
   51.0f / 64,  19.0f / 64, 59.0f / 64, 27.0f / 64,  49.0f / 64, 17.0f / 64, 57.0f / 64, 25.0f / 64,
   15.0f / 64,  47.0f / 64,  7.0f / 64, 39.0f / 64,  13.0f / 64, 45.0f / 64,  5.0f / 64, 37.0f / 64,
   63.0f / 64,  31.0f / 64, 55.0f / 64, 23.0f / 64,  61.0f / 64, 29.0f / 64, 53.0f / 64, 21.0f / 64,
};

DitherResult BayerDither(DitherJob& job, RgbaF* px) {
  DitherResult res;
  if (job.palette == nullptr || job.palette->count <= 0) {
    res.error = "bayer: no palette";
    return res;
  }
  // A REFERENCE, not a copy, and this is worth spelling out because getting it wrong
  // crashes in a way that looks nothing like a memory bug.
  //
  // Palette is { PaletteEntry entries[65536]; int count; bool; }, and PaletteEntry is
  // four doubles -- so Palette is 2 MiB.  Writing `const Palette pal = *job.palette;`
  // puts 2 MiB on the *stack*, and the default 1 MiB thread stack overflows
  // immediately.  The process dies with STATUS_STACK_OVERFLOW (0xC00000FD) partway
  // through the call, before a single pixel is touched, which reads like a
  // wild pointer rather than a copy.  The palette is already owned by the caller
  // and outlives the call, so bind to it.
  const Palette& pal = *job.palette;
  const std::size_t n = job.width * job.height;
  if (job.width < 8 || job.height < 8) {
    res.error = "bayer: needs at least 8x8";
    return res;
  }
  // The matrix is tiled, so it repeats every 8 pixels rather than being indexed by a
  // global position.  That is what makes it "ordered" and why the result has an 8x8
  // texture: the same decision is made in the same place in every tile.
  // One error buffer per channel, n floats each -- NOT 3n.  An earlier draft
  // allocated n*3 and then indexed the first n entries, which is harmless, but
  // reading the layout wrong is how the classic diffusion kernels get a threefold
  // error-propagation bug that looks like "slightly too noisy".
  std::vector<float> err_r(n, 0.0f), err_g(n, 0.0f), err_b(n, 0.0f);

  for (int f = 0; f < job.frames; ++f) {
    // One frame, one error buffer.  NOT shared across the batch -- see rd_plugin.h.
    std::fill(err_r.begin(), err_r.end(), 0.0f);
    std::fill(err_g.begin(), err_g.end(), 0.0f);
    std::fill(err_b.begin(), err_b.end(), 0.0f);
    RgbaF* frame = px + static_cast<std::size_t>(f) * n;
    for (std::size_t y = 0; y < job.height; ++y) {
      const std::size_t row = y * job.width;
      for (std::size_t x = 0; x < job.width; ++x) {
        const std::size_t i = row + x;
        // The error buffer is a function of (x, y), not of the loop counters' current
        // value after the write below.  Reading it before the neighbour writes of the
        // *previous* pixel have landed is the whole point, so this read has to come
        // first and the writes have to be to i+1 and i+width, never to i itself.
        float r = frame[i].r + err_r[i];
        float g = frame[i].g + err_g[i];
        float b = frame[i].b + err_b[i];
        // The threshold, as a brightness offset.  Diffusion scales it so that
        // --diffusion 0 reduces to plain nearest-colour, which is the behaviour
        // ImageMagick's -define dither:diffusion-amount=0 gives.
        const float th =
            static_cast<float>((kBayer8[(y & 7) * 8 + (x & 7)] - 0.5) * job.diffusion);
        r += th; g += th; b += th;
        const int best = NearestEntry(pal, r, g, b);
        const PaletteEntry& c = pal.entries[best];
        // Push the difference onto the neighbours.  The 7/16, 3/16, 5/16, 1/16
        // weights are Floyd-Steinberg; a Bayer matrix does not strictly need error
        // diffusion, but carrying a little of it is what stops the ordered texture
        // from looking like a grid.
        const float er = r - static_cast<float>(c.r);
        const float eg = g - static_cast<float>(c.g);
        const float eb = b - static_cast<float>(c.b);
        const std::size_t xm = x > 0 ? i - 1 : i;
        const std::size_t xp = x + 1 < job.width ? i + 1 : i;
        const std::size_t xd = i + job.width;
        err_r[xm] += er * 0.1875f; err_g[xm] += eg * 0.1875f; err_b[xm] += eb * 0.1875f;
        err_r[xp] += er * 0.1875f; err_g[xp] += eg * 0.1875f; err_b[xp] += eb * 0.1875f;
        if (xd < n) {
          err_r[xd] += er * 0.125f; err_g[xd] += eg * 0.125f; err_b[xd] += eb * 0.125f;
          if (xp < n) {
            err_r[xp + job.width] += er * 0.0625f; err_g[xp + job.width] += eg * 0.0625f;
            err_b[xp + job.width] += eb * 0.0625f;
          }
        }
        // Write the palette entry exactly.  This is the line that makes the output
        // 16 colours rather than 16 colours plus rounding.
        frame[i].r = static_cast<float>(c.r);
        frame[i].g = static_cast<float>(c.g);
        frame[i].b = static_cast<float>(c.b);
      }
    }
  }
  return res;
}

const bool kRegistered = [] {
  return RegisterDither({"bayer", "ordered 8x8 + light diffusion (example plugin)", BayerDither});
}();

}  // namespace
}  // namespace rd
