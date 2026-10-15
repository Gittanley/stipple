// SPDX-License-Identifier: GPL-3.0-or-later
// rd_dither_common.h -- pieces shared by the non-Riemersma dither algorithms.
//
// NOT a public header.  It lives in src/ rather than include/ because the plugin
// seam is rd_plugin.h and nothing outside this directory should depend on the
// internals of one particular set of algorithms.
//
// The one thing that must not be duplicated is the nearest-colour search.  Every
// algorithm here quantises by finding the closest palette entry, and if two copies
// of that rule ever disagreed -- a tie broken differently, a distance computed in a
// different precision -- then two algorithms that should agree on flat colour would
// pick different entries, and the difference would show up as "dither A is subtly
// wrong" with nothing to point at.  One copy, in one place.
#ifndef RD_DITHER_COMMON_H_
#define RD_DITHER_COMMON_H_

#include "rd_plugin.h"

#include <cstddef>
#include <vector>

namespace rd {
namespace dither_common {

// Index of the palette entry closest to (r, g, b).
//
// Squared Euclidean distance, no sqrt: the square root is monotonic, so it cannot
// change which entry wins, and skipping it is a measurable saving when this runs per
// pixel per frame.  This is the same rule ImageMagick uses to build the palette in
// the first place, which is why quantising to a palette built by IM reproduces IM's
// own choice of entry.
//
// The tie-break is the lowest index, via strict less-than.  That is not cosmetic:
// the video path re-runs from a checkpoint, and a tie broken differently on a second
// pass would make --resume produce a different file from the original run.
//
// NOT a copy of the palette.  Palette is { PaletteEntry entries[65536]; int; bool; }
// and PaletteEntry is four doubles, so Palette is 2 MiB.  Binding one by value puts
// 2 MiB on the stack, the default 1 MiB thread stack overflows immediately, and the
// process dies with STATUS_STACK_OVERFLOW before touching a pixel -- which reads
// like a wild pointer rather than a copy.  Always take it by reference.
inline int NearestEntry(const Palette& pal, float r, float g, float b) {
  int best = 0;
  float best_d = -1.0f;
  for (int i = 0; i < pal.count; ++i) {
    const PaletteEntry& e = pal.entries[i];
    const float dr = r - static_cast<float>(e.r);
    const float dg = g - static_cast<float>(e.g);
    const float db = b - static_cast<float>(e.b);
    const float d = dr * dr + dg * dg + db * db;
    if (best_d < 0.0f || d < best_d) {
      best_d = d;
      best = i;
    }
  }
  return best;
}

// One float per pixel per channel, zeroed.
//
// A separable buffer (three vectors of n, not one of 3n) so that a kernel which
// needs to reach two rows ahead -- Jarvis does -- can bounds-check per channel
// without a stride.  The classic diffusion kernels fail by a factor of three when
// this layout is read wrong, and the symptom is "slightly too noisy", which is very
// hard to trace back to an index.
struct ErrorPlane {
  std::vector<float> r, g, b;

  void Reset(std::size_t n) {
    r.assign(n, 0.0f);
    g.assign(n, 0.0f);
    b.assign(n, 0.0f);
  }
};

// Add `w * err` into channel `ch` of pixel `i`, if `i` is inside the frame.
//
// The bounds check is inside the add rather than at the call site, because the
// kernels below have edge weights that point at i-1 (left of column 0) and at
// i+2*width (two rows down, past the last row).  Clamping those to the frame is
// what ImageMagick's own kernels do, and getting it wrong writes outside the plane
// -- silently, in a vector, until it lands on the heap.
inline void Accumulate(ErrorPlane& e, std::size_t n, std::size_t i, float wr,
                       float wg, float wb, float er, float eg, float eb) {
  if (i >= n) return;
  e.r[i] += wr * er;
  e.g[i] += wg * eg;
  e.b[i] += wb * eb;
}

// Write the chosen palette entry exactly, leaving alpha alone.
//
// "Exactly" is the load-bearing word.  Blending towards the target, or writing the
// error-corrected value rather than the palette entry, is what makes an output
// "16 colours plus rounding" instead of 16 colours.
//
// Alpha is left untouched on purpose: it is not something a dither chooses between.
// `Palette::associate_alpha` says whether alpha was folded into the colour decision,
// and that decision was already made when the palette was built.
inline void WriteChosen(RgbaF& px, const PaletteEntry& c) {
  px.r = static_cast<float>(c.r);
  px.g = static_cast<float>(c.g);
  px.b = static_cast<float>(c.b);
}

// Shared preamble for every algorithm here: validate, and hand back the palette by
// reference.  Returns false with `res.error` set on anything unusable.
inline bool Prepare(DitherResult& res, const char* who, DitherJob& job,
                    const Palette*& pal_out) {
  if (job.palette == nullptr || job.palette->count <= 0) {
    res.error = std::string(who) + ": no palette";
    return false;
  }
  if (job.width == 0 || job.height == 0) {
    res.error = std::string(who) + ": empty frame";
    return false;
  }
  pal_out = job.palette;
  return true;
}

}  // namespace dither_common
}  // namespace rd

#endif  // RD_DITHER_COMMON_H_
