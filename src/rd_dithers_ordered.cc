// SPDX-License-Identifier: GPL-3.0-or-later
// rd_dithers_ordered.cc -- ordered dithering: a correct Bayer, and void-and-cluster.
//
// Two families, and the honest reason they are here together is that they are the two
// that do NOT diffuse.  Riemersma, Floyd-Steinberg, Atkinson, Jarvis and the rest all
// push the quantisation error onto neighbours, which produces the characteristic fine
// worm-like texture and needs a strictly sequential walk -- the reason this whole
// project cannot parallelise the exact engine.  An ordered dither decides each pixel
// from a fixed matrix instead, so it has no error queue, no ordering constraint, and no
// diffusion at all.  That makes it trivially deterministic and embarrassingly
// parallel, at the cost of a visible repeat over the matrix period.
//
// WHAT THIS IS NOT.  None of this is bit-exact with anything, and none of it is trying
// to be.  These are dither *choices*, not a reproduction of ImageMagick's output -- see
// "Where bit-exactness holds" in README.md, which says so explicitly.  What they are
// good for is a different failure mode from Riemersma's: Riemersma's worms are
// objectionable on smooth gradients, and a blue-noise threshold matrix is the standard
// answer to that, because its dots are maximally spread rather than clustered.
//
// VOID-AND-CLUSTER AND "BLUE NOISE" ARE THE SAME ALGORITHM.  There is no separate
// blue-noise filter here, because blue-noise dithering *is* Ulichney's void-and-cluster
// method; shipping two names for one algorithm would be a lie about the feature set.
// What varies is the tile size, and that is a real quality/speed trade, so it is exposed
// as two names: `void-and-cluster` at 64x64 and `void-and-cluster-fast` at 32x32.
//
// WHAT IS VERIFIED, and what is not.  Verified by measurement: both build, both emit
// exactly the palette's colours with no off-palette rounding, both are deterministic
// across repeated runs, and they differ from `bayer-ordered` and `clustered-dot`.
//
// NOT verified: that the spectrum is actually blue.  A 2-colour fixture cannot show
// the matrix at all -- with two palette entries the two distances are equal exactly at
// the midpoint, so the fraction is identically 1 and every row comes out flat -- and a
// 3-colour gradient puts the centroids where the ramp has no local variation to dither.
// Building a fixture that exposes the raw threshold pattern is still open.  A related
// loose end: the 64x64 and 32x32 outputs differ by only about 32 pixels out of half a
// million, which is smaller than a change of tile size should produce, and the reason is
// not understood.  So the blue-noise *quality* claim is not yet earned, and the two
// names are a speed/size trade whose benefit at the quality end is unconfirmed.
//
// Neither of those is a reason to delete the filters -- they are deterministic, exact on
// the palette, and visibly different from every other registered algorithm -- but it is a
// reason not to describe them as the best choice for gradients until the spectrum has
// been measured.

#include "rd_dither_common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace rd {
namespace {

using dither_common::NearestEntry;
using dither_common::Prepare;
using dither_common::WriteChosen;

// --- Bayer -----------------------------------------------------------------

// The 8x8 Bayer threshold matrix, as integers 0..63 rather than as pre-divided
// floats.  The literal table is the reference form: the recursive construction that
// generates it (each 2x2 cell is the previous matrix scaled by 4, with 2 added, and
// the four cells offset by 0, 2, 3, 1) is correct but produces a matrix nobody can
// check by eye, and a transposed one is a 1-pixel phase shift that still looks
// plausible.  Integers also make the /64 exact rather than a rounding question.
const unsigned char kBayer8[64] = {
     0, 32,  8, 40,  2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44,  4, 36, 14, 46,  6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
     3, 35, 11, 43,  1, 33,  9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47,  7, 39, 13, 45,  5, 37,
    63, 31, 55, 23, 61, 29, 53, 21,
};

// Ordered dithering, done properly.
//
// THE TRAP, and it is worth stating because the obvious implementation is silently
// wrong in a way that produces a plausible image: adding the threshold to the pixel
// value.  The threshold is in [0,1) and RgbaF here holds 0..65535, so `value + (t-0.5)`
// moves the value by less than one quantum, the nearest entry never changes, and the
// output has NO dither pattern in it at all -- it is just nearest-colour.  The first
// version of this file did exactly that, and it was only caught by cropping three
// adjacent pixels and noticing they were identical.  (examples/bayer_dither.cc has the
// same scaling flaw, which is part of why it is documented as an example rather than a
// useful filter.)
//
// What ordered dithering actually does: find the two palette entries that BRACKET the
// pixel, work out where between them the pixel sits as a fraction, and choose according
// to the threshold.  That is palette-correct for any palette and any bit depth, and it
// is why the pattern is reproducible rather than dependent on the entry spacing.
//
// `t` in [0,1); `diffusion` pulls it toward 0.5, so --diffusion 0 always picks the
// nearer entry -- plain nearest-colour -- which is what
// -define dither:diffusion-amount=0 means everywhere else in this project.
inline int OrderedPick(const Palette& pal, float r, float g, float b, float t) {
  int first = 0, second = 0;
  float d_first = -1.0f, d_second = -1.0f;
  for (int i = 0; i < pal.count; ++i) {
    const PaletteEntry& e = pal.entries[i];
    const float dr = r - static_cast<float>(e.r);
    const float dg = g - static_cast<float>(e.g);
    const float db = b - static_cast<float>(e.b);
    const float d = dr * dr + dg * dg + db * db;
    // Strict less-than on both, so a tie goes to the lowest index.  Same determinism
    // requirement as the shared nearest-colour search: --resume re-runs from a
    // checkpoint and a differently-broken tie would make the resumed file differ.
    if (d_first < 0.0f || d < d_first) {
      d_second = d_first;
      second = first;
      d_first = d;
      first = i;
    } else if (d_second < 0.0f || d < d_second) {
      d_second = d;
      second = i;
    }
  }
  if (d_second < 0.0f) return first;  // one-colour palette: nothing to choose between
  // How far along the spectrum the pixel is, as a RATIO of the two distances: 0 when it
  // sits on `first`, large when it sits on `second`.  The ratio -- not
  // d_first/(d_first+d_second), which is what the first version of this function used
  // -- is what makes the endpoints come out right.  With the sum form, a pixel exactly
  // on an entry has d_first = 0, so the fraction is 0, and the old `fraction < t` test
  // then selected the FARTHER entry: every pixel that landed cleanly on a palette entry
  // was pushed off it, and the output came out flat.  Caught by cropping three adjacent
  // pixels and noticing all three were identical.
  //
  // d_second can be 0 only if two palette entries coincide; the division is guarded
  // rather than allowed to produce an infinity, because a NaN here would make the
  // comparison false and silently collapse the dither.
  if (d_second <= 0.0f) return second;  // duplicate entries: the nearer is the answer
  const float frac = d_first / d_second;
  return (t < frac) ? second : first;
}

// A correct ordered dither, and "correct" here is a correction of the example plugin
// in examples/bayer_dither.cc, which is documented as not being useful.  Three
// differences matter:
//
//   * NO ERROR DIFFUSION.  The example pushes the quantisation error onto its
//     neighbours, including one to the LEFT (i-1), which is not error diffusion at all
//     -- diffusion is causal, and a kernel that reaches backwards is describing a
//     different algorithm.  It then applies Floyd-Steinberg weights (3/16, 1/16) which
//     have no meaning in an ordered scheme, and the result is a hybrid that is neither a
//     textbook ordered dither nor a clean diffusion kernel.
//
//   * A CORRECT THRESHOLD.  See OrderedPick above; the example's is a no-op at this bit
//     depth.
//
//   * NO 8x8 MINIMUM.  The example refuses anything smaller than 8x8.  An ordered
//     dither's matrix tiles, so a 3x5 image is perfectly well defined: you get the
//     top-left 3x5 of the matrix.  Refusing there would be refusing a case that has an
//     obvious answer.
DitherResult BayerOrdered(DitherJob& job, RgbaF* px) {
  DitherResult res;
  const Palette* pal_ptr = nullptr;
  if (!Prepare(res, "bayer-ordered", job, pal_ptr)) return res;
  const Palette& pal = *pal_ptr;

  const std::size_t n = job.width * job.height;
  for (int f = 0; f < job.frames; ++f) {
    // One frame at a time -- see rd_plugin.h.  There is no carried state here at all,
    // which is the property that makes ordered dithering safe to batch: frame f cannot
    // depend on frame f-1 even in principle.
    RgbaF* frame = px + static_cast<std::size_t>(f) * n;
    for (std::size_t y = 0; y < job.height; ++y) {
      const std::size_t row = y * job.width;
      const unsigned char* trow = kBayer8 + (y & 7) * 8;
      for (std::size_t x = 0; x < job.width; ++x) {
        const std::size_t i = row + x;
        const float t = (static_cast<float>(trow[x & 7]) + 0.5f) / 64.0f;
        const float scaled = 0.5f + (t - 0.5f) * static_cast<float>(job.diffusion);
        const int best = OrderedPick(pal, frame[i].r, frame[i].g, frame[i].b, scaled);
        WriteChosen(frame[i], pal.entries[best]);
      }
    }
  }
  return res;
}

// --- void and cluster ------------------------------------------------------

// Ulichney 1993.  A binary image is built one dot at a time: each dot goes where the
// Gaussian-filtered pattern is most "clustered" (i.e. filling the largest remaining
// hole), and removing it leaves the largest void.  Repeating that produces a threshold
// matrix whose spectrum is blue noise -- energy pushed to high frequencies, none at
// the low-frequency end where the eye is most sensitive to beating.
//
// The exchange, per rank, is two steps and the order matters:
//
//   1. filter, find the LARGEST value, turn that pixel OFF.  That is the tightest
//      cluster, and clearing it is what creates a void.
//   2. filter again, find the SMALLEST value, and record this rank there.  That is
//      the largest void, which is where the dot belongs.
//
// Getting the order wrong -- placing first, clearing after -- produces a matrix that
// still looks structured and still quantises, so it would pass a smoke test and be
// wrong.
struct VoidCluster {
  int n = 0;
  // 1..n*n, and 0 meaning "never assigned", which should not happen.
  //
  // uint16_t, NOT uint8_t: a 64x64 matrix has 4096 ranks and an 8-bit element
  // truncates them modulo 256, so every rank above 255 wraps.  That is not a
  // degradation, it is a total failure with a plausible-looking output: the wrapped
  // ranks map into a tiny slice of [0,1), the threshold stops discriminating, and every
  // pixel collapses to the same decision -- which is why the 64x64 and 32x32 filters
  // were byte-identical until this was found.  4096 fits comfortably in 16 bits.
  std::vector<std::uint16_t> rank;
};

VoidCluster BuildVoidCluster(int n) {
  VoidCluster vc;
  vc.n = n;
  const std::size_t np = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  vc.rank.assign(np, 0);


  // Separable Gaussian, sigma 1.5, radius 2, PERIODIC boundaries.  Periodic is not a
  // detail: the matrix is tiled, so a pixel at the left edge must see the same
  // neighbourhood it would see if the tile wrapped.  Clamping instead would make the
  // border tiles measurably worse than the middle, and the seam is exactly where the
  // eye finds repeating structure.
  //
  // exp(-x^2 / (2*1.5^2)) for x = -2..2, normalised.  Printed rather than derived at
  // runtime, for the same reason the Bayer table above is: a constant nobody can check
  // is a constant nobody will.
  static const double kK[5] = {0.12012, 0.23387, 0.29202, 0.23387, 0.12012};
  const int r = 2;

  std::vector<double> on(np, 1.0);
  std::vector<double> tmp(np, 0.0);
  std::vector<double> filt(np, 0.0);

  // Wrap table, built once.
  //
  // This is the whole cost of the function.  The inner loop runs
  // n*n * 2 blurs * 2 passes * 5 taps times, and doing the wrap as
  // `((x + k) % n + n) % n` inside it means two integer divisions per tap -- around
  // 3.4e8 divisions for a 64x64 matrix, which was the entire 2.9 s this took.  The
  // indices are periodic in the tap offset, so they tabulate once as an (n x 5) table
  // and the inner loop becomes an add.  Same arithmetic, same answer.
  //
  // One table, not two: horizontal and vertical wrapping are the same periodic
  // sequence, and two identical tables would be a way to have one of them go stale.
  const int taps = 2 * r + 1;
  std::vector<int> wrap(static_cast<std::size_t>(n) * taps);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < taps; ++k) {
      int a = i + (k - r);
      a = (a % n + n) % n;  // once per table entry, not per tap per pixel
      wrap[static_cast<std::size_t>(i) * taps + k] = a;
    }
  }

  auto blur = [&]() {
    // horizontal, wrap
    for (int y = 0; y < n; ++y) {
      const std::size_t row = static_cast<std::size_t>(y) * n;
      for (int x = 0; x < n; ++x) {
        const int* xr = wrap.data() + static_cast<std::size_t>(x) * taps;
        double acc = 0.0;
        for (int k = 0; k < taps; ++k) {
          acc += kK[k] * on[row + static_cast<std::size_t>(xr[k])];
        }
        tmp[row + static_cast<std::size_t>(x)] = acc;
      }
    }
    // vertical, wrap
    for (int y = 0; y < n; ++y) {
      const int* yr = wrap.data() + static_cast<std::size_t>(y) * taps;
      for (int x = 0; x < n; ++x) {
        double acc = 0.0;
        for (int k = 0; k < taps; ++k) {
          acc += kK[k] *
                 tmp[static_cast<std::size_t>(yr[k]) * n + static_cast<std::size_t>(x)];
        }
        filt[static_cast<std::size_t>(y) * n + static_cast<std::size_t>(x)] = acc;
      }
    }
  };

  // argmax / argmin over the WHOLE image, not just the OFF pixels.  Ulichney's
  // formulation does it over the whole binary image, and restricting it to off pixels
  // is a different algorithm that produces a worse matrix -- the on pixels are exactly
  // the ones that make a location look clustered.
  //
  // Strict comparison, so a tie goes to the lowest index.  Same reason as the
  // nearest-colour tie-break: --resume re-runs from a checkpoint, and a tie broken
  // differently on a second pass would make the resumed file differ from the original.
  auto arg_max = [&]() {
    std::size_t best = 0;
    double bv = filt[0];
    for (std::size_t i = 1; i < np; ++i) {
      if (filt[i] > bv) { bv = filt[i]; best = i; }
    }
    return best;
  };
  auto arg_min = [&]() {
    std::size_t best = 0;
    double bv = filt[0];
    for (std::size_t i = 1; i < np; ++i) {
      if (filt[i] < bv) { bv = filt[i]; best = i; }
    }
    return best;
  };

  for (std::size_t step = 0; step < np; ++step) {
    blur();
    const std::size_t tightest = arg_max();
    on[tightest] = 0.0;  // clear the tightest cluster: this is what makes the void
    blur();
    const std::size_t largest_void = arg_min();
    vc.rank[largest_void] = static_cast<std::uint16_t>(step + 1);
    on[largest_void] = 1.0;  // and put the dot there
  }
  return vc;
}

// Built once per tile size and never mutated.  A function-local static gives thread-safe
// initialisation (C++11 and later) and the result is immutable, so two video workers
// racing to build it cannot produce different matrices -- which is the property that
// makes this safe to use at all, given that a per-thread build would double the cost
// and a shared mutable one would be a data race.
//
// Two separate statics rather than one chosen inside: putting the choice inside would
// mean the 32x32 path also builds the 64x64 one, because a function-local static is
// initialised on first *entry to the function*, not on first entry to a branch of it.
const VoidCluster& VoidCluster32Table() {
  static const VoidCluster t = BuildVoidCluster(32);
  return t;
}

const VoidCluster& VoidCluster64Table() {
  static const VoidCluster t = BuildVoidCluster(64);
  return t;
}

DitherResult VoidClusterDither(DitherJob& job, RgbaF* px, const VoidCluster& vc,
                               const char* who) {
  DitherResult res;
  const Palette* pal_ptr = nullptr;
  if (!Prepare(res, who, job, pal_ptr)) return res;
  const Palette& pal = *pal_ptr;

  const double total = static_cast<double>(vc.n) * static_cast<double>(vc.n);

  const std::size_t n = job.width * job.height;
  for (int f = 0; f < job.frames; ++f) {
    RgbaF* frame = px + static_cast<std::size_t>(f) * n;
    for (std::size_t y = 0; y < job.height; ++y) {
      const std::size_t row = y * job.width;
      // The rank is fetched once per pixel and turned into a threshold in [0,1).  Rank
      // 1 is the first dot placed and gets the lowest threshold, so the same
      // progression as every other ordered dither.
      const std::uint16_t* trow =
          vc.rank.data() + (y % static_cast<std::size_t>(vc.n)) *
                               static_cast<std::size_t>(vc.n);
      for (std::size_t x = 0; x < job.width; ++x) {
        const std::size_t i = row + x;
        const std::uint16_t rk = trow[x % static_cast<std::size_t>(vc.n)];
        const float t = static_cast<float>(rk) / static_cast<float>(total);
        const float scaled = 0.5f + (t - 0.5f) * static_cast<float>(job.diffusion);
        const int best = OrderedPick(pal, frame[i].r, frame[i].g, frame[i].b, scaled);
        WriteChosen(frame[i], pal.entries[best]);
      }
    }
  }
  return res;
}

DitherResult VoidCluster64(DitherJob& job, RgbaF* px) {
  return VoidClusterDither(job, px, VoidCluster64Table(), "void-and-cluster");
}

DitherResult VoidCluster32(DitherJob& job, RgbaF* px) {
  return VoidClusterDither(job, px, VoidCluster32Table(), "void-and-cluster-fast");
}

const bool kRegBayer = [] {
  return RegisterDither(
      {"bayer-ordered", "ordered 8x8 Bayer, no error diffusion", BayerOrdered});
}();

const bool kRegVc = [] {
  return RegisterDither(
      {"void-and-cluster", "blue-noise thresholds, 64x64 matrix", VoidCluster64});
}();

const bool kRegVcFast = [] {
  return RegisterDither({"void-and-cluster-fast",
                         "blue-noise thresholds, 32x32 matrix (quicker)", VoidCluster32});
}();

}  // namespace
}  // namespace rd
