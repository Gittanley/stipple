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
// Building a fixture that exposes the raw threshold pattern is still open.  The
// related loose end -- "the 64x64 and 32x32 outputs differ by only about 32
// pixels out of half a million" -- is EXPLAINED and fixed: the rank builder
// started from an all-ON pattern, whose blur is uniform, so every exchange step
// landed on the same two pixels and the matrix came out degenerate (rank[n*n]
// at one position, 0 everywhere else) for both tile sizes, hence nearly
// identical outputs.  The builder now seeds deterministic proto-noise and runs
// the three phases described in BuildVoidCluster; --self-test asserts the rank
// spread.  The blue-noise spectrum itself is still not measured, so the
// quality claim remains open even though the ranks are now real.
//
// Neither of those is a reason to delete the filters -- they are deterministic, exact on
// the palette, and visibly different from every other registered algorithm -- but it is a
// reason not to describe them as the best choice for gradients until the spectrum has
// been measured.

#include "rd_dither_common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
// The exchange, used by the prefilter in BuildVoidCluster, is two steps and the
// order matters:
//
//   1. filter, find the LARGEST value among ON pixels, turn that pixel OFF.  That
//      is the tightest cluster, and clearing it is what creates a void.
//   2. filter again, find the SMALLEST value among OFF pixels, turn it ON.  That
//      is the largest void, which is where the dot belongs.
//
// Getting the order wrong -- placing first, clearing after -- produces a matrix that
// still looks structured and still quantises, so it would pass a smoke test and be
// wrong.  Ranks themselves are assigned afterwards, by the drain and refill phases
// documented in BuildVoidCluster, not by this exchange.
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

  // A deterministic proto-noise start, density ~0.1.
  //
  // NOT all-ON, which is what this used to be.  A blur of a uniform field is
  // uniform, so with `on` all 1.0 every step did the same thing: arg_max (strict
  // >, tie to index 0) returned 0, the clear at index 0 was undone by the
  // restore, arg_min returned 0, and rank[0] was overwritten np times.  The
  // matrix came out degenerate -- rank[0] == n*n, every other rank 0 -- which
  // is the "64x64 and 32x32 differ by only ~32 pixels" loose end in the header
  // comment above: both matrices were the same rank-0 pattern.  Caught by
  // --self-test's far-entry fraction check.
  //
  // The seed must be deterministic across processes: --resume re-runs the whole
  // program, and a std::random_device matrix would make the resumed file differ
  // from the original.  Fixed LCG constants, no seed input.
  std::vector<double> on(np, 1.0);
  {
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    for (std::size_t i = 0; i < np; ++i) {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      on[i] = ((s >> 33) % 10 == 0) ? 1.0 : 0.0;
    }
  }
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

  // argmax over the ON pixels only (the tightest cluster), argmin over the OFF
  // pixels only (the largest void) -- then REMOVE the argmax pixel and PLACE at
  // the argmin pixel.  The whole-image searches this used to run could return a
  // pixel whose state made the toggle a no-op, which is half of why the old
  // matrix was degenerate; the other half was the single exchange loop, which
  // cycled on the same pixel forever (see the phase comments below).
  //
  // Reference for the structure: Zoy's BlueNoise void-and-cluster
  // (pixelmager/bluenoisegen src_void_and_cluster/void_and_cluster.cpp, after
  // caca.zoy.org/study): prefilter to equilibrium, DRAIN the pattern assigning
  // ranks downward, then REFILL from the equilibrium clone assigning ranks
  // upward.  Each phase visits each pixel exactly once, so the ranks come out a
  // permutation of 1..n*n -- which is what a threshold matrix has to be.
  //
  // Strict comparison, so a tie goes to the lowest index.  Same reason as the
  // nearest-colour tie-break: --resume re-runs from a checkpoint, and a tie broken
  // differently on a second pass would make the resumed file differ from the original.
  auto arg_max_on = [&]() -> std::size_t {
    std::size_t best = np;  // np = "none": no ON pixel left
    double bv = 0.0;
    for (std::size_t i = 0; i < np; ++i) {
      if (on[i] <= 0.5) continue;
      if (best == np || filt[i] > bv) { bv = filt[i]; best = i; }
    }
    return best;
  };
  auto arg_min_off = [&]() -> std::size_t {
    std::size_t best = np;  // np = "none": no OFF pixel left
    double bv = 0.0;
    for (std::size_t i = 0; i < np; ++i) {
      if (on[i] > 0.5) continue;
      if (best == np || filt[i] < bv) { bv = filt[i]; best = i; }
    }
    return best;
  };

  // Prefilter: exchange (remove tightest ON, place at largest OFF void) until
  // the two land on the same pixel -- the pattern is then in equilibrium, the
  // state both phases below start from.  The reference loops until stable; the
  // bound here is a safety net so a build can never hang the first program that
  // touches a tile, and 4*n*n exchanges is far past where equilibrium arrives.
  for (std::size_t it = 0; it < 4 * np; ++it) {
    blur();
    const std::size_t tightest = arg_max_on();
    if (tightest == np) break;
    on[tightest] = 0.0;
    blur();
    const std::size_t largest_void = arg_min_off();
    if (largest_void == np) break;
    on[largest_void] = 1.0;
    if (tightest == largest_void) break;  // exchange was the identity: settled
  }

  // The equilibrium pattern, cloned: phase II refills exactly this state.
  const std::vector<double> equilibrium = on;

  // Phase I -- DRAIN.  The dots in the equilibrium pattern get the LOW ranks,
  // counting down: remove the tightest cluster, rank it, never re-add it.
  // Tightest-first means the redundant dot of each neighbourhood is claimed
  // while the neighbourhood still exists.
  std::size_t dots = 0;
  for (std::size_t i = 0; i < np; ++i) {
    if (on[i] > 0.5) ++dots;
  }
  std::size_t rank = dots;
  while (rank > 0) {
    --rank;
    blur();
    const std::size_t tightest = arg_max_on();
    if (tightest == np) break;  // cannot happen with rank<=dots; test catches it
    on[tightest] = 0.0;
    vc.rank[tightest] = static_cast<std::uint16_t>(rank + 1);  // 1-based
  }

  // Phase II -- REFILL.  Restore the equilibrium pattern and fill the largest
  // voids one dot at a time, ranks counting up to n*n.  A void is OFF by
  // definition of arg_min_off, so every placement lands on a not-yet-ranked
  // pixel: the two phases together rank every pixel exactly once.
  on = equilibrium;
  for (std::size_t r = dots; r < np; ++r) {
    blur();
    const std::size_t largest_void = arg_min_off();
    if (largest_void == np) break;  // pattern full before ranks ran out
    on[largest_void] = 1.0;
    vc.rank[largest_void] = static_cast<std::uint16_t>(r + 1);  // 1-based
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
