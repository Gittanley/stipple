// rd_types.h -- ImageMagick 7.1.2-31 Q16-HDRI numeric contract.
//
// Every constant and rounding rule in this header was verified against the
// *installed* build (C:\Program Files\ImageMagick-7.1.2-Q16-HDRI):
//
//   magick-baseconfig.h : MAGICKCORE_QUANTUM_DEPTH 16
//                         MAGICKCORE_HDRI_ENABLE   1
//                         MAGICKCORE_SIZEOF_FLOAT_T  4  ->  Quantum     == float
//                         MAGICKCORE_SIZEOF_DOUBLE_T 8  ->  MagickRealType == double
//
// The float/double split is the single most important fact for bit-exactness:
// ClampPixel() narrows to float, so every diffused value IM stores has already
// been rounded to float before it is widened back to double.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace rd {

inline constexpr int    kQuantumDepth = 16;
inline constexpr double kQuantumRange = 65535.0;              // QuantumRange
inline constexpr double kQuantumScale = 1.0 / 65535.0;        // QuantumScale
inline constexpr double kOpaqueAlpha  = 65535.0;              // OpaqueAlpha
inline constexpr double kMagickEpsilon = 1.0e-12;             // MagickEpsilon

// MagickSafeReciprocal(x): 1/x, saturating at 1/MagickEpsilon for tiny |x|.
inline double safe_reciprocal(double x) noexcept {
  const double sign = (x < 0.0) ? -1.0 : 1.0;
  return ((sign * x) >= kMagickEpsilon) ? 1.0 / x : sign * (1.0 / kMagickEpsilon);
}

// MagickCore/pixel-accessor.h: ClampPixel().  Returns Quantum == float.
//   if (pixel < 0.0)                      -> 0
//   if (pixel >= QuantumRange)             -> QuantumRange
//   else                                  -> (float) pixel      [HDRI]
inline float clamp_pixel(double v) noexcept {
  if (v < 0.0) return 0.0f;
  if (v >= kQuantumRange) return static_cast<float>(kQuantumRange);
  return static_cast<float>(v);
}

// MagickCore/quantum.h: ClampToQuantum().  HDRI build is a bare narrowing
// cast -- no clamping, no +0.5 rounding.
inline float clamp_to_quantum(double v) noexcept { return static_cast<float>(v); }

// MagickCore/quantize.c: DoublePixelPacket.  Error state is kept in double,
// exactly as IM does, because the error queue accumulates unbounded values.
struct RgbaD {
  double r = 0.0, g = 0.0, b = 0.0, a = 0.0;
};

// One pixel as stored by ImageMagick: Quantum == float per channel.
struct RgbaF {
  float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
};

// quantize.c: #define ErrorQueueLength 16
inline constexpr int kErrorQueueLength = 16;

// quantize.c: #define ErrorRelativeWeight MagickSafeReciprocal(16)
inline constexpr double kErrorRelativeWeight = 1.0 / 16.0;

// quantize.c: #define MaxTreeDepth 8
inline constexpr int kMaxTreeDepth = 8;

// magick-type.h: MaxColormapSize is 65536 for MAGICKCORE_QUANTUM_DEPTH == 16.
inline constexpr int kMaxColormapSize = 65536;

// ---------------------------------------------------------------------------
// Palette-lookup cache key
// ---------------------------------------------------------------------------
// quantize.c:RiemersmaDither() does not call ClosestColor() unconditionally:
//
//     i = CacheOffset(cube_info, &pixel);
//     if (p->cache[i] < 0) { ...ClosestColor...; p->cache[i] = color_number; }
//     index = p->cache[i];
//
// CacheOffset() keeps only the top 6 bits per channel, so two distinct diffused
// targets can share a key and the *first* one to touch that key decides the
// answer for both.  Reproducing ImageMagick bit-for-bit therefore requires the
// memo table, not just an equivalent nearest-colour search.  Omitting it is
// usually invisible and occasionally off by one palette index, which is exactly
// the kind of bug that makes "bit-perfect" claims false.
inline constexpr int kCacheShift = 2;  // quantize.c: non-Apple builds

// quantize.c: length = 1UL << (4*(8-CacheShift))
inline constexpr int kCacheEntries = 1 << (4 * (8 - kCacheShift));

// MagickCore/quantum.h: ScaleQuantumToChar() for MAGICKCORE_QUANTUM_DEPTH == 16
// with HDRI.  Note the float division and the float add.
inline unsigned char scale_quantum_to_char(float q) noexcept {
  if (std::isnan(q) || q <= 0.0f) return 0;
  const float scaled = q / 257.0f;
  if (scaled >= 255.0f) return 255;
  return static_cast<unsigned char>(scaled + 0.5f);
}

// quantize.c: CacheOffset() -- 6 bits per channel, 18 bits without alpha.
inline int cache_offset(bool associate_alpha, const RgbaD& p) noexcept {
  const unsigned int r = scale_quantum_to_char(clamp_pixel(p.r)) >> kCacheShift;
  const unsigned int g = scale_quantum_to_char(clamp_pixel(p.g)) >> kCacheShift;
  const unsigned int b = scale_quantum_to_char(clamp_pixel(p.b)) >> kCacheShift;
  int offset = static_cast<int>(r) | (static_cast<int>(g) << 6) |
               (static_cast<int>(b) << 12);
  if (associate_alpha) {
    const unsigned int a =
        scale_quantum_to_char(clamp_pixel(p.a)) >> kCacheShift;
    offset |= static_cast<int>(a) << 18;
  }
  return offset;
}

// A 16-entry palette entry, matching IM's PixelInfo colormap channels (double).
struct PaletteEntry {
  double r = 0.0, g = 0.0, b = 0.0, a = kOpaqueAlpha;
};

// The exponential-decay weights from GetQCubeInfo().  Reproduced with IM's
// exact expression so the bits match:
//
//   weight = 1.0;
//   for (i = 0; i < 16; i++) { weights[i] = SafeReciprocal(weight);
//                              weight *= exp(log(1.0/Erw)/(16-1.0)); }
//
// which evaluates to weights[i] == 16^(-i/15), i.e. weights[0] == 1 and
// weights[15] == 1/16.
inline void build_error_weights(double (&w)[kErrorQueueLength]) noexcept {
  double weight = 1.0;
  for (int i = 0; i < kErrorQueueLength; ++i) {
    w[i] = safe_reciprocal(weight);
    weight *= std::exp(std::log(1.0 / kErrorRelativeWeight) /
                       (kErrorQueueLength - 1.0));
  }
}

// quantize.c: AssociateAlphaPixel() -- premultiply by alpha unless opaque.
inline void associate_alpha_pixel(bool associate_alpha, const RgbaF& px,
                                  RgbaD* out) noexcept {
  out->a = static_cast<double>(px.a);
  if (!associate_alpha || px.a == static_cast<float>(kOpaqueAlpha)) {
    out->r = static_cast<double>(px.r);
    out->g = static_cast<double>(px.g);
    out->b = static_cast<double>(px.b);
    return;
  }
  const double alpha = kQuantumScale * static_cast<double>(px.a);
  out->r = alpha * static_cast<double>(px.r);
  out->g = alpha * static_cast<double>(px.g);
  out->b = alpha * static_cast<double>(px.b);
}

// quantize.c: AssociateAlphaPixelInfo() -- same rule for a colormap entry.
inline void associate_alpha_info(bool associate_alpha, const PaletteEntry& px,
                                 RgbaD* out) noexcept {
  out->a = px.a;
  if (!associate_alpha || px.a == kOpaqueAlpha) {
    out->r = px.r;
    out->g = px.g;
    out->b = px.b;
    return;
  }
  const double alpha = kQuantumScale * px.a;
  out->r = alpha * px.r;
  out->g = alpha * px.g;
  out->b = alpha * px.b;
}

}  // namespace rd
