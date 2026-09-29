// SPDX-License-Identifier: GPL-3.0-or-later
// rd_im.h -- ImageMagick bridge: decode, palette generation, encode.
//
// The palette is produced by ImageMagick's own QuantizeImage() with
// dither_method = RiemersmaDitherMethod.  That matters: quantize.c chooses the
// octree depth as Log4(colors)+2 and then decrements it when a dither method is
// selected, so a palette built with NoDitherMethod would be built from a
// *different* tree depth and would not match.  Letting IM build it also means
// IM's grayscale detection, SetGrayscaleImage() and colorspace normalisation
// come along for free and cannot drift.
#pragma once

#include <cstddef>
#include <string>

#include "rd_riemersma.h"
#include "rd_source.h"

namespace rd {

struct LoadedImage {
  void* handle = nullptr;  // Image*  (opaque here)
  std::size_t width = 0;
  std::size_t height = 0;
  bool has_alpha = false;
  int colorspace = 0;  // ColorspaceType
};

// Initialises the MagickCore environment exactly once per process.
// `magick_exe` should be the full path of the magick.exe that ships with the
// linked ImageMagick build; ImageMagick derives its configuration search path
// from it.  Pass an empty string only if MAGICK_HOME is already exported.
bool ImStartup(const std::string& magick_exe, std::string* error);
void ImShutdown();

// Reads `path` and normalises it the way quantize.c:ClassifyImageColors() would:
// a non-sRGB-compatible colorspace is transformed to sRGB so the pixels we hand
// to the dither are the pixels IM would have dithered.
bool ImLoad(const std::string& path, LoadedImage* out, std::string* error);

// Copies the decoded pixels into `store` (RGBA float, row major).
bool ImExtract(const LoadedImage& image, PixelStore* store, std::string* error);

// Runs IM's own quantizer on a clone and harvests the resulting colormap.
bool ImBuildPalette(const LoadedImage& image, int colors, Palette* out,
                    std::string* error);

// Runs the *complete* ImageMagick pipeline (quantize + Riemersma dither) on a
// clone and copies the dithered pixels into `out`.  This is exactly what
// `magick in -dither Riemersma -colors N out` produces, and it is what
// ImCompareAgainstReference() measures us against.
bool ImReferenceDither(const LoadedImage& image, int colors, PixelStore* out,
                       std::string* error);

// Pixel-absolute-error comparison of two same-geometry stores.
struct DiffResult {
  std::size_t differing_pixels = 0;
  std::size_t total_pixels = 0;
  int max_channel_delta = 0;  // in 16-bit code values
  double rmse = 0.0;
};
DiffResult CompareStores(const PixelStore& a, const PixelStore& b,
                         bool compare_alpha);

// Builds a palette and a colour tree directly from an RGBA-float buffer.
//
// Phase 2 needs this because the frames come from an ffmpeg rawvideo pipe: there
// is no file for ImageMagick to read, and round-tripping 30 frames through PNG
// just to throw the pixels away would dominate the setup.  The image is created
// with the same colourspace, depth and channel traits as a decoded sRGB frame so
// QuantizeImage() takes the same path it would for a file.
bool ImBuildPaletteFromPixels(const RgbaF* pixels, std::size_t width,
                              std::size_t height, int colors, Palette* palette,
                              ColorTree* tree, std::string* error);

// Reproduces the reference video pipeline's palette step as closely as is
// possible in-process: 8-bit, no alpha, the frames laid out horizontally the way
// `+append` does them, then QuantizeImage and a colormap deduplication.
//
// These three details are not cosmetic, and getting them wrong is why the default
// builder measured ~10 points of mean saturation below the reference:
//
//   * no alpha. quantize.c:3306 shortens the octree by one level when
//     image->alpha_trait != UndefinedPixelTrait, so carrying an alpha channel
//     builds a *different tree* than the reference does.
//   * 8-bit. The reference extracts PPM; QuantizeImage's colour-error test runs
//     at Quantum precision, so a 16-bit montage keeps splitting until the error
//     is negligible and settles on flatter centroids.
//   * horizontal append. Layout does not change the result -- verified identical
//     to four decimal places -- but matching it makes the two paths directly
//     comparable instead of merely similar.
bool ImBuildPaletteAppend8(const RgbaF* pixels, std::size_t width,
                           std::size_t height, int colors, Palette* palette,
                           ColorTree* tree, std::string* error);

// Writes the dithered pixels back over a clone of the loaded image and encodes
// it to `path`, using `format` when non-empty.
bool ImStore(const LoadedImage& image, const PixelStore& store,
             const std::string& path, const std::string& format,
             std::string* error);

void ImFree(LoadedImage* image);

// Adopts the colormap of an already-quantized image verbatim as the palette and
// builds the search tree from those colours.  This is what lets ImageMagick own
// palette generation while rdither owns the dithering; see --palette-from.
bool ImAdoptPaletteFromColormap(const LoadedImage& image, int colors,
                                Palette* palette, ColorTree* tree,
                                std::string* error);

// Writes the palette out so it can be inspected, diffed, and read back with
// --palette-from.  An indexed palette PNG for `.png`, a plain hex table for
// `.txt`/`.text`.
bool ImWritePalette(const Palette& palette, const std::string& path,
                    std::string* error);

// Reads a palette back from what ImWritePalette wrote, which is what makes
// --palette-import the inverse of --palette-export rather than a second way to
// do the same thing with a different spelling.
//
// Two formats, chosen by suffix, matching ImWritePalette exactly:
//
//   .txt / .text   The line-oriented Q16 form.  This is the LOSSLESS one: the
//                  values are written with %.1f and a Q16 channel is an integer,
//                  so the text is bit-exact and survives a round trip unchanged.
//                  The hex column is 8-bit and is ignored on read -- it is there
//                  for a human, and preferring it would silently quantise.
//   anything else  An image, read the way --palette-from already reads one: its
//                  unique colours become the palette.  That handles the PNG8
//                  1xN strip this module writes, and equally
//                  `magick ... -unique-colors` output, so a palette captured
//                  from the reference pipeline imports unchanged.  It is 8-bit
//                  by construction, because PNG's PLTE is 8-bit.
//
// `expect_colors` is the value of --colors.  It is validated, not applied: a
// file holding a different number of colours is REJECTED with both numbers in
// the message, because the two inputs disagree about the single most important
// property of a palette and silently preferring either one produces output the
// user did not ask for.  Pass <= 0 to accept whatever the file contains.
//
// `palette` is filled; `tree` is built from it, so the imported palette is
// immediately usable by every engine.  The tree is built from the PALETTE, not
// from the image's pixels -- that is the whole point of importing, and it is why
// this is not just --palette-from.
//
// associate_alpha is left false: neither format carries alpha.  A caller that
// wants an associated palette must say so, because assuming it would change
// every pixel's arithmetic.
bool ImReadPalette(const std::string& path, int expect_colors, Palette* palette,
                   ColorTree* tree, std::string* error);

// Writes an RgbaF buffer (Q16 range) out as a 16-bit image, for diagnostics.
bool ImWriteRgbaF(const RgbaF* pixels, std::size_t width, std::size_t height,
                 const char* path);

}  // namespace rd
