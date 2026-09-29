// rd_im.cpp -- ImageMagick bridge implementation (MagickCore C API).
#include "rd_im.h"

#include <MagickCore/MagickCore.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rd {
namespace {

bool g_started = false;

std::string DescribeException(ExceptionInfo* exception) {
  if (exception == nullptr || exception->reason == nullptr) {
    return "unknown ImageMagick error";
  }
  // GetLocaleExceptionMessage() is public API; AcquireExceptionMessage() and
  // RelinquishMagick() live in uninstalled private headers.
  const char* message =
      GetLocaleExceptionMessage(exception->severity, exception->reason);
  return (message != nullptr) ? std::string(message)
                              : std::string(exception->reason);
}

// MagickCore/colorspace-private.h:IssRGBCompatibleColorspace().  The real
// function is in a private header that the Windows package does not install,
// so it is reproduced here verbatim.
bool IsSrgbCompatible(ColorspaceType colorspace) {
  return (colorspace == sRGBColorspace) || (colorspace == RGBColorspace) ||
         (colorspace == Adobe98Colorspace) || (colorspace == ProPhotoColorspace) ||
         (colorspace == DisplayP3Colorspace) || (colorspace == scRGBColorspace) ||
         (colorspace == TransparentColorspace) ||
         (colorspace == GRAYColorspace) || (colorspace == LinearGRAYColorspace);
}

// RAII for ExceptionInfo so no early return leaks it.
class ScopedException {
 public:
  ScopedException() : info_(AcquireExceptionInfo()) {}
  ~ScopedException() {
    if (info_ != nullptr) DestroyExceptionInfo(info_);
  }
  ExceptionInfo* get() const { return info_; }

 private:
  ExceptionInfo* info_;
};

}  // namespace

bool ImStartup(const std::string& magick_exe, std::string* error) {
  if (g_started) return true;
  // ImageMagick resolves its configuration (policy.xml, coder modules) relative
  // to the application path handed to MagickCoreGenesis().  Passing nullptr
  // makes ReadImage work but leaves WriteImage silently producing nothing, so
  // the real magick.exe is supplied when it can be located.
  if (!magick_exe.empty()) {
    MagickCoreGenesis(magick_exe.c_str(), MagickFalse);
  } else {
    MagickCoreGenesis(nullptr, MagickFalse);
  }
  g_started = true;
  (void)error;
  return true;
}

void ImShutdown() {
  if (!g_started) return;
  MagickCoreTerminus();
  g_started = false;
}

bool ImLoad(const std::string& path, LoadedImage* out, std::string* error) {
  ScopedException exc;
  ImageInfo* info = AcquireImageInfo();
  if (info == nullptr) {
    *error = "AcquireImageInfo failed";
    return false;
  }
  const std::string spec = path;
  (void)CopyMagickString(info->filename, spec.c_str(), MagickPathExtent);
  Image* image = ReadImage(info, exc.get());
  if (image == nullptr) {
    *error = "cannot read '" + path + "': " + DescribeException(exc.get());
    info = DestroyImageInfo(info);
    return false;
  }

  // quantize.c:ClassifyImageColors() transforms to sRGB when the requested
  // colorspace is undefined and the image is not already sRGB compatible.
  if (!IsSrgbCompatible(image->colorspace)) {
    if (TransformImageColorspace(image, sRGBColorspace, exc.get()) ==
        MagickFalse) {
      *error = "colorspace transform failed: " + DescribeException(exc.get());
      image = DestroyImage(image);
      info = DestroyImageInfo(info);
      return false;
    }
  }

  out->handle = image;
  out->width = image->columns;
  out->height = image->rows;
  out->has_alpha = image->alpha_trait != UndefinedPixelTrait;
  out->colorspace = static_cast<int>(image->colorspace);
  info = DestroyImageInfo(info);
  return true;
}

bool ImExtract(const LoadedImage& loaded, PixelStore* store,
               std::string* error) {
  Image* image = static_cast<Image*>(loaded.handle);
  ScopedException exc;
  if ((store->width() != loaded.width) || (store->height() != loaded.height)) {
    *error = "pixel store geometry does not match the image";
    return false;
  }
  for (std::size_t y = 0; y < loaded.height; ++y) {
    const Quantum* q = GetAuthenticPixels(image, 0, static_cast<ssize_t>(y),
                                          loaded.width, 1, exc.get());
    if (q == nullptr) {
      *error = "GetAuthenticPixels failed: " + DescribeException(exc.get());
      return false;
    }
    RgbaF* dst = store->row(y);
    const std::size_t channels = GetPixelChannels(image);
    for (std::size_t x = 0; x < loaded.width; ++x) {
      const Quantum* p = q + x * channels;
      dst[x].r = static_cast<float>(GetPixelRed(image, p));
      dst[x].g = static_cast<float>(GetPixelGreen(image, p));
      dst[x].b = static_cast<float>(GetPixelBlue(image, p));
      dst[x].a = static_cast<float>(
          (loaded.has_alpha != false) ? GetPixelAlpha(image, p)
                                      : static_cast<Quantum>(kOpaqueAlpha));
    }
  }
  return true;
}

bool ImBuildPalette(const LoadedImage& loaded, int colors, Palette* out,
                    std::string* error) {
  Image* source = static_cast<Image*>(loaded.handle);
  ScopedException exc;
  Image* clone = CloneImage(source, 0, 0, MagickFalse, exc.get());
  if (clone == nullptr) {
    *error = "CloneImage failed: " + DescribeException(exc.get());
    return false;
  }
  ImageInfo* info = AcquireImageInfo();
  QuantizeInfo* qi = AcquireQuantizeInfo(info);
  qi->number_colors = static_cast<std::size_t>(colors);
  qi->tree_depth = 0;                        // let IM pick Log4(colors)+2 (-1)
  qi->colorspace = UndefinedColorspace;      // match the CLI default
  qi->dither_method = RiemersmaDitherMethod; // selects the shallower tree
  qi->measure_error = MagickFalse;

  const MagickBooleanType ok = QuantizeImage(qi, clone, exc.get());
  if (ok == MagickFalse) {
    *error = "QuantizeImage failed: " + DescribeException(exc.get());
    qi = DestroyQuantizeInfo(qi);
    info = DestroyImageInfo(info);
    clone = DestroyImage(clone);
    return false;
  }

  out->count = static_cast<int>(clone->colors);
  if (out->count <= 0 || out->count > kMaxColormapSize) {
    *error = "ImageMagick returned an unusable colormap size";
    qi = DestroyQuantizeInfo(qi);
    info = DestroyImageInfo(info);
    clone = DestroyImage(clone);
    return false;
  }
  // quantize.c:SetAssociatedAlpha(): alpha participates only when the source
  // image actually carries an alpha channel.
  out->associate_alpha = source->alpha_trait != UndefinedPixelTrait;
  for (int i = 0; i < out->count; ++i) {
    const PixelInfo& e = clone->colormap[i];
    out->entries[i].r = static_cast<double>(e.red);
    out->entries[i].g = static_cast<double>(e.green);
    out->entries[i].b = static_cast<double>(e.blue);
    out->entries[i].a = static_cast<double>(e.alpha);
  }

  qi = DestroyQuantizeInfo(qi);
  info = DestroyImageInfo(info);
  clone = DestroyImage(clone);
  return true;
}

bool ImReferenceDither(const LoadedImage& loaded, int colors, PixelStore* out,
                       std::string* error) {
  Image* source = static_cast<Image*>(loaded.handle);
  ScopedException exc;
  Image* clone = CloneImage(source, 0, 0, MagickFalse, exc.get());
  if (clone == nullptr) {
    *error = "CloneImage failed: " + DescribeException(exc.get());
    return false;
  }
  ImageInfo* info = AcquireImageInfo();
  QuantizeInfo* qi = AcquireQuantizeInfo(info);
  qi->number_colors = static_cast<std::size_t>(colors);
  qi->tree_depth = 0;
  qi->colorspace = UndefinedColorspace;
  qi->dither_method = RiemersmaDitherMethod;
  qi->measure_error = MagickFalse;
  const MagickBooleanType ok = QuantizeImage(qi, clone, exc.get());
  qi = DestroyQuantizeInfo(qi);
  info = DestroyImageInfo(info);
  if (ok == MagickFalse) {
    *error = "reference QuantizeImage failed: " + DescribeException(exc.get());
    clone = DestroyImage(clone);
    return false;
  }

  // The clone is now PseudoClass; read the *colormap colours* so the comparison
  // is on visible pixels rather than on indices.
  const bool has_alpha = source->alpha_trait != UndefinedPixelTrait;
  for (std::size_t y = 0; y < out->height(); ++y) {
    const Quantum* q = GetAuthenticPixels(clone, 0, static_cast<ssize_t>(y),
                                          out->width(), 1, exc.get());
    if (q == nullptr) {
      *error = "GetAuthenticPixels failed: " + DescribeException(exc.get());
      clone = DestroyImage(clone);
      return false;
    }
    RgbaF* dst = out->row(y);
    const std::size_t channels = GetPixelChannels(clone);
    for (std::size_t x = 0; x < out->width(); ++x) {
      const Quantum* p = q + x * channels;
      // clone->colormap is a PixelInfo* (MagickRealType channels), not Quantum*,
      // so the PseudoClass and DirectClass cases are read separately.
      if (clone->storage_class == PseudoClass) {
        const std::size_t entry = static_cast<std::size_t>(
            static_cast<double>(GetPixelIndex(clone, p)));
        if (entry < clone->colors) {
          const PixelInfo& e = clone->colormap[entry];
          dst[x].r = clamp_to_quantum(static_cast<double>(e.red));
          dst[x].g = clamp_to_quantum(static_cast<double>(e.green));
          dst[x].b = clamp_to_quantum(static_cast<double>(e.blue));
          dst[x].a = static_cast<float>(has_alpha
                                            ? GetPixelAlpha(clone, p)
                                            : static_cast<Quantum>(kOpaqueAlpha));
          continue;
        }
      }
      dst[x].r = static_cast<float>(GetPixelRed(clone, p));
      dst[x].g = static_cast<float>(GetPixelGreen(clone, p));
      dst[x].b = static_cast<float>(GetPixelBlue(clone, p));
      dst[x].a = static_cast<float>(has_alpha ? GetPixelAlpha(clone, p)
                                              : static_cast<Quantum>(kOpaqueAlpha));
    }
  }
  clone = DestroyImage(clone);
  return true;
}

DiffResult CompareStores(const PixelStore& a, const PixelStore& b,
                         bool compare_alpha) {
  DiffResult result;
  if ((a.width() != b.width()) || (a.height() != b.height())) return result;
  double sum_sq = 0.0;
  long long samples = 0;
  for (std::size_t y = 0; y < a.height(); ++y) {
    const RgbaF* pa = a.row(y);
    const RgbaF* pb = b.row(y);
    for (std::size_t x = 0; x < a.width(); ++x) {
      bool differs = false;
      const float ca[4] = {pa[x].r, pa[x].g, pa[x].b, pa[x].a};
      const float cb[4] = {pb[x].r, pb[x].g, pb[x].b, pb[x].a};
      const int channels = compare_alpha ? 4 : 3;
      for (int c = 0; c < channels; ++c) {
        const double delta =
            (static_cast<double>(ca[c]) - static_cast<double>(cb[c])) * kQuantumScale;
        sum_sq += delta * delta;
        ++samples;
        const int abs_delta = static_cast<int>(std::fabs(delta) * kQuantumRange + 0.5);
        if (abs_delta != 0) differs = true;
        if (abs_delta > result.max_channel_delta) {
          result.max_channel_delta = abs_delta;
        }
      }
      if (differs) ++result.differing_pixels;
      ++result.total_pixels;
    }
  }
  if (samples > 0) result.rmse = std::sqrt(sum_sq / static_cast<double>(samples));
  return result;
}

// Zero a row that QueueAuthenticPixels has just handed us, before any of it is
// assigned.
//
// Every write loop in this file produces four values per pixel -- R, G, B and
// usually A -- but ImageMagick 7's GetPixelChannels is not always four, and the
// loops that omit alpha entirely produce three.  Whatever the extra channels
// hold at that point is whatever the heap block held before, and the PNG coder's
// type optimisation scans the full channel set to choose between TrueColorAlpha,
// PaletteAlpha, Gray and Bilevel.  An image is therefore encoded differently
// depending on stale memory, which is how the same dithered bytes came out as a
// 1-bit Bilevel on one run and an 8-bit PaletteAlpha on the next.  It is not
// exotic: it was measured at 3 runs in 16, and it hits the GPU engines harder
// than the CPU ones only because their extra host allocations rearrange the heap.
//
// The fix is one memset per row.  It is written as a function rather than
// repeated because there are five call sites and the reasoning is long, and a
// long comment copied five times is five comments that will drift apart.
inline void ZeroQueuedRow(Quantum* q, std::size_t width, std::size_t channels) {
  std::memset(q, 0, channels * width * sizeof(Quantum));
}

bool ImStore(const LoadedImage& loaded, const PixelStore& store,
             const std::string& path, const std::string& format,
             std::string* error) {
  Image* source = static_cast<Image*>(loaded.handle);
  ScopedException exc;
  Image* out = CloneImage(source, 0, 0, MagickFalse, exc.get());
  if (out == nullptr) {
    *error = "CloneImage failed: " + DescribeException(exc.get());
    return false;
  }
  // RD_STORE_DUMP=<prefix> writes the store's raw bytes on entry and on exit, so
  // a store that is correct when the engine returns but wrong by the time it is
  // written is distinguishable from one that was always wrong.  That is the
  // difference between "the dither is broken" and "something clobbered the
  // buffer in between", which no amount of staring at the output image can tell.
  if (const char* prefix = std::getenv("RD_STORE_DUMP")) {
    auto dump = [&](const char* tag) {
      const std::string p = std::string(prefix) + tag + ".bin";
      FILE* f = std::fopen(p.c_str(), "wb");
      if (f == nullptr) return;
      const std::size_t n = store.width() * store.height() * sizeof(RgbaF);
      std::fwrite(store.data(), 1, n, f);
      std::fclose(f);
    };
    dump(".pre");
    {
      const std::string p = std::string(prefix) + ".meta.txt";
      FILE* f = std::fopen(p.c_str(), "wb");
      if (f != nullptr) {
        std::fprintf(f, "cs=%d has_alpha=%d channels=%lu w=%zu h=%zu\n",
                     static_cast<int>(out->colorspace), loaded.has_alpha ? 1 : 0,
                     static_cast<unsigned long>(GetPixelChannels(out)),
                     store.width(), store.height());
        std::fclose(f);
      }
    }
  }
  for (std::size_t y = 0; y < store.height(); ++y) {
    Quantum* q = QueueAuthenticPixels(out, 0, static_cast<ssize_t>(y),
                                      store.width(), 1, exc.get());
    if (q == nullptr) {
      *error = "QueueAuthenticPixels failed: " + DescribeException(exc.get());
      out = DestroyImage(out);
      return false;
    }
    const RgbaF* src = store.row(y);
    const std::size_t channels = GetPixelChannels(out);
    ZeroQueuedRow(q, store.width(), channels);
    for (std::size_t x = 0; x < store.width(); ++x) {
      Quantum* p = q + x * channels;
      SetPixelRed(out, static_cast<Quantum>(src[x].r), p);
      SetPixelGreen(out, static_cast<Quantum>(src[x].g), p);
      SetPixelBlue(out, static_cast<Quantum>(src[x].b), p);
      if (loaded.has_alpha) SetPixelAlpha(out, static_cast<Quantum>(src[x].a), p);
    }
    if (SyncAuthenticPixels(out, exc.get()) == MagickFalse) {
      *error = "SyncAuthenticPixels failed: " + DescribeException(exc.get());
      out = DestroyImage(out);
      return false;
    }
  }

  ImageInfo* info = AcquireImageInfo();
  if (const char* prefix = std::getenv("RD_STORE_DUMP")) {
    const std::string p = std::string(prefix) + ".post.bin";
    FILE* f = std::fopen(p.c_str(), "wb");
    if (f != nullptr) {
      std::fwrite(store.data(), 1, store.width() * store.height() * sizeof(RgbaF), f);
      std::fclose(f);
    }
  }
  if (!format.empty()) {
    // Setting info->magick makes WriteImage pick the coder explicitly, so the
    // output does not depend on the destination file extension.
    (void)CopyMagickString(info->magick, format.c_str(), MagickPathExtent);
  } else if (out->magick[0] != '\0') {
    // Otherwise inherit the coder from the image we cloned.
    (void)CopyMagickString(info->magick, out->magick, MagickPathExtent);
  }
  // WriteImage() clones image_info and then OVERWRITES the clone's filename
  // with image->filename (see constitute.c), so image_info->filename is only
  // consulted when reading.  Setting only image_info->filename leaves
  // image->filename empty, which the PNG coder happily writes to stdout while
  // still reporting success.  Both are set for clarity.
  (void)CopyMagickString(info->filename, path.c_str(), MagickPathExtent);
  (void)CopyMagickString(out->filename, path.c_str(), MagickPathExtent);
  const MagickBooleanType ok = WriteImage(info, out, exc.get());
  if (ok == MagickFalse) {
    *error = "cannot write '" + path + "': " + DescribeException(exc.get());
    info = DestroyImageInfo(info);
    out = DestroyImage(out);
    return false;
  }
  info = DestroyImageInfo(info);
  out = DestroyImage(out);
  return true;
}

// Copies the colormap ImageMagick just produced into `palette`, drops exact
// duplicates, and builds the search tree from the same pixels -- so the octree
// the dither descends and the palette it maps to come from one source.
//
// The dedup is what `magick ... -unique-colors` does, and the reference video
// pipeline depends on it: the quantizer emits exact duplicate colormap entries,
// and they waste palette slots.  Measured on a 1080p sample, `-colors 16` alone
// gave 21.4% mean saturation and `-colors 16 -unique-colors` gave 41.2%.
static bool HarvestColormap(const Image* image, const RgbaF* pixels,
                            std::size_t width, std::size_t height, int colors,
                            Palette* palette, ColorTree* tree,
                            std::string* error) {
  palette->count = static_cast<int>(image->colors);
  palette->associate_alpha = false;  // alpha is not part of the palette search
  if (palette->count <= 0 || palette->count > kMaxColormapSize) {
    *error = "ImageMagick returned an unusable colormap size";
    return false;
  }
  for (int i = 0; i < palette->count; ++i) {
    const PixelInfo& e = image->colormap[i];
    palette->entries[i].r = static_cast<double>(e.red);
    palette->entries[i].g = static_cast<double>(e.green);
    palette->entries[i].b = static_cast<double>(e.blue);
    palette->entries[i].a = static_cast<double>(e.alpha);
  }
  {
    int out = 0;
    for (int i = 0; i < palette->count; ++i) {
      bool dup = false;
      for (int k = 0; k < out; ++k) {
        if (palette->entries[k].r == palette->entries[i].r &&
            palette->entries[k].g == palette->entries[i].g &&
            palette->entries[k].b == palette->entries[i].b) {
          dup = true;
          break;
        }
      }
      if (!dup) palette->entries[out++] = palette->entries[i];
    }
    palette->count = out;
  }
  // Grayscale detection mirrors the CLI's.
  bool grayscale = true;
  for (std::size_t i = 0; i < width * height && grayscale; ++i) {
    grayscale = (pixels[i].r == pixels[i].g) && (pixels[i].g == pixels[i].b);
  }
  tree->Build(pixels, width, height, colors, /*associate_alpha=*/false, grayscale);
  return true;
}

bool ImBuildPaletteFromPixels(const RgbaF* pixels, std::size_t width,
                              std::size_t height, int colors, Palette* palette,
                              ColorTree* tree, std::string* error) {
  ScopedException exc;
  ImageInfo* info = AcquireImageInfo();
  Image* image = AcquireImage(info, exc.get());
  if (image == nullptr) {
    *error = "AcquireImage failed";
    info = DestroyImageInfo(info);
    return false;
  }
  // Match a decoded sRGB 16-bit frame so QuantizeImage() behaves identically.
  (void)CopyMagickString(image->filename, "video-frame", MagickPathExtent);
  (void)CopyMagickString(image->magick, "RGB", MagickPathExtent);
  image->colorspace = sRGBColorspace;
  image->alpha_trait = BlendPixelTrait;
  image->depth = 16;
  image->columns = width;
  image->rows = height;
  image->resolution.x = 72.0;
  image->resolution.y = 72.0;
  image->units = PixelsPerInchResolution;
  if (SetImageExtent(image, width, height, exc.get()) == MagickFalse) {
    *error = "SetImageExtent failed: " + DescribeException(exc.get());
    image = DestroyImage(image);
    info = DestroyImageInfo(info);
    return false;
  }
  const std::size_t channels = GetPixelChannels(image);
  for (std::size_t y = 0; y < height; ++y) {
    Quantum* q = QueueAuthenticPixels(image, 0, static_cast<ssize_t>(y), width, 1,
                                      exc.get());
    if (q == nullptr) {
      *error = "QueueAuthenticPixels failed: " + DescribeException(exc.get());
      image = DestroyImage(image);
      info = DestroyImageInfo(info);
      return false;
    }
    const RgbaF* src = pixels + y * width;
    ZeroQueuedRow(q, width, channels);
    for (std::size_t x = 0; x < width; ++x) {
      Quantum* p = q + x * channels;
      SetPixelRed(image, static_cast<Quantum>(src[x].r), p);
      SetPixelGreen(image, static_cast<Quantum>(src[x].g), p);
      SetPixelBlue(image, static_cast<Quantum>(src[x].b), p);
      SetPixelAlpha(image, static_cast<Quantum>(src[x].a), p);
    }
    if (SyncAuthenticPixels(image, exc.get()) == MagickFalse) {
      *error = "SyncAuthenticPixels failed: " + DescribeException(exc.get());
      image = DestroyImage(image);
      info = DestroyImageInfo(info);
      return false;
    }
  }

  QuantizeInfo* qi = AcquireQuantizeInfo(info);
  qi->number_colors = static_cast<std::size_t>(colors);
  qi->tree_depth = 0;
  qi->colorspace = UndefinedColorspace;
  qi->dither_method = RiemersmaDitherMethod;
  qi->measure_error = MagickFalse;
  const MagickBooleanType ok = QuantizeImage(qi, image, exc.get());
  qi = DestroyQuantizeInfo(qi);
  if (ok == MagickFalse) {
    *error = "QuantizeImage failed: " + DescribeException(exc.get());
    image = DestroyImage(image);
    info = DestroyImageInfo(info);
    return false;
  }

  if (!HarvestColormap(image, pixels, width, height, colors, palette, tree,
                       error)) {
    image = DestroyImage(image);
    info = DestroyImageInfo(info);
    return false;
  }
  image = DestroyImage(image);
  info = DestroyImageInfo(info);
  return true;
}

void ImFree(LoadedImage* image) {
  if (image == nullptr || image->handle == nullptr) return;
  Image* img = static_cast<Image*>(image->handle);
  img = DestroyImage(img);
  image->handle = nullptr;
}

bool ImWriteRgbaF(const RgbaF* pixels, std::size_t width, std::size_t height,
                 const char* path) {
  ScopedException exc;
  ImageInfo* info = AcquireImageInfo();
  Image* image = AcquireImage(info, exc.get());
  if (image == nullptr) {
    info = DestroyImageInfo(info);
    return false;
  }
  (void)CopyMagickString(image->filename, "montage", MagickPathExtent);
  (void)CopyMagickString(image->magick, "RGB", MagickPathExtent);
  image->colorspace = sRGBColorspace;
  image->alpha_trait = BlendPixelTrait;
  image->depth = 16;
  image->columns = width;
  image->rows = height;
  image->resolution.x = 72.0;
  image->resolution.y = 72.0;
  image->units = PixelsPerInchResolution;
  if (SetImageExtent(image, width, height, exc.get()) == MagickFalse) {
    image = DestroyImage(image);
    info = DestroyImageInfo(info);
    return false;
  }
  const std::size_t channels = GetPixelChannels(image);
  for (std::size_t y = 0; y < height; ++y) {
    Quantum* q = QueueAuthenticPixels(image, 0, static_cast<ssize_t>(y), width, 1,
                                      exc.get());
    if (q == nullptr) continue;
    const RgbaF* src = pixels + y * width;
    ZeroQueuedRow(q, width, channels);
    for (std::size_t x = 0; x < width; ++x) {
      Quantum* p = q + x * channels;
      SetPixelRed(image, static_cast<Quantum>(src[x].r), p);
      SetPixelGreen(image, static_cast<Quantum>(src[x].g), p);
      SetPixelBlue(image, static_cast<Quantum>(src[x].b), p);
      SetPixelAlpha(image, static_cast<Quantum>(src[x].a), p);
    }
    (void)SyncAuthenticPixels(image, exc.get());
  }
  {
    // WriteImage() overwrites image_info->filename with image->filename, so the
    // output name has to live on the image, exactly as ImStore does it.
    (void)CopyMagickString(image->filename, path, MagickPathExtent);
    (void)WriteImage(info, image, exc.get());
  }
  image = DestroyImage(image);
  info = DestroyImageInfo(info);
  return true;
}
// Adopts the colormap of an already-quantized image as the palette, instead of
// building one.  This is the seam that lets ImageMagick own palette generation
// while rdither owns the dithering:
//
//     magick f1..f30 +append -colors 16 -unique-colors palette.png
//     rdither --video --palette-from palette.png in.mp4 out.mp4
//
// The colormap is taken verbatim -- no re-quantization -- because re-quantizing is
// exactly what changes the palette, and keeping the one ImageMagick already
// produced is the entire point.  The search tree is then built from a swatch image
// holding those colours, so the tree the dither descends and the palette it maps
// to come from the same set.
bool ImAdoptPaletteFromColormap(const LoadedImage& image, int colors,
                                Palette* palette, ColorTree* tree,
                                std::string* error) {
  Image* img = static_cast<Image*>(image.handle);
  if (img == nullptr) {
    *error = "no loaded image";
    return false;
  }
  int n = 0;
  if (img->colors > 0 && img->colormap != nullptr) {
    n = static_cast<int>(img->colors);
    if (colors > 0 && n > colors) n = colors;
    for (int i = 0; i < n; ++i) {
      const PixelInfo& e = img->colormap[i];
      palette->entries[i].r = static_cast<double>(e.red);
      palette->entries[i].g = static_cast<double>(e.green);
      palette->entries[i].b = static_cast<double>(e.blue);
      palette->entries[i].a = static_cast<double>(e.alpha);
    }
  } else {
    // No colormap: collect the distinct pixel values instead.  This is what
    // `magick ... -unique-colors` emits, and also what rdither's own
    // --palette-export writes, so one reader handles both -- and any other tool's
    // palette image -- without depending on which coder produced it.
    const int limit = colors > 0 ? colors : kMaxColormapSize;
    n = 0;
    const std::size_t w = img->columns, h = img->rows;
    std::vector<Quantum> buffer(w * h * 3);
    if (ExportImagePixels(img, 0, 0, w, h, "RGB", QuantumPixel, buffer.data(),
                          nullptr) == MagickFalse) {
      *error = "ExportImagePixels failed on the palette image";
      return false;
    }
    for (std::size_t i = 0; i < w * h && n < limit; ++i) {
      const double r = static_cast<double>(buffer[3 * i + 0]);
      const double g = static_cast<double>(buffer[3 * i + 1]);
      const double b = static_cast<double>(buffer[3 * i + 2]);
      bool dup = false;
      for (int k = 0; k < n; ++k) {
        if (palette->entries[k].r == r && palette->entries[k].g == g &&
            palette->entries[k].b == b) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        palette->entries[n].r = r;
        palette->entries[n].g = g;
        palette->entries[n].b = b;
        palette->entries[n].a = static_cast<double>(kOpaqueAlpha);
        ++n;
      }
    }
    if (n == 0) {
      *error = "the image has neither a colormap nor readable pixels";
      return false;
    }
  }
  palette->count = n;
  palette->associate_alpha = false;
  // Drop exact duplicates, matching `-unique-colors`: the quantizer emits them,
  // and each one would otherwise waste a palette slot.
  {
    int out = 0;
    for (int i = 0; i < palette->count; ++i) {
      bool dup = false;
      for (int k = 0; k < out; ++k) {
        if (palette->entries[k].r == palette->entries[i].r &&
            palette->entries[k].g == palette->entries[i].g &&
            palette->entries[k].b == palette->entries[i].b) {
          dup = true;
          break;
        }
      }
      if (!dup) palette->entries[out++] = palette->entries[i];
    }
    palette->count = out;
  }
  if (palette->count <= 0) {
    *error = "the colormap deduplicated to nothing";
    return false;
  }

  // Swatch image: one block per palette entry, so the octree the dither searches
  // is built from exactly the colours it will map to.
  const int n_col = palette->count;
  const int block =
      std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n_col)))));
  const int rows = (n_col + block - 1) / block;
  const std::size_t w = static_cast<std::size_t>(block) * 8;
  const std::size_t h = static_cast<std::size_t>(rows) * 8;
  std::vector<RgbaF> swatch(w * h);
  bool grayscale = true;
  for (int i = 0; i < n_col; ++i) {
    const PaletteEntry& pe = palette->entries[i];
    if (pe.r != pe.g || pe.g != pe.b) grayscale = false;
    const std::size_t gx = static_cast<std::size_t>(i % block) * 8;
    const std::size_t gy = static_cast<std::size_t>(i / block) * 8;
    for (std::size_t y = 0; y < 8; ++y) {
      RgbaF* row = swatch.data() + (gy + y) * w + gx;
      for (std::size_t x = 0; x < 8; ++x) {
        row[x].r = static_cast<float>(pe.r);
        row[x].g = static_cast<float>(pe.g);
        row[x].b = static_cast<float>(pe.b);
        row[x].a = static_cast<float>(pe.a);
      }
    }
  }
  tree->Build(swatch.data(), w, h, n_col, /*associate_alpha=*/false, grayscale);
  return true;
}

// The reference video pipeline's palette step, in-process.  See the declaration
// in rd_im.h for why 8-bit, alpha-free and horizontally appended all matter.
bool ImBuildPaletteAppend8(const RgbaF* pixels, std::size_t width,
                           std::size_t height, int colors, Palette* palette,
                           ColorTree* tree, std::string* error) {
  ScopedException exc;
  ImageInfo* info = AcquireImageInfo();
  Image* image = AcquireImage(info, exc.get());
  if (image == nullptr) {
    *error = "AcquireImage failed";
    info = DestroyImageInfo(info);
    return false;
  }
  (void)CopyMagickString(image->filename, "video-palette", MagickPathExtent);
  (void)CopyMagickString(image->magick, "RGB", MagickPathExtent);
  image->colorspace = sRGBColorspace;
  // UndefinedPixelTrait, not BlendPixelTrait: quantize.c shortens the octree by
  // one level when an alpha channel is present, so an alpha-carrying palette
  // image is built from a different tree than the reference's PPM strip.
  image->alpha_trait = UndefinedPixelTrait;
  image->depth = 8;  // the reference extracts 8-bit PPM
  image->columns = width;
  image->rows = height;
  image->resolution.x = 72.0;
  image->resolution.y = 72.0;
  image->units = PixelsPerInchResolution;
  if (SetImageExtent(image, width, height, exc.get()) == MagickFalse) {
    *error = "SetImageExtent failed: " + DescribeException(exc.get());
    image = DestroyImage(image);
    info = DestroyImageInfo(info);
    return false;
  }
  const std::size_t channels = GetPixelChannels(image);
  for (std::size_t y = 0; y < height; ++y) {
    Quantum* q = QueueAuthenticPixels(image, 0, static_cast<ssize_t>(y), width, 1,
                                      exc.get());
    if (q == nullptr) {
      *error = "QueueAuthenticPixels failed: " + DescribeException(exc.get());
      image = DestroyImage(image);
      info = DestroyImageInfo(info);
      return false;
    }
    const RgbaF* src = pixels + y * width;
    ZeroQueuedRow(q, width, channels);
    for (std::size_t x = 0; x < width; ++x) {
      Quantum* p = q + x * channels;
      // Round to 8-bit precision but keep Q16 scale.  This is a Q16 build, so
      // Quantum spans 0..65535 and writing 0..255 directly makes the image 257x
      // too dark -- the octree then sees almost no detail and returns near-pure
      // primaries.  A PPM round trip multiplies by 257; so must this.
      const double r = static_cast<double>(src[x].r) / 257.0;
      const double g = static_cast<double>(src[x].g) / 257.0;
      const double b = static_cast<double>(src[x].b) / 257.0;
      const double cr = r < 0.0 ? 0.0 : (r > 255.0 ? 255.0 : r);
      const double cg = g < 0.0 ? 0.0 : (g > 255.0 ? 255.0 : g);
      const double cb = b < 0.0 ? 0.0 : (b > 255.0 ? 255.0 : b);
      SetPixelRed(image, static_cast<Quantum>(cr * 257.0), p);
      SetPixelGreen(image, static_cast<Quantum>(cg * 257.0), p);
      SetPixelBlue(image, static_cast<Quantum>(cb * 257.0), p);
    }
    if (SyncAuthenticPixels(image, exc.get()) == MagickFalse) {
      *error = "SyncAuthenticPixels failed: " + DescribeException(exc.get());
      image = DestroyImage(image);
      info = DestroyImageInfo(info);
      return false;
    }
  }

  QuantizeInfo* qi = AcquireQuantizeInfo(info);
  qi->number_colors = static_cast<std::size_t>(colors);
  qi->tree_depth = 0;  // let IM derive it, as the CLI does
  qi->colorspace = UndefinedColorspace;
  qi->dither_method = RiemersmaDitherMethod;
  qi->measure_error = MagickFalse;
  const MagickBooleanType ok = QuantizeImage(qi, image, exc.get());
  qi = DestroyQuantizeInfo(qi);
  if (ok == MagickFalse) {
    *error = "QuantizeImage failed: " + DescribeException(exc.get());
    image = DestroyImage(image);
    info = DestroyImageInfo(info);
    return false;
  }

  if (!HarvestColormap(image, pixels, width, height, colors, palette, tree,
                       error)) {
    image = DestroyImage(image);
    info = DestroyImageInfo(info);
    return false;
  }
  image = DestroyImage(image);
  info = DestroyImageInfo(info);
  return true;
}

// Writes the palette out so it can be inspected, version-controlled, and fed
// straight back in with --palette-from.
//
// Two formats, chosen by extension:
//   .png  an indexed palette image, which is what --palette-from reads back
//   .txt  a plain hex list, for diffing and for other tools
bool ImWritePalette(const Palette& palette, const std::string& path,
                    std::string* error) {
  if (palette.count <= 0) {
    *error = "palette is empty";
    return false;
  }
  // Each suffix test is guarded by its OWN length.  Guarding the pair with a
  // single `size() >= 4` is the bug this replaces: a path of exactly four
  // characters passes that check, fails the ".txt" test, and then evaluates
  // `path.compare(size() - 5, 5, ".text")` with size() - 5 underflowing a
  // size_t.  std::string::compare throws std::out_of_range for pos > size(),
  // nothing catches it, and the process dies with 0xC0000409 -- confirmed, not
  // inferred: `--palette-export abcd` terminated after building the palette.
  //
  // The failure was late and loud (a crash, after the expensive palette stage),
  // which is the only reason it is worth writing down: the underflow is silent
  // arithmetic and its consequence is a crash in a different function.
  const bool text =
      (path.size() >= 4 && path.compare(path.size() - 4, 4, ".txt") == 0) ||
      (path.size() >= 5 && path.compare(path.size() - 5, 5, ".text") == 0);
  if (text) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
      *error = "cannot open " + path + " for writing";
      return false;
    }
    std::fprintf(f, "# rdither palette, %d colours, Q16 (0..65535) and 8-bit hex\n",
                 palette.count);
    std::fprintf(f, "# count\tr\tg\tb\thex\n");
    for (int i = 0; i < palette.count; ++i) {
      const PaletteEntry& e = palette.entries[i];
      std::fprintf(f, "%d\t%.1f\t%.1f\t%.1f\t%02X%02X%02X\n", i, e.r, e.g, e.b,
                   static_cast<unsigned>(e.r / 257.0 + 0.5),
                   static_cast<unsigned>(e.g / 257.0 + 0.5),
                   static_cast<unsigned>(e.b / 257.0 + 0.5));
    }
    std::fclose(f);
    return true;
  }

  ScopedException exc;
  ImageInfo* info = AcquireImageInfo();
  Image* image = AcquireImage(info, exc.get());
  if (image == nullptr) {
    *error = "AcquireImage failed";
    info = DestroyImageInfo(info);
    return false;
  }
  const std::size_t n = static_cast<std::size_t>(palette.count);
  (void)CopyMagickString(image->filename, "rdither-palette", MagickPathExtent);
  (void)CopyMagickString(image->magick, "PNG", MagickPathExtent);
  image->colorspace = sRGBColorspace;
  image->alpha_trait = UndefinedPixelTrait;
  // PNG palette entries are 8-bit PLTE, so a palette PNG is 8-bit by
  // construction; 16-bit here would make the coder emit a DirectClass RGB image
  // with no colormap, which --palette-from could not read back.  Writing through
  // the PNG8: coder forces the indexed form.
  image->depth = 8;
  (void)CopyMagickString(image->magick, "PNG8", MagickPathExtent);
  image->columns = n;
  image->rows = 1;
  image->resolution.x = 72.0;
  image->resolution.y = 72.0;
  image->units = PixelsPerInchResolution;
  if (SetImageExtent(image, n, 1, exc.get()) == MagickFalse) {
    *error = "SetImageExtent failed: " + DescribeException(exc.get());
    image = DestroyImage(image);
    info = DestroyImageInfo(info);
    return false;
  }
  // Written as a plain 1 x N strip of the colours rather than a hand-built indexed
  // image.  Building PseudoClass by hand proved fragile: the PNG coder re-derives
  // the colormap and quietly dropped 16 entries to 4.  A plain strip needs no
  // colormap at all, and ImAdoptPaletteFromColormap() recovers the palette from
  // its unique pixel values -- which is exactly what `magick ... -unique-colors`
  // produces, so one reader handles both our files and the reference pipeline's.
  const std::size_t channels = GetPixelChannels(image);
  Quantum* q = QueueAuthenticPixels(image, 0, 0, n, 1, exc.get());
  if (q == nullptr) {
    *error = "QueueAuthenticPixels failed: " + DescribeException(exc.get());
    image = DestroyImage(image);
    info = DestroyImageInfo(info);
    return false;
  }
  ZeroQueuedRow(q, n, channels);
  for (std::size_t x = 0; x < n; ++x) {
    Quantum* p = q + x * channels;
    SetPixelRed(image, static_cast<Quantum>(palette.entries[x].r), p);
    SetPixelGreen(image, static_cast<Quantum>(palette.entries[x].g), p);
    SetPixelBlue(image, static_cast<Quantum>(palette.entries[x].b), p);
  }
  if (SyncAuthenticPixels(image, exc.get()) == MagickFalse) {
    *error = "SyncAuthenticPixels failed: " + DescribeException(exc.get());
    image = DestroyImage(image);
    info = DestroyImageInfo(info);
    return false;
  }
  // WriteImage overwrites image_info->filename with image->filename, so the output
  // name has to be carried on the image.
  (void)CopyMagickString(image->filename, ("PNG8:" + path).c_str(),
                         MagickPathExtent);
  const MagickBooleanType ok = WriteImage(info, image, exc.get());
  image = DestroyImage(image);
  info = DestroyImageInfo(info);
  if (ok == MagickFalse) {
    *error = "WriteImage failed for " + path;
    return false;
  }
  return true;
}

// Adopt the TREE's colormap into `palette`, discarding whatever ordering the
// caller arrived with.
//
// This is the step that makes an import an import rather than a copy.  Two
// independent reasons, and the second is the one that actually bit:
//
//  1. Precision.  ColorTree::Build is fed an RgbaF strip, and RgbaF is float, so
//     a Q16 value like 12347.8 becomes 12347.7998.  Adopting the tree's value
//     makes the palette agree with what the engines will actually compare
//     against.  Without this the run dies on
//     "colormap entry 0 differs (IM 12347.8000 / tree 12347.7998)".
//
//  2. ORDER.  The tree emits its colormap in its own traversal order, and every
//     engine indexes palette.entries[] with a color_number that refers to a
//     position in THAT order.  Keeping the file's order makes the walk write
//     palette[i] for a colour_number the tree assigned to a different entry:
//     every pixel still gets *a* palette colour, the palette still has the right
//     number of entries, and the output is quietly wrong.  This is invisible on
//     a derived palette, because ImageMagick's DefineImageColormap and this tree
//     are transliterations of one another and agree on order as well as values.
//     An imported file has no such guarantee and never had a reason to.
static void AdoptTreeColormap(Palette* palette, const ColorTree* tree) {
  const PaletteEntry* cm = tree->colormap();
  const int n = tree->color_count();
  for (int i = 0; i < palette->count && i < n; ++i) {
    palette->entries[i] = cm[i];
  }
  palette->count = n;
}

// Sort entries into a canonical order before the tree sees them.
//
// Without this, a .txt round trip is content-stable but ORDER-unstable, and the
// order is not cosmetic.  ColorTree::Build's traversal order depends on the order
// the colours arrive in, so feeding it an already-reordered palette re-reorders
// it again: measured, importing the exported palette swapped entries 11 and 12,
// and would have swapped them back on the next pass.  Since the dither breaks
// exact distance ties by INDEX, a palette whose order oscillates produces render
// output that oscillates with it -- so "export then import" would not reproduce
// the original image even though the colormap is provably the same set.
//
// A canonical order makes the whole thing idempotent, and costs nothing: the
// tree was going to discard the file's ordering anyway, so the only thing being
// given up is an ordering that was never stable in the first place.  Sorted by
// (r, g, b) with alpha last, which is total and obvious.
static void SortPaletteCanonical(Palette* palette) {
  std::sort(palette->entries, palette->entries + palette->count,
            [](const PaletteEntry& a, const PaletteEntry& b) {
              if (a.r != b.r) return a.r < b.r;
              if (a.g != b.g) return a.g < b.g;
              if (a.b != b.b) return a.b < b.b;
              return a.a < b.a;
            });
}

bool ImReadPalette(const std::string& path, int expect_colors, Palette* palette,
                   ColorTree* tree, std::string* error) {
  // Suffix dispatch, each test guarded by its own length.  The same shape as the
  // writer's, and for the same reason: a shared guard is what produced the
  // four-character-path crash this file used to have.
  const bool text =
      (path.size() >= 4 && path.compare(path.size() - 4, 4, ".txt") == 0) ||
      (path.size() >= 5 && path.compare(path.size() - 5, 5, ".text") == 0);

  if (text) {
    // Parse the Q16 text form.  Deliberately strict: a line that does not parse
    // is an error naming the line, because a palette that silently loses entries
    // produces a dither against the wrong number of colours and nothing about
    // the output says so.
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
      *error = "cannot open " + path + " for reading";
      return false;
    }
    palette->count = 0;
    palette->associate_alpha = false;
    char line[512];
    int lineno = 0;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
      ++lineno;
      if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
      int index = 0;
      double r = 0.0, g = 0.0, b = 0.0;
      char hex[32] = {0};
      // The hex column is read so the format is matched exactly, then ignored:
      // it is 8-bit and preferring it would quietly quantise a Q16 palette.
      const int got = std::sscanf(line, "%d %lf %lf %lf %31s", &index, &r, &g, &b,
                                  hex);
      if (got < 4) {
        std::fclose(f);
        *error = path + ": line " + std::to_string(lineno) +
                 " is not a palette row (want: index r g b [hex])";
        return false;
      }
      if (index < 0 || index >= kMaxColormapSize) {
        std::fclose(f);
        *error = path + ": line " + std::to_string(lineno) + " index " +
                 std::to_string(index) + " is out of range";
        return false;
      }
      // Assign by the file's own index rather than by arrival order, so a
      // reordered file still imports in the order it declares.
      palette->entries[index].r = r;
      palette->entries[index].g = g;
      palette->entries[index].b = b;
      palette->entries[index].a = kOpaqueAlpha;
      if (index + 1 > palette->count) palette->count = index + 1;
    }
    std::fclose(f);
    if (palette->count <= 0) {
      *error = path + " contains no palette rows";
      return false;
    }
    if (expect_colors > 0 && palette->count != expect_colors) {
      *error = path + " holds " + std::to_string(palette->count) +
               " colours but --colors is " + std::to_string(expect_colors) +
               "; the two disagree, so the palette was not used.  Pass the file's "
               "count as --colors, or pass --colors 0 to accept the file as it is.";
      return false;
    }
    // Snap every entry to the precision the engine can actually represent,
    // BEFORE the tree is built.  This is not cosmetic: the tree is built from an
    // RgbaF strip, and RgbaF is float, so a Q16 value like 12347.8 becomes
    // 12347.7998.  The palette would then hold the double while the tree held the
    // float, and every later consistency check failed on it --
    //
    //   error: colormap entry 0 differs
    //     (IM 12347.8000 / tree 12347.7998)
    //
    // A DERIVED palette never hits this, because ImageMagick's colormap and the
    // tree come from the same pixels and so already agree.  It is specific to
    // import, where the file's precision and the engine's precision are two
    // different things, and it is why the round trip has to reconcile them rather
    // than merely copy them.
    for (int i = 0; i < palette->count; ++i) {
      palette->entries[i].r = static_cast<double>(static_cast<float>(palette->entries[i].r));
      palette->entries[i].g = static_cast<double>(static_cast<float>(palette->entries[i].g));
      palette->entries[i].b = static_cast<double>(static_cast<float>(palette->entries[i].b));
    }
    // Build the tree from a synthetic 1xN strip of the palette, which is the same
    // trick the PNG8 writer uses and the same one HarvestColormap uses for a
    // colormap: ColorTree::Build reduces a pixel array, and the palette IS a
    // pixel array laid out as a 1xN row.  Doing it this way rather than adding a
    // palette-only Build() keeps one code path for "these are the colours",
    // which is the property that matters -- two ways to build a tree from a
    // palette is two ways for them to disagree.
    // Canonical order, immediately before the tree is fed.  Both the text and the
    // image branch need it, and it is the last point at which the file's own
    // ordering can still be discarded without consequences.
    SortPaletteCanonical(palette);
    std::vector<RgbaF> strip(static_cast<std::size_t>(palette->count));
    bool grayscale = true;
    for (int i = 0; i < palette->count; ++i) {
      const PaletteEntry& e = palette->entries[i];
      strip[static_cast<std::size_t>(i)].r = static_cast<float>(e.r);
      strip[static_cast<std::size_t>(i)].g = static_cast<float>(e.g);
      strip[static_cast<std::size_t>(i)].b = static_cast<float>(e.b);
      strip[static_cast<std::size_t>(i)].a = static_cast<float>(e.a);
      if (e.r != e.g || e.g != e.b) grayscale = false;
    }
    // colors == the palette's own count: the reduction must not merge or drop
    // entries the user just told us to keep.
    tree->Build(strip.data(), static_cast<std::size_t>(palette->count), 1,
                palette->count, /*associate_alpha=*/false, grayscale);
    if (tree->color_count() != palette->count) {
      *error = path + ": the tree reduced " + std::to_string(palette->count) +
               " colours to " + std::to_string(tree->color_count()) +
               "; the palette holds duplicate or unreachable entries";
      return false;
    }
    // Adopt the TREE's colormap, discarding the file's ordering.
    AdoptTreeColormap(palette, tree);
    return true;
  }

  // Image form: the same path --palette-from uses, so a PNG8 strip from
  // ImWritePalette and a `magick -unique-colors` file both import.
  LoadedImage loaded;
  if (!ImLoad(path, &loaded, error)) {
    *error = path + ": " + *error;
    return false;
  }
  const bool ok = ImAdoptPaletteFromColormap(loaded, expect_colors, palette, tree,
                                             error);
  ImFree(&loaded);
  if (!ok) {
    *error = path + ": " + *error;
    return false;
  }
  SortPaletteCanonical(palette);
  // Same reconciliation as the text branch.  ImAdoptPaletteFromColormap fills
  // the palette from the image's colormap and builds the tree from the same
  // pixels, so its ORDER is the image's, not the tree's -- and an 8-bit PNG strip
  // written by ImWritePalette preserves whatever order it was given.  Leaving
  // the two unreconciled is what made --palette-import fail with
  // "colormap entry 11 differs" while the identical file imported cleanly as
  // .txt.
  AdoptTreeColormap(palette, tree);
  return true;
}

}  // namespace rd
