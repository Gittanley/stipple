// SPDX-License-Identifier: GPL-3.0-or-later
// rd_octree.cpp -- transliteration of the parts of ImageMagick 7.1.2-31
// MagickCore/quantize.c that the Riemersma dither depends on.
#include "rd_octree.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace rd {
namespace {

// quantize.c: MagickSafeReciprocal().
inline double SafeReciprocal(double x) {
  const double sign = (x < 0.0) ? -1.0 : 1.0;
  return ((sign * x) >= kMagickEpsilon) ? 1.0 / x
                                       : sign * (1.0 / kMagickEpsilon);
}

// quantize.c: QuantizeErrorCompare().  Note it returns 0 for near-equal values,
// which makes qsort an unstable but deterministic ordering; the same comparator
// must be used or the threshold differs.
int QuantizeErrorCompare(const void* a, const void* b) {
  const double* p = static_cast<const double*>(a);
  const double* q = static_cast<const double*>(b);
  if (*p > *q) return 1;
  if (std::fabs(*q - *p) <= kMagickEpsilon) return 0;
  return -1;
}

// quantize.c: IsPixelEquivalent() reduces to exact channel equality for a
// DirectClass image with no black/index channel of its own.
inline bool PixelsEqual(const RgbaF& a, const RgbaF& b) {
  return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
}

}  // namespace

std::int32_t ColorTree::NewNode(std::uint32_t id, std::uint32_t level,
                                std::int32_t parent) {
  QNode node;
  node.id = id;
  node.level = level;
  node.parent = parent;
  for (int i = 0; i < kRgbaChildren; ++i) node.child[i] = -1;
  nodes_.push_back(node);
  return static_cast<std::int32_t>(nodes_.size() - 1);
}

std::size_t ColorTree::NodeId(const RgbaD& pixel, int index) const {
  std::size_t id =
      static_cast<std::size_t>((scale_quantum_to_char(clamp_pixel(pixel.r)) >> index) & 0x01);
  id |= static_cast<std::size_t>((scale_quantum_to_char(clamp_pixel(pixel.g)) >> index) & 0x01) << 1;
  id |= static_cast<std::size_t>((scale_quantum_to_char(clamp_pixel(pixel.b)) >> index) & 0x01) << 2;
  if (associate_alpha_) {
    id |= static_cast<std::size_t>((scale_quantum_to_char(clamp_pixel(pixel.a)) >> index) & 0x01) << 3;
  }
  return id;
}

void ColorTree::PruneChild(std::int32_t node) {
  const int children = Children();
  for (int i = 0; i < children; ++i) {
    const std::int32_t child = nodes_[node].child[i];
    if (child >= 0) PruneChild(child);
  }
  // The gate is quantize.c's `cube_info->nodes`, which counts allocated nodes
  // minus the ones already pruned -- *not* the size of the node array, because
  // ImageMagick never returns pruned nodes to a free list.  Using the array
  // size here over-prunes (a 2-colour reduction collapsed to a single colour).
  if (node_live_count_ > static_cast<std::size_t>(maximum_colors_)) {
    const std::int32_t parent = nodes_[node].parent;
    if (parent >= 0) {
      QNode& p = nodes_[parent];
      QNode& n = nodes_[node];
      p.number_unique += n.number_unique;
      for (int c = 0; c < 4; ++c) p.total_color[c] += n.total_color[c];
      p.child[n.id] = -1;
      // IM keeps the node allocated (nodes-- is a counter, not a free list), so
      // the node stays in the array; only the counter shrinks.  Reproduced by
      // decrementing below rather than erasing.
    }
    --node_live_count_;
  }
}

void ColorTree::PruneLevel(std::int32_t node) {
  const int children = Children();
  for (int i = 0; i < children; ++i) {
    const std::int32_t child = nodes_[node].child[i];
    if (child >= 0) PruneLevel(child);
  }
  if (static_cast<int>(nodes_[node].level) == depth_) PruneChild(node);
}

void ColorTree::PruneToCubeDepth(std::int32_t node) {
  const int children = Children();
  for (int i = 0; i < children; ++i) {
    const std::int32_t child = nodes_[node].child[i];
    if (child >= 0) PruneToCubeDepth(child);
  }
  if (static_cast<int>(nodes_[node].level) > depth_) PruneChild(node);
}

void ColorTree::Reduce(std::int32_t node) {
  const int children = Children();
  for (int i = 0; i < children; ++i) {
    const std::int32_t child = nodes_[node].child[i];
    if (child >= 0) Reduce(child);
  }
  if (nodes_[node].quantize_error <= pruning_threshold_) {
    PruneChild(node);
  } else {
    if (nodes_[node].number_unique > 0) ++color_count_;
    if (nodes_[node].quantize_error < next_threshold_) {
      next_threshold_ = nodes_[node].quantize_error;
    }
  }
}

std::size_t ColorTree::QuantizeErrorFlatten(std::int32_t node, std::size_t offset,
                                            std::vector<double>* out) const {
  if (offset >= node_live_count_) return 0;
  (*out)[offset] = nodes_[node].quantize_error;
  std::size_t n = 1;
  const int children = Children();
  for (int i = 0; i < children; ++i) {
    const std::int32_t child = nodes_[node].child[i];
    if (child >= 0) n += QuantizeErrorFlatten(child, offset + n, out);
  }
  return n;
}

void ColorTree::DefineImageColormap(std::int32_t node) {
  const int children = Children();
  for (int i = 0; i < children; ++i) {
    const std::int32_t child = nodes_[node].child[i];
    if (child >= 0) DefineImageColormap(child);
  }
  QNode& n = nodes_[node];
  if (n.number_unique == 0) return;
  if (color_count_ >= static_cast<int>(colormap_.size())) {
    colormap_.resize(colormap_.size() * 2 + 1);
  }
  PaletteEntry& q = colormap_[color_count_];
  double alpha = SafeReciprocal(static_cast<double>(n.number_unique));
  if (!associate_alpha_) {
    q.r = clamp_to_quantum(alpha * kQuantumRange * n.total_color[0]);
    q.g = clamp_to_quantum(alpha * kQuantumRange * n.total_color[1]);
    q.b = clamp_to_quantum(alpha * kQuantumRange * n.total_color[2]);
    q.a = kOpaqueAlpha;
  } else {
    const double opacity = alpha * kQuantumRange * n.total_color[3];
    q.a = clamp_to_quantum(opacity);
    if (q.a == kOpaqueAlpha) {
      q.r = clamp_to_quantum(alpha * kQuantumRange * n.total_color[0]);
      q.g = clamp_to_quantum(alpha * kQuantumRange * n.total_color[1]);
      q.b = clamp_to_quantum(alpha * kQuantumRange * n.total_color[2]);
    } else {
      const double gamma = SafeReciprocal(kQuantumScale * q.a);
      q.r = clamp_to_quantum(alpha * gamma * kQuantumRange * n.total_color[0]);
      q.g = clamp_to_quantum(alpha * gamma * kQuantumRange * n.total_color[1]);
      q.b = clamp_to_quantum(alpha * gamma * kQuantumRange * n.total_color[2]);
    }
  }
  n.color_number = static_cast<std::uint32_t>(color_count_);
  ++color_count_;
}

int ColorTree::ClosestColor(std::int32_t node, const RgbaD& target,
                            double* distance, int* color_number) const {
  const int children = Children();
  // Post-order: children 0..n-1, then the node itself.
  for (int i = 0; i < children; ++i) {
    const std::int32_t child = nodes_[node].child[i];
    if (child >= 0) ClosestColor(child, target, distance, color_number);
  }
  const QNode& n = nodes_[node];
  if (n.number_unique == 0) return *color_number;

  const PaletteEntry& p = colormap_[n.color_number];
  double alpha = 1.0;
  double beta = 1.0;
  if (associate_alpha_) {
    alpha = kQuantumScale * p.a;
    beta = kQuantumScale * target.a;
  }
  double pixel = alpha * p.r - beta * target.r;
  double d = pixel * pixel;
  if (d <= *distance) {
    pixel = alpha * p.g - beta * target.g;
    d += pixel * pixel;
    if (d <= *distance) {
      pixel = alpha * p.b - beta * target.b;
      d += pixel * pixel;
      if (d <= *distance) {
        if (associate_alpha_) {
          pixel = p.a - target.a;
          d += pixel * pixel;
        }
        if (d <= *distance) {
          *distance = d;
          *color_number = static_cast<int>(n.color_number);
        }
      }
    }
  }
  return *color_number;
}

void ColorTree::Build(const RgbaF* pixels, std::size_t width, std::size_t height,
                      int max_colors, bool associate_alpha, bool grayscale) {
  nodes_.clear();
  colormap_.assign(kMaxColormapSize, PaletteEntry{});
  associate_alpha_ = associate_alpha;
  // quantize.c:QuantizeImage() clamps the count it is handed before anything reads it:
  //
  //   maximum_colors=quantize_info->number_colors;
  //   if (maximum_colors == 0) maximum_colors=MaxColormapSize;
  //   if (maximum_colors > MaxColormapSize) maximum_colors=MaxColormapSize;
  //
  // This is not defensive tidiness.  maximum_colors_ is compared against
  // node_live_count_ through a static_cast<std::size_t>, and a value of 0 there makes
  // EVERY node merge into its parent -- PruneChild's gate `nodes > maximum_colors` is
  // `nodes > 0`, true for every node including the root's children -- so the tree
  // collapses to a single colour.  A NEGATIVE value becomes SIZE_MAX, so the gate is
  // false for every node and nothing is ever pruned; the reduce loop's
  // `while (color_count_ > maximum_colors_)` is then true for any count >= 0 and the
  // loop cannot terminate.  One place, before anything reads it, removes both.
  //
  // Unreachable today, and deliberately not treated as a live defect: rd_cli.cpp:831
  // rejects --colors outside 2..kMaxColormapSize, and rd_video.cpp:1349 takes
  // std::max(colors, palette_stage1_colors), so every call site passes at least 2.  It
  // is here because the hazard is a HANG and an INFINITE LOOP rather than a wrong
  // number, and because this is precisely the arithmetic upstream performs.
  maximum_colors_ = (max_colors <= 0) ? kMaxColormapSize : max_colors;
  if (maximum_colors_ > kMaxColormapSize) maximum_colors_ = kMaxColormapSize;
  node_live_count_ = 0;  // set to 1 below, once the root exists
  color_count_ = 0;
  pruning_threshold_ = 0.0;
  next_threshold_ = 0.0;

  // quantize.c:SetImageColormap(): depth = Log4(colormap size) + 2, minus one
  // when a dither method is selected, and MaxTreeDepth for greyscale input.
  // maximum_colors_, NOT max_colors: upstream's loop reads the clamped value, so a
  // caller passing 0 must give the same tree here as it does there.
  int depth = 1;
  for (std::size_t colors = static_cast<std::size_t>(maximum_colors_); colors != 0;
       ++depth) {
    colors >>= 2;
  }
  if (depth > 2) --depth;  // a dither method is always in play here
  // quantize.c:3306 -- and one level again when the image carries alpha:
  //   if ((image->alpha_trait != UndefinedPixelTrait) && (depth > 5)) depth--;
  // This was missing, so --colors >= 1024 on alpha input built a tree one level deeper
  // than the reference's.  On the image path the per-run colormap check at rd_cli.cpp
  // caught it (exit 4, "colormap entry N differs"); on the video path there is no such
  // check, so the palette came from IM's depth-5 tree while the colour_numbers and the
  // subtree partition came from a depth-6 tree, and the dither output was quietly
  // non-conformant.  verify.ps1 sweeps --colors (2,4,16,64,256); the largest of those
  // gives depth 6, which is exactly 5 after the decrement above -- so the alpha rule
  // could not fire in any test, even though an alpha fixture exists.
  if (associate_alpha_ && depth > 5) --depth;
  if (grayscale) depth = kMaxTreeDepth;
  if (depth > kMaxTreeDepth) depth = kMaxTreeDepth;
  if (depth < 2) depth = 2;
  depth_ = depth;

  root_ = NewNode(0, 0, -1);
  // GetQNodeInfo() increments cube_info->nodes for the root as well, so the
  // counter starts at 1.  Every threshold test in quantize.c is sensitive to
  // this off-by-one.
  node_live_count_ = 1;
  // quantize.c: GetQCubeInfo() does `cube_info->root->parent=cube_info->root`.
  // RiemersmaDither() relies on it: when the descent stops at the root it
  // searches root->parent, which must be the root rather than a null index.
  nodes_[root_].parent = root_;

  // ---- ClassifyImageColors() --------------------------------------------
  // ImageMagick splits this into two loops with *different* descent limits:
  //
  //   loop 1  level <= MaxTreeDepth, colours counted at level == MaxTreeDepth,
  //           and as soon as colours > maximum the tree is pruned to the cube
  //           depth and the loop breaks;
  //   loop 2  level <= cube_info->depth (which PruneLevel may already have
  //           reduced), colours counted at level == cube_info->depth, and no
  //           colour check at all.
  //
  // Reproducing the split matters: on a busy 800x600 image loop 1 overflows
  // MaxQNodes, PruneLevel lowers the depth, and loop 2 then builds a shallower
  // tree than loop 1 did.  Collapsing the two loops changes the colormap.
  const double midpoint = kQuantumRange / 2.0;
  const double error_alpha = 0.0;  // quantize.c sets this once, outside the loop
  std::size_t y = 0;

  // Processes one scanline.  `level_limit` and `colors_level` differ per loop.
  const auto process_row = [&](const RgbaF* row, int level_limit,
                               int colors_level) {
    std::size_t x = 0;
    while (x < width) {
      // Run-length encode the row: ImageMagick collapses runs of identical
      // adjacent pixels and weights them by `count`.
      std::size_t count = 1;
      while (x + count < width && PixelsEqual(row[x], row[x + count])) ++count;

      RgbaD pixel;
      associate_alpha_pixel(associate_alpha_, row[x], &pixel);

      int index = kMaxTreeDepth - 1;
      double bisect = (kQuantumRange + 1.0) / 2.0;
      double mid[4] = {midpoint, midpoint, midpoint, midpoint};
      std::int32_t node = root_;
      for (int level = 1; level <= level_limit; ++level) {
        bisect *= 0.5;
        const std::size_t id = NodeId(pixel, index);
        mid[0] += (id & 1) ? bisect : -bisect;
        mid[1] += (id & 2) ? bisect : -bisect;
        mid[2] += (id & 4) ? bisect : -bisect;
        mid[3] += (id & 8) ? bisect : -bisect;
        if (nodes_[node].child[id] < 0) {
          const std::int32_t created = NewNode(static_cast<std::uint32_t>(id),
                                               static_cast<std::uint32_t>(level), node);
          nodes_[node].child[id] = created;
          ++node_live_count_;
          if (level == colors_level) ++color_count_;
        }
        node = nodes_[node].child[id];
        const double e0 = kQuantumScale * (pixel.r - mid[0]);
        const double e1 = kQuantumScale * (pixel.g - mid[1]);
        const double e2 = kQuantumScale * (pixel.b - mid[2]);
        const double e3 = associate_alpha_ ? (kQuantumScale * (pixel.a - mid[3]))
                                           : error_alpha;
        double distance = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3;
        if (std::isnan(distance) != 0) distance = 0.0;
        nodes_[node].quantize_error +=
            static_cast<double>(count) * std::sqrt(distance);
        // ImageMagick adds the *child's* running total to the root here, which
        // double counts across siblings.  It is IM's behaviour, so it is kept.
        nodes_[root_].quantize_error += nodes_[node].quantize_error;
        --index;
      }
      QNode& leaf = nodes_[node];
      leaf.number_unique += count;
      leaf.total_color[0] += count * kQuantumScale * clamp_pixel(pixel.r);
      leaf.total_color[1] += count * kQuantumScale * clamp_pixel(pixel.g);
      leaf.total_color[2] += count * kQuantumScale * clamp_pixel(pixel.b);
      leaf.total_color[3] +=
          count * kQuantumScale *
          clamp_pixel(associate_alpha_ ? pixel.a : kOpaqueAlpha);
      x += count;
    }
  };

  // ---- loop 1 ------------------------------------------------------------
  for (; y < height; ++y) {
    if (node_live_count_ > static_cast<std::size_t>(kMaxQNodes)) {
      PruneLevel(root_);
      --depth_;
    }
    process_row(pixels + y * width, kMaxTreeDepth, kMaxTreeDepth);
    if (color_count_ > maximum_colors_) {
      PruneToCubeDepth(root_);
      break;
    }
  }
  ++y;

  // ---- loop 2 ------------------------------------------------------------
  for (; y < height; ++y) {
    if (node_live_count_ > static_cast<std::size_t>(kMaxQNodes)) {
      PruneLevel(root_);
      --depth_;
    }
    process_row(pixels + y * width, depth_, depth_);
  }

  // ---- ReduceImageColors() ----------------------------------------------
  if (color_count_ > maximum_colors_) {
    next_threshold_ = 0.0;
    if (node_live_count_ > 0) {
      std::vector<double> errors(node_live_count_, 0.0);
      (void)QuantizeErrorFlatten(root_, 0, &errors);
      const std::size_t n = node_live_count_;
      std::qsort(errors.data(), n, sizeof(double), QuantizeErrorCompare);
      const std::size_t aggressive = (110u * (static_cast<std::size_t>(maximum_colors_) + 1u)) / 100u;
      if (n > aggressive) next_threshold_ = errors[n - aggressive];
    }
  }
  while (color_count_ > maximum_colors_) {
    pruning_threshold_ = next_threshold_;
    next_threshold_ = nodes_[root_].quantize_error - 1.0;
    color_count_ = 0;
    Reduce(root_);
  }

  // ---- DefineImageColormap() --------------------------------------------
  color_count_ = 0;
  DefineImageColormap(root_);
}

}  // namespace rd
