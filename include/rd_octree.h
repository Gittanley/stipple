// SPDX-License-Identifier: GPL-3.0-or-later
// rd_octree.h -- faithful port of ImageMagick 7.1.2-31's quantize.c colour
// octree, to the depth the Riemersma dither actually depends on.
//
// Why this file exists
// --------------------
// quantize.c:RiemersmaDither() does not search the whole colormap:
//
//     node_info = p->root;
//     for (index=MaxTreeDepth-1; (ssize_t) index > 0; index--)
//     {
//       id = ColorToQNodeId(cube_info,&pixel,index);
//       if (node_info->child[id] == NULL) break;
//       node_info = node_info->child[id];
//     }
//     ClosestColor(image,p,node_info->parent);      // <-- a SUBTREE
//
// So the candidate set is the set of leaves carrying data *below a particular
// node*, not all N palette entries.  A plain nearest-colour scan over the
// palette therefore disagrees with ImageMagick on colour images (it happened to
// agree on greys, where the cube collapses).  Reproducing the tree is the only
// way to be bit-exact, so ClassifyImageColors / Reduce / DefineImageColormap are
// transliterated here.
//
// Verification: DefineImageColormap() in this port must produce exactly the
// colormap that ImageMagick's own QuantizeImage() produces for the same input.
// The CLI checks that on every run, which is a strong structural test of the
// whole tree.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rd_types.h"

namespace rd {

inline constexpr int kMaxQNodes = 266817;   // quantize.c: MaxQNodes
inline constexpr int kRgbChildren = 8;      // children when alpha is not used
inline constexpr int kRgbaChildren = 16;    // children when alpha is used

struct QNode {
  std::int32_t child[kRgbaChildren] = {};  // -1 == null
  std::int32_t parent = -1;
  double total_color[4] = {0.0, 0.0, 0.0, 0.0};
  double quantize_error = 0.0;
  std::uint64_t number_unique = 0;
  std::uint32_t color_number = 0;
  std::uint32_t id = 0;
  std::uint32_t level = 0;
};

class ColorTree {
 public:
  // Builds the tree, reduces it to at most `max_colors`, and fills `colormap`
  // exactly as quantize.c:DefineImageColormap() would.  `pixels` must be
  // row-major RGBA float in ImageMagick's Quantum range.
  //
  // `depth` follows quantize.c:Log4(max_colors)+2, decremented because a dither
  // method is selected, and forced to MaxTreeDepth for greyscale input.
  void Build(const RgbaF* pixels, std::size_t width, std::size_t height,
             int max_colors, bool associate_alpha, bool grayscale);

  int color_count() const { return color_count_; }
  const PaletteEntry* colormap() const { return colormap_.data(); }
  const std::vector<QNode>& nodes() const { return nodes_; }

  // quantize.c:ColorToQNodeId().
  std::size_t NodeId(const RgbaD& pixel, int index) const;

  // quantize.c:ClosestColor() restricted to the subtree rooted at
  // `node_info->parent`, with the traversal and the early exits reproduced.
  // `target` is cube_info->target, `distance`/`color_number` are the running
  // incumbent.  Returns the winning palette index.
  int ClosestColor(std::int32_t node, const RgbaD& target, double* distance,
                   int* color_number) const;

  bool associate_alpha() const { return associate_alpha_; }
  std::size_t node_count() const { return nodes_.size(); }

  // Accessors used by the dither's palette lookup.
  std::int32_t root() const { return root_; }
  std::int32_t Child(std::int32_t node, std::size_t id) const {
    return nodes_[node].child[id];
  }
  std::int32_t Parent(std::int32_t node) const { return nodes_[node].parent; }
  int depth() const { return depth_; }

 private:
  std::int32_t NewNode(std::uint32_t id, std::uint32_t level, std::int32_t parent);
  void PruneChild(std::int32_t node);
  void PruneLevel(std::int32_t node);
  void PruneToCubeDepth(std::int32_t node);
  void Reduce(std::int32_t node);
  std::size_t QuantizeErrorFlatten(std::int32_t node, std::size_t offset,
                                   std::vector<double>* out) const;
  void DefineImageColormap(std::int32_t node);
  int Children() const { return associate_alpha_ ? kRgbaChildren : kRgbChildren; }

  std::vector<QNode> nodes_;
  std::vector<PaletteEntry> colormap_;
  std::int32_t root_ = -1;
  int color_count_ = 0;
  int maximum_colors_ = 0;
  int depth_ = 3;
  // quantize.c: cube_info->nodes -- allocated nodes minus pruned ones.
  std::size_t node_live_count_ = 0;
  double pruning_threshold_ = 0.0;
  double next_threshold_ = 0.0;
  bool associate_alpha_ = false;
};

}  // namespace rd
