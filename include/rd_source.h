// SPDX-License-Identifier: GPL-3.0-or-later
// rd_source.h -- RAM-first pixel store with a memory-mapped disk fallback.
//
// The dither walk touches every pixel exactly once but in Hilbert order, so the
// access pattern is cache-hostile yet perfectly sequential in the order of use.
// That makes a flat, uniformly-addressable buffer the right abstraction: the
// same float* works whether the pixels live in the heap or in a page file.
//
// Budget policy (--max-ram-mb):
//   * footprint <= budget  -> anonymous heap allocation, zero file I/O
//   * footprint >  budget  -> a temp file mapped into the process address
//                             space, so the OS page cache + the disk provide
//                             the overflow capacity transparently
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "rd_types.h"

namespace rd {

// Defined in the CUDA translation unit; returns nullptr in a --no-cuda build.
// Declared here rather than included so rd_source.h stays free of cuda_runtime.h.
void* CudaAllocPinned(std::size_t bytes);
void CudaFreePinned(void* p);

// Interleaved RGBA float, row-major.  This is exactly one Quantum per channel.
using PixelBuffer = RgbaF*;

class PixelStore {
 public:
  // Creates a w*h RGBA-float store.  `max_ram_bytes` caps heap residency; when
  // the image is larger the store is backed by a temp file instead.  Returns
  // nullptr and fills `error` on failure.
  static PixelStore* Create(std::size_t width, std::size_t height,
                            std::size_t max_ram_bytes, std::string* error);

  ~PixelStore();

  PixelStore(const PixelStore&) = delete;
  PixelStore& operator=(const PixelStore&) = delete;

  std::size_t width() const { return width_; }
  std::size_t height() const { return height_; }
  std::size_t pixel_count() const { return width_ * height_; }
  std::size_t byte_size() const { return width_ * height_ * sizeof(RgbaF); }

  // True when the pixels are heap resident; false when disk backed.
  bool resident() const { return data_ != nullptr && !mapped_; }

  // True when the allocation is page-locked, so a CUDA engine can DMA straight
  // to and from it with no staging copy.  False in a --no-cuda build and on the
  // disk path -- file-backed pages must not be pinned, the OS may evict them.
  bool pinned() const { return pinned_; }

  // Human readable description of where the pixels live.
  const char* backing() const {
    return mapped_ ? "disk (memory-mapped spill file)" : "RAM";
  }
  const std::string& spill_path() const { return spill_path_; }

  // Direct base pointer.  Valid for the lifetime of the store.  The dither uses
  // this for its hot loop; nothing here is bounds-checked by design.
  RgbaF* data() { return data_; }
  const RgbaF* data() const { return data_; }

  RgbaF* row(std::size_t y) { return data_ + y * width_; }
  const RgbaF* row(std::size_t y) const { return data_ + y * width_; }

  // Prefetches the whole image into the page cache / heap.  Best effort: used
  // for the disk path so the sequential walk is not interleaved with first-touch
  // faults.  Returns false if the hint could not be applied.
  bool Prefetch();

 private:
  PixelStore() = default;

  std::size_t width_ = 0;
  std::size_t height_ = 0;
  RgbaF* data_ = nullptr;

  bool mapped_ = false;
  bool pinned_ = false;
  void* mapping_ = nullptr;   // HANDLE-ish base for UnmapViewOfFile
  std::size_t mapping_size_ = 0;
  std::string spill_path_;
};

}  // namespace rd
