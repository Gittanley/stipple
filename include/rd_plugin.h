// rd_plugin.h -- the seam for adding your own dither or filter.
//
// WHY THIS EXISTS
//
// rdither was built to reproduce one thing exactly: what
// `magick in.png -dither Riemersma -colors 16 out.png` does.  That is a good thing
// to be, and it is also a strange thing to *only* be, because everything around the
// dither -- the palette builder, the curve generator, the block partitioner, the
// decode/dither/encode pipeline, the crash-resume checkpoints -- is independent of
// which error-diffusion kernel you feed it.  None of that work should be locked
// behind one algorithm.
//
// So this file defines a small interface and a registry.  An algorithm registers
// itself, rdither looks it up by name from `--dither`, and everything else is
// unchanged.
//
// WHAT AN ALGORITHM HAS TO SUPPLY
//
// The interface is deliberately narrow, and it is narrow in a specific way: a dither
// receives a *fully built* palette and a precomputed traversal curve, and returns
// quantised pixels.  It does not get to choose its own palette, decode, or encode.
// That is not a limitation, it is the feature -- your kernel composes with the
// parts of this program that took twenty rounds to get right, instead of
// reimplementing them.
//
// Two entry points per algorithm, and the reason is bit-exactness.  `frames` is
// how many independent frames are handed over at once; the video pipeline gives a
// batch of 16 and each frame must keep its own error queue, because a frame dithered
// correctly must not depend on what was dithered before it.  An implementation that
// does not care (most do not) can ignore the parameter and treat the batch as one
// tall image.
//
// WHAT IS AND IS NOT PLUGGABLE HERE
//
// Static registration, not dynamic loading, and deliberately so.  A runtime-loaded
// plugin has to agree with the host on struct layout, calling convention, allocator,
// and C++ runtime -- and this program is built with MSVC + CUDA against a specific
// ImageMagick, so a `LoadLibrary` plugin from a different toolchain is a
// memory-corruption class of problem, not a feature.  Compile-time registration means
// your code is in the same binary, compiled by the same compiler, with the same
// headers.  Add a file to src/, add one line to CMakeLists.txt, rebuild.
//
// The examples directory has a complete, working second algorithm (Bayer ordered
// dithering, ~40 lines) that exists to prove the seam works end to end rather than to
// be useful.  Copy it and write your own.

#ifndef RD_PLUGIN_H_
#define RD_PLUGIN_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// The pixel and palette types are the pipeline's own, not copies.  A duplicate
// RgbaF here would be a second definition of the same struct in the same namespace,
// and a plugin that included both headers would fail to compile -- which is a poor
// first impression for the thing meant to make writing code easier.  Reusing them
// also means the plugin is handed memory the pipeline already owns, with no
// conversion on the way in or out.
#include "rd_riemersma.h"

namespace rd {

// The palette a plugin must quantise to.  `count` entries, in
// `entries[i].r / .g / .b`.
//
// ALWAYS BIND BY REFERENCE: `const Palette& pal = *job.palette;`
//
// This is the single most likely way to get a plugin wrong, and it fails in a way
// that looks nothing like the mistake.  Palette is `PaletteEntry entries[65536]` and
// PaletteEntry is four doubles, so **Palette is 2 MiB**.  Writing
// `const Palette pal = *job.palette;` puts 2 MiB on the stack, the default 1 MiB
// thread stack overflows immediately, and the process dies with
// STATUS_STACK_OVERFLOW (0xC00000FD) partway into your first line -- before touching
// a pixel, and with no message that mentions memory.  It was worth three debug
// cycles to find, so it is written down here.
//
// Alpha is not something a dither chooses between: `associate_alpha` in the palette
// says whether alpha was folded into the colour decision, and a plugin that ignores
// it will disagree with Riemersma on images that carry alpha.
using PluginPalette = Palette;

// Everything an algorithm is told about the job.
struct DitherJob {
  const Palette* palette = nullptr;
  // The Riemersma traversal order, if one was built.  May be null.
  //
  // An algorithm is free to ignore this and walk the image in whatever order it
  // likes -- Bayer, for instance, is a fixed matrix and has no use for a Hilbert
  // ordering.  It is offered because the hard part of an ordered-plus-diffusion
  // hybrid is the traversal, and that is already done.
  const CurveStepEntry* curve = nullptr;
  std::size_t curve_count = 0;
  std::size_t width = 0;
  std::size_t height = 0;
  int frames = 1;  // consecutive frames, each with an independent error queue
  // ImageMagick's -define dither:diffusion-amount, default 1.0.  Honour it: users
  // expect the same knob to mean the same thing across algorithms.
  double diffusion = 1.0;
  void* user = nullptr;  // reserved; always null in this version
};

// What an algorithm reports back.  `error` empty means success.
struct DitherResult {
  std::string error;
};

// --- the interface ---------------------------------------------------------

// Quantise `frames` consecutive frames of width*height RGBA-float pixels in place.
//
// This is the whole contract.  Implement it, and your algorithm works for single
// images and for video, on the CPU or the GPU, with palettes from ImageMagick or
// from rdither's own builder, with crash resume, for free.
//
// Rules that are not negotiable, because breaking them produces output that looks
// almost right:
//   * Pixel `i` of frame `f` is at `pixels[(f*width*height) + i]`.
//   * Do not read outside the frame you were given.  Frames are contiguous, so
//     index arithmetic that runs off the end of frame f reads frame f+1 and corrupts
//     it silently -- there is no bounds check and no crash.
//   * If the algorithm is stochastic, it must be deterministic.  Same input, same
//     output, every run: the video path relies on frame N being reproducible so
//     that --resume produces the same file.
struct DitherAlgorithm {
  const char* name;         // what --dither takes
  const char* description;  // one line, shown by --list-dithers
  // Returns an empty `error` on success.
  DitherResult (*run)(DitherJob& job, RgbaF* pixels);
};

// --- the registry ----------------------------------------------------------

// Add an algorithm.  Call from a static initialiser, or from main() before
// rdither runs.  Returns false (and registers nothing) if the name is empty or
// already taken, so a plugin cannot silently shadow a built-in.
bool RegisterDither(const DitherAlgorithm& algo);

// Look up by name.  Returns null when nothing is registered under that name.
const DitherAlgorithm* FindDither(const char* name);

// All registered algorithms, in registration order.
const std::vector<DitherAlgorithm>& RegisteredDithers();

// One-line descriptions of everything registered, for --help and --list-dithers.
std::string DitherListText();

// --- writing your own -----------------------------------------------------
//
// Three steps:
//
// 1. Copy examples/bayer_dither.cc to src/ and rename it.  It is a complete
//    working algorithm in about 40 lines.
//
// 2. Write the kernel.  The pattern is:
//
//      DitherResult MyDither(DitherJob& job, RgbaF* px) {
//        const Palette& pal = *job.palette;
//        const std::size_t n = job.width * job.height;
//        for (int f = 0; f < job.frames; ++f) {
//          RgbaF* frame = px + static_cast<std::size_t>(f) * n;
//          // ... your work on `frame` ...
//        }
//        return {};
//      }
//
//    The per-frame loop is not decoration.  Sharing one error buffer across the
//    batch would make frame f depend on frame f-1, which breaks both the parallel
//    block partition and crash resume.
//
// 3. Register it, in the same file:
//
//      static const bool kRegistered = [] {
//        return RegisterDither({"mine", "what it does", MyDither});
//      }();
//
//    Then add the file to RD_SOURCES in CMakeLists.txt and rebuild.
//
// A GPU algorithm does not go through this interface -- the GPU path is a set of
// CUDA kernels launched by the engine, not a function pointer, because the useful
// ones (the Riemersma walk) are stateful across a thread block with the error queue
// in registers.  Wrapping those in a portable interface would mean copying state to
// and from device memory per call and losing the reason they are fast.  If you want
// a GPU algorithm, add a kernel to src/rd_blocks_cuda.cu and dispatch it from
// rd_blocks_cuda.cu's gather/walk switch; the pattern to copy is
// RiemersmaBlocksCuda.  The host-side scaffolding -- palette, curve, batch
// management, pipeline, resume -- is reused either way.

}  // namespace rd

#endif  // RD_PLUGIN_H_
