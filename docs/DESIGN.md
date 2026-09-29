<!--
  DESIGN.md -- the long-form engineering record for rdither.

  This file is the whole story: how the dither is derived from ImageMagick's
  source, every performance round including the ones that failed, the bugs found
  along the way, and the measurements behind each decision.  It is long because it
  is a record, and it is kept because most of its value is in the negative results
  -- the things that were measured and did not work, so nobody repeats them.

  For how to build and use the program, see README.md at the repository root.
-->

# rdither -- design and engineering record
# rdither - ImageMagick 7.1.2-31 Q16-HDRI Riemersma dithering on CUDA


Phase 1: single-image, 16-colour reduction that is **bit-exact** with
`magick in.png -dither Riemersma -colors 16 out.png`, on a CPU reference engine
and a CUDA engine, with RAM-first / disk-fallback pixel storage.

Phase 2: video. `ffmpeg` decodes and encodes, `rdither` dithers, and the three
stages run **concurrently** — 4.1x on the pipeline as first shipped, 5.1x once
the queue was sized in slots rather than in frames (284.5 -> 69.5 s for 605
1080p frames, the same command line).

With the stages fully overlapped the run is ~99% dither-bound, and the biggest
single remaining line item is no longer decode or the dither: at
`--palette-frames all --palette-tile 512` the one-off palette build is 21.3 s of
a 69.5 s clip. `--palette-frames 30` removes most of it
for a palette that is visually indistinguishable.

```
bit-exact cases: 135 passed, 0 failed        (9 images x 5 colour counts x 3 engines)
```

Every case is confirmed twice: by `rdither --verify` (which runs ImageMagick's
own pipeline internally and diffs pixel-for-pixel) and independently by
`magick compare -metric AE`, which reports `0` in all 135 cases.

```
> rdither --colors 16 --verify in.png out.png
input      : in.png
geometry   : 800x600
alpha      : no
pixels     : RAM (7.3 MiB)
palette    : 16 colours from IM (associate_alpha=no)
curve      : level 10, 1048577 visits
tree       : depth 3, 121 nodes, 16 colours
colormap   : tree matches ImageMagick exactly
dither     : 65.11 ms (1 walk, 0.12 Mpixel/s aggregate, cpu engine)
verify     : AE=0/480000 pixels, max channel delta=0, RMSE=0.00000000 -> BIT-EXACT
```

---

## 1. Toolchain: why this is MSVC, not g++

The brief specified g++ (MinGW-w64). That cannot work here, and both halves of the
problem were verified on this machine rather than assumed:

| Check | Result |
|---|---|
| `nvcc -ccbin C:\msys64\ucrt64\bin\g++.exe a.cu` | `nvcc fatal : Host compiler targets unsupported OS.` |
| `nvcc a.cu` (default MSVC host) | compiles and links |
| MinGW `g++` linking `CORE_RL_Magick++_.lib` | `undefined reference to Magick::ColorGray::ColorGray(double)` … (Itanium mangling vs MSVC ABI) |

So: **nvcc on Windows only accepts MSVC as its host compiler**, and the
ImageMagick import libraries are MSVC-ABI, so MinGW cannot link them either.
`CMakeLists.txt` therefore uses MSVC + nvcc, and `build.cmd` drives it.

What *is* portable: `rd_types.h` (the Q16-HDRI numeric contract) and
`rd_riemersma.h`/`rd_octree.h` (the algorithm) are ImageMagick-free C++20 and
compile with `g++ -std=c++20`. Only the `rd_im.cpp` bridge and the CLI need MSVC.

## 2. What actually makes this bit-exact

Three things had to be right, and each one produced a *plausible but wrong*
result when missed. They are the reason to prefer a transliteration over a
reimplementation.

**(a) `Quantum` is `float`, not `double`.** `magick-baseconfig.h` reports
`MAGICKCORE_QUANTUM_DEPTH 16`, `MAGICKCORE_HDRI_ENABLE 1`, and
`MAGICKCORE_SIZEOF_FLOAT_T 4`, so `Quantum == float` while `MagickRealType ==
double`. `ClampPixel()` therefore *narrows through float*, and `RiemersmaDither`
stores the clamped result back into a `double` error queue:

```c
pixel.red=(double) ClampPixel(pixel.red);   /* double -> float -> double */
```

The diffusion state is double, but every value that survives a clamp has been
rounded to float. Reproducing that round trip is mandatory.

**(b) The palette lookup searches a subtree, not the palette.** This was the
expensive mistake. `ClosestColor()` is reached as
`ClosestColor(image,p,node_info->parent)`, so the candidate set is the leaves
carrying data *under one node* — not all N palette entries. A nearest-colour scan
over the palette agrees on greyscale (where the cube collapses) and disagrees on
colour: measured `AE=454/6144, RMSE=0.034` on a plasma image. The octree had to
be ported, which is what `src/rd_octree.cpp` is.

**(c) The lookup is memoised behind a 6-bit-per-channel key.** `CacheOffset()`
keeps only the top 6 bits per channel, so two distinct diffused targets can share
a key and *the first one to touch that key decides the answer for both*. Omitting
the table is usually invisible and occasionally off by one palette index.

Two further details in `ClassifyImageColors()` that change the colormap:

- The two row loops descend to **different depths** — loop 1 to `MaxTreeDepth`,
  loop 2 to the current (possibly already reduced) `cube_info->depth` — and only
  loop 1 checks `colors > maximum_colors`.
- `cube_info->nodes` counts the **root** and decrements on prune without ever
  returning nodes to a free list. Using the node-array size instead shifted
  every threshold and collapsed a 2-colour reduction to 1 colour.

Every run also re-derives the colormap from the ported tree and asserts it
equals ImageMagick's, which is a strong structural check on the whole port:

```
colormap   : tree matches ImageMagick exactly
```

## 3. Parallelism: the honest picture

Riemersma diffusion is a strict sequential chain — one 16-entry error queue, one
`(x,y)` cursor, each visit consuming the previous residual. **No intra-image
parallelisation can be bit-exact.** The only exact scaling axes are (i) building
the colour tree and (ii) running independent walks, i.e. video frames, each with
its own error queue and memo table.

Measured on this machine (GTX 1650 SUPER, sm_75), 256×256:

| Engine | Wall time | Aggregate throughput |
|---|---|---|
| CPU, 1 walk | 17.9 ms | 3.67 Mpixel/s |
| CUDA, 1 walk | 872 ms | 0.08 Mpixel/s |
| CUDA, 4 walks | 766 ms | 0.34 Mpixel/s |
| CUDA, 16 walks | 791 ms | 1.33 Mpixel/s |
| CUDA, 32 walks | 956 ms | 2.19 Mpixel/s |

So the GPU is ~48× *slower* than the CPU for a single image — one thread with no
parallelism, paying device-memory latency for the error queue, memo table and
tree. That is the expected result, not a defect, and the default engine is `cpu`.
What batching buys is real: 32 frames complete in roughly the time one takes,
~26× aggregate throughput, bit-exact per frame. That is the Phase 2 story.

On the device, all error-queue and distance arithmetic goes through
`__dadd_rn`/`__dmul_rn` so nvcc cannot contract `a*b+c` into an FMA and perturb
the last bit; the curve `level` is computed on the host for the same reason; and
the per-thread error queue and recursion stack live in global scratch, because
sm_75 caps local memory at 512 B/thread and keeping them local overflows the
stack.

## 4. Layout

```
include/rd_types.h        Q16-HDRI numeric contract: clamp, ScaleQuantumToChar,
                          CacheOffset, error weights, alpha association
include/rd_octree.h/.cpp  transliteration of quantize.c: ClassifyImageColors,
                          PruneChild/Level/ToCubeDepth, Reduce(+ImageColors),
                          DefineImageColormap, ClosestColor
include/rd_riemersma.h    Hilbert curve tables, DitherImage ordering, engine API
src/rd_riemersma_cpu.cpp  bit-exact CPU walk (the correctness oracle)
src/rd_riemersma_cuda.cu  CUDA walk, one thread per independent walk
src/rd_source.h/.cpp      RAM-first pixel store, memory-mapped disk spill
src/rd_im.h/.cpp          MagickCore bridge: decode, palette, encode, diff
src/rd_cli.cpp            command line
```

The Hilbert traversal is IM's `Riemersma()` recursion turned into an explicit
stack (`CurveStep` tables, `kLeaf`/`kNode`), so the same order is produced
sequentially on the host and on the device. `--dump-curve` prints the first
visits and the total.

## 5. Build

```bat
build.cmd

REM only if ImageMagick is somewhere unusual
set IMAGEMAGICK_ROOT=C:\Program Files\ImageMagick-7.1.2-Q16-HDRI
build.cmd
```

or by hand:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
      -DRD_IMAGEMAGICK_ROOT="C:/Program Files/ImageMagick-7.1.2-Q16-HDRI"
cmake --build build --config Release
```

`MAGICK_HOME` must point at the ImageMagick root at run time so coder modules
load. To build CPU-only, configure with `-DRD_WITH_CUDA=OFF`.

`CMAKE_CUDA_ARCHITECTURES` defaults to `75;90` (sm_75 is this GPU); override with
`-DCMAKE_CUDA_ARCHITECTURES=...`.

## 6. CLI

```
rdither [options] <input> <output>

  --colors N        palette size handed to ImageMagick (default 16)
  --engine E        cpu | cuda                     (default cpu)
  --diffusion X     IM's dither:diffusion-amount artifact (default 1.0)
  --max-ram-mb N    RAM budget before spilling pixels to disk (default 512)
  --max-vram-mb N   VRAM budget for the CUDA engine (0 = all free)
  --frames N        concurrent walks on the GPU (default 1)
  --format FMT      force the output coder, e.g. PNG, TIFF, MIFF
  --verify          diff against ImageMagick's own Riemersma output
  --dump-curve      print the curve level, visit count and first visits
  --dump-palette    print the harvested palette
  --quiet           suppress progress output
```

Exit codes: `0` ok, `1` error, `2` bad usage, `3` verification mismatch,
`4` the ported tree disagrees with ImageMagick's colormap.

### Examples

```bat
REM 16 colours, prove bit-exactness
rdither --colors 16 --verify photo.png out16.png

REM 256 colours
rdither --colors 256 photo.png out256.png

REM CUDA engine, 30 frames in flight (Phase 2 shape), 2 GiB VRAM cap
rdither --engine cuda --frames 30 --max-vram-mb 2048 frame.png out.png

REM keep the pixel buffer on disk: 64 MiB RAM budget
rdither --max-ram-mb 64 --colors 16 huge.tif out.png

REM match ImageMagick's diffusion strength
rdither --diffusion 0.8 --colors 16 in.png out.png

REM force the encoder regardless of extension
rdither --format TIFF in.png out.bin
```

Cross-check against the CLI:

```bat
magick photo.png -dither Riemersma -colors 16 ref.png
rdither --colors 16 --verify photo.png ours.png
magick compare -metric AE ours.png ref.png null:      REM -> 0
```

Full sweep (builds fixtures, runs 135 cases through both engines and the disk
spill path, checks each with `compare`):

```powershell
powershell -ExecutionPolicy Bypass -File verify.ps1
```

## 7. RAM-first, disk fallback

`PixelStore` holds the image as interleaved RGBA float — one `Quantum` per
channel, so the buffer *is* IM's pixel layout.

- footprint ≤ `--max-ram-mb` → `VirtualAlloc`, no file I/O;
- otherwise → a self-deleting temp file mapped into the address space, so the OS
  page cache and the disk provide the overflow transparently and the dither's
  hot loop is byte-identical in both cases.

`Prefetch()` touches one byte per 4 KiB page on the disk path so the sequential
Hilbert walk is not interleaved with first-touch faults. The CUDA engine requires
the frame in VRAM; `--max-vram-mb` checks the budget up front and `CudaMaxFrames()`
reports how many frames fit.

## 8. Assumptions

1. **Input is sRGB-compatible.** `ImLoad` applies the same
   `IssRGBCompatibleColorspace` → sRGB normalisation that
   `ClassifyImageColors()` does, so CMYK and friends are handled; the function
   itself is reproduced because it lives in an uninstalled private header.
2. **Grayscale detection is `r == g == b` for every pixel.** That selects
   `depth = MaxTreeDepth`, matching IM's `IdentifyImageGray` +
   `SetGrayscaleImage` decision. IM additionally runs `SetGrayscaleImage()`, a
   colorspace transform that this port does not reproduce; for genuinely gray
   input (R=G=B) the transform is near-identity and all greyscale fixtures verify
   at AE=0, but a borderline image could differ. Phase 2 should port
   `SetGrayscaleImage()` or call IM for it.
3. **`QuantizeErrorFlatten` reads uninitialised memory.** `ReduceImageColors`
   sizes its `quantize_error` array with `AcquireQuantumMemory` (not zeroed) and
   then indexes `nodes - 110*(max+1)/100`, which can land in the region
   `QuantizeErrorFlatten` never filled. This port zero-fills, which matches
   ImageMagick in practice (fresh pages) and matched all 135 cases, but it is
   technically reading uninitialised memory on IM's side and is not a guarantee.
4. **The palette comes from ImageMagick.** `ImBuildPalette` calls IM's own
   `QuantizeImage` with `dither_method = RiemersmaDitherMethod` and harvests
   `clone->colormap`. The dither method matters: `quantize.c` decrements the tree
   depth by one when a dither method is selected, so a palette built with
   `NoDitherMethod` would come from a different tree depth and would not match.
5. **Bit-exact means pixel colours.** The output is a truecolour image carrying
   the same RGBA values IM would produce. IM's own PNG is paletted, so the files
   are not byte-identical even though every pixel is; `compare -metric AE` is `0`.
6. **`measure_error` is off** (IM's default), so pixels are written from the
   colormap rather than keeping their error-diffused value.
7. **`associate_alpha`** is enabled when the source has an alpha channel, which
   switches `number_children` from 8 to 16 and premultiplies in
   `AssociateAlphaPixel`. Verified on a 60%-alpha plasma fixture.
8. **ffmpeg's command line is a moving target, and a removed option fails silently.**
   The two options this project depends on most — hardware decode and passthrough frame
   timing — have both been renamed or moved within ffmpeg's history, and the failure mode
   is always the same: ffmpeg exits during option splitting, writes zero bytes to the pipe,
   and the symptom appears somewhere else entirely. The frame-timing option alone has been
   responsible for three separate bugs here (see the crash-resume section, and round
   sixteen). `FpsMode()` now probes `ffmpeg -h long` once per run rather than assuming a
   spelling, and **any** option added to these commands should be treated the same way:
   verify it against the ffmpeg actually on `PATH`, not against the one this was written
   against.
9. **Every pixel-writing loop zeroes its row first (`ZeroQueuedRow`).** The loops assign
   four values per pixel — R, G, B, usually A — and some assign three, but
   `GetPixelChannels()` is not always four, and the extra channels then hold whatever the
   heap block held before. This is not theoretical: IM's PNG coder scans the full channel
   set to choose between TrueColorAlpha, PaletteAlpha, Gray and Bilevel, so stale memory
   changed the *encoding* of an otherwise identical image. Measured at 3 runs in 16, with
   the file differing on **every** run while the pixels were sometimes stable. It looked
   like an accelerator bug for a long time — it was chased through OpenCL barriers, bounds
   checks, buffer poisoning and finally a disassembly theory, all of which were
   reasonable and all of which were in the wrong component. `tools\probe-determinism.ps1`
   now guards it: a cross-engine bit-exactness test is *structurally unable* to see a
   fault both engines share, so each engine is also compared against itself across
   repeated runs. Note that "byte-identical output files" is the wrong assertion to
   make while testing this: ImageMagick writes a `png:tIME` chunk, so two runs a
   second apart differ in bytes with identical pixels. The determinism probe therefore
   hashes decoded pixels, and hashes the file *after* `-strip` — raw bytes differing
   while stripped bytes match is the timestamp doing its job, and the test reports it
   rather than failing on it. `png:exclude-chunk=date,time` would give reproducible
   file hashes, and is deliberately not the default, because the bar set in item 5
   above is matching ImageMagick's pixels rather than out-formatting it.
10. **The video engine's uint16 output buffer is pre-filled, because one pixel per
    frame is never dithered.** `Riemersma()` visits `4^L - 1` cells and the trailing
    `ForgetGravity` visit lands on the first one again, so the grid's *last* cell is
    never visited — which is why a 1920x1080 frame at level 11 leaves exactly one
    pixel at its source value, and why ImageMagick itself does. The float path gets
    this for free, because its buffer is `COPY_HOST_PTR` from the caller's frame. A
    freshly allocated uint16 output buffer does not: the scatter writes only pixels
    some visit owns, so the unvisited one keeps whatever the driver left there, and
    that goes straight into the encoder. One pixel in 2,073,600 is invisible, which is
    why it survived; what it is not is harmless, since uninitialised memory means the
    output is not reproducible.

    The OpenCL engine copies the source into the output buffer before the scatter, so
    its unvisited pixel agrees with its own float path's. **The CUDA engine has the
    same exposure** — `d_u16_buf` is a bare `cudaMalloc` at
    `src/rd_blocks_cuda.cu`, and `BlkScatterKernel` writes only owned pixels — and is
    deliberately *not* fixed yet. That engine is verified bit-exact, and changing its
    output, even for one pixel and even toward correctness, should be a measured
    decision rather than a drive-by. The fix is one `cudaMemcpyAsync` from the input
    or one `cudaMemsetAsync`.

    Found by `RD_OCL_CHECK_U16`, which runs the OpenCL data path and the float path
    over one input and diffs them — worth having precisely because video's acceptance
    bar is "almost frame exact", which is far too loose to see a single pixel of
    uninitialised memory. See docs/OPENCL.md, including the three earlier versions of
    that check that each reported a kernel bug and were themselves the bug.
11. **The video encoder is fed the source's true average frame rate, not
    `r_frame_rate`.** `r_frame_rate` is the *maximum instantaneous* rate — a
    container hint, not a summary — and it is the right number only when the source is
    constant rate. On a variable-rate source every frame in the output got that
    duration, so the output was the wrong length and, because the audio is
    stream-copied at its own timing, the A/V was desync by the same factor.
    Measured: a 3.0333 s VFR source came out 1.632 s long, 1.86x short.

    What fixed it is not clever: measure the frame timings and use
    `frames / total_duration`. For constant-rate input the two numbers are identical,
    so the common case is unaffected, and the measurement is a demux pass over the
    container's packet index rather than a decode. rdither now reports a VFR source
    and says whether it kept the per-frame cadence or flattened to the average —
    because a flattened VFR render is frame-for-frame identical to a correct one and
    nothing in the output distinguishes them.

    Restoring the *per-frame* cadence is implemented (`--video-preserve-vfr`) and is
    **off by default**: the mechanism is proven in isolation, but no genuinely
    variable-timestamp fixture could be built on this machine to confirm it end to
    end. See docs/OPENCL.md for the three silent failure modes involved — timebase
    too coarse, unquoted commas in the filter expression, and the fact that a frame's
    `duration` field is meaningless once a filter has run — and for the fourth that
    is still open.
12. **`-shortest` is used only where it trims audio, never where it would trim video.**
    The flag stops output at the shorter stream, which is right when the audio runs
    long and wrong when the video does: in that direction it *deletes video frames*.
    That is not a rounding artefact, it is frames the user asked to have dithered,
    and it scales with the clip. Measured on a 30 s excerpt of a phone capture —
    895 video frames against 30.0128 s of audio — the output came out with 892.
    Three frames gone, every run; `--no-audio`, which drops `-shortest` from the
    command, gave 895.

    Which direction applies is now decided by comparing the measured video and audio
    durations rather than assumed, and rdither says so when it declines to use the
    flag. Trailing video with no audio is handled by every player and costs nothing;
    deleted frames are not recoverable.

    Found only by running the reporter's own file. The rate fix above was measured,
    correct, and insufficient, and was reported as done — the fixture it was verified
    on had a 0.57-second frame in it, while the real input is a mild three-valued
    jitter around 30 fps. A fixture differing from the real input in *kind* rather
    than degree hides a second bug behind the first one's fix.
13. **Video geometry is the DISPLAYED geometry, not the coded geometry.** ffmpeg's
    rawvideo output auto-rotates by default, so a stream with a 90 or 270 degree
    display matrix — every portrait phone video — arrives at the pipeline already
    turned, at width/height swapped. The decoder never passed `-noautorotate`, so
    rdither was sizing its buffers from the coded dimensions.

    The pixel *count* is identical either way, so no bounds check detects it; only the
    row width is wrong, and the frame is read back sliced differently. The output is
    therefore a correct image, sheared, with no rotation tag to compensate. Measured on
    a HONOR/HUAWEI capture tagged `rotation=-90`: a 1080x1920 portrait image written
    into a 1920x1080 buffer. Upright it scores 0.037 RMSE against the source, which is
    the dither and nothing else; sheared, 0.297.

    The correction is made once, where `VideoInfo` is consumed, and every downstream
    user — buffers, the raw pipe's `-s`, the encoder, all three dither engines — takes
    the displayed frame. Rotating in rather than tagging through is forced: ffmpeg 9
    exposes `-display_rotation` as an **input** option only and has removed
    `-metadata:s:v rotate=`, so there is no way to carry the matrix to a re-encoded
    output in any container. The dither therefore runs on upright pixels, which
    rotates the diffusion direction relative to ImageMagick's pipeline; that changes
    which pixels differ, not whether the result is a correct dither, and it is
    identical across engines.

    Found by the reporter, who diagnosed it from the file's own metadata while two
    plausible video bugs were being chased. Worth recording: `magick`'s own stats
    output was enough to see the 90-degree rotation, and nobody had read it.

## 9. Approximate parallel engine (`--engine approx`) — measured, and a negative result

`src/rd_approx_cuda.cu` implements a **fully parallel** Riemersma approximation
with no scan, no blocks and no carried state. IM's walk is a 16-tap FIR
error-feedback modulator, and the residual obeys a relation that does not involve
the quantiser:

```
x - q = (I - H) e     =>     e = (I - H)^-1 (x - q)
```

Substituting into `v = x + H e` and using `H (I-H)^-1 = (I-H)^-1 - I` collapses
the recursion to a fixed point whose every operation is a convolution along the
curve index:

```
t = q + F (x - q),   F = (I - H)^-1 truncated to --approx-taps,   q <- Q(t)
```

The sequential answer is a *fixed point* of this map, so it is an exact solver
whose only question is convergence. One thread per pixel, one D-tap convolution
per sweep, shared-memory tiled so global traffic is `(TILE+D)/TILE` per sample
instead of `D` per sample.

### What the measurements actually show

800x600 plasma, 16 colours, GTX 1650 SUPER:

| sweeps | AE vs IM | RMSE | AE vs cache-free oracle |
|---|---|---|---|
| 1 | 298101 (62.1%) | 0.1391 | — |
| 8 | 113895 (23.7%) | 0.0744 | 13240 (2.76%) |
| 32 | 96780 (20.2%) | 0.0694 | 11304 (2.36%) |
| 64 | 84438 (17.6%) | 0.0650 | 9797 (2.04%) |
| 128 | 72430 (15.1%) | 0.0602 | 8312 (1.73%) |

**Three findings, one of which refutes the design prediction.**

1. **The AE floor is ImageMagick's memo cache, not the parallelism.** Deleting
   the 6-bit key memo table from the *sequential* oracle already moves 28565 of
   480000 pixels (5.95%), RMSE 0.0357. No order-free iteration can reproduce a
   first-touch-wins cache, so ~6% is a hard floor on any parallel scheme.

2. **Convergence is ~0.19%/sweep, not 0.541^N.** The linearised iteration matrix
   is `I - F` with `|I - F| <= G/(1-G) = 0.541`, which predicted ~25 sweeps to
   sub-LSB. Measured RMSE falls only 0.831x per 112 sweeps. The linearised
   contraction governs the *continuous* residual; AE is decided by discrete
   palette indices, and a pixel only changes when the continuous iterate crosses
   a decision boundary. The iterate is long since inside every boundary, so the
   flip rate is governed by the shrinking boundary-straddling population, not by
   0.541^k. Extrapolating, AE 0 would need thousands of sweeps.

3. **The error is insensitive to both filter length and precision**, which is
   what identifies (2) as the binding constraint:

   | taps | AE fp64 | AE fp32 |
   |---|---|---|
   | 16 | 84990 | 80783 |
   | 32 | 95223 | 89742 |
   | 64 | 96774 | 91186 |
   | 128 | 96780 | 91185 |
   | 256 | 96780 | 91185 |

   Saturates at ~19% from `D = 64` onward, and `D = 16` is *better* than
   `D = 256`. The `rho^D` truncation schedule and the fp32/fp64 choice are both
   irrelevant next to quantiser stagnation. (Restoring the per-step
   `ClampPixel()` saturation that the inverse-filter form drops was also tried and
   changed nothing measurable.)

### Performance

| | time | note |
|---|---|---|
| sequential CPU oracle | 85.3 ms | 5.6 Mpixel/s |
| sequential CUDA oracle | 3184.6 ms | one thread, no parallelism |
| approx, 8 sweeps, fp32 | 302.5 ms | of which ~254 ms is one-time setup |
| approx, marginal per sweep | 6.1 ms (fp32) / 14.5 ms (fp64) | |
| approx, marginal per sweep, linear lookup | 6.1 ms | the octree lookup is only ~0.5 ms/sweep |

The ~250 ms one-time cost is the host-side construction of the 4^level+1 entry
Hilbert visit order, which depends only on the geometry and therefore amortises
across a video sequence. Even so, at 8 sweeps the engine is **~3.5x slower than
the sequential CPU oracle** for a single image.

### Verdict

The formula is sound and the implementation is bit-for-bit reproducible, but as a
*replacement* for the sequential walk on this workload it does not pay off:

- it is slower than the CPU oracle per image,
- it cannot be bit-exact by construction (the cache),
- and its error floor is set by quantiser stagnation, not by anything tunable.

The practical scaling lever remains **frame-level parallelism**, and that is
something the CPU already does well: frames are independent, so N frames scale
across N cores with no approximation at all, while the GPU walk has a single
thread of useful work. Keep `--engine cpu` (or `--engine cuda --frames N`) as the
production path; `--engine approx` is a research probe and an AE/RMSE baseline.

A useful side-finding: `DitherImage()` genuinely **revisits cells**. The trailing
`RiemersmaDither(ForgetGravity)` lands on a cell the recursion already visited
(1 duplicate for a square image, 2 for a non-square one), so IM re-dithers it. A
parallel scatter must therefore carry an ownership map — last-writer-wins is
nondeterministic — which the implementation does.

## 9b. `--engine blocks` — the engine to use when the *look* is the goal

The section above treats bit-exactness as the objective and concludes the
parallel formulas do not pay. If instead you want the Riemersma **appearance**,
the right tool is much cheaper, because one fact settles it: the error queue is
exactly 16 deep, so a block of `B >= 16` positions is **bit-identical to the
sequential walk from its 16th position onward**. Only the first 16 positions of
each block can be perturbed by a wrong incoming state, and the perturbation is
bounded by `(G/(1-G)) * Delta/2 ~ 0.27 Delta` — under a third of a
quantisation step, so most of those pixels do not even change palette index.

`src/rd_blocks_cuda.cu` therefore runs every block from a zeroed queue: one
thread per block, one pass, no iteration, no scan, no prefix. `B` trades seam
error against parallelism, and the device is nowhere near saturated, so `B` can
be chosen freely.

Measured on 800x600 plasma, 16 colours (AE against the *cache-free* oracle, i.e.
isolating the seam error from ImageMagick's memo table):

| B | blocks | AE vs cache-free | AE vs IM | kernel ms/frame |
|---|---|---|---|---|
| 16 | 30000 | 5042 (1.05%) | 10.7% | 20.7 |
| 32 | 15000 | 3593 (0.75%) | 8.8% | 24.1 |
| 128 | 3750 | 1262 (0.26%) | 7.7% | 39.1 |
| 512 | 938 | 284 (0.06%) | 6.8% | 114.2 |
| 4096 | 118 | 61 (0.013%) | 6.4% | 280.9 |

The error halves with `B` exactly as the 16-seam argument predicts, and the
~6.4% floor against ImageMagick is the memo table, which is irreducible and
irrelevant to the appearance.

**Visual equivalence**, on a smooth 256x256 image where banding is the failure
mode to watch for:

| | unique colours | flat-crop σ | PSNR vs source |
|---|---|---|---|
| source | 65536 | 0.0390 | — |
| `-dither None` (no dither) | 16 | 0.0454 | 25.93 |
| ImageMagick Riemersma | 16 | 0.0479 | 25.70 |
| blocks B=32 | 16 | 0.0478 | 25.81 |
| blocks B=128 | 16 | 0.0483 | 25.73 |
| blocks B=512 | 16 | 0.0482 | 25.71 |

The flat-region texture (the thing that reads as "dithered" rather than
"posterised") matches ImageMagick to within 1%, and PSNR is indistinguishable
from it — while the seam error is 0.06% of pixels. **Recommended: `--blocks 128`**
— 0.26% seam error, 39 ms/frame, 2x the CPU oracle, with the texture intact.
`--blocks 32` is 3.4x faster if you want the extra speed; `--blocks 512` if you
want the seams gone.

Practical notes:

- **Measure the steady state, not the first call.** The engine prints its own
  phase breakdown to stderr; for 1920x1080 it reads roughly
  `curve=61 ms  alloc+upload=177 ms  kernels+IO=101 ms`.  The `alloc+upload`
  block is dominated by first-touch CUDA context creation and ~100 MB of
  `cudaMalloc`, and is paid **once per process** — it is not per-frame work.  A
  single-shot CUDA run therefore looks *slower* than the CPU oracle even though
  the kernel is 3.3x faster.  Use `--frames N` to measure:
  - CPU oracle: ~300 ms/frame
  - blocks B=32, kernel only: ~97 ms/frame (**~3.3x**)
- The `curve` term is the visit-order build.  It was 267 ms and is now 61 ms
  because the order is produced by the closed-form Hilbert point evaluated over
  a thread pool instead of a 4.19M-step serial recursion.  It depends only on
  the geometry, so in a sequence it is paid once (and `--frames N` amortises it
  within a single call).
- The closed form is **verified, not assumed**: `--self-test` compares it against
  the recursion position by position for levels 1..8 and reports
  `0 of 4^level positions DIFFER` at every level.  Getting there exposed two real
  facts about ImageMagick's curve, both now documented in `rd_riemersma.h`:
  - the recursion emits `4^level - 1` leaf visits, not `4^level`; the leaves cover
    every grid cell **except the last one**, which is therefore never dithered at
    all (that is the "one unvisited pixel" — ImageMagick does the same), and
  - the trailing `ForgetGravity` visit lands on the **origin**, because the
    Hilbert path closes on itself, so the origin is genuinely dithered twice.
- Blocks have no single visit order, so there is no memo table; the lookup is
  performed exactly every time. That is the *more* accurate nearest-colour rule
  (ImageMagick quantises the search key to 6 bits per channel).
- **Known cosmetic artifact:** the block engine emits one colour outside the
  palette (17 unique instead of 16) because the never-visited last cell keeps its
  source value. 1 pixel in 65536 on a square image; the bit-exact engines
  reproduce ImageMagick exactly here (AE=0) and are unaffected.

## 10. Phase 2 — video (ffmpeg pipeline)

```
rdither --video --colors 16 --blocks 128 --palette-frames 30 --batch-frames 32 in.mp4 out.mp4
```

### Why ffmpeg owns decode and encode

ImageMagick reads video one frame at a time through its ffmpeg delegate, building
a full `Image` (pixel cache, property bags, colormap) per frame. For 1920x1080
that is the dominant cost and it is pure overhead. The pipeline is instead three
processes with rawvideo in between and nothing touching the filesystem:

```
ffmpeg -i in  -f rawvideo -pix_fmt rgba64le -   ->  pipe
rdither  (GPU block engine, one launch per batch)
                                                     ->  pipe
ffmpeg -f rawvideo -pix_fmt rgba64le -s WxH -r F -i -  out
```

`rgba64le` is 16 bits per channel little-endian, which maps one-to-one onto
ImageMagick's Q16 `Quantum` (0..65535) with no scaling. Note that ffmpeg and
ImageMagick may differ by one LSB in YUV→RGB conversion; that is a decoder
difference, not a dither difference, and it applies equally to a reference
pipeline built on ImageMagick's own video reader.

### Palette from N frames

`--palette-frames N` (0 = all) samples N evenly spaced frames with an ffmpeg
`select` filter, takes a `--palette-tile` lattice sample of each (default 128, never
upscaled), tiles them into one montage, and runs **ImageMagick's own**
`QuantizeImage` with `RiemersmaDitherMethod` over that. One palette for the whole
sequence, which is what stops temporal colour pumping. The tree the dither
searches is built from the same montage, so the candidate sets match the palette.
The sample is a lattice rather than a box average on purpose — see the desaturation
section below.

### Pipelined: decode, dither and encode all run at once

The pipeline is four stages on six threads plus three child processes:

```
ffmpeg (decode) -> reader -> [ N host workers + 1 GPU worker ] -> writer -> ffmpeg (encode)
                          \_______ batch queue, depth chosen at start _________/
```

Slots are allocated up front and recycled, so the steady-state allocation is zero
and nothing is written to disk. Because the dither is a pure function of a frame
and the palette, batches are independent, so any worker may take any batch and the
CPU/GPU split is a *scheduling* decision rather than a correctness one.

#### The host block walk is not bit-identical to the CUDA walk

An earlier revision of this document claimed the split was verified invisible
(AE = 0). **That claim was measured on black frames and was worthless.** The
video path had a bug that made every output frame black (below), and comparing two
black videos trivially yields AE = 0. Measured properly, on real content, same
palette, lossless codec:

| comparison | metric |
|---|---|
| video (CUDA) vs video (host block walk only) | **AE = 32747, 1.58% of pixels** |
| video (CUDA) vs single-image `--engine blocks` | RMSE 0.032 |
| video (host block walk) vs single-image `--engine blocks` | RMSE 0.108 |

So the host walk diverges from the verified CUDA walk on ~1.6% of pixels. That is
invisible — the dither pattern is high frequency and the errors are isolated
pixels — but it is not bit-exact, and since the CUDA walk is the one verified
against ImageMagick (135/135), the host walk is the incorrect one. `rdither` now
prints a warning whenever host workers are used, and the default is
`--cpu-threads 0`. **The bit-exact guarantee covers the sequential CPU oracle and
the CUDA engines, not the host block walk.** The cause is not yet diagnosed;
`RiemersmaBlocksCpu` is only reachable from `--video --cpu-threads N` and so is
not covered by `verify.ps1`, which is how it went unnoticed.

#### A slot is a batch, so the queue must be sized in slots

The first version sized the queue as `ceil(workers * 2 / batch_frames)`, on the
reasoning that workers demand a couple of batches each. That is wrong: **a slot
already is a batch**, so at 1080p with `batch=16` and eleven workers the formula
returned **2 slots** — nine workers blocked on the free-slot queue while the GPU,
which needs 1 s per batch, idled 14 s out of every 15. Measured, the run retired
32 frames per 15 s = 2.13 fps, and 605 / 2.13 = 284 s, which is the wall time to
the decimal. Raising `--mem-fraction` did not help, which is what proved it was
the formula and not the RAM cap.

The rule is now the other way round: size the queue from the RAM budget, then trim
the worker count to the slots that exist, and give **every worker its own slot**.
A worker with nothing to work on is worth nothing, so a slot is the unit of
currency. The floor is 3, because a 2-slot pipeline also couples the reader to the
writer, and holding `--mem-fraction` constant keeps the accounting honest:
`in_flight_bytes()` now reports the depth actually allocated, not the pre-trim
value (it used to print "queue 2" next to "5316 MiB in flight" in the same line).

### Measured, 1920x1080, 605 frames, 16 colours, B=32, batch 16

| configuration | wall | ms/frame | queue | cpu/gpu |
|---|---|---|---|---|
| before the fix (10 host + GPU, 2 slots) | 284.5 s | 470.3 | 2 | 557/48 |
| 10 host + GPU, queue fixed | 62.0 s | 102.5 | 7 | 304/301 |
| **GPU only — the default** | **55.5 s** | **91.7** | 3 | 0/605 |
| 10 host + GPU, deep queue (0.9 RAM) | 55.9 s | 92.5 | 11 | 397/208 |

**5.1× end to end**, of which the queue fix is 4.6×. The GPU's share went from 48
frames to 301 with the same worker configuration, which is the whole point: the
GPU was never the bottleneck, it was never fed.

Measured on the shipped command line itself, `--palette-frames all
--palette-tile 512` included: **69.5 s** for 605 frames, of which 21.3 s is the
one-off palette and 48.2 s (79.7 ms/frame) is the pipelined steady state. That
79.7 ms/frame matches the 78.75 ms/frame implied by the all-in per-launch
`(curve + alloc+upload + dither) / batch`, i.e. the stages are now ~99% overlapped
and what remains is the dither launch overhead described below.

The honest finding is that **the host pool does not pay for itself here.** Its
block walk costs ~937 ms/frame against the GPU's 63.5 — the per-pixel error
scatter is 16 random writes into a frame-sized buffer, which is memory-latency
bound, and one thread per frame cannot hide that the way a thousand CUDA threads
can. Ten of them do not out-run one GPU, and they consume the bandwidth the GPU
needs. On a 60-frame clip this looked catastrophic (115 vs 219 ms/frame); over 605
frames the gap is only ~4%, because the short clip was dominated by startup and
palette cost. Either way the default is now **GPU-first**: `--cpu-threads -1`
means GPU-only when a GPU is present, and every core but two when it is not.
`--cpu-threads N` still opts into the pool, and `--mem-fraction` becomes
necessary if you do, since each 1080p batch-16 slot costs 759 MiB.

#### What is left, and it is per-launch overhead

The `[blocks]` line reports `dither+IO` only. The all-in per-frame cost is
`(curve + alloc+upload + dither) / batch` = 78.75 ms, against the 63.5 the summary
implies. The 19% gap is per-launch overhead paid on all 38 batches:

- **the curve is rebuilt and re-uploaded every launch** (90 ms × 38 = 3.4 s) even
  though it depends only on the frame geometry. The host workers already cache it
  per thread; the device path does not.
- **uploads are pageable**, 154 ms per batch for 133 MB ≈ 865 MB/s, where pinned
  host memory should reach several GB/s. `cudaHostRegister` on each slot is a
  small change and should remove most of that 5.8 s.

Dither stage only, versus the sequential CPU oracle at 318 ms/frame:

| batch | dither ms | ms/frame | vs batch=1 |
|---|---|---|---|
| 1 | 10231 | 170.5 | 1.00× |
| 4 | 6416 | 106.9 | 1.59× |
| 16 | 5172 | 86.2 | 1.98× |
| 32 | 5121 | 85.3 | 2.00× |

Batching contributes 2.0×, saturating at batch 16. Worth keeping, but it is a
launch-overhead effect and it is not where the pipeline's win comes from.

### "Dull and grey" has two independent causes, and the encoder is the bigger one

Measured on the output, same palette and same dither, only the encoder varying:

| encoder | unique colours out | drift from true dither (RMSE / PSNR) | 2 s size |
|---|---|---|---|
| libx264 yuv420p (old default) | 36356 | 0.01873 / 34.5 dB | 2.0 MiB |
| **libx264 yuv444p (default now)** | 30691 | **0.00789 / 42.1 dB** | 2.6 MiB |
| ffv1 yuv444p (`--video-lossless`) | **65** | 0 (reference) | 55.5 MiB |

A per-pixel dither pattern is the worst case for chroma subsampling, because 4:2:0
averages colour over 2x2 blocks — it averages away the very alternation the dither
is made of. Only lossless keeps the palette exactly, at ~28x the bitrate; yuv444p
is 2.4x closer to the true dither for 30% more bytes, which is the right trade for
hours of footage. The default is now yuv444p, and any subsampled or lossy
combination prints a warning naming the consequence, because otherwise it is
invisible until someone says the colours look flat.

The *other* cause is in the palette itself, and on grey-heavy footage it is real:
the same montage yields 84% mean saturation on ordinary footage and 49% on a clip
that is 80% desaturated. That part is about the palette, and it is what the
sections below address.

### `--im-palette`: one command, ImageMagick's palette, our dither

This is the answer to the palette problem, and it closes a gap that a long series of
attempts did not. `--im-palette` reproduces the reference pipeline's palette step
*in process*, then GPU-dithers the video:

```
rdither --video --engine blocks --colors 16 --im-palette input.mp4 out.mp4
```

Equivalent to the two-step form, without the intermediate PNG or the 30 PPM files
on disk:

```
magick f1..f30 +append -colors 16 -unique-colors palette.png
rdither --video --palette-from palette.png input.mp4 out.mp4
```

Three details of ImageMagick's behaviour were responsible for the whole
discrepancy, and all three are non-obvious:

1. **`alpha_trait` must be `UndefinedPixelTrait`.** `quantize.c:3306`:
   ```c
   if ((image->alpha_trait != UndefinedPixelTrait) && (depth > 5)) depth--;
   ```
   A palette image carrying an alpha channel is quantized from a *different tree
   depth* than one without. The reference reads PPM, which has no alpha.
2. **8-bit precision, at Q16 scale.** This is a Q16-HDRI build, so `Quantum` spans
   0..65535. Feeding 16-bit montage data makes `QuantizeImage`'s colour-error test
   keep subdividing until the error is negligible, and it settles on flatter
   centroids. The reference extracts 8-bit PPM. (Writing 0..255 without scaling by
   257 is worse still — the image comes out 257x too dark and the octree returns
   near-pure primaries, which is exactly what a first attempt at this did.)
3. **The octree is spatial, so layout matters.** The same 30 frames in a 6x5 grid
   and in one `+append` strip are not a repacking of the same data; the tree
   subdivides differently and the colormap differs. `--im-palette` builds the strip.

Measured against a reference pipeline built exactly along these lines (30 sampled
8-bit PPM frames, `+append -colors 16 -unique-colors`):

| | colours | mean saturation | near-neutral |
|---|---|---|---|
| reference pipeline's palette | 15 | 52.6% | — |
| rdither default builder | 16 | 41.7% | 9 |
| **`--im-palette`** | **15** | **53.5%** | 6 |

(the 1-point difference is the metric reading 16-bit Q values against an 8-bit PNG;
the 15 swatches agree entry for entry, `1B1D1A`, `FD0000`, `F706F9`, `00FFFE` and
so on). The palette then survives the dither intact: under `--video-lossless` the
output carries 27 unique colours from a 15-colour palette, the surplus being the
RGB→YUV→RGB round trip plus the error-diffusion intermediate.

**When to use which.** `--im-palette` when you want ImageMagick's palette semantics
in one command. `--palette-from` when you already have a palette PNG, or when the
palette comes from elsewhere. The default builder when you want the path that is
verifiable bit-exact against `magick -dither Riemersma -colors 16` in the same
process — note that this is the *only* one of the three that is, because the other
two deliberately reproduce 8-bit no-alpha quantization instead.

### Let ImageMagick own the palette: `--palette-from`

The palette search below went a long way and did not fully close the gap to a
reference ImageMagick pipeline that produced visibly better palettes. Rather than
keep guessing, this is now a seam: ImageMagick generates the palette, rdither does
the dithering. The two jobs are cleanly separable, because the palette step is
cheap (30 frames) and the per-frame dithering is the expensive part — and it is
exactly where rdither's GPU engine is worth 5x over the serial path.

```
magick f1..f30 +append -colors 16 -unique-colors palette.png
rdither --video --engine blocks --colors 16 --palette-from palette.png in.mp4 out.mp4
```

`--palette-from` takes the image's colormap **verbatim** — no re-quantization,
because re-quantizing is precisely what changes the palette, and keeping the one
ImageMagick already produced is the entire point. Exact duplicates are dropped
(the same thing `-unique-colors` does), and the octree the dither descends is
built from a swatch image of those colours, so the tree and the palette come from
the same set. The run reports the adopted palette's saturation so it is still
measurable.

Measured against a reference pipeline built exactly along these lines (30 sampled
8-bit PPM frames, `+append -colors 16 -unique-colors`):

| | colours | mean saturation | near-neutral |
|---|---|---|---|
| reference pipeline's palette PNG | 15 | 52.6% | — |
| rdither's own palette builder | 16 | 41.7% | 9 |
| **`--palette-from` (adopting the reference palette)** | 15 | **53.5%** | 6 |

and the dither output holds 19 unique colours from a 15-colour palette under
`--video-lossless` (the extra 4 are the RGB→YUV→RGB round trip), so the palette
survives the dither intact.

**When to use which.** If the palette ImageMagick builds from your footage is good
enough, use `--palette-from` and skip the guesswork. If you want one command and no
intermediate file, use the built-in builder and read the reported saturation; the
built-in path is also the only one that can be verified bit-exact against
`magick -dither Riemersma -colors 16` in the same process.

### The palette sampler was silently broken (`-vsync 0`)

The single most damaging bug in this file, and it explains "dull and grey" on
footage that is mostly grey. The palette decoder ran:

```
ffmpeg -i in.mp4 -vf select=not(mod(n\,2)) -frames:v 30 -f rawvideo -pix_fmt rgba64le -
```

`select` leaves timestamp gaps, and ffmpeg's default frame-rate conversion re-fills
them, so `-frames:v 30` yields **the first 30 frames in time order, not the 30
selected ones**. Every evenly-spaced sample beyond the first contiguous run was
silently dropped. On an 80%-grey clip that means the sampler returned *only grey
frames* and the palette came out 100% neutral — `7015, 7732, 6907`, R≈G≈B.

Fixed by adding `-vsync 0` (passthrough) to both palette decoders. The bug was
found by dumping the montage (`RD_DUMP_MONTAGE=path`, a 16-bit PNG of exactly what
the quantizer sees), diffing it cell-by-cell against `magick`'s own montage of the
same frames, and finding cells 2..29 completely different. Cell 0 matched; cell 29
should have been a 98%-saturated frame and was 3%.

| | before | after |
|---|---|---|
| montage, 30 frames, 80%-grey clip | 4.0% mean sat, 14/16 near-neutral | **42.2%, 8/16** |
| cell 29 (a vivid frame) | 3% sat | 98.5% sat |
| montage vs `magick` montage, cell-by-cell | cells 2–29 mismatched | RMSE 0.006 (bit depth only) |
| rdither palette vs `magick f -colors 16 -unique-colors` on the same montage | — | 42.2% vs 41.2%, i.e. **identical** |

Two earlier results in this document were measured through that bug and are
**withdrawn**: the "cliff" in `--palette-frames` (it was the sampler degenerating to
"first N frames" at some N and to a correct `step=1` at others) and the rejection
of two-stage pooling (the pool path had the same two defects, so its numbers say
nothing either way). The pool mode also still carries the shell-quote defect that
made its `select` string unparseable; it needs re-measuring before any conclusion
is drawn from it.

### Palette: full resolution, one quantization, no pre-reduction

`--palette-tile 0` (the default) copies each sampled frame whole into the montage,
so the quantizer sees every pixel, exactly like `magick f1..f30 +append -colors 16`.
Measured on the 80%-grey clip, 30 frames, 16 colours:

| cell | montage MiB | palette ms | mean saturation | near-neutral |
|---|---|---|---|---|
| tile 512 | 256.0 | 4714 | 42.6% | 8 |
| tile 256 | 64.0 | 2036 | 42.7% | 8 |
| tile 128 | 16.0 | 1428 | 42.7% | 8 |
| **full resolution** | 949.2 | 30172 | 42.2% | 8 |

Once the sampler is correct, **cell size stops mattering** (42.2–42.7%) — which is
itself the confirmation that the earlier variation was a sampling artefact rather
than a property of the imagery. Full resolution costs 21x the memory and 20x the
time for nothing measurable, so the practical default is a modest tile; `-palette-tile
0` is available for anyone who wants the literal `+append` semantics.

`-unique-colors` is not cosmetic, by the way: on the same montage, `-colors 16`
alone measures 21.4% mean saturation and `-colors 16 -unique-colors` measures
41.2%, because the quantizer emits duplicate colormap entries. Reading
`image->colormap` after `QuantizeImage` already gives the deduplicated result,
which is why rdither matches the `-unique-colors` figure and not the other one.

The proposal was: sample frames, quantize *each frame* to the colour count
separately, save those palettes as images, combine them into one big image, and
quantize that again. Implemented as `--palette-mode pool`, with stage 1 reducing
each sampled frame at full resolution to `--palette-stage1-colors` (default 64),
those swatches deduplicated by `--palette-dedup` RGB distance, and stage 2 running
ImageMagick's own quantizer over the result.

### Two-stage palette: implemented, and its evaluation withdrawn

The proposal was: sample frames, quantize *each frame* to the colour count on its
own, save those palettes as images, combine them into one big image, and quantize
that again. Implemented as `--palette-mode pool`, with `--palette-stage1-colors`,
`--palette-dedup` and `--palette-max-samples`. It is off by default.

**These numbers are void.** The pool path carried both defects described above, so
the table below compared a broken sampler against a broken sampler, and the
montage row was itself measured through the missing `-vsync 0`:

| | grey clip | vivid clip |
|---|---|---|
| montage, pixel-weighted (default) | 48.8% | 83.7% |
| pool, no dedup | 4.3% | - |
| pool, dedup 800 / 2600 / 6000 / 12000 | 4.0% / 4.0% / 4.5% / 5.4% | - |
| pool, stage 1 = 16 / 32 / 64 / 128 | 4.3% / 2.8% / 3.9% / 3.9% | 63.6% at s1=64 |

Its evaluation is **withdrawn**. The pool path carried both defects described
above - the unparseable `select` string and the missing `-vsync 0` - so every
number measured against it described a broken sampler rather than the idea. One
structural observation survives and is independent of sampling: pre-reducing a
frame to N colours yields N cluster *centroids*, and a second pass averages
centroids again, which is a route to desaturation that does not need grey footage
to show up. That predicts the scheme is worse, but it is a prediction, not a
measurement, and the honest position is that the mode is unvalidated. It is off
by default. Fix the pool decoder and re-measure before trusting either claim.

The per-frame weighting argument that motivated it - "each frame contributes the
same number of samples" - is true and irrelevant to grey-heavy footage, since 80% of the
frames being grey still leaves 80% of the swatches grey. That part of the
reasoning was sound; it just cannot out-weigh the averaging.

### The real desaturation bug was in the montage, and it was mine

The montage box-filtered each frame into its tile, averaging ~14 source pixels per
output pixel. The comment in the code claimed this "keeps the palette
representative". That is backwards for this job: **averaging distinct hues pulls
each sample toward the local mean, i.e. toward grey**, so the filter was
desaturating the palette before the quantizer ever saw it. It is now a lattice
sample — one real, unmodified source pixel per tile position:

| | grey clip | vivid clip | cost |
|---|---|---|---|
| box filter, tile 256 (was) | 44.5% | 85.0% | 3686 ms |
| lattice, tile 128 (now) | **52.0%** | **84.1%** | **2355 ms** |

Cheaper *and* less grey, because `--palette-tile 128` also gives each frame 4x less
pixel mass, which is a mild form of the frame-weighting that pool mode overdid.
Tile 16/32/64/128/256 sweep spans 46.9–52.0% (grey) and 80.0–84.1% (vivid), so 128
is the default.

Note what is *not* fixed: a grey frame's pixels really are grey, so with 80% of
the footage grey the octree still gives grey most of the slots, and that is
arguably correct — 16 colours for footage that is mostly grey should be mostly
grey, or the grey parts will look terrible. The principled next step is
**distinct-look sampling**: use each frame's own quick 16-colour reduction as a
cheap *signature* to skip near-duplicate frames, then feed only the accepted
frames' raw lattice pixels. That gets frame-diversity weighting without a second
round of averaging. Not implemented.

Every run now reports mean saturation and the number of **near-neutral** entries
(saturation < 0.2). Mean saturation alone cannot see this failure: three greys and
thirteen vivid colours still average high, which is why an earlier revision of
this document drew the wrong conclusion from it. `--palette-only` reports the
palette and stops, so a palette experiment costs seconds instead of a full render.

### Options

```
--video                ffmpeg pipeline instead of a single image
--palette-frames N     frames sampled for the palette (0 or all = every frame)
--palette-tile N       lattice-sample tile edge per frame (default 128)
--palette-mode M       pool (opt-in, measured worse) or montage (default)
--palette-stage1-colors N  pool mode: colours per frame in stage 1 (default 64)
--palette-dedup N     pool mode: RGB distance for merging swatches (default 2600)
--palette-max-samples N  pool mode: ceiling on stage-1 frames (default 64)
--batch-frames N       frames per GPU launch (default 16, saturates ~16)
--cpu-threads N        -1 auto (GPU only if a GPU is present), 0 = GPU only,
                       >0 exact host worker count
--host-grace-ms N      how long a host worker waits for the GPU first (0 = 60)
--no-gpu               host workers only
--mem-fraction F       share of physical RAM the queue may use (default 1/3);
                       raise it if you opt into --cpu-threads, since each 1080p
                       batch-16 slot costs 759 MiB and the pool is trimmed to fit
--max-ram-mb N         hard RAM ceiling, as for single images
--video-codec C        default libx264
--video-pix-fmt F      default yuv444p -- yuv420p destroys the palette
--video-lossless       ffv1 yuv444p: palette survives exactly, ~28x bitrate
--palette-only         build and report the palette, then stop
--crf N / --preset P   encoder quality
```

Every run also reports the palette's **mean saturation**, because a palette that
has quietly gone flat is otherwise only visible by eye, and `--palette-only`
exists so that number can be measured in a second instead of costing a full
render.

Exit codes as before; `0` on success. `ffmpeg`/`ffprobe` are found on `PATH` or
overridden with `FFMPEG`/`FFPROBE`.

`--palette-frames all` is capped at a 64 Mpixel montage: a full-length clip would
otherwise try to build a ~19 GB image, so past the cap frames are subsampled
rather than the run being refused.

## 11. The dither, as mathematics

Everything below is transcribed from `quantize.c` (7.1.2-31) and is what rdither
reproduces bit for bit. Line numbers refer to that file.

### Constants

```
ErrorQueueLength    = 16                        (line 217)
ErrorRelativeWeight = MagickSafeReciprocal(16)  = 1/16   (line 218)
```

### The error weights

`GetQCubeInfo` builds the 16 weights once (lines 2108-2113):

```
weight := 1
for i in 0 .. 15:
    weights[i] := 1 / weight
    weight    := weight * exp( ln(1/ErrorRelativeWeight) / 15 )
               =         weight * exp( ln(16) / 15 )
```

Unrolling, with `r := 16^(1/15) ≈ 1.20300`:

```
weights[i] = r^(-i) = 16^(-i/15)
```

so the vector is a geometric decay, heaviest on the **oldest** surviving error:

| i | 0 | 1 | 2 | 8 | 14 | 15 |
|---|---|---|---|---|---|---|
| `weights[i] = 16^(-i/15)` | 1.000000 | 0.831238 | 0.690956 | 0.227931 | 0.075189 | 0.062500 |

`weights[15] = 16^(-1) = 1/16` exactly, because the loop's last multiplication is
discarded.

### Per-pixel update

At each visited pixel with source value `s` and cached colormap entry `c`
(lines 1723-1795):

**1. Accumulate the correction** (lines 1723-1734). For each channel `k`:

```
s_k  ←  s_k  +  ErrorRelativeWeight · diffusion · Σ_{i=0}^{15} weights[i] · e_{i,k}
     =         s_k  +  (1/16) · diffusion · Σ_{i=0}^{15} 16^(-i/15) · e_{i,k}
```

`diffusion` is the `dither:diffusion-amount` artifact, default `1.0`. It scales the
whole correction linearly. With `q := 16^(-1/15) ≈ 0.831238`, the filter's total
gain is

```
G = (1/16) · Σ_{i=0}^{15} q^i = (1/16) · (1 - q^16)/(1 - q)
  = (1/16) · 5.617668 = 0.351104
```

so about 35% of the queued error is fed back per pixel, spread across a 16-entry
geometric memory. Riemersma is not a local two-neighbour spreader: the **oldest**
queued error carries 16× the weight of the newest, which is what gives the
pattern its characteristic long tail and why it does not look like Floyd–Steinberg.

**2. Clamp** (lines 1735-1739): `s_k ← ClampPixel(s_k)`. In this Q16-HDRI build
`Quantum` is `float`, so the clamp narrows through single precision — which is why
the arithmetic order is load-bearing for bit-exactness.

**3. Map to the palette** (lines 1740-1767). `i = CacheOffset(cube_info, &s)` is a
memo key built from the *clamped* value. On a miss, descend the colour tree from
the root for `index = 7 .. 1`:

```
node ← root
for index in 7 down to 1:
    id ← ColorToQNodeId(cube_info, &s, index)
    if node.child[id] is NULL: break
    node ← node.child[id]
ClosestColor(image, p, node.parent)
```

The search is over the **subtree** rooted at `node.parent`, not the whole palette —
a subtlety that a plain nearest-colour scan does not reproduce.

**4. Emit** (lines 1772-1782): write `colormap[index]` into the pixel.

**5. Push** (lines 1788-1795):

```
e ← e[1 .. 15]            (memmove; the oldest entry is discarded)
e[15] ← s - c             (the new error, appended at the tail)
```

So `e_0` is the oldest surviving error and `e_15` the newest, and step 1 weights
them in exactly the opposite order.

### The block-parallel decomposition

The queue is **exactly 16 deep** and is wholly replaced by step 5 of every pixel.
Therefore after 16 steps the walk's entire future is independent of the state it
started from:

> A block of `B ≥ 16` consecutive curve positions produces output identical to the
> sequential walk from its 16th position onward.

Only a block's first 16 positions can be perturbed by a wrong boundary state, and
only if the incoming queue is non-zero. Starting every block from a zeroed queue
therefore confines the defect to 16 positions per block, damped by the
`weights[i] ≤ 1` and `G ≈ 5.62` bounds above. This is what makes `--engine blocks`
visually equivalent to the sequential walk while being embarrassingly parallel —
one thread per (frame, block) for the walk, one per (frame, position) for the
gather and scatter.

The consequence worth being explicit about: `--engine blocks` is **not** bit-exact
against ImageMagick. The 135/135 exactness claim covers the sequential CPU oracle
and the CUDA engines; the block engine trades that guarantee for speed, by design.

### Traversal

The curve is ImageMagick's own recursive visit order, compacted to an index array
(`HilbertPoint` / `BuildCurveIndex`) so the GPU can follow it by array lookup
rather than by recursion. Two facts about it that are easy to get wrong and are
verified by `--self-test`:

- The recursion emits `4^level - 1` leaves, so the last grid cell is **never
  dithered** and keeps its source value.
- The trailing `ForgetGravity` visit lands on the **origin** — the path closes on
  itself — which is why the visit count is `pixels + 1`.

- ~~Video palette generation.~~ **Done** -- see section 10: 30 sampled frames,
  box-filtered into a montage, quantised by ImageMagick's own `QuantizeImage`.
- ~~Frame batching.~~ **Done** -- the block engine processes distinct frames as
  (frame, block) work items, 2.0x over single-frame, and the ffmpeg pipeline
  feeds it real frames.
- ~~Decode/encode overlap.~~ **Done** -- reader thread, batch queue, N host workers
  plus a GPU worker, writer thread. 5.1x over the original serial pipeline on 605
  frames (284.5 -> 55.5 s); see the table in section 10. Five bugs worth
  recording, all found by hanging, stalling, or *looking at the output*:
  the slot array was sized from a *stale* depth and indexed out of bounds; EOF woke
  only `cv_ready` so the writer slept forever; an encoder failure left the reader
  blocked on a queue nobody would drain; the queue was sized in *frames* when a
  slot is a *batch*, starving the GPU of 9 workers' worth of throughput; and the
  reader read rgba64le into `raw` **without ever calling `RawToFloats`**, so the
  engine dithered the zero-initialised `pixels` vector and every frame came out
  black. There is now a 600 s stall watchdog, because an 18000-frame job must not
  hang silently.
- **The verification method was wrong, and that is the important lesson.** For
  several revisions the video path was checked by diffing *the video output against
  another video run*, which cannot detect a fault common to both. The black-frame
  bug survived a dozen such checks. Video output is now validated against ground
  truth: `--palette-frames 1` makes the video palette identical to the single-image
  palette, so a video frame can be compared to the verified single-image engine, and
  frame alignment is checked by confirming video frame *N* matches source frame *N*
  and not its neighbours.
- **The palette is still flatter than ideal on grey-heavy footage, and the
  remaining fix is distinct-look sampling.** A grey frame's pixels really are grey,
  so with 80% of the footage grey the octree gives grey most slots — arguably
  correct, but the user reports it as dull. The principled approach is to use each
  frame's own cheap 16-colour reduction as a *signature* to skip near-duplicate
  frames, then feed only accepted frames' raw lattice pixels: frame-diversity
  weighting without a second round of averaging. The two-stage scheme just
  implemented and rejected is a worse version of this idea.
### Per-launch overhead: persistent device state

Every launch used to rebuild everything. At batch 16 / 1080p that cost, per launch:
~60 ms of host work building the visit curve, **8 `cudaMalloc` + 8 `cudaFree`**
(one of them 531 MB), and five uploads. On a 605-frame clip that was 38 launches x
~260 ms = ~10 s of a 49 s dither stage, and essentially all of it waste: the curve,
the palette, the tree and the error weights are functions of the geometry and the
palette, both constant for a whole clip. Only the pixel buffer changes.

So they are built once and kept (`DeviceState` in `rd_blocks_cuda.cu`). The key
covers geometry, batch capacity, block size and an FNV hash of the palette entries
plus the tree's node and colour counts -- so a caller that reuses a palette object
with new contents rebuilds rather than silently reusing stale device state. The
per-launch line now reports `setup=0.0 ms (cached)`.

The key is on **capacity**, not the exact frame count, which matters: a 605-frame
clip at batch 16 ends on a 13-frame batch, and keying on the exact count rebuilt
every buffer -- 447 ms -- for that one launch.

Pixels now move through **pinned staging memory on a dedicated stream**
(`cudaHostAlloc` + `cudaMemcpyAsync`). Pageable H2D transfers are staged
through a small internal bounce buffer by the driver, which caps them; pinned
memory removes that ceiling and lets the copy overlap. If the pin fails (no
pageable RAM left, say) it falls back to a direct copy from the caller's memory --
bandwidth loss, never correctness. The two mid-pipeline `cudaDeviceSynchronize`
calls became one `cudaStreamSynchronize` at the end.

| | dither stage, 605 frames @1080p | ms/frame | fps |
|---|---|---|---|
| before | 61108 ms | 101.0 | 9.9 |
| **after** | **43480 ms** | **71.9** | **13.4** |

**1.38x**, with no change to the arithmetic -- 135/135 bit-exact before and after,
which is the check that matters for a change of this kind.

### The one that mattered: the error queue belongs in registers

The persistent-cache work above took the dither stage to 72 ms/frame. The next
measurement said something was still wrong: roughly 2.7 GB of device traffic per
launch at an effective **2.6 GB/s**, far below any modern GPU. So the kernels were
not bandwidth-bound -- they were *amplification*-bound.

The error queue is 16 entries x 4 channels = 64 doubles, and **every position
shifts all 64**. It lived in global memory, one 512-byte slot per work item, so
each thread touched `32 x 512 B x 2 = 32 KB` per block of B=32 to process **512 B**
of pixel data -- 64x more traffic than useful work, re-read and re-written at every
step.

As a local array of constant size with fully unrolled constant indices it stays in
registers. The arithmetic is untouched (same `__dadd_rn` / `__dmul_rn`, same
order), so the output is bit-identical -- 135/135 before and after, which is the
check that matters for a change like this.

| | dither+IO per 16-frame batch | ms/frame |
|---|---|---|
| queue in global memory | 1030 ms | 64.4 |
| **queue in registers** | **314 ms** | **19.6** |

**3.3x**, and it also frees 531 MB of VRAM at batch 16, because the per-work-item
state array no longer exists.

### Two defaults that turned out to matter more than any tuning knob

**`--preset veryfast` instead of `medium`.** Once the dither stopped being the
bottleneck, the encoder's preset became the lever -- and not through its own busy
time, which barely moves (21.8 s at medium, 22.3 s at veryfast), but through how
fast it drains the pipe and so how long the writer thread blocks:

| preset | wall | fps | size |
|---|---|---|---|
| medium | 30.7 s | 19.7 | 29.3 MiB |
| fast | 27.0 s | 22.4 | 29.2 MiB |
| **veryfast** | **23.9 s** | **25.3** | **24.9 MiB** |
| superfast | 23.7 s | 25.6 | 52.8 MiB |

`veryfast` is both faster than `medium` and *smaller*; `superfast` buys 1% more speed
for twice the bitrate, a bad trade on a 3-hour render.

**`--palette-tile 128` instead of full resolution.** The palette build was 25-49 s
of a ~30 s run -- the largest line item left. Tile size costs 36x and buys nothing:

| tile | palette ms | mean saturation |
|---|---|---|
| 0 (full resolution) | 49201 | 84.7% |
| 512 | 5617 | 84.6% |
| 256 | 2243 | 84.7% |
| **128** | **1363** | **84.9%** |

A full-resolution montage is 949 MiB to build a palette that is measurably no
better. The default is now 128: **1.5 s instead of 25-49 s.**

### Where it ended up

605 frames, 1920x1080, 16 colours, `--engine blocks`, B=32, batch 16, defaults:

```
palette  : 16 colours from 30 sampled frames, 128x128 montage, 1.5 s, 84.9% saturation
frames   : 605 in 24.81 s (24.4 fps)
busy time: palette 1497 | decode 16599 | dither 20346 | encode 22699 | wall 24808 ms
output   : 25.4 MiB
```

Against the 284.5 s the same clip originally took, that is **11.5x**. The stages are
now balanced -- decode 16.6 s, dither 20.3 s, encode 22.7 s of worker-time inside a
24.8 s wall -- so no single stage dominates, and the remaining gains are not in this
pipeline any more. Getting past it needs more than one of each stage: parallel
decode, and either a faster encoder or several concurrent chains.

`--self-test` PASSED and 135/135 bit-exact throughout, including after every change
above.

to genuinely saturated, and is **not implemented**.

### `--palette-export`: write the palette out

`--palette-export PATH` writes the palette after it is built, so it can be
inspected, diffed, kept under version control, and fed straight back in with
`--palette-from`. Two formats, by extension:

- `.png` -- a 1 x N strip of the palette colours. Not a hand-built indexed image:
  that was tried and the PNG coder silently re-derived the colormap and dropped 16
  entries to 4. A plain strip needs no colormap, and the reader recovers the
  palette from its unique pixel values -- which is exactly what
  `magick ... -unique-colors` emits, so **one reader handles our files, the
  reference pipeline's, and anyone else's**.
- `.txt` -- a plain hex table with both Q16 and 8-bit values, for diffing and for
  other tools.

The round trip is lossless: 16 colours exported, 16 adopted, mean saturation
unchanged at 84.2%.

### Crash and interruption protection

- **`Child::Close()` now terminates the child if it is still running.** It
  previously only closed handles, which left ffmpeg alive and orphaned on any
  early exit -- so an error on a three-hour job leaked processes still holding the
  output file. A child that has already exited is left alone, so the normal path
  still gets ffmpeg's real exit code and its chance to finalise the container.
- **Ctrl-C unwinds cleanly.** A console handler sets an atomic flag and nothing
  else, because doing real work in a control handler is unsafe; the reader, worker
  and writer threads notice it at their next checkpoint and unwind through the
  normal shutdown path, which closes the encoder's stdin so ffmpeg finalises what
  it has. Exit code is 130, the conventional 128 + SIGINT, and the message says
  plainly that the output holds the frames completed so far.
- **A 600 s stall watchdog**, so a wedged decoder or encoder reports instead of
  hanging silently.
- CUDA allocation failures return error strings rather than aborting, and the
  pinned-staging allocation degrades to a pageable copy.

Verified: a hard kill mid-render leaves **0** orphaned ffmpeg processes, because
closing the inherited pipe handles gives the children EOF. The graceful Ctrl-C
path is implemented and compiles into the pipeline, but a real console control
event could not be raised from the test harness, so it is **unexercised** -- treat
it as reviewed rather than verified.

- **Per-launch GPU overhead, ~19% of the dither stage.** The curve is rebuilt and
  re-uploaded on every launch though it depends only on geometry (3.4 s over 38
  batches), and uploads are pageable at ~865 MB/s instead of pinned (5.8 s).
  Together about 9 s of the 55.5 s. `cudaHostRegister` per slot plus a
  geometry-keyed device-side curve cache are the two fixes.
- **Decode is no longer the bottleneck** and does not need attention: at 605
  frames, dither is 63.5 ms/frame against decode's 25, so hiding decode fully was
  worth far less than fixing the queue was.
- **The dither inner loop.** The tree descent was ~36 float divisions per pixel
  per block, because `ColorToQNodeId` recomputed the four `ScaleQuantumToChar`
  values at each of the 8 descent levels. Hoisting them out leaves 4 and is
  bit-identical (AE = 0, 135/135 still exact) -- but it bought **no** measurable
  wall-clock, which says the kernel is memory-bound, not division-bound. So this
  line of attack is exhausted; a per-block memo table is the only remaining idea.
- **The palette cost.** 2261 ms of a 8746 ms clip is the one-off palette build.
  The octree walk is single-threaded on the host and is the obvious GPU target.
- **Multi-GPU / streams** for the batch axis. The persistent device cache and pinned
  staging are done; what is *not* done is running several batches concurrently on
  separate CUDA streams, which is what would take GPU utilisation from one batch at
  a time to saturated. That needs N device states rather than one, and N times the
  VRAM for the pixel buffers.
- **Multi-GPU.**
  differ) and is not reachable from `verify.ps1`, which is how it went unnoticed.
- **The host block walk is not bit-exact** against the CUDA walk (~1.6% of pixels
  you use `--im-palette` or `--palette-from`, both of which now match it.
- **The palette is still flatter than the reference on grey-heavy footage** unless
  remaining idea.
  Transfers are no longer the bottleneck. A per-block memo table is the main
- **The dither kernels are memory-bound** (~890 ms of each 1030 ms launch).
  constant 100% load needs N device states on N streams, not one.
  pinned staging took the dither stage 1.38x faster (101 -> 72 ms/frame), but
- **GPU utilisation is still one batch at a time.** The persistent device cache and
  produce frames.
  its durability and its verification are done; the segment decode does not
- **Crash resume is built but does not work** -- see above. The checkpoint format,

### Round two: the writer thread, and one negative result

With the dither no longer the bottleneck, the writer thread became the thing to
measure, so it now reports its own two halves -- they scale differently and only
one of them is a candidate for moving to the GPU:

```
writer 15146 ms of that encode stage: 3663 ms float->uint16 on the host,
                                       11483 ms pushing the pipe
```

**A 16 MiB pipe buffer instead of 1 MiB.** The writer hands 33 MB frames to the
encoder, and with a 1 MiB buffer it blocked on backpressure every frame, so the
measured "pipe" stage was context switching rather than the encoder. The buffer is
the handoff window between the three stages, so it is sized like one. The pipe
stage fell 985 -> 789 ms on a 60-frame clip.

**Parallel float->uint16.** 3663 ms of pure host work, proportional to pixels, on
the one thread that also pushes the encoder's pipe. The batch's frames are
independent, so it is split across three cores. Same arithmetic per pixel, so the
output is bit-identical; only the thread it happens on changes.

Together with the encoder preset change, the encode stage went **22699 -> 15146 ms**
on 605 frames.

**Double buffering the GPU transfers: measured, no gain, reverted.** Transfers are
now the larger half of the dither stage (~141 ms of transfers against ~170 ms of
kernel per batch), so overlapping batch N+1's upload with batch N's kernels is the
obvious move. Implemented with two pixel buffers on two CUDA streams, it measured
**20668 ms of dither against 20334 ms** for one buffer: nothing, for ~1 GB of VRAM
and ~1 GB of pinned host memory.

The reason is worth recording, because it is not obvious: overlapping needs two
launches in flight at once, and the caller is *synchronous* -- it
`cudaStreamSynchronize`s before returning, so the next launch cannot be issued
until the previous download has landed. Two buffers and one thread is just two
buffers. Reverted to one, and the code says so.

### Where the time goes now

605 frames, 1920x1080, 16 colours, defaults:

```
frames   : 605 in 24.09 s (25.1 fps)
busy time: palette 1066 | decode 16219 | dither 20315 | encode 15146 | wall 24086
```

The encode stage is no longer what limits the run, and the pipeline got slightly
*less* well overlapped as it got faster (91% -> 84%): the GPU worker is 84% busy and
neither of the two remaining stages can absorb the slack. The wall is now set by
the dither at 20315 ms with 11483 ms of pipe and 16219 ms of decode underneath it.

The remaining wins are architectural rather than tuning, and neither is built:

- **Two GPU worker threads with independent device states.** This is what the
  double-buffering experiment was reaching for, and it is the one that would
  actually take the GPU to full utilisation: with two calls in flight, one is
  downloading while the other computes. It needs a state *pool* and per-state
  locking rather than the single mutex the worker holds today.
- **Parallel decode.** 16219 ms for 605 frames is one ffmpeg process. Splitting the
  input into ranges decoded by several would take the floor down further, but it
  is the same segmenting machinery the unfinished `--resume` path needs.

### Round three: the GPU writes the encoder's bytes directly

The writer's 3649 ms of float->uint16 conversion was pure host work, and the
download that fed it moved 16 bytes per pixel to deliver 8. Both halves of that
were removable at once: have the scatter kernel emit packed rgba64le.

`BlkScatterKernel` now takes an optional `std::uint16_t*` and, when it is given,
writes four uint16 per pixel -- exactly the byte layout `rgb64le`/`rgba64le` wants
-- instead of a float4. The float4 buffer then becomes purely the *gather's*
input, so it is not written back at all, and the download is halved. The host
conversion disappears: **3663 ms -> 0 ms of 605 frames**, plus a third of the
per-batch transfer.

Two details that are load-bearing and easy to get wrong:

- **Alpha has to be written too.** The float path leaves it at 65535 unless alpha
  is associated, and the encoder consumes these bytes directly, so there is no
  second chance to fill it in.
- **The download size is `pixel_used * 4 * sizeof(uint16_t)`, not `bytes`.** My
  first version reused the float4 byte count, which reads four times past the end
  of the buffer. The resulting error does not surface where it is caused: the
  first batch completes, the *next* batch's launch check reports
  `block walk launch failed: invalid argument`. Hence the `cudaGetErrorString` in
  the walk launch check, which is how it was found at all.

**Verifying it.** The device's conversion is `double -> float -> truncating
uint16`, the same three steps `FloatsToRaw` performs, but "the same" is a claim
about two compilers, so it is measured rather than argued. Two independent checks:

- `RD_CHECK_U16=1` makes the kernel write both buffers and download both, then
  compares all 33177600 samples of a 4-frame batch against what `FloatsToRaw` would
  compute: **0 differ**. It costs a full host pass over the batch, so it is off
  unless the variable is set.
- `--gpu-float-out` selects the old path end to end. Run both with
  `--video-lossless` and compare the *decoded pixels*, not the files:
  `PSNR r:inf g:inf b:inf a:inf average:inf`.

That second point is a trap worth recording. Comparing `md5` of the two `.mkv`
files looks like the obvious test and is **worthless**: Matroska embeds a random
SegmentUID and a writing timestamp, so two runs can never produce equal files
even when every pixel matches. It "failed" for several rounds here and looked
exactly like a real mismatch. Compare decoded frames, or the hash of the raw
dump.

There was also a plain bug in the wiring, found by the A/B: the writer decided to
skip its conversion from "this batch went to the GPU" rather than "this batch's
`b.raw` was written by the device". Those differ whenever the GPU returns float4
instead -- `--gpu-float-out`, or any future path that does not convert on the
device -- and the encoder was then handed an uninitialised buffer.

Result on 605 frames:

```
frames   : 605 in 21.68 s (27.9 fps)      (was 24.06 s / 25.1 fps)
busy time: palette 1066 | decode 14826 | dither 18696 | encode 12419 | wall 21681
writer 12419 ms of that encode stage: 0 ms float->uint16 on the host, 12419 ms pipe
```

`encode` 15288 -> 12419, `dither` 20446 -> 18696, and 25.1 -> 27.9 fps overall.
Bit-exactness held throughout: 135/135 and `--self-test` PASSED.
### Round four: two GPU workers -- another negative result

Round two's double-buffering failed for a specific, diagnosable reason: the caller
is synchronous, so there was never a second launch in flight to overlap with. The
fix for *that* is not two buffers, it is two calls -- which means one independent
device state per GPU worker, each with its own buffers, stream and mutex, so that
one call is downloading while the other computes. That is what the state pool in
`rd_blocks_cuda.cu` is, and it is the thing round two was reaching for.

It works, and it does not help. On 605 frames, three runs of each:

```
gpu-workers=1 : 27.9 / 28.1 fps   dither worker-time 18933 / 18835 ms
gpu-workers=2 : 27.8 / 28.1 fps   dither worker-time 19408 / 19382 ms
```

Identical throughput, and the dither stage gets slightly *worse* in aggregate --
about 570 ms more worker-time, which is contention, not throughput. The conclusion
is the opposite of the round-two hypothesis: the dither stage is **saturated, not
transfer-bound**. Adding a second worker cannot fill a pipeline that has no gaps to
fill, and the transfer chain is already overlapping with the reader and writer on
the host side. So the ~141 ms-of-transfers figure that motivated both attempts was
a bad inference from a per-batch total, not a measurement of idle transfer time.

The pool and `--gpu-workers` are kept -- a negative result worth re-checking on
different hardware should be re-checkable -- but the default is **1**, so the
ordinary path does not pay for a second copy of the per-batch device buffers
(~1.9 GiB of VRAM and ~0.5 GiB pinned at 1080p and 16 frames per batch).

### Round five: three cheap things, and what they proved

All three were cheap. Two of them are negative results, and together with the
hardware they point at one specific fix.

**The machine.** Xeon E5-2620 v3, 6 cores / 12 threads at 2.40 GHz, 15.84 GiB RAM,
**GTX 1650 SUPER with 4.00 GiB VRAM**. The VRAM figure retroactively justifies
`--gpu-workers 1`: two states would have wanted ~2.7 GiB of device memory plus
~1 GiB pinned, which was never going to sit comfortably on a 4 GiB card.

**A deeper queue does nothing.** `--queue-depth` 3, 4, 5, 6, 8 gave 27.6, 27.9,
27.8, 27.9, 28.0 fps, and the GPU worker stayed idle 45-47% of its stage time at
every depth. So the starvation was never a buffering problem.

**ffmpeg thread caps do nothing.** Decode 4 / encode 6, 6 / 4, 8 / 2, 1 / 10,
10 / 1, 3 / 3 all landed within 27.6-28.0 fps, with `decode` pinned at ~14.7 s
every time. Checked against ffmpeg directly rather than through rdither:

```
h264 decode, 605 x 1080p, rawvideo to null:
  -threads  1 :  5363 ms      -threads  6 :  5012 ms
  -threads  2 :  4895 ms      -threads 12 :  5035 ms
```

h264 decoding barely scales on this clip -- 1.1x from one thread to twelve -- so
there was no parallelism to win back. The knob is kept because it costs nothing and
the answer is clip-dependent, but it is not where the time is.

**The reader's widening, parallelised: +0.8 fps.** This one is real but small, and
the reason it is small is the interesting part. Three reps each:

```
  rep1 :  rt=1: 27.7 fps   rt=6: 28.5 fps
  rep2 :  rt=1: 27.6 fps   rt=6: 28.6 fps
  rep3 :  rt=1: 27.9 fps   rt=6: 28.2 fps
```

Six threads instead of one buys 0.8 fps, not the 5x the arithmetic suggests. It is
**memory-bandwidth-bound**: 605 frames is two billion uint16->float conversions, each
reading 8 bytes and writing 16, so ~48 GB of traffic against a memory system that
delivers a small fraction of that per core. Adding cores adds contention, not
throughput. The `decode` stage figure gets *worse* with more threads for the same
reason, which is why the naive reading of that number is misleading.

**What the three together establish.** ffmpeg needs ~5.0 s to decode the clip. The
reader's stage was taking ~14.7 s. So roughly 10 s of it was never ffmpeg at all --
it was the widening loop and the 20 GB of pipe traffic, and it is bandwidth-bound,
which is exactly why neither more slots nor more threads nor more GPU workers moved
it. The only fix is to not do the work: upload the uint16 bytes as they arrive and
let the gather kernel widen them, which removes ~48 GB of host memory traffic and
halves the H2D at the same time. That is the uint16 upload, still the largest single
win available, and still gated on the video path having no automated bit-exactness
check.

28.3 fps after this round, 135/135 and `--self-test` PASSED.
### Round six: the widening is gone, and RD_TRACE now sees everything

**28.3 -> 33.8 fps**, and peak host memory fell from 2919 MiB to 1906 MiB at the
same time. 135/135 and `--self-test` PASSED. This is the change rounds two through
five were all circling.

**What it is.** ffmpeg delivers `rgba64le`. The reader used to widen that to float4
before the device ever saw it, and the dither kernel wanted float4. Now the uint16
bytes go up as they arrive and `BlkGatherU16Kernel` widens them in the kernel:

```
dst[0] = static_cast<T>(static_cast<float>(p[0]));
```

which is the *same* expression `RawToFloats` used. It is exact, not approximately
exact: every uint16 is representable in a float's 24-bit mantissa, so there is
nothing to round. The dither sees bit-identical input either way.

**Why it was worth it.** The loop was the reader's dominant cost and it was
bandwidth-bound, not compute-bound -- two billion conversions over 605 frames, each
reading 8 bytes and writing 16, so ~48 GB of host memory traffic. That single fact
explains every previous result: six threads on it bought 0.8 fps (round five) because
the bus was the constraint; a deeper queue did nothing (round five) because the
reader was not buffer-starved; two GPU workers did nothing (round four) because the
device was waiting on batches that did not exist yet. The device was idle 47% of its
stage time. It is now **29%**, and the dither stage fell 20545 -> 13326 ms.

**Three buffers became two.** A slot holds `in16`, `out16`, and -- only when a host
path needs it -- `pixels`. Splitting `raw` into in and out is not extra memory: the
old code had the reader fill one buffer and the device later overwrite it with the
result, which only worked because the reader's copy was dead by then. And on the
pure GPU path the float4 buffer is no longer allocated at all, 16 bytes per pixel
per slot. The RAM budget had to be taught the same arithmetic (`per_batch_bytes()`),
because a budget derived from a stale number is exactly the hole it exists to close.

There is a trap in that, and it is worth writing down: the float buffer is needed
whenever *any* host path can run, not only when the GPU is off. `--cpu-threads 1`
alongside a GPU still dithers from float4, so keying the decision on `use_gpu` alone
left `b.pixels` empty and handed the host worker a null pointer. Caught by running
the mixed configuration, not by reading the code.

**Verifying it.** The video path still has no automated bit-exactness check, so this
was gated on the decoded-raw A/B, comparing the two genuinely different code paths:

```
--video-lossless, default          vs   --video-lossless --gpu-float-out
PSNR r:inf g:inf b:inf a:inf average:inf
```

Bit-identical, 60/60 frames, and the device path was also faster (24.3 vs 18.4 fps on
the short clip). Both host paths were then run explicitly -- `--cpu-threads 2` with a
GPU, and `--no-gpu` -- to confirm the conditional allocation did not break them.

**Where the time is now.**

```
frames   : 605 in 17.90 s (33.8 fps)
busy time: palette 1070 | decode 16801 | dither 13326 | encode 10604 | wall 17900
[ram]    12367 MiB free at start; queue 1519 + reserve 1274 = ~2793 MiB peak
```

Decode is now the largest stage at 16801 ms, and the wall is only 1099 ms above it,
so the pipeline is overlapping at ~94% -- up from 83%. ffmpeg itself needs ~5.0 s
to decode this clip, so roughly 11.8 s of the reader is 20 GB of rgba64le coming
through a pipe. That is now the thing to attack, and the options are a larger pipe
buffer still, or skipping the pipe for shared memory.

**RD_TRACE fixed.** It used to live at the call sites, which meant it covered the
palette decoder and nothing else: the decode and encode command lines were invisible.
That is not a cosmetic gap -- it is why the round-five thread-cap result could not be
confirmed from inside the tool and needed a hand-written ffmpeg experiment to check.
The trace now lives in `Child::Start`, with a `role` label, so a new spawn site cannot
be added without being traced:

```
[spawn:probe]           ffprobe ... -show_entries stream=width,height,...
[spawn:palette-frames]  ffmpeg ... -vf select=not(mod(n\,30)) -frames:v 2 ...
[spawn:video-decode]    ffmpeg ... -threads 10 -i "..." -f rawvideo -pix_fmt rgba64le -
[spawn:video-encode]    ffmpeg ... -threads 1 -f rawvideo -pix_fmt rgba64le -s 1920x1080 ...
```

Which immediately confirmed the round-five caps were reaching ffmpeg all along, so
that negative result was real.
### Round seven: page-locked batch buffers, and a buffer size that did not matter

**33.8 -> 36.2 fps**, and peak host memory fell again, 1911 -> 1666 MiB. 135/135 and
`--self-test` PASSED, and the bit-exactness gate was re-run across the two paths
(`PSNR average:inf`).

**First, a negative result that redirected the work.** The reader's stage was 16.8 s
while ffmpeg needs only ~5.0 s to decode the clip, so ~11.8 s looked like pipe
transport. The obvious lever was the pipe buffer, already 16 MiB. Raising it to
64 MiB measured **33.9 vs 33.8 fps** -- nothing. So the reader is not transport-bound;
it is ffmpeg converting and writing 20 GB of `rgba64le`, and no buffer size touches
that. (Trying to confirm this with `ffmpeg ... | findstr` over 20 GB of binary was a
mistake: it hung the shell, because that is a PowerShell-native-command pipeline
buffering the whole stream. Measured in-process instead.)

**The real cost, found by reading the data path.** The reader deposited each frame in
pageable `in16`; the GPU worker then `memcpy`'d it into pinned staging before the H2D.
That is **20 GB of host-to-host copy on the GPU worker's critical path, achieving
nothing** -- the same class of mistake as round two's double buffering, and it survived
because it was written before the uint16 upload made the upload cheap.

The fix is to have the reader write *directly* into page-locked memory, so the copy
engine DMAs from exactly where the decoder put the bytes. `CudaAllocPinned` /
`CudaFreePinned` are exposed for that, `BlockOptions` carries `in_u16_pinned` and
`out_u16_pinned`, and a pinned `out16` also makes the download a true async DMA rather
than one through the driver's bounce buffer.

```
frames   : 605 in 16.70 s (36.2 fps)      (was 17.90 s / 33.8 fps)
busy time: palette 1052 | decode 16133 | dither 9042 | encode 11087 | wall 16704
```

The dither stage fell **13326 -> 9042 ms**, which is more than the memcpy alone could
explain: the pinned destination helped the download too. It is now the *smallest* of
the three stages, having been the largest for three rounds.

**Memory went down again, and that is not automatic.** Pinned memory cannot be paged
out, so making the batch buffers pinned is exactly the sort of change that can quietly
invalidate a RAM budget. Two things keep it honest:

- `per_batch_bytes()` counts what is actually allocated, and the `[ram]` line reports
  queue + reserve, so the projection and the allocations cannot drift apart.
- With the staging buffer no longer needed on this path, its allocation is skipped
  entirely -- 265 MiB of pinned host memory that would otherwise have been reserved
  for a memcpy that no longer happens. That is where the 1911 -> 1666 MiB came from.

`Batch` became move-only to own the pinned pointers safely, which is why its copy
constructor is deleted and both move operations are written out: a defaulted move
would have copied the pointers and left the source still holding them, double-freeing
on destruction. That also forced `NewBatch()` away from `vector<Batch>(1, std::move(b))`,
whose count/value constructor reaches for copy.

**Where the time is now.**

```
frames   : 605 in 16.70 s (36.2 fps)
busy time: palette 1052 | decode 16133 | dither 9042 | encode 11087 | wall 16704
```

Decode is the floor at 16133 ms and the wall is 571 ms above it, so the stages overlap
at ~96%. 18001 frames is about 8.3 minutes.

The next target is the remaining ~11 s of the reader, which is ffmpeg's `yuv420p` ->
`rgba64le` conversion and the write that follows it. The one clean idea left is to ask
for `rgb48le` instead: 6 bytes per pixel instead of 8, a 25% cut in both the
conversion's output and the pipe traffic, and lossless here because the source carries
no alpha and the device already writes an opaque alpha itself. The device-side
widening would then have to synthesise the alpha channel, which is one line in the
gather kernel. Not tried, and worth roughly 2-3 s.
### Round eight: `rgb48le` -- 25% less traffic, and not the same picture

**Reverted. This is the round where being nearly right was the problem.**

The reader's stage is 16108 ms and ffmpeg needs ~5.0 s to decode, so ~11 s is
`yuv420p` -> `rgba64le` conversion and the write after it. That work scales with the
output width: `rgba64le` is 8 bytes per pixel, `rgb48le` is 6. The source carries no
alpha, ffmpeg writes a constant 65535 for it, the dither ignores it and writes its own
65535 back -- so it looked like a free 25% cut in both the conversion's output and the
pipe traffic, with a one-line change in the gather kernel to synthesise the alpha.

The reasoning was sound. The premise was not. **ffmpeg's `rgb48le` and `rgba64le`
conversions do not produce the same RGB.** Measured on frame 0 of the bench clip:

```
frame 0 RGB samples differing between ffmpeg's rgba64le and rgb48le: 4525709 of 6220800
  first: px 0 ch 0 : rgba64le=31831 rgb48le=31611
```

Three quarters of the samples differ, by small amounts -- the 3-channel and 4-channel
scaler paths round differently. The end-to-end check caught it too: `--video-lossless`
on both paths gave `PSNR average:21.8`, nowhere near the `inf` that a genuine no-op
gives.

So the saving is real but it is paid for by changing every pixel of the image. That is
not an optimisation, it is a different picture: this tool's entire value is that the
dither is bit-exact, and a 25% speedup bought with a 3% colour shift would have shipped
looking fine in a spot check and been wrong in every frame. Reverted, with the numbers
kept in the source at the decision point so the next person does not re-derive the idea.

Two lessons worth recording, because both are traps this project has walked into before:

- **The bit-exactness gate earned its keep.** `verify.ps1` covers the single-image path
  and says nothing about this, but the decoded-raw A/B compared two genuinely different
  paths and said `21.8 dB` immediately. Without it this would have been a plausible
  looking 25% win.
- **"The alpha is constant, so it does not matter" was a true statement about the wrong
  thing.** Alpha really is inert. The problem was never the alpha; it was that changing
  the *pixel format* changes how ffmpeg rounds the other three channels. The check
  belonged on ffmpeg's output, not on the channel semantics.

**Unchanged, and confirmed after the revert:**

```
frames   : 605 in 16.67 s (36.3 fps)
busy time: palette 1038 | decode 16108 | dither 9013 | encode 10988 | wall 16673
135/135, --self-test PASSED, peak RAM 1665 MiB
```

**What is actually left.** The reader's 16.1 s is now the whole problem, and it is one
ffmpeg process doing three things: h264 decode (~5 s), colour conversion, and writing
20 GB down a pipe. The pipe size is not it (64 MiB measured: no change). The format is
not available (above). What remains is:

- **Parallel decode**, which is the only thing that can move the floor, and which is
  blocked on the segmenting machinery `--resume` needs.
- **A cheaper colour conversion.** swscale's `yuv420p` -> `rgba64le` may not be the
  fastest available path; `-sws_flags` has options worth measuring. But note the trap
  above: any flag that changes rounding changes the output, so this has to be measured
  against the reference, not against "looks the same".
- **Moving the conversion off ffmpeg entirely** -- decode to `yuv420p` (a quarter of the
  bytes) and do the YUV->RGB on the device, which has spare capacity now that the dither
  is the smallest stage. That is a real change to the arithmetic, and reproducing
  swscale's exact rounding on the device is the hard part; it would need the same
  treatment as the dither kernel.
### Round nine: the machine is full, and that is the answer

This round set out to build parallel decode -- N ffmpeg processes, each taking a frame
range, so N pipes. It never got built, because the premise was measured first and
failed.

**Splitting the decode across processes barely helps.** Same clip, output to the NUL
device so no pipe is involved:

```
1 process, whole clip     :  5273 ms
2 processes, in parallel  :  4791 ms    (9% -- and each range re-decodes the prefix)
4 processes, in parallel  :  5077 ms    (worse than 2)
```

**The decisive test was running whole pipelines concurrently.** If the machine had
headroom, one pipeline plus a second would sum:

```
1 concurrent: 35.8 fps                    aggregate 35.8
2 concurrent: 19.2 fps + 19.5 fps         aggregate 38.7
3 concurrent: 12.9 fps + 12.9 fps + 13.0  aggregate 38.8
```

Aggregate throughput **saturates at ~38.8 fps** and does not move again. One pipeline
in isolation does 36.3 fps, which is **93% of everything this machine can do** at this
resolution and colour depth. Parallel decode could therefore win at most 7%, and N
decoders would each re-decode their prefix through the `select` filter -- so the cost
side gets worse exactly as the benefit side vanishes.

**What the ceiling actually is.** The reader moves 20 GB of `rgba64le` per 605 frames
in ~15.9 s, about 1.26 GB/s, and adding processes does not raise it. So it is not
per-pipe syscall overhead that could be parallelised away. Writing the same stream to a
file instead of a pipe is *slower*, not faster:

```
60-frame clip, 949 MB:  -> file 1046 ms = 0.93 GB/s     (pipe is ~1.26 GB/s)
```

Which means the limit is the machine's data-movement capability, not the transport
mechanism. A shared-memory or file-based redesign -- the two things that would move a
pipe-bound reader -- cannot help, because the pipe is already the faster of the two.
A file-based scheme is also unusable for the real workload: 2-3 hours of raw
`rgba64le` is hundreds of gigabytes, which is why it is not on the table regardless of
speed.

**So the honest end state is this.** The reader is ~11 s of ffmpeg converting and
writing 20 GB, and that is the machine, not the code. The only remaining lever is fewer
bytes per pixel, and both routes to that are closed:

- `rgb48le` at 6 bytes instead of 8 is a 25% cut and is **not** the same image --
  ffmpeg's 3-channel and 4-channel scaler paths round differently (round eight).
- decoding to `yuv420p` at 1.5 bytes per pixel is a 75% cut and would take the pipe from
  ~10.6 s to ~1.6 s, which is the only remaining change worth more than a few percent.
  It requires reproducing swscale's chroma upsampling and matrix rounding in a CUDA
  kernel, bit-exactly, and round eight is the standing evidence of how easy it is to
  get that subtly wrong: a 3% colour shift would look fine in a spot check and be wrong
  in every frame.

**What this means for the recommendation.** At 36.3 fps on a 6-core Haswell Xeon with a
GTX 1650 SUPER, a 18001-frame 1080p clip takes about 8.3 minutes. The pipeline is at
~93% of the machine's measured ceiling, the stages overlap at ~96%, and no stage is
saturated or starved. Further work on this machine has to change the *data volume*,
not the concurrency.
### Round ten: the encoder is a CPU-allocation knob, and `ultrafast` is both faster and truer

The encoder was the second-largest stage (10.3 s of a 17.0 s wall) and the obvious
remaining target. What the sweep found is stranger than expected.

**The encode stage time barely moves; the reader moves enormously.**

```
setting            encode_ms   decode_ms   wall_ms   fps
veryfast crf 18       10355       16154      16948    35.8
ultrafast crf 18       9825       14247      14817    41.0
medium   crf 18        9416       23078      25116    24.1
```

`medium` has the *fastest* encode of the three and the slowest wall by 50%. `ultrafast`
barely changes the encode time yet gains 5.2 fps. Neither is explained by encoding
speed: x264's lookahead and CABAC work at preset medium competes with the decoder for
the same 6 cores, and the reader is the pipeline's floor, so starving it is what costs
the time. `--preset` on this machine is a **CPU allocation knob**, not an encode-speed
knob, and that is only visible because the reader is instrumented separately.

**Fidelity was measured, not assumed, and the result inverts the intuition.** PSNR of
each encoded output against an ffv1 (lossless) render of the same dithered frames --
same `yuv444p`, so this isolates x264's compression:

| setting | PSNR vs true dither | unique colours | size (605 frames) | fps |
|---|---|---|---|---|
| veryfast crf 23 | 35.87 dB | 58008 / 64479 / 62769 | 15.2 MB | 36.4 |
| veryfast crf 18 (default) | 38.59 dB | 44898 / 55418 / 50214 | 25.4 MB | 35.8 |
| veryfast crf 12 | 42.47 dB | 25314 / 36316 / 29725 | 41.8 MB | 35.5 |
| **ultrafast crf 18** | **47.30 dB** | **7961 / 13505 / 4366** | 73.6 MB | **41.0** |

**`ultrafast` is both the fastest and the most faithful**, by 8.7 dB, while producing
*fewer* unique colours than the setting that compresses better. Both facts point the
same way once the content is taken into account. A 16-colour Riemersma dither is an
image of flat blocks carrying a high-frequency single-pixel pattern, and that pattern
sits exactly on block edges -- which is what deblocking and the rate-distortion
machinery exist to smooth over. At crf 18 those tools are actively damaging this
content: they move pixels away from their true palette values and invent intermediate
tones between them. `ultrafast` disables most of them, so the flat blocks survive
intact. Its rate control is then wasteful, and the waste shows up as a 2.9x larger
file -- which on an image made of 16 colours is cheap insurance.

So "worse compressor" and "worse for this tool" are not the same thing, and the usual
reason to avoid `ultrafast` does not apply when the input is already 16 colours.

**`crf` is the quality dial and it is free.** Encode times across crf 12/18/23 are
within noise of each other (10286 / 10355 / 10417 ms), while fidelity moves 6.6 dB.
Higher crf is also *worse* here in the intuitive direction: crf 23 is both the least
faithful and barely the fastest, so there is no reason to raise it.

**Not changed.** The user asked for the preset to be left alone, and that instruction
stands -- but it was given before any of this was measured, and it was presumably
based on the usual "ultrafast looks worse" reasoning, which this content inverts. The
default remains `veryfast` / crf 18. The table above is the decision.

For scale: on the 18001-frame clip, `ultrafast` crf 18 is about 2.2 GB against 755 MB,
on a machine with 47 GiB free. `--video-lossless` remains available and is the only
setting that is exactly the dither.
### Round eleven: profiling the dither, and two exact results

**Both profilers are unusable on this machine, and the workaround is worth recording.**
`ncu` 2026.3.1 fails with `ERR_NVGPUCTRPERM` -- the driver reserves performance
counters for administrators. `nsys` 2026.3.2 injects cleanly (the app runs, produces a
correct 2.8 MB output) and records *no CUDA data at all*, neither kernels nor API calls.
So kernel timings come from `cudaEvent` pairs around the three launches instead, gated
on `RD_KERNEL_TIMING=1`. Same answer, no privileges, no replay.

**Where the dither stage actually goes.** Steady state, 33.18 M pixels per launch:

| input | gather | walk | scatter | sum |
|---|---|---|---|---|
| rgba64 | 6.0 | 172.5 | 5.2 | 183.7 ms |
| yuv444 | 5.6 | 356.2 | 5.2 | 367.0 ms |
| yuv444-prepass | 7.7 | 351.5 | 5.2 | 364.4 ms |

So **the walk is 88-95% of the dither stage**, and the gather and scatter are noise.
This is what the last three rounds could not see: the yuv path's penalty is not in the
conversion at all, it is entirely in the walk -- which is why the coalesced pre-pass
changed nothing. Two hypotheses killed, and the cost located.

**Result 1 -- hoisting the loop-invariant error weights: 5%, bit-exact.** The queue
weight `w = kErb * diffusion * weights[k]` depends only on `k` and on `diffusion`, both
fixed for the whole launch, yet it was recomputed inside the per-pixel loop: 32 double
multiplies per pixel to produce 16 values that never change. On a GeForce card FP64
runs at 1/64 rate, so those were expensive. Hoisted: walk 172.5 -> 163.8 ms, 135/135
and `--self-test` unchanged.

**Result 2 -- dropping the dead alpha queue: 54% WORSE. Reverted.** 16 of the 64
doubles in the register queue are the alpha error, and the only read of them is under
`if (assoc)`, which is false for video -- they are initialised, shifted and stored 33
million times per launch for nothing. Templating the kernel on `assoc` to drop them
looked like an obvious 25% register saving. It cost **163.8 -> 251.6 ms**. The reason is
the interesting part: nvcc only keeps such a local array in registers because the
stride is a power of two, so each index is a shift. At stride 3 every index becomes a
multiply, the array is no longer provably register-resident, and it spills. Reverted,
with the reason left in the source so nobody tries it again.

**Neither result reaches the user.** dither 9052 -> 8739 ms, and the wall did not move:

```
palette 1060 | decode 16353 | dither 8739 | encode 10139 | wall 17014    35.6 fps
```

The reader is the floor by 7.6 s. **Doubling the walk's speed would change the default
throughput by nothing at all.** This is the clearest statement of where the ceiling is:
the dither is 88-95% one kernel, and there is still slack to spare underneath it.

**What is left in the walk.** It is latency-bound, not throughput-bound: 163.8 ms for
33.18 M pixels is ~11800 cycles per pixel, against a few hundred for the arithmetic. The
64-double register queue is what allows so little occupancy. The one large lever left
is reducing that register pressure, and the single attempt at it backfired by 54%.

There is also a colour-index cache in `d_select_index` -- a 6-bit-per-channel hash
lookup that would remove the octree descent and the ancestor scan entirely -- and the
walk passes it as `nullptr`. It is an *approximation*, so it is unusable here, but it is
why the search is worth looking at: the yuv path's 2x walk penalty is data-dependent
inside that search, which is the same reason the coalesced gather could not have helped.
### Round twelve: NVDEC, and the answer to "something other than ffmpeg"

**Yes, there is another decoder, and it is bit-exact.** The GPU has NVDEC (Turing, compute
7.5) and ffmpeg exposes `h264_cuvid` and `-hwaccel cuda`. Wired in behind
`--no-hwaccel` to turn it off.

```
h264 decode, 605 x 1080p, no transport:
  software        5041-5139 ms
  NVDEC -> null   1118-1158 ms   = 541 fps
```

H.264 reconstruction is *normative* -- the spec fixes it -- so a conformant hardware
decoder must produce identical YUV. Verified rather than assumed: the first 10 frames
decoded both ways and written as rgba64le hash identically
(`MD5 1FD40474DE85816DECB2CBFCEF58F5`). ffmpeg still does the swscale conversion
afterwards, so the RGB the dither receives is untouched.

**And it changes nothing in the pipeline.**

```
input        nvdec    decode   dither   encode     wall   fps
rgba64       on      16233     8863    10510    16839   35.8
rgba64       off     16189     8739    10157    16937   35.8
yuv444       on       9229    15299    12829    16756   35.8
yuv444       off     10813    14929     9690    17864   33.8
```

Three of four configurations land on the same number, which is the whole finding. The
software decode was *already hidden*: ffmpeg decodes frame k+1 while the pipe drains
frame k, so removing the decode cost changes nothing when the pipe is the limit. NVDEC
only pays where the reader is not already the bottleneck (the yuv444 row: 10813 ->
9229 ms), and there the dither takes over as the limit anyway.

**NVDEC made the palette stage worse, and was reverted there.** It has to create a CUDA
context per process, a few hundred milliseconds, and that decoder only produces ~30
frames. Measured 1057 -> 1650 ms. The main reader decodes 605 frames and is a different
story; the two decoders are now deliberately different.

**The wall is whichever stage is largest, and that is now explicit.** rgba64 is
reader-bound (16.2 s); yuv444 is dither-bound (15.3 s). Both land at ~16.8 s. There is
no configuration in reach that beats ~35.8 fps without reducing the *largest* stage,
and the two candidates are the ones this document has been chipping at: the pipe
(0.86 GB/s, and a file is slower at 0.93 GB/s so it is not the transport) and the walk
(latency-bound, 11800 cycles per pixel, both optimisation attempts failed).

**What "not ffmpeg" would actually mean.** Dropping ffmpeg for the CUDA video decoder
API would remove the pipe, because NVDEC writes into device memory and a 9.35 GB
download over PCIe is ~1.3 s against the pipe's ~11 s. Reader 16.2 s -> ~2.5 s, and
then the wall becomes the encoder at 10.2 s, or 8.8 s if the encoder is also fed
`yuv444p` (3.5 GB instead of 9.35 GB, via a palette->YUV table the scatter indexes).
That is a path to roughly 55-60 fps.

The blocker is unchanged and specific: the YCbCr->RGB conversion would have to be done
on the device and must **reproduce swscale's rounding bit-exactly**, including 4:2:0
chroma upsampling, the fixed-point coefficients, and the clip. This document has
already been bitten twice by exactly this class of problem -- the two scaler paths
disagreeing on three quarters of samples, and `rgb48le` being a 25% "win" that changed
every pixel. The work is better isolated now than it was (the gather is 6 ms and the
conversion is not the bottleneck), and the verification loop is tight -- compare one
kernel's output against ffmpeg's rgba64le for a frame, iterate. But it is a large piece
of work, not a flag.

One caveat recorded for honesty: absolute fps on this machine drifts downward under
sustained load -- 35.7, 34.3, 32.8 over three back-to-back runs late in a long session,
against 35.6-36.1 early on. Every comparison that decided something here was A/B'd
interleaved within a short window, so the conclusions hold; the absolute numbers do not
travel.
### Round thirteen: the walk is at a local optimum, proved with counters

Nsight Compute (from an elevated shell) on `BlkIndexWalkKernel`, 16 frames of 1080p:

```
gpu__time_duration.sum                                 172.7 / 168.7 / 169.8 ms
launch__registers_per_thread                           255        <- the ceiling
sm__warps_active.avg.pct_of_peak_sustained_active      19.5 %
```

**255 registers per thread is the architectural maximum.** The compiler is pinned
against it and spilling. The 64-double register queue is 128 of those registers by
itself, so only ~8 warps per SM fit and there is almost nothing to hide the 16-deep
serial error-diffusion chain behind. That is the 11,800 cycles per pixel, and it
confirms the occupancy hypothesis -- which had been the leading explanation since
round eleven.

**And it cannot be exploited.** `__launch_bounds__` caps registers to trade them for
occupancy. Swept:

```
unconstrained 163.8 | bounds(128,1) 172.0 | (128,2) 167.2 | (128,3) 302.9
(128,4) 367.8 | (128,6) 475.7
```

At 3 blocks the cap falls below the 128 registers the queue needs, it spills to local
memory, and the 64x traffic disaster that the register queue was introduced to fix comes
straight back -- 1.9x to 2.9x slower. The attribute was removed; unconstrained is the
best measured configuration.

Combined with round eleven's result, both routes out are closed:

- **Fewer registers**: the only smaller version is dropping the 16 unused alpha entries
  (64 -> 48 doubles), which measured **54% worse**. A stride of 3 instead of 4 stops
  nvcc proving the array is register-resident, so it spills anyway.
- **Fewer entries or narrower types**: both change the arithmetic, so bit-exactness
  with the reference dither goes with them.

So the 19.5% occupancy is a consequence of the algorithm's state, not an oversight, and
the walk is at a genuine local optimum. It is not improvable by tuning; only a different
algorithm would move it, and that means giving up bit-exactness.

**What this changes about priorities.** The walk is 88-95% of a stage that is 8.9 s of a
17.0 s wall, with the reader at 16.4 s. There is now 7.5 s of slack under the dither
stage, and no way to spend it. The pipeline is pinned at ~35.5 fps by whichever stage is
largest, and the largest is a pipe carrying 9.35 GB at 0.86 GB/s.

Verified: 135/135, `--self-test` PASSED, 605 frames, 35.4 fps mean over three reps.
Note the walk measures 167.3 ms here against 163.8 earlier and the machine drifts
several percent under sustained load, so these are the same number within noise.
### Round fourteen: the decoder replacement -- what I got, and why I stopped

The goal was to drop ffmpeg for the CUDA video decoder API, so NVDEC writes into
device memory, the 9.35 GB pipe disappears, and the reader goes 16.4 s -> ~2.5 s.
**I did not finish it.** What follows is what was established, because the two facts
that gate the remaining work are not obvious and are worth more than a half-built path.

**The easy half is done and shipped.** Hardware decode is wired in, on by default,
`--no-hwaccel` to disable. It is bit-exact, verified by hashing: 10 frames decoded
software and by NVDEC into rgba64le give the same MD5. H.264 reconstruction is
normative, so a conformant decoder must agree. It changes nothing in the pipeline,
because the software decode was already hidden behind the pipe -- and it makes the
*palette* stage worse (1057 -> 1650 ms, NVDEC's per-process CUDA context dominating 30
frames), so it is off there.

**The blocker is now precisely located, and it is not the matrix.** Two measurements:

1. **The 4:2:0 chroma upsampling is a wide symmetric FIR.** Not nearest-neighbour, not
   bilinear. Recovered by inverting swscale's own 16-bit RGB back to the chroma it must
   have used, on a synthetic sawtooth: the recovered chroma *lags the input, plateaus
   and attenuates*, which is the signature of a filter several taps wide. So this is
   swscale's `initFilter()` coefficient generation, not a two-tap average.

2. **The 8-bit-in / 16-bit-out matrix is not any of the four classic integer forms.**
   0 / 404 exact matches across unclipped samples, for `(298*(Y-16) + 409*(V-128)) >> 8`
   and three 16-bit scalings of it.

The method is the reusable part, so it is kept as a tool: `tools/probe-swscale.ps1`
generates synthetic planes, runs ffmpeg, inverts its output, and reports exact-match
rates so candidate formulas can be tested instead of guessed. Both experiments above
are encoded in it and re-run in seconds.

**What remains, honestly scoped.** Read `libswscale` for this exact build
(`N-122031-gc4d22f2d2c`, 2025-12-08) -- `initFilter()` in `swscale.c` for the chroma
filter, `yuv2rgb.c` and `output.c` for the 8->16 bit matrix -- and port the fixed point
to CUDA. Then the harness above is the acceptance test: one kernel's output against
ffmpeg's rgba64le for a single frame, iterate. This is several more rounds of work, and
it is the whole remaining value: without it the pipe stays and the reader stays at
16.4 s.

**A cheaper alternative was tested and is dead.** Since NVDEC frees CPU, parallel range
decoders might have paid for themselves. They do not: aggregate throughput is 35.5 /
36.7 / 31.5 fps at 1 / 2 / 3 concurrent runs, and three concurrent is *worse* than two
(three CUDA contexts on a 4 GiB card, each holding ~1.3 GB of device buffers). The
machine's ceiling has not moved.

So: the pipe is the reader's limit, the pipe can only go if the conversion moves to the
device, and the conversion cannot move without swscale's exact arithmetic. That chain
is where this stops.

> **Superseded by round seventeen.** The first two links held; the third was wrong. The
> matrix is exactly reproducible, and the thing actually blocking the 4:4:4 path was a
> plane stride that made every frame after the first read the wrong planes. The 42% win
> needed neither the FIR nor anything else that was assumed here. The measuring method
> was right, though: every conclusion below is marked "measured", and the one that was
> measured *worst* -- the wide-FIR finding -- is the one that pointed at the wrong file.
### Crash resume (`--segment-frames` / `--resume`) -- WORKS

**Status: working, and verified bit-exact end to end.** It was documented here for
several rounds as "BUILT, NOT WORKING" with the cause unknown. It is now fixed, and
the thing that was wrong was neither the checkpoint format nor the `select` filter.

```
--segment-frames 15, 60-frame source:  4 segments of ~690 KB, joined, exit 0
crash-resumed (killed after 2 segments) vs single-pass, both --video-lossless:
  PSNR r:inf g:inf b:inf a:inf average:inf
```

The filter was a red herring. Running the exact argv that `rdither` builds, through
`Start-Process` with an argument array so that nothing reparses it, yields 45 frames
from `select=gte(n\,15)` -- the command line was always fine. `RD_TRACE` covering
every spawn (round six) is what made that a two-minute check instead of a guess.

**Three separate bugs, and the segments were only the last one to show.**

- **The writer threw away the batch it was waiting for.** The segment boundary was
  tested as `frames_dithered >= frames_in_segment`, evaluated *before* taking the batch
  off the `done` queue. It fired the instant a 16-frame batch landed on a 15-frame
  segment, so the writer broke out and wrote nothing. That is the 585-byte container
  header, and it is why every segment looked empty even when the dither was visibly
  working. The fix is to judge the boundary on frames *written*, after the write.
- **A batch straddling the boundary was written whole**, duplicating frames that
  belonged to the next segment. The write is now clamped to the frames remaining in
  the segment.
- **The reader had no segment limit at all**, so it decoded to end of file: a
  15-frame segment pulled 30 frames (visible as `x16` then `x14` in the dither log).
  That wasted the decode and, worse, left the decoder to be torn down mid-stream, so
  ffmpeg exited with a crash code and the run reported failure even though the
  segments were intact. The reader now stops at the segment boundary and reports EOF
  for the segment, not just for the clip.

Passthrough frame timing was also missing from the main decoder. It is a no-op without a
filter, and the segment path is the only place one is used, which is exactly why the bug
could sit undetected in plain runs -- the same trap the two palette decoders had already
fallen into. (That option has since been renamed and moved; see round sixteen.)

**The durability layer was correct throughout.** Temp file, `FlushFileBuffers`,
`MOVEFILE_WRITE_THROUGH`, directory flush; on load the job settings and every
recorded segment's existence and exact byte length are verified. A kill mid-segment
leaves the last part file unrecorded and it is simply redone.

**One gap, stated rather than glossed:** the checkpoint records the input's size and
mtime, the geometry, and the settings, but *not* the palette. The palette is rebuilt
from the same input and the same options, both of which are verified, so in practice
it cannot drift -- but a change to the palette code or to the ImageMagick build between
runs would not be caught. Recording a hash of the palette entries would close it.

### Round sixteen: ffmpeg 9 deleted `-vsync`, and the replacement is not a substitution

Upgrading ffmpeg to 9.0.2 broke every video run. The symptom was one line:

```
source     : input2.mp4
video      : 1920x1080, h264, yuv420p, 605 frames @ 60.000 fps
error: ffmpeg produced no frames for the palette sample
```

which points at the palette, at the clip, and at nothing to do with ffmpeg's version. The
actual cause is that **`-vsync` was removed outright**, not deprecated: ffmpeg dies during
option splitting with `Unrecognized option 'vsync'`, the pipe gets zero bytes, and the
palette sampler correctly reports that it read no frames. `ffprobe` still works, which is
why the frame count printed normally two lines above the error -- the file is fine, one of
the two tools had stopped accepting the arguments being handed to it.

**All three decoders were affected, and only the first one was visible.** The two palette
decoders run before the pipeline starts, so the run died there. Had the palette been
skipped, the main reader would have failed the same way. The crash-resume section above
records how this same option already went missing from the main decoder once; the fix now
covers all three call sites rather than one.

**The obvious fix is wrong, in a way that fails just as quietly.** `-fps_mode passthrough`
is the same value: `parse_and_set_vsync` in `ffmpeg_opt.c` maps `passthrough` to
`VSYNC_PASSTHROUGH`, exactly what `-vsync 0` meant. But it is declared

```c
{ "fps_mode", OPT_TYPE_STRING, OPT_VIDEO | OPT_EXPERT | OPT_PERSTREAM | OPT_OUTPUT, ... }
```

so it is an **output** option, while `-vsync` was global and every one of these commands
carried it *in front of* `-i`. Substituting the name leaves the position wrong, and ffmpeg
rejects the whole command with `Option fps_mode ... cannot be applied to input url`. The
spelling and the position have to change together.

They are provably otherwise equivalent: both set the same field,
`ofp->fps.vsync_method = opts->vsync_method` (`ffmpeg_filter.c:984`), consumed by the same
switch at 2669. And measured, not assumed -- 10 frames of `tests/clip1920.mp4` as
`rgba64le` give md5 `1FD40474DE85816DECB2CDCBFCEF58F5`, the value the old `-vsync 0`
spelling was recorded at in round twelve.

So the code carries two fragments instead of one and picks between them with a cached
one-time probe of `ffmpeg -h long` (`-h` alone will not do: `fps_mode` is `OPT_EXPERT`, and
the short listing hides expert options). This keeps the binary working against both old
and new ffmpeg rather than trading one for the other. Measured cost 63 ms once per run, of
which 52 ms is bare process startup -- 0.06% of a 17 s render.

**Verified, on the new build:**

| | |
|---|---|
| the reported command (`input2.mp4`, `--im-palette`, 605 frames) | 605 frames out, 29.9 fps, 20.25 s |
| `verify.ps1` | 135/135 bit-exact |
| segmented vs single-shot, both `--video-lossless` | decoded pixels md5-identical |
| NVDEC on vs off | decoded pixels md5-identical |
| path matrix: `--palette-mode pool`, default builder, `--no-hwaccel`, `yuv444`, `yuv444-prepass`, `--palette-only` | 60/60 frames each, no errors |

The segmented-versus-single-shot row is the one that matters most, because the segment path
is the only user of the main decoder's `-vf` filter, and `-vf` is precisely what makes
passthrough load-bearing rather than cosmetic.

Two things worth recording. `--palette-mode pool` is the only way to reach the first
decoder at all, and `-pix_fmt rgba64le` on the second is unreachable from the CLI --
`palette_depth` has no flag, so that branch is dead code. A fix applied to only the paths
a default run touches would have passed every smoke test above.

Separately: ffmpeg 9.0.2's swscale output is **byte-identical** to the previous
`N-122031` master build, so every measurement in this document still stands. An apparent
hash change was a 30-character mis-transcription of a 32-character MD5 in my own notes.

### Round seventeen: the matrix is solvable, and the 4:4:4 path was broken

**35 -> 49 fps, and the 4:4:4 input path had been producing garbage since it was written.**

#### The matrix

Round fourteen measured the 8-bit-in/16-bit-out conversion and could not find it: the
four classic integer forms all scored 0/376. The reason is now clear -- they are all
8-bit forms, and swscale's intermediate is **15-bit**:

```
Ys = (Y - 128) * 512                                     yuv2rgba64_full_X_c_template
Y  = (Ys + 0x10000 - 8192) * 9539 + (1<<13) - (1<<29)     output.c:1410-1418
R  = clip16((Vs * 13075      + Y) >> 14)                 yuv2rgb.c:786-791
G  = clip16((Vs * -6660 + Us * -3209 + Y) >> 14)
B  = clip16((Us * 16525             + Y) >> 14)
```

Those six constants are `roundToInt16(v * (1 << 13))` applied to `cy`, `oy` and the
BT.601 `inv_table` entries, for limited range with contrast = saturation = 65536 and
brightness = 0. Found by reading `yuv2rgb.c:786` and `output.c:1385`, not by fitting --
and the reading matters, because **two of the six were wrong when derived by hand**
(`v2g` and `u2b`) and would have produced a near-miss that looks right by eye.

Verified, not asserted, by `tools/probe_swscale_matrix.cpp`:

| | |
|---|---|
| full 220x225 (Y,V) grid, R/G/B | 0 mismatches of 49,500 each |
| a real 1920x1080 frame | 0 mismatches of 2,073,600 |
| out-of-range samples | all inside [0, 65535] |

The probe is the acceptance test and runs in 0.16 s, which is what made this tractable;
the PowerShell equivalent takes minutes. It also carries the three experiments that
contradicted my first two hypotheses, and a frame-by-frame differ, because the bug
below was invisible to every per-frame check.

#### The bug

With the matrix exact, the 4:4:4 path was still no faster: the walk kernel measured
**348 ms against 167 ms**, and stayed there. Every explanation was wrong:

- not the matrix -- the probe says the conversion is exact
- not the tree -- the exported palettes are byte-identical (`D81E76A3...`)
- not the chroma filter -- still 348 ms on a native 4:4:4 source, where no filter runs
- not the launch -- same grid, same threads, same `d_cx`, same coefficients

The `d_cx` contents had to be identical, and the walk had to be identical. It was not,
which is only possible if `d_cx` was not identical. And it was: **planar 4:4:4 is three
planes per frame, so frame *f*'s luma starts at `f * 3 * pixel_pitch`, not
`f * pixel_pitch`.** The gather used the latter, so frame 0 was correct and every frame
after it read the previous frame's chroma planes as luma. A frame-by-frame differ put it
beyond doubt: frame 0 identical, 567 of 605 frames differing, 99% of components, worst
delta 65281, PSNR -43 dB. Both 4:4:4 kernels had it.

That also explains the walk regression without any hand-waving. Garbage pixel values send
the octree search down a much deeper path, and the walk is FP64-bound on a card that
runs FP64 at 1/64 rate -- so a 2x cost for meaningless input is exactly what should
happen. **The walk was never precision-sensitive; it was reading the wrong bytes.**

#### What it is worth

605 frames, 1920x1080, 16 colours, defaults, three interleaved reps:

| source | `--input-mode` | fps | decode | dither | encode | wall |
|---|---|---|---|---|---|---|
| 4:2:0 | `rgba64` (was default) | 34.6 | 16358 | 8862 | 11116 | 17036 |
| 4:2:0 | **`yuv444` (now default)** | **49.0** | **5596** | 8864 | 11336 | **12066** |
| 4:2:0 | `yuv444-prepass` | 49.4 | 5694 | 8953 | 11343 | 12147 |
| 4:4:4 | `rgba64` | 33.1 | | | | |
| 4:4:4 | `yuv444-prepass` | **49.0** | | | | |

**+42%**, and the reader is no longer the floor: decode falls 66% to 5.6 s while the
dither (8.9 s) and the encoder (11.3 s) are untouched, so the wall is now set by the
encoder's own 9.35 GB. The 4:2:0 chroma FIR is *not* needed for this and was never
needed -- the win was always about bytes, and the blocker was a bug, not a filter.

`yuv444` is the default over the prepass despite the prepass measuring ~1% faster: the
prepass allocates a second device buffer of 6 bytes/px per queue slot, which the RAM
budget does not account for and which is ~2.4 GB at 4K on a 4 GB card.

#### What it costs

For a **4:4:4 source the two modes are bit-identical**, verified over all 605 frames.
For a **4:2:0 source they are not**: letting ffmpeg reconstruct 4:4:4 and then
converting applies a different chroma reconstruction than swscale's fused 4:2:0 -> RGBA
path, and once Riemersma's error feedback has amplified the difference, **0.67% of output
components change (34.3 dB)**. That is a real change to the image and it is why
`--input-mode rgba64` is kept and documented rather than deleted.

Verified alongside: 135/135 bit-exact, `--self-test` PASSED, 605 frames in and out,
segmented-vs-single-shot decoded pixels identical, and the two 4:4:4 kernels -- written
as independent transcriptions of the model -- agree bit-for-bit.

**The lesson worth keeping.** Round fourteen measured the same conversion and could not
crack it, and concluded the remaining work was the chroma filter. The filter was never
the problem; a wrong plane stride was, and it had been sitting in the code the whole
time, looking exactly like a precision effect because a 2x slowdown on nonsense input is
precisely what a precision effect would look like. Three of my four hypotheses were
testable and three were wrong. The thing that found it was a frame-by-frame differ on
the *dithered output*, after per-frame checks had already said "identical" twice.

## 12. Still open

- **The walk's cost is the palette's tree, and it is content-dependent.** Measured on
  the 18001-frame 1080p clip that motivated round eighteen: the walk kernel takes
  **387 ms** there against 166-181 ms on `tests/L605.mp4`, at identical geometry and
  with an identical tree *size*. The mechanism is nodes-visited-per-pixel in the octree
  search -- **3.71 against 2.63** (compile with `-DRD_CC_STATS=ON`, then
  `RD_CC_STATS=1`, to reproduce). That ratio accounts for the whole dither-stage
  difference. It is not fixable inside the search: ImageMagick's own `ClosestColor`
  has no bound to borrow (it recurses every child and prunes only on the per-channel
  `distance <=` test), the cell bound is identically zero because the descent follows
  the target's own bits, and the 6-bit memo table is a first-touch-wins approximation
  that cannot be used without breaking bit-exactness. What *is* left is the palette:
  a tree built from 16 colours where 10 are near-neutral greys is very lopsided, and
  that is what makes the searches expensive. Deferred, deliberately.
- **The encoder, not the reader, is now the ceiling.** With the reader down to 5.6 s,
  the 18001-frame clip's wall is set by pushing 9.35 GB of `rgba64le` into ffmpeg's
  stdin: 373 s, or 48.3 fps. Even an infinitely fast dither leaves the wall at
  `palette + encode` ~= 400 s, so **~45 fps is the ceiling until the output pipe
  shrinks**, and the only way to shrink it is an exact device RGB->YUV.
- **The encoder is now the ceiling, and it is 94% of the wall.** Round nineteen halved
  the dither stage, 485 -> 236 s on the 18001-frame clip, and the wall did not move:
  the GPU had been hiding the host, and once it stopped, decode and encode absorbed
  the slack. What is left is 298 GB of `rgba64le` per 18001 frames pushed at ~0.6 GB/s.
  Shrinking it means piping `yuv444p` (3 bytes/px against 8) and converting RGB->YUV on
  the device, which needs its own exactness work -- the matrix is solved, the reverse
  direction is not, and it changes what `--video-lossless` means.
  **Solved in round twenty**; the encoder is no longer the wall.
- **The reader is the wall, and it is starved rather than slow.** Round twenty took the
  18001-frame render from 533 s to 399.5 s by moving RGB->YUV onto the device, leaving
  the reader's 393745 ms as the wall -- but a bare `-f null -` decode of the same file
  is 1.46 ms/frame against the stage's 21.9, so it is not decoding slowly, it is being
  denied cores by x264. Profiling confirms the machine is CPU-saturated (10.2 of 12
  logical cores) and that the encoder alone asks for 6 of the 6 physical ones, putting a
  **~51 fps floor** under the `veryfast` default against the 45.1 achieved. Thread caps
  and process priority both make it worse, not better. The two things that would move
  it are the encoder preset (round twenty-one: 45.1 -> 88.8 fps, and slightly better
  quality, for 2.27x file size) and halving the input bytes by decoding `yuv420p`
  directly instead of upsampling to 4:4:4 on the CPU, which would need the 4:2:0 chroma
  filter ported and would move the palette slightly.
- **Crash resume works** (see above) and is verified bit-exact, but the checkpoint does  not record a hash of the palette, so a change to the palette code or the ImageMagick
  build between an interrupted run and its resume would not be detected.
- **GPU utilisation is not the gap it looked like.** Two GPU worker threads with
  independent device states were built, and measured no faster than one (rounds two
  and four). The persistent device cache, pinned staging, the register-resident
  error queue and the device-side uint16 output took the dither stage from 101 to
  30 ms/frame; what is left is saturated work, not idle transfer time.
- **Parallel decode is not worth building on this machine.** Measured, not assumed:
  aggregate throughput saturates at ~38.8 fps whether one, two or three pipelines run,
  and a single pipeline already reaches ~93% of that. The segmenting machinery it would
  have needed now works (see crash resume), so this is a capacity limit, not a missing
  feature.
- **The palette stage is linear in sample count and quality is flat past ~30
  frames** (84.7% at 30 samples, 84.9% at 4096). A palette sample costs about the
  same whether the tile is 128 or 16, because it is dominated by ImageMagick's
  per-call overhead; batching several tiles into one quantize call would cut the loop
  cost and is not done.
- **The remaining win is now the *output*, not the input.** Round seventeen took the
  reader's decode stage from 16.4 s to 5.6 s, so the reader is no longer the floor. What
  is left is the encoder, which pushes 9.35 GB of `rgba64le` into ffmpeg's stdin at
  ~0.86 GB/s and costs ~11.3 s. Feeding it `yuv444p` instead would cut that to 3.5 GB,
  at the cost of an exact RGB->YUV conversion on the device and of `--video-lossless`
  no longer being a 1:1 comparison. Not started; the shape of the work is now known
  rather than guessed, which is the whole change from round fourteen.
- **The 4:2:0 chroma FIR is still not ported, and no longer blocks anything.** The
  42% win did not need it -- the reader's problem was bytes, not filtering. It is only
  needed to make `--input-mode yuv444` bit-identical to the `rgba64` reference for 4:2:0
  sources, which currently differ by 0.67% of components (34.3 dB). The matrix half of
  that is solved and proven; the filter half is `initFilter()` in `swscale.c`.
- **The host block walk is not bit-exact** against the CUDA walk (~1.6% of pixels
  differ) and is not reachable from `verify.ps1`, which is how it went unnoticed.
- **`verify.ps1` does not cover the video path at all.** Reader and writer changes have
  no automated bit-exactness check behind them, and the worst bugs in this project have
  all been in that path: the passthrough-frame-timing option twice, a black video, a
  discarded batch at segment boundaries, its own deletion in ffmpeg 9, and a plane
  stride that made 567 of 605 frames garbage. The gate that catches them is a
  decoded-raw A/B against the previous build, run by hand. Automating it is the
  highest-value remaining work that is not a performance change -- and round seventeen
  is the argument for it, because a frame-by-frame differ on the dithered output found a
  bug that two per-frame comparisons had both called identical. The table in round
  sixteen is the shape it should take.
- **Multi-GPU.**

### Round eighteen: the palette stage was a full decode in disguise

Asked whether ImageMagick quantizes every frame. It does not, and the answer is worth
recording because what the stage *is* turned out to matter:

**ImageMagick is called once per render.** With `--im-palette` the only call is
`ImBuildPaletteAppend8` -> a single `QuantizeImage()` on one montage of the sampled
frames (`rd_video.cpp:1117`). All 18001 output frames are dithered by
`RiemersmaBlocksCuda` -- rdither's own port of `RiemersmaDither()`, which is exactly
why it is bit-exact and ~40x faster than asking IM to do it per frame. IM would be
called per *sampled* frame only in `--palette-mode pool` (30 calls), and per frame
only in `--verify`'s single-image reference.

**But the palette stage was decoding the entire clip.** It samples with
`-vf select=not(mod(n,600))` and the obvious objection is that `select` should skip the
frames it drops. It does not: the filter runs *after* h264 decoding, so ffmpeg decodes
all 18001 frames and throws away 99.8% of the result. Measured on the 18001-frame clip,
the stage reported **26627 ms** and a bare `-f null -` decode of the same file
measured **26.2 s** -- the stage *was* a full decode, to keep 30 frames.

Replaced with one short `-ss`-per-sample invocation, so 30 frames are decoded instead
of 18001. **26.4 s -> 7.0 s, and the palette is byte-identical** (md5
`403536190F7D26E14...` from both paths, 22.4% mean saturation, 10 near-neutral either
way). That is ~19 s off a 540 s render, about +3.7%, on a stage that is serial and so
pure gain.

**The first attempt at it was silently wrong, and only hashing caught it.** Seeking to
each sample's *midpoint* produced a different palette (21.2% vs 22.4%) with no error
anywhere -- 30 frames still arrived, just the wrong 30. `accurate_seek` discards frames
whose presentation time is strictly *before* the target, so the target must be the
frame's own timestamp; the midpoint lands on the next frame. Confirmed by hashing
frame 600 both ways: `-ss 10.000000` matches `select=eq(n,600)`, `-ss 10.008333` does
not. The code now aims a quarter of a frame early, so float rounding in `idx/fps`
cannot overshoot. The kind of bug this project keeps meeting: right shape, plausible
numbers, wrong answer, and only a byte-level comparison sees it.

Gated to clips over 1200 frames (`RD_PALETTE_SEEK=0` forces the old path), because
below that the process spawns cost more than the decode they avoid -- a 605-frame clip
decodes in 0.5 s and its palette stage is untouched at 1067 ms. `--palette-mode pool`
still uses the decode path and still pays the full decode.

### Round nineteen: the octree search, flattened -- 2.06x on the dither, bit-identical

The walk's cost was never the arithmetic. Fitting `t = a + b * nodes_per_pixel` across
two clips of identical geometry gives a **negative intercept** (2.63 nodes/px and
10.4 ms/frame against 3.71 and 24.2) -- the signature of a cost that is not work. What
it was: `ClosestColor` is a chain of dependent loads through a 128-byte node, and
adjacent pixels take different paths, so a warp pays for the union of its lanes'
traversals. More tree, more divergence, superlinearly more time.

**The fix is to run the traversal once, on the host, and ship the answer as data.**
ImageMagick visits a node's children in index order and then the node itself, so a
node's candidates are *its children's lists concatenated, then its own colour* -- and
a leaf wins ties, so the order is the answer, not just the set. With at most 16
colours that ordered list is a presence mask plus a nibble-packed array of palette
indices: **12 bytes per node**, register-resident. `BuildFlatSearch()` assembles it by
executing the same post-order walk, and the device scans the mask instead of chasing
pointers.

| | recursive | flat | |
|---|---|---|---|
| walk, L605 | 189.0 ms | **93.0** | 2.0x |
| walk, 18001-frame clip | 287.1 ms | **91.5** | 3.1x |
| registers / stack | 255 / 80 B | **166 / 0** | |
| dither stage, 3000 frames | 67059 ms | **37146** | 1.80x |
| full 18001-frame dither | 484974 ms | **235547** | **2.06x** |

The two clips now cost the *same* (93.0 against 91.5 ms), which is the real proof: the
sensitivity to tree shape was the recursion, and it is gone. Registers fell by 89
because the recursion's stack frame is gone too.

**Bit-identical, and it took two bugs to get there.** `RD_FLAT_SEARCH=0` forces the
recursive path so the two can be compared on real video, decoded output byte for byte.
The first version folded a child's list into its parent when the child was *pushed*,
before it had been processed, so it merged empty lists. The second computed each
child's positions from zero and combined with `|=`, so a second child collided with
the first's slots and its nibbles overwrote them instead of concatenating. Both
produce a search with the right shape, a plausible-looking render, an unchanged
palette, and **the wrong order -- which is the wrong answer**, because ties go to the
last leaf visited. Neither showed up in 135/135 image cases; only the decoded A/B
caught them. `verify.ps1` passes on both versions because the image fixtures happen
not to hit the colliding shapes.

Palettes above 16 colours do not fit the nibble packing, so `BuildFlatSearch` returns
false, `g_flat_search` stays null and the recursive path serves them. The branch is on
a device global set once per clip, so it is warp-uniform and costs no divergence.
Verified at 16, 32, 64 and 200 colours.

**The end-to-end gain on a long clip is small, and the reason is the honest finding of
this round.** On the 18001-frame clip the dither stage halved, 484974 -> 235547 ms, and
the wall went 540 -> 533 s. The other stages absorbed it: with the GPU no longer the
constraint, decode rose 234 -> 391 s and encode 373 -> 500 s, because the host threads
and ffmpeg were previously being given slack and now are not. **The encoder is now 94%
of the wall** -- 18001 frames of `rgba64le` is 298 GB pushed at ~0.6 GB/s -- and that
is the only stage left worth attacking. Writing the output to the HDD rather than the
SSD makes almost no difference (42.0 against 37.4 fps, and the sign flipped between
runs), so it is pipe bandwidth and ffmpeg's encode, not the disk.

### Round twenty: the encoder stops converting on the CPU -- 33.8 -> 45.1 fps

Round nineteen halved the dither stage and the wall did not move, because the GPU had
been *hiding* the host. With it no longer the constraint, decode and encode absorbed the
slack they had been given, and **the encode stage became 94% of the wall**: 18001 frames
of `rgba64le` is 298 GB handed to ffmpeg, which converts it to `yuv444p` on the CPU
before encoding.

The measurement that says *what* to fix: the writer was blocked at **597 MB/s**, below
the pipe's own 0.86 GB/s. So the transport was never the constraint -- the host-side
conversion was. Shrinking the pipe alone would have moved the same CPU work onto our
writer thread and won nothing. **Both halves have to move together**, and they can:
the scatter now writes planar 8-bit 4:4:4 straight out of the palette, doing the
RGB->YUV on the device, and ffmpeg is fed `yuv444p` with nothing to convert. 3 bytes
per pixel against 8, so 2.67x less to push, download, and convert.

| 18001 frames | before | after | |
|---|---|---|---|
| encode | 499964 ms | **159813** | 3.1x |
| dither | 235547 | 213688 | |
| decode | 390699 | 393745 | now the wall |
| **wall** | **533 s** | **399.5 s** | **33.8 -> 45.1 fps** |

**The fidelity cost is zero, and for a reason worth recording.** The RGB->YUV->RGB round
trip *already happened* -- ffmpeg converted our dithered RGB before writing the file, so
the palette entries in the file were already not the entries we chose. This only
decides who rounds. Confirmed rather than assumed: both paths produce **256 unique
colours** in the decoded output, which is the 8-bit ceiling, so the palette was never
what the output was limited by. A decoded A/B against the old path gives **41.69 dB**
(RMS 54 of 65535).

`RD_YUV444_OUT=0` restores rgba64le for comparison, which is how the A/B was run.

The remaining stage is the reader at 393745 ms, and part of that is queue-blocking
rather than decoding -- a bare `-f null -` decode of this file is 1.46 ms/frame against
the stage's 21.9. So the pipeline is now bounded by how well the host and the GPU
overlap, not by either alone.

### Encoder preset: `veryfast` is the default, and it is the wrong one

`--preset` is left at `veryfast` because it is the project's default, not because it is
the best choice. **Measured on the 18001-frame 1080p clip, `ultrafast` is both faster
and slightly higher quality, and the only thing it costs is file size.** This is
recorded here rather than acted on, since the preset has been an explicit standing
instruction.

Quality, as PSNR against a lossless (`--video-lossless`, ffv1 yuv444p) render of the
same dithered frames, 60 frames at 1080p:

| `--preset` | PSNR vs ffv1 | file, 60 frames | encoder CPU, 300 frames | file, 18001 frames |
|---|---|---|---|---|
| `veryfast` (default) | 29.394 dB | 2.63 MB | 37.6 core-s | 1067 MB |
| `superfast` | 29.446 dB | 5.35 MB | 28.1 core-s | — |
| **`ultrafast`** | **29.466 dB** | 8.09 MB | **13.1 core-s** | **2424 MB** |

Faster presets predict less, so on dithered content -- where the picture is a small
number of flat colours meeting hard edges -- they invent fewer colours at those edges.
That is why the quality column goes *up* as the preset gets faster, and it reproduces
what round ten found on the earlier build.

**The end-to-end effect is much larger than the encode stage alone, and the reason is
not obvious.** Full clip, 18001 frames, nothing else changed:

| | `veryfast` | `ultrafast` |
|---|---|---|
| decode | 393745 ms | **113478** |
| dither | 213688 | 202015 |
| encode | 159813 | 97956 |
| **fps** | **45.1** | **88.8** |

The encode stage barely moves; the **reader** more than halves. The pipeline is
CPU-bound (see below) and x264 at `veryfast` wants 6 cores on its own, so the decoder
was being starved: it reported 393745 ms for work that takes ffmpeg 65.7 ms/frame's
worth standalone. Freeing the encoder's CPU does not speed up decoding -- it stops the
encoder taking it.

### Where the CPU goes, and what the ceiling is

Sampled every 3 s during a render: **mean 10.2 of 12 logical cores, 16 of 17 samples
above 5.5 physical cores.** Per stage, standalone:

| stage | CPU | cores demanded |
|---|---|---|
| x264 encode (`veryfast`) | 2268 core-s | **6.03** |
| h264 decode + 4:4:4 | 371 core-s | 0.71 |
| rdither (reader, writer, worker) | — | ~0.4 |

With `veryfast` the wall is therefore an arithmetic fact rather than an engineering
one: roughly 2100 physical core-seconds of work on 6 cores is a **~51 fps floor**, and
the build achieves 45.1, i.e. 88% of it. The remaining 12% is the serial palette
stage and imperfect overlap. Things tried against CPU starvation and rejected:

| | result |
|---|---|
| `--encode-threads` 2/3/4/5 | every cap was *slower* than auto (57.6 fps against 48.5-56.9) |
| demote the encoder process to `BelowNormal` | **worse**: 44.3 -> 38.2 fps. The encoder is a real throughput stage, not noise; slowing it just back-pressures the dither |
| `-tune zerolatency` | 17% less encoder CPU standalone, but the encoder is not what binds |

The honest conclusion: with `veryfast` fixed, ~51 fps is the ceiling and it is set by
x264's appetite for cores. Only the preset (or hardware encoding, which cannot carry
4:4:4 and so would cost the palette) moves it.

Round twenty-two replaced the estimates in the table above with a direct measurement, and
the conclusion did not change: **x264 is ~89% of all CPU in the pipeline** (1240 of 1590
core-s on the 8000-frame slice), rdither is 7.6%, and h264 decode is 1.8%. Everything this
project can actually control is ~9% of the bill.

### Round twenty-one: the palette stage, parallelised

The palette is a serial prefix that gates the whole pipeline, and in the seek mode it
was 30 sequential ffmpeg spawns at ~230 ms each -- 7 s of a 400 s render spent on
process startup and seek latency with the rest of the machine idle. Each sample is an
independent seek to a different timestamp and owns exactly one montage cell, so the
writes are disjoint and no lock is needed.

| 18001 frames, palette only | time |
|---|---|
| one decode pass, `select` filter | 27.3 s |
| 30 sequential seeks | 7.0 s |
| **30 seeks, 6 at a time** | **3.7 s** |

Byte-identical palette (`md5` equal to the sequential version, 22.4% mean saturation,
10 near-neutral). Concurrency is capped at 6: on a spinning disk thirty simultaneous
seeks thrash the head, and past six the returns flatten. Below 5000 frames the decode
path is used instead, since the spawns cost more than the decode they avoid.

### Round twenty-two: decoding `yuv420p` directly -- 2.3x cheaper, and worth nothing

The standing blocker on making the decoder cheaper was that 4:2:0 chroma has to be
reconstructed, and swscale's FIR for that was unported. The cheap way round it is not to
reconstruct it *like swscale* but to reconstruct it *differently* and let the picture move
a little -- which the standing "almost frame exact" tolerance allows.

`--input-mode yuv420` does that: ffmpeg emits the source format untouched, the pipe carries
1.5 bytes per pixel instead of 3, and the device does a 2x bilinear chroma upsample before
the same bit-exact matrix. The **palette is unaffected by construction** -- the palette
samplers run their own ffmpeg with their own `-pix_fmt rgba` and never read `input_mode` --
and measured identical (same 30 samples, same 25.7% saturation).

Every part of it worked. The decoder is **2.3x faster** (160.6 s -> 45.9 s of stage time,
3.6x on the slice) and the pipe carries 52 GB instead of 104.

**And the wall did not improve.** Four separate measurements, in four different directions:

| measurement | yuv444 | yuv420 | |
|---|---|---|---|
| A/B, 2 reps, straight after a build | 45.7 | 41.8 | -8.5% |
| budget probe, 1 rep, idle machine | 47.4 | 52.0 | +9.7% |
| interleaved, 2 reps | 41.6 | 47.5 | +13.9% |
| ...the same run, cold first rep dropped | 49.0 | 47.5 | **-3.0%** |

The last row is the honest one. That run's `yuv444` samples were 34.3 and 49.0 fps -- a 43%
spread inside one arm -- so the sign of the result is being decided by machine state, not by
the mode. Any of those four numbers could be quoted to argue either case.

The mechanism is not in doubt, though, because it does not depend on fps at all. Measuring
**total CPU** rather than wall time, across all four configurations:

| configuration | rdither | ffmpeg (decode + x264) | total core-s | avg cores busy |
|---|---|---|---|---|
| yuv444, auto threads | 121 | 1465.6 | **1586.5** | 9.15 |
| yuv420, auto threads | 102.8 | 1489.9 | **1592.7** | 10.01 |
| yuv444, `--encode-threads 4` | 117.9 | 1474.4 | **1592.3** | 8.75 |
| yuv420, `--encode-threads 4` | 99.3 | 1510.4 | **1609.7** | 9.52 |

**Total work is conserved to within 1.5%.** Removing 115 s of decoder stage time did not
remove 115 s of CPU, because the encoder is elastic: given a reader that needs less, x264
takes more threads and burns the difference, plus thread overhead. The wall is core-seconds
divided by cores available, and neither factor moved.

This is the same conclusion as round twenty, with a much better instrument. The older table
in *Where the CPU goes* estimated the split; this measures it, and the split is the answer:

| | core-s | share |
|---|---|---|
| x264 encode, `veryfast` | ~1240 | **~89%** |
| rdither (reader, writer, worker) | 121 | 7.6% |
| h264 decode + 4:4:4 | ~29 | 1.8% |

**89% of all CPU in this pipeline is the encoder**, and the encoder is preset-locked by
standing instruction. Even a zero-cost decode and a zero-cost dither -- both impossible --
would cap out at an 11% wall improvement. There is no fps left outside the preset. Every
other knob is already recorded as measured-neutral: `--queue-depth` 3..8, `--batch-frames
32`, ffmpeg thread caps, two GPU workers, 64 MiB pipe buffer, and `--encode-threads` 2/3/4/5
(re-confirmed above: every cap is worse than auto, and by *total CPU*, not just wall time).

Fidelity for the record: **25.8 dB mean, 13.3 dB worst** against the `yuv444` path, with a
`yuv444`-vs-`yuv444` determinism control at inf and the old `rgba64` comparison at 134 dB, so
the metric is sound. That is worse than "a small PSNR error" implies, and the per-channel
columns say why: `mse_u`/`mse_v` sit at 40/38 dB, so the *input* is close and the damage is
the quantizer. A 1-unit chroma difference flips a pixel to a different palette entry, and
the gap between two 16-colour entries is far larger than the gap between two 4:2:0
reconstructions. A 16-colour dither punishes any chroma-side approximation
disproportionately, which is also why porting swscale's exact FIR could not rescue this: it
would fix the input, and the input was never the problem.

The mode ships **off by default** (`input_mode = "yuv444"`) and is not recommended. It is
opt-in, it is guarded, and its help text says plainly that its decoder is 2.3x faster while
its total CPU is not.

### Two guards added, because the opt-in path could corrupt silently

`--input-mode yuv420` reads Y at full size and Cb/Cr at a quarter of it, indexed by
`x >> 1` / `y >> 1`. That is only meaningful for a source that really is 4:2:0 and really is
even in both axes. Handed 4:2:2, 4:4:4, or an odd width it reads the wrong bytes and
produces plausible-looking garbage -- the same failure class as the 4:4:4 plane stride and
the input/output size collision, both of which this codebase has already paid for once each.
It now refuses instead:

```
[video] --input-mode yuv420 needs a 4:2:0 source; this one is yuv422p. Use yuv444 instead.
```

`--input-mode` also validates its argument rather than falling through to the interleaved
path on a typo, since a silently-defaulted option is indistinguishable in the output from a
correct one.

### Two lessons, both about instrumentation

- **Stage times are not CPU.** `decode 160 s -> 46 s` reads like a 114 s saving and means
  nothing: on a saturated machine that stage time is *slack*, not work, and the encoder
  consumes it. Round twenty-two's headline number was a stage time.
- **A 43% spread inside one arm invalidates an A/B.** The runs that decided the sign differed
  by that much, and averaging across a 2-rep design does not rescue a single-digit-percent
  effect measured on a machine whose load drifts by tens of percent.
  `tools/probe-cpubudget.ps1` therefore reports total core-seconds beside fps: core-seconds
  did not move, and that settles the question without needing fps to be trustworthy.

## 13. Where the remaining time is
605 frames, 1920x1080, 16 colours, defaults (round seventeen):

```
frames   : 605 in 12.07 s (49.0 fps)
busy time: palette 1050 | decode 5596 | dither 8864 | encode 11336 | wall 12066
```

**The reader is no longer the floor.** Decode fell from 16358 ms to 5596 ms, so the
wall is now set by the encoder at 11336 ms -- and the two stages it was previously
waiting on are the ones that moved. The pipeline is limited by its slowest stage, which
is what a pipeline is supposed to look like, and for the first time since round one that
stage is the writer rather than the reader.

That reframes what is left. There is no longer an input-side problem: the 9.35 GB
`rgba64le` crossing the decode pipe is gone, replaced by 3.5 GB of `yuv444p`. What
remains is the *output* pipe, which moves the same 9.35 GB into the encoder at the same
~0.86 GB/s and costs ~11.3 s. That is now the single largest item, and the same
bytes-not-concurrency argument applies to it.

**On the machine's capacity.** Round nine measured aggregate throughput saturating at
~38.8 fps across one, two and three concurrent pipelines, and concluded a single
pipeline was at ~93% of the hardware's limit. That was measured with the old 9.35 GB
input pipe, and a single pipeline now does 49 fps -- so the ceiling was not the machine,
it was the reader. The conclusion was right about the method and wrong about the
number, which is worth recording: a saturation measurement bounds the thing you
measured, and this one was measuring a transport that was about to be replaced.

Things that were measured and did **not** help, so they need not be retried:

| | result |
|---|---|
| deeper queue (`--queue-depth` 3..8) | 27.6-28.0 fps, no change at any depth |
| ffmpeg thread caps (decode/encode splits) | all within 27.6-28.0 fps |
| parallelising the reader's widening | +0.8 fps, and the loop is now gone entirely |
| two GPU worker threads | no gain; the device is saturated, not transfer-bound |
| 64 MiB pipe buffer | no change |
| `-avioflags direct` | within noise |
| `rgb48le` instead of `rgba64le` | 25% faster, **different image** -- rejected |
| parallel decode | reader-bound at the time; the ceiling was transport, not CPU |
| `--input-mode yuv444` as opt-in | **re-measured and adopted** -- it was broken, see round seventeen |
| the error weights in shared memory | removes a spill but frees **no** registers; neutral |
| `--queue-depth 4`, `--gpu-workers 2`, `--batch-frames 32` | all within 1% on 18001-frame content, not just the bench clip |
| `--palette-frames 60` / `100` | 22.4% -> 24.5% -> 23.2% mean saturation; the low saturation is the imagery, not the sampling |
| `--input-mode yuv420` (round twenty-two) | decoder **2.3x faster**, total CPU **unchanged**, wall within noise either way. 89% of CPU is the encoder |

**Done, and it was on this list for the wrong reason.** The remaining change worth more
than a few percent was always about *bytes on the input pipe*, and the stated blocker
was bit-exactness of swscale's arithmetic. The matrix turned out to be exactly
reproducible (round seventeen), but it was not what was actually in the way -- a plane
stride was, and the option was silently producing garbage while being measured. Decode
fell 16.4 s -> 5.6 s. The remaining item on this list is now the *output* pipe.

For 2-3 hour 1080p footage, ~49 fps means roughly 6.1 minutes per 18000 frames.

### Round twenty-four: audio, more palette samples, and a plugin seam

Three changes, each of which changed a default.

**Audio is now carried through.** Stream copy, so it costs nothing: `-map 0:v -map 1:a?
-c:a copy -shortest`, with the source as a second `-i`.  The `-map` is not decoration --
without it ffmpeg's default stream selection picks by type ranking, which is how a cover
art or data track ends up as the video.  Verified by `volumedetect`: mean -21.1 dB and
max -13.0 dB on both sides, so the content is copied and not merely a stream header.  A
silent source reports it and writes video only; `--no-audio` is the flag rather than
`--audio` so no existing command line changes.

**The palette samples as many frames as a 60 s budget allows, up to 256.**  The old
default of 30 came from the reference pipeline, which quantises 30 *named* files.  Here
the 30 were a number of equally spaced *points*, and on a 3-hour clip all 30 can land in
the dull stretches.  Reproduced deliberately: a clip that is grey for 1750 of 1801 frames
gives **0.0% mean saturation at 30 samples and 41.6% at 60** -- the colour is simply not
in the palette, because 30 evenly spaced samples stop at frame 1740 and the colour starts
at 1750.

A budget rather than a count, because the cost is one ffmpeg spawn plus one seek per
sample (measured 0.12 s, six at a time), so 60 s buys about 500 samples and a slow disk
should take fewer rather than take longer.

`--palette-max-samples` defaulted to 64, which was a category error: 64 was set for the
*pooled* palette path, where each sample costs a full-resolution quantize (~1.0 s), and
it was silently capping the *montage* path, where each costs 0.12 s.  Now 256.  The
`'all' reduced to N` message also lied -- it named the montage area cap when
`palette_max_samples` was the binding limit.

Note the plateau in the table above: 22.4% at 30, 24.2% at 128, 23.9% at 256.  Past ~60
samples the answer stops changing on grey-heavy material.  More samples cannot help once
the colour has been seen; if the reported saturation is still low, the imagery is grey.

**A plugin seam, so the program is not only Riemersma.**  `include/rd_plugin.h` defines a
dither interface and a registry; `--dither NAME` selects; `examples/bayer_dither.cc` is a
complete working second algorithm in ~40 lines, in the default build so the seam is
exercised rather than theoretical.  A typo'd name fails loudly, because silently running
Riemersma looks exactly like "the plugin did nothing".

Registration is at compile time on purpose.  A `LoadLibrary` plugin has to agree with the
host on struct layout, calling convention and allocator, and this is built with a specific
MSVC + CUDA against a specific ImageMagick -- so that is a memory-corruption class of
problem dressed as a feature.

The trap, now written into both files: `Palette` is 2 MiB (`entries[65536]` of four
doubles), so `const Palette pal = *job.palette;` puts 2 MiB on the stack and the process
dies with `STATUS_STACK_OVERFLOW` before touching a pixel, with no message mentioning
memory.  Bind by reference.

### HEVC, measured rather than assumed

Already worked as *input* with no code change -- the decoder hands the file to ffmpeg and
ffmpeg picks the decoder.  Verified end to end, 300/300 frames, and NVDEC handles it too.
(The 299/300 I first measured was an ffprobe race, not a defect: `-count_frames` was read
before the encoder process had finished writing.)

As *output*, `--video-codec libx265` works and is measurably *worse* here: encode goes
4.3 s -> 16.1 s for 300 frames, 3.8x, for 12% smaller files (19.0 MB -> 16.8 MB).  Since
x264 is ~89% of all CPU, that is a large net loss.  It also preserves the palette *worse*,
not better: 1344 unique colours against x264's 832 in frame 100, i.e. x265 invents
*more*.  Not recommended.

### OpenCL: present, unused

Works on this machine (OpenCL 3.0, "NVIDIA CUDA" platform, GTX 1650 SUPER) with nothing
installed, because Windows ships the ICD loader and the driver supplies the vendor
library.  The registry key most guides tell you to check,
`HKLM\SOFTWARE\Khronos\OpenCL\Vendors`, is *empty* and OpenCL still works -- modern
drivers register through the loader's own enumeration.  `tools/probe-opencl.exe` asks the
loader, not the registry.

No engine yet, deliberately.  The arithmetic is portable (OpenCL's default is
correctly-rounded double, which is exactly what `__dadd_rn`/`__dmul_rn` exist to pin down
in CUDA), but the port is the whole of `rd_cuda_common.cuh` plus the walk, the gather and
the scatter -- and this codebase's record is that two separate versions of that work
passed 135/135 while being wrong in ways only a decoded A/B caught.  A bit-exactness
claim that has not been verified end to end against ImageMagick is not a claim.  Written
up in `docs/OPENCL.md`.
