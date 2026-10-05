# OpenCL: the engine, and how it was verified

## Status

`--engine opencl` runs images and video, and on **both opaque and alpha** input it
is bit-identical to `--engine blocks` (CUDA) everywhere this project can measure:

| | probe | result |
|---|---|---|
| images, 36 cells | `tools\probe-opencl-exact.ps1` | 36/36 identical |
| images, 135 cases vs ImageMagick | `verify.ps1` | 135/135 |
| run-to-run stability | `tools\probe-determinism.ps1` | 6/6, 12 runs each |
| video, 60 frames of 1080p lossless, **both** data paths | `tools\probe-video-exact.ps1` | 2/2 identical — `rgba64le` and `yuv444p` |
| the `rgba64le` data path | `RD_OCL_CHECK_U16=1` | 36/36 exact against the float path |

This was not true a few hours ago, and an earlier version of this file said
"bit-identical on opaque input only" while carrying a long account of an open bug
that was blamed on this engine. The bug was real; the blame was not. It was in the
image writer, and it is fixed. The full account is below, because the reasoning that
pointed at the wrong component is the interesting part — as is the fact that the
instrument, not the code, was wrong four separate times during this work.

Bit-identical is the correct bar here, and it is a bar this project can actually
test. `rd_riemersma.h` specifies `RiemersmaBlocksCpu` as having "the same partition
and the same arithmetic as RiemersmaBlocksCuda", so the blocks engine is defined by
a property that two independent implementations can be compared on. A new
accelerator port is either bit-identical or it is wrong; "close" is not available.

## The bug that was not this engine's, and how it was actually found

### The wrong answer, held confidently for several hours

The engine appeared to produce different output between runs on alpha input: 3 distinct
results over 25 runs, against 1 over 25 for CUDA and 1 over 12 for the CPU engine, on
the same image. Every hypothesis that could be tested from the host side had been
tested and eliminated — queue ordering (`clFinish` barriers, kept in the code because
they are correct), out-of-range scatter writes (a bounds check, also kept), palette
size, uninitialised buffers (a poison fill, `RD_OCL_POISON`, which left opaque output
bit-identical and did not shrink the alpha spread at all, proving nothing was read
before it was written).

That left a device-side miscompilation of the FP64 error-queue kernel as the standing
suspect, untestable on a machine with one OpenCL vendor. An engine guard was added:
`RiemersmaBlocksOpencl` **refused alpha input** rather than emit output it could not
reproduce.

### The mistake

The measurement instrument was the problem, and it was the problem from the first
table in that account. Every determinism check hashed the **rendered file's decoded
RGBA**. That makes the writer part of the measurement. Chasing the engine, and the
engine's device memory, while the file was the thing being hashed could only ever have
found a fault somewhere; it found a real one, just not the one that was being looked
for.

### What settled it

Dump the buffers. `RD_OCL_DUMP=<prefix>` writes the device and host buffers back to
the host as raw bytes; `RD_STORE_DUMP=<prefix>` does the same for the `PixelStore` as
`ImStore` receives it. Over 16 runs of a 96x64 greyscale+alpha image:

| buffer | distinct hashes / 16 runs |
|---|---|
| `cx` — gather output | 1 |
| `idx` — the walk's chosen palette indices | 1 |
| `owner` — the scatter's owner table | 1 |
| `pal` — the palette as the device sees it | 1 |
| `pix` — device pixels after the scatter | 1 |
| `host` — readback into `batch`, what the writer receives | 1 (byte-identical to `pix`) |
| store, on entry to `ImStore` | 1 |
| store, on exit from `ImStore` | 1 |
| image colorspace, alpha trait, channel count | 1 |
| **the PNG file, hashed as bytes** | **14** |
| **the PNG file, decoded to RGBA** | 7–8 |

The engine was deterministic the whole time, and correct: against CUDA on that image,
0 of 24576 decoded bytes differ. The store was intact when it reached the writer. The
image handed to `WriteImage` had identical properties every run. And the output file
differed **every single run** — and not only in its pixels:

```
run  1:  Colorspace sRGB   Type PaletteAlpha   Depth 4/8-bit
run  2:  Colorspace Gray   Type Bilevel        Depth 1-bit
```

ImageMagick was choosing a different *encoding* each time, from identical input.

### The cause

`ImStore` wrote R, G, B and A per pixel and nothing else — but `GetPixelChannels`
reports **5** on this image. The fifth channel was never assigned, so it held whatever
that heap block held before. The PNG coder's type optimisation scans the full channel
set, so the encoder's choice of TrueColorAlpha / PaletteAlpha / Gray / Bilevel
depended on uninitialised memory. Which is also why the GPU engines hit it far more
often than the CPU engine: their extra host allocations rearrange the heap, so the
stale value differs. It was never engine-specific in any meaningful sense — the CPU
engine simply got lucky, at 1-in-16 rather than 3-in-16.

The fix is one `memset` per row in `ImStore`, zeroing the whole row before filling the
four channels the dither actually produces.

After it, 20 runs each of `opencl`, `blocks` and `cpu`, on alpha and on opaque input:
**1 distinct result each**, and OpenCL still byte-identical to CUDA.

### What was kept, and what was thrown away

Kept, because each is correct and each cost an assumption:

- the `clFinish` barriers between gather, walk and scatter;
- the scatter's bounds check on the curve index;
- `RD_OCL_DUMP` and `RD_STORE_DUMP`, which are how any of this is checkable again.

Thrown away: the alpha refusal, and `RD_OCL_POISON`. The refusal blocked a working
path on a phantom. The poison fill answered its question — nothing is read before it
is written — and keeping a branch on buffer creation alive to re-ask a settled
question is a cost with no remaining benefit.

### The test that would have caught it

`tools\probe-determinism.ps1` runs the same command 12 times per case and requires one
distinct decoded pixel set. It exists because a test that compares two
implementations **cannot** see a fault they share — OpenCL and CUDA were both wrong
in the same way at the same time, so the cross-engine sweep was blind to it and only
caught it by running enough cells for the coin flips to disagree. Comparing an
implementation against itself across time is the only thing that sees this class.

It reports pixel hashes and file hashes separately, so a failure says which moved.
A file hash that varies while pixels do not is a different bug from pixels varying,
and the two were conflated for most of the investigation above.

"File hash" there means the file *after* `-strip`. ImageMagick writes a `png:tIME`
chunk, so raw output bytes differ between any two runs a second apart even when every
pixel matches — measured here as 5–11 distinct raw files per 12 runs, with 1 distinct
pixel set and 1 distinct stripped file. That is PNG recording when it was written, and
failing a test on it would be failing the test for being right. `png:exclude-chunk=
date,time` suppresses it if byte-reproducible files are ever wanted; the default is
left alone, because matching ImageMagick's *pixels* is the stated bar.

The lesson worth keeping: *"identical input, different output"* is a claim about a
measurement, and a measurement that passes through a file format is measuring the
writer too. When a bug appears to live in an accelerator, dump the buffers before
concluding it does — the GPU is where you are expected to look, and that is exactly
why looking there is expensive.

Single-image is not the whole story: the engine also runs the video pipeline
(`--video --engine opencl`), and is bit-identical to CUDA there too. What carries
over is the `rgba64le` data path; what does not is the planar-YUV gathers.

## Video: the rgba64le data path

`--engine opencl --video --input-mode rgba64` with `RD_YUV444_OUT=0` runs the
whole pipeline — reader, batcher, GPU workers, encoder — on OpenCL. Measured on
`tests\clip1920.mp4`, 60 frames of 1920x1080, lossless, both engines through the
same palette and the same 16-frame batches:

```
decoded RGBA   497.7 MB   (60 frames x 1920x1080 x 4)
blocks  sha256 FB9E83D42F4B2279F2DC2DAABCA8E296C252B201557B4ED3A28D6D241EA697ED
opencl  sha256 FB9E83D42F4B2279F2DC2DAABCA8E296C252B201557B4ED3A28D6D241EA697ED
```

Identical, not "almost frame exact". That is a stronger result than the video bar
normally asks for, and it is expected: the two engines are defined to have the same
partition and the same arithmetic, and the block partition is per-frame independent,
so a frame cannot differ without the walk differing.

### What the video port added

**A device-state cache.** One context and device, shared; one command queue, program
and kernel set per slot, lazily built under a mutex. Two reasons the kernels are
per-slot and not shared: `clSetKernelArg` *mutates* the kernel object, so two
threads setting arguments on one kernel is a data race, and sharing the program is
the natural next saving and exactly the wrong one. Once built, a slot runs with no
lock held at all — every call left on the hot path is thread-safe in OpenCL 1.2, and
the one that is not (`clCreateCommandQueue`) only ever happens under the mutex.

**`gather_u16` and `scatter_u16`.** rgba64le in, rgba64le out — 8 bytes per pixel
against float4's 16, in both directions, which deletes the host's uint16→float
widening (the reader's dominant cost, ~48 GB of traffic over 605 frames) and its
float→uint16 conversion (measured at 3649 ms), and halves the H2D and the download.

The scatter's cast is the one thing that must not be simplified:

```c
dst[..] = (ushort)(float)e[0];      // correct: double -> float -> truncating uint16
dst[..] = (ushort)e[0];            // WRONG: double -> uint16, one rounding fewer
```

CUDA spells it `static_cast<std::uint16_t>(static_cast<float>(p[i]))` and so must
this. The one-step form is a different operation — the `double→float` step rounds,
and the truncation then lands on a different value — and it differs from CUDA on
exactly the samples where that rounding crosses an integer boundary: a silent,
scattered, last-bit divergence visible only as compression noise in an encoded file.

### How the data path is verified

`RD_OCL_CHECK_U16=1` runs the pipeline twice over the same input and diffs the
results: 36/36 cases exact (six images × three palette sizes × two block lengths).
The float path is the reference because it is *already* verified bit-identical to
CUDA on all 36 cells, so the check needs no new fixture and no new reference.

Getting that check right took three attempts, and the failures are the useful part,
because all three reported a *kernel* bug and all three were the instrument:

1. It converted pass one's **output** to uint16 and dithered that. That is dithering
   an already-dithered image — two unrelated computations. Thousands of differences,
   none of them real.
2. It compared the uint16 path against the float path on the image's own pixels. An
   image loaded from PNG is HDRI float, and its samples are generally not integers:
   8-bit sRGB 128 arrives as 32896.06, which rgba64le cannot represent. Lossless
   against lossy.
3. It read `want16` from `batch` before the readback that puts pass one's result
   there, so it compared pass two's output against the **input** — differing on
   every pixel the dither actually changed. 18432 differences on a case whose
   kernels are correct.

The lesson is the same one as the PNG-hashing mistake at the top of this file, and
it is worth stating once: *an instrument can be the thing that varies.* Three
consecutive alarming results were all the check being wrong, and the one real bug it
finally found was a single pixel per frame.

### The bug it did find

`Riemersma()` visits `4^L - 1` cells and the trailing `ForgetGravity` visit lands on
the first one again, so the grid's **last cell is never dithered** — which is why a
1920x1080 frame at level 11 leaves exactly one pixel at its source value, and why
ImageMagick itself does.

The float path gets that for free: its buffer is `COPY_HOST_PTR` from the caller's
frame, so the unvisited pixel still holds its source. A freshly allocated uint16
buffer holds whatever the driver left there, and that goes straight into the encoder.
One pixel in 2,073,600 — invisible, which is why it survived — but uninitialised
memory, so the output is not reproducible.

Fixed here by copying the source into the output buffer before the scatter runs, so
the uint16 path's unvisited pixel agrees with the float path's exactly.

**The CUDA engine has the same exposure**: `d_u16_buf` is a bare `cudaMalloc` and its
scatter writes only owned pixels. Not fixed as part of this change, because that
engine is verified bit-exact and changing its output — even for one pixel, even
toward correctness — deserves its own before/after measurement. It is a one-line
`cudaMemsetAsync` or a `cudaMemcpyAsync` from the input, and it should be a
deliberate decision rather than a drive-by.

## Planar 4:4:4: ported. Planar 4:2:0 and the prepass: still refused.

`--input-mode yuv444` and the planar 4:4:4 **output** kernel are now ported, and are
bit-identical to CUDA -- verified per pixel on 60 frames of 1080p, alongside the
`rgba64le` path, in `tools\probe-video-exact.ps1`, which now sweeps both.

They are `gather_yuv444` and `scatter_yuv444` in `rd_opencl.cpp`, transliterations of
`BlkGatherYuv444Kernel` and `BlkScatterYuv444Kernel` plus `d_rgb_to_yuv444` in
`rd_blocks_cuda.cu`. The swscale constants were **copied, not derived** -- see the note
on `sws_yuv_to_rgb16`, and `tools/probe_swscale_matrix.cpp`, which pins them against
ffmpeg's own output. They compiled and matched bit for bit on the first run, which is
what copying buys and what deriving would not have.

Two things that are easy to get wrong and are commented at the definitions:

  * **Three planes per frame, so frame f's luma starts at f * 3 * npix.** Using
    f * npix reads frame f-1's U and V as if they were frame f's luma -- correct for
    frame 0, nonsense for every frame after, and invisible because frame 0 is all
    anyone looked at.
  * **The output is pre-filled from the input** with a straight device-to-device copy
    before the scatter runs, so the pixel the curve never visits keeps its source
    value. On this path that is free: a planar 4:4:4 source and a planar 4:4:4
    output have the same layout, so the source's own three bytes are the right
    answer. This is the same fault the CUDA engine had and fixed separately -- see
    `BlkFillUnvisitedKernel`.

`--input-mode yuv420` and `yuv444-prepass` are **still refused with a message**, for
the original reason: each is a bit-exact transliteration of a specific libswscale
routine, and quietly reading rgba64le instead would produce a plausible picture that
came from a different decoder, with the difference attributed to the dither. The
refusal is checked at parse time, before the palette stage runs, and again in the
engine as a backstop for a direct caller.

### What porting 4:4:4 was worth

This was the whole of the OpenCL video gap, and getting the number right mattered more
than the code, because the gap had been attributed to the engine. 600 frames of 1080p60,
3 interleaved runs each, on the machine this was written on (Xeon E5-2620 v3, GTX 1650
SUPER):

| | CUDA | OpenCL | ratio |
|---|---|---|---|
| **before** -- CUDA on yuv444p, OpenCL confined to rgba64le | 14.2 s | 31.3 s | **2.20x** |
| both forced onto `rgba64le`, to isolate the engine | 32.8 s | 35.5 s | **1.08x** |
| **after** -- both on yuv444p, 3 B/px | 15.6 s | 19.6 s | **1.26x** |

The middle row is the one that matters: with the data path held equal, the engines were
already within 8% of each other. The entire 2.2x was bytes on the wire -- 8 per pixel
against 3. Per stage, from `RD_OCL_TIMING=1` over 38 batches, OpenCL's three kernels

CORRECTION: there is no `RD_OCL_TIMING` in this codebase. It appears nowhere under
`src/` or `include/` -- the only occurrence anywhere is the citation itself and a
comment in `tools/probe-opencl.cpp`. So the per-stage comparison below was not produced
by a flag a reader can set, and the 6410 ms / 6124 ms pair cannot be reproduced or
checked by anyone following this document.

What DOES exist now, added after that text was written: `RD_OCL_IO=1`, which prints an
`[io]` line with five separate byte buckets and a conservation assertion comparing the
measured H2D total against `width x height x frames x bytes_per_pixel` computed from the
run's own inputs. That checks the byte volume exactly and is immune to the run-to-run
drift documented elsewhere -- which is the property a per-stage millisecond comparison
lacked. It does not break the stage down the way the quoted numbers do, so this
measurement is recorded as unreproducible rather than replaced.
*plus* its readback came to 6410 ms against CUDA's 6124 ms of whole-batch dither+IO, so
the walk was never the problem.

The `1.26x` rather than `1.08x` is real, and is the price of the new path: the planar
kernels do the YCbCr<->RGB conversion on the device and OpenCL's is a little less
efficient than CUDA's. Run-to-run spread on this machine is around 15%, so read that as
1.2-1.3x rather than a precise figure.

Reach remains the stronger argument for OpenCL anyway: on a part with weak
double-precision, the walk -- which is where essentially all the work is -- is the
operation the silicon is worst at.


## Rotation: the bug that was not VFR either

Reported as "it wasn't VFR — probably the rotation tag". It was.

### The bug

ffmpeg's rawvideo output **auto-rotates by default**, and rdither's decoder never
passed `-noautorotate`. So for a stream carrying a 90 or 270 degree display matrix —
every portrait phone video — the decoder emitted **1080x1920** frames while rdither
sized every buffer from `width`/`height` in the stream header, which are the **coded**
**1920x1080**.

The pixel *count* is identical (2073600 either way), so **no bounds check can detect
it**. The only thing wrong is the row width, so the frame is read back sliced
differently and the picture comes out sheared. The output also carried no rotation tag,
since the pixels were re-encoded from raw and nothing copied the side data.

Measured on a HONOR/HUAWEI phone capture tagged `rotation=-90`: the output was the
1080x1920 portrait image written into a 1920x1080 buffer. Reinterpreting the output
bytes at the width the decoder actually used halved the error against the source
(0.074 versus 0.153 RMSE), which is what identified it.

After the fix, the same 5 s excerpt:

```
[video] source carries a -90 degree display matrix; the decoder rotates, so the
frames are processed and written as 1080x1920 upright with no rotation tag.
```

| reading of the output | RMSE vs source |
|---|---|
| upright (1080x1920) | **0.037** — the dither, and nothing else |
| landscape (1920x1080) | 0.297 — eight times worse |

### Why rotate in rather than tag through

The alternative is `-noautorotate` on the decoder plus a copy of the display matrix to
the output, which keeps the pixels as stored. That is more faithful and it is not
available: **ffmpeg 9 exposes `-display_rotation` as an input option only**, and
`-metadata:s:v:0 rotate=90` was removed — applying the input option to the output fails
with "you are trying to apply an input option to an output file". Rotating the pixels
in is therefore the only route that works in every container, and it needs no output
tag at all, because the frame is already upright.

The cost is that the dither runs on upright pixels rather than as-stored ones, so the
error-diffusion direction is rotated relative to ImageMagick's own pipeline. That
changes which pixels differ, not whether the output is a correct 16-colour dither, and
it applies identically to every engine so cross-engine comparison is unaffected.

### A probe that fails quietly

Two wrong forms of the ffprobe query were tried before the right one, and both are
worse than an outright error:

- `side_data=rotation` (space-separated, unqualified) is **accepted, matches nothing,
  and returns the stream fields with rotation silently absent** — exit 0, no warning,
  every rotated input then processed at the wrong geometry.
- `stream_side_data=rotation` without a colon re-emits the whole stream block instead
  of the side data.

The working form needs a **colon** between the sections:

```
-show_entries stream=width,height,...:stream_side_data=rotation
```

A probe that exits 0 while quietly omitting the field it exists to fetch is the worst
failure mode available. Check that a new probe field actually appears in the output
rather than assuming the query worked.

The alpha branch is the one the original 36-case sweep never executed, and adding it
is what found all of the above. That is the argument for keeping it in the suite: the
case that finds the bug is the case the coverage was missing.

## Variable frame rate: the real bug was not the rate

Reported as "VFR does not render correctly". It took two rounds to find, because the
first fix was correct and did not address what the reporter could see.

### Round 1: the rate, which was a real bug

The encoder is fed `-r <info.fps>`, and `info.fps` comes from `r_frame_rate`. For a
constant-rate source that is the rate. For a variable one, `r_frame_rate` is the
**maximum instantaneous** rate -- a container hint, not a summary. Every frame in
the output got that duration, so the output was the wrong length and, with the audio
stream-copied at its own timing, A/V desynced by the same factor. Measured on a
synthetic VFR fixture (three concatenated segments at 30, 12 and 60 fps, 102 frames):

| | frames | timeline | avg rate |
|---|---|---|---|
| source | 102 | 3.0333 s | 3060/91 = 33.6 |
| before | 102 | **1.632 s** | 60/1 |
| after | 102 | **3.004 s** | 23841/709 = 33.63 |

1.86x too short before. The fix is to measure the source's frame timings and re-emit
at the true average (`frames / total_duration`). Constant-rate inputs are unaffected --
the two numbers are identical -- and the measurement is a demux pass over the
container's packet index, not a decode.

### Round 2: the reporter's own file, and the bug that was left

Tested against a real phone capture (HONOR/HUAWEI `mcpro24fps`, 2 min 52 s, "Frame
rate mode: Variable", min 29.713 / max 29.890 fps), the rate fix was already working --
VFR detected, 29.803 fps average, output genuinely constant rate. **It was still
wrong**, and the reason had nothing to do with frame rate.

`-shortest` was in the encoder command. It stops output at the shorter of the two
streams, which is the right tool in exactly one direction:

- **Audio longer than the video** -- the trailing audio is what should go. This is
  what the flag was originally added for.

- **Video longer than the audio** -- `-shortest` **deletes video frames** to make the
  tracks match.

The second case is the one that occurs in practice, and the loss scales with the clip.
A 30 s excerpt: source 895 video frames against 30.0128 s of audio, output **892**.
Three frames deleted, every run. `--no-audio`, which removes `-shortest` from the
command entirely, gave 895.

The asymmetry is not subtle. Trailing video past trailing audio is handled by every
player and costs nothing; deleted frames are frames the user asked to have dithered,
gone. So `-shortest` is now used **only where it trims audio**, decided by comparing
the measured video and audio durations, and rdither says so when it declines:

    [video] source video is 30.0308 s but its audio is only 30.0128 s; not using
    -shortest, which would delete 0.5 video frames to match.

After the fix, the same excerpt: **895 frames in, 895 frames out**, timeline 30.0308 s
against the source's 30.0304 s, and a single PTS delta -- constant rate, as asked.

### The lesson

The first fix was measured, correct, and insufficient, and it was reported as done.
The reporter's file was nothing like the fixture: theirs is a mild three-valued
jitter around 30 fps from a phone encoder, mine contained a 0.57-second frame. A
fixture that differs from the real input in *kind* rather than degree will hide a
second bug behind the first one's fix, and the way out is the reporter's own file
rather than a better guess at a synthetic one.

### Per-frame cadence: still flattened, and why that is a separate question

The output is constant rate at the source's true average, which is what "render at
constant fps" asks for and what duration and A/V sync require. The *per-frame* motion
cadence is flattened, so a variable-rate capture is resampled once. Restoring it needs
a piecewise `setpts` re-applied at the encoder, because a rawvideo pipe carries no
timestamps:

    settb=expr=1/<timescale>,setpts='<piecewise PTS as a function of N>'/TB

plus `-fps_mode passthrough -video_track_timescale <t>`. That is **proven in
isolation** -- 102 frames through a raw pipe come out with PTS running 0 -> 3.017 s
against a 3.0333 s source -- but **not end to end**, because no genuinely
variable-timestamp *fixture* could be built on this machine: lavfi, `setpts`, concat,
ffv1/matroska and libx264/mp4 all came back constant-rate. It ships as
`--video-preserve-vfr`, **off**. Shipping an unverified path as the default is how the
next person loses a day.

Three of the four traps involved fail *silently*:

- **The timebase.** `setpts` works in the input's timebase, which for rawvideo is
  `1/declared-rate`. At 1/60 s, frames 65 us apart collapse onto one tick and ffmpeg
  drops them: 102 in, 100 out, timeline quietly short. `settb` must come first.
- **The expression must be quoted.** `-vf` splits on commas as filter separators, so
  `if(lt(N,29),...)` is parsed as a filter named `29)`.
- **Measure PTS, not durations.** Every frame's `duration` field stays at the input
  rate no matter what `setpts` does. Summing packet durations reported a 1.6 s
  timeline for a 50.5 s one, and produced two entirely wrong conclusions before the
  PTS were read directly. That field is not a reliable measurement of anything once a
  filter has run. A fourth trap, specific to h264 output: packet PTS go up and down
  because that is B-frame *decode* order, so a delta analysis over unsorted PTS is
  meaningless. Sort first.
- **The open one:** whether the muxer honours the reassigned durations at all. In the
  isolated pipe test the PTS survived; in every encoder-side fixture attempt they were
  flattened back to the input rate. That needs a VFR fixture that can be produced.

## The three places the dialect bit

- `__dadd_rn` / `__dmul_rn` become `+` and `*`. OpenCL C's default rounding for
  double is correctly rounded and the default build does not contract, so these are
  the same operations — *provided* `-cl-fast-relaxed-math` stays off.
- `-cl-mad-disable` was tried and deliberately **not** used: the NVIDIA driver
  rejects the option outright ("Don't understand command line argument"), which
  aborts every build rather than protecting one. Contraction is left to the
  default. **This is a stated residual risk**: a driver that contracts double
  arithmetic by default would differ from CUDA on the near-tie pixels and nowhere
  else, which is the signature of a 1-ULP disagreement. It is the first thing to
  try on a machine that reports a small scattered difference, and the program
  build log is captured (`OpenCLBuildLog()`) precisely so a driver's behaviour is
  visible rather than inferred.
- `local` is a **reserved address-space keyword** in OpenCL C. `const int local = ...`
  is a parse error. The CUDA original's `local` is renamed `blk`.

The address-space qualifier is also part of the pointer *type*, so a helper taking a
buffer must write `__global` on the parameter. In CUDA a bare `const T*` just works;
here it does not compile.

## Why the build log exists

`OpenCLBuildLog()` returns the driver's `CL_PROGRAM_BUILD_LOG` unconditionally, and
the CLI prints it on failure. "The driver rejected the source" and "the driver
accepted the source and compiled it into something that computes a different
answer" look identical from outside a driver that does not tell you. Four of the
compile errors above were only findable because the log was there.

## Remaining scope, honestly

- **The planar-YUV gathers.** `--input-mode yuv420` / `yuv444` / `yuv444-prepass` and
  the planar 4:4:4 output kernel are refused with a message. Each is a bit-exact
  transliteration of a specific libswscale routine on the CUDA side, so this is real
  work rather than a flag — but it is mechanical, and the shape is already proven by
  the `rgba64le` path. Until it is done, `--engine opencl --video` needs
  `--input-mode rgba64` and `RD_YUV444_OUT=0`.
- **Palettes above 16 colours.** The port carries the flattened search index only.
  CUDA falls back to the recursive `ClosestColor` when a palette does not fit the
  4-bit nibble packing; this engine refuses with a message instead, because
  silently computing something else would redefine what `--engine` means.
- **Page-locked host memory has no OpenCL equivalent.** The video pipeline's
  zero-copy upload does not transfer, so the `rgba64le` path pays a staging memcpy
  that the CUDA path does not. It is still a net win — it deletes the host's
  uint16↔float conversions, which are the larger cost — but it is not the same win.
- **Performance is measured on one machine, and only on that machine.** The
  before/after table in the planar 4:4:4 section above is the real figure: **1.26x**
  of CUDA's wall clock, both engines on `yuv444p`, 600 frames of 1080p60, 3
  interleaved runs. It is quoted as "1.2-1.3x" in the README because a single
  machine cannot support three digits. An earlier version of this bullet said
  performance was unknown and quoted a 10-frame 160x90 run; that clip is almost
  entirely process start-up and driver init, so the number was meaningless in both
  directions, and the bullet predated the steady-state measurement. What remains
  true is the scope limit: sub-5% differences on this machine sit inside the fps
  noise, and the figure says nothing at all about a non-NVIDIA device - see
  "Intel and AMD" above.
- **Multi-frame batching** (`--frames N`) is plumbed through `options.frames` and is
  exercised by the video path above (16-frame batches, 60 frames of 1080p), but
  `--frames N` on the *image* path has not been compared against CUDA.

## Intel and AMD: unmeasured, and why that is the whole question

Everything above was measured on **one machine, with an NVIDIA driver**. So the
correctness result transfers — the arithmetic is the same kernels and the same
partition on any conformant device — but the *performance* result does not
transfer at all, and performance is the only reason to use this engine on a
non-NVIDIA card.

The cost centre is FP64. The Riemersma error queue carries the error as a double,
so the walk is double-precision throughout, and a consumer GPU's FP64 rate is
usually **1/32 or 1/64** of its FP32 rate. That ratio is exactly where NVIDIA,
AMD and Intel differ most, and it is not a detail: it can turn a large win into a
large loss. So "OpenCL is 1.2-1.3x of CUDA" is a true statement about an NVIDIA
driver and says nothing about an iGPU.

Two things make this cheap to answer rather than a research project:

- The engine is **bit-identical to CUDA on every cell measured** (36/36 images, 2/2
  video data paths). There is no porting work to do first. If your card is fast
  enough, it is already correct.
- `tools\probe-opencl.exe` reports your devices and their `cl_khr_fp64` support
  with no SDK and no build, in under a second.

What would decide it, and has not been done here: run `tools\probe-opencl-exact.ps1`
on the machine in question, which reports both correctness and per-engine timings
per cell, then time a real clip with each engine on the same data path. The
comparison is only meaningful on the same path — see the rgba64le section above
for the 2.67x that was pure data-path artefact.

Until someone runs that, the honest statement is: **the OpenCL engine is verified
correct and is the supported route to a non-NVIDIA GPU; it has not been shown to be
faster than CPU on any of them.**

## Checking any machine

`tools\probe-opencl.exe` reports platforms and devices with no SDK and no
`windows.h`. It is the right first thing to run on a machine where `--engine
opencl` says it found nothing.

Note the trap: `HKLM\SOFTWARE\Khronos\OpenCL\Vendors` is **empty on this machine**
and OpenCL works anyway. Modern drivers register through the loader's own
enumeration, so an empty vendors key does not mean no OpenCL. Ask the loader.

