# rdither

GPU-accelerated reimplementation of ImageMagick's Riemersma error-diffusion dither,
for images and video. Bit-exact with ImageMagick where that is possible, and
deliberately approximate where it is not — [see exactly which is which](#where-bit-exactness-holds).

> ### Made by an AI — "Space Bunny"
>
> This repository was written by an AI assistant operating under the name **Space
> Bunny**, working with the project owner. There is no human author on the commits.
>
> What that means for you, stated plainly rather than buried:
>
> - **The measurements are real.** Every number in this README and in `docs/DESIGN.md`
>   came from running the code on the machine it was written on. None of it is
>   estimated or illustrative.
> - **The verification is real too, and it is the point.** 135 bit-exact cases against
>   ImageMagick, a 54-cell cross-engine bit-exactness sweep, a run-to-run determinism
>   probe, and a video frame-accounting check. See [Correctness](#correctness).
> - **Expect some things to be wrong.** An AI will confidently report work as finished
>   before the check that would prove it has run. That happened repeatedly during
>   development — including a video bug that was reported fixed twice while a second,
>   unrelated fault was still present. The commits and the issue history are the
>   record; treat anything surprising as a bug report rather than a question of
>   intent.
>
> You are welcome to use it, audit it, or throw it away. If you find a fault, the
> useful thing to say is *what you measured*, not *what you expected*.

## Licence

**GPL-3.0-or-later.** Full text in [LICENSE](LICENSE), verbatim as published by the
Free Software Foundation.

    Copyright (c) 2026 Gittanley, prompt operator
    SPDX-License-Identifier: GPL-3.0-or-later

All source code here was written by an AI assistant operating under the name
**"Space Bunny"**, directed by the prompt operator — see the note at the top of this
file, and the SPDX identifier in the header of every source file. The copyright is the
prompt operator's because a purely AI-generated work has no copyright owner in most
jurisdictions, and the human here supplied the goals, the bug diagnoses, the rejected
approaches and the decisions about what ships. `LICENSE` itself is kept exactly as
gnu.org publishes it, unaltered, because that is the form licence-detection tooling
expects.

You may use, study, modify and redistribute this freely, and **anything you build on
it must also be free and open source** — that is what copyleft means here, and it is
the part most worth having. There is no fee, no registration, and no requirement that
you credit me.

Two things it deliberately does *not* do, because a licence that did would not be open
source:

- **It does not stop you using the finished tool commercially.** If you need to forbid
  that, you need a proprietary licence, and you lose the open-source status. The GPL
  trades that off on purpose.
- **It does not stop proprietary *tools* from touching the code.** Compiling it with
  NVIDIA's CUDA toolkit, or linking against ImageMagick, does not make your build a
  derivative. Only redistributing the result would.

The patent grant in GPL-3.0 is why I chose it over GPL-2.0: you cannot be sued for
using this.

If you point it at an image with the sequential walk and ask for 16 colours, you get
back the same 16 colours ImageMagick would have produced, the same pixels, byte for
byte — not "similar", not "close enough to pass". That is the design goal, and it is
checked on every build.

```
rdither --colors 16 photo.png out.png
```

**That guarantee is narrower than it sounds, and the rest of this file is where it
gets precise.** The fast GPU engines and the video pipeline are *not* bit-exact with
ImageMagick, by construction. See [Where bit-exactness holds](#where-bit-exactness-holds)
before you judge any output.

Decode, dither and encode all run at once, at about 51 fps on a 2014-era 6-core Xeon
with a GTX 1650 SUPER. So 5 minutes of **60 fps** footage took 5 minutes 53 seconds —
about 1.17x its own duration, while 30 fps footage would come out faster than real
time. What that number does *not* measure is the GPU —
[two thirds of it is decode and encode](#what-the-fps-number-means-in-practice), so a
faster CPU, RAM or disk makes it faster with the same GPU.

---

## Why it exists

ImageMagick's Riemersma dither is slow enough that using it on a long video is not
practical. Reimplementing it naively is also a trap: the algorithm is *not* what makes
it slow, the palette lookup is, and the obvious ways to speed that up change the output.
Almost every optimisation attempted here either made no difference or quietly changed
the image. The ones that worked are in [docs/DESIGN.md](docs/DESIGN.md), and so is the
list of ones that did not, which is the more useful half.

The result is a program that is fast because of what was measured rather than what was
assumed, and that tells you when it is guessing.

---

## Install

You need three things:

| | |
|---|---|
| **Visual Studio 2022** | any edition, with the "Desktop development with C++" workload |
| **CUDA Toolkit** | 12.x or 13.x, plus an NVIDIA driver. Only for the GPU engines |
| **ImageMagick 7 Q16-HDRI** | the **MSVC** build (`...-dll.exe` or `-static.exe`) |

Then:

```
build.cmd
```

The script checks all three before compiling and tells you exactly which one is missing
and where to get it. It finishes by dithering a test image and comparing against
ImageMagick, so a green run means the build is correct, not merely that it compiled.

Useful options:

```
build.cmd --no-cuda     CPU only.  Same program, much faster to compile
build.cmd --clean        delete the build directory and start over
```

If ImageMagick is installed somewhere unusual:

```
set IMAGEMAGICK_ROOT=C:\path\to\ImageMagick-7.x.x-Q16-HDRI
build.cmd
```

### MinGW and clang will not work

nvcc on Windows accepts only MSVC as its host compiler, and the ImageMagick import
libraries are MSVC-ABI. g++ mangles the `Magick::` symbols with the Itanium ABI and every
reference comes back undefined. This is a toolchain constraint, not a preference.

---

## Use

### Images

```
rdither --colors 16 photo.png out.png

# check the result against ImageMagick instead of trusting it
rdither --colors 16 --verify photo.png out.png

# GPU.  'blocks' is the one to use: same output, far faster
rdither --colors 16 --engine blocks photo.png out.png
```

On a machine with no CUDA — an AMD or Intel GPU — use `opencl` instead. It is the
same block partition and the same arithmetic reached through OpenCL, so the output
is identical to `blocks` on the same frame:

```
rdither --colors 16 --engine opencl photo.png out.png
```

Currently single-image only; the video path still uses CUDA. See
[docs/OPENCL.md](docs/OPENCL.md), which is also where the verification is described.

### Video

```
rdither --video --engine blocks --colors 16 in.mp4 out.mkv
```

Decode, dither and encode run concurrently, so the reported fps is the whole pipeline's
throughput rather than one stage's. Audio from the source is carried across untouched
(`--no-audio` to drop it).

### What the fps number means in practice

The measurement below is on a real clip, not a synthetic one:

```
rdither --video --engine blocks --colors 16 input.mp4 out.mkv
```

`input.mp4` is 18001 frames, 1920x1080, 60 fps, 300.032 s — h264 yuv420p, bt709,
tv range, 10.2 Mbps, 373 MB, with stereo AAC. Machine: **Xeon E5-2620 v3 @ 2.40 GHz**,
6 physical cores / 12 threads, 15.8 GB RAM, **GTX 1650 SUPER**. Input mode, pixel format,
preset, queue depth, workers and batch size were all left at their defaults.

```
palette    : 16 colours from 256 sampled frame(s), 128x128 montage, 64.0 MiB, 28676.7 ms, mean saturation 22.1%, 9 near-neutral
frames     : 18001 in 352.51 s (51.1 fps)
busy time  : palette 28677 | decode 270944 | dither 222857 | encode 153089 | wall 352507 ms
```

**Read that as elapsed time, not as a benchmark.** 18000 frames at 60 fps is 5 minutes
of footage, and it finished in 352 seconds. So:

> **5 minutes of 60 fps video takes about 5 minutes 53 seconds to render.** You wait
> roughly six minutes, and the file is complete, seekable and openable in an editor
> when the command returns. Nothing is queued up behind it.

The same ratio holds wherever you point it — roughly **1.17x the footage's own
duration**. That means:

- **60 fps footage takes about 17% longer than it plays.** A 3-hour 60 fps clip is
  about 3 hours 31 minutes. A 30-second insert takes 35 seconds.
- **30 fps footage comes out well ahead of real time** — 30 frames per second of
  footage against a ~51 fps pipeline, so roughly **0.59x**, and a 3-hour 30 fps clip
  lands in about 1 hour 46 minutes. Same machine, same numbers, opposite conclusion.

So whether this is faster or slower than real time depends entirely on your source
frame rate, and it is worth checking rather than assuming.

#### This clip is not a real-world benchmark, and here is the measurement that says so

Do not quote the 51 fps at anyone. That clip is a **DaVinci Resolve render using the
YouTube 1080p preset** — a low-bitrate, already-compressed, heavily re-encoded file —
and the content is a **grey, low-saturation extreme hyperlapse at 2000% speed**, so
almost nothing in it resembles real footage. It is the longest clip this project was
measured on, and it is unrepresentative in ways the tool can partly quantify.

The palette line is the tell: **mean saturation 22.1%, and 9 of the 16 palette entries
near-neutral.** Nine greys in a 16-colour palette is not what most video looks like. A
palette sampled from saturated, colourful material would spend that budget on hues
instead, and hue transitions are more expensive to encode than flat greys — so the
encode half of this measurement is flattered by the content in a way yours would not be.

What survives the criticism is the part that does not depend on the picture, and that
is the number worth having:

#### The dither cost is fixed, and it is not close to being the bottleneck

The walk kernel does identical work on every pixel of every frame — a fixed 16-deep
error-queue shift and a palette search whose length is the palette size. There is no
data-dependent branch and no early exit, so **content does not change the dither's
cost.** Measured here:

| | |
|---|---|
| per frame, 1920x1080, 16 colours, B=512 | **~10 ms** |
| pixels dithered | 2,073,600 x 18,001 = **37.3 gigapixel** |
| implied rate | ~207 Mpixel/s, **100 fps of dithering** |
| against 60 fps footage | **1.7x real time, on its own** |

A synthetic `testsrc2` clip measured 9.6-10.9 ms/frame on the same machine — the same
number as the grey hyperlapse, which is the point. Your content will give you the same
dither time. What your content *will* change is decode and encode.

#### Which is why the machine matters more than the GPU

The `busy time` line is *thread*-time summed across all six cores, so the stages total
647 s against a 352 s wall and you cannot divide them to get each stage's share of it.
As a rough split of that thread-time, though, decode is 271 s, dither 223 s, encode
153 s — **two thirds of the work is not GPU work at all.**

So the same GPU on a better machine finishes faster without the dither doing a single
extra FLOP. Decode and encode are bound by memory bandwidth, storage throughput and
single-thread CPU — and this machine has a **2014-era server Xeon at 2.4 GHz**, which
is very likely the single biggest thing making it slow. A machine with the identical
GPU but a modern CPU, NVMe storage and more memory bandwidth will render this
materially faster. The converse also holds: if you are already faster than real time,
a better GPU will not meaningfully change your edit session. Look at which stage is
largest for your content before buying anything.

**On a segment you can work before the whole thing lands:** `--segment-frames N` writes
separate muxed segment files as it goes, so for a long job the early ones are already
on disk and playable while the later ones render. The default single-file mode writes
one complete file on return; if you interrupt it, that file holds the frames completed
so far and `--resume` continues from there.

The reported wall **includes the palette stage** — 28.7 s of it here, about 8%. It did
not for a while: the denominator was measured from inside the pipeline, which starts
after the palette is already built, so two runs whose palettes differed by 25 s
reported the same fps. If a number here looks worse than one you remember, this is why.

### Rotated video

A stream with a 90 or 270 degree display matrix — every portrait phone video — used to
come out **sheared**. The picture was correct but sliced as if it were the wrong width,
and the output had no rotation tag to compensate.

The cause was silent. ffmpeg's rawvideo output auto-rotates by default, so the decoder
handed the pipeline 1080x1920 frames, while rdither sized every buffer from `width` and
`height` in the stream header, which are the **coded** 1920x1080. The pixel *count* is
identical either way, so no bounds check could detect it — only the row width was
wrong, and the frame was read back sliced differently.

rdither now uses the **displayed** geometry everywhere: buffers, the raw pipe's `-s`,
the encoder, and all three dither engines. The output carries upright pixels and no
rotation tag, so it is correct in any container. It says so when it happens:

```
[video] source carries a -90 degree display matrix; the decoder rotates, so the
frames are processed and written as 1080x1920 upright with no rotation tag.
```

Rotating the pixels in, rather than copying the display matrix to the output, is forced
rather than preferred: ffmpeg 9 exposes `-display_rotation` as an **input** option only
and has removed `-metadata:s:v rotate=`, so there is no way to carry the matrix to a
re-encoded output. The cost is that the error diffusion runs on upright pixels, so its
direction is rotated relative to ImageMagick's own pipeline — that changes which pixels
differ, not whether the result is a correct 16-colour dither.

**There is no regression test for this.** Every clip in `tests/` is unrotated, and
ffmpeg cannot *write* a display matrix at all, so a fixture cannot be generated the
obvious way. It is the one known gap in the suite, and the reason to suspect rotation
first when a rendered video looks wrong.

### Variable frame rate

A VFR source used to render at the wrong length. Three separate faults, all of which
present as "doesn't render correctly":

**The rate.** The encoder was fed `r_frame_rate`, which for a variable-rate source is
the *maximum instantaneous* rate, not an average — a container hint, not a summary. A
3.03 s source came out 1.63 s long, with matching A/V desync. rdither now measures the
real frame timings and re-emits at the true average. Constant-rate inputs are
unaffected: the two numbers are identical.

**`-shortest`.** That flag stops output at the shorter of the two streams, which is
right when the audio runs long and actively wrong the other way: when the **video** is
longer, it *deletes video frames* to make the tracks match. On a 30 s excerpt with 895
frames against 30.0128 s of audio, the output had 892 — three frames gone, every run,
scaling with the clip. It is now used only where it trims audio, decided by comparing
the measured durations.

**Per-frame cadence is still flattened.** The output is constant rate at the true
average, which fixes duration and A/V sync but resamples the motion once. Restoring
the cadence needs a piecewise `setpts` re-applied at the encoder, because a rawvideo
pipe carries no timestamps. `--video-preserve-vfr` does this and is **off by default**:
the mechanism is proven in isolation, but it is not verified end to end, because no
genuinely variable-timestamp fixture could be produced on this machine.

rdither reports what it did, because a flattened VFR render is frame-for-frame
indistinguishable from a correct one and nothing in the output says which you got:

```
[video] variable frame rate: 5128 frames in 322 constant-rate runs, 172.0645 s
total, 29.803 fps average (the container declares 29.833333333).
[video]   re-emitted at the true average, 29.803 fps.  Duration and A/V sync match
the source; per-frame motion cadence is flattened.
```

## Where bit-exactness holds

This is the most important section in the file, because "bit-exact" is true of some
engines here and false of others, and the difference is not a detail.

| What you run | Engine it uses | vs ImageMagick | Tested by |
|---|---|---|---|
| `--engine cpu` | sequential walk, on the CPU | **bit-exact** | 135 cases, `verify.ps1` |
| `--engine cuda` | sequential walk, on the GPU | **bit-exact** | 135 cases, `verify.ps1` |
| `--engine blocks` | block-parallel walk, CUDA | **not exact** — 1.7% of pixels differ | determinism only |
| `--engine opencl` | block-parallel walk, OpenCL | **not exact** — 1.7% too, being identical to `blocks` | `blocks` on 54 cells + determinism |
| `--engine approx` | iterative solver | **not exact**, by design | not covered |
| `--video` (any engine) | block-parallel, always | **not exact** | frame accounting, cross-engine |
| `--dither bayer\|atkinson\|jarvis\|floyd-steinberg\|clustered-dot` | their own kernels | **not exact**, and not trying to be | determinism only |

**So: images on `cpu` or `cuda` are bit-exact. GPU *block* engines and all of video
are not.** One design decision causes that, and it is not a bug:

**The block walk is an approximation, on purpose.** Riemersma diffusion is a strictly
sequential walk: one 16-entry error queue, one cursor, each step consuming the last
step's residual. To parallelise it, a block of N positions restarts its queue from
zero, so only the first 16 positions of each block can be perturbed. That is a bounded,
*localised* defect — a slightly different speckle at block seams, not banding and not
drift — and it is measured against ImageMagick on a 200x150 gradient:

| `--blocks` | pixels differing from ImageMagick | RMSE |
|---|---|---|
| 32 | 1819 of 30001 (6.1%) | 0.0239 |
| **512** (default) | **501 of 30001 (1.7%)** | **0.0107** |

Larger blocks carry error further before restarting, so they land nearer the reference.
512 is the default because it is the faithful one: `--blocks 32` is 11% quicker on the
dither but carries 3.4× the deviation (RMSE 0.0239 against 0.0107). Since the dither is
roughly a third to a half of the pipeline depending on content, that 11% is a few
percent end to end — which is a deliberate trade, and `--blocks 32` is there if you
want the other end of it. **No block size makes the walk exact — only cutting it does.**

**Video always uses the block walk.** The pipeline is built around the block partition:
each work item is a batch of frames dithered as independent blocks, which is what gives
it its parallelism. A sequential walk cannot be batched, so the video path is the
approximation by necessity, and that is the honest answer to "is video bit-exact?" —
**no**.

There is no bit-exact video mode, and `--no-gpu` is not one. `--no-gpu` runs the *same*
block walk on host workers: same partition, same approximation, just off the GPU. It is
the way to check the GPU is not lying to you, not the way to get a different answer.
If per-pixel agreement with ImageMagick matters more than throughput, the dither has to
be cut rather than parallelised, and that is an architectural change to the video
pipeline — not a flag. It is not built.

**And video is not bit-exact with its own input either**, which is a separate point
worth making plainly. Beyond the dither, the default encoder is lossy `libx264`, and
`--video-pix-fmt yuv420p` subsamples chroma over 2×2 blocks, averaging away the dither
itself. Measured on 1080p at 16 colours: `ffv1 yuv444p` → 56 unique colours in the
output, `libx264 yuv444p crf 12` → 15191, `libx264 yuv420p` → 36031. Use
`--video-lossless` when you care what the encoder did to the palette; it costs about
28× the bitrate.

**What *is* verified for the GPU engines**: that `blocks` and `opencl` agree with each
other byte-for-byte, per pixel, on 54 cases including alpha input, and on 60 frames of
1080p video — and that both are deterministic across repeated runs. The OpenCL port
was built against that bar specifically, because `rd_riemersma.h` defines the two
engines as having the same partition and the same arithmetic, which makes "identical or
wrong" the only available outcome. Neither claim is that they match ImageMagick.

### Progress and ETA

Long stages report themselves, with an ETA:

```
[palette] 128/256  50%  9.1/s  eta 14s  14.0s elapsed
[video] 7984/18001  44%  48.7/s  eta 3m47s  2m44s elapsed  read 8000
```

On a terminal the line is redrawn in place, about four times a second. Redirected to
a file it becomes one newline-terminated line every ten seconds, because a log file
full of carriage returns is not a log file. `--quiet` prints none of it.

`[video]` counts frames **written to the output**, and `read` is how far the decoder
has got. The gap between them is the depth of the in-flight queue; a `read` that
climbs while the bar does not means the bottleneck is downstream, not in decoding.
While the bar is drawing, the per-batch timing lines are suppressed, since they
would land on the bar's own line — `RD_TRACE=1` brings them back.

### Interrupting and resuming

A long render can be split, and a run that dies mid-way does not have to start over:

```
rdither --video --engine blocks --colors 16 --segment-frames 2000 in.mp4 out.mkv
rdither --video --engine blocks --colors 16 --resume out.mkv
```

Each segment is verified on resume, so a truncated output is reported rather than
accepted.

### All the options

```
rdither --help
```

---

## Two things worth knowing before you judge the output

**The encoder can throw the palette away.** A 16-colour image re-encoded as lossy 4:2:0
does not have 16 colours any more — chroma subsampling averages the dither's per-pixel
alternation away, and lossy compression invents colours around every transition. Measured
on this pipeline: 16 colours in, 36356 out at `yuv420p`, 30445 at `libx264 yuv444p crf 18`,
56 at `ffv1 yuv444p`. If the result looks flat and grey, this is usually why, and the fix
is `--video-lossless` (about 28× the bitrate) or accepting some loss. rdither warns you
when the chosen settings cannot carry the palette.

**The palette comes from sampling, so it can miss.** For video it samples frames across
the whole clip — as many as a 60-second budget allows, up to 256 — and reports the mean
saturation it got. That number is the thing to check. If it comes back low, the footage
really is grey, and more samples will not fix it: on grey-heavy material saturation sits
at 22–24% whether you sample 30 frames or 256.

---

## Extending it

rdither reproduces one algorithm exactly, but almost nothing around it is specific to
Riemersma. The palette builder, the curve generator, the block partitioner, the
concurrent pipeline and the crash-resume checkpoints all work with any dither you supply.

Add one with `--dither`:

```
rdither --list-dithers                      # what is registered
rdither --dither bayer --colors 16 in.png out.png
```

`examples/bayer_dither.cc` is a complete, working second algorithm in about 40 lines,
built into the default binary. It exists to prove the seam works rather than to be useful
— ordered dithering has visible 8×8 texture where Riemersma has none.

Writing your own is three steps: copy that file, write the kernel, add it to
`RD_SOURCES` in `CMakeLists.txt`. [include/rd_plugin.h](include/rd_plugin.h) has the
interface and, more usefully, the two mistakes that cost real time to diagnose.

Registration is at compile time, not runtime loading, on purpose. A `LoadLibrary` plugin
built with a different toolchain has to agree with the host on struct layout, calling
convention and allocator, and this program is built with a specific MSVC + CUDA against a
specific ImageMagick. Same-binary means the same compiler, the same headers, and none of
that class of problem.

---

## Layout

```
include/          headers; rd_plugin.h is the extension interface
src/              the implementation
examples/         bayer_dither.cc -- a worked example of a plugin
tools/            measurement scripts (the record of how the numbers were got)
docs/DESIGN.md    the full engineering record
docs/OPENCL.md    the OpenCL engine: what it is, how it is verified, what is left
verify.ps1        135 bit-exact cases against ImageMagick
build.cmd         build, with dependency checks
```

`rd_cuda_common.cuh` and `rd_riemersma.h` are the two files where the actual algorithm
lives. They are commented at the level of "why this and not the obvious thing", because
every one of those choices is load-bearing for bit-exactness.

---

## Correctness

`verify.ps1` renders 135 cases and compares each against ImageMagick's own output, pixel
by pixel, with zero tolerance. It is the first thing to run when something looks wrong
and the last thing to run before committing.

```
> powershell -File verify.ps1
bit-exact cases: 135 passed, 0 failed
```

**Read that number for what it is.** Those 135 cases run on `cpu`, `cuda` and
`cpu --max-ram-mb 1` — the *sequential* walk, the one that is supposed to be exact. The
suite is the reason the exactness claim is trustworthy, and it is also the reason the
inexactness claims are not in any way tentative: those come from a separate measurement
against ImageMagick's output, not from this suite, because this suite does not touch
those engines at all. See [Where bit-exactness holds](#where-bit-exactness-holds).

Beyond that suite, `verify.ps1` also runs:

| Check | What it proves | Result |
|---|---|---|
| `tools\probe-determinism.ps1` | repeated runs give identical pixels and identical files | 6/6 |
| `tools\probe-opencl-exact.ps1` | OpenCL == CUDA, per pixel, on 54 image cells | 54/54 |
| `tools\probe-video-exact.ps1` | OpenCL == CUDA on 60 frames of 1080p, and no frames lost | identical |

---

## Further reading

[docs/DESIGN.md](docs/DESIGN.md) — the engineering record: the mathematics, the twenty-odd
optimisation rounds, the two real bugs found in this codebase's own new code, the
measurements, and a table of everything that was tried and did not work.

[docs/OPENCL.md](docs/OPENCL.md) — the OpenCL engine: what it covers, the 54 image cells
and 60 video frames that verify it against CUDA, the four bugs the port actually had
(including one that turned out not to be in OpenCL at all), and why it is currently
2.7× slower than CUDA.
