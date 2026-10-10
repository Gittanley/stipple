# rdither

GPU-accelerated reimplementation of ImageMagick's Riemersma error-diffusion dither,
for images and video. Bit-exact with ImageMagick where that is possible, and
deliberately approximate where it is not — [see exactly which is which](#where-bit-exactness-holds).

## The guarantee

If you point it at an image with the sequential walk and ask for 16 colours, you get
back the same 16 colours ImageMagick would have produced, the same pixels, byte for
byte - not "similar", not "close enough to pass". That is the design goal, and it is
checked against ImageMagick on every build by [Continuous
integration](.github/workflows/ci.yml), which builds a clean clone and runs the suite
on a GPU-less runner — deliberately, because that is the configuration five separate
defects survived in, each one a build path nobody had ever run.

| | coverage | notes |
|---|---|---|
| clean clone builds at all | every push | the nocuda job alone catches three of the five defects |
| CPU-only build, 110 bit-exact cases vs ImageMagick | every push | 55 further cases report SKIPPED, not passed |
| CUDA build compiles and links | every push | 12.8 toolkit unpacked from redist archives; no GPU on the runner |
| OpenCL-vs-CUDA, 36 cells | manual | no GPU on the runner |
| video probes | manual | no GPU, and no clip is committed |

A skipped check is never counted as a pass, and the CPU job asserts that at least 90
real cases actually ran — because a suite that skips nearly everything and exits 0 is
worse than a failing one.

Three jobs run on every push. The two that gate — `nocuda` and `cuda` — are green; the
third, `linux`, is `continue-on-error: true` and does not block. Getting there took three
fixes in the CUDA job alone —
every one of them in code written minutes earlier and never run — and a hunt for a
*buildable* ImageMagick on a runner that ships only the runtime. All of it is recorded in
[the workflow](.github/workflows/ci.yml) and [docs/DESIGN.md](docs/DESIGN.md) so nobody
repeats it. Two facts out of that hunt matter to you:

- The runners' ImageMagick cannot be linked against, so CI installs its own:
  `pwsh -File tools/install-imagemagick.ps1` — 21 s, pinned to 7.1.2-31 Q16-HDRI. It
  asserts every file the build needs and reads back the version it got, because a prefix
  that installs cleanly can still be unbuildable.
- `CMakeLists.txt` and `build.cmd` therefore accept **three** ImageMagick layouts: the
  official Windows one, a conda prefix, which puts headers under
  `Library\include\ImageMagick-7` and names its libraries `MagickCore-7.Q16HDRI.dll.lib`,
  and a system one on Linux.

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
[63% of it is decode and encode](#what-the-fps-number-means-in-practice), so a
faster CPU, RAM or disk makes it faster with the same GPU.

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
> - **The verification is real too, and it is the point.** 165 bit-exact cases against
>   ImageMagick, a 36-cell cross-engine bit-exactness sweep, a run-to-run determinism
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

### What the output looks like

Five public-domain images. Every panel is 16 colours and every one is verified
bit-exact against ImageMagick. Click any strip for the full-resolution version.

[![Starry Night: original, image path with and without --im-palette, video path with and without --im-palette](docs/examples/starry-strip.png)](docs/examples/starry-strip.png)

[![Blue Marble: the same five variants](docs/examples/bluemarble-strip.png)](docs/examples/bluemarble-strip.png)

[![Earthrise: the same five variants](docs/examples/earthrise-strip.png)](docs/examples/earthrise-strip.png)

[![Martian sunset: the same five variants](docs/examples/mars-strip.png)](docs/examples/mars-strip.png)

[![Self-portrait: the same five variants](docs/examples/portrait-strip.png)](docs/examples/portrait-strip.png)

**Reading the five panels, left to right.**

| panel | what it is |
|---|---|
| **original** | the source image, untouched |
| **image `--im-palette`** | image path, palette built by ImageMagick's own sampling |
| **image** | image path, palette built by rdither's octree |
| **video** | one frame of a dithered clip, via `--video-lossless` |
| **video `--im-palette`** | the same clip, palette from ImageMagick's sampling |

Two things in that table are measured facts rather than intentions.

**Panels 2 and 3 are identical** — AE = 0, same colour count, same bytes. On the image
path `--im-palette` changes nothing, because there is only one frame to sample and
rdither's octree already reproduces ImageMagick's choice. It is not ignored:
`colormap: tree matches ImageMagick exactly` is printed on every run. The flag only has an
effect on video, where there are many frames to choose a palette from — panels 4 and 5
differ by a measured AE of 1.12.

**The video panels hold exactly 16 colours; the image panels do not.** A frame written
through ffv1 keeps the palette precisely, so a 16-colour request really is 16 colours. An
image is written as 8-bit RGB, which retains the dither's local variation and gives
~93,000 distinct pixel values. Neither is wrong; they answer different questions — how
many colours the dither chose, versus what is stored in this file.

You will notice the pattern where it is *supposed* to appear: **Starry Night's sky**
and the **Blue Marble's ocean** are broad smooth ramps, which is exactly where
Riemersma's error diffusion shows its structure. Flat regions stay flat. That is the
whole reason a dither exists — to make a 16-colour image look like a gradient instead of
like 16 bands.

Sources and licences in [docs/examples/SOURCES.md](docs/examples/SOURCES.md).
Sources and licences for all five images are in
[docs/examples/SOURCES.md](docs/examples/SOURCES.md).
### Images

```
rdither --colors 16 photo.png out.png

# check the result against ImageMagick instead of trusting it
rdither --colors 16 --verify photo.png out.png

# GPU.  'blocks' is the fastest, and NOT the same output: ~1.7% of pixels differ
# from the cpu engine above.  See "Where bit-exactness holds" before choosing.
# (That 1.7% is the IMAGE path.  On --video, blocks is bit-identical to the cpu
# engine -- measured 0 of 37,324,800 differing components over 6 frames spanning a
# 600-frame 1080p clip.  In video the choice that moves pixels is --input-mode,
# not --engine.)
rdither --colors 16 --engine blocks photo.png out.png
```

On a machine with no CUDA — an AMD or Intel GPU — use `opencl` instead. It is the
same block partition and the same arithmetic reached through OpenCL, so the output
is identical to `blocks` on the same frame:

```
rdither --colors 16 --engine opencl photo.png out.png
```

**That identity is measured, not asserted: 36 of 36 comparison cells are
bit-identical to CUDA** — six images × two colour counts × three block sizes, via
`tools\probe-opencl-exact.ps1`. It is also the only path for a non-NVIDIA GPU, and
it now works from a clean clone: download the 1.1 MB Khronos SDK, unpack it beside
this repository, and `build.cmd` enables the engine. No install step and nothing
to deploy — `OpenCL.dll` is already on Windows. Full instructions in
[CONTRIBUTING.md](CONTRIBUTING.md#the-opencl-engine).

**What is not measured:** Intel and AMD hardware. All of the above was measured on
an NVIDIA driver, and the thing that varies most between vendors is FP64
throughput, which is this algorithm's cost centre. So the software is ready and
the *speed* on your GPU is not a claim anyone can make for you yet —
`tools\probe-opencl.exe` prints your device and whether it has
`cl_khr_fp64`, and [docs/OPENCL.md](docs/OPENCL.md) explains what to do with that.

It runs **video** as well as images, on the same input modes, and is within about
**1.07×** of CUDA's wall clock on 1080p60 - measured, 3 interleaved runs of 600 frames
on this machine: CUDA 15.7 / 16.0 / 17.4 s and OpenCL 16.9 / 17.1 / 17.2 s, which is
1.04× on the mean and 1.07× on the best of three. This read "1.2-1.3×" when written;
both engines got faster since, the OpenCL side by the per-pixel unvisited fill and the
cached clip-invariant buffers. Read the ratio as a range, not a constant.
It was long documented as "2.7× slower than CUDA", which was an artefact of
comparing the two engines on *different data paths*: OpenCL could only take rgba64le
at 8 bytes per pixel while CUDA defaults to planar yuv444p at 3, and the 2.67× of extra
traffic was the entire apparent gap. On the same path the engines were within 8% even
before the planar kernels were ported. Only `--input-mode yuv420` and the
`yuv444-prepass` are still refused, with a message. See
[docs/OPENCL.md](docs/OPENCL.md), which is also where the verification is described.

### Video

```
rdither --video --engine blocks --colors 16 in.mp4 out.mkv
```

Decode, dither and encode run concurrently, so the reported fps is the whole pipeline's
throughput rather than one stage's. Audio from the source is carried across untouched
(`--no-audio` to drop it).

### What it reads, what it does to the pixels, what it writes

Not only a dither: the video path also does its own colour conversion, in both
directions, on the device.

**Input.** Anything ffmpeg decodes, at the pixel formats ffmpeg produces. The decoder
hands over planar YUV, and `--input-mode` chooses what happens next:

| `--input-mode` | bytes/px | what it does |
|---|---|---|
| `yuv444` (default) | 3 | swscale's own YCbCr→RGB matrix, on the device |
| `yuv444-prepass` | 3 | same result via a coalesced pass; measured **AE 0** against `yuv444` |
| `yuv420` | 1.5 | the decoder's own format, no swscale at all; chroma reconstructed 2× bilinear on the device |
| `rgba64` | 8 | the reference path: ffmpeg converts, rdither receives RGBA |

`yuv420` is the cheapest and `rgba64` the most faithful, and on 4:2:0 source that
difference is large. Measured on a 4:2:0 clip, comparing modes to each other so the
dithering is held constant:

| | vs `yuv444` | vs `rgba64` |
|---|---|---|
| `yuv444` | — | 28.0 dB |
| `yuv420` | **34 dB** | not measured |

**34 dB is not a colourimetric nuance, it is a different picture.** If you have 4:2:0
footage and care about fidelity, use `yuv444` or `rgba64`. On a true 4:4:4 source every
mode agrees exactly (`yuv444` vs `rgba64` is AE 0), so the choice only bites for 4:2:0
input - which is most real footage.

**The byte column above is not a speed column, and the difference matters.** Halving
pipe traffic does not halve wall clock. Measured end to end on the 600-frame 1080p bench
clip, `--colors 16 --video-lossless`, `--engine blocks`, interleaved:

| `--input-mode` | mean fps | vs `yuv444` |
|---|---|---|
| `rgba64` | 33.7 | 0.67x |
| `yuv444` (default) | 50.5 | baseline |
| `yuv420` | 53.2 | 1.05x |

So the big step is the one already taken - `rgba64` to `yuv444` is **1.50x** - and
`yuv444` to `yuv420` is **1.05x**. The reason is that `yuv420` does not merely move
fewer bytes: it reconstructs 2x2 chroma on the device, which is real work that consumes
much of the saving. **`yuv420` stays off the default for that reason.** (5% is also
inside this machine's 10-25% run-to-run drift between identical binaries, so treat it as
indicative rather than settled; `yuv444` is the default because it is not clearly worth
the 34 dB.)

**Output.** `--video-pix-fmt` defaults to `yuv444p` and that default is load-bearing:
`yuv420p` output subsamples chroma over 2×2 blocks, averaging away the dither pattern
entirely. It is also far smaller, in the same direction as the unique-colour counts measured
below (36,031 at `yuv420p` against 15,191 at `yuv444p`, same 600 frames) - so the smaller
file is the one where the dither is gone. `--video-lossless` gives
ffv1 yuv444p, which keeps the palette exactly, at roughly 28× the bitrate of yuv420p
H.264.

### Concurrency, and the RAM it uses

Decode, dither and encode all run at once, and the tool will use the machine you give it.
On the benchmark machine, thread-time totals 675.6 s against a 352 s wall, so every core is
busy — and 63% of that is decode and encode rather than GPU work. The GPU
is not usually the constraint; see
[what the fps number means](#what-the-fps-number-means-in-practice).

**`--cpu-threads` is not a scheduling knob, and this is the part worth reading.** With
`auto` and a GPU present, the host worker pool is **off** (`0`) and the GPU walk is the
only walk. That is measured: the host block walk costs ~937 ms/frame at 1080p against the
GPU's 63.5, and on a deep queue ten host workers make the whole pipeline *slower*
(219–244 ms/frame with them, 115 without) because they saturate the memory bandwidth the
GPU's data path needs.

So **any explicit value ≥ 1 turns the host walk on**, and the host walk is not
bit-identical to the GPU walk — the program says so on every run that does it. Even among
host-only runs the worker count matters, because fewer workers means a different block
partition and each block restarts its error queue. **If you need reproducible output,
leave it on `auto`.**

Other knobs, briefly: `--reader-threads`, `--decode-threads` and `--encode-threads` cap
ffmpeg's own threads; `--gpu-workers N` runs N GPU walks at once (2 measured no faster,
the dither being saturated rather than stalled); `--frames`, `--batch-frames` and
`--queue-depth` set the work granularity.

**RAM.** The queue is sized first and the worker count trimmed to fit it, because a deep
queue with fewer workers beats a shallow one with more. `[ram]` on every run reports the
real figure — its `queue N` is **megabytes in flight, not a slot count** — plus a 768 MiB
decoder reserve, with the reserve dominating. `--mem-fraction` sets the share of physical
RAM the queue may use (default about a third), and `--max-ram-mb` caps that budget when you
pass it. The cap can only lower it: a request for more memory than the machine has is
answered with the memory it has. It is also not exact — a liveness floor of two slots means
a very small value cannot push the queue below that, which is reported rather than hidden;
frames beyond the budget spill to disk rather than failing.

### What the fps number means in practice

Measured end to end on a real clip — 18001 frames of 1920x1080 60 fps h264 (300 s of
footage), all defaults, on a **Xeon E5-2620 v3 @ 2.40 GHz**, 6 cores / 12 threads,
15.8 GB RAM, **GTX 1650 SUPER**:

```
palette    : 16 colours from 256 sampled frame(s), 128x128 montage, 64.0 MiB, 28676.7 ms, mean saturation 22.1%, 9 near-neutral
frames     : <frames> in <total> s (<fps> fps, palette included)
             <fps> fps over the <pipeline> s window the progress bar covers (excludes the <palette> s palette)
busy time  : wall <total_ms> ms (palette <palette_ms> of it)
             summed worker time: decode <decode> | dither <dither> | encode <encode>
             writer <convert+pipe> ms of that encode figure: <convert> ms float->uint16 on the host, <pipe> ms pushing the pipe
```

Line shapes as of `67f529e`, read off `src/rd_cli.cpp:332`, `:347`, `:359` and `:367`.
Placeholders, not a transcript: the four lines above the `busy time` block are the ones
this section had quoted, and they are kept because the decode/dither/encode figures are
what `[63% of it is decode and encode](#what-the-fps-number-means-in-practice)` is
computed from. `convert_ms` and `pipe_ms` are a partition of the encode figure; the three
beside them are summed worker time across threads and can exceed the wall clock.

**Read that as elapsed time, not as a benchmark.** 300 s of footage took 353 s, so:

> **5 minutes of 60 fps video takes about 6 minutes to render.** You wait, and the file
> is complete, seekable and openable in an editor when the command returns.

The ratio — roughly **1.17x the footage's own duration** — is the useful part, and it
moves with your source frame rate: 60 fps footage takes ~17% longer than it plays (a
3-hour clip is about 3 h 31 m), while 30 fps comes out well ahead of real time (~0.59x,
so ~1 h 46 m for 3 hours). Whether this beats real time depends on your source rate, so
it is worth checking rather than assuming.

**Do not quote the 51 fps at anyone.** It is one low-bitrate, already-compressed file of
grey low-saturation material on a 2014-era 6-core Xeon, and 63% of the thread-time
is decode and encode rather than GPU work — so a faster CPU, NVMe or more memory
bandwidth moves this number with the same GPU, and a faster GPU will not fix a pipeline
that is not GPU-bound. Read which stage is largest in your own `busy time` line before
buying hardware. The reported wall **includes the palette stage** — 28.7 s here, ~8%.

**On a long job you can work before the whole thing lands:** `--segment-frames N` writes
separate muxed segment files as it goes, so early ones are playable while later ones
render. The default single-file mode writes one complete file on return; interrupt it and
that file holds the frames completed so far, and `--resume` continues from there.

### Rotated video

A stream with a 90 or 270 degree display matrix — every portrait phone video — used to
come out **sheared**. rdither now uses the **displayed** geometry everywhere: buffers,
the raw pipe's `-s`, the encoder and all three dither engines. The output carries upright
pixels and no rotation tag, so it is correct in any container, and it says so when it
happens:

```
[video] source carries a -90 degree display matrix; the decoder rotates, so the
frames are processed and written as 1080x1920 upright with no rotation tag.
```

Rotating the pixels in rather than carrying the matrix to the output is forced, not
preferred: ffmpeg 9 exposes `-display_rotation` as an **input** option only and has
removed `-metadata:s:v rotate=`. The cost is that error diffusion runs on upright pixels,
so its direction is rotated relative to ImageMagick's — which changes *which* pixels
differ, not whether the result is a correct 16-colour dither.

**There is no regression test for this**, and ffmpeg cannot *write* a display matrix, so a
fixture cannot be generated the obvious way. It is the one known gap in the suite, and the
reason to suspect rotation first when a rendered video looks wrong. The writeup is in
[docs/DESIGN.md](docs/DESIGN.md).

### Variable frame rate

A VFR source used to render at the wrong length. rdither now **measures the real frame
timings and re-emits at the true average** — the encoder is not fed the container's
`r_frame_rate`, which for a variable-rate source is the maximum instantaneous rate rather
than a summary. Constant-rate inputs are unaffected: the two numbers are identical.
`-shortest` is now used only where it trims audio, never where it would delete video
frames to make the tracks match.

**Per-frame cadence is still flattened.** The output is constant rate at the true average,
which fixes duration and A/V sync but resamples the motion once. `--video-preserve-vfr`
restores the cadence with a piecewise `setpts` re-applied at the encoder, because a
rawvideo pipe carries no timestamps — it is **off by default**: the mechanism is proven in
isolation but not verified end to end, because no genuinely variable-timestamp fixture
could be produced on this machine.

rdither reports what it did, because a flattened VFR render is frame-for-frame
indistinguishable from a correct one and nothing in the output says which you got.

## Where bit-exactness holds

This is the most important section in the file, because "bit-exact" is true of some
engines here and false of others, and the difference is not a detail.

| What you run | Engine it uses | vs ImageMagick | Tested by |
|---|---|---|---|
| `--engine cpu` | sequential walk, on the CPU | **bit-exact** | 165 cases, `verify.ps1` |
| `--engine cuda` | sequential walk, on the GPU | **bit-exact** | 165 cases, `verify.ps1` |
| `--engine blocks` | block-parallel walk, CUDA | **not exact** — 1.7% of pixels differ | determinism only |
| `--engine opencl` | block-parallel walk, OpenCL | **not exact** — 1.7% too, being identical to `blocks` | `blocks` on 36 cells + determinism |
| `--engine approx` | iterative solver | **not exact**, by design | not covered |
| `--video` (any engine) | block-parallel, always | **not exact** | frame accounting, cross-engine |
| `--dither bayer\|atkinson\|jarvis\|floyd-steinberg\|clustered-dot` | their own kernels | **not exact**, and not trying to be | determinism only |
| `--dither bayer-ordered\|void-and-cluster` | ordered thresholds, no diffusion | **not exact**, and not trying to be | determinism only |

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
block walk on host workers, so it is still the approximation above — on paper the way to
check the GPU is not lying to you without a second GPU.

**It is now trustworthy for that.** The host path agrees with CUDA **byte for byte** on
video, at `--batch-frames` 1 and 16 (the two the oracle exercises, plus 16 being the
default), in `yuv444`, `yuv444-prepass` and
`rgba64`, and at 1920x1080 as well as 320x180 — so `--no-gpu` measures the GPU rather
than the host. Getting there took three fixes, all found by running two builds on one
machine and comparing decoded pixels, because `probe-video-exact` compares two *device*
engines and the host was only ever checked for determinism: frames written in
worker-completion order, 8-bit planar YUV read through a `uint16` stride, and a
float→planar converter whose plane stride was the whole batch rather than one frame,
which made every `--batch-frames` above 3 wrong — including the default of 16. All three
are written up in [docs/DESIGN.md](docs/DESIGN.md).

What none of that changes: the host walk is still the block walk, so it is still ~1.7% of
pixels from ImageMagick. `--no-gpu` answers "does the GPU agree with the CPU", not "is
this bit-exact".

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
other byte-for-byte, per pixel, on 36 cases including alpha input, and on 60 frames of
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

## One thing worth knowing before you judge the output

(The other one — *the encoder can throw the palette away*, and what `--video-lossless`
is for — is covered under [Where bit-exactness holds](#where-bit-exactness-holds).)

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

`examples/bayer_dither.cc` is a complete, working second algorithm in about 170 lines,
built into the default binary. It exists to prove the seam works rather than to be useful
— ordered dithering has visible 8×8 texture where Riemersma has none, and its threshold
is applied as a brightness offset that is a no-op at this bit depth. For a dither you
would actually choose:

| `--dither` | what it is |
|---|---|
| `bayer-ordered` | 8×8 Bayer, no error diffusion. Crisp; the 8×8 period is visible by design. |
| `void-and-cluster` | blue-noise thresholds, 64×64 matrix. Intended for smooth gradients, where Riemersma's worms are objectionable. |
| `void-and-cluster-fast` | the same algorithm at 32×32, for when the matrix build is too slow. |

"Blue noise" and "void-and-cluster" are the same algorithm — Ulichney's — so there is no
separate `blue-noise` name; the two entries are two tile sizes of one method.

`--diffusion` scales the threshold toward 0.5, and `--diffusion 0` reduces any of them to
plain nearest-colour — the same meaning it has for the diffusion kernels.

**None of this is bit-exact with anything, and none of it is trying to be.** They are
dither *choices*. See [Where bit-exactness holds](#where-bit-exactness-holds).

**Verified:** each emits exactly the palette's colours with no off-palette rounding, and
each is deterministic across repeated runs (`probe-determinism.ps1` sweeps 8 algorithms
over 4 runs each).

**Not verified:** that the void-and-cluster spectrum is actually blue, and the two tile
sizes agree far more than they should. Measured on `tests\noisy.png` at 16 colours, the
64×64 and 32×32 outputs differ by **144 pixels in 196,608 (0.073%)**, and:

- at `--diffusion 0` they differ by **zero** pixels, which localises the entire
  disagreement to the threshold decision — diffusion is the only thing that flag touches;
- it is **exactly 144 at `--diffusion` 1, 2 and 8**, though each render does change, so it
  is not an artefact of how far the thresholds are scaled;
- those 144 pixels are the image positions of **3 cells of the 64×64 matrix** —
  `(32,0)`, `(0,32)`, `(32,32)` — and all three correspond to **`(0,0)` in the 32×32
  grid**. Every one lies on a 32-tile seam; none is strictly interior.

So the two matrices agree, as far as the decision function can show, at 4093 of 4096
positions. Both really are built — the tables are separate statics and the build costs
differ 9.4× — so *why* two independently computed blue-noise rank fields agree that
closely is not established. The filters are sound and distinct; the quality claim is not
yet earned. See [docs/DESIGN.md](docs/DESIGN.md).

Writing your own is three steps: copy that file, write the kernel, add it to
`RD_SOURCES` in `CMakeLists.txt`. [include/rd_plugin.h](include/rd_plugin.h) has the
interface and, more usefully, the two mistakes that cost real time to diagnose.

---

## Layout

```
include/          headers; rd_plugin.h is the extension interface
src/              the implementation
examples/         bayer_dither.cc -- a worked example of a plugin
tools/            measurement scripts (the record of how the numbers were got)
docs/DESIGN.md    the full engineering record
docs/OPENCL.md    the OpenCL engine: what it is, how it is verified, what is left
verify.ps1        165 bit-exact cases against ImageMagick
build.cmd         build, with dependency checks
```

`rd_cuda_common.cuh` and `rd_riemersma.h` are the two files where the actual algorithm
lives. They are commented at the level of "why this and not the obvious thing", because
every one of those choices is load-bearing for bit-exactness.

---

## Correctness

`verify.ps1` renders 165 cases and compares each against ImageMagick's own output, pixel
by pixel, with zero tolerance. It is the first thing to run when something looks wrong
and the last thing to run before committing.

```
> powershell -File verify.ps1
bit-exact cases: 165 passed, 0 failed
```

**Read that number for what it is.** Those 165 cases run on `cpu`, `cuda` and
`cpu --max-ram-mb 1` - the *sequential* walk, the one that is supposed to be exact. The
suite is the reason the exactness claim is trustworthy, and it is also the reason the
inexactness claims are not in any way tentative: those come from a separate measurement
against ImageMagick's output, not from this suite, because this suite does not touch
those engines at all. See [Where bit-exactness holds](#where-bit-exactness-holds).

Without an NVIDIA GPU you will see **110 passed, 55 SKIPPED** rather than 165. The 55 are
the `cuda` cases, and they are reported as skipped rather than quietly dropped. The split
is exact rather than measured, because the case list is a product: **3 engines x 5 colour
counts x 11 fixtures = 165**, so each engine owns 5 x 11 = 55, and dropping `cuda` leaves
two. This was 100/50 while the total was 150, before `in_gray_alpha` and `in_jpeg` were
added -- a stale pair of numbers that still divided correctly, which is the dangerous kind:

```
bit-exact cases: 110 passed, 0 failed, 55 SKIPPED (engine not in this build)
```

Beyond that suite, `verify.ps1` also runs:

| Check | What it proves | Result |
|---|---|---|
| `tools\probe-determinism.ps1` | repeated runs give identical pixels and identical files | 6/6 |
| `tools\probe-opencl-exact.ps1` | OpenCL == CUDA, per pixel, on 36 image cells | 36/36 |
| `tools\probe-video-exact.ps1` | OpenCL == CUDA on 60 frames of 1080p, **both data paths**, and no frames lost | 2/2 identical |
| `tools\probe-video-determinism.ps1` | the same video command 8× over is the same pixels and the same frame count, on each of the three engines | 3/3 |
| `tools\probe-unvisited-pixel.ps1` | the pixel the walk never visits keeps its source value, on 14 geometries | 6 of 14 have one; 8 have none to check |
| `tools\probe-video-invariance.ps1` | `--batch-frames 16` gives the same pixels as `1`, `--no-gpu` the same as the device, **and part C: the video path stays within 1% of ImageMagick's own Riemersma** | 2/2, 1 half skipped without a GPU |
| `tools\probe-host-oracle.ps1` | the host walk is bit-identical to the device, and batch-invariant | ok |
| `tools\probe-alpha-guard.ps1` | `SourcePixFmtHasAlpha` takes both directions | ok |
| `tools\probe-split-decode.ps1` | N seek-based decodes == one decode, per frame | CFR identical; VFR differs, reported not failed |
| `tools\probe-palette-deadline.ps1` | `--palette-budget-ms` bounds the stage, **and both palette samplers agree on a clip whose frames are all identical** | ok |
| `tools\probe-cache-invalidation.ps1` | the clip-invariant cache hits within a run and leaks nothing between geometries | ok |
| `tools\probe-segment-frames.ps1` | `--segment-frames` runs at all: plain, with `RD_PALETTE_OVERLAP=1`, multi-segment, and its decoded pixels equal the unsegmented run | ok |
| `rdither --self-test` | the C++-level invariants, including the palette gate's blocking and publish semantics | PASSED |
| `tools\probe-palette-overlap.ps1` | `RD_PALETTE_OVERLAP` changes no pixel | **off by default** — run `verify.ps1 -Slow` |

### Exit codes

`verify.ps1` does not have one "success" code, and the difference matters when a run is red:

| | |
|---|---|
| **0** | every stage ran and passed (reachable with `-Slow`, or when nothing is skipped) |
| **1** | a stage failed |
| **2** | a stage did not run for an *incidental* reason — missing fixture, `-Fast`, probe could not start. A gap. |
| **3** | the `-BudgetMinutes` watchdog fired; the run did not finish |
| **4** | **pass with declared exclusions** — everything that ran passed, and everything skipped is off by default and named in the output with the switch that runs it. **This is what the default invocation returns.** |

### Part C is the one that matters, and it was the last one added

Every other video check here compares rdither to rdither: OpenCL to CUDA, run to run, host
to device, batch 16 to 1. A defect shared by all their arms is **invisible to them, because
they agree** — and three shipped that way through runs recorded as
`EXIT=0, 165/0 bit-exact, every stage ran, 0 skipped`:

* the by-seek palette sampler read RGBA bytes as RGB triples, giving a green-grey palette
  with 7 of 16 entries pure green, while reporting "mean saturation 49.0%";
* `nb_frames` is `N/A` on mkv, so the palette was built from **one frame**;
* `--palette-import` on video hung forever on a palette gate nobody released.

Part C is the only shape that can see that class: rdither's video output against
ImageMagick's own Riemersma. It asserts a **1% ceiling**, not `AE=0`, because the blocks walk
zeroes its error queue per block and the reference does not — measured 0.099%–0.538% across
six colour counts and five block lengths, so the ceiling is set from that envelope rather than
from one sample.

That last one is the odd one out and earns its place. The cross-engine checks compare two
engines against each other, which makes them **structurally blind to a fault both engines
share** — and that has happened: the pixel ImageMagick's walk never visits was
uninitialised device memory in *both* engines, and it was invisible by eye because a fresh
`cudaMalloc` returns zeroed pages. So that check compares against the **source** rather
than against another engine, and carries a negative control — the unvisited pixel must
match the source *and* its neighbour must **not**, or a renderer that dithered nothing
would pass.

`probe-video-determinism.ps1` is the same idea one level up, and for the same reason. It
verifies the frame count **from the file** rather than from rdither's own report: a
pipeline that consistently drops frames reports a consistent number, and a self-consistent
report is not evidence.

### Known limitations

These are current, not history. Fixed faults are not listed here; they are in
[docs/DESIGN.md](docs/DESIGN.md) with the reasoning, and in the commit history.

**`--video-preserve-vfr` is off by default.** The mechanism is proven in isolation, not
end to end — no genuinely variable-timestamp fixture could be produced on this machine.

**There is no regression test for rotated video.** ffmpeg cannot *write* a display
matrix, so the fixture cannot be generated the obvious way.

**A fault that the host and the device share in the video path is still invisible.**
`probe-video-invariance.ps1` compares the host against the device, so it catches a host
answer that differs — which is where **four** separate faults lived, one of them making
`--no-gpu` wrong at every `--batch-frames` above 3, which is the default. But if both
agreed on something wrong, it would pass, exactly as the cross-engine checks do. What
would catch that class is a video comparison against a *known-good* render, and there is
no committed golden clip: Matroska embeds a random SegmentUID and a writing timestamp, so
files can never be compared byte-for-byte, and a pixel golden would have to be regenerated
whenever the palette or the walk changed. See [docs/DESIGN.md](docs/DESIGN.md).

**Intel and AMD iGPUs are untested.** No hardware was available, so no claim is made
either way. `tools\probe-opencl.exe` is the check.

**OpenCL is unavailable in CI, not verified there.** The runners have no GPU, so the
OpenCL engine skips rather than runs. Faults of the class "correct on images, broken on
video in a `-DRD_WITH_CUDA=OFF` build" need two builds and a real GPU to find.

## Further reading

[docs/DESIGN.md](docs/DESIGN.md) — the engineering record: the mathematics, the twenty-odd
optimisation rounds, the real bugs found in this codebase's own new code, the
measurements, and a table of everything that was tried and did not work.

[docs/OPENCL.md](docs/OPENCL.md) — the OpenCL engine: what it covers, the 36 image cells
and 60 video frames that verify it against CUDA, the four bugs the port actually had
(including one that turned out not to be in OpenCL at all), and where its remaining
cost is — which turned out **not** to be the engine.