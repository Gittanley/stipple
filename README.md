# rdither

Bit-exact GPU reimplementation of ImageMagick's Riemersma error-diffusion dither, for
images and video.

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

If you point it at an image and ask for 16 colours, you get back the same 16 colours
ImageMagick would have produced, the same pixels, byte for byte — not "similar", not
"close enough to pass". That is the design goal, and it is checked on every build.

```
rdither --colors 16 photo.png out.png
```

For a 3-hour video it runs about 56 frames per second on a modest 6-core machine, with
decode, dither and encode all happening at once.

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

A 18001-frame 1080p60 clip, 16 colours, on 6 physical cores:

```
frames     : 18001 in 349.97 s (51.4 fps)
busy time  : palette 27719 | decode 257150 | dither 226332 | encode 161856 | wall 349973 ms
```

The reported wall **includes the palette stage**. It did not for a while — the
denominator was measured from inside the pipeline, which starts after the palette is
already built, so two runs whose palettes differed by 25 s reported the same fps. If
a number here looks worse than one you remember, this is why.

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

---

## Further reading

[docs/DESIGN.md](docs/DESIGN.md) — the engineering record: the mathematics, the twenty-odd
optimisation rounds, the two real bugs found in this codebase's own new code, the
measurements, and a table of everything that was tried and did not work.

[docs/OPENCL.md](docs/OPENCL.md) — the OpenCL engine: what it covers, the 36 bit-exact
comparisons that verify it, the four bugs the port actually had, and what is left
(video, and palettes above 16).
