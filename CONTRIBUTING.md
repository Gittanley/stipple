# Contributing

## What this is

A bit-exact reimplementation of ImageMagick's Riemersma error-diffusion dither, for
images and video, on CUDA and OpenCL. It reproduces
`magick in.png -dither Riemersma -colors 16 out.png` pixel for pixel on the sequential
engines, and is deliberately approximate on the parallel ones — see
[Where bit-exactness holds](README.md#where-bit-exactness-holds) before judging any
output, and before claiming anything about it.

## Build

Windows only, and the reason is a toolchain constraint rather than a preference: nvcc
accepts only MSVC as its host compiler, and the ImageMagick import libraries are MSVC-ABI.
MinGW and clang do not work.

You need:

| | |
|---|---|
| **Visual Studio 2022** | any edition, with the "Desktop development with C++" workload |
| **CUDA toolkit** | 13.x, matching your driver |
| **ImageMagick** | 7.x, **Q16-HDRI** — the Q8 build will not match bit for bit |
| **OpenCL** | headers and loader; the `OpenCL-SDK-*-Win-x64` tree in some working copies is a local convenience and is not part of this repository |

```
git clone https://github.com/Gittanley/stipple
cd stipple
build.cmd
```

If ImageMagick is installed somewhere unusual:

```
set IMAGEMAGICK_ROOT=C:\path\to\ImageMagick-7.x.x-Q16-HDRI
build.cmd
```

## Test

```
powershell -File verify.ps1
```

Exit 0 means everything that could run, ran and passed. The suite is honest about the
rest: a check that cannot run says so and is reported as **skipped**, never as a pass and
never as a failure. From a clean clone you should expect:

| Check | On a fresh clone |
|---|---|
| 135 bit-exact cases vs ImageMagick | runs — `verify.ps1` generates its own fixtures with `magick` |
| unvisited-pixel provenance, 9 geometries | runs — also generates its own fixtures |
| image determinism | **skipped** — needs fixtures from `tools\probe-opencl-exact.ps1` |
| OpenCL-vs-CUDA, 54 image cells | **skipped** — same reason |
| video, both data paths + determinism | **skipped** — needs a video clip, which is not committed |

Run `tools\probe-opencl-exact.ps1` once to generate the image fixtures and the first two
skips go away. The video skips stay until you point the probes at a clip of your own; the
probes take a `-Clip` argument and say so when it is missing.

Exit 2 from any probe means "cannot run" and is deliberately distinct from exit 1, "ran
and differed". A test that cannot run and says so is the difference between a missing
input and a silent hole in the suite.

## Conventions that are load-bearing

**The encoder preset stays `veryfast`.** It was measured as both faster *and* smaller
than `medium` (see [docs/DESIGN.md](docs/DESIGN.md) §10). Do not change it as an
"improvement" without that measurement, because the obvious reading is that `medium`
must be better.

**A measurement without its conditions is not evidence.** Every performance number in
these docs records the machine, the command, and the content it was measured on. Several
numbers in earlier drafts had to be thrown out because the data path differed between the
two things being compared — the same arithmetic, a different pipe. If you add a number,
add the conditions with it.

**Check what your instrument measures before believing it.** Six times in this project's
history the *instrument* was wrong while the code was fine. The two that cost the most:
`magick compare -metric AE` reports **one frame** on a multi-frame file, not the clip
(it printed `378618 (0.18259)`, and 0.18259 × 2073600 is exactly one 1920×1080 frame, so
every AE figure taken that way understated the damage by about 60×); and a determinism
probe with a **fixed** temp path collides with a second concurrent instance, which
produced output that read like "CUDA failed, OpenCL passed". Both are written up in the
comments where they can bite again.

**Copy, do not derive, when bit-exactness depends on a constant.** The libswscale matrix
in `rd_blocks_cuda.cu` / `rd_opencl.cpp` and the `octicost`/weights constants are
*transliterated* from their sources, and `tools\probe_swscale_matrix.cpp` prints them for
checking. Two of the six were got wrong by hand before they were pinned. If you re-derive
one, you will produce something that looks right and is not.

**A test that cannot fail is worse than no test.** If you add a check, make sure it goes
red when the thing it checks is broken — and if you deliberately report a known defect
instead of failing, say so in the output, because a green light that means "I looked away"
is how suites rot.

## Adding a dither algorithm

Three steps, and the seam is [include/rd_plugin.h](include/rd_plugin.h):

1. Copy `examples/bayer_dither.cc` to `src/` and rename it.
2. Write the kernel. The one rule that is easy to get wrong: `frames` is how many
   independent frames you are handed, and each must keep its own state. Sharing a buffer
   across the batch makes frame *f* depend on frame *f−1*, which breaks the parallel block
   partition and `--resume`.
3. Register it in the same file and add the file to `RD_SOURCES` in `CMakeLists.txt`.

Then add it to `tools\probe-determinism.ps1`'s algorithm list. That is not optional
paperwork: the engine sweep varies `--engine`, and every plugin is CPU code reached by
the same engine, so a broken algorithm is otherwise indistinguishable from a correct one.

Two traps that have already been hit, both in shipped-looking code:

- **A threshold in the wrong units is a no-op.** Adding `(t - 0.5)` to a pixel whose
  channels are 0–65535 changes the value by less than one quantum, the nearest palette
  entry never changes, and the output contains no dither at all. Threshold against the
  two *bracketing* entries instead.
- **A rank that does not fit its type.** `unsigned char` holding ranks up to 4096
  truncates them, and the symptom is not a degraded pattern but two filters that should
  differ producing byte-identical output.

## Known open defects

Listed in [docs/DESIGN.md](docs/DESIGN.md) §12 and in the header comment of
`tools\probe-video-determinism.ps1`. The short version: `--no-gpu` **video** is not
trustworthy — it is non-deterministic and misreads the default `yuv444` input. It is fine
on images, where it is bit-exact against CUDA. Treat it as CUDA-only for video until
`tools\probe-video-determinism.ps1` passes for the `cpu` case.
