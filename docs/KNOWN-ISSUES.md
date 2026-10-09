# Known issues

Findings that are **measured and confirmed but deliberately not patched**, with the
measurement that confirms each one. A falsified claim belongs here rather than inside a
fix, where it would be invisible to the next reader.

---

## 1. The default palette sample count truncates coverage, and lands all-grey on last-third colour

**Status:** open. Confirmed by measurement 2026-10-08. Not patched -- every fix changes
which frames reach the quantiser and therefore changes every video output.

**Severity:** silent wrong output. Right frame count, exit 0, plausible picture, greyscale.

### The measurement

Fixture: 720 frames of 1080p, built for this test. Frames 0-479 flat grey, 480-599 red,
600-719 cyan. Colour occupies the last 240 frames = **33% of the clip**, all of it in the
final third. Built with three `color` lavfi inputs and `concat`, encoded `libx264 -g 30`.

`rdither --video --colors 16 --engine cuda --palette-only --palette-max-samples N`:

| samples | colours | mean saturation | near-neutral | montage | clip covered |
|---:|---:|---:|---:|---:|---:|
| **256 (the default)** | **1** | **0.0%** | 1 | 64.0 MiB | **71%** |
| 128 | 2 | 50.0% | 1 | 33.0 MiB | 89% |
| 64 | 3 | 66.7% | 1 | 16.0 MiB | 98% |
| 32 | 3 | 66.7% | 1 | 9.0 MiB | 98% |
| 16 | 3 | 66.7% | 1 | 4.0 MiB | 100% |

At the default the palette is **one grey colour** and the whole clip renders greyscale.
Lowering the sample count *fixes* it.

### Cause

`src/rd_video.cpp:1569` computes the stride as

```c
const long long step = (info.frames > 0) ? (info.frames / want) : 1;
snprintf(filter, ..., "select=not(mod(n\\,%lld))", step < 1 ? 1 : step);
```

which spreads samples across the *whole* clip, and then `:1617` passes
`-frames:v want`, which makes ffmpeg emit the **first `want` matches and stop**. With
`step = 720/256 = 2` the matches run 0,2,4,...,718 but only the first 256 are taken, so the
**highest frame sampled is 510**. Frames 511-719 are never decoded into the palette.

The two mechanisms contradict each other: a finer stride spreads samples more thinly, but
the count cap truncates proportionally earlier. **Coverage improves as the sample count
falls**, which is the opposite of what the flag means.

| samples | step | last frame sampled | coverage |
|---:|---:|---:|---:|
| 256 | 2 | 510 | 71.1% |
| 128 | 5 | 635 | 88.9% |
| 64 | 11 | 693 | 97.8% |
| 32 | 22 | 682 | 97.8% |
| 16 | 45 | 675 | 100% |

### Confirmed to change output -- MEASURED, not asserted

An earlier draft of this file claimed "every fix changes every video output" without
testing it. It has now been measured, and the claim holds -- but by a smaller margin than
"changes" implies on the palette, and by a larger one on the picture.

Chain, each link measured separately:

| link | evidence | result |
|---|---|---|
| sampled pixels differ | montage SHA256, truncated vs full | `DA6DA4F801DA12F4` vs `B42DEC6A10B9D6CC` |
| palette entries differ | `--dump-palette`, entry by entry | **16 of 16** differ, max component delta **271.4** of 65535 (0.4%) |
| output pixels differ | `tools\probe-frame-diff.ps1` | **0.97%-1.09%** of components per frame, max delta **52652** (80% of full range) |

The last row is the one that matters. A palette moving 0.4% per component produces output
that moves **80% of range** on the pixels that move, because Riemersma error feedback
propagates a changed palette entry into every later pixel in its neighbourhood. So the
change is invisible in the palette and unmissable in the picture -- which is why a palette
metric (saturation 84.5% vs 84.4%) is the wrong instrument for judging it, and why the
fix cannot be waved through on "the colours barely moved".

Per-frame divergence also **increases along the clip**, which is the shape of the defect:

```
frame   0   0.972%      frame 359   1.092%
frame 120   0.972%      frame 479   1.092%
frame 240   0.972%      frame 599   1.092%
```

Frames past ~350 are the ones the truncated palette never sampled, so they are the ones it
fits worst. A comparison that read only frame 0 would under-report this by 11%.

### How this was measured, and how it was first measured wrongly

The first attempt hashed both outputs (`MD5=64b14f…` vs `MD5=951c97…`) and then decoded
**both files to raw rgb48le to count differing pixels: 7119.1 MB each, 14.2 GB total**,
on a machine whose owner has already had one crash. The hash had already answered "they
differ"; the full decode was for magnitude, and could have been got from three frames.

`tools\probe-frame-diff.ps1` now does it in **71 MB per side** by sampling six frames
spread across the clip (first, last, and evenly spaced between) and comparing byte by
byte, reporting the count and the worst delta.

**A whole-stream hash is the wrong instrument for this question**, for two reasons that
were both paid for on 2026-10-08:

1. It is all-or-nothing. It cannot say *where* or *how much*, and those are exactly what
   decide whether a change is safe.
2. It cannot distinguish one pixel from a cascade. Here the two ends of that range are
   "1.2% of the picture moved by 80% of range" and "identical".

Six controls were run before this probe was trusted, covering exit 0 (identical pair),
exit 1 (the defect pair), exit 2 (missing file, non-video input, `-Frames 1` rejected), and
a 3-frame clip where the request for 6 degrades to sampling all 3 rather than lying about
coverage.

### Why it stayed hidden

It needs colour concentrated in the final third. On footage whose colour is distributed,
truncation costs some diversity and nothing visibly breaks. The repo's bench clip measures
**84.4-84.6% mean saturation across 16 -> 256 samples** -- flat -- because one scene of the
project's own fixture has nothing to lose from sampling fewer frames.

`src/rd_video.cpp:1550-1555` already predicted this and named the reason the earlier
30-vs-4096 evidence was worthless: that measurement was taken on a clip *already saturated*,
so it recorded a ceiling, not a floor. The prediction is confirmed here.

**Consequence for the sample-count default.** There is no evidence that 256 samples buys
anything, and direct evidence that at 256 it loses the entire final third of the clip. Any
future discussion of lowering the default has to start from this table, not from the bench
clip's flat saturation curve.

### Candidate fixes, none applied

Each changes which frames reach the quantiser, so each changes palettes and every video
output. This is a decision about output, not about performance.

1. **Strided to the end.** Keep the count, choose the stride so the last sample lands on the
   final frame (`step = (frames-1) / (want-1)`), letting `-frames:v want` no longer
   truncate short. Fixes coverage; changes every existing palette.
2. **One sample per stride position, cap the count instead of the positions.** Same effect,
   different code shape.
3. **Sample by seek** (`sample_by_seek`, `rd_video.cpp:1622`) does reach the end, but only
   engages above 5000 frames, so this 720-frame clip never takes it.

### Not yet tested

- Whether the by-seek path has the same truncation. Its stride arithmetic differs (one seek
  per sample rather than a `select` stride), so it may be immune, and that has not been checked.
- Behaviour on VFR sources, where `info.frames` may be an estimate rather than a count.
- ~~Whether an alpha-bearing source makes `SourcePixFmtHasAlpha` correct~~ -- **resolved
  2026-10-08.** `tools\make-video-fixtures.ps1` now builds `clip_alpha.mkv` (`yuva420p` in
  ffv1) and asserts both that the fixture carries alpha and that rdither keeps the
  4-channel palette path for it. See item 3.

---

## 2. `nb_frames` is read but never counted, so a container that declares none yields a 1-frame palette

**Status:** open. Confirmed by measurement 2026-10-08. Not patched -- see "why not".

**Severity:** silent wrong output, same class as item 1. Right frame count in the file, exit 0,
palette built from one frame.

### The measurement

Built a `yuva420p` / ffv1 clip (needed for item 3 below, and the first time this
repository had any fixture carrying alpha):

```
ffprobe nb_frames      : N/A      <- what rdither reads
ffprobe -count_frames  : 20       <- reality
rdither prints         : 320x240, ffv1, yuva420p, 0 frames @ 10.000 fps
rdither prints         : [video] palette: sampling 1 frame(s) of 0
```

The palette was built from **one** frame and the clip rendered 20, exit 0.

### Cause

`rd_video.cpp` takes `info.frames` from ffprobe's declared `stream=nb_frames` and never
falls back to `-count_frames`. For a container that does not declare the field the value is
`N/A` -> 0, and the clamp chain at `rd_video.cpp:1274-1276` degrades `want` to its floor of 1.

### Why it is narrow but not exotic

`nb_frames` is populated by mp4/h264 (the project's shipped shape) and absent from
mkv/ffv1. So the trigger is "an mkv", which is a container this project already uses for its
own fixtures -- `tests\clip1920_audio.mkv` and the new alpha fixture are both mkv. The
18001-frame clip and the 600-frame bench clip are mp4 and are unaffected.

### Why not patched

A `nb_frames` -> `-count_frames` fallback means a **real** frame count on every such source,
which changes `want`, which changes the montage, which changes the palette and therefore
every video output from those sources. Same trade as item 1, same reason to defer: it is a
decision about output, not about performance. Unlike item 1 this one has a defensible
correct answer -- "count the frames" is right where "sample the whole clip" is arguable --
so it is the more likely of the two to be approved.

### Candidate fix, not applied

In `VideoProbeInfo`, when the declared `nb_frames` is absent or 0, re-probe the same stream
with `-count_frames`. Cheap (one extra demux pass, no decode), and the packet-index pass
already performed for VFR detection is the same shape of work.

### Not yet tested

- Whether `-count_frames` on a long mkv is fast enough to be acceptable. The VFR probe
  already walks the packet index, so it may be, but it has not been measured.
- Whether any real mkv source in this project's intended use declares `nb_frames`.

- ~~Whether the by-seek path has the same truncation~~ -- **resolved 2026-10-08: it is NOT affected.** Above 5000 frames the palette samples by seek, and each sample is an independent `-ss` to `s * step`, so the last one lands near the clip's end. There is no `-frames:v` count to truncate the run. The defect is confined to the sequential path, which is the short-clip path.
- Whether an alpha-bearing source makes `SourcePixFmtHasAlpha` correct. **No fixture in the
  repository carries an alpha channel**, so that branch has never executed. See the
  `9df7e34` commit message.
- Behaviour on VFR sources, where `info.frames` may be an estimate rather than a count.
---

## 3. `SourcePixFmtHasAlpha` had no fixture and no check -- now it has both

**Status:** closed 2026-10-08. Not a defect in the code; a gap in the checks around it.

### What was wrong

`SourcePixFmtHasAlpha` (`rd_video.cpp`) gates the 3-channel palette path added in `9df7e34`.
It is deny-by-default, so an unrecognised format keeps 4 channels and keeps today's
behaviour. That is the right default, and it was **completely untested**: no fixture in the
repository carried an alpha channel, so the deny branch had never executed. A guard that
defaults wrongly and is never exercised looks exactly like one that works.

The `9df7e34` commit message says so, and said so at the time. This closes it.

### What exists now

`tools\make-video-fixtures.ps1` builds a third fixture, `clip_alpha.mkv` -- `yuva420p` in
ffv1, 320x180, 30 frames -- and asserts three things:

1. **It carries alpha.** `pix_fmt` must match `yuva*`. This is the assertion that matters,
   because the failure mode is an encoder that accepts `-pix_fmt yuva420p` and emits
   something else: `libvpx-vp9` was tried and silently produced `yuv420p`, a file with the
   right shape and no alpha. A fixture that looks healthy and cannot test the thing it
   exists to test is worse than no fixture, because it reports a pass.
2. **rdither keeps the 4-channel palette path** for it, checked by tracing the actual
   `palette-frames` spawn and asserting it does not ask for `rgb24`.
3. It emits `CLIP_ALPHA=` for `verify.ps1` to read back, matching the existing
   `CLIP=` / `CLIP_AUDIO=` convention.

Verified in both directions: an alpha source (`yuva420p`) gets `-pix_fmt rgba`, and the
no-alpha source (`yuv420p`) gets `-pix_fmt rgb24`. A guard proven only in one direction is
half a guard.

### Note on the alpha path being provably unchanged

For an alpha source `pal_ch == 4`, so the format string is `" -pix_fmt rgba"` -- which is
exactly what the code hardcoded before `9df7e34`. The alpha path cannot have regressed, by
inspection rather than by test. That is why this is a check gap and not a suspected defect.

### A regression introduced and fixed while adding this

Adding the guard assertion exposed that `-Ffmpeg <bad path>` fell through to whatever
`ffmpeg` was on PATH and **exited 0**. That is the floor this script is supposed to have,
and it was missing. Now:

```
-Ffmpeg C:\nonexistent\ffmpeg.exe   ->  "cannot run: the ffmpeg given with -Ffmpeg does
                                          not exist"   exit 2
```

Both paths re-verified after the fix: bad path exits 2, normal run exits 0.


---

## 4. The palette/reader overlap is implemented, correct, and MEASURED SLOWER

**Status:** implemented and byte-exact; **shipped opt-in** because it is a regression when
on. Not a defect in the code -- a defect in the premise the change was built on.

### What was built

`RD_PALETTE_OVERLAP=1` starts the palette build on its own thread (`src/rd_cli.cpp`) so the
reader fills its bounded queue while it runs. `PaletteGate` (`include/rd_video.h`) carries the
in-flight palette; the dither workers call `gate.Wait()` immediately before dispatch, so a
worker holding a decoded batch waits for the palette it is about to use. The report and both
exit paths join the thread.

### The measurement, which is the finding

Two clips, so the finding is not one machine's mood on one afternoon.
`--colors 16 --engine cuda --video-lossless`:

| clip | arm | fps | palette | decode | **dither** | encode | wall |
|---|---|---:|---:|---:|---:|---:|---:|
| `testsrc` 300 f, 0.3 MiB | default | 38.9 | 3517 | 3596 | **2999** | 2266 | 7711 |
| `testsrc` 300 f | `=1` | **25.5** | 3951 | 4144 | **6059** | 2454 | 11744 |
| `cpubench` 600 f, 14.8 MiB | default | 47.7 | 3708 | 8227 | **6203** | 4373 | 12568 |
| `cpubench` 600 f | `=1` | **37.3** | 4197 | 5125 | **9331** | 8100 | 16069 |

**34% slower on one clip, 22% on the other. The palette stage gets slower under concurrency in
both** (3708 -> 4197 ms), and the wall gets worse in both.

### The mechanism, which the worker-time breakdown makes visible

**Dither time roughly doubles: 6203 -> 9331 ms on `cpubench`.** The palette thread is not
filling idle capacity, it is taking cores away from the dither workers, so total work goes *up*
and every later stage waits behind it. That is the whole of the loss, and it is visible in one
number that the earlier measurements never looked at.

Decode time *appears* to drop (8227 -> 5125 ms). That is not a saving: the reader blocks
earlier against a full queue and so does less work before being throttled. Reading it as a win
would be reading a bottleneck as an improvement.

### Why, and why my earlier contention measurement did not find it

An earlier measurement ran the two **ffmpeg decoders** concurrently and found no contention
(palette 1864 -> 1800 ms, reader 4179 -> 4240 ms). That was true and it was too narrow: what
actually contends is the palette's **ImageMagick quantise** over a 64 MiB montage against a
reader pipeline that **already saturates every core**. The reader is the pipeline's floor, so
running a CPU-heavy stage beside it does not fill idle time -- there is no idle time. The two
overlap by contending.

The same pattern is recorded elsewhere in this file's history: a 1.7% figure that was really
the image path, and a saturation metric that could not see the failure it existed to catch.
The general failure is measuring a *component* and generalising it to the *system*.

### The probe's ratio is not a reliable direction, and that is worth knowing

An earlier run of `tools\probe-palette-overlap.ps1` on its own 1080p fixture reported
**1.452x -- the overlap winning**, against 0.78x measured here minutes later on `cpubench`.
Both clips lose when measured with the worker-time breakdown above; the probe run was wrong,
or at least not measuring a stable quantity.

So the probe's **timing ratio must not be read as directional.** What it does assert -- the
arms differ by far more than this machine's drift, so the flag demonstrably reached the code
-- is sound and is the part that gates. A probe that reported a confident sign here would be
reporting noise, and a reader who trusted it would draw the wrong conclusion, which is how
the 1.452x nearly became the documented result.

### Correctness, which does hold

`tools\probe-palette-overlap.ps1`: **0 of 37,324,800 components differ** across six frames
spread over the clip, worst delta 0. The two arms also differ in wall time by far more than
this machine's 10-25% drift, which is what proves the flag reached the code -- without that,
a flag that silently did nothing would make the A/B compare one binary with itself and report
a perfect pass. A `--colors 32` negative control confirms the comparison detects differences.

### Why opt-in rather than reverted

The code is correct and the arithmetic can hold on a machine that is not already saturated
(a GPU with idle cores, or a reader bound on I/O rather than CPU). Shipping a measured 22%
regression as the **default** is not defensible, so the flag is inverted to
`RD_PALETTE_OVERLAP=1` and costs nothing when unset.

### What the spec got wrong

`docs\superpowers\specs\2026-10-08-palette-read-overlap-design.md` predicted a 1091 ms
(8.4%) win. The mechanism it assumed -- the reader runs at full speed beside the palette --
is exactly what does not happen here. Section 6 of that spec listed CPU contention as
"resolved" on the strength of a decoder-only measurement, which is the error above.

### Not tested

- Whether a machine with idle cores sees the predicted win. Untestable here: this one's reader
  saturates.
- The reader-fails-while-palette-builds path, and the interrupt-during-palette-window path, are
  implemented but not exercised by any probe.

---

## 5. `--palette-budget-ms` bounded nothing until 2026-10-09, and the fix makes output load-dependent

**Status:** fixed. The flag now bounds the stage. The price is that a *lowered* budget makes
output a function of machine speed, which is recorded here rather than left implicit.

### What it did

`rd_video.cpp:1690` computed a sample COUNT from a hardcoded per-sample cost:

```cpp
const int affordable = static_cast<int>((budget_s - 0.02) / 0.12);
```

No clock was ever read. `budget_s` was never compared against elapsed time anywhere in the
sampling path, so there was no deadline: the count was chosen up front and the stage then ran
for as long as it took. Meanwhile `rd_cli.cpp:125` documented it as "time budget for that
sampling" and README.md:582 as "as many as a 60-second budget allows".

Two things were wrong with the surrounding comment, and the first is the interesting one. It
claimed the budget "degrades gracefully -- a slow disk takes fewer samples instead of taking
the same time it always took." That is backwards: a slow disk took the SAME number of samples
for LONGER. The flag had a coverage effect and no time effect, and the comment attributed the
time effect to it.

It was also **inert at the default**. `affordable = (60 - 0.02) / 0.12 = 500`, a 600-frame clip
offers 600, and `--palette-max-samples` caps at 256, so 256 wins and the budget never binds.
It only takes effect below about 30.7 s.

### What it does now

A real deadline, checked in both samplers, never allowed to take the sample count below one,
and reported on stderr when it fires. `RD_PALETTE_DEADLINE_MS` overrides the deadline without
touching the count, which is what makes it testable at all: `--palette-budget-ms 1` collapses
the arithmetic to one sample and so cannot distinguish "the count was reduced" from "the
deadline fired".

Truncation is safe to do because the montage is padded rather than left short --
`rd_video.cpp:2045-2064` repeats the last sampled tile into every unfilled cell, precisely
because "Black is a real colour to the octree, so those cells spend palette entries on nothing."

### The cost, stated plainly

**A real deadline is load-dependent output.** Fewer samples means a different montage, so a
different palette, so different pixels. On this machine, which is rarely quiet, a lowered
budget therefore stops being reproducible.

This is bounded: at the default the flag is still inert, `--palette-max-samples` binds first,
and the default path's output is unchanged. It bites only when the budget is deliberately
lowered below ~30.7 s. The alternative -- leaving the flag lying -- was rejected because a
flag that bounds nothing is not a smaller problem than a flag that bounds time; it is a
problem nobody can see.

### Not tested

- Behaviour when the deadline lands mid-montage on a clip where the by-seek workers finish out
  of order. `sampled` counts LEADING filled cells, so a gap truncates at the gap rather than
  at the deadline; that is the pre-existing rule and the probe exercises only the common case
  where every seek succeeds.
- Whether 0.12 s per sample is still the right constant. It is unmeasured and it is the
  load-dependent part of the count.
