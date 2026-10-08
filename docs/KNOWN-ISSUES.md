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

- Whether the by-seek path has the same truncation (its stride arithmetic differs).
- Whether an alpha-bearing source makes `SourcePixFmtHasAlpha` correct. **No fixture in the
  repository carries an alpha channel**, so that branch has never executed. See the
  `9df7e34` commit message.
- Behaviour on VFR sources, where `info.frames` may be an estimate rather than a count.