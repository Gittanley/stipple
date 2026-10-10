# Look cheatsheet

Every knob that changes how your video *looks*, what it costs, and how to tell whether it
worked. Written 2026-10-10 after a session where the answer to "why does my video look like
that" was guessed four times before it was measured.

**Read this before changing a flag.** Most "the dither got weaker" reports are a *palette*
or *encoder* effect wearing a dither's clothes. Section 3 tells them apart.

---

## 1. The one command you probably want

```
rdither.exe --video --engine blocks --colors 16 --im-palette \
            --video-pix-fmt yuv420p --palette-frames 128 in.mp4 out.mp4
```

Four flags, each earning its place:

| flag | why |
|---|---|
| `--engine blocks` | GPU. Same partition as `cuda` on the video path. |
| `--im-palette` | palette built the way ImageMagick's own `+append -colors N -unique-colors` builds it. |
| `--video-pix-fmt yuv420p` | **the big one.** See section 2. |
| `--palette-frames 128` | keeps whole-clip coverage, avoids the palette crowding that 256 causes. See section 4. |

Measured against `J:\video renders\final_result_FHD_60fps.mp4` (the reference you like),
mean absolute pixel delta at frame 30: **4.95** for 128, 5.38 for 256, 6.81 for 30.

---

## 2. Why your video looks "grainy" — and the one flag that fixes it

**This is the single biggest lever and it is not the dither.**

| pixel format | unique colours in one frame | look |
|---|---:|---|
| `yuv444p` (default) | 28,012 | every dither dot preserved exactly; busy, speckled |
| `yuv420p` | 8,877 | chroma averaged over 2x2; flat, clean, smooth |

Proof it is the format and not the dither: take an existing `yuv444p` render and
**re-encode it to 4:2:0 without touching a pixel of the dither**. Unique colours fall
28,012 → 8,877. The look changes completely.

The reference file you like is `yuv420p`. Every render you did before this session was
`yuv444p`. That alone is most of the difference you were seeing.

```c
--video-pix-fmt yuv420p      // clean and flat; chroma is subsampled
--video-pix-fmt yuv444p      // exact; busier and noisier   <- current default
--video-lossless             // ffv1 yuv444p, exact AND lossless, ~28x bitrate
```

**The cost, stated plainly:** 4:2:0 throws away chroma resolution, so the 16-colour palette
does not survive *exactly*. The tool warns about this on every 4:2:0 run and it is correct
to. You accept it knowingly or you pay the bitrate.

### Lower bitrate flattens further

The reference is **12.4 Mbit/s**; yours was **16.7**. Lower bitrate means x264 discards more
of the dither speckle.

```
--crf 24        # flatter, smoother, closer to the reference
--crf 18        # current default
```

---

## 3. "The dither looks weak/strong" — diagnose before you touch a knob

Run this and read the **swatches** line. Every video run prints it:

```
palette    : 16 colours from 128 sampled frame(s), ... mean saturation 24.3%, 9 near-neutral
swatches   : 0E0D0C 1A2B28 2E312F 261D16 4E2319 284637 504E36 32524E
swatches   : 4E514E 6E706F 5F645B 323A44 8C8874 6D7687 AFB3B3 6C816A
```

| what you see | what it means | fix |
|---|---|---|
| swatches are mostly grey (`0E0D0C`, `2E312F`, `6E706F`…) | correct — your footage *is* grey. Dithering between two near-identical greys is invisible **by construction**. | nothing is broken. See section 4 for more contrast. |
| swatches are wildly saturated / all one hue | **broken palette.** | check section 5. |
| palette line says a **high** mean saturation but the picture is grey | **broken palette.** A wrong read inflates the number. | section 5. |
| picture is speckled/busy | format, not dither | section 2. |

**Mean saturation has no baseline.** 49% is fine for a forest, terrible for a face. The
swatches are the real signal; the number is a hint. Do not tune to the number.

### Dither strength itself

```
--diffusion 1.0     # current default. IM's dither:diffusion-amount. Unchanged since day one.
--blocks 512        # curve positions per walk block. current default.
--blocks 32         # 11% faster, deviates 3.4x more from IM's output
```

`--blocks` and `--diffusion` have **not** changed since the initial commit. If the dither
"feels different", the palette or the pixel format moved, not these.

To confirm the engine is exact, on any frame:

```
rdither.exe --colors 16 --verify frame.png out.png
```

`BIT-EXACT` means byte-identical to ImageMagick. On the `cpu` engine that check passes with
`AE=0, max channel delta=0`. The GPU engines are *not* bit-exact — `--blocks 512` deviates
~1.7% of pixels from IM on a gradient, `--blocks 32` ~6.1%.

---

## 4. `--palette-frames`: the coverage-vs-contrast trade

**Measured on your clip** (`orig.mp4`, 18005 frames, 18,005 = 5 min):

| frames | palette mean pair-distance | dark entries | max luminance | cost |
|---:|---:|---:|---:|---|
| 16 | 109.3 | 8 | 215 | fast |
| 30 | 104.4 | 8 | 215 | fast |
| 60 | 108.3 | 9 | 215 | fast |
| 128 | 92.7 | 7 | 178 | ~11 s |
| **256 (default)** | **93.6** | **6** | **175** | ~25 s |

Fewer samples → wider spread, more contrast, keeps a bright highlight. 256 crowds the
palette into the mid-tones.

### Do NOT use small values on a long clip

`--palette-frames 30` on an 18,005-frame clip reads the **first 30 frames** and pads the
rest by repeating the last tile. You asked for 30 and got **15 distinct samples plus 15
copies**. Observed result: two bogus near-white entries (`D7D7D7`, `A7A9A2`), mean
luminance dropped 87.7 → 73.1, and the picture got *harsher*, not better.

This is `docs/KNOWN-ISSUES.md` item 1. Small sample counts are only correct for **short
clips**, which is why `--help` says 30 "reproduces the reference pipeline" — a pipeline
that quantises 30 named *files*, not 30 frames of one long clip.

**Rule: `--palette-frames 128` is the low end you should use on a clip over a minute.**

---

## 5. Symptom → cause quick table

| symptom | cause | fix |
|---|---|---|
| everything green-grey, unrecognisable | **palette read wrongly** (channel mismatch) | see the note below; this was a real bug, fixed 2026-10-09 |
| palette line shows *higher* saturation than usual, picture worse | same bug — the wrong read *inflates* saturation | same |
| speckled, busy, noisy | `yuv444p` | `--video-pix-fmt yuv420p` |
| flat, dull, few colours | `yuv420p` already; or `--crf` too low | raise `--crf`, or `yuv444p` |
| dither invisible | palette too grey / colours too close | fewer samples (§4) or more colours |
| only 15 of 30 samples used | padding; clip too long for that count | raise `--palette-frames` |
| colour missing from last third of clip | coverage truncation | raise `--palette-frames`; see KNOWN-ISSUES 1 |
| `--verify` fails on CPU but passes elsewhere | engine divergence, not a regression | use `--engine cpu` for exactness |

---

## 6. Knobs NOT to touch without reading

| flag | note |
|---|---|
| `--palette-tile` | measured **no effect** on your clip (23.9% at 128, 180 and 360 alike). `--palette-tile 0` collapses to 3 samples and takes 162 s. Leave at default. |
| `--palette-max-samples` | caps at 256 by default; rarely the thing you want. |
| `--input-mode` | affects decode path and speed, and `yuv420` vs `yuv444` differ by ~34 dB. Leave alone unless chasing speed. |
| `--queue-depth`, `--batch-frames` | throughput only, no visual effect. |

---

## 7. Cost reference (this machine, measured)

| operation | wall |
|---|---|
| full 18005-frame render, `--video-lossless` | ~185 s |
| full 18005-frame render, `yuv420p` h264 | ~170 s |
| palette stage alone, 256 samples | ~25 s |
| palette stage alone, 128 samples | ~11 s |
| one 1080p 60-frame render, cuda | ~2.5 s |
| one 1080p 60-frame render, cpu | ~2.3 s |

Palette stage scales roughly linearly in samples; the render stage does not care.

---

## 8. The one thing that was actually broken

On 2026-10-09 the by-seek palette sampler asked ffmpeg for **4 channels** (`rgba`) while
reading into a **3-channel** buffer, so on any clip over 5000 frames every montage cell was
an RGBA byte stream misread as RGB triples. Result: green-grey palette, 7 of 16 entries pure
green, while reporting "mean saturation 49.0%" the entire time.

Fixed, and gated by `tools/probe-palette-deadline.ps1`, which now builds a 5100-frame clip
where every frame is identical — so both samplers must agree, and a misread cannot fake it.

Full writeup: `notes/palette-byseek-channel-mismatch.md`.