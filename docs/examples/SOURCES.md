# Image sources and licences

Every file in this directory is a **dithered output produced by `rdither`** from a
photograph or painting that is in the **public domain**. Nothing here is
third-party source code, and no licence other than the public domain applies to
the underlying images.

The `*.png` files are the full-size results; the `*.thumb.png` files are
**independently dithered at 480 px**, not downscaled copies of the large ones, so
a thumbnail is a real result at that size rather than a shrunken approximation.

| file | source | author | licence |
|---|---|---|---|
| `starry.png` | *The Starry Night* (1889), via Wikimedia Commons | Vincent van Gogh | public domain |
| `bluemarble.png` | *The Blue Marble*, Apollo 17, via Wikimedia Commons | NASA | public domain |
| `earthrise.png` | *Earthrise*, Apollo 8, via Wikimedia Commons | NASA / Frank Borman | public domain |
| `mars.png` | *Martian Sunset*, via Wikimedia Commons | NASA / JPL | public domain |
| `portrait.png` | *Self-Portrait* (1856), via Wikimedia Commons | Konstantin Makovsky | public domain |

`video-frame.png` is a single frame extracted from a dithered video clip — the same
*Starry Night* source, run through the video path rather than the image path:

```
ffmpeg -loop 1 -i starry.jpg -t 2 -r 25 -vf scale=1280:-2 -pix_fmt yuv444p clip.mp4
rdither --video --video-lossless --colors 16 clip.mp4 clip.mkv
ffmpeg -i clip.mkv -vf "select=eq(n\,25)" -frames:v 1 video-frame.png
```

`--video-lossless` matters: with libx264 the encoder invents colours around every
transition (measured 30,691 unique colours from 16), so the frame would not be a
16-colour image any more.

Regenerate any of the stills with:

```
rdither --colors 16 <source.jpg> <name>.png
```

## Why these images

They were chosen for the properties a *dithering* test actually needs — detail,
flat regions, shading, and texture — not for familiarity:

| image | what it stresses |
|---|---|
| `starry` | smooth sky gradients — where error diffusion is most visible, plus heavy brush texture |
| `bluemarble` | broad saturated colour fields and gentle ramps; almost no hard edges |
| `earthrise` | mostly flat black sky against a small bright object; extreme dynamic range |
| `mars` | a single dominant hue ramp with fine atmospheric detail |
| `portrait` | skin tones and soft shading |

## A note on the PNG encoding

The files are stored exactly as `rdither` writes them. Several size reductions
were measured and **rejected**:

* `oxipng --opt max` — **0.0% saved**. The output is already at its lossless floor.
* Re-quantising to a 16-entry palette (PNG8) — **39% smaller but lossy**; it
  changes pixels (measured AE ≈ 0.025 on *Starry Night*) and is therefore
  unacceptable in a repository whose central claim is bit-exactness.
* Lossless PNG8 — both **larger** than the original (2.85 MB vs 1.68 MB) and
  still not pixel-identical.

The dithered images contain ~6,875 unique colours, because dithering deliberately
introduces local variation that defeats palette-based compression. That is the
algorithm working as intended, not an inefficiency to optimise away.