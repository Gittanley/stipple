# Image sources and licences

Every file in this directory is a **dithered output produced by `rdither`** from a
photograph or painting that is in the **public domain**. Nothing here is
third-party source code, and no licence other than the public domain applies to
the underlying images.

Each `*-strip.png` is a five-panel comparison, left to right:

1. `original` — the source, untouched
2. `image --im-palette` — image path, palette from ImageMagick's own sampling
3. `image` — image path, palette from rdither's octree
4. `video` — one frame of a dithered clip
5. `video --im-palette` — the same clip, palette from ImageMagick's sampling

Panels 2 and 3 are byte-identical: on the image path `--im-palette` is a no-op,
because there is one frame to sample and the octree already reproduces
ImageMagick's choice. The flag matters on video, where many frames are sampled.

| file | source | author | licence |
|---|---|---|---|
| `starry-strip.png` | *The Starry Night* (1889), via Wikimedia Commons | Vincent van Gogh | public domain |
| `bluemarble-strip.png` | *The Blue Marble*, Apollo 17, via Wikimedia Commons | NASA | public domain |
| `earthrise-strip.png` | *Earthrise*, Apollo 8, via Wikimedia Commons | NASA / Frank Borman | public domain |
| `mars-strip.png` | *Martian Sunset*, via Wikimedia Commons | NASA / JPL | public domain |
| `portrait-strip.png` | *Self-Portrait* (1856), via Wikimedia Commons | Konstantin Makovsky | public domain |

To regenerate a strip:

```
# panels 1-3, the image path
magick source.jpg -resize 400x -strip p1.png
rdither --colors 16 --im-palette source.jpg p2.png
rdither --colors 16 source.jpg p3.png

# panels 4-5, the video path
ffmpeg -loop 1 -i source.jpg -t 1 -r 10 -vf scale=400:-2 -pix_fmt yuv444p -color_range tv v.mp4
rdither --video --video-lossless --colors 16 v.mp4 v4.mkv
rdither --video --video-lossless --im-palette --colors 16 v.mp4 v5.mkv
ffmpeg -i v4.mkv -frames:v 1 p4.png
ffmpeg -i v5.mkv -frames:v 1 p5.png
```

`--video-lossless` matters: with libx264 the encoder invents colours around every
transition (measured 30,691 unique colours from 16), so the frame would not be a
16-colour image any more.

`-color_range tv` also matters. A clip encoded full-range comes out as `yuvj444p`,
which the video path crashes on — see the known issues in the README.

## A note on the PNG encoding

The strips are stored as `rdither` writes them, after `oxipng --opt max`. Several
reductions were measured and **rejected**:

* re-quantising to a 16-entry palette — smaller, but lossy (measured AE ≈ 0.025), and
  unacceptable in a repository whose central claim is bit-exactness;
* lossless palette PNG — larger, and still not pixel-identical.

Dithering deliberately introduces local variation that defeats palette-based
compression. That is the algorithm working, not an inefficiency to optimise away.

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

## Benchmark clip

`tools\bench.ps1` measures throughput on a **real** clip, because an fps figure taken
from a `lavfi` gradient describes a gradient. The `video-fps` suite refuses to run
without `-ClipFile` and refuses to run against the synthetic fixture, for the same
reason the image suites report dropped configurations instead of hiding them.

The clip is **not committed** — 529.9 MB — and is fetched on demand:

| field | value |
|---|---|
| title | *Waves off of dock at Boston harbor* |
| url | <https://upload.wikimedia.org/wikipedia/commons/a/ac/Waves_off_of_dock_at_Boston_harbor._1080P_60fps._No_Audio.ogg> |
| description page | <https://commons.wikimedia.org/wiki/File:Waves_off_of_dock_at_Boston_harbor._1080P_60fps._No_Audio.ogg> |
| author | Adam S. Keck (Commons user *Ghostis*) |
| licence | **CC BY-SA 4.0** |
| copyright status | **copyrighted** — *not* public domain, unlike every image above |
| size | 555,660,139 bytes (529.9 MB) |
| sha256 | `F2F353C3D88AE91D620AB1F57E6D8043688AB62E76EAEFC6C0CF6E7B257BDF55` |
| measured properties | 1920x1080, `r_frame_rate` 60000/1001, **3643** frames, 60.777383 s, Theora |

Properties are from `ffprobe -count_frames` on the downloaded file, not from the file
name: the name says "1080P 60fps", the container says 60000/1001.

Because the licence is share-alike rather than public domain, this clip is kept out of
the repository rather than committed alongside the examples; a redistributed derivative
would carry the same licence.

Benchmarks run against a **6-second excerpt**, so that a sweep over four engines and
three repetitions costs minutes instead of an hour:

| field | value |
|---|---|
| sha256 | `E0BF918DFBC3CA8E5C1D981D94E28E6AA36F1E9DB8BB47D56603033347F97644` |
| properties | 1920x1080, **360** frames, 6.006 s, `r_frame_rate` 19001/317 |
| encode | libx264 `-preset veryfast -crf 12 -pix_fmt yuv444p` |

```
ffmpeg -ss 10 -i waves.ogg -t 6 -an -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv444p waves_6s.mkv
```

The excerpt is re-encoded rather than stream-copied because `ogg`/`theora` does not cut
on a frame boundary predictably. `-crf 12 -pix_fmt yuv444p` keeps the source's own
colour space and introduces no chroma subsampling, so the frames reaching `rdither` are
not already degraded before the dithering is measured.