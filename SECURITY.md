# Security Policy

## Reporting a vulnerability

**Do not open a public issue for a security problem.** Use GitHub's private reporting on
this repository — the "Security" tab → "Report a vulnerability" — so a fix can be
prepared before the details are public.

Please include: what you built it from, the exact command, what you expected, and what
happened. If a crash is involved, the command line and the ImageMagick and CUDA versions
are worth more than a description of the symptom.

There is no bug bounty and no SLA. This is one machine's worth of work on a spare-time
project, and a slow honest reply beats a fast non-reply.

## What is in scope

- **Image parsing.** `rdither` decodes with ffmpeg and reads images with ImageMagick,
  both of which have had serious historical vulnerabilities. A file that crashes the
  process, hangs it, or makes it allocate without bound is worth reporting — the input
  formats are untrusted by construction.
- **Output handling.** Palette and path handling, checkpoint and resume files, and the
  encoder pipe. Anything that writes outside its output path.
- **The build.** A modified or unexpected source in the dependency chain.

## What is not a vulnerability

- **Malformed input rejected with an error.** `rdither` refuses geometry it cannot
  handle and says why. That is the intended behaviour.
- **Crashes on a GPU that reports insufficient double precision.** Detected and
  refused, not papered over.
- **The known open defects in [docs/DESIGN.md](docs/DESIGN.md) §12** — chiefly that
  `--no-gpu` video is non-deterministic. Those are documented, not hidden, and are
  already tracked.
- **Anything you can only trigger by editing the source to do it.**

## Scope of the code

The repository contains no network code, no server component, and no updater. The
attack surface is: files you are asked to decode, files you are asked to write, and the
build.

`FFmpeg-master/` and `OpenCL-SDK-*/` appear in some working copies and are **not** part
of this repository. Nothing here is vendored, and a report about a bug in either belongs
to that project, not to this one.
