# SPDX-License-Identifier: GPL-3.0-or-later
<#
  probe-video-pixels.ps1 -- hash the DECODED PIXELS of a video render, so that a change
  can be shown to preserve output rather than merely to remain self-consistent.

  WHY THIS EXISTS.  Every video check in this repository compares a render against
  ITSELF:

    probe-video-determinism.ps1   run vs run,       same binary
    probe-video-invariance.ps1    part A: batch 16 vs 1, same binary
                                  part B: --no-gpu vs device, same binary
    probe-unvisited-pixel.ps1     source value kept at one pixel

  None of them compares against anything that existed before the change.  So a change
  that alters every pixel CONSISTENTLY passes all of them: host agrees with device, batch 1
  agrees with batch 16, eight runs agree with each other, and the picture is different
  from yesterday's.  The only check with an external oracle is the bit-exact sweep, and it
  runs on IMAGES, not video.

  That is a real gap rather than a theoretical one.  The change this was written for --
  making the host walk use the flat search -- is argued to be output-identical because
  both paths compute the same nearest palette entry.  That is an ARGUMENT.  This turns it
  into a measurement.

  WHY DECODED PIXELS AND NOT THE FILE.  Container bytes are not stable: an encoder writes
  timestamps, padding and rate-control decisions that can differ between two runs of
  identical pixels.  Hashing the .mkv would therefore report a difference where the picture
  is identical, and -- worse -- a file-hash check invites people to "fix" a real pixel
  change by loosening the comparison.  So each render is decoded with

      ffmpeg -f rawvideo -pix_fmt <fmt> -

  and the RAW FRAME BYTES are hashed.  Same pixels, same hash, always.  This is the same
  choice probe-determinism.ps1 and probe-video-invariance.ps1 already make, and for the
  same reason.

  USAGE.  Two modes, and the mode is not inferred, because inferring it is how a baseline
  silently becomes "the current output" and asserts nothing:

    # 1. WRITE the baseline.  Use this BEFORE making a change.
    pwsh -File tools\probe-video-pixels.ps1 -Write

    # 2. COMPARE against it.  Use this AFTER, and after every risky edit.
    pwsh -File tools\probe-video-pixels.ps1

  Exit codes: 0 all cases match (or the baseline was written), 1 a case differs or failed,
  2 cannot run (no rdither, no ffmpeg, no clip, or no baseline to compare against).
#>
param(
  [string]$Rdither = "",
  [string]$Clip = "",
  [string]$Baseline = "",
  [switch]$Write,
  # The colour counts to cover.  16 is the only one the OpenCL engine accepts, so its rows
  # are expected to be ABSENT above 16 rather than to fail -- see the engine handling
  # below.
  [int[]]$Colors = @(16, 32, 64),
  [int]$Frames = 60
)

$ErrorActionPreference = 'Continue'

if (-not $Rdither) { $Rdither = Join-Path (Split-Path $PSScriptRoot -Parent) 'build\Release\rdither.exe' }
if (-not (Test-Path $Rdither)) { "cannot run: no rdither at $Rdither"; exit 2 }

if (-not $Clip) {
  # tests\clip1920.mp4 when present, since that is the fixture the rest of the suite uses.
  # Deliberately NOT silently substituting a generated clip: a baseline hashed against one
  # clip and compared against another would report differences that are the clip's, not
  # the code's.
  $Clip = Join-Path (Split-Path $PSScriptRoot -Parent) 'tests\clip1920.mp4'
}
if (-not (Test-Path $Clip)) { "cannot run: no clip at $Clip"; exit 2 }

if (-not (Get-Command ffmpeg -EA SilentlyContinue)) { "cannot run: no ffmpeg on PATH"; exit 2 }
if (-not (Get-Command ffprobe -EA SilentlyContinue)) { "cannot run: no ffprobe on PATH"; exit 2 }

if (-not $Baseline) {
  $Baseline = Join-Path (Split-Path $PSScriptRoot -Parent) 'artifacts\video_pixels_baseline.txt'
}

# Unique work directory, per the reason probe-opencl-exact.ps1 records at length: two
# concurrent instances sharing one directory compare against each other's half-written
# files, which reads as an arithmetic disagreement between engines.
$Dir = Join-Path $env:TEMP ('rdpix_' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $Dir | Out-Null

# Hash the DECODED frame bytes.  rgb48le rather than rgba64le: it is still lossless for
# every value rdither writes (16 bits per channel), and it halves the bytes hashed, which
# matters because this runs several renders of 60 frames at 1080p.  rgba64le would be
# equally correct and slower for no gain.
function Get-PixelHash([string]$Path) {
  $raw = Join-Path $Dir ('px_' + [Guid]::NewGuid().ToString('N').Substring(0, 8) + '.raw')
  & ffmpeg -y -v error -i $Path -f rawvideo -pix_fmt rgb48le $raw 2>$null
  if (-not (Test-Path $raw)) { return $null }
  # Count the frames actually decoded rather than trusting the container's claim: a
  # truncated render would otherwise hash to "a short but consistent" picture and pass.
  $nb = & ffprobe -v error -select_streams v:0 -count_frames `
        -show_entries stream=nb_read_frames -of 'default=nw=1:nk=1' $Path 2>$null
  $frameBytes = (Get-Item $raw).Length
  $frames = 0
  if ($nb -match '(\d+)') { $frames = [int]$Matches[1] }
  $h = (Get-FileHash $raw -Algorithm SHA256).Hash
  Remove-Item $raw -EA SilentlyContinue
  [pscustomobject]@{ Hash = $h; Frames = $frames; Bytes = $frameBytes }
}

# The engines, and how each is expected to behave.  OpenCL REFUSES above 16 colours
# (rd_opencl.cpp: "this engine implements the flat search only, so --colors must be 16 or
# fewer"), and that refusal is correct behaviour rather than a gap, so a refusal is
# recorded as such and compared like any other outcome.  If a future change makes OpenCL
# accept 32, this probe reports the CHANGE -- which is exactly what it is for.
$engines = @('cpu', 'cuda', 'opencl')

$results = [ordered]@{}

foreach ($c in $Colors) {
  foreach ($e in $engines) {
    $key = "$c/$e"
    $out = Join-Path $Dir ("${c}_${e}.mkv")
    Remove-Item $out -EA SilentlyContinue

    $text = (& $Rdither --video --colors $c --engine $e --video-lossless --no-audio `
             $Clip $out 2>&1) -join "`n"
    $rc = $LASTEXITCODE

    if ($rc -ne 0 -or -not (Test-Path $out)) {
      # Record the REFUSAL as the outcome.  The identifying line is the one carrying the
      # engine's reason -- `--engine opencl with --video supports ...`, or the dither
      # refusal -- and it is NOT the first non-blank line of the output, which is
      # `source : <path>`.  Taking the first line produced reasons like
      #
      #     REFUSED: source     : J:\...\tests\clip1920.mp4
      #
      # which record that a file exists rather than why the engine declined, and a refusal
      # you cannot read is a refusal you cannot act on.  So: a line mentioning error or
      # refusal if there is one, and only then the first non-blank line.
      $lines = @(($text -split "`r?`n") | Where-Object { $_ -match '\S' })
      $reason = ($lines | Where-Object { $_ -match 'error|refus|must be|not support' } |
                 Select-Object -First 1)
      if ($null -eq $reason) { $reason = ($lines | Select-Object -First 1) }
      if ($null -eq $reason) { $reason = "exit $rc, no output" }
      $results[$key] = "REFUSED: " + $reason.Trim()
      Remove-Item $out -EA SilentlyContinue
      continue
    }

    $px = Get-PixelHash $out
    if ($null -eq $px) {
      $results[$key] = "NO-PIXELS: ffmpeg decoded nothing from $out"
      continue
    }
    $results[$key] = "{0}|{1}|{2}" -f $px.Hash, $px.Frames, $px.Bytes
    Remove-Item $out -EA SilentlyContinue
  }
}

Remove-Item -Recurse -Force $Dir -EA SilentlyContinue

if ($Write) {
  $header = @(
    "# Video pixel baseline -- tools\probe-video-pixels.ps1 -Write"
    "# clip        : $Clip"
    "# pixels      : decoded rgb48le, NOT container bytes"
    "# cases       : $Colors"
    "# rdither     : $Rdither"
    "# This file is written BEFORE a risky change and compared AFTER it."
    "# If a change is INTENDED to alter output, regenerate deliberately and say so."
  )
  ($header + ($results.GetEnumerator() | ForEach-Object { "$($_.Key) $($_.Value)" })) |
    Set-Content -Path $Baseline -Encoding utf8
  "wrote baseline: $Baseline"
  $results.GetEnumerator() | ForEach-Object { "  {0,-12} {1}" -f $_.Key, $_.Value }
  exit 0
}

if (-not (Test-Path $Baseline)) {
  "cannot run: no baseline at $Baseline"
  "Write one BEFORE making the change you want to check:"
  "  pwsh -File tools\probe-video-pixels.ps1 -Write"
  exit 2
}

$expected = @{}
foreach ($line in (Get-Content $Baseline)) {
  if ($line -match '^\s*#' -or $line -notmatch '\S') { continue }
  # "key value" -- key has no space, value is the rest.
  if ($line -match '^(\S+)\s+(.+)$') { $expected[$Matches[1]] = $Matches[2].Trim() }
}

if ($expected.Count -eq 0) {
  "cannot run: baseline at $Baseline has no cases in it"
  exit 2
}

"video pixel comparison against $Baseline"
"  each case is the SHA256 of the DECODED frame bytes"
$fail = 0
$missing = 0
foreach ($k in $expected.Keys) {
  $want = $expected[$k]
  $got = if ($results.Contains($k)) { $results[$k] } else { $null }
  if ($null -eq $got) {
    "  MISSING  $k -- the baseline has this case and this run produced none"
    $missing++
    continue
  }
  if ($got -eq $want) {
    "  ok       $k"
  } else {
    "  DIFFER   $k"
    "      baseline: $want"
    "      now     : $got"
    $fail++
  }
}
foreach ($k in $results.Keys) {
  if (-not $expected.Contains($k)) {
    "  NEW      $k -- this case is not in the baseline (an engine gained or lost a"
    "            configuration).  Recorded, not failed, because whether that is intended"
    "            is a question about the change and not about this probe."
    "            $k = $($results[$k])"
  }
}

""
if ($fail -gt 0 -or $missing -gt 0) {
  "FAILED: $fail differing, $missing missing."
  "If the change was meant to alter output, regenerate the baseline deliberately and"
  "record why in the commit message.  A baseline quietly rewritten to match new output"
  "is worse than no baseline, because it looks like evidence."
  exit 1
}
"all $($expected.Count) case(s) byte-identical to the baseline."
exit 0