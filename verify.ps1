# SPDX-License-Identifier: GPL-3.0-or-later
<#
  verify.ps1 -- bit-exactness sweep against ImageMagick's own Riemersma dither.

  Every case is checked twice:
    1. rdither --verify, which runs IM's full pipeline internally and diffs
       pixel-for-pixel;
    2. `magick compare -metric AE`, an independent path through the CLI.

  Exit code 0 means every case reported AE=0 on both engines.
#>
param(
  [string]$Rdither   = ".\build\Release\rdither.exe",
  [string]$Magick    = "magick",
  [int[]] $Colors    = @(2, 4, 16, 64, 256),
  [string]$ImageDir  = "tests"
)

# `magick compare -metric AE` reports its metric on stderr, which PowerShell
# would otherwise promote to a terminating error under Stop.
$ErrorActionPreference = "Continue"
$env:MAGICK_HOME = if ($env:MAGICK_HOME) { $env:MAGICK_HOME } else { "C:\Program Files\ImageMagick-7.1.2-Q16-HDRI" }

if (-not (Test-Path $Rdither)) { throw "rdither not found at $Rdither (run build.bat first)" }

# ---- fixtures ---------------------------------------------------------------
$fixtures = @{
  "t_1x1"     = @("-size", "1x1",   "xc:red")
  "t_3x5"     = @("-size", "3x5",   "plasma:fractal")
  "t_17x13"   = @("-size", "17x13", "gradient:red-blue")
  "t_wide"    = @("-size", "640x64", "plasma:fractal")
  "in_grad"   = @("-size", "64x64", "gradient:black-white", "-colorspace", "sRGB")
  "in_plasma" = @("-size", "96x64", "plasma:fractal", "-colorspace", "sRGB")
  "in_shapes" = @("-size", "64x64", "xc:gray50", "-fill", "red", "-draw", "circle 32,32 32,8")
}

New-Item -ItemType Directory -Force $ImageDir | Out-Null
foreach ($name in $fixtures.Keys) {
  $path = Join-Path $ImageDir "$name.png"
  if (-not (Test-Path $path)) {
    & $Magick @($fixtures[$name]) $path
  }
}
# Alpha and noise cases are derived from a base image.
if (-not (Test-Path "$ImageDir\in_alpha.png")) {
  & $Magick "$ImageDir\in_plasma.png" -alpha set -channel A -evaluate set 60% +channel "$ImageDir\in_alpha.png"
}
if (-not (Test-Path "$ImageDir\noisy.png")) {
  & $Magick -size 512x384 plasma:fractal -attenuate 0.4 +noise Gaussian -colorspace sRGB "$ImageDir\noisy.png"
}
# A JPEG fixture, and the reason it exists.
#
# Every other fixture here is a PNG, which is exactly why a real defect survived
# for so long.  ImStore() cloned the INPUT image and inherited its coder, so dither
# a JPEG to a path called .png and the file on disk was JPEG bytes (FF D8 FF) --
# lossy re-compression of pixels that were already an exact 16-colour palette.  The
# file came back with 93,377 distinct colours where ImageMagick's own Riemersma
# output has 16.  A PNG input cannot reach it, because a PNG clone already names
# the PNG coder, so no PNG-only fixture could ever have caught it.
#
# --verify missed it too: it compared the in-memory store rather than the file that
# was written, and reported AE=0 on that broken file.  Both are fixed; this fixture
# is what keeps the JPEG path covered.
$jpg = "$ImageDir\in_jpeg.jpg"
if (-not (Test-Path $jpg)) {
  # Hard edges on a flat field, because ringing around an edge is what produces
  # source colours outside the palette and makes the failure visible.
  & $Magick -size 256x192 "xc:#204080" -fill "#e8c040" -draw "rectangle 40,40 120,120" `
            -fill "#f0f0f0" -draw "circle 190,140 190,60" -quality 92 $jpg
}

$images = @("t_1x1", "t_3x5", "t_17x13", "t_wide", "in_grad", "in_plasma",
            "in_shapes", "in_alpha", "noisy", "in_jpeg")

# Source extension per fixture.  Everything is PNG except in_jpeg, which is a JPEG
# on purpose -- see the comment where it is generated.  A non-PNG input is the only
# thing that reaches the coder-inheritance defect.
$ext = @{ "in_jpeg" = "jpg" }

$engines = @(
  @{ name = "cpu";        args = @("--engine", "cpu") },
  @{ name = "cuda";       args = @("--engine", "cuda") },
  @{ name = "cpu/disk";   args = @("--engine", "cpu", "--max-ram-mb", "1") }
)

# An engine this build cannot run makes its cases unaskable, not failed.  Without
# this, `build.cmd --no-cuda` produced "90 passed, 45 failed" and exited 1: the 45
# were exactly the cuda engine's share (9 images x 5 colour counts), every one of them
# a refusal to start rather than a wrong pixel.  A build configuration the project
# documents as supported was reporting a failing suite.
. "$PSScriptRoot\tools\rd-engine-probe.ps1"
$avail = Get-RdEngineAvailability -Rdither $Rdither -Engines @('cuda') -Fixture (Join-Path $ImageDir "in_grad.png")
Write-RdEngineSkipReport -Availability $avail | Out-Null
$skipEngines = @($avail.Keys | Where-Object { $avail[$_] })
$activeEngines = @($engines | Where-Object { $skipEngines -notcontains $_.name })

$pass = 0; $fail = 0; $skipped = 0
foreach ($img in $images) {
  $e = if ($ext.ContainsKey($img)) { $ext[$img] } else { "png" }
  $src = Join-Path $ImageDir "$img.$e"
  foreach ($c in $Colors) {
    $ref = Join-Path $ImageDir "$($img)_$($c)_ref.png"
    & $Magick $src -dither Riemersma -colors $c $ref | Out-Null
    foreach ($e in $engines) {
      if ($skipEngines -contains $e.name) { $skipped++; continue }
      $out = Join-Path $ImageDir "$($img)_$($c)_$($e.name -replace '/','_').png"
      $text = (& $Rdither @($e.args) --colors $c --verify --quiet $src $out 2>&1 | Out-String)
      $internal = $text -match "BIT-EXACT"
      # Independent confirmation through the ImageMagick CLI.  `compare -metric`
      # prints to stderr, so it is funnelled through cmd to keep the number
      # clean instead of wrapped in a PowerShell ErrorRecord.
      $aeText = (& cmd /c "`"$Magick`" compare -metric AE `"$out`" `"$ref`" null: 2>&1" | Out-String).Trim()
      $ae = ($aeText -split '\s+')[0]
      $ok = $internal -and ($ae -eq "0")
      if ($ok) { $pass++ } else {
        $fail++
        Write-Host ("FAIL {0,-10} colors={1,-4} {2,-9} internal={3} compare_AE={4}" -f `
                    $img, $c, $e.name, $internal, $ae) -ForegroundColor Red
        if ($text.Trim()) { Write-Host "     $($text.Trim())" }
      }
    }
  }
}

Write-Host ""
if ($skipped -gt 0) {
  Write-Host "bit-exact cases: $pass passed, $fail failed, $skipped SKIPPED (engine not in this build)" -ForegroundColor Yellow
  Write-Host "  The $skipped skipped cases are NOT passes. On a build with that engine they run." -ForegroundColor Yellow
} else {
  Write-Host "bit-exact cases: $pass passed, $fail failed"
}
if ($fail -gt 0) { exit 1 }

# Bit-exactness against ImageMagick is a comparison between two implementations, so
# it is structurally unable to see a fault that BOTH have.  The unfilled-pixel-channel
# bug in the image writer was exactly that: it made the encoder pick a different PNG
# type from stale heap memory, and it was invisible here for as long as both paths
# were wrong in the same way at the same time.
#
# So this also runs each engine against ITSELF, repeatedly.  It needs fixtures that
# only tools\probe-opencl-exact.ps1 generates, hence the wrapper: if those are absent
# the determinism probe exits 2 ("cannot run"), and that is reported as skipped
# rather than as a pass and not as a failure.  A check that cannot run and says so is
# the difference between a missing fixture and a silent hole in the suite.
# The image determinism probe reads c_plasma and c_ashape from a shared fixture
# directory, and this used to PRINT an instruction to run tools\probe-opencl-exact.ps1
# first.  No automated run reads instructions, so on every clean clone -- which is every
# runner -- the probe skipped, and a skip is not a failure, so nothing went red and
# nothing said so.  The same shape as the missing video fixtures: a check that cannot run
# and does not say why is a hole wearing a pass's clothes.
#
# The fixtures are generated here, by the script that already defines them, rather than
# by a new one -- two copies of a fixture definition would drift.  -FixturesOnly stops
# before the 54-cell cross-engine comparison, which is not this suite's business and
# whose exit code would then have to be interpreted here.
$oclFx = ''
$oclGen = Join-Path $PSScriptRoot 'tools\probe-opencl-exact.ps1'
if (Test-Path $oclGen) {
  $g = & pwsh -NoProfile -File $oclGen -FixturesOnly 2>&1 | Out-String
  $gCode = $LASTEXITCODE
  foreach ($line in ($g -split "`r?`n")) {
    if ($line -match '^FIXTURE_DIR=(.+)$') { $oclFx = $Matches[1].Trim() }
    elseif ($line.Trim()) { Write-Host "  $($line.TrimEnd())" }
  }
  if ($gCode -ne 0) { $oclFx = '' }
}

$det = Join-Path $PSScriptRoot 'tools\probe-determinism.ps1'
if (Test-Path $det) {
  Write-Host ""
  $detArgs = @('-Rdither', $Rdither)
  if ($oclFx) { $detArgs += @('-FixtureDir', $oclFx) }
  & pwsh -NoProfile -File $det @detArgs
  switch ($LASTEXITCODE) {
    0 { }
    2 { Write-Host "determinism: SKIPPED (cannot run -- the probe printed the reason above)" -ForegroundColor Yellow }
    default { Write-Host "determinism: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

# The video pipeline is a third comparison, and neither of the two above can stand
# in for it: the image probes never build a batch, never run the reader, and never
# touch the uint16 data path, which is the part that differs most between an image
# and a video frame.  It also decodes ~450 MB, so it is kept out of the way rather
# than run by default -- but it is wired in, because a check nothing invokes is a
# check that rots.  Exit 2 from the probe means "cannot run" (no clip), and that is
# reported as skipped, never as a pass and never as a failure.
# Fixtures for the two clip-taking probes.  tests\clip1920.mp4 and
# tests\clip1920_audio.mkv are large local files that are deliberately untracked, so
# every clean clone -- which is every automated run -- reported the video checks as
# SKIPPED.  Installing ffmpeg did not change that; it only moved the reason.  With
# ffmpeg present the runner said "no clip at tests\clip1920.mp4" instead of "ffmpeg not
# found", and the video path still had no coverage at all.
#
# So when the real fixtures are absent, GENERATE small ones rather than skipping.
# Nothing binary enters the repository, and the checks still assert what they assert:
# cross-engine bit identity, run-to-run determinism, and no frame lost to -shortest.
# What is given up is SIZE, and that is a real loss: a bug that only appears at
# 1920x1080 with 60 frames would not be caught by a 320x180, 30-frame fixture.  This
# narrows the blind spot; it does not close it, and saying so is cheaper than letting
# someone assume the video path is now fully covered.
$genClip = ''
$genAudio = ''
$needFixtures = -not (Test-Path (Join-Path $PSScriptRoot 'tests\clip1920.mp4')) -or
                -not (Test-Path (Join-Path $PSScriptRoot 'tests\clip1920_audio.mkv'))
if ($needFixtures) {
  $mk = Join-Path $PSScriptRoot 'tools\make-video-fixtures.ps1'
  if (Test-Path $mk) {
    Write-Host ""
    $fx = & pwsh -NoProfile -File $mk 2>&1 | Out-String
    $fxCode = $LASTEXITCODE
    foreach ($line in ($fx -split "`r?`n")) {
      if ($line -match '^(CLIP|CLIP_AUDIO)=(.+)$') {
        if ($Matches[1] -eq 'CLIP') { $genClip = $Matches[2].Trim() } else { $genAudio = $Matches[2].Trim() }
      } elseif ($line.Trim()) { Write-Host "  $($line.TrimEnd())" }
    }
    if ($fxCode -ne 0) {
      # Not fatal.  The probes keep their own defaults, hit the missing clip, and exit
      # 2, which is reported as SKIPPED -- the same honest outcome as before, and the
      # message above has already said why.
      $genClip = ''
      $genAudio = ''
    }
  }
}

$vid = Join-Path $PSScriptRoot 'tools\probe-video-exact.ps1'
if (Test-Path $vid) {
  Write-Host ""
  $vidArgs = @('-Rdither', $Rdither)
  if ($genClip) { $vidArgs += @('-Clip', $genClip) }
  if ($genAudio) { $vidArgs += @('-AudioClip', $genAudio) }
  & pwsh -NoProfile -File $vid @vidArgs
  switch ($LASTEXITCODE) {
    0 { }
    # Not "no tests\clip1920.mp4".  The probe exits 2 for three different reasons --
    # no clip, no ffmpeg, or no pair of engines that both produced frames -- and it
    # prints which one above.  Naming the clip here regardless meant that once the
    # probe learned to skip an engine for want of a device, this line went on
    # reporting a missing file that was present, pointing at the wrong thing.
    2 { Write-Host "video: SKIPPED (cannot run -- the probe printed the reason above)" -ForegroundColor Yellow }
    default { Write-Host "video: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

# And a fourth, which is the odd one out: it does not compare two engines, it
# compares the output against the SOURCE.  That is deliberate, and it is the only
# shape of check that could have caught the unvisited-pixel bug -- both engines
# were reading their own uninitialised memory, so the two comparisons above
# agreed with each other while both were wrong.  It also builds its own fixtures
# from lavfi, so it needs nothing from tests\ except ffmpeg on PATH.
$unv = Join-Path $PSScriptRoot 'tools\probe-unvisited-pixel.ps1'
if (Test-Path $unv) {
  Write-Host ""
  & pwsh -NoProfile -File $unv -Rdither $Rdither
  switch ($LASTEXITCODE) {
    0 { }
    2 { Write-Host "unvisited pixel: SKIPPED (cannot run -- the probe printed the reason above)" -ForegroundColor Yellow }
    default { Write-Host "unvisited pixel: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

# And a fifth: the image determinism probe again, pointed at VIDEO.  The image one
# is a self-comparison across time rather than an engine-against-engine comparison,
# which is the only shape of check that sees a fault both engines share -- and two
# separate bugs lived there.  The video pipeline had no such check at all, so a
# reintroduction of the -shortest frame-deleting class in the palette or read stage
# would not have been caught by anything: probe-video-exact compares the engines
# against each other, and both would drop the same frames.
$vdet = Join-Path $PSScriptRoot 'tools\probe-video-determinism.ps1'
if (Test-Path $vdet) {
  Write-Host ""
  $vdetArgs = @('-Rdither', $Rdither)
  if ($genAudio) { $vdetArgs += @('-Clip', $genAudio) }
  & pwsh -NoProfile -File $vdet @vdetArgs
  switch ($LASTEXITCODE) {
    0 { }
    2 { Write-Host "video determinism: SKIPPED (cannot run -- the probe printed the reason above)" -ForegroundColor Yellow }
    default { Write-Host "video determinism: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}
exit 0
