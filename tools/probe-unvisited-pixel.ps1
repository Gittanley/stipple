# Is the pixel the Riemersma walk never visits written from the source?
#
# This is a regression test for a bug that shipped, and the reason it shipped is
# the most useful thing about it: every other check in this project was blind to it.
#
# THE BUG.  BuildCurveIndex enumerates ImageMagick's closed-form Hilbert path over
# [0, 4^level - 1) and keeps only the cells inside the image, and for some
# geometries that leaves exactly one in-bounds cell unvisited.  The GPU scatters
# write only where `owner[pixel] == i`, so that pixel is never written.  In the
# float output path that is harmless: `batch` arrives pre-filled with the source
# frame, so the pixel keeps its source value -- which is what ImageMagick's own
# recursion leaves there, because the recursion genuinely never visits it.  In the
# uint16 and planar-4:4:4 output paths the buffer is a bare cudaMalloc with no
# COPY_HOST_PTR, so the same pixel was uninitialised device memory, and it was
# encoded into the file.
#
# It is one pixel in ~10^6, so it looks fine.  That is the trap: it is not a visual
# bug, it is a *reproducibility* bug.  The value would differ between runs, between
# drivers, and between CUDA and OpenCL.
#
# And on a fresh allocation it does not even look wrong.  With the fill disabled,
# this test measures the output pixel as YUV 0,0,0 against a source of 170,166,16 --
# cudaMalloc handed back zeroed pages, so a one-frame clip reads as a clean, if
# slightly dark, result.  The corruption only becomes visible once the buffer is
# reused across batches, where the pixel carries the previous batch's data instead.
# That is why eyeballing the output found nothing, and why the check below compares
# against the source rather than against "does it look plausible".
#
# Nothing in the suite noticed, because:
#
#   * 1920x1080 -- the geometry of every long clip measured in this project,
#     including the five-minute benchmark -- has NO unvisited pixel at all.
#   * verify.ps1 tests `cpu`, `cuda` and `cpu/disk`, which are the sequential walk
#     and never consult the owner map.
#   * probe-opencl-exact.ps1 and probe-video-exact.ps1 compare CUDA against OpenCL.
#     Before the fix both were reading their own uninitialised memory, so they
#     could still agree with each other -- the test is structurally blind to a bug
#     both engines share.
#
# So this test does the only thing that can catch it: it asks what the pixel
# *should* be, and compares against the source.  It does not compare two engines.
#
# WHAT IT CHECKS, on each geometry:
#
#   1. rdither reports the unvisited pixel, and names it.
#   2. In the output, that pixel equals the SOURCE's pixel exactly.
#   3. Its immediate neighbour does NOT equal the source, because the neighbour WAS
#      dithered.  Without this the test would also pass if rdither simply never
#      dithered anything, or if it copied the whole frame through untouched.
#
# Point 3 is the one that makes the test mean something.  A test that only checked
# the source match would be satisfied by a renderer that does nothing.
#
# Exit 2 means "cannot run" (no ffmpeg, no rdither), deliberately distinct from
# exit 1, "ran and the pixel is wrong".

param(
  [string]$Rdither = "",
  [string]$Ffmpeg  = "",
  [int]$Colors = 16,
  # Geometries to sweep.  These are the ones that were measured, not a guess.
  #
  # The summary below used to say "the first six have no unvisited pixel and the last four
  # do", which described a ten-geometry list.  Four more were added at the end without
  # updating it, and the last run reports "6 of 6 geometries asserted and ok, 0 failed, 8
  # have no unvisited pixel to check" -- so it is SIX with one and EIGHT with none.  Do not
  # describe this list by position; read it.  Both halves matter -- a geometry that should
  # report one and does not is a silent regression too.
  #
  # 4096x2160 was MISSING and is in the break set.  The condition is derived, not
  # guessed: the missing cell is (2^level - 1, 0), so it is inside the image exactly
  # when `width` is a power of two AND width >= height.  4096 is a power of two, so
  # DCI 4K has one unvisited pixel and was therefore untested.  33x17 is the mirror
  # case -- level 6 puts the missing cell at (63,0), x=63 >= 33, so it has NONE, and
  # the old comment at rd_blocks_cuda.cu:1058 claiming "33x17 -> 1 at (31, 0)" cannot
  # be reproduced from ComputeCurveLevel.  (31,0) is the level-5 cell.
  #
  # Four geometries were added while landing the fused gather: 512x480, 256x192, 320x240
  # and 480x512.  Two of them (512x480, 256x192) have an unvisited pixel and were NOT in
  # the set, so the earlier claim that only the four >= 1024-wide geometries have one was
  # wrong by omission -- the rule held, the enumeration was incomplete.  The other two
  # are the falsifying side: 320x240 (320 not a power of two) and 480x512 (taller than
  # wide) must have none, and the run asserts that they do.
  #
  # 512x480 was briefly believed to falsify the rule, on the strength of an IMAGE-path
  # run that printed no unvisited line.  That was a wrong inference from a gated log:
  # the report is guarded on the list being non-empty, so no line means the print did not
  # run.  It is in the list now so the doubt stays recorded where it can be re-checked.
  [string[]]$Geometries = @(
    "1920x1080", "1280x720", "768x1024", "720x1280", "640x360", "854x480",
    "1024x768", "1024x1024", "2048x2048", "4096x2160",
    "512x480", "256x192", "320x240", "480x512"
  ),
  [int]$Frames = 8
)

$ErrorActionPreference = 'Stop'

if (-not $Rdither) {
  $Rdither = Join-Path (Split-Path $PSScriptRoot -Parent) 'build\Release\rdither.exe'
}
if (-not (Test-Path $Rdither)) { Write-Host "cannot run: $Rdither not built"; exit 2 }

if (-not $Ffmpeg) {
  $c = Get-Command ffmpeg -ErrorAction SilentlyContinue
  if ($c) { $Ffmpeg = $c.Source }
}
if (-not $Ffmpeg -or -not (Test-Path $Ffmpeg)) {
  Write-Host "cannot run: ffmpeg not found"
  exit 2
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("rd-unvisited-" + [Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# This test hardcodes --engine blocks (the fault it is looking for is a cudaMalloc
# with no fill), so on a build without CUDA every one of the 9 geometries failed with
# "rdither exited 1" and verify.ps1 reported "0 of 9 cases ok, 0 skipped" and exited 1.
# Zero ok AND zero skipped is the shape that is hardest to read correctly: it is not a
# pass, not a skip, and not a clean failure either.  The engine has to be established
# BEFORE the loop, once, and reported as this probe's own "cannot run" (exit 2), which
# verify.ps1 already maps to SKIPPED.
#
# The availability check needs one readable image.  ffmpeg is already a hard
# requirement of this probe, so a single-frame PNG is made here rather than adding a
# dependency on ImageMagick, which this probe does not otherwise use.
$availPng = Join-Path $tmp 'avail.png'
& $Ffmpeg -v error -y -f lavfi -i 'testsrc=size=64x64:rate=1' -frames:v 1 $availPng 2>$null
if (-not (Test-Path $availPng)) {
  Write-Host "cannot run: could not make a probe image for the engine check"
  exit 2
}
. "$PSScriptRoot\rd-engine-probe.ps1"
$avail = Get-RdEngineAvailability -Rdither $Rdither -Engines @('blocks', 'cpu') -Fixture $availPng
# blocks is preferred, because the fault this probe was written for was a cudaMalloc
# with no fill.  But refusing to run without it meant this probe reported SKIPPED on
# every automated run -- the hosted runners have no GPU -- so the video path had no
# coverage at all.  What remains on the host engine is still the comparison that
# caught the original bug: output against SOURCE, across nine geometries, including
# the three that DO have an unvisited pixel.  The reduced scope is stated on stdout
# rather than being discovered later by reading a tally.
#
# The line below used to read the other way round -- `if ($avail['blocks']) { $engine =
# 'cpu' } else { $engine = 'blocks' }` -- which is the opposite of the comment directly
# above it, and it meant the device path was NEVER chosen.  Measured on a
# -DRD_WITH_CUDA=OFF build: all nine geometries printed "no unvisited pixel", $pass
# reached 9, and the probe exited 0 having asserted nothing at all, because the cpu walk
# never prints a marker.  A probe that cannot fail on the configuration CI runs is worse
# than no probe, because it is counted.  Found by an audit that asked which probe it
# trusted least; the answer was this one, for a reason nobody had read the code to check.
if ($avail['blocks']) { $engine = 'blocks' } else { $engine = 'cpu' }
if ($avail[$engine]) {
  Write-Host "cannot run: no usable engine -- $($avail[$engine])"
  exit 2
}
if ($engine -eq 'cpu') {
  Write-Host ("engine     : cpu -- no usable blocks device, so the GPU half of this" +
              " check is NOT covered.  Comparing output against SOURCE still is.")
}

# Read one pixel as planar YUV444.
#
# The `format=yuv444p` in front of the crop is not optional.  Cropping a 4:2:0
# source down to a single pixel asks for 1x1 chroma, which does not exist, and
# ffmpeg fails the filter graph with "Invalid too big or non positive size" --
# an error that looks like a bad crop argument rather than a subsampling problem.
function Get-PixelYuv444([string]$Path, [int]$X, [int]$Y) {
  $out = Join-Path $tmp 'px.bin'
  if (Test-Path $out) { Remove-Item $out -Force }
  $a = @('-v','error','-i',$Path,'-vf',"format=yuv444p,crop=1:1:$X`:$Y",
         '-frames:v','1','-f','rawvideo','-pix_fmt','yuv444p','-y',$out)
  $p = Start-Process -FilePath $Ffmpeg -ArgumentList $a -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0 -or -not (Test-Path $out)) { return $null }
  $b = [IO.File]::ReadAllBytes($out)
  if ($b.Length -lt 3) { return $null }
  Remove-Item $out -Force -ErrorAction SilentlyContinue
  return @([int]$b[0], [int]$b[1], [int]$b[2])
}

$pass = 0; $fail = 0; $skipped = 0; $none = 0
$results = @()

foreach ($g in $Geometries) {
  $parts = $g.Split('x')
  $w = [int]$parts[0]; $h = [int]$parts[1]
  $clip = Join-Path $tmp "in_$g.mp4"
  $dith = Join-Path $tmp "out_$g.mkv"

  # A synthetic clip is right here even though it is not right for a throughput
  # benchmark: this test asks about one pixel's provenance, and what that pixel
  # *is* does not matter.  What matters is that the walk never visits it, and
  # that is a property of the geometry.
  #
  # -frames:v alone, no -t.  The lavfi source is endless so the frame cap is
  # sufficient, and it avoids formatting a duration as a decimal -- which on this
  # machine's locale produces "0,267", and ffmpeg rejects a comma there with
  # "Invalid duration for option t".  A test that skips every case because of a
  # locale is worse than no test: it exits 0 and looks green.
  $gen = @('-y','-v','error','-f','lavfi','-i',"testsrc2=size=$g`:rate=30",
           '-frames:v',"$Frames",
           '-c:v','libx264','-preset','ultrafast','-crf','20','-pix_fmt','yuv420p',$clip)
  $gp = Start-Process -FilePath $Ffmpeg -ArgumentList $gen -NoNewWindow -PassThru -Wait
  if ($gp.ExitCode -ne 0 -or -not (Test-Path $clip)) {
    Write-Host ("  {0,-10} SKIP  could not build fixture" -f $g)
    $skipped++; continue
  }

  # --video-lossless, because this test is about pixels.  With the default
  # libx264 yuv444p the encoder invents colours around every transition (measured
  # 30691 unique colours from a 16-colour palette), and then a mismatch here would
  # say nothing about the dither.
  $log = Join-Path $tmp "log_$g.txt"
  # Not $args: that is an automatic variable, and shadowing it works right up
  # until something reads it.
  $rargs = @('--video','--engine',$engine,'--video-lossless','--colors',"$Colors",$clip,$dith)
  $rp = Start-Process -FilePath $Rdither -ArgumentList $rargs -NoNewWindow -PassThru -Wait `
          -RedirectStandardError $log -RedirectStandardOutput ($log + '.out')
  $text = (Get-Content $log -Raw -ErrorAction SilentlyContinue)

  if ($rp.ExitCode -ne 0) {
    Write-Host ("  {0,-10} FAIL  rdither exited {1}" -f $g, $rp.ExitCode)
    $fail++; $results += "$g : rdither exit $($rp.ExitCode)"
    continue
  }

  # Parse the note.  "the N it does not reach keep their source value" -- N is the
  # count, and the first index is printed after "First at index I (X,Y)".
  $m = [regex]::Match($text, 'the\s+(\d+)\s+it does not reach')
  if (-not $m.Success) {
    # No unvisited pixel for this geometry.  That is legitimate -- most of the nine
    # are like it -- so it is not a FAILURE.  But it is NOT A PASS EITHER, and counting
    # it as one is what let this probe report "9 of 9 cases ok" on a build where it
    # asserted nothing at all: the marker never appears, every geometry takes this
    # branch, and $pass reaches 9.  A geometry with nothing to check is counted
    # separately and named in the summary, so the denominator is the number of
    # geometries actually asserted rather than the number tried.
    Write-Host ("  {0,-10} info  no unvisited pixel -- nothing to assert here" -f $g)
    $none++
    $results += "$g : none"
    continue
  }

  $count = [int]$m.Groups[1].Value
  $m2 = [regex]::Match($text, 'First at index (-?\d+) \((\d+),(\d+)\)')
  if (-not $m2.Success) {
    Write-Host ("  {0,-10} FAIL  reports {1} unvisited but does not name the pixel" -f $g, $count)
    $fail++; $results += "$g : reported $count, pixel unnamed"
    continue
  }
  $px = [int]$m2.Groups[2].Value
  $py = [int]$m2.Groups[3].Value

  # Left neighbour, for the negative control below.
  $nx = if ($px -gt 0) { $px - 1 } else { $px + 1 }

  $srcU = Get-PixelYuv444 $clip $px $py
  $outU = Get-PixelYuv444 $dith $px $py
  $srcN = Get-PixelYuv444 $clip $nx $py
  $outN = Get-PixelYuv444 $dith $nx $py

  if ($null -eq $srcU -or $null -eq $outU -or $null -eq $srcN -or $null -eq $outN) {
    Write-Host ("  {0,-10} SKIP  could not read pixels back" -f $g)
    $skipped++; continue
  }

  $uMatch = ($srcU[0] -eq $outU[0]) -and ($srcU[1] -eq $outU[1]) -and ($srcU[2] -eq $outU[2])
  $nDithered = -not (($srcN[0] -eq $outN[0]) -and ($srcN[1] -eq $outN[1]) -and ($srcN[2] -eq $outN[2]))

  if ($uMatch -and $nDithered) {
    Write-Host ("  {0,-10} ok    {1} unvisited at ({2},{3}); source value kept, neighbour dithered" -f `
                $g, $count, $px, $py)
    $pass++
    $results += "$g : ok, $count unvisited at ($px,$py)"
  } elseif (-not $uMatch) {
    Write-Host ("  {0,-10} FAIL  ({1},{2}) src YUV {3} != out YUV {4} -- uninitialised memory" -f `
                $g, $px, $py, ($srcU -join ','), ($outU -join ','))
    $fail++
    $results += "$g : FAIL, ($px,$py) src=$($srcU -join ',') out=$($outU -join ',')"
  } else {
    # The unvisited pixel is right, but the neighbour matches the source too --
    # so either nothing was dithered or the whole frame was copied through.  That
    # is a failure of the test's own premise, and it is reported as one rather
    # than passed, because a green test that cannot fail is worse than no test.
    Write-Host ("  {0,-10} FAIL  unvisited pixel matches, but neighbour ({1},{2}) also matches the source -- nothing was dithered" -f `
                $g, $nx, $py)
    $fail++
    $results += "$g : FAIL, neighbour not dithered, test cannot distinguish"
  }
}

Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
$summary = ("Unvisited-pixel provenance: {0} of {1} geometries asserted and ok," +
            " {2} failed, {3} have no unvisited pixel to check, {4} skipped.")
Write-Host ($summary -f $pass, ($pass + $fail), $fail, $none, $skipped)
if ($fail -gt 0) {
  Write-Host ""
  foreach ($r in $results) { if ($r -like '*FAIL*') { Write-Host "  $r" } }
  exit 1
}
# Nothing ran is not a pass.  Exiting 0 here would let a locale bug, a missing
# ffmpeg or a broken fixture report as a green test, which is the specific way
# this kind of check rots: it stops being run because it never goes red.
if (($pass + $fail) -eq 0) {
  Write-Host "cannot run: every case skipped, so nothing was actually checked."
  exit 2
}
exit 0
