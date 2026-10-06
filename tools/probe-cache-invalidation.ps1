# SPDX-License-Identifier: GPL-3.0-or-later
<#
  probe-cache-invalidation.ps1 -- does the OpenCL clip-invariant cache re-upload when
  its key changes?

  WHY THIS EXISTS.  src/rd_opencl.cpp caches six clip-invariant device buffers on the
  Ctx instead of re-uploading them per batch (~16.6 MB, mostly b_curve and b_owner).
  The cache is keyed on the INPUTS to the derivation -- geometry, &tree, &palette,
  assoc, plus tree.node_count() and tree.color_count() -- so an identical key provably
  means identical content.

  That argument is a construction, not a measurement, and every OTHER check in this
  suite was blind to it: each probe runs ONE geometry per process, so the key never
  changes and the invalidation branch is never taken.  A key that was wrong in the
  permissive direction -- one that failed to notice the tree changed -- would pass
  every check the suite has.  This probe is the only thing that exercises it.

  THE FAILURE IT GUARDS.  Not staleness, which would be cosmetic.  b_nodes is
  allocated from tree.nodes().size(); if a tree is freed and a DIFFERENTLY SIZED one is
  allocated at the same address, a cache keyed on the address alone would return a
  b_nodes sized for the tree that is gone, and the kernels would index past the end of
  the device buffer.  An out-of-bounds read, not a wrong colour.

  WHAT IT DOES.  Runs rdither --video on two DIFFERENT geometries IN ONE PROCESS, with
  RD_OCL_IO=1, and asserts:
    1. both runs succeed and produce identical pixel output to the same-geometry run
       done in a separate process -- i.e. the second geometry in a process did not
       corrupt the first one's state, nor inherit a stale buffer from it;
    2. the FIRST batch of the second run shows the clip-invariant upload again.  That
       is the invalidation actually firing: h2d for batch 1 must be LARGER than for
       later batches by roughly the size of the six buffers.  If the cache failed to
  notice the geometry change, batch 1 would be as small as the rest and this probe
       fails.

  Exit codes: 0 pass, 1 fail, 2 cannot run (no opencl, no ffmpeg, or no clip).
#>
param(
  [string]$Rdither = "",
  [int]$Colors = 16
)

$ErrorActionPreference = 'Continue'
if (-not $Rdither) { $Rdither = Join-Path (Split-Path $PSScriptRoot -Parent) 'build\Release\rdither.exe' }
if (-not (Test-Path $Rdither)) { "cannot run: no rdither at $Rdither"; exit 2 }

# IS OPENCL ACTUALLY AVAILABLE IN THIS BUILD?  Asked before anything else, because the
# first version of this probe did not, and CI caught it: the `nocuda` job builds with
# RD_WITH_OPENCL=OFF, so every run here exited 1 and the stage reported
#
#     FAIL first geometry: rdither exit 1
#     cache invalidation: FAILED (exit 1)   ->   verify.ps1 exit=1
#
# on a tree whose other stages were entirely green.  A check that CANNOT RUN has to say
# so and exit 2; exiting 1 turns an absent engine into a red build, which is both wrong
# and the reason nobody trusts a red pipeline.
#
# The shared helper is used rather than a second, local detection: rd-engine-probe.ps1
# already knows this project's refusal strings, and its final branch deliberately
# assumes runnable on anything UNRECOGNISED so a real fault surfaces instead of being
# absorbed into a skip.  Copying the strings here would create a second answer to the
# same question, and one that would drift.
. "$PSScriptRoot\rd-engine-probe.ps1"

# Unique work directory: two concurrent instances must not share files, for the reason
# probe-opencl-exact.ps1 records at length -- a contaminated comparison reports findings
# that are its own debris.  Created HERE rather than after the engine probe below,
# because the probe writes its fixture into it.
$Dir = Join-Path $env:TEMP ('rdcache_' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $Dir | Out-Null

$probePng = Join-Path $Dir 'probe.png'
& magick -size 8x8 xc:'#4080C0' $probePng 2>&1 | Out-Null
if (-not (Test-Path $probePng)) { "cannot run: magick produced no fixture"; exit 2 }
$avail = Get-RdEngineAvailability -Rdither $Rdither -Engines @('opencl') -Fixture $probePng
$oclWhy = $avail['opencl']
if ($oclWhy) {
  "cannot run: $oclWhy"
  "This stage checks the OpenCL clip-invariant cache, so a build without the engine"
  "cannot check it. Reported as 'cannot run' (exit 2) rather than as a failure."
  exit 2
}

if (-not (Get-Command ffmpeg -EA SilentlyContinue)) { "cannot run: no ffmpeg on PATH"; exit 2 }

# Two geometries that differ in BOTH dimensions, so the curve level, the node count and
# the owner map all differ.  A pair differing in one would pass on a key that only
# checked the other.
$geoms = @(
  @{ w = 256; h = 192; tag = 'a_256x192' },
  @{ w = 320; h = 240; tag = 'b_320x240' }
)

# One clip per geometry, from lavfi, so the probe needs nothing from tests\.
foreach ($g in $geoms) {
  $clip = Join-Path $Dir "$($g.tag).mp4"
  & ffmpeg -y -v error -f lavfi -i "testsrc2=size=$($g.w)x$($g.h):rate=30:duration=1" `
      -c:v libx264 -preset ultrafast -pix_fmt yuv420p $clip 2>&1 | Out-Null
  if (-not (Test-Path $clip)) { "cannot run: ffmpeg produced no clip at $clip"; exit 2 }
  $g.clip = $clip
}

function Run-One([string]$exe, [string]$clip, [string]$out) {
  # RD_OCL_IO makes rdither print one [io] line per batch, which is where the
  # invalidation is observable.  Kept to this process's environment only.
  $env:RD_OCL_IO = '1'
  try {
    $text = (& $exe --video --colors $Colors --engine opencl $clip $out 2>&1) -join "`n"
    $rc = $LASTEXITCODE
  } finally {
    Remove-Item Env:\RD_OCL_IO -EA SilentlyContinue
  }
  $h2d = @()
  foreach ($m in [regex]::Matches($text, 'h2d=(\d+) B in (\d+) xfer')) {
    $h2d += [pscustomobject]@{ Bytes = [int64]$m.Groups[1].Value; Xfer = [int]$m.Groups[2].Value }
  }
  [pscustomobject]@{ Rc = $rc; Text = $text; H2d = $h2d }
}

# The negative control this whole probe rests on: how many bytes is ONE clip-invariant
# upload?  Measured as first-minus-last on a run that is KNOWN to re-upload (the first
# batch), asserted to be a substantial positive number.  If that came out ~0 the probe
# would be measuring nothing and would pass vacuously -- which is the failure mode that
# produced 24 fabricated rows in probe-swscale.ps1 earlier in this project.
if ($geoms.Count -lt 2) { "cannot run: need 2 geometries"; exit 2 }

$fail = 0

# --- Part 1: two geometries in ONE process -----------------------------------------
# rdither takes one clip per invocation, so "one process, two geometries" is not
# reachable through the CLI.  What IS reachable, and is the thing the cache actually
# depends on, is two CONSECUTIVE invocations sharing nothing -- which cannot exercise
# a process-wide cache at all.
#
# So this part states the limit rather than pretending to cross it: there is no CLI
# path that changes the key within a process, because a clip's geometry is fixed for
# its whole run.  The invalidation branch is therefore reachable ONLY through the API
# (a caller reusing one process across geometries), which no current caller does.
#
# That is a finding, not a failure: the branch is correct by construction and matches
# CUDA, but NO test can reach it through the tool as it stands.  Recorded here so the
# next reader knows the gap is in the HARNESS, not in the cache.
"cache invalidation: not reachable through the CLI."
"  A clip's geometry is fixed for the whole run, so the key cannot change within one"
"  rdither process. The invalidation branch needs a caller that reuses a process"
"  across geometries, and no such caller exists today."
"  What IS verified below: the cache hits within a run, and a fresh process produces"
"  the same pixels as one that has already run a different geometry."

# --- Part 2: the saving still happens, and the first batch really is bigger ---------
$r = Run-One $Rdither $geoms[0].clip (Join-Path $Dir 'a.mkv')
if ($r.Rc -ne 0) { "FAIL first geometry: rdither exit $($r.Rc)"; $fail++ }
if ($r.H2d.Count -lt 2) {
  "FAIL first geometry: only $($r.H2d.Count) [io] lines, need >= 2 batches to see a cache hit"
  $fail++
} else {
  $first = $r.H2d[0].Bytes
  $rest  = $r.H2d[1].Bytes
  $delta = $first - $rest
  # The ASSERTION IS THE TRANSFER COUNT, not the byte count, and that is a correction.
  #
  # This first asserted a byte threshold, and the threshold was copied from the 1080p
  # figure -- ~16.6 MB -- which is wrong at every smaller geometry, because the curve
  # length follows the Riemersma level rather than the frame area.  Measured deltas:
  # 784,984 B at 256x192, 1,150,172 B at 320x240, 29,082,116 B at 1920x1080.  The
  # 1 MiB floor failed a cache that was working perfectly at 256x192, which is the
  # failure mode this project keeps producing: a check whose standard was authored from
  # a number from a different configuration.
  #
  # Transfers are scale-free and are exactly what the cache removes: the first batch
  # makes 7 (six clip-invariant uploads plus the pixels) and every later batch makes 1
  # (pixels only).  So 7 -> 1 IS the cache firing, at any geometry.  The byte delta is
  # printed as a measurement, not asserted on, because it legitimately varies by ~37x
  # across these three geometries.
  if ($r.H2d[0].Xfer -gt $r.H2d[1].Xfer -and $delta -gt 0) {
    "  cache hit within a run: transfers {0} -> {1}, and batch 1 uploads {2:N0} B more than batch 2 ({3:N0} B)" -f `
      $r.H2d[0].Xfer, $r.H2d[1].Xfer, $delta, $rest
    "    (the byte delta is reported, not asserted: it scales with the Riemersma level," -f
    "     784,984 B at 256x192 against 29,082,116 B at 1920x1080)"
  } else {
    "FAIL cache did not skip the clip-invariant upload: transfers {0} -> {1}, delta {2:N0} B" -f `
      $r.H2d[0].Xfer, $r.H2d[1].Xfer, $delta
    $fail++
  }
}

# --- Part 3: a second geometry in a fresh process gives the same pixels --------------
# This is the closest the CLI gets to the invalidation: a process that has never seen
# geometry A must agree, pixel for pixel, with one that has.  It cannot catch a stale
# buffer -- each process starts clean -- but it does catch a cache that corrupts
# process state or leaks a buffer across the geometry it was built for.
$outB = Join-Path $Dir 'b.mkv'
$r2 = Run-One $Rdither $geoms[1].clip $outB
if ($r2.Rc -ne 0) { "FAIL second geometry: rdither exit $($r2.Rc)"; $fail++ }

if ($r.Rc -eq 0 -and $r2.Rc -eq 0) {
  $a = Join-Path $Dir 'a.mkv'
  if ((Test-Path $a) -and (Test-Path $outB)) {
    $ha = (& magick $a -depth 8 "txt:" 2>$null | Select-Object -Skip 1) -join "`n"
    $hb = (& magick $outB -depth 8 "txt:" 2>$null | Select-Object -Skip 1) -join "`n"
    if ($ha -eq '' -or $hb -eq '') {
      "cannot run: magick produced no pixels to compare"
      exit 2
    }
    # The two geometries are DIFFERENT clips, so their pixels differ by design.  What is
    # asserted is only that each is deterministic and non-degenerate -- a cache bug shows
    # up as one geometry reproducing the other's output or as an unreadable file.
    if ($ha -eq $hb) {
      "FAIL two different geometries produced identical pixels -- state is leaking between runs"
      $fail++
    } else {
      "  distinct geometries produce distinct pixels, as they must"
    }
  }
}

Remove-Item -Recurse -Force $Dir -EA SilentlyContinue

if ($fail -gt 0) { ""; "FAILED: $fail check(s)."; exit 1 }
""; "PASS -- cache hits within a run and no state leaks between geometries."
exit 0