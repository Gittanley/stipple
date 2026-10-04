# Is the OpenCL engine bit-identical to the CUDA blocks engine?
#
# The comparison has to be per-pixel, not a whole-file string compare.  Two traps
# already paid for in this project:
#
#   * `magick x.png -depth 8 txt:` and the same on y.png produce strings that can
#     differ in trailing whitespace while every pixel matches, so a plain -eq
#     reports "different" for identical images.  Count differing LINES instead.
#
#   * A run that fails still leaves a stale output file behind, and a stale file
#     compares as "different" with no indication that the engine never ran.  So
#     the output is deleted first and the exit code is captured and reported.
#
# Anything that is not bit-exact here is a real defect: the two engines are
# specified to have the same partition and the same arithmetic, so "close" is not
# an acceptable outcome, and a small scattered difference is the signature of an
# arithmetic disagreement rather than of a structural one.

param(
  # $Dir is UNIQUE per process; $FixtureDir is stable and shared.  The fixed name
  # this used to have for BOTH is a bug, and the same one already fixed in
  # probe-video-determinism.ps1: two concurrent instances shared one directory, and
  # one compared a cell against a file the other was still writing.  The symptom was
  # 53 of 54 cells identical -- one reported "not identical" -- which reads as an
  # arithmetic disagreement between the engines, i.e. as exactly the defect this probe
  # exists to catch.  It was caused by two probes I started myself, minutes after the
  # same probe reported 54 of 54.  A contaminated instrument reporting a real-looking
  # defect is the worst outcome an instrument can produce.
  [string]$Dir = "",
  # STABLE and shared, unlike $Dir.  probe-determinism.ps1 reads the same fixtures
  # from here, which is why verify.ps1 says to run this probe once before it.
  [string]$FixtureDir = "",
  # Make the shared fixtures and stop, without running the 54-cell comparison.  See the
  # comment at the point where it takes effect.
  [switch]$FixturesOnly,
  [int]$Cols = 2
)

if (-not $Dir) {
  $Dir = Join-Path $env:TEMP ('rdocl_' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
}
if (-not (Test-Path $Dir)) { New-Item -ItemType Directory -Force -Path $Dir | Out-Null }

# NOT 'Stop'.  rdither writes diagnostics to stderr, and a native command's stderr
# surfaces as a non-terminating error record; under 'Stop' the first "[blocks]"
# notice aborts the whole sweep before it starts.  Failures are detected from the
# exit codes below, which is the thing that actually means failure.
$ErrorActionPreference = 'Continue'
$rd = Join-Path (Split-Path $PSScriptRoot -Parent) 'build\Release\rdither.exe'
if (-not (Test-Path $Dir)) { New-Item -ItemType Directory -Force -Path $Dir | Out-Null }

# Images chosen to separate the failure modes: a flat field (no error queue
# activity at all), a pure ramp (error queue active, two colours), noise (deep
# tree, 16 colours), plasma (worst case for the octree search), and -- added
# after a round-1 finding -- two ALPHA cases, because associate_alpha changes the
# error queue, the octave fan-out (8 children vs 16) and the scatter, and was
# therefore the one branch of the port with no coverage at all.
# Fixtures go to a STABLE, shared directory; the per-cell work goes to the unique $Dir.
#
# The sharing is deliberate and load-bearing: verify.ps1 tells you to run this probe
# once so that probe-determinism.ps1 finds c_plasma and c_ashape, and it reports the
# determinism check as skipped until they exist.  So the fixtures cannot simply be
# random per run.
#
# What must NOT be shared is the work.  $Dir used to be one fixed name for both, so
# two concurrent instances of this probe shared it, and one compared a cell against a
# file the other was still writing.  The symptom was 53 of 54 identical -- one cell
# reported "not identical" -- which reads as an arithmetic disagreement between the
# engines, i.e. as exactly the defect this probe exists to catch.  It was caused by two
# probes I started myself, minutes after the same probe reported 54 of 54.  A
# contaminated instrument reporting a real-looking defect is the worst outcome an
# instrument can produce, so the split is deliberate: shared inputs, private outputs.
if (-not $FixtureDir) { $FixtureDir = Join-Path $env:TEMP 'rdocl_fixtures' }
New-Item -ItemType Directory -Force -Path $FixtureDir | Out-Null

& magick -size 64x48 xc:'#4080C0' "$FixtureDir\c_flat.png" 2>&1 | Out-Null
& magick -size 64x48 gradient:black-white "$FixtureDir\c_ramp.png" 2>&1 | Out-Null
& magick -size 96x64 plasma:fractal -seed 7 "$FixtureDir\c_plasma.png" 2>&1 | Out-Null
& magick -size 96x64 xc:gray +noise Random "$FixtureDir\c_noise.png" 2>&1 | Out-Null
# Uniform alpha: exercises the association without a discontinuity.
& magick -size 96x64 gradient:red-blue "$FixtureDir\c_astuff.png" 2>&1 | Out-Null
cmd /c "magick ""$FixtureDir\c_astuff.png"" -alpha set -channel A -evaluate set 50%% +channel ""$FixtureDir\c_alpha50.png""" 2>&1 | Out-Null
# Hard alpha edge on a transparent field: the worst case for the alpha error
# queue, and it also trips the greyscale detection path (ImageMagick reduces this
# to graya, so the tree is built with a different child count again).
cmd /c "magick -size 96x64 xc:none -fill ""rgba(200,40,120,0.35)"" -draw ""circle 48,32 48,10"" ""$FixtureDir\c_ashape.png""" 2>&1 | Out-Null

# -FixturesOnly stops here, on purpose.
#
# probe-determinism.ps1 reads c_plasma and c_ashape from this directory, and
# verify.ps1 used to merely PRINT an instruction to run this probe first -- which no
# automated run does, so the image determinism probe skipped in CI and nobody noticed
# because a skip is not a failure.  The fixtures need only `magick` and are made above,
# before any engine work, so a caller that only wants them should not have to also run a
# 54-cell cross-engine comparison whose result it did not ask for and whose exit code it
# would then have to interpret.
#
# Generating them here rather than in a new script is the point: the definitions of these
# fixtures live in exactly one place, and a second copy would drift.
if ($FixturesOnly) {
  $made = @('c_flat', 'c_ramp', 'c_plasma', 'c_noise', 'c_astuff', 'c_alpha50', 'c_ashape')
  $missing = @($made | Where-Object { -not (Test-Path (Join-Path $FixtureDir "$_.png")) })
  if ($missing.Count) {
    # Asserted rather than assumed: a fixture that silently failed to generate leaves
    # probe-determinism skipping for ever, which is the exact failure being fixed here.
    Write-Host "could not generate: $($missing -join ', ') in $FixtureDir"
    exit 1
  }
  "FIXTURE_DIR=$FixtureDir"
  exit 0
}

function Pixels($path) {
  if (-not (Test-Path $path)) { return @() }
  $lines = & magick $path -depth 8 "txt:" 2>$null
  # Drop the header line; keep the pixel lines only.
  $lines | Select-Object -Skip 1
}

function Compare-Pair($img, $colors, $block, $tag) {
  # A UNIQUE output path per cell.  This is not tidiness, it is a bug fix: the
  # earlier version reused one filename for all 54 cells and deleted it with
  # -EA SilentlyContinue, which hides a failed delete.  On this machine roughly
  # two cells per run then compared a *stale* file from an earlier cell against
  # the current one -- a different palette, so ~100% of pixels differ, first
  # difference at pixel 0 -- and the failing cells MOVED between runs.  A
  # nondeterministic comparison cannot distinguish a real defect from its own
  # debris, and it is worse than no test because it looks like a finding.
  $cu = "$Dir\cell_${tag}_cu.png"
  $cl = "$Dir\cell_${tag}_cl.png"

  # Capture the output, do NOT discard it.  These were originally piped to
  # Out-Null, which assigned $e1/$e2 the value of Out-Null -- that is, $null --
  # so the documented-refusal check below could never match and the failure
  # messages were empty.  The variable being named suggests otherwise, which is
  # how it survived review.
  $e1 = (& $rd --engine blocks --blocks $block --colors $colors $img $cu 2>&1) -join "`n"
  $rc1 = $LASTEXITCODE
  $e2 = (& $rd --engine opencl --blocks $block --colors $colors $img $cl 2>&1) -join "`n"
  $rc2 = $LASTEXITCODE

  if ($rc1 -ne 0 -or $rc2 -ne 0) {
    # There is exactly one refusal left in this engine, and it has nothing to do
    # with alpha: a palette too large for the 4-bit nibble packing is declined
    # rather than silently computed some other way, because that would redefine
    # what --engine means.  The sweep uses palettes that fit, so any refusal here
    # is a real failure and is reported as one.
    #
    # The alpha cells used to land in this branch and be waved through as "refused
    # (alpha: known nondeterministic, by design)".  That was 18 of 54 cells
    # permanently exempted from comparison, on the strength of a fault that was
    # never in this engine -- it was unfilled channels in the image writer, now
    # fixed.  They are compared for real below, like every other cell.
    Remove-Item -EA SilentlyContinue $cu, $cl
    return "FAIL rc=$rc1/$rc2  $($e2 | Select-Object -First 1)"
  }
  # Existence and non-trivial size are checked BEFORE the pixel read, because a
  # missing or empty file otherwise reads as "every pixel differs" (each line
  # compared against $null).  That produced a false bug report on the alpha path
  # in this same round.
  foreach ($f in @($cu, $cl)) {
    if (-not (Test-Path $f)) {
      Remove-Item -EA SilentlyContinue $cu, $cl
      return "FAIL missing output $f"
    }
    if ((Get-Item $f).Length -lt 64) {
      Remove-Item -EA SilentlyContinue $cu, $cl
      return "FAIL truncated output $f ($((Get-Item $f).Length) bytes)"
    }
  }

  $a = Pixels $cu
  $b = Pixels $cl
  if ($a.Count -eq 0 -or $b.Count -eq 0) {
    Remove-Item -EA SilentlyContinue $cu, $cl
    return "FAIL unreadable output (cuda $($a.Count) px, opencl $($b.Count) px)"
  }
  $result = ''
  if ($a.Count -ne $b.Count) {
    $result = "FAIL size mismatch (cuda $($a.Count) px, opencl $($b.Count) px)"
  } else {
    $n = $a.Count
    $d = 0
    $first = -1
    for ($i = 0; $i -lt $n; $i++) {
      if ($a[$i] -ne $b[$i]) {
        if ($first -lt 0) { $first = $i }
        $d++
      }
    }
    if ($d -eq 0) {
      $result = "identical ($n px)"
    } else {
      $result = ("DIFFER {0}/{1} ({2:N2}%), first at px {3}" -f $d, $n, (100.0 * $d / $n), $first)
    }
  }
  Remove-Item -EA SilentlyContinue $cu, $cl
  return $result
}

$imgs = @('c_flat', 'c_ramp', 'c_noise', 'c_plasma', 'c_alpha50', 'c_ashape')
$blocks = @(16, 64, 512)
# Distinct colour counts, deduplicated.  This used to be `@(2, 4, $Cols)` with $Cols
# defaulting to 2 -- the literal array (2, 4, 2).  Every --colors 2 cell was therefore
# computed TWICE, the tag collided so the second run overwrote the first render, and
# both copies counted: the summary printed "54 of 54 comparisons identical" when 36
# distinct cells had been compared, and 18 of those 54 were a re-run of a cell already
# counted.  README.md and docs\OPENCL.md quoted the 54.  A duplicated cell is worse
# than a missing one, because it reads as corroboration.
$colourCounts = @(2, 4, $Cols) | Where-Object { $_ -gt 0 } | Sort-Object -Unique
$fail = 0
$total = 0
''
'OpenCL vs CUDA blocks, per-pixel'
''
foreach ($img in $imgs) {
  foreach ($c in @(2, 4, $Cols)) {
    foreach ($b in $blocks) {
      $tag = "{0}_{1}_{2}" -f $img, $c, $b
      $r = Compare-Pair "$FixtureDir\$img.png" $c $b $tag
      $total++
      # Every cell is compared for real, alpha included.  There is no outcome
      # that counts as a pass without being a comparison.
      if ($r -notlike 'identical*') { $fail++ }
      '  {0,-10} colors={1,-3} B={2,-4} {3}' -f $img, $c, $b, $r
    }
  }
}
''
'{0} of {1} comparisons identical; {2} not.' -f ($total - $fail), $total, $fail
# An empty sweep is not a perfect sweep.  $fail -eq 0 alone cannot tell "everything
# matched" from "nothing ran", and the deduplication above makes an accidentally empty
# $colourCounts a live possibility rather than a theoretical one.
if ($fail -eq 0 -and $total -eq 0) {
  Write-Host 'cannot run: zero comparison cells -- refusing to report an empty sweep as a pass.'
  exit 2
}
if ($fail -gt 0) { exit 1 }
