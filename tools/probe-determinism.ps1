# Is the same command twice in a row the same file, every time?
#
# This exists because of a bug that bit-exactness testing structurally CANNOT
# find.  The unfilled-channel fault in ImStore (see ZeroQueuedRow in src/rd_im.cpp)
# made the PNG coder choose a different encoding -- TrueColorAlpha, PaletteAlpha,
# Gray, Bilevel -- from whatever stale heap memory happened to be in the fifth
# channel.  Both engines went through the same writer, so comparing OpenCL against
# CUDA found nothing: they were wrong in the same way at the same time, roughly a
# third of runs' worth of agreement.  The probe-opencl-exact.ps1 sweep only caught
# it because it happened to run enough cells for the coin flips to disagree.
#
# A test that compares two implementations cannot see a fault they share.  This
# one compares an implementation against ITSELF across time, which is the only
# thing that sees it.  Two rules follow, and both are load-bearing:
#
#   * Repeat a case enough times to see an intermittent fault.  Once is not a
#     test of determinism; it is a test of luck.  The default here is 12, and the
#     fault showed up in 3 runs of 16 -- so 12 is comfortably past the point where
#     a regression cannot hide.
#
#   * Hash the DECODED PIXELS, and hash the file WITH METADATA STRIPPED.  Not the
#     raw file.  ImageMagick writes a `png:tIME` chunk recording when the file was
#     written, so two runs a second apart produce different bytes from identical
#     pixels -- measured here at 5 distinct files over 6 runs, with 1 distinct
#     pixel set and 1 distinct stripped file.  That is PNG doing its job, not a
#     fault, and no image tool promises byte-identical files across time.  (If
#     byte-reproducible output is wanted, `png:exclude-chunk=date,time` on the
#     write does it; it is not the default here because this project's stated bar
#     is matching ImageMagick's *pixels*, per DESIGN.md section 8 item 5.)
#
#     Comparing stripped bytes rather than giving up on the file is what makes the
#     second half of this test worth running: "the encoder picked a different type
#     again" is a real bug and shows up as differing stripped files, while a
#     timestamp shows up as differing raw files with identical stripped ones.
#
# Every engine is swept, not just the GPU ones.  The fault was worse on OpenCL only
# because its extra host allocations rearranged the heap; it was present on all of
# them, and a test that only watches the fast path will call the bug flaky.

param(
  # The SHARED fixture directory, matching tools\probe-opencl-exact.ps1's -FixtureDir.
  # These are inputs, generated once and only read here, so sharing them is correct and
  # is what verify.ps1's "run probe-opencl-exact.ps1 first" instruction relies on.
  [string]$FixtureDir = "",
  [int]$Runs = 12
)

# Per-run OUTPUT directory, unique per process.  This used to be one fixed name shared
# with probe-opencl-exact.ps1, so two concurrent probes used one directory and compared
# against each other's half-written files.  Shared inputs, private outputs.
if (-not $FixtureDir) { $FixtureDir = Join-Path $env:TEMP 'rdocl_fixtures' }
$Dir = Join-Path $env:TEMP ('rddet_' + [Guid]::NewGuid().ToString('N').Substring(0, 8))

# NOT 'Stop'.  rdither writes diagnostics to stderr, and a native command's stderr
# surfaces as a non-terminating error record; under 'Stop' the first "[blocks]"
# notice aborts the sweep before it starts.  Failure is detected from exit codes
# and from the decoded hash count, which are the things that actually mean failure.
$ErrorActionPreference = 'Continue'
$rd = Join-Path (Split-Path $PSScriptRoot -Parent) 'build\Release\rdither.exe'
if (-not (Test-Path $rd)) { "missing $rd -- build first"; exit 1 }
if (-not (Test-Path $FixtureDir)) { New-Item -ItemType Directory -Force -Path $FixtureDir | Out-Null }
# BOTH directories are created here.  Creating only $FixtureDir -- which is what an
# earlier version of this split did, by replacing this line instead of adding to it --
# leaves $Dir nonexistent, so every per-run output path is a write into a directory
# that was never made, and all 14 cases report "produced no output".  The symptom
# names fourteen defects (three engines, eight algorithms) and the cause is one
# missing line here.  A missing New-Item is the cheapest possible mistake and the most
# expensive one to read, because the failure is attributed to the code under test.
if (-not (Test-Path $Dir)) { New-Item -ItemType Directory -Force -Path $Dir | Out-Null }

# Same images the OpenCL sweep uses, so a regression in either is the same
# regression.  c_ashape and c_alpha50 carry alpha; c_plasma and c_noise do not.
$imgs = @('c_plasma', 'c_ashape')
$engines = @('cpu', 'blocks', 'opencl')

# Missing fixtures are "cannot test", which is not the same as "failed", and
# reporting one as the other is how a broken test gets trusted.  Exit 2 says so.
$missing = @($imgs | Where-Object { -not (Test-Path "$FixtureDir\$_.png") })
if ($missing.Count -gt 0) {
  ''
  # $FixtureDir, NOT $Dir.  This said $Dir, which is this run's own private output
  # directory and by design never holds a fixture -- so the message pointed the reader
  # at a directory that could not contain the thing it was asking for, and the obvious
  # next step (go and look there) finds nothing.  Found by reading the diff of my own
  # FixtureDir split, not by a failing run: the message is only printed when the
  # fixtures are genuinely missing, which is exactly when nobody is watching closely.
  "cannot run: no fixture for $($missing -join ', ') in $FixtureDir"
  'these are generated by tools\probe-opencl-exact.ps1 -- run that once first'
  exit 2
}

# Decode to raw RGBA through cmd /c rather than the PowerShell pipeline.  A native
# command's stdout piped through PowerShell is re-encoded, which turns the binary
# into text and destroys the very thing being hashed.
function Get-PixelHash([string]$png) {
  $raw = [IO.Path]::ChangeExtension($png, '.raw')
  Remove-Item -EA SilentlyContinue $raw
  cmd /c "magick ""$png"" -depth 8 ""rgba:$raw""" 2>$null | Out-Null
  if (-not (Test-Path $raw)) { return $null }
  $h = (Get-FileHash $raw -Algorithm SHA256).Hash
  Remove-Item -EA SilentlyContinue $raw
  return $h
}

# The same file with metadata removed, so a tIME difference does not register as an
# encoder difference.  -strip rewrites through IM rather than editing bytes, so it
# also re-runs the encoder -- which means this hash is only meaningful because the
# pixels have already been checked: if the pixels are stable and this varies, the
# encoder really is choosing differently.
function Get-StrippedHash([string]$png) {
  $tmp = [IO.Path]::ChangeExtension($png, '.strip.png')
  Remove-Item -EA SilentlyContinue $tmp
  cmd /c "magick ""$png"" -strip ""$tmp""" 2>$null | Out-Null
  if (-not (Test-Path $tmp)) { return $null }
  $h = (Get-FileHash $tmp -Algorithm SHA256).Hash
  Remove-Item -EA SilentlyContinue $tmp
  return $h
}

# An engine that cannot run at all makes determinism unaskable, not failed.  Two
# different reasons, both environmental rather than a defect:
#
#   "this build has no OpenCL"  -- src/rd_opencl.cpp's #else stub, because the SDK is
#                                  gitignored and absent from a fresh clone
#   "no CUDA device"            -- compiled in, but this machine has no NVIDIA GPU
#
# Both used to report as "12 of 12 runs produced no output" and then FAIL, so a fresh
# clone with no OpenCL SDK exited 1 on a suite whose other 12 cases were green.  A
# stranger's first impression of the project was a broken test suite.
#
# The matching lives in tools\rd-engine-probe.ps1, shared with verify.ps1.  It is one
# file because it is a list of exact diagnostic strings, and an unrecognised error
# deliberately counts as RUNNABLE -- a broken engine also produces no output, and a
# looser rule would let a real defect hide behind a skip.
. "$PSScriptRoot\rd-engine-probe.ps1"
$avail = Get-RdEngineAvailability -Rdither $rd -Engines $engines -Fixture "$FixtureDir\$($imgs[0]).png"
Write-RdEngineSkipReport -Availability $avail | Out-Null
$skipEngines = @{}
foreach ($k in $avail.Keys) { if ($avail[$k]) { $skipEngines[$k] = $avail[$k] } }
$anyUsable = @($engines | Where-Object { -not $skipEngines.ContainsKey($_) }).Count -gt 0
if (-not $anyUsable) {
  ''
  'No engine in this build can run, so there is nothing to test.  Exit 2.'
  exit 2
}

$fail = 0
$total = 0
''
"Determinism: $Runs runs per case, decoded pixels"
''

foreach ($img in $imgs) {
  $src = "$FixtureDir\$img.png"
  foreach ($eng in $engines) {
    if ($skipEngines.ContainsKey($eng)) { continue }
    $total++
    $pixelHashes = @{}
    $strippedHashes = @{}
    $rawHashes = @{}
    $bad = 0
    for ($i = 1; $i -le $Runs; $i++) {
      $out = "$Dir\det_${img}_${eng}_$i.png"
      # Delete first.  A run that fails leaves a stale file, and a stale file
      # hashes fine -- so a crashed engine would report as a stable engine, which
      # is the precise opposite of the truth.
      Remove-Item -EA SilentlyContinue $out
      & $rd --engine $eng --blocks 16 --colors 2 $src $out 2>$null | Out-Null
      if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { $bad++; continue }
      $ph = Get-PixelHash $out
      $sh = Get-StrippedHash $out
      if ($null -eq $ph -or $null -eq $sh) { $bad++; continue }
      $pixelHashes[$ph] = 1
      $strippedHashes[$sh] = 1
      $rawHashes[(Get-FileHash $out -Algorithm SHA256).Hash] = 1
      Remove-Item -EA SilentlyContinue $out
    }
    $np = $pixelHashes.Count
    $ns = $strippedHashes.Count
    $nr = $rawHashes.Count
    $tag = "{0,-8} {1,-7}" -f $img, $eng
    if ($bad -gt 0) {
      $fail++
      "  $tag  FAIL  $bad of $Runs runs produced no output"
    } elseif ($np -ne 1) {
      $fail++
      "  $tag  FAIL  $np distinct pixel sets over $Runs runs -- NOT DETERMINISTIC"
    } elseif ($ns -ne 1) {
      $fail++
      "  $tag  FAIL  pixels stable but $ns distinct files after -strip -- the encoder is the variable"
    } else {
      # Raw files differing while stripped files match is the tIME chunk, which is
      # PNG working as intended.  Worth surfacing, never worth failing on.
      $note = if ($nr -eq 1) { '' } else { "   ($nr raw files differ: png:tIME only)" }
      "  $tag  ok    1 pixel set, 1 stripped file, $Runs runs$note"
    }
  }
}

# The non-Riemersma algorithms, which the sweep above never touches: it varies the
# ENGINE, and every one of these is CPU code reached by the same engine.  They differ
# from each other in a way an engine sweep cannot see, so without this a broken
# algorithm would be indistinguishable from a correct one.
#
# Fewer runs, because a defect in a closed-form threshold matrix is a type error rather
# than a race -- it either truncates or it does not, on every run.  One such bug was
# `unsigned char` holding ranks up to 4096, which made two supposedly different void-
# and-cluster tile sizes byte-identical; 4 runs is ample for that, and the matrix build
# is the slowest thing in the suite.
''
$algos = @('floyd-steinberg', 'atkinson', 'jarvis', 'clustered-dot',
           'bayer', 'bayer-ordered', 'void-and-cluster', 'void-and-cluster-fast')
$algoRuns = [Math]::Min($Runs, 4)
"Determinism: $algoRuns runs per algorithm, non-Riemersma"
''
foreach ($algo in $algos) {
  $total++
  $hashes2 = @{}
  $bad2 = 0
  # $FixtureDir, not $Dir.  This is the algorithm block's INPUT, and it was the one
  # reference the $Dir -> $FixtureDir split missed.  It reported as eight algorithms
  # failing with "produced no output" when all eight are fine.
  $src2 = "$FixtureDir\c_plasma.png"
  for ($i = 1; $i -le $algoRuns; $i++) {
    $o = "$Dir\algo_${algo}_$i.png"
    Remove-Item -EA SilentlyContinue $o
    & $rd --dither $algo --colors 4 $src2 $o 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $o)) { $bad2++; continue }
    $h2 = Get-PixelHash $o
    if ($null -eq $h2) { $bad2++; continue }
    $hashes2[$h2] = 1
    Remove-Item -EA SilentlyContinue $o
  }
  if ($bad2 -gt 0) {
    $fail++
    "  {0,-22} FAIL  $bad2 of $algoRuns runs produced no output" -f $algo
  } elseif ($hashes2.Count -ne 1) {
    $fail++
    "  {0,-22} FAIL  $($hashes2.Count) distinct pixel sets over $algoRuns runs" -f $algo
  } else {
    "  {0,-22} ok    1 pixel set, $algoRuns runs" -f $algo
  }
}

''
if ($fail -eq 0) {
  "{0} of {1} cases deterministic." -f $total, $total
  exit 0
}
# $fail, NOT ($total - $fail).  This said "($total - $fail) of $total cases FAILED",
# so a run with 2 of 14 cases failing printed "12 of 14 cases FAILED" -- it counted
# every PASSING case as a failure, directly above a table showing 12 ok lines and
# 2 FAIL lines.  The output contradicted itself and the error ran in the direction
# that makes a mostly-healthy suite look broken.
#
# Line 193 above had the same expression and was only ever correct by accident:
# it is reached solely when $fail -eq 0, where $total - 0 happens to equal $total.
# Writing $fail in both places removes the coincidence.
"{0} of {1} cases FAILED." -f $fail, $total
exit 1
