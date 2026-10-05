# probe-host-oracle.ps1 -- is the CPU host path bit-identical to the CUDA blocks path,
# and is it invariant to batch size?
#
# WHAT THIS IS FOR
# ----------------
# Every other comparison in this project puts two implementations of the same thing
# against each other.  This one puts the HOST against the DEVICE, which is the
# comparison that actually answers "is the CPU path right": the blocks engine is the
# one with a known-correct unvisited-pixel fill, and the host path is supposed to agree
# with it exactly.
#
# Two properties, both of which have been defects at least once:
#
#   1. host(batch 1) == device          the host dither is right
#   2. host(batch 1) == host(batch 16) the host dither does not depend on batch size,
#                                     which is where the batch-planar bug lived
#
# WHY IT IS REWRITTEN, since the arithmetic did not change
# -------------------------------------------------------
# The old version had NO EXIT STATEMENT AT ALL.  Its exit code was therefore whatever
# the last native command happened to leave in $LASTEXITCODE -- ffmpeg, inside
# Get-DecHash, on the last of three files -- so it exited 0 having printed
#
#   HOST-ORACLE: 1 MISMATCH(ES) -- R1 must not land on top of this
#
# and also exited 0 on the stronger line it had for exactly this case:
#
#   the device produced no hash -- the oracle is VOID      ($fail = 99)
#
# An oracle that reports a mismatch and returns success is worse than no oracle,
# because it is read as a clean run.  Measured on the pre-rewrite file: -Clip does not
# exist as a parameter here, so the demonstration is the missing source image, below.
#
# Three more things were open in the same file:
#
#   * $rc / $rc1 / $rc16 were PRINTED and never tested, so an rdither that exited 1 and
#     wrote nothing still reached the comparison;
#   * the file opened with four hardcoded absolute paths -- a CUDA dir, an ImageMagick
#     prefix, a prepended PATH and `Set-Location 'J:\ai code gen\dither 2'` -- with no
#     $ErrorActionPreference, so on any other machine Set-Location printed an error,
#     carried on, and then ran .\.build\Release\rdither.exe against the wrong root
#     while the stale $LASTEXITCODE made the three $rc values read 0;
#   * nothing checked that the deterministic source image had been produced at all, so
#     a failing `magick` fed the probes a stale file from a previous run.
#
# EXIT CODES, the convention the rest of the suite uses
# ---------------------------------------------------
#   0  both properties held
#   1  a run failed, or a property did not hold
#   2  nothing measurable -- no magick, no ffmpeg, no rdither, no source image, or no
#      usable device engine.  "Cannot run", never 0.
#
# Usage
#   .\tools\probe-host-oracle.ps1
#   .\tools\probe-host-oracle.ps1 -Rdither build\Release\rdither.exe -Colors 16

param(
  [string]$Rdither = "",
  [string]$Magick = "",
  [string]$Ffmpeg = "",
  [int]$Colors = 16,
  [int]$Width = 320,
  [int]$Height = 180,
  [int]$CpuThreads = 4,
  [string]$Work = "",
  # Hang guard per run.  Without one a wedged rdither hangs this script forever, which
  # is indistinguishable from a slow machine.  900 s matches the ceiling rdither puts on
  # its own ffmpeg calls.
  [int]$TimeoutSec = 900
)

$ErrorActionPreference = 'Stop'

# ---- resolve the three things this probe cannot work without ------------------
if (-not $Rdither) {
  $Rdither = Join-Path (Split-Path $PSScriptRoot -Parent) 'build\Release\rdither.exe'
}
if (-not (Test-Path $Rdither)) { "cannot run: $Rdither not built"; exit 2 }

function Resolve-Tool([string]$Name, [string]$Given, [string]$Hint) {
  if ($Given) {
    if (-not (Test-Path $Given)) { "cannot run: -$Hint '$Given' does not exist"; exit 2 }
    return $Given
  }
  $c = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue |
       Select-Object -First 1
  if (-not $c) { "cannot run: $Name not on PATH"; "  pass -$Hint with a full path"; exit 2 }
  return $c.Source
}
$Magick = Resolve-Tool 'magick' $Magick 'Magick'
$Ffmpeg = Resolve-Tool 'ffmpeg' $Ffmpeg 'Ffmpeg'

if (-not $Work) { $Work = Join-Path ([IO.Path]::GetTempPath()) 'rd_hostref' }
New-Item -ItemType Directory -Force -Path $Work | Out-Null

# ---- the source frame --------------------------------------------------------
# 320x180: small enough that a host-vs-device comparison costs almost nothing, and
# non-power-of-two on both axes so the curve does NOT cover the frame exactly once --
# which is the case where the scatter's `owner[pixel] == i` filter actually does work.
$src = Join-Path $Work 'ref.png'
# DETERMINISTIC source.  This was +noise Gaussian with no -seed, which meant a
# different image on every run: the three hashes below agreed with each other but
# CHANGED between runs, so the probe could not detect a change in the code under test
# at all.  A self-consistent check that cannot see its own subject is worse than none.
Remove-Item -EA SilentlyContinue $src
& $Magick -size "${Width}x${Height}" plasma:fractal -seed 12345 -attenuate 0.4 `
    +noise Gaussian -colorspace sRGB $src 2>&1 | Out-Null
# CHECKED, not assumed.  The old version used whatever file happened to be at this path,
# so a failed `magick` silently compared this run against the last run's source.
if (-not (Test-Path $src)) {
  "cannot run: magick did not produce the source image at $src"
  '  the oracle needs a frame to dither; without one there is nothing to compare.'
  exit 2
}
"source: ${Width}x${Height} (deliberately not a power of two on either axis)"

# ---- is there a device engine to be the oracle? --------------------------------
# The blocks engine is the reference: it is the engine with a known-correct
# unvisited-pixel fill and the one the host path is supposed to agree with.  If this
# build cannot run it, there is no oracle and the honest answer is "cannot run" -- not
# a comparison of the host against itself.
. "$PSScriptRoot\rd-engine-probe.ps1"
$avail = Get-RdEngineAvailability -Rdither $Rdither -Engines @('blocks') -Fixture $src
Write-RdEngineSkipReport -Availability $avail | Out-Null
if ($avail['blocks']) {
  "cannot run: no usable device engine -- $($avail['blocks'])"
  '  the host path is only measurable against the device path, so with neither there'
  '  is no oracle.  This is deliberately not a pass.'
  exit 2
}

# ---- one run, with its exit code ---------------------------------------------
function Invoke-Rd([string[]]$RdArgs, [string]$Out) {
  Remove-Item -EA SilentlyContinue $Out
  # rdither's own stage lines are redirected, not inherited.  The old version piped them
  # to Out-Null; letting them reach the console buries this file's verdict in forty
  # lines of per-stage timing, which is the output a reader actually skims.
  $tag = [IO.Path]::GetFileNameWithoutExtension($Out)
  $so = Join-Path $Work "$tag.out"
  $se = Join-Path $Work "$tag.err"
  $p = Start-Process -FilePath $Rdither -ArgumentList $RdArgs -PassThru -NoNewWindow `
        -RedirectStandardOutput $so -RedirectStandardError $se
  if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    try { $p.Kill($true) } catch { try { $p.Kill() } catch { } }
    try { $p.WaitForExit(10000) | Out-Null } catch { }
    return [pscustomobject]@{ Rc = -1; Out = ''; Why = "rdither did not exit within $TimeoutSec s and was killed" }
  }
  $rc = $p.ExitCode
  $why = $null
  if ($rc -ne 0) { $why = "rdither exited $rc" }
  elseif (-not (Test-Path $Out)) { $why = 'rdither exited 0 but wrote no output' }
  return [pscustomobject]@{ Rc = $rc; Out = $Out; Why = $why }
}

# The output path is the SECOND argument and is deliberately outside the array.  With
# $ErrorActionPreference = 'Stop' set above, the first version of these three lines put
# it inside -- Invoke-Rd @('...', $src, (Join-Path $Work 'dev.mp4')) -- so the array
# became the whole argument list, $Out bound empty, and
# `Remove-Item -EA SilentlyContinue $Out` died on "cannot bind argument to parameter
# 'Path' because it is an empty string".  An unbound parameter is the same class of
# defect as the -Args trap in probe-swscale.ps1: a silent empty where a value was
# required, and the only reason it was loud is the Stop preference above.
$devOut  = Join-Path $Work 'dev.mp4'
$b1Out   = Join-Path $Work 'host_b1.mp4'
$b16Out  = Join-Path $Work 'host_b16.mp4'

$dev  = Invoke-Rd @('--video','--colors',"$Colors",'--engine','blocks',
                     '--batch-frames','1',$src,$devOut) $devOut
$h1   = Invoke-Rd @('--video','--colors',"$Colors",'--engine','cpu',
                     '--cpu-threads',"$CpuThreads",'--batch-frames','1',
                     $src,$b1Out) $b1Out
$h16  = Invoke-Rd @('--video','--colors',"$Colors",'--engine','cpu',
                     '--cpu-threads',"$CpuThreads",'--batch-frames','16',
                     $src,$b16Out) $b16Out

# The exit codes were printed and never tested.  This is where they are tested.
$failed = @()
foreach ($pair in @(@('device',$dev), @('host batch 1',$h1), @('host batch 16',$h16))) {
  if ($pair[1].Why) { $failed += "$($pair[0]): $($pair[1].Why)" }
}
"device exit=$($dev.Rc)   host batch1 exit=$($h1.Rc)   host batch16 exit=$($h16.Rc)"
""
if ($failed.Count -gt 0) {
  "HOST-ORACLE: FAILED -- $($failed.Count) of 3 runs produced no output."
  foreach ($f in $failed) { "  $f" }
  '  Nothing was compared.  This used to print the three exit codes and carry on.'
  exit 1
}

# ---- compare decoded pixels, not container bytes -------------------------------
# h264/x264 is lossy so the FILES will differ for reasons that have nothing to do with
# the dither.  ffmpeg md5 over the rawvideo stream, streaming, so no multi-GB raw dump.
function Get-DecHash([string]$f) {
  if (-not (Test-Path $f)) { return $null }
  $h = (& $Ffmpeg -v error -i $f -f rawvideo -pix_fmt rgba -f md5 - 2>$null | Out-String).Trim()
  # ffmpeg's exit code AND the shape of its output.  A non-zero exit with a leftover
  # hash on stdout is not a measurement.
  if ($LASTEXITCODE -ne 0) { return $null }
  if ($h -notmatch '^MD5=([0-9a-f]{32})') { return $null }
  return $Matches[1]
}
$devH  = Get-DecHash $dev.Out
$h1H   = Get-DecHash $h1.Out
$h16H  = Get-DecHash $h16.Out

"device    md5 $devH"
"host b1   md5 $h1H"
"host b16  md5 $h16H"
""

# Nothing to compare.  This is the case the old file called "the oracle is VOID" and
# then exited 0.
if ($null -eq $devH -or $null -eq $h1H -or $null -eq $h16H) {
  $unhashed = @()
  if ($null -eq $devH) { $unhashed += 'device' }
  if ($null -eq $h1H)  { $unhashed += 'host b1' }
  if ($null -eq $h16H) { $unhashed += 'host b16' }
  "HOST-ORACLE: cannot run -- ffmpeg produced no usable hash for $($unhashed -join ', ')"
  '  Every run exited 0 and wrote a file, so this is ffmpeg refusing the file, not the'
  '  dither.  The old file called this "the oracle is VOID" and still exited 0.'
  exit 2
}

$fail = 0
if ($h1H -ne $devH)  { "  MISMATCH: host batch 1 differs from device"; $fail++ }
if ($h16H -ne $h1H)  { "  MISMATCH: host batch 16 differs from host batch 1 -- the host path is not batch-invariant"; $fail++ }
if ($h1H -eq $h16H)  { "  host is batch-invariant (b1 == b16)" }
if ($fail -eq 0) { "HOST-ORACLE: host == device, and host is batch-invariant"; exit 0 }
"HOST-ORACLE: $fail MISMATCH(ES) -- this must not be left on top of a green run"
exit 1