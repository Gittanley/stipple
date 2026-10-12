# probe-quiet-flag.ps1 -- does --quiet actually quiet?
#
# WHY THIS EXISTS.  `--quiet` did nothing on the video path: 11 lines on stdout with
# the flag, 0 on the image path with the same flag.  The image path gates every print
# on opt.quiet and the video path gated none of them.  It survived because no stage ran
# the flag on video, and because "it compiles and exits 0" says nothing about output.
#
# WHAT IT ASSERTS, per path:
#
#   --quiet      stdout is EMPTY
#   no --quiet   stdout is NOT empty
#   both         the output file exists, and the exit code is 0
#
# The second half is the half that matters.  A probe that only checks "quiet is silent"
# passes on a binary that prints nothing ever, which is the same lie in a new direction.
#
# STDERR IS NOT ASSERTED and deliberately so: the encoder, RAM-budget and
# "no audio stream" warnings are diagnostics, not progress, and --quiet is not meant to
# swallow them (`curl -s` vs `curl -S`).  If that decision is ever reversed, this probe
# is where the change belongs.
#
# No clip is committed to this repository (.gitignore excludes *.mkv and *.mp4), so the
# fixture is built here, the way every other probe does it.
#
# Exit codes:  0 pass   1 FAIL   2 cannot run (reason printed)

param(
  [string]$Rdither = "",
  [string]$Ffmpeg  = ""
)

$ErrorActionPreference = 'Stop'

function Resolve-Tool([string]$name, [string]$given) {
  if ($given -and (Test-Path $given)) { return (Resolve-Path $given).Path }
  $c = Get-Command $name -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  return ""
}

$exe = Resolve-Tool 'rdither.exe' $Rdither
$ffmpeg = Resolve-Tool 'ffmpeg.exe' $Ffmpeg

if (-not $exe) { "quiet flag: no rdither given and none on PATH; cannot run"; exit 2 }
if (-not $ffmpeg) { "quiet flag: no ffmpeg on PATH; cannot run"; exit 2 }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("rd_quiet_" + [Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
  $src = Join-Path $tmp 'fixture.mkv'
  & $ffmpeg -v error -y -f lavfi -i 'testsrc=size=160x90:rate=10:duration=1' `
                   -c:v ffv1 -pix_fmt yuv444p $src 2>&1 | Out-Null
  if (-not (Test-Path $src)) {
    "quiet flag: could not build a fixture with ffmpeg; cannot run"
    exit 2
  }
  $img = Join-Path $tmp 'in.png'
  & $ffmpeg -v error -y -f lavfi -i 'testsrc=size=64x32:rate=1:duration=1' `
                   -frames:v 1 $img 2>&1 | Out-Null
  if (-not (Test-Path $img)) {
    "quiet flag: could not build a PNG fixture with ffmpeg; cannot run"
    exit 2
  }

  # Runs one case and returns [stdoutLines, exitCode, outPath].
  # stdout ONLY: 2>$null drops stderr deliberately, see the header.
  function Invoke-Case([string]$outName, [string[]]$args_, [string]$inputPath) {
    $out = Join-Path $tmp $outName
    $r = & $exe @args_ $inputPath $out 2>$null
    return ,@(@($r).Count, $LASTEXITCODE, $out)
  }

  # Accumulator is `$problems`, NOT `$fail`: a `$script:fail` anywhere in a function
  # called from script scope is the SAME variable as a plain `$fail` here, so the first
  # version of this probe reset its own accumulator and printed "ok" over a FAIL.
  $problems = 0
  function Check([string]$label, [int]$quietLines, [int]$loudLines, [int]$code,
                 [string]$out, [bool]$expectOutput) {
    $bad = 0
    if ($quietLines -ne 0) { $bad++ }
    if ($loudLines -eq 0) { $bad++ }
    if ($code -ne 0) { $bad++ }
    if ($expectOutput -and -not (Test-Path $out)) { $bad++ }
    $mark = if ($bad -eq 0) { 'ok  ' } else { 'FAIL' }
    Write-Host ("  {0} {1,-22} quiet {2} line(s), loud {3} line(s), exit {4}, output {5}" -f `
                $mark, $label, $quietLines, $loudLines, $code,
                $(if ($expectOutput) { Test-Path $out } else { 'n/a' }))
    return $bad
  }

  $vbase = @('--video','--engine','cpu','--colors','16')
  # expectOutput is false for --palette-only, which stops before the dither and so
  # writes nothing. Asserting a file there would be asserting the opposite of correct.
  $cases = @(
    @{ label = 'video';            quiet = @($vbase + @('--quiet'));                  out = 'vq.mkv';  want = $true  }
    @{ label = 'video segmented';  quiet = @($vbase + @('--quiet','--segment-frames','4')); out = 'sq.mkv';  want = $true  }
    @{ label = 'video palette-only'; quiet = @($vbase + @('--quiet','--palette-only')); out = 'po.mkv';  want = $false }
  )

  foreach ($c in $cases) {
    $q = Invoke-Case $c.out $c.quiet $src
    $l = Invoke-Case ($c.out + '.loud') ($c.quiet | Where-Object { $_ -ne '--quiet' }) $src
    $problems += Check $c.label $q[0] $l[0] $q[1] $q[2] $c.want
  }

  # The image path, which was already correct. Kept in the probe so a regression there
  # is caught by the same stage rather than being discovered by a user.
  $ib = @('--colors','16')
  $q = Invoke-Case 'iq.png' ($ib + @('--quiet')) $img
  $l = Invoke-Case 'iq.png.loud' $ib $img
  $problems += Check 'image' $q[0] $l[0] $q[1] $q[2] $true

  if ($problems -eq 0) {
    ""
    "quiet flag: ok -- silent when asked, still reporting otherwise, outputs written"
    exit 0
  }
  ""
  "quiet flag: FAILED ($problems problem(s))"
  exit 1
}
finally {
  Remove-Item $tmp -Recurse -Force -EA SilentlyContinue
}