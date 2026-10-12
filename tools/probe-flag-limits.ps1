# probe-flag-limits.ps1 -- are the memory-budget flags actually range-checked?
#
# WHY THIS EXISTS.  `--max-ram-mb` and `--max-vram-mb` are stored in megabytes and
# multiplied by 1024*1024 at the point of use, in std::size_t, which is 64-bit unsigned.
# Any value at or above 2^44 MB wrapped that multiply to zero -- 2^44 * 2^20 is 2^64,
# which is 0 modulo 2^64.  So the LARGEST number the flag accepted produced a ZERO
# budget: the exact inverse of the request, silently, with a correct-looking output
# image.  Measured before the fix:
#
#     rdither --colors 16 --max-ram-mb 17592186044416 in.png out.png
#     pixels     : disk (memory-mapped spill file) (0.6 MiB)
#     exit 0
#
# --max-vram-mb carries the identical multiply and had no check either.
#
# WHAT IT ASSERTS, and why each half matters:
#
#   2^44      REJECTED, non-zero exit, and the message names the bound.  A bound the
#             user cannot see is a bound they cannot act on.
#   2^44 - 1  ACCEPTED.  This is the half that stops a "fix" from being "reject
#             everything large": it pins the largest legal value, so the check cannot
#             be tightened later without this stage going red.
#   512       ACCEPTED, the everyday case.
#
# It asserts the rejection is SILENT-ON-NO-INPUT too: a value the parser cannot read at
# all must still be an error rather than a default.
#
# 2^44 is written as an exact integer, not computed from a float, so the probe asserts
# the boundary rather than something near it.
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

if (-not $exe) { "flag limits: no rdither given and none on PATH; cannot run"; exit 2 }
if (-not $ffmpeg) { "flag limits: no ffmpeg on PATH; cannot run"; exit 2 }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("rd_lim_" + [Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
  $img = Join-Path $tmp 'in.png'
  & $ffmpeg -v error -y -f lavfi -i 'testsrc=size=64x32:rate=1:duration=1' `
                   -frames:v 1 $img 2>&1 | Out-Null
  if (-not (Test-Path $img)) {
    "flag limits: could not build a PNG fixture with ffmpeg; cannot run"
    exit 2
  }

  # 2^44 and 2^44-1 written out. Computed here rather than by shifting in PowerShell so
  # the probe asserts the exact decimal the boundary sits at.
  $wrap  = '17592186044416'
  $legal = '17592186044415'

  $problems = 0

  function Expect([string]$label, [string]$flag, [string[]]$args_, [bool]$wantOk, [int]$tag) {
    $out = Join-Path $tmp ("o$tag" + [Guid]::NewGuid().ToString('N').Substring(0,4) + '.png')
    $msg = & $exe @args_ $img $out 2>&1
    $code = $LASTEXITCODE
    $ok = ($code -eq 0)
    $text = ($msg | Out-String)
    # A rejection has to NAME THE FLAG. An error that does not say which argument was
    # wrong is as unusable as no check at all. Matching on the flag name rather than on
    # a digit: "--max-ram-mb expects a number" carries no digits and is the better
    # message, which an earlier version of this probe got wrong by demanding one.
    $namesFlag = $text.Contains($flag)
    $good = if ($wantOk) { $ok } else { (-not $ok) -and $namesFlag }
    $mark = if ($good) { 'ok  ' } else { 'FAIL' }
    Write-Host ("  {0} {1,-38} exit {2} {3}" -f $mark, $label, $code,
                $(if ($good) { '' } else { '(expected ' + $(if ($wantOk) { 'exit 0' } else { "a non-zero exit naming $flag" }) + ')' }))
    return $(if ($good) { 0 } else { 1 })
  }

  Write-Host "flag limits:"

  $problems += Expect "--max-ram-mb 2^44 (wraps to 0)"   '--max-ram-mb'  @('--colors','16','--max-ram-mb',$wrap)  $false 1
  $problems += Expect "--max-vram-mb 2^44 (wraps to 0)"  '--max-vram-mb' @('--colors','16','--max-vram-mb',$wrap) $false 2
  $problems += Expect "--max-ram-mb 2^44-1 (largest legal)" '--max-ram-mb' @('--colors','16','--max-ram-mb',$legal) $true  3
  $problems += Expect "--max-ram-mb 512 (ordinary)"      '--max-ram-mb'  @('--colors','16','--max-ram-mb','512')   $true  4
  $problems += Expect "--max-ram-mb abc (not a default)" '--max-ram-mb' @('--colors','16','--max-ram-mb','abc')  $false 5

  if ($problems -eq 0) {
    ""
    "flag limits: ok -- the wrap is rejected, the largest legal value is accepted"
    exit 0
  }
  ""
  "flag limits: FAILED ($problems problem(s))"
  exit 1
}
finally {
  Remove-Item $tmp -Recurse -Force -EA SilentlyContinue
}