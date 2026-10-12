# probe-palette-parity.ps1 -- do PNG and .txt palette imports behave the SAME way?
#
# WHY THIS EXISTS.  `--palette-import` had two readers that disagreed.  The `.txt` branch
# validated --colors against the file's own colour count and refused a mismatch, naming
# both numbers.  The image branch did not: it clamped and compared the CLAMPED count, so
# `if (colors > 0 && n > colors) n = colors;` guaranteed `n == colors` whenever the image
# held more than asked for, and any check written after that point compared the file with
# itself.  A 6-colour PNG at --colors 16 was accepted, exit 0, dithered to 6, no message --
# while the same palette as .txt was refused.
#
# The bug was not "a missing check". It was an ASYMMETRY between two paths that should
# agree, which is why this asserts agreement rather than asserting either behaviour:
# a probe that pinned the .txt behaviour alone would still be green on a binary where
# the image branch had drifted further.
#
# WHAT IT ASSERTS, for BOTH formats:
#
#   matching --colors    accepted, exit 0
#   mismatched --colors  refused, non-zero exit, message names BOTH numbers
#   and, crucially: the two formats reach the SAME verdict for the same inputs
#
# It builds its own palettes with the binary under test, so it is not asserting anything
# about fixtures it did not create, and a change to the exporter cannot silently make
# this probe pass on a stale file.
#
# No clip is committed to this repository (.gitignore excludes *.mkv and *.mp4), so the
# fixture is built here.
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

if (-not $exe) { "palette parity: no rdither given and none on PATH; cannot run"; exit 2 }
if (-not $ffmpeg) { "palette parity: no ffmpeg on PATH; cannot run"; exit 2 }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("rd_par_" + [Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
  $src = Join-Path $tmp 'fixture.mkv'
  & $ffmpeg -v error -y -f lavfi -i 'testsrc=size=160x90:rate=10:duration=1' `
                   -c:v ffv1 -pix_fmt yuv444p $src 2>&1 | Out-Null
  if (-not (Test-Path $src)) {
    "palette parity: could not build a fixture with ffmpeg; cannot run"
    exit 2
  }

  # A 16-colour palette, written by the binary under test in both formats. 16 is chosen
  # because the bug needed "fewer colours than asked for" to be invisible: a palette with
  # MORE than --colors would have tripped the clamp and produced the right answer by
  # accident.
  $palPng = Join-Path $tmp 'p16.png'
  $palTxt = Join-Path $tmp 'p16.txt'
  & $exe --video --engine cpu --colors 16 --palette-export $palPng $src (Join-Path $tmp 'seed.mkv') 2>&1 | Out-Null
  & $exe --video --engine cpu --colors 16 --palette-export $palTxt $src (Join-Path $tmp 'seed.mkv') 2>&1 | Out-Null
  if (-not (Test-Path $palPng) -or -not (Test-Path $palTxt)) {
    "palette parity: could not export both palette formats; cannot run"
    exit 2
  }
  $rows = @(Get-Content $palTxt | Where-Object { $_ -notmatch '^#' -and $_.Trim() })
  # The palette's ACTUAL colour count, read back from what was just written. Hardcoding
  # 16 was wrong on the first run of this probe: a 160x90 lavfi fixture quantises to 14
  # colours, so --colors 16 was genuinely a mismatch and BOTH formats correctly refused
  # it. A probe that assumes its own fixture's colour count is testing the fixture, not
  # the reader.
  $n = $rows.Count
  if ($n -lt 1) { "palette parity: exported palette is empty; cannot run"; exit 2 }
  # A mismatch value that cannot collide with the real one, and is not a multiple of it
  # either, so nothing can make the two agree by arithmetic accident.
  $m = $n + 48
  Write-Host ("palette parity: fixtures built, palette holds {0} colours (mismatch uses --colors {1})" -f $n, $m)

  $problems = 0

  # Returns [exitCode, message]. --palette-import is video-only, which is why a clip is
  # needed here at all.
  function Try-Import([string]$pal, [int]$colors) {
    $out = Join-Path $tmp ([Guid]::NewGuid().ToString('N').Substring(0,6) + '.mkv')
    $m = & $exe --video --engine cpu --colors $colors --palette-import $pal $src $out 2>&1
    return ,@($LASTEXITCODE, ($m | Out-String))
  }

  $results = @{}
  foreach ($fmt in @('png','txt')) {
    $pal = if ($fmt -eq 'png') { $palPng } else { $palTxt }

    $r = Try-Import $pal $n
    $results["$fmt-match"] = $r
    $good = ($r[0] -eq 0)
    Write-Host ("  {0} {1} at matching --colors {2} -> exit {3} (expect 0)" -f `
                $(if ($good) { 'ok  ' } else { 'FAIL' }), $fmt, $n, $r[0])
    if (-not $good) { $problems++ }

    $r = Try-Import $pal $m
    $results["$fmt-mismatch"] = $r
    $text = $r[1]
    $namesBoth = $text.Contains([string]$n) -and $text.Contains([string]$m)
    $good = ($r[0] -ne 0) -and $namesBoth
    Write-Host ("  {0} {1} at mismatched --colors {2} -> exit {3}, names both numbers: {4} (expect non-zero, both)" -f `
                $(if ($good) { 'ok  ' } else { 'FAIL' }), $fmt, $m, $r[0], $namesBoth)
    if (-not $good) { $problems++ }
  }

  # The parity assertion itself, which is the one that would have caught the original.
  foreach ($k in @('match','mismatch')) {
    $a = $results["png-$k"][0] -eq 0
    $b = $results["txt-$k"][0] -eq 0
    $same = ($a -eq $b)
    Write-Host ("  {0} png and txt AGREE on $k (png accepted={1}, txt accepted={2})" -f `
                $(if ($same) { 'ok  ' } else { 'FAIL' }), $a, $b)
    if (-not $same) { $problems++ }
  }

  # Two cases this probe did not cover when it was first written, and both were found
  # by bugs it sailed straight past.

  # ROUND-TRIP STABILITY.  The .txt form used to be described as lossless in
  # include/rd_im.h, on the grounds that %.1f of a Q16 channel is exact because a Q16
  # channel is an integer.  Both halves were false: in this build the channels are not
  # integers (44 of 48 values in an exported 16-colour palette were fractional), and
  # export -> import -> export changed every one of sixteen rows.  A description is not
  # evidence, so this asserts it.
  $rt1 = Join-Path $tmp 'rt1.txt'
  $rt2 = Join-Path $tmp 'rt2.txt'
  & $exe --video --engine cpu --colors $n --palette-export $rt1 $src (Join-Path $tmp 'rt1.mkv') 2>&1 | Out-Null
  # The count to import with must be read back from what was WRITTEN, not from $n.
  # Asking for --colors 14 yields 13 actual colours here: the canonical sort and the
  # tree's dedup drop duplicates, so the requested count is an upper bound rather than
  # a promise. Using $n made the import refuse its own export -- which is the
  # validation working correctly and the probe guessing wrongly. This is the second
  # time in this file that assumed a colour count instead of measuring it.
  $rtCount = 0
  if (Test-Path $rt1) {
    $rtCount = @(Get-Content $rt1 | Where-Object { $_ -notmatch '^#' -and $_.Trim() }).Count
  }
  & $exe --video --engine cpu --colors $rtCount --palette-import $rt1 --palette-export $rt2 $src (Join-Path $tmp 'rt2.mkv') 2>&1 | Out-Null
  if ((Test-Path $rt1) -and (Test-Path $rt2)) {
    # TWO properties, not one, because the first version of this assertion compared the
    # files byte-for-byte and was WRONG: the first import re-sorts canonically, so row 0
    # legitimately moves. That is SortPaletteCanonical doing its job, not data loss.
    # Comparing bytes reported "16 rows differ" on a format that had lost nothing.
    #
    #   (a) VALUES are preserved -- the colour set is unchanged. This is the lossless
    #       claim, and it is the one that was false under %.1f.
    #   (b) the SECOND pass is byte-identical -- normalisation is idempotent, so repeated
    #       export/import cycles converge instead of oscillating.
    $vals1 = @(Get-Content $rt1 | Where-Object { $_ -notmatch '^#' -and $_.Trim() } |
               ForEach-Object { (($_ -split '\s+') | Where-Object { $_ -ne '' })[1..3] -join ',' } | Sort-Object)
    $vals2 = @(Get-Content $rt2 | Where-Object { $_ -notmatch '^#' -and $_.Trim() } |
               ForEach-Object { (($_ -split '\s+') | Where-Object { $_ -ne '' })[1..3] -join ',' } | Sort-Object)
    $good = (($vals1 -join '|') -eq ($vals2 -join '|'))
    Write-Host ("  {0} txt round trip preserves every colour value" -f $(if ($good) { 'ok  ' } else { 'FAIL' }))
    if (-not $good) { $problems++ }

    $rt3 = Join-Path $tmp 'rt3.txt'
    & $exe --video --engine cpu --colors $rtCount --palette-import $rt2 --palette-export $rt3 $src (Join-Path $tmp 'rt3.mkv') 2>&1 | Out-Null
    $good = (Test-Path $rt3) -and ((Get-Content $rt2) -join '|') -eq ((Get-Content $rt3) -join '|')
    Write-Host ("  {0} txt normalisation is idempotent (pass 2 == pass 3 byte for byte)" -f `
                $(if ($good) { 'ok  ' } else { 'FAIL' }))
    if (-not $good) { $problems++ }
  } else {
    Write-Host "  FAIL txt round trip: could not produce both exports"
    $problems++
  }

  # A GAP in a palette file.  This probe originally passed green while that bug was
  # live, because it only ever compared a matching count against a mismatched one and
  # never asked whether the FILE was valid: a gap used to import with a BLACK entry,
  # because PaletteEntry's defaults are 0,0,0,opaque and nothing was checking.
  $gapped = Join-Path $tmp 'gap.txt'
  $body = Get-Content $rt1 | Where-Object { $_ -notmatch '^#' -and $_.Trim() }
  ($body | Where-Object { ($_ -split '\s+')[0] -ne '1' }) | Out-File -Encoding ascii $gapped
  $out = Join-Path $tmp 'gap.mkv'
  $msg = & $exe --video --engine cpu --colors ($n - 1) --palette-import $gapped $src $out 2>&1
  $code = $LASTEXITCODE
  $good = ($code -ne 0) -and (($msg | Out-String).Contains('index'))
  Write-Host ("  {0} txt with a missing index is refused by name (exit {1})" -f `
              $(if ($good) { 'ok  ' } else { 'FAIL' }), $code)
  if (-not $good) { $problems++ }

  if ($problems -eq 0) {
    ""
    "palette parity: ok -- both formats accept a match and refuse a mismatch identically"
    exit 0
  }
  ""
  "palette parity: FAILED ($problems problem(s))"
  exit 1
}
finally {
  Remove-Item $tmp -Recurse -Force -EA SilentlyContinue
}