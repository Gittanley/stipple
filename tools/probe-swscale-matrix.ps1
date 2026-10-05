# probe-swscale-matrix.ps1 -- determine the exact structure of swscale's 8-bit YUV
# -> 16-bit RGB conversion, empirically.
#
# Why this exists
# ---------------
# probe-swscale.ps1 established that the mapping is none of the four classic
# integer forms.  That rules out a formula but says nothing about what to build.
# To port it we need to know its *shape*:
#
#   * is it separable, R = f(Y) + g(V)?   (then it is two 256-entry tables, tiny)
#   * or is it a genuine 2-D function of (Y,V)?
#   * is R independent of U, as BT.601 requires?
#   * what is the output quantisation?  (a table path gives coarse power-of-two steps)
#
# All of that is one measurement each, and ffmpeg is the oracle for all of them.
# yuv444p is used throughout so that no chroma *filter* is in play -- this isolates
# the colour matrix, which is the part that has to be reproduced on the device.
#
# EXIT CODES, and the two things that were wrong before
# -----------------------------------------------------
#   0  all four questions were measured, and each printed its own verdict
#   1  a measurement was refused: the apparatus contradicted itself, or a number the
#      conclusion rests on came out as zero
#   2  nothing could be measured: no ffmpeg, or ffmpeg wrote nothing usable
#
# 1. IT DID NOT RUN.  It had no exit statement and it never finished either.
#
#    Convert-Yuv444 returned a byte[] through the PowerShell pipeline, which UNROLLS
#    an array into its elements: the caller received a System.Object[] of 524288
#    boxed bytes rather than a System.Byte[].  [BitConverter]::ToUInt16 then had to
#    coerce that object[] to byte[] on every one of the 131072 calls, each conversion
#    walking all 524288 elements.  Measured on this machine: 200 calls take 11888 ms
#    on the object[] and 2.3 ms on a real byte[], a factor of ~5000, which puts the
#    full grid at about 2.1 HOURS.  So this file has never produced its output on any
#    machine, and nothing noticed, because a file that hangs is at least not green.
#
#    The cause is the same family as the `param([string[]]$Args)` trap in
#    probe-swscale.ps1 and the unbound `$Out` I wrote myself in probe-host-oracle.ps1:
#    the value ARRIVES, and is not what the code assumed it was, and nothing says so.
#    `return ,$bytes` is the fix; the comma suppresses the unroll.
#
# 2. IT HAD NO VERDICT.  Q1 printed "R(Y,V) - R(Y,Vmid) independent of Y : 49500 /
#    49500" and left it to the reader to notice that this is the same as fully
#    separable.  A number with no word attached to it is not a conclusion, and the
#    reader who skims is the one who is supposed to act on it.  Every question below
#    now ends in a word derived from its own numbers.
#
# Usage
# -----
#   .\tools\probe-swscale-matrix.ps1 -Ffmpeg "C:\path\to\ffmpeg.exe"
param(
  [string]$Ffmpeg = "ffmpeg",
  [string]$Work = "$env:TEMP\swscale-matrix",
  # Hang guard per ffmpeg invocation, and a floor on what it must produce.  Both were
  # missing: the first made a wedged ffmpeg indistinguishable from a slow machine, and
  # the second is what lets "the grid was written and nothing came back" reach the
  # scoring loops as an array of zeros.
  [int]$FfmpegTimeoutSec = 180
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Work | Out-Null

if (-not (Get-Command $Ffmpeg -ErrorAction SilentlyContinue) -and -not (Test-Path $Ffmpeg)) {
  Write-Host "cannot run: ffmpeg not found at '$Ffmpeg'."
  Write-Host 'swscale''s conversion is the entire subject of this file; with no ffmpeg there'
  Write-Host 'is nothing to measure.  Pass -Ffmpeg with a full path.'
  exit 2
}

# Runs one planar yuv444p frame through ffmpeg and returns the rgba64le bytes AS A
# byte[].  The comma in `return ,$bytes` is load-bearing; see the header.
function Convert-Yuv444([byte[]]$yuv, [int]$w, [int]$h, [string]$tag) {
  $raw = "$Work\in_$tag.yuv"
  $outFile = "$Work\out_$tag.rgba"
  [System.IO.File]::WriteAllBytes($raw, $yuv)
  # Deleted BEFORE the spawn, so "the file is there" can only mean "ffmpeg just wrote
  # it" rather than "the previous run's file is still here".
  Remove-Item -EA SilentlyContinue $outFile
  $p = Start-Process -FilePath $Ffmpeg -WindowStyle Hidden -PassThru -ArgumentList @(
    '-v','error','-f','rawvideo','-pix_fmt','yuv444p','-s',"${w}x${h}",
    '-i',$raw,'-f','rawvideo','-pix_fmt','rgba64le','-y',$outFile)
  if (-not $p.WaitForExit($FfmpegTimeoutSec * 1000)) {
    try { $p.Kill($true) } catch { try { $p.Kill() } catch { } }
    try { $p.WaitForExit(10000) | Out-Null } catch { }
    throw "${tag}: ffmpeg did not exit within $FfmpegTimeoutSec s and was killed"
  }
  if ($p.ExitCode -ne 0) { throw "${tag}: ffmpeg failed : exit $($p.ExitCode)" }
  if (-not (Test-Path $outFile)) { throw "${tag}: ffmpeg exited 0 but wrote no $outFile" }
  $bytes = [System.IO.File]::ReadAllBytes($outFile)
  # The expected length is checked, not assumed.  A short read here used to index
  # past the end of a 512 KiB array and score whatever was there.
  $want = $w * $h * 8
  if ($bytes.Length -lt $want) {
    throw "${tag}: ffmpeg returned $($bytes.Length) bytes for $w x $h, expected $want"
  }
  return ,$bytes
}

# The full legal Y and V ranges for limited-range BT.601.
$Y0 = 16; $Y1 = 235
$V0 = 16; $V1 = 240
$NY = $Y1 - $Y0 + 1   # 220
$NV = $V1 - $V0 + 1   # 225

# ---------------------------------------------------------------------------
# Grid 1: every (Y,V) pair, U held at 128.  Gives R(Y,V) and G(Y,V) directly.
#   256 x 256 covers 220*225 = 49500 pairs with room to spare.
#
#   The surplus WRAPS: column $col >= $NY repeats the pattern from $col - NY, and
#   row $row >= $NV likewise, so the last 36 columns and 31 rows overwrite cells the
#   readers below have already consumed or do not read at all.  It is harmless and
#   deliberate, and it is written down here because "the grid is 256 wide and NY is
#   220" reads like an off-by-36 until you know.
# ---------------------------------------------------------------------------
$W = 256; $H = 256; $PS = $W * $H
$b = New-Object byte[] ($PS * 3)
for ($row = 0; $row -lt $H; $row++) {
  $v = $V0 + ($row % $NV)
  for ($col = 0; $col -lt $W; $col++) {
    $y = $Y0 + ($col % $NY)
    $i = $row * $W + $col
    $b[$i]            = [byte]$y
    $b[$PS + $i]      = [byte]128
    $b[2 * $PS + $i]  = [byte]$v
  }
}
$o = Convert-Yuv444 $b $W $H "grid"

# R and G indexed by (y, v)
$R = New-Object 'int[,]' $NY, $NV
$G = New-Object 'int[,]' $NY, $NV
for ($row = 0; $row -lt $H; $row++) {
  $vi = $row % $NV
  for ($col = 0; $col -lt $W; $col++) {
    $yi = $col % $NY
    $i = $row * $W + $col
    $R[$yi,$vi] = [BitConverter]::ToUInt16($o, $i * 8)
    $G[$yi,$vi] = [BitConverter]::ToUInt16($o, $i * 8 + 2)
  }
}
# If the conversion produced nothing, every cell is 0 and every question below is
# answered "yes" for the wrong reason -- separability holds trivially of a constant
# grid.  Checked here, once, where it can still be named.
# $R[($NY-1),($NV-1)] and not $R[$NY-1,$NV-1]: inside a multi-dimensional index
# PowerShell parses the whole comma expression first and then fails on op_Subtraction
# between an int[] and an int.  The parentheses are not stylistic.
if ($R[0,0] -eq 0 -and $R[($NY - 1),($NV - 1)] -eq 0) {
  Write-Host ''
  Write-Host 'cannot run: the 256 x 256 grid came back as all zeros, so there is nothing'
  Write-Host 'to measure.  Separability of a constant grid is true and means nothing.'
  exit 2
}

# ---------------------------------------------------------------------------
# Q1  Is R separable?  R(Y,V) - R(Y,Vmid) should not depend on Y if R = f(Y) + g(V).
# ---------------------------------------------------------------------------
$vMid = [int](($NV - 1) / 2)
$sepR = 0; $sepRtot = 0
$sepG = 0; $sepGtot = 0
for ($yi = 0; $yi -lt $NY; $yi++) {
  $baseR = $R[$yi,$vMid]; $baseG = $G[$yi,$vMid]
  for ($vi = 0; $vi -lt $NV; $vi++) {
    $sepRtot++
    if (($R[$yi,$vi] - $baseR) -eq ($R[0,$vi] - $R[0,$vMid])) { $sepR++ }
    $sepGtot++
    if (($G[$yi,$vi] - $baseG) -eq ($G[0,$vi] - $G[0,$vMid])) { $sepG++ }
  }
}
Write-Host ""
Write-Host "Q1  separability of the 8-bit -> 16-bit output (U held at 128)"
Write-Host ("    R(Y,V) - R(Y,Vmid) independent of Y : {0,6} / {1,6}" -f $sepR, $sepRtot)
Write-Host ("    G(Y,V) - G(Y,Vmid) independent of Y : {0,6} / {1,6}" -f $sepG, $sepGtot)
# The word, derived from the numbers rather than left to the reader.
$q1 = if ($sepR -eq $sepRtot) { 'R is EXACTLY separable: R = f(Y) + g(V)' }
      else { "R is NOT separable ($($sepRtot - $sepR) of $sepRtot pairs disagree), so it is a genuine 2-D function of (Y,V)" }
$q1g = if ($sepG -eq $sepGtot) { 'G is EXACTLY separable' }
       else { "G is NOT separable ($($sepGtot - $sepG) of $sepGtot pairs disagree)" }
Write-Host "    VERDICT: $q1"
Write-Host "    VERDICT: $q1g"

# ---------------------------------------------------------------------------
# Q2  Quantisation.  The largest common divisor of every R value and of every
#     adjacent difference tells us whether this is a table path (coarse steps) or
#     full 16-bit arithmetic.
# ---------------------------------------------------------------------------
# The gcd calls take a TEMPORARY, and are written without parentheses.  Both matter,
# and neither is a style choice:
#
#   * `Gcd($gAll, $R[$yi,$vi])` throws "cannot process argument transformation on
#     parameter 'a'.  Cannot convert the System.Object[] value to type System.Int32".
#     `$R[$yi,$vi]` is a perfectly good Int32 when assigned -- `x = $R[0,0]` gives 12 --
#     but as an element of a parenthesised command argument list the comma inside the
#     index is taken as part of the argument collection and the value arrives as an
#     Object[].  Measured: `Gcd($g, ($R[0,0]))` throws while `Gcd $g $v`, with
#     $v = $R[0,0], returns 12.  Same family again: the value arrives, and is not what
#     the code assumed, and the binder's complaint is about a type rather than about
#     the index.
#   * This is the FOURTH latent defect in this file, and the only reason any of the
#     first three was ever found: the Q1 hang meant control never reached this line.
#     A fix that makes a file runnable is also a fix that exposes what was behind it.
function Gcd([int]$a, [int]$b) { while ($b -ne 0) { $t = $a % $b; $a = $b; $b = $t }; return [Math]::Abs($a) }
$gAll = 0; $gDiff = 0
for ($yi = 0; $yi -lt $NY; $yi++) {
  for ($vi = 0; $vi -lt $NV; $vi++) {
    $v = $R[$yi,$vi]
    $gAll = Gcd $gAll $v
  }
  if ($yi -gt 0) {
    for ($vi = 0; $vi -lt $NV; $vi++) {
      # ($yi - 1) parenthesised.  int[,] subscripts are parsed as a single expression
      # first, so `$R[$yi - 1,$vi]` is `int[] - int` and throws on op_Subtraction; every
      # subscript below is parenthesised for that reason rather than for looks.
      $d = $R[$yi,$vi] - $R[($yi - 1),$vi]
      $gDiff = Gcd $gDiff $d
    }
  }
}
Write-Host ""
Write-Host "Q2  quantisation of R"
# gcd == 0 is what an all-zero grid produces, and "every R value is divisible by 0"
# is not a measurement.  It is excluded above, so a 0 here means no measurement.
if ($gAll -eq 0) {
  Write-Host '    VERDICT: gcd 0 -- no R value was measured, so quantisation is unknown.'
  Write-Host ''
  Write-Host 'FAILED: Q2 rests on a zero.  Nothing about the output quantisation is established.'
  exit 1
}
Write-Host ("    gcd of all R values          : {0}" -f $gAll)
Write-Host ("    gcd of adjacent-Y differences: {0}" -f $gDiff)
$q2 = if ($gAll -gt 1) { "a TABLE path: every R is a multiple of $gAll" }
      else { 'full 16-bit arithmetic: the gcd of all R values is 1' }
Write-Host "    VERDICT: $q2"

# ---------------------------------------------------------------------------
# Q3  Is R independent of U, and B a function of (Y,U)?  Same grid, U swept.
# ---------------------------------------------------------------------------
$b2 = New-Object byte[] ($PS * 3)
for ($row = 0; $row -lt $H; $row++) {
  $u = $V0 + ($row % $NV)
  for ($col = 0; $col -lt $W; $col++) {
    $y = $Y0 + ($col % $NY)
    $i = $row * $W + $col
    $b2[$i]            = [byte]$y
    $b2[$PS + $i]      = [byte]$u
    $b2[2 * $PS + $i]  = [byte]128
  }
}
$o2 = Convert-Yuv444 $b2 $W $H "grid_u"

# B indexed by (y, u)
$B = New-Object 'int[,]' $NY, $NV
$uMid = [int](($NV - 1) / 2)
for ($row = 0; $row -lt $H; $row++) {
  $ui = $row % $NV
  for ($col = 0; $col -lt $W; $col++) {
    $B[($col % $NY),$ui] = [BitConverter]::ToUInt16($o2, ($row * $W + $col) * 8 + 4)
  }
}
$sepB = 0; $sepBtot = 0
for ($yi = 0; $yi -lt $NY; $yi++) {
  $base = $B[$yi,$uMid]
  for ($ui = 0; $ui -lt $NV; $ui++) {
    $sepBtot++
    if (($B[$yi,$ui] - $base) -eq ($B[0,$ui] - $B[0,$uMid])) { $sepB++ }
  }
}
# R on the U-swept grid should match R on the V-swept grid wherever V=128 and U=128.
# This is the apparatus checking ITSELF: the two grids hold the same 220 pixels under
# identical conditions, so any disagreement means the measurement is not what it
# claims to be, and every other number in this file is then suspect.
$uAt128 = 128 - $V0
$vAt128 = 128 - $V0
$dup = 0; $dupTot = 0
for ($yi = 0; $yi -lt $NY; $yi++) {
  $dupTot++
  if ($R[$yi,$vAt128] -eq [BitConverter]::ToUInt16($o2, ((($uAt128) * $W) + $yi) * 8)) { $dup++ }
}
Write-Host ""
Write-Host "Q3  cross-checks"
Write-Host ("    B(Y,U) separable in U         : {0,6} / {1,6}" -f $sepB, $sepBtot)
Write-Host ("    R identical at U=128 both grids: {0,6} / {1,6}" -f $dup, $dupTot)
$q3b = if ($sepB -eq $sepBtot) { 'B = f(Y) + h(U)' }
       else { "B is NOT separable in U ($($sepBtot - $sepB) of $sepBtot pairs disagree)" }
Write-Host "    VERDICT: $q3b"
if ($dup -ne $dupTot) {
  Write-Host ("    VERDICT: the two grids DISAGREE on R at $($dupTot - $dup) of $dupTot shared")
  Write-Host '             pixels, so the apparatus is not measuring a single function and'
  Write-Host '             no conclusion from this file may be used.'
  Write-Host ''
  Write-Host 'FAILED: self-consistency check.  This is a fault in the measurement, not a'
  Write-Host 'finding about swscale.'
  exit 1
}
Write-Host '    VERDICT: R is independent of U, as BT.601 requires'

# ---------------------------------------------------------------------------
# Q4  If separable, the two tables are just the columns.  Print their step
#     structure so the port can be checked by eye.
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Q4  separable model, g(V) = R(Y,V) - R(Y,Vmid) for Y=$($Y0):"
$steps = @()
for ($vi = 0; $vi -lt 12; $vi++) { $steps += ("{0,7}" -f ($R[0,$vi] - $R[0,$vMid])) }
Write-Host ("    V={0,4}.. : {1}" -f $V0, ($steps -join ' '))
$steps = @()
for ($yi = 0; $yi -lt 12; $yi++) { $steps += ("{0,7}" -f ($R[$yi,$vMid] - $R[0,$vMid])) }
Write-Host ("    Y={0,4}.. : {1}" -f $Y0, ($steps -join ' '))

# ---------------------------------------------------------------------------
# Summary.  Four questions, four measured denominators, one exit code.  All four
# questions ran or this file already exited above, so reaching here is the pass.
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("Summary: 4 of 4 questions measured -- R separability, R quantisation," +
            " B separability, R/U independence.")
Write-Host "         Q1 $q1"
Write-Host "         Q2 $q2"
Write-Host "         Q3 $q3b"
Write-Host '         Q4 step structure printed above for a port to check against.'
Write-Host ""
exit 0