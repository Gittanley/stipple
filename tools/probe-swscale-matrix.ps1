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
# Usage
# -----
#   .\tools\probe-swscale-matrix.ps1 -Ffmpeg "C:\path\to\ffmpeg.exe"
param(
  [string]$Ffmpeg = "ffmpeg",
  [string]$Work = "$env:TEMP\swscale-matrix"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Work | Out-Null

# Runs one planar yuv444p frame through ffmpeg and returns the rgba64le bytes.
function Convert-Yuv444([byte[]]$yuv, [int]$w, [int]$h, [string]$tag) {
  $raw = "$Work\in_$tag.yuv"
  [System.IO.File]::WriteAllBytes($raw, $yuv)
  $p = Start-Process -FilePath $Ffmpeg -WindowStyle Hidden -PassThru -ArgumentList @(
    '-v','error','-f','rawvideo','-pix_fmt','yuv444p','-s',"${w}x${h}",
    '-i',$raw,'-f','rawvideo','-pix_fmt','rgba64le','-y',"$Work\out_$tag.rgba")
  if (-not $p.WaitForExit(180000)) { $p.Kill(); throw "ffmpeg timed out" }
  if ($p.ExitCode -ne 0) { throw "ffmpeg failed on $tag : exit $($p.ExitCode)" }
  return [System.IO.File]::ReadAllBytes("$Work\out_$tag.rgba")
}

# The full legal Y and V ranges for limited-range BT.601.
$Y0 = 16; $Y1 = 235
$V0 = 16; $V1 = 240
$NY = $Y1 - $Y0 + 1   # 220
$NV = $V1 - $V0 + 1   # 225

# ---------------------------------------------------------------------------
# Grid 1: every (Y,V) pair, U held at 128.  Gives R(Y,V) and G(Y,V) directly.
#   256 x 256 covers 220*225 = 49500 pairs with room to spare.
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

# R indexed by (y, v)
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

# ---------------------------------------------------------------------------
# Q2  Quantisation.  The largest common divisor of every R value and of every
#     adjacent difference tells us whether this is a table path (coarse steps) or
#     full 16-bit arithmetic.
# ---------------------------------------------------------------------------
function Gcd([int]$a, [int]$b) { while ($b -ne 0) { $t = $a % $b; $a = $b; $b = $t }; return [Math]::Abs($a) }
$gAll = 0; $gDiff = 0
for ($yi = 0; $yi -lt $NY; $yi++) {
  for ($vi = 0; $vi -lt $NV; $vi++) {
    $gAll = Gcd($gAll, $R[$yi,$vi])
  }
  if ($yi -gt 0) {
    for ($vi = 0; $vi -lt $NV; $vi++) {
      $d = $R[$yi,$vi] - $R[$yi - 1,$vi]
      $gDiff = Gcd($gDiff, $d)
    }
  }
}
Write-Host ""
Write-Host "Q2  quantisation of R"
Write-Host ("    gcd of all R values          : {0}" -f $gAll)
Write-Host ("    gcd of adjacent-Y differences: {0}" -f $gDiff)

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
    $B[$col % $NY, $ui] = [BitConverter]::ToUInt16($o2, ($row * $W + $col) * 8 + 4)
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
Write-Host ""
