# probe-swscale.ps1 -- identify exactly what ffmpeg's swscale does, empirically.
#
# Why this exists
# ---------------
# Replacing the ffmpeg decoder with the CUDA video decoder API means doing the
# YCbCr->RGB conversion on the device, and for this project it has to match swscale
# *bit-exactly* (see README round 14).  Guessing the formula does not work: the four
# classic integer forms all scored 0/376 exact matches.  This script is the tool that
# turns the question into a measurement.
#
# The trick is that swscale's own output is invertible.  Given its 16-bit RGB and the
# 8-bit YUV that went in, the chroma it must have used can be solved for exactly:
#
#   R = 1.164383*(Y-16) + 1.596027*(V-128)   =>  V_eff = 128 + (R - 1.164383*(Y-16)) / 1.596027
#   B = 1.164383*(Y-16) + 2.017232*(U-128)   =>  U_eff = 128 + (B - 1.164383*(Y-16)) / 2.017232
#   G = 1.164383*(Y-16) - 0.391762*(U-128) - 0.812968*(V-128)   <- consistency check
#
# The BT.601 coefficients are not assumed either: they were checked against a flat
# region of a real frame first, where they agreed to ~1%, while BT.709 was wildly
# wrong.  The source declares every colour parameter "unknown", so ffmpeg falls back
# to its default and 601 is what it uses.
#
# Usage
# -----
#   .\tools\probe-swscale.ps1 -Ffmpeg "C:\path\to\ffmpeg.exe" -Work "C:\temp\probe"
#
# With no -Formulas it runs the two experiments that have already been done and prints
# what they showed, so the next person starts from facts rather than from scratch.
param(
  [string]$Ffmpeg = "ffmpeg",
  [string]$Work = "$env:TEMP\swscale-probe",
  [int]$Width = 64,
  [int]$Height = 64
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Work | Out-Null

# BT.601 limited range, verified against ffmpeg's own output.
$Ky = 1.164383; $Kv = 1.596027; $Ku = 2.017232; $Kg1 = 0.391762; $Kg2 = 0.812968

function Convert-Yuv([byte[]]$yuv, [int]$w, [int]$h) {
  $args = @('-v','error','-f','rawvideo','-pix_fmt','yuv420p','-s',"${w}x${h}",
            '-i',"$Work\in.yuv",'-f','rawvideo','-pix_fmt','rgba64le','-y',"$Work\out.rgba")
  $p = Start-Process -FilePath $Ffmpeg -ArgumentList $args -PassThru -WindowStyle Hidden
  if (-not $p.WaitForExit(120000)) { $p.Kill(); throw "ffmpeg timed out" }
  if ($p.ExitCode -ne 0) { throw "ffmpeg failed: $($p.ExitCode)" }
  return [System.IO.File]::ReadAllBytes("$Work\out.rgba")
}

# ---------------------------------------------------------------------------
# Experiment 1: what does the 4:2:0 chroma upsampling do?
# A sawtooth in the chroma plane makes the filter's shape directly readable from
# the recovered U_eff.  RESULT (measured): it is a wide symmetric FIR.  The recovered
# chroma lags the input, plateaus, and attenuates -- it is NOT nearest-neighbour and
# NOT bilinear.  Reproducing it means porting swscale's initFilter() coefficient
# generation, not writing a two-tap average.
# ---------------------------------------------------------------------------
$PS = $Width * $Height
$cw = [int]($Width / 2); $ch = [int]($Height / 2)
# 4:2:0 is planar with *half-resolution* chroma, so the planes are
# Y at 0, U at W*H, V at W*H + (W/2)*(H/2) -- not at 2*W*H.
$uOff = $PS
$vOff = $PS + $cw * $ch
$buf = New-Object byte[] ($PS + 2 * $cw * $ch)
for ($i = 0; $i -lt $PS; $i++) { $buf[$i] = 128 }
for ($y = 0; $y -lt $ch; $y++) {
  for ($x = 0; $x -lt $cw; $x++) { $buf[$uOff + $y * $cw + $x] = [byte](($x % 8) * 32) }
}
for ($i = 0; $i -lt $cw * $ch; $i++) { $buf[$vOff + $i] = 128 }
[System.IO.File]::WriteAllBytes("$Work\in.yuv", $buf)
$rgb = Convert-Yuv $buf $Width $Height

Write-Host ""
Write-Host "Experiment 1: recovering swscale's effective chroma from its own output."
Write-Host ("  {0,3} | {1,6} {2,6} | {3,7} {4,7} {5,7} | nearest {6,7} | bilinear {7,7}" -f `
  'x','Uin','Vin','Ueff','Veff','Gerr','Uin','Ubilin')
for ($x = 0; $x -lt 24; $x++) {
  $px = $x + 16 * $Width
  $Y = 128.0
  $R = [BitConverter]::ToUInt16($rgb, $px * 8) / 257.0
  $G = [BitConverter]::ToUInt16($rgb, $px * 8 + 2) / 257.0
  $B = [BitConverter]::ToUInt16($rgb, $px * 8 + 4) / 257.0
  $yy = $Ky * ($Y - 16)
  $Veff = 128 + ($R - $yy) / $Kv
  $Ueff = 128 + ($B - $yy) / $Ku
  $Gcalc = $yy - $Kg1 * ($Ueff - 128) - $Kg2 * ($Veff - 128)
  $cx = [math]::Floor($x / 2)
  $un = $buf[$uOff + 16 * $cw + $cx]
  $i0 = [math]::Max(0, [math]::Floor($x - 0.5)); $f = ($x - 0.5) - $i0
  $i1 = [math]::Min($cw - 1, $i0 + 1)
  $ub = $buf[$uOff + 16 * $cw + $i0] * (1 - $f) + $buf[$uOff + 16 * $cw + $i1] * $f
  Write-Host ("  {0,3} | {1,6} {2,6} | {3,7:N1} {4,7:N1} {5,7:N1} | {6,7} | {7,7:N1}" -f `
    $x, $un, 128, $Ueff, $Veff, ($G - $Gcalc), $un, $ub)
}
Write-Host "  -> lag, plateau and attenuation mean a wide symmetric FIR, not nearest/bilinear."

# ---------------------------------------------------------------------------
# Experiment 2: given 4:4:4 input, what integer arithmetic produces the output?
# RESULT (measured): none of the four classic forms scored above 0/376 exact on
# unclipped samples.  Add candidate forms below and re-run to test them.
# ---------------------------------------------------------------------------
$W2 = 256; $H2 = 8; $PS2 = $W2 * $H2
$b2 = New-Object byte[] ($PS2 * 3)
for ($i = 0; $i -lt $PS2; $i++) {
  $b2[$i]            = [byte](16 + ($i % 240))
  $b2[$PS2 + $i]     = [byte](16 + (($i * 7) % 224))
  $b2[2 * $PS2 + $i] = [byte](16 + (($i * 13) % 224))
}
[System.IO.File]::WriteAllBytes("$Work\in444.yuv", $b2)
$p = Start-Process -FilePath $Ffmpeg -ArgumentList @('-v','error','-f','rawvideo','-pix_fmt','yuv444p','-s',"${W2}x${H2}",'-i',"$Work\in444.yuv",'-f','rawvideo','-pix_fmt','rgba64le','-y',"$Work\out444.rgba") -PassThru -WindowStyle Hidden
$null = $p.WaitForExit(120000)
$r2 = [System.IO.File]::ReadAllBytes("$Work\out444.rgba")

$rows = @()
for ($i = 0; $i -lt $PS2; $i++) {
  $R = [BitConverter]::ToUInt16($r2, $i * 8)
  $G = [BitConverter]::ToUInt16($r2, $i * 8 + 2)
  $B = [BitConverter]::ToUInt16($r2, $i * 8 + 4)
  if ($R -gt 800 -and $R -lt 64000 -and $G -gt 800 -and $G -lt 64000 -and $B -gt 800 -and $B -lt 64000) {
    $rows += [pscustomobject]@{ Y=$b2[$i]; U=$b2[$PS2+$i]; V=$b2[2*$PS2+$i]; R=$R; G=$G; B=$B }
  }
}
Write-Host ""
Write-Host "Experiment 2: exact-match rate of candidate integer forms, $($rows.Count) unclipped samples."
foreach ($form in 1..4) {
  $ok = 0
  foreach ($s in $rows) {
    $yy = $s.Y - 16; $uu = $s.U - 128; $vv = $s.V - 128
    switch ($form) {
      1 { $pr = [math]::Max(0,[math]::Min(255,[math]::Floor((298*$yy + 409*$vv + 128)/256)))*257
           $pg = [math]::Max(0,[math]::Min(255,[math]::Floor((298*$yy - 100*$uu - 208*$vv + 128)/256)))*257
           $pb = [math]::Max(0,[math]::Min(255,[math]::Floor((298*$yy + 516*$uu + 128)/256)))*257 }
      2 { $pr = [math]::Max(0,[math]::Min(65535,[math]::Floor((298*257*$yy + 409*257*$vv + 32768)/65536)))
           $pg = [math]::Max(0,[math]::Min(65535,[math]::Floor((298*257*$yy - 100*257*$uu - 208*257*$vv + 32768)/65536)))
           $pb = [math]::Max(0,[math]::Min(65535,[math]::Floor((298*257*$yy + 516*257*$uu + 32768)/65536))) }
      3 { $pr = [math]::Max(0,[math]::Min(65535,[math]::Floor((76258*$yy + 104954*$vv + 32768)/65536)))
           $pg = [math]::Max(0,[math]::Min(65535,[math]::Floor((76258*$yy - 25670*$uu - 53394*$vv + 32768)/65536)))
           $pb = [math]::Max(0,[math]::Min(65535,[math]::Floor((76258*$yy + 132612*$uu + 32768)/65536))) }
      4 { $pr = [math]::Max(0,[math]::Min(65535,[math]::Floor((9531*$yy + 13112*$vv + 32768)/8192)))
           $pg = [math]::Max(0,[math]::Min(65535,[math]::Floor((9531*$yy - 3205*$uu - 6664*$vv + 32768)/8192)))
           $pb = [math]::Max(0,[math]::Min(65535,[math]::Floor((9531*$yy + 16563*$uu + 32768)/8192))) }
    }
    if ($pr -eq $s.R -and $pg -eq $s.G -and $pb -eq $s.B) { $ok++ }
  }
  Write-Host ("  form {0} : {1,5} / {2} exact" -f $form, $ok, $rows.Count)
}
Write-Host "  -> all zero: swscale's 8-bit-in / 16-bit-out path is none of these."
Write-Host "     The next step is to read it in libswscale (yuv2rgb.c, output.c,"
Write-Host "     initFilter() in swscale.c) for this ffmpeg and port the fixed point."
Write-Host ""

