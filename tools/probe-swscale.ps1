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
# EXIT CODES, and why this file has them
# --------------------------------------
#   0  both experiments ran and produced the measured result
#   1  an experiment could not run, or produced no rows to score, or the measured
#      result does not support the conclusion it prints
#   2  ffmpeg is not available -- nothing here is answerable
#
# It had no exit code at all before.  Its two `throw`s only ever fired on ffmpeg
# itself, so every conclusion below was printed unconditionally: a sweep that
# measured ZERO rows printed "form 1 : 0 / 0 exact" four times and then declared
# "-> all zero: swscale's 8-bit-in / 16-bit-out path is none of these", which is a
# vacuous 0/0 satisfying the conclusion for the wrong reason.  Zero rows does not
# mean the forms are wrong; it means nothing was measured.  Those are now different
# outcomes with different exits.
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
  [int]$Height = 64,
  # Hang guard for each ffmpeg invocation.  Experiment 2 discarded the WaitForExit
  # boolean entirely (`$null =`), never killed on timeout and never checked the exit
  # code, and it did not delete its output file first -- so a hung ffmpeg left the
  # PREVIOUS run's out444.rgba in place and the probe scored that, silently, as if
  # it were fresh.
  [int]$FfmpegTimeoutSec = 120
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Work | Out-Null

# BT.601 limited range, verified against ffmpeg's own output.
$Ky = 1.164383; $Kv = 1.596027; $Ku = 2.017232; $Kg1 = 0.391762; $Kg2 = 0.812968

# Every ffmpeg run goes through here, so no caller can forget the three things that
# were each individually missed: delete the output first, wait with a timeout, kill on
# timeout, and check the exit code.
#
# $Output is deleted BEFORE the spawn rather than after, because that is the only
# ordering in which "the file is there" means "ffmpeg wrote it just now".
#
# The parameter is $FfmpegArgs and NOT $Args.  $args is a PowerShell automatic
# variable, and declaring `param([string[]]$Args, ...)` binds NOTHING: the caller
# passes -Args @('-v','error',...) and the body receives an empty array, with no error
# and no warning.  Measured, and it was found here because ffmpeg then exited 1 having
# been handed no arguments at all.  A silent empty argument list is precisely the
# failure this file exists to prevent, so it is checked rather than assumed.
function Invoke-Ffmpeg {
  param([string[]]$FfmpegArgs, [string]$Output, [string]$What)

  if ($null -eq $FfmpegArgs -or $FfmpegArgs.Count -eq 0) {
    throw "${What}: no ffmpeg arguments reached Invoke-Ffmpeg, so nothing was run"
  }
  if ($Output) { Remove-Item -EA SilentlyContinue $Output }
  $p = Start-Process -FilePath $Ffmpeg -ArgumentList $FfmpegArgs -PassThru -WindowStyle Hidden

  # The boolean is the whole point of the overload.  Discarding it is what let a hung
  # ffmpeg fall through to a read of whatever was already on disk.
  # ${What}: the braces are required -- "$What:" is a scoped-variable reference in
  # PowerShell and is a parse error, which is how this line was found.
  if (-not $p.WaitForExit($FfmpegTimeoutSec * 1000)) {
    try { $p.Kill($true) } catch { try { $p.Kill() } catch { } }
    try { $p.WaitForExit(10000) | Out-Null } catch { }
    throw "${What}: ffmpeg did not exit within $FfmpegTimeoutSec s and was killed"
  }
  $code = $p.ExitCode
  if ($code -ne 0) { throw "${What}: ffmpeg failed with exit $code" }
  if ($Output -and -not (Test-Path $Output)) {
    throw "${What}: ffmpeg exited 0 but wrote no $Output"
  }
  if (-not $Output) { return }
  return [System.IO.File]::ReadAllBytes($Output)
}

if (-not (Get-Command $Ffmpeg -ErrorAction SilentlyContinue) -and -not (Test-Path $Ffmpeg)) {
  "cannot run: ffmpeg not found at '$Ffmpeg'."
  'swscale behaviour is the entire subject of this probe; without ffmpeg there is'
  'nothing to measure and nothing to conclude.  Pass -Ffmpeg with a full path.'
  exit 2
}

function Convert-Yuv([byte[]]$yuv, [int]$w, [int]$h) {
  $args = @('-v','error','-f','rawvideo','-pix_fmt','yuv420p','-s',"${w}x${h}",
            '-i',"$Work\in.yuv",'-f','rawvideo','-pix_fmt','rgba64le','-y',"$Work\out.rgba")
  return Invoke-Ffmpeg -FfmpegArgs $args -Output "$Work\out.rgba" -What 'experiment 1'
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

# The recovery below reads pixel 16 rows down and up to 24 columns across, so a
# geometry that cannot supply those rows measures nothing.  Checked rather than
# assumed, because a short buffer here reads out of range and prints numbers anyway.
$probeRow = 16
if ($Height -le $probeRow -or $Width -lt 24 -or $ch -le $probeRow) {
  "cannot run: -Width/-Height ($Width x $Height) cannot supply experiment 1's probe row $probeRow."
  "It needs Height > $probeRow and Width >= 24.  Nothing was measured."
  exit 2
}
if ($null -eq $rgb -or $rgb.Length -lt $PS * 8) {
  "experiment 1: ffmpeg returned $($rgb.Length) bytes for $PS pixels, expected $($PS * 8)."
  'Nothing was measured, so no conclusion is available.'
  exit 1
}

Write-Host ""
Write-Host "Experiment 1: recovering swscale's effective chroma from its own output."
Write-Host ("  {0,3} | {1,6} {2,6} | {3,7} {4,7} {5,7} | nearest {6,7} | bilinear {7,7}" -f `
  'x','Uin','Vin','Ueff','Veff','Gerr','Uin','Ubilin')
for ($x = 0; $x -lt 24; $x++) {
  $px = $x + $probeRow * $Width
  $Y = 128.0
  $R = [BitConverter]::ToUInt16($rgb, $px * 8) / 257.0
  $G = [BitConverter]::ToUInt16($rgb, $px * 8 + 2) / 257.0
  $B = [BitConverter]::ToUInt16($rgb, $px * 8 + 4) / 257.0
  $yy = $Ky * ($Y - 16)
  $Veff = 128 + ($R - $yy) / $Kv
  $Ueff = 128 + ($B - $yy) / $Ku
  $Gcalc = $yy - $Kg1 * ($Ueff - 128) - $Kg2 * ($Veff - 128)
  $cx = [math]::Floor($x / 2)
  $un = $buf[$uOff + $probeRow * $cw + $cx]
  $i0 = [math]::Max(0, [math]::Floor($x - 0.5)); $f = ($x - 0.5) - $i0
  $i1 = [math]::Min($cw - 1, $i0 + 1)
  $ub = $buf[$uOff + $probeRow * $cw + $i0] * (1 - $f) + $buf[$uOff + $probeRow * $cw + $i1] * $f
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
# Checked, not assumed: a short read would index past the end of $r2 and the scoring
# loop would compare against garbage rather than fail.
$r2 = Invoke-Ffmpeg -FfmpegArgs @('-v','error','-f','rawvideo','-pix_fmt','yuv444p','-s',"${W2}x${H2}",
                            '-i',"$Work\in444.yuv",'-f','rawvideo','-pix_fmt','rgba64le','-y',
                            "$Work\out444.rgba") `
      -Output "$Work\out444.rgba" -What 'experiment 2'
if ($r2.Length -lt $PS2 * 8) {
  "experiment 2: ffmpeg returned $($r2.Length) bytes for $PS2 pixels, expected $($PS2 * 8)."
  'The input was not converted, so there is nothing to score and no conclusion.'
  exit 1
}

$rows = @()
for ($i = 0; $i -lt $PS2; $i++) {
  $R = [BitConverter]::ToUInt16($r2, $i * 8)
  $G = [BitConverter]::ToUInt16($r2, $i * 8 + 2)
  $B = [BitConverter]::ToUInt16($r2, $i * 8 + 4)
  if ($R -gt 800 -and $R -lt 64000 -and $G -gt 800 -and $G -lt 64000 -and $B -gt 800 -and $B -lt 64000) {
    $rows += [pscustomobject]@{ Y=$b2[$i]; U=$b2[$PS2+$i]; V=$b2[2*$PS2+$i]; R=$R; G=$G; B=$B }
  }
}

# Zero rows is "nothing measured", not "the forms are all wrong".  This is the check
# whose absence produced the vacuous 0/0 that satisfied the conclusion below.
if ($rows.Count -eq 0) {
  ""
  "experiment 2: ffmpeg's $W2 x $H2 yuv444p -> rgba64le conversion left 0 of $PS2 samples"
  'inside the unclipped window, so there is nothing to score.'
  'The conclusion below is NOT available: 0 / 0 would satisfy it for the wrong reason.'
  exit 1
}

Write-Host ""
Write-Host "Experiment 2: exact-match rate of candidate integer forms, $($rows.Count) unclipped samples."
$scores = @{}
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
  $scores[$form] = $ok
  Write-Host ("  form {0} : {1,5} / {2} exact" -f $form, $ok, $rows.Count)
}

# The conclusion is now derived from the scores rather than printed regardless of them.
$anyHit = @($scores.Keys | Where-Object { $scores[$_] -gt 0 })
if ($anyHit.Count -eq 0) {
  Write-Host "  -> all four scored 0 of $($rows.Count): swscale's 8-bit-in / 16-bit-out path is"
  Write-Host "     none of these, on $($rows.Count) samples that were actually measured."
  Write-Host "     The next step is to read it in libswscale (yuv2rgb.c, output.c,"
  Write-Host "     initFilter() in swscale.c) for this ffmpeg and port the fixed point."
} else {
  $top = ($anyHit | Sort-Object { -$scores[$_] } | Select-Object -First 1)
  Write-Host "  -> form(s) $($anyHit -join ', ') score above zero; best is form $top at $($scores[$top]) / $($rows.Count)."
  Write-Host '     A candidate matched, so "none of these" is NOT available and this file'
  Write-Host '     cannot rank the rest for you: add the next candidate below and re-run.'
  Write-Host "     Exiting 1 because the conclusion this script exists to print is now false."
  Write-Host ""
  exit 1
}
Write-Host ""
exit 0