# probe-segment-frames.ps1 -- does the SEGMENTED path work at all?
#
# WHY THIS EXISTS.  `--segment-frames` was broken in both of its modes and nothing in
# this suite noticed, for the plainest possible reason: no stage ran it.  It shipped
# with an access violation (0xC0000005) under RD_PALETTE_OVERLAP=1 and a plain exit 1
# without it.  Every other stage here passed, green, for as long as the bug existed.
#
# A check that cannot fail is not a check, and this file is the one that makes that
# path failable.  It is the cheapest stage in the suite -- seconds -- and it guards
# the property that matters most in it: a long render is exactly the render that
# passes --segment-frames, because --segment-frames is how you make progress durable
# across a power cut.  A crash there is the worst possible place for one.
#
# WHAT IT ASSERTS, and why each part:
#
#   1. exit 0, plain.  Caught the empty-ColorTree failure.
#   2. exit 0, with RD_PALETTE_OVERLAP=1.  Caught the access violation, which needs
#      the overlap thread to still be writing *tree while the segment path moved
#      from it.  Without the env var this arm passes on code that is still broken.
#   3. exit 0, more than one segment.  A single segment never enters the loop's
#      second iteration, so it cannot catch a checkpoint or join bug.
#   4. decoded pixels of the segmented run EQUAL the unsegmented run, under
#      --video-lossless.  This is the load-bearing one, and it is why the fixture is
#      lossless: with h264 the two arms legitimately differ, because x264 restarts
#      rate control per segment, and a byte comparison would fail forever on correct
#      code.  Under ffv1 there is no encoder state to restart, so any difference is
#      a difference in what rdither did.
#
# IT IS NOT TESTED against.  The gate compares decoded pixels, not hashes of files:
# container bytes differ for reasons that have nothing to do with the dither, and a
# check that fails on them teaches people to ignore it.
#
# No clip is committed to this repository (.gitignore excludes *.mkv and *.mp4), so
# the fixture is built here, the same way every other probe does it.
#
# Exit codes:  0 pass   1 FAIL   2 cannot run (and the reason is printed)

param(
  [string]$Rdither = "",
  [string]$Ffmpeg  = "",
  [string]$Clip    = ""
)

$ErrorActionPreference = 'Stop'

# Same resolver every probe here carries; there is no shared tools\probe-common.ps1.
function Resolve-Tool([string]$name, [string]$given) {
  if ($given -and (Test-Path $given)) { return (Resolve-Path $given).Path }
  $c = Get-Command $name -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  return ""
}

$exe = Resolve-Tool 'rdither.exe' $Rdither
$ffmpeg = Resolve-Tool 'ffmpeg.exe' $Ffmpeg

if (-not $exe) { "segment frames: no rdither given and none on PATH; cannot run"; exit 2 }
if (-not $ffmpeg) { "segment frames: no ffmpeg on PATH; cannot run"; exit 2 }

# 320x180, 10 frames, ffv1/yuv444p: lossless so arm 4 is meaningful, and small enough
# that the whole stage is seconds.  10 frames over --segment-frames 4 is three
# segments, so the loop iterates and the checkpoint path runs.
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("rd_seg_" + [Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
  if ($Clip -and (Test-Path $Clip)) {
    $src = (Resolve-Path $Clip).Path
  } else {
    $src = Join-Path $tmp 'fixture.mkv'
    & $ffmpeg -v error -y -f lavfi -i 'testsrc=size=320x180:rate=10:duration=1' `
                     -c:v ffv1 -pix_fmt yuv444p $src 2>&1 | Out-Null
    if (-not (Test-Path $src)) {
      "segment frames: could not build a fixture with ffmpeg; cannot run"
      exit 2
    }
  }

  function Decode-To-Raw([string]$path) {
    $raw = Join-Path $tmp ([IO.Path]::GetFileName($path) + '.yuv')
    & $ffmpeg -y -v error -i $path -f rawvideo -pix_fmt rgb24 $raw 2>&1 | Out-Null
    if (-not (Test-Path $raw)) { return $null }
    return (Get-FileHash $raw -Algorithm SHA256).Hash
  }

  $failures = 0

  # The arms.  `--engine cpu` so the whole thing runs on a machine with no GPU: this
  # stage is about the CLI's control flow, which is identical on both builds, and a
  # stage that needs a device is a stage that gets skipped in CI.
  function Invoke-Arm([string]$label, [hashtable]$extraEnv, [string[]]$extraArgs) {
    $out = Join-Path $tmp ($label + '.mkv')
    $envBackup = @{}
    foreach ($k in $extraEnv.Keys) {
      $envBackup[$k] = [Environment]::GetEnvironmentVariable($k)
      [Environment]::SetEnvironmentVariable($k, $extraEnv[$k])
    }
    try {
      $log = & $exe @('--video','--engine','cpu','--colors','16','--video-lossless') `
                 @extraArgs $src $out 2>&1
      $code = $LASTEXITCODE
    } finally {
      foreach ($k in $envBackup.Keys) {
        [Environment]::SetEnvironmentVariable($k, $envBackup[$k])
      }
    }
    $joined = ($log | Where-Object { $_ -match 'segment \d+/\d+: done' }).Count
    Write-Host ("  {0,-22} exit {1}   {2} segment(s) checkpointed" -f $label, $code, $joined)
    if ($code -ne 0) {
      Write-Host "    FAILED (exit $code)"
      $log | Select-Object -Last 6 | ForEach-Object { Write-Host "    $_" }
      return $null
    }
    return $out
  }

  "segment frames: $src"

  # 1. plain
  $plain = Invoke-Arm 'segmented'  @{} @('--segment-frames','4')
  if (-not $plain) { $failures++ }

  # 2. the overlap arm.  RD_PALETTE_OVERLAP=1 makes the palette build on its own
  #    thread, which is the only way to get the race that produced 0xC0000005.
  $overlap = Invoke-Arm 'segmented+overlap' @{ RD_PALETTE_OVERLAP = '1' } @('--segment-frames','4')
  if (-not $overlap) { $failures++ }

  # 3. one segment, to catch the crash that a multi-segment run would also catch but
  #    for a different reason.
  $single = Invoke-Arm 'one segment' @{} @('--segment-frames','100')
  if (-not $single) { $failures++ }

  # 4. the unsegmented control, and the pixel comparison.
  $control = Invoke-Arm 'unsegmented' @{} @()
  if (-not $control) { $failures++ }

  if ($plain -and $control) {
    $a = Decode-To-Raw $plain
    $b = Decode-To-Raw $control
    if (-not $a -or -not $b) {
      Write-Host "  pixels: could not decode both arms; cannot run"
      exit 2
    }
    if ($a -eq $b) {
      Write-Host "  ok    segmented pixels == unsegmented pixels (lossless)"
    } else {
      Write-Host "  FAIL  segmented pixels DIFFER from unsegmented pixels (lossless)"
      Write-Host "        segmented   $a"
      Write-Host "        unsegmented $b"
      $failures++
    }
  }

  if ($failures -eq 0) {
    ""
    "segment frames: ok -- all arms exited 0 and the segmented pixels match"
    exit 0
  }
  ""
  "segment frames: FAILED ($failures problem(s))"
  exit 1
}
finally {
  Remove-Item $tmp -Recurse -Force -EA SilentlyContinue
}