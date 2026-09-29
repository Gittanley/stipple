# Is --engine opencl bit-identical to --engine blocks on the VIDEO pipeline?
#
# The image engine has tools\probe-opencl-exact.ps1.  This is the same idea for
# video, and it exists for the same reason: the two engines are *defined* to have
# the same partition and the same arithmetic, so a frame dithered on either is the
# same frame, and anything else is a defect rather than a tolerance to negotiate.
#
# Measured on tests\clip1920.mp4 -- 60 frames of 1920x1080, lossless, same palette,
# same 16-frame batches -- the decoded RGBA is 447.8 MB and the two SHA-256 hashes
# are equal.  Identical, not "almost frame exact", which is a stronger result than
# the video bar normally asks for and the one the engine spec implies.
#
# Two things this deliberately does NOT do:
#
#   * Compare the encoded FILES.  Matroska embeds a random SegmentUID and a writing
#     timestamp, so two runs can never produce equal files even when every pixel
#     matches; that has already "failed" for several rounds in this project.  The
#     frames are decoded and the raw dump is hashed instead.
#
#   * Report a frame count or timing difference as a failure.  A short clip is
#     dominated by process start-up and driver init, and a browser running video
#     alongside puts sub-5% fps differences inside the noise.  This test is about
#     pixels.  Numbers are printed for information and nothing is concluded from
#     them.
#
# Exit 2 means "cannot run" (no clip, no ffmpeg, or the fixtures are absent), which
# is deliberately distinct from exit 1, "ran and differed".  A test that cannot run
# and says so is the difference between a missing input and a real regression.

param(
  [string]$Clip = "",
  [int]$Colors = 16,
  # A second fixture that carries audio, run with audio ENABLED.
  #
  # It exists because the audio path was completely untested: both clips in tests/
  # were silent, so `-map 1:a?`, `-c:a copy` and `-shortest` were exercised by no
  # test at all -- and that is exactly where a frame-deleting bug lived.  -shortest
  # stops output at the shorter of the two streams, so when the VIDEO is the longer
  # one it deletes video frames to make the tracks match.  Measured: a 30 s excerpt
  # with 895 video frames against 30.0128 s of audio came out with 892.  A silent
  # fixture cannot catch that however many times it is rendered.
  [string]$AudioClip = ""
)

$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
$rd = Join-Path $root 'build\Release\rdither.exe'
if (-not (Test-Path $rd)) { "missing $rd -- build first"; exit 1 }

if (-not $Clip) { $Clip = Join-Path $root 'tests\clip1920.mp4' }
if (-not (Test-Path $Clip)) {
  "cannot run: no clip at $Clip"
  'expected tests\clip1920.mp4, or pass -Clip <path>'
  exit 2
}
if (-not $AudioClip) { $AudioClip = Join-Path $root 'tests\clip1920_audio.mkv' }
if (-not (Test-Path $AudioClip)) {
  "cannot run: no audio fixture at $AudioClip"
  'expected tests\clip1920_audio.mkv -- the audio path has no other coverage'
  exit 2
}

# The OpenCL video path carries the rgba64le data path only, so both engines are run
# with the settings that select it.  Running them with different settings would
# compare a different question than the one being asked here.
$env:RD_YUV444_OUT = '0'
$tmp = Join-Path $env:TEMP 'rdvid_exact'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

'OpenCL vs CUDA blocks, video pipeline, decoded frames'
''
$fail = 0
foreach ($eng in @('blocks', 'opencl')) {
  $out = Join-Path $tmp "clip_$eng.mkv"
  Remove-Item -EA SilentlyContinue $out
  $text = & $rd --video --engine $eng --colors $Colors --input-mode rgba64 `
                 --video-lossless --no-audio $Clip $out 2>&1 | Out-String
  $rc = $LASTEXITCODE
  if ($rc -ne 0 -or -not (Test-Path $out)) {
    $fail++
    "  $eng  FAILED to run (exit $rc)"
    ($text -split "`n" | Where-Object { $_.Trim() } | Select-Object -First 3) |
      ForEach-Object { "        $($_)" }
    continue
  }
  $fps = if ($text -match 'frames\s+:\s+(\d+) in [\d.]+ s \(([\d.]+) fps\)') {
    "  $($Matches[1]) frames, $($Matches[2]) fps"
  } else { "" }
  "  $eng  ran ok$fps"
}

if ($fail -gt 0) { "`n$fail engine(s) failed to run."; exit 1 }

# Decode both to raw RGBA and compare hashes.  Decoding goes through cmd /c, not the
# PowerShell pipeline: a native command's stdout piped through PowerShell is
# re-encoded, which turns binary into text and destroys the thing being hashed.
$hashes = @{}
foreach ($eng in @('blocks', 'opencl')) {
  $mkv = Join-Path $tmp "clip_$eng.mkv"
  $raw = Join-Path $tmp "clip_$eng.raw"
  Remove-Item -EA SilentlyContinue $raw
  cmd /c "magick ""$mkv"" -depth 8 ""rgba:$raw""" 2>$null | Out-Null
  if (-not (Test-Path $raw)) {
    "  $eng  decode FAILED (is ImageMagick on PATH?)"
    $fail++
    continue
  }
  $h = (Get-FileHash $raw -Algorithm SHA256).Hash
  $hashes[$eng] = $h
  "  $eng  decoded $([math]::Round((Get-Item $raw).Length / 1MB, 1)) MB  sha256 $h"
}

Remove-Item -EA SilentlyContinue (Join-Path $tmp 'clip_*.raw')

if ($fail -gt 0) { "`n$fail engine(s) failed."; exit 1 }

# FRAME ACCOUNTING, on the clip that carries audio.  This is the check that would
# have caught the -shortest bug, and it is separate from the hash comparison above
# because the two fail differently: the hash comparison sees the two ENGINES agree,
# which they did -- both deleted the same frames.  A shared fault is invisible to a
# test that compares implementations against each other, which is the same structural
# blind spot as the uninitialised-channel bug and the same reason
# probe-determinism.ps1 exists.
''
'Frame accounting with audio enabled (-shortest is the risk)'
''
function Count-Frames([string]$f) {
  $n = cmd /c "ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 ""$f""" 2>$null
  if (-not $n) { return -1 }
  return [int]$n
}
$srcFrames = Count-Frames $AudioClip
$audioOut = Join-Path $tmp 'audio_out.mkv'
Remove-Item -EA SilentlyContinue $audioOut
& $rd --video --colors $Colors --video-lossless $AudioClip $audioOut 2>&1 | Out-Null
$outFrames = Count-Frames $audioOut
Remove-Item -EA SilentlyContinue $audioOut

if ($srcFrames -lt 0 -or $outFrames -lt 0) {
  "  frame counting FAILED (is ffprobe on PATH?)"
  $fail++
} elseif ($outFrames -ne $srcFrames) {
  "  FAIL  source has $srcFrames frames, output has $outFrames"
  "        an output with FEWER frames than its source means the pipeline is"
  "        deleting video -- -shortest does exactly that when the video outlasts"
  "        the audio"
  $fail++
} else {
  "  ok  source $srcFrames frames, output $srcFrames frames -- none lost"
}
if ($hashes['blocks'] -ne $hashes['opencl']) {
  ''
  'DIFFER: the two engines did not produce the same frames.'
  exit 1
}
''
'IDENTICAL: every decoded frame matches.'
exit 0
