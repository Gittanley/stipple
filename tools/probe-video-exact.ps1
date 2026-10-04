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
  [string]$AudioClip = "",
  # Which rdither to test.  verify.ps1 passes its own -Rdither through.  This probe
  # had no such parameter and hardcoded build\Release\rdither.exe, so pointing the
  # suite at another build tested a different binary here than everywhere else.
  [string]$Rdither = ""
)

$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
if (-not $Rdither) { $Rdither = Join-Path $root 'build\Release\rdither.exe' }
$rd = $Rdither
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

# BOTH data paths, because the engine now has two and only one of them was being
# checked.
#
# The planar 4:4:4 path is CUDA's default and is the one that matters: it is 3 bytes
# per pixel against rgba64le's 8, and being unable to use it is what made this engine
# look 2.2x slower than CUDA when it is within about 8% on the same path.  It is also
# a DIFFERENT code path -- its gather and scatter are different kernels with their own
# constants -- so a pass on rgba64le says nothing about it.
#
# Both engines are always run with the settings that select the path under test.
# Running them with different settings would compare a different question than the one
# being asked here, which is exactly the mistake that produced the 2.2x figure.
$env:RD_YUV444_OUT = '0'
$tmp = Join-Path $env:TEMP 'rdvid_exact'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# 'rgba64' keeps the pre-existing route (RD_YUV444_OUT=0, --input-mode rgba64).
# 'yuv444'  drops RD_YUV444_OUT so the planar output kernel is used, and asks for
#           the planar input mode.
$paths = @(
  @{ name = 'rgba64le (8 B/px)'; mode = 'rgba64'; yuv = '0' },
  @{ name = 'yuv444p  (3 B/px)'; mode = 'yuv444'; yuv = $null }
)

$fail = 0
$compared = 0
$skipped = 0
foreach ($p in $paths) {
  if ($null -eq $p.yuv) { Remove-Item Env:\RD_YUV444_OUT -EA SilentlyContinue }
  else { $env:RD_YUV444_OUT = $p.yuv }

  "data path: $($p.name)"
  ''
  $ran = @()
  foreach ($eng in @('blocks', 'opencl')) {
    $out = Join-Path $tmp "clip_$eng.mkv"
    Remove-Item -EA SilentlyContinue $out
    $text = & $rd --video --engine $eng --colors $Colors --input-mode $p.mode `
                   --video-lossless --no-audio $Clip $out 2>&1 | Out-String
    $rc = $LASTEXITCODE
    if ($rc -ne 0 -or -not (Test-Path $out)) {
      # An engine with no device is NOT a failure.  `--engine opencl` on a machine
      # with no usable OpenCL exits nonzero BY DESIGN: rd_video.cpp refuses rather
      # than falling back to CUDA, so that `--engine opencl` cannot quietly end up
      # measuring the other engine.  This probe used to call that a failure, and the
      # only reason that was quiet is that the same run used to exit 0 having written
      # nothing at all.  Now that a run which dithers no frames is a hard error, the
      # two cases have to be told apart or the suite fails on every GPU-less runner --
      # which is every runner CI has.
      $why = (($text -split "`n" | Where-Object { $_ -match 'error:' } |
               Select-Object -First 1) -replace '.*error:\s*', '').Trim()
      if ($text -match 'requested for video but' -or
          $text -match 'this build has no (OpenCL|CUDA)' -or
          $text -match 'no CUDA device') {
        $skipped++
        "  $eng  SKIPPED (no usable device: $why)"
      } else {
        $fail++
        "  $eng  FAILED to run (exit $rc)"
        ($text -split "`n" | Where-Object { $_.Trim() } | Select-Object -First 3) |
          ForEach-Object { "        $($_)" }
      }
      continue
    }
    $ran += $eng
    $fps = if ($text -match 'frames\s+:\s+(\d+) in [\d.]+ s \(([\d.]+) fps\)') {
      "  $($Matches[1]) frames, $($Matches[2]) fps"
    } else { "" }
    # An engine that ran ZERO frames has not run.  `--engine opencl` on a machine
    # with no OpenCL device exits 0 and leaves the "ran ok" line without a frame
    # count, because the engine reports itself unavailable and the pipeline hands
    # the work to the host workers rather than refusing.  Counting that as a run
    # made this probe fail on every GPU-less machine -- including CI, which has no
    # GPU -- with a misleading "decode FAILED (is ImageMagick on PATH?)" pointing at
    # the wrong thing entirely.
    #
    # The fix is to require frames, and to say which engine is absent rather than
    # treating it as a defect.  An absent engine is a SKIP, matching what the image
    # suite already does and reports as SKIPPED rather than passed.
    if ($text -notmatch 'frames\s+:\s+([1-9]\d*)\s+in') {
      $skipped++
      "  $eng  SKIPPED (no frames: this build has no usable $eng device)"
      $ran = @($ran | Where-Object { $_ -ne $eng })
      continue
    }
    "  $eng  ran ok$fps"
  }
  if ($ran.Count -ne 2) { ''; continue }

  # Decode both to raw RGBA and compare hashes.  Decoding goes through cmd /c, not
  # the PowerShell pipeline: a native command's stdout piped through PowerShell is
  # re-encoded, which turns binary into text and destroys the thing being hashed.
  $hashes = @{}
  foreach ($eng in $ran) {
    $mkv = Join-Path $tmp "clip_$eng.mkv"
  # ffmpeg streams the decoded frames straight into md5.  The obvious alternative,
  # `magick "$mkv" -depth 8 "rgba:$raw"`, decodes EVERY frame into a host buffer and
  # writes it out: measured 1,865.7 MiB peak working set and a 449 MiB file for the
  # 60-frame 1080p clip, repeated once per engine.  On a longer clip it is GB-scale --
  # this is where an 8 GB spike came from.  This form writes no file and holds one
  # frame.  It hashes every decoded sample of every frame, which is what the raw
  # file was hashed for.
  $h = (& ffmpeg -v error -i $mkv -f rawvideo -pix_fmt rgba -f md5 - 2>$null | Out-String).Trim()
  if (-not $h) {
    "  $eng  decode FAILED (is ffmpeg on PATH?)"
    $fail++
    continue
  }
  $hashes[$eng] = $h
  "  $eng  decoded via ffmpeg md5  $h"
}

  Remove-Item -EA SilentlyContinue (Join-Path $tmp 'clip_*.raw')

  if ($hashes['blocks'] -ne $hashes['opencl']) {
    $fail++
    "  DIFFER on $($p.name):"
    "    the two engines are defined to have the same partition and the same"
    "    arithmetic, so any difference is a defect, not a tolerance to negotiate."
    "    A difference in the swscale constants or the plane stride shows up as a hue"
    "    shift that looks like a dither bug."
  } else {
    $compared++
    '  IDENTICAL: every decoded frame matches.'
  }
  ''
}
Remove-Item Env:\RD_YUV444_OUT -EA SilentlyContinue

if ($fail -gt 0) { "`n$fail check(s) failed."; exit 1 }
if ($compared -eq 0) {
  # Exit 2 is the "cannot run" convention used across this suite, distinct from a
  # pass and from a failure.  An absent engine is a skip, never a pass.
  "`ncannot run: no data path produced two comparable engines. ($skipped engine run(s) skipped for want of a device)"
  exit 2
}
"$compared of $($paths.Count) data paths identical."
if ($skipped -gt 0) {
  "$skipped engine run(s) SKIPPED for want of a device.  These are not passes."
}

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
if ($fail -gt 0) { ''; exit 1 }
''
"Frame accounting ok.  $compared of $($paths.Count) data paths bit-identical."
exit 0
