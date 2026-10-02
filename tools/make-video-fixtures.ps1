<#
.SYNOPSIS
    Generate small video fixtures for the checks that cannot run in a clean clone.

.DESCRIPTION
    Three of the video checks report SKIPPED in CI, and ffmpeg was not the reason.
    With ffmpeg installed the runner log says:

        cannot run: no clip at tests\clip1920.mp4
        cannot run: no video fixture (tests\clip1920_audio.mkv)

    because tests\clip1920.mp4 and tests\clip1920_audio.mkv are large local files that
    are deliberately not tracked.  That is why the video path had no automated
    coverage at all, and it is why the --cpu-threads 0 access violation could sit in
    the tree through a green CI run: the code was never reached.

    So the fixtures are generated instead of committed.  Nothing binary enters the
    repository, the clips are a couple of hundred kilobytes rather than tens of
    megabytes, and the check runs on any machine that has ffmpeg -- which verify.ps1
    now requires anyway.

    The clips are deliberately SMALL, and that is a real trade rather than an
    oversight.  What each check asserts is size-independent: cross-engine bit
    identity, run-to-run determinism, and that no frame is lost to -shortest.  What
    is lost is the chance of a bug that only appears at 1920x1080 with 60 frames.
    A bug of that shape would need a bigger fixture to catch, and the honest
    statement is that this narrows the blind spot rather than closing it.

.PARAMETER OutDir
    Where to write.  Defaults to a directory under the system temp dir.

.PARAMETER Ffmpeg
    Path to ffmpeg.  Defaults to whatever is on PATH.

.EXAMPLE
    pwsh -File tools\make-video-fixtures.ps1
#>
[CmdletBinding()]
param(
  [string]$OutDir = "",
  [string]$Ffmpeg = ""
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if (-not $Ffmpeg) {
  $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
  if (-not $cmd) {
    Write-Host 'cannot run: ffmpeg not found on PATH'
    exit 2
  }
  $Ffmpeg = $cmd.Source
}
if (-not $OutDir) {
  $OutDir = Join-Path ([IO.Path]::GetTempPath()) 'rd_video_fixtures'
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# ffprobe sits beside ffmpeg in every build that has ffmpeg, and it is the tool that
# can count frames.  The first version passed the probe arguments to FFMPEG, which
# cannot answer them, so the count came back empty and every fixture looked like it
# had 0 frames.  Resolved explicitly rather than assumed, and checked, because a
# missing ffprobe here means the assertions below cannot be made at all.
$ffprobe = Join-Path (Split-Path $Ffmpeg -Parent) 'ffprobe.exe'
if (-not (Test-Path $ffprobe)) {
  $cmd = Get-Command ffprobe -ErrorAction SilentlyContinue
  if ($cmd) { $ffprobe = $cmd.Source }
}
if (-not (Test-Path $ffprobe)) {
  Write-Host "cannot run: no ffprobe beside $Ffmpeg, so the fixtures cannot be verified"
  exit 2
}

# 320x180, 30 frames at 30 fps = exactly 1.000 s of video.  testsrc because it has
# hard edges and gradients, which is what makes an unvisited pixel visible at all:
# a flat colour would dither identically everywhere and prove nothing.
$Size = '320x180'
$Rate = 30
$Frames = 30

# The audio fixture's shape is not arbitrary.  -shortest stops the output at the
# shorter stream, and the frame-deleting bug it caused only appears when the VIDEO is
# not the shorter one.  The measured case in the README was 895 video frames (29.83 s)
# against 30.0128 s of audio, which came out with 892.  So audio here is deliberately
# the longer stream: 1.05 s against 1.000 s of video.
$AudioSeconds = 1.05

function Invoke-Ffmpeg([string[]]$FfmpegArgs, [string]$What) {
  # ffmpeg's own message is captured and shown.  The first version piped it to
  # Out-Null, so a failure reported only "ffmpeg failed to build the audio clip" --
  # which is the least useful sentence available, because the real cause (an input
  # option bound to the wrong file) is on the line that was thrown away.
  $out = & $Ffmpeg @FfmpegArgs 2>&1 | Out-String
  if ($LASTEXITCODE -ne 0) {
    $detail = ($out -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -First 2) -join ' / '
    throw "ffmpeg failed to build the ${What}: $detail"
  }
}

$clip = Join-Path $OutDir 'clip.mp4'
$clipAudio = Join-Path $OutDir 'clip_audio.mkv'

Write-Host "video fixtures -> $OutDir"

# h264 in mp4 for the silent one: it is the same shape as the real clip, so the
# decode path exercised is the one the project actually ships against.
Invoke-Ffmpeg @('-y', '-hide_banner', '-loglevel', 'error',
  '-f', 'lavfi', '-i', "testsrc=size=$Size`:rate=$Rate",
  '-frames:v', "$Frames", '-pix_fmt', 'yuv420p', '-c:v', 'libx264',
  $clip) 'silent clip'

# ffv1 in mkv for the audio one, matching tests\clip1920_audio.mkv.  Both -f lavfi
# inputs come FIRST and every output option after them: an output option that appears
# before the second -i is bound to that INPUT by ffmpeg, which fails with "Option
# frames:v cannot be applied to input url sine=...".  -frames:v and not -t: the lavfi
# source is endless, so a duration cap would depend on the muxer rounding, and this
# fixture should have an exact frame count.
Invoke-Ffmpeg @('-y', '-hide_banner', '-loglevel', 'error',
  '-f', 'lavfi', '-i', "testsrc=size=$Size`:rate=$Rate",
  '-f', 'lavfi', '-i', "sine=frequency=440:sample_rate=48000:duration=$AudioSeconds",
  '-frames:v', "$Frames", '-pix_fmt', 'yuv444p', '-c:v', 'ffv1',
  '-c:a', 'pcm_s16le', '-shortest',
  $clipAudio) 'audio clip'

# Assert what was built rather than trusting exit 0.  A fixture with the wrong shape
# would make every check that uses it quietly meaningless, which is worse than no
# fixture: it would report a pass.
foreach ($pair in @(@{ path = $clip; want = $Frames; what = 'silent clip' },
                    @{ path = $clipAudio; want = $Frames; what = 'audio clip' })) {
  if (-not (Test-Path $pair.path)) { throw "the $($pair.what) was not created" }
  $counted = & $ffprobe -v error -select_streams v:0 -count_frames `
    -show_entries stream=nb_read_frames -of default=nw=1:nk=1 $pair.path 2>&1 | Out-String
  $n = [int](($counted -replace '\D', ''))
  if ($n -ne $pair.want) {
    throw "the $($pair.what) has $n frames, expected $($pair.want) (ffprobe said '$counted')"
  }
  $kb = [math]::Round((Get-Item $pair.path).Length / 1KB, 1)
  Write-Host ("  {0,-12} {1,4} frames  {2,7} kB  {3}" -f $pair.what, $n, $kb, (Split-Path $pair.path -Leaf))
}
if (-not (& $ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=nw=1 $clipAudio 2>&1)) {
  throw 'the audio clip has no audio stream, which is the whole point of it'
}
Write-Host '  audio stream present on the audio clip'

# Emitted as parseable lines so verify.ps1 can read the paths back rather than
# re-deriving them and getting a different answer.
Write-Host "CLIP=$clip"
Write-Host "CLIP_AUDIO=$clipAudio"
exit 0
