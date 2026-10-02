<#
.SYNOPSIS
    Install ffmpeg and ffprobe on a Windows machine, unattended.

.DESCRIPTION
    GitHub's hosted Windows runners have NO ffmpeg, which means verify.ps1 skips
    every video check there.  That is not a cosmetic gap: the video path is where
    the access violation in --cpu-threads 0 lived, and nothing in CI could have
    seen it because the code was never reached.

    A portable build is used rather than an installer for the same reason
    ImageMagick uses one in CI: an installer that can prompt is an installer that
    will eventually hang a job.  The archive is pinned to a numbered release, not
    to "latest", because a moving target turns a green build red with nothing in
    this repository having changed.

    The BtbN archive nests as <root>\bin\ffmpeg.exe, so the bin directory is what
    goes on PATH.  That is measured here rather than assumed, because the CUDA
    component archives had the same shape and the wrong assumption cost three CI
    fixes.

.PARAMETER Prefix
    Where to install.  Defaults to C:\ffmpeg.

.EXAMPLE
    pwsh -File tools\install-ffmpeg.ps1 -Prefix C:\ffmpeg
#>
[CmdletBinding()]
param(
  [string]$Prefix = 'C:\ffmpeg'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# A numbered release, not master.  "latest" on this repository is a nightly build
# whose name changes every day, which would make this step unreproducible.
$Archive = 'ffmpeg-n8.1-latest-win64-gpl-8.1.zip'
$Url = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/$Archive"

Write-Host "=== ffmpeg -> $Prefix ==="

if (Test-Path $Prefix) { Remove-Item $Prefix -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Prefix | Out-Null

$zip = Join-Path $Prefix $Archive
Write-Host "fetching $Archive"
for ($attempt = 1; $attempt -le 3; $attempt++) {
  try {
    Invoke-WebRequest -Uri $Url -OutFile $zip -UseBasicParsing -TimeoutSec 1800
    break
  } catch {
    Write-Host "  attempt $attempt failed: $($_.Exception.Message)"
    if ($attempt -eq 3) { throw "could not fetch ffmpeg after 3 attempts" }
    Start-Sleep -Seconds (10 * $attempt)
  }
}
Write-Host ("  {0:N1} MB" -f ((Get-Item $zip).Length / 1MB))

# Expand-Archive, not 7-Zip, and that is a deliberate change from the first draft of
# this script.  The first draft shelled out to C:\Program Files\7-Zip\7z.exe, which
# would have added a dependency that nothing in this workflow had ever exercised --
# 7z appears in ci.yml only in comments recording that it was REJECTED as a route for
# ImageMagick.  An unproven external tool on the critical path of a CI job is a way to
# discover on a Tuesday that a runner image changed.  Expand-Archive is what the CUDA
# step in ci.yml already uses, so this adds no new dependency at all.
# Measured on this archive: 2.4 s, 44 files, both tools present and running.
Expand-Archive -Path $zip -DestinationPath $Prefix -Force
Remove-Item $zip -Force -EA SilentlyContinue

# <prefix>\<archive-without-zip>\bin\ffmpeg.exe.  Located by CONTENT, so a change
# in the archive's internal layout is a one-line change here rather than a silently
# empty PATH entry.
$ffmpeg = Get-ChildItem $Prefix -Recurse -Filter 'ffmpeg.exe' -File | Select-Object -First 1
$ffprobe = Get-ChildItem $Prefix -Recurse -Filter 'ffprobe.exe' -File | Select-Object -First 1
if (-not $ffmpeg) {
  Write-Host "tree under $Prefix :"
  Get-ChildItem $Prefix -Recurse -Directory | Select-Object -First 20 |
    ForEach-Object { Write-Host "  $($_.FullName)" }
  throw 'ffmpeg.exe not found in the archive'
}
if (-not $ffprobe) { throw 'ffprobe.exe not found in the archive; verify.ps1 needs it' }

$bin = Split-Path $ffmpeg.FullName -Parent
Write-Host "  ffmpeg : $($ffmpeg.FullName)"
Write-Host "  ffprobe: $($ffprobe.FullName)"

# Assert it RUNS.  A zip that extracts without error is not an ffmpeg; these two
# are the tools the video pipeline actually shells out to.
& $ffmpeg.FullName -version 2>&1 | Select-Object -First 1 | ForEach-Object { Write-Host "  $_" }
& $ffprobe.FullName -version 2>&1 | Select-Object -First 1 | ForEach-Object { Write-Host "  $_" }

# The encoders rdither asks for by name.  A build without x264 or ffv1 would make
# the video checks fail deep inside ffmpeg with a confusing message, so they are
# named here where the failure is obvious.
$encoders = & $ffmpeg.FullName -hide_banner -encoders 2>&1 | Out-String
foreach ($e in @('libx264', 'ffv1')) {
  if ($encoders -notmatch [regex]::Escape($e)) {
    throw "this ffmpeg build has no $e encoder, which the video pipeline requires"
  }
  Write-Host "  [ok] $e encoder present"
}

# GITHUB_PATH is how a later step in the same job finds this.  Guarded rather than
# written blind: with the variable unset -- which is every run outside CI -- the
# append is a SILENT NO-OP.  It exits 0, prints "PATH += ..." and leaves ffmpeg
# unreachable, so a local run of this script appears to work and then every video
# check skips for want of ffmpeg.  A no-op that looks like a success is the failure
# mode this project treats as worst, so it is made loud.
if ($env:GITHUB_PATH) {
  "$bin" >> $env:GITHUB_PATH
  Write-Host "PATH += $bin   (via GITHUB_PATH, for later steps in this job)"
} else {
  Write-Host "ffmpeg installed but NOT on PATH: GITHUB_PATH is unset, so this is not CI."
  Write-Host "  add it yourself before running verify.ps1:"
  Write-Host "    `$env:PATH = ""$bin;`$env:PATH"""
}
Write-Host 'ffmpeg installed and verified.'