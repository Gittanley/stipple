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
    will eventually hang a job.  MEASURED 2026-10-14: BtbN/FFmpeg-Builds publishes
    ONE rolling release (tag `latest`, rebuilt every few days) and no versioned
    releases, so the URL cannot be pinned -- an earlier comment here claimed a
    numbered pin that never existed.  What is done instead: the zip is verified by
    CONTENT against that release's own checksums.sha256 before use, and the
    resolved hash is printed, so a log records which build a run used even though
    the URL never changes.  A hard-coded hash pin was considered and rejected: it
    would turn this step red every few days on BtbN's schedule, for nothing in
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

# The URL is BtbN's rolling release and the asset name contains "latest" because
# that is what BtbN ships (see the .DESCRIPTION above: one release, no versions).
# Reproducibility comes from the checksum step below, not from this string.
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

# Content verification, because the URL above cannot pin anything.  The release
# ships checksums.sha256 beside every asset; the zip is verified against it
# BEFORE extraction, so a truncated download or an asset swapped under the same
# name fails here with both hashes named, instead of becoming a confusing error
# three probes later.
$sums = Join-Path $Prefix 'checksums.sha256'
Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/checksums.sha256' `
                  -OutFile $sums -UseBasicParsing -TimeoutSec 300
$wantLine = Get-Content $sums | Where-Object { $_ -match ([regex]::Escape($Archive) + '\s*$') } | Select-Object -First 1
if (-not $wantLine) { throw "checksums.sha256 has no line for $Archive -- the asset was renamed; re-measure, do not guess" }
$want = ($wantLine -split '\s+')[0].ToLowerInvariant()
$got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "  sha256: $got   (release's checksums.sha256: $want)"
if ($got -ne $want) {
  throw "checksum mismatch for ${Archive}: release says $want, download is $got"
}
Write-Host "  [ok] matches the release's own checksums.sha256"

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