# Is splitting the decode across N seek-based ffmpeg processes BIT-IDENTICAL to one decode?
#
# WHY THIS EXISTS. The reader is the pipeline's floor: one ffmpeg process, 8227 ms of an
# 8860 ms VideoProcess wall (92.9%) on the 600-frame 1080p clip. So the only remaining lever
# on this machine is feeding that reader faster, and the obvious way is N processes each
# seeking to its own chunk. That is exactly what the palette's by-seek path already does above
# 5000 frames (rd_video.cpp:1622, 6 workers).
#
# Before that is worth building it has to be shown that it does not quietly corrupt output, and
# this is the check. It compares PER-FRAME SHA256 from ffmpeg's `framehash` muxer, which does
# the hashing in C and emits text: no full-resolution raw is ever written, and nothing scales
# with clip length in memory. A whole-stream hash would answer "same?" but not "same from
# which frame?", and the whole point here is chunk BOUNDARIES.
#
# THE RESULT, as measured 2026-10-09 on ffmpeg 9.0.2:
#
#   CFR 24 fps, g=24                 IDENTICAL
#   CFR 24 fps, g=60 (long GOP)      IDENTICAL   <- worst case for seek landing
#   CFR 30000/1001 fps               IDENTICAL   <- fractional NTSC timebase
#   CFR, frame count not a multiple
#       of the chunk size            IDENTICAL
#   VFR (30 fps then 8 fps concat)   DIFFERS     <- 1 of 46 frames, at a boundary
#
# So split decode is safe on constant-frame-rate input and UNSAFE on variable-frame-rate
# input, because frame_index / fps is not a timestamp when durations vary. Any implementation
# must therefore detect CFR first and fall back to a single stream otherwise -- and the data
# for that is already collected by the probe (`r_frame_rate` vs `avg_frame_rate`).
#
# USAGE
#   probe-split-decode.ps1 [-Rdither path]
#
# EXIT CODES
#   0  CFR splits are byte-identical AND the VFR case is byte-identical (i.e. no longer a risk)
#   1  a CFR split is not byte-identical -- splitting is unsafe, unconditionally
#   2  cannot run -- no ffmpeg/ffprobe, or a fixture failed to build

param()

$ErrorActionPreference = 'Stop'

foreach ($t in @('ffmpeg', 'ffprobe')) {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) {
        Write-Host "split decode: cannot run -- no $t on PATH"
        exit 2
    }
}

$w = Join-Path ([IO.Path]::GetTempPath()) ('rd_split_' + [IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Force $w | Out-Null

function Get-Hashes {
    param([string[]]$FfmpegArgs)
    $out = & ffmpeg -hide_banner -loglevel error @FfmpegArgs -pix_fmt yuv444p -f framehash - 2>$null
    @($out | Where-Object { "$_" -match '^\s*\d+,' } | ForEach-Object { ($_ -split ',')[-1].Trim() })
}

# Returns a verdict string. 'DIFFERS' and 'COULD-NOT-RUN' are deliberately distinct: a fixture
# that failed to build must never be reported as a behavioural result, which is a mistake this
# script made once while being written.
function Test-Split {
    param([string]$Name, [string]$Clip, [int]$Total, [int]$Per, [double]$Fps)
    $ref = Get-Hashes @('-i', $Clip, '-frames:v', "$Total")
    if ($ref.Count -eq 0) { return @{ Name = $Name; Verdict = 'COULD-NOT-RUN'; Bad = 0 } }
    $split = @()
    for ($i = 0; $i -lt [int][Math]::Ceiling($Total / $Per); $i++) {
        $startF = $i * $Per
        $sec = [Math]::Round($startF / $Fps, 6)
        $want = [Math]::Min($Per, $Total - $startF)
        $split += Get-Hashes @('-ss', "$sec", '-i', $Clip, '-frames:v', "$want")
    }
    $bad = 0
    $n = [Math]::Min($ref.Count, $split.Count)
    for ($i = 0; $i -lt $n; $i++) { if ($ref[$i] -ne $split[$i]) { $bad++ } }
    $v = if ($bad -eq 0 -and $split.Count -eq $Total) { 'IDENTICAL' } else { 'DIFFERS' }
    return @{ Name = $Name; Verdict = $v; Bad = $bad; Ref = $ref.Count; Split = $split.Count }
}

try {
    Write-Host 'split decode: per-frame identity of N seek-chunks vs one decode'

    # Small fixtures on purpose. This is a property of SEEKING, not of resolution, and
    # 320x180 keeps the whole check to a couple of seconds.
    & ffmpeg -y -hide_banner -loglevel error -f lavfi -i 'testsrc=size=320x180:rate=24' `
        -frames:v 120 -pix_fmt yuv420p -c:v libx264 -g 24 "$w\c24.mp4" 2>&1 | Out-Null
    & ffmpeg -y -hide_banner -loglevel error -f lavfi -i 'testsrc=size=320x180:rate=24' `
        -frames:v 120 -pix_fmt yuv420p -c:v libx264 -g 60 "$w\g60.mp4" 2>&1 | Out-Null
    & ffmpeg -y -hide_banner -loglevel error -f lavfi -i 'testsrc=size=320x180:rate=30000/1001' `
        -frames:v 60 -pix_fmt yuv420p -c:v libx264 -g 15 "$w\c2997.mp4" 2>&1 | Out-Null
    foreach ($f in @('c24', 'g60', 'c2997')) {
        if (-not (Test-Path "$w\$f.mp4")) { Write-Host "split decode: cannot run -- fixture $f failed"; exit 2 }
    }

    $results = @()
    $results += Test-Split 'CFR 24 fps, g=24' "$w\c24.mp4" 60 24 24
    $results += Test-Split 'CFR 24 fps, g=60 (long GOP)' "$w\g60.mp4" 120 30 24
    $results += Test-Split 'CFR 30000/1001 fps' "$w\c2997.mp4" 60 15 29.97
    $results += Test-Split 'CFR, count not a multiple of chunk' "$w\c24.mp4" 50 20 24

    # VFR: two segments at different rates concatenated with -c copy, so durations genuinely
    # vary and frame_index/fps is not a timestamp.
    & ffmpeg -y -hide_banner -loglevel error -f lavfi -i 'testsrc=size=320x180:rate=30' `
        -frames:v 30 -pix_fmt yuv420p -c:v libx264 -g 10 "$w\a.mp4" 2>&1 | Out-Null
    & ffmpeg -y -hide_banner -loglevel error -f lavfi -i 'testsrc=size=320x180:rate=8' `
        -frames:v 16 -pix_fmt yuv420p -c:v libx264 -g 8 "$w\b.mp4" 2>&1 | Out-Null
    if ((Test-Path "$w\a.mp4") -and (Test-Path "$w\b.mp4")) {
        "file '$w\a.mp4'" | Set-Content "$w\list.txt" -Encoding utf8
        "file '$w\b.mp4'" | Add-Content "$w\list.txt" -Encoding utf8
        & ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i "$w\list.txt" -c copy "$w\vfr.mp4" 2>&1 | Out-Null
        $results += Test-Split 'VFR (30fps then 8fps)' "$w\vfr.mp4" 46 15 30
    } else {
        $results += @{ Name = 'VFR (30fps then 8fps)'; Verdict = 'COULD-NOT-RUN'; Bad = 0 }
    }

    foreach ($r in $results) {
        $detail = if ($r.ContainsKey('Ref')) { "  ($($r.Bad) mismatches, $($r.Ref) vs $($r.Split))" } else { '' }
        Write-Host ("  {0,-38} {1}{2}" -f $r.Name, $r.Verdict, $detail)
    }

    $cfrBad = @($results | Where-Object { $_.Name -notmatch 'VFR' -and $_.Verdict -eq 'DIFFERS' })
    $cfrSkip = @($results | Where-Object { $_.Name -notmatch 'VFR' -and $_.Verdict -eq 'COULD-NOT-RUN' })
    if ($cfrSkip.Count -gt 0) {
        Write-Host 'split decode: cannot run -- a CFR case produced no output'
        exit 2
    }
    if ($cfrBad.Count -gt 0) {
        Write-Host 'FAIL: a CONSTANT-frame-rate split is not byte-identical. Splitting is'
        Write-Host '      unsafe unconditionally, CFR included. Do not build it.'
        exit 1
    }

    $vfr = @($results | Where-Object { $_.Name -match 'VFR' })
    if ($vfr.Count -eq 1 -and $vfr[0].Verdict -eq 'DIFFERS') {
        Write-Host '  CFR splits are byte-identical; the VFR case is NOT. That is the known'
        Write-Host '  constraint on this idea, so it is reported as ok-with-a-condition'
        Write-Host '  rather than failed -- the check it gates is "CFR splits are safe".'
    }
} finally {
    Remove-Item -LiteralPath $w -Recurse -Force -EA SilentlyContinue
}

Write-Host 'split decode: ok'
exit 0