# Compare two video files by SAMPLED FRAMES SPREAD ACROSS THE CLIP, and report HOW MUCH
# differs.
#
# WHY THIS EXISTS, because a whole-stream hash is the wrong instrument and using one is
# how this session wasted 14 GB and still learned nothing about magnitude.
#
# Two facts, both learned the hard way on 2026-10-08:
#
#   1. A single MD5 over the whole stream is all-or-nothing. It answers "do these differ?"
#      and nothing else -- not WHERE, and not HOW MUCH. Those two facts decide whether a
#      change is safe, and the hash cannot give either. Rendering 600 frames to compare
#      them produced 7119.1 MB per side, 14.2 GB in total, to learn something two hash
#      lines had already said.
#
#   2. A whole-file hash also cannot tell "one pixel moved" from "error diffusion cascaded
#      and a fifth of the picture moved". Those demand opposite answers about whether a
#      palette change is acceptable. Measured on the sample-truncation defect: a palette
#      differing by at most 271 of 65535 per component -- 0.4% -- produced 1.23% of
#      OUTPUT pixels differing with a max delta of 52652, i.e. 80% of full range.
#      Magnitude is the finding; a hash discards it.
#
# WHY THE SAMPLES SPREAD ACROSS THE CLIP, because sampling only the head is blind to the
# most common shape of fault here. On the sample-truncation defect the per-frame divergence
# INCREASED along the clip:
#
#     frame   0   0.972% of components differ
#     frame   1   1.020%
#     frame   2   1.074%
#     frame 599   1.340%   (measured separately, against the same pair)
#
# because the defect is that later frames were never sampled -- so the palette they were
# dithered with is the furthest from the palette they should have had. A probe reading
# frame 0 alone would under-report the fault by 27%. The default here therefore takes the
# FIRST, the LAST, and evenly spaced frames between, so a late-only defect cannot hide.
#
# USAGE
#   probe-frame-diff.ps1 -A <file> -B <file> [-Frames 6] [-Labels "before","after"]
#
# EXIT CODES
#   0  every sampled frame is byte-identical
#   1  they differ; per-frame counts and the worst delta are printed
#   2  cannot run -- a file is missing, ffmpeg is absent, the frame count is unknown, or
#      the decode did not yield exactly the frames asked for. "Cannot run" is deliberately
#      not 0: a probe that cannot run must not be readable as a pass, which is the same
#      floor probe-video-invariance.ps1 applies.
#
# WHAT IT DELIBERATELY DOES NOT DO
#   * It does not hash. It compares, so it can count.
#   * It does not decode the whole file to disk. Only the sampled frames are materialised,
#     so peak usage is the sample buffers (~12.4 MB per frame per side), not the file size.
#     Six frames of 1080p rgb48le is ~75 MB per side against 7119.1 MB for a full decode.
#   * It is NOT a substitute for a whole-clip claim. Six frames cannot prove bit-exactness
#     over 600. For that, use probe-video-pixels.ps1, which hashes decoded pixels. This
#     probe answers the opposite question: when two outputs DIFFER, by how much and where.

param(
    [Parameter(Mandatory = $true)][string]$A,
    [Parameter(Mandatory = $true)][string]$B,
    [int]$Frames = 6,
    [string[]]$Labels = @('A', 'B')
)

$ErrorActionPreference = 'Stop'

if ($Frames -lt 2) {
    Write-Error "-Frames must be at least 2 (a single frame cannot span a clip)"
    exit 2
}
foreach ($p in @($A, $B)) {
    if (-not (Test-Path -LiteralPath $p)) {
        "frame diff: cannot run -- '$p' does not exist"
        exit 2
    }
}

# rgb48le is 3 channels x 2 bytes = 6 B/px = 12,441,600 B for 1920x1080.  rgb24's 3 B/px is
# the wrong constant and makes an N-frame sample look like 2N -- which is an arithmetic
# error this session made, and it briefly looked like a broken select filter.
$FrameBytes = 1920 * 1080 * 3 * 2

function Find-Ffmpeg {
    $c = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
    if ($c -and (Test-Path -LiteralPath $c)) { return $c }
    $hard = 'C:\progi\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe'
    if (Test-Path -LiteralPath $hard) { return $hard }
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { return 'ffmpeg' }
    return $null
}
function Find-Ffprobe {
    $c = (Get-Command ffprobe -ErrorAction SilentlyContinue).Source
    if ($c -and (Test-Path -LiteralPath $c)) { return $c }
    $hard = 'C:\progi\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe'
    if (Test-Path -LiteralPath $hard) { return $hard }
    if (Get-Command ffprobe -ErrorAction SilentlyContinue) { return 'ffprobe' }
    return $null
}

$ffmpeg = Find-Ffmpeg
if (-not $ffmpeg) { "frame diff: cannot run -- no ffmpeg found"; exit 2 }
$ffprobe = Find-Ffprobe
if (-not $ffprobe) { "frame diff: cannot run -- no ffprobe found (frame count is needed to spread the samples)"; exit 2 }

# Frame count of A.  Only A is counted, and the decode of B asserts the same number of
# frames came back, so a length mismatch surfaces as exit 2 rather than as a difference.
function Get-FrameCount {
    param([string]$Path, [string]$Probe)
    $out = & $Probe -v error -select_streams v:0 -count_frames `
                     -show_entries stream=nb_read_frames -of 'default=nw=1:nk=1' $Path 2>$null
    $v = 0
    if ($out -and ([int]::TryParse(($out | Select-Object -First 1).Trim(), [ref]$v))) { return $v }
    return 0
}

$total = Get-FrameCount -Path $A -Probe $ffprobe
if ($total -le 0) { "frame diff: cannot run -- could not count frames in '$A'"; exit 2 }

$want = [Math]::Min($Frames, $total)
if ($want -lt 2) {
    "frame diff: cannot run -- only $total frame(s); cannot span a clip"
    exit 2
}

# FIRST, LAST, and evenly spaced between.  Integer arithmetic only, and the set is
# de-duplicated and sorted so the expected count is exact.
$idx = New-Object 'System.Collections.Generic.List[int]'
$idx.Add(0)
$idx.Add($total - 1)
if ($want -gt 2) {
    for ($i = 1; $i -lt ($want - 1); $i++) {
        $idx.Add([int][Math]::Round($i * ($total - 1) / ($want - 1)))
    }
}
$idx = @($idx | Sort-Object -Unique)
$want = $idx.Count

$terms = ($idx | ForEach-Object { "eq(n\,$_)" }) -join '+'
$filter = "select='$terms'"

function Get-FrameSamples {
    param([string]$Path, [string]$FF, [string]$Filt, [int]$Expect, [string]$Filter2)
    $out = Join-Path ([IO.Path]::GetTempPath()) ("framediff_" + [IO.Path]::GetRandomFileName() + ".raw")
    try {
        & $FF -v error -i $Path -vf $Filter2 -fps_mode passthrough `
               -frames:v $Expect -pix_fmt rgb48le -f rawvideo -y $out 2>$null
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $out)) { return $null }
        $len = (Get-Item -LiteralPath $out).Length
        # ASSERT the decode produced exactly the frames asked for.  Trusting the frame
        # count without this check is how a mismatched comparison becomes a fake result.
        if ($len -ne ($Expect * $FrameBytes)) { return $null }
        return ,$out
    } catch { return $null }
}

$sa = Get-FrameSamples -Path $A -FF $ffmpeg -Filt $filter -Expect $want -Filter2 $filter
if (-not $sa) { "frame diff: cannot run -- '$A' did not decode to exactly $want whole frame(s)"; exit 2 }
$sb = Get-FrameSamples -Path $B -FF $ffmpeg -Filt $filter -Expect $want -Filter2 $filter
if (-not $sb) {
    Remove-Item -LiteralPath $sa -Force -EA SilentlyContinue
    "frame diff: cannot run -- '$B' did not decode to exactly $want whole frame(s)"
    exit 2
}

try {
    $la = (Get-Item -LiteralPath $sa).Length
    $lb = (Get-Item -LiteralPath $sb).Length
    if ($la -ne $lb) {
        "frame diff: cannot run -- sample sizes differ ($la vs $lb)"
        exit 2
    }
    $n = [int]($la / $FrameBytes)

    $lbl = @($Labels) -join ',' -split ','
    $la0 = if ($lbl.Count -ge 1 -and $lbl[0].Trim()) { $lbl[0].Trim() } else { 'A' }
    $lb0 = if ($lbl.Count -ge 2 -and $lbl[1].Trim()) { $lbl[1].Trim() } else { 'B' }

    "frame diff: $n frame(s) of 1920x1080 rgb48le sampled across a $total-frame clip"
    "  indices: $($idx -join ', ')   bytes: $([Math]::Round($la/1MB,2)) MB per side"
    "  comparing: $la0  vs  $lb0"

    $totalDiff = 0
    $worst = 0
    $worstFrame = -1
    $fa = [IO.File]::ReadAllBytes($sa)
    $fb = [IO.File]::ReadAllBytes($sb)
    $comps = 1920 * 1080 * 3
    for ($f = 0; $f -lt $n; $f++) {
        $off = $f * $FrameBytes
        $bad = 0
        $frameMax = 0
        for ($i = 0; $i -lt $FrameBytes; $i += 2) {
            $va = [int]$fa[$off + $i] -bor ([int]$fa[$off + $i + 1] -shl 8)
            $vb = [int]$fb[$off + $i] -bor ([int]$fb[$off + $i + 1] -shl 8)
            if ($va -ne $vb) {
                $bad++
                $d = [Math]::Abs($va - $vb)
                if ($d -gt $frameMax) { $frameMax = $d }
            }
        }
        $totalDiff += $bad
        if ($frameMax -gt $worst) { $worst = $frameMax; $worstFrame = $idx[$f] }
        "  frame {0,-5} components differing {1,9} of {2} ({3,8}%)   max delta {4}" -f $idx[$f], $bad, $comps, [Math]::Round(100.0 * $bad / $comps, 4), $frameMax
    }
    ""
    "  TOTAL components differing: $totalDiff    worst delta: $worst of 65535 (frame $worstFrame)"
    if ($totalDiff -eq 0) {
        "frame diff: all $n sampled frame(s) are byte-identical"
        exit 0
    }
    "frame diff: $totalDiff component(s) differ across $n sampled frame(s) -- worst delta $worst of 65535"
    "  NOTE: $n of $total frames sampled. For a whole-clip equality claim use tools\probe-video-pixels.ps1."
    exit 1
} finally {
    Remove-Item -LiteralPath $sa, $sb -Force -EA SilentlyContinue
}