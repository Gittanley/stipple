# Does --palette-budget-ms actually bound the palette stage?
#
# WHY THIS PROBE EXISTS. The flag is documented as a time budget -- `rd_cli.cpp:125` says
# "time budget for that sampling (default 60000)" and README.md:582 says "as many as a
# 60-second budget allows". It did not do that. `rd_video.cpp:1690` computed a sample COUNT
# from a hardcoded 0.12 s per sample:
#
#     const int affordable = static_cast<int>((budget_s - 0.02) / 0.12);
#
# No clock was ever read, so there was no deadline: the count was decided up front and the
# stage then ran for as long as it took. On a slow machine it overran the budget; the
# comment above it claimed the opposite ("a slow disk takes fewer samples instead of taking
# the same time it always took"), which is backwards.
#
# WHY IT NEEDS AN ENV SEAM TO BE TESTABLE. `--palette-budget-ms 1` cannot distinguish the two
# behaviours, because that value collapses the arithmetic to `affordable = max(1, negative)
# = 1`, i.e. one sample regardless of any deadline. So lowering the flag proves nothing.
# `RD_PALETTE_DEADLINE_MS` overrides ONLY the deadline and never the count, which is what
# makes "the deadline fired" observable while the requested sample count stays fixed. It
# follows the existing seam naming (`RD_PALETTE_OVERLAP`, `RD_PALETTE_SEEK`, `RD_TRACE`).
#
# WHAT IS ASSERTED, and what deliberately is NOT:
#   - the deadline REDUCES the number of distinct samples placed. That is the flag working.
#   - the render still succeeds and still produces a palette. A deadline that returns an
#     error, or that leaves the palette empty, fails here.
#   - the output is NOT asserted to be identical to an unbudgeted run, because it cannot be:
#     fewer samples is a different montage, therefore a different palette, therefore different
#     pixels. That is the accepted consequence of a real deadline and the reason this flag
#     stays off the default path -- see docs\KNOWN-ISSUES.md.
#
# EXIT CODES
#   0  the deadline bounded the stage, and the render still succeeded
#   1  the deadline did not reduce the sample count, OR the bounded render failed
#   2  cannot run -- no ffmpeg/ffprobe, no rdither, or no fixture

param(
    [string]$Rdither = '',
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'

foreach ($t in @('ffmpeg', 'ffprobe')) {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) {
        Write-Host "palette deadline: cannot run -- no $t on PATH"
        exit 2
    }
}
if (-not $Rdither) {
    $root = Split-Path $PSScriptRoot -Parent
    $p = Join-Path $root 'build\Release\rdither.exe'
    if (Test-Path -LiteralPath $p) { $Rdither = $p }
}
if (-not $Rdither -or -not (Test-Path -LiteralPath $Rdither)) {
    Write-Host 'palette deadline: cannot run -- rdither.exe not found'
    exit 2
}

if (-not $OutDir) {
    $OutDir = Join-Path ([IO.Path]::GetTempPath()) ('rd_deadline_' + [IO.Path]::GetRandomFileName())
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function New-Fixture {
    param([string]$Dir)
    # TWO clips, because there are TWO samplers and the deadline had to be added to both.
    #
    # `seq`: 300 frames of 1080p, so the sequential sampler has real work to be cut short of.
    # `seek`: the by-seek path engages on `info.frames > 5000` (rd_video.cpp:1663) and
    # RD_PALETTE_SEEK can only DISABLE it, never force it. So this one is 5100 frames -- and
    # at 320x180 that is a few MB and a couple of seconds to make, which is the cheap way to
    # reach a path that otherwise needs a 2.8-minute 1080p clip. Testing only the sequential
    # arm would leave the six concurrent seek workers, which is where the deadline is most
    # likely to be wrong, completely unexercised.
    $seq = Join-Path $Dir 'seq.mp4'
    & ffmpeg -y -hide_banner -loglevel error -f lavfi -i 'testsrc=size=1920x1080:rate=30' `
        -frames:v 300 -pix_fmt yuv420p -c:v libx264 -g 30 $seq 2>&1 | Out-Null
    $seek = Join-Path $Dir 'seek.mp4'
    & ffmpeg -y -hide_banner -loglevel error -f lavfi -i 'testsrc=size=320x180:rate=30' `
        -frames:v 5100 -pix_fmt yuv420p -c:v libx264 -g 60 $seek 2>&1 | Out-Null
    if (-not (Test-Path -LiteralPath $seq)) { return $null }
    if (-not (Test-Path -LiteralPath $seek)) { return $null }
    return @{ seq = $seq; seek = $seek }
}

# Returns the number of DISTINCT samples the run reports, or $null on failure.
function Get-Sampled {
    param([string]$Clip, [string]$Out, [string]$DeadlineMs)
    $prev = $env:RD_PALETTE_DEADLINE_MS
    if ($DeadlineMs) { $env:RD_PALETTE_DEADLINE_MS = $DeadlineMs }
    else { Remove-Item Env:\RD_PALETTE_DEADLINE_MS -EA SilentlyContinue }
    try {
        $text = (& $Rdither --video --colors 16 --engine cuda --video-lossless `
                     --frames 60 $Clip $Out 2>&1 | Out-String)
    } finally {
        if ($null -eq $prev) { Remove-Item Env:\RD_PALETTE_DEADLINE_MS -EA SilentlyContinue }
        else { $env:RD_PALETTE_DEADLINE_MS = $prev }
    }
    if ($LASTEXITCODE -ne 0) { return [pscustomobject]@{ Rc = $LASTEXITCODE; Sampled = -1; Text = $text } }
    $m = [regex]::Match($text, 'palette\s*:\s*(\d+) colours from (\d+)')
    if (-not $m.Success) {
        $m = [regex]::Match($text, 'palette_sampled|(\d+) sampled frame')
    }
    $sampled = if ($m.Success) { [int]$m.Groups[$m.Groups.Count - 1].Value } else { -1 }
    return [pscustomobject]@{ Rc = 0; Sampled = $sampled; Text = $text }
}

try {
    Write-Host 'palette deadline: does --palette-budget-ms bound the stage?'

    $fx = New-Fixture -Dir $OutDir
    if (-not $fx) { Write-Host 'palette deadline: cannot run -- could not build a fixture'; exit 2 }

    $failed = $false
    foreach ($arm in @(@('sequential', $fx.seq), @('by-seek', $fx.seek))) {
        $label = $arm[0]
        $clip = $arm[1]

        # Unbounded, so we know the count the deadline has to beat.
        $free = Get-Sampled -Clip $clip -Out (Join-Path $OutDir "$label-free.mkv") -DeadlineMs $null
        if ($free.Rc -ne 0) {
            Write-Host "palette deadline: cannot run -- the unbounded $label render exited $($free.Rc)"
            exit 2
        }
        Write-Host ("  {0,-11} unbounded   : {1} sampled frame(s)" -f $label, $free.Sampled)

        # Deadline of 1 ms. It cannot fire before the first sample is placed, so the floor
        # is one sample rather than zero.
        $tight = Get-Sampled -Clip $clip -Out (Join-Path $OutDir "$label-tight.mkv") -DeadlineMs '1'
        if ($tight.Rc -ne 0) {
            Write-Host "  FAIL  the $label deadline-bounded render FAILED (exit $($tight.Rc))"
            Write-Host '        A deadline that breaks the render is worse than no deadline.'
            ($tight.Text -split "`n" | Where-Object { $_ -match '\S' } | Select-Object -First 3) |
                ForEach-Object { Write-Host "          $($_.Trim())" }
            $failed = $true
            continue
        }
        Write-Host ("  {0,-11} deadline 1ms : {1} sampled frame(s)" -f $label, $tight.Sampled)

        if ($tight.Sampled -lt 1) {
            Write-Host "  FAIL  the bounded $label run reported no palette at all."
            $failed = $true
            continue
        }
        if ($tight.Sampled -ge $free.Sampled) {
            Write-Host "  FAIL  a 1 ms deadline still placed $($tight.Sampled) $label samples,"
            Write-Host "        against $($free.Sampled) unbounded. The deadline did nothing, so"
            Write-Host '        --palette-budget-ms still does not bound the stage.'
            $failed = $true
            continue
        }
        Write-Host ("  {0,-11} ok           : bounded {1} -> {2}" -f $label, $free.Sampled, $tight.Sampled)
    }
    if ($failed) { exit 1 }
} finally {
    Remove-Item -LiteralPath $OutDir -Recurse -Force -EA SilentlyContinue
}

Write-Host 'palette deadline: ok'
exit 0