# Does RD_PALETTE_OVERLAP=0 produce pixel-identical output to the default?
#
# WHY THIS SHAPE.  The dangerous failure for an A/B of "same output, faster" is a flag
# that silently does nothing: the two arms become the same binary, the pixels match
# perfectly, and the probe reports PASS while proving nothing.  So this probe asserts the
# arms DIFFER IN TIMING first, and only then compares pixels.  A timing assertion is the
# cheapest possible proof that the flag reached the code.
#
# Pixel comparison is tools\probe-frame-diff.ps1, which samples six frames spread across the
# clip.  Spread, not the head: on the sample-truncation defect the divergence INCREASED along
# the clip (0.972% at frame 0, 1.092% from frame 359), so a head-only sample under-reports
# by 11% and a late-only fault would be invisible.
#
# USAGE
#   probe-palette-overlap.ps1 [-Frames 6] [-Rdither path] [-OutDir path]
#
# EXIT CODES
#   0  overlap and no-overlap produce identical pixels, and the arms measurably differ
#   1  the pixels differ, OR the arms did not differ in timing (flag did nothing), OR the
#      negative control failed to detect a known difference
#   2  cannot run -- no ffmpeg, no rdither, no fixture, or a short clip.  "Cannot run" is
#      deliberately not 0: a probe that cannot run must not be readable as a pass.

param(
    [int]$Frames = 6,
    [string]$Rdither = '',
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'

function Resolve-Tool {
    param([string]$Given, [string]$Name)
    if ($Given) {
        if (Test-Path -LiteralPath $Given) { return $Given }
        if (Get-Command $Given -ErrorAction SilentlyContinue) { return $Given }
        return $null
    }
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($hard in @("C:\progi\ffmpeg-master-latest-win64-gpl-shared\bin\$Name.exe")) {
        if (Test-Path -LiteralPath $hard) { return $hard }
    }
    return $null
}

$ffmpeg = Resolve-Tool -Given '' -Name 'ffmpeg'
if (-not $ffmpeg) { Write-Host 'palette overlap: cannot run -- no ffmpeg'; exit 2 }
$ffprobe = Resolve-Tool -Given '' -Name 'ffprobe'
if (-not $ffprobe) {
    $sib = Join-Path (Split-Path $ffmpeg -Parent) 'ffprobe.exe'
    if (Test-Path -LiteralPath $sib) { $ffprobe = $sib }
}
if (-not $ffprobe) { Write-Host 'palette overlap: cannot run -- no ffprobe'; exit 2 }

if (-not $Rdither) {
    $root = Split-Path $PSScriptRoot -Parent
    foreach ($cand in @('build\Release\rdither.exe', 'build\Debug\rdither.exe')) {
        $p = Join-Path $root $cand
        if (Test-Path -LiteralPath $p) { $Rdither = $p; break }
    }
}
if (-not $Rdither -or -not (Test-Path -LiteralPath $Rdither)) {
    Write-Host 'palette overlap: cannot run -- rdither.exe not found'
    exit 2
}

$diff = Join-Path $PSScriptRoot 'probe-frame-diff.ps1'
if (-not (Test-Path -LiteralPath $diff)) {
    Write-Host 'palette overlap: cannot run -- tools\probe-frame-diff.ps1 is missing'
    exit 2
}

if (-not $OutDir) {
    $OutDir = Join-Path ([IO.Path]::GetTempPath()) ('rd_overlap_' + [IO.Path]::GetRandomFileName())
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function New-Fixture {
    param([string]$Dir)
    # 1920x1080, and that is not a preference.  probe-frame-diff asserts that the decoded
    # output is a whole number of rgb48le frames against a HARD-CODED 1920x1080 frame, so a
    # smaller clip makes it exit 2 ("did not decode to exactly N whole frames") and the
    # pixel comparison silently cannot run.
    #
    # It is also the geometry the spec's numbers were measured on.  A 640x360 clip makes the
    # palette ~74% of the run and the reader trivial -- the opposite of the real ratio, and
    # the overlap's sign flips with it.
    $clip = Join-Path $Dir 'overlap.mp4'
    & $ffmpeg -y -hide_banner -loglevel error -f lavfi -i 'testsrc=size=1920x1080:rate=30' `
        -frames:v 300 -pix_fmt yuv420p -c:v libx264 -g 30 $clip 2>&1 | Out-Null
    if (-not (Test-Path -LiteralPath $clip)) { return $null }
    $n = & $ffprobe -v error -select_streams v:0 -count_frames `
        -show_entries stream=nb_read_frames -of 'default=nw=1:nk=1' $clip 2>$null
    return ,@($clip, [int]$n)
}

# Renders one arm and returns its wall time in ms, taken from rdither's OWN `frames :` line
# rather than a stopwatch, so the number is the one the program reports.
function Invoke-Arm {
    param([string]$Clip, [string]$Out, [bool]$NoOverlap, [int]$Colors = 16)
    $prev = $env:RD_PALETTE_OVERLAP
    if ($NoOverlap) { $env:RD_PALETTE_OVERLAP = '1' } else { Remove-Item Env:\RD_PALETTE_OVERLAP -EA SilentlyContinue }
    try {
        $text = (& $Rdither --video --colors $Colors --engine $script:Engine `
                     --video-lossless $Clip $Out 2>&1 | Out-String)
    } finally {
        if ($null -eq $prev) { Remove-Item Env:\RD_PALETTE_OVERLAP -EA SilentlyContinue }
        else { $env:RD_PALETTE_OVERLAP = $prev }
    }
    $code = $LASTEXITCODE
    if ($code -ne 0) { return $null }
    if ($text -notmatch 'frames\s*:\s*(\d+) in ([0-9.]+) s \(([0-9.]+) fps') { return $null }
    return [double]$Matches[3]
}

try {
    Write-Host 'palette overlap: same pixels with and without the overlap'

    $fx = New-Fixture -Dir $OutDir
    if (-not $fx) { Write-Host 'palette overlap: cannot run -- could not build a fixture'; exit 2 }
    $clip = $fx[0]
    Write-Host "  fixture: $($fx[1]) frames of 1920x1080"
    if ($fx[1] -lt 120) {
        Write-Host "palette overlap: cannot run -- only $($fx[1]) frames; need 120+"
        exit 2
    }

    # Engine chosen BY PROBE, not by assumption.  This used to hardcode cuda, so on a
    # machine without it every render fails, all four arms come back null, and the probe
    # exits 2 -- which the gate reports as SKIPPED.  That reads as "nothing to check here"
    # rather than "the engine this probe demanded does not exist", which is the same
    # cannot-run-reported-as-covered trap that the coverage line already exists to catch.
    # One two-frame render settles it.
    $script:Engine = $null
    foreach ($candidate in @('cuda', 'cpu')) {
        $probeOut = Join-Path $OutDir 'engine-probe.mkv'
        & $Rdither --video --colors 16 --engine $candidate --video-lossless `
                   --frames 2 $clip $probeOut *> $null
        if ($LASTEXITCODE -eq 0) { $script:Engine = $candidate; break }
    }
    if (-not $script:Engine) {
        Write-Host 'palette overlap: cannot run -- no usable engine (tried cuda, then cpu)'
        exit 2
    }
    Write-Host "  engine     : $script:Engine"

    # --- 1. TIMING FIRST -------------------------------------------------------
    # Interleaved, so a drift in machine speed cannot masquerade as the flag working.
    $onTimes = @(); $offTimes = @()
    for ($rep = 0; $rep -lt 2; $rep++) {
        $onTimes  += Invoke-Arm -Clip $clip -Out (Join-Path $OutDir "on_$rep.mkv")  -NoOverlap $false
        $offTimes += Invoke-Arm -Clip $clip -Out (Join-Path $OutDir "off_$rep.mkv") -NoOverlap $true
    }
    # Inverted logic caught here: Where-Object returns NOTHING when every arm succeeded, and
    # ($null -eq $nothing) is true, so the original test exited 2 -- on success.
    $missing = @($onTimes + $offTimes | Where-Object { $null -eq $_ })
    if ($missing.Count -gt 0) {
        Write-Host "palette overlap: cannot run -- $($missing.Count) of 4 renders produced no parsable timing line"
        exit 2
    }
    $onMean = ($onTimes  | Measure-Object -Average).Average
    $offMean = ($offTimes | Measure-Object -Average).Average
    Write-Host ("  overlap ON  : {0:N1} fps  ({1:N1}, {2:N1})" -f $onMean, $onTimes[0], $onTimes[1])
    Write-Host ("  overlap OFF : {0:N1} fps  ({1:N1}, {2:N1})" -f $offMean, $offTimes[0], $offTimes[1])

    $gain = if ($offMean -gt 0) { $onMean / $offMean } else { 0 }
    Write-Host ("  ratio       : {0:N3}x  (overlap / default)" -f $gain)
    if ([Math]::Abs($gain - 1.0) -lt 0.02) {
        Write-Host 'FAIL: the two arms are within 2% -- RD_PALETTE_OVERLAP changed nothing,'
        Write-Host '      so a pixel comparison between them would prove nothing.'
        exit 1
    }
    # Reported, NOT gated -- and deliberately printed with no verdict attached.
    #
    # This ratio is NOT a stable direction.  An earlier run of this probe reported 1.452x
    # (the overlap "winning") while measurements taken minutes later on a real clip gave
    # 0.78x and a second clip gave 0.66x, and every one of those runs was slower on both
    # arms than its neighbours.  Two reps of a contended render is not enough to resolve a
    # 20% effect on a machine that drifts 10-25% between identical binaries.
    #
    # So the direction is NOT gated and this probe does NOT claim one.  What it does assert,
    # and what gates, is: identical pixels, and arms that differ by enough to prove the flag
    # reached the code at all.  The measured direction -- slower, on both clips, because the
    # palette thread takes cores from the dither workers -- is recorded in
    # docs\KNOWN-ISSUES.md item 4.  Do not "fix" this by asserting $gain -lt 1; that would
    # encode a noise reading as a contract, and the next machine to run it would fail the gate.
    Write-Host '  (the ratio above is NOT a stable direction -- see KNOWN-ISSUES.md item 4;' 
    Write-Host '   this probe gates pixel-identity and flag-reachability, not performance)'

    # --- 2. PIXELS -------------------------------------------------------------
    $a = Join-Path $OutDir 'on_0.mkv'
    $b = Join-Path $OutDir 'off_0.mkv'
    & pwsh -NoProfile -File $diff -A $a -B $b -Labels 'overlap-on','overlap-off' -Frames $Frames
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: the overlap changes the pixels -- that is the whole contract broken'
        exit 1
    }

    # --- 3. NEGATIVE CONTROL ---------------------------------------------------
    # A comparison that has never been seen failing cannot tell "identical" from "not
    # looking".  --colors 32 is a known-different render of the same clip.
    $c = Invoke-Arm -Clip $clip -Out (Join-Path $OutDir 'ctl32.mkv') -NoOverlap $false -Colors 32
    if ($null -eq $c) { Write-Host 'palette overlap: cannot run -- the negative control did not render'; exit 2 }
    & pwsh -NoProfile -File $diff -A $a -B (Join-Path $OutDir 'ctl32.mkv') `
        -Labels 'colors-16','colors-32' -Frames $Frames | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host 'FAIL: the negative control reported IDENTICAL for --colors 16 vs 32.'
        Write-Host '      The comparison is not detecting differences, so step 2 proves nothing.'
        exit 1
    }
    Write-Host '  negative control: --colors 32 correctly reported as different'
} finally {
    Remove-Item -LiteralPath $OutDir -Recurse -Force -EA SilentlyContinue
}

Write-Host 'palette overlap: ok'
exit 0