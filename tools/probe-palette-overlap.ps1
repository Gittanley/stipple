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
# Returns the run's OWN stdout, not a stopwatch reading and not a bare fps. The caller needs
# the whole text because the reachability assertion reads a line out of it, and a stopwatch
# cannot tell a flag that worked from a machine that happened to be quiet.
function Invoke-Arm {
    # The parameter is named for what it DOES, not for the flag's polarity.  It was
    # `$NoOverlap` while the body already set RD_PALETTE_OVERLAP=1, so `-NoOverlap $false`
    # meant "turn it ON" while reading as "turn it off" -- and it did turn it off, because
    # the body branches on the value.  The two arms ran swapped and the probe compared the
    # default against itself.  The timing assertion never noticed; the marker did, in ~4 s.
    param([string]$Clip, [string]$Out, [bool]$Overlap, [int]$Colors = 16)
    $prev = $env:RD_PALETTE_OVERLAP
    if ($Overlap) { $env:RD_PALETTE_OVERLAP = '1' } else { Remove-Item Env:\RD_PALETTE_OVERLAP -EA SilentlyContinue }
    try {
        $text = (& $Rdither --video --colors $Colors --engine $script:Engine `
                     --video-lossless $Clip $Out 2>&1 | Out-String)
    } finally {
        if ($null -eq $prev) { Remove-Item Env:\RD_PALETTE_OVERLAP -EA SilentlyContinue }
        else { $env:RD_PALETTE_OVERLAP = $prev }
    }
    if ($LASTEXITCODE -ne 0) { return $null }
    if ($text -notmatch 'frames\s*:') { return $null }
    return $text
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

    # --- 1. REACHABILITY, DETERMINISTICALLY -------------------------------------
    # This used to be a TIMING assertion: render both arms twice, require their wall times to
    # differ by more than 2%.  That is not a usable gate on this machine and never was, and
    # the reason is arithmetic rather than bad luck.  Run-to-run drift here is 10-25%; this
    # feature's effect is 22-34%.  Those ranges overlap, so a single A/B pair carries no
    # information about whether the flag did anything.  Three runs of this probe duly
    # disagreed: 0.78x, 1.45x, 0.66x.
    #
    # So the flag now announces itself on stdout ("built concurrently with the reader"), and
    # that line is the assertion.  It costs nothing, it cannot be produced by a busy machine,
    # and it cannot pass by accident.  The second rep went with the timing check, which halves
    # the probe's render cost -- the reps existed only to average a number nothing gates on.
    $onText  = Invoke-Arm -Clip $clip -Out (Join-Path $OutDir 'on.mkv')  -Overlap $true
    $offText = Invoke-Arm -Clip $clip -Out (Join-Path $OutDir 'off.mkv') -Overlap $false
    foreach ($pair in @(@('on', $onText), @('off', $offText))) {
        if ($null -eq $pair[1]) {
            Write-Host "palette overlap: cannot run -- the $($pair[0]) arm produced no output"
            exit 2
        }
    }
    $marker = 'built concurrently with the reader'
    if ($onText -notmatch [regex]::Escape($marker)) {
        Write-Host 'FAIL: RD_PALETTE_OVERLAP=1 did NOT take the concurrent path.'
        Write-Host '      Without this, a flag that silently stopped working would make the'
        Write-Host '      pixel comparison below compare one binary against itself and pass.'
        exit 1
    }
    if ($offText -match [regex]::Escape($marker)) {
        Write-Host 'FAIL: the default arm took the concurrent path. The flag is not opt-in,'
        Write-Host '      which means every render pays for it and the gate proves nothing.'
        exit 1
    }
    Write-Host "  reachability: marker present with =1, absent by default  (deterministic)"

    # Timings are still printed, because they are the evidence for KNOWN-ISSUES item 4, but
    # they are NOT asserted on and carry no verdict: the drift above is larger than the
    # effect.  Do not "fix" this by asserting $onFps -lt $offFps; that would encode noise as a
    # contract and the gate would fail on whichever machine happened to be busiest.
    $onFps  = if ($onText  -match '\(([0-9.]+) fps') { [double]$Matches[1] } else { 0 }
    $offFps = if ($offText -match '\(([0-9.]+) fps') { [double]$Matches[1] } else { 0 }
    Write-Host ("  observed    : overlap {0:N1} fps vs default {1:N1} fps -- NOT ASSERTED" -f $onFps, $offFps)

    # --- 2. PIXELS -------------------------------------------------------------
    $a = Join-Path $OutDir 'on.mkv'
    $b = Join-Path $OutDir 'off.mkv'
    & pwsh -NoProfile -File $diff -A $a -B $b -Labels 'overlap-on','overlap-off' -Frames $Frames
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: the overlap changes the pixels -- that is the whole contract broken'
        exit 1
    }

    # --- 3. NEGATIVE CONTROL ---------------------------------------------------
    # A comparison that has never been seen failing cannot tell "identical" from "not
    # looking".  --colors 32 is a known-different render of the same clip.
    $c = Invoke-Arm -Clip $clip -Out (Join-Path $OutDir 'ctl32.mkv') -Overlap $false -Colors 32
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