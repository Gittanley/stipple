# Does SourcePixFmtHasAlpha (rd_video.cpp) actually take the 4-channel path on an alpha
# source, and the 3-channel path on an opaque one?
#
# WHY THIS IS A SEPARATE PROBE.  The first version of this check lived inside
# make-video-fixtures.ps1, which is the wrong home and was silently doing nothing.
# verify.ps1:301 only invokes that generator when tests\clip1920.mp4 or clip1920_audio.mkv
# is MISSING; both are committed, so on any normal checkout the generator never runs and
# every assertion inside it is dead.  The suite came back EXIT=0 with the check never
# having executed -- the exact "green means nothing" failure this project has hit before.
# A check that only runs on some machines is not a check.
#
# And the parser made it worse: verify.ps1:311 matches '^(CLIP|CLIP_AUDIO)=', so even when
# the generator did run, CLIP_ALPHA= would have been printed and ignored.
#
# So this builds its own 16x16 clip (a few ms), traces rdither's actual palette-decoder
# spawn, and asserts the format requested.  It needs no 1080p fixture, no engine parity
# and no GPU, so it can run on every build including a GPU-less CI runner -- the same
# reasoning probe-video-invariance.ps1 gives for existing on the host path.
#
# WHAT IT CHECKS
#   A. an alpha source (yuva420p) must NOT get `-pix_fmt rgb24`
#   B. an opaque source (yuv420p) MUST get `-pix_fmt rgb24`
#
# Both directions matter.  A guard proven only in one direction is half a guard: the deny
# branch is what protects transparency, and the allow branch is what makes the 530 MB
# saving real.
#
# EXIT CODES
#   0  the guard behaved correctly in both directions
#   1  it did not -- the requested format is named
#   2  cannot run: no ffmpeg, no rdither, or the fixture would not build.  "Cannot run"
#      is not 0, for the reason in the header.
#
# WHAT IT DELIBERATELY DOES NOT DO
#   It does not check that alpha SURVIVES the render.  rdither's output is yuv444p and
#   carries no alpha (see the run log: "yuv444p" out of a yuva420p in), which is a separate
#   question and a separate decision.  This probe asks only whether the guard picks the
#   right channel count, which is the thing the 9df7e34 commit could have got wrong.

param(
    [string]$Rdither = '',
    [string]$Ffmpeg = ''
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

$ffmpeg = Resolve-Tool -Given $Ffmpeg -Name 'ffmpeg'
if (-not $ffmpeg) { Write-Host 'alpha guard: cannot run -- no ffmpeg'; exit 2 }
$ffprobe = Resolve-Tool -Given '' -Name 'ffprobe'
if (-not $ffprobe) {
    $sib = Join-Path (Split-Path $ffmpeg -Parent) 'ffprobe.exe'
    if (Test-Path -LiteralPath $sib) { $ffprobe = $sib }
}
if (-not $ffprobe) { Write-Host 'alpha guard: cannot run -- no ffprobe'; exit 2 }

if (-not $Rdither) {
    # $PSScriptRoot is <repo>\tools, so ONE Split-Path is the repo root.  Splitting twice
    # lands above it and makes every build look unbuilt.
    $root = Split-Path $PSScriptRoot -Parent
    foreach ($cand in @('build\Release\rdither.exe', 'build\Debug\rdither.exe', 'build\rdither.exe')) {
        $p = Join-Path $root $cand
        if (Test-Path -LiteralPath $p) { $Rdither = $p; break }
    }
}
if (-not $Rdither -or -not (Test-Path -LiteralPath $Rdither)) {
    Write-Host 'alpha guard: cannot run -- rdither.exe not found'
    exit 2
}

$work = Join-Path ([IO.Path]::GetTempPath()) ('rd_alpha_guard_' + [IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Force -Path $work | Out-Null
$fail = 0

function New-Clip {
    param([string]$Name, [string]$PixFmt)
    $out = Join-Path $work $Name
    & $ffmpeg -y -hide_banner -loglevel error -f lavfi -i 'testsrc=size=160x120:rate=10' `
        -frames:v 4 -pix_fmt $PixFmt -c:v ffv1 $out 2>&1 | Out-Null
    if (-not (Test-Path -LiteralPath $out)) { return $null }
    $fmt = (& $ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt `
               -of 'default=nw=1:nk=1' $out 2>$null | Out-String).Trim()
    return ,@($out, $fmt)
}

# What format did rdither's palette decoder actually ask ffmpeg for?
function Get-RequestedPixFmt {
    param([string]$Clip)
    $prev = $env:RD_TRACE
    $env:RD_TRACE = '1'
    $text = (& $Rdither --video --colors 16 --engine cuda --palette-only `
                 $Clip (Join-Path $work 'probe_out.mkv') 2>&1 | Out-String)
    if ($null -eq $prev) { Remove-Item Env:\RD_TRACE -EA SilentlyContinue } else { $env:RD_TRACE = $prev }
    if ($text -match '\[spawn:palette-frames\].*?-pix_fmt\s+(\S+)') { return $Matches[1].Trim() }
    return $null
}

try {
    Write-Host 'alpha guard: SourcePixFmtHasAlpha, both directions'

    # --- B first: the opaque path.  If this cannot be built or traced there is no point
    # --- continuing, because A alone would be a one-sided check.
    $opaque = New-Clip -Name 'opaque.mkv' -PixFmt 'yuv420p'
    if (-not $opaque) { Write-Host 'alpha guard: cannot run -- could not build the opaque fixture'; exit 2 }
    Write-Host "  opaque fixture: pix_fmt=$($opaque[1])"
    $askedOpaque = Get-RequestedPixFmt -Clip $opaque[0]
    if (-not $askedOpaque) {
        Write-Host 'alpha guard: cannot run -- no palette-frames spawn was traced on the opaque clip'
        exit 2
    }
    Write-Host "  opaque source requested: -pix_fmt $askedOpaque"
    if ($askedOpaque -eq 'rgb24') {
        Write-Host '  ok: opaque source takes the 3-channel path'
    } else {
        Write-Host "  FAIL: opaque source asked for '$askedOpaque', expected rgb24 -- the 3-channel path is dead"
        $fail++
    }

    # --- A: the alpha path.  This is the one that protects transparency.
    $alpha = New-Clip -Name 'alpha.mkv' -PixFmt 'yuva420p'
    if (-not $alpha) { Write-Host 'alpha guard: cannot run -- could not build the alpha fixture'; exit 2 }
    Write-Host "  alpha fixture : pix_fmt=$($alpha[1])"
    if ($alpha[1] -notlike 'yuva*') {
        Write-Host "  FAIL: the alpha fixture's pix_fmt is '$($alpha[1])', not a yuva* format -- it carries no alpha, so the deny branch cannot be exercised"
        $fail++
    } else {
        $askedAlpha = Get-RequestedPixFmt -Clip $alpha[0]
        if (-not $askedAlpha) {
            Write-Host 'alpha guard: cannot run -- no palette-frames spawn was traced on the alpha clip'
            exit 2
        }
        Write-Host "  alpha source requested: -pix_fmt $askedAlpha"
        if ($askedAlpha -eq 'rgb24') {
            Write-Host "  FAIL: alpha source asked for rgb24 -- SourcePixFmtHasAlpha returned false and real transparency would be flattened to a constant"
            $fail++
        } elseif ($askedAlpha -eq 'rgba') {
            Write-Host '  ok: alpha source keeps the 4-channel path'
        } else {
            Write-Host "  NOTE: alpha source asked for '$askedAlpha' -- neither rgb24 nor rgba, so the guard is not doing what it claims"
            $fail++
        }
    }
} finally {
    Remove-Item -LiteralPath $work -Recurse -Force -EA SilentlyContinue
}

if ($fail -eq 0) { Write-Host 'alpha guard: ok'; exit 0 }
Write-Host "alpha guard: $fail FAILED"
exit 1