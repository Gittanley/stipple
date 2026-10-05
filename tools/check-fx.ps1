# check-fx.ps1 -- compile-check the Reshade effect.
#
# WHY THIS EXISTS, AND WHY IT IS MORE THAN "RUN A HLSL COMPILER".
#
# A .fx that does not compile is a .fx that looks done, and the effect is not part
# of the CMake build, so nothing else would ever notice.  An earlier version of
# this checker compiled the shader with dxc, it passed, the shader was shipped --
# and it produced SEVEN errors on a real Reshade install.  A HLSL compile is
# necessary and is nowhere near sufficient, because Reshade parses the file with
# its own preprocessor and effect parser and enforces conventions no HLSL
# compiler has an opinion about.
#
# So this does three things, in order:
#
#   1. Pattern-checks the Reshade conventions dxc structurally cannot catch:
#      semicolon-separated uniform annotations, no `compile`, no bare sampler,
#      no 2.x screen-size names.  Each of these was a real error.
#   2. Compiles the shader against the REAL ReShade.fxh, with the real
#      __RESHADE__ / BUFFER_* defines it expects.  This validates the actual
#      ReShade API usage -- ReShade::BackBuffer, PostProcessVS -- rather than a
#      shim, so a wrong Reshade call is caught here instead of on a user's PC.
#   3. Still strips the two things dxc cannot parse at all: the `technique`
#      block and the `ui_` annotations.  Both are Reshade effect syntax that
#      dxc rejects outright, and both are checked by (1) instead.
#
# EXIT CODES
#   0  both stages ran
#   1  a violation, a missing file, or the shader did not compile
#   2  no dxc.exe -- stage 1 ran and passed, stage 2 is UNVERIFIED.  This used
#      to be 0, which is the one thing a "cannot run" must never be.
#
# NOTHING INVOKES THIS SCRIPT
# ==========================
# It is not in verify.ps1, not in .github\workflows\ci.yml, and not in any hook.
# A green suite therefore says NOTHING about reshade\rdither_riemersma.fx, and
# the paragraph above is the only thing suggesting otherwise.
#
# That is left as it is rather than quietly fixed, because wiring it in has a
# cost that has to be paid deliberately: every CI run would come to depend on a
# Windows SDK being present, and on a runner without one this file now exits 2
# and turns a green pipeline red over a file that is not part of the build.  If
# you wire it in, gate the CI step on a dxc-presence check and read 2 as
# "unverified here", not as a failure.  A green pipeline that had silently
# skipped this file would be exactly the problem the exit codes above exist to
# prevent.

param(
  # Empty means "search for it" -- see Find-Dxc below.  This used to be a hardcoded
  # absolute path to one Windows SDK version:
  #   C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe
  # A machine with any other SDK has no dxc there, so stage 2 became a silent no-op;
  # and installing a NEWER SDK never changed it, so this compiled against whatever
  # that one pinned version shipped, indefinitely.  A named path that does not exist
  # is now an error rather than a skip: a caller who asked for a compiler and got
  # nothing has been told the wrong thing.
  [string]$Dxc = '',
  [int]$Width = 1920,
  [int]$Height = 1080
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$fx = Join-Path $root 'reshade\rdither_riemersma.fx'
$fxh = Join-Path $root 'reshade\ReShade.fxh'

if (-not (Test-Path $fx)) { Write-Host "[check-fx] FAIL: $fx not found"; exit 1 }
if (-not (Test-Path $fxh)) {
  Write-Host "[check-fx] FAIL: $fxh not found."
  Write-Host '          It is vendored (SPDX CC0-1.0) from a Reshade 3.x install and'
  Write-Host '          is what makes stage 2 real rather than a shim.'
  exit 1
}

$raw = Get-Content -Raw -LiteralPath $fx

# ---- 1. Reshade conventions dxc cannot catch ------------------------------
$violations = @()

# 1a. Uniform annotations are semicolon-separated.  A comma gives
#     "syntax error: unexpected ','" on every annotated uniform.
foreach ($m in [regex]::Matches($raw, '(?s)<[^<>]*ui_type[^<>]*>')) {
  if ($m.Value -match ',\s*[A-Za-z_]+\s*=') {
    $line = ($raw.Substring(0, $m.Index) -split "`n").Count
    $violations += "line ${line}: comma inside a uniform annotation -- Reshade separates these with SEMICOLONS"
  }
}

# 1b. `compile` is fxc effect syntax; Reshade's parser rejects it outright.
#
# This rule used `-match` plus `$Matches[0].Index`.  $Matches holds the matched TEXT,
# not a Match object, so `.Index` on a string is $null, `Substring(0, $null)` becomes
# `Substring(0, 0)`, and "" splits into exactly one element -- so this rule reported
# "line 1" for every violation it has ever found, in any file.  The other three rules
# below read `$m.Index` off a real Match from [regex]::Matches, which is why only this
# one was wrong: one rule reaching for the offset by a different mechanism.
#
# Measured: planting `PixelShader = compile ps_RDither();` on line 426 of a 430-line
# shader printed "line 1: 'compile' in the technique block".  A defect report that
# names the wrong line is a false claim about the reader's own file, and it was false
# in every run.  [ \t]*, not \s*: .NET's \s spans newlines, so `\s*` after `^` can
# consume whole lines and move the match start; harmless once the offset comes from a
# Match, but wrong, and it would mislead the next edit here.
foreach ($m in [regex]::Matches($raw, '(?m)^[ \t]*(VertexShader|PixelShader)[ \t]*=[ \t]*compile\b')) {
  $line = ($raw.Substring(0, $m.Index) -split "`n").Count
  $violations += "line ${line}: 'compile' in the technique block -- fxc syntax, rejected by Reshade; name the entry point directly"
}

# 1c. A locally declared sampler must carry an EQUALS:  { Texture = texLUT; }
#     Both `Texture X;` (no equals) and the bare `sampler X : SAMPLER0;` were
#     real errors on this install.  NEITHER FORM APPEARS IN THE SHADER ANY MORE --
#     it takes ReShade::BackBuffer from ReShade.fxh and declares no sampler at all
#     -- so these two rules now guard nothing that is present.  They are kept
#     deliberately: each is a trap this shader fell into and then removed, and a
#     rule with no current instance is what a guard is for.  If a sampler is ever
#     added back, the correct form is `sampler X { Texture = texLUT; }` and these
#     will say so instead of letting Reshade's parser say it as an X3000.
foreach ($m in [regex]::Matches($raw, '(?s)\{[^}]*\bTexture\b\s+[^;=]+;')) {
  $line = ($raw.Substring(0, $m.Index) -split "`n").Count
  $violations += "line ${line}: '{ Texture X; }' needs an EQUALS -- the form this install uses is '{ Texture = texLUT; }'"
}
foreach ($m in [regex]::Matches($raw, '(?m)^[ \t]*sampler[ \t]+(\w+)[ \t]*(?::[ \t]*\w+[ \t]*)?;')) {
  $line = ($raw.Substring(0, $m.Index) -split "`n").Count
  $violations += "line ${line}: bare 'sampler $($m.Groups[1].Value);' -- use ReShade::BackBuffer from ReShade.fxh instead"
}

# 1d. The 2.x screen-size names.  ReShade 3.x calls these BUFFER_*.
#
# The guard was `(?!.*//)`, which asserts that the symbol is not preceded by `//`
# ANYWHERE ON THE LINE -- including in a trailing comment.  That is much broader than
# the intent, and it disables the rule for real code.  Measured on this file with
# `float w2 = BACKBUFFER_WIDTH; // 2.x spelling` inserted before the technique:
#
#   (?m)^(?!.*//).*BACKBUFFER_WIDTH\b   matches 0 lines
#   (?m)^.*BACKBUFFER_WIDTH\b          matches 2 lines
#
# The rule below skips only lines that ARE comments -- leading `//` or `/*` -- which is
# what it has to do, because this shader's own header documents all four traps in `//`
# comments and quotes the forbidden spellings verbatim.  Verified by planting each
# violation: the four rules fire on a planted `compile`, a braced sampler without an
# equals, a bare `sampler X : SAMPLER0;`, and `BACKBUFFER_WIDTH` both on its own line
# and after code on a line with a trailing comment.
#
# The old guard was not merely redundant.  It is why the trailing-comment case reached
# dxc at all: on a machine with a Windows SDK the shader still fails to compile ("use
# of undeclared identifier 'BACKBUFFER_WIDTH'", exit 1), so the defect was caught by
# stage 2 -- and stage 2 is skipped, with exit 2, on every machine without one, where
# the violation would then ship.  A rule that only works on the machines that happen
# to have a compiler is not a rule.
foreach ($sym in @('BACKBUFFER_WIDTH', 'BACKBUFFER_HEIGHT')) {
  foreach ($m in [regex]::Matches($raw, "(?m)^[ \t]*(?!//)(?!/\*)(?!.*\*\/).*\b$sym\b")) {
    $line = ($raw.Substring(0, $m.Index) -split "`n").Count
    $violations += "line ${line}: '$sym' is the 2.x spelling; ReShade 3.x provides BUFFER_WIDTH / BUFFER_HEIGHT"
  }
}

if ($violations.Count -gt 0) {
  Write-Host '[check-fx] FAIL: Reshade convention violations dxc cannot catch:'
  $violations | ForEach-Object { Write-Host "    $_" }
  Write-Host '          These produced real errors on a Reshade install while dxc passed the file.'
  exit 1
}
Write-Host '[check-fx] Reshade conventions OK (annotations, no compile, no bare sampler, no 2.x names).'
Write-Host '          This half needs no compiler, so it ran whether or not a dxc was found.'

# ---- locate dxc, now that stage 1 has run ----------------------------------
#
# Stage 1 deliberately runs FIRST.  The absent-compiler check used to sit above it and
# exit the whole file, so on a machine with no Windows SDK the convention checks -- the
# ones that catch the very errors dxc passed -- did not run either.  A missing compiler
# should cost exactly the coverage that depends on it.
#
# Newest SDK first, not a pinned one.  The path this replaces was pinned to
# 10.0.26100.0, which is a silent no-op on every machine without that exact SDK and a
# stale compiler on every machine that has since installed a newer one.
function Find-Dxc {
  param([string]$Explicit)

  if ($Explicit) {
    if (-not (Test-Path $Explicit)) {
      throw "-Dxc was given as '$Explicit' and no such file exists.  A named compiler that is not there is a mistake in the caller, not a reason to skip."
    }
    return [pscustomobject]@{ Path = (Resolve-Path $Explicit).Path; Source = '-Dxc' }
  }
  $onPath = Get-Command dxc -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($onPath) { return [pscustomobject]@{ Path = $onPath.Source; Source = 'PATH' } }

  $roots = @()
  foreach ($pf in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
    if ($pf) { $roots += (Join-Path $pf 'Windows Kits\10\bin') }
  }
  $roots += 'C:\Program Files (x86)\Windows Kits\10\bin'
  $roots += 'C:\Program Files\Windows Kits\10\bin'
  $found = @()
  foreach ($r in @($roots | Where-Object { $_ -and (Test-Path $_) } | Sort-Object -Unique)) {
    $found += @(Get-ChildItem -Path $r -Recurse -Filter 'dxc.exe' -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Directory.Name -eq 'x64' })
  }
  if ($found.Count -gt 0) {
    $best = $found | Sort-Object `
      @{ Expression = { if ($_.FullName -match 'bin\\([\d.]+)\\') { [version]$Matches[1] } else { [version]'0.0' } }; Descending = $true },
      @{ Expression = { $_.FullName }; Descending = $false } | Select-Object -First 1
    return [pscustomobject]@{ Path = $best.FullName; Source = 'Windows Kits search (newest x64)' }
  }
  return $null
}

$dxcInfo = $null
try { $dxcInfo = Find-Dxc -Explicit $Dxc } catch { Write-Host "[check-fx] FAIL: $($_.Exception.Message)"; exit 1 }
if (-not $dxcInfo) {
  Write-Host ''
  Write-Host '[check-fx] SKIP: no dxc.exe found.  Stage 1 above DID run; stage 2 did not.' -ForegroundColor Yellow
  Write-Host '          Searched: PATH, then every x64\dxc.exe under' -ForegroundColor Yellow
  Write-Host '            %ProgramFiles(x86)%\Windows Kits\10\bin\*\x64' -ForegroundColor Yellow
  Write-Host '            %ProgramFiles%\Windows Kits\10\bin\*\x64' -ForegroundColor Yellow
  Write-Host '          The HLSL compile against the real ReShade headers is therefore' -ForegroundColor Yellow
  Write-Host '          UNVERIFIED.  Exiting 2, "cannot run", not 0: this used to exit 0' -ForegroundColor Yellow
  Write-Host '          here, which is the inverse of the convention the rest of the suite' -ForegroundColor Yellow
  Write-Host '          uses and reads in a log exactly like a pass.' -ForegroundColor Yellow
  exit 2
}
$Dxc = $dxcInfo.Path
Write-Host "[check-fx] stage 2: dxc via $($dxcInfo.Source)"
Write-Host "          $Dxc"
try {
  $ver = (& $Dxc --version 2>&1 | Select-Object -First 1)
  if ($ver) { Write-Host "          $ver" }
} catch { Write-Host '          (dxc --version produced no output; continuing)' }

# ---- 2. compile against the real ReShade headers --------------------------
$tmp = Join-Path $env:TEMP 'rdither_fx_check'
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp | Out-Null

# Stage a dxc-readable copy of ReShade's real header.  ReShade::BackBuffer and
# friends are declared with ReShade's own `sampler` type, which dxc cannot parse
# -- but the NAMES are the thing worth validating, so they are kept and only the
# underlying objects are swapped for ones dxc understands.  A shader that
# misspelled ReShade::BackBuffer would still fail here; one that merely uses a
# legacy intrinsic would not, which is the right way round.
$h = Get-Content -Raw -LiteralPath $fxh
$h = $h -replace '(?s)texture BackBufferTex : COLOR;.*?sampler DepthBuffer \{ Texture = DepthBufferTx \};', 'STUB'
$h = $h -replace '(?s)texture BackBufferTex : COLOR;.*?sampler DepthBuffer \{ Texture = DepthBufferTex; \};', @'
	// dxc-readable stand-ins for ReShade's own sampler type.  Named BackBuffer /
	// DepthBuffer because that is what the rest of this header, and the shader,
	// refer to -- the *Tex names are ReShade effect syntax and mean nothing here.
	// The SamplerState is deliberately NOT declared here: this block sits inside
	// `namespace ReShade`, so it would be ReShade::_rd_ss and unreachable from the
	// shim macros below, which expand in the shader's own (global) scope.
	Texture2D<float4> BackBuffer : register(t0);
	Texture2D<float4> DepthBuffer : register(t1);
'@
Set-Content -LiteralPath (Join-Path $tmp 'ReShade.fxh') -Value $h -Encoding utf8

# tex2D / tex2Dlod are SM3-SM5 spellings.  dxc emits DXIL only and rejects every
# legacy intrinsic, so both map onto the modern form.  Third such limitation,
# after the technique block and the ui_ annotations, and the only one that
# touches the shader's own body.  _rd_ss must be at global scope for the reason
# given above.
$shim = @'
SamplerState _rd_ss : register(s0);
#define tex2Dlod(s, c) ((s).Sample(_rd_ss, (c).xy))
#define tex2D(s, c)    ((s).Sample(_rd_ss, (c).xy))
'@

$s = $raw
$s = ($s -split '(?m)^technique ')[0]                 # dxc cannot parse the technique block
$s = $s -replace '<\s*\r?\n\s*ui_type[^>]*>', ''      # nor the ui_ annotations
foreach ($u in @('numColors', 'paletteBits', 'outputBits')) {
  $s = $s -replace "(?m)^uniform int\s+$u.*$", "uniform int $u;"
}
$s = $s -replace '(?m)^uniform float\s+strength.*$', 'uniform float strength;'
$s = $s -replace '(?m)^(\s*)#include "ReShade.fxh"', ('$1' + $shim + "`n" + '$1#include "ReShade.fxh"')
$body = Join-Path $tmp 'body.hlsl'
Set-Content -LiteralPath $body -Value $s -Encoding utf8

function Invoke-Dxc([string]$Profile, [string]$Entry, [string]$Out) {
  $log = Join-Path $tmp "$Entry.log"
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    # The defines ReShade's own headers expect.  __RESHADE__ gates the version
    # check inside ReShade.fxh ("ReShade 3.0+ is required"), and the BUFFER_*
    # group is what ReShade.fxh itself uses for AspectRatio and PixelSize --
    # supplying them here is also what proved the shader's own screen-size name
    # was wrong.
    & $Dxc -T $Profile -E $Entry -I $tmp -D '__RESHADE__=30000' `
           -D "BUFFER_WIDTH=$Width" -D "BUFFER_HEIGHT=$Height" `
           -D "BUFFER_RCP_WIDTH=$(1.0 / $Width)" -D "BUFFER_RCP_HEIGHT=$(1.0 / $Height)" `
           -Fo (Join-Path $tmp $Out) $body 2>&1 | Out-File -FilePath $log -Encoding utf8
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $prev
  }
  # dxc's diagnostic shape is  path:line:col: error: message  .  Matching exactly
  # that stops the effect-syntax WARNING from being read as a failure.
  $real = @(Get-Content $log -ErrorAction SilentlyContinue |
            Where-Object { $_ -match ':\d+:\d+:\s*error:' })
  if ($code -ne 0 -or $real.Count -gt 0) {
    Write-Host "[check-fx] FAIL: $Entry ($Profile) did not compile (exit $code)"
    $real | Select-Object -First 15 | ForEach-Object { Write-Host "    $_" }
    return $false
  }
  return $true
}

# The pixel shader is the one that matters; it references ReShade::BackBuffer and
# so is a real test of the Reshade API rather than of plain HLSL.
$okPs = Invoke-Dxc 'ps_6_0' 'PS_RDither' 'ps.dxil'
if (-not $okPs) { exit 1 }

Write-Host '[check-fx] OK: the shader compiles against the real ReShade 3.x headers.'
Write-Host '          ReShade::BackBuffer usage validated. The technique block and the'
Write-Host '          ui_ annotations are still checked by pattern only (stage 1).'
Write-Host '          NOTE: nothing in this repository runs this script, so a green'
Write-Host '          verify.ps1 says nothing about this file.  See the header.'
exit 0
