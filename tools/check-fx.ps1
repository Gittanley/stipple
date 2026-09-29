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

param(
  [string]$Dxc = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe',
  [int]$Width = 1920,
  [int]$Height = 1080
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$fx = Join-Path $root 'reshade\rdither_riemersma.fx'
$fxh = Join-Path $root 'reshade\ReShade.fxh'

if (-not (Test-Path $Dxc)) {
  Write-Host "[check-fx] SKIP: dxc.exe not found at $Dxc"
  exit 0
}
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
if ($raw -match '(?m)^\s*(VertexShader|PixelShader)\s*=\s*compile\b') {
  $line = ($raw.Substring(0, $Matches[0].Index) -split "`n").Count
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
foreach ($m in [regex]::Matches($raw, '(?m)^\s*sampler\s+(\w+)\s*(?::\s*\w+\s*)?;')) {
  $line = ($raw.Substring(0, $m.Index) -split "`n").Count
  $violations += "line ${line}: bare 'sampler $($m.Groups[1].Value);' -- use ReShade::BackBuffer from ReShade.fxh instead"
}

# 1d. The 2.x screen-size names.  ReShade 3.x calls these BUFFER_*.
foreach ($sym in @('BACKBUFFER_WIDTH', 'BACKBUFFER_HEIGHT')) {
  foreach ($m in [regex]::Matches($raw, "(?m)^(?!.*//).*\b$sym\b")) {
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
exit 0
