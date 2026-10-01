<#
.SYNOPSIS
    Install ImageMagick 7.1.2-31 Q16-HDRI on a Windows machine with no GUI.

.DESCRIPTION
    The official Windows installer cannot be driven unattended: under /S it opens a
    directory prompt and waits for a human, which on a CI runner means the job
    times out with the installer named as an orphan.  Five routes were measured
    before this one, and all five are recorded in .github/workflows/ci.yml.

    conda works because it has no installer at all -- it unpacks files.  This
    script fetches micromamba (the static, self-contained build; the conda-forge
    one is not, and dies with STATUS_DLL_NOT_FOUND because libcurl and friends
    are not beside it), creates a prefix, and then ASSERTS that the prefix is
    actually buildable rather than assuming it.

    The assertions are the point.  "micromamba exited 0" is not evidence that the
    toolchain works; a prefix with headers but no import library links no better
    than the runtime-only copy the runner already ships.  So every file the build
    needs is checked by name, and the failing case prints what it did find.

.PARAMETER Prefix
    Where to install.  Defaults to C:\im.

.PARAMETER Channel
    conda channel.  Defaults to conda-forge, which carries a win-64 build of the
    exact version this project is pinned to.

.EXAMPLE
    pwsh -File tools/install-imagemagick.ps1 -Prefix C:\im
#>
[CmdletBinding()]
param(
  [string]$Prefix = 'C:\im',
  [string]$Channel = 'conda-forge'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Pinned deliberately rather than to "latest".  A Q8 build compiles and then
# fails every comparison, and "latest" would let a new release turn a green build
# red with nothing in this repository having changed.  The conda spelling of the
# version uses an underscore: 7.1.2_31, NOT 7.1.2-31, and micromamba's solver
# rejects the hyphenated form with a confusing "does not exist" rather than
# falling back to the right build.
$Version = '7.1.2_31'

Write-Host "=== ImageMagick $Version Q16-HDRI -> $Prefix ==="

# --- micromamba ----------------------------------------------------------------
$mm = Join-Path $env:RUNNER_TEMP 'micromamba.exe'
if (-not (Test-Path $mm)) {
  # The self-contained build from micro.mamba.pm.  Its Library\bin already holds
  # the DLLs it needs, so it runs from a bare directory with nothing beside it.
  $archive = Join-Path $env:RUNNER_TEMP 'micromamba.tar.bz2'
  Write-Host 'fetching micromamba'
  Invoke-WebRequest -Uri 'https://micro.mamba.pm/api/micromamba/win-64/latest' `
    -OutFile $archive -UseBasicParsing -TimeoutSec 900

  $sevenZip = 'C:\Program Files\7-Zip\7z.exe'
  if (-not (Test-Path $sevenZip)) { throw "7-Zip not found at $sevenZip" }
  $stage = Join-Path $env:RUNNER_TEMP 'mm'
  if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $stage | Out-Null

  # A .tar.bz2 three levels deep: 7z peels each layer in turn.
  & $sevenZip e $archive "-o$env:RUNNER_TEMP" -y | Out-Null
  & $sevenZip x (Join-Path $env:RUNNER_TEMP 'micromamba.tar') "-o$stage" -y | Out-Null
  & $sevenZip e (Join-Path $stage 'micromamba.tar') "-o$stage" -y | Out-Null

  $found = Join-Path $stage 'Library\bin\micromamba.exe'
  if (-not (Test-Path $found)) { throw "micromamba.exe not extracted to $found" }
  Copy-Item $found $mm -Force
}

$ver = (& $mm --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $ver) {
  throw "micromamba did not run (exit $LASTEXITCODE). The self-contained build bundles its DLLs; if this is the conda-forge build it fails with STATUS_DLL_NOT_FOUND."
}
Write-Host "micromamba $ver"

# --- the environment -----------------------------------------------------------
$env:MAMBA_ROOT_PREFIX = Join-Path $env:RUNNER_TEMP 'mamba-root'
if (Test-Path $Prefix) { Remove-Item $Prefix -Recurse -Force }

$sw = [Diagnostics.Stopwatch]::StartNew()
& $mm create -y -p $Prefix -c $Channel "imagemagick=$Version" 2>&1 |
  ForEach-Object { Write-Host "  $_" }
$rc = $LASTEXITCODE
$sw.Stop()
Write-Host ("micromamba create exit={0} in {1:N0}s" -f $rc, $sw.Elapsed.TotalSeconds)
if ($rc -ne 0) { throw "micromamba create failed with exit $rc" }

# --- assert it can actually be built against ------------------------------------
# Each of these is a hard #error or a link failure twenty steps later, so they are
# checked here where the message can name what is missing.
$inc  = Join-Path $Prefix 'Library\include\ImageMagick-7'
$lib  = Join-Path $Prefix 'Library\lib'
$bin  = Join-Path $Prefix 'Library\bin'
$exe  = Join-Path $bin 'magick.exe'

$need = @(
  @{ Path = (Join-Path $inc 'MagickCore\MagickCore.h'); What = 'MagickCore header' },
  @{ Path = (Join-Path $inc 'Magick++.h');           What = 'Magick++ header' },
  @{ Path = (Join-Path $lib 'MagickCore-7.Q16HDRI.dll.lib'); What = 'MagickCore import library' },
  @{ Path = (Join-Path $lib 'MagickWand-7.Q16HDRI.dll.lib'); What = 'MagickWand import library' },
  @{ Path = $exe; What = 'magick.exe (the compare reference)' }
)
foreach ($n in $need) {
  if (-not (Test-Path $n.Path)) {
    Write-Host "MISSING: $($n.What) at $($n.Path)"
    Write-Host "prefix tree, depth 2:"
    Get-ChildItem $Prefix -Directory -EA SilentlyContinue | ForEach-Object { Write-Host "  $($_.FullName)" }
    if (Test-Path $lib) {
      Write-Host "contents of ${lib}:"
      Get-ChildItem $lib -EA SilentlyContinue | Select-Object -First 20 |
        ForEach-Object { Write-Host "  $($_.Name)" }
    }
    throw "$($n.What) is missing.  A prefix that installs cleanly is not
necessarily buildable -- this asserts it."
  }
  Write-Host "  [ok] $($n.What)"
}

# The version string decides bit-exactness, so read it rather than trust the spec
# that was requested.  micromamba resolved the `agpl` build variant, not the one
# named first in a search; assuming the request was honoured would be a guess.
$reported = (& $exe --version 2>&1 | Out-String)
Write-Host $reported.Trim()
if ($reported -notmatch '7\.1\.2-31') {
  throw "expected ImageMagick 7.1.2-31, but the prefix reports:`n$reported"
}
if ($reported -notmatch 'Q16' -or $reported -notmatch 'HDRI') {
  throw "expected a Q16 HDRI build. A Q8 build compiles and then fails every comparison:`n$reported"
}

# The core DLL is version-suffixed in this layout (MagickCore-7.Q16HDRI-10.dll) and
# there are 224 dependent DLLs beside it, so the whole directory goes on PATH.
# Copying one file would produce an exe that dies at startup.
"IM_ROOT=$Prefix"            >> $env:GITHUB_ENV
"IMAGEMAGICK_ROOT=$Prefix"   >> $env:GITHUB_ENV
"MAGICK_HOME=$Prefix"        >> $env:GITHUB_ENV
"$bin"                       >> $env:GITHUB_PATH

Write-Host "IMAGEMAGICK_ROOT=$Prefix"
Write-Host "PATH += $bin"
Write-Host 'ImageMagick installed and verified buildable.'