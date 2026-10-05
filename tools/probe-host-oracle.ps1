$env:CUDA_PATH = 'C:\Users\RatOrMouse\.cache\opencode\rt2\cuda'
$env:MAGICK_HOME = 'C:\Program Files\ImageMagick-7.1.2-Q16-HDRI'
$env:PATH = "$env:CUDA_PATH\bin;C:\fftest2\ffmpeg-n8.1-latest-win64-gpl-8.1\bin;$env:MAGICK_HOME;$env:PATH"
Set-Location 'J:\ai code gen\dither 2'

$work = Join-Path $env:TEMP 'rd_hostref'
New-Item -ItemType Directory -Force -Path $work | Out-Null

# 320x180: small enough that a host-vs-device comparison costs almost nothing, and
# non-power-of-two on both axes so the curve does NOT cover the frame exactly once --
# which is the case where the scatter's `owner[p] == i` filter actually does work.
$src = Join-Path $work 'ref.png'
# DETERMINISTIC source.  This was +noise Gaussian with no -seed, which meant a
# different image on every run: the three hashes below agreed with each other but
# CHANGED between runs, so the probe could not detect a change in the code under test
# at all.  A self-consistent check that cannot see its own subject is worse than none.
& magick -size 320x180 plasma:fractal -seed 12345 -attenuate 0.4 +noise Gaussian -colorspace sRGB $src
"source: 320x180 (deliberately not a power of two on either axis)"

# The CUDA blocks engine is the reference: it is the engine with a known-correct
# unvisited-pixel fill and the one the host path is supposed to agree with.
& .\build\Release\rdither.exe --video --colors 16 --engine blocks --batch-frames 1 `
    $src (Join-Path $work 'dev.mp4') *>&1 | Out-Null
$rc = $LASTEXITCODE

# The host path: cpu workers, no GPU, batch 1 first (where the batch-planar bug was
# invisible) and then a multi-frame batch (where it was visible).
& .\build\Release\rdither.exe --video --colors 16 --engine cpu --cpu-threads 4 `
    --batch-frames 1 $src (Join-Path $work 'host_b1.mp4') *>&1 | Out-Null
$rc1 = $LASTEXITCODE

& .\build\Release\rdither.exe --video --colors 16 --engine cpu --cpu-threads 4 `
    --batch-frames 16 $src (Join-Path $work 'host_b16.mp4') *>&1 | Out-Null
$rc16 = $LASTEXITCODE

"device exit=$rc   host batch1 exit=$rc1   host batch16 exit=$rc16"
""

# Compare decoded pixels, not container bytes: h264/x264 is lossy so the FILES will
# differ for reasons that have nothing to do with the dither.  ffmpeg md5 over the
# rawvideo stream, streaming, so no multi-GB raw dump.
function Get-DecHash([string]$f) {
  if (-not (Test-Path $f)) { return 'MISSING' }
  $h = (& ffmpeg -v error -i $f -f rawvideo -pix_fmt rgba -f md5 - 2>$null | Out-String).Trim()
  if ($h) { return $h } else { return 'NO-HASH' }
}
$dev = Get-DecHash (Join-Path $work 'dev.mp4')
$h1  = Get-DecHash (Join-Path $work 'host_b1.mp4')
$h16 = Get-DecHash (Join-Path $work 'host_b16.mp4')

"device    md5 $dev"
"host b1   md5 $h1"
"host b16  md5 $h16"
""

$fail = 0
if ($dev -eq 'NO-HASH' -or $dev -eq 'MISSING') { '  device produced no hash -- the oracle is VOID'; $fail = 99 }
if ($h1 -ne $dev)  { "  MISMATCH: host batch 1 differs from device"; $fail++ }
if ($h16 -ne $h1)  { "  MISMATCH: host batch 16 differs from host batch 1 -- the host path is not batch-invariant"; $fail++ }
if ($h1 -eq $h16)  { "  host is batch-invariant (b1 == b16)" }
if ($fail -eq 0) { "HOST-ORACLE: host == device, and host is batch-invariant" }
else { "HOST-ORACLE: $fail MISMATCH(ES) -- R1 must not land on top of this" }
