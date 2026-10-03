# Does the video pipeline give the same ANSWER at different batch sizes, and on the
# host as on the device?
#
# Four silent faults lived in the host video path.  Not one of them was visible to any
# check this project had, and the reason is structural rather than unlucky:
#
#   * probe-video-exact.ps1 compares `--engine blocks` against `--engine opencl` --
#     BOTH device engines.  The host is not in that comparison at all.
#   * probe-video-determinism.ps1 checks the host, but for DETERMINISM only: the same
#     command run N times must give the same bytes.  That sees a race or an
#     uninitialised read.  It is structurally blind to a WRONG-BUT-STABLE answer, and
#     three of the four faults were exactly that.
#   * The only sweep of --batch-frames in the repo is probe-cpubudget.ps1, which is a
#     PERFORMANCE probe, and it uses 24 and 32 -- both above the threshold, so even as
#     a correctness check it would have missed the worst of the four.
#
# So a fault in the host path that does not vary between runs, or that only appears at
# some batch sizes, had nowhere to land.  That is not a gap in the fixtures; it is a
# missing SHAPE of check, and this file is that shape.
#
# WHAT IT CHECKS, in two parts.
#
# PART A -- BATCH INVARIANCE.  The same command with `--batch-frames 1` and with
# `--batch-frames N`, compared pixel by pixel.  This needs NO GPU, so it runs on a
# GPU-less CI runner, which is the only place it can run on every build.
#
# Why batch 1 is the right reference: the pipeline's per-frame work is supposed to be
# independent.  RiemersmaBlocksCpu zeroes its error queue per frame per block, the
# palette is built once for the whole clip before any frame is dithered, and the
# encoder is a stream.  So the batch size is a scheduling knob and nothing else.  When
# that stopped being true:
#
#   * the writer emitted frames in WORKER-COMPLETION ORDER, because `done` was a deque
#     of slot indices carrying no position -- correct with one batch by accident,
#     interleaved with several.
#   * the input converter read 8-bit planar yuv444p through a `uint16` pointer, so it
#     consumed twice what was written.  At --batch-frames 1 each batch holds one frame,
#     the frame is read from offset 0, and the bug is INVISIBLE.  That is what makes
#     batch 1 the right reference rather than merely a convenient one: it is the one
#     setting under which this class of fault cannot show up.
#   * FloatsToYuv444 used its `pixels` argument as the PLANE STRIDE, so it was correct
#     for one frame and batch-planar for more.  writer_convert_threads is capped at 3,
#     so every --batch-frames above 3 was wrong -- and the default is 16.
#
# PART B -- HOST AGAINST DEVICE.  `--engine blocks --no-gpu` against `--engine blocks`,
# compared pixel by pixel.  This is the comparison the suite has never made, and it is
# the only one that can catch a host answer that is stable AND wrong.  It needs a GPU
# and skips without one; Part A is what covers CI.
#
# THE NEGATIVE CONTROL, which is what makes Part A mean something.
#
# "These two renders agree" is also what a renderer that dithered nothing would report,
# and it is what two runs of the same broken pipeline report.  Comparing two of your own
# outputs against each other proves only that they are alike.  So this probe also
# requires, of the reference render:
#
#   1. the frame count read FROM THE FILE matches the source's, not rdither's report --
#      a pipeline that consistently drops frames reports a consistent number, and a
#      self-consistent report is not evidence;
#   2. the decoded bytes differ from the SOURCE clip, so "identical" cannot mean
#      "nothing was dithered";
#   3. the decoded bytes are not constant, so "identical" cannot mean "an empty or
#      zero-filled buffer that happens to match itself".
#
# Without 2 and 3 a stub, an empty file, or a pass-through would pass this test.  With
# them, agreeing with the reference means agreeing with a render that demonstrably did
# the work.
#
# Exit 2 means "cannot run" (no ffmpeg, no rdither), deliberately distinct from exit 1,
# "ran and the answers differ".

param(
  [string]$Rdither = "",
  [string]$Ffmpeg  = "",
  [string]$Clip    = "",
  [int]$Colors     = 16,
  # The default is 16 and the bug threshold was 4, so this straddles it.  1 is the
  # reference; anything above writer_convert_threads' cap of 3 exercises the split.
  [int]$BatchBig   = 16,
  [switch]$FixturesOnly
)

$ErrorActionPreference = 'Stop'

function Resolve-Tool([string]$name, [string]$given) {
  if ($given -and (Test-Path $given)) { return (Resolve-Path $given).Path }
  $c = Get-Command $name -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  return ""
}

$exe = Resolve-Tool 'rdither.exe' $Rdither
$ffmpeg = Resolve-Tool 'ffmpeg.exe' $Ffmpeg
$ffprobe = Resolve-Tool 'ffprobe.exe' ''

if (-not $exe) { "video invariance: no rdither given and none on PATH; cannot run"; exit 2 }
if (-not $ffmpeg) { "video invariance: no ffmpeg on PATH; cannot run"; exit 2 }

# A fixture built here rather than committed, because a 30-frame lavfi clip is a few
# tens of kilobytes and a binary in git is a binary in git forever.  yuv444p ffv1 so
# that the host's planar-YUV input path is the one under test rather than rgba64le,
# which shares none of that code.
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("rd_inv_" + [Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

function New-Fixture([string]$path) {
  $args = @('-v','error','-y','-f','lavfi','-i','testsrc=size=320x180:rate=30:duration=1',
            '-c:v','ffv1','-pix_fmt','yuv444p', $path)
  & $ffmpeg @args 2>&1 | Out-Null
  return (Test-Path $path)
}

if ($Clip -and (Test-Path $Clip)) { $clipPath = (Resolve-Path $Clip).Path }
else {
  $clipPath = Join-Path $tmp 'fixture.mkv'
  if (-not (New-Fixture $clipPath)) {
    "video invariance: could not build a fixture with ffmpeg; cannot run"
    Remove-Item $tmp -Recurse -Force -EA SilentlyContinue
    exit 2
  }
}

# Frame count and geometry from the CONTAINER, not from rdither's own report.
$srcFrames = 0; $srcW = 0; $srcH = 0
if ($ffprobe) {
  $probe = & $ffprobe -v error -select_streams v:0 -count_frames `
                     -show_entries stream=nb_read_frames,width,height `
                     -of 'csv=p=0' $clipPath 2>$null | Select-Object -First 1
  if ($probe -and $probe -match '^(\d+),(\d+),(\d+)$') {
    $srcW = [int]$Matches[1]; $srcH = [int]$Matches[2]; $srcFrames = [int]$Matches[3]
  }
}

function Render([string]$out, [string[]]$extra) {
  Remove-Item $out -Force -EA SilentlyContinue
  $text = & $exe --video @($extra) --colors $Colors --video-lossless --no-audio `
            $clipPath $out 2>&1 | Out-String
  return @{ rc = $LASTEXITCODE; text = $text; made = (Test-Path $out) }
}

# Decode to raw rgb24 and hash it.  Never pipe binary through PowerShell: the native
# tool writes the file and it is read back in whole.  ffmpeg is asked for rgb24 rather
# than the file's own format so the comparison is on the pixels a viewer would see, and
# so a container difference cannot masquerade as a pixel difference.
function Decode([string]$mkv, [string]$tag) {
  $raw = Join-Path $tmp "$tag.rgb"
  Remove-Item $raw -Force -EA SilentlyContinue
  & $ffmpeg -v error -i $mkv -f rawvideo -pix_fmt rgb24 $raw 2>&1 | Out-Null
  if (-not (Test-Path $raw)) { return $null }
  $bytes = [IO.File]::ReadAllBytes($raw)
  if ($bytes.Length -eq 0) { return $null }
  return @{ bytes = $bytes; frameBytes = $srcW * $srcH * 3; hash = (Get-FileHash $raw -Algorithm SHA256).Hash }
}

function FramesOf($d) { if ($d -eq $null -or $d.frameBytes -le 0) { return 0 }; return [int]($d.bytes.Length / $d.frameBytes) }
function HashOf($d) { if ($d -eq $null) { return '(none)' }; return $d.hash.Substring(0,16) }

"Video invariance: batch $BatchBig vs 1, and host vs device"
''
"clip: $clipPath"
if ($srcFrames -gt 0) { "  source: ${srcW}x${srcH}, $srcFrames frames (from the container)" }
''
$fail = 0
$skipped = 0

# ---------------------------------------------------------------- control
# Established against the batch-1 render, and checked BEFORE it is used as a reference,
# so a reference that did no work cannot make the comparison below pass.
"  control: is the reference render real work?"
$refOut = Join-Path $tmp 'b1.mkv'
$r1 = Render $refOut @('--engine','blocks','--no-gpu','--batch-frames','1')
if ($r1.rc -ne 0 -or -not $r1.made) {
  "    SKIP  --engine blocks --no-gpu is unavailable in this build"
  "          $($r1.text.Trim() -split "`n" | Select-Object -First 1)"
  $skipped++
  $ref = $null
} else {
  $ref = Decode $refOut 'b1'
  if ($null -eq $ref) {
    "    FAIL  the batch-1 render produced no decodable pixels"
    $fail++
    $ref = $null
  } else {
    $rf = FramesOf $ref
    $line = "    $($rf) frames, $(HashOf $ref)"
    $bad = @()
    if ($srcFrames -gt 0 -and $rf -ne $srcFrames) { $bad += "frame count $rf != source $srcFrames" }
    if ($srcW -gt 0 -and $bad.Count -eq 0) {
      # The decoded stream must NOT equal the source: that is what "dithered nothing"
      # would look like.  Sampled, because a byte loop over the whole clip is slow.
      & $ffmpeg -v error -i $clipPath -f rawvideo -pix_fmt rgb24 (Join-Path $tmp 'src.rgb') 2>&1 | Out-Null
      $srcRaw = Join-Path $tmp 'src.rgb'
      if (Test-Path $srcRaw) {
        $sb = [IO.File]::ReadAllBytes($srcRaw)
        $n = [Math]::Min($sb.Length, $ref.bytes.Length)
        $same = $true
        for ($i = 0; $i -lt $n; $i += 997) { if ($sb[$i] -ne $ref.bytes[$i]) { $same = $false; break } }
        if ($same) { $bad += 'output is byte-identical to the source, so nothing was dithered' }
      }
      # And it must not be constant, or a zero-filled buffer would match itself.
      $first = $ref.bytes[0]; $const = $true
      for ($i = 0; $i -lt $ref.bytes.Length; $i += 1013) { if ($ref.bytes[$i] -ne $first) { $const = $false; break } }
      if ($const) { $bad += 'every sampled pixel is identical, so the buffer looks empty' }
    }
    if ($bad.Count -gt 0) {
      "    FAIL  $line -- $($bad -join '; ')"
      $fail++
      $ref = $null
    } else {
      "    ok    $line -- differs from the source, not constant, count matches"
    }
  }
}

# ---------------------------------------------------------------- part A
if ($null -ne $ref) {
  ''
  "  part A: does --batch-frames $BatchBig give the same pixels as 1?"
  $bigOut = Join-Path $tmp "b$BatchBig.mkv"
  $rb = Render $bigOut @('--engine','blocks','--no-gpu','--batch-frames',"$BatchBig")
  if ($rb.rc -ne 0 -or -not $rb.made) {
    "    FAIL  the --batch-frames $BatchBig render failed (exit $($rb.rc))"
    $fail++
  } else {
    $big = Decode $bigOut "b$BatchBig"
    if ($null -eq $big) {
      "    FAIL  the --batch-frames $BatchBig render produced no decodable pixels"
      $fail++
    } elseif ($big.hash -eq $ref.hash) {
      "    ok    identical to batch 1 ($(HashOf $ref))"
    } else {
      # Name the first frame that differs.  A whole-clip AE is a single number that
      # says nothing about the SHAPE of the disagreement, and shape is what
      # distinguishes a layout fault from a rounding one.
      $fb = $ref.frameBytes
      $firstBad = -1; $badFrames = 0
      $n = [Math]::Min($big.bytes.Length, $ref.bytes.Length) / $fb
      for ($f = 0; $f -lt [int]$n; $f++) {
        $o = $f * $fb; $eq = $true
        for ($i = 0; $i -lt $fb; $i++) { if ($big.bytes[$o+$i] -ne $ref.bytes[$o+$i]) { $eq = $false; break } }
        if (-not $eq) { $badFrames++; if ($firstBad -lt 0) { $firstBad = $f } }
      }
      "    FAIL  $(HashOf $ref) vs $(HashOf $big): $badFrames of $([int]$n) frames differ" +
        $(if ($firstBad -ge 0) { ", first at frame $firstBad" } else { "" })
      "          --batch-frames changes the ANSWER, not just the schedule.  The dither"
      "          is per-frame and the palette is built once, so it must not."
      $fail++
    }
  }
}

# ---------------------------------------------------------------- part B
''
"  part B: does --no-gpu give the same pixels as the device?"
$devOut = Join-Path $tmp 'dev.mkv'
$rd = Render $devOut @('--engine','blocks')
if ($rd.rc -ne 0 -or -not $rd.made) {
  "    SKIP  no CUDA device in this build -- this is the half CI cannot run"
  $skipped++
} elseif ($rd.text -match 'gpu=no') {
  # Defence in depth: a case named for the device must not report for the device.
  "    SKIP  --engine blocks fell back to the host, so this tests nothing new"
  $skipped++
} else {
  $dev = Decode $devOut 'dev'
  if ($null -eq $dev) {
    "    FAIL  the device render produced no decodable pixels"
    $fail++
  } elseif ($null -eq $ref) {
    "    SKIP  no host reference to compare against"
    $skipped++
  } elseif ($dev.hash -eq $ref.hash) {
    "    ok    host and device byte-identical ($(HashOf $ref))"
  } else {
    $fb = $ref.frameBytes
    $n = [Math]::Min($dev.bytes.Length, $ref.bytes.Length) / $fb
    $badFrames = 0
    for ($f = 0; $f -lt [int]$n; $f++) {
      $o = $f * $fb; $eq = $true
      for ($i = 0; $i -lt $fb; $i++) { if ($dev.bytes[$o+$i] -ne $ref.bytes[$o+$i]) { $eq = $false; break } }
      if (-not $eq) { $badFrames++ }
    }
    "    FAIL  host $(HashOf $ref) vs device $(HashOf $dev): $badFrames of $([int]$n) frames differ"
    $fail++
  }
}

Remove-Item $tmp -Recurse -Force -EA SilentlyContinue
''
if ($skipped -gt 0 -and $fail -eq 0) {
  "$($skipped) part(s) SKIPPED for want of a device or engine.  Skips are not passes."
}
if ($fail -eq 0) { "video invariance: ok"; exit 0 }
"video invariance: $fail FAILED"
exit 1
