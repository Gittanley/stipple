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
#
# PART C -- THE ONLY STAGE WITH AN EXTERNAL REFERENCE.  Added 2026-10-10.
#
# Parts A and B compare rdither to rdither.  So does every other video stage in this
# project: probe-video-exact is blocks vs opencl, probe-video-determinism is run vs run,
# probe-host-oracle is host vs device.  Each asks "do two of OUR answers agree?" -- and a
# defect shared by every arm is invisible to all of them, because they would agree.
#
# That is not a hypothetical, it is what happened twice.  The palette sampler's by-seek arm
# asked ffmpeg for 4 channels and read into a 3-channel buffer, so every montage cell was an
# RGBA stream read as RGB and 7 of 16 palette entries came out pure green.  It shipped
# through a gate run recorded as "EXIT=0, 165/0 bit-exact, every stage ran, 0 skipped",
# because the sequential sampler and the by-seek sampler were BOTH misreading and therefore
# AGREEING.  `nb_frames` absent on mkv gave a 1-frame palette the same way.  So did
# `--palette-import` on video, which hung on every arm identically.
#
# Part C is the shape that can see it: rdither's video output against ImageMagick's own
# Riemersma, on the same source frame, through an independent `magick compare`.
#
# WHY IT ASSERTS A BOUND AND NOT `AE=0`, because `AE=0` is unachievable and asserting it
# would make this stage permanently red:
#
#   RiemersmaBlocksCpu zeroes the error queue at every BLOCK boundary
#   (rd_riemersma_cpu.cpp:603-604).  RiemersmaWalkCpu -- the image path -- keeps one queue
#   for the whole frame (rd_riemersma_cpu.cpp:344-345), and so does ImageMagick.  At 320x180
#   with the default `--blocks 512` that is 113 error-queue restarts per frame.
#
#   Measured, both sides dithering against the SAME forced palette
#   (notes/video-block-queue-restart.md):
#     image path                        7.5 px  (0.013%)
#     video, --blocks 512 (default)   203.6 px  (0.354%)
#     video, --blocks 16              309.8 px  (0.538%)
#
# Monotonic in block size, converging toward the image path -- the signature of the restart.
# `--help` has always admitted the cost ("--blocks 32 deviates 3.4x more from IM's output").
#
# So the bound is what has value.  A palette regression or a channel misread moves it by an
# ORDER OF MAGNITUDE -- the green-grey bug was not 0.4% off, it was 7 of 16 entries pure
# green -- while the block structure moves it by a factor of 1.5.  A ceiling of 1.0% sits
# between those, set from the worst value measured across six colour counts and five block
# lengths rather than from one sample.
#
# CPU ENGINE ONLY.  The GPU engines deviate from IM by design (~1.7% at --blocks 512, ~6.1%
# at --blocks 32), so asserting them here would report a documented, intended difference as a
# failure on every run.  `--input-mode rgba64` because yuv444 converts on the device and
# yuv420 is 34 dB away by design; the host path is the one where exactness is defined.

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
    # Only claim the checks that actually RAN.  This used to print "differs from the
    # source, not constant, count matches" unconditionally -- including when ffprobe was
    # unavailable and the whole source-comparison block had been skipped -- so the
    # control announced a verification it had not performed.
    $checked = @()
    if ($srcFrames -gt 0) {
      $checked += 'frame count matches the source'
      if ($rf -ne $srcFrames) { $bad += "frame count $rf != source $srcFrames" }
    } else {
      $bad += 'source frame count unavailable (ffprobe?), so the count was NOT checked'
    }
    if ($srcW -gt 0 -and $bad.Count -eq 0) {
      # The decoded stream must NOT equal the source: that is what "dithered nothing"
      # would look like.  Sampled, because a byte loop over the whole clip is slow.
      & $ffmpeg -v error -i $clipPath -f rawvideo -pix_fmt rgb24 (Join-Path $tmp 'src.rgb') 2>&1 | Out-Null
      $srcRaw = Join-Path $tmp 'src.rgb'
      if (-not (Test-Path $srcRaw)) {
        $bad += 'could not decode the source, so "differs from the source" was NOT checked'
      } else {
        $checked += 'differs from the source'
        $sb = [IO.File]::ReadAllBytes($srcRaw)
        $n = [Math]::Min($sb.Length, $ref.bytes.Length)
        $same = $true
        for ($i = 0; $i -lt $n; $i += 997) { if ($sb[$i] -ne $ref.bytes[$i]) { $same = $false; break } }
        if ($same) { $bad += 'output is byte-identical to the source, so nothing was dithered' }
      }
      # And it must not be constant, or a zero-filled buffer would match itself.
      $checked += 'not constant'
      $first = $ref.bytes[0]; $const = $true
      for ($i = 0; $i -lt $ref.bytes.Length; $i += 1013) { if ($ref.bytes[$i] -ne $first) { $const = $false; break } }
      if ($const) { $bad += 'every sampled pixel is identical, so the buffer looks empty' }
    }
    if ($bad.Count -gt 0) {
      "    FAIL  $line -- $($bad -join '; ')"
      $fail++
      $ref = $null
    } else {
      # Name the checks that ran, from the list built above -- not a fixed phrase.
      "    ok    $line -- verified: $($checked -join ', ')"
    }
  }
}

# ---------------------------------------------------------------- part A
if ($null -eq $ref) {
  # The control did not produce a reference, so Part A cannot run.  It used to simply
  # not appear: no heading, no count, no line.  The batch half is the one that needs no
  # GPU and therefore the only half that can run on CI, so its disappearance has to be
  # visible rather than inferred from a missing section.
  ''
  '  part A: does --batch-frames 16 give the same pixels as 1?'
  '    SKIP  no reference render, so there is nothing to compare against'
  $skipped++
}
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
      # A frame-count difference is its OWN fault, not a pixel one.  The lengths are
      # compared first because the loop below clamps to the shorter of the two, so a
      # dropped frame used to print "0 of N frames differ" -- a verdict that named the
      # wrong fault while still failing.  Shape is what distinguishes a layout fault
      # from a rounding one, so it is worth getting right.
      $fb = $ref.frameBytes
      if ($big.bytes.Length -ne $ref.bytes.Length) {
        "    FAIL  $(HashOf $ref) vs $(HashOf $big): FRAME COUNT differs --" +
          " $(FramesOf $ref) frames against $(FramesOf $big)."
        "          --batch-frames must not change how many frames are written."
        $fail++
      } else {
        # Name the first frame that differs.  A whole-clip AE is a single number that
        # says nothing about the SHAPE of the disagreement.
        $firstBad = -1; $badFrames = 0
        $n = [int]($big.bytes.Length / $fb)
        for ($f = 0; $f -lt $n; $f++) {
          $o = $f * $fb; $eq = $true
          for ($i = 0; $i -lt $fb; $i++) { if ($big.bytes[$o+$i] -ne $ref.bytes[$o+$i]) { $eq = $false; break } }
          if (-not $eq) { $badFrames++; if ($firstBad -lt 0) { $firstBad = $f } }
        }
        "    FAIL  $(HashOf $ref) vs $(HashOf $big): $badFrames of $n frames differ" +
          $(if ($firstBad -ge 0) { ", first at frame $firstBad" } else { "" })
        "          --batch-frames changes the ANSWER, not just the schedule.  The dither"
        "          is per-frame and the palette is built once, so it must not."
        $fail++
      }
    }
  }
}

# ---------------------------------------------------------------- part B
''
"  part B: does --no-gpu give the same pixels as the device?"
$devOut = Join-Path $tmp 'dev.mkv'
$rd = Render $devOut @('--engine','blocks')
if ($rd.rc -ne 0 -or -not $rd.made) {
  # Read what the run SAID, not merely that it exited non-zero.  This branch used to
  # report EVERY failure as "no CUDA device", which turned a driver fault, a crash, an
  # out-of-memory or a bad flag into a skip -- and the probe exited 0.  Demonstrated
  # with a stub that succeeded for --no-gpu and returned 255 with
  # CUDA_ERROR_LAUNCH_FAILED here: the output read
  #     SKIP  no CUDA device in this build
  #     video invariance: ok      EXITCODE=0
  # which is the same mistake probe-video-determinism.ps1 and probe-video-exact.ps1
  # already avoid by grepping stderr for the refusal strings.  This file was written
  # after both of them and dropped the check.
  if ($rd.text -match 'no CUDA device' -or $rd.text -match 'requested for video but') {
    "    SKIP  no CUDA device in this build -- this is the half CI cannot run"
    $skipped++
  } else {
    $fail++
    "    FAIL  the device render failed (exit $($rd.rc)) and it is NOT a missing device:"
    foreach ($dl in ($rd.text -split "`n" | Where-Object { $_ -match '\S' } | Select-Object -First 3)) {
      "            $($dl.Trim())"
    }
  }
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

# ---------------------------------------------------------------- part C
#
# The only part with an EXTERNAL reference.  Parts A and B, and every other video stage in
# this project, compare rdither to rdither -- so a defect shared by all of their arms is
# invisible to them, because they agree.  Three have shipped that way; see the header.
#
# Cost: one extra 12-frame render plus a few `magick` calls, a couple of seconds.  It runs on
# every build and needs no GPU, because it uses the host engine.
''
"  part C: does the video path match ImageMagick's own Riemersma?"
$magick = Resolve-Tool 'magick.exe' ''
if (-not $magick) {
  '    SKIP  magick is not on PATH, so there is no reference implementation to compare to'
  $skipped++
} else {
  $cOut = Join-Path $tmp 'oracle.mkv'
  # An rgb24 fixture is what makes this an ORACLE rather than another self-comparison: it
  # has no YCbCr->RGB step anywhere, so the only thing that can differ from ImageMagick is
  # the dither.  With a yuv420p fixture a colour-conversion difference would swamp the thing
  # being measured.
  $oracleClip = Join-Path $tmp 'oracle_src.mkv'
  & $ffmpeg -v error -y -f lavfi -i "testsrc=size=${srcW}x${srcH}:rate=30:duration=1" `
      -frames:v 12 -pix_fmt rgb24 -c:v ffv1 $oracleClip 2>&1 | Out-Null
  if (-not (Test-Path $oracleClip)) {
    '    SKIP  could not build an rgb24 oracle fixture'
    $skipped++
  } else {
    # Palette from the IMAGE path, handed to the video path, so both dither against the same
    # colours -- otherwise this compares two different palettes and reports the difference
    # as a dither fault.  --colors MUST equal the file's count: rdither refuses a mismatch,
    # and a refusal here looks exactly like a divergence.
    $palTxt = Join-Path $tmp 'oracle_pal.txt'
    $pimg = Join-Path $tmp 'oracle_pal.png'
    & $ffmpeg -v error -y -i $oracleClip -frames:v 1 -pix_fmt rgb24 $pimg 2>&1 | Out-Null
    $null = & $exe --colors $Colors --palette-export $palTxt $pimg (Join-Path $tmp 'pal_out.png') 2>&1
    $palCount = 0
    if (Test-Path $palTxt) {
      $palCount = @(Get-Content $palTxt | Where-Object { $_ -notmatch '^#' -and $_.Trim() }).Count
    }
    if ($palCount -lt 1) {
      '    FAIL  could not export a palette to compare against'
      $fail++
    } else {
      $rc2 = & $exe --video --engine cpu --colors $palCount --palette-import $palTxt `
                --input-mode rgba64 --video-lossless --no-audio `
                $oracleClip $cOut 2>&1 | Out-String
      if (-not (Test-Path $cOut)) {
        '    FAIL  the oracle render produced no output'
        $fail++
      } else {
        # Force ImageMagick onto OUR palette with -remap, so a palette difference cannot be
        # mistaken for a dither difference.  Four earlier attempts at this comparison left
        # this step out, which is why they reported divergences that were really just two
        # different palettes.
        $hexes = @(Get-Content $palTxt | Where-Object { $_ -notmatch '^#' -and $_.Trim() } |
                   ForEach-Object { ($_ -split "`t")[-1].Trim() })
        $gif = Join-Path $tmp 'oracle_pal.gif'
        $gargs = @('-size', "$($hexes.Count)x1", 'xc:black')
        for ($k = 0; $k -lt $hexes.Count; $k++) {
          $gargs += @('-fill', "#$($hexes[$k])", '-draw', "point $k,0")
        }
        & $magick @gargs $gif 2>&1 | Out-Null

        $srcPng = Join-Path $tmp 'oracle_src0.png'
        & $ffmpeg -v error -y -i $oracleClip -frames:v 1 -pix_fmt rgb24 $srcPng 2>&1 | Out-Null
        $refPng = Join-Path $tmp 'oracle_ref.png'
        & $magick $srcPng -dither Riemersma -remap $gif $refPng 2>&1 | Out-Null
        $gotPng = Join-Path $tmp 'oracle_got.png'
        & $ffmpeg -v error -y -i $cOut -frames:v 1 -pix_fmt rgb24 $gotPng 2>&1 | Out-Null

        if (-not (Test-Path $refPng) -or -not (Test-Path $gotPng)) {
          '    FAIL  could not produce both sides of the comparison'
          $fail++
        } else {
          # The CEILING, and why it is a ceiling rather than AE=0, is in the header: the
          # blocks walk zeroes its error queue per block and the reference does not.
          #
          # MEASURED ENVELOPE, not a guess and not one sample.  Every point is rgb24 in
          # ffv1, palette exported from the image path and forced onto ImageMagick with
          # -remap, so both sides dither the same colours and only the dither differs.
          # Across colour count, at --blocks 512:
          #
          #     --colors  4 ( 4 entries)  0.421%      --colors 32 (24 entries)  0.110%
          #     --colors  8 ( 8 entries)  0.339%      --colors 64 (45 entries)  0.099%
          #     --colors 12 (11 entries)  0.353%
          #     --colors 16 (14 entries)  0.354%
          #
          # Across block length, at 14 palette entries -- the dominant axis:
          #
          #     --blocks   16   0.538%     --blocks  512   0.354%
          #     --blocks   32   0.513%     --blocks 4096   0.348%
          #     --blocks   64   0.460%
          #
          # So the worst observed value across both axes is 0.538%.  The ceiling is 1.0%:
          # roughly 1.9x the worst case measured, which is enough that the number does not
          # move when the machine is busy, and tight enough that a regression has to be a
          # real one to get through.
          #
          # IT WAS 2.0% BEFORE, which was a number picked off a single 0.354% sample --
          # 5.6x headroom, so anything up to 1.9% would have passed.  A ceiling with that
          # much slack is not a guard.  The envelope above is what makes 1.0 defensible, and
          # it took six colour counts and five block lengths to earn it; one run does not.
          $CeilingPct = 1.0
          $aeText = (& cmd /c "`"$magick`" compare -metric AE `"$gotPng`" `"$refPng`" null: 2>&1" | Out-String).Trim()
          # This ImageMagick reports a NORMALISED metric -- "229.48 (0.00398402)" -- not a
          # bare integer.  The count is the first token and the fraction is the second; a
          # check written against the integer form alone reads the fraction and passes
          # everything, which is how this was nearly shipped.
          $tok = ($aeText -split '\s+')
          $pct = $null
          if ($tok.Count -ge 2 -and $tok[1] -match '^\(([0-9.]+)\)$') {
            $pct = 100.0 * [double]$Matches[1]
          } elseif ($tok.Count -ge 2 -and $tok[0] -match '^[0-9]+(\.[0-9]+)?$') {
            # The first token must be a NUMBER before it is divided.  It did not have to
            # be, and that made this stage unfailable.
            #
            # `magick compare` writes its error to stderr, so a missing or unreadable
            # image yields text rather than a metric:
            #
            #     compare: unable to open image 'x.png' null:
            #
            # Stripping the non-numerics from that leaves "", `[double]''` is 0, and 0 is
            # under any ceiling, so a BROKEN COMPARISON REPORTED A PASS.  Measured, and it
            # is the worst possible failure for the one stage in this suite that has an
            # external reference: breaking the tool that does the measuring turned the
            # check green.
            #
            # The sibling branch above, added for the same reason, already fails closed --
            # this one did not, and the author of this line was also the author of that
            # one.
            $den = $srcW * $srcH
            if ($den -gt 0) { $pct = 100.0 * [double]$tok[0] / $den }
          }
          if ($null -eq $pct) {
            "    FAIL  could not read the comparison metric ('$aeText')"
            $fail++
          } elseif ($pct -le $CeilingPct) {
            "    ok    {0:N3}% of pixels differ from ImageMagick's Riemersma (ceiling {1}%)" -f $pct, $CeilingPct
            "          same $palCount colours forced on both sides, so this measures the"
            "          dither.  Baseline ~0.40%: RiemersmaBlocksCpu zeroes its error queue per"
            "          block (rd_riemersma_cpu.cpp:604) and the reference does not."
          } else {
            "    FAIL  {0:N3}% of pixels differ from ImageMagick's Riemersma, ceiling {1}%" -f $pct, $CeilingPct
            "          This is the only video stage with an external reference.  Every other"
            "          one compares rdither to rdither and cannot see a fault they share."
            $fail++
          }
        }
      }
    }
  }
}

Remove-Item $tmp -Recurse -Force -EA SilentlyContinue
''
if ($skipped -gt 0 -and $fail -eq 0) {
  "$($skipped) part(s) SKIPPED for want of a device or engine.  Skips are not passes."
}
# Nothing ran is not a clean sweep.  $fail -eq 0 cannot tell "both halves agreed" from
# "both halves were skipped", and verify.ps1 maps exit 0 to an empty branch, so the
# driver contributes no line and the log cannot be distinguished from a good run.  This
# probe had no floor; probe-unvisited-pixel.ps1 and probe-video-exact.ps1 both do.
if ($skipped -ge 2 -and $fail -eq 0) {
  "video invariance: cannot run -- every part was skipped, so nothing was checked."
  exit 2
}
if ($fail -eq 0) { "video invariance: ok"; exit 0 }
"video invariance: $fail FAILED"
exit 1
