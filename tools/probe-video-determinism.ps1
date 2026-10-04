# Is the VIDEO pipeline the same every time you run it?
#
# The image equivalent is tools\probe-determinism.ps1, and this exists for the same
# reason, one level up.  A test that compares two implementations cannot see a fault
# they share.  That is not theoretical here -- it is how two separate bugs survived:
#
#   * The unfilled fifth channel in ImStore.  Both engines went through the same
#     writer, so OpenCL-vs-CUDA agreed while both were wrong.
#   * The unvisited pixel in the uint16/4:4:4 output.  CUDA's buffer was a bare
#     cudaMalloc and OpenCL's was created without COPY_HOST_PTR, so BOTH were reading
#     uninitialised device memory and the cross-engine test could not have seen it.
#     See tools\probe-unvisited-pixel.ps1, which compares against the source instead.
#
# Both were found by accident or by a check aimed somewhere else.  Neither was found
# by the cross-engine comparison, which is the check this project runs most.
#
# THE BUG THIS WOULD HAVE CAUGHT DIRECTLY.  -shortest stops the output at whichever
# track ends first, and when the VIDEO is the longer one it deletes VIDEO frames to
# make the tracks match.  Measured: a 30 s excerpt with 895 video frames against
# 30.0128 s of audio came out with 892.  Every existing check was blind to it:
# both engines dropped the same three frames, so OpenCL-vs-CUDA agreed, and both
# fixtures in tests\ were silent, so -map 1:a? and -c:a copy were exercised by
# nothing.  Frame accounting was bolted on afterwards, in probe-video-exact.ps1, as a
# separate concern -- which it should be, but it means a reintroduction of the same
# class of bug in the palette or read stage would not be caught by it either.
#
# So this probe checks, per engine, over N runs:
#
#   1. the decoded PIXEL stream is identical every run
#   2. the FRAME COUNT is identical every run
#   3. the pipeline actually produced output on every run
#
# Deliberately NOT compared: the encoded file's bytes.  Matroska embeds a random
# SegmentUID and a writing timestamp, so two runs can never produce equal files even
# when every pixel matches -- that has already "failed" for several rounds in this
# project.  The pixels and the frame count are the things that are supposed to be
# reproducible.
#
# The audio clip is used rather than the silent one specifically because that is where
# the frame-deleting bug lived.  A silent fixture cannot exercise -shortest at all,
# however many times it is rendered.
#
# Exit 2 means "cannot run" (no clip), which is deliberately distinct from exit 1,
# "ran and differed".

param(
  [string]$Clip = "",
  # 8 stays.  It looked like 8 was too few, and the measurement says otherwise for the
  # path this probe actually samples.
  #
  # On the default yuv444p input the `cpu` case diverges on 11 of 24 runs -- a rate of
  # 0.458 -- so an 8-run sample misses it with probability 0.542^7, about 1.4%.  The
  # suite detects this reliably at 8, and raising it to 24 measured 21 s against 62 s
  # for a gain from 98.6% to 99.9%.  Not worth 3x the time.
  #
  # The confusion worth recording: on --input-mode rgba64 the same case diverges on
  # only 6 of 24 runs, a rate of 0.25, where 8 runs miss it 13% of the time.  Both
  # figures are real and they are for different paths.  The rgba64 number is the one
  # that made this look urgent, and it is the one this probe never takes.
  [int]$Runs = 8,
  [int]$Colors = 16,
  [string]$Rdither = "",
  [string]$Magick = ""
)

$ErrorActionPreference = 'Continue'

if (-not $Rdither) {
  $Rdither = Join-Path (Split-Path $PSScriptRoot -Parent) 'build\Release\rdither.exe'
}
if (-not (Test-Path $Rdither)) { "missing $Rdither -- build first"; exit 1 }

if (-not $Clip) {
  # Audio first, deliberately: this is the fixture the -shortest bug needed and the
  # silent one cannot stand in for it.
  foreach ($c in @('tests\clip1920_audio.mkv', 'tests\clip1920.mp4')) {
    $p = Join-Path (Split-Path $PSScriptRoot -Parent) $c
    if (Test-Path $p) { $Clip = $p; break }
  }
}
if (-not $Clip -or -not (Test-Path $Clip)) {
  ''
  'cannot run: no video fixture (tests\clip1920_audio.mkv)'
  exit 2
}

if (-not $Magick) {
  $m = Get-Command magick -ErrorAction SilentlyContinue
  if ($m) { $Magick = $m.Source }
}
if (-not $Magick) { "cannot run: magick not on PATH"; exit 2 }

# A UNIQUE directory, per process.  The fixed name this used to have is a bug: two
# instances of this probe -- which happens the moment anything runs the suite twice, or
# a suite overlaps with a manual run -- share one directory, and then one deletes
# frames.raw while the other is hashing it.  The symptom is not a test failure, it is a
# crash: "The process cannot access the file because it is being used by another
# process", followed by a null index and a cascade of nonsense.  Observed exactly that,
# and it was misread at first as CUDA and OpenCL becoming non-deterministic, which is
# the sort of conclusion a flaky instrument invites.
$tmp = Join-Path $env:TEMP ('rdvid_det_' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# Decode to raw RGBA.  Through cmd /c, not the PowerShell pipeline: a native command's
# stdout piped through PowerShell is re-encoded, which turns the binary into text and
# destroys the thing being hashed.
function Get-DecodedHash([string]$mkv) {
  # ffmpeg streams the decoded frames into md5: no file, one frame held at a time.
  # The obvious alternative, `magick "$mkv" -depth 8 "rgba:$raw"`, decodes EVERY frame
  # into a host buffer first -- measured 1,865.7 MiB peak working set and a 449 MiB
  # file for the 60-frame 1080p clip, and this probe calls it once per run per engine.
  # It hashed every decoded sample, which md5 over the rawvideo stream also does.
  $h = (& ffmpeg -v error -i $mkv -f rawvideo -pix_fmt rgba -f md5 - 2>$null | Out-String).Trim()
  if (-not $h) { return $null }
  # The md5 stream carries every decoded sample of every frame, which is exactly what
  # hashing the raw file covered.  A frame count is reported alongside so a truncated
  # decode cannot masquerade as a stable one.
  $n = 0
  try {
    $n = [int](& ffprobe -v error -select_streams v:0 -count_frames `
                 -show_entries stream=nb_read_frames -of csv=p=0 $mkv 2>$null | Out-String).Trim()
  } catch { $n = 0 }
  return @{ hash = $h; bytes = $n }
}

# Frame count, from the file rather than from rdither's own report: a pipeline that
# consistently drops three frames would report a consistent number, and a report that
# agrees with itself is not evidence.
function Get-FrameCount([string]$mkv) {
  $n = cmd /c "ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 ""$mkv""" 2>$null
  if (-not $n) { return -1 }
  return [int]$n
}

# Engine configurations, named by what they exercise rather than by the flag alone.
# The OpenCL one drops RD_YUV444_OUT so the planar output kernel is used, since that
# is a different code path with its own constants from the rgba64le route.
$cases = @(
  @{ name = 'cuda';    args = @('--engine', 'blocks') },
  @{ name = 'opencl';  args = @('--engine', 'opencl', '--input-mode', 'yuv444'); env = $true },
  # Left as '--engine blocks --no-gpu' on purpose, after trying '--engine cpu' here
  # and measuring what that did.  It looked like a cleanup: --no-gpu asks for the CUDA
  # blocks engine and then declines to use a GPU, which is a strange way to say "host".
  # But two things came out of the experiment:
  #
  #  * --engine blocks FALLS BACK TO THE HOST when there is no CUDA device -- measured,
  #    on a -DRD_WITH_CUDA=OFF build -- so this case was never blocked by the missing
  #    engine in the first place.  The change was not what let it run.
  #  * This case was the one carrying the host-path faults, and the KNOWN DEFECT
  #    reporting used to be keyed to the case NAME.  Renaming the engine would have
  #    silently moved those defects out of "reported, not failed" and into a hard
  #    failure -- and --engine cpu is deterministic, so the case that failed would have
  #    been a different code path than the one being reported on.  All of those faults
  #    are fixed and the bypass is gone, so the case name no longer carries that weight;
  #    the note is kept because the fragility was real, not because it still applies.
  #
  # So it stays.  The real gap is that `--engine blocks` falling back to the host with
  # no diagnostic is its own defect: rd_video.cpp refuses exactly that fallback for
  # OpenCL, because a silent fallback makes --engine X quietly measure something else.
  @{ name = 'cpu';     args = @('--engine', 'blocks', '--no-gpu') }
)

"Video determinism: $Runs runs per case, decoded pixels + frame count"
''
"clip: $Clip"
''
$fail = 0
$skipped = 0
$total = 0

foreach ($c in $cases) {
  $total++
  # The `cpu` case was a KNOWN, UNFIXED BUG and was reported rather than failed so the
  # rest of the sweep stayed usable while it was open.  It is FIXED and the bypass is
  # GONE: a divergence in this case is now a hard failure, which is the only way the
  # check can catch the next one.  The comment below used to say "if this case starts
  # PASSING, remove the bypass" -- it does, so it was removed.
  #
  # What it WAS.  THREE defects, not one, and separating them was the hard part --
  # fixing the first exposed the other two rather than clearing the case.
  #
  # 1. OUTPUT FORMAT (FIXED).  `--no-gpu` never implemented the host-side
  #    float -> planar-4:4:4 conversion the GPU path does on the device.
  #    FloatsToRawParallel produced rgba64le at 8 bytes per pixel while the writer
  #    measured the frame with `out_frame_bytes = out_yuv444 ? pixels * 3 : ...`
  #    and spawned the encoder with `-pix_fmt yuv444p`, which is 3.  The host pushed
  #    8 into a pipe declared as 3; the encoder read rgba64le as planar YUV and the
  #    stream desynchronised, so a red band came out green.
  #
  # 2. INPUT FORMAT (FIXED).  `RawToFloatsParallel` took a `const std::uint16_t*`, but
  #    the decoder delivers **8-bit planar** yuv444p -- `in_frame_bytes` is `pixels * 3`
  #    bytes while the converter read 16-bit elements, consuming twice what was written
  #    and leaving the tail as whatever the buffer held.  The device's
  #    `d_sws_yuv_to_rgb16` is now transliterated to the host as `SwsYuvToRgb16` so the
  #    engines agree bit for bit.
  #
  # 3. NON-DETERMINISM (FIXED).  Frames were written to the encoder in worker-completion
  #    order: `done` was a deque of slot indices carrying no position, so the writer took
  #    `done.front()`.  With one batch that was correct by accident; with several it
  #    interleaved them.  `Batch::first_frame` now carries the position and the writer
  #    waits for its turn.
  #
  # A FOURTH surfaced while fixing the second, and it was the worst of them: the
  # host writer's plane stride was the whole BATCH rather than one frame, so
  # `FloatsToYuv444Parallel` wrote every frame's Y then every frame's U then every
  # frame's V.  `writer_convert_threads` is capped at 3, so every `--batch-frames` above
  # 3 was wrong -- including the default of 16.  A wrong-but-STABLE result, so nothing
  # here could ever have seen it.
  #
  # Verified after the fix: `--no-gpu` and CUDA agree BYTE FOR BYTE on video at every
  # --batch-frames from 1 to 30, in yuv444, yuv444-prepass and rgba64, at 1920x1080 as
  # well as 320x180.  The writeups are in docs/DESIGN.md.
  #
  # Ruled out by measurement along the way, each of which was a wrong guess first:
  # the palette (identical -- 3 colours, 100% mean saturation, both engines), the
  # flat octree search (`RiemersmaBlocksCpu` == CUDA, AE 0 on images at 16 and 256
  # colours), and worker count (varies at `--cpu-threads 1`).
  #
  # A note on measuring this, because it cost an hour.  `magick compare -metric AE`
  # on a multi-frame file reports ONE frame, not the clip: it printed
  # "378618 (0.18259)", and 0.18259 x 2073600 -- exactly one 1920x1080 frame -- is
  # 378619.  Over 60 frames the same count would be 0.00304.  Every AE figure
  # recorded while investigating this understated the damage by a factor of about
  # 60, and conclusions drawn from them about magnitude were wrong.  Count bytes
  # over the whole decode instead -- and do not byte-loop 500 MB in PowerShell, which
  # is slow enough to time out.

  if ($c.env) { Remove-Item Env:\RD_YUV444_OUT -EA SilentlyContinue }
  else { $env:RD_YUV444_OUT = '0' }

  $hashes = @{}
  $counts = @{}
  $bad = 0
  $fcFail = 0
  $unavailable = ''
  for ($i = 1; $i -le $Runs; $i++) {
    $out = Join-Path $tmp "det_$($c.name)_$i.mkv"
    # Delete first: a crashed run leaves a stale file, and a stale file hashes fine,
    # so a failing engine would report as a stable one -- the precise opposite.
    Remove-Item -EA SilentlyContinue $out
    # stderr is CAPTURED, not sent to $null.  It used to be discarded, which is exactly
    # why an engine with no device could not be told from a crashed one: both arrived
    # as "nonzero exit", and both counted as failures.  On a runner with no GPU -- which
    # is every runner CI has -- `--engine opencl` refuses BY DESIGN, so this probe would
    # have failed there rather than skipping.
    $text = & $Rdither --video @($c.args) --colors $Colors --video-lossless --no-audio `
              $Clip $out 2>&1 | Out-String
    $rc = $LASTEXITCODE
    if ($text -match 'requested for video but' -or
        $text -match 'this build has no (OpenCL|CUDA)' -or
        $text -match 'no CUDA device') {
      $unavailable = (($text -split "`r?`n" | Where-Object { $_ -match 'error:' } |
                       Select-Object -First 1) -replace '.*error:\s*', '').Trim()
      break
    }
    # A case named for an engine must not report a result for an engine it did not
    # use.  `--engine blocks` on a build with no CUDA device used to FALL BACK TO THE
    # HOST with no diagnostic, so on the CI runner the case named `cuda` was quietly
    # measuring the host.  That is how the host faults below were first noticed: the
    # same bug, counted twice, once failed and once excused.
    #
    # rdither now REFUSES that fallback rather than making it, so the refusal message
    # above normally catches this case first.  This check is kept as a backstop rather
    # than deleted along with the symptom: it reads what the run actually did
    # (`gpu=yes x1` / `gpu=no`) instead of trusting a flag, so it still holds if a
    # silent fallback ever reappears somewhere else.
    if ($c.name -eq 'cuda' -and $text -match 'gpu=no') {
      $unavailable = 'no CUDA device, and --engine blocks fell back to the host, so ' +
                     'this case did not test the engine it is named for'
      break
    }
    if ($rc -ne 0 -or -not (Test-Path $out)) { $bad++; continue }
    $d = Get-DecodedHash $out
    if ($null -eq $d) { $bad++; continue }
    $hashes[$d.hash] = $d.bytes
    # A -1 here means ffprobe FAILED, not that the clip has one frame or minus one
    # frames.  Folding the sentinel into the tally made every failing run report the
    # same "count", so $counts.Count stayed 1, the frame-count check passed, and the
    # probe exited 0 having verified NONE of the thing it exists to verify.  Measured
    # with a stub ffprobe that exits 0 printing nothing: all three cases reported
    # "(-1 frames)" and "3 of 3 cases deterministic", exit 0.
    #
    # This is precisely the -shortest frame-deleting class the probe was written for,
    # and probe-video-exact.ps1 already guards the same sentinel two hundred lines
    # away.  One probe handled it; the other did not.
    $fc = Get-FrameCount $out
    if ($fc -lt 0) { $fcFail++ } else { $counts[$fc] = 1 }
    Remove-Item -EA SilentlyContinue $out
  }
  Remove-Item Env:\RD_YUV444_OUT -EA SilentlyContinue

  $tag = "{0,-8}" -f $c.name
  if ($unavailable) {
    $skipped++
    "  $tag SKIPPED (no usable device: $unavailable)"
    continue
  }
  $verdict = $null
  if ($bad -gt 0) {
    $verdict = @($bad, $Runs, "FAIL", "runs produced no output")
  } elseif ($hashes.Count -ne 1) {
    $verdict = @($hashes.Count, $Runs, "FAIL", "distinct decoded pixel sets -- NOT DETERMINISTIC")
  } elseif ($fcFail -gt 0) {
    $verdict = @($fcFail, $Runs, "FAIL",
                "frame count could not be read from the file (ffprobe?) -- the frame count is one of the three things this probe exists to check, so this is a failure and not a skip")
  } elseif ($counts.Count -ne 1) {
    $verdict = @($counts.Count, $Runs, "FAIL",
                "pixels stable but frame count varies: $((($counts.Keys) | Sort-Object) -join ', ')")
  }

  if ($null -ne $verdict) {
    $fail++
    "  $tag  $($verdict[2])  $($verdict[0]) of $($verdict[1]) $($verdict[3])"
  } else {
    $n = ($counts.Keys)[0]
    $mb = [math]::Round((($hashes.GetEnumerator())[0].Value) / 1MB, 1)
    # "no divergence in N samples", not "deterministic".  A run of identical results
    # shows that nothing diverged while it was watched; it is not a proof, and the
    # wording should not read as one.  The number is on the line so the reader can
    # judge the sample rather than take the verdict on trust.
    "  $tag  ok    1 pixel set over $Runs runs ($mb MB decoded, $n frames) -- no " +
      "divergence observed"
  }
}

Remove-Item $tmp -Recurse -Force -EA SilentlyContinue

''
# A SKIPPED case is not a deterministic case.  The tally below divided by $total,
# which counts every case including the ones that never ran, so a GPU-less build
# printed "3 of 3 cases deterministic" immediately under a line reading
# "cuda  SKIPPED (no usable device)".  That is a false claim in the one line a reader
# skims, and it is the same mistake as the mislabelled FAILED tally below -- a tally
# that miscounts its own coverage is worse than no tally.  Found by reading a real
# CI-configuration log, where the skip was on screen directly above the claim.
$ran = $total - $skipped
if ($fail -eq 0) {
  if ($skipped -gt 0) {
    "{0} of {1} cases deterministic, {2} SKIPPED for want of a device.  Skips are not passes." -f $ran, $total, $skipped
  } else {
    "{0} of {1} cases deterministic." -f $ran, $total
  }
  exit 0
}
# "2 of 3 cases FAILED" was printed when 2 of 3 cases PASSED: the count was
# ($total - $fail) under a label that said FAILED.  It has been misread at least twice
# in this session, once by me while checking whether the OpenCL fault was a real
# failure, and it would be misread by anyone skimming a green run.  A tally that
# mislabels its own direction is worse than no tally.
if ($fail -eq 0) { "all {0} cases deterministic." -f $total } else { "{0} of {1} cases FAILED." -f $fail, ($total - $skipped) }
if ($skipped -gt 0) { "{0} of {1} case(s) SKIPPED for want of a device.  These are not passes." -f $skipped, $total }
exit 1
