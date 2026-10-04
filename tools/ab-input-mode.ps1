# Interleaved A/B: does --input-mode yuv420 actually help, and is it worth adopting?
#
# Two measurements of the same comparison now disagree, in opposite directions:
#
#   round twenty-two, 2 reps, straight after a build   yuv444 45.7  yuv420 41.8  (-8.5%)
#   budget probe,   1 rep, idle machine                 yuv444 47.4  yuv420 52.0  (+9.7%)
#
# Both are single-digit-percent effects on a machine whose absolute fps drifts several
# percent under load, which is exactly the regime where a one-shot number means nothing.
# So: alternate the two configurations and repeat, so that any drift in machine load
# hits both arms equally instead of whichever ran during the busy patch.
#
# THREE THINGS THIS SCRIPT NOW CHECKS THAT IT DID NOT
# =====================================================
#
# 1. PIXELS, within each arm.  It compared fps only.  Two reps of the SAME
#    configuration that disagree on a single pixel means the pipeline is
#    nondeterministic, and an fps delta measured across such a pair is a comparison
#    of two different answers.  That is now a failure, not a footnote.  Reps of one
#    arm are hashed over every decoded sample and must agree; every rep of an arm
#    must hash identically.
#
# 2. PIXELS, across the arms.  Reported, always, because yuv420 and yuv444 are NOT
#    a like-for-like swap: 4:2:0 has half the chroma resolution, so the flag changes
#    the picture.  The script therefore prints, before any timing conclusion, whether
#    the two arms' outputs differ -- so nobody reads a performance delta as a free
#    win.  This is reported rather than asserted, because a difference here is the
#    documented purpose of the flag, not a defect.
#
# 3. THE CORE-SECONDS FIGURE IS A DIFFERENT NUMBER FROM THE ONE IN THE HEADER BELOW.
#    The old header cited "~1590 core-s, conserved across all four configurations"
#    from tools\probe-cpubudget.ps1.  That number was produced by taking the maximum
#    over samples of the sum of the CPU of the children that happened to be alive,
#    which drops every ffmpeg child that had already exited -- the palette decode
#    and the palette seeks -- and so understates the total.  Measured on this machine
#    on two identical runs of one command (both 26.6 fps, so comparable):
#
#        max of live sum        rd 22.80  ff  62.60  total  85.4 core-s   3.35 cores avg
#        sum over all children  rd 23.52  ff  76.89  total 100.4 core-s   4.06 cores avg
#
#    So the ~1590 figure is a floor, not a measurement, and the "conserved" claim
#    cannot be re-derived from it.  The accounting now lives in
#    tools\rd-cpu-meter.ps1 and this script uses it; the number has to be re-measured
#    before anything is concluded from core utilisation here.  That is why the print
#    below says the core column is a NEW baseline rather than comparing against 1590.
#
# A MISSED NUMBER IS NOT A ZERO.  The old code initialised $fps to 0.0 when the
# regex missed, averaged it in with the real values, and published a delta computed
# from it.  A run that produced no "(N.N fps)" line is now a failed run: the arm is
# marked VOID, the row is printed with n/a rather than a digit, and the script exits
# non-zero rather than reporting a comparison built on a missing number.

param(
  [string]$Clip = "$env:TEMP\rd420\u8000.mp4",
  [int]$Reps    = 3,
  # Loop tick for the CPU meter; also the bound on how stale the last reading of a
  # child can be.  See tools\rd-cpu-meter.ps1.
  [int]$PollMs = 200,
  # How often to ask WMI for an unseen child.  See tools\rd-cpu-meter.ps1 for the
  # measured cost (24.0 ms per filtered call on this machine).
  [int]$ChildEveryMs = 1200,
  # Hang guard.  900 s matches the ceiling rdither puts on its own ffmpeg calls
  # (src/rd_cli.cpp:337).  0 waits forever, which is what this used to do.
  [int]$TimeoutSec = 900
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$rd = Join-Path $root 'build\Release\rdither.exe'
$work = "$env:TEMP\rd420"
$out = "$work\ab_out.mkv"
$log = "$work\ab.log"
$err = "$work\ab.err"

# Never created by anything.  A run with any other -Clip failed inside Start-Process
# with a redirection error naming the LOG file, so the missing directory presented as
# a problem with a file that does not exist.
if (-not (Test-Path $work)) { New-Item -ItemType Directory -Force -Path $work | Out-Null }
. "$PSScriptRoot\rd-cpu-meter.ps1"

if (-not (Test-Path $rd)) { throw "rdither not found: $rd" }
if (-not (Test-Path $Clip)) {
  "cannot run: no clip at $Clip"
  'this probe measures the video pipeline, so there is nothing to measure without one.'
  exit 2
}
if ($Reps -lt 1) { throw "-Reps must be at least 1 (got $Reps)" }

$arms = @(
  @{ Mode = 'yuv444' },
  @{ Mode = 'yuv420' }
)

# Hash the DECODED PIXELS of the whole output, so "the same output" means the same
# picture rather than the same container bytes.
#
# WHY ffmpeg AND NOT IMAGEMICK.  The first version of this used
# `magick "$mkv" -depth 8 "rgba:$raw"`, which decodes EVERY frame to raw RGBA: for the
# 605-frame 1920x1080 clip that is 4.67 GB per rep, measured (2.5 GB written before I
# stopped it), and 20 GB across four reps.  ImageMagick is also the memory-hungry
# process in this repository -- 8 GB of RAM on this machine per the owner -- so a check
# that runs it four times back to back is the wrong instrument even when it finishes.
# `ffmpeg -f rawvideo -pix_fmt rgba -f md5 -` hashes every decoded sample of every
# frame, streams rather than buffers, writes no file, and measured 11.3 s for the same
# clip.  It also removes this probe's dependency on magick entirely.
#
# WHY THE WHOLE STREAM AND NOT A SAMPLELED FRAME.  A cheaper option is
# `magick "$mkv[0]"`, and it was tried first.  It returns one frame in ~1.1 s -- and
# `magick "$mkv[0]" "$mkv[300]"` returns two frames that are BYTE-IDENTICAL, while
# ffmpeg's own frame-accurate extraction of frames 0, 300 and 604 of the same file
# gives three different images.  So ImageMagick was handing back the same frame for
# every selector and a frame-sampled check would have silently compared frame 0 and
# nothing else, while claiming to compare the output.  Unverifiable instrumentation is
# worse than a slower verifiable one.
#
# `-pix_fmt rgba` so the hash is over materialised RGB rather than over whatever
# sample format the container declares: the point is "the same picture".
#
# Non-zero exit and empty output on a file ffmpeg cannot open, which is checked, so a
# failed hash can never be mistaken for two matching ones.
function Get-VideoPixelHash([string]$mkv) {
  $out = (& ffmpeg -v error -i $mkv -f rawvideo -pix_fmt rgba -f md5 - 2>$null) -join ''
  if ($LASTEXITCODE -ne 0) { return $null }
  if ($out -notmatch 'MD5=([0-9a-f]{32})') { return $null }
  return $Matches[1]
}

$res  = @{ yuv444 = @(); yuv420 = @() }
$util = @{ yuv444 = @(); yuv420 = @() }
$hash = @{ yuv444 = @(); yuv420 = @() }
$repFiles = @{ yuv444 = @(); yuv420 = @() }
$void = @()

for ($r = 1; $r -le $Reps; $r++) {
  foreach ($arm in $arms) {
    # A per-rep output path.  The old script reused one $out for every rep and then
    # deleted it, so the two arms were being compared through whichever file happened
    # to be left over -- and the pixel check below compares files, which makes that
    # fatal rather than untidy.
    $this = "$($work)\ab_$($arm.Mode)_$($r).mkv"
    Remove-Item -EA SilentlyContinue $this
    $a = @('--video','--engine','blocks','--blocks','32','--im-palette','--colors','16',
           '--input-mode',$arm.Mode,$Clip,$this)
    $p = Start-Process -FilePath $rd -ArgumentList $a -PassThru -NoNewWindow `
          -RedirectStandardOutput $log -RedirectStandardError $err

    $m = Measure-RdRunCpu -Process $p -ChildEveryMs $ChildEveryMs -PollMs $PollMs -TimeoutSec $TimeoutSec

    # Exit code, which was never read.  A run that failed immediately used to
    # contribute fps 0.0 and a real-looking core-s total to the mean.
    $exitCode = $null
    try { $exitCode = $p.ExitCode } catch { $exitCode = $null }

    $text = (Get-Content $log -EA SilentlyContinue) -join "`n"
    $fps = $null
    if ($text -match '\(([\d.]+) fps\)') { $fps = [double]$Matches[1] }

    $why = $null; $cls = ''
    if ($exitCode -ne 0) { $why = "rdither exited $exitCode"; $cls = 'fail' }
    elseif ($null -eq $fps) { $why = 'no "(N.N fps)" in the output'; $cls = 'fail' }
    elseif ($m.ChildPids -eq 0) {
      $why = "no ffmpeg child observed in $($m.Wall) s at $ChildEveryMs ms, so core-s is unmeasured"
      $cls = 'cannot-run'
    }
    elseif (-not (Test-Path $this)) { $why = 'rdither exited 0 but wrote no output file'; $cls = 'fail' }

    if ($why) {
      Write-Host ('rep {0} {1,-7} {2,6} s  VOID[{3}]: {4}' -f $r, $arm.Mode, $m.Wall, $cls, $why) -ForegroundColor Red
      $void += [pscustomobject]@{ Rep = $r; Arm = $arm.Mode; Class = $cls; Why = $why }
      continue
    }

    $h = Get-VideoPixelHash $this
    if ($null -eq $h) {
      # The file exists and rdither exited 0, but nothing could hash it.  Treating
      # that as "no difference" would be the same error as treating a missing number
      # as zero, one level down.
      Write-Host ('rep {0} {1,-7} VOID[fail]: wrote {2} but ffmpeg could not decode it, so its pixels are unverified' -f `
        $r, $arm.Mode, (Split-Path $this -Leaf)) -ForegroundColor Red
      $void += [pscustomobject]@{ Rep = $r; Arm = $arm.Mode; Class = 'fail'
                                Why = 'output written but not decodable, so its pixels are unverified' }
      continue
    }

    # Average logical cores busy = total core-seconds / wall.  Computed HERE, not read
    # off the meter: Measure-RdRunCpu returns ParentCpu, ChildCpu and Wall and does not
    # return a Cores property, so `$m.Cores` was $null, and this line then fed $null into
    # the per-arm mean.  The mean printed as empty and the percentage guard below
    # correctly refused to divide by it -- which is how a null got this far without a
    # crash, and is the only reason it was noticed at all.
    $total = $m.ParentCpu + $m.ChildCpu
    $cores = if ($m.Wall -gt 0) { [math]::Round($total / $m.Wall, 2) } else { $null }

    $res[$arm.Mode]  += $fps
    $util[$arm.Mode] += $cores
    $hash[$arm.Mode] += $h
    $repFiles[$arm.Mode] += $this
    'rep {0} {1,-7} {2,6:N1} fps  wall {3,6:N1} s  rd {4,7:N1} ff {5,7:N1} tot {6,7:N1} core-s  {7,5:N2} cores avg' -f `
      $r, $arm.Mode, $fps, $m.Wall, $m.ParentCpu, $m.ChildCpu, $total, $cores
  }
}

''
'=== means over {0} interleaved rep(s) ===' -f $Reps
foreach ($mode in @('yuv444','yuv420')) {
  $f = $res[$mode]; $c = $util[$mode]
  if ($f.Count -eq 0) {
    '  {0,-7} no usable runs' -f $mode
    continue
  }
  '  {0,-7} fps {1,6:N1}  (runs: {2})' -f $mode, ($f | Measure-Object -Average).Average,
      (($f | ForEach-Object { '{0:N1}' -f $_ }) -join ', ')
  # Guarded on $c too, not just on $f: an empty cores array formats as an empty string
  # and an all-empty average divides by nothing, which is what the first version of
  # this line did after a $m.Cores that never existed was averaged in as $null.
  if ($c.Count -gt 0) {
    '  {0,-7} cores avg {1,5:N2}  (runs: {2})' -f $mode, ($c | Measure-Object -Average).Average,
        (($c | ForEach-Object { '{0:N2}' -f $_ }) -join ', ')
  } else {
    '  {0,-7} cores avg not measured on any run' -f $mode
  }
}

# Within each arm, every rep must be the same pixels.  It used to be assumed.
'=== pixels ==='
$bad = @()
foreach ($mode in @('yuv444','yuv420')) {
  $hs = @($hash[$mode] | Where-Object { $null -ne $_ })
  if ($hs.Count -eq 0) { '  {0,-7} no output hashed' -f $mode; continue }
  $distinct = @($hs | Sort-Object -Unique).Count
  if ($distinct -ne 1) {
    '  {0,-7} {1} DISTINCT pixel sets over {2} runs -- the pipeline is NOT reproducible' -f `
      $mode, $distinct, $hs.Count
    $bad += $mode
  } else {
    '  {0,-7} 1 pixel set over {1} run(s)' -f $mode, $hs.Count
  }
}
foreach ($i in 0..($repFiles['yuv444'].Count - 1)) {
  if ($i -lt $repFiles['yuv420'].Count) {
    $h4 = $hash['yuv444'][$i]; $h2 = $hash['yuv420'][$i]
    if ($null -ne $h4 -and $null -ne $h2 -and $h4 -ne $h2) {
      '  across arms: rep {0} yuv444 and yuv420 differ ({1} vs {2})' -f ($i + 1), $h4.Substring(0,12), $h2.Substring(0,12)
    }
  }
}
if ($hash['yuv444'].Count -gt 0 -and $hash['yuv420'].Count -gt 0) {
  $anyDiff = $false
  for ($i = 0; $i -lt [math]::Min($hash['yuv444'].Count, $hash['yuv420'].Count); $i++) {
    if ($hash['yuv444'][$i] -ne $hash['yuv420'][$i]) { $anyDiff = $true }
  }
  if ($anyDiff) {
    ''
    'yuv420 does NOT reproduce yuv444''s pixels, which is what 4:2:0 chroma subsampling'
    'is for.  So this is not a like-for-like swap: the fps comparison below measures a'
    'change of input mode, not a free speedup, and any decision to adopt it is a'
    'decision to change the picture as well.'
  }
}

# ---- the comparison --------------------------------------------------------
$avail = @('yuv420','yuv444') | Where-Object { $res[$_].Count -gt 0 }
if ($avail.Count -lt 2) {
  ''
  'cannot compare: only ' + $avail.Count + ' of 2 arms produced a usable run.'
  'A one-arm comparison has no delta, and printing one against a missing arm is how'
  'this script used to publish a figure built on an unparsed 0.0.'
  exit 2
}

$a = ($res['yuv420'] | Measure-Object -Average).Average
$b = ($res['yuv444'] | Measure-Object -Average).Average
$ca = ($util['yuv420'] | Measure-Object -Average).Average
$cb = ($util['yuv444'] | Measure-Object -Average).Average
''
# The old line divided by $b with no guard, and $b was a mean over values that could
# all be 0.0.  Both the absolute difference and the ratio are printed, and the ratio is
# omitted rather than invented when the divisor is not positive.
if ($b -gt 0) {
  'yuv420 vs yuv444: {0:+0.0;-0.0;0.0} fps ({1:+0.0;-0.0;0.0}%)' -f ($a - $b), (100*($a-$b)/$b)
} else {
  'yuv420 vs yuv444: {0:+0.0;-0.0;0.0} fps  (no percentage: the yuv444 mean is {1}, and a ratio against zero has no meaning)' -f ($a - $b), $b
}
if ($cb -gt 0) {
  'core utilisation: {0:+0.00;-0.00;0.00} cores ({1:+0.0;-0.0;0.0}%)' -f ($ca - $cb), (100*($ca-$cb)/$cb)
} else {
  'core utilisation: {0:+0.00;-0.00;0.00} cores  (no percentage: the yuv444 mean is {1})' -f ($ca - $cb), $cb
}
''
'The core-seconds column above is a NEW baseline, not a comparison against the ~1590'
'figure quoted at the top of this file: that figure came from an accounting that'
'dropped every ffmpeg child which had already exited.  Re-measure before concluding'
'anything about whether core-seconds moved.  See tools\rd-cpu-meter.ps1.'
'This is a {0}-rep design on a single-digit-percent effect; the spread between reps' -f $Reps
'is printed above and is the honest measure of how much this design can resolve.'

foreach ($f in $repFiles.Values) { foreach ($x in $f) { Remove-Item -EA SilentlyContinue $x } }

if ($bad.Count -gt 0) {
  ''
  "FAILED: $($bad -join ', ') produced more than one pixel set across reps of the same"
  'configuration.  The pipeline is not reproducible, so the fps comparison above is a'
  'comparison of different answers and cannot be read as a speedup.'
  exit 1
}
if ($void.Count -gt 0) {
  ''
  "VOID: $($void.Count) of $($Reps * $arms.Count) runs produced no usable measurement."
  foreach ($v in $void) { '  rep {0} {1,-7} [{2}] {3}' -f $v.Rep, $v.Arm, $v.Class, $v.Why }
  'Refusing to report a delta built from the runs that did produce a number.'
  if (@($void | Where-Object { $_.Class -eq 'fail' }).Count -gt 0) { exit 1 }
  exit 2
}
exit 0