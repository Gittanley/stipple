# Does --blocks N change throughput?  A cheap, decisive version.
#
# The previous attempt at this ran 12 passes over an 8000-frame clip -- about half an
# hour -- to answer a question about one stage.  Unnecessary: block length only
# affects the dither, so a 1000-frame clip measures it just as well, and the palette
# is pinned to 30 samples so it does not vary between arms.
#
# The earlier script hung rather than answering, so this one also prints a line per rep
# as it goes, so partial results are visible if it is cut short.
#
# The question: the user ran a bare `--video` (block length defaults to 128) and got
# 55.5 fps, while every benchmark in this repo's history used `--blocks 32`.  If 32 is
# genuinely slower, every benchmark number in the record is pessimistic.
#
# Reports two walls, because conflating them is how this script used to misreport:
#
#   dither     rdither's own `dither` stage, from its `busy time` line.  The only
#              thing block length can affect, and the number that answers the question.
#   pipeline   rdither's own `wall`, from the same line.  Comparable with every
#              `wall` figure in docs/DESIGN.md, because it is the same quantity.
#   outer      this script's stopwatch, from Start-Process to process exit.  NOT
#              comparable with any `wall` in the docs: it additionally spans process
#              spawn, CUDA context creation, CLI parsing, the ffprobe feature probes
#              and teardown.  Measured gap on this machine, one 605-frame 1080p run:
#              outer 13.13 s against pipeline 11.24 s, so about 1.9 s -- 14% -- of
#              which the pipeline figure knows nothing.
#
# The old version printed the outer stopwatch in a column headed `TrueS` and called the
# other figure "what the user experiences", which is not a comparison anyone could make
# with the record.  It also parsed `wall` into a variable and then never used it.

param(
  [string]$Clip = "$env:TEMP\rd420\v1000.mp4",
  [int]$Reps    = 2,
  # Minimum VALID reps per arm before that arm may be ranked.  Default is $Reps, i.e.
  # every rep must have produced a parsed, positive dither time.  Lower it to accept a
  # design where some reps are expected to fail; the count of valid reps is always
  # printed either way.
  [int]$MinReps = 0,
  # Hang guard.  WaitForExit() with no timeout meant a wedged rdither hung this probe
  # forever and reported nothing, which is the failure mode this file was written to
  # escape.  900 s matches the ceiling rdither puts on its own ffmpeg calls
  # (src/rd_cli.cpp:337).  A rep measured at 13 s on an idle machine has 69x headroom.
  [int]$TimeoutSec = 900
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$rd = Join-Path $root 'build\Release\rdither.exe'
$work = "$env:TEMP\rd420"
$out = "$work\blk2_out.mkv"
$log = "$work\blk2.log"
$err = "$work\blk2.err"

# Never created by anything.  Every default path above lives in this directory, so a
# run that got as far as Start-Process died on a redirection error naming the LOG file
# rather than on the missing directory -- and a run using the default -Clip died on the
# clip check first and never got that far at all.
if (-not (Test-Path $work)) { New-Item -ItemType Directory -Force -Path $work | Out-Null }

if ($MinReps -lt 1) { $MinReps = $Reps }
if (-not (Test-Path $rd)) { throw "rdither not found: $rd" }
if (-not (Test-Path $Clip)) {
  "cannot run: no clip at $Clip"
  'this probe measures the video pipeline, so there is nothing to measure without one.'
  exit 2
}
if ($Reps -lt 1) { throw "-Reps must be at least 1 (got $Reps)" }

$arms = @(
  @{ Name = 'default(128)'; E = @() },
  @{ Name = 'blocks 32';    E = @('--blocks','32') },
  @{ Name = 'blocks 256';   E = @('--blocks','256') },
  @{ Name = 'blocks 512';   E = @('--blocks','512') }
)

# One rep.  Returns the parsed numbers, or nulls plus the reason, so a caller can
# decide instead of averaging a missing value into a mean.
function Measure-Rep {
  param([string]$Name, [int]$Index, [string[]]$Extra)

  Remove-Item -EA SilentlyContinue $out
  $a = @('--video','--engine','blocks','--im-palette','--colors','16','--no-hwaccel',
         '--palette-frames','30') + $Extra + @($Clip, $out)
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $p = Start-Process -FilePath $rd -ArgumentList $a -PassThru -NoNewWindow `
        -RedirectStandardOutput $log -RedirectStandardError $err

  # A timeout, and a kill.  WaitForExit() with no argument is the difference between
  # "this probe reports a failed rep" and "this probe never finishes".
  if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    try { $p.Kill($true) } catch { try { $p.Kill() } catch { } }
    try { $p.WaitForExit(15000) | Out-Null } catch { }
    $sw.Stop()
    return [pscustomobject]@{ Ok = $false; Why = "rdither did not exit within $TimeoutSec s and was killed"; Wall = $sw.Elapsed.TotalSeconds }
  }
  $sw.Stop()

  # The exit code.  It was never read, so a run that failed on a bad argument produced
  # no dither line, no wall line, and two 0.0 values that were then averaged in with
  # the real ones.
  $exitCode = $null
  try { $exitCode = $p.ExitCode } catch { $exitCode = $null }

  $text = (Get-Content $log -EA SilentlyContinue) -join "`n"

  $d = $null
  if ($text -match 'dither\s+(\d+)\s*\|') { $d = [double]$Matches[1] }
  $w = $null
  if ($text -match 'wall\s+(\d+)\s+ms')   { $w = [double]$Matches[1] / 1000.0 }

  $why = $null
  if ($null -ne $exitCode -and $exitCode -ne 0) { $why = "rdither exited $exitCode" }
  elseif ($null -eq $d) { $why = 'no "dither NNNN |" in the output' }
  elseif ($d -le 0) {
    # A dither stage of 0 ms is not a fast dither; with --engine blocks over a video
    # clip it is a line this script failed to read.  Either way it must not become the
    # best result in the table, which is precisely what sorting it into first place did.
    $why = "the dither stage reported $d ms, which is a missing measurement rather than a fast stage"
  }
  elseif ($null -eq $w) { $why = 'no "wall NNNN ms" in the output' }

  [pscustomobject]@{
    Ok = ($null -eq $why); Why = $why
    Dither = $d; Pipeline = $w; Outer = $sw.Elapsed.TotalSeconds; ExitCode = $exitCode
  }
}

$rows = @()
foreach ($arm in $arms) {
  $dith = @(); $pipes = @(); $outers = @(); $bad = @()
  for ($r = 1; $r -le $Reps; $r++) {
    $m = Measure-Rep -Name $arm.Name -Index $r -Extra $arm.E
    if (-not $m.Ok) {
      # Printed, named, and NOT averaged.  A missing rep used to become a 0.0 that
      # counted three times over: once in the mean, once in the minimum, once in the
      # maximum -- and the minimum is the column the ranking sorted on.
      'rep {0} {1,-14} VOID: {2}' -f $r, $arm.Name, $m.Why | Write-Host
      $bad += $m.Why
      continue
    }
    $dith += $m.Dither; $pipes += $m.Pipeline; $outers += $m.Outer
    'rep {0} {1,-14} dither {2,6} ms   pipeline {3,6:N2} s   outer {4,6:N2} s' -f `
      $r, $arm.Name, $m.Dither, $m.Pipeline, $m.Outer | Write-Host
  }
  $rows += [pscustomobject]@{
    Config   = $arm.Name
    Valid    = $dith.Count
    Void     = $bad.Count
    # int, not a rounded double: Format-Table then renders 5465 rather than 5465,00,
    # and the two are different-looking numbers for the same measurement.
    Dither   = if ($dith.Count) { [int][math]::Round(($dith | Measure-Object -Average).Average, 0) } else { $null }
    DMin     = if ($dith.Count) { [int][math]::Round(($dith | Measure-Object -Minimum).Minimum, 0) } else { $null }
    DMax     = if ($dith.Count) { [int][math]::Round(($dith | Measure-Object -Maximum).Maximum, 0) } else { $null }
    Pipeline = if ($pipes.Count) { [math]::Round(($pipes | Measure-Object -Average).Average, 2) } else { $null }
    Outer    = if ($outers.Count) { [math]::Round(($outers | Measure-Object -Average).Average, 2) } else { $null }
  }
}

''
# The clip is named, not a frame count.  The old line hardcoded "1000 frames" while
# printing whatever -Clip was given, so a run on a 10-frame clip reported "means over
# 1 reps of 1000 frames" -- a claim in the output about the input that was simply false.
'=== means over {0} rep(s) of {1} ===' -f $Reps, (Split-Path $Clip -Leaf)
$rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
'  pipeline = rdither''s own `wall`, the quantity every `wall` in docs/DESIGN.md quotes.'
'  outer    = this script''s stopwatch, which also spans process spawn, CUDA context'
'             creation, CLI parsing, the ffprobe feature probes and teardown.  It is'
'             not the same measurement and the two columns must not be differenced.'

# ---- the ranking ----------------------------------------------------------
#
# The old ranking was `Sort-Object Dither | Select-Object -First 1`, on a column that
# could hold 0.0 for a rep that never ran.  Ascending sort, first element: a missing
# measurement is the smallest number in the table and therefore always wins.  So the
# ranking below refuses to run unless every arm has at least -MinReps valid reps and a
# strictly positive mean, and then picks the minimum by explicit comparison rather than
# by sorting, so there is no path by which an unranked value can be promoted.
$unranked = @($rows | Where-Object { $_.Valid -lt $MinReps -or $null -eq $_.Dither -or $_.Dither -le 0 })
if ($unranked.Count -gt 0) {
  ''
  "CANNOT RANK: $($unranked.Count) of $($rows.Count) arms have fewer than $MinReps valid reps, or a non-positive dither mean."
  foreach ($u in $unranked) {
    '  {0,-14} valid reps {1}/{2}, mean dither {3} ms' -f $u.Config, $u.Valid, $Reps, `
      $(if ($null -eq $u.Dither) { 'n/a' } else { $u.Dither })
  }
  'No best/worst is named.  Naming one over a void arm is the failure being fixed.'
  exit 1
}

$ranked = @($rows)
$best = $ranked[0]; $worst = $ranked[0]
foreach ($x in $ranked) {
  if ($x.Dither -lt $best.Dither)  { $best = $x }
  if ($x.Dither -gt $worst.Dither) { $worst = $x }
}

''
# Divided by the WORST, and only when it is positive.  The old line was
# 100*($bestD - $worstD)/$worstD, i.e. always negative or zero, labelled as if it were
# a magnitude: an all-zero sweep divided by zero and threw, and a sweep where one arm
# failed to parse reported exactly -100.0% -- a fabricated number, produced by the
# sentinel this script has just spent the rest of its length refusing to publish.
if ($worst.Dither -gt 0) {
  $spreadMs = $worst.Dither - $best.Dither
  'dither stage: {0} {1:N0} ms  vs  {2} {3:N0} ms' -f $best.Config, $best.Dither, $worst.Config, $worst.Dither
  'dither spread: {0:N0} ms, which is {1:N1}% of the worst arm ({2})' -f `
    $spreadMs, (100 * $spreadMs / $worst.Dither), $worst.Dither
} else {
  # Unreachable given the gate above, and kept because a guard that cannot be reached
  # is indistinguishable from a guard that was never written.
  "dither spread: undefined -- the worst arm reports $($worst.Dither) ms and a ratio against zero has no meaning."
  exit 1
}

if ($Reps -lt 2) {
  ''
  'Only 1 rep: the per-arm minimum and maximum are the same sample, so the spread'
  'above is not an estimate of run-to-run variation and must not be read as one.'
}
Remove-Item -EA SilentlyContinue $out
exit 0