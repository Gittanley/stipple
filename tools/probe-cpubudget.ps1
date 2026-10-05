# Where do the core-seconds go, and is any fps left?
#
# Round twenty-two's result looked paradoxical: cutting the decoder's work by 114 s
# made the wall 9% WORSE.  The explanation is that the encoder's thread count is
# elastic, so freed reader CPU does not reduce the wall -- it gets absorbed, and the
# encoder's own thread overhead grows.  The question this answers is therefore not
# "is the reader slow" but "what fraction of total CPU is anything we control".
#
# So per configuration this reports:
#   wall, fps          -- the outcome
#   rd core-s          -- rdither's own CPU: reader threads, palette, setup
#   ff core-s          -- ffmpeg's CPU: the palette decoders, the video decoder and
#                         x264 (x264 runs in the encoder child, in-process)
#   total core-s       -- the number that actually sets the wall
#   cores              -- total core-s / wall, i.e. average logical cores busy
#
# HOW CORE-SECONDS ARE COUNTED, and why the old comment here was wrong
# ====================================================================
# The previous version of this header said:
#
#   "Core-seconds are cumulative TotalProcessorTime, read while the processes are
#    alive, so only the final sample matters and no polling error accumulates."
#
# and the code underneath did something else entirely: it took the MAXIMUM over
# samples of the sum of the CPU of the children that happened to be ALIVE at that
# instant.  Those are different quantities, and the max is not an approximation of
# the cumulative total -- it discards work.  rdither spawns ffmpeg for the palette
# decode (src/rd_video.cpp:1274) and one ffmpeg per palette seek (:1730, six at a
# time) BEFORE the long-lived video decoder (:2310) and encoder (:2517).  By the
# time the decode/encode pair is running, the palette children are gone, and their
# CPU has left the sum for good.  It also collapsed two phases with nothing in
# common -- the palette phase and the encode phase -- down to whichever was larger.
#
# Measured, on this machine, two identical runs of the same command back to back
# (both 26.6 fps, so the runs are comparable):
#
#                        rd core-s    ff core-s   total core-s   cores avg
#   max of live sum        22.8         62.6          85.4          3.35
#   sum over all children  23.52        76.89         100.4         4.06
#                                          ^^^^^^^^ 14.2% of the total, lost
#
# The loss is not symmetric noise.  It removes exactly the work that does NOT
# overlap the encoder -- the reader-side work this probe exists to compare -- so a
# change that made the reader cheaper read as "core-seconds did not move".  Two
# conclusions in docs/DESIGN.md rest on that number (:2835 "Total work is conserved
# to within 1.5%", and :2895-2896 "core-seconds did not move, and that settles the
# question"), so it is not cosmetic.  Those two claims were measured with the
# quantity below; they need re-measuring with this one, which is a different number
# and a larger one.
#
# The accounting now lives in tools\rd-cpu-meter.ps1, shared with
# tools\ab-input-mode.ps1, which carried the same loop and the same defect.  It is
# per-PID, cumulative, and sums every child ever observed; see that file for the
# three measurements that make it exact rather than approximate, and for the cost it
# adds to the machine being measured.
#
# Wall comes from a local stopwatch, not (StartTime, ExitTime): those come from the
# process object after exit and are empty on this runtime.  rdither's own `wall` in
# its `busy time` line is a DIFFERENT number -- it excludes process start, CLI
# parsing, the ffprobe feature probes and teardown -- so it is printed separately
# and never silently substituted.

param(
  [string]$Clip = "$env:TEMP\rd420\u8000.mp4",
  [switch]$Sweep,
  # How often to ask WMI for a child process not yet seen.  Measured cost of one
  # filtered Get-CimInstance Win32_Process on this machine: 24.0 ms.  Lower means
  # fewer missed short-lived children and more load on the machine under
  # measurement.  ChildQueries and MaxChildGapS are printed per configuration
  # so the cost is visible in the output.
  #
  # WAS 1200 ms, and that was too coarse to bound the error.  A child that is not
  # enumerated before it exits contributes ZERO, not a partial figure, so the loss
  # bound is (gap x logical cores) per miss.  At 1200 ms on 12 cores that is 14.4
  # core-s per miss against a measured total of ~137 core-s -- the bound was worth
  # ten percent of the whole measurement.  The first run with the old value
  # recorded MaxChildGapS 11.4 s in a 23.8 s run, i.e. nine consecutive queries
  # that saw no child at all, which is most of the run unobserved.
  #
  # 250 ms costs 24 ms of one core per query, about 1.2 percent of the machine
  # under measurement, and bounds a miss at 3.0 core-s.  It is deliberately not
  # tighter: the number being defended here is a share, and 1.2 percent of one
  # core cannot move a share by a meaningful amount.
  [int]$ChildEveryMs = 250,
  # Loop tick.  Only two things happen this often: noticing that rdither exited, and
  # taking the last readable CPU reading of each child.  Tighter means a tighter
  # bound on the last one, at no measurable cost -- a HasExited check and three
  # property reads.  The bound it sets is printed whenever it is not zero.
  [int]$PollMs = 200,
  # Hang guard.  0 waits forever, which is what the old loop did: a wedged rdither
  # meant this probe never reported anything rather than reporting a failure.
  # 900 s matches the ceiling rdither itself puts on its own ffmpeg calls
  # (src/rd_cli.cpp:337), so it is a precedent in the tree rather than a number
  # invented here.  A run measured at 13 s on an idle machine has 69x of headroom.
  [int]$TimeoutSec = 900
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$rd = Join-Path $root 'build\Release\rdither.exe'
$work = "$env:TEMP\rd420"
$out = "$work\probe_out.mkv"
$log = "$work\probe.log"
$err = "$work\probe.err"

# This directory is the default home of the clip this script measures, and it did
# not exist: nothing created it, so a run with any other -Clip failed inside
# Start-Process with a redirection error naming the LOG file rather than the missing
# directory, and a run with the default -Clip died on the clip check first and never
# got that far.  Both hid the same fact.
if (-not (Test-Path $work)) { New-Item -ItemType Directory -Force -Path $work | Out-Null }

. "$PSScriptRoot\rd-cpu-meter.ps1"

if (-not (Test-Path $rd)) { throw "rdither not found: $rd" }
if (-not (Test-Path $Clip)) {
  "cannot run: no clip at $Clip"
  'this probe measures the video pipeline, so there is nothing to measure without one.'
  exit 2
}

function Measure-Config {
  param([string]$Label, [string[]]$Extra)

  Remove-Item -EA SilentlyContinue $out

  # --blocks 32 throughout, because the established A/B for this regime was taken
  # with it and a default-B run is not comparable to those numbers.  Flag drift
  # between probes is how a measurement ends up answering a question nobody asked.
  $a = @('--video','--engine','blocks','--blocks','32','--im-palette','--colors','16') +
       $Extra + @($Clip, $out)
  $p = Start-Process -FilePath $rd -ArgumentList $a -PassThru -NoNewWindow `
        -RedirectStandardOutput $log -RedirectStandardError $err

  $m = Measure-RdRunCpu -Process $p -ChildEveryMs $ChildEveryMs -PollMs $PollMs -TimeoutSec $TimeoutSec

  # Exit code.  It was never read before: a run that failed three seconds in
  # produced a row with fps 0.0 and a plausible-looking core-s total, and the sweep
  # carried on and reported a comparison built from it.
  $exitCode = $null
  try { $exitCode = $p.ExitCode } catch { $exitCode = $null }

  $text = (Get-Content $log -EA SilentlyContinue) -join "`n"

  # A missed fps is NOT a zero.  The old code wrote 0.0 and averaged it in with the
  # real numbers, and the formatted row printed an unmarked "0.0" that reads as a
  # measurement.  It is carried here as a null plus a flag, formatted as n/a, and
  # the caller refuses to publish a table containing one.
  $fps = $null
  # The closing bracket is optional and the label is not matched.  It used to require
# `fps)` immediately, which was true until the summary line gained a parenthetical
# label -- it now reads `(32.0 fps, palette included)` -- so this regex matched
# nothing and every configuration VOIDed.  A missed rate is not a zero, so refusing
# to publish was the correct behaviour, but the cause was a message changing shape
# under a probe that reads it as text.
#
# The PARENTHESIS is what selects the rate, and it has to stay the selector: the
# summary prints two rates, and only the parenthesised one covers the whole job
# (the other is the progress bar's window, which excludes the palette build). A
# looser match on a bare number would take whichever came first, and the two
# differ by ~28 percent on this clip.
if ($text -match '\(([\d.]+) fps[,)]') { $fps = [double]$Matches[1] }

  $stage = (($text -split "`n" | Where-Object { $_ -match 'busy time' }) -join ' ') -replace '\s+',' '

  # rdither's own wall, from inside the process.  Different from the stopwatch and
  # not comparable with it; printed so the gap is visible instead of assumed.
  $inner = $null
  if ($text -match '\|\s*wall\s+(\d+)\s+ms') { $inner = [double]$Matches[1] / 1000.0 }

  [pscustomobject]@{
    Label     = $Label
    ExitCode  = $exitCode
    Wall      = $m.Wall
    InnerWall = if ($null -eq $inner) { 'n/a' } else { [math]::Round($inner, 1) }
    Fps       = $fps
    RdCoreS   = $m.ParentCpu
    FfCoreS   = $m.ChildCpu
    Total     = [math]::Round($m.ParentCpu + $m.ChildCpu, 1)
    Cores     = if ($m.Wall -gt 0) { [math]::Round(($m.ParentCpu + $m.ChildCpu) / $m.Wall, 2) } else { 0 }
    ChildPids = $m.ChildPids
    ChildOk   = $m.ChildFinalised
    Queries   = $m.ChildQueries
    GapS      = $m.MaxChildGapS
    Stage     = $stage.Trim()
  }
}

$configs = if ($Sweep) {
  # Everything still in the pipeline that the summary records as untried or
  # untested in the current regime.  batch/queue/reader are the overlap knobs;
  # they are the only things left that change the wall without touching the preset.
  @(
    @{ L = 'default (yuv444)';        E = @() },
    @{ L = 'queue-depth 4';           E = @('--queue-depth','4') },
    @{ L = 'queue-depth 6';           E = @('--queue-depth','6') },
    @{ L = 'batch-frames 24';         E = @('--batch-frames','24') },
    @{ L = 'batch-frames 32';         E = @('--batch-frames','32') },
    @{ L = 'reader-threads 3';        E = @('--reader-threads','3') },
    @{ L = 'reader-threads 5';        E = @('--reader-threads','5') },
    @{ L = 'queue 4 + batch 24';      E = @('--queue-depth','4','--batch-frames','24') },
    @{ L = 'queue 6 + reader 3';      E = @('--queue-depth','6','--reader-threads','3') }
  )
} else {
  @(
    @{ L = 'yuv444 (default)';       E = @('--input-mode','yuv444') },
    @{ L = 'yuv420';                 E = @('--input-mode','yuv420') },
    @{ L = 'yuv444 enc-threads 4';   E = @('--input-mode','yuv444','--encode-threads','4') },
    @{ L = 'yuv420 enc-threads 4';   E = @('--input-mode','yuv420','--encode-threads','4') }
  )
}

$rows = @()
$void = @()
foreach ($c in $configs) {
  $r = Measure-Config -Label $c.L -Extra $c.E
  $rows += $r

  # Why this row is unusable, named, rather than published as a number.  Ordered by
  # how much it would have misled: a crashed run first, because it produces a
  # plausible core-s total and a 0.0 fps.
  # Why this row is unusable, named, rather than published as a number.  Ordered by
  # how much it would have misled: a crashed run first, because it produces a
  # plausible core-s total and a 0.0 fps.
  #
  # `fail` and `cannot-run` are kept apart because they are different exits.  A run
  # that exited non-zero is a failure.  A run from which no child could be observed
  # is this machine at this scale being unable to answer the question -- the clip is
  # too short to outlast one sampling interval -- which is exit 2, "cannot run",
  # the convention the rest of this suite uses.  Neither is exit 0.
  $why = $null; $cls = ''
  if ($r.ExitCode -ne 0) { $why = "rdither exited $($r.ExitCode)"; $cls = 'fail' }
  elseif ($null -eq $r.Fps) { $why = 'no "(N.N fps)" in the output'; $cls = 'fail' }
  elseif ($r.ChildPids -eq 0) {
    # Not a zero.  Zero means "measured, and the children used no CPU", which is
    # not a thing that happens.  No child observed means the ff column is
    # unmeasured, and an unmeasured column that prints as 0.0 is how this probe
    # used to understate a total by two thirds.  On a short clip this is the
    # expected outcome, not a defect in the pipeline: ffmpeg's children here live
    # for a few hundred ms each, so a run shorter than -ChildEveryMs cannot see
    # one.  Longer clip, or a smaller -ChildEveryMs.
    $why = "no ffmpeg child was ever observed in $($r.Wall) s of sampling at " +
           "$ChildEveryMs ms, so ff core-s is unmeasured rather than zero"
    $cls = 'cannot-run'
  }

  $fpsText = if ($null -eq $r.Fps) { 'n/a' } else { '{0:N1}' -f $r.Fps }
  $line = '{0,-22} {1,6:N1} s  {2,6} fps  rd {3,7} ff {4,7} tot {5,7} core-s  {6,5:N2} cores avg' -f `
    $r.Label, $r.Wall, $fpsText, $r.RdCoreS, $r.FfCoreS, $r.Total, $r.Cores
  if ($why) {
    Write-Host ($line + "   VOID[$cls]: " + $why) -ForegroundColor Red
    $void += @{ Label = $r.Label; Why = $why; Class = $cls }
  } else {
    Write-Host $line
  }

  # A child seen but not yet exited when the last read happened is NOT void.  It is
  # a lower bound, and the size of the bound is knowable: the read is at most one
  # poll interval stale, so at most (poll interval x logical cores) core-s is
  # missing for it.  Published with the bound rather than discarded, because the
  # bound is what lets a reader decide whether the row is good enough.  On the real
  # workload it is 0 -- measured 3 of 3 children finalised at exit on a 605-frame
  # 1080p clip -- so this line is normally just the confirmation of that.
  $boundNote = ''
  if ($r.ChildPids -gt $r.ChildOk) {
    $worst = ($r.ChildPids - $r.ChildOk) * ($PollMs / 1000.0) * [Environment]::ProcessorCount
    $boundNote = "   ff core-s is a LOWER BOUND: $($r.ChildPids - $r.ChildOk) of " +
                 "$($r.ChildPids) children were last read within $PollMs ms of exit, " +
                 "so at most $([math]::Round($worst,1)) core-s may be missing"
    Write-Host ('    children {0} (finalised at exit: {1})' -f $r.ChildPids, $r.ChildOk) -ForegroundColor Yellow
    Write-Host ('    ' + $boundNote.Trim()) -ForegroundColor Yellow
  } else {
    '    children {0} (all finalised at exit)' -f $r.ChildPids
  }
  '    inner wall {0,6} s   WMI queries {1}   longest gap between samples that saw a child {2} s' -f `
    $r.InnerWall, $r.Queries, $r.GapS
}

''
'=== stage breakdown (ms), in the order printed ==='
$rows | ForEach-Object { '{0,-22} {1}' -f $_.Label, $_.Stage }
''
'outer wall is this script''s stopwatch: process start, CLI parse, the ffprobe'
'feature probes, and teardown are inside it and not inside rdither''s own `wall`.'
'The two are printed separately above and must not be compared with each other.'

if ($void.Count -gt 0) {
  ''
  "VOID: $($void.Count) of $($rows.Count) configurations produced no usable measurement."
  foreach ($v in $void) { "  {0,-22} [{1}] {2}" -f $v.Label, $v.Class, $v.Why }
  'Refusing to publish a table that mixes real rows with rows whose numbers are missing.'
  if (@($void | Where-Object { $_.Class -eq 'fail' }).Count -gt 0) { exit 1 }
  exit 2
}
exit 0