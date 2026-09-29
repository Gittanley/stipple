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
# The budget probe also showed total CPU is conserved across all four configurations at
# ~1590 core-s, with the wall decided purely by average cores busy (9.15 for yuv444,
# 10.01 for yuv420).  That is a testable prediction: if the mechanism is that a faster
# reader keeps the dither and encoder fed more continuously, then yuv420's win should
# show up as higher average core utilisation, and it should be reproducible.

param(
  [string]$Clip = "$env:TEMP\rd420\u8000.mp4",
  [int]$Reps    = 3
)

$ErrorActionPreference = 'Stop'
$rd = Join-Path (Split-Path $PSScriptRoot -Parent) 'build\Release\rdither.exe'
$out = "$env:TEMP\rd420\ab_out.mkv"
$log = "$env:TEMP\rd420\ab.log"
$err = "$env:TEMP\rd420\ab.err"

if (-not (Test-Path $Clip)) { throw "clip not found: $Clip" }

$arms = @(
  @{ Mode = 'yuv444' },
  @{ Mode = 'yuv420' }
)

$res = @{ yuv444 = @(); yuv420 = @() }
$util = @{ yuv444 = @(); yuv420 = @() }

for ($r = 1; $r -le $Reps; $r++) {
  foreach ($arm in $arms) {
    Remove-Item -EA SilentlyContinue $out
    $a = @('--video','--engine','blocks','--blocks','32','--im-palette','--colors','16',
           '--input-mode',$arm.Mode,$Clip,$out)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $rd -ArgumentList $a -PassThru -NoNewWindow `
          -RedirectStandardOutput $log -RedirectStandardError $err
    $rdCpu = 0.0; $ffCpu = 0.0
    while (-not $p.HasExited) {
      $me = Get-Process -Id $p.Id -EA SilentlyContinue
      if ($me) { $rdCpu = $me.TotalProcessorTime.TotalSeconds }
      $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" -EA SilentlyContinue |
                Select-Object -ExpandProperty ProcessId)
      $sum = 0.0
      foreach ($k in $kids) {
        $kp = Get-Process -Id $k -EA SilentlyContinue
        if ($kp) { $sum += $kp.TotalProcessorTime.TotalSeconds }
      }
      if ($sum -gt $ffCpu) { $ffCpu = $sum }
      Start-Sleep -Milliseconds 1200
    }
    $sw.Stop()
    $p.WaitForExit()

    $text = (Get-Content $log -EA SilentlyContinue) -join "`n"
    $fps = 0.0
    if ($text -match '\(([\d.]+) fps\)') { $fps = [double]$Matches[1] }
    $wall = $sw.Elapsed.TotalSeconds
    $cores = if ($wall -gt 0) { ($rdCpu + $ffCpu) / $wall } else { 0 }

    $res[$arm.Mode] += $fps
    $util[$arm.Mode] += $cores
    'rep {0} {1,-7} {2,6:N1} fps  wall {3,6:N1} s  tot {4,7:N0} core-s  {5,5:N2} cores avg' -f `
      $r, $arm.Mode, $fps, $wall, ($rdCpu + $ffCpu), $cores
    Remove-Item -EA SilentlyContinue $out
  }
}

''
'=== means over {0} interleaved reps ===' -f $Reps
foreach ($m in @('yuv444','yuv420')) {
  $f = $res[$m]; $c = $util[$m]
  '  {0,-7} fps {1,6:N1}  (runs: {2})' -f $m, ($f | Measure-Object -Average).Average,
      (($f | ForEach-Object { '{0:N1}' -f $_ }) -join ', ')
  '  {0,-7} cores avg {1,5:N2}  (runs: {2})' -f $m, ($c | Measure-Object -Average).Average,
      (($c | ForEach-Object { '{0:N2}' -f $_ }) -join ', ')
}
$a = ($res['yuv420'] | Measure-Object -Average).Average
$b = ($res['yuv444'] | Measure-Object -Average).Average
''
'yuv420 vs yuv444: {0:+0.0;-0.0;0.0} fps ({1:+0.0;-0.0;0.0}%)' -f ($a - $b), (100*($a-$b)/$b)
$ca = ($util['yuv420'] | Measure-Object -Average).Average
$cb = ($util['yuv444'] | Measure-Object -Average).Average
'core utilisation: {0:+0.00;-0.00;0.00} cores ({1:+0.0;-0.0;0.0}%)' -f ($ca - $cb), (100*($ca-$cb)/$cb)
