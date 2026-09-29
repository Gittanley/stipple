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
# Reports the dither stage (what block length actually affects) and the true wall
# (what the user experiences), because on a CPU-bound pipeline those two can disagree:
# a faster dither helps only if the dither is the critical path.

param(
  [string]$Clip = "$env:TEMP\rd420\v1000.mp4",
  [int]$Reps    = 2
)

$ErrorActionPreference = 'Stop'
$rd = Join-Path (Split-Path $PSScriptRoot -Parent) 'build\Release\rdither.exe'
$out = "$env:TEMP\rd420\blk2_out.mkv"
$log = "$env:TEMP\rd420\blk2.log"
$err = "$env:TEMP\rd420\blk2.err"

if (-not (Test-Path $Clip)) { throw "clip not found: $Clip" }

$arms = @(
  @{ Name = 'default(128)'; E = @() },
  @{ Name = 'blocks 32';    E = @('--blocks','32') },
  @{ Name = 'blocks 256';   E = @('--blocks','256') },
  @{ Name = 'blocks 512';   E = @('--blocks','512') }
)

$rows = @()
foreach ($arm in $arms) {
  $dith = @(); $walls = @()
  for ($r = 1; $r -le $Reps; $r++) {
    Remove-Item -EA SilentlyContinue $out
    $a = @('--video','--engine','blocks','--im-palette','--colors','16','--no-hwaccel',
           '--palette-frames','30') + $arm.E + @($Clip, $out)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $rd -ArgumentList $a -PassThru -NoNewWindow `
          -RedirectStandardOutput $log -RedirectStandardError $err
    $p.WaitForExit()
    $sw.Stop()
    $text = (Get-Content $log -EA SilentlyContinue) -join "`n"
    $d = 0.0; $w = 0.0
    if ($text -match 'dither\s+(\d+)\s*\|')  { $d = [double]$Matches[1] }
    if ($text -match 'wall\s+(\d+)\s+ms')    { $w = [double]$Matches[1] }
    $dith += $d
    $walls += $sw.Elapsed.TotalSeconds
    'rep {0} {1,-14} dither {2,6} ms   true {3,6:N2} s' -f $r, $arm.Name, $d, $sw.Elapsed.TotalSeconds
  }
  $rows += [pscustomobject]@{
    Config  = $arm.Name
    Dither  = [math]::Round(($dith  | Measure-Object -Average).Average, 0)
    DMin    = [math]::Round(($dith  | Measure-Object -Minimum).Minimum, 0)
    DMax    = [math]::Round(($dith  | Measure-Object -Maximum).Maximum, 0)
    TrueS   = [math]::Round(($walls | Measure-Object -Average).Average, 2)
  }
  Remove-Item -EA SilentlyContinue $out
}

''
'=== means over {0} reps of 1000 frames ===' -f $Reps
$rows | Sort-Object Dither | Format-Table -AutoSize | Out-String -Width 200
$best = ($rows | Sort-Object Dither | Select-Object -First 1).Config
$worst = ($rows | Sort-Object Dither -Descending | Select-Object -First 1).Config
$bd = ($rows | Where-Object Config -eq $best).Dither
$wd = ($rows | Where-Object Config -eq $worst).Dither
''
'dither spread: {0} {1:N0} ms  vs  {2} {3:N0} ms   ({4:+0.0;-0.0}%)' -f `
  $best, $bd, $worst, $wd, (100*($bd-$wd)/$wd)
