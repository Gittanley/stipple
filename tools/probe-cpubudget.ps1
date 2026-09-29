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
#   ff core-s          -- ffmpeg's CPU: decode AND x264 (x264 runs in-process)
#   total core-s       -- the number that actually sets the wall
#   cores              -- total core-s / wall, i.e. average logical cores busy
#
# Wall comes from a local stopwatch, not (StartTime, ExitTime): those come from the
# process object after exit and are empty on this runtime, which is how the first
# version of this probe reported blank walls and zero cores while still producing
# correct core-seconds.  Core-seconds are cumulative TotalProcessorTime, read while
# the processes are alive, so only the final sample matters and no polling error
# accumulates.

param(
  [string]$Clip = "$env:TEMP\rd420\u8000.mp4",
  [switch]$Sweep
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$rd = Join-Path $root 'build\Release\rdither.exe'
$out = "$env:TEMP\rd420\probe_out.mkv"
$log = "$env:TEMP\rd420\probe.log"
$err = "$env:TEMP\rd420\probe.err"

if (-not (Test-Path $Clip)) { throw "clip not found: $Clip" }

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

  $rdCpu = 0.0
  $ffCpu = 0.0
  $sw = [Diagnostics.Stopwatch]::StartNew()

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
  # A missing fps must become 0, not $null: the report line formats with {2,6:N1},
  # and -f throws on a null operand, which kills the whole sweep after the first
  # config rather than reporting it.  Zero also shows up honestly in the mean.
  $fps = 0.0
  if ($text -match '\(([\d.]+) fps\)') { $fps = [double]$Matches[1] }
  $stage = (($text -split "`n" | Where-Object { $_ -match 'busy time' }) -join ' ') -replace '\s+',' '

  $wall = $sw.Elapsed.TotalSeconds
  [pscustomobject]@{
    Label   = $Label
    Wall    = [math]::Round($wall, 1)
    Fps     = $fps
    RdCoreS = [math]::Round($rdCpu, 1)
    FfCoreS = [math]::Round($ffCpu, 1)
    Total   = [math]::Round($rdCpu + $ffCpu, 1)
    Cores   = if ($wall -gt 0) { [math]::Round(($rdCpu + $ffCpu) / $wall, 2) } else { 0 }
    Stage   = $stage.Trim()
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
foreach ($c in $configs) {
  $r = Measure-Config -Label $c.L -Extra $c.E
  $rows += $r
  '{0,-22} {1,6:N1} s {2,6:N1} fps  rd {3,7} ff {4,7} tot {5,7} core-s  {6,5:N2} cores avg' -f `
    $r.Label, $r.Wall, $r.Fps, $r.RdCoreS, $r.FfCoreS, $r.Total, $r.Cores
}

''
'=== stage breakdown (ms), in the order printed ==='
$rows | ForEach-Object { '{0,-22} {1}' -f $_.Label, $_.Stage }
