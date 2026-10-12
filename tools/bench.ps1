<#
.SYNOPSIS
  rdither benchmark harness. Measures; never asserts; never gates.

.DESCRIPTION
  This is NOT a test. It always exits 0, even when a suite explodes, because a
  benchmark that can fail is a test and a test that measures is a mess. Nothing here
  is wired into verify.ps1 and nothing should be.

  It exists because this machine drifts 10-25% between identical binaries -- stated in
  README.md and in artifacts/colors32_findings.md -- and the previous measurement in
  that file used TWO RUNS PER CELL. At that sample size a 25% drift is invisible, so
  the table cannot distinguish a real effect from noise, and its conclusions about the
  blocks/approx crossover rest on numbers that a louder neighbour could have produced.

  DESIGN, and why each part is here:

    * READ THE INTERNAL TIMER, NOT THE WALL CLOCK. rdither prints `dither :  X ms`
      for the stage under test. The wall clock includes palette build, decode, encode
      and OS scheduling. colors32_findings.md already says to read `dither :`; this
      makes it the default rather than advice.

    * INTERLEAVE AND SHUFFLE, DO NOT RUN IN BLOCKS. The single worst mistake available
      in a benchmark is running config A 20 times, then config B 20 times. Drift is
      monotonic on a busy box, so it lands entirely on whichever config went first and
      the result is a measurement of when it ran, not of what it ran. Every round
      reshuffles, so drift is spread across all configs equally.

    * MEDIAN AND IQR, NOT MEAN AND STDErr. The drift is multiplicative and heavy
      tailed. A mean of a drifted sample is biased toward whichever neighbour was
      loudest; the median is not. IQR is reported so a wide spread is visible rather
      than hidden inside a confident-looking mean.

    * A CONTROL RUN THROUGHOUT. One config is re-measured at the start, the middle and
      the end. Its spread is the run's OWN drift, measured rather than assumed, so the
      report can say "this difference is inside the noise" instead of leaving the
      reader to guess. Any result smaller than the control spread is not a result.

    * WARMUP DISCARDED. The first run of any process pays DLL load, driver init and
      first-touch page faults. Those are real costs for a one-shot command and
      irrelevant to a steady-state loop, so they are measured and thrown away.

    * MACHINE STATE AT BOTH ENDS. A benchmark run on a box that was busy is
      self-indicting only if it recorded that it was busy.

.PARAMETER Suite
  dither        image engines x colour counts. Milliseconds per run, so this is cheap.
  video-ram     video throughput against --max-ram-mb. Depth is budget-bound here.
  video-threads video throughput against --cpu-threads.
  crossover     the blocks/approx crossover, the question colors32_findings.md left open.
  all           every suite above, in that order.

.PARAMETER Reps
  Runs per configuration. 20 is the default because at 10-25% drift that is roughly
  what it takes to resolve a 5% difference. Fewer than 8 makes the IQR meaningless.
  Do not report a difference smaller than the control spread, whatever Reps is.

.EXAMPLE
  pwsh -File tools\bench.ps1 -Suite crossover -Reps 20
  pwsh -File tools\bench.ps1 -Suite all -Reps 20 -Out artifacts\bench-2026-10-12.md
#>
param(
  [ValidateSet('dither','video-ram','video-threads','crossover','all')]
  [string]$Suite = 'dither',
  [int]$Reps = 20,
  [string]$Rdither = "",
  [string]$Ffmpeg  = "",
  [string]$Out     = ""
)

$ErrorActionPreference = 'Stop'

function Resolve-Tool([string]$name, [string]$given) {
  if ($given -and (Test-Path $given)) { return (Resolve-Path $given).Path }
  $c = Get-Command $name -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  return ""
}

# --- statistics -------------------------------------------------------------------
# Median and quartiles, by the nearest-rank method on the sorted sample. Chosen over
# an interpolated percentile because at Reps=20 an interpolated quartile invents
# precision the sample does not have.
function Get-Stats([double[]]$values) {
  if (-not $values -or $values.Count -eq 0) { return $null }
  $s = @($values | Sort-Object)
  $n = $s.Count
  $mid = [int][Math]::Floor($n / 2)
  $median = if ($n % 2 -eq 1) { $s[$mid] } else { ($s[$mid - 1] + $s[$mid]) / 2.0 }
  $q = {
    param([double]$p)
    $idx = [int][Math]::Ceiling($p * $n) - 1
    if ($idx -lt 0) { $idx = 0 }
    if ($idx -ge $n) { $idx = $n - 1 }
    return $s[$idx]
  }
  $q1 = & $q 0.25
  $q3 = & $q 0.75
  return [pscustomobject]@{
    n     = $n
    min   = [math]::Round($s[0], 3)
    max   = [math]::Round($s[$n - 1], 3)
    median= [math]::Round($median, 3)
    q1    = [math]::Round($q1, 3)
    q3    = [math]::Round($q3, 3)
    iqr   = [math]::Round($q3 - $q1, 3)
    spread= [math]::Round((($s[$n - 1] - $s[0]) / $s[0]) * 100.0, 1)
  }
}

function Get-MachineState {
  $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1)
  $os  = Get-CimInstance Win32_OperatingSystem
  $load = (Get-Counter '\Processor(_Total)\% Processor Time' -EA SilentlyContinue).CounterSamples[0]
  return @(
    "cpu      : $($cpu.Name.Trim()), $($cpu.NumberOfCores)c/$($cpu.NumberOfLogicalProcessors)t",
    "free mem : $([math]::Round($os.FreePhysicalMemory/1MB, 1)) GB free of $([math]::Round($os.TotalVisibleMemorySize/1MB,1)) GB",
    "load     : $(if ($load) { "$([math]::Round($load.CookedValue,1))% busy" } else { 'unknown' })"
  )
}

# --- the runner --------------------------------------------------------------------
# One measurement. Returns the internal `dither :` figure in ms when there is one,
# and the wall clock in ms otherwise. Which of the two is returned is recorded per
# suite, because a reader who does not know which one a column holds cannot use it.
function Measure-One([string[]]$args_, [string]$inputPath, [string]$outPath, [string]$timer) {
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $r = & $exe @args_ $inputPath $outPath 2>&1
  $code = $LASTEXITCODE
  $sw.Stop()
  if ($code -ne 0) { return $null }
  if ($timer -eq 'internal') {
    $m = $r | Select-String -Pattern 'dither\s+:\s+([0-9.]+)\s*ms'
    if ($m) { return [double]$m.Matches[0].Groups[1].Value }
    return $null
  }
  return [math]::Round($sw.Elapsed.TotalMilliseconds, 3)
}

# Runs every config Reps times, reshuffling each round. Returns a hashtable keyed by
# config label -> double[].
function Invoke-Sweep([object[]]$configs, [int]$reps, [scriptblock]$run) {
  $results = @{}
  foreach ($c in $configs) { $results[$c.Label] = [System.Collections.Generic.List[double]]::new() }
  for ($round = 0; $round -lt $reps; $round++) {
    # Reshuffle EVERY round. Running configs in blocks attributes all drift to
    # whichever config was measured first; see the header.
    $order = @($configs | Get-Random)
    foreach ($c in $order) {
      $v = & $run $c
      if ($null -ne $v) { $results[$c.Label].Add([double]$v) }
    }
    Write-Progress -Activity "round $($round+1)/$reps" -Status ($c.Label) `
                   -PercentComplete ([int](100 * ($round + 1) / $reps))
  }
  Write-Progress -Activity "done" -Completed
  return $results
}

function Write-Table([object]$results) {
  $lines = @()
  $lines += "| config | n | median | IQR | min | max | spread |"
  $lines += "|---|---:|---:|---:|---:|---:|---:|"
  foreach ($k in ($results.Keys | Sort-Object)) {
    $st = Get-Stats $results[$k]
    if (-not $st) { continue }
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | {6}% |" -f `
               $k, $st.n, $st.median, $st.iqr, $st.min, $st.max, $st.spread)
  }
  return $lines
}

# --- setup -------------------------------------------------------------------------
$exe = Resolve-Tool 'rdither.exe' $Rdither
$ffmpeg = Resolve-Tool 'ffmpeg.exe' $Ffmpeg
if (-not $exe) { Write-Host "bench: no rdither given and none on PATH; nothing to do"; exit 0 }
if (-not $ffmpeg) { Write-Host "bench: no ffmpeg on PATH; image suites unavailable"; }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("rd_bench_" + [Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$report = [System.Collections.Generic.List[string]]::new()
$report.Add("# rdither benchmark")
$report.Add("")
$report.Add("Generated $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') by tools\bench.ps1. **A measurement, not a gate.**")
$report.Add("Suite: $Suite   Reps per configuration: $Reps")
$report.Add("")
$report.Add("Read the medians against the control spread below before reading any difference")
$report.Add("as real. A gap smaller than the control spread is noise, whatever the sample size.")
$report.Add("")
$report.Add("## Machine, before")
$report += (Get-MachineState | ForEach-Object { "- $_" })
$report.Add("")

try {
  # Fixtures are BUILT here, not committed: .gitignore excludes *.png inputs' larger
  # relatives and *.mp4/*.mkv entirely.
  $imgBig   = Join-Path $tmp 'noisy.png'
  $imgSmall = Join-Path $tmp 'plasma.png'
  if ($ffmpeg) {
    & $ffmpeg -y -v error -f lavfi -i 'plasma:fractal' -colorspace sRGB -size 512x384 $imgSmall 2>&1 | Out-Null
    & $ffmpeg -y -v error -f lavfi -i 'plasma:fractal' -attenuate 0.4 +noise Gaussian `
             -colorspace sRGB -size 512x384 $imgBig 2>&1 | Out-Null
  }
  $clip = Join-Path $tmp 'hd.mkv'
  if ($ffmpeg) {
    & $ffmpeg -y -v error -f lavfi -i 'testsrc=size=1920x1080:rate=30:duration=3' `
             -c:v ffv1 -pix_fmt yuv444p $clip 2>&1 | Out-Null
  }

  $runSuites = if ($Suite -eq 'all') { @('dither','crossover','video-ram','video-threads') } else { @($Suite) }

  # --- CONTROL -----------------------------------------------------------------------
  # One configuration, re-measured at the start and end of EVERY suite. Its spread is
  # this run's own drift, measured rather than assumed, and it is the yardstick every
  # other number in the report is read against. Without it the report is a table of
  # numbers with no way to tell a result from a neighbour's fan.
  #
  # It is NOT discarded into a pipeline the way the first version of this script did:
  # that version computed nothing and wrote a report promising a drift figure it never
  # produced, which is worse than not claiming one at all.
  $controlValues = [System.Collections.Generic.List[double]]::new()
  $controlOut = Join-Path $tmp 'ctl.png'
  function Measure-Control {
    $v = Measure-One @('--colors','16','--engine','blocks') $imgSmall $controlOut 'internal'
    if ($null -ne $v) { $controlValues.Add([double]$v) }
  }
  function Get-DriftNote {
    $st = Get-Stats $controlValues.ToArray()
    if (-not $st) { return "  control: no samples, so NOTHING in this report can be distinguished from noise." }
    return ("  control: n={0} median={1} IQR={2} max/min spread={3}%. **Any difference smaller" + `
            " than that is noise, whatever the sample size.**") -f $st.n, $st.median, $st.iqr, $st.spread
  }

  foreach ($s in $runSuites) {
    $report.Add("## Suite: $s")
    $report.Add("")

    switch ($s) {
      'dither' {
        if (-not (Test-Path $imgSmall)) { $report.Add("skipped: no ffmpeg, no fixture"); break }
        $configs = @()
        foreach ($img in @(@{n='small';p=$imgSmall}, @{n='big';p=$imgBig})) {
          foreach ($c in 2,4,16,32,64,256) {
            foreach ($e in 'cpu','blocks') {
              $configs += @{ Label = "$($img.n) colors=$c engine=$e"
                             Args  = @('--colors',"$c",'--engine',$e)
                             Input = $img.p }
            }
          }
        }
        $outPath = Join-Path $tmp 'o.png'
        # Discarded warmup: DLL load, driver init, first-touch page faults.
        Measure-Control
        $res = Invoke-Sweep $configs $Reps { param($c)
                   Measure-One $c.Args $c.Input $outPath 'internal' }
        $report.Add("Timer: the internal `dither :` figure, not wall clock.")
        $report.Add((Write-Table $res) -join [Environment]::NewLine)
        $report.Add((Get-DriftNote))
        $report.Add("")
        break
      }

      'crossover' {
        if (-not (Test-Path $imgSmall)) { $report.Add("skipped: no ffmpeg, no fixture"); break }
        # The open question from colors32_findings.md: where does blocks lose to
        # approx? That file measured 2 runs per cell and put the crossover between 16
        # and 24. Finer colour steps around there, and enough reps to survive drift.
        $configs = @()
        foreach ($img in @(@{n='small';p=$imgSmall}, @{n='big';p=$imgBig})) {
          foreach ($c in 12,16,18,20,22,24,28,32,48,64) {
            foreach ($e in 'blocks','approx') {
              $configs += @{ Label = "$($img.n) colors=$c engine=$e"
                             Args  = @('--colors',"$c",'--engine',$e)
                             Input = $img.p }
            }
          }
        }
        $outPath = Join-Path $tmp 'o.png'
        Measure-Control
        $res = Invoke-Sweep $configs $Reps { param($c)
                   Measure-One $c.Args $c.Input $outPath 'internal' }
        $report.Add("Timer: the internal `dither :` figure. Read each image's column")
        $report.Add("separately -- colors32_findings.md's own finding is that the")
        $report.Add("crossover is DATA-DEPENDENT, not colour-dependent, so averaging the")
        $report.Add("two images would destroy the very thing being measured.")
        $report.Add((Write-Table $res) -join [Environment]::NewLine)
        $report.Add((Get-DriftNote))
        $report.Add("")
        Measure-Control
        break
      }

      'video-ram' {
        if (-not (Test-Path $clip)) { $report.Add("skipped: no ffmpeg, no fixture"); break }
        # --max-ram-mb only reaches the video queue when passed, and only CAPS what the
        # physical-memory derivation allows. Depth is budget-bound only with enough
        # workers, so cpu-threads is pinned high here.
        $configs = @()
        foreach ($mb in 200,400,800,1600,3200,6400) {
          $configs += @{ Label = "video max-ram-mb=$mb"
                         Args  = @('--video','--engine','cpu','--cpu-threads','12',
                                   '--colors','16','--max-ram-mb',"$mb")
                         Input = $clip }
        }
        $outPath = Join-Path $tmp 'o.mkv'
        Measure-Control
        $res = Invoke-Sweep $configs $Reps { param($c)
                   Measure-One $c.Args $c.Input $outPath 'wall' }
        $report.Add("Timer: WALL CLOCK, because the quantity of interest is throughput,")
        $report.Add("not a stage. No internal timer covers the whole pipeline.")
        $report.Add((Write-Table $res) -join [Environment]::NewLine)
        $report.Add((Get-DriftNote))
        $report.Add("")
        Measure-Control
        $report.Add("Also worth recording alongside this table, since the flag caps")
        $report.Add("DEPTH rather than bytes: the `queue=N` from each run.")
        $report.Add("")
        break
      }

      'video-threads' {
        if (-not (Test-Path $clip)) { $report.Add("skipped: no ffmpeg, no fixture"); break }
        $configs = @()
        foreach ($t in 0,2,4,8,12,16) {
          $configs += @{ Label = "video cpu-threads=$t"
                         Args  = @('--video','--engine','cpu','--cpu-threads',"$t",'--colors','16')
                         Input = $clip }
        }
        $outPath = Join-Path $tmp 'o.mkv'
        Measure-Control
        $res = Invoke-Sweep $configs $Reps { param($c)
                   Measure-One $c.Args $c.Input $outPath 'wall' }
        $report.Add("Timer: WALL CLOCK. cpu-threads=0 means GPU-only on a GPU build, so")
        $report.Add("that row is NOT a thread-count data point on a CUDA build -- it is a")
        $report.Add("different engine, and comparing it to the others would be a category")
        $report.Add("error. Run this suite on a --no-cuda build if you want threads alone.")
        $report.Add((Write-Table $res) -join [Environment]::NewLine)
        $report.Add((Get-DriftNote))
        $report.Add("")
        Measure-Control
        break
      }
    }
  }

  $report.Add("## Machine, after")
  $report += (Get-MachineState | ForEach-Object { "- $_" })
  $report.Add("")
  $report.Add("## What this run does NOT establish")
  $report.Add("")
  $report.Add("- Whether a per-node lower bound in the octree is worth building. That needs")
  $report.Add("  instrumentation of ClosestColor to count visits into subtrees that cannot")
  $report.Add("  beat the current best, which is a CODE CHANGE, not a benchmark. This")
  $report.Add("  harness cannot answer it and does not pretend to.")
  $report.Add("- Anything about the flat search above 16 colours. It is structurally absent")
  $report.Add("  (`colours > 16` returns false), so there is nothing here to compare against.")
  $report.Add("- Performance on any machine but this one. The drift figure is local.")

  $text = $report -join "`r`n"
  if ($Out) {
    $dir = Split-Path -Parent $Out
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Set-Content -Path $Out -Value $text
    Write-Host "bench: wrote $Out"
  } else {
    Write-Host $text
  }
}
finally {
  Remove-Item $tmp -Recurse -Force -EA SilentlyContinue
}

# Always 0. A benchmark that can fail is a test.
exit 0