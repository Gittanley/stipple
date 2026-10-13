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
  # TAGS select suites. Any suite carrying one of these runs; suites carry more than
  # one so a narrow request is cheap to express:
  #
  #   image   dither, crossover      milliseconds per run, internal timer
  #   video   video-ram, video-threads   1080p lavfi fixture, wall clock
  #   fps     video-fps             a REAL clip, -Clip required
  #   ram     video-ram             only the --max-ram-mb sweep
  #   threads video-threads         only the --cpu-threads sweep
  #   all                          every suite
  #
  # -Tags was added because the alternative was a fixed list in which every suite is
  # coupled to every other: asking one question meant paying for all five, which is how a
  # "how long does this take" answer turns into forty minutes.
  [string]$Tags = 'image',
  [int]$Reps = 20,
  # Defaults to the project's own build, the same default verify.ps1 uses. The first
  # version defaulted to "" and fell back to PATH, so the first run of this harness
  # printed "no rdither given and none on PATH" and exited in under a second -- a bench
  # that silently measures nothing is worse than one that refuses to start.
  [string]$Rdither = ".\build\Release\rdither.exe",
  [string]$Ffmpeg  = "",
  # The real-clip throughput suite. -ClipFile points at a file on disk; see
  # docs/examples/SOURCES.md for where the benchmark clip comes from and why it is not
  # committed. Without it this suite does not run rather than quietly substituting a
  # synthetic fixture, because an fps figure is the one number README states as fact.
  #
  # It is -ClipFile and not -Clip because PowerShell variable names are CASE-INSENSITIVE.
  # The parameter was originally `$Clip` and the synthetic lavfi fixture was `$clip`, and
  # the fixture assignment -- which runs earlier, for the video-ram suite -- overwrote the
  # parameter. A run given a real 6-second 1080p clip reported a 90-frame synthetic one,
  # with its path and SHA printed as if all were well. The suite's guard also passed,
  # because by then $Clip held the fixture and the fixture exists.
  [string]$ClipFile = "",
  [string]$Out     = ""
)

$ErrorActionPreference = 'Stop'
# Numbers in the report must be culture-independent. This box runs a locale where
# [math]::Round renders 45.35 as "45,35" -- unreadable in a table and unparseable by
# anything reading the report back. Set before any arithmetic happens, not inside the
# formatter, because Get-Stats rounds as it computes.
[System.Threading.Thread]::CurrentThread.CurrentCulture =
  [System.Globalization.CultureInfo]::InvariantCulture

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
  $cpu = (Get-CimInstance Win32_Processor -EA SilentlyContinue | Select-Object -First 1)
  $os  = Get-CimInstance Win32_OperatingSystem -EA SilentlyContinue
  # Every one of these is optional. The first run of this harness died on
  # Get-Counter returning nothing: -EA SilentlyContinue suppresses the COMMAND's error
  # but not the property access on a null result, and $ErrorActionPreference = 'Stop'
  # turns that into a terminating one. A machine-state note is not worth killing a run
  # over -- an unavailable counter must degrade to 'unknown', not abort.
  $loadText = 'unknown'
  try {
    $c = Get-Counter '\Processor(_Total)\% Processor Time' -EA Stop
    if ($c -and $c.CounterSamples -and $c.CounterSamples.Count -gt 0) {
      $loadText = "$([math]::Round($c.CounterSamples[0].CookedValue,1))% busy"
    }
  } catch { $loadText = 'unknown' }
  return @(
    "cpu      : $(if ($cpu) { "$($cpu.Name.Trim()), $($cpu.NumberOfCores)c/$($cpu.NumberOfLogicalProcessors)t" } else { 'unknown' })",
    "free mem : $(if ($os) { "$([math]::Round($os.FreePhysicalMemory/1MB, 1)) GB free of $([math]::Round($os.TotalVisibleMemorySize/1MB,1)) GB" } else { 'unknown' })",
    "load     : $loadText"
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
# Calls Measure-One directly rather than through a [scriptblock] parameter. The
# scriptblock form -- Invoke-Sweep $configs $reps { param($c) ... } -- was the last thing
# standing between this script and a correct table: it produced a $results with 2 keys
# from 24 configurations, and no error, which is the exact failure the dropped-run counter
# below was added to catch. One less indirection.
function Invoke-Sweep([object[]]$configs, [int]$reps, [string]$inputPath,
                      [string]$outPath, [string]$timer) {
  $results = @{}
  $dropped = 0
  $firstDrop = ''
  foreach ($c in $configs) { $results[$c.Label] = [System.Collections.Generic.List[double]]::new() }
  for ($round = 0; $round -lt $reps; $round++) {
    # Reshuffle EVERY round. Running configs in blocks attributes all drift to
    # whichever config was measured first; see the header.
    #
    # -Count is REQUIRED. `Get-Random` defaults to -Count 1, so `$configs | Get-Random`
    # returns ONE random configuration, not a shuffled list -- which made every round
    # measure a single config and left 22 of 24 rows empty, with dropped=0, because
    # nothing was failing: almost nothing was RUNNING.
    $order = @($configs | Get-Random -Count $configs.Count)
    foreach ($c in $order) {
      $inPath = if ($c.Input) { $c.Input } else { $inputPath }
      $v = Measure-One $c.Args $inPath $outPath $timer
      # A run that produced no measurement is DROPPED, and COUNTED. Counting it matters:
      # an earlier version dropped 22 of 24 configurations and wrote a two-row table with
      # no indication that anything was wrong.
      if ($null -eq $v -or [double]$v -le 0) {
        $dropped++
        if (-not $firstDrop) { $firstDrop = $c.Label }
      } else {
        # Assigned to a local first. `$results[$key].Add(x)` does NOT do what it reads
        # like: on this PowerShell it mutates one key and silently discards the rest,
        # which is how 24 configurations produced a two-row table with dropped=0.
        # Reproduced in isolation and fixed by binding the list to a variable first.
        $list = $results[$c.Label]
        $list.Add([double]$v)
      }
    }
    Write-Progress -Activity "round $($round+1)/$reps" -PercentComplete ([int](100 * ($round + 1) / $reps))
  }
  Write-Progress -Activity "done" -Completed
  $script:dropped = $dropped
  $script:firstDrop = $firstDrop
  return $results
}

# Probes each engine once and returns those that run. This is what replaced the
# -Engines parameter, and it is also the thing that makes the OpenCL question
# answerable: opencl was the arm silently missing from every table, and now it is
# either measured or reported as unavailable, with the reason.
function Detect-Engines([string]$exePath, [string]$probePng) {
  $known = @('cpu','blocks','cuda','opencl','approx')
  $ok = [System.Collections.Generic.List[string]]::new()
  $why = @{}
  foreach ($e in $known) {
    if (-not $probePng -or -not (Test-Path $probePng)) { $ok.Add($e); continue }
    $msg = & $exePath --colors 16 --engine $e $probePng (Join-Path ([IO.Path]::GetTempPath()) ("probe_" + $e + ".png")) 2>&1
    if ($LASTEXITCODE -eq 0) {
      $ok.Add($e)
    } else {
      $why[$e] = (($msg | Select-String 'error' | Select-Object -First 1).Line)
    }
  }
  if ($why.Count -gt 0) {
    Write-Host "bench: engines unavailable on this machine, and therefore not measured:"
    foreach ($k in $why.Keys) { Write-Host "        $k : $why[$k]" }
  }
  Write-Host ("bench: engines measured: " + ($ok -join ', '))
  return @($ok)
}

function Write-Table([object]$results) {
  $lines = @()
  $lines += "| config | n | median | IQR | min | max | spread |"
  $lines += "|---|---:|---:|---:|---:|---:|---:|"
  $shown = 0
  foreach ($k in ($results.Keys | Sort-Object)) {
    $st = Get-Stats $results[$k]
    if (-not $st) { continue }
    $shown++
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | {6}% |" -f `
               $k, $st.n, $st.median, $st.iqr, $st.min, $st.max, $st.spread)
  }
  if ($shown -lt $script:configCount) {
    $lines += ""
    $lines += ("**{0} of {1} configurations produced NO measurement.** First: `{2}`." -f `
               ($script:configCount - $shown), $script:configCount, $script:firstDrop)
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
foreach ($_m in (Get-MachineState)) { $report.Add("- $_m") }
$report.Add("")

try {
  # Fixtures are BUILT here, not committed. TWO IMAGES ON PURPOSE, and the pair is the
  # whole experiment: artifacts/colors32_findings.md's central finding is that dither
  # cost tracks TREE SIZE, not colour count, so a suite that measured one image would
  # find no effect even if the effect were real. Measured on this machine with
  # `magick -format %k`: `gradients` yields ~600 distinct colours, `testsrc` plus noise
  # ~28,900 -- a wider spread than the 16-vs-195,464 pair the original used, which is the
  # right direction.
  #
  # NO `plasma` FILTER: this ffmpeg build does not have one, and an earlier version of
  # this line asked for `plasma=fractal:size=...`, which failed twice over -- wrong
  # option syntax first, then no such filter -- leaving both fixtures absent while the
  # suite reported "skipped" and wrote a report with no numbers in it. The fixture check
  # below is the guard that should have caught that on the first run.
  $imgBig   = Join-Path $tmp 'big.png'
  $imgSmall = Join-Path $tmp 'small.png'
  if ($ffmpeg) {
    & $ffmpeg -y -v error -f lavfi -i 'gradients=size=512x384' -frames:v 1 $imgSmall 2>&1 | Out-Null
    & $ffmpeg -y -v error -f lavfi -i 'testsrc=size=512x384' `
             -vf 'noise=alls=24:allf=t+u' -frames:v 1 $imgBig 2>&1 | Out-Null
  }
  if (-not (Test-Path $imgSmall) -or -not (Test-Path $imgBig)) {
    Write-Host "bench: could not build the image fixtures with ffmpeg ($ffmpeg); NOTHING MEASURED"
    Write-Host "        tried: gradients=size=512x384, and testsrc=size=512x384 + noise=alls=24:allf=t+u"
  }
  $synthClip = Join-Path $tmp 'hd.mkv'
  if ($ffmpeg) {
    & $ffmpeg -y -v error -f lavfi -i 'testsrc=size=1920x1080:rate=30:duration=3' `
             -c:v ffv1 -pix_fmt yuv444p $synthClip 2>&1 | Out-Null
  }

  # Engines are DETECTED, not passed. `-Engines cpu,opencl` was tried and removed:
  # `pwsh -File script.ps1` does not bind a comma list to a parameter the way an
  # interactive shell does, so the argument arrived mangled and every configuration
  # failed with no obvious cause. Probing each engine once is cheaper than explaining
  # PowerShell's argument binding, and it answers the question the parameter existed for:
  # which engines this machine can actually run.
  $Engines = & Detect-Engines $exe $imgSmall

$runSuites = if ($Tags -contains 'all') {
    @('dither','crossover','video-ram','video-threads','video-fps')
  } else {
    $map = [ordered]@{
      'dither'        = @('image')
      'crossover'     = @('image')
      'video-ram'     = @('video','ram')
      'video-threads' = @('video','threads')
      'video-fps'     = @('video','fps')
    }
    @($map.GetEnumerator() | Where-Object {
        $tag = $_.Value
        ($tag | Where-Object { $Tags -contains $_ }).Count -gt 0
      } | ForEach-Object { $_.Key })
  }
if ($runSuites.Count -eq 0) {
  Write-Host "bench: no suite carries any of: $($Tags -join ', ')"
  Write-Host "        tags are: image, video, fps, ram, threads, all"
  exit 0
}
$script:wantedEngines = $Engines
$script:unavailable = @()

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
        if (-not (Test-Path $imgSmall)) { $report.Add("skipped: image fixtures unavailable"); break }
        $configs = @()
        foreach ($img in @(@{n='small';p=$imgSmall}, @{n='big';p=$imgBig})) {
          foreach ($c in 2,4,16,32,64,256) {
            # opencl used to be added only where a comment believed it could run.
            # The harness DROPS any config that cannot, reporting the shortfall rather
            # than hiding it, so the filter was never what made it safe.
            $engines = @($script:wantedEngines)
            # NO opencl filter here, and there was one until this run measured it wrong.
            # The filter read "opencl refuses --colors above 16", which is not what the
            # code does: rd_opencl.cpp:630 tests the RESOLVED PALETTE, not the request.
            # Filtering on the request therefore deleted rows that were perfectly valid
            # --colors 20 measured opencl fine on an image whose palette resolved to 12 --
            # while rows the filter kept were not thereby safe. Predicating a measurement
            # on a cause that is wrong hides exactly the case worth seeing, and Invoke-Sweep
            # already counts and reports genuine failures, so nothing is lost by letting
            # opencl run and fail honestly.
            foreach ($e in $engines) {
              $configs += @{ Label = "$($img.n) colors=$c engine=$e"
                             Args  = @('--colors',"$c",'--engine',$e)
                             Input = $img.p }
            }
          }
        }
        $outPath = Join-Path $tmp 'o.png'
        # Discarded warmup: DLL load, driver init, first-touch page faults.
        Measure-Control
        $script:configCount = $configs.Count
        $res = Invoke-Sweep $configs $Reps $null $outPath 'internal'
        $report.Add("Timer: the internal `dither :` figure, not wall clock.")
        $report.Add((Write-Table $res) -join [Environment]::NewLine)
        $report.Add((Get-DriftNote))
        $report.Add("")
        break
      }

      'crossover' {
        if (-not (Test-Path $imgSmall)) { $report.Add("skipped: image fixtures unavailable"); break }
        # The open question from colors32_findings.md: where does blocks lose to
        # approx? That file measured 2 runs per cell and put the crossover between 16
        # and 24. Finer colour steps around there, and enough reps to survive drift.
        $configs = @()
        foreach ($img in @(@{n='small';p=$imgSmall}, @{n='big';p=$imgBig})) {
          foreach ($c in 12,16,18,20,22,24,28,32,48,64) {
            foreach ($e in @($script:wantedEngines)) {
              $configs += @{ Label = "$($img.n) colors=$c engine=$e"
                             Args  = @('--colors',"$c",'--engine',$e)
                             Input = $img.p }
            }
          }
        }
        $outPath = Join-Path $tmp 'o.png'
        Measure-Control
        $script:configCount = $configs.Count
        $res = Invoke-Sweep $configs $Reps $null $outPath 'internal'
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
        if (-not (Test-Path $synthClip)) { $report.Add("skipped: image fixtures unavailable"); break }
        # --max-ram-mb only reaches the video queue when passed, and only CAPS what the
        # physical-memory derivation allows. Depth is budget-bound only with enough
        # workers, so cpu-threads is pinned high here.
        $configs = @()
        foreach ($mb in 200,400,800,1600,3200,6400) {
          $configs += @{ Label = "video max-ram-mb=$mb"
                         Args  = @('--video','--engine','cpu','--cpu-threads','12',
                                   '--colors','16','--max-ram-mb',"$mb")
                         Input = $synthClip }
        }
        $outPath = Join-Path $tmp 'o.mkv'
        Measure-Control
        $script:configCount = $configs.Count
        $res = Invoke-Sweep $configs $Reps $synthClip $outPath 'wall'
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

      'video-fps' {
        # THE suite README's headline number depends on, and the only one that measures
        # a real clip. The other video suites use a 90-frame lavfi fixture, which is fine
        # for comparing configurations against each other and worthless as a throughput
        # claim about real footage.
        #
        # -Clip is REQUIRED here. If it is absent the suite says so and stops rather than
        # measuring the synthetic fixture and reporting it as footage, because that is
        # exactly how an unverifiable number gets written down.
        if (-not $ClipFile -or -not (Test-Path $ClipFile)) {
          $report.Add("not run: -ClipFile was not given, or the file is absent.")
          $report.Add("        This suite measures a REAL clip and will not substitute a")
          $report.Add("        synthetic one.  See docs\examples\SOURCES.md for the clip this")
          $report.Add("        repository benchmarks against, its URL and its SHA-256.")
          break
        }
        # Guard against the specific failure this suite already committed once: being
        # handed the synthetic fixture and reporting it as footage. Comparing paths is
        # cheap and the failure was silent, so it gets an explicit refusal.
        if ($ClipFile -eq $synthClip) {
          $report.Add("not run: -ClipFile is the synthetic lavfi fixture ($synthClip).")
          $report.Add("        An fps figure from that fixture describes a gradient, not a")
          $report.Add("        video, and it is not comparable to anything README says.")
          break
        }
        # Report the clip's own properties. An fps figure without the geometry it was
        # measured at is not a measurement, it is a rumour.
        $ffprobe = Resolve-Tool 'ffprobe.exe' ''
        if ($ffprobe) {
          $info = & $ffprobe -v error -select_streams v:0 -count_frames `
                            -show_entries stream=width,height,nb_read_frames,r_frame_rate `
                            -of 'csv=p=0' $ClipFile 2>$null | Select-Object -First 1
          $report.Add("clip: $ClipFile")
          $report.Add("      $info  (width,height,frames,frame_rate)")
        }
        $sha = (Get-FileHash $ClipFile -Algorithm SHA256).Hash
        $report.Add("      sha256 $sha")

        $configs = @()
        # --colors 16 is the default and the README figure's setting. --blocks is the
        # default video engine. Both are stated rather than assumed.
        # Every detected engine, not a hand-picked two. Opencl was left out of this list
        # when the suite was written, which is how the one arm that WINS at 1080p went
        # unmeasured: it carries ~160 ms of fixed cost per process and loses to cpu on
        # any small image, so it looks worthless in the image suites and is not.
        foreach ($e in @($script:wantedEngines)) {
          $configs += @{ Label = "fps engine=$e colors=16"
                         Args  = @('--video','--engine',$e,'--colors','16')
                         Input = $ClipFile }
        }
        $outPath = Join-Path $tmp 'fps.mkv'
        Measure-Control
        $res = Invoke-Sweep $configs $Reps $ClipFile $outPath 'wall'
        $report.Add("Timer: WALL CLOCK over the whole clip, which is the quantity")
        $report.Add("README quotes. Reps per configuration: $Reps.")
        $report.Add((Write-Table $res) -join [Environment]::NewLine)
        $report.Add((Get-DriftNote))
        $report.Add("")
        Measure-Control
        break
      }

      'video-threads' {
        if (-not (Test-Path $synthClip)) { $report.Add("skipped: image fixtures unavailable"); break }
        $configs = @()
        foreach ($t in 0,2,4,8,12,16) {
          $configs += @{ Label = "video cpu-threads=$t"
                         Args  = @('--video','--engine','cpu','--cpu-threads',"$t",'--colors','16')
                         Input = $synthClip }
        }
        $outPath = Join-Path $tmp 'o.mkv'
        Measure-Control
        $script:configCount = $configs.Count
        $res = Invoke-Sweep $configs $Reps $synthClip $outPath 'wall'
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
  foreach ($_m in (Get-MachineState)) { $report.Add("- $_m") }
  $report.Add("")
  $report.Add("## What this run does NOT establish")
  $report.Add("")
  $report.Add("- Whether a per-node lower bound in the octree is worth building. That needs")
  $report.Add("  instrumentation of ClosestColor to count visits into subtrees that cannot")
  $report.Add("  beat the current best, which is a CODE CHANGE, not a benchmark. This")
  $report.Add("  harness cannot answer it and does not pretend to.")
  $report.Add("- Anything about the flat search on a PALETTE larger than 16 entries. That is")
  $report.Add("  structurally absent (`tree.color_count() > 16` returns false, rd_opencl.cpp:630),")
  $report.Add("  so there is nothing here to compare against. Note the predicate is the resolved")
  $report.Add("  palette, NOT `--colors`: a request above 16 that resolves to 12 entries runs,")
  $report.Add("  and a request below 16 that resolves to 17 is refused. Rows missing for")
  $report.Add("  engine=opencl at high --colors are that refusal, not a harness failure.")
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
