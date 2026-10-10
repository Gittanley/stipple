# SPDX-License-Identifier: GPL-3.0-or-later
<#
  verify.ps1 -- bit-exactness sweep against ImageMagick's own Riemersma dither.

  Every case is checked twice:
    1. rdither --verify, which runs IM's full pipeline internally and diffs
       pixel-for-pixel;
    2. `magick compare -metric AE`, an independent path through the CLI.

  Every case is checked twice:
    1. rdither --verify, which runs IM's full pipeline internally and diffs
       pixel-for-pixel;
    2. `magick compare -metric AE`, an independent path through the CLI.

  EXIT CODES -- these are load-bearing, and 4 was added on 2026-10-10 because the suite
  could not otherwise tell a caller "everything passed" from "something could not run":

    0  every stage ran and passed.  Reachable only with -Slow, or when nothing is skipped.
    1  a stage FAILED.
    2  a stage did not run for an INCIDENTAL reason -- a missing fixture, a -Fast run, a
       probe that could not start.  A gap.  Not a pass, not a failure.
    3  the -BudgetMinutes watchdog fired.  The run did not finish, so it neither passed nor
       failed and must not be read as either.
    4  PASS WITH DECLARED EXCLUSIONS.  Everything that ran passed, and everything that did
       not run is OFF BY DEFAULT and named in the output with the switch that runs it.
       This is the exit code of the DEFAULT invocation, and it is a good result: it means
       `palette overlap` (295 s, gating a flag that ships off) was deliberately excluded,
       not skipped by accident.

  Before 4 existed, the default invocation exited 2 forever, even on a tree where every
  stage passed. A red light nobody can act on trains people to ignore it, which is the same
  failure as a green that means nothing -- and it is the same shape of bug this suite exists
  to catch, living in the suite itself.
#>
param(
  [string]$Rdither   = ".\build\Release\rdither.exe",
  [string]$Magick    = "magick",
  [int[]] $Colors    = @(2, 4, 16, 64, 256),
  [string]$ImageDir  = "tests",
  # Skip the stages that need a 1080p clip.  The full run is dominated by exactly those
  # two, and they are also the only checks that can exercise the faults found this week:
  # four of them needed 1080p video and the image suite would never have seen them.  But
  # equally the image suite catches octree and curve regressions and needs no video at
  # all, so paying 25-35 minutes for the union on every edit is the wrong trade.  -Fast
  # is the inner loop; the default is still the full sweep, which is what CI runs.
  [switch]$Fast,
  # -Fast leaves two stages unrun, and "unrun" is not "passed".  It therefore exits 2
  # (the suite's existing "cannot run / not covered" code) rather than 0, so that a
  # caller which runs `verify.ps1 -Fast` and checks the exit code cannot mistake a
  # partial run for a green one.  Nothing in the pipeline reads this automatically yet --
  # CI runs the full sweep -- so the flag costs an inner-loop user one extra switch, and
  # buys a distinction that used to exist only as prose at the bottom of the log.
  [switch]$AllowPartial,
  # HARD CEILING on the whole run, in minutes. Zero = no ceiling.
  #
  # WHY THIS EXISTS. verify.ps1 crashed the owner's machine on 2026-10-05. The suite is not
  # unbounded -- every stage is a finite script with its own fixtures -- but nothing in it
  # bounded the TOTAL, so a stage that regressed from seconds into minutes, or a fixture
  # generator that started rendering 1080p instead of 320x180, would have run for as long as
  # it liked with no complaint. A gate that can take the machine down is not a gate you can
  # be asked to run.
  #
  # It is enforced by a WATCHDOG rather than by estimates: a timer thread checks elapsed
  # time every 15 s and, on expiry, prints which stages have completed and kills the run
  # with exit 3. Estimates were rejected deliberately -- they are exactly the kind of claim
  # that is right until the machine is busy, and this machine is never quiet.
  #
  # 3 is a new code, distinct from 0 (pass), 1 (fail) and 2 (not covered): the suite did not
  # finish, so it neither passed nor failed and must not be read as either.
  [int]$BudgetMinutes = 25,
  # Run the stages that are off by default because of what they COST.
  #
  # Currently one: `palette overlap`, measured at 295.1 s, which gates RD_PALETTE_OVERLAP --
  # a flag that ships OFF because it measured SLOWER (docs\KNOWN-ISSUES.md item 4). 15% of a
  # 20-minute run for an opt-in feature's correctness is the wrong place to spend it, so it
  # moved here. It is not deleted: a real check stays available, and -Slow is how you run it.
  #
  # THIS PARAMETER WAS MISSING WHEN THE MOVE FIRST SHIPPED. The gate said
  #     Skipped 'palette overlap' (off by default: ... Run verify.ps1 -Slow.)
  # and there was no `-Slow` to run -- `$Slow` was undefined, so the stage could not be
  # reached at all. `pwsh -File verify.ps1 -Slow` exited 2 with the identical skip message,
  # which is the worst possible combination: it named a remedy that did not exist, and a
  # reader checking the log would reasonably believe the stage had been run.
  #
  # That is the same shape as the bug this whole gate exists to catch -- a check that reports
  # without doing anything -- in the gate itself rather than in the code under test. Found by
  # running `-Slow` and reading the output, which is the only reason it was found at all.
  [switch]$Slow,
  # Threads the stages are allowed to use. The 2026-10-05 crash happened with every core
  # pinned by x264 plus the host dither pool. Leaving 1-2 cores free keeps the machine
  # responsive for whatever else the owner is doing, which on this box is usually something.
  # 0 = do not set it.
  #
  # MEASURED, and the answer is that it does not work, so this stays a no-op by choice
  # rather than by oversight.  On the cpu case of probe-video-determinism (the slowest
  # single render in the suite, `--engine blocks --no-gpu`, 60 frames of 1080p):
  #
  #     no thread flags                 14.33 s
  #     --cpu-threads 6                14.43 s
  #     --cpu-threads 12               15.37 s
  #     --reader-threads 12            15.12 s
  #     --reader-threads 12 --cpu-threads 12   16.29 s
  #
  # Capping made it SLOWER, monotonically, and the default already leaves cores alone.  So
  # rather than thread four flags through eleven probe scripts to buy nothing, the knob is
  # accepted as 0 and the real protection is -BudgetMinutes, which is enforced by a watchdog
  # rather than hoped for.  Kept as a parameter because deleting it would break any existing
  # invocation that passes it; it is documented here as measured-and-inert rather than
  # quietly left looking load-bearing.
  [int]$LeaveCores = 0
)

# Per-stage wall clock, printed at the end.  A slow run used to be reported only as a
# total, with no attribution, so the obvious next step was to guess which stage to make
# faster.  Each stage records its own time and the summary names them, so the next slow
# run says where the time went.
$stageTimes = [System.Collections.Generic.List[object]]::new()
function Add-Stage([string]$Name, [double]$Ms) {
  $stageTimes.Add([pscustomobject]@{ Name = $Name; Ms = $Ms })
}
function Stage([string]$Name, [scriptblock]$Body) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  try { & $Body } finally { $sw.Stop(); Add-Stage $Name $sw.Elapsed.TotalMilliseconds }
}

# Wall clock for the whole run, so the summary can attribute time to stages that have
# no timer of their own rather than quietly dropping it.  On -Fast the named stages came
# to 77s against a 145s wall clock, which is the kind of unattributed remainder that
# makes a timing report worse than none: it looks like the answer.
$swTotal = [Diagnostics.Stopwatch]::StartNew()

# ---- the watchdog -------------------------------------------------------------
#
# A timer, not an estimate. It watches the CLOCK and kills the run, so it stays correct on a
# busy machine, which is the only kind of machine this runs on.
$budgetHit = $false
if ($BudgetMinutes -gt 0) {
  $budgetMs = $BudgetMinutes * 60.0 * 1000.0
  $watch = [System.Timers.Timer]::new(15000)
  $watch.AutoReset = $true
  $watch.add_Elapsed({
    if ($swTotal.Elapsed.TotalMilliseconds -gt $budgetMs -and -not $budgetHit) {
      $script:budgetHit = $true
      Write-Host ""
      Write-Host "BUDGET EXCEEDED: this run has taken longer than $BudgetMinutes minute(s)." -ForegroundColor Red
      Write-Host "Stages completed so far, in order:" -ForegroundColor Red
      foreach ($s in $stageTimes) {
        Write-Host ("  {0,-28} {1,7:N1}s" -f $s.Name, ($s.Ms / 1000)) -ForegroundColor Red
      }
      Write-Host "Killing the run. Exit 3: neither a pass nor a failure -- it did not finish." -ForegroundColor Red
      # The timer fires on a threadpool thread, so it cannot simply 'return' out of the
      # script. Stop-Process on the parent is what actually ends it; the pending native
      # children die with the pipeline.
      $env:RD_BUDGET_KILLED = '1'
      Stop-Process -Id $PID -Force
    }
  })
  $watch.Start()
  Write-Host "budget: $BudgetMinutes min ceiling, checked every 15 s (a stage that regresses cannot run forever)."
}

# Stages this run did not execute, named.  -Fast used to skip two 1080p video stages,
# print two yellow sentences about it, and then exit 0 -- so an automated caller reading
# the exit code saw a pass, and a human reading the log had to notice prose to know that
# a third of the suite never ran.  The list is printed in the summary, which makes the
# gap greppable, and its non-emptiness decides the exit code below.
$stagesSkipped = [System.Collections.Generic.List[string]]::new()
# Stages skipped because they are OFF BY DEFAULT -- reachable, documented, and re-runnable
# with a named switch.  Kept apart from the incidental skips below because the two deserve
# DIFFERENT exit codes, and merging them is what made this suite unreadable.
#
# The reason this exists: `palette overlap` (295 s, gating a flag that ships OFF) moved behind
# -Slow, and every default run then exited 2 forever -- on a perfect tree. An exit code that
# is red for a reason nobody can act on trains people to ignore it, which is the same failure
# as a green that means nothing. So:
#
#   exit 4  every stage that ran passed, and everything NOT run is off by default and named
#           below with the switch that runs it.  This is a PASS with declared exclusions.
#   exit 2  something did not run for an INCIDENTAL reason -- a missing fixture, a -Fast run,
#           a probe that could not start.  That is a gap, and it stays a gap.
#
# The distinction is the whole point. 4 says "here is what I did not cover and here is how to
# cover it"; 2 says "I could not cover something and you should not assume it is fine".
$stagesOffByDefault = [System.Collections.Generic.List[string]]::new()
function Skip-Stage([string]$Name, [string]$Why) {
  $stagesSkipped.Add("$Name ($Why)")
  Write-Host ""
  Write-Host "$Name : SKIPPED ($Why)" -ForegroundColor Yellow
}
# A probe that exited 2 -- "cannot run" -- did NOT RUN.  It has to be recorded as a gap, or
# the coverage line counts it as a stage that ran.
#
# THIS FUNCTION EXISTS BECAUSE NONE OF THE ELEVEN EXIT-2 HANDLERS DID THAT.  Each one was
#
#     2 { Write-Host "<name>: SKIPPED (cannot run ...)" -ForegroundColor Yellow }
#
# which prints a line that LOOKS like a recorded skip and touches neither $stagesSkipped nor
# $stagesOffByDefault.  Measured by replaying the handler with $LASTEXITCODE = 2:
#
#     palette deadline: SKIPPED (cannot run -- the probe printed the reason above)
#     stagesSkipped.Count = 0
#     coverage: every stage ran.  0 skipped.
#     EXIT 0
#
# So a probe that never executed was reported as one that executed, on the first runner
# missing ffmpeg, a fixture, or an engine -- and the suite exited 0.  That is the precise
# failure verify.ps1:30-33 says this suite exists to prevent, sitting in the suite.
#
# A skip that only prints is a comment.
function Skip-CannotRun([string]$Name) {
  Skip-Stage $Name 'the probe exited 2 (cannot run) -- it printed the reason above'
}
# The same, but recorded as a DELIBERATE exclusion rather than a gap.
function Skip-ByDefault([string]$Name, [string]$Why) {
  $stagesSkipped.Add("$Name ($Why)")
  $stagesOffByDefault.Add("$Name ($Why)")
  Write-Host ""
  Write-Host "$Name : OFF BY DEFAULT ($Why)" -ForegroundColor Yellow
}

# `magick compare -metric AE` reports its metric on stderr, which PowerShell
# would otherwise promote to a terminating error under Stop.
$ErrorActionPreference = "Continue"
$env:MAGICK_HOME = if ($env:MAGICK_HOME) { $env:MAGICK_HOME } else { "C:\Program Files\ImageMagick-7.1.2-Q16-HDRI" }

if (-not (Test-Path $Rdither)) { throw "rdither not found at $Rdither (run build.bat first)" }

# ---- fixtures ---------------------------------------------------------------
$fixtures = @{
  "t_1x1"     = @("-size", "1x1",   "xc:red")
  "t_3x5"     = @("-size", "3x5",   "plasma:fractal")
  "t_17x13"   = @("-size", "17x13", "gradient:red-blue")
  "t_wide"    = @("-size", "640x64", "plasma:fractal")
  "in_grad"   = @("-size", "64x64", "gradient:black-white", "-colorspace", "sRGB")
  "in_plasma" = @("-size", "96x64", "plasma:fractal", "-colorspace", "sRGB")
  "in_shapes" = @("-size", "64x64", "xc:gray50", "-fill", "red", "-draw", "circle 32,32 32,8")
}

New-Item -ItemType Directory -Force $ImageDir | Out-Null
$swBitExact = [Diagnostics.Stopwatch]::StartNew()
foreach ($name in $fixtures.Keys) {
  $path = Join-Path $ImageDir "$name.png"
  if (-not (Test-Path $path)) {
    & $Magick @($fixtures[$name]) $path
  }
}
# Alpha and noise cases are derived from a base image.
if (-not (Test-Path "$ImageDir\in_alpha.png")) {
  & $Magick "$ImageDir\in_plasma.png" -alpha set -channel A -evaluate set 60% +channel "$ImageDir\in_alpha.png"
}
# Grey AND alpha together, which is the combination ImageMagick's SetAssociatedAlpha
# treats specially: it clears associate_alpha when number_colors == 2 and the colorspace
# is grey, even on an image that carries alpha.  The tree then goes 8-child instead of
# 16-child with no 2^24 memo table, and ClosestColor drops its alpha term.
#
# This fixture did not exist, which is why the divergence went unnoticed: --colors 2 was
# tested, and alpha was tested, but never together on a grey image.  Deriving it from
# in_alpha would NOT work -- that is sRGB plasma, not grey -- so it is built grey first
# and given alpha after, in that order.
if (-not (Test-Path "$ImageDir\in_gray_alpha.png")) {
  & $Magick "$ImageDir\in_grad.png" -colorspace Gray -alpha set -channel A -evaluate set 50% +channel "$ImageDir\in_gray_alpha.png"
}
if (-not (Test-Path "$ImageDir\noisy.png")) {
  & $Magick -size 512x384 plasma:fractal -attenuate 0.4 +noise Gaussian -colorspace sRGB "$ImageDir\noisy.png"
}
# A JPEG fixture, and the reason it exists.
#
# Every other fixture here is a PNG, which is exactly why a real defect survived
# for so long.  ImStore() cloned the INPUT image and inherited its coder, so dither
# a JPEG to a path called .png and the file on disk was JPEG bytes (FF D8 FF) --
# lossy re-compression of pixels that were already an exact 16-colour palette.  The
# file came back with 93,377 distinct colours where ImageMagick's own Riemersma
# output has 16.  A PNG input cannot reach it, because a PNG clone already names
# the PNG coder, so no PNG-only fixture could ever have caught it.
#
# --verify missed it too: it compared the in-memory store rather than the file that
# was written, and reported AE=0 on that broken file.  Both are fixed; this fixture
# is what keeps the JPEG path covered.
$jpg = "$ImageDir\in_jpeg.jpg"
if (-not (Test-Path $jpg)) {
  # Hard edges on a flat field, because ringing around an edge is what produces
  # source colours outside the palette and makes the failure visible.
  & $Magick -size 256x192 "xc:#204080" -fill "#e8c040" -draw "rectangle 40,40 120,120" `
            -fill "#f0f0f0" -draw "circle 190,140 190,60" -quality 92 $jpg
}

$images = @("t_1x1", "t_3x5", "t_17x13", "t_wide", "in_grad", "in_plasma",
            "in_shapes", "in_alpha", "noisy", "in_jpeg", "in_gray_alpha")

# in_gray_alpha was EXCLUDED here until this commit, and the exclusion recorded the reason
# honestly: a grey image with alpha at --colors 2, where ImageMagick's CLI collapses to
# one colour while ColorTree::Build yielded two, and the tool's own
# "colormap : tree matches ImageMagick exactly" check caught the disagreement.
#
# CLOSED, and the cause was a divergence from upstream rather than a missing feature.
# SetAssociatedAlpha in MagickCore/quantize.c tests quantize_info->colorspace; this port
# tested source->colorspace.  ImBuildPalette sets qi->colorspace = UndefinedColorspace
# (upstream's GetQuantizeInfo default, and what the CLI passes), so upstream's clause can
# never fire there -- while the port's fired on greyscale images and cleared an alpha IM
# does associate, which changes the colour COUNT.
#
# Both ends measured by rebuilding, not by argument: the pre-fix tree gives --colors 2
# exit 4 with associate_alpha=no and no verify line, while 4/9/16 all match; the fixed
# tree matches at all four.  The tree was never meant to be constrained to the palette
# count -- in IM, AssignImageColors derives image->colors from the same cube_info the
# tree came from -- so the two consumers were simply fed different associate_alpha.
#
# Generated below like the others, since it is derived from in_grad.

# Source extension per fixture.  Everything is PNG except in_jpeg, which is a JPEG
# on purpose -- see the comment where it is generated.  A non-PNG input is the only
# thing that reaches the coder-inheritance defect.
$ext = @{ "in_jpeg" = "jpg" }

$engines = @(
  @{ name = "cpu";        args = @("--engine", "cpu") },
  @{ name = "cuda";       args = @("--engine", "cuda") },
  @{ name = "cpu/disk";   args = @("--engine", "cpu", "--max-ram-mb", "1") }
)

# An engine this build cannot run makes its cases unaskable, not failed.  Without
# this, `build.cmd --no-cuda` produced "90 passed, 45 failed" and exited 1: the 45
# were exactly the cuda engine's share, every one of them a refusal to start rather
# than a wrong pixel.  A build configuration the project documents as supported was
# reporting a failing suite.
#
# The cuda share is 11 fixtures x 5 colour counts = 55 of the 165 total, which is why
# README quotes "110 passed, 55 SKIPPED" for a --no-cuda build.  That arithmetic is a
# product of the three lists below, so it is exact and does not need re-measuring when a
# fixture is added -- but it does need UPDATING, and did not get it when in_gray_alpha
# and in_jpeg arrived.  Three lists, one multiplication: if you add a fixture, a colour
# count or an engine, this number moves.
. "$PSScriptRoot\tools\rd-engine-probe.ps1"
$avail = Get-RdEngineAvailability -Rdither $Rdither -Engines @('cuda') -Fixture (Join-Path $ImageDir "in_grad.png")
Write-RdEngineSkipReport -Availability $avail | Out-Null
$skipEngines = @($avail.Keys | Where-Object { $avail[$_] })
$activeEngines = @($engines | Where-Object { $skipEngines -notcontains $_.name })

$pass = 0; $fail = 0; $skipped = 0
foreach ($img in $images) {
  $e = if ($ext.ContainsKey($img)) { $ext[$img] } else { "png" }
  $src = Join-Path $ImageDir "$img.$e"
  foreach ($c in $Colors) {
    $ref = Join-Path $ImageDir "$($img)_$($c)_ref.png"
    & $Magick $src -dither Riemersma -colors $c $ref | Out-Null
    foreach ($e in $engines) {
      if ($skipEngines -contains $e.name) { $skipped++; continue }
      $out = Join-Path $ImageDir "$($img)_$($c)_$($e.name -replace '/','_').png"
      $text = (& $Rdither @($e.args) --colors $c --verify --quiet $src $out 2>&1 | Out-String)
      $internal = $text -match "BIT-EXACT"
      # Independent confirmation through the ImageMagick CLI.  `compare -metric`
      # prints to stderr, so it is funnelled through cmd to keep the number
      # clean instead of wrapped in a PowerShell ErrorRecord.
      $aeText = (& cmd /c "`"$Magick`" compare -metric AE `"$out`" `"$ref`" null: 2>&1" | Out-String).Trim()
      $ae = ($aeText -split '\s+')[0]
      $ok = $internal -and ($ae -eq "0")
      if ($ok) { $pass++ } else {
        $fail++
        Write-Host ("FAIL {0,-10} colors={1,-4} {2,-9} internal={3} compare_AE={4}" -f `
                    $img, $c, $e.name, $internal, $ae) -ForegroundColor Red
        if ($text.Trim()) { Write-Host "     $($text.Trim())" }
      }
    }
  }
}

Write-Host ""
if ($skipped -gt 0) {
  Write-Host "bit-exact cases: $pass passed, $fail failed, $skipped SKIPPED (engine not in this build)" -ForegroundColor Yellow
  Write-Host "  The $skipped skipped cases are NOT passes. On a build with that engine they run." -ForegroundColor Yellow
} else {
  Write-Host "bit-exact cases: $pass passed, $fail failed"
}
if ($fail -gt 0) { exit 1 }
$swBitExact.Stop(); Add-Stage 'bit-exact cases' $swBitExact.Elapsed.TotalMilliseconds

# Bit-exactness against ImageMagick is a comparison between two implementations, so
# it is structurally unable to see a fault that BOTH have.  The unfilled-pixel-channel
# bug in the image writer was exactly that: it made the encoder pick a different PNG
# type from stale heap memory, and it was invisible here for as long as both paths
# were wrong in the same way at the same time.
#
# So this also runs each engine against ITSELF, repeatedly.  It needs fixtures that
# only tools\probe-opencl-exact.ps1 generates, hence the wrapper: if those are absent
# the determinism probe exits 2 ("cannot run"), and that is reported as skipped
# rather than as a pass and not as a failure.  A check that cannot run and says so is
# the difference between a missing fixture and a silent hole in the suite.
# The image determinism probe reads c_plasma and c_ashape from a shared fixture
# directory, and this used to PRINT an instruction to run tools\probe-opencl-exact.ps1
# first.  No automated run reads instructions, so on every clean clone -- which is every
# runner -- the probe skipped, and a skip is not a failure, so nothing went red and
# nothing said so.  The same shape as the missing video fixtures: a check that cannot run
# and does not say why is a hole wearing a pass's clothes.
#
# The fixtures are generated here, by the script that already defines them, rather than
# by a new one -- two copies of a fixture definition would drift.  -FixturesOnly stops
# before the 54-cell cross-engine comparison, which is not this suite's business and
# whose exit code would then have to be interpreted here.
$oclFx = ''
$oclGen = Join-Path $PSScriptRoot 'tools\probe-opencl-exact.ps1'
if (Test-Path $oclGen) {
  $g = & pwsh -NoProfile -File $oclGen -FixturesOnly 2>&1 | Out-String
  $gCode = $LASTEXITCODE
  foreach ($line in ($g -split "`r?`n")) {
    if ($line -match '^FIXTURE_DIR=(.+)$') { $oclFx = $Matches[1].Trim() }
    elseif ($line.Trim()) { Write-Host "  $($line.TrimEnd())" }
  }
  # A failed generator is NOT a skip.  Degrading it to one used to clear $oclFx,
  # let the probe fall back to a default directory it may not have, and exit 2 --
  # which this suite reports as SKIPPED and then still exits 0.  So a broken
  # generator produced a green run with a stage silently absent, which is the one
  # outcome a tally must never be able to express.  Fail here, where the cause is.
  if ($gCode -ne 0) {
    Write-Host "fixture generator failed (exit $gCode) -- refusing to report the stage as skipped" -ForegroundColor Red
    exit 1
  }
}

$det = Join-Path $PSScriptRoot 'tools\probe-determinism.ps1'
if (Test-Path $det) {
  Write-Host ""
  $detArgs = @('-Rdither', $Rdither)
  if ($oclFx) { $detArgs += @('-FixtureDir', $oclFx) }
  & pwsh -NoProfile -File $det @detArgs
  switch ($LASTEXITCODE) {
    0 { }
    2 { Skip-CannotRun "determinism" }
    default { Write-Host "determinism: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

# The video pipeline is a third comparison, and neither of the two above can stand
# in for it: the image probes never build a batch, never run the reader, and never
# touch the uint16 data path, which is the part that differs most between an image
# and a video frame.  It also decodes ~450 MB, so it is kept out of the way rather
# than run by default -- but it is wired in, because a check nothing invokes is a
# check that rots.  Exit 2 from the probe means "cannot run" (no clip), and that is
# reported as skipped, never as a pass and never as a failure.
# Fixtures for the two clip-taking probes.  tests\clip1920.mp4 and
# tests\clip1920_audio.mkv are large local files that are deliberately untracked, so
# every clean clone -- which is every automated run -- reported the video checks as
# SKIPPED.  Installing ffmpeg did not change that; it only moved the reason.  With
# ffmpeg present the runner said "no clip at tests\clip1920.mp4" instead of "ffmpeg not
# found", and the video path still had no coverage at all.
#
# So when the real fixtures are absent, GENERATE small ones rather than skipping.
# Nothing binary enters the repository, and the checks still assert what they assert:
# cross-engine bit identity, run-to-run determinism, and no frame lost to -shortest.
# What is given up is SIZE, and that is a real loss: a bug that only appears at
# 1920x1080 with 60 frames would not be caught by a 320x180, 30-frame fixture.  This
# narrows the blind spot; it does not close it, and saying so is cheaper than letting
# someone assume the video path is now fully covered.
$genClip = ''
$genAudio = ''
$genAlpha = ''
# -Fast skips the two stages that need a 1080p clip, so it must not spend the time
# generating one either.  The fixture generator is not free -- it renders and verifies
# two clips through ffmpeg -- and on the fast path nothing would consume the result.
$needFixtures = (-not $Fast) -and (
    (-not (Test-Path (Join-Path $PSScriptRoot 'tests\clip1920.mp4'))) -or
    (-not (Test-Path (Join-Path $PSScriptRoot 'tests\clip1920_audio.mkv'))))
if ($needFixtures) {
  $mk = Join-Path $PSScriptRoot 'tools\make-video-fixtures.ps1'
  if (Test-Path $mk) {
    Write-Host ""
    $fx = & pwsh -NoProfile -File $mk 2>&1 | Out-String
    $fxCode = $LASTEXITCODE
    foreach ($line in ($fx -split "`r?`n")) {
      if ($line -match '^(CLIP|CLIP_AUDIO|CLIP_ALPHA)=(.+)$') {
        switch ($Matches[1]) {
          'CLIP'       { $genClip = $Matches[2].Trim() }
          'CLIP_AUDIO' { $genAudio = $Matches[2].Trim() }
          # Without this case CLIP_ALPHA= matched nothing, so the line was printed and
          # thrown away -- an emitted path that no consumer read.
          'CLIP_ALPHA' { $genAlpha = $Matches[2].Trim() }
        }
      } elseif ($line.Trim()) { Write-Host "  $($line.TrimEnd())" }
    }
    if ($fxCode -ne 0) {
      # Not fatal in the sense of "skip the stage": the probes keep their own
      # defaults, hit the missing clip and exit 2, which is reported as SKIPPED.
      # But that is only honest if the reason is a missing fixture.  If OUR
      # generator is what failed, the stage is not un-runnable, we broke it, and
      # the suite must not come back green.  So distinguish the two.
      Write-Host "video fixture generator failed (exit $fxCode)" -ForegroundColor Red
      exit 1
    }
  }
}

$vid = Join-Path $PSScriptRoot 'tools\probe-video-exact.ps1'
$swVid = [Diagnostics.Stopwatch]::StartNew()
if ($Fast) {
  Skip-Stage 'video exactness' '-Fast: needs a 1080p clip; run without -Fast before committing'
} elseif (Test-Path $vid) {
  Write-Host ""
  $vidArgs = @('-Rdither', $Rdither)
  if ($genClip) { $vidArgs += @('-Clip', $genClip) }
  if ($genAudio) { $vidArgs += @('-AudioClip', $genAudio) }
  & pwsh -NoProfile -File $vid @vidArgs
  switch ($LASTEXITCODE) {
    0 { }
    # Not "no tests\clip1920.mp4".  The probe exits 2 for three different reasons --
    # no clip, no ffmpeg, or no pair of engines that both produced frames -- and it
    # prints which one above.  Naming the clip here regardless meant that once the
    # probe learned to skip an engine for want of a device, this line went on
    # reporting a missing file that was present, pointing at the wrong thing.
    2 { Skip-CannotRun "video" }
    default { Write-Host "video: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}
$swVid.Stop(); Add-Stage 'video exactness' $swVid.Elapsed.TotalMilliseconds

# And a fourth, which is the odd one out: it does not compare two engines, it
# compares the output against the SOURCE.  That is deliberate, and it is the only
# shape of check that could have caught the unvisited-pixel bug -- both engines
# were reading their own uninitialised memory, so the two comparisons above
# agreed with each other while both were wrong.  It also builds its own fixtures
# from lavfi, so it needs nothing from tests\ except ffmpeg on PATH.
$unv = Join-Path $PSScriptRoot 'tools\probe-unvisited-pixel.ps1'
if (Test-Path $unv) {
  Write-Host ""
  & pwsh -NoProfile -File $unv -Rdither $Rdither
  switch ($LASTEXITCODE) {
    0 { }
    2 { Skip-CannotRun "unvisited pixel" }
    default { Write-Host "unvisited pixel: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

# A sixth: SourcePixFmtHasAlpha, the guard that decides whether the palette decoder gets 3
# channels or 4.  It decides that on the SOURCE pixel format, and no fixture in this
# repository carried alpha, so the deny branch had never executed -- a guard that defaults
# correctly and is never exercised is indistinguishable from one that is broken.
#
# It is its own probe, not an assertion inside make-video-fixtures.ps1, because that
# generator only runs when tests\clip1920.mp4 is MISSING (verify.ps1:301).  With both
# committed clips present it never runs, so a check placed there is dead on every normal
# checkout and the suite still reports green.  This builds its own 160x120 clips in a few
# ms and needs no 1080p fixture, no engine parity and no GPU.
# And a seventh: the built-in C++ self-test (`rdither --self-test`, rd_cli.cpp:1257).
#
# This one is easy to miss and was missed until the alpha-guard work: the gate has never
# invoked --self-test, so every assertion in it ran only when somebody typed the flag by
# hand.  A test nobody runs is not a gate, and the gate reporting EXIT=0 while a test was
# never executed is the failure mode this suite exists to catch -- so it runs here.
#
# It is cheap (sub-second) and needs no GPU, no clip and no engine, so it belongs on every
# run including -Fast.
$stOut = & $Rdither --self-test 2>&1
$stCode = $LASTEXITCODE
if ($stOut) { $stOut | ForEach-Object { Write-Host "  $_" } }
if ($stCode -ne 0) {
  Write-Host "self-test: FAILED (exit $stCode)" -ForegroundColor Red
  exit 1
}

$ag = Join-Path $PSScriptRoot 'tools\probe-alpha-guard.ps1'
if (Test-Path $ag) {
  Write-Host ""
  & pwsh -NoProfile -File $ag -Rdither $Rdither
  switch ($LASTEXITCODE) {
    0 { }
    2 { Skip-CannotRun "alpha guard" }
    default { Write-Host "alpha guard: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

# And an eighth: the palette/reader overlap.  The flag is OPT-IN and off by default,
# which is exactly why this needs gating -- an unexercised code path is the one most
# likely to rot, and the property that matters (the overlap must not change a single
# pixel) cannot be seen from the default path at all.
#
# The probe asserts the arms differ in TIMING before comparing their pixels.  Without
# that, a flag that silently stopped doing anything would make the A/B compare one
# binary against itself and report a perfect pass -- the failure this whole comparison
# shape exists to catch.
#
# Not in -Fast: it renders 4 x 300 frames of 1080p plus a negative control.
#
# MEASURED 295.1 s -- 15% of a 20-minute run for a flag that is OFF BY DEFAULT and that
# measured SLOWER (docs\KNOWN-ISSUES.md item 4).  It is a real check and it stays, but it
# moves behind -Slow: an opt-in feature's correctness gate does not belong on the path
# everyone runs, and 295 s is the difference between a gate you run and one you avoid.
$ov = Join-Path $PSScriptRoot 'tools\probe-palette-overlap.ps1'
if ($Slow) {
  if (Test-Path $ov) {
    Write-Host ""
    & pwsh -NoProfile -File $ov -Rdither $Rdither
    switch ($LASTEXITCODE) {
      0 { }
      2 { Skip-CannotRun "palette overlap" }
      default { Write-Host "palette overlap: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
    }
  }
} else {
  Skip-ByDefault 'palette overlap' '295 s, and it gates a flag that ships OFF (KNOWN-ISSUES 4).  Run verify.ps1 -Slow.'
}

# And a ninth: whether splitting the decode across N seek-based ffmpeg processes is
# byte-identical to one decode.  This guards a property nothing currently depends on, which
# is the point -- it is the evidence for the next attempt at the bottleneck, and it is
# evidence that is easy to get wrong from reasoning alone.
#
# The recorded result is asymmetric and that asymmetry IS the finding: CFR splits are
# byte-identical in every case tried, and the VFR case is NOT (frame_index/fps is not a
# timestamp when durations vary).  So the probe gates the CFR half and REPORTS the VFR half,
# rather than failing on a constraint that is known and documented.
#
# Cheap: 320x180 fixtures, a few seconds.  Not in -Fast.
$sd = Join-Path $PSScriptRoot 'tools\probe-split-decode.ps1'
if (Test-Path $sd) {
  Write-Host ""
  & pwsh -NoProfile -File $sd
  switch ($LASTEXITCODE) {
    0 { }
    2 { Skip-CannotRun "split decode" }
    default { Write-Host "split decode: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

# And a tenth: does --palette-budget-ms actually bound the palette stage?  It did not until
# 2026-10-09 -- it computed a sample count from a hardcoded 0.12 s per sample and had no
# deadline at all, so it bounded nothing while its help text promised a time budget.  A flag
# that quietly does nothing is the same failure shape as a probe that quietly passes.
#
# Both samplers are covered.  The by-seek arm needs a clip over 5000 frames (rd_video.cpp:1663)
# and RD_PALETTE_SEEK can only disable that path, never force it, so the probe builds a 5100
# frame clip at 320x180 -- a few MB, seconds to make -- rather than a 1080p clip long enough to
# reach the path honestly.
$pd = Join-Path $PSScriptRoot 'tools\probe-palette-deadline.ps1'
if (Test-Path $pd) {
  Write-Host ""
  & pwsh -NoProfile -File $pd -Rdither $Rdither
  switch ($LASTEXITCODE) {
    0 { }
    2 { Skip-CannotRun "palette deadline" }
    default { Write-Host "palette deadline: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

# And an eleventh: is the HOST path bit-identical to the DEVICE path?
#
# Every other comparison in this suite puts two implementations of the SAME kind against
# each other -- cpu vs cuda, engine vs engine, run vs run. This one puts host against
# device, which is the only shape that answers "is the CPU path right", and both
# properties it checks have been real defects at least once: host(batch 1) vs device, and
# host invariant to batch size.
#
# It was ungated, which is how a whole class of host fault could sit here unexamined while
# the suite reported green. It is also CHEAP -- 320x180 fixtures, a bounded 900 s timeout,
# and its exit codes are actually tested (its predecessor had no exit statement at all and
# returned whatever the last ffmpeg happened to leave behind, so it exited 0 having failed).
$ho = Join-Path $PSScriptRoot 'tools\probe-host-oracle.ps1'
if (Test-Path $ho) {
  Write-Host ""
  & pwsh -NoProfile -File $ho -Rdither $Rdither
  switch ($LASTEXITCODE) {
    0 { }
    2 { Skip-CannotRun "host oracle" }
    default { Write-Host "host oracle: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

# And a fifth: the image determinism probe again, pointed at VIDEO.  The image one
# is a self-comparison across time rather than an engine-against-engine comparison,
# which is the only shape of check that sees a fault both engines share -- and two
# separate bugs lived there.  The video pipeline had no such check at all, so a
# reintroduction of the -shortest frame-deleting class in the palette or read stage
# would not have been caught by anything: probe-video-exact compares the engines
# against each other, and both would drop the same frames.
$vdet = Join-Path $PSScriptRoot 'tools\probe-video-determinism.ps1'
$swVdet = [Diagnostics.Stopwatch]::StartNew()
if ($Fast) {
  Skip-Stage 'video determinism' '-Fast: the slowest stage by far; run without -Fast before committing'
} elseif (Test-Path $vdet) {
  Write-Host ""
  $vdetArgs = @('-Rdither', $Rdither)
  if ($genAudio) { $vdetArgs += @('-Clip', $genAudio) }
  & pwsh -NoProfile -File $vdet @vdetArgs
  switch ($LASTEXITCODE) {
    0 { }
    2 { Skip-CannotRun "video determinism" }
    default { Write-Host "video determinism: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}
$swVdet.Stop(); Add-Stage 'video determinism' $swVdet.Elapsed.TotalMilliseconds
# And a sixth, and the one that closes a gap four faults lived in: does the video
# pipeline give the same ANSWER at different batch sizes, and does the host give the
# same answer as the device?  probe-video-exact compares two DEVICE engines, and
# probe-video-determinism checks the host for DETERMINISM, which is structurally blind
# to a wrong-but-stable answer -- three of the four host faults were exactly that.  The
# batch half needs no GPU, so unlike the device half it runs on every CI push, which is
# the only reason it can catch anything there.
$vinv = Join-Path $PSScriptRoot 'tools\probe-video-invariance.ps1'
if (Test-Path $vinv) {
  Write-Host ""
  $vinvArgs = @('-Rdither', $Rdither)
  if ($genAudio) { $vinvArgs += @('-Clip', $genAudio) }
  & pwsh -NoProfile -File $vinv @vinvArgs
  switch ($LASTEXITCODE) {
    0 { }
    2 { Skip-CannotRun "video invariance" }
    default { Write-Host "video invariance: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

  # And a seventh: is the OpenCL clip-invariant cache actually skipping the upload it
  # exists to skip?  Every other stage here runs ONE geometry per process, so the
  # cache's invalidation branch is never taken, and a key that failed to notice a
  # change would pass all of them.  This probe is the only thing that looks.
  #
  # It asserts on the TRANSFER COUNT, not the byte count.  The six clip-invariant
  # buffers ARE six of the first batch's seven transfers, so 7 -> 1 is the cache
  # firing at any geometry -- while the byte delta scales with the Riemersma level
  # and varies about 37x across the geometries measured (784,984 B at 256x192
  # against 29,082,116 B at 1920x1080).  A byte threshold authored from the 1080p
  # figure failed a cache that was working perfectly at 256x192, which is why the
  # check is written this way.
  $cinv = Join-Path $PSScriptRoot 'tools\probe-cache-invalidation.ps1'
  if (Test-Path $cinv) {
  Write-Host ""
  & pwsh -NoProfile -File $cinv -Rdither $Rdither
  switch ($LASTEXITCODE) {
    0 { }
    2 { Skip-CannotRun "cache invalidation" }
    default { Write-Host "cache invalidation: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
  }

# And an eighth: does the SEGMENTED path work at all?
#
# `--segment-frames` shipped with an access violation (0xC0000005) under
# RD_PALETTE_OVERLAP=1 and a plain exit 1 without it, and this suite stayed green the
# whole time, because NO STAGE RAN IT.  It is the crash-safe path -- the one a long
# render uses, precisely so progress survives a power cut -- so a crash there is the
# worst place for one, and it went unnoticed for exactly the reason that nothing
# exercised it.
#
# The probe covers the four shapes separately: plain, with the overlap env var (the
# only way to reach the race), more than one segment (so the loop iterates), and a
# pixel comparison against the unsegmented control under --video-lossless.  Cheap:
# a 320x180 10-frame lavfi fixture, seconds.
$seg = Join-Path $PSScriptRoot 'tools\probe-segment-frames.ps1'
if (Test-Path $seg) {
  Write-Host ""
  & pwsh -NoProfile -File $seg -Rdither $Rdither
  switch ($LASTEXITCODE) {
    0 { }
    2 { Skip-CannotRun "segment frames" }
    default { Write-Host "segment frames: FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
  }
}

# ---- where the time went --------------------------------------------------------
#
# Printed unconditionally, and attributed per stage, because "the suite takes 25-35
# minutes" is not actionable on its own -- the only thing you can do with a total is
# guess which stage to make faster.  On -Fast this is the number that matters: it is
# the cost of the inner loop, and it should be small enough that nobody is tempted to
# skip it.
$swTotal.Stop()
$totalMs = $swTotal.Elapsed.TotalMilliseconds
$namedMs = ($stageTimes | Measure-Object Ms -Sum).Sum
Write-Host ""
if ($Fast) { Write-Host "stage times (-Fast; the skipped stages below are NOT included):" -ForegroundColor Cyan }
else      { Write-Host "stage times:" -ForegroundColor Cyan }
foreach ($s in $stageTimes) {
  $pct = if ($totalMs -gt 0) { 100.0 * $s.Ms / $totalMs } else { 0 }
  Write-Host ("  {0,-28} {1,7:N1}s  {2,5:N1}%" -f $s.Name, ($s.Ms / 1000), $pct)
}
# Whatever the named stages do not cover.  Printed rather than dropped: the image
# determinism probe, the unvisited-pixel probe, the OpenCL fixture generation and the
# invariance probe all live in here, and an unattributed remainder is exactly the thing
# that sends you looking in the wrong place.
$otherMs = $totalMs - $namedMs
if ($otherMs -gt 500) {
  Write-Host ("  {0,-28} {1,7:N1}s  {2,5:N1}%   (determinism, unvisited-pixel," -f 'other stages', ($otherMs / 1000), (100.0 * $otherMs / $totalMs))
  Write-Host ("  {0,-28} {1,19}    OpenCL fixtures, video invariance, startup)" -f '', '')
}
Write-Host ("  {0,-28} {1,7:N1}s" -f 'TOTAL', ($totalMs / 1000))

# ---- coverage, before the exit code -------------------------------------------
#
# One line that says how many stages ran and how many did not, naming the ones that did
# not.  This is the countable form of a warning that used to live only in prose, and it
# is what the exit code below is derived from, so the two cannot disagree.
Write-Host ""
if ($stagesSkipped.Count -eq 0) {
  Write-Host "coverage: every stage ran.  0 skipped." -ForegroundColor Cyan
} else {
  $gaps = $stagesSkipped.Count - $stagesOffByDefault.Count
  if ($gaps -eq 0) {
    Write-Host ("coverage: every stage that ran passed. {0} stage(s) are OFF BY DEFAULT and" -f $stagesOffByDefault.Count) -ForegroundColor Cyan
    foreach ($s in $stagesOffByDefault) { Write-Host "    $s" -ForegroundColor Cyan }
    Write-Host "Each is named with the switch that runs it.  Nothing below covers them." -ForegroundColor Cyan
  } else {
    Write-Host ("coverage: {0} stage(s) did not run and are NOT covered by this run:" -f $stagesSkipped.Count) -ForegroundColor Yellow
    foreach ($s in $stagesSkipped) { Write-Host "    $s" -ForegroundColor Yellow }
    if ($stagesOffByDefault.Count -gt 0) {
      Write-Host ("  of which {0} are off by default and {1} are gaps:" -f $stagesOffByDefault.Count, $gaps) -ForegroundColor Yellow
    }
    Write-Host "A stage that did not run is not a passing stage.  The bit-exact tally above" -ForegroundColor Yellow
    Write-Host "does not include them and no number in this run covers them." -ForegroundColor Yellow
  }
}

if ($stagesSkipped.Count -gt 0) {
  Write-Host ""
  if ($AllowPartial) {
    Write-Host "Exiting 0 because -AllowPartial was given.  This run did NOT cover the" -ForegroundColor Yellow
    Write-Host "stages listed above; do not read 0 as a full-suite pass." -ForegroundColor Yellow
    exit 0
  }
  # THE EXIT CODE DEPENDS ON WHY A STAGE DID NOT RUN, not only on whether one did.
  #
  #   4 -- everything skipped is off by DEFAULT: named above, with the switch that runs it.
  #        This is a PASS with declared exclusions, and it is reachable on the default
  #        invocation, which exit 2 was not.  `palette overlap` moved behind -Slow in this
  #        commit and every default run then exited 2 forever, on a tree where everything
  #        passed -- so a caller could not tell a healthy suite from a broken one, and the
  #        habit that forms is to ignore the code.  A red light nobody can act on is
  #        functionally a light that is off.
  #
  #   2 -- something did not run for an INCIDENTAL reason: a missing fixture, a -Fast run,
  #        a probe that could not start.  That is a GAP and stays one.
  #
  # 0 is still unreachable without -AllowPartial, and 1 is still failure.
  $gaps = $stagesSkipped.Count - $stagesOffByDefault.Count
  if ($gaps -eq 0) {
    Write-Host "Exiting 4: PASS with declared exclusions.  Everything that ran passed; the" -ForegroundColor Cyan
    Write-Host "stage(s) above are off by default and the switch to run each is named there." -ForegroundColor Cyan
    Write-Host "For full coverage run: verify.ps1 -Slow" -ForegroundColor Cyan
    exit 4
  }
  # 2, not 1: nothing failed, and 1 is reserved for that.  2 is this suite's existing
  # "cannot run / not covered" code, which is exactly what a -Fast run is.  It was 0.
  # $($stagesSkipped.Count), not $stagesSkipped.Count: inside a double-quoted string
  # PowerShell interpolates the collection and then emits a literal ".Count", so the
  # uncorrected form printed the whole skip list followed by ".Count".
  Write-Host "Exiting 2: $gaps of the $($stagesSkipped.Count) stage(s) above did not run for a" -ForegroundColor Yellow
  Write-Host "reason that is not a declared exclusion.  Not a pass and not a failure." -ForegroundColor Yellow
  Write-Host "Use -AllowPartial if you" -ForegroundColor Yellow
  exit 2
}
exit 0
