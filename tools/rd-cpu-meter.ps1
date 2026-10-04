# SPDX-License-Identifier: GPL-3.0-or-later
<#
  rd-cpu-meter.ps1 -- CPU accounting for one rdither run and the ffmpeg children it
  spawns.  Dot-sourced by tools\probe-cpubudget.ps1 and tools\ab-input-mode.ps1.

  WHY IT IS ITS OWN FILE, AND WHY IT IS NOT THE FOUR LINES IT REPLACES
  =====================================================================
  Both of those scripts used to carry the same loop, and both computed

      child core-s = MAX over samples of ( sum of TotalProcessorTime of the
                     children that happen to be alive at that instant )

  That quantity is not the work the children did.  It is the most the children
  were doing simultaneously at the best moment.  Two separate losses follow, and
  they are both large here:

    1. A child that EXITS leaves the sum for good.  rdither spawns ffmpeg for
       the palette decode (src/rd_video.cpp:1274) and one ffmpeg per palette seek
       (:1730, six at a time) BEFORE it spawns the long-lived video decoder
       (:2310) and encoder (:2517).  Every one of those early children is gone by
       the time the decode/encode pair is running, so its CPU is subtracted from
       the total and never comes back.

    2. Even for children that never overlap, the MAX picks one instant.  The
       palette phase and the encode phase have almost nothing in common, and the
       larger of the two is reported as if it were the whole.

  Measured on a controlled process tree whose parent ran two 4-second CPU-bound
  children SEQUENTIALLY (4.9531 + 4.5938 = 9.5469 core-s of real work):

      max-of-live-sum      4.55 core-s     52% short
      this file            9.48 core-s      0.7% short, and see below on why

  The bias is not symmetric noise.  Losing work that has already finished can only
  ever move the total DOWN, and the work it loses is exactly the work that does
  NOT overlap the encoder -- which is the decoder-side work
  tools/probe-cpubudget.ps1 exists to compare.  A change that made the reader
  cheaper therefore shows up as "core-seconds did not move".  Two conclusions in
  docs/DESIGN.md rest on that number (:2835 and :2895-2896), so it is not a
  cosmetic defect.

  HOW THIS FILE COMPUTES IT INSTEAD
  =================================
  Per-PID, cumulative, and the total is a SUM over every child ever observed, so
  an exited child stays in the answer.  Three measured facts make this exact
  rather than approximate:

    * `Win32_Process.KernelModeTime` + `UserModeTime` is the same cumulative
      counter as `Get-Process`'s TotalProcessorTime.  Measured on this machine,
      the two agreed to 0.0000 s on a live child.  So one CIM row carries both
      the discovery and the CPU, and the per-child `Get-Process` round trip that
      the old loop made is gone.

    * `Start-Process -PassThru` keeps a handle, and the handle's
      TotalProcessorTime is still readable AFTER the process exits.  Measured:
      2.890625 s read from the exited object.  So rdither's OWN cpu needs no
      polling at all -- it is read once, after exit, and is exact.

    * A handle obtained with [Diagnostics.Process]::GetProcessById is readable
      after the CHILD exits, but only while THIS script is still running --
      measured: 2.7187500 s read at t=4.7 s from a child that had been gone for
      2 s, and '' once the parent had exited too.  So each child is read one
      final time the moment it is seen to have exited, while the handle is still
      good, and that value is frozen.  That is where the 0.7% in the table above
      goes: nothing is lost, it is just read at exit rather than at exit minus
      one sampling interval.

  WHAT THIS COSTS
  ===============
  One `Get-CimInstance Win32_Process` per -ChildEveryMs.  Measured on this
  machine: 24.0 ms per filtered call (20 calls), against 0.58 ms for a
  `Get-Process` and 39.4 ms for raw WMI through ManagementObjectSearcher, so this
  is the cheapest route available.  At the 1200 ms default that is 2.0% duty on
  one core, and the default is deliberately NOT raised: the discovery cadence is
  the only thing that decides whether a short-lived child is seen at all, and
  rdither's palette-seek children are short-lived.  -ChildEveryMs exists so the
  cost can be traded deliberately; ChildQueries and MaxChildGapS are reported so
  the trade is visible in the output instead of being a comment.

  If a run produced NO child at all, the child figure is unmeasured rather than
  zero, and the caller must treat it as such -- see tools\probe-cpubudget.ps1.
#>

function Measure-RdRunCpu {
  param(
    # The Start-Process -PassThru object for rdither.  Must not be disposed by
    # the caller before this returns: the handle is what makes the parent's
    # post-exit CPU readable.
    [Parameter(Mandatory = $true)]
    [System.Diagnostics.Process]$Process,

    # How often to ask WMI for a child that has not been seen before.  Lower =
    # fewer missed short-lived children and more load on the machine being
    # measured.  See the header for the measured cost.
    [int]$ChildEveryMs = 1200,

    # Loop tick.  Independent of the child cadence, because the per-child CPU
    # comes from a handle this function already holds rather than from a query.
    [int]$PollMs = 200,

    # 0 means wait as long as it takes.  Non-zero is a hang guard: the process
    # and its children are killed and TimedOut is set, because a probe that waits
    # forever reports nothing at all, which is worse than reporting a failure.
    [int]$TimeoutSec = 0
  )

  $ErrorActionPreference = 'Continue'

  # pid -> @{ Cpu; Final; Name; Handle }.  Keyed by pid, never summed on the fly:
  # the whole point is that a pid keeps its CPU after it leaves the sum.
  $kids = @{}
  $childQueries = 0
  $childSamples = 0
  $nextChildAt = 0.0
  $lastChildAt = -1.0
  $maxGap = 0.0
  $timedOut = $false

  $sw = [Diagnostics.Stopwatch]::StartNew()
  while (-not $Process.HasExited) {
    $t = $sw.Elapsed.TotalSeconds
    if ($TimeoutSec -gt 0 -and $t -gt $TimeoutSec) { $timedOut = $true; break }

    if ($t -ge $nextChildAt) {
      $nextChildAt = $t + ($ChildEveryMs / 1000.0)
      $childQueries++
      $rows = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($Process.Id)" -ErrorAction SilentlyContinue)
      if ($rows.Count -gt 0) {
        $childSamples++
        if ($lastChildAt -ge 0) {
          $gap = $t - $lastChildAt
          if ($gap -gt $maxGap) { $maxGap = $gap }
        }
        $lastChildAt = $t
        foreach ($c in $rows) {
          $id  = [int]$c.ProcessId
          $cpu = ([double]$c.KernelModeTime + [double]$c.UserModeTime) / 1e7
          if (-not $kids.ContainsKey($id)) {
            $h = $null
            try { $h = [Diagnostics.Process]::GetProcessById($id) } catch { $h = $null }
            $kids[$id] = @{ Cpu = $cpu; Final = $null; Name = $c.Name; Handle = $h }
          } elseif ($cpu -lt $kids[$id].Cpu) {
            # Cumulative CPU cannot decrease within one process, so it did.  The
            # pid was recycled and this is a different program; keeping the old
            # number would attribute another process's work to this run.
            if ($kids[$id].Handle) { $kids[$id].Handle.Dispose() }
            $h = $null
            try { $h = [Diagnostics.Process]::GetProcessById($id) } catch { $h = $null }
            $kids[$id] = @{ Cpu = $cpu; Final = $null; Name = $c.Name; Handle = $h }
          }
        }
      }
    }

    # Read every known child from its handle.  This is the accurate part: no
    # query, and it still works for a child that has already exited as long as
    # this script is the one holding the handle.
    foreach ($id in @($kids.Keys)) {
      $e = $kids[$id]
      if ($null -ne $e.Handle) {
        try {
          $v = $e.Handle.TotalProcessorTime.TotalSeconds
          if ($null -ne $v) { $e.Cpu = $v }
          # Freeze the LAST readable value.  Once the handle goes bad -- which
          # happens when this script's own parent goes away -- it reads empty, and
          # an empty read folded into a total would be a silent zero.
          if ($e.Handle.HasExited) { $e.Final = $e.Cpu }
        } catch { }
      }
    }

    Start-Sleep -Milliseconds $PollMs
  }
  $sw.Stop()

  if ($timedOut) {
    # Kill the tree, not just the parent: ffmpeg inherits a pipe from rdither and
    # would otherwise outlive it and keep the output file locked.
    try { $Process.Kill($true) } catch { try { $Process.Kill() } catch { } }
    try { $Process.WaitForExit(15000) | Out-Null } catch { }
  } else {
    $Process.WaitForExit()
  }

  # rdither's own CPU.  Read once, after exit, from the handle Start-Process kept:
  # exact, and free -- no polling of the parent at all.
  $parentCpu = 0.0
  $parentCpuRead = $true
  try {
    $parentCpu = $Process.TotalProcessorTime.TotalSeconds
    if ($null -eq $parentCpu) { $parentCpu = 0.0; $parentCpuRead = $false }
  } catch { $parentCpuRead = $false }

  # One last sweep, for children that were still running at the final tick.
  foreach ($id in @($kids.Keys)) {
    $e = $kids[$id]
    if ($null -ne $e.Handle -and $null -eq $e.Final) {
      try {
        $v = $e.Handle.TotalProcessorTime.TotalSeconds
        if ($null -ne $v) { $e.Cpu = $v; $e.Final = $v }
      } catch { }
    }
  }

  $childCpu = 0.0
  $finalised = 0
  $open = @()
  foreach ($id in @($kids.Keys)) {
    $e = $kids[$id]
    if ($null -ne $e.Final) { $childCpu += $e.Final; $finalised++ }
    else { $childCpu += $e.Cpu; $open += $id }
    if ($e.Handle) { $e.Handle.Dispose() }
  }

  [pscustomobject]@{
    Wall              = [math]::Round($sw.Elapsed.TotalSeconds, 2)
    ParentCpu         = [math]::Round($parentCpu, 2)
    ParentCpuRead     = $parentCpuRead
    ChildCpu          = [math]::Round($childCpu, 2)
    ChildPids         = $kids.Count
    ChildFinalised    = $finalised
    ChildNotFinalised = $open.Count
    ChildQueries      = $childQueries
    ChildSamples      = $childSamples
    MaxChildGapS      = [math]::Round($maxGap, 1)
    TimedOut          = $timedOut
  }
}