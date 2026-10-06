# SPDX-License-Identifier: GPL-3.0-or-later
<#
  rd-engine-probe.ps1 -- shared helper: which engines can this build actually run?

  Dot-sourced by verify.ps1 and by tools\probe-determinism.ps1.  It exists as one
  file because the check is a list of exact diagnostic strings, and two hand-kept
  copies of a list of strings is a list that will drift.  It already nearly did: the
  determinism probe grew its own copy while this was being written.

  WHY THE CHECK CANNOT BE LOOSER.  "The engine produced no output" does not mean
  "the engine is not in this build" -- it is equally what a genuinely broken engine
  does, and a broken engine is the thing this suite exists to catch.  So the match is
  on the exact messages below, and anything unrecognised is reported as RUNNABLE, so
  the caller runs it and reports the real failure.  Only these strings buy a skip.

  The two strings are deliberate parts of the code's contract, not incidental text:
  the first is the #else stub in src\rd_opencl.cpp, the second is rd_cli.cpp's own
  refusal before it touches a device.  If either is reworded, this silently stops
  skipping and the suite starts failing again -- which is a loud, harmless failure,
  and much better than a loose match that hides a real defect.
#>

function Get-RdEngineAvailability {
  <#
    .SYNOPSIS
      For each engine name, run it once and decide whether this build can run it.

    .PARAMETER Rdither
      Path to rdither.exe.

    .PARAMETER Engines
      Engine names, e.g. @('cpu', 'cuda', 'blocks', 'opencl').

    .PARAMETER Fixture
      A readable input image, used for the one-shot trial run.

    .OUTPUTS
      A hashtable: engine name -> $null when runnable, or a reason string when not.
  #>
  param(
    [Parameter(Mandatory = $true)][string]$Rdither,
    [Parameter(Mandatory = $true)][string[]]$Engines,
    [Parameter(Mandatory = $true)][string]$Fixture
  )

  $result = @{}
  $scratch = [IO.Path]::GetTempFileName()
  $png = [IO.Path]::ChangeExtension($scratch, '.png')
  Remove-Item -EA SilentlyContinue $scratch

  foreach ($eng in $Engines) {
    Remove-Item -EA SilentlyContinue $png
    # $Rdither is QUOTED, and that is the whole fix.  Unquoted, PowerShell parses the
    # leading -R of a value like /home/runner/work/dither 2/build/rdither.exe as the
    # parameter -Rdither, and then tries to cast the NEXT token to the type of that
    # nonexistent parameter.  On this machine the path contains a space and the word
    # "code", so the fragment it tried to convert was literally `code`:
    #
    #     The input string 'code' was not in a correct format.
    #
    # which names a directory rather than the bug, and kills the whole function --
    # including the caller that only wanted to know an engine was unavailable.  The
    # same class of defect as the -RD_WITH_CUDA typo below: one character, and the
    # failure points somewhere else entirely.  Quoting is also just correct; the bare
    # form only worked when the path had no space in it.
    $msg = (& $Rdither --engine $eng --colors 2 $Fixture $png 2>&1) -join "`n"
    $rc = $LASTEXITCODE
    $made = Test-Path $png
    Remove-Item -EA SilentlyContinue $png

    if ($rc -eq 0 -and $made) { $result[$eng] = $null; continue }

    if ($msg -match 'this build has no OpenCL') {
      $result[$eng] = 'this build has no OpenCL (configure with -DRD_WITH_OPENCL=ON)'
    } elseif ($msg -match 'this build has no CUDA') {
      # This said -RD_WITH_CUDA=ON, ONE dash short of the flag.  PowerShell parses a
      # leading -R as a parameter name, so it tried to convert the fragment to an int
      # and the whole function died with
      #
      #     The input string 'code' was not in a correct format.
      #
      # 'code' is the tail of the path "...\dither code gen\dither 2\...", so the
      # message named a directory rather than the bug.  It only fires on the CUDA
      # branch, which is the one verify.ps1 calls at startup, so a build without CUDA
      # raised it -- and the caller saw a non-terminating-looking exit rather than a
      # skip report.  Same single-character shape as the other five defects in ci.yml's
      # header: a value nobody typed twice the same way.
      $result[$eng] = 'this build has no CUDA (configure with -DRD_WITH_CUDA=ON)'
    } elseif ($msg -match 'no CUDA device') {
      $result[$eng] = 'no CUDA device on this machine'
    } else {
      # Unknown.  Assume runnable so the caller surfaces the real error.
      $result[$eng] = $null
    }
  }
  return $result
}

function Write-RdEngineSkipReport {
  <#
    .SYNOPSIS
      Print which engines are unavailable, so reduced coverage is never silent.
  #>
  param([Parameter(Mandatory = $true)]$Availability)

  $skipped = @($Availability.Keys | Where-Object { $Availability[$_] } | Sort-Object)
  if ($skipped.Count -eq 0) { return $false }

  Write-Host ""
  Write-Host "Engines this build cannot run. Every case below that needs one is" -ForegroundColor Yellow
  Write-Host "reported as SKIPPED, not as passed -- the coverage is genuinely smaller" -ForegroundColor Yellow
  Write-Host "on this machine than on a CUDA + OpenCL one:" -ForegroundColor Yellow
  foreach ($k in $skipped) {
    Write-Host ("  {0,-8} {1}" -f $k, $Availability[$k]) -ForegroundColor Yellow
  }
  return $true
}
