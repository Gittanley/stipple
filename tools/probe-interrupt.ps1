# probe-interrupt.ps1 -- Ctrl+Break during a video encode must EXIT, not hang.
#
# Regression probe for the writer self-deadlock.  The writer's wait loop took
# `std::unique_lock lock(mu)` and then, in the Interrupted() branch and the
# 600 s stall branch, built a SECOND lock_guard on the same non-recursive
# mutex (rd_video.cpp, the branch that sets `failed = true`).  With `lock`
# already holding mu that self-lock wedges the writer while mu stays held:
# every reader blocks on mu, reader.join() never returns, and the process hangs
# until killed -- on every Ctrl-C during video, which is the common case.
#
# Method:
#   * start rdither via System.Diagnostics.Process (shares our console, own
#     process group == pid);
#   * wait --delay-ms so the encode is inside the writer loop (the clip's
#     palette stage takes ~1.1 s and the encode ~1.2 s after it, so 1500 ms
#     lands mid-encode; 3000 ms lands AFTER the run finished -- measured, the
#     first version of this probe failed exactly that way);
#   * GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, child pid).  Ctrl+C cannot be
#     addressed to one group on Windows, Ctrl+Break can, and ConsoleHandler in
#     rd_video.cpp accepts CTRL_BREAK_EVENT the same as Ctrl+C;
#   * require the child to exit with code 130 (ReportInterrupted) inside
#     --timeout-sec.  A hang past the timeout is exactly the old bug.
#
# Exit codes: 0 pass, 1 fail (hang, wrong exit, or event not delivered),
#             2 setup problem (missing exe/clip).

param(
    [int]$DelayMs = 1500,
    [int]$TimeoutSec = 60
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Exe = Join-Path $Root 'build\Release\rdither.exe'
$Clip = Join-Path $Root 'tests\clip1920.mp4'

if (-not (Test-Path $Exe)) {
    Write-Host "probe-interrupt: missing $Exe (build first)"
    exit 2
}
if (-not (Test-Path $Clip)) {
    Write-Host "probe-interrupt: no clip at $Clip"
    exit 2
}

Add-Type -Namespace RdProbe -Name Native -MemberDefinition @'
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
public struct STARTUPINFOW {
    public int cb;
    public string lpReserved;
    public string lpDesktop;
    public string lpTitle;
    public int dwX; public int dwY; public int dwXSize; public int dwYSize;
    public int dwXCountChars; public int dwYCountChars;
    public int dwFillAttribute; public int dwFlags;
    public short wShowWindow; public short cbReserved2;
    public IntPtr lpReserved2; public IntPtr hStdInput;
    public IntPtr hStdOutput; public IntPtr hStdError;
}
[StructLayout(LayoutKind.Sequential)]
public struct PROCESS_INFORMATION {
    public IntPtr hProcess; public IntPtr hThread;
    public int dwProcessId; public int dwThreadId;
}
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool CreateProcess(
    string lpApplicationName, string lpCommandLine,
    IntPtr lpProcessAttributes, IntPtr lpThreadAttributes,
    bool bInheritHandles, uint dwCreationFlags,
    IntPtr lpEnvironment, string lpCurrentDirectory,
    ref STARTUPINFOW lpStartupInfo, out PROCESS_INFORMATION lpProcessInformation);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool CloseHandle(IntPtr hObject);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool GenerateConsoleCtrlEvent(uint dwCtrlEvent, uint dwProcessGroupId);
[DllImport("kernel32.dll")]
public static extern bool WaitForSingleObject(IntPtr hHandle, uint dwMilliseconds);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool GetExitCodeProcess(IntPtr hProcess, out uint lpExitCode);
'@

$CTRL_BREAK_EVENT = 1
$CREATE_NEW_PROCESS_GROUP = 0x00000200
$INFINITE = 0xFFFFFFFF

$out = Join-Path $env:TEMP 'rdither_interrupt_probe.mkv'
Remove-Item -EA SilentlyContinue $out

Write-Host "probe-interrupt: encoding $Clip, break after $DelayMs ms"

# CREATE_NEW_PROCESS_GROUP is the load-bearing flag, and it is why this probe
# uses CreateProcess instead of Start-Process: without a NEW group the child
# joins the POWERSHHELL process group, so GenerateConsoleCtrlEvent aimed at it
# either goes nowhere (child pid is not a group id) or -- measured on the first
# version -- hits the PowerShell host itself, which drops into its debugger and
# the probe never gets to wait on the child.  With the flag, the child's group
# id IS its pid, the break reaches only the child, and Ctrl+C is disabled for
# the new group by Windows (irrelevant here: ConsoleHandler accepts
# CTRL_BREAK_EVENT too).
$si = New-Object RdProbe.Native+STARTUPINFOW
$si.cb = [Runtime.InteropServices.Marshal]::SizeOf($si)
# Quoted paths: the repo lives under "J:\ai code gen\...", and an unquoted
# command line splits the clip at the space (rdither: unexpected argument).
$cmd = "`"$Exe`" --video `"$Clip`" `"$out`" --colors 16 --video-lossless --no-audio --engine cpu"
$pi = New-Object RdProbe.Native+PROCESS_INFORMATION
# lpApplicationName MUST be the exe, not $null: with a null app name the
# marshaller passes an empty string, CreateProcess reads it as a path, and the
# call fails with Win32 ERROR_PATH_NOT_FOUND (3).  Measured both variants.
$ok = [RdProbe.Native]::CreateProcess($Exe, $cmd, [IntPtr]::Zero, [IntPtr]::Zero,
                                       $false, $CREATE_NEW_PROCESS_GROUP, [IntPtr]::Zero,
                                       $Root, [ref]$si, [ref]$pi)
if (-not $ok) {
    $err = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    Write-Host "probe-interrupt: CreateProcess failed (Win32 $err)"
    exit 2
}
$hProc = $pi.hProcess
$childPid = $pi.dwProcessId
[RdProbe.Native]::CloseHandle($pi.hThread) | Out-Null

Start-Sleep -Milliseconds $DelayMs

# Still alive before the break?  A finished encode measures nothing: a too-large
# --delay is a timing miss, not a pass.  (WAIT_OBJECT_0 = 0 = "signaled" = the
# process exited; WAIT_TIMEOUT = 0x102 = still running.)
if ([RdProbe.Native]::WaitForSingleObject($hProc, 0) -eq 0) {
    $code0 = 0
    [RdProbe.Native]::GetExitCodeProcess($hProc, [ref]$code0) | Out-Null
    Write-Host "probe-interrupt: FAIL -- process already exited (code $code0) before the break; lower -DelayMs"
    [RdProbe.Native]::CloseHandle($hProc) | Out-Null
    exit 1
}

# Group id == child pid (created with CREATE_NEW_PROCESS_GROUP above): the break
# reaches only the child, never this PowerShell host.
$ok = [RdProbe.Native]::GenerateConsoleCtrlEvent($CTRL_BREAK_EVENT, [uint32]$childPid)
if (-not $ok) {
    $err = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    Write-Host "probe-interrupt: FAIL -- GenerateConsoleCtrlEvent failed (Win32 $err); console setup cannot deliver the break"
    [RdProbe.Native]::CloseHandle($hProc) | Out-Null
    exit 1
}

$stillWaiting = [RdProbe.Native]::WaitForSingleObject($hProc, [uint32]($TimeoutSec * 1000)) -ne 0
if ($stillWaiting) {
    Write-Host "probe-interrupt: FAIL -- still running $TimeoutSec s after Ctrl+Break (writer deadlock, the bug this probe exists for)"
    # taskkill: the raw handle has no Kill(); /T also sweeps ffmpeg children
    # the hung run left behind.
    & taskkill /PID $childPid /F /T 2>$null | Out-Null
    [RdProbe.Native]::CloseHandle($hProc) | Out-Null
    exit 1
}

$code = 0
if (-not [RdProbe.Native]::GetExitCodeProcess($hProc, [ref]$code)) {
    $code = -1
}
[RdProbe.Native]::CloseHandle($hProc) | Out-Null
if ($code -ne 130) {
    Write-Host "probe-interrupt: FAIL -- exited in time but with code $code (want 130 from ReportInterrupted)"
    exit 1
}

Write-Host "probe-interrupt: ok -- exited 130 within $TimeoutSec s of Ctrl+Break"
exit 0
