@echo off
REM ===========================================================================
REM build.cmd -- build rdither.
REM
REM If this script fails, the last few lines it printed say which of the things
REM below was missing or wrong.  Look for the [FAIL] marker; everything is
REM reported in the order a missing dependency is most likely to bite.
REM
REM WHAT YOU NEED
REM   1. Visual Studio 2022, with the "Desktop development with C++" workload.
REM      (nvcc on Windows only accepts MSVC as its host compiler, and the
REM      ImageMagick import libraries are MSVC-ABI, so MinGW/g++ will not work.)
REM   2. CUDA Toolkit, for the GPU engines.
REM   3. ImageMagick 7 Q16-HDRI, the MSVC build.
REM
REM If ImageMagick is not where this script expects, set IMAGEMAGICK_ROOT:
REM   set IMAGEMAGICK_ROOT=C:\Program Files\ImageMagick-7.1.2-Q16-HDRI
REM   build.cmd
REM
REM To build without the GPU (CPU engine only, and much faster to compile):
REM   build.cmd --no-cuda
REM ===========================================================================
setlocal enabledelayedexpansion
set "RD_N=0"

call :step "Checking what is installed"

REM --- ImageMagick ---------------------------------------------------------
REM The MSVC build only; a MinGW one ships .a import libraries instead of .lib
REM and cannot be linked by MSVC.
REM
REM Initialised with the "set /p" form rather than a plain `set "X=%Y%"`, because
REM that form expands a missing %Y% to the literal text "Y" -- and it also
REM does not create the variable at all when %Y% is empty, which makes a
REM conditional fall into the wrong branch.  `if defined` is the reliable test.
set "RD_IMAGE_MAGICK=C:\Program Files\ImageMagick-7.1.2-Q16-HDRI"
if defined IMAGEMAGICK_ROOT set "RD_IMAGE_MAGICK=%IMAGEMAGICK_ROOT%"

REM Two layouts are accepted, tested by CONTENT rather than by path shape, because
REM a conda/mamba prefix puts headers under Library\include\ImageMagick-7 and
REM names its import libraries MagickCore-7.Q16HDRI.dll.lib instead of
REM CORE_RL_MagickCore_.lib.  That layout is worth supporting: it is the only
REM ImageMagick on Windows that installs unattended, which is what CI needs.
REM See the ImageMagick section of CMakeLists.txt, which probes the same way.
if exist "%RD_IMAGE_MAGICK%\include\MagickCore\MagickCore.h" (
  set "RD_IM_FLAVOUR=windows"
  set "RD_IM_BIN=%RD_IMAGE_MAGICK%"
) else if exist "%RD_IMAGE_MAGICK%\Library\include\ImageMagick-7\MagickCore\MagickCore.h" (
  set "RD_IM_FLAVOUR=conda"
  set "RD_IM_BIN=%RD_IMAGE_MAGICK%\Library\bin"
) else (
  echo.
  echo   [FAIL] ImageMagick not found.
  echo.
  echo     Looked for, in both accepted layouts:
  echo       %RD_IMAGE_MAGICK%\include\MagickCore\MagickCore.h
  echo       %RD_IMAGE_MAGICK%\Library\include\ImageMagick-7\MagickCore\MagickCore.h
  echo.
  echo     rdither needs ImageMagick 7 Q16-HDRI, the MSVC dll or static build.
  echo     build.  Download it from:
  echo         https://imagemagick.org/script/download.php
  echo     and pick "ImageMagick-7.x.x-Q16-HDRI-x64-dll.exe".
  echo.
  echo     If yours is installed somewhere else, point at it before running:
  echo         set IMAGEMAGICK_ROOT=C:\path\to\ImageMagick-7.x.x-Q16-HDRI
  echo         build.cmd
  echo.
  echo     A conda or mamba prefix also works, and installs unattended:
  echo         micromamba create -p C:\im -c conda-forge imagemagick=7.1.2_31
  echo         set IMAGEMAGICK_ROOT=C:\im
  echo         build.cmd
  echo.
  set "RC=1"
  goto :done
)
echo   [ok] ImageMagick: %RD_IMAGE_MAGICK%  (%RD_IM_FLAVOUR% layout)

REM --- Visual Studio --------------------------------------------------------
REM vswhere ships with VS 2022 and is the reliable way to find the install.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSPATH="
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * ^
      -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
      -property installationPath`) do set "VSPATH=%%i"
)
if "%VSPATH%"=="" (
  REM vswhere can be missing on a stripped install; fall back to the usual paths.
  for /d %%d in ("%ProgramFiles%\Microsoft Visual Studio\2022\*" ^
                 "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\*") do (
    if exist "%%d\VC\Auxiliary\Build\vcvars64.bat" set "VSPATH=%%~d"
  )
)
if "%VSPATH%"=="" (
  echo.
  echo   [FAIL] Visual Studio 2022 with the C++ tools not found.
  echo.
  echo     rdither needs:
  echo       * Visual Studio 2022, any edition
  echo       * the "Desktop development with C++" workload
  echo         -- MSVC v143 build tools and the Windows 10/11 SDK
  echo
  echo     Visual Studio Installer ^(the setup app^) -^>
  echo       Modify -^> "Desktop development with C++" -^> Install.
  echo     Or from a command prompt:
  echo       winget install Microsoft.VisualStudio.2022.Community ^
  echo         --override "--add Microsoft.VisualStudio.Workload.VCTools"
  echo.
  echo     This is checked by looking for vcvars64.bat under the install.
  echo.
  set "RC=1"
  goto :done
)
echo   [ok] Visual Studio: %VSPATH%

REM --- CUDA, unless skipped -------------------------------------------------
set "WITH_CUDA=ON"
:parse
if "%~1"=="" goto :parsed
if /i "%~1"=="--no-cuda" (
  set "WITH_CUDA=OFF"
  shift
  goto :parse
)
if /i "%~1"=="--clean" (
  echo   --clean: removing the build directory
  if exist build rmdir /s /q build
  shift
  goto :parse
)
echo   [warn] unknown option "%~1" ignored
shift
goto :parse
:parsed

REM CUDA_PATH first, then the conventional install location.
REM
REM The comments have to live OUTSIDE the parenthesised block below.  A REM inside one
REM is not a comment to cmd: it is parsed as a command, which aborted the block early
REM and made the whole check silently do nothing.  That is a nasty failure because the
REM script carries on and reports the toolkit as missing with no indication that the
REM test itself never ran.
REM
REM CUDA_PATH is what NVIDIA documents, what cmake's FindCUDAToolkit reads, and what CI
REM sets after unpacking the redist component archives into a scratch directory.  It was
REM not consulted at all before, so the CUDA CI job could not have worked even with the
REM archives merged correctly: the glob finds nothing on a hosted runner, and the
REM script reported "The CUDA Toolkit was not found" while a perfectly good toolkit sat
REM in %RUNNER_TEMP%\cuda.
REM
REM The install-location glob stays as the fallback because that is where a normal
REM machine keeps it, and an explicit CUDA_PATH should win when both exist.
set "CUDA_PATH_FOUND="
if /i not "%WITH_CUDA%"=="OFF" (
  if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\nvcc.exe" set "CUDA_PATH_FOUND=%CUDA_PATH%"
  )
  REM !CUDA_PATH_FOUND! and not %CUDA_PATH_FOUND%.
  REM
  REM cmd expands %VAR% for a WHOLE parenthesised block when it reaches the opening
  REM bracket, before any line inside it has run.  So the test below saw the variable
  REM as empty even though the line above had just set it, the glob always ran, and
  REM the install-directory copy always won.  An explicit CUDA_PATH was silently
  REM ignored.  Delayed expansion (!) re-reads the value at execution time, and this
  REM script already enables it on line 23.
  REM
  REM Verified with a standalone repro: with %FOUND% the "saw EMPTY" branch printed;
  REM with !FOUND! it printed "saw SET".
  if "!CUDA_PATH_FOUND!"=="" (
    for /d %%d in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
      if exist "%%d\bin\nvcc.exe" set "CUDA_PATH_FOUND=%%~d"
    )
  )
)
if /i "%WITH_CUDA%"=="ON" (
  if "!CUDA_PATH_FOUND!"=="" (
    echo.
    echo   [FAIL] The CUDA Toolkit was not found.
    echo.
    echo     Looked for, in order:
    echo       CUDA_PATH\bin\nvcc.exe
    echo       "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe"
    echo.
    echo     CUDA_PATH is currently: "%CUDA_PATH%"
    echo.
    echo     Without it, only the CPU engine can be built.  Two choices:
    echo.
    echo       1. Install CUDA, needed for --engine cuda and --engine blocks:
    echo            https://developer.nvidia.com/cuda-downloads
    echo          Version 12.x or 13.x, and an NVIDIA driver that supports it.
    echo
    echo       2. Build CPU-only right now:
    echo            build.cmd --no-cuda
    echo          (same program, --engine cpu.  The GPU engines are much faster
    echo           on video, so this is mainly useful for checking that a build
    echo           works at all..
    echo.
    set "RC=1"
    goto :done
  )
  echo   [ok] CUDA Toolkit: %CUDA_PATH_FOUND%
) else (
  echo   [ok] CUDA skipped, building the CPU engine only
)

REM --- configure ------------------------------------------------------------
call :step "Configuring the build (CMake)"

REM Work from the script directory and use relative paths.  %~dp0 ends in a
REM backslash and this project may sit under a path with spaces, which mangles
REM quoted -S/-B arguments when they are passed as absolute paths.
pushd "%~dp0"

set "CMAKE_GEN=Visual Studio 17 2022"
set "CMAKE_ARCH=-A x64"
if exist build\CMakeCache.txt (
  for /f "tokens=2 delims==" %%k in ('findstr /b "CMAKE_GENERATOR:" build\CMakeCache.txt') do set "PREV_GEN=%%k"
  if defined PREV_GEN set "CMAKE_GEN=!PREV_GEN!"
)

if "%WITH_CUDA%"=="OFF" (
  echo   cmake -S . -B build -G "!CMAKE_GEN!" !CMAKE_ARCH! -DRD_WITH_CUDA=OFF -DRD_IMAGEMAGICK_ROOT="%RD_IMAGE_MAGICK%"
  cmake -S . -B build -G "!CMAKE_GEN!" !CMAKE_ARCH! -DRD_WITH_CUDA=OFF -DRD_IMAGEMAGICK_ROOT="%RD_IMAGE_MAGICK%"
) else (
  echo   cmake -S . -B build -G "!CMAKE_GEN!" !CMAKE_ARCH! -DRD_IMAGEMAGICK_ROOT="%RD_IMAGE_MAGICK%"
  cmake -S . -B build -G "!CMAKE_GEN!" !CMAKE_ARCH! -DRD_IMAGEMAGICK_ROOT="%RD_IMAGE_MAGICK%"
)
if errorlevel 1 (
  echo.
  echo   [FAIL] CMake could not configure the build.
  echo.
  echo     The messages just above say why.  The usual causes are an
  echo     ImageMagick path that is not the MSVC build, or a Visual Studio
  echo     install without the C++ workload.
  echo     For a clean slate: build.cmd --clean
  echo.
  popd
  set "RC=1"
  goto :done
)
echo   [ok] configured

REM --- compile --------------------------------------------------------------
call :step "Compiling ^(this takes a minute or two^)"

REM No extra verbosity flag here.  CMake already passes MSBuild /v:m, and adding
REM "-- -v" on top made MSBuild reject the command line outright (MSB1016: "specify
REM a verbosity level"), because it had been given two conflicting -v switches and
REM the bare -v is not a valid MSBuild one.  At /v:m a failed compile prints its
REM error and the file and line, which is what a person reading this needs; the
REM full command line is available by hand:
REM     cmake --build build --config Release --verbose
cmake --build build --config Release --parallel
if errorlevel 1 (
  echo.
  echo   [FAIL] Compilation failed.  The errors are in the output above.
  echo.
  echo     They are usually one of:
  echo       * "cannot open include file 'cuda_runtime.h'"
  echo         -^> CUDA is not on PATH for this shell.  Open a "x64 Native Tools
  echo            Command Prompt for VS 2022" and run build.cmd again.
  echo       * "cannot open input file 'CORE_RL_MagickCore_.lib'"
  echo         -^> ImageMagick is not the MSVC build.  See the note above.
  echo       * "identifier X is undefined" in a .cu file
  echo         -^> a CUDA kernel was changed in a way that broke the host/device
  echo            split; the full text above names the file and line.
  echo     For a clean slate: build.cmd --clean
  echo.
  popd
  set "RC=1"
  goto :done
)
echo   [ok] compiled

REM --- run it ---------------------------------------------------------------
call :step "Smoke test ^(build a palette and check it against ImageMagick^)"

REM Under the conda layout the DLLs are NOT beside magick.exe's own root, and
REM rdither.exe will not start unless Library\bin is on PATH -- there are 224
REM dependent DLLs there, not one.  Prepending it is what makes the smoke test
REM below a real test rather than a failure to load.
if "%RD_IM_FLAVOUR%"=="conda" set "PATH=%RD_IM_BIN%;%PATH%"
set "MAGICK_HOME=%RD_IMAGE_MAGICK%"
if not exist "%TEMP%\rdither_smoke.png" (
  "%RD_IM_BIN%\magick.exe" -size 64x64 gradient:black-white "%TEMP%\rdither_smoke.png"
)
if not exist "%TEMP%\rdither_smoke.png" (
  echo   [FAIL] could not create a test image with magick.exe
  echo     The ImageMagick install at "%RD_IMAGE_MAGICK%" looks incomplete.
  popd
  set "RC=1"
  goto :done
)

build\Release\rdither.exe --colors 16 --verify "%TEMP%\rdither_smoke.png" "%TEMP%\rdither_smoke_out.png"
if errorlevel 1 (
  echo.
  echo   [FAIL] The program built, but its output is not bit-exact against
  echo     ImageMagick's Riemersma dither.
  echo.
  echo     This is a real problem, not a flaky test: the whole point of the
  echo     program is that it reproduces `magick -dither Riemersma` exactly.
  echo     It usually means a compiler setting changed -- most often FMA
  echo     contraction, which changes the error accumulation.  Check that
  echo     CMakeLists.txt still has -fmad=false for CUDA and /fp:precise for
  echo     the host compiler.
  echo.
  popd
  set "RC=1"
  goto :done
)
echo   [ok] bit-exact against ImageMagick

popd

call :step "Done"
echo.
echo   rdither.exe  :  %~dp0build\Release\rdither.exe
echo.
echo   It needs MAGICK_HOME at run time, because the ImageMagick DLLs and the
echo   format modules load from the install directory:
echo.
echo       set MAGICK_HOME=%RD_IMAGE_MAGICK%
if "%RD_IM_FLAVOUR%"=="conda" (
  echo.
  echo   Conda layout: the DLLs are in Library\bin rather than beside
echo   magick.exe, so PATH needs that directory too:
  echo.
  echo       set PATH=%RD_IM_BIN%;%%PATH%%
)
echo.
echo   Quick check:
echo       rdither.exe --list-dithers
echo       rdither.exe --colors 16 input.png output.png
echo.
echo   Full documentation: README.md
echo.

set "RC=0"
goto :done

REM --- helpers --------------------------------------------------------------
REM Prints a step header with an incrementing number.  RD_N is a single global
REM seeded once, at the top of this script, and only ever touched here.
REM
REM It was a per-call-site `set "STEP=0"` first, and a call site that forgot it
REM printed "[~1]": set /a on an empty variable computes "0-1", which is not a
REM number, and cmd substitutes ~ for a token it cannot parse.  One seed at the
REM top, one increment here, no way to forget.
:step
set /a RD_N+=1
echo.
echo [%RD_N%] %~1
echo.
goto :eof

:done
echo.
if "%RC%"=="0" (
  echo BUILD SUCCEEDED.
) else (
  echo BUILD FAILED ^(exit %RC%^).
)
echo.
endlocal & exit /b %RC%
