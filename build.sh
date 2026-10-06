#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# ============================================================================
#  build.sh -- build rdither on Linux.
#
#  *** UNVERIFIED. THIS SCRIPT HAS NEVER BEEN RUN. ***
#
#  It was written on Windows by reading build.cmd and CMakeLists.txt. No Linux
#  machine has ever executed it, and this repository has no Linux CI job, so
#  nothing here has been observed to work. Treat it as a first draft to be
#  corrected against a real error message, not as a supported build path.
#
#  THE KNOWN BLOCKER, which is in CMakeLists.txt and not in this script:
#  CMakeLists.txt recognises exactly TWO ImageMagick layouts, `windows` and
#  `conda`, and both expect MSVC import libraries:
#
#      CORE_RL_MagickCore_.lib
#      MagickCore-7.Q16HDRI.dll.lib
#      magick.exe
#      file(GLOB _rd_im_dlls ".../*.dll")
#
#  A Linux ImageMagick ships none of those: the libraries are libMagickCore-7.Q16HDRI.so,
#  the binary has no .exe, and there are no DLLs to copy. So this script will reach
#  CMake and CMake will reject the ImageMagick, whatever this file does. Adding the
#  `linux` flavour to CMakeLists.txt is the actual work; this script is the wrapper
#  around it.
#
#  WHY THIS EXISTS ANYWAY. OpenCL is the primary accelerated path on Linux --
#  there is no CUDA there at all, and Intel/AMD iGPU need it too -- so "no Linux
#  build" is a real gap, not a cosmetic one. A draft that fails loudly and says
#  why is more use than nothing, PROVIDED it is marked as unverified, which is
#  what the banner above is for.
#
#  WHAT TO NEED
#    1. CMake >= 3.18, a C++17 compiler (gcc or clang), and make or ninja.
#    2. ImageMagick 7 with development headers.  Debian/Ubuntu:
#         sudo apt install cmake build-essential libmagickwand-dev
#       (that package name has changed across releases; if it is not found, ask
#        apt for the ImageMagick 7 dev package your release ships)
#    3. ffmpeg, for the video suite.
#    4. Optional, for the OpenCL engine: an OpenCL ICD.  Headers alone are not
#       enough -- on a CPU-only box `pocl-opencl-icd` provides a working
#       OpenCL-on-CPU runtime, which is enough to BUILD and to run the suite.
#
#  USAGE
#    ./build.sh                # configure and build
#    ./build.sh --no-cuda      # CPU + OpenCL only; no CUDA toolkit needed
#    ./build.sh --clean        # remove the build directory first
#
#  ENV
#    IMAGEMAGICK_ROOT   install prefix, if pkg-config cannot find it. Point it
#                       at the directory CONTAINING include/ and lib/, e.g.
#                       /usr or /opt/imagemagick.
#    JOBS               parallel compile jobs. Defaults to nproc.
#    RD_OPENCL_SDK      root of the Khronos OpenCL SDK, if it is not on the
#                       default search path.
# ============================================================================

set -u

RD_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${RD_HERE}/build"

WITH_CUDA=ON
DO_CLEAN=0

for arg in "$@"; do
  case "$arg" in
    --no-cuda) WITH_CUDA=OFF ;;
    --clean)   DO_CLEAN=1 ;;
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

fail() { echo; echo "[FAIL] $*" >&2; echo; exit 1; }
step() { echo; echo "--- $* ---"; }

# --- dependencies ------------------------------------------------------------
step "Checking the toolchain"

MISSING=""
command -v cmake >/dev/null 2>&1 || MISSING="$MISSING cmake"
command -v c++    >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || MISSING="$MISSING a C++ compiler"
[ "$WITH_CUDA" = "ON" ] && { command -v nvcc >/dev/null 2>&1 || MISSING="$MISSING nvcc"; }

if [ -n "$MISSING" ]; then
  echo "[FAIL] Missing:$MISSING"
  echo "  On Debian/Ubuntu:"
  echo "    sudo apt install cmake build-essential"
  echo "  Build without the GPU engines if you only need the CPU and OpenCL ones:"
  echo "    ./build.sh --no-cuda"
  exit 1
fi
echo "  cmake   : $(cmake --version | head -1)"
echo "  compiler: $( (c++ --version 2>/dev/null || g++ --version) | head -1)"

# --- ImageMagick -------------------------------------------------------------
# build.cmd tests two Windows layouts by CONTENT. The Linux equivalent is
# pkg-config, which is what CMake's FindImageMagick-equivalent logic wants and
# what the `linux` flavour added to CMakeLists.txt is meant to consume. Until
# that flavour exists this check is advisory: it tells the user what is wrong
# before CMake does, and it cannot make the build succeed on its own.
step "Locating ImageMagick 7"

IM_OK=0
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists MagickCore-7.Q16HDRI 2>/dev/null; then
  echo "  pkg-config: $(pkg-config --modversion MagickCore-7.Q16HDRI)"
  IM_OK=1
elif command -v MagickCore-config >/dev/null 2>&1; then
  echo "  MagickCore-config: $(MagickCore-config --version 2>/dev/null)"
  IM_OK=1
elif command -v Magick-config >/dev/null 2>&1; then
  echo "  Magick-config: $(Magick-config --version 2>/dev/null)"
  IM_OK=1
elif [ -n "${IMAGEMAGICK_ROOT:-}" ] && [ -f "${IMAGEMAGICK_ROOT}/include/ImageMagick-7/MagickCore/MagickCore.h" ]; then
  echo "  IMAGEMAGICK_ROOT=${IMAGEMAGICK_ROOT}"
  IM_OK=1
fi

if [ "$IM_OK" = "0" ]; then
  echo "  [warn] ImageMagick 7 development files were not found."
  echo "         This script is expected to FAIL at the configure step, because"
  # Single quotes, NOT double.  Inside double quotes bash performs command
  # substitution on backticks, so `linux` and `windows` were EXECUTED as commands:
  #
  #     ./build.sh: line 125: linux: command not found
  #     ./build.sh: line 126: windows: command not found
  #
  # and `conda` -- which does exist on some machines -- ran `conda --version` and
  # printed its entire help text.  A reader looking for the ImageMagick error found
  # a wall of conda usage instead, because the line that should have named the cause
  # had replaced it with noise.  Single quotes suppress substitution entirely.
  echo '         CMakeLists.txt has no `linux` ImageMagick flavour yet -- it knows'
  echo '         only `windows` and `conda`, both of which want MSVC .lib files.'
  echo "         See the banner at the top of this file."
  echo "         On Debian/Ubuntu, try:"
  echo "           sudo apt install libmagickwand-dev"
else
  echo "  [ok] ImageMagick development files found"
fi

# --- configure ---------------------------------------------------------------
if [ "$DO_CLEAN" = "1" ]; then
  step "Removing $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

step "Configuring (CMake)"

CMAKE_ARGS=(-S "$RD_HERE" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)
[ "$WITH_CUDA" = "OFF" ] && CMAKE_ARGS+=(-DRD_WITH_CUDA=OFF)
[ -n "${IMAGEMAGICK_ROOT:-}" ] && CMAKE_ARGS+=(-DRD_IMAGEMAGICK_ROOT="$IMAGEMAGICK_ROOT")
[ -n "${RD_OPENCL_SDK:-}" ] && CMAKE_ARGS+=(-DRD_OPENCL_SDK="$RD_OPENCL_SDK")

printf '  cmake'
for a in "${CMAKE_ARGS[@]}"; do printf ' %s' "$a"; done
printf '\n'

if ! cmake "${CMAKE_ARGS[@]}"; then
  fail "CMake could not configure the build."
fi
echo "  [ok] configured"

# --- compile -----------------------------------------------------------------
step "Compiling"

JOBS="${JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) )}"
echo "  using $JOBS parallel jobs"
cmake --build "$BUILD_DIR" --config Release --parallel "$JOBS" \
  || fail "Compilation failed. The errors above name the file and line."

BIN="$BUILD_DIR/rdither"
[ -x "$BIN" ] || BIN="$BUILD_DIR/Release/rdither"

echo
if [ -x "$BIN" ]; then
  echo "BUILD SUCCEEDED."
  echo
  echo "  binary : $BIN"
  echo "  quick check:"
  echo "    $BIN --list-dithers"
  echo "    $BIN --colors 16 input.png output.png"
  echo
  echo "  Full suite (needs ffmpeg, and a GPU for the CUDA and OpenCL engines):"
  echo "    pwsh -File ./verify.ps1"
  echo
  echo "  REMINDER: this build path has never been verified. Treat the first"
  echo "  failures as expected and report them."
else
  fail "the build reported success but no binary was found (looked for $BUILD_DIR/rdither and $BUILD_DIR/Release/rdither)."
fi