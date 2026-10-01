cmake_minimum_required(VERSION 3.24)
project(rdither LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Toolchain note
# ---------------------------------------------------------------------------
# This project targets the ImageMagick 7.1.2-31 *MSVC* build that ships for
# Windows.  nvcc on Windows accepts only MSVC as its host compiler ("nvcc fatal
# : Host compiler targets unsupported OS" for MinGW-w64), and the Magick++
# import libraries are MSVC-ABI, so g++/MinGW cannot link them either -- it
# mangles Magick:: symbols with the Itanium ABI and every reference comes back
# undefined.  MSVC is therefore mandatory for the full binary.  The
# ImageMagick-free core (rd_types.h + rd_riemersma.h) is portable C++20 and can
# be compiled with g++ -std=c++20; see README.md.
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RD_WITH_CUDA "Build the CUDA engine" ON)
# The OpenCL engine.  ON by default and found automatically, because its whole
# purpose is reach: a machine with a working GPU and no CUDA SDK should still get
# a GPU engine.  It links the ICD loader import library -- which is part of
# Windows, so the runtime dependency is not the SDK -- and needs only its headers
# at build time.  If no SDK is found the build continues without it and the
# engine reports "not available" at runtime, which is a feature that is off
# rather than a broken build.
option(RD_WITH_OPENCL "Build the OpenCL engine" ON)
set(RD_OPENCL_SDK "" CACHE PATH
    "Root of the Khronos OpenCL SDK (the folder containing include/CL and lib)")
# Per-pixel atomics that report how many octree nodes a search visits.  Off by
# default and not runtime-switchable: with them compiled in they cost a measured
# 2.5x on the dither stage.  See RD_CC_COUNT in include/rd_cuda_common.cuh.
option(RD_CC_STATS "Count octree search nodes (diagnostic; slows the walk ~2.5x)" OFF)
set(RD_IMAGEMAGICK_ROOT "" CACHE PATH
    "Root of the ImageMagick 7.1.2-31 Q16-HDRI installation")

# --- ImageMagick -----------------------------------------------------------
# TWO LAYOUTS ARE SUPPORTED, and which one is in use is decided by looking at the
# tree rather than by asking the user to say so.
#
#   official Windows installer        conda/mamba prefix
#     include/MagickCore/*.h            Library/include/ImageMagick-7/MagickCore/*.h
#     lib/CORE_RL_MagickCore_.lib        Library/lib/MagickCore-7.Q16HDRI.dll.lib
#     magick.exe                        Library/bin/magick.exe
#
# The second exists because it is the only ImageMagick on Windows that installs
# unattended.  The official installer opens a directory prompt under /S on a clean
# machine and waits for a human; six separate attempts to work around that are
# recorded in .github/workflows/ci.yml.  `micromamba create imagemagick=7.1.2_31`
# takes fifteen seconds, needs no GUI, and ships the same 7.1.2-31 Q16-HDRI
# version -- verified here to compile, link, and run a real Riemersma quantize
# against MagickCore, not merely to be present on disk.
if(NOT RD_IMAGEMAGICK_ROOT)
  if(DEFINED ENV{IMAGEMAGICK_ROOT})
    set(RD_IMAGEMAGICK_ROOT "$ENV{IMAGEMAGICK_ROOT}")
  else()
    file(GLOB _rd_im_dirs "C:/Program Files/ImageMagick-*")
    foreach(_dir ${_rd_im_dirs})
      if(EXISTS "${_dir}/include/MagickCore/MagickCore.h")
        set(RD_IMAGEMAGICK_ROOT "${_dir}")
      endif()
    endforeach()
  endif()
endif()

if(NOT RD_IMAGEMAGICK_ROOT)
  message(FATAL_ERROR
    "ImageMagick not found.  Either install it, or set IMAGEMAGICK_ROOT to the "
    "install root, or pass -DRD_IMAGEMAGICK_ROOT=<path>.  A conda/mamba prefix "
    "works: micromamba create -p C:/im -c conda-forge imagemagick=7.1.2_31")
endif()

# Layout probe, by content.  RD_IM_FLAVOUR is "windows" or "conda".
if(EXISTS "${RD_IMAGEMAGICK_ROOT}/include/MagickCore/MagickCore.h")
  set(RD_IM_FLAVOUR "windows")
  set(RD_IM_INCLUDE "${RD_IMAGEMAGICK_ROOT}/include")
  set(RD_IM_LIB     "${RD_IMAGEMAGICK_ROOT}/lib")
  set(RD_IM_BIN      "${RD_IMAGEMAGICK_ROOT}")
  set(RD_IM_MAGICK_EXE "${RD_IMAGEMAGICK_ROOT}/magick.exe")
  # The official build names its import libraries per quantum; CORE_RL is Q16 HDRI.
  set(RD_IM_CORE_LIB "CORE_RL_MagickCore_.lib")
  set(RD_IM_WAND_LIB "CORE_RL_MagickWand_.lib")
  # Its DLLs sit one level up from lib/, unversioned.
  file(GLOB _rd_im_dlls "${RD_IMAGEMAGICK_ROOT}/*.dll")
elseif(EXISTS "${RD_IMAGEMAGICK_ROOT}/Library/include/ImageMagick-7/MagickCore/MagickCore.h")
  set(RD_IM_FLAVOUR "conda")
  set(RD_IM_INCLUDE "${RD_IMAGEMAGICK_ROOT}/Library/include/ImageMagick-7")
  set(RD_IM_LIB     "${RD_IMAGEMAGICK_ROOT}/Library/lib")
  set(RD_IM_BIN      "${RD_IMAGEMAGICK_ROOT}/Library/bin")
  set(RD_IM_MAGICK_EXE "${RD_IMAGEMAGICK_ROOT}/Library/bin/magick.exe")
  set(RD_IM_CORE_LIB "MagickCore-7.Q16HDRI.dll.lib")
  set(RD_IM_WAND_LIB "MagickWand-7.Q16HDRI.dll.lib")
  # The conda DLL carries a build suffix, e.g. MagickCore-7.Q16HDRI-10.dll.
  file(GLOB _rd_im_dlls "${RD_IM_BIN}/MagickCore*.dll")
else()
  message(FATAL_ERROR
    "ImageMagick headers not found under ${RD_IMAGEMAGICK_ROOT}.  Expected either\n"
    "  ${RD_IMAGEMAGICK_ROOT}/include/MagickCore/MagickCore.h          (Windows installer)\n"
    "  ${RD_IMAGEMAGICK_ROOT}/Library/include/ImageMagick-7/MagickCore/MagickCore.h  (conda)")
endif()

# A conda build is driven by pkg-config, so magick-config.h does not carry the
# quantum settings itself and REFUSES to compile without them -- it is a hard
# #error, not a warning.  These five come from
# Library/lib/pkgconfig/ImageMagick-7.Q16HDRI.pc, read rather than guessed:
#   Cflags: -DMAGICKCORE_HDRI_ENABLE=1 -DMAGICKCORE_QUANTUM_DEPTH=16 -D_DLL -D_MT
#           -DMAGICKCORE_CHANNEL_MASK_DEPTH=32
if(RD_IM_FLAVOUR STREQUAL "conda")
  set(RD_IM_DEFINES
    MAGICKCORE_HDRI_ENABLE=1
    MAGICKCORE_QUANTUM_DEPTH=16
    MAGICKCORE_CHANNEL_MASK_DEPTH=32
    _DLL
    _MT)
endif()

message(STATUS "ImageMagick root:  ${RD_IMAGEMAGICK_ROOT}  (${RD_IM_FLAVOUR} layout)")
message(STATUS "ImageMagick exe:   ${RD_IM_MAGICK_EXE}")

# --- CUDA ------------------------------------------------------------------
if(RD_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "No CUDA compiler found; building the CPU engine only.")
    set(RD_WITH_CUDA OFF)
  else()
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # sm_75 covers the GTX 1650 SUPER present on this machine; 90 is forward
      # looking for Hopper.  Override with -DCMAKE_CUDA_ARCHITECTURES=...
      set(CMAKE_CUDA_ARCHITECTURES 75 90)
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # FMA contraction and fast-math would break bit-exactness on the device, so
    # they are disabled through the CUDA flag variables.  These are plain nvcc
    # tokens; passing -fmad through target_compile_options makes CMake merge it
    # into an existing -Xcompiler list with commas, which nvcc rejects.
    string(APPEND CMAKE_CUDA_FLAGS " -fmad=false")
    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler=/fp:precise")
    string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -Xcompiler=/fp:precise")
  endif()
endif()

# --- OpenCL ----------------------------------------------------------------
if(RD_WITH_OPENCL)
  if(NOT RD_OPENCL_SDK)
    # Look for a Khronos SDK in the places someone actually puts one, and REMEMBER
    # WHICH PLACES THOSE WERE so the not-found message can list them.  No registry
    # probing on purpose: an SDK is a build-time input, and a machine that has never
    # downloaded one has nothing to find.
    #
    # This list used to be a single glob of the source directory, while the comment
    # above it promised "beside the source, and in the usual C:/OpenCL* and per-user
    # download places".  None of those three was searched.  So the instruction in the
    # comment was the one that did not work, and the place that did work was the one
    # the comment never mentioned -- found by unpacking the official release zip beside
    # a clean clone, exactly as the comment said to, and being told "not found".
    #
    # Every entry is a specific, narrow glob.  This never walks a user profile: the
    # Downloads folder is checked directly and nothing above it.
    set(_rd_ocl_roots
      "${CMAKE_CURRENT_SOURCE_DIR}"          # inside the repository
      "${CMAKE_CURRENT_SOURCE_DIR}/.."       # beside the repository
      "${CMAKE_CURRENT_SOURCE_DIR}/../..")    # repository nested one level down
    if(WIN32)
      list(APPEND _rd_ocl_roots "C:/")       # the usual C:/OpenCL-SDK*
      if(DEFINED ENV{USERPROFILE})
        list(APPEND _rd_ocl_roots "$ENV{USERPROFILE}/Downloads")
      endif()
    endif()
    set(_rd_ocl_searched "")
    foreach(_root ${_rd_ocl_roots})
      list(APPEND _rd_ocl_searched "${_root}")
      if(RD_OPENCL_SDK)
        continue()                           # first hit wins
      endif()
      file(GLOB _rd_ocl_dirs "${_root}/OpenCL-SDK*")
      foreach(_d ${_rd_ocl_dirs})
        # Headers only.  A directory with headers but no import library is a real and
        # common state (a partial unzip), and it gets its own specific warning below
        # rather than being silently skipped here.
        if(IS_DIRECTORY "${_d}/include/CL" AND EXISTS "${_d}/include/CL/cl.h")
          set(RD_OPENCL_SDK "${_d}")
        endif()
      endforeach()
    endforeach()
  endif()
  if(RD_OPENCL_SDK AND EXISTS "${RD_OPENCL_SDK}/include/CL/cl.h")
    message(STATUS "OpenCL SDK: ${RD_OPENCL_SDK}")
    # The import library, not the DLL.  OpenCL.dll is the ICD loader, which is an
    # OS component: it is present whether or not any vendor driver is installed,
    # and it is what loads the vendor library.  So the SDK supplies headers and an
    # import lib, and no runtime redistributable is needed -- which is why
    # "install the SDK" is not a step for the end user.
    set(_rd_ocl_lib "${RD_OPENCL_SDK}/lib/OpenCL.lib")
    if(EXISTS "${_rd_ocl_lib}")
      set(RD_OPENCL_ENABLED ON)
    else()
      message(WARNING
        "OpenCL headers found at ${RD_OPENCL_SDK} but ${_rd_ocl_lib} is missing; "
        "building without the OpenCL engine.")
    endif()
  else()
    # A "not found" that does not say where it looked is the same failure as a silent
    # one: the reader cannot act on it.  So list every directory searched, name what a
    # usable folder must contain, and give the download.  This is the message a
    # contributor sees first, so it has to be enough to fix the problem without
    # reading any documentation.
    set(_rd_ocl_list "")
    foreach(_p ${_rd_ocl_searched})
      set(_rd_ocl_list "${_rd_ocl_list}      ${_p}\n")
    endforeach()
    message(STATUS
      "OpenCL SDK not found; building without the OpenCL engine.\n"
      "  Looked for a folder named  OpenCL-SDK*  containing  include/CL/cl.h  in:\n"
      "${_rd_ocl_list}"
      "  Get it from https://github.com/KhronosGroup/OpenCL-SDK/releases\n"
      "  (the OpenCL-SDK-*-Win-x64.zip asset), unpack it into any of the places\n"
      "  above, or pass -DRD_OPENCL_SDK=<path> explicitly.\n"
      "  The engine is a separate accelerator, so the build is still complete and\n"
      "  usable without it -- this is a missing optional feature, not a failure.")
  endif()
endif()

# --- sources ---------------------------------------------------------------
set(RD_SOURCES
  src/rd_source.cpp
  src/rd_im.cpp
  src/rd_octree.cpp
  src/rd_riemersma_cpu.cpp
  src/rd_video.cpp
  src/rd_progress.cpp
  src/rd_opencl.cpp
  src/rd_checkpoint.cpp
  src/rd_plugin.cpp
  src/rd_dithers.cc
  src/rd_dithers_ordered.cc
  src/rd_cli.cpp)

# The example dither.  Registered like any other algorithm, and it is here in the
# default build rather than behind an option on purpose: a plugin seam that nobody
# can see working is a plugin seam that does not get tested.  Delete this line and
# examples\bayer_dither.cc if you do not want it -- nothing else depends on it.
list(APPEND RD_SOURCES examples/bayer_dither.cc)

# Always compiled, and its entire body sits inside #ifndef RD_WITH_CUDA, so it
# contributes nothing to a CUDA build.  It is here because the seven CUDA entry points
# that rd_video.cpp and rd_cli.cpp call are defined in .cu files this build omits, and
# a declared-but-undefined symbol is a link error.  Same reason the #else half of
# src/rd_opencl.cpp exists.
list(APPEND RD_SOURCES src/rd_cuda_stub.cpp)

if(RD_WITH_CUDA)
  list(APPEND RD_SOURCES src/rd_riemersma_cuda.cu src/rd_approx_cuda.cu
                         src/rd_blocks_cuda.cu)
endif()

add_executable(rdither ${RD_SOURCES})

target_include_directories(rdither PRIVATE include "${RD_IM_INCLUDE}")
target_link_directories(rdither PRIVATE "${RD_IM_LIB}")

# The MSVC import libraries are named per-quantum.  Two naming schemes, picked by
# the layout probe above: the official build says CORE_RL_MagickCore_.lib, conda
# says MagickCore-7.Q16HDRI.dll.lib.  Hardcoding either one is what made the conda
# toolchain unusable until the probe existed.
target_link_libraries(rdither PRIVATE "${RD_IM_CORE_LIB}" "${RD_IM_WAND_LIB}")

if(RD_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(rdither PRIVATE CUDA::cudart)
  target_compile_definitions(rdither PRIVATE RD_WITH_CUDA=1)
  if(RD_CC_STATS)
    target_compile_definitions(rdither PRIVATE RD_CC_STATS=1)
  endif()
endif()

if(RD_OPENCL_ENABLED)
  target_include_directories(rdither PRIVATE "${RD_OPENCL_SDK}/include")
  target_link_libraries(rdither PRIVATE "${_rd_ocl_lib}")
  target_compile_definitions(rdither PRIVATE RD_WITH_OPENCL=1)
  message(STATUS "OpenCL engine: enabled")
endif()

target_compile_definitions(rdither PRIVATE
  $<$<CONFIG:Release>:NDEBUG>
  _CRT_SECURE_NO_WARNINGS
  NOMINMAX
  RD_DEFAULT_MAGICK_EXE="${RD_IM_MAGICK_EXE}")

# The conda layout's magick-config.h #errors out unless it is told the quantum
# settings, which its pkgconfig file carries.  Applied here rather than baked into
# the source so the Windows layout keeps compiling with none of them.
if(RD_IM_DEFINES)
  target_compile_definitions(rdither PRIVATE ${RD_IM_DEFINES})
endif()

if(MSVC)
  # /fp:precise is the default and does not contract a*b+c into FMA, which the
  # bit-exact error accumulation depends on.  Never switch to /fp:fast here.
  # These options are CXX-only: handing raw /W4 to nvcc produces an unusable
  # command line.
  target_compile_options(rdither PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/fp:precise>)
endif()

# The import libraries live in lib/, the matching DLLs one level up.  Only the
# core DLL is required; coder modules are loaded on demand from MODULES_PATH.
#
# Copying is a convenience for the Windows layout, where the DLL sits beside the
# import library and there is nothing else to arrange.  Under the conda layout
# there are 224 dependent DLLs in the same directory and the core one is
# version-suffixed (MagickCore-7.Q16HDRI-10.dll), so copying a single file would
# produce an exe that dies at startup on a missing dependency.  That case is
# handled by putting the prefix's Library/bin on PATH instead -- see the message
# below, which names the directory to add rather than leaving it to be discovered.
if(_rd_im_dlls)
  add_custom_command(TARGET rdither POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_rd_im_dlls}
            "$<TARGET_FILE_DIR:rdither>"
    VERBATIM)
endif()
if(RD_IM_FLAVOUR STREQUAL "conda")
  message(STATUS
    "ImageMagick DLL directory (add to PATH before running rdither):\n    ${RD_IM_BIN}")
endif()
