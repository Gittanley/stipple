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

if(NOT RD_IMAGEMAGICK_ROOT OR NOT EXISTS "${RD_IMAGEMAGICK_ROOT}/include/MagickCore/MagickCore.h")
  message(FATAL_ERROR
    "ImageMagick not found.  Pass -DRD_IMAGEMAGICK_ROOT=\"C:/Program Files/ImageMagick-7.1.2-Q16-HDRI\"")
endif()
message(STATUS "ImageMagick root: ${RD_IMAGEMAGICK_ROOT}")

set(RD_IM_INCLUDE "${RD_IMAGEMAGICK_ROOT}/include")
set(RD_IM_LIB     "${RD_IMAGEMAGICK_ROOT}/lib")

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
    # Look for a Khronos SDK unpacked beside the source, and in the usual
    # "C:/OpenCL*" and per-user download places.  No registry probing on purpose:
    # an SDK is a build-time input, and a machine that has never downloaded one
    # has nothing to find.
    file(GLOB _rd_ocl_local "${CMAKE_CURRENT_SOURCE_DIR}/OpenCL-SDK*")
    foreach(_d ${_rd_ocl_local})
      if(IS_DIRECTORY "${_d}/include/CL")
        set(RD_OPENCL_SDK "${_d}")
      endif()
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
    message(STATUS
      "OpenCL SDK not found; building without the OpenCL engine.  Pass "
      "-DRD_OPENCL_SDK=<path> to enable it.  The engine is a separate accelerator, "
      "so the build is still complete and usable without it.")
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

if(RD_WITH_CUDA)
  list(APPEND RD_SOURCES src/rd_riemersma_cuda.cu src/rd_approx_cuda.cu
                         src/rd_blocks_cuda.cu)
endif()

add_executable(rdither ${RD_SOURCES})

target_include_directories(rdither PRIVATE include "${RD_IM_INCLUDE}")
target_link_directories(rdither PRIVATE "${RD_IM_LIB}")

# The MSVC import libraries are named per-quantum, e.g. CORE_RL_MagickCore_.
target_link_libraries(rdither PRIVATE
  CORE_RL_MagickCore_.lib
  CORE_RL_MagickWand_.lib)

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
  RD_DEFAULT_MAGICK_EXE="${RD_IMAGEMAGICK_ROOT}/magick.exe")

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
add_custom_command(TARGET rdither POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${RD_IMAGEMAGICK_ROOT}/CORE_RL_MagickCore_.dll"
          "$<TARGET_FILE_DIR:rdither>"
  VERBATIM)
