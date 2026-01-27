# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Check for ROCM_PATH or HIP_PATH
message(STATUS "Enabling HIP toolkit")
if(DEFINED ROCM_PATH)
  set(HIP_PATH ${ROCM_PATH})
elseif(DEFINED ENV{ROCM_PATH})
  set(HIP_PATH $ENV{ROCM_PATH})
elseif(DEFINED ENV{HIP_PATH})
  set(HIP_PATH $ENV{HIP_PATH})
elseif(EXISTS "/opt/rocm")
  set(HIP_PATH "/opt/rocm")
else()
  message(FATAL_ERROR "ROCM_PATH or HIP_PATH not set. Please set one of them to point to your ROCm installation.")
endif()
set(CMAKE_HIP_COMPILER_ROCM_ROOT ${HIP_PATH})
message(STATUS "Using ROCm installation: ${HIP_PATH}")

# Add HIP_PATH to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH};${HIP_PATH}/lib/cmake;${HIP_PATH}/llvm/bin")

# Set HIP standard to match C++ standard if not already set
if(NOT DEFINED CMAKE_HIP_STANDARD)
  set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "C++ standard for HIP" FORCE)
endif()
set(CMAKE_HIP_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard for HIP" FORCE)

# Check if HIP is available and enable it if found
set(CMAKE_HIP_COMPILER "${HIP_PATH}/llvm/bin/clang++" CACHE FILEPATH "Path to HIP compiler")
include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
else()
  message(FATAL_ERROR "HIP language not found. Please check your HIP/ROCm installation.")
endif()

# Find HIP package
set(HIP_PLATFORM "amd")
find_package(hip REQUIRED CONFIG PATHS ${ROCM_PATH} ${ROCM_PATH}/lib/cmake/hip)

# Minimum supported HIP version for HYPRE
set(REQUIRED_HIP_VERSION "5.2.0")

if(NOT DEFINED hip_VERSION)
  message(WARNING
    "Cannot detect HIP version from the 'hip' package. Skipping the minimum version check. "
    "Proceed at your own risk!!!")
else()
  if(hip_VERSION VERSION_LESS REQUIRED_HIP_VERSION)
    message(FATAL_ERROR
      "HYPRE requires HIP >= ${REQUIRED_HIP_VERSION}, but found ${hip_VERSION}.")
  endif()
endif()

# Function to detect GPU architectures using rocm-smi
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  message(STATUS "Detecting GPU architectures using rocm-smi...")

  # Execute rocm-smi to get GPU architecture info
  execute_process(
    COMMAND rocm-smi --showUniqueId --format=json
    OUTPUT_VARIABLE ROCM_SMI_OUTPUT
    RESULT_VARIABLE ROCM_SMI_RESULT
    ERROR_VARIABLE ROCM_SMI_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(NOT ROCM_SMI_RESULT EQUAL 0)
    message(WARNING "rocm-smi failed to execute: ${ROCM_SMI_ERROR}")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "Default HIP architectures" FORCE)
    message(STATUS "Setting CMAKE_HIP_ARCHITECTURES to default 'gfx90a'")
    return()
  endif()

  # Parse JSON output to extract GPU architectures
  include(Jsoncpp)
  jsoncpp_parse(json_output "${ROCM_SMI_OUTPUT}")

  # Extract GPU architecture codes
  set(GPU_ARCHS "")
  foreach(gpu ${json_output@/gpu/@})
    get_property(gpu_arch PROPERTY GPU ARCHITECTURE "${gpu}")
    if(NOT gpu_arch STREQUAL "")
      list(APPEND GPU_ARCHS ${gpu_arch})
    endif()
  endforeach()

  list(REMOVE_DUPLICATES GPU_ARCHS)

  if(GPU_ARCHS)
    string(REPLACE ";" "," GPU_ARCHS_STR "${GPU_ARCHS}")
    set(CMAKE_HIP_ARCHITECTURES "${GPU_ARCHS_STR}" CACHE STRING "Detected HIP architectures" FORCE)
    message(STATUS "Detected GPU architectures: ${CMAKE_HIP_ARCHITECTURES}")
  else()
    message(WARNING "No GPUs detected. Setting CMAKE_HIP_ARCHITECTURES to default 'gfx90a'")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "Default HIP architectures" FORCE)
  endif()
else()
  # Remove duplicates from the pre-set CMAKE_HIP_ARCHITECTURES
  string(REPLACE "," ";" HIP_ARCH_LIST "${CMAKE_HIP_ARCHITECTURES}")
  list(REMOVE_DUPLICATES HIP_ARCH_LIST)
  string(REPLACE ";" "," HIP_ARCH_STR "${HIP_ARCH_LIST}")
  set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCH_STR}" CACHE STRING "Detected HIP architectures" FORCE)
  message(STATUS "CMAKE_HIP_ARCHITECTURES is explicitly set to: ${CMAKE_HIP_ARCHITECTURES}")
endif()

# Validate mixed-architecture sets: do not mix gfx9 (CDNA) with gfx10+/gfx11 (RDNA)
# If multiple architectures are specified, fail fast on unsupported mixes
string(REPLACE "," ";" _HYPRE_HIP_ARCH_LIST_CHECK "${CMAKE_HIP_ARCHITECTURES}")
list(LENGTH _HYPRE_HIP_ARCH_LIST_CHECK _HYPRE_HIP_ARCH_COUNT)
if(_HYPRE_HIP_ARCH_COUNT GREATER 1)
  set(_HYPRE_HAS_GFX9 FALSE)
  set(_HYPRE_HAS_GFX10PLUS FALSE)
  foreach(_HYPRE_ARCH_ITEM IN LISTS _HYPRE_HIP_ARCH_LIST_CHECK)
    # Extract numeric base from entries like "gfx90a", "gfx1100"
    string(REGEX MATCH "gfx?([0-9a-fA-F]+)" _HYPRE_DUMMY "${_HYPRE_ARCH_ITEM}")
    set(_HYPRE_GFX_ID_STR "${CMAKE_MATCH_1}")
    string(REGEX REPLACE "[^0-9]" "" _HYPRE_GFX_BASE_ID_STR "${_HYPRE_GFX_ID_STR}")
    if(_HYPRE_GFX_BASE_ID_STR MATCHES "^[0-9]+$")
      string(REGEX REPLACE "^0+" "" _HYPRE_GFX_ID_NO_ZEROS "${_HYPRE_GFX_BASE_ID_STR}")
      if(_HYPRE_GFX_ID_NO_ZEROS STREQUAL "")
        set(_HYPRE_GFX_ID_NO_ZEROS "0")
      endif()
      math(EXPR _HYPRE_GFX_ID_INT "${_HYPRE_GFX_ID_NO_ZEROS}")
      if(_HYPRE_GFX_ID_INT LESS 1000)
        set(_HYPRE_HAS_GFX9 TRUE)
      else()
        set(_HYPRE_HAS_GFX10PLUS TRUE)
      endif()
    endif()
  endforeach()
  if(_HYPRE_HAS_GFX9 AND _HYPRE_HAS_GFX10PLUS)
    message(FATAL_ERROR
      "HYPRE does not support building a single binary for mixed HIP architectures: "
      "found both gfx9 (CDNA) and gfx10+ (RDNA) in CMAKE_HIP_ARCHITECTURES='${CMAKE_HIP_ARCHITECTURES}'.\n"
      "Please specify a single architecture via -DCMAKE_HIP_ARCHITECTURES=<gfx-id> (e.g., gfx90a or gfx1100) "
      "and build separate binaries for each architecture family.")
  endif()
endif()
set_property(TARGET ${PROJECT_NAME} PROPERTY HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
set(GPU_BUILD_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE INTERNAL "GPU targets to compile for")
set(GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE INTERNAL "AMD GPU targets to compile for")

# Check if user specified either WARP_SIZE or WAVEFRONT_SIZE
if(DEFINED HYPRE_WAVEFRONT_SIZE)
  set(HYPRE_WARP_SIZE ${HYPRE_WAVEFRONT_SIZE} CACHE STRING "GPU warp size")
  message(STATUS "Using user-specified wavefront size: ${HYPRE_WAVEFRONT_SIZE}")

elseif(NOT DEFINED HYPRE_WARP_SIZE)
  # Auto-detect warp size based on gpu arch if not specified
  set(FOUND_CDNA_CARD FALSE)
  set(DETECTED_ARCHITECTURES "") # To collect and report all detected architectures

  # CMAKE_HIP_ARCHITECTURES typically contains a semicolon-separated list (e.g., "gfx90a;gfx942")
  if(DEFINED CMAKE_HIP_ARCHITECTURES AND NOT "${CMAKE_HIP_ARCHITECTURES}" STREQUAL "")
    foreach(ARCH_ITEM IN LISTS CMAKE_HIP_ARCHITECTURES)
      # Only process if we haven't already found an old GFX card
      if(NOT FOUND_CDNA_CARD)
        # Extract the numeric part from the GFX architecture string (e.g., "gfx90a" -> "90a")
        # Note: CMAKE_HIP_ARCHITECTURES usually contains only the ID, not "gfx" prefix,
        # but regex is robust if it does.
        string(REGEX MATCH "gfx?([0-9a-fA-F]+)" _dummy "${ARCH_ITEM}")
        set(GFX_ID_STR "${CMAKE_MATCH_1}") # e.g., "90a", "942"

        # Extract only the leading numeric part for comparison
        # (e.g., "90a" -> "90", "942" -> "942")
        string(REGEX REPLACE "[^0-9]" "" GFX_BASE_ID_STR "${GFX_ID_STR}")

        set(CURRENT_GFX_ID_INT 0)
        if(GFX_BASE_ID_STR MATCHES "^[0-9]+$")
          # Remove leading zeros to ensure correct integer comparison (e.g., "0900" -> "900")
          string(REGEX REPLACE "^0+" "" GFX_ID_STR_NO_LEADING_ZERO "${GFX_BASE_ID_STR}")
          if(NOT GFX_ID_STR_NO_LEADING_ZERO EQUAL "")
            set(CURRENT_GFX_ID_INT "${GFX_ID_STR_NO_LEADING_ZERO}")
          endif()
       endif()

        list(APPEND DETECTED_ARCHITECTURES "gfx${GFX_ID_STR}") # Add full GFX string to list for reporting
        message(STATUS "Processing CMAKE_HIP_ARCHITECTURES entry: ${ARCH_ITEM} -> Numeric base: ${CURRENT_GFX_ID_INT}")

        # If any detected GFX card has a numeric ID less than 1000
        if(CURRENT_GFX_ID_INT LESS 1000)
          set(FOUND_CDNA_CARD TRUE)
        endif()
      endif()
    endforeach()

    # Set the final HYPRE_WARP_SIZE based on the flag
    if(FOUND_CDNA_CARD)
      set(HYPRE_WARP_SIZE 64 CACHE STRING "GPU wavefront size (detected at least one GFX < 1000)")
      message(STATUS "HYPRE_WARP_SIZE set to 64 for architectures: ${DETECTED_ARCHITECTURES}")
    else()
      # Only reached here if all detected GFX cards are 1000 or greater
      set(HYPRE_WARP_SIZE 32 CACHE STRING "GPU wavefront size (all detected GFX >= 1000)")
      message(STATUS "HYPRE_WARP_SIZE set to 32 for architectures: ${DETECTED_ARCHITECTURES}")
    endif()

  else()
    message(FATAL_ERROR "CMAKE_HIP_ARCHITECTURES not found or empty!")
  endif()

else()
  message(STATUS "Using user-specified wavefront size: ${HYPRE_WARP_SIZE}")
endif()

# Set WAVEFRONT_SIZE to match WARP_SIZE for consistency
set(HYPRE_WAVEFRONT_SIZE ${HYPRE_WARP_SIZE} CACHE STRING "GPU wavefront size (alias for WARP_SIZE)")
mark_as_advanced(HYPRE_WAVEFRONT_SIZE)

# Collection of ROCm optional libraries
set(ROCM_LIBS "")

# Function to find and add libraries
function(find_and_add_rocm_library LIB_NAME)
  string(TOUPPER ${LIB_NAME} LIB_NAME_UPPER)
  set(HYPRE_ENABLE_VAR "HYPRE_ENABLE_${LIB_NAME_UPPER}")
  if(${HYPRE_ENABLE_VAR})
    set(HYPRE_USING_${LIB_NAME_UPPER} ON CACHE INTERNAL "")
    find_package(${LIB_NAME} REQUIRED)
    if(TARGET roc::${LIB_NAME})
      message(STATUS "Found target: roc::${LIB_NAME}")
      # Append the library variable that hypre expects (except for rocThrust which is header-only).
      # rocThrust is header-only and should be linked as a target.
      if(NOT ${LIB_NAME_UPPER} STREQUAL ROCTHRUST)
        list(APPEND ROCM_LIBS ${${LIB_NAME}_LIBRARY})
      else()
        list(APPEND ROCM_LIBS roc::${LIB_NAME})
      endif()
    else()
      #message(WARNING "roc::${LIB_NAME} target not found. Attempting manual linking.")
      find_library(${LIB_NAME}_LIBRARY ${LIB_NAME} HINTS ${HIP_PATH}/lib ${HIP_PATH}/lib64)
      if(${LIB_NAME}_LIBRARY)
        message(STATUS "Found ${LIB_NAME} library: ${${LIB_NAME}_LIBRARY}")
        if(NOT ${LIB_NAME_UPPER} STREQUAL ROCTHRUST)
          add_library(roc::${LIB_NAME} UNKNOWN IMPORTED)
          set_target_properties(roc::${LIB_NAME} PROPERTIES
            IMPORTED_LOCATION "${${LIB_NAME}_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${HIP_PATH}/include")
          list(APPEND ROCM_LIBS roc::${LIB_NAME})
        else()
          # rocThrust is header-only; if the package didn't provide a target, fall back
          # to the HIP include directory already added below.
          message(WARNING "roc::${LIB_NAME} target not found; relying on ${HIP_PATH}/include for rocThrust headers")
        endif()
      else()
        message(FATAL_ERROR "Could not find ${LIB_NAME} library. Please check your ROCm installation.")
      endif()
    endif()
    set(ROCM_LIBS ${ROCM_LIBS} PARENT_SCOPE)
  endif()
endfunction()

# Find and add libraries
find_and_add_rocm_library(rocblas)
find_and_add_rocm_library(rocsparse)
find_and_add_rocm_library(rocrand)
find_and_add_rocm_library(rocsolver)
find_and_add_rocm_library(rocthrust)

if(HYPRE_ENABLE_GPU_PROFILING)
  set(HYPRE_USING_ROCTX ON CACHE BOOL "" FORCE)
  find_library(ROCTRACER_LIBRARY
     NAMES libroctracer64.so
     PATHS ${HIP_PATH}/lib ${HIP_PATH}/lib64
     NO_DEFAULT_PATH)
  if(ROCTRACER_LIBRARY)
    message(STATUS "ROCTracer library found in ${ROCTRACER_LIBRARY}")
    list(APPEND ROCM_LIBS ${ROCTRACER_LIBRARY})
  else()
    message(WARNING "ROCTracer library not found. GPU profiling may not work correctly.")
  endif()

  find_library(ROCTX_LIBRARY
     NAMES libroctx64.so
     PATHS ${HIP_PATH}/lib ${HIP_PATH}/lib64
     NO_DEFAULT_PATH)
  if(ROCTX_LIBRARY)
    message(STATUS "ROC-TX library found in ${ROCTX_LIBRARY}")
    list(APPEND ROCM_LIBS ${ROCTX_LIBRARY})
  else()
    message(WARNING "ROC-TX library not found. GPU profiling may not work correctly.")
  endif()
endif()

# Add HIP include directory
target_include_directories(${PROJECT_NAME} PUBLIC ${HIP_PATH}/include)

# Link HIP libraries to the target
target_link_libraries(${PROJECT_NAME} PUBLIC ${ROCM_LIBS})
message(STATUS "Linking to ROCm libraries: ${ROCM_LIBS}")

# Turn on Position Independent Code (PIC) by default
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# Signal to downstream targets that they need PIC
set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE TRUE)

# Ensure LTO-specific flags are included
if (HYPRE_ENABLE_LTO AND NOT MSVC)
  if (CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(STATUS "Enabling Device LTO")
  else ()
    message(FATAL_ERROR "HIP Device LTO available only with Clang")
  endif ()

  # HIP compilation options
  target_compile_options(${PROJECT_NAME}
    PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc -foffload-lto>
      $<$<COMPILE_LANGUAGE:CXX>:-foffload-lto>
    INTERFACE
      $<$<COMPILE_LANGUAGE:HIP>:-foffload-lto>
      $<$<COMPILE_LANGUAGE:CXX>:-foffload-lto>
  )

  # Link options need to be more specific
  target_link_options(${PROJECT_NAME}
    PRIVATE
      -fgpu-rdc
      -foffload-lto
      --hip-link
      -Wl,--no-as-needed  # Ensure all symbols are kept
    INTERFACE
      -foffload-lto
  )
endif ()

# Print HIP info
if (DEFINED hip_VERSION)
  message(STATUS "HIP version: ${hip_VERSION}")
endif()
message(STATUS "HIP C++ standard: ${CMAKE_HIP_STANDARD}")
message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
if (DEFINED CMAKE_HIP_FLAGS)
  message(STATUS "HIP common flags: ${CMAKE_HIP_FLAGS}")
endif()
