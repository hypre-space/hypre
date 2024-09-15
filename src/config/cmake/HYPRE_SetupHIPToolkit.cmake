# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# This handles the non-compiler aspect of the HIP toolkit.
# Uses cmake find_package to locate the AMD HIP tools
# for shared libraries. Otherwise for static libraries, assumes
# the libraries are located in ${ROCM_PATH}/lib or ${ROCM_PATH}/lib64.
# Please set environment variable ROCM_PATH or HIP_PATH.

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
message(STATUS "Using ROCm installation: ${HIP_PATH}")

# Add HIP_PATH to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH})

# Check if HIP is available and enable it if found
include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
else()
  message(FATAL_ERROR "HIP language not found. Please check your HIP/ROCm installation.")
endif()

# Find HIP package
find_package(hip REQUIRED CONFIG)

# Collection of ROCm optional libraries
set(ROCM_LIBS "")

# Function to find and add libraries
function(find_and_add_rocm_library LIB_NAME)
  string(TOUPPER ${LIB_NAME} LIB_NAME_UPPER)
  set(HYPRE_ENABLE_VAR "HYPRE_ENABLE_${LIB_NAME_UPPER}")
  if(${HYPRE_ENABLE_VAR})
    set(HYPRE_USING_${LIB_NAME_UPPER} ON CACHE BOOL "" FORCE)
    find_package(${LIB_NAME} REQUIRED)
    if(TARGET roc::${LIB_NAME})
      message(STATUS "roc::${LIB_NAME} target found")
      list(APPEND ROCM_LIBS roc::${LIB_NAME})
    else()
      #message(WARNING "roc::${LIB_NAME} target not found. Attempting manual linking.")
      find_library(${LIB_NAME}_LIBRARY ${LIB_NAME} HINTS ${HIP_PATH}/lib ${HIP_PATH}/lib64)
      if(${LIB_NAME}_LIBRARY)
        message(STATUS "Found ${LIB_NAME} library: ${${LIB_NAME}_LIBRARY}")
        add_library(roc::${LIB_NAME} UNKNOWN IMPORTED)
        set_target_properties(roc::${LIB_NAME} PROPERTIES
          IMPORTED_LOCATION "${${LIB_NAME}_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${HIP_PATH}/include")
        list(APPEND ROCM_LIBS roc::${LIB_NAME})
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

if(HYPRE_ENABLE_GPU_PROFILING)
  set(HYPRE_USING_ROCTRACER ON CACHE BOOL "" FORCE)
  find_library(ROCTRACER_LIBRARY
     NAMES libroctracer64.so
     PATHS ${HIP_PATH}/lib ${HIP_PATH}/lib64
     NO_DEFAULT_PATH)
  if(ROCTRACER_LIBRARY)
    message(STATUS "ROCm tracer library found in ${ROCTRACER_LIBRARY}")
    list(APPEND ROCM_LIBS ${ROCTRACER_LIBRARY})
  else()
    message(WARNING "ROCm tracer library not found. GPU profiling may not work correctly.")
  endif()
endif()

# Add HIP include directory
target_include_directories(HYPRE PUBLIC ${HIP_PATH}/include)

# Link HIP libraries to the target
target_link_libraries(HYPRE PUBLIC ${ROCM_LIBS})
message(STATUS "Linking to ROCm libraries: ${ROCM_LIBS}")

# Set HIP-specific variables
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fPIC" CACHE STRING "HIP compiler flags" FORCE)

# Print HIP info
message(STATUS "HIP Standard: ${CMAKE_HIP_STANDARD}")
message(STATUS "HIP FLAGS: ${CMAKE_HIP_FLAGS}")
