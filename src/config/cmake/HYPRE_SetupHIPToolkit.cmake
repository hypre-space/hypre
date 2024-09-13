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
if(DEFINED ENV{ROCM_PATH})
  set(HIP_PATH $ENV{ROCM_PATH})
elseif(DEFINED ENV{HIP_PATH})
  set(HIP_PATH $ENV{HIP_PATH})
else()
  message(FATAL_ERROR "Neither ROCM_PATH nor HIP_PATH environment variable is set. Please set one of them to point to your ROCm installation.")
endif()

# Add HIP_PATH to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH})

# Find HIP package
find_package(hip REQUIRED CONFIG)

# Set HIP-specific variables
set(CMAKE_HIP_ARCHITECTURES "gfx940" CACHE STRING "target HIP architectures")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fPIC" CACHE STRING "HIP compiler flags" FORCE)

# Collection of HIP optional libraries
set(EXPORT_INTERFACE_HIP_LIBS "")

if(HYPRE_SHARED OR WIN32)
  set(HYPRE_HIP_TOOLKIT_STATIC FALSE)
else()
  set(HYPRE_HIP_TOOLKIT_STATIC TRUE)
endif()

# Function to find and add libraries
function(find_and_add_hip_library LIB_NAME)
  if(HYPRE_ENABLE_${LIB_NAME})
    set(HYPRE_USING_${LIB_NAME} ON CACHE BOOL "" FORCE)
    if(HYPRE_HIP_TOOLKIT_STATIC)
      list(APPEND EXPORT_INTERFACE_HIP_LIBS roc::${LIB_NAME}_static)
    else()
      list(APPEND EXPORT_INTERFACE_HIP_LIBS roc::${LIB_NAME})
    endif()
    set(EXPORT_INTERFACE_HIP_LIBS ${EXPORT_INTERFACE_HIP_LIBS} PARENT_SCOPE)
  endif()
endfunction()

# Find and add libraries
find_and_add_hip_library(rocblas)
find_and_add_hip_library(rocsparse)
find_and_add_hip_library(rocrand)
find_and_add_hip_library(rocsolver)

if(HYPRE_ENABLE_GPU_PROFILING)
  set(HYPRE_USING_ROCTRACER ON CACHE BOOL "" FORCE)
  find_library(ROCTRACER_LIBRARY
     NAMES libroctracer64.so
     PATHS ${HIP_PATH}/lib ${HIP_PATH}/lib64
     NO_DEFAULT_PATH)
  if(ROCTRACER_LIBRARY)
    message(STATUS "ROCm tracer library found in ${ROCTRACER_LIBRARY}")
    list(APPEND EXPORT_INTERFACE_HIP_LIBS ${ROCTRACER_LIBRARY})
  else()
    message(WARNING "ROCm tracer library not found. GPU profiling may not work correctly.")
  endif()
endif()

# Make EXPORT_INTERFACE_HIP_LIBS available to parent scope
if(PARENT_SCOPE)
  set(EXPORT_INTERFACE_HIP_LIBS ${EXPORT_INTERFACE_HIP_LIBS} PARENT_SCOPE)
else()
  set(EXPORT_INTERFACE_HIP_LIBS ${EXPORT_INTERFACE_HIP_LIBS})
endif()