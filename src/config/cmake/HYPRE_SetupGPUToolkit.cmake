# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Enable CXX language
enable_language(CXX)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 14)
  set(CMAKE_CXX_STANDARD 14) # Enforce C++14 at least
endif()
set_property(TARGET HYPRE PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
message(STATUS "Enabling support for CXX.")
message(STATUS "Using CXX standard: C++${CMAKE_CXX_STANDARD}")

# Set GPU-related variables
set(HYPRE_USING_GPU ON CACHE BOOL "" FORCE)
set(HYPRE_USING_HOST_MEMORY OFF CACHE BOOL "" FORCE)

if(HYPRE_ENABLE_UNIFIED_MEMORY)
  set(HYPRE_USING_UNIFIED_MEMORY ON CACHE BOOL "" FORCE)
else()
  set(HYPRE_USING_DEVICE_MEMORY ON CACHE BOOL "" FORCE)
endif()

# Check if examples are enabled, but not unified memory
if(HYPRE_BUILD_EXAMPLES AND NOT HYPRE_ENABLE_UNIFIED_MEMORY)
  message(WARNING "Running the examples on GPUs requires Unified Memory! Examples will not be built!")
  set(HYPRE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
endif()

# Add any extra CXX compiler flags
if(NOT HYPRE_WITH_EXTRA_CXXFLAGS STREQUAL "")
  string(REPLACE " " ";" HYPRE_WITH_EXTRA_CXXFLAGS_LIST ${HYPRE_WITH_EXTRA_CXXFLAGS})
  target_compile_options(HYPRE PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${HYPRE_WITH_EXTRA_CXXFLAGS_LIST}>)
endif()

# Include the toolkit setup file for the selected GPU architecture
if(HYPRE_WITH_CUDA)
  include(HYPRE_SetupCUDAToolkit)

elseif(HYPRE_WITH_HIP)
  include(HYPRE_SetupHIPToolkit)

elseif(HYPRE_WITH_SYCL)
  message(STATUS "Enabling SYCL toolkit")
  enable_language(SYCL)
  include(HYPRE_SetupSYCLToolkit)
  set(EXPORT_DEVICE_LIBS ${EXPORT_INTERFACE_SYCL_LIBS})

else()
  message(FATAL_ERROR "Neither CUDA nor HIP nor SYCL is enabled. Please enable one of them.")
endif()
