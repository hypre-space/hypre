# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Enable CXX language
enable_language(CXX)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(HYPRE_ENABLE_SYCL)
  # We enforce the use of Intel's oneAPI DPC++/C++ Compiler
  if(NOT CMAKE_CXX_COMPILER MATCHES "dpcpp|icpx")
    message(FATAL_ERROR "SYCL requires DPC++ or Intel C++ compiler")
  endif()

  # Enforce C++17 at least for SYCL
  if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
  endif()
else()
  # Enforce C++14 at least for CUDA and HIP
  if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 14)
    set(CMAKE_CXX_STANDARD 14)
  endif()
endif()

# Set C++ standard for HYPRE
set_property(TARGET HYPRE PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})

# Add C++ standard library to interface
if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND WIN32))
  # MSVC and Intel on Windows link the C++ standard library automatically
  message(STATUS "${CMAKE_CXX_COMPILER_ID} on Windows: C++ standard library linked automatically")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang" AND APPLE)
  # Apple Clang specifically uses c++
  target_link_libraries(HYPRE INTERFACE "-lc++")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "XL|XLClang")
  # IBM XL C++ needs `-libmc++`
  target_link_libraries(HYPRE INTERFACE "-libmc++ -lstdc++")
else()
  # Most other compilers use stdc++
  if(NOT (MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND WIN32)))
    target_link_libraries(HYPRE INTERFACE "-lstdc++")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|Intel|PGI|NVHPC|XL|XLClang")
      message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}. Attempting to link -lstdc++")
    endif()
  endif()
endif()
message(STATUS "C++ standard library configuration completed for ${CMAKE_CXX_COMPILER_ID}")

# Print C++ info
message(STATUS "Enabling support for CXX.")
message(STATUS "Using CXX standard: C++${CMAKE_CXX_STANDARD}")

# Set GPU-related variables
set(HYPRE_USING_GPU ON CACHE INTERNAL "")
set(HYPRE_USING_HOST_MEMORY OFF CACHE INTERNAL "")
set(HYPRE_ENABLE_HOST_MEMORY OFF CACHE BOOL "Use host memory" FORCE)

if(HYPRE_ENABLE_UNIFIED_MEMORY)
  set(HYPRE_USING_UNIFIED_MEMORY ON CACHE INTERNAL "")
else()
  set(HYPRE_USING_DEVICE_MEMORY ON CACHE INTERNAL "")
endif()

# Check if examples are enabled, but not unified memory
if(HYPRE_BUILD_EXAMPLES AND NOT HYPRE_ENABLE_UNIFIED_MEMORY)
  message(WARNING "Running the examples on GPUs requires Unified Memory!\nExamples will not be built!\n")
  set(HYPRE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
endif()

# Add any extra CXX compiler flags
if(NOT HYPRE_WITH_EXTRA_CXXFLAGS STREQUAL "")
  string(REPLACE " " ";" HYPRE_WITH_EXTRA_CXXFLAGS_LIST ${HYPRE_WITH_EXTRA_CXXFLAGS})
  target_compile_options(HYPRE PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${HYPRE_WITH_EXTRA_CXXFLAGS_LIST}>)
endif()

# Include the toolkit setup file for the selected GPU architecture
if(HYPRE_ENABLE_CUDA)
  include(HYPRE_SetupCUDAToolkit)

elseif(HYPRE_ENABLE_HIP)
  include(HYPRE_SetupHIPToolkit)

elseif(HYPRE_ENABLE_SYCL)
  include(HYPRE_SetupSYCLToolkit)
  set(EXPORT_DEVICE_LIBS ${EXPORT_INTERFACE_SYCL_LIBS})

else()
  message(FATAL_ERROR "Neither CUDA nor HIP nor SYCL is enabled. Please enable one of them.")
endif()
