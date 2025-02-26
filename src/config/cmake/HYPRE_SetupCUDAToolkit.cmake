# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

message(STATUS "Enabling CUDA toolkit")

# Check for CUDA_PATH, CUDA_HOME or CUDA_DIR
if(DEFINED CUDAToolkit_ROOT)
  set(CUDA_DIR ${CUDAToolkit_ROOT})
elseif(DEFINED ENV{CUDAToolkit_ROOT})
  set(CUDA_DIR $ENV{CUDAToolkit_ROOT})
elseif(DEFINED CUDA_DIR)
  set(CUDA_DIR ${CUDA_DIR})
elseif(DEFINED ENV{CUDA_DIR})
  set(CUDA_DIR $ENV{CUDA_DIR})
elseif(DEFINED CUDA_PATH)
  set(CUDA_DIR ${CUDA_PATH})
elseif(DEFINED ENV{CUDA_PATH})
  set(CUDA_DIR $ENV{CUDA_PATH})
elseif(DEFINED CUDA_HOME)
  set(CUDA_DIR ${CUDA_HOME})
elseif(DEFINED ENV{CUDA_HOME})
  set(CUDA_DIR $ENV{CUDA_HOME})
elseif(EXISTS "/opt/cuda")
  set(CUDA_DIR "/opt/cuda")
elseif(EXISTS "/usr/bin/nvcc")
  set(CUDA_DIR "/usr")
else()
  message(FATAL_ERROR "CUDA_PATH or CUDA_HOME not set. Please set one of them to point to your CUDA installation.")
endif()
message(STATUS "Using CUDA installation: ${CUDA_DIR}")

# Specify the path to the custom nvcc compiler
if(WIN32)
  set(CMAKE_CUDA_COMPILER "${CUDA_DIR}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler")
else()
  set(CMAKE_CUDA_COMPILER "${CUDA_DIR}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()

# Specify the CUDA Toolkit root directory
set(CUDAToolkit_ROOT "${CUDA_DIR}" CACHE PATH "Path to the CUDA toolkit")

# Optionally, prioritize the custom CUDA path in CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${CUDA_DIR}")

# Set CUDA standard to match C++ standard if not already set
if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Visual Studio does not support CMAKE_CUDA_HOST_COMPILER
if (NOT MSVC)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "CXX compiler used by CUDA" FORCE)
endif()

# Check if CUDA is available and enable it if found
include(CheckLanguage)
check_language(CUDA)
if(DEFINED CMAKE_CUDA_COMPILER)
   enable_language(CUDA)
else()
  message(FATAL_ERROR "CUDA language not found. Please check your CUDA installation.")
endif()

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Add a dummy cuda target if it doesn't exist (avoid error when building with BLT dependencies)
if(NOT TARGET cuda)
  add_library(cuda INTERFACE)
endif()

# Detection CUDA architecture if not given by the user
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
  message(STATUS "Detecting CUDA GPU architectures using nvidia-smi...")

  # Platform-specific NVIDIA smi command
  if (WIN32)
    # Try multiple possible locations on Windows
    find_program(NVIDIA_SMI_CMD
      NAMES nvidia-smi.exe
      PATHS
      "${CUDA_DIR}/bin"
      "$ENV{ProgramFiles}/NVIDIA Corporation/NVSMI"
      "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v*/bin"
      "$ENV{ProgramW6432}/NVIDIA Corporation/NVSMI"
      NO_DEFAULT_PATH
    )

    if(NOT NVIDIA_SMI_CMD)
      find_program(NVIDIA_SMI_CMD nvidia-smi.exe)
    endif()
    message(STATUS "Found nvidia-smi: ${NVIDIA_SMI_CMD}")
  else()
    set(NVIDIA_SMI_CMD "nvidia-smi")
  endif()

  if(NOT NVIDIA_SMI_CMD)
    message(WARNING "nvidia-smi not found. Using default CUDA architecture 70.")
    set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "Default CUDA architectures" FORCE)
  else()
    # Execute nvidia-smi to get GPU compute capabilities
    execute_process(
      COMMAND ${NVIDIA_SMI_CMD} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE NVIDIA_SMI_OUTPUT
      RESULT_VARIABLE NVIDIA_SMI_RESULT
      ERROR_VARIABLE NVIDIA_SMI_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT NVIDIA_SMI_RESULT EQUAL 0)
      message(WARNING "${NVIDIA_SMI_CMD} failed to execute: ${NVIDIA_SMI_ERROR}")
      set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "Default CUDA architectures" FORCE)
      message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES to default '70'")
    else()
      # Clean the output (remove extra newlines and spaces)
      string(STRIP "${NVIDIA_SMI_OUTPUT}" CUDA_ARCHS)     # Remove trailing/leading whitespaces
      string(REPLACE "." "" CUDA_ARCHS "${CUDA_ARCHS}")   # Replace '.' with nothing to format '7.0' as '70'
      string(REPLACE "\n" ";" CUDA_ARCHS "${CUDA_ARCHS}") # Replace newline with semicolon for list format

      # Remove any duplicates CUDA archictectures
      list(REMOVE_DUPLICATES CUDA_ARCHS)

      if(CUDA_ARCHS)
        string(REPLACE ";" "," CUDA_ARCHS_STR "${CUDA_ARCHS}")
        set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHS_STR}" CACHE STRING "Detected CUDA architectures" FORCE)
        message(STATUS "Detected CUDA GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
      else()
        message(WARNING "No GPUs detected. Setting CMAKE_CUDA_ARCHITECTURES to default '70'")
        set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "Default CUDA architectures" FORCE)
      endif()
    endif()
  endif()
else()
  # Remove duplicates from the pre-set CMAKE_CUDA_ARCHITECTURES
  string(REPLACE "," ";" CUDA_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")
  list(REMOVE_DUPLICATES CUDA_ARCH_LIST)
  string(REPLACE ";" "," CUDA_ARCH_STR "${CUDA_ARCH_LIST}")
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_STR}" CACHE STRING "Detected CUDA architectures" FORCE)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES is already set to: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Set the CUDA architectures to the HYPRE target
set_property(TARGET HYPRE PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# Show CUDA Toolkit location
if(CUDAToolkit_FOUND)
  if (CUDAToolkit_ROOT)
    message(STATUS "CUDA Toolkit found at: ${CUDAToolkit_ROOT}")
  endif()
  message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA Toolkit include directory: ${CUDAToolkit_INCLUDE_DIRS}")
  message(STATUS "CUDA Toolkit library directory: ${CUDAToolkit_LIBRARY_DIR}")
else()
  message(FATAL_ERROR "CUDA Toolkit not found")
endif()

# Check for Thrust headers
find_path(THRUST_INCLUDE_DIR thrust/version.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}/cuda-thrust
  PATH_SUFFIXES thrust
  NO_DEFAULT_PATH
)

if(THRUST_INCLUDE_DIR)
  message(STATUS "CUDA Thrust headers found in: ${THRUST_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "CUDA Thrust headers not found! Please specify -DTHRUST_INCLUDE_DIR.")
endif()

# Collection of CUDA optional libraries
set(CUDA_LIBS "")

# Function to handle CUDA libraries
function(find_and_add_cuda_library LIB_NAME HYPRE_ENABLE_VAR)
  string(TOUPPER ${LIB_NAME} LIB_NAME_UPPER)
  if(${HYPRE_ENABLE_VAR})
    set(HYPRE_USING_${LIB_NAME_UPPER} ON CACHE INTERNAL "")

    # Use CUDAToolkit to find the component
    find_package(CUDAToolkit REQUIRED COMPONENTS ${LIB_NAME})

    if(TARGET CUDA::${LIB_NAME})
      message(STATUS "Found ${LIB_NAME_UPPER} library")
      list(APPEND CUDA_LIBS CUDA::${LIB_NAME})
    else()
      message(STATUS "CUDA::${LIB_NAME} target not found. Attempting manual linking.")
      find_library(${LIB_NAME}_LIBRARY ${LIB_NAME} HINTS ${CUDAToolkit_LIBRARY_DIR})
      if(${LIB_NAME}_LIBRARY)
        message(STATUS "Found ${LIB_NAME_UPPER} library: ${${LIB_NAME}_LIBRARY}")
        add_library(CUDA::${LIB_NAME} UNKNOWN IMPORTED)
        set_target_properties(CUDA::${LIB_NAME} PROPERTIES
          IMPORTED_LOCATION "${${LIB_NAME}_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}")
        list(APPEND CUDA_LIBS CUDA::${LIB_NAME})
      else()
        message(FATAL_ERROR "Could not find ${LIB_NAME_UPPER} library. Please check your CUDA installation.")
      endif()
    endif()

    set(CUDA_LIBS ${CUDA_LIBS} PARENT_SCOPE)
  endif()
endfunction()

# Handle CUDA libraries
list(APPEND CUDA_LIBS CUDA::cudart) # Add cudart first since other CUDA libraries may depend on it
find_and_add_cuda_library(cusparse HYPRE_ENABLE_CUSPARSE)
find_and_add_cuda_library(curand HYPRE_ENABLE_CURAND)
find_and_add_cuda_library(cublas HYPRE_ENABLE_CUBLAS)
find_and_add_cuda_library(cublasLt HYPRE_ENABLE_CUBLAS)
find_and_add_cuda_library(cusolver HYPRE_ENABLE_CUSOLVER)

# Handle GPU Profiling with nvToolsExt
if(HYPRE_ENABLE_GPU_PROFILING)
  find_library(NVTX_LIBRARY nvToolsExt HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib)
  if(NVTX_LIBRARY)
    message(STATUS "Found NVTX library")
    set(HYPRE_USING_NVTX ON CACHE BOOL "" FORCE)
    list(APPEND CUDA_LIBS ${NVTX_LIBRARY})
  else()
    message(FATAL_ERROR "NVTX library not found! Make sure CUDA is installed correctly.")
  endif()
endif()

# Add CUDA Toolkit include directories to the target
target_include_directories(HYPRE PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

# Add Thrust include directory
target_include_directories(HYPRE PUBLIC ${THRUST_INCLUDE_DIR})

# Link CUDA libraries to the target
target_link_libraries(HYPRE PUBLIC ${CUDA_LIBS})
message(STATUS "Linking to CUDA libraries: ${CUDA_LIBS}")

# Set additional CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Ensure LTO-specific flags are included
if (HYPRE_ENABLE_LTO AND CUDAToolkit_VERSION VERSION_LESS 11.2)
  # See https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto
  message(WARNING "Device LTO not available on CUDAToolkit_VERSION (${CUDAToolkit_VERSION}) < 11.2. Turning it off...")

elseif (HYPRE_ENABLE_LTO AND CMAKE_VERSION VERSION_LESS 3.25)
  # See https://gitlab.kitware.com/cmake/cmake/-/commit/96bc59b1ca01be231347404d178445263687dd22
  message(WARNING "Device LTO not available with CUDA on CMAKE_VERSION (${CMAKE_VERSION}) < 3.25. Turning it off...")

elseif (HYPRE_ENABLE_LTO)
  message(STATUS "Enabling Device LTO")

  # Enable LTO for the target
  set_target_properties(${PROJECT_NAME} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

# Print CUDA info
message(STATUS "CUDA C++ standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
