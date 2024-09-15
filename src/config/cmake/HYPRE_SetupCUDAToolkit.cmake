# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Collection of CUDA optional libraries
set(CUDA_LIBS "")

# Check if CUDA is available and enable it if found
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "CUDA language not found. Please check your CUDA installation.")
endif()

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUDA standard to match C++ standard if not already set
if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

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
  message(FATAL_ERROR "CUDA Thrust headers not found! Please check your CUDA installation.")
endif()

# Function to handle CUDA libraries
function(find_and_add_cuda_library LIB_NAME HYPRE_ENABLE_VAR)
  if(${HYPRE_ENABLE_VAR})
    string(TOUPPER ${LIB_NAME} LIB_NAME_UPPER)
    set(HYPRE_USING_${LIB_NAME_UPPER} ON CACHE BOOL "" FORCE)
    if(TARGET CUDAToolkit::${LIB_NAME})
      message(STATUS "CUDAToolkit::${LIB_NAME} target found")
      list(APPEND CUDA_LIBS CUDAToolkit::${LIB_NAME})
    else()
      message(WARNING "CUDAToolkit::${LIB_NAME} target not found. Attempting manual linking.")
      find_library(${LIB_NAME}_LIBRARY ${LIB_NAME} HINTS ${CUDAToolkit_LIBRARY_DIR})
      if(${LIB_NAME}_LIBRARY)
        message(STATUS "Found ${LIB_NAME} library: ${${LIB_NAME}_LIBRARY}")
        add_library(CUDAToolkit::${LIB_NAME} UNKNOWN IMPORTED)
        set_target_properties(CUDAToolkit::${LIB_NAME} PROPERTIES
          IMPORTED_LOCATION "${${LIB_NAME}_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}")
        list(APPEND CUDA_LIBS CUDAToolkit::${LIB_NAME})
      else()
        message(FATAL_ERROR "Could not find ${LIB_NAME} library. Please check your CUDA installation.")
      endif()
    endif()
    set(CUDA_LIBS ${CUDA_LIBS} PARENT_SCOPE)
  endif()
endfunction()

# Handle CUDA libraries
find_and_add_cuda_library(cusparse HYPRE_ENABLE_CUSPARSE)
find_and_add_cuda_library(curand HYPRE_ENABLE_CURAND)
find_and_add_cuda_library(cublas HYPRE_ENABLE_CUBLAS)
find_and_add_cuda_library(cublasLt HYPRE_ENABLE_CUBLAS)
find_and_add_cuda_library(cusolver HYPRE_ENABLE_CUSOLVER)

# Handle GPU Profiling with nvToolsExt
if(HYPRE_ENABLE_GPU_PROFILING)
  find_package(nvToolsExt REQUIRED)
  set(HYPRE_USING_NVTX ON CACHE BOOL "" FORCE)
  list(APPEND CUDA_LIBS CUDA::nvToolsExt)
endif()

# Return the list of CUDA libraries
set(EXPORT_INTERFACE_CUDA_LIBS ${CUDA_LIBS})

# Add CUDA Toolkit include directories to the target  
target_include_directories(HYPRE PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

# Add Thrust include directory
target_include_directories(HYPRE PUBLIC ${THRUST_INCLUDE_DIR})

# Link CUDA libraries to the target
target_link_libraries(HYPRE PUBLIC ${CUDA_LIBS})

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=${CMAKE_CXX_COMPILER} -expt-extended-lambda")

# Print CUDA info
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "NVCC FLAGS: ${CUDA_NVCC_FLAGS}")
