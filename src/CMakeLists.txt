# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

cmake_minimum_required(VERSION 3.21)
message(STATUS "CMake Executable: ${CMAKE_COMMAND}")
message(STATUS "CMake Version: ${CMAKE_VERSION}")

# Include hypre's CMake utilities
include("${CMAKE_CURRENT_SOURCE_DIR}/config/cmake/HYPRE_CMakeUtilities.cmake")

# Set cmake policies
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 NEW) # Ensure CUDA toolkit is found
endif()
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW) # CUDA_ARCHITECTURES is allowed to be set in cache
endif()

# The version number.
set(HYPRE_VERSION 2.31.0)
set(HYPRE_NUMBER  23100)
set(HYPRE_DATE    2024/02/14)
set(HYPRE_TIME    00:00:00)
set(HYPRE_BUGS    https://github.com/hypre-space/hypre/issues)
set(HYPRE_SRCDIR  "${PROJECT_SOURCE_DIR}")

# Display the hypre version
setup_git_version_info("${CMAKE_CURRENT_SOURCE_DIR}/../.git")
if (GIT_VERSION_FOUND)
  message(STATUS "Hypre Version: ${HYPRE_DEVELOP_STRING} (${HYPRE_BRANCH_NAME})")
else()
  message(STATUS "NOTE: Could not find .git directory")
  message(STATUS "Hypre Version: ${HYPRE_VERSION}")
endif()

# Set the project name
set(PROJECT_NAME HYPRE)
project(${PROJECT_NAME}
  VERSION ${HYPRE_VERSION}
  LANGUAGES C)

# Create the HYPRE library object
add_library(${PROJECT_NAME})

# We use C99 by default, but users are free to specify any newer standard version
set(CMAKE_C_STANDARD 99 CACHE STRING "Standard to use for the C compiler")

# Raise an error if the source and binary directories are the same
if (${HYPRE_SOURCE_DIR} STREQUAL ${HYPRE_BINARY_DIR})
  message(FATAL_ERROR "In-place build not allowed! Please use a separate build directory. See the Users Manual or INSTALL file for details.")
endif ()

# Ensure RPATH/RUNPATH is set properly during the build
set(CMAKE_SKIP_BUILD_RPATH FALSE)                       # Includes rpath in the binaries being built
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)               # Use different paths for build and install
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)             # Use link-time paths
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")  # Where binaries can look for shared libraries

# Set cmake module path
set(CMAKE_MODULE_PATH "${HYPRE_SOURCE_DIR}/config/cmake" "${CMAKE_MODULE_PATH}")

# Set default installation directory, but provide a means for users to change
set(HYPRE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/hypre" CACHE PATH
    "Installation directory for HYPRE")
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${HYPRE_INSTALL_PREFIX}" CACHE INTERNAL "" FORCE)
endif ()

# Set default compile optimization flag
set(HYPRE_BUILD_TYPE "Release" CACHE STRING
    "Optimization flags: set to Debug, Release, RelWithDebInfo, or MinSizeRel")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "${HYPRE_BUILD_TYPE}" CACHE INTERNAL "" FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}") 

# Configuration options
option(HYPRE_ENABLE_SHARED               "Build a shared library" ${BUILD_SHARED_LIBS})
option(HYPRE_ENABLE_BIGINT               "Use long long int for HYPRE_Int" OFF)
option(HYPRE_ENABLE_MIXEDINT             "Use long long int for HYPRE_BigInt, int for HYPRE_INT" OFF)
option(HYPRE_ENABLE_SINGLE               "Use float for HYPRE_Real" OFF)
option(HYPRE_ENABLE_LONG_DOUBLE          "Use long double for HYPRE_Real" OFF)
option(HYPRE_ENABLE_COMPLEX              "Use complex values" OFF)
option(HYPRE_ENABLE_HYPRE_BLAS           "Use internal BLAS library" ON)
option(HYPRE_ENABLE_HYPRE_LAPACK         "Use internal LAPACK library" ON)
option(HYPRE_ENABLE_PERSISTENT_COMM      "Use persistent communication" OFF)
option(HYPRE_ENABLE_FEI                  "Use FEI" OFF) # TODO: Add this cmake feature
option(HYPRE_WITH_MPI                    "Compile with MPI" ON)
option(HYPRE_WITH_GPU_AWARE_MPI          "Compile with device aware GPU support" OFF)
option(HYPRE_WITH_OPENMP                 "Use OpenMP" OFF)
option(HYPRE_ENABLE_HOPSCOTCH            "Use hopscotch hashing with OpenMP" OFF)
option(HYPRE_WITH_SUPERLU                "Use TPL SuperLU" OFF)
option(HYPRE_WITH_DSUPERLU               "Use TPL SuperLU_Dist" OFF)
option(HYPRE_WITH_MAGMA                  "Use TPL MAGMA" OFF)
option(HYPRE_WITH_CALIPER                "Use Caliper" OFF)
option(HYPRE_PRINT_ERRORS                "Print HYPRE errors" OFF)
option(HYPRE_TIMING                      "Use HYPRE timing routines" OFF)
option(HYPRE_BUILD_EXAMPLES              "Build examples" OFF)
option(HYPRE_BUILD_TESTS                 "Build tests" OFF)
option(HYPRE_USING_HOST_MEMORY           "Use host memory" ON)
# GPU options
option(HYPRE_WITH_CUDA                   "Use CUDA. Require cuda-8.0 or higher" OFF)
option(HYPRE_WITH_HIP                    "Use HIP" OFF)
option(HYPRE_WITH_SYCL                   "Use SYCL" OFF)
option(HYPRE_ENABLE_UNIFIED_MEMORY       "Use unified memory for allocating the memory" OFF)
option(HYPRE_ENABLE_DEVICE_MALLOC_ASYNC  "Use device async malloc" OFF)
# CUDA options
option(HYPRE_ENABLE_CUDA_STREAMS         "Use CUDA streams" ON)
option(HYPRE_ENABLE_CUSPARSE             "Use cuSPARSE" ON)
option(HYPRE_ENABLE_CUSOLVER             "Use cuSOLVER" OFF)
option(HYPRE_ENABLE_DEVICE_POOL          "Use device memory pool" OFF)
option(HYPRE_ENABLE_CUBLAS               "Use cuBLAS" OFF)
option(HYPRE_ENABLE_CURAND               "Use cuRAND" ON)
option(HYPRE_ENABLE_GPU_PROFILING        "Use NVTX on CUDA" OFF)
# HIP options
option(HYPRE_ENABLE_ROCBLAS              "Use rocBLAS" ON)
option(HYPRE_ENABLE_ROCSPARSE            "Use rocSPARSE" ON)
option(HYPRE_ENABLE_ROCRAND              "Use rocRAND" ON)
option(HYPRE_ENABLE_ROCSOLVER            "Use rocSOLVER" ON)
# oneAPI options
option(HYPRE_ENABLE_ONEMKLSPARSE         "Use oneMKL sparse" ON)
option(HYPRE_ENABLE_ONEMKLBLAS           "Use oneMKL blas" ON)
option(HYPRE_ENABLE_ONEMKLRAND           "Use oneMKL rand" ON)
# Umpire resource management options
option(HYPRE_WITH_UMPIRE                 "Use Umpire Allocator for device and unified memory" OFF)
option(HYPRE_WITH_UMPIRE_HOST            "Use Umpire Allocator for host memory" OFF)
option(HYPRE_WITH_UMPIRE_DEVICE          "Use Umpire Allocator for device memory" OFF)
option(HYPRE_WITH_UMPIRE_UM              "Use Umpire Allocator for unified memory" OFF)
option(HYPRE_WITH_UMPIRE_PINNED          "Use Umpire Allocator for pinned memory" OFF)
# TPL options
option(TPL_UMPIRE_LIBRARIES              "List of absolute paths to Umpire link libraries [].")
option(TPL_UMPIRE_INCLUDE_DIRS           "List of absolute paths to Umpire include directories [].")
option(TPL_SUPERLU_LIBRARIES             "List of absolute paths to SuperLU link libraries [].")
option(TPL_SUPERLU_INCLUDE_DIRS          "List of absolute paths to SuperLU include directories [].")
option(TPL_DSUPERLU_LIBRARIES            "List of absolute paths to SuperLU_Dist link libraries [].")
option(TPL_DSUPERLU_INCLUDE_DIRS         "List of absolute paths to SuperLU_Dist include directories [].")
option(TPL_MAGMA_LIBRARIES               "List of absolute paths to MAGMA link libraries [].")
option(TPL_MAGMA_INCLUDE_DIRS            "List of absolute paths to MAGMA include directories [].")
option(TPL_BLAS_LIBRARIES                "Optional list of absolute paths to BLAS libraries, otherwise use FindBLAS to locate [].")
option(TPL_LAPACK_LIBRARIES              "Optional list of absolute paths to LAPACK libraries, otherwise use FindLAPACK to locate [].")
option(TPL_FEI_INCLUDE_DIRS              "List of absolute paths to FEI include directories [].")
# Extra flags
set(HYPRE_WITH_EXTRA_CFLAGS              "" CACHE STRING "Define extra C compile flags")
set(HYPRE_WITH_EXTRA_CXXFLAGS            "" CACHE STRING "Define extra CXX compile flags")
set(HYPRE_SYCL_TARGET                    "" CACHE STRING "Target SYCL architecture, e.g. 'spir64_gen'.")
set(HYPRE_SYCL_TARGET_BACKEND            "" CACHE STRING "Additional SYCL backend options, e.g. '-device 12.1.0,12.4.0'.")

# Set config name values
set_conditional_option(ENABLE "" SHARED)
set_conditional_option(ENABLE "" BIGINT)
set_conditional_option(ENABLE "" MIXEDINT)
set_conditional_option(ENABLE "" SINGLE)
set_conditional_option(ENABLE "" LONG_DOUBLE)
set_conditional_option(ENABLE "" COMPLEX)
set_conditional_option(ENABLE USING HYPRE_BLAS)
set_conditional_option(ENABLE USING HYPRE_LAPACK)
set_conditional_option(ENABLE "" HOPSCOTCH)
set_conditional_option(WITH USING OPENMP)
set_conditional_option(WITH USING CUDA)
set_conditional_option(WITH USING HIP)
set_conditional_option(WITH USING SYCL)
set_conditional_option(WITH USING MAGMA)
set_conditional_option(WITH USING CALIPER)

if (HYPRE_WITH_MPI)
  set(HYPRE_HAVE_MPI ON CACHE BOOL "" FORCE)
  set(HYPRE_SEQUENTIAL OFF CACHE BOOL "" FORCE)
else ()
  set(HYPRE_SEQUENTIAL ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_SUPERLU)
  set(HYPRE_USING_SUPERLU ON CACHE BOOL "" FORCE)
  add_compile_definitions(HAVE_SUPERLU)
endif ()

if (HYPRE_WITH_DSUPERLU)
  set(HYPRE_USING_DSUPERLU ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_HYPRE_BLAS OFF CACHE BOOL "" FORCE)
  set(HYPRE_USING_HYPRE_LAPACK OFF CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_FEI)
  set(HYPRE_USING_FEI ON CACHE BOOL "" FORCE)
  message(WARNING "CMake support for FEI is not complete!")
endif ()

# FEI doesn't currently compile with shared
if (HYPRE_SHARED OR HYPRE_BIGINT OR HYPRE_SINGLE OR HYPRE_LONG_DOUBLE)
  set(HYPRE_USING_FEI OFF CACHE BOOL "" FORCE)
  set(HYPRE_ENABLE_FEI OFF CACHE BOOL "" FORCE)
endif ()

if (HYPRE_SEQUENTIAL)
  set(HYPRE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_UMPIRE)
  set(HYPRE_USING_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE_DEVICE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE_UM ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_UMPIRE_HOST)
  set(HYPRE_USING_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE_HOST ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_UMPIRE_DEVICE)
  set(HYPRE_USING_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE_DEVICE ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_UMPIRE_UM)
  set(HYPRE_USING_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE_UM ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_UMPIRE_PINNED)
  set(HYPRE_USING_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE_PINNED ON CACHE BOOL "" FORCE)
endif ()

# Ensure that HYPRE_ENABLE_SHARED and HYPRE_SHARED do not conflict with BUILD_SHARED_LIBS
if(DEFINED CACHE{BUILD_SHARED_LIBS} AND (DEFINED CACHE{HYPRE_ENABLE_SHARED} OR DEFINED CACHE{HYPRE_SHARED}))
  if ((HYPRE_ENABLE_SHARED AND NOT BUILD_SHARED_LIBS) OR
      (NOT HYPRE_ENABLE_SHARED AND BUILD_SHARED_LIBS))
    message(FATAL_ERROR "Incompatible options: HYPRE_ENABLE_SHARED (${HYPRE_ENABLE_SHARED}) and BUILD_SHARED_LIBS (${BUILD_SHARED_LIBS}) must have the same value.")
  elseif ((HYPRE_SHARED AND NOT BUILD_SHARED_LIBS) OR
          (NOT HYPRE_SHARED AND BUILD_SHARED_LIBS))
    message(FATAL_ERROR "Incompatible options: HYPRE_SHARED (${HYPRE_SHARED}) and BUILD_SHARED_LIBS (${BUILD_SHARED_LIBS}) must have the same value.")
  endif()
endif()

# Set MPI compile flags
if (HYPRE_WITH_MPI)
  configure_mpi_target()
endif ()

# Set OpenMP compile flags
if (HYPRE_USING_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_C)
endif (HYPRE_USING_OPENMP)

# Set GPU compile flags
if (HYPRE_WITH_CUDA OR HYPRE_WITH_HIP OR HYPRE_WITH_SYCL)
  include(HYPRE_SetupGPUToolkit)
else ()
  message(STATUS "GPU support disabled")
  set(HYPRE_USING_HOST_MEMORY ON CACHE BOOL "" FORCE)
endif ()

# Add any extra C compiler flags HYPRE_WITH_EXTRA_CFLAGS
if (NOT HYPRE_WITH_EXTRA_CFLAGS STREQUAL "")
  string(REPLACE " " ";" HYPRE_WITH_EXTRA_CFLAGS_LIST ${HYPRE_WITH_EXTRA_CFLAGS})
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:${HYPRE_WITH_EXTRA_CFLAGS_LIST}>")
endif ()

# Set library build type (must appear before add_library calls)
set(BUILD_SHARED_LIBS ${HYPRE_SHARED} CACHE INTERNAL "Build shared libraries" FORCE)

# Set output directory for the library
set_target_properties(HYPRE PROPERTIES
  VERSION "${HYPRE_VERSION}"
  SOVERSION "${HYPRE_VERSION_MAJOR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Set headers
set(HYPRE_MAIN_HEADERS
  ${CMAKE_CURRENT_BINARY_DIR}/HYPRE_config.h
  HYPREf.h
  HYPRE.h
)
list(APPEND HYPRE_HEADERS ${HYPRE_MAIN_HEADERS})

# TPL variables
set(HYPRE_DEPENDENCY_DIRS "" CACHE INTERNAL "List of absolute paths to TPL directories")
set(HYPRE_NEEDS_CXX OFF CACHE INTERNAL "Flag to indicate if C++ is needed for TPLs")

# Setup TPLs
setup_fei()
setup_tpl_or_internal(blas)
setup_tpl_or_internal(lapack)
setup_tpl(superlu)
setup_tpl(dsuperlu)
setup_tpl(magma)
setup_tpl(caliper)
setup_tpl(umpire)

# Some TPLs need C++ to be enabled
if(HYPRE_NEEDS_CXX)
  enable_language(CXX)
endif()

# Print the directories used to hint the user about the active TPLs
message(STATUS "Dependency directories: ${HYPRE_DEPENDENCY_DIRS}")

# Configure a header file to pass CMake settings to the source code
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config/HYPRE_config.h.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/HYPRE_config.h"
  )

# Headers and sources: remaining subdirectories
set(HYPRE_DIRS utilities multivector krylov seq_mv seq_block_mv parcsr_mv parcsr_block_mv distributed_matrix IJ_mv matrix_matrix distributed_ls parcsr_ls struct_mv struct_ls sstruct_mv sstruct_ls)
foreach (DIR IN LISTS HYPRE_DIRS)
  add_subdirectory(${DIR})
  target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${DIR}>)
endforeach ()

# Set the language for GPU sources (has to be done after the subdirectories are added)
if (HYPRE_WITH_CUDA)
  set_source_files_properties(${HYPRE_GPU_SOURCES} PROPERTIES LANGUAGE CUDA)
elseif (HYPRE_WITH_HIP)
  set_source_files_properties(${HYPRE_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
elseif (HYPRE_WITH_SYCL)
  set_source_files_properties(${HYPRE_GPU_SOURCES} PROPERTIES LANGUAGE SYCL)
endif()

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blas>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lapack>
    $<INSTALL_INTERFACE:include>
)

# Set file properties for MSVC
if (MSVC)
  target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
  if (MSVC_VERSION LESS 1928) # Visual Studio 2019 version 16.8 claims full C11 support
    # Use the C++ compiler to compile these files to get around lack of C99 support
    set_source_files_properties(utilities/hopscotch_hash.c       PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(utilities/merge_sort.c           PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(seq_mv/csr_matop.c               PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_mv/par_csr_matop.c        PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_mv/par_csr_matvec.c       PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/ams.c                  PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/aux_interp.c           PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/par_add_cycle.c        PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/par_amg_setup.c        PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/par_coarsen.c          PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/par_cgc_coarsen.c      PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/par_jacobi_interp.c    PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/par_mgr_setup.c        PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/par_rap.c              PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/par_relax.c            PROPERTIES COMPILE_FLAGS /TP)
    set_source_files_properties(parcsr_ls/par_strength.c         PROPERTIES COMPILE_FLAGS /TP)
  endif()
  if (MSVC_VERSION LESS 1900) #1900 is studio 2015, next older is 1800 which is studio 2013
    #Fix issue with visual studio 2013
    set_source_files_properties(struct_ls/pfmg3_setup_rap.c      PROPERTIES COMPILE_FLAGS /Od)
  endif()
endif ()

# Build FEI, if requested (to be phased out)
if (HYPRE_USING_FEI)
  add_subdirectory(FEI_mv)
endif ()

# Build the examples directory, if requested
if (HYPRE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif ()

# Build the test directory, if requested
if (HYPRE_BUILD_TESTS)
  add_subdirectory(test)
endif ()

# Installation
include(GNUInstallDirs)
install(TARGETS HYPRE
  EXPORT HYPRETargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES ${HYPRE_HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Export the package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/HYPREConfigVersion.cmake"
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config/HYPREConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/HYPREConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/HYPREConfig.cmake
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/HYPREConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/HYPREConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HYPRE
)
# Install the CMake export file
install(EXPORT HYPRETargets
  FILE HYPRETargets.cmake
  NAMESPACE HYPRE::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/HYPRE"
)

# Export from build tree
export(EXPORT HYPRETargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/HYPRETargets.cmake"
  NAMESPACE HYPRE::
)

# Declare an alias so that consumers can depend on HYPRE::HYPRE target
# also when using HYPRE via add_directory or FetchContent
add_library(HYPRE::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Export the package
export(PACKAGE ${PROJECT_NAME})

# Set the base options for printing
set(BASE_OPTIONS
    BUILD_SHARED_LIBS
    HYPRE_ENABLE_SHARED
    HYPRE_ENABLE_BIGINT
    HYPRE_ENABLE_MIXEDINT
    HYPRE_ENABLE_SINGLE
    HYPRE_ENABLE_LONG_DOUBLE
    HYPRE_ENABLE_COMPLEX
    HYPRE_ENABLE_HYPRE_BLAS
    HYPRE_ENABLE_HYPRE_LAPACK
    HYPRE_ENABLE_PERSISTENT_COMM
    HYPRE_ENABLE_FEI
    HYPRE_ENABLE_HOPSCOTCH
    HYPRE_USING_GPU
    HYPRE_WITH_MPI
    HYPRE_WITH_OPENMP
)

# Set the dependency libraries options for printing
set(TPL_OPTIONS
    HYPRE_WITH_CALIPER
    HYPRE_WITH_UMPIRE
    HYPRE_WITH_UMPIRE_HOST
    HYPRE_WITH_UMPIRE_DEVICE
    HYPRE_WITH_UMPIRE_UM
    HYPRE_WITH_UMPIRE_PINNED
    HYPRE_WITH_MAGMA
    HYPRE_WITH_DSUPERLU
)

# Set the GPU options for printing
if(HYPRE_WITH_CUDA)
  set(GPU_OPTIONS
      HYPRE_WITH_CUDA
      HYPRE_WITH_GPU_AWARE_MPI
      HYPRE_ENABLE_CUDA_STREAMS
      HYPRE_ENABLE_CUSPARSE
      HYPRE_ENABLE_CUSOLVER
      HYPRE_ENABLE_DEVICE_POOL
      HYPRE_ENABLE_CUBLAS
      HYPRE_ENABLE_CURAND
      HYPRE_ENABLE_GPU_PROFILING
  )
elseif(HYPRE_WITH_HIP)
  set(GPU_OPTIONS
      HYPRE_WITH_HIP
      HYPRE_WITH_GPU_AWARE_MPI
      HYPRE_ENABLE_ROCBLAS
      HYPRE_ENABLE_ROCSPARSE
      HYPRE_ENABLE_ROCRAND
      HYPRE_ENABLE_ROCSOLVER
      HYPRE_ENABLE_GPU_PROFILING
  )
elseif(HYPRE_WITH_SYCL)
  set(GPU_OPTIONS
      HYPRE_WITH_SYCL
      HYPRE_WITH_GPU_AWARE_MPI
      HYPRE_ENABLE_ONEMKLSPARSE
      HYPRE_ENABLE_ONEMKLBLAS
      HYPRE_ENABLE_ONEMKLRAND
      HYPRE_ENABLE_GPU_PROFILING
  )
endif()

# Print the status of the options for debugging purposes
print_option_status(${BASE_OPTIONS} ${GPU_OPTIONS} ${TPL_OPTIONS})
