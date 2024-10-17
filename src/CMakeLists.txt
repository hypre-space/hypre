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
set(HYPRE_VERSION 2.32.0)
set(HYPRE_NUMBER  23200)
set(HYPRE_DATE    2024/10/08)
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
set(CMAKE_SKIP_BUILD_RPATH FALSE)                         # Includes rpath in the binaries being built
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)                 # Use different paths for build and install
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)               # Use link-time paths
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")    # Where binaries can look for shared libraries
if(NOT BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS OFF)                              # Build static library by default
endif()

# Set cmake module path
set(CMAKE_MODULE_PATH "${HYPRE_SOURCE_DIR}/config/cmake" "${CMAKE_MODULE_PATH}")

# Set default installation directory, but provide a means for users to change
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR NOT DEFINED HYPRE_INSTALL_PREFIX)
  if (NOT DEFINED HYPRE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/hypre" CACHE PATH "Installation directory for HYPRE" FORCE)
  else()
    set(CMAKE_INSTALL_PREFIX "${HYPRE_INSTALL_PREFIX}" CACHE PATH "Installation directory for HYPRE" FORCE)
  endif()
else()
  if (NOT DEFINED HYPRE_INSTALL_PREFIX)
    set(HYPRE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Installation directory for HYPRE" FORCE)
  endif()
endif()

# Set default compile optimization flag
if(NOT CMAKE_BUILD_TYPE AND NOT DEFINED HYPRE_BUILD_TYPE)
  set(HYPRE_BUILD_TYPE "Release" CACHE STRING
      "Optimization flags: set to Debug, Release, RelWithDebInfo, or MinSizeRel")
endif()

# Set the possible values of build type for cmake-gui
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
  "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

# If CMAKE_BUILD_TYPE is not set, use HYPRE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${HYPRE_BUILD_TYPE}" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

# Print the build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Base boolean configuration options
set_hypre_option(BASE HYPRE_ENABLE_SHARED               "Build a shared library" ${BUILD_SHARED_LIBS})
set_hypre_option(BASE HYPRE_ENABLE_BIGINT               "Use long long int for HYPRE_Int and HYPRE_BigInt" OFF)
set_hypre_option(BASE HYPRE_ENABLE_MIXEDINT             "Use long long int for HYPRE_BigInt, int for HYPRE_Int" OFF)
set_hypre_option(BASE HYPRE_ENABLE_SINGLE               "Use float for HYPRE_Real" OFF)
set_hypre_option(BASE HYPRE_ENABLE_LONG_DOUBLE          "Use long double for HYPRE_Real" OFF)
set_hypre_option(BASE HYPRE_ENABLE_COMPLEX              "Use complex values" OFF)
set_hypre_option(BASE HYPRE_ENABLE_INTERNAL_BLAS        "Use internal BLAS library" ON)
set_hypre_option(BASE HYPRE_ENABLE_INTERNAL_LAPACK      "Use internal LAPACK library" ON)
set_hypre_option(BASE HYPRE_ENABLE_PERSISTENT_COMM      "Use persistent communication" OFF)
set_hypre_option(BASE HYPRE_ENABLE_FEI                  "Use FEI" OFF) # TODO: Add this cmake feature
set_hypre_option(BASE HYPRE_ENABLE_HOPSCOTCH            "Use hopscotch hashing with OpenMP" OFF)
set_hypre_option(BASE HYPRE_WITH_OPENMP                 "Use OpenMP" OFF)
set_hypre_option(BASE HYPRE_WITH_MPI                    "Compile with MPI" ON)
set_hypre_option(BASE HYPRE_PRINT_ERRORS                "Print HYPRE errors" OFF)
set_hypre_option(SKIP HYPRE_TIMING                      "Use HYPRE timing routines" OFF)
set_hypre_option(BASE HYPRE_BUILD_EXAMPLES              "Build examples" OFF)
set_hypre_option(BASE HYPRE_BUILD_TESTS                 "Build tests" OFF)
set_hypre_option(BASE HYPRE_USING_HOST_MEMORY           "Use host memory" ON)
# GPU options
set_hypre_option(GPU  HYPRE_WITH_CUDA                   "Use CUDA. Require cuda-8.0 or higher" OFF)
set_hypre_option(GPU  HYPRE_WITH_HIP                    "Use HIP" OFF)
set_hypre_option(GPU  HYPRE_WITH_SYCL                   "Use SYCL" OFF)
set_hypre_option(GPU  HYPRE_WITH_GPU_AWARE_MPI          "Compile with device aware GPU support" OFF)
set_hypre_option(GPU  HYPRE_ENABLE_UNIFIED_MEMORY       "Use unified memory for allocating the memory" OFF)
set_hypre_option(GPU  HYPRE_ENABLE_DEVICE_MALLOC_ASYNC  "Use device async malloc" OFF)
set_hypre_option(GPU  HYPRE_ENABLE_GPU_PROFILING        "Use NVTX on CUDA or rocTX on HIP" OFF)
# CUDA options
set_hypre_option(CUDA HYPRE_ENABLE_CUDA_STREAMS         "Use CUDA streams" ON)
set_hypre_option(CUDA HYPRE_ENABLE_CUSPARSE             "Use cuSPARSE" ON)
set_hypre_option(CUDA HYPRE_ENABLE_CUSOLVER             "Use cuSOLVER" ON)
set_hypre_option(CUDA HYPRE_ENABLE_CUBLAS               "Use cuBLAS" ON)
set_hypre_option(CUDA HYPRE_ENABLE_CURAND               "Use cuRAND" ON)
set_hypre_option(CUDA HYPRE_ENABLE_DEVICE_POOL          "Use CUB's device memory pool" OFF)
# HIP options
set_hypre_option(HIP  HYPRE_ENABLE_ROCBLAS              "Use rocBLAS" ON)
set_hypre_option(HIP  HYPRE_ENABLE_ROCSPARSE            "Use rocSPARSE" ON)
set_hypre_option(HIP  HYPRE_ENABLE_ROCRAND              "Use rocRAND" ON)
set_hypre_option(HIP  HYPRE_ENABLE_ROCSOLVER            "Use rocSOLVER" ON)
# oneAPI options
set_hypre_option(SYCL HYPRE_ENABLE_ONEMKLSPARSE         "Use oneMKL sparse" ON)
set_hypre_option(SYCL HYPRE_ENABLE_ONEMKLBLAS           "Use oneMKL blas" ON)
set_hypre_option(SYCL HYPRE_ENABLE_ONEMKLRAND           "Use oneMKL rand" ON)
# TPL options
set_hypre_option(TPL  HYPRE_WITH_UMPIRE                 "Use Umpire Allocator for device and unified memory" OFF)
set_hypre_option(TPL  HYPRE_WITH_UMPIRE_HOST            "Use Umpire Allocator for host memory" OFF)
set_hypre_option(TPL  HYPRE_WITH_UMPIRE_PINNED          "Use Umpire Allocator for pinned memory" OFF)
set_hypre_option(TPL  HYPRE_WITH_UMPIRE_DEVICE          "Use Umpire Allocator for device memory" OFF)
set_hypre_option(TPL  HYPRE_WITH_UMPIRE_UM              "Use Umpire Allocator for unified memory" OFF)
set_hypre_option(TPL  HYPRE_WITH_SUPERLU                "Use TPL SuperLU" OFF)
set_hypre_option(TPL  HYPRE_WITH_DSUPERLU               "Use TPL SuperLU_Dist" OFF)
set_hypre_option(TPL  HYPRE_WITH_MAGMA                  "Use TPL MAGMA" OFF)
set_hypre_option(TPL  HYPRE_WITH_CALIPER                "Use TPL Caliper" OFF)
set_hypre_option(SKIP TPL_UMPIRE_LIBRARIES              "Absolute paths to Umpire link libraries [Optional]." "")
set_hypre_option(SKIP TPL_UMPIRE_INCLUDE_DIRS           "Absolute paths to Umpire include directories [Optional]." "")
set_hypre_option(SKIP TPL_SUPERLU_LIBRARIES             "Absolute paths to SuperLU link libraries [Optional]." "")
set_hypre_option(SKIP TPL_SUPERLU_INCLUDE_DIRS          "Absolute paths to SuperLU include directories [Optional]." "")
set_hypre_option(SKIP TPL_DSUPERLU_LIBRARIES            "Absolute paths to SuperLU_Dist link libraries [Optional]." "")
set_hypre_option(SKIP TPL_DSUPERLU_INCLUDE_DIRS         "Absolute paths to SuperLU_Dist include directories [Optional]." "")
set_hypre_option(SKIP TPL_MAGMA_LIBRARIES               "Absolute paths to MAGMA link libraries [Optional]." "")
set_hypre_option(SKIP TPL_MAGMA_INCLUDE_DIRS            "Absolute paths to MAGMA include directories [Optional]." "")
set_hypre_option(SKIP TPL_CALIPER_LIBRARIES             "Absolute paths to Caliper link libraries [Optional]." "")
set_hypre_option(SKIP TPL_CALIPER_INCLUDE_DIRS          "Absolute paths to Caliper include directories [Optional]." "")
set_hypre_option(SKIP TPL_BLAS_LIBRARIES                "Absolute paths to BLAS link libraries [Optional]." "")
set_hypre_option(SKIP TPL_LAPACK_LIBRARIES              "Absolute paths to LAPACK link libraries [Optional]." "")
set_hypre_option(SKIP TPL_FEI_INCLUDE_DIRS              "Absolute paths to FEI include directories [Optional]." "")
# Extra flags
set(HYPRE_WITH_EXTRA_CFLAGS              "" CACHE STRING "Define extra C compile flags")
set(HYPRE_WITH_EXTRA_CXXFLAGS            "" CACHE STRING "Define extra CXX compile flags")
set(HYPRE_SYCL_TARGET                    "" CACHE STRING "Target SYCL architecture, e.g. 'spir64_gen'.")
set(HYPRE_SYCL_TARGET_BACKEND            "" CACHE STRING "Additional SYCL backend options, e.g. '-device 12.1.0,12.4.0'.")

# Set config name values
set_internal_hypre_option(ENABLE "" SHARED)
set_internal_hypre_option(ENABLE "" BIGINT)
set_internal_hypre_option(ENABLE "" MIXEDINT)
set_internal_hypre_option(ENABLE "" SINGLE)
set_internal_hypre_option(ENABLE "" LONG_DOUBLE)
set_internal_hypre_option(ENABLE "" COMPLEX)
set_internal_hypre_option(ENABLE USING HYPRE_BLAS)
set_internal_hypre_option(ENABLE USING HYPRE_LAPACK)
set_internal_hypre_option(ENABLE "" HOPSCOTCH)
set_internal_hypre_option(ENABLE USING GPU_STREAMS)
set_internal_hypre_option(ENABLE USING DEVICE_POOL)
set_internal_hypre_option(WITH USING OPENMP)
set_internal_hypre_option(WITH USING CUDA)
set_internal_hypre_option(WITH USING HIP)
set_internal_hypre_option(WITH USING SYCL)
set_internal_hypre_option(WITH USING MAGMA)
set_internal_hypre_option(WITH USING CALIPER)

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
  set(HYPRE_WITH_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE_HOST ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_UMPIRE_DEVICE)
  set(HYPRE_WITH_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE_DEVICE ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_UMPIRE_UM)
  set(HYPRE_WITH_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE_UM ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_UMPIRE_PINNED)
  set(HYPRE_WITH_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE ON CACHE BOOL "" FORCE)
  set(HYPRE_USING_UMPIRE_PINNED ON CACHE BOOL "" FORCE)
endif ()

# Check for conflicting options
ensure_options_match(HYPRE_ENABLE_SHARED BUILD_SHARED_LIBS)
ensure_options_match(HYPRE_INSTALL_PREFIX CMAKE_INSTALL_PREFIX)
ensure_options_match(HYPRE_BUILD_TYPE CMAKE_BUILD_TYPE)
ensure_options_differ(HYPRE_WITH_CUDA HYPRE_WITH_HIP)
ensure_options_differ(HYPRE_WITH_CUDA HYPRE_WITH_SYCL)
ensure_options_differ(HYPRE_WITH_CUDA HYPRE_BIGINT)
ensure_options_differ(HYPRE_WITH_HIP HYPRE_WITH_SYCL)
ensure_options_differ(HYPRE_WITH_HIP HYPRE_BIGINT)

# Set MPI compile flags
if (HYPRE_WITH_MPI)
  configure_mpi_target()
endif ()

# Set OpenMP compile flags
if (HYPRE_USING_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_C)
endif (HYPRE_USING_OPENMP)

# Setup GPU build options: CUDA, HIP, SYCL are mutually exclusive
if (HYPRE_WITH_CUDA OR HYPRE_WITH_HIP OR HYPRE_WITH_SYCL)
  include(HYPRE_SetupGPUToolkit)
else ()
  message(STATUS "GPU support not enabled")
  set(HYPRE_USING_HOST_MEMORY ON CACHE BOOL "" FORCE)
endif ()

# Add any extra C compiler flags HYPRE_WITH_EXTRA_CFLAGS
if (NOT HYPRE_WITH_EXTRA_CFLAGS STREQUAL "")
  string(REPLACE " " ";" HYPRE_WITH_EXTRA_CFLAGS_LIST ${HYPRE_WITH_EXTRA_CFLAGS})
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:${HYPRE_WITH_EXTRA_CFLAGS_LIST}>")
endif ()

# Set output directory for the library
set_target_properties(HYPRE PROPERTIES
  VERSION "${HYPRE_VERSION}"
  SOVERSION "${HYPRE_VERSION_MAJOR}${HYPRE_VERSION_MINOR}"
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
set(HYPRE_DEPENDENCY_DIRS "" CACHE INTERNAL "List of absolute paths to third-party libraries (TPLs) directories")
set(HYPRE_NEEDS_CXX OFF CACHE INTERNAL "Flag to indicate if C++ is needed for TPLs")

# Setup third-party libraries (TPLs)
setup_fei()
setup_tpl_or_internal(blas)
setup_tpl_or_internal(lapack)
setup_tpl(superlu)
setup_tpl(dsuperlu)
setup_tpl(magma)
setup_tpl(caliper)
setup_tpl(umpire)

# Some TPLs need C++ to be enabled (GPU already enables C++)
if(HYPRE_NEEDS_CXX AND NOT HYPRE_USING_GPU)
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
  INSTALL_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/HYPREConfig.cmake"
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/HYPREConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/HYPREConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/HYPRE"
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

# Print status of the build options
print_option_status(${BASE_OPTIONS} ${GPU_OPTIONS} ${TPL_OPTIONS})
