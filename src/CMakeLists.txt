# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

cmake_minimum_required(VERSION 3.13...3.16)

if (${CMAKE_VERSION} VERSION_LESS 3.16)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else ()
  cmake_policy(VERSION 3.16)
endif ()

# The version number.
set(HYPRE_VERSION 2.22.1)
set(HYPRE_NUMBER  22201)
set(HYPRE_DATE    2021/08/20)
set(HYPRE_TIME    00:00:00)
set(HYPRE_BUGS    https://github.com/hypre-space/hypre/issues)
set(HYPRE_SRCDIR  "${PROJECT_SOURCE_DIR}")

set(PROJECT_NAME HYPRE)
project(${PROJECT_NAME}
  VERSION ${HYPRE_VERSION}
  LANGUAGES C)

if (${HYPRE_SOURCE_DIR} STREQUAL ${HYPRE_BINARY_DIR})
  message(FATAL_ERROR "In-place build not allowed! Please use a separate build directory. See the Users Manual or INSTALL file for details.")
endif ()

# Set cmake module path
set(CMAKE_MODULE_PATH "${HYPRE_SOURCE_DIR}/config/cmake" "${CMAKE_MODULE_PATH}")
include(HYPRE_CMakeUtilities)

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

# Configuration options
option(HYPRE_ENABLE_SHARED           "Build a shared library" OFF)
option(HYPRE_ENABLE_BIGINT           "Use long long int for HYPRE_Int" OFF)
option(HYPRE_ENABLE_MIXEDINT         "Use long long int for HYPRE_BigInt, int for HYPRE_INT" OFF)
option(HYPRE_ENABLE_SINGLE           "Use float for HYPRE_Real" OFF)
option(HYPRE_ENABLE_LONG_DOUBLE      "Use long double for HYPRE_Real" OFF)
option(HYPRE_ENABLE_COMPLEX          "Use complex values" OFF)
option(HYPRE_ENABLE_HYPRE_BLAS       "Use internal BLAS library" ON)
option(HYPRE_ENABLE_HYPRE_LAPACK     "Use internal LAPACK library" ON)
option(HYPRE_ENABLE_PERSISTENT_COMM  "Use persistent communication" OFF)
option(HYPRE_ENABLE_FEI              "Use FEI" OFF) # TODO: Add this cmake feature
option(HYPRE_WITH_MPI                "Compile with MPI" ON)
option(HYPRE_WITH_OPENMP             "Use OpenMP" OFF)
option(HYPRE_ENABLE_HOPSCOTCH        "Use hopscotch hashing with OpenMP" OFF)
option(HYPRE_WITH_DSUPERLU           "Use TPL SuperLU_Dist" OFF)
option(HYPRE_WITH_CALIPER            "Use Caliper" OFF)  # TODO: Finish this cmake feature
option(HYPRE_PRINT_ERRORS            "Print HYPRE errors" OFF)
option(HYPRE_TIMING                  "Use HYPRE timing routines" OFF)
option(HYPRE_BUILD_EXAMPLES          "Build examples" OFF)
option(HYPRE_BUILD_TESTS             "Build tests" OFF)
option(HYPRE_USING_HOST_MEMORY       "Use host memory" ON)
set(HYPRE_WITH_EXTRA_CFLAGS       "" CACHE STRING "Define extra C compile flags")
set(HYPRE_WITH_EXTRA_CXXFLAGS     "" CACHE STRING "Define extra CXX compile flags")
# CUDA options
option(HYPRE_WITH_CUDA               "Use CUDA. Require cuda-8.0 or higher" OFF)
option(HYPRE_ENABLE_UNIFIED_MEMORY   "Use unified memory for allocating the memory" OFF)
option(HYPRE_ENABLE_CUDA_STREAMS     "Use CUDA streams" ON)
option(HYPRE_ENABLE_CUSPARSE         "Use cuSPARSE" ON)
option(HYPRE_ENABLE_DEVICE_POOL      "Use device memory pool" OFF)
option(HYPRE_ENABLE_CUBLAS           "Use cuBLAS" OFF)
option(HYPRE_ENABLE_CURAND           "Use cuRAND" ON)
option(HYPRE_ENABLE_GPU_PROFILING    "Use NVTX on CUDA" OFF)
set(HYPRE_CUDA_SM "70" CACHE STRING  "Target CUDA architecture.")

option(TPL_DSUPERLU_LIBRARIES        "List of absolute paths to SuperLU_Dist link libraries [].")
option(TPL_DSUPERLU_INCLUDE_DIRS     "List of absolute paths to SuperLU_Dist include directories [].")
option(TPL_BLAS_LIBRARIES            "Optional list of absolute paths to BLAS libraries, otherwise use FindBLAS to locate [].")
option(TPL_LAPACK_LIBRARIES          "Optional list of absolute paths to LAPACK libraries, otherwise use FindLAPACK to locate [].")

# Set config name values
if (HYPRE_ENABLE_SHARED)
  set(HYPRE_SHARED ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_BIGINT)
  set(HYPRE_BIGINT ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_MIXEDINT)
  set(HYPRE_MIXEDINT ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_SINGLE)
  set(HYPRE_SINGLE ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_LONG_DOUBLE)
  set(HYPRE_LONG_DOUBLE ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_COMPLEX)
  set(HYPRE_COMPLEX ON CACHE BOOL "" FORCE)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(HYPRE_DEBUG ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_HYPRE_BLAS)
  set(HYPRE_USING_HYPRE_BLAS ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_HYPRE_LAPACK)
  set(HYPRE_USING_HYPRE_LAPACK ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_PERSISTENT_COMM)
  set(HYPRE_USING_PERSISTENT_COMM ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_MPI)
  set(HYPRE_HAVE_MPI ON CACHE BOOL "" FORCE)
  set(HYPRE_SEQUENTIAL OFF CACHE BOOL "" FORCE)
else ()
  set(HYPRE_SEQUENTIAL ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_WITH_OPENMP)
  set(HYPRE_USING_OPENMP ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_HOPSCOTCH)
  set(HYPRE_HOPSCOTCH ON CACHE BOOL "" FORCE)
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

if (HYPRE_WITH_CALIPER)
  set(HYPRE_USING_CALIPER ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_SHARED OR HYPRE_BIGINT OR HYPRE_SINGLE OR HYPRE_LONG_DOUBLE)
  # FEI doesn't currently compile with shared
  set(HYPRE_USING_FEI OFF CACHE BOOL "" FORCE)
endif ()

if (HYPRE_SEQUENTIAL)
  set(HYPRE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
endif ()

# CUDA
if (HYPRE_WITH_CUDA)
  enable_language(CXX)
  message(STATUS "Enabled support for CXX.")

  # Enforce C++11
  if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 11)
    set(CMAKE_CXX_STANDARD 11)
  endif ()
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  message(STATUS "Using CXX standard: c++${CMAKE_CXX_STANDARD}")

  # Use ${CMAKE_CXX_COMPILER} as the cuda host compiler.
  if (NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif ()

  # Add any extra CXX compiler flags HYPRE_WITH_EXTRA_CXXFLAGS
  if (NOT HYPRE_WITH_EXTRA_CXXFLAGS STREQUAL "")
    string(REPLACE " " ";" HYPRE_WITH_EXTRA_CXXFLAGS "${HYPRE_WITH_EXTRA_CXXFLAGS}")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${HYPRE_WITH_EXTRA_CXXFLAGS}>")
  endif ()

  # Check if CUDA is available, then enable it
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)

    enable_language(CUDA)
    message(STATUS "Enabled support for CUDA.")

    if (NOT CMAKE_CUDA_STANDARD OR CMAKE_CUDA_STANDARD EQUAL 98)
      set(CMAKE_CUDA_STANDARD 11)
    endif ()

    set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "" FORCE)

    set(HYPRE_USING_CUDA ON CACHE BOOL "" FORCE)
    set(HYPRE_USING_GPU ON CACHE BOOL "" FORCE)

    if (HYPRE_ENABLE_UNIFIED_MEMORY)
      set(HYPRE_USING_UNIFIED_MEMORY ON CACHE BOOL "" FORCE)
    else ()
      set(HYPRE_USING_DEVICE_MEMORY ON CACHE BOOL "" FORCE)
    endif ()

    # Check if examples are enabled, but not unified memory
    if (HYPRE_BUILD_EXAMPLES AND NOT HYPRE_ENABLE_UNIFIED_MEMORY)
      message(WARNING "Running the examples on GPUs requires Unified Memory!
        Examples will not be built!")
      set(HYPRE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    endif ()

    if (CMAKE_VERSION VERSION_LESS 3.18.0)
      add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${HYPRE_CUDA_SM}>")
    else ()
      set(CMAKE_CUDA_ARCHITECTURES "${HYPRE_CUDA_SM}")
    endif ()
    message(STATUS "Using CUDA architecture: ${HYPRE_CUDA_SM}")

    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-expt-extended-lambda>")

    set(HYPRE_USING_HOST_MEMORY OFF CACHE BOOL "" FORCE)

    if (HYPRE_ENABLE_CUDA_STREAMS)
      set(HYPRE_USING_CUDA_STREAMS ON CACHE BOOL "" FORCE)
    endif (HYPRE_ENABLE_CUDA_STREAMS)

    if (HYPRE_ENABLE_DEVICE_POOL)
      set(HYPRE_USING_DEVICE_POOL ON CACHE BOOL "" FORCE)
    endif (HYPRE_ENABLE_DEVICE_POOL)

    # TODO Eventually should require cmake>=3.17
    # and use cmake's FindCUDAToolkit. Now collect
    # CUDA optional libraries.
    include(HYPRE_SetupCUDAToolkit)
  else ()
    message(WARNING "No CUDA support!")
    set(HYPRE_USING_HOST_MEMORY ON CACHE BOOL "" FORCE)
  endif (CMAKE_CUDA_COMPILER)
endif (HYPRE_WITH_CUDA)

# Add any extra C compiler flags HYPRE_WITH_EXTRA_CFLAGS
if (NOT HYPRE_WITH_EXTRA_CFLAGS STREQUAL "")
  string(REPLACE " " ";" HYPRE_WITH_EXTRA_CFLAGS "${HYPRE_WITH_EXTRA_CFLAGS}")
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:${HYPRE_WITH_EXTRA_CFLAGS}>")
endif ()

# Set library build type (must appear before add_library calls)
if (HYPRE_SHARED)
  set(BUILD_SHARED_LIBS ON CACHE INTERNAL "" FORCE)
else ()
  set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "" FORCE)
endif ()

# Create the HYPRE library object
add_library(${PROJECT_NAME})

# Headers and sources
set(HYPRE_HEADERS "")

# Headers and sources: .
set(HYPRE_MAIN_HEADERS
  ${CMAKE_CURRENT_BINARY_DIR}/HYPRE_config.h
  HYPREf.h
  HYPRE.h
  )

set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HYPRE_MAIN_HEADERS})

# Headers and sources: blas
if (HYPRE_USING_HYPRE_BLAS)
  add_subdirectory(blas)
else ()
  # Use TPL_BLAS_LIBRARIES if set.
  if (TPL_BLAS_LIBRARIES)
    message(STATUS "Using TPL_BLAS_LIBRARIES='${TPL_BLAS_LIBRARIES}'")
    target_link_libraries(${PROJECT_NAME} PUBLIC "${TPL_BLAS_LIBRARIES}")
  else ()
    # Find system blas
    find_package(BLAS REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC "${BLAS_LIBRARIES}")
  endif ()
  target_compile_definitions(${PROJECT_NAME} PUBLIC "USE_VENDOR_BLAS")
endif ()

# Headers and sources: lapack
if (HYPRE_USING_HYPRE_LAPACK)
  add_subdirectory(lapack)
else ()
  # Use TPL_LAPACK_LIBRARIES if set.
  if (TPL_LAPACK_LIBRARIES)
    message(STATUS "Using TPL_LAPACK_LIBRARIES='${TPL_LAPACK_LIBRARIES}'")
    target_link_libraries(${PROJECT_NAME} PUBLIC "${TPL_LAPACK_LIBRARIES}")
  else ()
    # Find system lapack
    find_package(LAPACK REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC "${LAPACK_LIBRARIES}")
  endif ()
endif ()

# Find DSUPERLU, if requested
if (HYPRE_USING_DSUPERLU)
  if (NOT TPL_DSUPERLU_LIBRARIES)
    message(FATAL_ERROR "TPL_DSUPERLU_LIBRARIES option should be set for SuperLU_Dist support.")
  endif ()

  if (NOT TPL_DSUPERLU_INCLUDE_DIRS)
    message(FATAL_ERROR "TPL_DSUPERLU_INCLUDE_DIRS option be set for SuperLU_Dist support.")
  endif ()

  foreach (dir ${TPL_DSUPERLU_INCLUDE_DIRS})
    if (NOT EXISTS ${dir})
      message(FATAL_ERROR "SuperLU_Dist include directory not found: ${dir}")
    endif ()
    set(CMAKE_C_FLAGS "-I${dir} ${CMAKE_C_FLAGS}")
  endforeach ()
  message(STATUS "Enabled support for using DSUPERLU.")
  set(DSUPERLU_FOUND TRUE)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${TPL_DSUPERLU_LIBRARIES} stdc++)
  target_include_directories(${PROJECT_NAME} PUBLIC ${TPL_DSUPERLU_INCLUDE_DIRS})
endif (HYPRE_USING_DSUPERLU)

if (DSUPERLU_FOUND)
  set(HYPRE_USING_DSUPERLU TRUE)
endif ()

if (HYPRE_USING_CUDA)
  target_link_libraries(${PROJECT_NAME} PUBLIC "${EXPORT_INTERFACE_CUDA_LIBS}")
  if (HYPRE_HAVE_MPI)
    target_include_directories(${PROJECT_NAME} PUBLIC
      ${MPI_CXX_INCLUDE_DIRS})
  endif ()
endif ()

# Configure a header file to pass CMake settings to the source code
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config/HYPRE_config.h.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/HYPRE_config.h"
  )

# Headers and sources: remaining subdirectories
set(HYPRE_DIRS utilities multivector krylov seq_mv parcsr_mv parcsr_block_mv distributed_matrix IJ_mv matrix_matrix distributed_ls parcsr_ls struct_mv struct_ls sstruct_mv sstruct_ls)
foreach (DIR IN LISTS HYPRE_DIRS)
  add_subdirectory(${DIR})
  target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${DIR}>)
endforeach ()

# BINARY must be first in order to get the correct HYPRE_config.h file
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${HYPRE_BINARY_DIR}>
  $<BUILD_INTERFACE:${HYPRE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blas>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lapack>
  $<INSTALL_INTERFACE:include>
  )

if (HYPRE_USING_CUDA)
  set_source_files_properties(${HYPRE_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif ()

# Set MPI compile flags
if (NOT HYPRE_SEQUENTIAL)
  find_program(MPIEXEC_EXECUTABLE NAMES mpiexec mpirun)
  find_package(MPI REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_C)
endif (NOT HYPRE_SEQUENTIAL)

# Set OpenMP compile flags
if (HYPRE_USING_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_C)
endif (HYPRE_USING_OPENMP)

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

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME}
  EXPORT HYPRETargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES ${HYPRE_HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  HYPREConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
  )

install(EXPORT HYPRETargets
  FILE HYPRETargets.cmake
  NAMESPACE HYPRE::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/HYPRE"
  )

configure_package_config_file(
  config/HYPREConfig.cmake.in HYPREConfig.cmake
  INSTALL_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/HYPREConfig.cmake"
  )
install(
  FILES
      "${CMAKE_CURRENT_BINARY_DIR}/HYPREConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/HYPREConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/HYPRE"
  )

export(EXPORT HYPRETargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/HYPRETargets.cmake"
  NAMESPACE HYPRE::
  )

export(PACKAGE ${PROJECT_NAME})
