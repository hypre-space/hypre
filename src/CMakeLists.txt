# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

cmake_minimum_required (VERSION 2.8.8)
if (POLICY CMP0074)
   cmake_policy(SET CMP0074 NEW) 
endif (POLICY CMP0074)

project (hypre)

# The version number.
set (HYPRE_VERSION 2.18.2)
set (HYPRE_DATE    2019/10/28)
set (HYPRE_TIME    00:00:00)
set (HYPRE_BUGS    hypre-support@llnl.gov)
set (HYPRE_SRCDIR  "${PROJECT_SOURCE_DIR}")

if (${hypre_SOURCE_DIR} STREQUAL ${hypre_BINARY_DIR})
  message(FATAL_ERROR "In-place build not allowed! Please use a separate build directory. See the Users Manual or INSTALL file for details.")
endif ()

# Set cmake module path 
set(CMAKE_MODULE_PATH ${hypre_SOURCE_DIR}/config/cmake) 
include(hypre_CMakeUtilities)

# Set default installation directory, but provide a means for users to change
set (HYPRE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/hypre" CACHE PATH
  "Installation directory for HYPRE")
set (CMAKE_INSTALL_PREFIX "${HYPRE_INSTALL_PREFIX}" CACHE INTERNAL "" FORCE)

# Set default compile optimization flag
set (HYPRE_BUILD_TYPE "Release" CACHE STRING
  "Optimization flags: set to Debug, Release, RelWithDebInfo, or MinSizeRel")
set (CMAKE_BUILD_TYPE "${HYPRE_BUILD_TYPE}" CACHE INTERNAL "" FORCE)

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
option(HYPRE_ENABLE_GLOBAL_PARTITION "Use global partitioning" OFF)
option(HYPRE_ENABLE_FEI              "Use FEI" OFF) # TODO: Add this cmake feature
option(HYPRE_WITH_MPI                "Compile with MPI" ON)
option(HYPRE_WITH_OPENMP             "Use OpenMP" OFF)
option(HYPRE_WITH_HOPSCOTCH          "Use hopscotch hashing with OpenMP" OFF)
option(HYPRE_WITH_DSUPERLU           "Use TPL SuperLU_Dist" OFF)
option(HYPRE_WITH_CALIPER            "Use Caliper" OFF)  # TODO: Finish this cmake feature
option(HYPRE_PRINT_ERRORS            "Print HYPRE errors" OFF)
option(HYPRE_TIMING                  "Use HYPRE timing routines" OFF)
option(HYPRE_BUILD_EXAMPLES          "Build examples" OFF)
option(HYPRE_BUILD_TESTS             "Build tests" OFF)

option(TPL_DSUPERLU_LIBRARIES        "List of absolute paths to SuperLU_Dist link libraries [].")
option(TPL_DSUPERLU_INCLUDE_DIRS     "List of absolute paths to SuperLU_Dist include directories [].")

# Set config name values
if (HYPRE_ENABLE_SHARED)
  set(HYPRE_SHARED ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_BIGINT)
  set(HYPRE_BIGINT ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_MIXEDINT)
  set(HYPRE_MIXEDINT ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_SINGLE)
  set(HYPRE_SINGLE ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_LONG_DOUBLE)
  set(HYPRE_LONG_DOUBLE ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_COMPLEX)
  set(HYPRE_COMPLEX ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_HYPRE_BLAS)
  set(HYPRE_USING_HYPRE_BLAS ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_HYPRE_LAPACK)
  set(HYPRE_USING_HYPRE_LAPACK ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_PERSISTENT_COMM)
  set(HYPRE_USING_PERSISTENT_COMM ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_GLOBAL_PARTITION)
  set(HYPRE_NO_GLOBAL_PARTITION OFF CACHE BOOL "" FORCE)
elseif()
  set(HYPRE_NO_GLOBAL_PARTITION ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_WITH_MPI)
  set(HYPRE_SEQUENTIAL OFF CACHE BOOL "" FORCE)
elseif()
  set(HYPRE_SEQUENTIAL ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_WITH_OPENMP)
  set (HYPRE_USING_OPENMP ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_WITH_HOPSCOTCH)
  set (HYPRE_HOPSCOTCH ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_WITH_DSUPERLU)
  set (HYPRE_USING_DSUPERLU ON CACHE BOOL "" FORCE)
  set (HYPRE_USING_HYPRE_BLAS OFF CACHE BOOL "" FORCE)
  set (HYPRE_USING_HYPRE_LAPACK OFF CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_FEI)
  set (HYPRE_USING_FEI ON CACHE BOOL "" FORCE)
  message(WARNING "CMake support for FEI is not complete!")
endif()

if (HYPRE_WITH_CALIPER)
  set (HYPRE_USING_CALIPER ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_SHARED OR HYPRE_BIGINT OR HYPRE_SINGLE OR HYPRE_LONG_DOUBLE)
  # FEI doesn't currently compile with shared
  set (HYPRE_USING_FEI OFF CACHE BOOL "" FORCE)
endif ()

if (HYPRE_SEQUENTIAL)
  set (HYPRE_NO_GLOBAL_PARTITION OFF CACHE BOOL "" FORCE)
  set (HYPRE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
endif ()

# Headers and sources
set (HYPRE_HEADERS "")
set (HYPRE_SOURCES "")

# Headers and sources: .
list (APPEND HYPRE_HEADERS
  "${PROJECT_BINARY_DIR}/HYPRE_config.h"
  HYPREf.h
  HYPRE.h
)

# This is a list of TPLs that are used by all targets
set(TPL_LIBRARIES "")

# This is a list of linker flags to be used with TPLs for all targets
set(TPL_LINKER_FLAGS "")

# Headers and sources: blas
if (HYPRE_USING_HYPRE_BLAS)
  add_subdirectory(blas) 
else()
  # Find system blas
  find_package(BLAS REQUIRED)
  list(APPEND TPL_LIBRARIES ${BLAS_LIBRARIES})
  set(CMAKE_C_FLAGS "-DUSE_VENDOR_BLAS ${CMAKE_C_FLAGS}")
endif()

# Headers and sources: lapack
if (HYPRE_USING_HYPRE_LAPACK)
  add_subdirectory(lapack) 
else()
  # Find system lapack
  find_package(LAPACK REQUIRED)
  list(APPEND TPL_LIBRARIES ${LAPACK_LIBRARIES})
endif()

# Find DSUPERLU, if requested 
if (HYPRE_USING_DSUPERLU)
  if (NOT TPL_DSUPERLU_LIBRARIES)
    message(FATAL_ERROR "TPL_DSUPERLU_LIBRARIES option should be set for SuperLU_Dist support.")
  endif()

  if (NOT TPL_DSUPERLU_INCLUDE_DIRS)
    message(FATAL_ERROR "TPL_DSUPERLU_INCLUDE_DIRS option be set for SuperLU_Dist support.")
  endif()
  foreach(dir ${TPL_DSUPERLU_INCLUDE_DIRS})
    if (NOT EXISTS ${dir})
      message(FATAL_ERROR "SuperLU_Dist include directory not found: ${dir}")
    endif()
    set(CMAKE_C_FLAGS "-I${dir} ${CMAKE_C_FLAGS}")
  endforeach()
  message("-- Enabled support for using DSUPERLU.")
  set(DSUPERLU_FOUND TRUE)
  list(APPEND TPL_LIBRARIES ${TPL_DSUPERLU_LIBRARIES} stdc++)
endif()

if (TPL_DSUPERLU_INCLUDE_DIRS)
  include_directories(${TPL_DSUPERLU_INCLUDE_DIRS})  
endif ()

if(DSUPERLU_FOUND)
  set(HYPRE_USING_DSUPERLU TRUE)
endif()


# Headers and sources: remaining subdirectories
set(HYPRE_DIRS utilities multivector krylov seq_mv parcsr_mv parcsr_block_mv distributed_matrix IJ_mv matrix_matrix distributed_ls parcsr_ls struct_mv struct_ls sstruct_mv sstruct_ls)
foreach(DIR IN LISTS HYPRE_DIRS)
  add_subdirectory(${DIR})
endforeach()

# Configure a header file to pass CMake settings to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/config/HYPRE_config.h.cmake.in"
  "${PROJECT_BINARY_DIR}/HYPRE_config.h"
  )

# BINARY must be first in order to get the correct HYPRE_config.h file
include_directories(${hypre_BINARY_DIR})
include_directories(${hypre_SOURCE_DIR})
include_directories(blas)
include_directories(lapack)
# Add remaining subdirectories
foreach(DIR IN LISTS HYPRE_DIRS)
  include_directories(${DIR})
endforeach()
include_directories(distributed_ls/Euclid)
include_directories(distributed_ls/ParaSails)

# Set library build type
if (HYPRE_SHARED)
  set (BUILD_SHARED_LIBS ON CACHE INTERNAL "" FORCE)
else ()
  set (BUILD_SHARED_LIBS OFF CACHE INTERNAL "" FORCE)
endif ()

# Set MPI compile flags
if (NOT HYPRE_SEQUENTIAL)
  find_package (MPI)
  if ((MPI_C_FOUND) AND (NOT CMAKE_C_COMPILER STREQUAL MPI_C_COMPILER))
    include_directories (${MPI_C_INCLUDE_PATH})
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}")
    list(APPEND TPL_LIBRARIES ${MPI_C_LIBRARIES})
  endif ()
endif (NOT HYPRE_SEQUENTIAL)

# Set OpenMP compile flags
if (HYPRE_USING_OPENMP)
  find_package (OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  endif (OPENMP_FOUND)
endif (HYPRE_USING_OPENMP)

if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  # Use the C++ compiler to compile these files to get around lack of C99 support
  set_source_files_properties (utilities/hypre_hopscotch_hash.c PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (utilities/hypre_merge_sort.c     PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (seq_mv/csr_matop.c               PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_mv/par_csr_matop.c        PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_mv/par_csr_matvec.c       PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/ams.c                  PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/aux_interp.c           PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/par_add_cycle.c        PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/par_amg_setup.c        PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/par_coarsen.c          PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/par_cgc_coarsen.c      PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/par_jacobi_interp.c    PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/par_mgr_setup.c        PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/par_rap.c              PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/par_relax.c            PROPERTIES COMPILE_FLAGS /TP)
  set_source_files_properties (parcsr_ls/par_strength.c         PROPERTIES COMPILE_FLAGS /TP)
  #Fix issue with visual studio 2013
  set_source_files_properties (struct_ls/pfmg3_setup_rap.c      PROPERTIES COMPILE_FLAGS /Od)
endif ()

if (HYPRE_USING_FEI)
  add_subdirectory(FEI_mv)
endif ()

# Build the examples directory, if requested
if (HYPRE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Build the test directory, if requested
if (HYPRE_BUILD_TESTS)
  add_subdirectory(test)
endif()

# Cleanup the TPL list
list(REMOVE_DUPLICATES TPL_LIBRARIES)

add_library (HYPRE ${HYPRE_SOURCES} ${HYPRE_HEADERS} ${FEI_LIBS})

target_link_libraries(HYPRE PUBLIC ${TPL_LIBRARIES})

install (TARGETS HYPRE DESTINATION lib)
install (FILES ${HYPRE_HEADERS} DESTINATION include)

