# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# This handles the non-compiler aspect of the SYCL toolkit.
message(STATUS "Enabling SYCL toolkit")

# limit C++ errors to one
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel|Clang")
  target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ferror-limit=1>)
endif()

# Find Intel SYCL
find_package(IntelSYCL REQUIRED)

# Set up SYCL flags
if(IntelSYCL_FOUND)
  # Standard SYCL flags
  target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsycl>)
  target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsycl-unnamed-lambda>)
  target_link_options(${PROJECT_NAME} PUBLIC -fsycl)
  target_link_options(${PROJECT_NAME} PUBLIC -fsycl-device-code-split=per_kernel)
  target_link_options(${PROJECT_NAME} PUBLIC -Wl,--no-relax)

  # Use either user-specified target or IntelSYCL's default
  if(HYPRE_SYCL_TARGET)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsycl-targets=${HYPRE_SYCL_TARGET}>)
    target_link_options(${PROJECT_NAME} PUBLIC -fsycl-targets=${HYPRE_SYCL_TARGET})
  elseif(INTEL_SYCL_TARGETS)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsycl-targets=${INTEL_SYCL_TARGETS}>)
    target_link_options(${PROJECT_NAME} PUBLIC -fsycl-targets=${INTEL_SYCL_TARGETS})
  endif()

  # Use either user-specified backend or IntelSYCL's default
  if(HYPRE_SYCL_TARGET_BACKEND)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Xsycl-target-backend=${HYPRE_SYCL_TARGET_BACKEND}>)
    target_link_options(${PROJECT_NAME} PUBLIC -Xsycl-target-backend=${HYPRE_SYCL_TARGET_BACKEND})
  elseif(INTEL_SYCL_BACKEND)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Xsycl-target-backend=${INTEL_SYCL_BACKEND}>)
    target_link_options(${PROJECT_NAME} PUBLIC -Xsycl-target-backend=${INTEL_SYCL_BACKEND})
  endif()
endif()

# Find Intel DPCT
if(NOT DEFINED DPCTROOT)
  if(DEFINED ENV{DPCTROOT})
    set(DPCTROOT $ENV{DPCTROOT})
  elseif(DEFINED ENV{DPCT_BUNDLE_ROOT})
    set(DPCTROOT $ENV{DPCT_BUNDLE_ROOT})
  elseif(DEFINED ENV{ONEAPI_ROOT} AND EXISTS "$ENV{ONEAPI_ROOT}/dpcpp-ct/latest")
    set(DPCTROOT "$ENV{ONEAPI_ROOT}/dpcpp-ct/latest")
  endif()
endif()

# Check if DPCT is found
if(NOT EXISTS "${DPCTROOT}/include/dpct/dpct.hpp")
  message(FATAL_ERROR "Could not find DPCT installation. Please set DPCTROOT")
endif()

# Add DPCT include directory
target_include_directories(${PROJECT_NAME} PUBLIC "${DPCTROOT}/include")
message(STATUS "DPCT include directory: ${DPCTROOT}/include")

if (HYPRE_ENABLE_ONEMKLSPARSE)
  set(HYPRE_USING_ONEMKLSPARSE ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_ONEMKLBLAS)
  set(HYPRE_USING_ONEMKLBLAS ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_ONEMKLRAND)
  set(HYPRE_USING_ONEMKLRAND ON CACHE BOOL "" FORCE)
endif()

# Setup MKL
if (HYPRE_USING_ONEMKLSPARSE OR HYPRE_USING_ONEMKLBLAS OR HYPRE_USING_ONEMKLRAND)
  # Find MKL
  set(MKL_LINK dynamic)
  set(MKL_THREADING sequential)
  #set(ENABLE_TRY_SYCL_COMPILE ON) # This option slows down the build
  find_package(MKL CONFIG REQUIRED HINTS "$ENV{MKLROOT}/lib/cmake/mkl")

  # Add all required MKL components explicitly
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
    $<LINK_ONLY:MKL::MKL_SYCL::BLAS>
    $<LINK_ONLY:MKL::MKL_SYCL::LAPACK>
    $<LINK_ONLY:MKL::MKL_SYCL::SPARSE>
    $<LINK_ONLY:MKL::MKL_SYCL::VM>
    $<LINK_ONLY:MKL::MKL_SYCL::RNG>
    $<LINK_ONLY:MKL::MKL_SYCL::STATS>
    $<LINK_ONLY:MKL::MKL_SYCL::DATA_FITTING>
  )

  # Ensure compile options and include directories are properly propagated
  target_compile_options(${PROJECT_NAME} PUBLIC
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
  )
  target_include_directories(${PROJECT_NAME} PUBLIC
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()
