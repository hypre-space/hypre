# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# This handles the non-compiler aspect of the SYCL toolkit.

if (HYPRE_ENABLE_ONEMKLSPARSE)
  set(HYPRE_USING_ONEMKLSPARSE ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_ONEMKLBLAS)
  set(HYPRE_USING_ONEMKLBLAS ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_ONEMKLRAND)
  set(HYPRE_USING_ONEMKLRAND ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_USING_ONEMKLSPARSE OR HYPRE_USING_ONEMKLBLAS OR HYPRE_USING_ONEMKLRAND)
  set(MKL_LINK static)
  set(MKL_THREADING sequential)
  find_package(MKL CONFIG REQUIRED HINTS "$ENV{MKLROOT}/lib/cmake/mkl")
  target_compile_options(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  target_include_directories(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(${PROJECT_NAME} PUBLIC $<LINK_ONLY:MKL::MKL>)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC $ENV{DPLROOT}/include)

if (HYPRE_SYCL_TARGET)
  target_compile_options(${PROJECT_NAME} PUBLIC -fsycl-targets=${HYPRE_SYCL_TARGET})
  target_link_options(${PROJECT_NAME} PUBLIC -fsycl-targets=${HYPRE_SYCL_TARGET})
endif()

if (HYPRE_SYCL_TARGET_BACKEND)
  target_compile_options(${PROJECT_NAME} PUBLIC -fsycl-targets=${HYPRE_SYCL_TARGET_BACKEND})
  target_link_options(${PROJECT_NAME} PUBLIC -fsycl-targets=${HYPRE_SYCL_TARGET_BACKEND})
endif()
