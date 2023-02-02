# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# This handles the non-compiler aspect of the CUDA toolkit.
# Uses cmake find_package to locate the NVIDIA CUDA C tools 
# for shared libraries. Otherwise for static libraries, assumes
# the libraries are located in ${CUDA_TOOLKIT_ROOT_DIR}/lib64.
# Please set cmake variable CUDA_TOOLKIT_ROOT_DIR. 

# Collection of CUDA optional libraries
set(EXPORT_INTERFACE_CUDA_LIBS "")

if (NOT CUDA_FOUND)
  find_package(CUDA REQUIRED)
endif ()

if (CMAKE_VERSION VERSION_LESS 3.17)

  if (HYPRE_ENABLE_CUSPARSE)
    set(HYPRE_USING_CUSPARSE ON CACHE BOOL "" FORCE)
    if (HYPRE_SHARED)
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS ${CUDA_cusparse_LIBRARY})
    else ()
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparse_static.a)
    endif ()
  endif ()

  if (HYPRE_ENABLE_CURAND)
    set(HYPRE_USING_CURAND ON CACHE BOOL "" FORCE)
    if (HYPRE_SHARED)
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS ${CUDA_curand_LIBRARY})
    else ()
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand_static.a)
    endif ()
  endif ()

  if (HYPRE_ENABLE_CUBLAS)
    set(HYPRE_USING_CUBLAS ON CACHE BOOL "" FORCE)
    if (HYPRE_SHARED)
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS ${CUDA_CUBLAS_LIBRARIES})
    else ()
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a)
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublasLt_static.a)
    endif (HYPRE_SHARED)
  endif (HYPRE_ENABLE_CUBLAS)

  if (HYPRE_ENABLE_CUSOLVER)
    set(HYPRE_USING_CUSOLVER ON CACHE BOOL "" FORCE)
    if (HYPRE_SHARED)
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS ${CUDA_cusolver_LIBRARY})
    else ()
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusolver_static.a)
    endif ()
  endif ()

  if (NOT HYPRE_SHARED)
    list(APPEND EXPORT_INTERFACE_CUDA_LIBS
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a)
  endif ()

  if (HYPRE_ENABLE_GPU_PROFILING)
    set(HYPRE_USING_NVTX ON CACHE BOOL "" FORCE)
    find_library(NVTX_LIBRARY
       NAME libnvToolsExt.so
       PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    message(STATUS "NVidia tools extension library found in " ${NVTX_LIBRARY})
    list(APPEND EXPORT_INTERFACE_CUDA_LIBS ${NVTX_LIBRARY})
  endif (HYPRE_ENABLE_GPU_PROFILING)

else()

  find_package(CUDAToolkit REQUIRED)

  if (HYPRE_SHARED OR WIN32)
    set(HYPRE_CUDA_TOOLKIT_STATIC FALSE)
  else()
    set(HYPRE_CUDA_TOOLKIT_STATIC TRUE)
  endif()

  if (HYPRE_ENABLE_CUSPARSE)
    set(HYPRE_USING_CUSPARSE ON CACHE BOOL "" FORCE)
    if (HYPRE_CUDA_TOOLKIT_STATIC)
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS CUDA::cusparse_static)
    else ()
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS CUDA::cusparse)
    endif ()
  endif ()

  if (HYPRE_ENABLE_CURAND)
    set(HYPRE_USING_CURAND ON CACHE BOOL "" FORCE)
    if (HYPRE_CUDA_TOOLKIT_STATIC)
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS CUDA::curand_static)
    else ()
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS CUDA::curand)
    endif ()
  endif ()

  if (HYPRE_ENABLE_CUBLAS)
    set(HYPRE_USING_CUBLAS ON CACHE BOOL "" FORCE)
    if (HYPRE_CUDA_TOOLKIT_STATIC)
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS CUDA::cublas_static CUDA::cublasLt_static)
    else ()
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS CUDA::cublas CUDA::cublasLt)
    endif (HYPRE_CUDA_TOOLKIT_STATIC)
  endif (HYPRE_ENABLE_CUBLAS)

  if (HYPRE_ENABLE_CUSOLVER)
    set(HYPRE_USING_CUSOLVER ON CACHE BOOL "" FORCE)
    if (HYPRE_CUDA_TOOLKIT_STATIC)
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS CUDA::cusolver_static)
    else ()
      list(APPEND EXPORT_INTERFACE_CUDA_LIBS CUDA::cusolver)
    endif ()
  endif ()

  if (HYPRE_CUDA_TOOLKIT_STATIC)
    list(APPEND EXPORT_INTERFACE_CUDA_LIBS CUDA::culibos)
  endif (HYPRE_CUDA_TOOLKIT_STATIC)

  if (HYPRE_ENABLE_GPU_PROFILING)
    set(HYPRE_USING_NVTX ON CACHE BOOL "" FORCE)
    list(APPEND EXPORT_INTERFACE_CUDA_LIBS CUDA::nvToolsExt)
  endif (HYPRE_ENABLE_GPU_PROFILING)

endif()
