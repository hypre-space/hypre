# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Ensure that environment variable $MKLROOT is set

# Collection of SYCL/oneAPI optional libraries
set(EXPORT_INTERFACE_SYCL_LIBS "")

if (HYPRE_ENABLE_ONEMKLSPARSE)
  set(HYPRE_USING_ONEMKLSPARSE ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_ONEMKLRNG)
  set(HYPRE_USING_ONEMKLRNG ON CACHE BOOL "" FORCE)
endif ()

if (HYPRE_ENABLE_ONEMKLBLAS)
  set(HYPRE_USING_ONEMKLBLAS ON CACHE BOOL "" FORCE)
endif ()

if ((HYPRE_ENABLE_ONEMKLSPARSE) OR (HYPRE_ENABLE_ONEMKLRNG) OR (HYPRE_ENABLE_ONEMKLBLAS))
  if (HYPRE_SHARED)
    list(APPEND EXPORT_INTERFACE_SYCL_LIBS
      "$ENV{MKLROOT}/lib/intel64/libmkl_sycl.so")
    list(APPEND EXPORT_INTERFACE_SYCL_LIBS
      "$ENV{MKLROOT}/lib/intel64/libmkl_core.so")
  else ()
    list(APPEND EXPORT_INTERFACE_SYCL_LIBS
      "$ENV{MKLROOT}/lib/intel64/libmkl_sycl.a")
    list(APPEND EXPORT_INTERFACE_SYCL_LIBS
      "$ENV{MKLROOT}/lib/intel64/libmkl_core.a")
  endif ()
endif ()
