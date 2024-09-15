# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

if(HYPRE_WITH_CUDA)
  # Enforce C++11 at least
  if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 11)
    set(CMAKE_CXX_STANDARD 11)
  endif ()
  include(HYPRE_SetupCUDAToolkit)

elseif(HYPRE_WITH_HIP)
  # Enforce C++14 at least
  if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 14)
    set(CMAKE_CXX_STANDARD 14)
  endif ()
  include(HYPRE_SetupHIPToolkit)

elseif(HYPRE_WITH_SYCL)
  # Enforce C++14 at least
  if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 14)
    set(CMAKE_CXX_STANDARD 14)
  endif ()

  message(STATUS "Enabling SYCL toolkit")
  enable_language(SYCL)
  include(HYPRE_SetupSYCLToolkit)
  set(EXPORT_DEVICE_LIBS ${EXPORT_INTERFACE_SYCL_LIBS})

else()
  message(FATAL_ERROR "Neither CUDA nor HIP nor SYCL is enabled. Please enable one of them.")
endif()
