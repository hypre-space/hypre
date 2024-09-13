# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

if(HYPRE_WITH_CUDA)
  include(HYPRE_SetupCUDAToolkit)
elseif(HYPRE_WITH_HIP)
  include(HYPRE_SetupHIPToolkit)
else()
  message(FATAL_ERROR "Neither CUDA nor HIP is enabled. Please enable one of them.")
endif()