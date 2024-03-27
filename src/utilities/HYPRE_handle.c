/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_handle utility functions
 *
 *****************************************************************************/

#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * HYPRE_SetSpTransUseVendor
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SetSpTransUseVendor( HYPRE_Int use_vendor )
{
   return hypre_SetSpTransUseVendor(use_vendor);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetSpMVUseVendor
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SetSpMVUseVendor( HYPRE_Int use_vendor )
{
   return hypre_SetSpMVUseVendor(use_vendor);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetSpGemmUseVendor
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SetSpGemmUseVendor( HYPRE_Int use_vendor )
{
   return hypre_SetSpGemmUseVendor(use_vendor);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetUseGpuRand
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SetUseGpuRand( HYPRE_Int use_gpu_rand )
{
   return hypre_SetUseGpuRand(use_gpu_rand);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetGPUAwareMPI
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SetGpuAwareMPI( HYPRE_Int use_gpu_aware_mpi )
{
   return hypre_SetGpuAwareMPI(use_gpu_aware_mpi);
}
