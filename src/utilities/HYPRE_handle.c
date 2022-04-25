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
 * HYPRE_SetSpMVUseCusparse
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SetSpMVUseCusparse( HYPRE_Int use_cusparse )
{
   return hypre_SetSpMVUseCusparse(use_cusparse);
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

