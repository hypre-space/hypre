/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
 * HYPRE_SetSpGemmUseCusparse
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SetSpGemmUseCusparse( HYPRE_Int use_cusparse )
{
   return hypre_SetSpGemmUseCusparse(use_cusparse);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetUseGpuRand
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SetUseGpuRand( HYPRE_Int use_curand )
{
   return hypre_SetUseGpuRand(use_curand);
}

