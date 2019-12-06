/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * IJMatrix_ParCSR interface
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"

#include "../HYPRE.h"

HYPRE_Int
hypre_IJMatrixSetValuesParCSRDevice_v1( hypre_IJMatrix       *matrix,
                                        HYPRE_Int             nelms,
                                        const HYPRE_BigInt   *rows,
                                        const HYPRE_BigInt   *cols,
                                        const HYPRE_Complex  *values )
{
}

__global__ void
hypreCUDAKernel_IJSetValuesParCSR(HYPRE_Int 

HYPRE_Int
hypre_IJMatrixSetValuesParCSRDevice( hypre_IJMatrix       *matrix,
                                     HYPRE_Int             nrows,
                                     HYPRE_Int            *ncols,
                                     const HYPRE_BigInt   *rows,
                                     const HYPRE_Int      *row_indexes,
                                     const HYPRE_BigInt   *cols,
                                     const HYPRE_Complex  *values )
{
}

