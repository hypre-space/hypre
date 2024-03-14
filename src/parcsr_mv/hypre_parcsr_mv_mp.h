/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

/* Mixed precision function protos */
/* hypre_parcsr_mv_mp.h */

#ifdef HYPRE_MIXED_PRECISION
HYPRE_Int
hypre_ParVectorCopy_mp( hypre_ParVector *x,
                     hypre_ParVector *y );

HYPRE_Int
hypre_ParVectorAxpy_mp( HYPRE_Complex    alpha,
                     hypre_ParVector *x,
                     hypre_ParVector *y );

HYPRE_Int
hypre_ParVectorConvert_mp (hypre_ParVector *v,
                    HYPRE_Precision new_precision );

HYPRE_Int
hypre_ParCSRMatrixConvert_mp (hypre_ParCSRMatrix *A,
                       HYPRE_Precision new_precision );

#endif
