/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* Mixed precision function protos */
/* parcsr_mv_mp.c */

#ifdef HYPRE_MIXED_PRECISION
HYPRE_Int
hypre_ParVectorCopy_mp( hypre_ParVector *x,
                        hypre_ParVector *y );

HYPRE_Int
hypre_ParVectorAxpy_mp( hypre_long_double    alpha,
                        hypre_ParVector *x,
                        hypre_ParVector *y );

HYPRE_Int
hypre_ParVectorConvert_mp ( hypre_ParVector *v,
                            HYPRE_Precision new_precision );

HYPRE_Int
hypre_ParCSRMatrixConvert_mp ( hypre_ParCSRMatrix *A,
                               HYPRE_Precision new_precision );

hypre_ParCSRMatrix*
hypre_ParCSRMatrixClone_mp(hypre_ParCSRMatrix   *A, HYPRE_Precision new_precision);

HYPRE_Int
hypre_ParCSRMatrixCopy_mp( hypre_ParCSRMatrix *A,
                           hypre_ParCSRMatrix *B );

#endif
