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
hypre_ParVectorAxpy_mp( hypre_double    alpha,
                        hypre_ParVector *x,
                        hypre_ParVector *y );

#endif
