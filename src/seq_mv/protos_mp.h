/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* Mixed precision function protos */

/* seq_mv_mp.c */

#ifdef HYPRE_MIXED_PRECISION
HYPRE_Int
hypre_SeqVectorCopy_mp( hypre_Vector_mp *x,
                        hypre_Vector_mp *y );

HYPRE_Int
hypre_SeqVectorAxpy_mp( hypre_double alpha,
                        hypre_Vector_mp *x,
                        hypre_Vector_mp *y     );

#endif
