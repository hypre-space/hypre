/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* Mixed precision function protos */
/* struct_mv_mp.c */

#if defined(HYPRE_MIXED_PRECISION)
HYPRE_Int
hypre_StructVectorCopy_mp( hypre_StructVector_mp *x,
                     hypre_StructVector_mp *y );
HYPRE_Int
hypre_StructVectorConvert_mp( hypre_StructVector_mp *x,
                     HYPRE_Precision new_precision);
#endif

