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
hypre_StructVectorCopy_mp( hypre_StructVector *x,
                           hypre_StructVector *y );
#endif

