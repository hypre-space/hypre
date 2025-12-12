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
hypre_SeqVectorCopy_mp( hypre_Vector *x,
                        hypre_Vector *y );

HYPRE_Int
hypre_SeqVectorAxpy_mp( hypre_double alpha,
                        hypre_Vector *x,
                        hypre_Vector *y     );

HYPRE_Int
hypre_CSRMatrixConvert_mp ( hypre_CSRMatrix *A,
                            HYPRE_Precision new_precision);

HYPRE_Int
hypre_SeqVectorConvert_mp ( hypre_Vector *v,
                            HYPRE_Precision new_precision);

HYPRE_Int
hypre_CSRMatrixCopy_mp( hypre_CSRMatrix *A, hypre_CSRMatrix *B);

hypre_CSRMatrix*
hypre_CSRMatrixClone_mp( hypre_CSRMatrix *A, HYPRE_Precision new_precision );

HYPRE_Int
hypre_RealArrayCopyHost_mp(HYPRE_Precision precision_x, void *x, HYPRE_Precision precision_y, void *y, HYPRE_Int n);

HYPRE_Int
hypre_RealArrayCopy_mp(HYPRE_Precision precision_x, void *x, HYPRE_MemoryLocation location_x, HYPRE_Precision precision_y, void *y, HYPRE_MemoryLocation location_y, HYPRE_Int n);

void *
hypre_RealArrayClone_mp(HYPRE_Precision precision_x, void *x, HYPRE_MemoryLocation location_x, HYPRE_Precision new_precision, HYPRE_MemoryLocation new_location, HYPRE_Int n);

HYPRE_Int
hypre_RealArrayAxpynHost_mp(HYPRE_Precision precision_x, hypre_long_double alpha, void *x, HYPRE_Precision precision_y, void *y, HYPRE_Int n);

HYPRE_Int
hypre_RealArrayAxpyn_mp(HYPRE_Precision precision_x, void *x, HYPRE_Precision precision_y, void *y, HYPRE_MemoryLocation location, HYPRE_Int n, hypre_long_double alpha);

#endif
