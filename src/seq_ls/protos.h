/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* direct_solver.c */
hypre_DirectSolverData* hypre_DirectSolverCreate( hypre_DirectSolverBackend backend,
                                                  hypre_DirectSolverMethod method,
                                                  hypre_MatrixType mat_type, HYPRE_Int size,
                                                  HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_DirectSolverDestroy( hypre_DirectSolverData *data );
HYPRE_Int hypre_DirectSolverInitialize( hypre_DirectSolverData *data );
HYPRE_Int hypre_DirectSolverSetup( hypre_DirectSolverData *data,
                                   void *vA, hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverSolve( hypre_DirectSolverData *data,
                                   void *vA, hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverInvert( hypre_DirectSolverData *data,
                                    void *vA, void **vAinv_ptr );

/* ubatched_direct_vendor.c */
#if defined (HYPRE_USING_CUDA) || defined (HYPRE_USING_HIP)
HYPRE_Int hypre_UBatchedDenseDirectVendorSetup( hypre_DirectSolverData *data,
                                                hypre_UBatchedDenseMatrix *A,
                                                hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_UBatchedDenseDirectVendorSolve( hypre_DirectSolverData *data,
                                                hypre_UBatchedDenseMatrix *A,
                                                hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_UBatchedDenseDirectVendorInvert( hypre_DirectSolverData *data,
                                                 hypre_UBatchedDenseMatrix *A,
                                                 hypre_UBatchedDenseMatrix **Ainv_ptr );
#endif

/* ubatched_direct_custom.c */
HYPRE_Int hypre_UBatchedDenseDirectCustomSetup( hypre_DirectSolverData *data,
                                                hypre_UBatchedDenseMatrix *A,
                                                hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_UBatchedDenseDirectCustomSolve( hypre_DirectSolverData *data,
                                                hypre_UBatchedDenseMatrix *A,
                                                hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_UBatchedDenseDirectCustomInvert( hypre_DirectSolverData *data,
                                                 hypre_UBatchedDenseMatrix *A,
                                                 hypre_UBatchedDenseMatrix **Ainv_ptr );

/* ubatched_direct_magma.c */
#if defined (HYPRE_USING_MAGMA)
void*     hypre_UBatchedDenseDirectMagmaCreate( void );
HYPRE_Int hypre_UBatchedDenseDirectMagmaDestroy( hypre_DirectSolverData *data );
HYPRE_Int hypre_UBatchedDenseDirectMagmaSetup( hypre_DirectSolverData *data,
                                               hypre_UBatchedDenseMatrix *A,
                                               hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_UBatchedDenseDirectMagmaSolve( void *vdata,
                                               hypre_UBatchedDenseMatrix *A,
                                               hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_UBatchedDenseDirectMagmaInvert( void *vdata,
                                                hypre_UBatchedDenseMatrix *A,
                                                hypre_UBatchedDenseMatrix **Ainv_ptr );
#endif
