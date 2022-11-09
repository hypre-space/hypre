/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* direct_solver.c */
void* hypre_DirectSolverCreate( hypre_DirectSolverBackend backend, hypre_DirectSolverMethod method,
                                HYPRE_Int info_size, HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_DirectSolverDestroy ( void *vdata );
HYPRE_Int hypre_DirectSolverInitialize ( void* vdata );
HYPRE_Int hypre_DirectSolverSetup ( void *vdata, hypre_DenseMatrix *A,
                                    hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverSolve ( void *vdata, hypre_DenseMatrix *A,
                                    hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverInvert ( void *vdata, hypre_DenseMatrix *A,
                                     hypre_DenseMatrix *Ainv );

/* direct_vendor.c */
#if defined (HYPRE_USING_CUDA)
HYPRE_Int hypre_DirectSolverSetupVendor ( hypre_DirectSolverData *data, hypre_DenseMatrix *A,
                                          hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverSolveVendor ( hypre_DirectSolverData *data, hypre_DenseMatrix *A,
                                          hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverInvertVendor ( hypre_DirectSolverData *data, hypre_DenseMatrix *A,
                                           hypre_DenseMatrix *Ainv );
#endif

/* direct_custom.c */
HYPRE_Int hypre_DirectSolverSetupCustom ( hypre_DirectSolverData *data, hypre_DenseMatrix *A,
                                          hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverSolveCustom ( hypre_DirectSolverData *data, hypre_DenseMatrix *A,
                                          hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverInvertCustom ( hypre_DirectSolverData *data, hypre_DenseMatrix *A,
                                           hypre_DenseMatrix *Ainv );

/* direct_magma.c */
#if defined (HYPRE_USING_MAGMA)
void*     hypre_DirectSolverCreateMagma ( void );
HYPRE_Int hypre_DirectSolverDestroyMagma ( void *direct_vdata );
HYPRE_Int hypre_DirectSolverSetupMagma ( void *direct_vdata, hypre_DenseMatrix *A,
                                         hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverSolveMagma ( void *direct_vdata, hypre_DenseMatrix *A,
                                         hypre_Vector *f, hypre_Vector *u );
#endif
