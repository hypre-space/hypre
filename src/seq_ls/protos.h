/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* direct_solver.c */
void*     hypre_DirectSolverCreate ( HYPRE_Int option );
HYPRE_Int hypre_DirectSolverDestroy ( void *vdata );
HYPRE_Int hypre_DirectSolverSetup ( void *direct_vdata, hypre_DenseMatrix *A, hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverSolve ( void *direct_vdata, hypre_DenseMatrix *A, hypre_Vector *f, hypre_Vector *u );

/* direct_cusolver.c */
#if defined (HYPRE_USING_CUSOLVER)
void*     hypre_DirectSolverCreateCuSolver ( void );
HYPRE_Int hypre_DirectSolverDestroyCuSolver ( void *direct_vdata );
HYPRE_Int hypre_DirectSolverSetupCuSolver ( void *direct_vdata, hypre_DenseMatrix *A, hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverSolveCuSolver ( void *direct_vdata, hypre_DenseMatrix *A, hypre_Vector *f, hypre_Vector *u );
#endif

/* direct_magma.c */
#if defined (HYPRE_USING_MAGMA)
void*     hypre_DirectSolverCreateMagma ( void );
HYPRE_Int hypre_DirectSolverDestroyMagma ( void *direct_vdata );
HYPRE_Int hypre_DirectSolverSetupMagma ( void *direct_vdata, hypre_DenseMatrix *A, hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_DirectSolverSolveMagma ( void *direct_vdata, hypre_DenseMatrix *A, hypre_Vector *f, hypre_Vector *u );
#endif
