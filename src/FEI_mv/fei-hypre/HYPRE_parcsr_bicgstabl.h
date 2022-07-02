/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRBiCGSTABL interface
 *
 *****************************************************************************/

#ifndef __CGSTABL__
#define __CGSTABL__

#ifdef __cplusplus
extern "C" {
#endif

extern int HYPRE_ParCSRBiCGSTABLCreate( MPI_Comm comm, HYPRE_Solver *solver );

extern int HYPRE_ParCSRBiCGSTABLDestroy( HYPRE_Solver solver );

extern int HYPRE_ParCSRBiCGSTABLSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRBiCGSTABLSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRBiCGSTABLSetTol( HYPRE_Solver solver, double tol );

extern int HYPRE_ParCSRBiCGSTABLSetSize( HYPRE_Solver solver, int size );

extern int HYPRE_ParCSRBiCGSTABLSetMaxIter( HYPRE_Solver solver, int max_iter );

extern int HYPRE_ParCSRBiCGSTABLSetStopCrit( HYPRE_Solver solver, int stop_crit );

extern int HYPRE_ParCSRBiCGSTABLSetPrecond( HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void               *precond_data );

extern int HYPRE_ParCSRBiCGSTABLSetLogging( HYPRE_Solver solver, int logging);

extern int HYPRE_ParCSRBiCGSTABLGetNumIterations(HYPRE_Solver solver,
                                                 int *num_iterations);

extern int HYPRE_ParCSRBiCGSTABLGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                       double *norm );

#ifdef __cplusplus
}
#endif
#endif

