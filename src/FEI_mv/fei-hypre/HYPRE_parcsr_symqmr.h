/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRSymQMR interface
 *
 *****************************************************************************/

#ifndef __SYMQMR__
#define __SYMQMR__

#ifdef __cplusplus
extern "C" {
#endif

extern int HYPRE_ParCSRSymQMRCreate( MPI_Comm comm, HYPRE_Solver *solver );

extern int HYPRE_ParCSRSymQMRDestroy( HYPRE_Solver solver );

extern int HYPRE_ParCSRSymQMRSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRSymQMRSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRSymQMRSetTol( HYPRE_Solver solver, double tol );

extern int HYPRE_ParCSRSymQMRSetMaxIter( HYPRE_Solver solver, int max_iter );

extern int HYPRE_ParCSRSymQMRSetStopCrit( HYPRE_Solver solver, int stop_crit );

extern int HYPRE_ParCSRSymQMRSetPrecond( HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void                *precond_data );

extern int HYPRE_ParCSRSymQMRSetLogging( HYPRE_Solver solver, int logging);

extern int HYPRE_ParCSRSymQMRGetNumIterations(HYPRE_Solver solver,
                                              int *num_iterations);

extern int HYPRE_ParCSRSymQMRGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                       double *norm );

#ifdef __cplusplus
}
#endif
#endif

