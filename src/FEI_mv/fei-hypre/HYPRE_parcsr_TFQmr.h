/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRTFQmr interface
 *
 *****************************************************************************/

#ifndef __TFQMR__
#define __TFQMR__

#ifdef __cplusplus
extern "C" {
#endif

extern int HYPRE_ParCSRTFQmrCreate( MPI_Comm comm, HYPRE_Solver *solver );

extern int HYPRE_ParCSRTFQmrDestroy( HYPRE_Solver solver );

extern int HYPRE_ParCSRTFQmrSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRTFQmrSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRTFQmrSetTol( HYPRE_Solver solver, double tol );

extern int HYPRE_ParCSRTFQmrSetMaxIter( HYPRE_Solver solver, int max_iter );

extern int HYPRE_ParCSRTFQmrSetStopCrit( HYPRE_Solver solver, int stop_crit );

extern int HYPRE_ParCSRTFQmrSetPrecond( HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void               *precond_data );

extern int HYPRE_ParCSRTFQmrSetLogging( HYPRE_Solver solver, int logging);

extern int HYPRE_ParCSRTFQmrGetNumIterations(HYPRE_Solver solver,
                                                 int *num_iterations);

extern int HYPRE_ParCSRTFQmrGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                       double *norm );

#ifdef __cplusplus
}
#endif
#endif

