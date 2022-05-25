/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRBiCGS interface
 *
 *****************************************************************************/

#ifndef __BICGS__
#define __BICGS__

#ifdef __cplusplus
extern "C" {
#endif

extern int HYPRE_ParCSRBiCGSCreate( MPI_Comm comm, HYPRE_Solver *solver );

extern int HYPRE_ParCSRBiCGSDestroy( HYPRE_Solver solver );

extern int HYPRE_ParCSRBiCGSSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRBiCGSSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRBiCGSSetTol( HYPRE_Solver solver, double tol );

extern int HYPRE_ParCSRBiCGSSetMaxIter( HYPRE_Solver solver, int max_iter );

extern int HYPRE_ParCSRBiCGSSetStopCrit( HYPRE_Solver solver, int stop_crit );

extern int HYPRE_ParCSRBiCGSSetPrecond( HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void                *precond_data );

extern int HYPRE_ParCSRBiCGSSetLogging( HYPRE_Solver solver, int logging);

extern int HYPRE_ParCSRBiCGSGetNumIterations(HYPRE_Solver solver,
                                             int *num_iterations);

extern int HYPRE_ParCSRBiCGSGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                         double *norm );

#ifdef __cplusplus
}
#endif
#endif

