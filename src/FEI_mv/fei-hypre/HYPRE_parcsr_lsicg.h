/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRLSICG interface
 *
 *****************************************************************************/

#ifndef __LSICG__
#define __LSICG__

#ifdef __cplusplus
extern "C" {
#endif

extern int HYPRE_ParCSRLSICGCreate(MPI_Comm comm, HYPRE_Solver *solver);

extern int HYPRE_ParCSRLSICGDestroy(HYPRE_Solver solver);

extern int HYPRE_ParCSRLSICGSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRLSICGSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRLSICGSetTol(HYPRE_Solver solver, double tol);

extern int HYPRE_ParCSRLSICGSetMaxIter(HYPRE_Solver solver, int max_iter);

extern int HYPRE_ParCSRLSICGSetStopCrit(HYPRE_Solver solver, int stop_crit);

extern int HYPRE_ParCSRLSICGSetPrecond(HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void *precond_data );

extern int HYPRE_ParCSRLSICGSetLogging(HYPRE_Solver solver, int logging);

extern int HYPRE_ParCSRLSICGGetNumIterations(HYPRE_Solver solver,
                                             int *num_iterations);

extern int HYPRE_ParCSRLSICGGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                         double *norm );

#ifdef __cplusplus
}
#endif
#endif

