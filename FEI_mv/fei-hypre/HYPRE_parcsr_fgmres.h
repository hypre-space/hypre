/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_ParCSRFGMRES interface
 *
 *****************************************************************************/

#ifndef __FGMRESH__
#define __FGMRESH__

#ifdef __cplusplus
extern "C" {
#endif

extern int HYPRE_ParCSRFGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver );

extern int HYPRE_ParCSRFGMRESDestroy( HYPRE_Solver solver );

extern int HYPRE_ParCSRFGMRESSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRFGMRESSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x );

extern int HYPRE_ParCSRFGMRESSetKDim(HYPRE_Solver solver, int kdim);

extern int HYPRE_ParCSRFGMRESSetTol(HYPRE_Solver solver, double tol);

extern int HYPRE_ParCSRFGMRESSetMaxIter(HYPRE_Solver solver, int max_iter);

extern int HYPRE_ParCSRFGMRESSetStopCrit(HYPRE_Solver solver, int stop_crit);

extern int HYPRE_ParCSRFGMRESSetPrecond(HYPRE_Solver  solver,
          int (*precond)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void *precond_data);

extern int HYPRE_ParCSRFGMRESSetLogging(HYPRE_Solver solver, int logging);

extern int HYPRE_ParCSRFGMRESGetNumIterations(HYPRE_Solver solver,
                                              int *num_iterations);

extern int HYPRE_ParCSRFGMRESGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                          double *norm );

#ifdef __cplusplus
}
#endif
#endif

