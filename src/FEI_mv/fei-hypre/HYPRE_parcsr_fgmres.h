/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





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

int HYPRE_ParCSRFGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver );

int HYPRE_ParCSRFGMRESDestroy( HYPRE_Solver solver );

int HYPRE_ParCSRFGMRESSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x );

int HYPRE_ParCSRFGMRESSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x );

int HYPRE_ParCSRFGMRESSetKDim(HYPRE_Solver solver, int kdim);

int HYPRE_ParCSRFGMRESSetTol(HYPRE_Solver solver, double tol);

int HYPRE_ParCSRFGMRESSetMaxIter(HYPRE_Solver solver, int max_iter);

int HYPRE_ParCSRFGMRESSetStopCrit(HYPRE_Solver solver, int stop_crit);

int HYPRE_ParCSRFGMRESSetPrecond(HYPRE_Solver  solver,
          int (*precond)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void *precond_data);

int HYPRE_ParCSRFGMRESSetLogging(HYPRE_Solver solver, int logging);

int HYPRE_ParCSRFGMRESGetNumIterations(HYPRE_Solver solver,
                                              int *num_iterations);

int HYPRE_ParCSRFGMRESGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                          double *norm );

int HYPRE_ParCSRFGMRESUpdatePrecondTolerance(HYPRE_Solver  solver,
                             int (*set_tolerance)(HYPRE_Solver sol, double));

#ifdef __cplusplus
}
#endif

#endif

