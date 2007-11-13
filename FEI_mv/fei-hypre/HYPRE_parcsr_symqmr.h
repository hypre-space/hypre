/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/






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

