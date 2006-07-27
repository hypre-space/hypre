/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



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

