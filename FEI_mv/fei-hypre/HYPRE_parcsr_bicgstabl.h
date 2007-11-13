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

