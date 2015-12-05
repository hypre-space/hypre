/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_GMRES interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_GMRESDestroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
int 
HYPRE_ParCSRGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_GMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_GMRESSetup( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_GMRESSetup( solver,
                             A,
                             b,
                             x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_GMRESSolve( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_GMRESSolve( solver,
                             A,
                             b,
                             x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetKDim, HYPRE_GMRESGetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESSetKDim( HYPRE_Solver solver,
                          int             k_dim    )
{
   return( hypre_GMRESSetKDim( (void *) solver, k_dim ) );
}

int
HYPRE_GMRESGetKDim( HYPRE_Solver solver,
                          int           * k_dim    )
{
   return( hypre_GMRESGetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetTol, HYPRE_GMRESGetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( hypre_GMRESSetTol( (void *) solver, tol ) );
}

int
HYPRE_GMRESGetTol( HYPRE_Solver solver,
                         double           * tol    )
{
   return( hypre_GMRESGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetConvergenceFactorTol, HYPRE_GMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESSetConvergenceFactorTol( HYPRE_Solver solver,
                         double             cf_tol    )
{
   return( hypre_GMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

int
HYPRE_GMRESGetConvergenceFactorTol( HYPRE_Solver solver,
                         double           * cf_tol    )
{
   return( hypre_GMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetMinIter, HYPRE_GMRESGetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( hypre_GMRESSetMinIter( (void *) solver, min_iter ) );
}

int
HYPRE_GMRESGetMinIter( HYPRE_Solver solver,
                             int        * min_iter )
{
   return( hypre_GMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetMaxIter, HYPRE_GMRESGetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( hypre_GMRESSetMaxIter( (void *) solver, max_iter ) );
}

int
HYPRE_GMRESGetMaxIter( HYPRE_Solver solver,
                             int        * max_iter )
{
   return( hypre_GMRESGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetStopCrit, HYPRE_GMRESGetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESSetStopCrit( HYPRE_Solver solver,
                              int          stop_crit )
{
   return( hypre_GMRESSetStopCrit( (void *) solver, stop_crit ) );
}

int
HYPRE_GMRESGetStopCrit( HYPRE_Solver solver,
                              int        * stop_crit )
{
   return( hypre_GMRESGetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetRelChange, HYPRE_GMRESGetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESSetRelChange( HYPRE_Solver solver,
                         int                rel_change )
{
   return( hypre_GMRESSetRelChange( (void *) solver, rel_change ) );
}

int
HYPRE_GMRESGetRelChange( HYPRE_Solver solver,
                         int              * rel_change )
{
   return( hypre_GMRESGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESSetPrecond( HYPRE_Solver          solver,
                             HYPRE_PtrToSolverFcn  precond,
                             HYPRE_PtrToSolverFcn  precond_setup,
                             HYPRE_Solver          precond_solver )
{
   return( hypre_GMRESSetPrecond( (void *) solver,
                                  precond, precond_setup,
                                  (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( hypre_GMRESGetPrecond( (void *)     solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetPrintLevel, HYPRE_GMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESSetPrintLevel( HYPRE_Solver solver,
                        int          level )
{
   return( hypre_GMRESSetPrintLevel( (void *) solver, level ) );
}

int
HYPRE_GMRESGetPrintLevel( HYPRE_Solver solver,
                        int        * level )
{
   return( hypre_GMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetLogging, HYPRE_GMRESGetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESSetLogging( HYPRE_Solver solver,
                     int          level )
{
   return( hypre_GMRESSetLogging( (void *) solver, level ) );
}

int
HYPRE_GMRESGetLogging( HYPRE_Solver solver,
                     int        * level )
{
   return( hypre_GMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( hypre_GMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESGetConverged
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESGetConverged( HYPRE_Solver  solver,
                         int                *converged )
{
   return( hypre_GMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_GMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( hypre_GMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESGetResidual
 *--------------------------------------------------------------------------*/

int HYPRE_GMRESGetResidual( HYPRE_Solver solver, void **residual )
{
   /* returns a pointer to the residual vector */
   return hypre_GMRESGetResidual( (void *) solver, residual );
}

