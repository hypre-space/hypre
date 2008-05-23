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
 * HYPRE_FlexGMRES interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESDestroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
int 
HYPRE_ParCSRFlexGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_FlexGMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_FlexGMRESSetup( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_FlexGMRESSetup( solver,
                             A,
                             b,
                             x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_FlexGMRESSolve( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_FlexGMRESSolve( solver,
                             A,
                             b,
                             x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetKDim, HYPRE_FlexGMRESGetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESSetKDim( HYPRE_Solver solver,
                          int             k_dim    )
{
   return( hypre_FlexGMRESSetKDim( (void *) solver, k_dim ) );
}

int
HYPRE_FlexGMRESGetKDim( HYPRE_Solver solver,
                          int           * k_dim    )
{
   return( hypre_FlexGMRESGetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetTol, HYPRE_FlexGMRESGetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( hypre_FlexGMRESSetTol( (void *) solver, tol ) );
}

int
HYPRE_FlexGMRESGetTol( HYPRE_Solver solver,
                         double           * tol    )
{
   return( hypre_FlexGMRESGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetAbsoluteTol, HYPRE_FlexGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESSetAbsoluteTol( HYPRE_Solver solver,
                         double             a_tol    )
{
   return( hypre_FlexGMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

int
HYPRE_FlexGMRESGetAbsoluteTol( HYPRE_Solver solver,
                         double           * a_tol    )
{
   return( hypre_FlexGMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetConvergenceFactorTol, HYPRE_FlexGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESSetConvergenceFactorTol( HYPRE_Solver solver,
                         double             cf_tol    )
{
   return( hypre_FlexGMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

int
HYPRE_FlexGMRESGetConvergenceFactorTol( HYPRE_Solver solver,
                         double           * cf_tol    )
{
   return( hypre_FlexGMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetMinIter, HYPRE_FlexGMRESGetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( hypre_FlexGMRESSetMinIter( (void *) solver, min_iter ) );
}

int
HYPRE_FlexGMRESGetMinIter( HYPRE_Solver solver,
                             int        * min_iter )
{
   return( hypre_FlexGMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetMaxIter, HYPRE_FlexGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( hypre_FlexGMRESSetMaxIter( (void *) solver, max_iter ) );
}

int
HYPRE_FlexGMRESGetMaxIter( HYPRE_Solver solver,
                             int        * max_iter )
{
   return( hypre_FlexGMRESGetMaxIter( (void *) solver, max_iter ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESSetPrecond( HYPRE_Solver          solver,
                             HYPRE_PtrToSolverFcn  precond,
                             HYPRE_PtrToSolverFcn  precond_setup,
                             HYPRE_Solver          precond_solver )
{
   return( hypre_FlexGMRESSetPrecond( (void *) solver,
                                  precond, precond_setup,
                                  (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( hypre_FlexGMRESGetPrecond( (void *)     solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetPrintLevel, HYPRE_FlexGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESSetPrintLevel( HYPRE_Solver solver,
                        int          level )
{
   return( hypre_FlexGMRESSetPrintLevel( (void *) solver, level ) );
}

int
HYPRE_FlexGMRESGetPrintLevel( HYPRE_Solver solver,
                        int        * level )
{
   return( hypre_FlexGMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetLogging, HYPRE_FlexGMRESGetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESSetLogging( HYPRE_Solver solver,
                     int          level )
{
   return( hypre_FlexGMRESSetLogging( (void *) solver, level ) );
}

int
HYPRE_FlexGMRESGetLogging( HYPRE_Solver solver,
                     int        * level )
{
   return( hypre_FlexGMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( hypre_FlexGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESGetConverged
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESGetConverged( HYPRE_Solver  solver,
                         int                *converged )
{
   return( hypre_FlexGMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_FlexGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( hypre_FlexGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

int HYPRE_FlexGMRESGetResidual( HYPRE_Solver solver, void **residual )
{
   /* returns a pointer to the residual vector */
   return hypre_FlexGMRESGetResidual( (void *) solver, residual );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetModifyPC
 *--------------------------------------------------------------------------*/
 

int HYPRE_FlexGMRESSetModifyPC( HYPRE_Solver  solver,
                                int (*modify_pc)(HYPRE_Solver, int, double) )

{
   return hypre_FlexGMRESSetModifyPC( (void *) solver, modify_pc);
   
}




