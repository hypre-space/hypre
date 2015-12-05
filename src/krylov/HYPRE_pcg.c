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
 * $Revision: 2.13 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_PCG interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_PCGCreate does not exist.  Call the appropriate function which
 * also specifies the vector type, e.g. HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * HYPRE_PCGDestroy
 *--------------------------------------------------------------------------*/

/*
int 
HYPRE_PCGDestroy( HYPRE_Solver solver )*/
/* >>> This is something we can't do without knowing the vector_type.
   We can't save it in and pull it out of solver because that isn't
   really a known struct. */
/*
{
   if ( vector_type=="ParCSR" ) {
      return HYPRE_ParCSRPCGDestroy( HYPRE_Solver solver );
   }
   else {
      return 0;
   }
}*/

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_PCGSetup( HYPRE_Solver solver,
                HYPRE_Matrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{
   return( hypre_PCGSetup( solver,
                           A,
                           b,
                           x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_PCGSolve( HYPRE_Solver solver,
                HYPRE_Matrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{
   return( hypre_PCGSolve( (void *) solver,
                           (void *) A,
                           (void *) b,
                           (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetTol, HYPRE_PCGGetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGSetTol( HYPRE_Solver solver,
                 double             tol    )
{
   return( hypre_PCGSetTol( (void *) solver, tol ) );
}

int
HYPRE_PCGGetTol( HYPRE_Solver solver,
                 double           * tol    )
{
   return( hypre_PCGGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetAbsoluteTolFactor, HYPRE_PCGGetAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGSetAbsoluteTolFactor( HYPRE_Solver solver,
                               double abstolf )
{
   return( hypre_PCGSetAbsoluteTolFactor( (void *) solver, abstolf ) );
}

int
HYPRE_PCGGetAbsoluteTolFactor( HYPRE_Solver solver,
                               double * abstolf )
{
   return( hypre_PCGGetAbsoluteTolFactor( (void *) solver, abstolf ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetConvergenceFactorTol, HYPRE_PCGGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGSetConvergenceFactorTol( HYPRE_Solver solver,
                                  double cf_tol )
{
   return hypre_PCGSetConvergenceFactorTol( (void *) solver,
                                            cf_tol   );
}

int
HYPRE_PCGGetConvergenceFactorTol( HYPRE_Solver solver,
                                  double * cf_tol )
{
   return hypre_PCGGetConvergenceFactorTol( (void *) solver,
                                            cf_tol   );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetMaxIter, HYPRE_PCGGetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGSetMaxIter( HYPRE_Solver solver,
                     int                max_iter )
{
   return( hypre_PCGSetMaxIter( (void *) solver, max_iter ) );
}

int
HYPRE_PCGGetMaxIter( HYPRE_Solver solver,
                     int              * max_iter )
{
   return( hypre_PCGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetStopCrit, HYPRE_PCGGetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGSetStopCrit( HYPRE_Solver solver,
                      int          stop_crit )
{
   return( hypre_PCGSetStopCrit( (void *) solver, stop_crit ) );
}

int
HYPRE_PCGGetStopCrit( HYPRE_Solver solver,
                      int        * stop_crit )
{
   return( hypre_PCGGetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetTwoNorm, HYPRE_PCGGetTwoNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGSetTwoNorm( HYPRE_Solver solver,
                     int                two_norm )
{
   return( hypre_PCGSetTwoNorm( (void *) solver, two_norm ) );
}

int
HYPRE_PCGGetTwoNorm( HYPRE_Solver solver,
                     int              * two_norm )
{
   return( hypre_PCGGetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetRelChange, HYPRE_PCGGetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGSetRelChange( HYPRE_Solver solver,
                       int                rel_change )
{
   return( hypre_PCGSetRelChange( (void *) solver, rel_change ) );
}

int
HYPRE_PCGGetRelChange( HYPRE_Solver solver,
                       int              * rel_change )
{
   return( hypre_PCGGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGSetPrecond( HYPRE_Solver         solver,
                     HYPRE_PtrToSolverFcn precond,
                     HYPRE_PtrToSolverFcn precond_setup,
                     HYPRE_Solver         precond_solver )
{
   return( hypre_PCGSetPrecond( (void *) solver,
                                precond, precond_setup,
                                (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGGetPrecond( HYPRE_Solver  solver,
                     HYPRE_Solver *precond_data_ptr )
{
   return( hypre_PCGGetPrecond( (void *)     solver,
                                (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetLogging, HYPRE_PCGGetLogging
 * SetLogging sets both the print and log level, for backwards compatibility.
 * Soon the SetPrintLevel call should be deleted.
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGSetLogging( HYPRE_Solver solver,
                     int          level )
{
   return ( hypre_PCGSetLogging( (void *) solver, level ) );
}

int
HYPRE_PCGGetLogging( HYPRE_Solver solver,
                     int        * level )
{
   return ( hypre_PCGGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetPrintLevel, HYPRE_PCGGetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGSetPrintLevel( HYPRE_Solver solver,
                        int          level )
{
   return( hypre_PCGSetPrintLevel( (void *) solver, level ) );
}

int
HYPRE_PCGGetPrintLevel( HYPRE_Solver solver,
                        int        * level )
{
   return( hypre_PCGGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGGetNumIterations( HYPRE_Solver  solver,
                           int                *num_iterations )
{
   return( hypre_PCGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGGetConverged
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGGetConverged( HYPRE_Solver  solver,
                       int                *converged )
{
   return( hypre_PCGGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_PCGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                       double             *norm   )
{
   return( hypre_PCGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGGetResidual
 *--------------------------------------------------------------------------*/

int HYPRE_PCGGetResidual( HYPRE_Solver solver, void **residual )
{
   /* returns a pointer to the residual vector */
   return hypre_PCGGetResidual( (void *) solver, residual );
}

