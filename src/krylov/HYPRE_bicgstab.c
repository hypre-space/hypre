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
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_BiCGSTAB interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABCreate does not exist.  Call the appropriate function which
 * also specifies the vector type, e.g. HYPRE_ParCSRBiCGSTABCreate
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_BiCGSTABDestroy( HYPRE_Solver solver )
{
   return( hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_BiCGSTABSetup( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_BiCGSTABSetup( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_BiCGSTABSolve( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_BiCGSTABSolve( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( hypre_BiCGSTABSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetConvergenceFactorTol( HYPRE_Solver solver,
                         double             cf_tol    )
{
   return( hypre_BiCGSTABSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( hypre_BiCGSTABSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( hypre_BiCGSTABSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetStopCrit( HYPRE_Solver solver,
                              int          stop_crit )
{
   return( hypre_BiCGSTABSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetPrecond( HYPRE_Solver         solver,
                                HYPRE_PtrToSolverFcn precond,
                                HYPRE_PtrToSolverFcn precond_setup,
                                HYPRE_Solver         precond_solver )
{
   return( hypre_BiCGSTABSetPrecond( (void *) solver,
                                     precond, precond_setup,
                                     (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( hypre_BiCGSTABGetPrecond( (void *)     solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetLogging( HYPRE_Solver solver,
                             int logging)
{
   return( hypre_BiCGSTABSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetPrintLevel( HYPRE_Solver solver,
                             int print_level)
{
   return( hypre_BiCGSTABSetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( hypre_BiCGSTABGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( hypre_BiCGSTABGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABGetResidual( HYPRE_Solver  solver,
                            void             **residual  )
{
   return( hypre_BiCGSTABGetResidual( (void *) solver, residual ) );
}
