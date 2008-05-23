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
 * HYPRE_SStructFlexGMRES interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESCreate( MPI_Comm             comm,
                          HYPRE_SStructSolver *solver )
{
   hypre_FlexGMRESFunctions * fgmres_functions =
      hypre_FlexGMRESFunctionsCreate(
         hypre_CAlloc, hypre_SStructKrylovFree, hypre_SStructKrylovCommInfo,
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovCreateVectorArray,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovClearVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (HYPRE_SStructSolver) hypre_FlexGMRESCreate( fgmres_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructFlexGMRESDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_FlexGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructFlexGMRESSetup( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( HYPRE_FlexGMRESSetup( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructFlexGMRESSolve( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( HYPRE_FlexGMRESSolve( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESSetKDim( HYPRE_SStructSolver solver,
                           int                 k_dim )
{
   return( HYPRE_FlexGMRESSetKDim( (HYPRE_Solver) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESSetTol( HYPRE_SStructSolver solver,
                          double              tol )
{
   return( HYPRE_FlexGMRESSetTol( (HYPRE_Solver) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESSetAbsoluteTol( HYPRE_SStructSolver solver,
                          double              tol )
{
   return( HYPRE_FlexGMRESSetAbsoluteTol( (HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESSetMinIter( HYPRE_SStructSolver solver,
                              int                 min_iter )
{
   return( HYPRE_FlexGMRESSetMinIter( (HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESSetMaxIter( HYPRE_SStructSolver solver,
                              int                 max_iter )
{
   return( HYPRE_FlexGMRESSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESSetStopCrit( HYPRE_SStructSolver solver,
                               int                 stop_crit )
{
   return( HYPRE_FlexGMRESSetStopCrit( (HYPRE_Solver) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESSetPrecond( HYPRE_SStructSolver          solver,
                              HYPRE_PtrToSStructSolverFcn  precond,
                              HYPRE_PtrToSStructSolverFcn  precond_setup,
                              void *          precond_data )
{
   return( HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                  (HYPRE_PtrToSolverFcn) precond,
                                  (HYPRE_PtrToSolverFcn) precond_setup,
                                  (HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESSetLogging( HYPRE_SStructSolver solver,
                              int                 logging )
{
   return( HYPRE_FlexGMRESSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESSetPrintLevel( HYPRE_SStructSolver solver,
                              int                 level )
{
   return( HYPRE_FlexGMRESSetPrintLevel( (HYPRE_Solver) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESGetNumIterations( HYPRE_SStructSolver  solver,
                                    int                 *num_iterations )
{
   return( HYPRE_FlexGMRESGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                double              *norm )
{
   return( HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFlexGMRESGetResidual( HYPRE_SStructSolver  solver,
                                void              **residual )
{
   return( HYPRE_FlexGMRESGetResidual( (HYPRE_Solver) solver, residual ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetModifyPC
 *--------------------------------------------------------------------------*/
 

int HYPRE_SStructFlexGMRESSetModifyPC( HYPRE_SStructSolver  solver,
                                      HYPRE_PtrToModifyPCFcn modify_pc)

{
   return ( HYPRE_FlexGMRESSetModifyPC( (HYPRE_Solver) solver,  (HYPRE_PtrToModifyPCFcn) modify_pc));
   
}

