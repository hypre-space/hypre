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





#include "headers.h"

/*==========================================================================*/
/** Creates a new FlexGMRES solver object.

{\bf Input files:}
headers.h

@return Error code.

@param comm [IN]
  MPI communicator
@param solver [OUT]
  solver structure

@see HYPRE_StructFlexGMRESDestroy */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructFlexGMRESCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   hypre_FlexGMRESFunctions * fgmres_functions =
      hypre_FlexGMRESFunctionsCreate(
         hypre_CAlloc, hypre_StructKrylovFree,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovCreateVectorArray,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );

   *solver = ( (HYPRE_StructSolver) hypre_FlexGMRESCreate( fgmres_functions ) );

   return 0;
}

/*==========================================================================*/
/*==========================================================================*/
/** Destroys a FlexGMRES solver object.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure

@see HYPRE_StructFlexGMRESCreate */
/*--------------------------------------------------------------------------*/

int 
HYPRE_StructFlexGMRESDestroy( HYPRE_StructSolver solver )
{
   return( hypre_FlexGMRESDestroy( (void *) solver ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Precomputes tasks necessary for doing the solve.  This routine
ensures that the setup for the preconditioner is also called.

NOTE: This is supposed to be an optional call, but currently is required.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param A [IN]
  coefficient matrix
@param b [IN]
  right-hand-side vector
@param x [IN]
  unknown vector

@see HYPRE_StructFlexGMRESSolve */
/*--------------------------------------------------------------------------*/

int 
HYPRE_StructFlexGMRESSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( HYPRE_FlexGMRESSetup( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Performs the FlexGMRES linear solve.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param A [IN]
  coefficient matrix
@param b [IN]
  right-hand-side vector
@param x [IN]
  unknown vector

@see HYPRE_StructFlexGMRESSetup */
/*--------------------------------------------------------------------------*/

int 
HYPRE_StructFlexGMRESSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( HYPRE_FlexGMRESSolve( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set the stopping tolerance.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param tol [IN]
  FlexGMRES solver tolerance

@see HYPRE_StructFlexGMRESSolve, HYPRE_StructFlexGMRESSetup   */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructFlexGMRESSetTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( HYPRE_FlexGMRESSetTol( (HYPRE_Solver) solver, tol ) );
}
/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set the absolute stopping tolerance.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param tol [IN]
  FlexGMRES solver tolerance

@see HYPRE_StructFlexGMRESSolve, HYPRE_StructFlexGMRESSetup   */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructFlexGMRESSetAbsoluteTol( HYPRE_StructSolver solver,
                       double             atol    )
{
   return( HYPRE_FlexGMRESSetAbsoluteTol( (HYPRE_Solver) solver, atol ) );
}
/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set the maximum number of iterations.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param max_iter [IN]
  FlexGMRES solver maximum number of iterations

@see HYPRE_StructFlexGMRESSolve, HYPRE_StructFlexGMRESSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructFlexGMRESSetMaxIter( HYPRE_StructSolver solver,
                           int                max_iter )
{
   return( HYPRE_FlexGMRESSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Sets the dimension of the Krylov subspace.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param k_dim [IN]
  FlexGMRES dimension of the Krylov subspace

@see HYPRE_StructFlexGMRESSolve, HYPRE_StructFlexGMRESSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructFlexGMRESSetKDim( HYPRE_StructSolver solver,
                           int                k_dim )
{
   return( HYPRE_FlexGMRESSetKDim( (HYPRE_Solver) solver, k_dim ) );
}
/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Sets the precondioner to use in FlexGMRES.  The Default is no
preconditioner, i.e. the solver is just conjugate gradients (CG).

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param precond [IN]
  pointer to the preconditioner solve function
@param precond_setup [IN]
  pointer to the preconditioner setup function
@param precond_solver [IN/OUT]
  preconditioner solver structure

@see HYPRE_StructFlexGMRESSolve, HYPRE_StructFlexGMRESSetup*/
/*--------------------------------------------------------------------------*/

int
HYPRE_StructFlexGMRESSetPrecond( HYPRE_StructSolver         solver,
                           HYPRE_PtrToStructSolverFcn precond,
                           HYPRE_PtrToStructSolverFcn precond_setup,
                           HYPRE_StructSolver         precond_solver )
{
   return( HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                (HYPRE_PtrToSolverFcn) precond,
                                (HYPRE_PtrToSolverFcn) precond_setup,
                                (HYPRE_Solver) precond_solver ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set the type of logging to do.  Currently, if parameter
logging is set to 0, no logging is done.  If set to 1, the norms and
relative norms for each iteration are saved.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param logging [IN]
  integer indicating what type of logging to do

@see HYPRE_StructFlexGMRESSolve, HYPRE_StructFlexGMRESSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructFlexGMRESSetLogging( HYPRE_StructSolver solver,
                           int                logging )
{
   return( HYPRE_FlexGMRESSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Currently, if parameter print_level is set to 0, no printing 
is allowed.  If set to 1, printing takes place.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param logging [IN]
  integer allowing printing to take place

@see HYPRE_StructFlexGMRESSolve, HYPRE_StructFlexGMRESSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructFlexGMRESSetPrintLevel( HYPRE_StructSolver solver,
                           int                print_level )
{
   return( HYPRE_FlexGMRESSetPrintLevel( (HYPRE_Solver) solver, print_level ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Gets the number of iterations done in the solve.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN]
  solver structure
@param num_iterations [OUT]
  number of iterations

@see HYPRE_StructFlexGMRESSolve, HYPRE_StructFlexGMRESSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructFlexGMRESGetNumIterations( HYPRE_StructSolver  solver,
                                 int                *num_iterations )
{
   return( HYPRE_FlexGMRESGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Gets the final relative residual norm for the solve.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN]
  solver structure
@param norm [OUT]
  final relative residual norm

@see HYPRE_StructFlexGMRESSolve, HYPRE_StructFlexGMRESSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
   return( HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetModifyPC
 *--------------------------------------------------------------------------*/
 

int HYPRE_StructFlexGMRESSetModifyPC( HYPRE_StructSolver  solver,
                                      HYPRE_PtrToModifyPCFcn modify_pc)

{
   return ( HYPRE_FlexGMRESSetModifyPC( (HYPRE_Solver) solver,  (HYPRE_PtrToModifyPCFcn) modify_pc));
   
}

