/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





#include "headers.h"

/*==========================================================================*/
/** Creates a new GMRES solver object.

{\bf Input files:}
headers.h

@return Error code.

@param comm [IN]
  MPI communicator
@param solver [OUT]
  solver structure

@see HYPRE_StructGMRESDestroy */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGMRESCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   hypre_GMRESFunctions * gmres_functions =
      hypre_GMRESFunctionsCreate(
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

   *solver = ( (HYPRE_StructSolver) hypre_GMRESCreate( gmres_functions ) );

   return 0;
}

/*==========================================================================*/
/*==========================================================================*/
/** Destroys a GMRES solver object.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure

@see HYPRE_StructGMRESCreate */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructGMRESDestroy( HYPRE_StructSolver solver )
{
   return( hypre_GMRESDestroy( (void *) solver ) );
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

@see HYPRE_StructGMRESSolve */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructGMRESSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( HYPRE_GMRESSetup( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Performs the GMRES linear solve.

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

@see HYPRE_StructGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructGMRESSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( HYPRE_GMRESSolve( (HYPRE_Solver) solver,
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
  GMRES solver tolerance

@see HYPRE_StructGMRESSolve, HYPRE_StructGMRESSetup   */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGMRESSetTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( HYPRE_GMRESSetTol( (HYPRE_Solver) solver, tol ) );
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
  GMRES solver tolerance

@see HYPRE_StructGMRESSolve, HYPRE_StructGMRESSetup   */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGMRESSetAbsoluteTol( HYPRE_StructSolver solver,
                       double             atol    )
{
   return( HYPRE_GMRESSetAbsoluteTol( (HYPRE_Solver) solver, atol ) );
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
  GMRES solver maximum number of iterations

@see HYPRE_StructGMRESSolve, HYPRE_StructGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGMRESSetMaxIter( HYPRE_StructSolver solver,
                           HYPRE_Int          max_iter )
{
   return( HYPRE_GMRESSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
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
  GMRES dimension of the Krylov subspace

@see HYPRE_StructGMRESSolve, HYPRE_StructGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGMRESSetKDim( HYPRE_StructSolver solver,
                           HYPRE_Int          k_dim )
{
   return( HYPRE_GMRESSetKDim( (HYPRE_Solver) solver, k_dim ) );
}
/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Sets the preconditioner to use in GMRES.  The Default is no
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

@see HYPRE_StructGMRESSolve, HYPRE_StructGMRESSetup*/
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGMRESSetPrecond( HYPRE_StructSolver         solver,
                           HYPRE_PtrToStructSolverFcn precond,
                           HYPRE_PtrToStructSolverFcn precond_setup,
                           HYPRE_StructSolver         precond_solver )
{
   return( HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
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

@see HYPRE_StructGMRESSolve, HYPRE_StructGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGMRESSetLogging( HYPRE_StructSolver solver,
                           HYPRE_Int          logging )
{
   return( HYPRE_GMRESSetLogging( (HYPRE_Solver) solver, logging ) );
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

@see HYPRE_StructGMRESSolve, HYPRE_StructGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGMRESSetPrintLevel( HYPRE_StructSolver solver,
                           HYPRE_Int          print_level )
{
   return( HYPRE_GMRESSetPrintLevel( (HYPRE_Solver) solver, print_level ) );
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

@see HYPRE_StructGMRESSolve, HYPRE_StructGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGMRESGetNumIterations( HYPRE_StructSolver  solver,
                                 HYPRE_Int          *num_iterations )
{
   return( HYPRE_GMRESGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
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

@see HYPRE_StructGMRESSolve, HYPRE_StructGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGMRESGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
   return( HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}


