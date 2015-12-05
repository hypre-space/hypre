/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/





#include "headers.h"

/*==========================================================================*/
/** Creates a new LGMRES solver object.

{\bf Input files:}
headers.h

@return Error code.

@param comm [IN]
  MPI communicator
@param solver [OUT]
  solver structure

@see HYPRE_StructLGMRESDestroy */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   hypre_LGMRESFunctions * lgmres_functions =
      hypre_LGMRESFunctionsCreate(
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

   *solver = ( (HYPRE_StructSolver) hypre_LGMRESCreate( lgmres_functions ) );

   return 0;
}

/*==========================================================================*/
/*==========================================================================*/
/** Destroys a LGMRES solver object.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure

@see HYPRE_StructLGMRESCreate */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructLGMRESDestroy( HYPRE_StructSolver solver )
{
   return( hypre_LGMRESDestroy( (void *) solver ) );
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

@see HYPRE_StructLGMRESSolve */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructLGMRESSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( HYPRE_LGMRESSetup( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Performs the LGMRES linear solve.

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

@see HYPRE_StructLGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructLGMRESSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( HYPRE_LGMRESSolve( (HYPRE_Solver) solver,
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
  LGMRES solver tolerance

@see HYPRE_StructLGMRESSolve, HYPRE_StructLGMRESSetup   */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESSetTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( HYPRE_LGMRESSetTol( (HYPRE_Solver) solver, tol ) );
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
  LGMRES solver tolerance

@see HYPRE_StructLGMRESSolve, HYPRE_StructLGMRESSetup   */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESSetAbsoluteTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( HYPRE_LGMRESSetAbsoluteTol( (HYPRE_Solver) solver, tol ) );
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
  LGMRES solver maximum number of iterations

@see HYPRE_StructLGMRESSolve, HYPRE_StructLGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESSetMaxIter( HYPRE_StructSolver solver,
                           HYPRE_Int          max_iter )
{
   return( HYPRE_LGMRESSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Sets the dimension of the  approximation subspace.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
 solver structure
@param k_dim [IN]
  LGMRES dimension of the approximation space

@see HYPRE_StructLGMRESSolve, HYPRE_StructLGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESSetKDim( HYPRE_StructSolver solver,
                           HYPRE_Int          k_dim )
{
   return( HYPRE_LGMRESSetKDim( (HYPRE_Solver) solver, k_dim ) );
}



/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Sets the number of augmentation vectors.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param aug_dim [IN]
  LGMRES dimension of the Krylov subspace

@see HYPRE_StructLGMRESSolve, HYPRE_StructLGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESSetAugDim( HYPRE_StructSolver solver,
                           HYPRE_Int          aug_dim )
{
   return( HYPRE_LGMRESSetAugDim( (HYPRE_Solver) solver, aug_dim ) );
}


/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Sets the precondioner to use in LGMRES.  The Default is no
preconditioner, i.e. the solver is just LGMRES.

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

@see HYPRE_StructLGMRESSolve, HYPRE_StructLGMRESSetup*/
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESSetPrecond( HYPRE_StructSolver         solver,
                           HYPRE_PtrToStructSolverFcn precond,
                           HYPRE_PtrToStructSolverFcn precond_setup,
                           HYPRE_StructSolver         precond_solver )
{
   return( HYPRE_LGMRESSetPrecond( (HYPRE_Solver) solver,
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

@see HYPRE_StructLGMRESSolve, HYPRE_StructLGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESSetLogging( HYPRE_StructSolver solver,
                           HYPRE_Int          logging )
{
   return( HYPRE_LGMRESSetLogging( (HYPRE_Solver) solver, logging ) );
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

@see HYPRE_StructLGMRESSolve, HYPRE_StructLGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESSetPrintLevel( HYPRE_StructSolver solver,
                           HYPRE_Int          print_level )
{
   return( HYPRE_LGMRESSetPrintLevel( (HYPRE_Solver) solver, print_level ) );
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

@see HYPRE_StructLGMRESSolve, HYPRE_StructLGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESGetNumIterations( HYPRE_StructSolver  solver,
                                 HYPRE_Int          *num_iterations )
{
   return( HYPRE_LGMRESGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
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

@see HYPRE_StructLGMRESSolve, HYPRE_StructLGMRESSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructLGMRESGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
   return( HYPRE_LGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}


