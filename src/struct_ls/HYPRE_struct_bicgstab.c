/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/





#include "headers.h"

/*==========================================================================*/
/** Creates a new BiCGSTAB solver object.

{\bf Input files:}
headers.h

@return Error code.

@param comm [IN]
  MPI communicator
@param solver [OUT]
  solver structure

@see HYPRE_StructBiCGSTABDestroy */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBiCGSTABCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   hypre_BiCGSTABFunctions * bicgstab_functions =
      hypre_BiCGSTABFunctionsCreate(
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );

   *solver = ( (HYPRE_StructSolver) hypre_BiCGSTABCreate( bicgstab_functions ) );

   return 0;
}

/*==========================================================================*/
/*==========================================================================*/
/** Destroys a BiCGSTAB solver object.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure

@see HYPRE_StructBiCGSTABCreate */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructBiCGSTABDestroy( HYPRE_StructSolver solver )
{
   return( hypre_BiCGSTABDestroy( (void *) solver ) );
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

@see HYPRE_StructBiCGSTABSolve */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructBiCGSTABSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( HYPRE_BiCGSTABSetup( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Performs the BiCGSTAB linear solve.

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

@see HYPRE_StructBiCGSTABSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructBiCGSTABSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( HYPRE_BiCGSTABSolve( (HYPRE_Solver) solver,
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
  BiCGSTAB solver tolerance

@see HYPRE_StructBiCGSTABSolve, HYPRE_StructBiCGSTABSetup   */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBiCGSTABSetTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( HYPRE_BiCGSTABSetTol( (HYPRE_Solver) solver, tol ) );
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
  BiCGSTAB solver tolerance

@see HYPRE_StructBiCGSTABSolve, HYPRE_StructBiCGSTABSetup   */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBiCGSTABSetAbsoluteTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( HYPRE_BiCGSTABSetAbsoluteTol( (HYPRE_Solver) solver, tol ) );
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
  BiCGSTAB solver maximum number of iterations

@see HYPRE_StructBiCGSTABSolve, HYPRE_StructBiCGSTABSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBiCGSTABSetMaxIter( HYPRE_StructSolver solver,
                           HYPRE_Int          max_iter )
{
   return( HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}


/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Sets the precondioner to use in BiCGSTAB.  The Default is no
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

@see HYPRE_StructBiCGSTABSolve, HYPRE_StructBiCGSTABSetup*/
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBiCGSTABSetPrecond( HYPRE_StructSolver         solver,
                           HYPRE_PtrToStructSolverFcn precond,
                           HYPRE_PtrToStructSolverFcn precond_setup,
                           HYPRE_StructSolver         precond_solver )
{
   return( HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
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

@see HYPRE_StructBiCGSTABSolve, HYPRE_StructBiCGSTABSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBiCGSTABSetLogging( HYPRE_StructSolver solver,
                           HYPRE_Int          logging )
{
   return( HYPRE_BiCGSTABSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Sets the print level.  Currently, if parameter
is set to 0, no printing is done. 

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param logging [IN]
  integer indicating the print level

@see HYPRE_StructBiCGSTABSolve, HYPRE_StructBiCGSTABSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBiCGSTABSetPrintLevel( HYPRE_StructSolver solver,
                            	HYPRE_Int level)
{
   return( HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver) solver, level ) );
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

@see HYPRE_StructBiCGSTABSolve, HYPRE_StructBiCGSTABSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBiCGSTABGetNumIterations( HYPRE_StructSolver  solver,
                                 HYPRE_Int          *num_iterations )
{
   return( HYPRE_BiCGSTABGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
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

@see HYPRE_StructBiCGSTABSolve, HYPRE_StructBiCGSTABSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
   return( HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}


/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Gets the residual .

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN]
  solver structure
@param residual [OUT]
  residual vector

@see HYPRE_StructBiCGSTABSolve, HYPRE_StructBiCGSTABSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBiCGSTABGetResidual( HYPRE_StructSolver  solver,
                                 void  **residual)
{
   return( HYPRE_BiCGSTABGetResidual( (HYPRE_Solver) solver, residual ) );
}

