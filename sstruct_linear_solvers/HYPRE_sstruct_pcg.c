/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructPCG interface
 *
 *****************************************************************************/

#include "headers.h"

/*==========================================================================*/
/** Creates a new PCG solver object.

{\bf Input files:}
headers.h

@return Error code.

@param comm [IN]
  MPI communicator
@param solver [OUT]
  solver structure

@see HYPRE_SStructPCGDestroy */

/*--------------------------------------------------------------------------
 * HYPRE_SStructPCGCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructPCGCreate( MPI_Comm             comm,
                          HYPRE_SStructSolver *solver )
{
   hypre_PCGFunctions * pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_SStructKrylovFree,
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovClearVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (HYPRE_SStructSolver) hypre_PCGCreate( pcg_functions ) );

   return 0;
}

/*==========================================================================*/
/*==========================================================================*/
/** Destroys a PCG solver object.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure

@see HYPRE_SStructPCGCreate */

/*--------------------------------------------------------------------------
 * HYPRE_SStructPCGDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructPCGDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_PCGDestroy( (void *) solver ) );
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

@see HYPRE_SStructPCGSolve */
/*--------------------------------------------------------------------------
 * HYPRE_SStructPCGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructPCGSetup( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( hypre_PCGSetup( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*==========================================================================*/
/** Performs the PCG linear solve.

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

@see HYPRE_SStructPCGSetup */
/*--------------------------------------------------------------------------
 * HYPRE_SStructPCGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructPCGSolve( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( hypre_PCGSolve( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*==========================================================================*/
/** (Optional) Set the stopping tolerance.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param tol [IN]
  PCG solver tolerance

@see HYPRE_SStructPCGSolve, HYPRE_SStructPCGSetup   */
/*--------------------------------------------------------------------------
 * HYPRE_SStructPCGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructPCGSetTol( HYPRE_SStructSolver solver,
                          double              tol )
{
   return( hypre_PCGSetTol( (void *) solver, tol ) );
}

/*==========================================================================*/
/** (Optional) Set the maximum number of iterations.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param max_iter [IN]
  PCG solver maximum number of iterations

@see HYPRE_SStructPCGSolve, HYPRE_SStructPCGSetup */
/*--------------------------------------------------------------------------
 * HYPRE_SStructPCGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructPCGSetMaxIter( HYPRE_SStructSolver solver,
                              int                 max_iter )
{
   return( hypre_PCGSetMaxIter( (void *) solver, max_iter ) );
}

/*==========================================================================*/
/** (Optional) Set the type of norm to use in the stopping criteria.
If parameter two\_norm is set to 0, the preconditioner norm is used.
If set to 1, the two-norm is used.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param two_norm [IN]
  boolean indicating whether or not to use the two-norm

@see HYPRE_SStructPCGSolve, HYPRE_SStructPCGSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_SStructPCGSetTwoNorm( HYPRE_SStructSolver solver,
                           int                two_norm )
{
   return( hypre_PCGSetTwoNorm( (void *) solver, two_norm ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set whether or not to do an additional relative change
stopping test.  If parameter rel\_change is set to 0, no additional
stopping test is done.  If set to 1, the additional test is done.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param rel_change [IN]
  boolean indicating whether or not to do relative change test

@see HYPRE_SStructPCGSolve, HYPRE_SStructPCGSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_SStructPCGSetRelChange( HYPRE_SStructSolver solver,
                             int                rel_change )
{
   return( hypre_PCGSetRelChange( (void *) solver, rel_change ) );
}

/*==========================================================================*/
/** (Optional) Sets the precondioner to use in PCG.  The Default is no
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

@see HYPRE_SStructPCGSolve, HYPRE_SStructPCGSetup*/
/*--------------------------------------------------------------------------
 * HYPRE_SStructPCGSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructPCGSetPrecond( HYPRE_SStructSolver          solver,
                              HYPRE_PtrToSStructSolverFcn  precond,
                              HYPRE_PtrToSStructSolverFcn  precond_setup,
                              void *          precond_data )
{
   return( hypre_PCGSetPrecond( (void *) solver,
                                  precond, precond_setup, precond_data ) );
}

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

@see HYPRE_SStructPCGSolve, HYPRE_SStructPCGSetup */
/*--------------------------------------------------------------------------
 * HYPRE_SStructPCGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructPCGSetLogging( HYPRE_SStructSolver solver,
                              int                 logging )
{
   return( hypre_PCGSetLogging( (void *) solver, logging ) );
}

/*==========================================================================*/
/** (Optional) Gets the number of iterations done in the solve.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN]
  solver structure
@param num_iterations [OUT]
  number of iterations

@see HYPRE_SStructPCGSolve, HYPRE_SStructPCGSetup */
/*--------------------------------------------------------------------------
 * HYPRE_SStructPCGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructPCGGetNumIterations( HYPRE_SStructSolver  solver,
                                    int                 *num_iterations )
{
   return( hypre_PCGGetNumIterations( (void *) solver, num_iterations ) );
}

/*==========================================================================*/
/** (Optional) Gets the final relative residual norm for the solve.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN]
  solver structure
@param norm [OUT]
  final relative residual norm

@see HYPRE_SStructPCGSolve, HYPRE_SStructPCGSetup */
/*--------------------------------------------------------------------------
 * HYPRE_SStructPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructPCGGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                double              *norm )
{
   return( hypre_PCGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
