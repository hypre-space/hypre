/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

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

int
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

int 
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

int 
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

int 
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

int
HYPRE_StructGMRESSetTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( HYPRE_GMRESSetTol( (HYPRE_Solver) solver, tol ) );
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

int
HYPRE_StructGMRESSetMaxIter( HYPRE_StructSolver solver,
                           int                max_iter )
{
   return( HYPRE_GMRESSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}


/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Sets the precondioner to use in GMRES.  The Default is no
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

int
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

int
HYPRE_StructGMRESSetLogging( HYPRE_StructSolver solver,
                           int                logging )
{
   return( HYPRE_GMRESSetLogging( (HYPRE_Solver) solver, logging ) );
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

int
HYPRE_StructGMRESGetNumIterations( HYPRE_StructSolver  solver,
                                 int                *num_iterations )
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

int
HYPRE_StructGMRESGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
   return( HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}


