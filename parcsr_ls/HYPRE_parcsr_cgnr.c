/*BHEADER**********************************************************************
 * (c) 1998-2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_ParCSRCGNR interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRCGNRCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_CGNRFunctions * cgnr_functions =
      hypre_CGNRFunctionsCreate(
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvec,
         hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd,
         hypre_ParKrylovCopyVector, hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup,
         hypre_ParKrylovIdentity, hypre_ParKrylovIdentity );

   *solver = ( (HYPRE_Solver) hypre_CGNRCreate( cgnr_functions) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRCGNRDestroy( HYPRE_Solver solver )
{
   return( hypre_CGNRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRCGNRSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_CGNRSetup( solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRCGNRSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_CGNRSolve( solver,
                            (HYPRE_Matrix) A,
                            (HYPRE_Vector) b,
                            (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRCGNRSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( hypre_CGNRSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRCGNRSetMinIter( HYPRE_Solver solver,
                             int                min_iter )
{
   return( hypre_CGNRSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRCGNRSetMaxIter( HYPRE_Solver solver,
                             int                max_iter )
{
   return( hypre_CGNRSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRCGNRSetStopCrit( HYPRE_Solver solver,
                             int                stop_crit )
{
   return( hypre_CGNRSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRCGNRSetPrecond( HYPRE_Solver         solver,
                            HYPRE_PtrToParSolverFcn precond,
                            HYPRE_PtrToParSolverFcn precondT,
                            HYPRE_PtrToParSolverFcn precond_setup,
                            HYPRE_Solver         precond_solver )
{
   return( hypre_CGNRSetPrecond( (void *) solver,
                                 precond, precondT, precond_setup,
                                 (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRCGNRGetPrecond( HYPRE_Solver   solver,
                            HYPRE_Solver  *precond_data_ptr )
{
   return( hypre_CGNRGetPrecond( (void *)         solver,
                                 (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRCGNRSetLogging( HYPRE_Solver solver,
                             int logging)
{
   return( hypre_CGNRSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRCGNRGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( hypre_CGNRGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( hypre_CGNRGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
