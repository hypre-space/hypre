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
 * HYPRE_ParCSRGMRES interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_GMRESFunctions * gmres_functions =
      hypre_GMRESFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );

   *solver = ( (HYPRE_Solver) hypre_GMRESCreate( gmres_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_GMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRGMRESSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( hypre_GMRESSetup( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRGMRESSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( hypre_GMRESSolve( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetKDim( HYPRE_Solver solver,
                          int             k_dim    )
{
   return( hypre_GMRESSetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( hypre_GMRESSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( hypre_GMRESSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( hypre_GMRESSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetStopCrit( HYPRE_Solver solver,
                              int          stop_crit )
{
   return( hypre_GMRESSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetPrecond( HYPRE_Solver          solver,
                             HYPRE_PtrToParSolverFcn  precond,
                             HYPRE_PtrToParSolverFcn  precond_setup,
                             HYPRE_Solver          precond_solver )
{
   return( hypre_GMRESSetPrecond( (void *) solver,
                                  precond, precond_setup,
                                  (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( hypre_GMRESGetPrecond( (void *)     solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetLogging( HYPRE_Solver solver,
                             int logging)
{
   return( hypre_GMRESSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( hypre_GMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( hypre_GMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
