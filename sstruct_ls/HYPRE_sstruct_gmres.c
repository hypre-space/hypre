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
 * HYPRE_SStructGMRES interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGMRESCreate( MPI_Comm             comm,
                          HYPRE_SStructSolver *solver )
{
   *solver = ( (HYPRE_SStructSolver) hypre_GMRESCreate( ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructGMRESDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_GMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructGMRESSetup( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( hypre_GMRESSetup( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructGMRESSolve( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( hypre_GMRESSolve( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGMRESSetKDim( HYPRE_SStructSolver solver,
                           int                 k_dim )
{
   return( hypre_GMRESSetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGMRESSetTol( HYPRE_SStructSolver solver,
                          double              tol )
{
   return( hypre_GMRESSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGMRESSetMinIter( HYPRE_SStructSolver solver,
                              int                 min_iter )
{
   return( hypre_GMRESSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGMRESSetMaxIter( HYPRE_SStructSolver solver,
                              int                 max_iter )
{
   return( hypre_GMRESSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGMRESSetStopCrit( HYPRE_SStructSolver solver,
                               int                 stop_crit )
{
   return( hypre_GMRESSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGMRESSetPrecond( HYPRE_SStructSolver          solver,
                              HYPRE_PtrToSStructSolverFcn  precond,
                              HYPRE_PtrToSStructSolverFcn  precond_setup,
                              void                        *precond_data )
{
   return( hypre_GMRESSetPrecond( (void *) solver,
                                  precond, precond_setup, precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGMRESSetLogging( HYPRE_SStructSolver solver,
                              int                 logging )
{
   return( hypre_GMRESSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGMRESGetNumIterations( HYPRE_SStructSolver  solver,
                                    int                 *num_iterations )
{
   return( hypre_GMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGMRESGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                double              *norm )
{
   return( hypre_GMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
