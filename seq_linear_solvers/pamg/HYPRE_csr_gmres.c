/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_CSRGMRES interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESCreate( HYPRE_Solver *solver )
{
   *solver = ( (HYPRE_Solver) hypre_GMRCreate( ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_GMRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRGMRESSetup( HYPRE_Solver solver,
                        HYPRE_CSRMatrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_GMRSetup( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRGMRESSolve( HYPRE_Solver solver,
                        HYPRE_CSRMatrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_GMRSolve( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetKDim( HYPRE_Solver solver,
                          int             k_dim    )
{
   return( hypre_GMRSetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( hypre_GMRSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( hypre_GMRSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( hypre_GMRSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetStopCrit( HYPRE_Solver solver,
                              int          stop_crit )
{
   return( hypre_GMRSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetPrecond( HYPRE_Solver  solver,
                             int (*precond)      (HYPRE_Solver sol, 
					 	  HYPRE_CSRMatrix matrix,
						  HYPRE_Vector b,
						  HYPRE_Vector x),
                             int (*precond_setup)(HYPRE_Solver sol, 
					 	  HYPRE_CSRMatrix matrix,
						  HYPRE_Vector b,
						  HYPRE_Vector x),
                             void               *precond_data )
{
   return( hypre_GMRSetPrecond( (void *) solver,
                                precond, precond_setup, precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( hypre_GMRGetPrecond( (void *)     solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetLogging( HYPRE_Solver solver,
                             int logging)
{
   return( hypre_GMRSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( hypre_GMRGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( hypre_GMRGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
