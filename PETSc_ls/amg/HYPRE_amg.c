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
 * HYPRE_AMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_AMGInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Solver
HYPRE_AMGInitialize( MPI_Comm comm )
{
   return ( (HYPRE_Solver) hypre_AMGInitialize( comm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGFinalize
 *--------------------------------------------------------------------------*/

int 
HYPRE_AMGFinalize( HYPRE_Solver solver )
{
   return( hypre_AMGFinalize( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_AMGSetup( HYPRE_Solver solver,
                HYPRE_Matrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{
   return( hypre_AMGSetup( (void *) solver,
                           (hypre_Matrix *) A,
                           (hypre_Vector *) b,
                           (hypre_Vector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_AMGSolve( HYPRE_Solver solver,
                HYPRE_Matrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{
   return( hypre_AMGSolve( (void *) solver,
                           (hypre_Matrix *) A,
                           (hypre_Vector *) b,
                           (hypre_Vector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetTol( HYPRE_Solver solver,
                 double           tol    )
{
   return( hypre_AMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetMaxIter( HYPRE_Solver solver,
                     int              max_iter  )
{
   return( hypre_AMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_AMGSetZeroGuess( HYPRE_Solver solver )
{
   return( hypre_AMGSetZeroGuess( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGGetNumIterations( HYPRE_Solver  solver,
                           int              *num_iterations )
{
   return( hypre_AMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                       double       *relative_residual_norm )
{
   return( hypre_AMGGetFinalRelativeResidualNorm( (void *) solver,
                                                  relative_residual_norm ) );
}

