/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_StructJacobi interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiInitialize( MPI_Comm            comm,
                              HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_JacobiInitialize( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiFinalize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructJacobiFinalize( HYPRE_StructSolver solver )
{
   return( hypre_JacobiFinalize( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructJacobiSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( hypre_JacobiSetup( (void *) solver,
                              (hypre_StructMatrix *) A,
                              (hypre_StructVector *) b,
                              (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructJacobiSolve( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return( hypre_JacobiSolve( (void *) solver,
                              (hypre_StructMatrix *) A,
                              (hypre_StructVector *) b,
                              (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiSetTol( HYPRE_StructSolver solver,
                          double             tol    )
{
   return( hypre_JacobiSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiSetMaxIter( HYPRE_StructSolver solver,
                              int                max_iter  )
{
   return( hypre_JacobiSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructJacobiSetZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_JacobiSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructJacobiSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_JacobiSetZeroGuess( (void *) solver, 0 ) );
}





/* NOT YET IMPLEMENTED */

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiGetNumIterations( HYPRE_StructSolver  solver,
                                 int                *num_iterations )
{
#if 0
   return( hypre_JacobiGetNumIterations( (void *) solver, num_iterations ) );
#endif
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
#if 0
   return( hypre_JacobiGetFinalRelativeResidualNorm( (void *) solver, norm ) );
#endif
   return 0;
}
