/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_StructSMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGInitialize
 *--------------------------------------------------------------------------*/

HYPRE_StructSolver
HYPRE_StructSMGInitialize( MPI_Comm comm )
{
   return ( (HYPRE_StructSolver) hypre_SMGInitialize( comm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGFinalize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSMGFinalize( HYPRE_StructSolver solver )
{
   return( hypre_SMGFinalize( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSMGSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( hypre_SMGSetup( (void *) solver,
                           (hypre_StructMatrix *) A,
                           (hypre_StructVector *) b,
                           (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSMGSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( hypre_SMGSolve( (void *) solver,
                           (hypre_StructMatrix *) A,
                           (hypre_StructVector *) b,
                           (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SMGSetMemoryUse
 *--------------------------------------------------------------------------*/

int
HYPRE_SMGSetMemoryUse( HYPRE_StructSolver solver,
                       int              memory_use )
{
   return( hypre_SMGSetMemoryUse( (void *) solver, memory_use ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SMGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_SMGSetTol( HYPRE_StructSolver solver,
                 double           tol    )
{
   return( hypre_SMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SMGSetMaxIter( HYPRE_StructSolver solver,
                     int              max_iter  )
{
   return( hypre_SMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_SMGSetZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_SMGSetZeroGuess( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SMGSetNumPreRelax
 *
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

int
HYPRE_SMGSetNumPreRelax( HYPRE_StructSolver solver,
                         int                num_pre_relax )
{
   return( hypre_SMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_SMGSetNumPostRelax( HYPRE_StructSolver solver,
                          int                num_post_relax )
{
   return( hypre_SMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_SMGGetNumIterations( HYPRE_StructSolver  solver,
                           int                *num_iterations )
{
   return( hypre_SMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_SMGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                       double             *norm   )
{
   return( hypre_SMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

