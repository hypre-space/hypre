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
 * HYPRE_StructPFMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGInitialize( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_PFMGInitialize( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGFinalize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructPFMGFinalize( HYPRE_StructSolver solver )
{
   return( hypre_PFMGFinalize( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructPFMGSetup( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix A,
                       HYPRE_StructVector b,
                       HYPRE_StructVector x      )
{
   return( hypre_PFMGSetup( (void *) solver,
                            (hypre_StructMatrix *) A,
                            (hypre_StructVector *) b,
                            (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructPFMGSolve( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix A,
                       HYPRE_StructVector b,
                       HYPRE_StructVector x      )
{
   return( hypre_PFMGSolve( (void *) solver,
                            (hypre_StructMatrix *) A,
                            (hypre_StructVector *) b,
                            (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetTol( HYPRE_StructSolver solver,
                        double             tol    )
{
   return( hypre_PFMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetMaxIter( HYPRE_StructSolver solver,
                            int                max_iter  )
{
   return( hypre_PFMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetRelChange( HYPRE_StructSolver solver,
                              int                rel_change  )
{
   return( hypre_PFMGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructPFMGSetZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_PFMGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructPFMGSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_PFMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetRelaxType( HYPRE_StructSolver solver,
                              int                relax_type )
{
   return( hypre_PFMGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetNumPreRelax( HYPRE_StructSolver solver,
                                int                num_pre_relax )
{
   return( hypre_PFMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetNumPostRelax( HYPRE_StructSolver solver,
                                 int                num_post_relax )
{
   return( hypre_PFMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetDxyz
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetDxyz( HYPRE_StructSolver  solver,
                         double             *dxyz   )
{
   return( hypre_PFMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetLogging( HYPRE_StructSolver solver,
                            int                logging )
{
   return( hypre_PFMGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGGetNumIterations( HYPRE_StructSolver  solver,
                                  int                *num_iterations )
{
   return( hypre_PFMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                              double             *norm   )
{
   return( hypre_PFMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

