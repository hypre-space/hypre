/******************************************************************************
 *
 * HYPRE_SStructFAC Routines
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACCreate( MPI_Comm comm, HYPRE_SStructSolver *solver )
{
   *solver = ( (HYPRE_SStructSolver) hypre_FACCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACDestroy
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACDestroy2( HYPRE_SStructSolver solver )
{
   return( hypre_FACDestroy2( (void *) solver ) );
}

int
HYPRE_SStructFACSetup2( HYPRE_SStructSolver  solver,
                        HYPRE_SStructMatrix  A,
                        HYPRE_SStructVector  b,
                        HYPRE_SStructVector  x )
{
   return( hypre_FacSetup2( (void *) solver,
                           (hypre_SStructMatrix *)  A,
                           (hypre_SStructVector *)  b,
                           (hypre_SStructVector *)  x ) );
}

int
HYPRE_SStructFACSolve3(HYPRE_SStructSolver solver,
                       HYPRE_SStructMatrix A,
                       HYPRE_SStructVector b,
                       HYPRE_SStructVector x)
{
   return( hypre_FACSolve3((void *) solver,
                           (hypre_SStructMatrix *)  A,
                           (hypre_SStructVector *)  b,
                           (hypre_SStructVector *)  x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetTol( HYPRE_SStructSolver solver,
                        double             tol    )
{
   return( hypre_FACSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACSetPLevels( HYPRE_SStructSolver  solver,
                            int                  nparts,
                            int                 *plevels)
{
   return( hypre_FACSetPLevels( (void *) solver, nparts, plevels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetPRefinements( HYPRE_SStructSolver  solver,
                                 int                  nparts,
                                 int                (*rfactors)[3] )
{
   return( hypre_FACSetPRefinements( (void *)         solver,
                                                      nparts,
                                                      rfactors ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetMaxLevels( HYPRE_SStructSolver solver,
                              int                 max_levels  )
{
   return( hypre_FACSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetMaxIter( HYPRE_SStructSolver solver,
                            int                max_iter  )
{
   return( hypre_FACSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetRelChange( HYPRE_SStructSolver solver,
                              int                rel_change  )
{
   return( hypre_FACSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_FACSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetNonZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_FACSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetRelaxType( HYPRE_SStructSolver solver,
                              int                relax_type )
{
   return( hypre_FACSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACSetNumPreRelax( HYPRE_SStructSolver solver,
                                int                num_pre_relax )
{
   return( hypre_FACSetNumPreSmooth( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetNumPostRelax( HYPRE_SStructSolver solver,
                                 int                num_post_relax )
{
   return( hypre_FACSetNumPostSmooth( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetCoarseSolverType( HYPRE_SStructSolver solver,
                                     int                 csolver_type)
{
   return( hypre_FACSetCoarseSolverType( (void *) solver, csolver_type) );
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetLogging( HYPRE_SStructSolver solver,
                            int                logging )
{
   return( hypre_FACSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACGetNumIterations( HYPRE_SStructSolver  solver,
                                  int                *num_iterations )
{
   return( hypre_FACGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                              double             *norm   )
{
   return( hypre_FACGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}


