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
 *
 *****************************************************************************/

#include "headers.h"

#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#endif

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridInitialize( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_HybridInitialize( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridFinalize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructHybridFinalize( HYPRE_StructSolver solver )
{
   return( hypre_HybridFinalize( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructHybridSetup( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return( hypre_HybridSetup( (void *) solver,
                              (hypre_StructMatrix *) A,
                              (hypre_StructVector *) b,
                              (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructHybridSolve( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return( hypre_HybridSolve( (void *) solver,
                              (hypre_StructMatrix *) A,
                              (hypre_StructVector *) b,
                              (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetTol( HYPRE_StructSolver solver,
                          double             tol    )
{
   return( hypre_HybridSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetConvergenceTol( HYPRE_StructSolver solver,
                                     double             cf_tol    )
{
   return( hypre_HybridSetConvergenceTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetMaxDSIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetMaxDSIterations( HYPRE_StructSolver solver,
                                      int                max_ds_its    )
{
   return( hypre_HybridSetMaxDSIterations( (void *) solver, max_ds_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetMaxMGIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetMaxMGIterations( HYPRE_StructSolver solver,
                                      int                max_mg_its    )
{
   return( hypre_HybridSetMaxMGIterations( (void *) solver, max_mg_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetTwoNorm( HYPRE_StructSolver solver,
                              int                two_norm    )
{
   return( hypre_HybridSetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetRelChange( HYPRE_StructSolver solver,
                                int                rel_change    )
{
   return( hypre_HybridSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetLogging( HYPRE_StructSolver solver,
                              int                logging    )
{
   return( hypre_HybridSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridGetNumIterations( HYPRE_StructSolver solver,
                                    int               *num_its    )
{
   return( hypre_HybridGetNumIterations( (void *) solver, num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetNumDSIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridGetNumDSIterations( HYPRE_StructSolver solver,
                                      int               *num_ds_its    )
{
   return( hypre_HybridGetNumDSIterations( (void *) solver, num_ds_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetNumMGIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridGetNumMGIterations( HYPRE_StructSolver solver,
                                      int               *num_mg_its    )
{
   return( hypre_HybridGetNumMGIterations( (void *) solver, num_mg_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridGetFinalRelativeResidualNorm( HYPRE_StructSolver solver,
                                                double            *norm    )
{
   return( hypre_HybridGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

