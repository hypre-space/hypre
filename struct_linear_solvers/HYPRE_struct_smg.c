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

#define NO_PTHREAD_MANGLING

#include "headers.h"
#include "threading.h"

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

typedef struct {
   HYPRE_StructSolver solver;
   HYPRE_StructMatrix A;
   HYPRE_StructVector b;
   HYPRE_StructVector x;
   int               *returnvalue;
} HYPRE_StructSMGSetupArgs;

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

void
HYPRE_StructSMGSetupVoidPtr( void *argptr )
{
   HYPRE_StructSMGSetupArgs *localargs = (HYPRE_StructSMGSetupArgs *) argptr;

   *(localargs->returnvalue) = HYPRE_StructSMGSetup( localargs->solver,
                                                     localargs->A,
                                                     localargs->b,
                                                     localargs->x );
}

int 
HYPRE_StructSMGSetupPush( HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x      )
{
   HYPRE_StructSMGSetupArgs  pushargs;
   int                       i;
   int                       returnvalue;

   pushargs.solver = solver;
   pushargs.A      = A;
   pushargs.b      = b;
   pushargs.x      = x;
   pushargs.returnvalue = (int *) malloc(sizeof(int));

   for (i=0; i<NUM_THREADS; i++)
      hypre_work_put( HYPRE_StructSMGSetupVoidPtr, (void *)&pushargs);

   hypre_work_wait();

   returnvalue = *(pushargs.returnvalue);

   free( pushargs.returnvalue );

   return returnvalue;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSolve
 *--------------------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolver solver;
   HYPRE_StructMatrix A;
   HYPRE_StructVector b;
   HYPRE_StructVector x;
   int               *returnvalue;
} HYPRE_StructSMGSolveArgs;

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

void
HYPRE_StructSMGSolveVoidPtr( void *argptr )
{
   HYPRE_StructSMGSolveArgs *localargs = (HYPRE_StructSMGSolveArgs *) argptr;

   *(localargs->returnvalue) = HYPRE_StructSMGSolve( localargs->solver,
                                                     localargs->A,
                                                     localargs->b,
                                                     localargs->x );
}

int 
HYPRE_StructSMGSolvePush( HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x      )
{
   HYPRE_StructSMGSolveArgs  pushargs;
   int                       i;
   int                       returnvalue;

   pushargs.solver = solver;
   pushargs.A      = A;
   pushargs.b      = b;
   pushargs.x      = x;
   pushargs.returnvalue = (int *) malloc(sizeof(int));

   for (i=0; i<NUM_THREADS; i++)
      hypre_work_put( HYPRE_StructSMGSolveVoidPtr, (void *)&pushargs);

   hypre_work_wait();

   returnvalue = *(pushargs.returnvalue);

   free( pushargs.returnvalue );

   return returnvalue;
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMemoryUse
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetMemoryUse( HYPRE_StructSolver solver,
                             int                memory_use )
{
   return( hypre_SMGSetMemoryUse( (void *) solver, memory_use ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( hypre_SMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetMaxIter( HYPRE_StructSolver solver,
                           int                max_iter  )
{
   return( hypre_SMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetRelChange( HYPRE_StructSolver solver,
                             int                rel_change  )
{
   return( hypre_SMGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructSMGSetZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_SMGSetZeroGuess( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPreRelax
 *
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetNumPreRelax( HYPRE_StructSolver solver,
                               int                num_pre_relax )
{
   return( hypre_SMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetNumPostRelax( HYPRE_StructSolver solver,
                                int                num_post_relax )
{
   return( hypre_SMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetLogging( HYPRE_StructSolver solver,
                           int                logging )
{
   return( hypre_SMGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGGetNumIterations( HYPRE_StructSolver  solver,
                                 int                *num_iterations )
{
   return( hypre_SMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
   return( hypre_SMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

