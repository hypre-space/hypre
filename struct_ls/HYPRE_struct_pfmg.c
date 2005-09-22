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
 * HYPRE_StructPFMGCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_PFMGCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructPFMGDestroy( HYPRE_StructSolver solver )
{
   return( hypre_PFMGDestroy( (void *) solver ) );
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
 * HYPRE_StructPFMGSetTol, HYPRE_StructPFMGGetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetTol( HYPRE_StructSolver solver,
                        double             tol    )
{
   return( hypre_PFMGSetTol( (void *) solver, tol ) );
}

int
HYPRE_StructPFMGGetTol( HYPRE_StructSolver solver,
                        double           * tol    )
{
   return( hypre_PFMGGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxIter, HYPRE_StructPFMGGetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetMaxIter( HYPRE_StructSolver solver,
                            int                max_iter  )
{
   return( hypre_PFMGSetMaxIter( (void *) solver, max_iter ) );
}

int
HYPRE_StructPFMGGetMaxIter( HYPRE_StructSolver solver,
                            int              * max_iter  )
{
   return( hypre_PFMGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxLevels, HYPRE_StructPFMGGetMaxLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetMaxLevels( HYPRE_StructSolver solver,
                              int                max_levels  )
{
   return( hypre_PFMGSetMaxLevels( (void *) solver, max_levels ) );
}

int
HYPRE_StructPFMGGetMaxLevels( HYPRE_StructSolver solver,
                              int              * max_levels  )
{
   return( hypre_PFMGGetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelChange, HYPRE_StructPFMGGetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetRelChange( HYPRE_StructSolver solver,
                              int                rel_change  )
{
   return( hypre_PFMGSetRelChange( (void *) solver, rel_change ) );
}

int
HYPRE_StructPFMGGetRelChange( HYPRE_StructSolver solver,
                              int              * rel_change  )
{
   return( hypre_PFMGGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetZeroGuess, HYPRE_StructPFMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructPFMGSetZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_PFMGSetZeroGuess( (void *) solver, 1 ) );
}

int
HYPRE_StructPFMGGetZeroGuess( HYPRE_StructSolver solver,
                              int * zeroguess )
{
   return( hypre_PFMGGetZeroGuess( (void *) solver, zeroguess ) );
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
 * HYPRE_StructPFMGSetRelaxType, HYPRE_StructPFMGGetRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetRelaxType( HYPRE_StructSolver solver,
                              int                relax_type )
{
   return( hypre_PFMGSetRelaxType( (void *) solver, relax_type) );
}

int
HYPRE_StructPFMGGetRelaxType( HYPRE_StructSolver solver,
                              int              * relax_type )
{
   return( hypre_PFMGGetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRAPType, HYPRE_StructPFMGGetRAPType
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetRAPType( HYPRE_StructSolver solver,
                            int                rap_type )
{
   return( hypre_PFMGSetRAPType( (void *) solver, rap_type) );
}

int
HYPRE_StructPFMGGetRAPType( HYPRE_StructSolver solver,
                            int              * rap_type )
{
   return( hypre_PFMGGetRAPType( (void *) solver, rap_type) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPreRelax, HYPRE_StructPFMGGetNumPreRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetNumPreRelax( HYPRE_StructSolver solver,
                                int                num_pre_relax )
{
   return( hypre_PFMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

int
HYPRE_StructPFMGGetNumPreRelax( HYPRE_StructSolver solver,
                                int              * num_pre_relax )
{
   return( hypre_PFMGGetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPostRelax, HYPRE_StructPFMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetNumPostRelax( HYPRE_StructSolver solver,
                                 int                num_post_relax )
{
   return( hypre_PFMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

int
HYPRE_StructPFMGGetNumPostRelax( HYPRE_StructSolver solver,
                                 int              * num_post_relax )
{
   return( hypre_PFMGGetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetSkipRelax, HYPRE_StructPFMGGetSkipRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetSkipRelax( HYPRE_StructSolver solver,
                              int                skip_relax )
{
   return( hypre_PFMGSetSkipRelax( (void *) solver, skip_relax) );
}

int
HYPRE_StructPFMGGetSkipRelax( HYPRE_StructSolver solver,
                              int              * skip_relax )
{
   return( hypre_PFMGGetSkipRelax( (void *) solver, skip_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetDxyz, HYPRE_StructPFMGGetDxyz
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetDxyz( HYPRE_StructSolver  solver,
                         double             *dxyz   )
{
   return( hypre_PFMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetLogging, HYPRE_StructPFMGGetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetLogging( HYPRE_StructSolver solver,
                            int                logging )
{
   return( hypre_PFMGSetLogging( (void *) solver, logging) );
}

int
HYPRE_StructPFMGGetLogging( HYPRE_StructSolver solver,
                            int              * logging )
{
   return( hypre_PFMGGetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetPrintLevel, HYPRE_StructPFMGGetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPFMGSetPrintLevel( HYPRE_StructSolver solver,
                            int                  print_level )
{
   return( hypre_PFMGSetPrintLevel( (void *) solver, print_level) );
}

int
HYPRE_StructPFMGGetPrintLevel( HYPRE_StructSolver solver,
                            int                * print_level )
{
   return( hypre_PFMGGetPrintLevel( (void *) solver, print_level) );
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

