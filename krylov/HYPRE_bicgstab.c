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
 * HYPRE_BiCGSTAB interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABCreate does not exist.  Call the appropriate function which
 * also specifies the vector type, e.g. HYPRE_ParCSRBiCGSTABCreate
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_BiCGSTABDestroy( HYPRE_Solver solver )
{
   return( hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_BiCGSTABSetup( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_BiCGSTABSetup( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_BiCGSTABSolve( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_BiCGSTABSolve( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( hypre_BiCGSTABSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( hypre_BiCGSTABSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( hypre_BiCGSTABSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetStopCrit( HYPRE_Solver solver,
                              int          stop_crit )
{
   return( hypre_BiCGSTABSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetPrecond( HYPRE_Solver         solver,
                                HYPRE_PtrToSolverFcn precond,
                                HYPRE_PtrToSolverFcn precond_setup,
                                HYPRE_Solver         precond_solver )
{
   return( hypre_BiCGSTABSetPrecond( (void *) solver,
                                     precond, precond_setup,
                                     (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( hypre_BiCGSTABGetPrecond( (void *)     solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABSetLogging( HYPRE_Solver solver,
                             int logging)
{
   return( hypre_BiCGSTABSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( hypre_BiCGSTABGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_BiCGSTABGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( hypre_BiCGSTABGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
