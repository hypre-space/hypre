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

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridCreate( HYPRE_Solver *solver )
{
   *solver = ( (HYPRE_Solver) hypre_AMGHybridCreate( ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRHybridDestroy( HYPRE_Solver solver )
{
   return( hypre_AMGHybridDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRHybridSetup( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   return( hypre_AMGHybridSetup( (void *) solver,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b,
                              (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRHybridSolve( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   return( hypre_AMGHybridSolve( (void *) solver,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b,
                              (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetTol( HYPRE_Solver solver,
                          double             tol    )
{
   return( hypre_AMGHybridSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetConvergenceTol( HYPRE_Solver solver,
                                     double             cf_tol    )
{
   return( hypre_AMGHybridSetConvergenceTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetDSCGMaxIter( HYPRE_Solver solver,
                                  int                dscg_max_its )
{
   return( hypre_AMGHybridSetDSCGMaxIter( (void *) solver, dscg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetPCGMaxIter( HYPRE_Solver solver,
                                 int                pcg_max_its )
{
   return( hypre_AMGHybridSetPCGMaxIter( (void *) solver, pcg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetTwoNorm( HYPRE_Solver solver,
                              int                two_norm    )
{
   return( hypre_AMGHybridSetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetRelChange( HYPRE_Solver solver,
                                int                rel_change    )
{
   return( hypre_AMGHybridSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetPrecond( HYPRE_Solver         solver,
                              HYPRE_PtrToParSolverFcn precond,
                              HYPRE_PtrToParSolverFcn precond_setup,
                              HYPRE_Solver         precond_solver )
{
   return( hypre_AMGHybridSetPrecond( (void *) solver,
                                   precond, precond_setup,
                                   (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetLogging( HYPRE_Solver solver,
                              int                logging    )
{
   return( hypre_AMGHybridSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetPLogging( HYPRE_Solver solver,
                              int               plogging    )
{
   return( hypre_AMGHybridSetPLogging( (void *) solver, plogging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetStrongThreshold( HYPRE_Solver solver,
                              double            strong_threshold    )
{
   return( hypre_AMGHybridSetStrongThreshold( (void *) solver, 
		strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetMaxRowSum( HYPRE_Solver solver,
                              double             max_row_sum    )
{
   return( hypre_AMGHybridSetMaxRowSum( (void *) solver, max_row_sum ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetTruncFactor( HYPRE_Solver solver,
                              double              trunc_factor    )
{
   return( hypre_AMGHybridSetTruncFactor( (void *) solver, trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetMaxLevels( HYPRE_Solver solver,
                              int                max_levels    )
{
   return( hypre_AMGHybridSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMeasureType
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetMeasureType( HYPRE_Solver solver,
                              int                measure_type    )
{
   return( hypre_AMGHybridSetMeasureType( (void *) solver, measure_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetCoarsenType( HYPRE_Solver solver,
                              int                coarsen_type    )
{
   return( hypre_AMGHybridSetCoarsenType( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleType
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetCycleType( HYPRE_Solver solver,
                              int                cycle_type    )
{
   return( hypre_AMGHybridSetCycleType( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetNumGridSweeps( HYPRE_Solver solver,
                              int               *num_grid_sweeps    )
{
   return( hypre_AMGHybridSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetGridRelaxType( HYPRE_Solver solver,
                              int               *grid_relax_type    )
{
   return( hypre_AMGHybridSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetGridRelaxPoints( HYPRE_Solver solver,
                              int              **grid_relax_points    )
{
   return( hypre_AMGHybridSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridSetRelaxWeight( HYPRE_Solver solver,
                              double             *relax_weight    )
{
   return( hypre_AMGHybridSetRelaxWeight( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridGetNumIterations( HYPRE_Solver solver,
                                    int               *num_its    )
{
   return( hypre_AMGHybridGetNumIterations( (void *) solver, num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridGetDSCGNumIterations( HYPRE_Solver solver,
                                        int               *dscg_num_its )
{
   return( hypre_AMGHybridGetDSCGNumIterations( (void *) solver, dscg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridGetPCGNumIterations( HYPRE_Solver solver,
                                       int               *pcg_num_its )
{
   return( hypre_AMGHybridGetPCGNumIterations( (void *) solver, pcg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRHybridGetFinalRelativeResidualNorm( HYPRE_Solver solver,
                                                double            *norm    )
{
   return( hypre_AMGHybridGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

