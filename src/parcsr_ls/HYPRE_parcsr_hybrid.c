/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridCreate( HYPRE_Solver *solver )
{
   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   *solver = ( (HYPRE_Solver) hypre_AMGHybridCreate( ) );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridDestroy( HYPRE_Solver solver )
{
   return ( hypre_AMGHybridDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetup( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   return ( hypre_AMGHybridSetup( (void *) solver,
                                  (hypre_ParCSRMatrix *) A,
                                  (hypre_ParVector *) b,
                                  (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSolve( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   return ( hypre_AMGHybridSolve( (void *) solver,
                                  (hypre_ParCSRMatrix *) A,
                                  (hypre_ParVector *) b,
                                  (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetTol( HYPRE_Solver solver,
                          HYPRE_Real   tol    )
{
   return ( hypre_AMGHybridSetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetAbsoluteTol( HYPRE_Solver solver,
                                  HYPRE_Real   tol    )
{
   return ( hypre_AMGHybridSetAbsoluteTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetConvergenceTol( HYPRE_Solver solver,
                                     HYPRE_Real   cf_tol    )
{
   return ( hypre_AMGHybridSetConvergenceTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetDSCGMaxIter( HYPRE_Solver solver,
                                  HYPRE_Int    dscg_max_its )
{
   return ( hypre_AMGHybridSetDSCGMaxIter( (void *) solver, dscg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetPCGMaxIter( HYPRE_Solver solver,
                                 HYPRE_Int    pcg_max_its )
{
   return ( hypre_AMGHybridSetPCGMaxIter( (void *) solver, pcg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetSetupType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetSetupType( HYPRE_Solver solver,
                                HYPRE_Int    setup_type )
{
   return ( hypre_AMGHybridSetSetupType( (void *) solver, setup_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetSolverType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetSolverType( HYPRE_Solver solver,
                                 HYPRE_Int    solver_type )
{
   return ( hypre_AMGHybridSetSolverType( (void *) solver, solver_type ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetRecomputeResidual( HYPRE_Solver  solver,
                                        HYPRE_Int     recompute_residual )
{
   return ( hypre_AMGHybridSetRecomputeResidual( (void *) solver, recompute_residual ) );
}

HYPRE_Int
HYPRE_ParCSRHybridGetRecomputeResidual( HYPRE_Solver  solver,
                                        HYPRE_Int    *recompute_residual )
{
   return ( hypre_AMGHybridGetRecomputeResidual( (void *) solver, recompute_residual ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetRecomputeResidualP( HYPRE_Solver  solver,
                                         HYPRE_Int     recompute_residual_p )
{
   return ( hypre_AMGHybridSetRecomputeResidualP( (void *) solver, recompute_residual_p ) );
}

HYPRE_Int
HYPRE_ParCSRHybridGetRecomputeResidualP( HYPRE_Solver  solver,
                                         HYPRE_Int    *recompute_residual_p )
{
   return ( hypre_AMGHybridGetRecomputeResidualP( (void *) solver, recompute_residual_p ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetKDim( HYPRE_Solver solver,
                           HYPRE_Int    k_dim    )
{
   return ( hypre_AMGHybridSetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetTwoNorm( HYPRE_Solver solver,
                              HYPRE_Int    two_norm    )
{
   return ( hypre_AMGHybridSetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetStopCrit( HYPRE_Solver solver,
                               HYPRE_Int    stop_crit    )
{
   return ( hypre_AMGHybridSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetRelChange( HYPRE_Solver solver,
                                HYPRE_Int    rel_change    )
{
   return ( hypre_AMGHybridSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetPrecond( HYPRE_Solver         solver,
                              HYPRE_PtrToParSolverFcn precond,
                              HYPRE_PtrToParSolverFcn precond_setup,
                              HYPRE_Solver         precond_solver )
{
   return ( hypre_AMGHybridSetPrecond( (void *) solver,
                                       (HYPRE_Int (*)(void*, void*, void*, void*) ) precond,
                                       (HYPRE_Int (*)(void*, void*, void*, void*) ) precond_setup,
                                       (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetLogging( HYPRE_Solver solver,
                              HYPRE_Int    logging    )
{
   return ( hypre_AMGHybridSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetPrintLevel( HYPRE_Solver solver,
                                 HYPRE_Int    print_level    )
{
   return ( hypre_AMGHybridSetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetStrongThreshold( HYPRE_Solver solver,
                                      HYPRE_Real   strong_threshold    )
{
   return ( hypre_AMGHybridSetStrongThreshold( (void *) solver,
                                               strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetMaxRowSum( HYPRE_Solver solver,
                                HYPRE_Real   max_row_sum    )
{
   return ( hypre_AMGHybridSetMaxRowSum( (void *) solver, max_row_sum ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetTruncFactor( HYPRE_Solver solver,
                                  HYPRE_Real   trunc_factor    )
{
   return ( hypre_AMGHybridSetTruncFactor( (void *) solver, trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPMaxElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetPMaxElmts( HYPRE_Solver solver,
                                HYPRE_Int    p_max    )
{
   return ( hypre_AMGHybridSetPMaxElmts( (void *) solver, p_max ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetMaxLevels( HYPRE_Solver solver,
                                HYPRE_Int    max_levels    )
{
   return ( hypre_AMGHybridSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMeasureType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetMeasureType( HYPRE_Solver solver,
                                  HYPRE_Int    measure_type    )
{
   return ( hypre_AMGHybridSetMeasureType( (void *) solver, measure_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetCoarsenType( HYPRE_Solver solver,
                                  HYPRE_Int    coarsen_type    )
{
   return ( hypre_AMGHybridSetCoarsenType( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetInterpType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetInterpType( HYPRE_Solver solver,
                                 HYPRE_Int    interp_type    )
{
   return ( hypre_AMGHybridSetInterpType( (void *) solver, interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetCycleType( HYPRE_Solver solver,
                                HYPRE_Int    cycle_type    )
{
   return ( hypre_AMGHybridSetCycleType( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetNumGridSweeps( HYPRE_Solver solver,
                                    HYPRE_Int   *num_grid_sweeps    )
{
   return ( hypre_AMGHybridSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetGridRelaxType( HYPRE_Solver solver,
                                    HYPRE_Int   *grid_relax_type    )
{
   return ( hypre_AMGHybridSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetGridRelaxPoints( HYPRE_Solver solver,
                                      HYPRE_Int  **grid_relax_points    )
{
   return ( hypre_AMGHybridSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetNumSweeps( HYPRE_Solver solver,
                                HYPRE_Int    num_sweeps    )
{
   return ( hypre_AMGHybridSetNumSweeps( (void *) solver, num_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleNumSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetCycleNumSweeps( HYPRE_Solver solver,
                                     HYPRE_Int    num_sweeps,
                                     HYPRE_Int    k )
{
   return ( hypre_AMGHybridSetCycleNumSweeps( (void *) solver, num_sweeps, k ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetRelaxType( HYPRE_Solver solver,
                                HYPRE_Int    relax_type    )
{
   return ( hypre_AMGHybridSetRelaxType( (void *) solver, relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetCycleRelaxType( HYPRE_Solver solver,
                                     HYPRE_Int    relax_type,
                                     HYPRE_Int    k )
{
   return ( hypre_AMGHybridSetCycleRelaxType( (void *) solver, relax_type, k ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxOrder
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetRelaxOrder( HYPRE_Solver solver,
                                 HYPRE_Int    relax_order    )
{
   return ( hypre_AMGHybridSetRelaxOrder( (void *) solver, relax_order ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetKeepTranspose
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetKeepTranspose( HYPRE_Solver solver,
                                    HYPRE_Int    keepT    )
{
   return ( hypre_AMGHybridSetKeepTranspose( (void *) solver, keepT ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxCoarseSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetMaxCoarseSize( HYPRE_Solver solver,
                                    HYPRE_Int    max_coarse_size    )
{
   return ( hypre_AMGHybridSetMaxCoarseSize( (void *) solver, max_coarse_size ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMinCoarseSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetMinCoarseSize( HYPRE_Solver solver,
                                    HYPRE_Int    min_coarse_size    )
{
   return ( hypre_AMGHybridSetMinCoarseSize( (void *) solver, min_coarse_size ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetSeqThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetSeqThreshold( HYPRE_Solver solver,
                                   HYPRE_Int    seq_threshold    )
{
   return ( hypre_AMGHybridSetSeqThreshold( (void *) solver, seq_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetRelaxWt( HYPRE_Solver solver,
                              HYPRE_Real   relax_wt    )
{
   return ( hypre_AMGHybridSetRelaxWt( (void *) solver, relax_wt ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetLevelRelaxWt( HYPRE_Solver solver,
                                   HYPRE_Real   relax_wt,
                                   HYPRE_Int    level )
{
   return ( hypre_AMGHybridSetLevelRelaxWt( (void *) solver, relax_wt, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetOuterWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetOuterWt( HYPRE_Solver solver,
                              HYPRE_Real   outer_wt    )
{
   return ( hypre_AMGHybridSetOuterWt( (void *) solver, outer_wt ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLevelOuterWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetLevelOuterWt( HYPRE_Solver solver,
                                   HYPRE_Real   outer_wt,
                                   HYPRE_Int    level )
{
   return ( hypre_AMGHybridSetLevelOuterWt( (void *) solver, outer_wt, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetRelaxWeight( HYPRE_Solver solver,
                                  HYPRE_Real  *relax_weight    )
{
   return ( hypre_AMGHybridSetRelaxWeight( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetOmega
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetOmega( HYPRE_Solver solver,
                            HYPRE_Real  *omega    )
{
   return ( hypre_AMGHybridSetOmega( (void *) solver, omega ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetAggNumLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetAggNumLevels( HYPRE_Solver solver,
                                   HYPRE_Int    agg_num_levels    )
{
   return ( hypre_AMGHybridSetAggNumLevels( (void *) solver, agg_num_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetAggInterpType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetAggInterpType( HYPRE_Solver solver,
                                    HYPRE_Int    agg_interp_type    )
{
   return ( hypre_AMGHybridSetAggInterpType( (void *) solver, agg_interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumPaths
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetNumPaths( HYPRE_Solver solver,
                               HYPRE_Int    num_paths    )
{
   return ( hypre_AMGHybridSetNumPaths( (void *) solver, num_paths ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumFunctions
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetNumFunctions( HYPRE_Solver solver,
                                   HYPRE_Int    num_functions    )
{
   return ( hypre_AMGHybridSetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNodal
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetNodal( HYPRE_Solver solver,
                            HYPRE_Int    nodal    )
{
   return ( hypre_AMGHybridSetNodal( (void *) solver, nodal ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetDofFunc
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetDofFunc( HYPRE_Solver solver,
                              HYPRE_Int   *dof_func    )
{
   return ( hypre_AMGHybridSetDofFunc( (void *) solver, dof_func ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNonGalerkTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridSetNonGalerkinTol( HYPRE_Solver solver,
                                     HYPRE_Int   nongalerk_num_tol,
                                     HYPRE_Real  *nongalerkin_tol)
{
   return ( hypre_AMGHybridSetNonGalerkinTol( (void *) solver, nongalerk_num_tol, nongalerkin_tol ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridGetNumIterations( HYPRE_Solver solver,
                                    HYPRE_Int   *num_its    )
{
   return ( hypre_AMGHybridGetNumIterations( (void *) solver, num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridGetDSCGNumIterations( HYPRE_Solver solver,
                                        HYPRE_Int   *dscg_num_its )
{
   return ( hypre_AMGHybridGetDSCGNumIterations( (void *) solver, dscg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridGetPCGNumIterations( HYPRE_Solver solver,
                                       HYPRE_Int   *pcg_num_its )
{
   return ( hypre_AMGHybridGetPCGNumIterations( (void *) solver, pcg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRHybridGetFinalRelativeResidualNorm( HYPRE_Solver solver,
                                                HYPRE_Real  *norm    )
{
   return ( hypre_AMGHybridGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}


HYPRE_Int
HYPRE_ParCSRHybridGetSetupSolveTime( HYPRE_Solver solver,
                                     HYPRE_Real  *time    )
{
   return ( hypre_AMGHybridGetSetupSolveTime( (void *) solver, time ) );
}
