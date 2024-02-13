/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_MGRCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRCreate( HYPRE_Solver *solver )
{
   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   *solver = ( (HYPRE_Solver) hypre_MGRCreate( ) );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRDestroy( HYPRE_Solver solver )
{
   return ( hypre_MGRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetup( HYPRE_Solver solver,
                HYPRE_ParCSRMatrix A,
                HYPRE_ParVector b,
                HYPRE_ParVector x      )
{
   if (!A)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   return ( hypre_MGRSetup( (void *) solver,
                            (hypre_ParCSRMatrix *) A,
                            (hypre_ParVector *) b,
                            (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSolve( HYPRE_Solver solver,
                HYPRE_ParCSRMatrix A,
                HYPRE_ParVector b,
                HYPRE_ParVector x      )
{
   if (!A)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!b)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (!x)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   return ( hypre_MGRSolve( (void *) solver,
                            (hypre_ParCSRMatrix *) A,
                            (hypre_ParVector *) b,
                            (hypre_ParVector *) x ) );
}

#ifdef HYPRE_USING_DSUPERLU
/*--------------------------------------------------------------------------
 * HYPRE_MGRDirectSolverCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRDirectSolverCreate( HYPRE_Solver *solver )
{
   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   *solver = ( (HYPRE_Solver) hypre_MGRDirectSolverCreate( ) );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRDirectSolverDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRDirectSolverDestroy( HYPRE_Solver solver )
{
   return ( hypre_MGRDirectSolverDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRDirectSolverSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRDirectSolverSetup( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x      )
{
   return ( hypre_MGRDirectSolverSetup( (void *) solver,
                                        (hypre_ParCSRMatrix *) A,
                                        (hypre_ParVector *) b,
                                        (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRDirectSolverSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRDirectSolverSolve( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x      )
{
   return ( hypre_MGRDirectSolverSolve( (void *) solver,
                                        (hypre_ParCSRMatrix *) A,
                                        (hypre_ParVector *) b,
                                        (hypre_ParVector *) x ) );
}
#endif

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCpointsByContiguousBlock
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetCpointsByContiguousBlock( HYPRE_Solver solver,
                                      HYPRE_Int  block_size,
                                      HYPRE_Int  max_num_levels,
                                      HYPRE_BigInt  *idx_array,
                                      HYPRE_Int  *block_num_coarse_points,
                                      HYPRE_Int  **block_coarse_indexes)
{
   return ( hypre_MGRSetCpointsByContiguousBlock( (void *) solver, block_size, max_num_levels,
                                                  idx_array, block_num_coarse_points, block_coarse_indexes));
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCpointsByBlock
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetCpointsByBlock( HYPRE_Solver solver,
                            HYPRE_Int  block_size,
                            HYPRE_Int  max_num_levels,
                            HYPRE_Int *block_num_coarse_points,
                            HYPRE_Int  **block_coarse_indexes)
{
   return ( hypre_MGRSetCpointsByBlock( (void *) solver, block_size, max_num_levels,
                                        block_num_coarse_points, block_coarse_indexes));
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCpointsByPointMarkerArray
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetCpointsByPointMarkerArray( HYPRE_Solver solver,
                                       HYPRE_Int  block_size,
                                       HYPRE_Int  max_num_levels,
                                       HYPRE_Int  *num_block_coarse_points,
                                       HYPRE_Int  **lvl_block_coarse_indexes,
                                       HYPRE_Int  *point_marker_array)
{
   return ( hypre_MGRSetCpointsByPointMarkerArray( (void *) solver, block_size, max_num_levels,
                                                   num_block_coarse_points, lvl_block_coarse_indexes, point_marker_array));
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetNonCpointsToFpoints
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetNonCpointsToFpoints( HYPRE_Solver solver, HYPRE_Int nonCptToFptFlag)
{
   return hypre_MGRSetNonCpointsToFpoints((void *) solver, nonCptToFptFlag);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetFSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetFSolver(HYPRE_Solver          solver,
                    HYPRE_PtrToParSolverFcn  fine_grid_solver_solve,
                    HYPRE_PtrToParSolverFcn  fine_grid_solver_setup,
                    HYPRE_Solver          fsolver )
{
   return ( hypre_MGRSetFSolver( (void *) solver,
                                 (HYPRE_Int (*)(void*, void*, void*, void*)) fine_grid_solver_solve,
                                 (HYPRE_Int (*)(void*, void*, void*, void*)) fine_grid_solver_setup,
                                 (void *) fsolver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetFSolverAtLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetFSolverAtLevel(HYPRE_Int     level,
                           HYPRE_Solver  solver,
                           HYPRE_Solver  fsolver )
{
   return ( hypre_MGRSetFSolverAtLevel( level,
                                        (void *) solver,
                                        (void *) fsolver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRBuildAff
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRBuildAff(HYPRE_ParCSRMatrix A,
                  HYPRE_Int *CF_marker,
                  HYPRE_Int debug_flag,
                  HYPRE_ParCSRMatrix *A_ff)
{
   return (hypre_MGRBuildAff(A, CF_marker, debug_flag, A_ff));
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCoarseSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetCoarseSolver(HYPRE_Solver          solver,
                         HYPRE_PtrToParSolverFcn  coarse_grid_solver_solve,
                         HYPRE_PtrToParSolverFcn  coarse_grid_solver_setup,
                         HYPRE_Solver          coarse_grid_solver )
{
   return ( hypre_MGRSetCoarseSolver( (void *) solver,
                                      (HYPRE_Int (*)(void*, void*, void*, void*)) coarse_grid_solver_solve,
                                      (HYPRE_Int (*)(void*, void*, void*, void*)) coarse_grid_solver_setup,
                                      (void *) coarse_grid_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetMaxCoarseLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetMaxCoarseLevels( HYPRE_Solver solver, HYPRE_Int maxlev )
{
   return hypre_MGRSetMaxCoarseLevels(solver, maxlev);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetBlockSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetBlockSize( HYPRE_Solver solver, HYPRE_Int bsize )
{
   return hypre_MGRSetBlockSize(solver, bsize );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetReservedCoarseNodes
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetReservedCoarseNodes( HYPRE_Solver solver, HYPRE_Int reserved_coarse_size,
                                 HYPRE_BigInt *reserved_coarse_indexes )
{
   return hypre_MGRSetReservedCoarseNodes(solver, reserved_coarse_size, reserved_coarse_indexes );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetReservedCpointsLevelToKeep
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetReservedCpointsLevelToKeep( HYPRE_Solver solver, HYPRE_Int level)
{
   return hypre_MGRSetReservedCpointsLevelToKeep((void *) solver, level);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetRestrictType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetRestrictType(HYPRE_Solver solver, HYPRE_Int restrict_type )
{
   return hypre_MGRSetRestrictType(solver, restrict_type );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelRestrictType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetLevelRestrictType( HYPRE_Solver solver, HYPRE_Int *restrict_type )
{
   return hypre_MGRSetLevelRestrictType( solver, restrict_type );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetFRelaxMethod
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetFRelaxMethod(HYPRE_Solver solver, HYPRE_Int relax_method )
{
   return hypre_MGRSetFRelaxMethod(solver, relax_method );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelFRelaxMethod
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetLevelFRelaxMethod( HYPRE_Solver solver, HYPRE_Int *relax_method )
{
   return hypre_MGRSetLevelFRelaxMethod( solver, relax_method );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelFRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetLevelFRelaxType( HYPRE_Solver solver, HYPRE_Int *relax_type )
{
   return hypre_MGRSetLevelFRelaxType( solver, relax_type );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCoarseGridMethod
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetCoarseGridMethod( HYPRE_Solver solver, HYPRE_Int *cg_method )
{
   return hypre_MGRSetCoarseGridMethod( solver, cg_method );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelFRelaxNumFunctions
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetLevelFRelaxNumFunctions( HYPRE_Solver solver, HYPRE_Int *num_functions )
{
   return hypre_MGRSetLevelFRelaxNumFunctions( solver, num_functions );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetRelaxType(HYPRE_Solver solver, HYPRE_Int relax_type )
{
   return hypre_MGRSetRelaxType(solver, relax_type );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetNumRelaxSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetNumRelaxSweeps( HYPRE_Solver solver, HYPRE_Int nsweeps )
{
   return hypre_MGRSetNumRelaxSweeps(solver, nsweeps);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelNumRelaxSweeps
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetLevelNumRelaxSweeps( HYPRE_Solver solver, HYPRE_Int *nsweeps )
{
   return hypre_MGRSetLevelNumRelaxSweeps(solver, nsweeps);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetInterpType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetInterpType( HYPRE_Solver solver, HYPRE_Int interpType )
{
   return hypre_MGRSetInterpType(solver, interpType);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelInterpType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetLevelInterpType( HYPRE_Solver solver, HYPRE_Int *interpType )
{
   return hypre_MGRSetLevelInterpType(solver, interpType);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetNumInterpSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetNumInterpSweeps( HYPRE_Solver solver, HYPRE_Int nsweeps )
{
   return hypre_MGRSetNumInterpSweeps(solver, nsweeps);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetNumRestrictSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetNumRestrictSweeps( HYPRE_Solver solver, HYPRE_Int nsweeps )
{
   return hypre_MGRSetNumRestrictSweeps(solver, nsweeps);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetTruncateCoarseGridThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetTruncateCoarseGridThreshold( HYPRE_Solver solver, HYPRE_Real threshold)
{
   return hypre_MGRSetTruncateCoarseGridThreshold( solver, threshold );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetBlockJacobiBlockSize
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetBlockJacobiBlockSize( HYPRE_Solver solver, HYPRE_Int blk_size )
{
   return hypre_MGRSetBlockJacobiBlockSize(solver, blk_size);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetFrelaxPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetFrelaxPrintLevel( HYPRE_Solver solver, HYPRE_Int print_level )
{
   return hypre_MGRSetFrelaxPrintLevel( solver, print_level );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCoarseGridPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetCoarseGridPrintLevel( HYPRE_Solver solver, HYPRE_Int print_level )
{
   return hypre_MGRSetCoarseGridPrintLevel( solver, print_level );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetPrintLevel( HYPRE_Solver solver, HYPRE_Int print_level )
{
   return hypre_MGRSetPrintLevel( solver, print_level );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetLogging( HYPRE_Solver solver, HYPRE_Int logging )
{
   return hypre_MGRSetLogging(solver, logging );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetMaxIter( HYPRE_Solver solver, HYPRE_Int max_iter )
{
   return hypre_MGRSetMaxIter( solver, max_iter );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetTol( HYPRE_Solver solver, HYPRE_Real tol )
{
   return hypre_MGRSetTol( solver, tol );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetMaxGlobalsmoothIters
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetMaxGlobalSmoothIters( HYPRE_Solver solver, HYPRE_Int max_iter )
{
   return hypre_MGRSetMaxGlobalSmoothIters(solver, max_iter);
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelsmoothIters
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetLevelSmoothIters( HYPRE_Solver solver,
                              HYPRE_Int *smooth_iters )
{
   return hypre_MGRSetLevelSmoothIters(solver, smooth_iters);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetGlobalsmoothType
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetGlobalSmoothType( HYPRE_Solver solver, HYPRE_Int smooth_type )
{
   return hypre_MGRSetGlobalSmoothType(solver, smooth_type);
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelsmoothType
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetLevelSmoothType( HYPRE_Solver solver,
                             HYPRE_Int *smooth_type )
{
   return hypre_MGRSetLevelSmoothType(solver, smooth_type);
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetGlobalSmoothCycle
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetGlobalSmoothCycle( HYPRE_Solver solver,
                               HYPRE_Int global_smooth_cycle )
{
   return hypre_MGRSetGlobalSmoothCycle(solver, global_smooth_cycle);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetPMaxElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetPMaxElmts( HYPRE_Solver solver, HYPRE_Int P_max_elmts )
{
   return hypre_MGRSetPMaxElmts(solver, P_max_elmts);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelPMaxElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetLevelPMaxElmts( HYPRE_Solver solver, HYPRE_Int *P_max_elmts )
{
   return hypre_MGRSetLevelPMaxElmts(solver, P_max_elmts);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRGetCoarseGridConvergenceFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRGetCoarseGridConvergenceFactor( HYPRE_Solver solver, HYPRE_Real *conv_factor )
{
   return hypre_MGRGetCoarseGridConvergenceFactor( solver, conv_factor );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRGetNumIterations( HYPRE_Solver solver, HYPRE_Int *num_iterations )
{
   return hypre_MGRGetNumIterations( solver, num_iterations );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRGetFinalRelativeResidualNorm( HYPRE_Solver solver, HYPRE_Real *res_norm )
{
   return hypre_MGRGetFinalRelativeResidualNorm(solver, res_norm);
}
