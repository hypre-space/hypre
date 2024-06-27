/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Mixed precision function protos
 *
 *****************************************************************************/

#ifndef HYPRE_PARCSR_LS_MP_HEADER
#define HYPRE_PARCSR_LS_MP_HEADER

#include "_hypre_parcsr_ls.h"

/* Mixed precision function protos */
/* hypre_parcsr_ls_mp.h */

#ifdef HYPRE_MIXED_PRECISION

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int HYPRE_BoomerAMGSetup_mp(HYPRE_Solver       solver,
                                  HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector    b,
                                  HYPRE_ParVector    x);

HYPRE_Int HYPRE_BoomerAMGSolve_mp(HYPRE_Solver       solver,
                                  HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector    b,
                                  HYPRE_ParVector    x);
HYPRE_Int HYPRE_MPAMGPrecSetup_mp(HYPRE_Solver       solver,
                                  HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector    b,
                                  HYPRE_ParVector    x);

HYPRE_Int HYPRE_MPAMGPrecSolve_mp(HYPRE_Solver       solver,
                                  HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector    b,
                                  HYPRE_ParVector    x);
HYPRE_Int hypre_MPAMGCycle_mp( void *amg_vdata, hypre_ParVector **F_array, hypre_ParVector **U_array );

HYPRE_Int HYPRE_MPAMGCreate_mp(HYPRE_Solver *solver);

HYPRE_Int HYPRE_MPAMGDestroy_mp(HYPRE_Solver solver);

HYPRE_Int HYPRE_MPAMGSetup_mp(HYPRE_Solver       solver,
                              HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector    b,
                              HYPRE_ParVector    x);

HYPRE_Int HYPRE_MPAMGSolve_mp(HYPRE_Solver       solver,
                              HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector    b,
                              HYPRE_ParVector    x);

HYPRE_Int HYPRE_MPAMGGetResidual_mp(HYPRE_Solver     solver,
                                    HYPRE_ParVector *residual);

HYPRE_Int HYPRE_MPAMGGetNumIterations_mp(HYPRE_Solver  solver,
                                         HYPRE_Int    *num_iterations);

HYPRE_Int HYPRE_MPAMGGetCumNnzAP_mp(HYPRE_Solver  solver,
                                    hypre_double *cum_nnz_AP);

HYPRE_Int HYPRE_MPAMGSetCumNnzAP_mp(HYPRE_Solver  solver,
                                    hypre_double  cum_nnz_AP);

HYPRE_Int HYPRE_MPAMGGetFinalRelativeResidualNorm_mp(HYPRE_Solver  solver,
                                                     HYPRE_Real   *rel_resid_norm);

HYPRE_Int HYPRE_MPAMGSetNumFunctions_mp(HYPRE_Solver solver,
                                        HYPRE_Int          num_functions);

HYPRE_Int HYPRE_MPAMGSetDofFunc_mp(HYPRE_Solver  solver,
                                   HYPRE_Int    *dof_func);

HYPRE_Int HYPRE_MPAMGSetTol_mp(HYPRE_Solver solver,
                               HYPRE_Real   tol);

HYPRE_Int HYPRE_MPAMGSetMaxIter_mp(HYPRE_Solver solver,
                                   HYPRE_Int          max_iter);

HYPRE_Int HYPRE_MPAMGSetMinIter_mp(HYPRE_Solver solver,
                                   HYPRE_Int    min_iter);

HYPRE_Int HYPRE_MPAMGSetMaxCoarseSize_mp(HYPRE_Solver solver,
                                         HYPRE_Int    max_coarse_size);

HYPRE_Int HYPRE_MPAMGSetMinCoarseSize_mp(HYPRE_Solver solver,
                                         HYPRE_Int    min_coarse_size);

HYPRE_Int HYPRE_MPAMGSetMaxLevels_mp(HYPRE_Solver solver,
                                     HYPRE_Int    max_levels);

HYPRE_Int HYPRE_MPAMGSetCoarsenCutFactor_mp(HYPRE_Solver solver,
                                            HYPRE_Int    coarsen_cut_factor);

HYPRE_Int HYPRE_MPAMGSetStrongThreshold_mp(HYPRE_Solver solver,
                                           HYPRE_Real   strong_threshold);

HYPRE_Int HYPRE_MPAMGSetMaxRowSum_mp(HYPRE_Solver solver,
                                     HYPRE_Real    max_row_sum);

HYPRE_Int HYPRE_MPAMGSetCoarsenType_mp(HYPRE_Solver solver,
                                       HYPRE_Int    coarsen_type);

HYPRE_Int HYPRE_MPAMGSetMeasureType_mp(HYPRE_Solver solver,
                                       HYPRE_Int    measure_type);

HYPRE_Int HYPRE_MPAMGSetAggNumLevels_mp(HYPRE_Solver solver,
                                        HYPRE_Int    agg_num_levels);

HYPRE_Int HYPRE_MPAMGSetNumPaths_mp(HYPRE_Solver solver,
                                    HYPRE_Int    num_paths);

HYPRE_Int HYPRE_MPAMGSetNodal_mp(HYPRE_Solver solver,
                                 HYPRE_Int    nodal);
HYPRE_Int HYPRE_MPAMGSetNodalDiag_mp(HYPRE_Solver solver,
                                     HYPRE_Int    nodal_diag);

HYPRE_Int HYPRE_MPAMGSetKeepSameSign_mp(HYPRE_Solver solver,
                                        HYPRE_Int    keep_same_sign);

HYPRE_Int HYPRE_MPAMGSetInterpType_mp(HYPRE_Solver solver,
                                      HYPRE_Int    interp_type);

HYPRE_Int HYPRE_MPAMGSetTruncFactor_mp(HYPRE_Solver solver,
                                       HYPRE_Real   trunc_factor);

HYPRE_Int HYPRE_MPAMGSetPMaxElmts_mp(HYPRE_Solver solver,
                                     HYPRE_Int    P_max_elmts);

HYPRE_Int HYPRE_MPAMGSetSepWeight_mp(HYPRE_Solver solver,
                                     HYPRE_Int    sep_weight);

HYPRE_Int HYPRE_MPAMGSetAggInterpType_mp(HYPRE_Solver solver,
                                         HYPRE_Int    agg_interp_type);

HYPRE_Int HYPRE_MPAMGSetAggTruncFactor_mp(HYPRE_Solver solver,
                                          HYPRE_Real   agg_trunc_factor);

HYPRE_Int HYPRE_MPAMGSetAggP12TruncFactor_mp(HYPRE_Solver solver,
                                             HYPRE_Real   agg_P12_trunc_factor);

HYPRE_Int HYPRE_MPAMGSetAggPMaxElmts_mp(HYPRE_Solver solver,
                                        HYPRE_Int    agg_P_max_elmts);

HYPRE_Int HYPRE_MPAMGSetAggP12MaxElmts_mp(HYPRE_Solver solver,
                                          HYPRE_Int    agg_P12_max_elmts);

HYPRE_Int HYPRE_MPAMGSetCycleType_mp(HYPRE_Solver solver,
                                     HYPRE_Int    cycle_type);
HYPRE_Int
HYPRE_MPAMGSetFCycle_mp( HYPRE_Solver solver,
                         HYPRE_Int    fcycle  );

HYPRE_Int HYPRE_MPAMGSetNumGridSweeps_mp(HYPRE_Solver  solver,
                                         HYPRE_Int    *num_grid_sweeps);

HYPRE_Int HYPRE_MPAMGSetNumSweeps_mp(HYPRE_Solver  solver,
                                     HYPRE_Int     num_sweeps);

HYPRE_Int HYPRE_MPAMGSetCycleNumSweeps_mp(HYPRE_Solver  solver,
                                          HYPRE_Int     num_sweeps,
                                          HYPRE_Int     k);

HYPRE_Int HYPRE_MPAMGSetRelaxType_mp(HYPRE_Solver  solver,
                                     HYPRE_Int     relax_type);

HYPRE_Int HYPRE_MPAMGSetCycleRelaxType_mp(HYPRE_Solver  solver,
                                          HYPRE_Int     relax_type,
                                          HYPRE_Int     k);

HYPRE_Int HYPRE_MPAMGSetRelaxOrder_mp(HYPRE_Solver  solver,
                                       HYPRE_Int     relax_order);

HYPRE_Int HYPRE_MPAMGSetGridRelaxPoints(HYPRE_Solver   solver,
                                            HYPRE_Int    **grid_relax_points);

HYPRE_Int HYPRE_MPAMGSetRelaxWt_mp(HYPRE_Solver  solver,
                                   HYPRE_Real    relax_weight);

HYPRE_Int HYPRE_MPAMGSetLevelRelaxWt_mp(HYPRE_Solver  solver,
                                        HYPRE_Real    relax_weight,
                                        HYPRE_Int     level);

HYPRE_Int HYPRE_MPAMGSetOuterWt_mp(HYPRE_Solver  solver,
                                   HYPRE_Real    omega);

HYPRE_Int HYPRE_MPAMGSetLevelOuterWt_mp(HYPRE_Solver  solver,
                                        HYPRE_Real    omega,
                                        HYPRE_Int     level);

HYPRE_Int HYPRE_MPAMGSetPrintLevel_mp(HYPRE_Solver solver,
                                      HYPRE_Int    print_level);

HYPRE_Int HYPRE_MPAMGSetLogging_mp(HYPRE_Solver solver,
                                   HYPRE_Int    logging);

HYPRE_Int HYPRE_MPAMGSetDebugFlag_mp(HYPRE_Solver solver,
                                     HYPRE_Int    debug_flag);

HYPRE_Int HYPRE_MPAMGSetRAP2_mp(HYPRE_Solver solver,
                                HYPRE_Int    rap2);

HYPRE_Int HYPRE_MPAMGSetModuleRAP2_mp(HYPRE_Solver solver,
                                      HYPRE_Int    mod_rap2);

HYPRE_Int HYPRE_MPAMGSetKeepTranspose_mp(HYPRE_Solver solver,
                                         HYPRE_Int    keepTranspose);
HYPRE_Int HYPRE_MPAMGSetSabs_mp (HYPRE_Solver solver,
                                 HYPRE_Int Sabs );

HYPRE_Int HYPRE_MPAMGSetPrecisionArray_mp (HYPRE_Solver solver,
                                           HYPRE_Precision *precision_array);

HYPRE_Int hypre_MPAMGSetupStats_mp (void * amg_vdata, hypre_ParCSRMatrix *A);

HYPRE_Int hypre_MPAMGWriteSolverParams_mp (void * data);


/* par_mpamg_mp.c */
/*void *hypre_MPAMGCreate_mp ( void );
HYPRE_Int hypre_MPAMGDestroy_mp ( void *data );
HYPRE_Int hypre_MPAMGSetMaxLevels_mp ( void *data, HYPRE_Int max_levels );
HYPRE_Int hypre_MPAMGGetMaxLevels_mp ( void *data, HYPRE_Int *max_levels );
HYPRE_Int hypre_MPAMGSetMaxCoarseSize_mp ( void *data, HYPRE_Int max_coarse_size );
HYPRE_Int hypre_MPAMGGetMaxCoarseSize_mp ( void *data, HYPRE_Int *max_coarse_size );
HYPRE_Int hypre_MPAMGSetMinCoarseSize_mp ( void *data, HYPRE_Int min_coarse_size );
HYPRE_Int hypre_MPAMGGetMinCoarseSize_mp ( void *data, HYPRE_Int *min_coarse_size );
HYPRE_Int hypre_MPAMGSetCoarsenCutFactor( void *data, HYPRE_Int coarsen_cut_factor );
HYPRE_Int hypre_MPAMGGetCoarsenCutFactor( void *data, HYPRE_Int *coarsen_cut_factor );
HYPRE_Int hypre_MPAMGSetStrongThreshold_mp ( void *data, HYPRE_Real strong_threshold );
HYPRE_Int hypre_MPAMGGetStrongThreshold_mp ( void *data, HYPRE_Real *strong_threshold );
HYPRE_Int hypre_MPAMGSetSabs_mp ( void *data, HYPRE_Int Sabs );
HYPRE_Int hypre_MPAMGSetMaxRowSum_mp ( void *data, HYPRE_Real max_row_sum );
HYPRE_Int hypre_MPAMGGetMaxRowSum_mp ( void *data, HYPRE_Real *max_row_sum );
HYPRE_Int hypre_MPAMGSetTruncFactor_mp ( void *data, HYPRE_Real trunc_factor );
HYPRE_Int hypre_MPAMGGetTruncFactor_mp ( void *data, HYPRE_Real *trunc_factor );
HYPRE_Int hypre_MPAMGSetPMaxElmts_mp ( void *data, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_MPAMGGetPMaxElmts_mp ( void *data, HYPRE_Int *P_max_elmts );
HYPRE_Int hypre_MPAMGSetInterpType_mp ( void *data, HYPRE_Int interp_type );
HYPRE_Int hypre_MPAMGGetInterpType_mp ( void *data, HYPRE_Int *interp_type );
HYPRE_Int hypre_MPAMGSetSepWeight_mp ( void *data, HYPRE_Int sep_weight );
HYPRE_Int hypre_MPAMGSetMinIter_mp ( void *data, HYPRE_Int min_iter );
HYPRE_Int hypre_MPAMGGetMinIter_mp ( void *data, HYPRE_Int *min_iter );
HYPRE_Int hypre_MPAMGSetMaxIter_mp ( void *data, HYPRE_Int max_iter );
HYPRE_Int hypre_MPAMGGetMaxIter_mp ( void *data, HYPRE_Int *max_iter );
HYPRE_Int hypre_MPAMGSetCoarsenType_mp ( void *data, HYPRE_Int coarsen_type );
HYPRE_Int hypre_MPAMGGetCoarsenType_mp ( void *data, HYPRE_Int *coarsen_type );
HYPRE_Int hypre_MPAMGSetMeasureType_mp ( void *data, HYPRE_Int measure_type );
HYPRE_Int hypre_MPAMGGetMeasureType_mp ( void *data, HYPRE_Int *measure_type );
HYPRE_Int hypre_MPAMGSetFCycle_mp ( void *data, HYPRE_Int fcycle );
HYPRE_Int hypre_MPAMGGetFCycle_mp ( void *data, HYPRE_Int *fcycle );
HYPRE_Int hypre_MPAMGSetCycleType_mp ( void *data, HYPRE_Int cycle_type );
HYPRE_Int hypre_MPAMGGetCycleType_mp ( void *data, HYPRE_Int *cycle_type );
HYPRE_Int hypre_MPAMGSetTol_mp ( void *data, HYPRE_Real tol );
HYPRE_Int hypre_MPAMGGetTol_mp ( void *data, HYPRE_Real *tol );
HYPRE_Int hypre_MPAMGSetNumSweeps_mp ( void *data, HYPRE_Int num_sweeps );
HYPRE_Int hypre_MPAMGSetCycleNumSweeps_mp ( void *data, HYPRE_Int num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_MPAMGGetCycleNumSweeps_mp ( void *data, HYPRE_Int *num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_MPAMGSetNumGridSweeps_mp ( void *data, HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_MPAMGGetNumGridSweeps_mp ( void *data, HYPRE_Int **num_grid_sweeps );
HYPRE_Int hypre_MPAMGSetRelaxType_mp ( void *data, HYPRE_Int relax_type );
HYPRE_Int hypre_MPAMGSetCycleRelaxType_mp ( void *data, HYPRE_Int relax_type, HYPRE_Int k );
HYPRE_Int hypre_MPAMGGetCycleRelaxType_mp ( void *data, HYPRE_Int *relax_type, HYPRE_Int k );
HYPRE_Int hypre_MPAMGSetRelaxOrder_mp ( void *data, HYPRE_Int relax_order );
HYPRE_Int hypre_MPAMGGetRelaxOrder_mp ( void *data, HYPRE_Int *relax_order );
HYPRE_Int hypre_MPAMGSetGridRelaxType_mp ( void *data, HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_MPAMGGetGridRelaxType_mp ( void *data, HYPRE_Int **grid_relax_type );
HYPRE_Int hypre_MPAMGSetGridRelaxPoints_mp ( void *data, HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_MPAMGGetGridRelaxPoints_mp ( void *data, HYPRE_Int ***grid_relax_points );
HYPRE_Int hypre_MPAMGSetRelaxWt_mp ( void *data, HYPRE_Real relax_weight );
HYPRE_Int hypre_MPAMGSetLevelRelaxWt_mp ( void *data, HYPRE_Real relax_weight, HYPRE_Int level );
HYPRE_Int hypre_MPAMGGetLevelRelaxWt_mp ( void *data, HYPRE_Real *relax_weight, HYPRE_Int level );
HYPRE_Int hypre_MPAMGSetOuterWt_mp ( void *data, HYPRE_Real omega );
HYPRE_Int hypre_MPAMGSetLevelOuterWt_mp ( void *data, HYPRE_Real omega, HYPRE_Int level );
HYPRE_Int hypre_MPAMGGetLevelOuterWt_mp ( void *data, HYPRE_Real *omega, HYPRE_Int level );
HYPRE_Int hypre_MPAMGSetLogging_mp ( void *data, HYPRE_Int logging );
HYPRE_Int hypre_MPAMGGetLogging_mp ( void *data, HYPRE_Int *logging );
HYPRE_Int hypre_MPAMGSetPrintLevel_mp ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_MPAMGGetPrintLevel_mp ( void *data, HYPRE_Int *print_level );
HYPRE_Int hypre_MPAMGSetNumIterations_mp ( void *data, HYPRE_Int num_iterations );
HYPRE_Int hypre_MPAMGSetDebugFlag_mp ( void *data, HYPRE_Int debug_flag );
HYPRE_Int hypre_MPAMGGetDebugFlag_mp ( void *data, HYPRE_Int *debug_flag );
HYPRE_Int hypre_MPAMGSetNumFunctions_mp ( void *data, HYPRE_Int num_functions );
HYPRE_Int hypre_MPAMGGetNumFunctions_mp ( void *data, HYPRE_Int *num_functions );
HYPRE_Int hypre_MPAMGSetNodal_mp ( void *data, HYPRE_Int nodal );
HYPRE_Int hypre_MPAMGSetNodalLevels_mp ( void *data, HYPRE_Int nodal_levels );
HYPRE_Int hypre_MPAMGSetNodalDiag_mp ( void *data, HYPRE_Int nodal );
HYPRE_Int hypre_MPAMGSetKeepSameSign_mp ( void *data, HYPRE_Int keep_same_sign );
HYPRE_Int hypre_MPAMGSetNumPaths_mp ( void *data, HYPRE_Int num_paths );
HYPRE_Int hypre_MPAMGSetAggNumLevels_mp ( void *data, HYPRE_Int agg_num_levels );
HYPRE_Int hypre_MPAMGSetAggInterpType_mp ( void *data, HYPRE_Int agg_interp_type );
HYPRE_Int hypre_MPAMGSetAggPMaxElmts_mp ( void *data, HYPRE_Int agg_P_max_elmts );
HYPRE_Int hypre_MPAMGSetAggP12MaxElmts_mp ( void *data, HYPRE_Int agg_P12_max_elmts );
HYPRE_Int hypre_MPAMGSetAggTruncFactor_mp ( void *data, HYPRE_Real agg_trunc_factor );
HYPRE_Int hypre_MPAMGSetAggP12TruncFactor_mp ( void *data, HYPRE_Real agg_P12_trunc_factor );
HYPRE_Int hypre_MPAMGSetDofFunc_mp ( void *data, HYPRE_Int *dof_func );
HYPRE_Int hypre_MPAMGGetNumIterations_mp ( void *data, HYPRE_Int *num_iterations );
HYPRE_Int hypre_MPAMGGetCumNumIterations_mp ( void *data, HYPRE_Int *cum_num_iterations );
HYPRE_Int hypre_MPAMGGetResidual_mp ( void *data, hypre_ParVector **resid );
HYPRE_Int hypre_MPAMGGetRelResidualNorm_mp ( void *data, HYPRE_Real *rel_resid_norm );
HYPRE_Int hypre_MPAMGSetRAP2_mp ( void *data, HYPRE_Int rap2 );
HYPRE_Int hypre_MPAMGSetModuleRAP2_mp ( void *data, HYPRE_Int mod_rap2 );
HYPRE_Int hypre_MPAMGSetKeepTranspose_mp ( void *data, HYPRE_Int keepTranspose );
HYPRE_Int hypre_MPAMGSetCumNnzAP_mp ( void *data, hypre_double cum_nnz_AP );
HYPRE_Int hypre_MPAMGGetCumNnzAP_mp ( void *data, hypre_double *cum_nnz_AP );
HYPRE_Int hypre_MPAMGSetPrecisionArray_mp (void *data, HYPRE_Precision *precision_array);*/

#ifdef __cplusplus
}
#endif

#endif


#endif
