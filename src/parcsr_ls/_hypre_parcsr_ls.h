/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <HYPRE_config.h>

#include "HYPRE_parcsr_ls.h"

#ifndef hypre_PARCSR_LS_HEADER
#define hypre_PARCSR_LS_HEADER

#include "_hypre_utilities.h"
#include "krylov.h"
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_lobpcg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { HYPRE_Int prev; HYPRE_Int next; } Link;

#ifndef hypre_ParAMG_DATA_HEADER
#define hypre_ParAMG_DATA_HEADER

#define CUMNUMIT


#include "par_csr_block_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_ParAMGData
 *--------------------------------------------------------------------------*/

typedef struct
{

   /* setup params */
   HYPRE_Int     max_levels;
   HYPRE_Real   strong_threshold;
   HYPRE_Real   strong_thresholdR; /* theta for build R: defines strong F neighbors */
   HYPRE_Real   filter_thresholdR; /* theta for filtering R  */
   HYPRE_Real   max_row_sum;
   HYPRE_Real   trunc_factor;
   HYPRE_Real   agg_trunc_factor;
   HYPRE_Real   agg_P12_trunc_factor;
   HYPRE_Real   jacobi_trunc_threshold;
   HYPRE_Real   S_commpkg_switch;
   HYPRE_Real   CR_rate;
   HYPRE_Real   CR_strong_th;
   HYPRE_Real   A_drop_tol;
   HYPRE_Int    A_drop_type;
   HYPRE_Int      measure_type;
   HYPRE_Int      setup_type;
   HYPRE_Int      coarsen_type;
   HYPRE_Int      P_max_elmts;
   HYPRE_Int      interp_type;
   HYPRE_Int      sep_weight;
   HYPRE_Int      agg_interp_type;
   HYPRE_Int      agg_P_max_elmts;
   HYPRE_Int      agg_P12_max_elmts;
   HYPRE_Int      restr_par;
   HYPRE_Int      is_triangular;
   HYPRE_Int      gmres_switch;
   HYPRE_Int      agg_num_levels;
   HYPRE_Int      num_paths;
   HYPRE_Int      post_interp_type;
   HYPRE_Int      num_CR_relax_steps;
   HYPRE_Int      IS_type;
   HYPRE_Int      CR_use_CG;
   HYPRE_Int      cgc_its;
   HYPRE_Int      max_coarse_size;
   HYPRE_Int      min_coarse_size;
   HYPRE_Int      seq_threshold;
   HYPRE_Int      redundant;
   HYPRE_Int      participate;
   HYPRE_Int      Sabs;

   /* solve params */
   HYPRE_Int      max_iter;
   HYPRE_Int      min_iter;
   HYPRE_Int      fcycle;
   HYPRE_Int      cycle_type;
   HYPRE_Int     *num_grid_sweeps;
   HYPRE_Int     *grid_relax_type;
   HYPRE_Int    **grid_relax_points;
   HYPRE_Int      relax_order;
   HYPRE_Int      user_coarse_relax_type;
   HYPRE_Int      user_relax_type;
   HYPRE_Int      user_num_sweeps;
   HYPRE_Real     user_relax_weight;
   HYPRE_Real     outer_wt;
   HYPRE_Real    *relax_weight;
   HYPRE_Real    *omega;
   HYPRE_Int      converge_type;
   HYPRE_Real     tol;

   /* problem data */
   hypre_ParCSRMatrix  *A;
   HYPRE_Int      num_variables;
   HYPRE_Int      num_functions;
   HYPRE_Int      nodal;
   HYPRE_Int      nodal_levels;
   HYPRE_Int      nodal_diag;
   HYPRE_Int      num_points;
   HYPRE_Int     *dof_func;
   HYPRE_Int     *dof_point;
   HYPRE_Int     *point_dof_map;

   /* data generated in the setup phase */
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   hypre_ParCSRMatrix **P_array;
   hypre_ParCSRMatrix **R_array;
   HYPRE_Int          **CF_marker_array;
   HYPRE_Int          **dof_func_array;
   HYPRE_Int          **dof_point_array;
   HYPRE_Int          **point_dof_map_array;
   HYPRE_Int            num_levels;
   HYPRE_Real         **l1_norms;

   /* Block data */
   hypre_ParCSRBlockMatrix **A_block_array;
   hypre_ParCSRBlockMatrix **P_block_array;
   hypre_ParCSRBlockMatrix **R_block_array;

   HYPRE_Int block_mode;

   /* data for more complex smoothers */
   HYPRE_Int            smooth_num_levels;
   HYPRE_Int            smooth_type;
   HYPRE_Solver        *smoother;
   HYPRE_Int            smooth_num_sweeps;
   HYPRE_Int            schw_variant;
   HYPRE_Int            schw_overlap;
   HYPRE_Int            schw_domain_type;
   HYPRE_Real           schwarz_rlx_weight;
   HYPRE_Int            schwarz_use_nonsymm;
   HYPRE_Int            ps_sym;
   HYPRE_Int            ps_level;
   HYPRE_Int            pi_max_nz_per_row;
   HYPRE_Int            eu_level;
   HYPRE_Int            eu_bj;
   HYPRE_Real           ps_threshold;
   HYPRE_Real           ps_filter;
   HYPRE_Real           pi_drop_tol;
   HYPRE_Real           eu_sparse_A;
   char                *euclidfile;

   HYPRE_Real          *max_eig_est;
   HYPRE_Real          *min_eig_est;
   HYPRE_Int            cheby_eig_est;
   HYPRE_Int            cheby_order;
   HYPRE_Int            cheby_variant;
   HYPRE_Int            cheby_scale;
   HYPRE_Real           cheby_fraction;
   HYPRE_Real         **cheby_ds;
   HYPRE_Real         **cheby_coefs;

   /* data needed for non-Galerkin option */
   HYPRE_Int           nongalerk_num_tol;
   HYPRE_Real         *nongalerk_tol;
   HYPRE_Real          nongalerkin_tol;
   HYPRE_Real         *nongal_tol_array;

   /* data generated in the solve phase */
   hypre_ParVector   *Vtemp;
   hypre_Vector      *Vtemp_local;
   HYPRE_Real        *Vtemp_local_data;
   HYPRE_Real         cycle_op_count;
   hypre_ParVector   *Rtemp;
   hypre_ParVector   *Ptemp;
   hypre_ParVector   *Ztemp;

   /* fields used by GSMG and LS interpolation */
   HYPRE_Int          gsmg;        /* nonzero indicates use of GSMG */
   HYPRE_Int          num_samples; /* number of sample vectors */

   /* log info */
   HYPRE_Int      logging;
   HYPRE_Int      num_iterations;
#ifdef CUMNUMIT
   HYPRE_Int      cum_num_iterations;
#endif
   HYPRE_Real   rel_resid_norm;
   hypre_ParVector *residual; /* available if logging>1 */

   /* output params */
   HYPRE_Int      print_level;
   char     log_file_name[256];
   HYPRE_Int      debug_flag;

   /* whether to print the constructed coarse grids BM Oct 22, 2006 */
   HYPRE_Int      plot_grids;
   char     plot_filename[251];

   /* coordinate data BM Oct 17, 2006 */
   HYPRE_Int      coorddim;
   float    *coordinates;

 /* data for fitting vectors in interpolation */
   HYPRE_Int          num_interp_vectors;
   HYPRE_Int          num_levels_interp_vectors; /* not set by user */
   hypre_ParVector  **interp_vectors;
   hypre_ParVector ***interp_vectors_array;
   HYPRE_Int          interp_vec_variant;
   HYPRE_Int          interp_vec_first_level;
   HYPRE_Real         interp_vectors_abs_q_trunc;
   HYPRE_Int          interp_vectors_q_max;
   HYPRE_Int          interp_refine;
   HYPRE_Int          smooth_interp_vectors;
   HYPRE_Real       *expandp_weights; /* currently not set by user */

   /* enable redundant coarse grid solve */
   HYPRE_Solver   coarse_solver;
   hypre_ParCSRMatrix  *A_coarse;
   hypre_ParVector  *f_coarse;
   hypre_ParVector  *u_coarse;
   MPI_Comm   new_comm;

   /* store matrix, vector and communication info for Gaussian elimination */
   HYPRE_Int   gs_setup;
   HYPRE_Real *A_mat, *A_inv;
   HYPRE_Real *b_vec;
   HYPRE_Int  *comm_info;

   /* information for multiplication with Lambda - additive AMG */
   HYPRE_Int      additive;
   HYPRE_Int      mult_additive;
   HYPRE_Int      simple;
   HYPRE_Int      add_last_lvl;
   HYPRE_Int      add_P_max_elmts;
   HYPRE_Real     add_trunc_factor;
   HYPRE_Int      add_rlx_type;
   HYPRE_Real     add_rlx_wt;
   hypre_ParCSRMatrix *Lambda;
   hypre_ParCSRMatrix *Atilde;
   hypre_ParVector *Rtilde;
   hypre_ParVector *Xtilde;
   HYPRE_Real *D_inv;

   /* Use 2 mat-mat-muls instead of triple product*/
   HYPRE_Int rap2;
   HYPRE_Int keepTranspose;
   HYPRE_Int modularized_matmat;
   /* information for preserving indexes as coarse grid points */
   HYPRE_Int C_point_keep_level;
   HYPRE_Int num_C_point_marker;
   HYPRE_Int   **C_point_marker_array;

#ifdef HYPRE_USING_DSUPERLU
 /* Parameters and data for SuperLU_Dist */
   HYPRE_Int dslu_threshold;
   HYPRE_Solver dslu_solver;
#endif

} hypre_ParAMGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_AMGData structure
 *--------------------------------------------------------------------------*/

/* setup params */

#define hypre_ParAMGDataRestriction(amg_data) ((amg_data)->restr_par)
#define hypre_ParAMGDataIsTriangular(amg_data) ((amg_data)->is_triangular)
#define hypre_ParAMGDataGMRESSwitchR(amg_data) ((amg_data)->gmres_switch)
#define hypre_ParAMGDataMaxLevels(amg_data) ((amg_data)->max_levels)
#define hypre_ParAMGDataStrongThreshold(amg_data) ((amg_data)->strong_threshold)
#define hypre_ParAMGDataStrongThresholdR(amg_data)((amg_data)->strong_thresholdR)
#define hypre_ParAMGDataFilterThresholdR(amg_data)((amg_data)->filter_thresholdR)
#define hypre_ParAMGDataSabs(amg_data) (amg_data->Sabs)
#define hypre_ParAMGDataMaxRowSum(amg_data) ((amg_data)->max_row_sum)
#define hypre_ParAMGDataTruncFactor(amg_data) ((amg_data)->trunc_factor)
#define hypre_ParAMGDataAggTruncFactor(amg_data) ((amg_data)->agg_trunc_factor)
#define hypre_ParAMGDataAggP12TruncFactor(amg_data) ((amg_data)->agg_P12_trunc_factor)
#define hypre_ParAMGDataJacobiTruncThreshold(amg_data) ((amg_data)->jacobi_trunc_threshold)
#define hypre_ParAMGDataSCommPkgSwitch(amg_data) ((amg_data)->S_commpkg_switch)
#define hypre_ParAMGDataInterpType(amg_data) ((amg_data)->interp_type)
#define hypre_ParAMGDataSepWeight(amg_data) ((amg_data)->sep_weight)
#define hypre_ParAMGDataAggInterpType(amg_data) ((amg_data)->agg_interp_type)
#define hypre_ParAMGDataCoarsenType(amg_data) ((amg_data)->coarsen_type)
#define hypre_ParAMGDataMeasureType(amg_data) ((amg_data)->measure_type)
#define hypre_ParAMGDataSetupType(amg_data) ((amg_data)->setup_type)
#define hypre_ParAMGDataPMaxElmts(amg_data) ((amg_data)->P_max_elmts)
#define hypre_ParAMGDataAggPMaxElmts(amg_data) ((amg_data)->agg_P_max_elmts)
#define hypre_ParAMGDataAggP12MaxElmts(amg_data) ((amg_data)->agg_P12_max_elmts)
#define hypre_ParAMGDataNumPaths(amg_data) ((amg_data)->num_paths)
#define hypre_ParAMGDataAggNumLevels(amg_data) ((amg_data)->agg_num_levels)
#define hypre_ParAMGDataPostInterpType(amg_data) ((amg_data)->post_interp_type)
#define hypre_ParAMGDataNumCRRelaxSteps(amg_data) ((amg_data)->num_CR_relax_steps)
#define hypre_ParAMGDataCRRate(amg_data) ((amg_data)->CR_rate)
#define hypre_ParAMGDataCRStrongTh(amg_data) ((amg_data)->CR_strong_th)
#define hypre_ParAMGDataADropTol(amg_data) ((amg_data)->A_drop_tol)
#define hypre_ParAMGDataADropType(amg_data) ((amg_data)->A_drop_type)
#define hypre_ParAMGDataISType(amg_data) ((amg_data)->IS_type)
#define hypre_ParAMGDataCRUseCG(amg_data) ((amg_data)->CR_use_CG)
#define hypre_ParAMGDataL1Norms(amg_data) ((amg_data)->l1_norms)
#define hypre_ParAMGDataCGCIts(amg_data) ((amg_data)->cgc_its)
#define hypre_ParAMGDataMaxCoarseSize(amg_data) ((amg_data)->max_coarse_size)
#define hypre_ParAMGDataMinCoarseSize(amg_data) ((amg_data)->min_coarse_size)
#define hypre_ParAMGDataSeqThreshold(amg_data) ((amg_data)->seq_threshold)

/* solve params */

#define hypre_ParAMGDataMinIter(amg_data) ((amg_data)->min_iter)
#define hypre_ParAMGDataMaxIter(amg_data) ((amg_data)->max_iter)
#define hypre_ParAMGDataFCycle(amg_data) ((amg_data)->fcycle)
#define hypre_ParAMGDataCycleType(amg_data) ((amg_data)->cycle_type)
#define hypre_ParAMGDataConvergeType(amg_data) ((amg_data)->converge_type)
#define hypre_ParAMGDataTol(amg_data) ((amg_data)->tol)
#define hypre_ParAMGDataNumGridSweeps(amg_data) ((amg_data)->num_grid_sweeps)
#define hypre_ParAMGDataUserCoarseRelaxType(amg_data) ((amg_data)->user_coarse_relax_type)
#define hypre_ParAMGDataUserRelaxType(amg_data) ((amg_data)->user_relax_type)
#define hypre_ParAMGDataUserRelaxWeight(amg_data) ((amg_data)->user_relax_weight)
#define hypre_ParAMGDataUserNumSweeps(amg_data) ((amg_data)->user_num_sweeps)
#define hypre_ParAMGDataGridRelaxType(amg_data) ((amg_data)->grid_relax_type)
#define hypre_ParAMGDataGridRelaxPoints(amg_data) \
((amg_data)->grid_relax_points)
#define hypre_ParAMGDataRelaxOrder(amg_data) ((amg_data)->relax_order)
#define hypre_ParAMGDataRelaxWeight(amg_data) ((amg_data)->relax_weight)
#define hypre_ParAMGDataOmega(amg_data) ((amg_data)->omega)
#define hypre_ParAMGDataOuterWt(amg_data) ((amg_data)->outer_wt)

/* problem data parameters */
#define  hypre_ParAMGDataNumVariables(amg_data)  ((amg_data)->num_variables)
#define hypre_ParAMGDataNumFunctions(amg_data) ((amg_data)->num_functions)
#define hypre_ParAMGDataNodal(amg_data) ((amg_data)->nodal)
#define hypre_ParAMGDataNodalLevels(amg_data) ((amg_data)->nodal_levels)
#define hypre_ParAMGDataNodalDiag(amg_data) ((amg_data)->nodal_diag)
#define hypre_ParAMGDataNumPoints(amg_data) ((amg_data)->num_points)
#define hypre_ParAMGDataDofFunc(amg_data) ((amg_data)->dof_func)
#define hypre_ParAMGDataDofPoint(amg_data) ((amg_data)->dof_point)
#define hypre_ParAMGDataPointDofMap(amg_data) ((amg_data)->point_dof_map)

/* data generated by the setup phase */
#define hypre_ParAMGDataCFMarkerArray(amg_data) ((amg_data)-> CF_marker_array)
#define hypre_ParAMGDataCPointMarkerArray(amg_data) ((amg_data)-> C_point_marker_array)
#define hypre_ParAMGDataAArray(amg_data) ((amg_data)->A_array)
#define hypre_ParAMGDataFArray(amg_data) ((amg_data)->F_array)
#define hypre_ParAMGDataUArray(amg_data) ((amg_data)->U_array)
#define hypre_ParAMGDataPArray(amg_data) ((amg_data)->P_array)
#define hypre_ParAMGDataRArray(amg_data) ((amg_data)->R_array)
#define hypre_ParAMGDataDofFuncArray(amg_data) ((amg_data)->dof_func_array)
#define hypre_ParAMGDataDofPointArray(amg_data) ((amg_data)->dof_point_array)
#define hypre_ParAMGDataPointDofMapArray(amg_data) \
((amg_data)->point_dof_map_array)
#define hypre_ParAMGDataNumLevels(amg_data) ((amg_data)->num_levels)
#define hypre_ParAMGDataSmoothType(amg_data) ((amg_data)->smooth_type)
#define hypre_ParAMGDataSmoothNumLevels(amg_data) \
((amg_data)->smooth_num_levels)
#define hypre_ParAMGDataSmoothNumSweeps(amg_data) \
((amg_data)->smooth_num_sweeps)
#define hypre_ParAMGDataSmoother(amg_data) ((amg_data)->smoother)
#define hypre_ParAMGDataVariant(amg_data) ((amg_data)->schw_variant)
#define hypre_ParAMGDataOverlap(amg_data) ((amg_data)->schw_overlap)
#define hypre_ParAMGDataDomainType(amg_data) ((amg_data)->schw_domain_type)
#define hypre_ParAMGDataSchwarzRlxWeight(amg_data) \
((amg_data)->schwarz_rlx_weight)
#define hypre_ParAMGDataSchwarzUseNonSymm(amg_data) \
((amg_data)->schwarz_use_nonsymm)
#define hypre_ParAMGDataSym(amg_data) ((amg_data)->ps_sym)
#define hypre_ParAMGDataLevel(amg_data) ((amg_data)->ps_level)
#define hypre_ParAMGDataMaxNzPerRow(amg_data) ((amg_data)->pi_max_nz_per_row)
#define hypre_ParAMGDataThreshold(amg_data) ((amg_data)->ps_threshold)
#define hypre_ParAMGDataFilter(amg_data) ((amg_data)->ps_filter)
#define hypre_ParAMGDataDropTol(amg_data) ((amg_data)->pi_drop_tol)
#define hypre_ParAMGDataEuclidFile(amg_data) ((amg_data)->euclidfile)
#define hypre_ParAMGDataEuLevel(amg_data) ((amg_data)->eu_level)
#define hypre_ParAMGDataEuSparseA(amg_data) ((amg_data)->eu_sparse_A)
#define hypre_ParAMGDataEuBJ(amg_data) ((amg_data)->eu_bj)

#define hypre_ParAMGDataMaxEigEst(amg_data) ((amg_data)->max_eig_est)
#define hypre_ParAMGDataMinEigEst(amg_data) ((amg_data)->min_eig_est)
#define hypre_ParAMGDataChebyOrder(amg_data) ((amg_data)->cheby_order)
#define hypre_ParAMGDataChebyFraction(amg_data) ((amg_data)->cheby_fraction)
#define hypre_ParAMGDataChebyEigEst(amg_data) ((amg_data)->cheby_eig_est)
#define hypre_ParAMGDataChebyVariant(amg_data) ((amg_data)->cheby_variant)
#define hypre_ParAMGDataChebyScale(amg_data) ((amg_data)->cheby_scale)
#define hypre_ParAMGDataChebyDS(amg_data) ((amg_data)->cheby_ds)
#define hypre_ParAMGDataChebyCoefs(amg_data) ((amg_data)->cheby_coefs)

/* block */
#define hypre_ParAMGDataABlockArray(amg_data) ((amg_data)->A_block_array)
#define hypre_ParAMGDataPBlockArray(amg_data) ((amg_data)->P_block_array)
#define hypre_ParAMGDataRBlockArray(amg_data) ((amg_data)->R_block_array)

#define hypre_ParAMGDataBlockMode(amg_data) ((amg_data)->block_mode)


/* data generated in the solve phase */
#define hypre_ParAMGDataVtemp(amg_data) ((amg_data)->Vtemp)
#define hypre_ParAMGDataVtempLocal(amg_data) ((amg_data)->Vtemp_local)
#define hypre_ParAMGDataVtemplocalData(amg_data) ((amg_data)->Vtemp_local_data)
#define hypre_ParAMGDataCycleOpCount(amg_data) ((amg_data)->cycle_op_count)
#define hypre_ParAMGDataRtemp(amg_data) ((amg_data)->Rtemp)
#define hypre_ParAMGDataPtemp(amg_data) ((amg_data)->Ptemp)
#define hypre_ParAMGDataZtemp(amg_data) ((amg_data)->Ztemp)

/* fields used by GSMG */
#define hypre_ParAMGDataGSMG(amg_data) ((amg_data)->gsmg)
#define hypre_ParAMGDataNumSamples(amg_data) ((amg_data)->num_samples)

/* log info data */
#define hypre_ParAMGDataLogging(amg_data) ((amg_data)->logging)
#define hypre_ParAMGDataNumIterations(amg_data) ((amg_data)->num_iterations)
#ifdef CUMNUMIT
#define hypre_ParAMGDataCumNumIterations(amg_data) ((amg_data)->cum_num_iterations)
#endif
#define hypre_ParAMGDataRelativeResidualNorm(amg_data) ((amg_data)->rel_resid_norm)
#define hypre_ParAMGDataResidual(amg_data) ((amg_data)->residual)

/* output parameters */
#define hypre_ParAMGDataPrintLevel(amg_data) ((amg_data)->print_level)
#define hypre_ParAMGDataLogFileName(amg_data) ((amg_data)->log_file_name)
#define hypre_ParAMGDataDebugFlag(amg_data)   ((amg_data)->debug_flag)

/* BM Oct 22, 2006 */
#define hypre_ParAMGDataPlotGrids(amg_data) ((amg_data)->plot_grids)
#define hypre_ParAMGDataPlotFileName(amg_data) ((amg_data)->plot_filename)

/* coordinates BM Oct 17, 2006 */
#define hypre_ParAMGDataCoordDim(amg_data) ((amg_data)->coorddim)
#define hypre_ParAMGDataCoordinates(amg_data) ((amg_data)->coordinates)


#define hypre_ParAMGNumInterpVectors(amg_data) ((amg_data)->num_interp_vectors)
#define hypre_ParAMGNumLevelsInterpVectors(amg_data) ((amg_data)->num_levels_interp_vectors)
#define hypre_ParAMGInterpVectors(amg_data) ((amg_data)->interp_vectors)
#define hypre_ParAMGInterpVectorsArray(amg_data) ((amg_data)->interp_vectors_array)
#define hypre_ParAMGInterpVecVariant(amg_data) ((amg_data)->interp_vec_variant)
#define hypre_ParAMGInterpVecFirstLevel(amg_data) ((amg_data)->interp_vec_first_level)
#define hypre_ParAMGInterpVecAbsQTrunc(amg_data) ((amg_data)->interp_vectors_abs_q_trunc)
#define hypre_ParAMGInterpVecQMax(amg_data) ((amg_data)->interp_vectors_q_max)
#define hypre_ParAMGInterpRefine(amg_data) ((amg_data)->interp_refine)
#define hypre_ParAMGSmoothInterpVectors(amg_data) ((amg_data)->smooth_interp_vectors)
#define hypre_ParAMGDataExpandPWeights(amg_data) ((amg_data)->expandp_weights)

#define hypre_ParAMGDataCoarseSolver(amg_data) ((amg_data)->coarse_solver)
#define hypre_ParAMGDataACoarse(amg_data) ((amg_data)->A_coarse)
#define hypre_ParAMGDataFCoarse(amg_data) ((amg_data)->f_coarse)
#define hypre_ParAMGDataUCoarse(amg_data) ((amg_data)->u_coarse)
#define hypre_ParAMGDataNewComm(amg_data) ((amg_data)->new_comm)
#define hypre_ParAMGDataRedundant(amg_data) ((amg_data)->redundant)
#define hypre_ParAMGDataParticipate(amg_data) ((amg_data)->participate)

#define hypre_ParAMGDataGSSetup(amg_data) ((amg_data)->gs_setup)
#define hypre_ParAMGDataAMat(amg_data) ((amg_data)->A_mat)
#define hypre_ParAMGDataAInv(amg_data) ((amg_data)->A_inv)
#define hypre_ParAMGDataBVec(amg_data) ((amg_data)->b_vec)
#define hypre_ParAMGDataCommInfo(amg_data) ((amg_data)->comm_info)

/* additive AMG parameters */
#define hypre_ParAMGDataAdditive(amg_data) ((amg_data)->additive)
#define hypre_ParAMGDataMultAdditive(amg_data) ((amg_data)->mult_additive)
#define hypre_ParAMGDataSimple(amg_data) ((amg_data)->simple)
#define hypre_ParAMGDataAddLastLvl(amg_data) ((amg_data)->add_last_lvl)
#define hypre_ParAMGDataMultAddPMaxElmts(amg_data) ((amg_data)->add_P_max_elmts)
#define hypre_ParAMGDataMultAddTruncFactor(amg_data) ((amg_data)->add_trunc_factor)
#define hypre_ParAMGDataAddRelaxType(amg_data) ((amg_data)->add_rlx_type)
#define hypre_ParAMGDataAddRelaxWt(amg_data) ((amg_data)->add_rlx_wt)
#define hypre_ParAMGDataLambda(amg_data) ((amg_data)->Lambda)
#define hypre_ParAMGDataAtilde(amg_data) ((amg_data)->Atilde)
#define hypre_ParAMGDataRtilde(amg_data) ((amg_data)->Rtilde)
#define hypre_ParAMGDataXtilde(amg_data) ((amg_data)->Xtilde)
#define hypre_ParAMGDataDinv(amg_data) ((amg_data)->D_inv)

/* non-Galerkin parameters */
#define hypre_ParAMGDataNonGalerkNumTol(amg_data) ((amg_data)->nongalerk_num_tol)
#define hypre_ParAMGDataNonGalerkTol(amg_data) ((amg_data)->nongalerk_tol)
#define hypre_ParAMGDataNonGalerkinTol(amg_data) ((amg_data)->nongalerkin_tol)
#define hypre_ParAMGDataNonGalTolArray(amg_data) ((amg_data)->nongal_tol_array)

#define hypre_ParAMGDataRAP2(amg_data) ((amg_data)->rap2)
#define hypre_ParAMGDataKeepTranspose(amg_data) ((amg_data)->keepTranspose)
#define hypre_ParAMGDataModularizedMatMat(amg_data) ((amg_data)->modularized_matmat)

/*indices for the dof which will keep coarsening to the coarse level */
#define hypre_ParAMGDataCPointKeepMarkerArray(amg_data) ((amg_data)-> C_point_marker_array)
#define hypre_ParAMGDataCPointKeepLevel(amg_data) ((amg_data)-> C_point_keep_level)
#define hypre_ParAMGDataNumCPointKeep(amg_data) ((amg_data)-> num_C_point_marker)

/* Parameters and data for SuperLU_Dist */
#ifdef HYPRE_USING_DSUPERLU
#define hypre_ParAMGDataDSLUThreshold(amg_data) ((amg_data)->dslu_threshold)
#define hypre_ParAMGDataDSLUSolver(amg_data) ((amg_data)->dslu_solver)
#endif

#endif




/* ads.c */
void *hypre_ADSCreate ( void );
HYPRE_Int hypre_ADSDestroy ( void *solver );
HYPRE_Int hypre_ADSSetDiscreteCurl ( void *solver , hypre_ParCSRMatrix *C );
HYPRE_Int hypre_ADSSetDiscreteGradient ( void *solver , hypre_ParCSRMatrix *G );
HYPRE_Int hypre_ADSSetCoordinateVectors ( void *solver , hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector *z );
HYPRE_Int hypre_ADSSetInterpolations ( void *solver , hypre_ParCSRMatrix *RT_Pi , hypre_ParCSRMatrix *RT_Pix , hypre_ParCSRMatrix *RT_Piy , hypre_ParCSRMatrix *RT_Piz , hypre_ParCSRMatrix *ND_Pi , hypre_ParCSRMatrix *ND_Pix , hypre_ParCSRMatrix *ND_Piy , hypre_ParCSRMatrix *ND_Piz );
HYPRE_Int hypre_ADSSetMaxIter ( void *solver , HYPRE_Int maxit );
HYPRE_Int hypre_ADSSetTol ( void *solver , HYPRE_Real tol );
HYPRE_Int hypre_ADSSetCycleType ( void *solver , HYPRE_Int cycle_type );
HYPRE_Int hypre_ADSSetPrintLevel ( void *solver , HYPRE_Int print_level );
HYPRE_Int hypre_ADSSetSmoothingOptions ( void *solver , HYPRE_Int A_relax_type , HYPRE_Int A_relax_times , HYPRE_Real A_relax_weight , HYPRE_Real A_omega );
HYPRE_Int hypre_ADSSetChebySmoothingOptions ( void *solver , HYPRE_Int A_cheby_order , HYPRE_Int A_cheby_fraction );
HYPRE_Int hypre_ADSSetAMSOptions ( void *solver , HYPRE_Int B_C_cycle_type , HYPRE_Int B_C_coarsen_type , HYPRE_Int B_C_agg_levels , HYPRE_Int B_C_relax_type , HYPRE_Real B_C_theta , HYPRE_Int B_C_interp_type , HYPRE_Int B_C_Pmax );
HYPRE_Int hypre_ADSSetAMGOptions ( void *solver , HYPRE_Int B_Pi_coarsen_type , HYPRE_Int B_Pi_agg_levels , HYPRE_Int B_Pi_relax_type , HYPRE_Real B_Pi_theta , HYPRE_Int B_Pi_interp_type , HYPRE_Int B_Pi_Pmax );
HYPRE_Int hypre_ADSComputePi ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *C , hypre_ParCSRMatrix *G , hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector *z , hypre_ParCSRMatrix *PiNDx , hypre_ParCSRMatrix *PiNDy , hypre_ParCSRMatrix *PiNDz , hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_ADSComputePixyz ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *C , hypre_ParCSRMatrix *G , hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector *z , hypre_ParCSRMatrix *PiNDx , hypre_ParCSRMatrix *PiNDy , hypre_ParCSRMatrix *PiNDz , hypre_ParCSRMatrix **Pix_ptr , hypre_ParCSRMatrix **Piy_ptr , hypre_ParCSRMatrix **Piz_ptr );
HYPRE_Int hypre_ADSSetup ( void *solver , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
HYPRE_Int hypre_ADSSolve ( void *solver , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
HYPRE_Int hypre_ADSGetNumIterations ( void *solver , HYPRE_Int *num_iterations );
HYPRE_Int hypre_ADSGetFinalRelativeResidualNorm ( void *solver , HYPRE_Real *rel_resid_norm );

/* ame.c */
void *hypre_AMECreate ( void );
HYPRE_Int hypre_AMEDestroy ( void *esolver );
HYPRE_Int hypre_AMESetAMSSolver ( void *esolver , void *ams_solver );
HYPRE_Int hypre_AMESetMassMatrix ( void *esolver , hypre_ParCSRMatrix *M );
HYPRE_Int hypre_AMESetBlockSize ( void *esolver , HYPRE_Int block_size );
HYPRE_Int hypre_AMESetMaxIter ( void *esolver , HYPRE_Int maxit );
HYPRE_Int hypre_AMESetTol ( void *esolver , HYPRE_Real tol );
HYPRE_Int hypre_AMESetRTol ( void *esolver , HYPRE_Real tol );
HYPRE_Int hypre_AMESetPrintLevel ( void *esolver , HYPRE_Int print_level );
HYPRE_Int hypre_AMESetup ( void *esolver );
HYPRE_Int hypre_AMEDiscrDivFreeComponent ( void *esolver , hypre_ParVector *b );
void hypre_AMEOperatorA ( void *data , void *x , void *y );
void hypre_AMEMultiOperatorA ( void *data , void *x , void *y );
void hypre_AMEOperatorM ( void *data , void *x , void *y );
void hypre_AMEMultiOperatorM ( void *data , void *x , void *y );
void hypre_AMEOperatorB ( void *data , void *x , void *y );
void hypre_AMEMultiOperatorB ( void *data , void *x , void *y );
HYPRE_Int hypre_AMESolve ( void *esolver );
HYPRE_Int hypre_AMEGetEigenvectors ( void *esolver , HYPRE_ParVector **eigenvectors_ptr );
HYPRE_Int hypre_AMEGetEigenvalues ( void *esolver , HYPRE_Real **eigenvalues_ptr );

/* amg_hybrid.c */
void *hypre_AMGHybridCreate ( void );
HYPRE_Int hypre_AMGHybridDestroy ( void *AMGhybrid_vdata );
HYPRE_Int hypre_AMGHybridSetTol ( void *AMGhybrid_vdata , HYPRE_Real tol );
HYPRE_Int hypre_AMGHybridSetAbsoluteTol ( void *AMGhybrid_vdata , HYPRE_Real a_tol );
HYPRE_Int hypre_AMGHybridSetConvergenceTol ( void *AMGhybrid_vdata , HYPRE_Real cf_tol );
HYPRE_Int hypre_AMGHybridSetNonGalerkinTol ( void *AMGhybrid_vdata , HYPRE_Int nongalerk_num_tol, HYPRE_Real *nongalerkin_tol );
HYPRE_Int hypre_AMGHybridSetDSCGMaxIter ( void *AMGhybrid_vdata , HYPRE_Int dscg_max_its );
HYPRE_Int hypre_AMGHybridSetPCGMaxIter ( void *AMGhybrid_vdata , HYPRE_Int pcg_max_its );
HYPRE_Int hypre_AMGHybridSetSetupType ( void *AMGhybrid_vdata , HYPRE_Int setup_type );
HYPRE_Int hypre_AMGHybridSetSolverType ( void *AMGhybrid_vdata , HYPRE_Int solver_type );
HYPRE_Int hypre_AMGHybridSetRecomputeResidual ( void *AMGhybrid_vdata , HYPRE_Int recompute_residual );
HYPRE_Int hypre_AMGHybridGetRecomputeResidual ( void *AMGhybrid_vdata , HYPRE_Int *recompute_residual );
HYPRE_Int hypre_AMGHybridSetRecomputeResidualP ( void *AMGhybrid_vdata , HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_AMGHybridGetRecomputeResidualP ( void *AMGhybrid_vdata , HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_AMGHybridSetKDim ( void *AMGhybrid_vdata , HYPRE_Int k_dim );
HYPRE_Int hypre_AMGHybridSetStopCrit ( void *AMGhybrid_vdata , HYPRE_Int stop_crit );
HYPRE_Int hypre_AMGHybridSetTwoNorm ( void *AMGhybrid_vdata , HYPRE_Int two_norm );
HYPRE_Int hypre_AMGHybridSetRelChange ( void *AMGhybrid_vdata , HYPRE_Int rel_change );
HYPRE_Int hypre_AMGHybridSetPrecond ( void *pcg_vdata , HYPRE_Int (*pcg_precond_solve )(void*,void*,void*,void*), HYPRE_Int (*pcg_precond_setup )(void*,void*,void*,void*), void *pcg_precond );
HYPRE_Int hypre_AMGHybridSetLogging ( void *AMGhybrid_vdata , HYPRE_Int logging );
HYPRE_Int hypre_AMGHybridSetPrintLevel ( void *AMGhybrid_vdata , HYPRE_Int print_level );
HYPRE_Int hypre_AMGHybridSetStrongThreshold ( void *AMGhybrid_vdata , HYPRE_Real strong_threshold );
HYPRE_Int hypre_AMGHybridSetMaxRowSum ( void *AMGhybrid_vdata , HYPRE_Real max_row_sum );
HYPRE_Int hypre_AMGHybridSetTruncFactor ( void *AMGhybrid_vdata , HYPRE_Real trunc_factor );
HYPRE_Int hypre_AMGHybridSetPMaxElmts ( void *AMGhybrid_vdata , HYPRE_Int P_max_elmts );
HYPRE_Int hypre_AMGHybridSetMaxLevels ( void *AMGhybrid_vdata , HYPRE_Int max_levels );
HYPRE_Int hypre_AMGHybridSetMeasureType ( void *AMGhybrid_vdata , HYPRE_Int measure_type );
HYPRE_Int hypre_AMGHybridSetCoarsenType ( void *AMGhybrid_vdata , HYPRE_Int coarsen_type );
HYPRE_Int hypre_AMGHybridSetInterpType ( void *AMGhybrid_vdata , HYPRE_Int interp_type );
HYPRE_Int hypre_AMGHybridSetCycleType ( void *AMGhybrid_vdata , HYPRE_Int cycle_type );
HYPRE_Int hypre_AMGHybridSetNumSweeps ( void *AMGhybrid_vdata , HYPRE_Int num_sweeps );
HYPRE_Int hypre_AMGHybridSetCycleNumSweeps ( void *AMGhybrid_vdata , HYPRE_Int num_sweeps , HYPRE_Int k );
HYPRE_Int hypre_AMGHybridSetRelaxType ( void *AMGhybrid_vdata , HYPRE_Int relax_type );
HYPRE_Int hypre_AMGHybridSetKeepTranspose ( void *AMGhybrid_vdata , HYPRE_Int keepT );
HYPRE_Int hypre_AMGHybridSetSplittingStrategy( void *AMGhybrid_vdata , HYPRE_Int splitting_strategy );
HYPRE_Int hypre_AMGHybridSetCycleRelaxType ( void *AMGhybrid_vdata , HYPRE_Int relax_type , HYPRE_Int k );
HYPRE_Int hypre_AMGHybridSetRelaxOrder ( void *AMGhybrid_vdata , HYPRE_Int relax_order );
HYPRE_Int hypre_AMGHybridSetMaxCoarseSize ( void *AMGhybrid_vdata , HYPRE_Int max_coarse_size );
HYPRE_Int hypre_AMGHybridSetMinCoarseSize ( void *AMGhybrid_vdata , HYPRE_Int min_coarse_size );
HYPRE_Int hypre_AMGHybridSetSeqThreshold ( void *AMGhybrid_vdata , HYPRE_Int seq_threshold );
HYPRE_Int hypre_AMGHybridSetNumGridSweeps ( void *AMGhybrid_vdata , HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_AMGHybridSetGridRelaxType ( void *AMGhybrid_vdata , HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_AMGHybridSetGridRelaxPoints ( void *AMGhybrid_vdata , HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_AMGHybridSetRelaxWeight ( void *AMGhybrid_vdata , HYPRE_Real *relax_weight );
HYPRE_Int hypre_AMGHybridSetOmega ( void *AMGhybrid_vdata , HYPRE_Real *omega );
HYPRE_Int hypre_AMGHybridSetRelaxWt ( void *AMGhybrid_vdata , HYPRE_Real relax_wt );
HYPRE_Int hypre_AMGHybridSetLevelRelaxWt ( void *AMGhybrid_vdata , HYPRE_Real relax_wt , HYPRE_Int level );
HYPRE_Int hypre_AMGHybridSetOuterWt ( void *AMGhybrid_vdata , HYPRE_Real outer_wt );
HYPRE_Int hypre_AMGHybridSetLevelOuterWt ( void *AMGhybrid_vdata , HYPRE_Real outer_wt , HYPRE_Int level );
HYPRE_Int hypre_AMGHybridSetNumPaths ( void *AMGhybrid_vdata , HYPRE_Int num_paths );
HYPRE_Int hypre_AMGHybridSetDofFunc ( void *AMGhybrid_vdata , HYPRE_Int *dof_func );
HYPRE_Int hypre_AMGHybridSetAggNumLevels ( void *AMGhybrid_vdata , HYPRE_Int agg_num_levels );
HYPRE_Int hypre_AMGHybridSetNumFunctions ( void *AMGhybrid_vdata , HYPRE_Int num_functions );
HYPRE_Int hypre_AMGHybridSetNodal ( void *AMGhybrid_vdata , HYPRE_Int nodal );
HYPRE_Int hypre_AMGHybridGetNumIterations ( void *AMGhybrid_vdata , HYPRE_Int *num_its );
HYPRE_Int hypre_AMGHybridGetDSCGNumIterations ( void *AMGhybrid_vdata , HYPRE_Int *dscg_num_its );
HYPRE_Int hypre_AMGHybridGetPCGNumIterations ( void *AMGhybrid_vdata , HYPRE_Int *pcg_num_its );
HYPRE_Int hypre_AMGHybridGetFinalRelativeResidualNorm ( void *AMGhybrid_vdata , HYPRE_Real *final_rel_res_norm );
HYPRE_Int hypre_AMGHybridSetup ( void *AMGhybrid_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
HYPRE_Int hypre_AMGHybridSolve ( void *AMGhybrid_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );

/* ams.c */
HYPRE_Int hypre_ParCSRRelax ( hypre_ParCSRMatrix *A , hypre_ParVector *f , HYPRE_Int relax_type , HYPRE_Int relax_times , HYPRE_Real *l1_norms , HYPRE_Real relax_weight , HYPRE_Real omega , HYPRE_Real max_eig_est , HYPRE_Real min_eig_est , HYPRE_Int cheby_order , HYPRE_Real cheby_fraction , hypre_ParVector *u , hypre_ParVector *v , hypre_ParVector *z );
hypre_ParVector *hypre_ParVectorInRangeOf ( hypre_ParCSRMatrix *A );
hypre_ParVector *hypre_ParVectorInDomainOf ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParVectorBlockSplit ( hypre_ParVector *x , hypre_ParVector *x_ [3 ], HYPRE_Int dim );
HYPRE_Int hypre_ParVectorBlockGather ( hypre_ParVector *x , hypre_ParVector *x_ [3 ], HYPRE_Int dim );
HYPRE_Int hypre_BoomerAMGBlockSolve ( void *B , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
HYPRE_Int hypre_ParCSRMatrixFixZeroRows ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRComputeL1Norms ( hypre_ParCSRMatrix *A , HYPRE_Int option , HYPRE_Int *cf_marker , HYPRE_Real **l1_norm_ptr );
HYPRE_Int hypre_ParCSRMatrixSetDiagRows ( hypre_ParCSRMatrix *A , HYPRE_Real d );
void *hypre_AMSCreate ( void );
HYPRE_Int hypre_AMSDestroy ( void *solver );
HYPRE_Int hypre_AMSSetDimension ( void *solver , HYPRE_Int dim );
HYPRE_Int hypre_AMSSetDiscreteGradient ( void *solver , hypre_ParCSRMatrix *G );
HYPRE_Int hypre_AMSSetCoordinateVectors ( void *solver , hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector *z );
HYPRE_Int hypre_AMSSetEdgeConstantVectors ( void *solver , hypre_ParVector *Gx , hypre_ParVector *Gy , hypre_ParVector *Gz );
HYPRE_Int hypre_AMSSetInterpolations ( void *solver , hypre_ParCSRMatrix *Pi , hypre_ParCSRMatrix *Pix , hypre_ParCSRMatrix *Piy , hypre_ParCSRMatrix *Piz );
HYPRE_Int hypre_AMSSetAlphaPoissonMatrix ( void *solver , hypre_ParCSRMatrix *A_Pi );
HYPRE_Int hypre_AMSSetBetaPoissonMatrix ( void *solver , hypre_ParCSRMatrix *A_G );
HYPRE_Int hypre_AMSSetInteriorNodes ( void *solver , hypre_ParVector *interior_nodes );
HYPRE_Int hypre_AMSSetProjectionFrequency ( void *solver , HYPRE_Int projection_frequency );
HYPRE_Int hypre_AMSSetMaxIter ( void *solver , HYPRE_Int maxit );
HYPRE_Int hypre_AMSSetTol ( void *solver , HYPRE_Real tol );
HYPRE_Int hypre_AMSSetCycleType ( void *solver , HYPRE_Int cycle_type );
HYPRE_Int hypre_AMSSetPrintLevel ( void *solver , HYPRE_Int print_level );
HYPRE_Int hypre_AMSSetSmoothingOptions ( void *solver , HYPRE_Int A_relax_type , HYPRE_Int A_relax_times , HYPRE_Real A_relax_weight , HYPRE_Real A_omega );
HYPRE_Int hypre_AMSSetChebySmoothingOptions ( void *solver , HYPRE_Int A_cheby_order , HYPRE_Int A_cheby_fraction );
HYPRE_Int hypre_AMSSetAlphaAMGOptions ( void *solver , HYPRE_Int B_Pi_coarsen_type , HYPRE_Int B_Pi_agg_levels , HYPRE_Int B_Pi_relax_type , HYPRE_Real B_Pi_theta , HYPRE_Int B_Pi_interp_type , HYPRE_Int B_Pi_Pmax );
HYPRE_Int hypre_AMSSetAlphaAMGCoarseRelaxType ( void *solver , HYPRE_Int B_Pi_coarse_relax_type );
HYPRE_Int hypre_AMSSetBetaAMGOptions ( void *solver , HYPRE_Int B_G_coarsen_type , HYPRE_Int B_G_agg_levels , HYPRE_Int B_G_relax_type , HYPRE_Real B_G_theta , HYPRE_Int B_G_interp_type , HYPRE_Int B_G_Pmax );
HYPRE_Int hypre_AMSSetBetaAMGCoarseRelaxType ( void *solver , HYPRE_Int B_G_coarse_relax_type );
HYPRE_Int hypre_AMSComputePi ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *G , hypre_ParVector *Gx , hypre_ParVector *Gy , hypre_ParVector *Gz , HYPRE_Int dim , hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_AMSComputePixyz ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *G , hypre_ParVector *Gx , hypre_ParVector *Gy , hypre_ParVector *Gz , HYPRE_Int dim , hypre_ParCSRMatrix **Pix_ptr , hypre_ParCSRMatrix **Piy_ptr , hypre_ParCSRMatrix **Piz_ptr );
HYPRE_Int hypre_AMSComputeGPi ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *G , hypre_ParVector *Gx , hypre_ParVector *Gy , hypre_ParVector *Gz , HYPRE_Int dim , hypre_ParCSRMatrix **GPi_ptr );
HYPRE_Int hypre_AMSSetup ( void *solver , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
HYPRE_Int hypre_AMSSolve ( void *solver , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
HYPRE_Int hypre_ParCSRSubspacePrec ( hypre_ParCSRMatrix *A0 , HYPRE_Int A0_relax_type , HYPRE_Int A0_relax_times , HYPRE_Real *A0_l1_norms , HYPRE_Real A0_relax_weight , HYPRE_Real A0_omega , HYPRE_Real A0_max_eig_est , HYPRE_Real A0_min_eig_est , HYPRE_Int A0_cheby_order , HYPRE_Real A0_cheby_fraction , hypre_ParCSRMatrix **A , HYPRE_Solver *B , HYPRE_PtrToSolverFcn *HB , hypre_ParCSRMatrix **P , hypre_ParVector **r , hypre_ParVector **g , hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector *r0 , hypre_ParVector *g0 , char *cycle , hypre_ParVector *z );
HYPRE_Int hypre_AMSGetNumIterations ( void *solver , HYPRE_Int *num_iterations );
HYPRE_Int hypre_AMSGetFinalRelativeResidualNorm ( void *solver , HYPRE_Real *rel_resid_norm );
HYPRE_Int hypre_AMSProjectOutGradients ( void *solver , hypre_ParVector *x );
HYPRE_Int hypre_AMSConstructDiscreteGradient ( hypre_ParCSRMatrix *A , hypre_ParVector *x_coord , HYPRE_BigInt *edge_vertex , HYPRE_Int edge_orientation , hypre_ParCSRMatrix **G_ptr );
HYPRE_Int hypre_AMSFEISetup ( void *solver , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x , HYPRE_Int num_vert , HYPRE_Int num_local_vert , HYPRE_BigInt *vert_number , HYPRE_Real *vert_coord , HYPRE_Int num_edges , HYPRE_BigInt *edge_vertex );
HYPRE_Int hypre_AMSFEIDestroy ( void *solver );
HYPRE_Int hypre_ParCSRComputeL1NormsThreads ( hypre_ParCSRMatrix *A , HYPRE_Int option , HYPRE_Int num_threads , HYPRE_Int *cf_marker , HYPRE_Real **l1_norm_ptr );
HYPRE_Int hypre_ParCSRRelaxThreads ( hypre_ParCSRMatrix *A , hypre_ParVector *f , HYPRE_Int relax_type , HYPRE_Int relax_times , HYPRE_Real *l1_norms , HYPRE_Real relax_weight , HYPRE_Real omega , hypre_ParVector *u , hypre_ParVector *Vtemp , hypre_ParVector *z );

/* aux_interp.c */
HYPRE_Int hypre_alt_insert_new_nodes ( hypre_ParCSRCommPkg *comm_pkg , hypre_ParCSRCommPkg *extend_comm_pkg , HYPRE_Int *IN_marker , HYPRE_Int full_off_procNodes , HYPRE_Int *OUT_marker );
HYPRE_Int hypre_big_insert_new_nodes ( hypre_ParCSRCommPkg *comm_pkg , hypre_ParCSRCommPkg *extend_comm_pkg , HYPRE_Int *IN_marker , HYPRE_Int full_off_procNodes , HYPRE_BigInt offset , HYPRE_BigInt *OUT_marker );
HYPRE_Int hypre_ssort ( HYPRE_BigInt *data , HYPRE_Int n );
HYPRE_Int hypre_index_of_minimum ( HYPRE_BigInt *data , HYPRE_Int n );
void hypre_swap_int ( HYPRE_BigInt *data , HYPRE_Int a , HYPRE_Int b );
void hypre_initialize_vecs ( HYPRE_Int diag_n , HYPRE_Int offd_n , HYPRE_Int *diag_ftc , HYPRE_BigInt *offd_ftc , HYPRE_Int *diag_pm , HYPRE_Int *offd_pm , HYPRE_Int *tmp_CF );
/*HYPRE_Int hypre_new_offd_nodes(HYPRE_Int **found , HYPRE_Int num_cols_A_offd , HYPRE_Int *A_ext_i , HYPRE_Int *A_ext_j, HYPRE_Int num_cols_S_offd, HYPRE_Int *col_map_offd, HYPRE_Int col_1, HYPRE_Int col_n, HYPRE_Int *Sop_i, HYPRE_Int *Sop_j, HYPRE_Int *CF_marker_offd );*/
HYPRE_Int hypre_exchange_marker(hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int *IN_marker, HYPRE_Int *OUT_marker);
HYPRE_Int hypre_exchange_interp_data( HYPRE_Int **CF_marker_offd, HYPRE_Int **dof_func_offd, hypre_CSRMatrix **A_ext, HYPRE_Int *full_off_procNodes, hypre_CSRMatrix **Sop, hypre_ParCSRCommPkg **extend_comm_pkg, hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int skip_fine_or_same_sign);
void hypre_build_interp_colmap(hypre_ParCSRMatrix *P, HYPRE_Int full_off_procNodes, HYPRE_Int *tmp_CF_marker_offd, HYPRE_BigInt *fine_to_coarse_offd);

/* block_tridiag.c */
void *hypre_BlockTridiagCreate ( void );
HYPRE_Int hypre_BlockTridiagDestroy ( void *data );
HYPRE_Int hypre_BlockTridiagSetup ( void *data , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
HYPRE_Int hypre_BlockTridiagSolve ( void *data , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
HYPRE_Int hypre_BlockTridiagSetIndexSet ( void *data , HYPRE_Int n , HYPRE_Int *inds );
HYPRE_Int hypre_BlockTridiagSetAMGStrengthThreshold ( void *data , HYPRE_Real thresh );
HYPRE_Int hypre_BlockTridiagSetAMGNumSweeps ( void *data , HYPRE_Int nsweeps );
HYPRE_Int hypre_BlockTridiagSetAMGRelaxType ( void *data , HYPRE_Int relax_type );
HYPRE_Int hypre_BlockTridiagSetPrintLevel ( void *data , HYPRE_Int print_level );

/* driver.c */
HYPRE_Int BuildParFromFile ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParDifConv ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParFromOneFile ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildRhsParFromOneFile ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix A , HYPRE_ParVector *b_ptr );
HYPRE_Int BuildParLaplacian9pt ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian27pt ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );

/* gen_redcs_mat.c */
HYPRE_Int hypre_seqAMGSetup ( hypre_ParAMGData *amg_data , HYPRE_Int p_level , HYPRE_Int coarse_threshold );
HYPRE_Int hypre_seqAMGCycle ( hypre_ParAMGData *amg_data , HYPRE_Int p_level , hypre_ParVector **Par_F_array , hypre_ParVector **Par_U_array );
HYPRE_Int hypre_GenerateSubComm ( MPI_Comm comm , HYPRE_Int participate , MPI_Comm *new_comm_ptr );
void hypre_merge_lists ( HYPRE_Int *list1 , HYPRE_Int *list2 , hypre_int *np1 , hypre_MPI_Datatype *dptr );

/* HYPRE_ads.c */
HYPRE_Int HYPRE_ADSCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_ADSDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ADSSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ADSSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ADSSetDiscreteCurl ( HYPRE_Solver solver , HYPRE_ParCSRMatrix C );
HYPRE_Int HYPRE_ADSSetDiscreteGradient ( HYPRE_Solver solver , HYPRE_ParCSRMatrix G );
HYPRE_Int HYPRE_ADSSetCoordinateVectors ( HYPRE_Solver solver , HYPRE_ParVector x , HYPRE_ParVector y , HYPRE_ParVector z );
HYPRE_Int HYPRE_ADSSetInterpolations ( HYPRE_Solver solver , HYPRE_ParCSRMatrix RT_Pi , HYPRE_ParCSRMatrix RT_Pix , HYPRE_ParCSRMatrix RT_Piy , HYPRE_ParCSRMatrix RT_Piz , HYPRE_ParCSRMatrix ND_Pi , HYPRE_ParCSRMatrix ND_Pix , HYPRE_ParCSRMatrix ND_Piy , HYPRE_ParCSRMatrix ND_Piz );
HYPRE_Int HYPRE_ADSSetMaxIter ( HYPRE_Solver solver , HYPRE_Int maxit );
HYPRE_Int HYPRE_ADSSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ADSSetCycleType ( HYPRE_Solver solver , HYPRE_Int cycle_type );
HYPRE_Int HYPRE_ADSSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_ADSSetSmoothingOptions ( HYPRE_Solver solver , HYPRE_Int relax_type , HYPRE_Int relax_times , HYPRE_Real relax_weight , HYPRE_Real omega );
HYPRE_Int HYPRE_ADSSetChebySmoothingOptions ( HYPRE_Solver solver , HYPRE_Int cheby_order , HYPRE_Int cheby_fraction );
HYPRE_Int HYPRE_ADSSetAMSOptions ( HYPRE_Solver solver , HYPRE_Int cycle_type , HYPRE_Int coarsen_type , HYPRE_Int agg_levels , HYPRE_Int relax_type , HYPRE_Real strength_threshold , HYPRE_Int interp_type , HYPRE_Int Pmax );
HYPRE_Int HYPRE_ADSSetAMGOptions ( HYPRE_Solver solver , HYPRE_Int coarsen_type , HYPRE_Int agg_levels , HYPRE_Int relax_type , HYPRE_Real strength_threshold , HYPRE_Int interp_type , HYPRE_Int Pmax );
HYPRE_Int HYPRE_ADSGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ADSGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *rel_resid_norm );

/* HYPRE_ame.c */
HYPRE_Int HYPRE_AMECreate ( HYPRE_Solver *esolver );
HYPRE_Int HYPRE_AMEDestroy ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESetup ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESolve ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESetAMSSolver ( HYPRE_Solver esolver , HYPRE_Solver ams_solver );
HYPRE_Int HYPRE_AMESetMassMatrix ( HYPRE_Solver esolver , HYPRE_ParCSRMatrix M );
HYPRE_Int HYPRE_AMESetBlockSize ( HYPRE_Solver esolver , HYPRE_Int block_size );
HYPRE_Int HYPRE_AMESetMaxIter ( HYPRE_Solver esolver , HYPRE_Int maxit );
HYPRE_Int HYPRE_AMESetTol ( HYPRE_Solver esolver , HYPRE_Real tol );
HYPRE_Int HYPRE_AMESetRTol ( HYPRE_Solver esolver , HYPRE_Real tol );
HYPRE_Int HYPRE_AMESetPrintLevel ( HYPRE_Solver esolver , HYPRE_Int print_level );
HYPRE_Int HYPRE_AMEGetEigenvalues ( HYPRE_Solver esolver , HYPRE_Real **eigenvalues );
HYPRE_Int HYPRE_AMEGetEigenvectors ( HYPRE_Solver esolver , HYPRE_ParVector **eigenvectors );

/* HYPRE_ams.c */
HYPRE_Int HYPRE_AMSCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_AMSDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_AMSSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSSetDimension ( HYPRE_Solver solver , HYPRE_Int dim );
HYPRE_Int HYPRE_AMSSetDiscreteGradient ( HYPRE_Solver solver , HYPRE_ParCSRMatrix G );
HYPRE_Int HYPRE_AMSSetCoordinateVectors ( HYPRE_Solver solver , HYPRE_ParVector x , HYPRE_ParVector y , HYPRE_ParVector z );
HYPRE_Int HYPRE_AMSSetEdgeConstantVectors ( HYPRE_Solver solver , HYPRE_ParVector Gx , HYPRE_ParVector Gy , HYPRE_ParVector Gz );
HYPRE_Int HYPRE_AMSSetInterpolations ( HYPRE_Solver solver , HYPRE_ParCSRMatrix Pi , HYPRE_ParCSRMatrix Pix , HYPRE_ParCSRMatrix Piy , HYPRE_ParCSRMatrix Piz );
HYPRE_Int HYPRE_AMSSetAlphaPoissonMatrix ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A_alpha );
HYPRE_Int HYPRE_AMSSetBetaPoissonMatrix ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A_beta );
HYPRE_Int HYPRE_AMSSetInteriorNodes ( HYPRE_Solver solver , HYPRE_ParVector interior_nodes );
HYPRE_Int HYPRE_AMSSetProjectionFrequency ( HYPRE_Solver solver , HYPRE_Int projection_frequency );
HYPRE_Int HYPRE_AMSSetMaxIter ( HYPRE_Solver solver , HYPRE_Int maxit );
HYPRE_Int HYPRE_AMSSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_AMSSetCycleType ( HYPRE_Solver solver , HYPRE_Int cycle_type );
HYPRE_Int HYPRE_AMSSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_AMSSetSmoothingOptions ( HYPRE_Solver solver , HYPRE_Int relax_type , HYPRE_Int relax_times , HYPRE_Real relax_weight , HYPRE_Real omega );
HYPRE_Int HYPRE_AMSSetChebySmoothingOptions ( HYPRE_Solver solver , HYPRE_Int cheby_order , HYPRE_Int cheby_fraction );
HYPRE_Int HYPRE_AMSSetAlphaAMGOptions ( HYPRE_Solver solver , HYPRE_Int alpha_coarsen_type , HYPRE_Int alpha_agg_levels , HYPRE_Int alpha_relax_type , HYPRE_Real alpha_strength_threshold , HYPRE_Int alpha_interp_type , HYPRE_Int alpha_Pmax );
HYPRE_Int HYPRE_AMSSetAlphaAMGCoarseRelaxType ( HYPRE_Solver solver , HYPRE_Int alpha_coarse_relax_type );
HYPRE_Int HYPRE_AMSSetBetaAMGOptions ( HYPRE_Solver solver , HYPRE_Int beta_coarsen_type , HYPRE_Int beta_agg_levels , HYPRE_Int beta_relax_type , HYPRE_Real beta_strength_threshold , HYPRE_Int beta_interp_type , HYPRE_Int beta_Pmax );
HYPRE_Int HYPRE_AMSSetBetaAMGCoarseRelaxType ( HYPRE_Solver solver , HYPRE_Int beta_coarse_relax_type );
HYPRE_Int HYPRE_AMSGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_AMSGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *rel_resid_norm );
HYPRE_Int HYPRE_AMSProjectOutGradients ( HYPRE_Solver solver , HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSConstructDiscreteGradient ( HYPRE_ParCSRMatrix A , HYPRE_ParVector x_coord , HYPRE_BigInt *edge_vertex , HYPRE_Int edge_orientation , HYPRE_ParCSRMatrix *G );
HYPRE_Int HYPRE_AMSFEISetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x , HYPRE_BigInt *EdgeNodeList_ , HYPRE_BigInt *NodeNumbers_ , HYPRE_Int numEdges_ , HYPRE_Int numLocalNodes_ , HYPRE_Int numNodes_ , HYPRE_Real *NodalCoord_ );
HYPRE_Int HYPRE_AMSFEIDestroy ( HYPRE_Solver solver );

/* HYPRE_parcsr_amg.c */
HYPRE_Int HYPRE_BoomerAMGCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_BoomerAMGDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BoomerAMGSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSolveT ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSetRestriction ( HYPRE_Solver solver , HYPRE_Int restr_par );
HYPRE_Int HYPRE_BoomerAMGSetIsTriangular ( HYPRE_Solver solver , HYPRE_Int is_triangular );
HYPRE_Int HYPRE_BoomerAMGSetGMRESSwitchR ( HYPRE_Solver solver , HYPRE_Int gmres_switch );
HYPRE_Int HYPRE_BoomerAMGSetMaxLevels ( HYPRE_Solver solver , HYPRE_Int max_levels );
HYPRE_Int HYPRE_BoomerAMGGetMaxLevels ( HYPRE_Solver solver , HYPRE_Int *max_levels );
HYPRE_Int HYPRE_BoomerAMGSetMaxCoarseSize ( HYPRE_Solver solver , HYPRE_Int max_coarse_size );
HYPRE_Int HYPRE_BoomerAMGGetMaxCoarseSize ( HYPRE_Solver solver , HYPRE_Int *max_coarse_size );
HYPRE_Int HYPRE_BoomerAMGSetMinCoarseSize ( HYPRE_Solver solver , HYPRE_Int min_coarse_size );
HYPRE_Int HYPRE_BoomerAMGGetMinCoarseSize ( HYPRE_Solver solver , HYPRE_Int *min_coarse_size );
HYPRE_Int HYPRE_BoomerAMGSetSeqThreshold ( HYPRE_Solver solver , HYPRE_Int seq_threshold );
HYPRE_Int HYPRE_BoomerAMGGetSeqThreshold ( HYPRE_Solver solver , HYPRE_Int *seq_threshold );
HYPRE_Int HYPRE_BoomerAMGSetRedundant ( HYPRE_Solver solver , HYPRE_Int redundant );
HYPRE_Int HYPRE_BoomerAMGGetRedundant ( HYPRE_Solver solver , HYPRE_Int *redundant );
HYPRE_Int HYPRE_BoomerAMGSetStrongThreshold ( HYPRE_Solver solver , HYPRE_Real strong_threshold );
HYPRE_Int HYPRE_BoomerAMGGetStrongThreshold ( HYPRE_Solver solver , HYPRE_Real *strong_threshold );
HYPRE_Int HYPRE_BoomerAMGSetStrongThresholdR ( HYPRE_Solver solver , HYPRE_Real strong_threshold );
HYPRE_Int HYPRE_BoomerAMGGetStrongThresholdR ( HYPRE_Solver solver , HYPRE_Real *strong_threshold );
HYPRE_Int HYPRE_BoomerAMGSetFilterThresholdR ( HYPRE_Solver solver , HYPRE_Real filter_threshold );
HYPRE_Int HYPRE_BoomerAMGGetFilterThresholdR ( HYPRE_Solver solver , HYPRE_Real *filter_threshold );
HYPRE_Int HYPRE_BoomerAMGSetGMRESSwitchR ( HYPRE_Solver solver , HYPRE_Int gmres_switch );
HYPRE_Int HYPRE_BoomerAMGSetSabs ( HYPRE_Solver solver , HYPRE_Int Sabs );
HYPRE_Int HYPRE_BoomerAMGSetMaxRowSum ( HYPRE_Solver solver , HYPRE_Real max_row_sum );
HYPRE_Int HYPRE_BoomerAMGGetMaxRowSum ( HYPRE_Solver solver , HYPRE_Real *max_row_sum );
HYPRE_Int HYPRE_BoomerAMGSetTruncFactor ( HYPRE_Solver solver , HYPRE_Real trunc_factor );
HYPRE_Int HYPRE_BoomerAMGGetTruncFactor ( HYPRE_Solver solver , HYPRE_Real *trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetPMaxElmts ( HYPRE_Solver solver , HYPRE_Int P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGGetPMaxElmts ( HYPRE_Solver solver , HYPRE_Int *P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetJacobiTruncThreshold ( HYPRE_Solver solver , HYPRE_Real jacobi_trunc_threshold );
HYPRE_Int HYPRE_BoomerAMGGetJacobiTruncThreshold ( HYPRE_Solver solver , HYPRE_Real *jacobi_trunc_threshold );
HYPRE_Int HYPRE_BoomerAMGSetPostInterpType ( HYPRE_Solver solver , HYPRE_Int post_interp_type );
HYPRE_Int HYPRE_BoomerAMGGetPostInterpType ( HYPRE_Solver solver , HYPRE_Int *post_interp_type );
HYPRE_Int HYPRE_BoomerAMGSetSCommPkgSwitch ( HYPRE_Solver solver , HYPRE_Real S_commpkg_switch );
HYPRE_Int HYPRE_BoomerAMGSetInterpType ( HYPRE_Solver solver , HYPRE_Int interp_type );
HYPRE_Int HYPRE_BoomerAMGSetSepWeight ( HYPRE_Solver solver , HYPRE_Int sep_weight );
HYPRE_Int HYPRE_BoomerAMGSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_BoomerAMGSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_BoomerAMGGetMaxIter ( HYPRE_Solver solver , HYPRE_Int *max_iter );
HYPRE_Int HYPRE_BoomerAMGSetCoarsenType ( HYPRE_Solver solver , HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_BoomerAMGGetCoarsenType ( HYPRE_Solver solver , HYPRE_Int *coarsen_type );
HYPRE_Int HYPRE_BoomerAMGSetMeasureType ( HYPRE_Solver solver , HYPRE_Int measure_type );
HYPRE_Int HYPRE_BoomerAMGGetMeasureType ( HYPRE_Solver solver , HYPRE_Int *measure_type );
HYPRE_Int HYPRE_BoomerAMGSetSetupType ( HYPRE_Solver solver , HYPRE_Int setup_type );
HYPRE_Int HYPRE_BoomerAMGSetOldDefault ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BoomerAMGSetFCycle ( HYPRE_Solver solver , HYPRE_Int fcycle );
HYPRE_Int HYPRE_BoomerAMGGetFCycle ( HYPRE_Solver solver , HYPRE_Int *fcycle );
HYPRE_Int HYPRE_BoomerAMGSetCycleType ( HYPRE_Solver solver , HYPRE_Int cycle_type );
HYPRE_Int HYPRE_BoomerAMGGetCycleType ( HYPRE_Solver solver , HYPRE_Int *cycle_type );
HYPRE_Int HYPRE_BoomerAMGSetConvergeType ( HYPRE_Solver solver , HYPRE_Int type );
HYPRE_Int HYPRE_BoomerAMGGetConvergeType ( HYPRE_Solver solver , HYPRE_Int *type );
HYPRE_Int HYPRE_BoomerAMGSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_BoomerAMGGetTol ( HYPRE_Solver solver , HYPRE_Real *tol );
HYPRE_Int HYPRE_BoomerAMGSetNumGridSweeps ( HYPRE_Solver solver , HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetNumSweeps ( HYPRE_Solver solver , HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetCycleNumSweeps ( HYPRE_Solver solver , HYPRE_Int num_sweeps , HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGGetCycleNumSweeps ( HYPRE_Solver solver , HYPRE_Int *num_sweeps , HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGInitGridRelaxation ( HYPRE_Int **num_grid_sweeps_ptr , HYPRE_Int **grid_relax_type_ptr , HYPRE_Int ***grid_relax_points_ptr , HYPRE_Int coarsen_type , HYPRE_Real **relax_weights_ptr , HYPRE_Int max_levels );
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxType ( HYPRE_Solver solver , HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_BoomerAMGSetRelaxType ( HYPRE_Solver solver , HYPRE_Int relax_type );
HYPRE_Int HYPRE_BoomerAMGSetCycleRelaxType ( HYPRE_Solver solver , HYPRE_Int relax_type , HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGGetCycleRelaxType ( HYPRE_Solver solver , HYPRE_Int *relax_type , HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGSetRelaxOrder ( HYPRE_Solver solver , HYPRE_Int relax_order );
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxPoints ( HYPRE_Solver solver , HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_BoomerAMGSetRelaxWeight ( HYPRE_Solver solver , HYPRE_Real *relax_weight );
HYPRE_Int HYPRE_BoomerAMGSetRelaxWt ( HYPRE_Solver solver , HYPRE_Real relax_wt );
HYPRE_Int HYPRE_BoomerAMGSetLevelRelaxWt ( HYPRE_Solver solver , HYPRE_Real relax_wt , HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetOmega ( HYPRE_Solver solver , HYPRE_Real *omega );
HYPRE_Int HYPRE_BoomerAMGSetOuterWt ( HYPRE_Solver solver , HYPRE_Real outer_wt );
HYPRE_Int HYPRE_BoomerAMGSetLevelOuterWt ( HYPRE_Solver solver , HYPRE_Real outer_wt , HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetSmoothType ( HYPRE_Solver solver , HYPRE_Int smooth_type );
HYPRE_Int HYPRE_BoomerAMGGetSmoothType ( HYPRE_Solver solver , HYPRE_Int *smooth_type );
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumLevels ( HYPRE_Solver solver , HYPRE_Int smooth_num_levels );
HYPRE_Int HYPRE_BoomerAMGGetSmoothNumLevels ( HYPRE_Solver solver , HYPRE_Int *smooth_num_levels );
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumSweeps ( HYPRE_Solver solver , HYPRE_Int smooth_num_sweeps );
HYPRE_Int HYPRE_BoomerAMGGetSmoothNumSweeps ( HYPRE_Solver solver , HYPRE_Int *smooth_num_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_BoomerAMGGetLogging ( HYPRE_Solver solver , HYPRE_Int *logging );
HYPRE_Int HYPRE_BoomerAMGSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_BoomerAMGGetPrintLevel ( HYPRE_Solver solver , HYPRE_Int *print_level );
HYPRE_Int HYPRE_BoomerAMGSetPrintFileName ( HYPRE_Solver solver , const char *print_file_name );
HYPRE_Int HYPRE_BoomerAMGSetDebugFlag ( HYPRE_Solver solver , HYPRE_Int debug_flag );
HYPRE_Int HYPRE_BoomerAMGGetDebugFlag ( HYPRE_Solver solver , HYPRE_Int *debug_flag );
HYPRE_Int HYPRE_BoomerAMGGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BoomerAMGGetCumNumIterations ( HYPRE_Solver solver , HYPRE_Int *cum_num_iterations );
HYPRE_Int HYPRE_BoomerAMGGetResidual ( HYPRE_Solver solver , HYPRE_ParVector *residual );
HYPRE_Int HYPRE_BoomerAMGGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *rel_resid_norm );
HYPRE_Int HYPRE_BoomerAMGSetVariant ( HYPRE_Solver solver , HYPRE_Int variant );
HYPRE_Int HYPRE_BoomerAMGGetVariant ( HYPRE_Solver solver , HYPRE_Int *variant );
HYPRE_Int HYPRE_BoomerAMGSetOverlap ( HYPRE_Solver solver , HYPRE_Int overlap );
HYPRE_Int HYPRE_BoomerAMGGetOverlap ( HYPRE_Solver solver , HYPRE_Int *overlap );
HYPRE_Int HYPRE_BoomerAMGSetDomainType ( HYPRE_Solver solver , HYPRE_Int domain_type );
HYPRE_Int HYPRE_BoomerAMGGetDomainType ( HYPRE_Solver solver , HYPRE_Int *domain_type );
HYPRE_Int HYPRE_BoomerAMGSetSchwarzRlxWeight ( HYPRE_Solver solver , HYPRE_Real schwarz_rlx_weight );
HYPRE_Int HYPRE_BoomerAMGGetSchwarzRlxWeight ( HYPRE_Solver solver , HYPRE_Real *schwarz_rlx_weight );
HYPRE_Int HYPRE_BoomerAMGSetSchwarzUseNonSymm ( HYPRE_Solver solver , HYPRE_Int use_nonsymm );
HYPRE_Int HYPRE_BoomerAMGSetSym ( HYPRE_Solver solver , HYPRE_Int sym );
HYPRE_Int HYPRE_BoomerAMGSetLevel ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetThreshold ( HYPRE_Solver solver , HYPRE_Real threshold );
HYPRE_Int HYPRE_BoomerAMGSetFilter ( HYPRE_Solver solver , HYPRE_Real filter );
HYPRE_Int HYPRE_BoomerAMGSetDropTol ( HYPRE_Solver solver , HYPRE_Real drop_tol );
HYPRE_Int HYPRE_BoomerAMGSetMaxNzPerRow ( HYPRE_Solver solver , HYPRE_Int max_nz_per_row );
HYPRE_Int HYPRE_BoomerAMGSetEuclidFile ( HYPRE_Solver solver , char *euclidfile );
HYPRE_Int HYPRE_BoomerAMGSetEuLevel ( HYPRE_Solver solver , HYPRE_Int eu_level );
HYPRE_Int HYPRE_BoomerAMGSetEuSparseA ( HYPRE_Solver solver , HYPRE_Real eu_sparse_A );
HYPRE_Int HYPRE_BoomerAMGSetEuBJ ( HYPRE_Solver solver , HYPRE_Int eu_bj );
HYPRE_Int HYPRE_BoomerAMGSetNumFunctions ( HYPRE_Solver solver , HYPRE_Int num_functions );
HYPRE_Int HYPRE_BoomerAMGGetNumFunctions ( HYPRE_Solver solver , HYPRE_Int *num_functions );
HYPRE_Int HYPRE_BoomerAMGSetNodal ( HYPRE_Solver solver , HYPRE_Int nodal );
HYPRE_Int HYPRE_BoomerAMGSetNodalLevels ( HYPRE_Solver solver , HYPRE_Int nodal_levels );
HYPRE_Int HYPRE_BoomerAMGSetNodalDiag ( HYPRE_Solver solver , HYPRE_Int nodal );
HYPRE_Int HYPRE_BoomerAMGSetDofFunc ( HYPRE_Solver solver , HYPRE_Int *dof_func );
HYPRE_Int HYPRE_BoomerAMGSetNumPaths ( HYPRE_Solver solver , HYPRE_Int num_paths );
HYPRE_Int HYPRE_BoomerAMGSetAggNumLevels ( HYPRE_Solver solver , HYPRE_Int agg_num_levels );
HYPRE_Int HYPRE_BoomerAMGSetAggInterpType ( HYPRE_Solver solver , HYPRE_Int agg_interp_type );
HYPRE_Int HYPRE_BoomerAMGSetAggTruncFactor ( HYPRE_Solver solver , HYPRE_Real agg_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAddTruncFactor ( HYPRE_Solver solver , HYPRE_Real add_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetMultAddTruncFactor ( HYPRE_Solver solver , HYPRE_Real add_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAggP12TruncFactor ( HYPRE_Solver solver , HYPRE_Real agg_P12_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAggPMaxElmts ( HYPRE_Solver solver , HYPRE_Int agg_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAddPMaxElmts ( HYPRE_Solver solver , HYPRE_Int add_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetMultAddPMaxElmts ( HYPRE_Solver solver , HYPRE_Int add_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxType ( HYPRE_Solver solver , HYPRE_Int add_rlx_type );
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxWt ( HYPRE_Solver solver , HYPRE_Real add_rlx_wt );
HYPRE_Int HYPRE_BoomerAMGSetAggP12MaxElmts ( HYPRE_Solver solver , HYPRE_Int agg_P12_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetNumCRRelaxSteps ( HYPRE_Solver solver , HYPRE_Int num_CR_relax_steps );
HYPRE_Int HYPRE_BoomerAMGSetCRRate ( HYPRE_Solver solver , HYPRE_Real CR_rate );
HYPRE_Int HYPRE_BoomerAMGSetCRStrongTh ( HYPRE_Solver solver , HYPRE_Real CR_strong_th );
HYPRE_Int HYPRE_BoomerAMGSetADropTol( HYPRE_Solver solver, HYPRE_Real A_drop_tol  );
HYPRE_Int HYPRE_BoomerAMGSetADropType( HYPRE_Solver solver, HYPRE_Int A_drop_type  );
HYPRE_Int HYPRE_BoomerAMGSetISType ( HYPRE_Solver solver , HYPRE_Int IS_type );
HYPRE_Int HYPRE_BoomerAMGSetCRUseCG ( HYPRE_Solver solver , HYPRE_Int CR_use_CG );
HYPRE_Int HYPRE_BoomerAMGSetGSMG ( HYPRE_Solver solver , HYPRE_Int gsmg );
HYPRE_Int HYPRE_BoomerAMGSetNumSamples ( HYPRE_Solver solver , HYPRE_Int gsmg );
HYPRE_Int HYPRE_BoomerAMGSetCGCIts ( HYPRE_Solver solver , HYPRE_Int its );
HYPRE_Int HYPRE_BoomerAMGSetPlotGrids ( HYPRE_Solver solver , HYPRE_Int plotgrids );
HYPRE_Int HYPRE_BoomerAMGSetPlotFileName ( HYPRE_Solver solver , const char *plotfilename );
HYPRE_Int HYPRE_BoomerAMGSetCoordDim ( HYPRE_Solver solver , HYPRE_Int coorddim );
HYPRE_Int HYPRE_BoomerAMGSetCoordinates ( HYPRE_Solver solver , float *coordinates );
HYPRE_Int HYPRE_BoomerAMGSetChebyOrder ( HYPRE_Solver solver , HYPRE_Int order );
HYPRE_Int HYPRE_BoomerAMGSetChebyFraction ( HYPRE_Solver solver , HYPRE_Real ratio );
HYPRE_Int HYPRE_BoomerAMGSetChebyEigEst ( HYPRE_Solver solver , HYPRE_Int eig_est );
HYPRE_Int HYPRE_BoomerAMGSetChebyVariant ( HYPRE_Solver solver , HYPRE_Int variant );
HYPRE_Int HYPRE_BoomerAMGSetChebyScale ( HYPRE_Solver solver , HYPRE_Int scale );
HYPRE_Int HYPRE_BoomerAMGSetInterpVectors ( HYPRE_Solver solver , HYPRE_Int num_vectors , HYPRE_ParVector *vectors );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecVariant ( HYPRE_Solver solver , HYPRE_Int num );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecQMax ( HYPRE_Solver solver , HYPRE_Int q_max );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecAbsQTrunc ( HYPRE_Solver solver , HYPRE_Real q_trunc );
HYPRE_Int HYPRE_BoomerAMGSetSmoothInterpVectors ( HYPRE_Solver solver , HYPRE_Int smooth_interp_vectors );
HYPRE_Int HYPRE_BoomerAMGSetInterpRefine ( HYPRE_Solver solver , HYPRE_Int num_refine );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecFirstLevel ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetAdditive ( HYPRE_Solver solver , HYPRE_Int additive );
HYPRE_Int HYPRE_BoomerAMGGetAdditive ( HYPRE_Solver solver , HYPRE_Int *additive );
HYPRE_Int HYPRE_BoomerAMGSetMultAdditive ( HYPRE_Solver solver , HYPRE_Int mult_additive );
HYPRE_Int HYPRE_BoomerAMGGetMultAdditive ( HYPRE_Solver solver , HYPRE_Int *mult_additive );
HYPRE_Int HYPRE_BoomerAMGSetSimple ( HYPRE_Solver solver , HYPRE_Int simple );
HYPRE_Int HYPRE_BoomerAMGGetSimple ( HYPRE_Solver solver , HYPRE_Int *simple );
HYPRE_Int HYPRE_BoomerAMGSetAddLastLvl ( HYPRE_Solver solver , HYPRE_Int add_last_lvl );
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkinTol ( HYPRE_Solver solver , HYPRE_Real nongalerkin_tol );
HYPRE_Int HYPRE_BoomerAMGSetLevelNonGalerkinTol ( HYPRE_Solver solver , HYPRE_Real nongalerkin_tol , HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkTol ( HYPRE_Solver solver , HYPRE_Int nongalerk_num_tol , HYPRE_Real *nongalerk_tol );
HYPRE_Int HYPRE_BoomerAMGSetRAP2 ( HYPRE_Solver solver , HYPRE_Int rap2 );
HYPRE_Int HYPRE_BoomerAMGSetModuleRAP2 ( HYPRE_Solver solver , HYPRE_Int mod_rap2 );
HYPRE_Int HYPRE_BoomerAMGSetKeepTranspose ( HYPRE_Solver solver , HYPRE_Int keepTranspose );
#ifdef HYPRE_USING_DSUPERLU
HYPRE_Int HYPRE_BoomerAMGSetDSLUThreshold ( HYPRE_Solver solver , HYPRE_Int slu_threshold );
#endif
HYPRE_Int HYPRE_BoomerAMGSetCpointsToKeep( HYPRE_Solver solver, HYPRE_Int cpt_coarse_level, HYPRE_Int num_cpt_coarse,HYPRE_Int *cpt_coarse_index);

/* HYPRE_parcsr_bicgstab.c */
HYPRE_Int HYPRE_ParCSRBiCGSTABCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRBiCGSTABSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetAbsoluteTol ( HYPRE_Solver solver , HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetStopCrit ( HYPRE_Solver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetResidual ( HYPRE_Solver solver , HYPRE_ParVector *residual );

/* HYPRE_parcsr_block.c */
HYPRE_Int HYPRE_BlockTridiagCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_BlockTridiagDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BlockTridiagSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_BlockTridiagSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_BlockTridiagSetIndexSet ( HYPRE_Solver solver , HYPRE_Int n , HYPRE_Int *inds );
HYPRE_Int HYPRE_BlockTridiagSetAMGStrengthThreshold ( HYPRE_Solver solver , HYPRE_Real thresh );
HYPRE_Int HYPRE_BlockTridiagSetAMGNumSweeps ( HYPRE_Solver solver , HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_BlockTridiagSetAMGRelaxType ( HYPRE_Solver solver , HYPRE_Int relax_type );
HYPRE_Int HYPRE_BlockTridiagSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );

/* HYPRE_parcsr_cgnr.c */
HYPRE_Int HYPRE_ParCSRCGNRCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRCGNRDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRCGNRSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCGNRSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCGNRSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRCGNRSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRCGNRSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRCGNRSetStopCrit ( HYPRE_Solver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRCGNRSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precondT , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRCGNRGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRCGNRSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRCGNRGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *norm );

/* HYPRE_parcsr_Euclid.c */
HYPRE_Int HYPRE_EuclidCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_EuclidDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_EuclidSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_EuclidSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector bb , HYPRE_ParVector xx );
HYPRE_Int HYPRE_EuclidSetParams ( HYPRE_Solver solver , HYPRE_Int argc , char *argv []);
HYPRE_Int HYPRE_EuclidSetParamsFromFile ( HYPRE_Solver solver , char *filename );
HYPRE_Int HYPRE_EuclidSetLevel ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_EuclidSetBJ ( HYPRE_Solver solver , HYPRE_Int bj );
HYPRE_Int HYPRE_EuclidSetStats ( HYPRE_Solver solver , HYPRE_Int eu_stats );
HYPRE_Int HYPRE_EuclidSetMem ( HYPRE_Solver solver , HYPRE_Int eu_mem );
HYPRE_Int HYPRE_EuclidSetSparseA ( HYPRE_Solver solver , HYPRE_Real sparse_A );
HYPRE_Int HYPRE_EuclidSetRowScale ( HYPRE_Solver solver , HYPRE_Int row_scale );
HYPRE_Int HYPRE_EuclidSetILUT ( HYPRE_Solver solver , HYPRE_Real ilut );

/* HYPRE_parcsr_flexgmres.c */
HYPRE_Int HYPRE_ParCSRFlexGMRESCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRFlexGMRESSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetKDim ( HYPRE_Solver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetAbsoluteTol ( HYPRE_Solver solver , HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetResidual ( HYPRE_Solver solver , HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetModifyPC ( HYPRE_Solver solver , HYPRE_PtrToModifyPCFcn modify_pc );

/* HYPRE_parcsr_gmres.c */
HYPRE_Int HYPRE_ParCSRGMRESCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRGMRESDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRGMRESSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRGMRESSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRGMRESSetKDim ( HYPRE_Solver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRGMRESSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRGMRESSetAbsoluteTol ( HYPRE_Solver solver , HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRGMRESSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRGMRESSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRGMRESSetStopCrit ( HYPRE_Solver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRGMRESSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRGMRESSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRGMRESGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRGMRESGetResidual ( HYPRE_Solver solver , HYPRE_ParVector *residual );


/*HYPRE_parcsr_cogmres.c*/
HYPRE_Int HYPRE_ParCSRCOGMRESCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRCOGMRESDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRCOGMRESSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCOGMRESSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCOGMRESSetKDim ( HYPRE_Solver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRCOGMRESSetCGS2 ( HYPRE_Solver solver , HYPRE_Int cgs2 );
HYPRE_Int HYPRE_ParCSRCOGMRESSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRCOGMRESSetAbsoluteTol ( HYPRE_Solver solver , HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRCOGMRESSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRCOGMRESSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRCOGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRCOGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRCOGMRESSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRCOGMRESSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRCOGMRESGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRCOGMRESGetResidual ( HYPRE_Solver solver , HYPRE_ParVector *residual );



/* HYPRE_parcsr_hybrid.c */
HYPRE_Int HYPRE_ParCSRHybridCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRHybridDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRHybridSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRHybridSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRHybridSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRHybridSetAbsoluteTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRHybridSetConvergenceTol ( HYPRE_Solver solver , HYPRE_Real cf_tol );
HYPRE_Int HYPRE_ParCSRHybridSetDSCGMaxIter ( HYPRE_Solver solver , HYPRE_Int dscg_max_its );
HYPRE_Int HYPRE_ParCSRHybridSetPCGMaxIter ( HYPRE_Solver solver , HYPRE_Int pcg_max_its );
HYPRE_Int HYPRE_ParCSRHybridSetSetupType ( HYPRE_Solver solver , HYPRE_Int setup_type );
HYPRE_Int HYPRE_ParCSRHybridSetSolverType ( HYPRE_Solver solver , HYPRE_Int solver_type );
HYPRE_Int HYPRE_ParCSRHybridSetKDim ( HYPRE_Solver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRHybridSetTwoNorm ( HYPRE_Solver solver , HYPRE_Int two_norm );
HYPRE_Int HYPRE_ParCSRHybridSetStopCrit ( HYPRE_Solver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRHybridSetRelChange ( HYPRE_Solver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_ParCSRHybridSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRHybridSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRHybridSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRHybridSetStrongThreshold ( HYPRE_Solver solver , HYPRE_Real strong_threshold );
HYPRE_Int HYPRE_ParCSRHybridSetMaxRowSum ( HYPRE_Solver solver , HYPRE_Real max_row_sum );
HYPRE_Int HYPRE_ParCSRHybridSetTruncFactor ( HYPRE_Solver solver , HYPRE_Real trunc_factor );
HYPRE_Int HYPRE_ParCSRHybridSetPMaxElmts ( HYPRE_Solver solver , HYPRE_Int p_max );
HYPRE_Int HYPRE_ParCSRHybridSetMaxLevels ( HYPRE_Solver solver , HYPRE_Int max_levels );
HYPRE_Int HYPRE_ParCSRHybridSetMeasureType ( HYPRE_Solver solver , HYPRE_Int measure_type );
HYPRE_Int HYPRE_ParCSRHybridSetCoarsenType ( HYPRE_Solver solver , HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_ParCSRHybridSetInterpType ( HYPRE_Solver solver , HYPRE_Int interp_type );
HYPRE_Int HYPRE_ParCSRHybridSetCycleType ( HYPRE_Solver solver , HYPRE_Int cycle_type );
HYPRE_Int HYPRE_ParCSRHybridSetNumGridSweeps ( HYPRE_Solver solver , HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_ParCSRHybridSetGridRelaxType ( HYPRE_Solver solver , HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_ParCSRHybridSetGridRelaxPoints ( HYPRE_Solver solver , HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_ParCSRHybridSetNumSweeps ( HYPRE_Solver solver , HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_ParCSRHybridSetCycleNumSweeps ( HYPRE_Solver solver , HYPRE_Int num_sweeps , HYPRE_Int k );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxType ( HYPRE_Solver solver , HYPRE_Int relax_type );
HYPRE_Int HYPRE_ParCSRHybridSetKeepTranspose ( HYPRE_Solver solver , HYPRE_Int keepT );
HYPRE_Int HYPRE_ParCSRHybridSetCycleRelaxType ( HYPRE_Solver solver , HYPRE_Int relax_type , HYPRE_Int k );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxOrder ( HYPRE_Solver solver , HYPRE_Int relax_order );
HYPRE_Int HYPRE_ParCSRHybridSetMaxCoarseSize ( HYPRE_Solver solver , HYPRE_Int max_coarse_size );
HYPRE_Int HYPRE_ParCSRHybridSetMinCoarseSize ( HYPRE_Solver solver , HYPRE_Int min_coarse_size );
HYPRE_Int HYPRE_ParCSRHybridSetSeqThreshold ( HYPRE_Solver solver , HYPRE_Int seq_threshold );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxWt ( HYPRE_Solver solver , HYPRE_Real relax_wt );
HYPRE_Int HYPRE_ParCSRHybridSetLevelRelaxWt ( HYPRE_Solver solver , HYPRE_Real relax_wt , HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRHybridSetOuterWt ( HYPRE_Solver solver , HYPRE_Real outer_wt );
HYPRE_Int HYPRE_ParCSRHybridSetLevelOuterWt ( HYPRE_Solver solver , HYPRE_Real outer_wt , HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxWeight ( HYPRE_Solver solver , HYPRE_Real *relax_weight );
HYPRE_Int HYPRE_ParCSRHybridSetOmega ( HYPRE_Solver solver , HYPRE_Real *omega );
HYPRE_Int HYPRE_ParCSRHybridSetAggNumLevels ( HYPRE_Solver solver , HYPRE_Int agg_num_levels );
HYPRE_Int HYPRE_ParCSRHybridSetNumPaths ( HYPRE_Solver solver , HYPRE_Int num_paths );
HYPRE_Int HYPRE_ParCSRHybridSetNumFunctions ( HYPRE_Solver solver , HYPRE_Int num_functions );
HYPRE_Int HYPRE_ParCSRHybridSetNodal ( HYPRE_Solver solver , HYPRE_Int nodal );
HYPRE_Int HYPRE_ParCSRHybridSetDofFunc ( HYPRE_Solver solver , HYPRE_Int *dof_func );
HYPRE_Int HYPRE_ParCSRHybridSetNonGalerkinTol ( HYPRE_Solver solver , HYPRE_Int nongalerk_num_tol, HYPRE_Real *nongalerkin_tol );
HYPRE_Int HYPRE_ParCSRHybridGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_its );
HYPRE_Int HYPRE_ParCSRHybridGetDSCGNumIterations ( HYPRE_Solver solver , HYPRE_Int *dscg_num_its );
HYPRE_Int HYPRE_ParCSRHybridGetPCGNumIterations ( HYPRE_Solver solver , HYPRE_Int *pcg_num_its );
HYPRE_Int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *norm );

/* HYPRE_parcsr_int.c */
HYPRE_Int hypre_ParSetRandomValues ( void *v , HYPRE_Int seed );
HYPRE_Int hypre_ParPrintVector ( void *v , const char *file );
void *hypre_ParReadVector ( MPI_Comm comm , const char *file );
HYPRE_Int hypre_ParVectorSize ( void *x );
HYPRE_Int hypre_ParCSRMultiVectorPrint ( void *x_ , const char *fileName );
void *hypre_ParCSRMultiVectorRead ( MPI_Comm comm , void *ii_ , const char *fileName );
HYPRE_Int aux_maskCount ( HYPRE_Int n , HYPRE_Int *mask );
void aux_indexFromMask ( HYPRE_Int n , HYPRE_Int *mask , HYPRE_Int *index );
HYPRE_Int HYPRE_TempParCSRSetupInterpreter ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_ParCSRSetupInterpreter ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_ParCSRSetupMatvec ( HYPRE_MatvecFunctions *mv );

/* HYPRE_parcsr_lgmres.c */
HYPRE_Int HYPRE_ParCSRLGMRESCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRLGMRESDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRLGMRESSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRLGMRESSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRLGMRESSetKDim ( HYPRE_Solver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRLGMRESSetAugDim ( HYPRE_Solver solver , HYPRE_Int aug_dim );
HYPRE_Int HYPRE_ParCSRLGMRESSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRLGMRESSetAbsoluteTol ( HYPRE_Solver solver , HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRLGMRESSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRLGMRESSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRLGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRLGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRLGMRESSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRLGMRESSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRLGMRESGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRLGMRESGetResidual ( HYPRE_Solver solver , HYPRE_ParVector *residual );

/* HYPRE_parcsr_ParaSails.c */
HYPRE_Int HYPRE_ParCSRParaSailsCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRParaSailsDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRParaSailsSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRParaSailsSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRParaSailsSetParams ( HYPRE_Solver solver , HYPRE_Real thresh , HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParCSRParaSailsSetFilter ( HYPRE_Solver solver , HYPRE_Real filter );
HYPRE_Int HYPRE_ParCSRParaSailsGetFilter ( HYPRE_Solver solver , HYPRE_Real *filter );
HYPRE_Int HYPRE_ParCSRParaSailsSetSym ( HYPRE_Solver solver , HYPRE_Int sym );
HYPRE_Int HYPRE_ParCSRParaSailsSetLoadbal ( HYPRE_Solver solver , HYPRE_Real loadbal );
HYPRE_Int HYPRE_ParCSRParaSailsGetLoadbal ( HYPRE_Solver solver , HYPRE_Real *loadbal );
HYPRE_Int HYPRE_ParCSRParaSailsSetReuse ( HYPRE_Solver solver , HYPRE_Int reuse );
HYPRE_Int HYPRE_ParCSRParaSailsSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_ParaSailsCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParaSailsDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParaSailsSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParaSailsSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParaSailsSetParams ( HYPRE_Solver solver , HYPRE_Real thresh , HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParaSailsSetThresh ( HYPRE_Solver solver , HYPRE_Real thresh );
HYPRE_Int HYPRE_ParaSailsGetThresh ( HYPRE_Solver solver , HYPRE_Real *thresh );
HYPRE_Int HYPRE_ParaSailsSetNlevels ( HYPRE_Solver solver , HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParaSailsGetNlevels ( HYPRE_Solver solver , HYPRE_Int *nlevels );
HYPRE_Int HYPRE_ParaSailsSetFilter ( HYPRE_Solver solver , HYPRE_Real filter );
HYPRE_Int HYPRE_ParaSailsGetFilter ( HYPRE_Solver solver , HYPRE_Real *filter );
HYPRE_Int HYPRE_ParaSailsSetSym ( HYPRE_Solver solver , HYPRE_Int sym );
HYPRE_Int HYPRE_ParaSailsGetSym ( HYPRE_Solver solver , HYPRE_Int *sym );
HYPRE_Int HYPRE_ParaSailsSetLoadbal ( HYPRE_Solver solver , HYPRE_Real loadbal );
HYPRE_Int HYPRE_ParaSailsGetLoadbal ( HYPRE_Solver solver , HYPRE_Real *loadbal );
HYPRE_Int HYPRE_ParaSailsSetReuse ( HYPRE_Solver solver , HYPRE_Int reuse );
HYPRE_Int HYPRE_ParaSailsGetReuse ( HYPRE_Solver solver , HYPRE_Int *reuse );
HYPRE_Int HYPRE_ParaSailsSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_ParaSailsGetLogging ( HYPRE_Solver solver , HYPRE_Int *logging );
HYPRE_Int HYPRE_ParaSailsBuildIJMatrix ( HYPRE_Solver solver , HYPRE_IJMatrix *pij_A );

/* HYPRE_parcsr_pcg.c */
HYPRE_Int HYPRE_ParCSRPCGCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRPCGDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRPCGSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPCGSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPCGSetTol ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRPCGSetAbsoluteTol ( HYPRE_Solver solver , HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRPCGSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRPCGSetStopCrit ( HYPRE_Solver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRPCGSetTwoNorm ( HYPRE_Solver solver , HYPRE_Int two_norm );
HYPRE_Int HYPRE_ParCSRPCGSetRelChange ( HYPRE_Solver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_ParCSRPCGSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRPCGGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRPCGSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRPCGSetLogging ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRPCGGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm ( HYPRE_Solver solver , HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRPCGGetResidual ( HYPRE_Solver solver , HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRDiagScaleSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector y , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRDiagScale ( HYPRE_Solver solver , HYPRE_ParCSRMatrix HA , HYPRE_ParVector Hy , HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSROnProcTriSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix HA , HYPRE_ParVector Hy , HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSROnProcTriSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix HA , HYPRE_ParVector Hy , HYPRE_ParVector Hx );

/* HYPRE_parcsr_pilut.c */
HYPRE_Int HYPRE_ParCSRPilutCreate ( MPI_Comm comm , HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRPilutDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRPilutSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPilutSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPilutSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRPilutSetDropTolerance ( HYPRE_Solver solver , HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRPilutSetFactorRowSize ( HYPRE_Solver solver , HYPRE_Int size );

/* HYPRE_parcsr_schwarz.c */
HYPRE_Int HYPRE_SchwarzCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_SchwarzDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_SchwarzSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_SchwarzSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
HYPRE_Int HYPRE_SchwarzSetVariant ( HYPRE_Solver solver , HYPRE_Int variant );
HYPRE_Int HYPRE_SchwarzSetOverlap ( HYPRE_Solver solver , HYPRE_Int overlap );
HYPRE_Int HYPRE_SchwarzSetDomainType ( HYPRE_Solver solver , HYPRE_Int domain_type );
HYPRE_Int HYPRE_SchwarzSetDomainStructure ( HYPRE_Solver solver , HYPRE_CSRMatrix domain_structure );
HYPRE_Int HYPRE_SchwarzSetNumFunctions ( HYPRE_Solver solver , HYPRE_Int num_functions );
HYPRE_Int HYPRE_SchwarzSetNonSymm ( HYPRE_Solver solver , HYPRE_Int use_nonsymm );
HYPRE_Int HYPRE_SchwarzSetRelaxWeight ( HYPRE_Solver solver , HYPRE_Real relax_weight );
HYPRE_Int HYPRE_SchwarzSetDofFunc ( HYPRE_Solver solver , HYPRE_Int *dof_func );

/* par_add_cycle.c */
HYPRE_Int hypre_BoomerAMGAdditiveCycle ( void *amg_vdata );
HYPRE_Int hypre_CreateLambda ( void *amg_vdata );
HYPRE_Int hypre_CreateDinv ( void *amg_vdata );

/* par_amg.c */
void *hypre_BoomerAMGCreate ( void );
HYPRE_Int hypre_BoomerAMGDestroy ( void *data );
HYPRE_Int hypre_BoomerAMGSetRestriction ( void *data , HYPRE_Int restr_par );
HYPRE_Int hypre_BoomerAMGSetIsTriangular ( void *data , HYPRE_Int is_triangular );
HYPRE_Int hypre_BoomerAMGSetGMRESSwitchR ( void *data , HYPRE_Int gmres_switch );
HYPRE_Int hypre_BoomerAMGSetMaxLevels ( void *data , HYPRE_Int max_levels );
HYPRE_Int hypre_BoomerAMGGetMaxLevels ( void *data , HYPRE_Int *max_levels );
HYPRE_Int hypre_BoomerAMGSetMaxCoarseSize ( void *data , HYPRE_Int max_coarse_size );
HYPRE_Int hypre_BoomerAMGGetMaxCoarseSize ( void *data , HYPRE_Int *max_coarse_size );
HYPRE_Int hypre_BoomerAMGSetMinCoarseSize ( void *data , HYPRE_Int min_coarse_size );
HYPRE_Int hypre_BoomerAMGGetMinCoarseSize ( void *data , HYPRE_Int *min_coarse_size );
HYPRE_Int hypre_BoomerAMGSetSeqThreshold ( void *data , HYPRE_Int seq_threshold );
HYPRE_Int hypre_BoomerAMGGetSeqThreshold ( void *data , HYPRE_Int *seq_threshold );
HYPRE_Int hypre_BoomerAMGSetRedundant ( void *data , HYPRE_Int redundant );
HYPRE_Int hypre_BoomerAMGGetRedundant ( void *data , HYPRE_Int *redundant );
HYPRE_Int hypre_BoomerAMGSetStrongThreshold ( void *data , HYPRE_Real strong_threshold );
HYPRE_Int hypre_BoomerAMGGetStrongThreshold ( void *data , HYPRE_Real *strong_threshold );
HYPRE_Int hypre_BoomerAMGSetStrongThresholdR ( void *data , HYPRE_Real strong_threshold );
HYPRE_Int hypre_BoomerAMGGetStrongThresholdR ( void *data , HYPRE_Real *strong_threshold );
HYPRE_Int hypre_BoomerAMGSetFilterThresholdR ( void *data , HYPRE_Real filter_threshold );
HYPRE_Int hypre_BoomerAMGGetFilterThresholdR ( void *data , HYPRE_Real *filter_threshold );
HYPRE_Int hypre_BoomerAMGSetSabs ( void *data , HYPRE_Int Sabs );
HYPRE_Int hypre_BoomerAMGSetMaxRowSum ( void *data , HYPRE_Real max_row_sum );
HYPRE_Int hypre_BoomerAMGGetMaxRowSum ( void *data , HYPRE_Real *max_row_sum );
HYPRE_Int hypre_BoomerAMGSetTruncFactor ( void *data , HYPRE_Real trunc_factor );
HYPRE_Int hypre_BoomerAMGGetTruncFactor ( void *data , HYPRE_Real *trunc_factor );
HYPRE_Int hypre_BoomerAMGSetPMaxElmts ( void *data , HYPRE_Int P_max_elmts );
HYPRE_Int hypre_BoomerAMGGetPMaxElmts ( void *data , HYPRE_Int *P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetJacobiTruncThreshold ( void *data , HYPRE_Real jacobi_trunc_threshold );
HYPRE_Int hypre_BoomerAMGGetJacobiTruncThreshold ( void *data , HYPRE_Real *jacobi_trunc_threshold );
HYPRE_Int hypre_BoomerAMGSetPostInterpType ( void *data , HYPRE_Int post_interp_type );
HYPRE_Int hypre_BoomerAMGGetPostInterpType ( void *data , HYPRE_Int *post_interp_type );
HYPRE_Int hypre_BoomerAMGSetSCommPkgSwitch ( void *data , HYPRE_Real S_commpkg_switch );
HYPRE_Int hypre_BoomerAMGGetSCommPkgSwitch ( void *data , HYPRE_Real *S_commpkg_switch );
HYPRE_Int hypre_BoomerAMGSetInterpType ( void *data , HYPRE_Int interp_type );
HYPRE_Int hypre_BoomerAMGGetInterpType ( void *data , HYPRE_Int *interp_type );
HYPRE_Int hypre_BoomerAMGSetSepWeight ( void *data , HYPRE_Int sep_weight );
HYPRE_Int hypre_BoomerAMGSetMinIter ( void *data , HYPRE_Int min_iter );
HYPRE_Int hypre_BoomerAMGGetMinIter ( void *data , HYPRE_Int *min_iter );
HYPRE_Int hypre_BoomerAMGSetMaxIter ( void *data , HYPRE_Int max_iter );
HYPRE_Int hypre_BoomerAMGGetMaxIter ( void *data , HYPRE_Int *max_iter );
HYPRE_Int hypre_BoomerAMGSetCoarsenType ( void *data , HYPRE_Int coarsen_type );
HYPRE_Int hypre_BoomerAMGGetCoarsenType ( void *data , HYPRE_Int *coarsen_type );
HYPRE_Int hypre_BoomerAMGSetMeasureType ( void *data , HYPRE_Int measure_type );
HYPRE_Int hypre_BoomerAMGGetMeasureType ( void *data , HYPRE_Int *measure_type );
HYPRE_Int hypre_BoomerAMGSetSetupType ( void *data , HYPRE_Int setup_type );
HYPRE_Int hypre_BoomerAMGGetSetupType ( void *data , HYPRE_Int *setup_type );
HYPRE_Int hypre_BoomerAMGSetFCycle ( void *data , HYPRE_Int fcycle );
HYPRE_Int hypre_BoomerAMGGetFCycle ( void *data , HYPRE_Int *fcycle );
HYPRE_Int hypre_BoomerAMGSetCycleType ( void *data , HYPRE_Int cycle_type );
HYPRE_Int hypre_BoomerAMGGetCycleType ( void *data , HYPRE_Int *cycle_type );
HYPRE_Int hypre_BoomerAMGSetConvergeType ( void *data , HYPRE_Int type );
HYPRE_Int hypre_BoomerAMGGetConvergeType ( void *data , HYPRE_Int *type );
HYPRE_Int hypre_BoomerAMGSetTol ( void *data , HYPRE_Real tol );
HYPRE_Int hypre_BoomerAMGGetTol ( void *data , HYPRE_Real *tol );
HYPRE_Int hypre_BoomerAMGSetNumSweeps ( void *data , HYPRE_Int num_sweeps );
HYPRE_Int hypre_BoomerAMGSetCycleNumSweeps ( void *data , HYPRE_Int num_sweeps , HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleNumSweeps ( void *data , HYPRE_Int *num_sweeps , HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetNumGridSweeps ( void *data , HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGGetNumGridSweeps ( void *data , HYPRE_Int **num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGSetRelaxType ( void *data , HYPRE_Int relax_type );
HYPRE_Int hypre_BoomerAMGSetCycleRelaxType ( void *data , HYPRE_Int relax_type , HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleRelaxType ( void *data , HYPRE_Int *relax_type , HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetRelaxOrder ( void *data , HYPRE_Int relax_order );
HYPRE_Int hypre_BoomerAMGGetRelaxOrder ( void *data , HYPRE_Int *relax_order );
HYPRE_Int hypre_BoomerAMGSetGridRelaxType ( void *data , HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_BoomerAMGGetGridRelaxType ( void *data , HYPRE_Int **grid_relax_type );
HYPRE_Int hypre_BoomerAMGSetGridRelaxPoints ( void *data , HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_BoomerAMGGetGridRelaxPoints ( void *data , HYPRE_Int ***grid_relax_points );
HYPRE_Int hypre_BoomerAMGSetRelaxWeight ( void *data , HYPRE_Real *relax_weight );
HYPRE_Int hypre_BoomerAMGGetRelaxWeight ( void *data , HYPRE_Real **relax_weight );
HYPRE_Int hypre_BoomerAMGSetRelaxWt ( void *data , HYPRE_Real relax_weight );
HYPRE_Int hypre_BoomerAMGSetLevelRelaxWt ( void *data , HYPRE_Real relax_weight , HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGGetLevelRelaxWt ( void *data , HYPRE_Real *relax_weight , HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetOmega ( void *data , HYPRE_Real *omega );
HYPRE_Int hypre_BoomerAMGGetOmega ( void *data , HYPRE_Real **omega );
HYPRE_Int hypre_BoomerAMGSetOuterWt ( void *data , HYPRE_Real omega );
HYPRE_Int hypre_BoomerAMGSetLevelOuterWt ( void *data , HYPRE_Real omega , HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGGetLevelOuterWt ( void *data , HYPRE_Real *omega , HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetSmoothType ( void *data , HYPRE_Int smooth_type );
HYPRE_Int hypre_BoomerAMGGetSmoothType ( void *data , HYPRE_Int *smooth_type );
HYPRE_Int hypre_BoomerAMGSetSmoothNumLevels ( void *data , HYPRE_Int smooth_num_levels );
HYPRE_Int hypre_BoomerAMGGetSmoothNumLevels ( void *data , HYPRE_Int *smooth_num_levels );
HYPRE_Int hypre_BoomerAMGSetSmoothNumSweeps ( void *data , HYPRE_Int smooth_num_sweeps );
HYPRE_Int hypre_BoomerAMGGetSmoothNumSweeps ( void *data , HYPRE_Int *smooth_num_sweeps );
HYPRE_Int hypre_BoomerAMGSetLogging ( void *data , HYPRE_Int logging );
HYPRE_Int hypre_BoomerAMGGetLogging ( void *data , HYPRE_Int *logging );
HYPRE_Int hypre_BoomerAMGSetPrintLevel ( void *data , HYPRE_Int print_level );
HYPRE_Int hypre_BoomerAMGGetPrintLevel ( void *data , HYPRE_Int *print_level );
HYPRE_Int hypre_BoomerAMGSetPrintFileName ( void *data , const char *print_file_name );
HYPRE_Int hypre_BoomerAMGGetPrintFileName ( void *data , char **print_file_name );
HYPRE_Int hypre_BoomerAMGSetNumIterations ( void *data , HYPRE_Int num_iterations );
HYPRE_Int hypre_BoomerAMGSetDebugFlag ( void *data , HYPRE_Int debug_flag );
HYPRE_Int hypre_BoomerAMGGetDebugFlag ( void *data , HYPRE_Int *debug_flag );
HYPRE_Int hypre_BoomerAMGSetGSMG ( void *data , HYPRE_Int par );
HYPRE_Int hypre_BoomerAMGSetNumSamples ( void *data , HYPRE_Int par );
HYPRE_Int hypre_BoomerAMGSetCGCIts ( void *data , HYPRE_Int its );
HYPRE_Int hypre_BoomerAMGSetPlotGrids ( void *data , HYPRE_Int plotgrids );
HYPRE_Int hypre_BoomerAMGSetPlotFileName ( void *data , const char *plot_file_name );
HYPRE_Int hypre_BoomerAMGSetCoordDim ( void *data , HYPRE_Int coorddim );
HYPRE_Int hypre_BoomerAMGSetCoordinates ( void *data , float *coordinates );
HYPRE_Int hypre_BoomerAMGSetNumFunctions ( void *data , HYPRE_Int num_functions );
HYPRE_Int hypre_BoomerAMGGetNumFunctions ( void *data , HYPRE_Int *num_functions );
HYPRE_Int hypre_BoomerAMGSetNodal ( void *data , HYPRE_Int nodal );
HYPRE_Int hypre_BoomerAMGSetNodalLevels ( void *data , HYPRE_Int nodal_levels );
HYPRE_Int hypre_BoomerAMGSetNodalDiag ( void *data , HYPRE_Int nodal );
HYPRE_Int hypre_BoomerAMGSetNumPaths ( void *data , HYPRE_Int num_paths );
HYPRE_Int hypre_BoomerAMGSetAggNumLevels ( void *data , HYPRE_Int agg_num_levels );
HYPRE_Int hypre_BoomerAMGSetAggInterpType ( void *data , HYPRE_Int agg_interp_type );
HYPRE_Int hypre_BoomerAMGSetAggPMaxElmts ( void *data , HYPRE_Int agg_P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetMultAddPMaxElmts ( void *data , HYPRE_Int add_P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetAddRelaxType ( void *data , HYPRE_Int add_rlx_type );
HYPRE_Int hypre_BoomerAMGSetAddRelaxWt ( void *data , HYPRE_Real add_rlx_wt );
HYPRE_Int hypre_BoomerAMGSetAggP12MaxElmts ( void *data , HYPRE_Int agg_P12_max_elmts );
HYPRE_Int hypre_BoomerAMGSetAggTruncFactor ( void *data , HYPRE_Real agg_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetMultAddTruncFactor ( void *data , HYPRE_Real add_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetAggP12TruncFactor ( void *data , HYPRE_Real agg_P12_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetNumCRRelaxSteps ( void *data , HYPRE_Int num_CR_relax_steps );
HYPRE_Int hypre_BoomerAMGSetCRRate ( void *data , HYPRE_Real CR_rate );
HYPRE_Int hypre_BoomerAMGSetCRStrongTh ( void *data , HYPRE_Real CR_strong_th );
HYPRE_Int hypre_BoomerAMGSetADropTol( void     *data, HYPRE_Real  A_drop_tol );
HYPRE_Int hypre_BoomerAMGSetADropType( void     *data, HYPRE_Int  A_drop_type );
HYPRE_Int hypre_BoomerAMGSetISType ( void *data , HYPRE_Int IS_type );
HYPRE_Int hypre_BoomerAMGSetCRUseCG ( void *data , HYPRE_Int CR_use_CG );
HYPRE_Int hypre_BoomerAMGSetNumPoints ( void *data , HYPRE_Int num_points );
HYPRE_Int hypre_BoomerAMGSetDofFunc ( void *data , HYPRE_Int *dof_func );
HYPRE_Int hypre_BoomerAMGSetPointDofMap ( void *data , HYPRE_Int *point_dof_map );
HYPRE_Int hypre_BoomerAMGSetDofPoint ( void *data , HYPRE_Int *dof_point );
HYPRE_Int hypre_BoomerAMGGetNumIterations ( void *data , HYPRE_Int *num_iterations );
HYPRE_Int hypre_BoomerAMGGetCumNumIterations ( void *data , HYPRE_Int *cum_num_iterations );
HYPRE_Int hypre_BoomerAMGGetResidual ( void *data , hypre_ParVector **resid );
HYPRE_Int hypre_BoomerAMGGetRelResidualNorm ( void *data , HYPRE_Real *rel_resid_norm );
HYPRE_Int hypre_BoomerAMGSetVariant ( void *data , HYPRE_Int variant );
HYPRE_Int hypre_BoomerAMGGetVariant ( void *data , HYPRE_Int *variant );
HYPRE_Int hypre_BoomerAMGSetOverlap ( void *data , HYPRE_Int overlap );
HYPRE_Int hypre_BoomerAMGGetOverlap ( void *data , HYPRE_Int *overlap );
HYPRE_Int hypre_BoomerAMGSetDomainType ( void *data , HYPRE_Int domain_type );
HYPRE_Int hypre_BoomerAMGGetDomainType ( void *data , HYPRE_Int *domain_type );
HYPRE_Int hypre_BoomerAMGSetSchwarzRlxWeight ( void *data , HYPRE_Real schwarz_rlx_weight );
HYPRE_Int hypre_BoomerAMGGetSchwarzRlxWeight ( void *data , HYPRE_Real *schwarz_rlx_weight );
HYPRE_Int hypre_BoomerAMGSetSchwarzUseNonSymm ( void *data , HYPRE_Int use_nonsymm );
HYPRE_Int hypre_BoomerAMGSetSym ( void *data , HYPRE_Int sym );
HYPRE_Int hypre_BoomerAMGSetLevel ( void *data , HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetThreshold ( void *data , HYPRE_Real thresh );
HYPRE_Int hypre_BoomerAMGSetFilter ( void *data , HYPRE_Real filter );
HYPRE_Int hypre_BoomerAMGSetDropTol ( void *data , HYPRE_Real drop_tol );
HYPRE_Int hypre_BoomerAMGSetMaxNzPerRow ( void *data , HYPRE_Int max_nz_per_row );
HYPRE_Int hypre_BoomerAMGSetEuclidFile ( void *data , char *euclidfile );
HYPRE_Int hypre_BoomerAMGSetEuLevel ( void *data , HYPRE_Int eu_level );
HYPRE_Int hypre_BoomerAMGSetEuSparseA ( void *data , HYPRE_Real eu_sparse_A );
HYPRE_Int hypre_BoomerAMGSetEuBJ ( void *data , HYPRE_Int eu_bj );
HYPRE_Int hypre_BoomerAMGSetChebyOrder ( void *data , HYPRE_Int order );
HYPRE_Int hypre_BoomerAMGSetChebyFraction ( void *data , HYPRE_Real ratio );
HYPRE_Int hypre_BoomerAMGSetChebyEigEst ( void *data , HYPRE_Int eig_est );
HYPRE_Int hypre_BoomerAMGSetChebyVariant ( void *data , HYPRE_Int variant );
HYPRE_Int hypre_BoomerAMGSetChebyScale ( void *data , HYPRE_Int scale );
HYPRE_Int hypre_BoomerAMGSetInterpVectors ( void *solver , HYPRE_Int num_vectors , hypre_ParVector **interp_vectors );
HYPRE_Int hypre_BoomerAMGSetInterpVecVariant ( void *solver , HYPRE_Int var );
HYPRE_Int hypre_BoomerAMGSetInterpVecQMax ( void *data , HYPRE_Int q_max );
HYPRE_Int hypre_BoomerAMGSetInterpVecAbsQTrunc ( void *data , HYPRE_Real q_trunc );
HYPRE_Int hypre_BoomerAMGSetSmoothInterpVectors ( void *solver , HYPRE_Int smooth_interp_vectors );
HYPRE_Int hypre_BoomerAMGSetInterpRefine ( void *data , HYPRE_Int num_refine );
HYPRE_Int hypre_BoomerAMGSetInterpVecFirstLevel ( void *data , HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetAdditive ( void *data , HYPRE_Int additive );
HYPRE_Int hypre_BoomerAMGGetAdditive ( void *data , HYPRE_Int *additive );
HYPRE_Int hypre_BoomerAMGSetMultAdditive ( void *data , HYPRE_Int mult_additive );
HYPRE_Int hypre_BoomerAMGGetMultAdditive ( void *data , HYPRE_Int *mult_additive );
HYPRE_Int hypre_BoomerAMGSetSimple ( void *data , HYPRE_Int simple );
HYPRE_Int hypre_BoomerAMGGetSimple ( void *data , HYPRE_Int *simple );
HYPRE_Int hypre_BoomerAMGSetAddLastLvl ( void *data , HYPRE_Int add_last_lvl );
HYPRE_Int hypre_BoomerAMGSetNonGalerkinTol ( void *data , HYPRE_Real nongalerkin_tol );
HYPRE_Int hypre_BoomerAMGSetLevelNonGalerkinTol ( void *data , HYPRE_Real nongalerkin_tol , HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetNonGalerkTol ( void *data , HYPRE_Int nongalerk_num_tol , HYPRE_Real *nongalerk_tol );
HYPRE_Int hypre_BoomerAMGSetRAP2 ( void *data , HYPRE_Int rap2 );
HYPRE_Int hypre_BoomerAMGSetModuleRAP2 ( void *data , HYPRE_Int mod_rap2 );
HYPRE_Int hypre_BoomerAMGSetKeepTranspose ( void *data , HYPRE_Int keepTranspose );
#ifdef HYPRE_USING_DSUPERLU
HYPRE_Int hypre_BoomerAMGSetDSLUThreshold ( void *data , HYPRE_Int slu_threshold );
#endif
HYPRE_Int hypre_BoomerAMGSetCpointsToKeep(void *data, HYPRE_Int cpt_coarse_level, HYPRE_Int  num_cpt_coarse, HYPRE_Int *cpt_coarse_index);

/* par_amg_setup.c */
HYPRE_Int hypre_BoomerAMGSetup ( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );

/* par_amg_solve.c */
HYPRE_Int hypre_BoomerAMGSolve ( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );

/* par_amg_solveT.c */
HYPRE_Int hypre_BoomerAMGSolveT ( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGCycleT ( void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array );
HYPRE_Int hypre_BoomerAMGRelaxT ( hypre_ParCSRMatrix *A , hypre_ParVector *f , HYPRE_Int *cf_marker , HYPRE_Int relax_type , HYPRE_Int relax_points , HYPRE_Real relax_weight , hypre_ParVector *u , hypre_ParVector *Vtemp );

/* par_cgc_coarsen.c */
HYPRE_Int hypre_BoomerAMGCoarsenCGCb ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , HYPRE_Int measure_type , HYPRE_Int coarsen_type , HYPRE_Int cgc_its , HYPRE_Int debug_flag , HYPRE_Int **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenCGC ( hypre_ParCSRMatrix *S , HYPRE_Int numberofgrids , HYPRE_Int coarsen_type , HYPRE_Int *CF_marker );
HYPRE_Int hypre_AmgCGCPrepare ( hypre_ParCSRMatrix *S , HYPRE_Int nlocal , HYPRE_Int *CF_marker , HYPRE_Int **CF_marker_offd , HYPRE_Int coarsen_type , HYPRE_Int **vrange );
//HYPRE_Int hypre_AmgCGCPrepare ( hypre_ParCSRMatrix *S , HYPRE_Int nlocal , HYPRE_Int *CF_marker , HYPRE_BigInt **CF_marker_offd , HYPRE_Int coarsen_type , HYPRE_BigInt **vrange );
HYPRE_Int hypre_AmgCGCGraphAssemble ( hypre_ParCSRMatrix *S , HYPRE_Int *vertexrange , HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd , HYPRE_Int coarsen_type , HYPRE_IJMatrix *ijG );
HYPRE_Int hypre_AmgCGCChoose ( hypre_CSRMatrix *G , HYPRE_Int *vertexrange , HYPRE_Int mpisize , HYPRE_Int **coarse );
HYPRE_Int hypre_AmgCGCBoundaryFix ( hypre_ParCSRMatrix *S , HYPRE_Int *CF_marker , HYPRE_Int *CF_marker_offd );

/* par_cg_relax_wt.c */
HYPRE_Int hypre_BoomerAMGCGRelaxWt ( void *amg_vdata , HYPRE_Int level , HYPRE_Int num_cg_sweeps , HYPRE_Real *rlx_wt_ptr );
HYPRE_Int hypre_Bisection ( HYPRE_Int n , HYPRE_Real *diag , HYPRE_Real *offd , HYPRE_Real y , HYPRE_Real z , HYPRE_Real tol , HYPRE_Int k , HYPRE_Real *ev_ptr );

/* par_cheby.c */
HYPRE_Int hypre_ParCSRRelax_Cheby_Setup ( hypre_ParCSRMatrix *A , HYPRE_Real max_eig , HYPRE_Real min_eig , HYPRE_Real fraction , HYPRE_Int order , HYPRE_Int scale , HYPRE_Int variant , HYPRE_Real **coefs_ptr , HYPRE_Real **ds_ptr );
HYPRE_Int hypre_ParCSRRelax_Cheby_Solve ( hypre_ParCSRMatrix *A , hypre_ParVector *f , HYPRE_Real *ds_data , HYPRE_Real *coefs , HYPRE_Int order , HYPRE_Int scale , HYPRE_Int variant , hypre_ParVector *u , hypre_ParVector *v , hypre_ParVector *r );

/* par_coarsen.c */
HYPRE_Int hypre_BoomerAMGCoarsen ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , HYPRE_Int CF_init , HYPRE_Int debug_flag , HYPRE_Int **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenRuge ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , HYPRE_Int measure_type , HYPRE_Int coarsen_type , HYPRE_Int debug_flag , HYPRE_Int **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenFalgout ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , HYPRE_Int measure_type , HYPRE_Int debug_flag , HYPRE_Int **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenHMIS ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , HYPRE_Int measure_type , HYPRE_Int debug_flag , HYPRE_Int **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenPMIS ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , HYPRE_Int CF_init , HYPRE_Int debug_flag , HYPRE_Int **CF_marker_ptr );

HYPRE_Int hypre_BoomerAMGCoarsenPMISDevice( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, HYPRE_Int CF_init, HYPRE_Int debug_flag, HYPRE_Int **CF_marker_ptr );

/* par_coarsen_device.c */
HYPRE_Int hypre_GetGlobalMeasureDevice( hypre_ParCSRMatrix *S, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int CF_init, HYPRE_Int aug_rand, HYPRE_Real *measure_diag, HYPRE_Real *measure_offd, HYPRE_Real *real_send_buf );

/* par_coarse_parms.c */
HYPRE_Int hypre_BoomerAMGCoarseParms ( MPI_Comm comm , HYPRE_Int local_num_variables , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int *CF_marker , HYPRE_Int **coarse_dof_func_ptr , HYPRE_BigInt **coarse_pnts_global_ptr );

/* par_coordinates.c */
float *GenerateCoordinates ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_BigInt nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Int coorddim );

/* par_cr.c */
HYPRE_Int hypre_BoomerAMGCoarsenCR1 ( hypre_ParCSRMatrix *A , HYPRE_Int **CF_marker_ptr , HYPRE_BigInt *coarse_size_ptr , HYPRE_Int num_CR_relax_steps , HYPRE_Int IS_type , HYPRE_Int CRaddCpoints );
HYPRE_Int hypre_cr ( HYPRE_Int *A_i , HYPRE_Int *A_j , HYPRE_Real *A_data , HYPRE_Int n , HYPRE_Int *cf , HYPRE_Int rlx , HYPRE_Real omega , HYPRE_Real tg , HYPRE_Int mu );
HYPRE_Int hypre_GraphAdd ( Link *list , HYPRE_Int *head , HYPRE_Int *tail , HYPRE_Int index , HYPRE_Int istack );
HYPRE_Int hypre_GraphRemove ( Link *list , HYPRE_Int *head , HYPRE_Int *tail , HYPRE_Int index );
HYPRE_Int hypre_IndepSetGreedy ( HYPRE_Int *A_i , HYPRE_Int *A_j , HYPRE_Int n , HYPRE_Int *cf );
HYPRE_Int hypre_IndepSetGreedyS ( HYPRE_Int *A_i , HYPRE_Int *A_j , HYPRE_Int n , HYPRE_Int *cf );
HYPRE_Int hypre_fptjaccr ( HYPRE_Int *cf , HYPRE_Int *A_i , HYPRE_Int *A_j , HYPRE_Real *A_data , HYPRE_Int n , HYPRE_Real *e0 , HYPRE_Real omega , HYPRE_Real *e1 );
HYPRE_Int hypre_fptgscr ( HYPRE_Int *cf , HYPRE_Int *A_i , HYPRE_Int *A_j , HYPRE_Real *A_data , HYPRE_Int n , HYPRE_Real *e0 , HYPRE_Real *e1 );
HYPRE_Int hypre_formu ( HYPRE_Int *cf , HYPRE_Int n , HYPRE_Real *e1 , HYPRE_Int *A_i , HYPRE_Real rho );
HYPRE_Int hypre_BoomerAMGIndepRS ( hypre_ParCSRMatrix *S , HYPRE_Int measure_type , HYPRE_Int debug_flag , HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepRSa ( hypre_ParCSRMatrix *S , HYPRE_Int measure_type , HYPRE_Int debug_flag , HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepHMIS ( hypre_ParCSRMatrix *S , HYPRE_Int measure_type , HYPRE_Int debug_flag , HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepHMISa ( hypre_ParCSRMatrix *S , HYPRE_Int measure_type , HYPRE_Int debug_flag , HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepPMIS ( hypre_ParCSRMatrix *S , HYPRE_Int CF_init , HYPRE_Int debug_flag , HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepPMISa ( hypre_ParCSRMatrix *S , HYPRE_Int CF_init , HYPRE_Int debug_flag , HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGCoarsenCR ( hypre_ParCSRMatrix *A , HYPRE_Int **CF_marker_ptr , HYPRE_BigInt *coarse_size_ptr , HYPRE_Int num_CR_relax_steps , HYPRE_Int IS_type , HYPRE_Int num_functions , HYPRE_Int rlx_type , HYPRE_Real relax_weight , HYPRE_Real omega , HYPRE_Real theta , HYPRE_Solver smoother , hypre_ParCSRMatrix *AN , HYPRE_Int useCG , hypre_ParCSRMatrix *S );

/* par_cycle.c */
HYPRE_Int hypre_BoomerAMGCycle ( void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array );

/* par_difconv.c */
HYPRE_ParCSRMatrix GenerateDifConv ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_BigInt nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Real *value );

/* par_gsmg.c */
HYPRE_Int hypre_ParCSRMatrixFillSmooth ( HYPRE_Int nsamples , HYPRE_Real *samples , hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , HYPRE_Int num_functions , HYPRE_Int *dof_func );
HYPRE_Real hypre_ParCSRMatrixChooseThresh ( hypre_ParCSRMatrix *S );
HYPRE_Int hypre_ParCSRMatrixThreshold ( hypre_ParCSRMatrix *A , HYPRE_Real thresh );
HYPRE_Int hypre_BoomerAMGCreateSmoothVecs ( void *data , hypre_ParCSRMatrix *A , HYPRE_Int num_sweeps , HYPRE_Int level , HYPRE_Real **SmoothVecs_p );
HYPRE_Int hypre_BoomerAMGCreateSmoothDirs ( void *data , hypre_ParCSRMatrix *A , HYPRE_Real *SmoothVecs , HYPRE_Real thresh , HYPRE_Int num_functions , HYPRE_Int *dof_func , hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGNormalizeVecs ( HYPRE_Int n , HYPRE_Int num , HYPRE_Real *V );
HYPRE_Int hypre_BoomerAMGFitVectors ( HYPRE_Int ip , HYPRE_Int n , HYPRE_Int num , const HYPRE_Real *V , HYPRE_Int nc , const HYPRE_Int *ind , HYPRE_Real *val );
HYPRE_Int hypre_BoomerAMGBuildInterpLS ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int num_smooth , HYPRE_Real *SmoothVecs , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpGSMG ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , hypre_ParCSRMatrix **P_ptr );

/* par_indepset.c */
HYPRE_Int hypre_BoomerAMGIndepSetInit ( hypre_ParCSRMatrix *S , HYPRE_Real *measure_array , HYPRE_Int seq_rand );
HYPRE_Int hypre_BoomerAMGIndepSet ( hypre_ParCSRMatrix *S , HYPRE_Real *measure_array , HYPRE_Int *graph_array , HYPRE_Int graph_array_size , HYPRE_Int *graph_array_offd , HYPRE_Int graph_array_offd_size , HYPRE_Int *IS_marker , HYPRE_Int *IS_marker_offd );

HYPRE_Int hypre_BoomerAMGIndepSetInitDevice( hypre_ParCSRMatrix *S, HYPRE_Real *measure_array, HYPRE_Int aug_rand);

HYPRE_Int hypre_BoomerAMGIndepSetDevice( hypre_ParCSRMatrix *S, HYPRE_Real *measure_diag, HYPRE_Real *measure_offd, HYPRE_Int graph_diag_size, HYPRE_Int *graph_diag, HYPRE_Int *IS_marker_diag, HYPRE_Int *IS_marker_offd, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int *int_send_buf );

/* par_interp.c */
HYPRE_Int hypre_BoomerAMGBuildInterp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpHE ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildDirInterp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildDirInterpDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );

HYPRE_Int hypre_BoomerAMGInterpTruncation ( hypre_ParCSRMatrix *P, HYPRE_Real trunc_factor, HYPRE_Int max_elmts );
HYPRE_Int hypre_BoomerAMGInterpTruncationDevice( hypre_ParCSRMatrix *P, HYPRE_Real trunc_factor, HYPRE_Int max_elmts );

HYPRE_Int hypre_BoomerAMGBuildInterpModUnk ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGTruncandBuild ( hypre_ParCSRMatrix *P , HYPRE_Real trunc_factor , HYPRE_Int max_elmts );
hypre_ParCSRMatrix *hypre_CreateC ( hypre_ParCSRMatrix *A , HYPRE_Real w );

HYPRE_Int hypre_BoomerAMGBuildInterpOnePnt( hypre_ParCSRMatrix  *A, HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, HYPRE_Int *col_offd_S_to_A, hypre_ParCSRMatrix **P_ptr);

/* par_jacobi_interp.c */
void hypre_BoomerAMGJacobiInterp ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **P , hypre_ParCSRMatrix *S , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int *CF_marker , HYPRE_Int level , HYPRE_Real truncation_threshold , HYPRE_Real truncation_threshold_minus );
void hypre_BoomerAMGJacobiInterp_1 ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **P , hypre_ParCSRMatrix *S , HYPRE_Int *CF_marker , HYPRE_Int level , HYPRE_Real truncation_threshold , HYPRE_Real truncation_threshold_minus , HYPRE_Int *dof_func , HYPRE_Int *dof_func_offd , HYPRE_Real weight_AF );
void hypre_BoomerAMGTruncateInterp ( hypre_ParCSRMatrix *P , HYPRE_Real eps , HYPRE_Real dlt , HYPRE_Int *CF_marker );
HYPRE_Int hypre_ParCSRMatrix_dof_func_offd ( hypre_ParCSRMatrix *A , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int **dof_func_offd );

/* par_laplace_27pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian27pt ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_BigInt nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Real *value );
HYPRE_Int hypre_map3 ( HYPRE_BigInt ix , HYPRE_BigInt iy , HYPRE_BigInt iz , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_BigInt *nx_part , HYPRE_BigInt *ny_part , HYPRE_BigInt *nz_part );

/* par_laplace_9pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian9pt ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int p , HYPRE_Int q , HYPRE_Real *value );
HYPRE_BigInt hypre_map2 ( HYPRE_BigInt ix , HYPRE_BigInt iy , HYPRE_Int p , HYPRE_Int q , HYPRE_BigInt nx , HYPRE_BigInt *nx_part , HYPRE_BigInt *ny_part );

/* par_laplace.c */
HYPRE_ParCSRMatrix GenerateLaplacian ( MPI_Comm comm , HYPRE_BigInt ix , HYPRE_BigInt ny , HYPRE_BigInt nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Real *value );
HYPRE_BigInt hypre_map ( HYPRE_BigInt ix , HYPRE_BigInt iy , HYPRE_BigInt iz , HYPRE_Int p, HYPRE_Int q, HYPRE_Int r, HYPRE_BigInt nx , HYPRE_BigInt ny, HYPRE_BigInt *nx_part, HYPRE_BigInt *ny_part, HYPRE_BigInt *nz_part );
HYPRE_ParCSRMatrix GenerateSysLaplacian ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_BigInt nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Int num_fun , HYPRE_Real *mtrx , HYPRE_Real *value );
HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_BigInt nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Int num_fun , HYPRE_Real *mtrx , HYPRE_Real *value );

/* par_lr_interp.c */
HYPRE_Int hypre_BoomerAMGBuildStdInterp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int sep_weight , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPIInterp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPICCInterp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildFFInterp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildFF1Interp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtInterp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );

/* par_multi_interp.c */
HYPRE_Int hypre_BoomerAMGBuildMultipass ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int P_max_elmts , HYPRE_Int weight_option , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );

/* par_nodal_systems.c */
HYPRE_Int hypre_BoomerAMGCreateNodalA ( hypre_ParCSRMatrix *A , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int option , HYPRE_Int diag_option , hypre_ParCSRMatrix **AN_ptr );
HYPRE_Int hypre_BoomerAMGCreateScalarCFS ( hypre_ParCSRMatrix *SN , HYPRE_Int *CFN_marker , HYPRE_Int *col_offd_SN_to_AN , HYPRE_Int num_functions , HYPRE_Int nodal , HYPRE_Int data , HYPRE_Int **dof_func_ptr , HYPRE_Int **CF_marker_ptr , HYPRE_Int **col_offd_S_to_A_ptr , hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateScalarCF ( HYPRE_Int *CFN_marker , HYPRE_Int num_functions , HYPRE_Int num_nodes , HYPRE_Int **dof_func_ptr , HYPRE_Int **CF_marker_ptr );

/* par_nongalerkin.c */
HYPRE_Int hypre_GrabSubArray ( HYPRE_Int *indices , HYPRE_Int start , HYPRE_Int end , HYPRE_BigInt *array , HYPRE_BigInt *output );
HYPRE_Int hypre_IntersectTwoArrays ( HYPRE_Int *x , HYPRE_Real *x_data , HYPRE_Int x_length , HYPRE_Int *y , HYPRE_Int y_length , HYPRE_Int *z , HYPRE_Real *output_x_data , HYPRE_Int *intersect_length );
HYPRE_Int hypre_IntersectTwoBigArrays ( HYPRE_BigInt *x , HYPRE_Real *x_data , HYPRE_Int x_length , HYPRE_BigInt *y , HYPRE_Int y_length , HYPRE_BigInt *z , HYPRE_Real *output_x_data , HYPRE_Int *intersect_length );
HYPRE_Int hypre_SortedCopyParCSRData ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B );
HYPRE_Int hypre_BoomerAMG_MyCreateS ( hypre_ParCSRMatrix *A , HYPRE_Real strength_threshold , HYPRE_Real max_row_sum , HYPRE_Int num_functions , HYPRE_Int *dof_func , hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSFromCFMarker(hypre_ParCSRMatrix    *A, HYPRE_Real strength_threshold, HYPRE_Real max_row_sum, HYPRE_Int *CF_marker, HYPRE_Int SMRK, hypre_ParCSRMatrix    **S_ptr);
HYPRE_Int hypre_NonGalerkinIJBufferInit ( HYPRE_Int *ijbuf_cnt , HYPRE_Int *ijbuf_rowcounter , HYPRE_Int *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBigBufferInit ( HYPRE_Int *ijbuf_cnt , HYPRE_Int *ijbuf_rowcounter , HYPRE_BigInt *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferNewRow ( HYPRE_BigInt *ijbuf_rownums , HYPRE_Int *ijbuf_numcols , HYPRE_Int *ijbuf_rowcounter , HYPRE_BigInt new_row );
HYPRE_Int hypre_NonGalerkinIJBufferCompressRow ( HYPRE_Int *ijbuf_cnt , HYPRE_Int ijbuf_rowcounter , HYPRE_Real *ijbuf_data , HYPRE_BigInt *ijbuf_cols , HYPRE_BigInt *ijbuf_rownums , HYPRE_Int *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferCompress ( HYPRE_Int ijbuf_size , HYPRE_Int *ijbuf_cnt , HYPRE_Int *ijbuf_rowcounter , HYPRE_Real **ijbuf_data , HYPRE_BigInt **ijbuf_cols , HYPRE_BigInt **ijbuf_rownums , HYPRE_Int **ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferWrite ( HYPRE_IJMatrix B , HYPRE_Int *ijbuf_cnt , HYPRE_Int ijbuf_size , HYPRE_Int *ijbuf_rowcounter , HYPRE_Real **ijbuf_data , HYPRE_BigInt **ijbuf_cols , HYPRE_BigInt **ijbuf_rownums , HYPRE_Int **ijbuf_numcols , HYPRE_BigInt row_to_write , HYPRE_BigInt col_to_write , HYPRE_Real val_to_write );
HYPRE_Int hypre_NonGalerkinIJBufferEmpty ( HYPRE_IJMatrix B , HYPRE_Int ijbuf_size , HYPRE_Int *ijbuf_cnt , HYPRE_Int ijbuf_rowcounter , HYPRE_Real **ijbuf_data , HYPRE_BigInt **ijbuf_cols , HYPRE_BigInt **ijbuf_rownums , HYPRE_Int **ijbuf_numcols );
hypre_ParCSRMatrix * hypre_NonGalerkinSparsityPattern(hypre_ParCSRMatrix *R_IAP, hypre_ParCSRMatrix *RAP, HYPRE_Int * CF_marker, HYPRE_Real droptol, HYPRE_Int sym_collapse, HYPRE_Int collapse_beta );
HYPRE_Int hypre_BoomerAMGBuildNonGalerkinCoarseOperator( hypre_ParCSRMatrix **RAP_ptr, hypre_ParCSRMatrix *AP, HYPRE_Real strong_threshold, HYPRE_Real max_row_sum, HYPRE_Int num_functions, HYPRE_Int * dof_func_value, HYPRE_Real S_commpkg_switch, HYPRE_Int * CF_marker, HYPRE_Real droptol, HYPRE_Int sym_collapse, HYPRE_Real lump_percent, HYPRE_Int collapse_beta );

/* par_rap.c */
HYPRE_Int hypre_BoomerAMGBuildCoarseOperator ( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , hypre_ParCSRMatrix **RAP_ptr );
HYPRE_Int hypre_BoomerAMGBuildCoarseOperatorKT ( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , HYPRE_Int keepTranspose, hypre_ParCSRMatrix **RAP_ptr );

/* par_rap_communication.c */
HYPRE_Int hypre_GetCommPkgRTFromCommPkgA ( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , HYPRE_Int *fine_to_coarse, HYPRE_Int *tmp_map_offd );
HYPRE_Int hypre_GenerateSendMapAndCommPkg ( MPI_Comm comm , HYPRE_Int num_sends , HYPRE_Int num_recvs , HYPRE_Int *recv_procs , HYPRE_Int *send_procs , HYPRE_Int *recv_vec_starts , hypre_ParCSRMatrix *A );

/* par_relax.c */
HYPRE_Int hypre_BoomerAMGRelax ( hypre_ParCSRMatrix *A , hypre_ParVector *f , HYPRE_Int *cf_marker , HYPRE_Int relax_type , HYPRE_Int relax_points , HYPRE_Real relax_weight , HYPRE_Real omega , HYPRE_Real *l1_norms , hypre_ParVector *u , hypre_ParVector *Vtemp , hypre_ParVector *Ztemp );
HYPRE_Int hypre_GaussElimSetup ( hypre_ParAMGData *amg_data , HYPRE_Int level , HYPRE_Int relax_type );
HYPRE_Int hypre_GaussElimSolve ( hypre_ParAMGData *amg_data , HYPRE_Int level , HYPRE_Int relax_type );

/* par_relax_interface.c */
HYPRE_Int hypre_BoomerAMGRelaxIF ( hypre_ParCSRMatrix *A , hypre_ParVector *f , HYPRE_Int *cf_marker , HYPRE_Int relax_type , HYPRE_Int relax_order , HYPRE_Int cycle_type , HYPRE_Real relax_weight , HYPRE_Real omega , HYPRE_Real *l1_norms , hypre_ParVector *u , hypre_ParVector *Vtemp , hypre_ParVector *Ztemp );

/* par_relax_more.c */
HYPRE_Int hypre_ParCSRMaxEigEstimate ( hypre_ParCSRMatrix *A , HYPRE_Int scale , HYPRE_Real *max_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCG ( hypre_ParCSRMatrix *A , HYPRE_Int scale , HYPRE_Int max_iter , HYPRE_Real *max_eig , HYPRE_Real *min_eig );
HYPRE_Int hypre_ParCSRRelax_Cheby ( hypre_ParCSRMatrix *A , hypre_ParVector *f , HYPRE_Real max_eig , HYPRE_Real min_eig , HYPRE_Real fraction , HYPRE_Int order , HYPRE_Int scale , HYPRE_Int variant , hypre_ParVector *u , hypre_ParVector *v , hypre_ParVector *r );
HYPRE_Int hypre_BoomerAMGRelax_FCFJacobi ( hypre_ParCSRMatrix *A , hypre_ParVector *f , HYPRE_Int *cf_marker , HYPRE_Real relax_weight , hypre_ParVector *u , hypre_ParVector *Vtemp );
HYPRE_Int hypre_ParCSRRelax_CG ( HYPRE_Solver solver , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u , HYPRE_Int num_its );
HYPRE_Int hypre_LINPACKcgtql1 ( HYPRE_Int *n , HYPRE_Real *d , HYPRE_Real *e , HYPRE_Int *ierr );
HYPRE_Real hypre_LINPACKcgpthy ( HYPRE_Real *a , HYPRE_Real *b );
HYPRE_Int hypre_ParCSRRelax_L1_Jacobi ( hypre_ParCSRMatrix *A , hypre_ParVector *f , HYPRE_Int *cf_marker , HYPRE_Int relax_points , HYPRE_Real relax_weight , HYPRE_Real *l1_norms , hypre_ParVector *u , hypre_ParVector *Vtemp );

/* par_rotate_7pt.c */
HYPRE_ParCSRMatrix GenerateRotate7pt ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int p , HYPRE_Int q , HYPRE_Real alpha , HYPRE_Real eps );

/* par_scaled_matnorm.c */
HYPRE_Int hypre_ParCSRMatrixScaledNorm ( hypre_ParCSRMatrix *A , HYPRE_Real *scnorm );

/* par_schwarz.c */
void *hypre_SchwarzCreate ( void );
HYPRE_Int hypre_SchwarzDestroy ( void *data );
HYPRE_Int hypre_SchwarzSetup ( void *schwarz_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );
HYPRE_Int hypre_SchwarzSolve ( void *schwarz_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );
HYPRE_Int hypre_SchwarzCFSolve ( void *schwarz_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u , HYPRE_Int *CF_marker , HYPRE_Int rlx_pt );
HYPRE_Int hypre_SchwarzSetVariant ( void *data , HYPRE_Int variant );
HYPRE_Int hypre_SchwarzSetDomainType ( void *data , HYPRE_Int domain_type );
HYPRE_Int hypre_SchwarzSetOverlap ( void *data , HYPRE_Int overlap );
HYPRE_Int hypre_SchwarzSetNumFunctions ( void *data , HYPRE_Int num_functions );
HYPRE_Int hypre_SchwarzSetNonSymm ( void *data , HYPRE_Int value );
HYPRE_Int hypre_SchwarzSetRelaxWeight ( void *data , HYPRE_Real relax_weight );
HYPRE_Int hypre_SchwarzSetDomainStructure ( void *data , hypre_CSRMatrix *domain_structure );
HYPRE_Int hypre_SchwarzSetScale ( void *data , HYPRE_Real *scale );
HYPRE_Int hypre_SchwarzReScale ( void *data , HYPRE_Int size , HYPRE_Real value );
HYPRE_Int hypre_SchwarzSetDofFunc ( void *data , HYPRE_Int *dof_func );

/* par_stats.c */
HYPRE_Int hypre_BoomerAMGSetupStats ( void *amg_vdata , hypre_ParCSRMatrix *A );
HYPRE_Int hypre_BoomerAMGWriteSolverParams ( void *data );

/* par_strength.c */
HYPRE_Int hypre_BoomerAMGCreateS ( hypre_ParCSRMatrix *A , HYPRE_Real strength_threshold , HYPRE_Real max_row_sum , HYPRE_Int num_functions , HYPRE_Int *dof_func , hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSabs ( hypre_ParCSRMatrix *A , HYPRE_Real strength_threshold , HYPRE_Real max_row_sum , HYPRE_Int num_functions , HYPRE_Int *dof_func , hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSCommPkg ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *S , HYPRE_Int **col_offd_S_to_A_ptr );
HYPRE_Int hypre_BoomerAMGCreate2ndS ( hypre_ParCSRMatrix *S , HYPRE_Int *CF_marker , HYPRE_Int num_paths , HYPRE_BigInt *coarse_row_starts , hypre_ParCSRMatrix **C_ptr );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker ( HYPRE_Int *CF_marker , HYPRE_Int num_var , HYPRE_Int *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker2 ( HYPRE_Int *CF_marker , HYPRE_Int num_var , HYPRE_Int *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCreateSDevice(hypre_ParCSRMatrix *A, HYPRE_Real strength_threshold, HYPRE_Real max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr);

/* par_sv_interp.c */
HYPRE_Int hypre_BoomerAMGSmoothInterpVectors ( hypre_ParCSRMatrix *A , HYPRE_Int num_smooth_vecs , hypre_ParVector **smooth_vecs , HYPRE_Int smooth_steps );
HYPRE_Int hypre_BoomerAMGCoarsenInterpVectors ( hypre_ParCSRMatrix *P , HYPRE_Int num_smooth_vecs , hypre_ParVector **smooth_vecs , HYPRE_Int *CF_marker , hypre_ParVector ***new_smooth_vecs , HYPRE_Int expand_level , HYPRE_Int num_functions );
HYPRE_Int hypre_BoomerAMG_GMExpandInterp ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **P , HYPRE_Int num_smooth_vecs , hypre_ParVector **smooth_vecs , HYPRE_Int *nf , HYPRE_Int *dof_func , HYPRE_Int **coarse_dof_func , HYPRE_Int variant , HYPRE_Int level , HYPRE_Real abs_trunc , HYPRE_Real *weights , HYPRE_Int q_max , HYPRE_Int *CF_marker , HYPRE_Int interp_vec_first_level );
HYPRE_Int hypre_BoomerAMGRefineInterp ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **P , HYPRE_BigInt *num_cpts_global , HYPRE_Int *nf , HYPRE_Int *dof_func , HYPRE_Int *CF_marker , HYPRE_Int level );

/* par_sv_interp_ln.c */
HYPRE_Int hypre_BoomerAMG_LNExpandInterp ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **P , HYPRE_BigInt *num_cpts_global , HYPRE_Int *nf , HYPRE_Int *dof_func , HYPRE_Int **coarse_dof_func , HYPRE_Int *CF_marker , HYPRE_Int level , HYPRE_Real *weights , HYPRE_Int num_smooth_vecs , hypre_ParVector **smooth_vecs , HYPRE_Real abs_trunc , HYPRE_Int q_max , HYPRE_Int interp_vec_first_level );

/* par_sv_interp_lsfit.c */
HYPRE_Int hypre_BoomerAMGFitInterpVectors ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **P , HYPRE_Int num_smooth_vecs , hypre_ParVector **smooth_vecs , hypre_ParVector **coarse_smooth_vecs , HYPRE_Real delta , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int *CF_marker , HYPRE_Int max_elmts , HYPRE_Real trunc_factor , HYPRE_Int variant , HYPRE_Int level );

/* partial.c */
HYPRE_Int hypre_BoomerAMGBuildPartialExtPIInterp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_BigInt *num_old_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialStdInterp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_BigInt *num_old_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int sep_weight , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialExtInterp ( hypre_ParCSRMatrix *A , HYPRE_Int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_BigInt *num_old_cpts_global , HYPRE_Int num_functions , HYPRE_Int *dof_func , HYPRE_Int debug_flag , HYPRE_Real trunc_factor , HYPRE_Int max_elmts , HYPRE_Int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );

/* par_vardifconv.c */
HYPRE_ParCSRMatrix GenerateVarDifConv ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_BigInt nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Real eps , HYPRE_ParVector *rhs_ptr );
HYPRE_Real afun ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real bfun ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real cfun ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real dfun ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real efun ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real ffun ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real gfun ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real rfun ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real bndfun ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );

/* par_vardifconv_rs.c */
HYPRE_ParCSRMatrix GenerateRSVarDifConv ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_BigInt nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Real eps , HYPRE_ParVector *rhs_ptr, HYPRE_Int type );
HYPRE_Real afun_rs ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real bfun_rs ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real cfun_rs ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real dfun_rs ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real efun_rs ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real ffun_rs ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real gfun_rs ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real rfun_rs ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );
HYPRE_Real bndfun_rs ( HYPRE_Real xx , HYPRE_Real yy , HYPRE_Real zz );


/* pcg_par.c */
void *hypre_ParKrylovCAlloc ( HYPRE_Int count , HYPRE_Int elt_size );
HYPRE_Int hypre_ParKrylovFree ( void *ptr );
void *hypre_ParKrylovCreateVector ( void *vvector );
void *hypre_ParKrylovCreateVectorArray ( HYPRE_Int n , void *vvector );
HYPRE_Int hypre_ParKrylovDestroyVector ( void *vvector );
void *hypre_ParKrylovMatvecCreate ( void *A , void *x );
HYPRE_Int hypre_ParKrylovMatvec ( void *matvec_data , HYPRE_Complex alpha , void *A , void *x , HYPRE_Complex beta , void *y );
HYPRE_Int hypre_ParKrylovMatvecT ( void *matvec_data , HYPRE_Complex alpha , void *A , void *x , HYPRE_Complex beta , void *y );
HYPRE_Int hypre_ParKrylovMatvecDestroy ( void *matvec_data );
HYPRE_Real hypre_ParKrylovInnerProd ( void *x , void *y );
HYPRE_Int hypre_ParKrylovMassInnerProd ( void *x , void **y, HYPRE_Int k, HYPRE_Int unroll, void *result );
HYPRE_Int hypre_ParKrylovMassDotpTwo ( void *x , void *y , void **z, HYPRE_Int k, HYPRE_Int unroll, void *result_x, void *result_y );
HYPRE_Int hypre_ParKrylovMassAxpy( HYPRE_Complex *alpha, void **x, void *y, HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Int hypre_ParKrylovCopyVector ( void *x , void *y );
HYPRE_Int hypre_ParKrylovClearVector ( void *x );
HYPRE_Int hypre_ParKrylovScaleVector ( HYPRE_Complex alpha , void *x );
HYPRE_Int hypre_ParKrylovAxpy ( HYPRE_Complex alpha , void *x , void *y );
HYPRE_Int hypre_ParKrylovCommInfo ( void *A , HYPRE_Int *my_id , HYPRE_Int *num_procs );
HYPRE_Int hypre_ParKrylovIdentitySetup ( void *vdata , void *A , void *b , void *x );
HYPRE_Int hypre_ParKrylovIdentity ( void *vdata , void *A , void *b , void *x );

/* schwarz.c */
HYPRE_Int hypre_AMGNodalSchwarzSmoother ( hypre_CSRMatrix *A , HYPRE_Int num_functions , HYPRE_Int option , hypre_CSRMatrix **domain_structure_pointer );
HYPRE_Int hypre_ParMPSchwarzSolve ( hypre_ParCSRMatrix *par_A , hypre_CSRMatrix *A_boundary , hypre_ParVector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , HYPRE_Real relax_wt , HYPRE_Real *scale , hypre_ParVector *Vtemp , HYPRE_Int *pivots , HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzSolve ( hypre_ParCSRMatrix *par_A , hypre_Vector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , HYPRE_Real relax_wt , hypre_Vector *aux_vector , HYPRE_Int *pivots , HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzCFSolve ( hypre_ParCSRMatrix *par_A , hypre_Vector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , HYPRE_Real relax_wt , hypre_Vector *aux_vector , HYPRE_Int *CF_marker , HYPRE_Int rlx_pt , HYPRE_Int *pivots , HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzFWSolve ( hypre_ParCSRMatrix *par_A , hypre_Vector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , HYPRE_Real relax_wt , hypre_Vector *aux_vector , HYPRE_Int *pivots , HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzCFFWSolve ( hypre_ParCSRMatrix *par_A , hypre_Vector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , HYPRE_Real relax_wt , hypre_Vector *aux_vector , HYPRE_Int *CF_marker , HYPRE_Int rlx_pt , HYPRE_Int *pivots , HYPRE_Int use_nonsymm );
HYPRE_Int transpose_matrix_create ( HYPRE_Int **i_face_element_pointer , HYPRE_Int **j_face_element_pointer , HYPRE_Int *i_element_face , HYPRE_Int *j_element_face , HYPRE_Int num_elements , HYPRE_Int num_faces );
HYPRE_Int matrix_matrix_product ( HYPRE_Int **i_element_edge_pointer , HYPRE_Int **j_element_edge_pointer , HYPRE_Int *i_element_face , HYPRE_Int *j_element_face , HYPRE_Int *i_face_edge , HYPRE_Int *j_face_edge , HYPRE_Int num_elements , HYPRE_Int num_faces , HYPRE_Int num_edges );
HYPRE_Int hypre_AMGCreateDomainDof ( hypre_CSRMatrix *A , HYPRE_Int domain_type , HYPRE_Int overlap , HYPRE_Int num_functions , HYPRE_Int *dof_func , hypre_CSRMatrix **domain_structure_pointer , HYPRE_Int **piv_pointer , HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AMGeAgglomerate ( HYPRE_Int *i_AE_element , HYPRE_Int *j_AE_element , HYPRE_Int *i_face_face , HYPRE_Int *j_face_face , HYPRE_Int *w_face_face , HYPRE_Int *i_face_element , HYPRE_Int *j_face_element , HYPRE_Int *i_element_face , HYPRE_Int *j_element_face , HYPRE_Int *i_face_to_prefer_weight , HYPRE_Int *i_face_weight , HYPRE_Int num_faces , HYPRE_Int num_elements , HYPRE_Int *num_AEs_pointer );
HYPRE_Int hypre_update_entry ( HYPRE_Int weight , HYPRE_Int *weight_max , HYPRE_Int *previous , HYPRE_Int *next , HYPRE_Int *first , HYPRE_Int *last , HYPRE_Int head , HYPRE_Int tail , HYPRE_Int i );
HYPRE_Int hypre_remove_entry ( HYPRE_Int weight , HYPRE_Int *weight_max , HYPRE_Int *previous , HYPRE_Int *next , HYPRE_Int *first , HYPRE_Int *last , HYPRE_Int head , HYPRE_Int tail , HYPRE_Int i );
HYPRE_Int hypre_move_entry ( HYPRE_Int weight , HYPRE_Int *weight_max , HYPRE_Int *previous , HYPRE_Int *next , HYPRE_Int *first , HYPRE_Int *last , HYPRE_Int head , HYPRE_Int tail , HYPRE_Int i );
HYPRE_Int hypre_matinv ( HYPRE_Real *x , HYPRE_Real *a , HYPRE_Int k );
HYPRE_Int hypre_parCorrRes ( hypre_ParCSRMatrix *A , hypre_ParVector *x , hypre_Vector *rhs , HYPRE_Real **tmp_ptr );
HYPRE_Int hypre_AdSchwarzSolve ( hypre_ParCSRMatrix *par_A , hypre_ParVector *par_rhs , hypre_CSRMatrix *domain_structure , HYPRE_Real *scale , hypre_ParVector *par_x , hypre_ParVector *par_aux , HYPRE_Int *pivots , HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AdSchwarzCFSolve ( hypre_ParCSRMatrix *par_A , hypre_ParVector *par_rhs , hypre_CSRMatrix *domain_structure , HYPRE_Real *scale , hypre_ParVector *par_x , hypre_ParVector *par_aux , HYPRE_Int *CF_marker , HYPRE_Int rlx_pt , HYPRE_Int *pivots , HYPRE_Int use_nonsymm );
HYPRE_Int hypre_GenerateScale ( hypre_CSRMatrix *domain_structure , HYPRE_Int num_variables , HYPRE_Real relaxation_weight , HYPRE_Real **scale_pointer );
HYPRE_Int hypre_ParAdSchwarzSolve ( hypre_ParCSRMatrix *A , hypre_ParVector *F , hypre_CSRMatrix *domain_structure , HYPRE_Real *scale , hypre_ParVector *X , hypre_ParVector *Vtemp , HYPRE_Int *pivots , HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParAMGCreateDomainDof ( hypre_ParCSRMatrix *A , HYPRE_Int domain_type , HYPRE_Int overlap , HYPRE_Int num_functions , HYPRE_Int *dof_func , hypre_CSRMatrix **domain_structure_pointer , HYPRE_Int **piv_pointer , HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParGenerateScale ( hypre_ParCSRMatrix *A , hypre_CSRMatrix *domain_structure , HYPRE_Real relaxation_weight , HYPRE_Real **scale_pointer );
HYPRE_Int hypre_ParGenerateHybridScale ( hypre_ParCSRMatrix *A , hypre_CSRMatrix *domain_structure , hypre_CSRMatrix **A_boundary_pointer , HYPRE_Real **scale_pointer );
/* RL */
HYPRE_Int hypre_BoomerAMGBuildRestrAIR( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Real filter_thresholdR, HYPRE_Int debug_flag, HYPRE_Int *col_offd_S_to_A, hypre_ParCSRMatrix **R_ptr, HYPRE_Int is_triangular, HYPRE_Int gmres_switch);

HYPRE_Int hypre_BoomerAMGBuildRestrDist2AIR( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Real filter_thresholdR, HYPRE_Int debug_flag, HYPRE_Int *col_offd_S_to_A, hypre_ParCSRMatrix **R_ptr, HYPRE_Int AIR1_5, HYPRE_Int is_triangular, HYPRE_Int gmres_switch);

HYPRE_Int hypre_BoomerAMGBuildRestrNeumannAIR( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int NeumannDeg, HYPRE_Real strong_thresholdR, HYPRE_Real filter_thresholdR, HYPRE_Int debug_flag, HYPRE_Int *col_offd_S_to_A, hypre_ParCSRMatrix **R_ptr);

#ifdef HYPRE_USING_DSUPERLU
/* superlu.c */
HYPRE_Int hypre_SLUDistSetup( HYPRE_Solver *solver, hypre_ParCSRMatrix *A, HYPRE_Int print_level);
HYPRE_Int hypre_SLUDistSolve( void* solver, hypre_ParVector *b, hypre_ParVector *x);
HYPRE_Int hypre_SLUDistDestroy( void* solver);
#endif

/* par_mgr.c */
void *hypre_MGRCreate ( void );
HYPRE_Int hypre_MGRDestroy ( void *mgr_vdata );
HYPRE_Int hypre_MGRCycle( void *mgr_vdata, hypre_ParVector **F_array, hypre_ParVector **U_array );
void *hypre_MGRCreateFrelaxVcycleData();
HYPRE_Int hypre_MGRDestroyFrelaxVcycleData( void *mgr_vdata );
HYPRE_Int hypre_MGRSetupFrelaxVcycleData( void *mgr_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u, HYPRE_Int level);
HYPRE_Int hypre_MGRFrelaxVcycle ( void *mgr_vdata, hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRSetCpointsByBlock( void *mgr_vdata, HYPRE_Int  block_size, HYPRE_Int  max_num_levels, HYPRE_Int *block_num_coarse_points, HYPRE_Int  **block_coarse_indexes);
HYPRE_Int hypre_MGRCoarsen(hypre_ParCSRMatrix *S,  hypre_ParCSRMatrix *A,HYPRE_Int final_coarse_size,HYPRE_Int *final_coarse_indexes,HYPRE_Int debug_flag,HYPRE_Int **CF_marker,HYPRE_Int last_level);
HYPRE_Int hypre_MGRSetReservedCoarseNodes(void      *mgr_vdata, HYPRE_Int reserved_coarse_size, HYPRE_Int *reserved_coarse_nodes);
HYPRE_Int hypre_MGRSetMaxGlobalsmoothIters( void *mgr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_MGRSetGlobalsmoothType( void *mgr_vdata, HYPRE_Int iter_type );
HYPRE_Int hypre_MGRSetNonCpointsToFpoints( void      *mgr_vdata, HYPRE_Int nonCptToFptFlag);

//HYPRE_Int hypre_MGRInitCFMarker(HYPRE_Int num_variables, HYPRE_Int *CF_marker, HYPRE_Int initial_coarse_size,HYPRE_Int *initial_coarse_indexes);
//HYPRE_Int hypre_MGRUpdateCoarseIndexes(HYPRE_Int num_variables, HYPRE_Int *CF_marker, HYPRE_Int initial_coarse_size,HYPRE_Int *initial_coarse_indexes);
HYPRE_Int hypre_MGRBuildInterp(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, HYPRE_Int *col_offd_S_to_A, hypre_ParCSRMatrix  **P, HYPRE_Int last_level, HYPRE_Int level, HYPRE_Int numsweeps);
//HYPRE_Int hypre_MGRBuildRestrictionToper(hypre_ParCSRMatrix *AT, HYPRE_Int *CF_marker, hypre_ParCSRMatrix *ST, HYPRE_Int *num_cpts_global,HYPRE_Int num_functions,HYPRE_Int *dof_func,HYPRE_Int debug_flag,HYPRE_Real trunc_factor, HYPRE_Int max_elmts, HYPRE_Int  *col_offd_ST_to_AT,hypre_ParCSRMatrix  **RT,HYPRE_Int last_level,HYPRE_Int level, HYPRE_Int numsweeps);
//HYPRE_Int hypre_BoomerAMGBuildInjectionInterp( hypre_ParCSRMatrix   *A, HYPRE_Int *CF_marker, HYPRE_Int *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int debug_flag,HYPRE_Int init_data,hypre_ParCSRMatrix  **P_ptr);
HYPRE_Int hypre_MGRSetCoarseSolver( void  *mgr_vdata, HYPRE_Int  (*coarse_grid_solver_solve)(void*,void*,void*,void*), HYPRE_Int  (*coarse_grid_solver_setup)(void*,void*,void*,void*), void  *coarse_grid_solver );
HYPRE_Int hypre_MGRSetup( void *mgr_vdata, hypre_ParCSRMatrix *A, hypre_ParVector    *f, hypre_ParVector    *u );
HYPRE_Int hypre_MGRSolve( void *mgr_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector  *u );
HYPRE_Int hypre_block_jacobi_scaling(hypre_ParCSRMatrix *A,hypre_ParCSRMatrix **B_ptr,void               *mgr_vdata,HYPRE_Int             debug_flag);
HYPRE_Int hypre_block_jacobi (hypre_ParCSRMatrix *A,hypre_ParVector    *f,hypre_ParVector    *u,HYPRE_Real         blk_size,HYPRE_Int           n_block,HYPRE_Int           left_size,HYPRE_Real *diaginv,hypre_ParVector    *Vtemp);
HYPRE_Int hypre_blockRelax_setup(hypre_ParCSRMatrix *A,HYPRE_Int blk_size, HYPRE_Int reserved_coarse_size, HYPRE_Real **diaginvptr);
HYPRE_Int hypre_blockRelax(hypre_ParCSRMatrix *A,hypre_ParVector *f,hypre_ParVector *u,HYPRE_Int blk_size,HYPRE_Int reserved_coarse_size,hypre_ParVector *Vtemp,hypre_ParVector *Ztemp);

HYPRE_Int hypre_MGRBuildAff( MPI_Comm comm, HYPRE_Int local_num_variables, HYPRE_Int num_functions,
HYPRE_Int *dof_func, HYPRE_Int *CF_marker, HYPRE_Int **coarse_dof_func_ptr, HYPRE_BigInt **coarse_pnts_global_ptr,
hypre_ParCSRMatrix *A, HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_f_ptr, hypre_ParCSRMatrix **A_ff_ptr );

HYPRE_Int hypre_MGRWriteSolverParams(void *mgr_vdata);
HYPRE_Int hypre_MGRSetAffSolverType( void *systg_vdata, HYPRE_Int *aff_solver_type );
HYPRE_Int hypre_MGRSetCoarseSolverType( void *systg_vdata, HYPRE_Int coarse_solver_type );
HYPRE_Int hypre_MGRSetCoarseSolverIter( void *systg_vdata, HYPRE_Int coarse_solver_iter );
HYPRE_Int hypre_MGRSetFineSolverIter( void *systg_vdata, HYPRE_Int fine_solver_iter );
HYPRE_Int hypre_MGRSetFineSolverMaxLevels( void *systg_vdata, HYPRE_Int fine_solver_max_levels );
HYPRE_Int hypre_MGRSetMaxCoarseLevels( void *mgr_vdata, HYPRE_Int maxlev );
HYPRE_Int hypre_MGRSetBlockSize( void *mgr_vdata, HYPRE_Int bsize );
HYPRE_Int hypre_MGRSetRelaxType( void *mgr_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_MGRSetFRelaxMethod( void *mgr_vdata, HYPRE_Int relax_method);
HYPRE_Int hypre_MGRSetRestrictType( void *mgr_vdata, HYPRE_Int interpType);
HYPRE_Int hypre_MGRSetInterpType( void *mgr_vdata, HYPRE_Int interpType);
HYPRE_Int hypre_MGRSetNumRelaxSweeps( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumInterpSweeps( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumRestrictSweeps( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetPrintLevel( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetLogging( void *mgr_vdata, HYPRE_Int logging );
HYPRE_Int hypre_MGRSetMaxIter( void *mgr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_MGRSetTol( void *mgr_vdata, HYPRE_Real tol );
// Accessor functions
HYPRE_Int hypre_MGRGetNumIterations( void *mgr_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_MGRGetFinalRelativeResidualNorm( void *mgr_vdata, HYPRE_Real *res_norm );


#ifdef __cplusplus
}
#endif

#endif

