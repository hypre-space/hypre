/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParAMG functions
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_BoomerAMGCreate( void )
{
   hypre_ParAMGData  *amg_data;
   hypre_Solver      *base;

   /* setup params */
   HYPRE_Int    max_levels;
   HYPRE_Int    max_coarse_size;
   HYPRE_Int    min_coarse_size;
   HYPRE_Int    coarsen_cut_factor;
   HYPRE_Real   strong_threshold;
   HYPRE_Real   strong_threshold_R;
   HYPRE_Real   filter_threshold_R;
   HYPRE_Int    Sabs;
   HYPRE_Real   max_row_sum;
   HYPRE_Real   trunc_factor;
   HYPRE_Real   agg_trunc_factor;
   HYPRE_Real   agg_P12_trunc_factor;
   HYPRE_Real   jacobi_trunc_threshold;
   HYPRE_Real   CR_rate;
   HYPRE_Real   CR_strong_th;
   HYPRE_Real   A_drop_tol;
   HYPRE_Int    A_drop_type;
   HYPRE_Int    interp_type;
   HYPRE_Int    sep_weight;
   HYPRE_Int    coarsen_type;
   HYPRE_Int    measure_type;
   HYPRE_Int    setup_type;
   HYPRE_Int    P_max_elmts;
   HYPRE_Int    num_functions;
   HYPRE_Int    nodal, nodal_levels, nodal_diag;
   HYPRE_Int    keep_same_sign;
   HYPRE_Int    num_paths;
   HYPRE_Int    agg_num_levels;
   HYPRE_Int    agg_interp_type;
   HYPRE_Int    agg_P_max_elmts;
   HYPRE_Int    agg_P12_max_elmts;
   HYPRE_Int    post_interp_type;
   HYPRE_Int    num_CR_relax_steps;
   HYPRE_Int    IS_type;
   HYPRE_Int    CR_use_CG;
   HYPRE_Int    cgc_its;
   HYPRE_Int    seq_threshold;
   HYPRE_Int    redundant;
   HYPRE_Int    rap2;
   HYPRE_Int    keepT;
   HYPRE_Int    modu_rap;

   /* solve params */
   HYPRE_Int    min_iter;
   HYPRE_Int    max_iter;
   HYPRE_Int    fcycle;
   HYPRE_Int    cycle_type;

   HYPRE_Int    converge_type;
   HYPRE_Real   tol;

   HYPRE_Int    num_sweeps;
   HYPRE_Int    relax_down;
   HYPRE_Int    relax_up;
   HYPRE_Int    relax_coarse;
   HYPRE_Int    relax_order;
   HYPRE_Real   relax_wt;
   HYPRE_Real   outer_wt;
   HYPRE_Real   nongalerkin_tol;
   HYPRE_Int    smooth_type;
   HYPRE_Int    smooth_num_levels;
   HYPRE_Int    smooth_num_sweeps;

   HYPRE_Int    variant, overlap, domain_type, schwarz_use_nonsymm;
   HYPRE_Real   schwarz_rlx_weight;
   HYPRE_Int    level, sym;
   HYPRE_Int    eu_level, eu_bj;
   HYPRE_Int    max_nz_per_row;
   HYPRE_Real   thresh, filter;
   HYPRE_Real   drop_tol;
   HYPRE_Real   eu_sparse_A;
   char    *euclidfile;
   HYPRE_Int    ilu_lfil;
   HYPRE_Int    ilu_type;
   HYPRE_Int    ilu_max_row_nnz;
   HYPRE_Int    ilu_max_iter;
   HYPRE_Real   ilu_droptol;
   HYPRE_Int    ilu_tri_solve;
   HYPRE_Int    ilu_lower_jacobi_iters;
   HYPRE_Int    ilu_upper_jacobi_iters;
   HYPRE_Int    ilu_reordering_type;
   HYPRE_Int    ilu_iter_setup_type;
   HYPRE_Int    ilu_iter_setup_option;
   HYPRE_Int    ilu_iter_setup_max_iter;
   HYPRE_Real   ilu_iter_setup_tolerance;

   HYPRE_Int    fsai_algo_type;
   HYPRE_Int    fsai_local_solve_type;
   HYPRE_Int    fsai_max_steps;
   HYPRE_Int    fsai_max_step_size;
   HYPRE_Int    fsai_max_nnz_row;
   HYPRE_Int    fsai_num_levels;
   HYPRE_Real   fsai_threshold;
   HYPRE_Int    fsai_eig_maxiter;
   HYPRE_Real   fsai_kap_tolerance;

   HYPRE_Int cheby_order;
   HYPRE_Int cheby_eig_est;
   HYPRE_Int cheby_variant;
   HYPRE_Int cheby_scale;
   HYPRE_Real cheby_eig_ratio;

   HYPRE_Int block_mode;

   HYPRE_Int    additive;
   HYPRE_Int    mult_additive;
   HYPRE_Int    simple;
   HYPRE_Int    add_last_lvl;
   HYPRE_Real   add_trunc_factor;
   HYPRE_Int    add_P_max_elmts;
   HYPRE_Int    add_rlx_type;
   HYPRE_Real   add_rlx_wt;

   /* log info */
   HYPRE_Int    num_iterations;
   HYPRE_Int    cum_num_iterations;
   HYPRE_Real   cum_nnz_AP;

   /* output params */
   HYPRE_Int    print_level;
   HYPRE_Int    logging;
   /* HYPRE_Int      cycle_op_count; */
   char     log_file_name[256];
   HYPRE_Int    debug_flag;

   char     plot_file_name[251] = {0};

   HYPRE_MemoryLocation memory_location = hypre_HandleMemoryLocation(hypre_handle());

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/

   /* setup params */
   max_levels = 25;
   max_coarse_size = 9;
   min_coarse_size = 0;
   seq_threshold = 0;
   redundant = 0;
   coarsen_cut_factor = 0;
   strong_threshold = 0.25;
   strong_threshold_R = 0.25;
   filter_threshold_R = 0.0;
   Sabs = 0;
   max_row_sum = 0.9;
   trunc_factor = 0.0;
   agg_trunc_factor = 0.0;
   agg_P12_trunc_factor = 0.0;
   jacobi_trunc_threshold = 0.01;
   sep_weight = 0;
   coarsen_type = 10;
   interp_type = 6;
   measure_type = 0;
   setup_type = 1;
   P_max_elmts = 4;
   agg_P_max_elmts = 0;
   agg_P12_max_elmts = 0;
   num_functions = 1;
   nodal = 0;
   nodal_levels = max_levels;
   nodal_diag = 0;
   keep_same_sign = 0;
   num_paths = 1;
   agg_num_levels = 0;
   post_interp_type = 0;
   agg_interp_type = 4;
   num_CR_relax_steps = 2;
   CR_rate = 0.7;
   CR_strong_th = 0;
   A_drop_tol = 0.0;
   A_drop_type = -1;
   IS_type = 1;
   CR_use_CG = 0;
   cgc_its = 1;

   variant = 0;
   overlap = 1;
   domain_type = 2;
   schwarz_rlx_weight = 1.0;
   smooth_num_sweeps = 1;
   smooth_num_levels = 0;
   smooth_type = 6;
   schwarz_use_nonsymm = 0;

   level = 1;
   sym = 0;
   thresh = 0.1;
   filter = 0.05;
   drop_tol = 0.0001;
   max_nz_per_row = 20;
   euclidfile = NULL;
   eu_level = 0;
   eu_sparse_A = 0.0;
   eu_bj = 0;
   ilu_lfil = 0;
   ilu_type = 0;
   ilu_max_row_nnz = 20;
   ilu_max_iter = 1;
   ilu_droptol = 0.01;
   ilu_tri_solve = 1;
   ilu_lower_jacobi_iters = 5;
   ilu_upper_jacobi_iters = 5;
   ilu_reordering_type = 1;
   ilu_iter_setup_type = 0;
   ilu_iter_setup_option = 10;
   ilu_iter_setup_max_iter = 20;
   ilu_iter_setup_tolerance = 1.e-3;

   /* FSAI smoother params */
#if defined (HYPRE_USING_CUDA) || defined (HYPRE_USING_HIP)
   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      fsai_algo_type = 3;
   }
   else
#endif
   {
      fsai_algo_type = hypre_NumThreads() > 4 ? 2 : 1;
   }
   fsai_local_solve_type = 0;
   fsai_max_steps = 4;
   fsai_max_step_size = 2;
   fsai_max_nnz_row = 8;
   fsai_num_levels = 1;
   fsai_threshold = 0.01;
   fsai_eig_maxiter = 5;
   fsai_kap_tolerance = 0.001;

   /* solve params */
   min_iter  = 0;
   max_iter  = 20;
   fcycle = 0;
   cycle_type = 1;
   converge_type = 0;
   tol = 1.0e-6;

   num_sweeps = 1;
   relax_down = 13;
   relax_up = 14;
   relax_coarse = 9;
   relax_order = 0;
   relax_wt = 1.0;
   outer_wt = 1.0;

   cheby_order = 2;
   cheby_variant = 0;
   cheby_scale = 1;
   cheby_eig_est = 10;
   cheby_eig_ratio = .3;

   block_mode = 0;

   additive = -1;
   mult_additive = -1;
   simple = -1;
   add_last_lvl = -1;
   add_trunc_factor = 0.0;
   add_P_max_elmts = 0;
   add_rlx_type = 18;
   add_rlx_wt = 1.0;

   /* log info */
   num_iterations = 0;
   cum_num_iterations = 0;
   cum_nnz_AP = -1.0;

   /* output params */
   print_level = 0;
   logging = 0;
   hypre_sprintf(log_file_name, "%s", "amg.out.log");
   /* cycle_op_count = 0; */
   debug_flag = 0;

   nongalerkin_tol = 0.0;

   rap2 = 0;
   keepT = 0;
   modu_rap = 0;

   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      keepT           =  1;
      modu_rap        =  1;
      coarsen_type    =  8;
      relax_down      = 18;
      relax_up        = 18;
      agg_interp_type =  7;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*-----------------------------------------------------------------------
    * Create the hypre_ParAMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = hypre_CTAlloc(hypre_ParAMGData, 1, HYPRE_MEMORY_HOST);
   base     = (hypre_Solver*) amg_data;

   /* Set base solver function pointers */
   hypre_SolverSetup(base)   = (HYPRE_PtrToSolverFcn)  HYPRE_BoomerAMGSetup;
   hypre_SolverSolve(base)   = (HYPRE_PtrToSolverFcn)  HYPRE_BoomerAMGSolve;
   hypre_SolverDestroy(base) = (HYPRE_PtrToDestroyFcn) HYPRE_BoomerAMGDestroy;

   /* memory location will be reset at the setup */
   hypre_ParAMGDataMemoryLocation(amg_data) = memory_location;

   hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = -1;
   hypre_ParAMGDataPartialCycleControl(amg_data) = -1;
   hypre_ParAMGDataMaxLevels(amg_data) =  max_levels;
   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = 9;
   hypre_ParAMGDataUserRelaxType(amg_data) = -1;
   hypre_ParAMGDataUserNumSweeps(amg_data) = -1;
   hypre_ParAMGDataUserRelaxWeight(amg_data) = relax_wt;
   hypre_ParAMGDataOuterWt(amg_data) = outer_wt;
   hypre_BoomerAMGSetMaxCoarseSize(amg_data, max_coarse_size);
   hypre_BoomerAMGSetMinCoarseSize(amg_data, min_coarse_size);
   hypre_BoomerAMGSetCoarsenCutFactor(amg_data, coarsen_cut_factor);
   hypre_BoomerAMGSetStrongThreshold(amg_data, strong_threshold);
   hypre_BoomerAMGSetStrongThresholdR(amg_data, strong_threshold_R);
   hypre_BoomerAMGSetFilterThresholdR(amg_data, filter_threshold_R);
   hypre_BoomerAMGSetSabs(amg_data, Sabs);
   hypre_BoomerAMGSetMaxRowSum(amg_data, max_row_sum);
   hypre_BoomerAMGSetTruncFactor(amg_data, trunc_factor);
   hypre_BoomerAMGSetAggTruncFactor(amg_data, agg_trunc_factor);
   hypre_BoomerAMGSetAggP12TruncFactor(amg_data, agg_P12_trunc_factor);
   hypre_BoomerAMGSetJacobiTruncThreshold(amg_data, jacobi_trunc_threshold);
   hypre_BoomerAMGSetSepWeight(amg_data, sep_weight);
   hypre_BoomerAMGSetMeasureType(amg_data, measure_type);
   hypre_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
   hypre_BoomerAMGSetInterpType(amg_data, interp_type);
   hypre_BoomerAMGSetSetupType(amg_data, setup_type);
   hypre_BoomerAMGSetPMaxElmts(amg_data, P_max_elmts);
   hypre_BoomerAMGSetAggPMaxElmts(amg_data, agg_P_max_elmts);
   hypre_BoomerAMGSetAggP12MaxElmts(amg_data, agg_P12_max_elmts);
   hypre_BoomerAMGSetNumFunctions(amg_data, num_functions);
   hypre_BoomerAMGSetNodal(amg_data, nodal);
   hypre_BoomerAMGSetNodalLevels(amg_data, nodal_levels);
   hypre_BoomerAMGSetNodal(amg_data, nodal_diag);
   hypre_BoomerAMGSetKeepSameSign(amg_data, keep_same_sign);
   hypre_BoomerAMGSetNumPaths(amg_data, num_paths);
   hypre_BoomerAMGSetAggNumLevels(amg_data, agg_num_levels);
   hypre_BoomerAMGSetAggInterpType(amg_data, agg_interp_type);
   hypre_BoomerAMGSetPostInterpType(amg_data, post_interp_type);
   hypre_BoomerAMGSetNumCRRelaxSteps(amg_data, num_CR_relax_steps);
   hypre_BoomerAMGSetCRRate(amg_data, CR_rate);
   hypre_BoomerAMGSetCRStrongTh(amg_data, CR_strong_th);
   hypre_BoomerAMGSetADropTol(amg_data, A_drop_tol);
   hypre_BoomerAMGSetADropType(amg_data, A_drop_type);
   hypre_BoomerAMGSetISType(amg_data, IS_type);
   hypre_BoomerAMGSetCRUseCG(amg_data, CR_use_CG);
   hypre_BoomerAMGSetCGCIts(amg_data, cgc_its);
   hypre_BoomerAMGSetVariant(amg_data, variant);
   hypre_BoomerAMGSetOverlap(amg_data, overlap);
   hypre_BoomerAMGSetSchwarzRlxWeight(amg_data, schwarz_rlx_weight);
   hypre_BoomerAMGSetSchwarzUseNonSymm(amg_data, schwarz_use_nonsymm);
   hypre_BoomerAMGSetDomainType(amg_data, domain_type);
   hypre_BoomerAMGSetSym(amg_data, sym);
   hypre_BoomerAMGSetLevel(amg_data, level);
   hypre_BoomerAMGSetThreshold(amg_data, thresh);
   hypre_BoomerAMGSetFilter(amg_data, filter);
   hypre_BoomerAMGSetDropTol(amg_data, drop_tol);
   hypre_BoomerAMGSetMaxNzPerRow(amg_data, max_nz_per_row);
   hypre_BoomerAMGSetEuclidFile(amg_data, euclidfile);
   hypre_BoomerAMGSetEuLevel(amg_data, eu_level);
   hypre_BoomerAMGSetEuSparseA(amg_data, eu_sparse_A);
   hypre_BoomerAMGSetEuBJ(amg_data, eu_bj);
   hypre_BoomerAMGSetILUType(amg_data, ilu_type);
   hypre_BoomerAMGSetILULevel(amg_data, ilu_lfil);
   hypre_BoomerAMGSetILUMaxRowNnz(amg_data, ilu_max_row_nnz);
   hypre_BoomerAMGSetILUDroptol(amg_data, ilu_droptol);
   hypre_BoomerAMGSetILUTriSolve(amg_data, ilu_tri_solve);
   hypre_BoomerAMGSetILULowerJacobiIters(amg_data, ilu_lower_jacobi_iters);
   hypre_BoomerAMGSetILUUpperJacobiIters(amg_data, ilu_upper_jacobi_iters);
   hypre_BoomerAMGSetILUMaxIter(amg_data, ilu_max_iter);
   hypre_BoomerAMGSetILULocalReordering(amg_data, ilu_reordering_type);
   hypre_BoomerAMGSetILUIterSetupType(amg_data, ilu_iter_setup_type);
   hypre_BoomerAMGSetILUIterSetupOption(amg_data, ilu_iter_setup_option);
   hypre_BoomerAMGSetILUIterSetupMaxIter(amg_data, ilu_iter_setup_max_iter);
   hypre_BoomerAMGSetILUIterSetupTolerance(amg_data, ilu_iter_setup_tolerance);
   hypre_BoomerAMGSetFSAIAlgoType(amg_data, fsai_algo_type);
   hypre_BoomerAMGSetFSAILocalSolveType(amg_data, fsai_local_solve_type);
   hypre_BoomerAMGSetFSAIMaxSteps(amg_data, fsai_max_steps);
   hypre_BoomerAMGSetFSAIMaxStepSize(amg_data, fsai_max_step_size);
   hypre_BoomerAMGSetFSAIMaxNnzRow(amg_data, fsai_max_nnz_row);
   hypre_BoomerAMGSetFSAINumLevels(amg_data, fsai_num_levels);
   hypre_BoomerAMGSetFSAIThreshold(amg_data, fsai_threshold);
   hypre_BoomerAMGSetFSAIEigMaxIters(amg_data, fsai_eig_maxiter);
   hypre_BoomerAMGSetFSAIKapTolerance(amg_data, fsai_kap_tolerance);

   hypre_BoomerAMGSetMinIter(amg_data, min_iter);
   hypre_BoomerAMGSetMaxIter(amg_data, max_iter);
   hypre_BoomerAMGSetCycleType(amg_data, cycle_type);
   hypre_BoomerAMGSetFCycle(amg_data, fcycle);
   hypre_BoomerAMGSetConvergeType(amg_data, converge_type);
   hypre_BoomerAMGSetTol(amg_data, tol);
   hypre_BoomerAMGSetNumSweeps(amg_data, num_sweeps);
   hypre_BoomerAMGSetCycleRelaxType(amg_data, relax_down, 1);
   hypre_BoomerAMGSetCycleRelaxType(amg_data, relax_up, 2);
   hypre_BoomerAMGSetCycleRelaxType(amg_data, relax_coarse, 3);
   hypre_BoomerAMGSetRelaxOrder(amg_data, relax_order);
   hypre_BoomerAMGSetRelaxWt(amg_data, relax_wt);
   hypre_BoomerAMGSetOuterWt(amg_data, outer_wt);
   hypre_BoomerAMGSetSmoothType(amg_data, smooth_type);
   hypre_BoomerAMGSetSmoothNumLevels(amg_data, smooth_num_levels);
   hypre_BoomerAMGSetSmoothNumSweeps(amg_data, smooth_num_sweeps);

   hypre_BoomerAMGSetChebyOrder(amg_data, cheby_order);
   hypre_BoomerAMGSetChebyFraction(amg_data, cheby_eig_ratio);
   hypre_BoomerAMGSetChebyEigEst(amg_data, cheby_eig_est);
   hypre_BoomerAMGSetChebyVariant(amg_data, cheby_variant);
   hypre_BoomerAMGSetChebyScale(amg_data, cheby_scale);

   hypre_BoomerAMGSetNumIterations(amg_data, num_iterations);

   hypre_BoomerAMGSetAdditive(amg_data, additive);
   hypre_BoomerAMGSetMultAdditive(amg_data, mult_additive);
   hypre_BoomerAMGSetSimple(amg_data, simple);
   hypre_BoomerAMGSetMultAddPMaxElmts(amg_data, add_P_max_elmts);
   hypre_BoomerAMGSetMultAddTruncFactor(amg_data, add_trunc_factor);
   hypre_BoomerAMGSetAddRelaxType(amg_data, add_rlx_type);
   hypre_BoomerAMGSetAddRelaxWt(amg_data, add_rlx_wt);
   hypre_ParAMGDataAddLastLvl(amg_data) = add_last_lvl;
   hypre_ParAMGDataLambda(amg_data) = NULL;
   hypre_ParAMGDataXtilde(amg_data) = NULL;
   hypre_ParAMGDataRtilde(amg_data) = NULL;
   hypre_ParAMGDataDinv(amg_data) = NULL;

#ifdef CUMNUMIT
   hypre_ParAMGDataCumNumIterations(amg_data) = cum_num_iterations;
#endif
   hypre_BoomerAMGSetPrintLevel(amg_data, print_level);
   hypre_BoomerAMGSetLogging(amg_data, logging);
   hypre_BoomerAMGSetPrintFileName(amg_data, log_file_name);
   hypre_BoomerAMGSetDebugFlag(amg_data, debug_flag);
   hypre_BoomerAMGSetRestriction(amg_data, 0);
   hypre_BoomerAMGSetIsTriangular(amg_data, 0);
   hypre_BoomerAMGSetGMRESSwitchR(amg_data, 64);

   hypre_BoomerAMGSetGSMG(amg_data, 0);
   hypre_BoomerAMGSetNumSamples(amg_data, 0);

   hypre_ParAMGDataAArray(amg_data) = NULL;
   hypre_ParAMGDataPArray(amg_data) = NULL;
   hypre_ParAMGDataRArray(amg_data) = NULL;
   hypre_ParAMGDataCFMarkerArray(amg_data) = NULL;
   hypre_ParAMGDataVtemp(amg_data)  = NULL;
   hypre_ParAMGDataRtemp(amg_data)  = NULL;
   hypre_ParAMGDataPtemp(amg_data)  = NULL;
   hypre_ParAMGDataZtemp(amg_data)  = NULL;
   hypre_ParAMGDataFArray(amg_data) = NULL;
   hypre_ParAMGDataUArray(amg_data) = NULL;
   hypre_ParAMGDataDofFunc(amg_data) = NULL;
   hypre_ParAMGDataDofFuncArray(amg_data) = NULL;
   hypre_ParAMGDataDofPointArray(amg_data) = NULL;
   hypre_ParAMGDataDofPointArray(amg_data) = NULL;
   hypre_ParAMGDataPointDofMapArray(amg_data) = NULL;
   hypre_ParAMGDataSmoother(amg_data) = NULL;
   hypre_ParAMGDataL1Norms(amg_data) = NULL;

   hypre_ParAMGDataABlockArray(amg_data) = NULL;
   hypre_ParAMGDataPBlockArray(amg_data) = NULL;
   hypre_ParAMGDataRBlockArray(amg_data) = NULL;

   /* this can not be set by the user currently */
   hypre_ParAMGDataBlockMode(amg_data) = block_mode;

   /* Stuff for Chebyshev smoothing */
   hypre_ParAMGDataMaxEigEst(amg_data) = NULL;
   hypre_ParAMGDataMinEigEst(amg_data) = NULL;
   hypre_ParAMGDataChebyDS(amg_data) = NULL;
   hypre_ParAMGDataChebyCoefs(amg_data) = NULL;

   /* BM Oct 22, 2006 */
   hypre_ParAMGDataPlotGrids(amg_data) = 0;
   hypre_BoomerAMGSetPlotFileName (amg_data, plot_file_name);

   /* BM Oct 17, 2006 */
   hypre_ParAMGDataCoordDim(amg_data) = 0;
   hypre_ParAMGDataCoordinates(amg_data) = NULL;

   /* for fitting vectors for interp */
   hypre_BoomerAMGSetInterpVecVariant(amg_data, 0);
   hypre_BoomerAMGSetInterpVectors(amg_data, 0, NULL);
   hypre_ParAMGNumLevelsInterpVectors(amg_data) = max_levels;
   hypre_ParAMGInterpVectorsArray(amg_data) = NULL;
   hypre_ParAMGInterpVecQMax(amg_data) = 0;
   hypre_ParAMGInterpVecAbsQTrunc(amg_data) = 0.0;
   hypre_ParAMGInterpRefine(amg_data) = 0;
   hypre_ParAMGInterpVecFirstLevel(amg_data) = 0;
   hypre_ParAMGNumInterpVectors(amg_data) = 0;
   hypre_ParAMGSmoothInterpVectors(amg_data) = 0;
   hypre_ParAMGDataExpandPWeights(amg_data) = NULL;

   /* for redundant coarse grid solve */
   hypre_ParAMGDataSeqThreshold(amg_data) = seq_threshold;
   hypre_ParAMGDataRedundant(amg_data) = redundant;
   hypre_ParAMGDataCoarseSolver(amg_data) = NULL;
   hypre_ParAMGDataACoarse(amg_data) = NULL;
   hypre_ParAMGDataFCoarse(amg_data) = NULL;
   hypre_ParAMGDataUCoarse(amg_data) = NULL;
   hypre_ParAMGDataNewComm(amg_data) = hypre_MPI_COMM_NULL;

   /* for Gaussian elimination coarse grid solve */
   hypre_ParAMGDataGSSetup(amg_data)          = 0;
   hypre_ParAMGDataGEMemoryLocation(amg_data) = HYPRE_MEMORY_UNDEFINED;
   hypre_ParAMGDataCommInfo(amg_data)         = NULL;
   hypre_ParAMGDataAMat(amg_data)             = NULL;
   hypre_ParAMGDataAWork(amg_data)            = NULL;
   hypre_ParAMGDataAPiv(amg_data)             = NULL;
   hypre_ParAMGDataBVec(amg_data)             = NULL;
   hypre_ParAMGDataUVec(amg_data)             = NULL;

   hypre_ParAMGDataNonGalerkinTol(amg_data) = nongalerkin_tol;
   hypre_ParAMGDataNonGalTolArray(amg_data) = NULL;

   hypre_ParAMGDataRAP2(amg_data)              = rap2;
   hypre_ParAMGDataKeepTranspose(amg_data)     = keepT;
   hypre_ParAMGDataModularizedMatMat(amg_data) = modu_rap;

   /* information for preserving indices as coarse grid points */
   hypre_ParAMGDataCPointsMarker(amg_data)      = NULL;
   hypre_ParAMGDataCPointsLocalMarker(amg_data) = NULL;
   hypre_ParAMGDataCPointsLevel(amg_data)       = 0;
   hypre_ParAMGDataNumCPoints(amg_data)         = 0;

   /* information for preserving indices as special fine grid points */
   hypre_ParAMGDataIsolatedFPointsMarker(amg_data) = NULL;
   hypre_ParAMGDataNumIsolatedFPoints(amg_data) = 0;

   hypre_ParAMGDataCumNnzAP(amg_data) = cum_nnz_AP;

#ifdef HYPRE_USING_DSUPERLU
   hypre_ParAMGDataDSLUThreshold(amg_data) = 0;
   hypre_ParAMGDataDSLUSolver(amg_data) = NULL;
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return (void *) amg_data;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGDestroy( void *data )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   if (amg_data)
   {
      HYPRE_Int     num_levels = hypre_ParAMGDataNumLevels(amg_data);
      HYPRE_Int     smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
      HYPRE_Solver *smoother = hypre_ParAMGDataSmoother(amg_data);
      void         *amg = hypre_ParAMGDataCoarseSolver(amg_data);
      MPI_Comm      new_comm = hypre_ParAMGDataNewComm(amg_data);
      HYPRE_Int    *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
      HYPRE_Int     i;
      HYPRE_MemoryLocation memory_location = hypre_ParAMGDataMemoryLocation(amg_data);

#ifdef HYPRE_USING_DSUPERLU
      // if (hypre_ParAMGDataDSLUThreshold(amg_data) > 0)
      if (hypre_ParAMGDataDSLUSolver(amg_data) != NULL)
      {
         hypre_SLUDistDestroy(hypre_ParAMGDataDSLUSolver(amg_data));
         hypre_ParAMGDataDSLUSolver(amg_data) = NULL;
      }
#endif

      if (hypre_ParAMGDataMaxEigEst(amg_data))
      {
         hypre_TFree(hypre_ParAMGDataMaxEigEst(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataMaxEigEst(amg_data) = NULL;
      }
      if (hypre_ParAMGDataMinEigEst(amg_data))
      {
         hypre_TFree(hypre_ParAMGDataMinEigEst(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataMinEigEst(amg_data) = NULL;
      }
      if (hypre_ParAMGDataNumGridSweeps(amg_data))
      {
         hypre_TFree(hypre_ParAMGDataNumGridSweeps(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataNumGridSweeps(amg_data) = NULL;
      }
      if (grid_relax_type)
      {
         HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
         if (grid_relax_type[1] == 15 || grid_relax_type[3] == 15 )
         {
            if (grid_relax_type[1] == 15)
            {
               for (i = 0; i < num_levels; i++)
               {
                  HYPRE_ParCSRPCGDestroy(smoother[i]);
               }
            }
            if (grid_relax_type[3] == 15 && grid_relax_type[1] != 15)
            {
               HYPRE_ParCSRPCGDestroy(smoother[num_levels - 1]);
            }
            hypre_TFree(smoother, HYPRE_MEMORY_HOST);
         }

         hypre_TFree(hypre_ParAMGDataGridRelaxType(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataGridRelaxType(amg_data) = NULL;
      }
      if (hypre_ParAMGDataRelaxWeight(amg_data))
      {
         hypre_TFree(hypre_ParAMGDataRelaxWeight(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataRelaxWeight(amg_data) = NULL;
      }
      if (hypre_ParAMGDataOmega(amg_data))
      {
         hypre_TFree(hypre_ParAMGDataOmega(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataOmega(amg_data) = NULL;
      }
      if (hypre_ParAMGDataNonGalTolArray(amg_data))
      {
         hypre_TFree(hypre_ParAMGDataNonGalTolArray(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataNonGalTolArray(amg_data) = NULL;
      }
      if (hypre_ParAMGDataDofFunc(amg_data))
      {
         hypre_IntArrayDestroy(hypre_ParAMGDataDofFunc(amg_data));
         hypre_ParAMGDataDofFunc(amg_data) = NULL;
      }
      for (i = 1; i < num_levels; i++)
      {
         hypre_ParVectorDestroy(hypre_ParAMGDataFArray(amg_data)[i]);
         hypre_ParVectorDestroy(hypre_ParAMGDataUArray(amg_data)[i]);

         if (hypre_ParAMGDataAArray(amg_data)[i])
         {
            hypre_ParCSRMatrixDestroy(hypre_ParAMGDataAArray(amg_data)[i]);
         }

         if (hypre_ParAMGDataPArray(amg_data)[i - 1])
         {
            hypre_ParCSRMatrixDestroy(hypre_ParAMGDataPArray(amg_data)[i - 1]);
         }

         if (hypre_ParAMGDataRestriction(amg_data))
         {
            if (hypre_ParAMGDataRArray(amg_data)[i - 1])
            {
               hypre_ParCSRMatrixDestroy(hypre_ParAMGDataRArray(amg_data)[i - 1]);
            }
         }

         hypre_IntArrayDestroy(hypre_ParAMGDataCFMarkerArray(amg_data)[i - 1]);

         /* get rid of any block structures */
         if (hypre_ParAMGDataABlockArray(amg_data)[i])
         {
            hypre_ParCSRBlockMatrixDestroy(hypre_ParAMGDataABlockArray(amg_data)[i]);
         }

         if (hypre_ParAMGDataPBlockArray(amg_data)[i - 1])
         {
            hypre_ParCSRBlockMatrixDestroy(hypre_ParAMGDataPBlockArray(amg_data)[i - 1]);
         }

         /* RL */
         if (hypre_ParAMGDataRestriction(amg_data))
         {
            if (hypre_ParAMGDataRBlockArray(amg_data)[i - 1])
            {
               hypre_ParCSRBlockMatrixDestroy(hypre_ParAMGDataRBlockArray(amg_data)[i - 1]);
            }
         }
      }
      if (hypre_ParAMGDataGridRelaxPoints(amg_data))
      {
         for (i = 0; i < 4; i++)
         {
            hypre_TFree(hypre_ParAMGDataGridRelaxPoints(amg_data)[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(hypre_ParAMGDataGridRelaxPoints(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataGridRelaxPoints(amg_data) = NULL;
      }

      hypre_ParCSRMatrixDestroy(hypre_ParAMGDataLambda(amg_data));

      if (hypre_ParAMGDataAtilde(amg_data))
      {
         hypre_ParCSRMatrix *Atilde = hypre_ParAMGDataAtilde(amg_data);
         hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(Atilde));
         hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(Atilde));
         hypre_TFree(Atilde, HYPRE_MEMORY_HOST);
      }

      hypre_ParVectorDestroy(hypre_ParAMGDataXtilde(amg_data));
      hypre_ParVectorDestroy(hypre_ParAMGDataRtilde(amg_data));

      if (hypre_ParAMGDataL1Norms(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            hypre_SeqVectorDestroy(hypre_ParAMGDataL1Norms(amg_data)[i]);
         }
         hypre_TFree(hypre_ParAMGDataL1Norms(amg_data), HYPRE_MEMORY_HOST);
      }

      if (hypre_ParAMGDataChebyCoefs(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            if (hypre_ParAMGDataChebyCoefs(amg_data)[i])
            {
               hypre_TFree(hypre_ParAMGDataChebyCoefs(amg_data)[i], HYPRE_MEMORY_HOST);
            }
         }
         hypre_TFree(hypre_ParAMGDataChebyCoefs(amg_data), HYPRE_MEMORY_HOST);
      }

      if (hypre_ParAMGDataChebyDS(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            hypre_SeqVectorDestroy(hypre_ParAMGDataChebyDS(amg_data)[i]);
         }
         hypre_TFree(hypre_ParAMGDataChebyDS(amg_data), HYPRE_MEMORY_HOST);
      }

      hypre_TFree(hypre_ParAMGDataDinv(amg_data), HYPRE_MEMORY_HOST);

      /* get rid of a fine level block matrix */
      if (hypre_ParAMGDataABlockArray(amg_data))
      {
         if (hypre_ParAMGDataABlockArray(amg_data)[0])
         {
            hypre_ParCSRBlockMatrixDestroy(hypre_ParAMGDataABlockArray(amg_data)[0]);
         }
      }

      /* see comments in par_coarsen.c regarding special case for CF_marker */
      if (num_levels == 1)
      {
         hypre_IntArrayDestroy(hypre_ParAMGDataCFMarkerArray(amg_data)[0]);
      }

      hypre_ParVectorDestroy(hypre_ParAMGDataVtemp(amg_data));
      hypre_TFree(hypre_ParAMGDataFArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParAMGDataUArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParAMGDataAArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParAMGDataABlockArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParAMGDataPBlockArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParAMGDataPArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParAMGDataCFMarkerArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_ParVectorDestroy(hypre_ParAMGDataRtemp(amg_data));
      hypre_ParVectorDestroy(hypre_ParAMGDataPtemp(amg_data));
      hypre_ParVectorDestroy(hypre_ParAMGDataZtemp(amg_data));

      if (hypre_ParAMGDataDofFuncArray(amg_data))
      {
         for (i = 1; i < num_levels; i++)
         {
            hypre_IntArrayDestroy(hypre_ParAMGDataDofFuncArray(amg_data)[i]);
         }
         hypre_TFree(hypre_ParAMGDataDofFuncArray(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataDofFuncArray(amg_data) = NULL;
      }
      if (hypre_ParAMGDataRestriction(amg_data))
      {
         hypre_TFree(hypre_ParAMGDataRBlockArray(amg_data), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_ParAMGDataRArray(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataRArray(amg_data) = NULL;
      }
      if (hypre_ParAMGDataDofPointArray(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            hypre_TFree(hypre_ParAMGDataDofPointArray(amg_data)[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(hypre_ParAMGDataDofPointArray(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataDofPointArray(amg_data) = NULL;
      }
      if (hypre_ParAMGDataPointDofMapArray(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            hypre_TFree(hypre_ParAMGDataPointDofMapArray(amg_data)[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(hypre_ParAMGDataPointDofMapArray(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataPointDofMapArray(amg_data) = NULL;
      }

      if (smooth_num_levels)
      {
         if ( hypre_ParAMGDataSmoothType(amg_data) == 7 ||
              hypre_ParAMGDataSmoothType(amg_data) == 17 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               HYPRE_ParCSRPilutDestroy(smoother[i]);
            }
         }
         else if ( hypre_ParAMGDataSmoothType(amg_data) == 8 ||
                   hypre_ParAMGDataSmoothType(amg_data) == 18 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               HYPRE_ParCSRParaSailsDestroy(smoother[i]);
            }
         }
         else if ( hypre_ParAMGDataSmoothType(amg_data) == 9 ||
                   hypre_ParAMGDataSmoothType(amg_data) == 19 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               HYPRE_EuclidDestroy(smoother[i]);
            }
         }
         else if ( hypre_ParAMGDataSmoothType(amg_data) == 4 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               HYPRE_FSAIDestroy(smoother[i]);
            }
         }
         else if ( hypre_ParAMGDataSmoothType(amg_data) == 5 ||
                   hypre_ParAMGDataSmoothType(amg_data) == 15 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               HYPRE_ILUDestroy(smoother[i]);
            }
         }
         else if ( hypre_ParAMGDataSmoothType(amg_data) == 6 ||
                   hypre_ParAMGDataSmoothType(amg_data) == 16 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               HYPRE_SchwarzDestroy(smoother[i]);
            }
         }
         hypre_TFree(hypre_ParAMGDataSmoother(amg_data), HYPRE_MEMORY_HOST);
      }
      hypre_ParVectorDestroy(hypre_ParAMGDataResidual(amg_data));
      hypre_ParAMGDataResidual(amg_data) = NULL;

      if ( hypre_ParAMGInterpVecVariant(amg_data) > 0 &&
           hypre_ParAMGNumInterpVectors(amg_data) > 0)
      {
         HYPRE_Int         num_vecs =  hypre_ParAMGNumInterpVectors(amg_data);
         hypre_ParVector **sm_vecs;
         HYPRE_Int         j, num_il;

         num_il = hypre_min(hypre_ParAMGNumLevelsInterpVectors(amg_data), num_levels);

         /* don't destroy lev = 0 - this was user input */
         for (i = 1; i < num_il; i++)
         {
            sm_vecs = hypre_ParAMGInterpVectorsArray(amg_data)[i];
            for (j = 0; j < num_vecs; j++)
            {
               hypre_ParVectorDestroy(sm_vecs[j]);
            }
            hypre_TFree(sm_vecs, HYPRE_MEMORY_HOST);
         }
         hypre_TFree(hypre_ParAMGInterpVectorsArray(amg_data), HYPRE_MEMORY_HOST);
      }

      hypre_BoomerAMGDestroy(amg);
      hypre_ParCSRMatrixDestroy(hypre_ParAMGDataACoarse(amg_data));
      hypre_ParVectorDestroy(hypre_ParAMGDataUCoarse(amg_data));
      hypre_ParVectorDestroy(hypre_ParAMGDataFCoarse(amg_data));

      /* destroy input CF_marker data */
      hypre_TFree(hypre_ParAMGDataCPointsMarker(amg_data), memory_location);
      hypre_TFree(hypre_ParAMGDataCPointsLocalMarker(amg_data), memory_location);
      hypre_TFree(hypre_ParAMGDataFPointsMarker(amg_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParAMGDataIsolatedFPointsMarker(amg_data), HYPRE_MEMORY_HOST);

      /* Direct solver for the coarsest level */
#if defined(HYPRE_USING_MAGMA)
      hypre_TFree(hypre_ParAMGDataAPiv(amg_data),  HYPRE_MEMORY_HOST);
#else
      hypre_TFree(hypre_ParAMGDataAPiv(amg_data),  hypre_ParAMGDataGEMemoryLocation(amg_data));
#endif
      hypre_TFree(hypre_ParAMGDataAMat(amg_data),  hypre_ParAMGDataGEMemoryLocation(amg_data));
      hypre_TFree(hypre_ParAMGDataAWork(amg_data), hypre_ParAMGDataGEMemoryLocation(amg_data));
      hypre_TFree(hypre_ParAMGDataBVec(amg_data),  hypre_ParAMGDataGEMemoryLocation(amg_data));
      hypre_TFree(hypre_ParAMGDataUVec(amg_data),  hypre_ParAMGDataGEMemoryLocation(amg_data));
      hypre_TFree(hypre_ParAMGDataCommInfo(amg_data), HYPRE_MEMORY_HOST);

      if (new_comm != hypre_MPI_COMM_NULL)
      {
         hypre_MPI_Comm_free(&new_comm);
      }

      hypre_TFree(amg_data, HYPRE_MEMORY_HOST);
   }
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetRestriction( void *data,
                               HYPRE_Int   restr_par )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* RL: currently, only 0: R = P^T
    *                     1: AIR
    *                     2: AIR-2
    *                     15: a special version of AIR-2 with less communication cost
    *                     k(k>=3,k!=15): Neumann AIR of degree k-3
    */
   if (restr_par < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataRestriction(amg_data) = restr_par;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetIsTriangular(void *data,
                               HYPRE_Int is_triangular )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataIsTriangular(amg_data) = is_triangular;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetGMRESSwitchR(void *data,
                               HYPRE_Int gmres_switch )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataGMRESSwitchR(amg_data) = gmres_switch;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMaxLevels( void *data,
                             HYPRE_Int   max_levels )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   HYPRE_Int old_max_levels;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_levels < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   old_max_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (old_max_levels < max_levels)
   {
      HYPRE_Real *relax_weight, *omega, *nongal_tol_array;
      HYPRE_Real relax_wt, outer_wt, nongalerkin_tol;
      HYPRE_Int i;
      relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);
      if (relax_weight)
      {
         relax_wt = hypre_ParAMGDataUserRelaxWeight(amg_data);
         relax_weight = hypre_TReAlloc(relax_weight,  HYPRE_Real,  max_levels, HYPRE_MEMORY_HOST);
         for (i = old_max_levels; i < max_levels; i++)
         {
            relax_weight[i] = relax_wt;
         }
         hypre_ParAMGDataRelaxWeight(amg_data) = relax_weight;
      }
      omega = hypre_ParAMGDataOmega(amg_data);
      if (omega)
      {
         outer_wt = hypre_ParAMGDataOuterWt(amg_data);
         omega = hypre_TReAlloc(omega,  HYPRE_Real,  max_levels, HYPRE_MEMORY_HOST);
         for (i = old_max_levels; i < max_levels; i++)
         {
            omega[i] = outer_wt;
         }
         hypre_ParAMGDataOmega(amg_data) = omega;
      }
      nongal_tol_array = hypre_ParAMGDataNonGalTolArray(amg_data);
      if (nongal_tol_array)
      {
         nongalerkin_tol = hypre_ParAMGDataNonGalerkinTol(amg_data);
         nongal_tol_array = hypre_TReAlloc(nongal_tol_array,  HYPRE_Real,  max_levels, HYPRE_MEMORY_HOST);
         for (i = old_max_levels; i < max_levels; i++)
         {
            nongal_tol_array[i] = nongalerkin_tol;
         }
         hypre_ParAMGDataNonGalTolArray(amg_data) = nongal_tol_array;
      }
   }
   hypre_ParAMGDataMaxLevels(amg_data) = max_levels;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetMaxLevels( void *data,
                             HYPRE_Int *  max_levels )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_levels = hypre_ParAMGDataMaxLevels(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMaxCoarseSize( void *data,
                                 HYPRE_Int   max_coarse_size )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_coarse_size < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMaxCoarseSize(amg_data) = max_coarse_size;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetMaxCoarseSize( void *data,
                                 HYPRE_Int *  max_coarse_size )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_coarse_size = hypre_ParAMGDataMaxCoarseSize(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMinCoarseSize( void *data,
                                 HYPRE_Int   min_coarse_size )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (min_coarse_size < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMinCoarseSize(amg_data) = min_coarse_size;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetMinCoarseSize( void *data,
                                 HYPRE_Int *  min_coarse_size )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *min_coarse_size = hypre_ParAMGDataMinCoarseSize(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSeqThreshold( void *data,
                                HYPRE_Int   seq_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (seq_threshold < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataSeqThreshold(amg_data) = seq_threshold;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSeqThreshold( void *data,
                                HYPRE_Int *  seq_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *seq_threshold = hypre_ParAMGDataSeqThreshold(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetRedundant( void *data,
                             HYPRE_Int   redundant )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (redundant < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataRedundant(amg_data) = redundant;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetRedundant( void *data,
                             HYPRE_Int *  redundant )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *redundant = hypre_ParAMGDataRedundant(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetCoarsenCutFactor( void       *data,
                                    HYPRE_Int   coarsen_cut_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (coarsen_cut_factor < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataCoarsenCutFactor(amg_data) = coarsen_cut_factor;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetCoarsenCutFactor( void       *data,
                                    HYPRE_Int  *coarsen_cut_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *coarsen_cut_factor = hypre_ParAMGDataCoarsenCutFactor(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetStrongThreshold( void     *data,
                                   HYPRE_Real    strong_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (strong_threshold < 0 || strong_threshold > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataStrongThreshold(amg_data) = strong_threshold;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetStrongThreshold( void     *data,
                                   HYPRE_Real *  strong_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *strong_threshold = hypre_ParAMGDataStrongThreshold(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetStrongThresholdR( void         *data,
                                    HYPRE_Real    strong_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (strong_threshold < 0 || strong_threshold > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataStrongThresholdR(amg_data) = strong_threshold;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetStrongThresholdR( void       *data,
                                    HYPRE_Real *strong_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *strong_threshold = hypre_ParAMGDataStrongThresholdR(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFilterThresholdR( void         *data,
                                    HYPRE_Real    filter_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (filter_threshold < 0 || filter_threshold > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataFilterThresholdR(amg_data) = filter_threshold;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetFilterThresholdR( void       *data,
                                    HYPRE_Real *filter_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *filter_threshold = hypre_ParAMGDataFilterThresholdR(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSabs( void         *data,
                        HYPRE_Int     Sabs )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataSabs(amg_data) = Sabs != 0;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMaxRowSum( void     *data,
                             HYPRE_Real    max_row_sum )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_row_sum <= 0 || max_row_sum > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMaxRowSum(amg_data) = max_row_sum;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetMaxRowSum( void     *data,
                             HYPRE_Real *  max_row_sum )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_row_sum = hypre_ParAMGDataMaxRowSum(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetTruncFactor( void     *data,
                               HYPRE_Real    trunc_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (trunc_factor < 0 || trunc_factor >= 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataTruncFactor(amg_data) = trunc_factor;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetTruncFactor( void     *data,
                               HYPRE_Real *  trunc_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *trunc_factor = hypre_ParAMGDataTruncFactor(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetPMaxElmts( void     *data,
                             HYPRE_Int    P_max_elmts )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (P_max_elmts < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataPMaxElmts(amg_data) = P_max_elmts;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetPMaxElmts( void     *data,
                             HYPRE_Int *  P_max_elmts )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *P_max_elmts = hypre_ParAMGDataPMaxElmts(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetJacobiTruncThreshold( void     *data,
                                        HYPRE_Real    jacobi_trunc_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (jacobi_trunc_threshold < 0 || jacobi_trunc_threshold >= 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataJacobiTruncThreshold(amg_data) = jacobi_trunc_threshold;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetJacobiTruncThreshold( void     *data,
                                        HYPRE_Real *  jacobi_trunc_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *jacobi_trunc_threshold = hypre_ParAMGDataJacobiTruncThreshold(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetPostInterpType( void     *data,
                                  HYPRE_Int    post_interp_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (post_interp_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataPostInterpType(amg_data) = post_interp_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetPostInterpType( void     *data,
                                  HYPRE_Int  * post_interp_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *post_interp_type = hypre_ParAMGDataPostInterpType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetInterpType( void     *data,
                              HYPRE_Int       interp_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }


   if ((interp_type < 0 || interp_type > 25) && interp_type != 100)

   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataInterpType(amg_data) = interp_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetInterpType( void     *data,
                              HYPRE_Int *     interp_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *interp_type = hypre_ParAMGDataInterpType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSepWeight( void     *data,
                             HYPRE_Int       sep_weight )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataSepWeight(amg_data) = sep_weight;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMinIter( void     *data,
                           HYPRE_Int       min_iter )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMinIter(amg_data) = min_iter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetMinIter( void     *data,
                           HYPRE_Int *     min_iter )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *min_iter = hypre_ParAMGDataMinIter(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMaxIter( void     *data,
                           HYPRE_Int     max_iter )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_iter < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMaxIter(amg_data) = max_iter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetMaxIter( void     *data,
                           HYPRE_Int *   max_iter )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_iter = hypre_ParAMGDataMaxIter(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetCoarsenType( void  *data,
                               HYPRE_Int    coarsen_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataCoarsenType(amg_data) = coarsen_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetCoarsenType( void  *data,
                               HYPRE_Int *  coarsen_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMeasureType( void  *data,
                               HYPRE_Int    measure_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMeasureType(amg_data) = measure_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetMeasureType( void  *data,
                               HYPRE_Int *  measure_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *measure_type = hypre_ParAMGDataMeasureType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSetupType( void  *data,
                             HYPRE_Int    setup_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataSetupType(amg_data) = setup_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSetupType( void  *data,
                             HYPRE_Int  *  setup_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *setup_type = hypre_ParAMGDataSetupType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetCycleType( void  *data,
                             HYPRE_Int    cycle_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (cycle_type < 0 || cycle_type > 2)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataCycleType(amg_data) = cycle_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetCycleType( void  *data,
                             HYPRE_Int *  cycle_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *cycle_type = hypre_ParAMGDataCycleType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFCycle( void     *data,
                          HYPRE_Int fcycle )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataFCycle(amg_data) = fcycle != 0;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetFCycle( void      *data,
                          HYPRE_Int *fcycle )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *fcycle = hypre_ParAMGDataFCycle(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetConvergeType( void     *data,
                                HYPRE_Int type  )
{
   /* type 0: default. relative over ||b||
    *      1:          relative over ||r0||
    */
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /*
   if ()
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   */

   hypre_ParAMGDataConvergeType(amg_data) = type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetConvergeType( void      *data,
                                HYPRE_Int *type  )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *type = hypre_ParAMGDataConvergeType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetTol( void     *data,
                       HYPRE_Real    tol  )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (tol < 0 || tol > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataTol(amg_data) = tol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetTol( void     *data,
                       HYPRE_Real *  tol  )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *tol = hypre_ParAMGDataTol(amg_data);

   return hypre_error_flag;
}

/* The "Get" function for SetNumSweeps is GetCycleNumSweeps. */
HYPRE_Int
hypre_BoomerAMGSetNumSweeps( void     *data,
                             HYPRE_Int      num_sweeps )
{
   HYPRE_Int i;
   HYPRE_Int *num_grid_sweeps;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (num_sweeps < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      hypre_ParAMGDataNumGridSweeps(amg_data) = hypre_CTAlloc(HYPRE_Int, 4, HYPRE_MEMORY_HOST);
   }

   num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);

   for (i = 0; i < 3; i++)
   {
      num_grid_sweeps[i] = num_sweeps;
   }
   num_grid_sweeps[3] = 1;

   hypre_ParAMGDataUserNumSweeps(amg_data) = num_sweeps;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetCycleNumSweeps( void     *data,
                                  HYPRE_Int      num_sweeps,
                                  HYPRE_Int      k )
{
   HYPRE_Int i;
   HYPRE_Int *num_grid_sweeps;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (num_sweeps < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (k < 1 || k > 3)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      num_grid_sweeps = hypre_CTAlloc(HYPRE_Int, 4, HYPRE_MEMORY_HOST);
      for (i = 0; i < 4; i++)
      {
         num_grid_sweeps[i] = 1;
      }
      hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;
   }

   hypre_ParAMGDataNumGridSweeps(amg_data)[k] = num_sweeps;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetCycleNumSweeps( void     *data,
                                  HYPRE_Int *    num_sweeps,
                                  HYPRE_Int      k )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *num_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data)[k];

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetNumGridSweeps( void     *data,
                                 HYPRE_Int      *num_grid_sweeps )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!num_grid_sweeps)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data))
   {
      hypre_TFree(hypre_ParAMGDataNumGridSweeps(amg_data), HYPRE_MEMORY_HOST);
   }
   hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetNumGridSweeps( void     *data,
                                 HYPRE_Int    ** num_grid_sweeps )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);

   return hypre_error_flag;
}

/* The "Get" function for SetRelaxType is GetCycleRelaxType. */
HYPRE_Int
hypre_BoomerAMGSetRelaxType( void     *data,
                             HYPRE_Int      relax_type )
{
   HYPRE_Int i;
   HYPRE_Int *grid_relax_type;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (relax_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      hypre_ParAMGDataGridRelaxType(amg_data) = hypre_CTAlloc(HYPRE_Int, 4, HYPRE_MEMORY_HOST);
   }
   grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);

   for (i = 0; i < 3; i++)
   {
      grid_relax_type[i] = relax_type;
   }
   grid_relax_type[3] = 9;
   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = 9;
   hypre_ParAMGDataUserRelaxType(amg_data) = relax_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetCycleRelaxType( void     *data,
                                  HYPRE_Int      relax_type,
                                  HYPRE_Int      k )
{
   HYPRE_Int i;
   HYPRE_Int *grid_relax_type;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (relax_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      grid_relax_type = hypre_CTAlloc(HYPRE_Int, 4, HYPRE_MEMORY_HOST);
      for (i = 0; i < 3; i++)
      {
         grid_relax_type[i] = 3;
      }
      grid_relax_type[3] = 9;
      hypre_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
   }

   hypre_ParAMGDataGridRelaxType(amg_data)[k] = relax_type;
   if (k == 3)
   {
      hypre_ParAMGDataUserCoarseRelaxType(amg_data) = relax_type;
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetCycleRelaxType( void     *data,
                                  HYPRE_Int    * relax_type,
                                  HYPRE_Int      k )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *relax_type = hypre_ParAMGDataGridRelaxType(amg_data)[k];

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetRelaxOrder( void     *data,
                              HYPRE_Int       relax_order)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataRelaxOrder(amg_data) = relax_order;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetRelaxOrder( void     *data,
                              HYPRE_Int     * relax_order)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *relax_order = hypre_ParAMGDataRelaxOrder(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetGridRelaxType( void     *data,
                                 HYPRE_Int      *grid_relax_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!grid_relax_type)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data))
   {
      hypre_TFree(hypre_ParAMGDataGridRelaxType(amg_data), HYPRE_MEMORY_HOST);
   }
   hypre_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = grid_relax_type[3];

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetGridRelaxType( void     *data,
                                 HYPRE_Int    ** grid_relax_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetGridRelaxPoints( void     *data,
                                   HYPRE_Int      **grid_relax_points )
{
   HYPRE_Int i;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!grid_relax_points)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxPoints(amg_data))
   {
      for (i = 0; i < 4; i++)
      {
         hypre_TFree(hypre_ParAMGDataGridRelaxPoints(amg_data)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParAMGDataGridRelaxPoints(amg_data), HYPRE_MEMORY_HOST);
   }
   hypre_ParAMGDataGridRelaxPoints(amg_data) = grid_relax_points;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetGridRelaxPoints( void     *data,
                                   HYPRE_Int    *** grid_relax_points )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetRelaxWeight( void     *data,
                               HYPRE_Real   *relax_weight )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!relax_weight)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataRelaxWeight(amg_data))
   {
      hypre_TFree(hypre_ParAMGDataRelaxWeight(amg_data), HYPRE_MEMORY_HOST);
   }
   hypre_ParAMGDataRelaxWeight(amg_data) = relax_weight;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetRelaxWeight( void     *data,
                               HYPRE_Real ** relax_weight )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetRelaxWt( void     *data,
                           HYPRE_Real    relax_weight )
{
   HYPRE_Int i, num_levels;
   HYPRE_Real *relax_weight_array;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      hypre_ParAMGDataRelaxWeight(amg_data) = hypre_CTAlloc(HYPRE_Real, num_levels, HYPRE_MEMORY_HOST);
   }

   relax_weight_array = hypre_ParAMGDataRelaxWeight(amg_data);
   for (i = 0; i < num_levels; i++)
   {
      relax_weight_array[i] = relax_weight;
   }

   hypre_ParAMGDataUserRelaxWeight(amg_data) = relax_weight;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetLevelRelaxWt( void    *data,
                                HYPRE_Real   relax_weight,
                                HYPRE_Int      level )
{
   HYPRE_Int i, num_levels;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1 || level < 0)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      hypre_ParAMGDataRelaxWeight(amg_data) = hypre_CTAlloc(HYPRE_Real, num_levels, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_levels; i++)
      {
         hypre_ParAMGDataRelaxWeight(amg_data)[i] = 1.0;
      }
   }

   hypre_ParAMGDataRelaxWeight(amg_data)[level] = relax_weight;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetLevelRelaxWt( void    *data,
                                HYPRE_Real * relax_weight,
                                HYPRE_Int      level )
{
   HYPRE_Int num_levels;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1 || level < 0)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *relax_weight = hypre_ParAMGDataRelaxWeight(amg_data)[level];

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetOmega( void     *data,
                         HYPRE_Real   *omega )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!omega)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataOmega(amg_data))
   {
      hypre_TFree(hypre_ParAMGDataOmega(amg_data), HYPRE_MEMORY_HOST);
   }
   hypre_ParAMGDataOmega(amg_data) = omega;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetOmega( void     *data,
                         HYPRE_Real ** omega )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *omega = hypre_ParAMGDataOmega(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetOuterWt( void     *data,
                           HYPRE_Real    omega )
{
   HYPRE_Int i, num_levels;
   HYPRE_Real *omega_array;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      hypre_ParAMGDataOmega(amg_data) = hypre_CTAlloc(HYPRE_Real, num_levels, HYPRE_MEMORY_HOST);
   }

   omega_array = hypre_ParAMGDataOmega(amg_data);
   for (i = 0; i < num_levels; i++)
   {
      omega_array[i] = omega;
   }
   hypre_ParAMGDataOuterWt(amg_data) = omega;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetLevelOuterWt( void    *data,
                                HYPRE_Real   omega,
                                HYPRE_Int      level )
{
   HYPRE_Int i, num_levels;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      hypre_ParAMGDataOmega(amg_data) = hypre_CTAlloc(HYPRE_Real, num_levels, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_levels; i++)
      {
         hypre_ParAMGDataOmega(amg_data)[i] = 1.0;
      }
   }

   hypre_ParAMGDataOmega(amg_data)[level] = omega;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetLevelOuterWt( void    *data,
                                HYPRE_Real * omega,
                                HYPRE_Int      level )
{
   HYPRE_Int num_levels;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *omega = hypre_ParAMGDataOmega(amg_data)[level];

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSmoothType( void     *data,
                              HYPRE_Int   smooth_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataSmoothType(amg_data) = smooth_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSmoothType( void     *data,
                              HYPRE_Int * smooth_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *smooth_type = hypre_ParAMGDataSmoothType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSmoothNumLevels( void     *data,
                                   HYPRE_Int   smooth_num_levels )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (smooth_num_levels < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataSmoothNumLevels(amg_data) = smooth_num_levels;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSmoothNumLevels( void     *data,
                                   HYPRE_Int * smooth_num_levels )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSmoothNumSweeps( void     *data,
                                   HYPRE_Int   smooth_num_sweeps )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (smooth_num_sweeps < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataSmoothNumSweeps(amg_data) = smooth_num_sweeps;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSmoothNumSweeps( void     *data,
                                   HYPRE_Int * smooth_num_sweeps )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *smooth_num_sweeps = hypre_ParAMGDataSmoothNumSweeps(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetLogging( void     *data,
                           HYPRE_Int       logging )
{
   /* This function should be called before Setup.  Logging changes
      may require allocation or freeing of arrays, which is presently
      only done there.
      It may be possible to support logging changes at other times,
      but there is little need.
   */
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataLogging(amg_data) = logging;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetLogging( void     *data,
                           HYPRE_Int     * logging )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *logging = hypre_ParAMGDataLogging(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetPrintLevel( void     *data,
                              HYPRE_Int print_level )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataPrintLevel(amg_data) = print_level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetPrintLevel( void     *data,
                              HYPRE_Int * print_level )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *print_level =  hypre_ParAMGDataPrintLevel(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetPrintFileName( void       *data,
                                 const char *print_file_name )
{
   hypre_ParAMGData  *amg_data =  (hypre_ParAMGData*)data;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if ( strlen(print_file_name) > 256 )
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", print_file_name);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetPrintFileName( void       *data,
                                 char ** print_file_name )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_sprintf( *print_file_name, "%s", hypre_ParAMGDataLogFileName(amg_data) );

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetNumIterations( void    *data,
                                 HYPRE_Int      num_iterations )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNumIterations(amg_data) = num_iterations;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetDebugFlag( void     *data,
                             HYPRE_Int       debug_flag )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataDebugFlag(amg_data) = debug_flag;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetDebugFlag( void     *data,
                             HYPRE_Int     * debug_flag )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *debug_flag = hypre_ParAMGDataDebugFlag(amg_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetGSMG( void       *data,
                        HYPRE_Int   par )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   amg_data->gsmg = par;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNumSamples( void *data,
                              HYPRE_Int   par )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   amg_data->num_samples = par;

   return hypre_error_flag;
}

/* BM Aug 25, 2006 */

HYPRE_Int
hypre_BoomerAMGSetCGCIts( void *data,
                          HYPRE_Int  its)
{
   HYPRE_Int ierr = 0;
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataCGCIts(amg_data) = its;
   return (ierr);
}

/* BM Oct 22, 2006 */
HYPRE_Int
hypre_BoomerAMGSetPlotGrids( void *data,
                             HYPRE_Int plotgrids)
{
   HYPRE_Int ierr = 0;
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataPlotGrids(amg_data) = plotgrids;
   return (ierr);
}

HYPRE_Int
hypre_BoomerAMGSetPlotFileName( void       *data,
                                const char *plot_file_name )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if ( strlen(plot_file_name) > 251 )
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   if (strlen(plot_file_name) == 0 )
   {
      hypre_sprintf(hypre_ParAMGDataPlotFileName(amg_data), "%s", "AMGgrids.CF.dat");
   }
   else
   {
      hypre_sprintf(hypre_ParAMGDataPlotFileName(amg_data), "%s", plot_file_name);
   }

   return hypre_error_flag;
}
/* Get the coarse grid hierarchy. Assumes cgrid is preallocated to the size of the local matrix.
 * Adapted from par_amg_setup.c, and simplified by ignoring printing in block mode.
 * We do a memcpy on the final grid hierarchy to avoid modifying user allocated data.
*/
HYPRE_Int
hypre_BoomerAMGGetGridHierarchy( void       *data,
                                 HYPRE_Int *cgrid )
{
   HYPRE_Int *ibuff = NULL;
   HYPRE_Int *wbuff, *cbuff, *tmp;
   HYPRE_Int local_size, lev_size, i, j, level, num_levels, block_mode;
   hypre_IntArray          *CF_marker_array;
   hypre_IntArray          *CF_marker_array_host;
   HYPRE_Int               *CF_marker;

   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!cgrid)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   block_mode = hypre_ParAMGDataBlockMode(amg_data);

   if ( block_mode)
   {
      hypre_ParCSRBlockMatrix **A_block_array;
      A_block_array = hypre_ParAMGDataABlockArray(amg_data);
      if (A_block_array == NULL)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid AMG data. AMG setup has not been called!!\n");
         return hypre_error_flag;
      }

      // get local size and allocate some memory
      local_size = hypre_CSRMatrixNumRows(hypre_ParCSRBlockMatrixDiag(A_block_array[0]));
      ibuff  = hypre_CTAlloc(HYPRE_Int, (2 * local_size), HYPRE_MEMORY_HOST);
      wbuff  = ibuff;
      cbuff  = ibuff + local_size;

      num_levels = hypre_ParAMGDataNumLevels(amg_data);
      for (level = (num_levels - 2); level >= 0; level--)
      {
         /* get the CF marker array on the host */
         CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data)[level];
         if (hypre_GetActualMemLocation(hypre_IntArrayMemoryLocation(CF_marker_array)) ==
             hypre_MEMORY_DEVICE)
         {
            CF_marker_array_host = hypre_IntArrayCloneDeep_v2(CF_marker_array, HYPRE_MEMORY_HOST);
         }
         else
         {
            CF_marker_array_host = CF_marker_array;
         }
         CF_marker = hypre_IntArrayData(CF_marker_array_host);

         /* swap pointers */
         tmp = wbuff;
         wbuff = cbuff;
         cbuff = tmp;

         lev_size = hypre_CSRMatrixNumRows(hypre_ParCSRBlockMatrixDiag(A_block_array[level]));

         for (i = 0, j = 0; i < lev_size; i++)
         {
            /* if a C-point */
            cbuff[i] = 0;
            if (CF_marker[i] > -1)
            {
               cbuff[i] = wbuff[j] + 1;
               j++;
            }
         }

         /* destroy copy host copy if necessary */
         if (hypre_GetActualMemLocation(hypre_IntArrayMemoryLocation(CF_marker_array)) ==
             hypre_MEMORY_DEVICE)
         {
            hypre_IntArrayDestroy(CF_marker_array_host);
         }
      }
   }
   else
   {
      hypre_ParCSRMatrix **A_array;
      A_array = hypre_ParAMGDataAArray(amg_data);
      if (A_array == NULL)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid AMG data. AMG setup has not been called!!\n");
         return hypre_error_flag;
      }

      // get local size and allocate some memory
      local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
      wbuff  = hypre_CTAlloc(HYPRE_Int, (2 * local_size), HYPRE_MEMORY_HOST);
      cbuff  = wbuff + local_size;

      num_levels = hypre_ParAMGDataNumLevels(amg_data);
      for (level = (num_levels - 2); level >= 0; level--)
      {
         /* get the CF marker array on the host */
         CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data)[level];
         if (hypre_GetActualMemLocation(hypre_IntArrayMemoryLocation(CF_marker_array)) ==
             hypre_MEMORY_DEVICE)
         {
            CF_marker_array_host = hypre_IntArrayCloneDeep_v2(CF_marker_array, HYPRE_MEMORY_HOST);
         }
         else
         {
            CF_marker_array_host = CF_marker_array;
         }
         CF_marker = hypre_IntArrayData(CF_marker_array_host);
         /* swap pointers */
         tmp = wbuff;
         wbuff = cbuff;
         cbuff = tmp;

         lev_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));

         for (i = 0, j = 0; i < lev_size; i++)
         {
            /* if a C-point */
            cbuff[i] = 0;
            if (CF_marker[i] > -1)
            {
               cbuff[i] = wbuff[j] + 1;
               j++;
            }
         }
         /* destroy copy host copy if necessary */
         if (hypre_GetActualMemLocation(hypre_IntArrayMemoryLocation(CF_marker_array)) ==
             hypre_MEMORY_DEVICE)
         {
            hypre_IntArrayDestroy(CF_marker_array_host);
         }
      }
   }
   // copy hierarchy into user provided array
   hypre_TMemcpy(cgrid, cbuff, HYPRE_Int, local_size, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   // free memory
   hypre_TFree(ibuff, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* BM Oct 17, 2006 */
HYPRE_Int
hypre_BoomerAMGSetCoordDim( void *data,
                            HYPRE_Int coorddim)
{
   HYPRE_Int ierr = 0;
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataCoordDim(amg_data) = coorddim;
   return (ierr);
}

HYPRE_Int
hypre_BoomerAMGSetCoordinates( void *data,
                               float *coordinates)
{
   HYPRE_Int ierr = 0;
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataCoordinates(amg_data) = coordinates;
   return (ierr);
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNumFunctions( void     *data,
                                HYPRE_Int       num_functions )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (num_functions < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNumFunctions(amg_data) = num_functions;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetNumFunctions( void     *data,
                                HYPRE_Int     * num_functions )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *num_functions = hypre_ParAMGDataNumFunctions(amg_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicate whether to use nodal systems function
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNodal( void     *data,
                         HYPRE_Int    nodal )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNodal(amg_data) = nodal;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate number of levels for nodal coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNodalLevels( void     *data,
                               HYPRE_Int    nodal_levels )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNodalLevels(amg_data) = nodal_levels;

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * Indicate how to treat diag for primary matrix with  nodal systems function
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNodalDiag( void     *data,
                             HYPRE_Int    nodal )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNodalDiag(amg_data) = nodal;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate whether to discard same sign coefficients in S for nodal>0
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetKeepSameSign( void      *data,
                                HYPRE_Int  keep_same_sign )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataKeepSameSign(amg_data) = keep_same_sign;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate the degree of aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNumPaths( void     *data,
                            HYPRE_Int       num_paths )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (num_paths < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNumPaths(amg_data) = num_paths;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of levels of aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggNumLevels( void     *data,
                                HYPRE_Int       agg_num_levels )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_num_levels < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAggNumLevels(amg_data) = agg_num_levels;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the interpolation used with aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggInterpType( void     *data,
                                 HYPRE_Int       agg_interp_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_interp_type < 0 || agg_interp_type > 9)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAggInterpType(amg_data) = agg_interp_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates max number of elements per row for aggressive coarsening
 * interpolation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggPMaxElmts( void     *data,
                                HYPRE_Int       agg_P_max_elmts )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_P_max_elmts < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAggPMaxElmts(amg_data) = agg_P_max_elmts;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates max number of elements per row for smoothed
 * interpolation in mult-additive or simple method
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetMultAddPMaxElmts( void     *data,
                                    HYPRE_Int       add_P_max_elmts )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (add_P_max_elmts < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataMultAddPMaxElmts(amg_data) = add_P_max_elmts;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates Relaxtion Type for Additive Cycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAddRelaxType( void     *data,
                                HYPRE_Int       add_rlx_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAddRelaxType(amg_data) = add_rlx_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates Relaxation Weight for Additive Cycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAddRelaxWt( void     *data,
                              HYPRE_Real       add_rlx_wt )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAddRelaxWt(amg_data) = add_rlx_wt;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates max number of elements per row for 1st stage of aggressive
 * coarsening two-stage interpolation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggP12MaxElmts( void     *data,
                                  HYPRE_Int       agg_P12_max_elmts )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_P12_max_elmts < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAggP12MaxElmts(amg_data) = agg_P12_max_elmts;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates truncation factor for aggressive coarsening interpolation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggTruncFactor( void     *data,
                                  HYPRE_Real  agg_trunc_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_trunc_factor < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAggTruncFactor(amg_data) = agg_trunc_factor;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the truncation factor for smoothed interpolation when using
 * mult-additive or simple method
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetMultAddTruncFactor( void     *data,
                                      HYPRE_Real      add_trunc_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (add_trunc_factor < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataMultAddTruncFactor(amg_data) = add_trunc_factor;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates truncation factor for 1 stage of aggressive coarsening
 * two stage interpolation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggP12TruncFactor( void     *data,
                                     HYPRE_Real  agg_P12_trunc_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_P12_trunc_factor < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAggP12TruncFactor(amg_data) = agg_P12_trunc_factor;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of relaxation steps for Compatible relaxation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNumCRRelaxSteps( void     *data,
                                   HYPRE_Int       num_CR_relax_steps )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (num_CR_relax_steps < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNumCRRelaxSteps(amg_data) = num_CR_relax_steps;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the desired convergence rate for Compatible relaxation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetCRRate( void     *data,
                          HYPRE_Real    CR_rate )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataCRRate(amg_data) = CR_rate;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the desired convergence rate for Compatible relaxation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetCRStrongTh( void     *data,
                              HYPRE_Real    CR_strong_th )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataCRStrongTh(amg_data) = CR_strong_th;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the drop tolerance for A-matrices from the 2nd level of AMG
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetADropTol( void     *data,
                            HYPRE_Real  A_drop_tol )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataADropTol(amg_data) = A_drop_tol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetADropType( void      *data,
                             HYPRE_Int  A_drop_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataADropType(amg_data) = A_drop_type;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicates which independent set algorithm is used for CR
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetISType( void     *data,
                          HYPRE_Int      IS_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (IS_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataISType(amg_data) = IS_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates whether to use CG for compatible relaxation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetCRUseCG( void     *data,
                           HYPRE_Int      CR_use_CG )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataCRUseCG(amg_data) = CR_use_CG;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetNumPoints( void     *data,
                             HYPRE_Int       num_points )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNumPoints(amg_data) = num_points;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetDofFunc( void                 *data,
                           HYPRE_Int            *dof_func)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_IntArrayDestroy(hypre_ParAMGDataDofFunc(amg_data));
   /* NOTE: size and memory location of hypre_IntArray will be set during AMG setup */
   if (dof_func == NULL)
   {
      hypre_ParAMGDataDofFunc(amg_data) = NULL;
   }
   else
   {
      hypre_ParAMGDataDofFunc(amg_data) = hypre_IntArrayCreate(-1);
      hypre_IntArrayData(hypre_ParAMGDataDofFunc(amg_data)) = dof_func;
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetPointDofMap( void     *data,
                               HYPRE_Int      *point_dof_map )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_TFree(hypre_ParAMGDataPointDofMap(amg_data), HYPRE_MEMORY_HOST);
   hypre_ParAMGDataPointDofMap(amg_data) = point_dof_map;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetDofPoint( void     *data,
                            HYPRE_Int      *dof_point )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_TFree(hypre_ParAMGDataDofPoint(amg_data), HYPRE_MEMORY_HOST);
   hypre_ParAMGDataDofPoint(amg_data) = dof_point;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetNumIterations( void     *data,
                                 HYPRE_Int      *num_iterations )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *num_iterations = hypre_ParAMGDataNumIterations(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetCumNumIterations( void     *data,
                                    HYPRE_Int      *cum_num_iterations )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
#ifdef CUMNUMIT
   *cum_num_iterations = hypre_ParAMGDataCumNumIterations(amg_data);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetResidual( void * data, hypre_ParVector ** resid )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *resid = hypre_ParAMGDataResidual( amg_data );
   return hypre_error_flag;
}


HYPRE_Int
hypre_BoomerAMGGetRelResidualNorm( void     *data,
                                   HYPRE_Real   *rel_resid_norm )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *rel_resid_norm = hypre_ParAMGDataRelativeResidualNorm(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetVariant( void     *data,
                           HYPRE_Int       variant)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (variant < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataVariant(amg_data) = variant;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetVariant( void     *data,
                           HYPRE_Int     * variant)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *variant = hypre_ParAMGDataVariant(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetOverlap( void     *data,
                           HYPRE_Int       overlap)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (overlap < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataOverlap(amg_data) = overlap;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetOverlap( void     *data,
                           HYPRE_Int     * overlap)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *overlap = hypre_ParAMGDataOverlap(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetDomainType( void     *data,
                              HYPRE_Int       domain_type)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (domain_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataDomainType(amg_data) = domain_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetDomainType( void     *data,
                              HYPRE_Int     * domain_type)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *domain_type = hypre_ParAMGDataDomainType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSchwarzRlxWeight( void     *data,
                                    HYPRE_Real schwarz_rlx_weight)
{
   hypre_ParAMGData  *amg_data =  (hypre_ParAMGData*)data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataSchwarzRlxWeight(amg_data) = schwarz_rlx_weight;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSchwarzRlxWeight( void     *data,
                                    HYPRE_Real   * schwarz_rlx_weight)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *schwarz_rlx_weight = hypre_ParAMGDataSchwarzRlxWeight(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSchwarzUseNonSymm( void     *data,
                                     HYPRE_Int use_nonsymm)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataSchwarzUseNonSymm(amg_data) = use_nonsymm;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSym( void     *data,
                       HYPRE_Int       sym)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataSym(amg_data) = sym;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetLevel( void     *data,
                         HYPRE_Int       level)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataLevel(amg_data) = level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetThreshold( void     *data,
                             HYPRE_Real    thresh)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataThreshold(amg_data) = thresh;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFilter( void     *data,
                          HYPRE_Real    filter)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataFilter(amg_data) = filter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetDropTol( void     *data,
                           HYPRE_Real    drop_tol)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataDropTol(amg_data) = drop_tol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMaxNzPerRow( void     *data,
                               HYPRE_Int       max_nz_per_row)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (max_nz_per_row < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataMaxNzPerRow(amg_data) = max_nz_per_row;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetEuclidFile( void     *data,
                              char     *euclidfile)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataEuclidFile(amg_data) = euclidfile;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetEuLevel( void     *data,
                           HYPRE_Int      eu_level)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataEuLevel(amg_data) = eu_level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetEuSparseA( void     *data,
                             HYPRE_Real    eu_sparse_A)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataEuSparseA(amg_data) = eu_sparse_A;

   return hypre_error_flag;
}


HYPRE_Int
hypre_BoomerAMGSetEuBJ( void     *data,
                        HYPRE_Int       eu_bj)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataEuBJ(amg_data) = eu_bj;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILUType( void     *data,
                           HYPRE_Int       ilu_type)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILUType(amg_data) = ilu_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILULevel( void     *data,
                            HYPRE_Int       ilu_lfil)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILULevel(amg_data) = ilu_lfil;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILUDroptol( void     *data,
                              HYPRE_Real       ilu_droptol)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILUDroptol(amg_data) = ilu_droptol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILUTriSolve( void     *data,
                               HYPRE_Int    ilu_tri_solve)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILUTriSolve(amg_data) = ilu_tri_solve;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILULowerJacobiIters( void     *data,
                                       HYPRE_Int    ilu_lower_jacobi_iters)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILULowerJacobiIters(amg_data) = ilu_lower_jacobi_iters;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILUUpperJacobiIters( void     *data,
                                       HYPRE_Int    ilu_upper_jacobi_iters)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILUUpperJacobiIters(amg_data) = ilu_upper_jacobi_iters;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILUMaxIter( void     *data,
                              HYPRE_Int       ilu_max_iter)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILUMaxIter(amg_data) = ilu_max_iter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILUMaxRowNnz( void     *data,
                                HYPRE_Int       ilu_max_row_nnz)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILUMaxRowNnz(amg_data) = ilu_max_row_nnz;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILULocalReordering( void     *data,
                                      HYPRE_Int       ilu_reordering_type)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILULocalReordering(amg_data) = ilu_reordering_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILUIterSetupType( void     *data,
                                    HYPRE_Int       ilu_iter_setup_type)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILUIterSetupType(amg_data) = ilu_iter_setup_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILUIterSetupOption( void     *data,
                                      HYPRE_Int       ilu_iter_setup_option)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILUIterSetupOption(amg_data) = ilu_iter_setup_option;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILUIterSetupMaxIter( void     *data,
                                       HYPRE_Int       ilu_iter_setup_max_iter)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILUIterSetupMaxIter(amg_data) = ilu_iter_setup_max_iter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetILUIterSetupTolerance( void     *data,
                                         HYPRE_Real       ilu_iter_setup_tolerance)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataILUIterSetupTolerance(amg_data) = ilu_iter_setup_tolerance;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFSAIAlgoType( void      *data,
                                HYPRE_Int  fsai_algo_type)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataFSAIAlgoType(amg_data) = fsai_algo_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFSAILocalSolveType( void      *data,
                                      HYPRE_Int  fsai_local_solve_type)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataFSAILocalSolveType(amg_data) = fsai_local_solve_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFSAIMaxSteps( void      *data,
                                HYPRE_Int  fsai_max_steps)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataFSAIMaxSteps(amg_data) = fsai_max_steps;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFSAIMaxStepSize( void      *data,
                                   HYPRE_Int  fsai_max_step_size)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataFSAIMaxStepSize(amg_data) = fsai_max_step_size;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFSAIMaxNnzRow( void      *data,
                                 HYPRE_Int  fsai_max_nnz_row)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataFSAIMaxNnzRow(amg_data) = fsai_max_nnz_row;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFSAINumLevels( void      *data,
                                 HYPRE_Int  fsai_num_levels)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataFSAINumLevels(amg_data) = fsai_num_levels;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFSAIThreshold( void      *data,
                                 HYPRE_Real fsai_threshold)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataFSAIThreshold(amg_data) = fsai_threshold;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFSAIEigMaxIters( void      *data,
                                   HYPRE_Int  fsai_eig_max_iters)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataFSAIEigMaxIters(amg_data) = fsai_eig_max_iters;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFSAIKapTolerance( void      *data,
                                    HYPRE_Real fsai_kap_tolerance)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataFSAIKapTolerance(amg_data) = fsai_kap_tolerance;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetChebyOrder( void     *data,
                              HYPRE_Int       order)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (order < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataChebyOrder(amg_data) = order;

   return hypre_error_flag;
}
HYPRE_Int
hypre_BoomerAMGSetChebyFraction( void     *data,
                                 HYPRE_Real  ratio)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (ratio <= 0.0 || ratio > 1.0 )
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataChebyFraction(amg_data) = ratio;

   return hypre_error_flag;
}
HYPRE_Int
hypre_BoomerAMGSetChebyEigEst( void     *data,
                               HYPRE_Int     cheby_eig_est)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (cheby_eig_est < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataChebyEigEst(amg_data) = cheby_eig_est;

   return hypre_error_flag;
}
HYPRE_Int
hypre_BoomerAMGSetChebyVariant( void     *data,
                                HYPRE_Int     cheby_variant)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataChebyVariant(amg_data) = cheby_variant;

   return hypre_error_flag;
}
HYPRE_Int
hypre_BoomerAMGSetChebyScale( void     *data,
                              HYPRE_Int     cheby_scale)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataChebyScale(amg_data) = cheby_scale;

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetInterpVectors
 * -used for post-interpolation fitting of smooth vectors
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoomerAMGSetInterpVectors(void *solver,
                                          HYPRE_Int  num_vectors,
                                          hypre_ParVector **interp_vectors)

{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) solver;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGInterpVectors(amg_data) =  interp_vectors;
   hypre_ParAMGNumInterpVectors(amg_data) = num_vectors;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetInterpVectorValues
 * -used for post-interpolation fitting of smooth vectors
 *--------------------------------------------------------------------------*/

/*HYPRE_Int hypre_BoomerAMGSetInterpVectorValues(void *solver,
                                    HYPRE_Int  num_vectors,
                                    HYPRE_Complex *interp_vector_values)

{
   hypre_ParAMGData *amg_data = solver;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGInterpVectors(amg_data) =  interp_vectors;
   hypre_ParAMGNumInterpVectors(amg_data) = num_vectors;

   return hypre_error_flag;
}*/

HYPRE_Int hypre_BoomerAMGSetInterpVecVariant(void *solver,
                                             HYPRE_Int  var)


{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) solver;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (var < 1)
   {
      var = 0;
   }
   if (var > 3)
   {
      var = 3;
   }

   hypre_ParAMGInterpVecVariant(amg_data) = var;

   return hypre_error_flag;

}

HYPRE_Int
hypre_BoomerAMGSetInterpVecQMax( void     *data,
                                 HYPRE_Int    q_max)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGInterpVecQMax(amg_data) = q_max;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetInterpVecAbsQTrunc( void     *data,
                                      HYPRE_Real    q_trunc)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGInterpVecAbsQTrunc(amg_data) = q_trunc;

   return hypre_error_flag;
}

HYPRE_Int hypre_BoomerAMGSetSmoothInterpVectors(void *solver,
                                                HYPRE_Int  smooth_interp_vectors)

{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) solver;
   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGSmoothInterpVectors(amg_data) = smooth_interp_vectors;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetInterpRefine( void     *data,
                                HYPRE_Int       num_refine )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGInterpRefine(amg_data) = num_refine;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetInterpVecFirstLevel( void     *data,
                                       HYPRE_Int  level )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGInterpVecFirstLevel(amg_data) = level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetAdditive( void *data,
                            HYPRE_Int   additive )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataAdditive(amg_data) = additive;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetAdditive( void *data,
                            HYPRE_Int *  additive )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *additive = hypre_ParAMGDataAdditive(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMultAdditive( void *data,
                                HYPRE_Int   mult_additive )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMultAdditive(amg_data) = mult_additive;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetMultAdditive( void *data,
                                HYPRE_Int *  mult_additive )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *mult_additive = hypre_ParAMGDataMultAdditive(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSimple( void *data,
                          HYPRE_Int   simple )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataSimple(amg_data) = simple;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSimple( void *data,
                          HYPRE_Int *  simple )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *simple = hypre_ParAMGDataSimple(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetAddLastLvl( void *data,
                              HYPRE_Int   add_last_lvl )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataAddLastLvl(amg_data) = add_last_lvl;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetNonGalerkinTol( void   *data,
                                  HYPRE_Real nongalerkin_tol)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;
   HYPRE_Int i, max_num_levels;
   HYPRE_Real *nongal_tol_array;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (nongalerkin_tol < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   max_num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   nongal_tol_array = hypre_ParAMGDataNonGalTolArray(amg_data);

   if (nongal_tol_array == NULL)
   {
      nongal_tol_array = hypre_CTAlloc(HYPRE_Real,  max_num_levels, HYPRE_MEMORY_HOST);
      hypre_ParAMGDataNonGalTolArray(amg_data) = nongal_tol_array;
   }
   hypre_ParAMGDataNonGalerkinTol(amg_data) = nongalerkin_tol;

   for (i = 0; i < max_num_levels; i++)
   {
      nongal_tol_array[i] = nongalerkin_tol;
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetLevelNonGalerkinTol( void   *data,
                                       HYPRE_Real   nongalerkin_tol,
                                       HYPRE_Int level)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;
   HYPRE_Real *nongal_tol_array;
   HYPRE_Int max_num_levels;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nongalerkin_tol < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   nongal_tol_array = hypre_ParAMGDataNonGalTolArray(amg_data);
   max_num_levels = hypre_ParAMGDataMaxLevels(amg_data);

   if (nongal_tol_array == NULL)
   {
      nongal_tol_array = hypre_CTAlloc(HYPRE_Real,  max_num_levels, HYPRE_MEMORY_HOST);
      hypre_ParAMGDataNonGalTolArray(amg_data) = nongal_tol_array;
   }

   if (level + 1 > max_num_levels)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   nongal_tol_array[level] = nongalerkin_tol;
   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetNonGalerkTol( void   *data,
                                HYPRE_Int   nongalerk_num_tol,
                                HYPRE_Real *nongalerk_tol)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataNonGalerkNumTol(amg_data) = nongalerk_num_tol;
   hypre_ParAMGDataNonGalerkTol(amg_data) = nongalerk_tol;
   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetRAP2( void      *data,
                        HYPRE_Int  rap2 )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataRAP2(amg_data) = rap2;
   return hypre_error_flag;
}


HYPRE_Int
hypre_BoomerAMGSetModuleRAP2( void      *data,
                              HYPRE_Int  mod_rap2 )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataModularizedMatMat(amg_data) = mod_rap2;
   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetKeepTranspose( void       *data,
                                 HYPRE_Int   keepTranspose)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataKeepTranspose(amg_data) = keepTranspose;
   return hypre_error_flag;
}

#ifdef HYPRE_USING_DSUPERLU
HYPRE_Int
hypre_BoomerAMGSetDSLUThreshold( void   *data,
                                 HYPRE_Int   dslu_threshold)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataDSLUThreshold(amg_data) = dslu_threshold;
   return hypre_error_flag;
}
#endif

HYPRE_Int
hypre_BoomerAMGSetCPoints(void         *data,
                          HYPRE_Int     cpt_coarse_level,
                          HYPRE_Int     num_cpt_coarse,
                          HYPRE_BigInt *cpt_coarse_index)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   HYPRE_BigInt     *C_points_marker = NULL;
   HYPRE_Int        *C_points_local_marker = NULL;
   HYPRE_Int         cpt_level;

   if (!amg_data)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! AMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (cpt_coarse_level < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! cpt_coarse_level < 0 !\n");
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   if (num_cpt_coarse < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! num_cpt_coarse < 0 !\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   HYPRE_MemoryLocation memory_location = hypre_ParAMGDataMemoryLocation(amg_data);

   /* free data not previously destroyed */
   if (hypre_ParAMGDataCPointsLevel(amg_data))
   {
      hypre_TFree(hypre_ParAMGDataCPointsMarker(amg_data), memory_location);
      hypre_TFree(hypre_ParAMGDataCPointsLocalMarker(amg_data), memory_location);
   }

   /* set Cpoint data */
   if (hypre_ParAMGDataMaxLevels(amg_data) < cpt_coarse_level)
   {
      cpt_level = hypre_ParAMGDataNumLevels(amg_data);
   }
   else
   {
      cpt_level = cpt_coarse_level;
   }

   if (cpt_level)
   {
      C_points_marker = hypre_CTAlloc(HYPRE_BigInt, num_cpt_coarse, memory_location);
      C_points_local_marker = hypre_CTAlloc(HYPRE_Int, num_cpt_coarse, memory_location);

      hypre_TMemcpy(C_points_marker, cpt_coarse_index, HYPRE_BigInt, num_cpt_coarse, memory_location,
                    HYPRE_MEMORY_HOST);
   }
   hypre_ParAMGDataCPointsMarker(amg_data)      = C_points_marker;
   hypre_ParAMGDataCPointsLocalMarker(amg_data) = C_points_local_marker;
   hypre_ParAMGDataNumCPoints(amg_data)         = num_cpt_coarse;
   hypre_ParAMGDataCPointsLevel(amg_data)       = cpt_level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFPoints(void         *data,
                          HYPRE_Int     isolated,
                          HYPRE_Int     num_points,
                          HYPRE_BigInt *indices)
{
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) data;
   HYPRE_BigInt       *marker = NULL;
   HYPRE_Int           i;

   if (!amg_data)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "AMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (num_points < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! negative number of points!\n");
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }


   if ((num_points > 0) && (!indices))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! indices not given!\n");
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   /* Set marker data */
   if (num_points > 0)
   {
      marker = hypre_CTAlloc(HYPRE_BigInt, num_points, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_points; i++)
      {
         marker[i] = indices[i];
      }
   }

   if (isolated)
   {
      /* Free data not previously destroyed */
      if (hypre_ParAMGDataIsolatedFPointsMarker(amg_data))
      {
         hypre_TFree(hypre_ParAMGDataIsolatedFPointsMarker(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataIsolatedFPointsMarker(amg_data) = NULL;
      }

      hypre_ParAMGDataNumIsolatedFPoints(amg_data)    = num_points;
      hypre_ParAMGDataIsolatedFPointsMarker(amg_data) = marker;
   }
   else
   {
      /* Free data not previously destroyed */
      if (hypre_ParAMGDataFPointsMarker(amg_data))
      {
         hypre_TFree(hypre_ParAMGDataFPointsMarker(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataFPointsMarker(amg_data) = NULL;
      }

      hypre_ParAMGDataNumFPoints(amg_data)    = num_points;
      hypre_ParAMGDataFPointsMarker(amg_data) = marker;
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetCumNnzAP( void       *data,
                            HYPRE_Real  cum_nnz_AP )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataCumNnzAP(amg_data) = cum_nnz_AP;

   return hypre_error_flag;
}


HYPRE_Int
hypre_BoomerAMGGetCumNnzAP( void       *data,
                            HYPRE_Real *cum_nnz_AP )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *cum_nnz_AP = hypre_ParAMGDataCumNnzAP(amg_data);

   return hypre_error_flag;
}
