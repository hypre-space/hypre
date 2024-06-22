/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "../parcsr_block_mv/par_csr_block_matrix.h"
#include "_hypre_utilities.hpp"

#define DEBUG 0
#define PRINT_CF 0
#define DEBUG_SAVE_ALL_OPS 0

/*****************************************************************************
 *
 * Routine for driving the setup phase of AMG
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetup( void               *amg_vdata,
                      hypre_ParCSRMatrix *A,
                      hypre_ParVector    *f,
                      hypre_ParVector    *u )
{
   MPI_Comm            comm = hypre_ParCSRMatrixComm(A);
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */
   HYPRE_Int            num_vectors;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   hypre_ParVector     *Vtemp = NULL;
   hypre_ParVector     *Rtemp = NULL;
   hypre_ParVector     *Ptemp = NULL;
   hypre_ParVector     *Ztemp = NULL;
   hypre_ParCSRMatrix **P_array;
   hypre_ParCSRMatrix **R_array;
   hypre_ParVector     *Residual_array;
   hypre_IntArray     **CF_marker_array;
   hypre_IntArray     **dof_func_array;
   hypre_IntArray      *dof_func;
   HYPRE_Int           *dof_func_data;
   HYPRE_Real          *relax_weight;
   HYPRE_Real          *omega;
   HYPRE_Real           schwarz_relax_wt = 1;
   HYPRE_Real           strong_threshold;
   HYPRE_Int            coarsen_cut_factor;
   HYPRE_Int            useSabs;
   HYPRE_Real           CR_strong_th;
   HYPRE_Real           max_row_sum;
   HYPRE_Real           trunc_factor, jacobi_trunc_threshold;
   HYPRE_Real           agg_trunc_factor, agg_P12_trunc_factor;
   HYPRE_Real           CR_rate;
   HYPRE_Int            relax_order;
   HYPRE_Int            max_levels;
   HYPRE_Int            amg_logging;
   HYPRE_Int            amg_print_level;
   HYPRE_Int            debug_flag;
   HYPRE_Int            dbg_flg;
   HYPRE_Int            local_num_vars;
   HYPRE_Int            P_max_elmts;
   HYPRE_Int            agg_P_max_elmts;
   HYPRE_Int            agg_P12_max_elmts;
   HYPRE_Int            IS_type;
   HYPRE_Int            num_CR_relax_steps;
   HYPRE_Int            CR_use_CG;
   HYPRE_Int            cgc_its; /* BM Aug 25, 2006 */
   HYPRE_Int            mult_additive = hypre_ParAMGDataMultAdditive(amg_data);
   HYPRE_Int            additive = hypre_ParAMGDataAdditive(amg_data);
   HYPRE_Int            simple = hypre_ParAMGDataSimple(amg_data);
   HYPRE_Int            add_last_lvl = hypre_ParAMGDataAddLastLvl(amg_data);
   HYPRE_Int            add_P_max_elmts = hypre_ParAMGDataMultAddPMaxElmts(amg_data);
   HYPRE_Int            keep_same_sign = hypre_ParAMGDataKeepSameSign(amg_data);
   HYPRE_Real           add_trunc_factor = hypre_ParAMGDataMultAddTruncFactor(amg_data);
   HYPRE_Int            add_rlx = hypre_ParAMGDataAddRelaxType(amg_data);
   HYPRE_Real           add_rlx_wt = hypre_ParAMGDataAddRelaxWt(amg_data);

   hypre_ParCSRBlockMatrix **A_block_array, **P_block_array, **R_block_array;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_ParAMGDataMemoryLocation(amg_data) = memory_location;
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memory_location);
#endif

   /* Local variables */
   HYPRE_Int           *CF_marker;
   hypre_IntArray      *CFN_marker = NULL;
   hypre_IntArray      *CF2_marker = NULL;
   hypre_IntArray      *CF3_marker = NULL;
   hypre_ParCSRMatrix  *S = NULL, *Sabs = NULL;
   hypre_ParCSRMatrix  *S2;
   hypre_ParCSRMatrix  *SN = NULL;
   hypre_ParCSRMatrix  *SCR;
   hypre_ParCSRMatrix  *P = NULL;
   hypre_ParCSRMatrix  *R = NULL;
   hypre_ParCSRMatrix  *A_H;
   hypre_ParCSRMatrix  *AN = NULL;
   hypre_ParCSRMatrix  *P1;
   hypre_ParCSRMatrix  *P2;
   hypre_ParCSRMatrix  *Pnew = NULL;
   HYPRE_Real          *SmoothVecs = NULL;
   hypre_Vector       **l1_norms = NULL;
   hypre_Vector       **cheby_ds = NULL;
   HYPRE_Real         **cheby_coefs = NULL;

   HYPRE_Int       old_num_levels, num_levels;
   HYPRE_Int       level;
   HYPRE_Int       local_size, i, row;
   HYPRE_BigInt    first_local_row;
   HYPRE_BigInt    coarse_size;
   HYPRE_Int       coarsen_type;
   HYPRE_Int       measure_type;
   HYPRE_Int       setup_type;
   HYPRE_BigInt    fine_size;
   HYPRE_Int       offset;
   HYPRE_Int       not_finished_coarsening = 1;
   HYPRE_Int       coarse_threshold = hypre_ParAMGDataMaxCoarseSize(amg_data);
   HYPRE_Int       min_coarse_size = hypre_ParAMGDataMinCoarseSize(amg_data);
   HYPRE_Int       seq_threshold = hypre_ParAMGDataSeqThreshold(amg_data);
   HYPRE_Int       j, k;
   HYPRE_Int       num_procs, my_id;
#if !defined(HYPRE_USING_GPU)
   HYPRE_Int       num_threads = hypre_NumThreads();
#endif
   HYPRE_Int      *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   HYPRE_Int       num_functions = hypre_ParAMGDataNumFunctions(amg_data);
   HYPRE_Int       nodal = hypre_ParAMGDataNodal(amg_data);
   HYPRE_Int       nodal_levels = hypre_ParAMGDataNodalLevels(amg_data);
   HYPRE_Int       nodal_diag = hypre_ParAMGDataNodalDiag(amg_data);
   HYPRE_Int       num_paths = hypre_ParAMGDataNumPaths(amg_data);
   HYPRE_Int       agg_num_levels = hypre_ParAMGDataAggNumLevels(amg_data);
   HYPRE_Int       agg_interp_type = hypre_ParAMGDataAggInterpType(amg_data);
   HYPRE_Int       sep_weight = hypre_ParAMGDataSepWeight(amg_data);
   hypre_IntArray *coarse_dof_func = NULL;
   HYPRE_BigInt    coarse_pnts_global[2];
   HYPRE_BigInt    coarse_pnts_global1[2];
   HYPRE_Int       num_cg_sweeps;

   HYPRE_Real *max_eig_est = NULL;
   HYPRE_Real *min_eig_est = NULL;

   HYPRE_Solver *smoother = hypre_ParAMGDataSmoother(amg_data);
   HYPRE_Int     smooth_type = hypre_ParAMGDataSmoothType(amg_data);
   HYPRE_Int     smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   HYPRE_Int     sym;
   HYPRE_Int     nlevel;
   HYPRE_Real    thresh;
   HYPRE_Real    filter;
   HYPRE_Real    drop_tol;
   HYPRE_Int     max_nz_per_row;
   char         *euclidfile;
   HYPRE_Int     eu_level;
   HYPRE_Int     eu_bj;
   HYPRE_Real    eu_sparse_A;
   HYPRE_Int     ilu_type;
   HYPRE_Int     ilu_lfil;
   HYPRE_Int     ilu_max_row_nnz;
   HYPRE_Int     ilu_max_iter;
   HYPRE_Int     ilu_tri_solve;
   HYPRE_Int     ilu_lower_jacobi_iters;
   HYPRE_Int     ilu_upper_jacobi_iters;
   HYPRE_Real    ilu_droptol;
   HYPRE_Int     ilu_reordering_type;
   HYPRE_Int     fsai_algo_type;
   HYPRE_Int     fsai_local_solve_type;
   HYPRE_Int     fsai_max_steps;
   HYPRE_Int     fsai_max_step_size;
   HYPRE_Int     fsai_max_nnz_row;
   HYPRE_Int     fsai_num_levels;
   HYPRE_Real    fsai_threshold;
   HYPRE_Int     fsai_eig_max_iters;
   HYPRE_Real    fsai_kap_tolerance;
   HYPRE_Int     needZ = 0;

   HYPRE_Int interp_type, restri_type;
   HYPRE_Int post_interp_type;  /* what to do after computing the interpolation matrix
                                   0 for nothing, 1 for a Jacobi step */

   /*for fittting interp vectors */
   /*HYPRE_Int                smooth_interp_vectors= hypre_ParAMGSmoothInterpVectors(amg_data); */
   HYPRE_Real         abs_q_trunc = hypre_ParAMGInterpVecAbsQTrunc(amg_data);
   HYPRE_Int                q_max = hypre_ParAMGInterpVecQMax(amg_data);
   HYPRE_Int                num_interp_vectors = hypre_ParAMGNumInterpVectors(amg_data);
   HYPRE_Int                num_levels_interp_vectors = hypre_ParAMGNumLevelsInterpVectors(amg_data);
   hypre_ParVector  **interp_vectors = hypre_ParAMGInterpVectors(amg_data);
   hypre_ParVector ***interp_vectors_array = hypre_ParAMGInterpVectorsArray(amg_data);
   HYPRE_Int                interp_vec_variant = hypre_ParAMGInterpVecVariant(amg_data);
   HYPRE_Int                interp_refine = hypre_ParAMGInterpRefine(amg_data);
   HYPRE_Int                interp_vec_first_level = hypre_ParAMGInterpVecFirstLevel(amg_data);
   HYPRE_Real        *expandp_weights =  hypre_ParAMGDataExpandPWeights(amg_data);

   /* parameters for non-Galerkin stuff */
   HYPRE_Int nongalerk_num_tol = hypre_ParAMGDataNonGalerkNumTol (amg_data);
   HYPRE_Real *nongalerk_tol = hypre_ParAMGDataNonGalerkTol (amg_data);
   HYPRE_Real nongalerk_tol_l = 0.0;
   HYPRE_Real *nongal_tol_array = hypre_ParAMGDataNonGalTolArray (amg_data);

   hypre_ParCSRBlockMatrix *A_H_block;

   HYPRE_Int       block_mode = 0;

   HYPRE_Int       mult_addlvl = hypre_max(mult_additive, simple);
   HYPRE_Int       addlvl = hypre_max(mult_addlvl, additive);
   HYPRE_Int       rap2 = hypre_ParAMGDataRAP2(amg_data);
   HYPRE_Int       keepTranspose = hypre_ParAMGDataKeepTranspose(amg_data);

   HYPRE_Int       local_coarse_size;
   HYPRE_Int       num_C_points_coarse      = hypre_ParAMGDataNumCPoints(amg_data);
   HYPRE_Int      *C_points_local_marker    = hypre_ParAMGDataCPointsLocalMarker(amg_data);
   HYPRE_BigInt   *C_points_marker          = hypre_ParAMGDataCPointsMarker(amg_data);
   HYPRE_Int       num_F_points             = hypre_ParAMGDataNumFPoints(amg_data);
   HYPRE_BigInt   *F_points_marker          = hypre_ParAMGDataFPointsMarker(amg_data);
   HYPRE_Int       num_isolated_F_points    = hypre_ParAMGDataNumIsolatedFPoints(amg_data);
   HYPRE_BigInt   *isolated_F_points_marker = hypre_ParAMGDataIsolatedFPointsMarker(amg_data);

   HYPRE_Int      *num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);
   HYPRE_Int       ns = num_grid_sweeps[1];
   HYPRE_Real      wall_time = 0.0;   /* for debugging instrumentation */
   HYPRE_Int       add_end;

#ifdef HYPRE_USING_DSUPERLU
   HYPRE_Int       dslu_threshold = hypre_ParAMGDataDSLUThreshold(amg_data);
#endif

   char            nvtx_name[1024];

   HYPRE_Real cum_nnz_AP = hypre_ParAMGDataCumNnzAP(amg_data);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /*A_new = hypre_CSRMatrixDeleteZeros(hypre_ParCSRMatrixDiag(A), 1.e-16);
   hypre_CSRMatrixPrint(A_new, "Atestnew"); */
   old_num_levels = hypre_ParAMGDataNumLevels(amg_data);
   max_levels = hypre_ParAMGDataMaxLevels(amg_data);
   add_end = hypre_min(add_last_lvl, max_levels - 1);
   if (add_end == -1) { add_end = max_levels - 1; }
   amg_logging = hypre_ParAMGDataLogging(amg_data);
   amg_print_level = hypre_ParAMGDataPrintLevel(amg_data);
   coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);
   measure_type = hypre_ParAMGDataMeasureType(amg_data);
   setup_type = hypre_ParAMGDataSetupType(amg_data);
   debug_flag = hypre_ParAMGDataDebugFlag(amg_data);
   relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);
   omega = hypre_ParAMGDataOmega(amg_data);
   sym = hypre_ParAMGDataSym(amg_data);
   nlevel = hypre_ParAMGDataLevel(amg_data);
   filter = hypre_ParAMGDataFilter(amg_data);
   thresh = hypre_ParAMGDataThreshold(amg_data);
   drop_tol = hypre_ParAMGDataDropTol(amg_data);
   max_nz_per_row = hypre_ParAMGDataMaxNzPerRow(amg_data);
   euclidfile = hypre_ParAMGDataEuclidFile(amg_data);
   eu_level = hypre_ParAMGDataEuLevel(amg_data);
   eu_sparse_A = hypre_ParAMGDataEuSparseA(amg_data);
   eu_bj = hypre_ParAMGDataEuBJ(amg_data);
   ilu_type = hypre_ParAMGDataILUType(amg_data);
   ilu_lfil = hypre_ParAMGDataILULevel(amg_data);
   ilu_max_row_nnz = hypre_ParAMGDataILUMaxRowNnz(amg_data);
   ilu_droptol = hypre_ParAMGDataILUDroptol(amg_data);
   ilu_tri_solve = hypre_ParAMGDataILUTriSolve(amg_data);
   ilu_lower_jacobi_iters = hypre_ParAMGDataILULowerJacobiIters(amg_data);
   ilu_upper_jacobi_iters = hypre_ParAMGDataILUUpperJacobiIters(amg_data);
   ilu_max_iter = hypre_ParAMGDataILUMaxIter(amg_data);
   ilu_reordering_type = hypre_ParAMGDataILULocalReordering(amg_data);
   fsai_algo_type = hypre_ParAMGDataFSAIAlgoType(amg_data);
   fsai_local_solve_type = hypre_ParAMGDataFSAILocalSolveType(amg_data);
   fsai_max_steps = hypre_ParAMGDataFSAIMaxSteps(amg_data);
   fsai_max_step_size = hypre_ParAMGDataFSAIMaxStepSize(amg_data);
   fsai_max_nnz_row = hypre_ParAMGDataFSAIMaxNnzRow(amg_data);
   fsai_num_levels = hypre_ParAMGDataFSAINumLevels(amg_data);
   fsai_threshold = hypre_ParAMGDataFSAIThreshold(amg_data);
   fsai_eig_max_iters = hypre_ParAMGDataFSAIEigMaxIters(amg_data);
   fsai_kap_tolerance = hypre_ParAMGDataFSAIKapTolerance(amg_data);
   interp_type = hypre_ParAMGDataInterpType(amg_data);
   restri_type = hypre_ParAMGDataRestriction(amg_data); /* RL */
   post_interp_type = hypre_ParAMGDataPostInterpType(amg_data);
   IS_type = hypre_ParAMGDataISType(amg_data);
   num_CR_relax_steps = hypre_ParAMGDataNumCRRelaxSteps(amg_data);
   CR_rate = hypre_ParAMGDataCRRate(amg_data);
   CR_use_CG = hypre_ParAMGDataCRUseCG(amg_data);
   cgc_its = hypre_ParAMGDataCGCIts(amg_data);

   relax_order = hypre_ParAMGDataRelaxOrder(amg_data);

   hypre_ParCSRMatrixSetNumNonzeros(A);
   hypre_ParCSRMatrixSetDNumNonzeros(A);
   hypre_ParAMGDataNumVariables(amg_data) = hypre_ParCSRMatrixNumRows(A);

   if (num_procs == 1) { seq_threshold = 0; }
   if (setup_type == 0) { return hypre_error_flag; }

   S = NULL;

   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   R_array = hypre_ParAMGDataRArray(amg_data);
   CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);
   dof_func_array = hypre_ParAMGDataDofFuncArray(amg_data);
   dof_func = hypre_ParAMGDataDofFunc(amg_data);
   local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   first_local_row = hypre_ParCSRMatrixFirstRowIndex(A);

   /* set size of dof_func hypre_IntArray if necessary */
   if (dof_func && hypre_IntArraySize(dof_func) < 0)
   {
      hypre_IntArraySize(dof_func) = local_size;
      hypre_IntArrayMemoryLocation(dof_func) = memory_location;
   }

   A_block_array = hypre_ParAMGDataABlockArray(amg_data);
   P_block_array = hypre_ParAMGDataPBlockArray(amg_data);
   R_block_array = hypre_ParAMGDataRBlockArray(amg_data);

   grid_relax_type[3] = hypre_ParAMGDataUserCoarseRelaxType(amg_data);

   /* Get the number of vector components when LHS/RHS are passed in */
   if ((f != NULL) && (u != NULL))
   {
      /* Verify that the number of vectors held by f and u match */
      if (hypre_ParVectorNumVectors(f) != hypre_ParVectorNumVectors(u))
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: num_vectors for RHS and LHS do not match!\n");
         return hypre_error_flag;
      }
      num_vectors = hypre_ParVectorNumVectors(f);
   }
   else
   {
      num_vectors = 1;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* change in definition of standard and multipass interpolation, by
      eliminating interp_type 9 and 5 and setting sep_weight instead
      when using separation of weights option */
   if (interp_type == 9)
   {
      interp_type = 8;
      sep_weight = 1;
   }
   else if (interp_type == 5)
   {
      interp_type = 4;
      sep_weight = 1;
   }

   /* Verify that if the user has selected the interp_vec_variant > 0
      (so GM or LN interpolation) then they have nodal coarsening
      selected also */
   if (interp_vec_variant > 0 && nodal < 1)
   {
      nodal = 1;
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "WARNING: Changing to node-based coarsening because LN of GM interpolation has been specified via HYPRE_BoomerAMGSetInterpVecVariant.\n");
   }

   /* Verify that settings are correct for solving systems */
   /* If the user has specified either a block interpolation or a block relaxation then
      we need to make sure the other has been chosen as well  - so we can be
      in "block mode" - storing only block matrices on the coarse levels*/
   /* Furthermore, if we are using systems and nodal = 0, then
      we will change nodal to 1 */
   /* probably should disable stuff like smooth number levels at some point */


   if (grid_relax_type[0] >= 20 && grid_relax_type[0] != 30 &&
       grid_relax_type[0] != 88 && grid_relax_type[0] != 89)
   {
      /* block relaxation chosen */
      if (!((interp_type >= 20 && interp_type != 100) || interp_type == 11 || interp_type == 10 ) )
      {
         hypre_ParAMGDataInterpType(amg_data) = 20;
         interp_type = hypre_ParAMGDataInterpType(amg_data) ;
      }

      for (i = 1; i < 3; i++)
      {
         if (grid_relax_type[i] < 20)
         {
            grid_relax_type[i] = 23;
         }
      }
      if (grid_relax_type[3] < 20)
      {
         grid_relax_type[3] = 29; /* GE */
      }

      block_mode = 1;
   }

   if ((interp_type >= 20 && interp_type != 100) || interp_type == 11 ||
       interp_type == 10 ) /* block interp choosen */
   {
      if (!(nodal))
      {
         hypre_ParAMGDataNodal(amg_data) = 1;
         nodal = hypre_ParAMGDataNodal(amg_data);
      }
      for (i = 0; i < 3; i++)
      {
         if (grid_relax_type[i] < 20)
         {
            grid_relax_type[i] = 23;
         }
      }

      if (grid_relax_type[3] < 20) { grid_relax_type[3] = 29; } /* GE */

      block_mode = 1;

   }

   hypre_ParAMGDataBlockMode(amg_data) = block_mode;


   /* end of systems checks */

   /* free up storage in case of new setup without previous destroy */

   if (A_array || A_block_array || P_array || P_block_array || CF_marker_array ||
       dof_func_array || R_array || R_block_array)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (A_array[j])
         {
            hypre_ParCSRMatrixDestroy(A_array[j]);
            A_array[j] = NULL;
         }

         if (A_block_array[j])
         {
            hypre_ParCSRBlockMatrixDestroy(A_block_array[j]);
            A_block_array[j] = NULL;
         }

         hypre_IntArrayDestroy(dof_func_array[j]);
         dof_func_array[j] = NULL;
      }

      for (j = 0; j < old_num_levels - 1; j++)
      {
         if (P_array[j])
         {
            hypre_ParCSRMatrixDestroy(P_array[j]);
            P_array[j] = NULL;
         }

         if (P_block_array[j])
         {
            hypre_ParCSRBlockMatrixDestroy(P_block_array[j]);
            P_block_array[j] = NULL;
         }
         /* RL */
         if (R_array[j])
         {
            hypre_ParCSRMatrixDestroy(R_array[j]);
            R_array[j] = NULL;
         }

         if (R_block_array[j])
         {
            hypre_ParCSRBlockMatrixDestroy(R_block_array[j]);
            R_block_array[j] = NULL;
         }
      }

      /* Special case use of CF_marker_array when old_num_levels == 1
         requires us to attempt this deallocation every time */
      hypre_IntArrayDestroy(CF_marker_array[0]);
      CF_marker_array[0] = NULL;

      for (j = 1; j < old_num_levels - 1; j++)
      {
         hypre_IntArrayDestroy(CF_marker_array[j]);
         CF_marker_array[j] = NULL;
      }
   }

   {
      MPI_Comm new_comm = hypre_ParAMGDataNewComm(amg_data);
      void *amg = hypre_ParAMGDataCoarseSolver(amg_data);
      if (hypre_ParAMGDataRtemp(amg_data))
      {
         hypre_ParVectorDestroy(hypre_ParAMGDataRtemp(amg_data));
         hypre_ParAMGDataRtemp(amg_data) = NULL;
      }
      if (hypre_ParAMGDataPtemp(amg_data))
      {
         hypre_ParVectorDestroy(hypre_ParAMGDataPtemp(amg_data));
         hypre_ParAMGDataPtemp(amg_data) = NULL;
      }
      if (hypre_ParAMGDataZtemp(amg_data))
      {
         hypre_ParVectorDestroy(hypre_ParAMGDataZtemp(amg_data));
         hypre_ParAMGDataZtemp(amg_data) = NULL;
      }

      if (hypre_ParAMGDataACoarse(amg_data))
      {
         hypre_ParCSRMatrixDestroy(hypre_ParAMGDataACoarse(amg_data));
         hypre_ParAMGDataACoarse(amg_data) = NULL;
      }

      if (hypre_ParAMGDataUCoarse(amg_data))
      {
         hypre_ParVectorDestroy(hypre_ParAMGDataUCoarse(amg_data));
         hypre_ParAMGDataUCoarse(amg_data) = NULL;
      }

      if (hypre_ParAMGDataFCoarse(amg_data))
      {
         hypre_ParVectorDestroy(hypre_ParAMGDataFCoarse(amg_data));
         hypre_ParAMGDataFCoarse(amg_data) = NULL;
      }

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
         hypre_MPI_Comm_free (&new_comm);
         hypre_ParAMGDataNewComm(amg_data) = hypre_MPI_COMM_NULL;
      }

      if (amg)
      {
         hypre_BoomerAMGDestroy (amg);
         hypre_ParAMGDataCoarseSolver(amg_data) = NULL;
      }

      hypre_TFree(hypre_ParAMGDataMaxEigEst(amg_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParAMGDataMinEigEst(amg_data), HYPRE_MEMORY_HOST);

      if (hypre_ParAMGDataChebyDS(amg_data))
      {
         for (i = 0; i < old_num_levels; i++)
         {
            hypre_SeqVectorDestroy(hypre_ParAMGDataChebyDS(amg_data)[i]);
         }
         hypre_TFree(hypre_ParAMGDataChebyDS(amg_data), HYPRE_MEMORY_HOST);
      }

      if (hypre_ParAMGDataChebyCoefs(amg_data))
      {
         for (i = 0; i < old_num_levels; i++)
         {
            hypre_TFree(hypre_ParAMGDataChebyCoefs(amg_data)[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(hypre_ParAMGDataChebyCoefs(amg_data), HYPRE_MEMORY_HOST);
      }

      if (hypre_ParAMGDataL1Norms(amg_data))
      {
         for (i = 0; i < old_num_levels; i++)
         {
            hypre_SeqVectorDestroy(hypre_ParAMGDataL1Norms(amg_data)[i]);
         }
         hypre_TFree(hypre_ParAMGDataL1Norms(amg_data), HYPRE_MEMORY_HOST);
      }
      if (smooth_num_levels && smoother)
      {
         if (smooth_num_levels > 1 &&
             smooth_num_levels > old_num_levels - 1)
         {
            smooth_num_levels = old_num_levels - 1;
         }
         if (hypre_ParAMGDataSmoothType(amg_data) == 7)
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               if (smoother[i])
               {
                  HYPRE_ParCSRPilutDestroy(smoother[i]);
                  smoother[i] = NULL;
               }
            }
         }
         else if (hypre_ParAMGDataSmoothType(amg_data) == 8)
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               if (smoother[i])
               {
                  HYPRE_ParCSRParaSailsDestroy(smoother[i]);
                  smoother[i] = NULL;
               }
            }
         }
         else if (hypre_ParAMGDataSmoothType(amg_data) == 9)
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               if (smoother[i])
               {
                  HYPRE_EuclidDestroy(smoother[i]);
                  smoother[i] = NULL;
               }
            }
         }
         else if (hypre_ParAMGDataSmoothType(amg_data) == 4)
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               if (smoother[i])
               {
                  HYPRE_FSAIDestroy(smoother[i]);
                  smoother[i] = NULL;
               }
            }
         }
         else if (hypre_ParAMGDataSmoothType(amg_data) == 5)
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               if (smoother[i])
               {
                  HYPRE_ILUDestroy(smoother[i]);
                  smoother[i] = NULL;
               }
            }
         }
         else if (hypre_ParAMGDataSmoothType(amg_data) == 6)
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               if (smoother[i])
               {
                  HYPRE_SchwarzDestroy(smoother[i]);
                  smoother[i] = NULL;
               }
            }
         }
         hypre_TFree(hypre_ParAMGDataSmoother(amg_data), HYPRE_MEMORY_HOST);
      }
      if ( hypre_ParAMGDataResidual(amg_data) )
      {
         hypre_ParVectorDestroy( hypre_ParAMGDataResidual(amg_data) );
         hypre_ParAMGDataResidual(amg_data) = NULL;
      }
   }

   if (A_array == NULL)
   {
      A_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels, HYPRE_MEMORY_HOST);
   }
   if (A_block_array == NULL)
   {
      A_block_array = hypre_CTAlloc(hypre_ParCSRBlockMatrix*, max_levels, HYPRE_MEMORY_HOST);
   }

   if (P_array == NULL && max_levels > 1)
   {
      P_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels - 1, HYPRE_MEMORY_HOST);
   }
   if (P_block_array == NULL && max_levels > 1)
   {
      P_block_array = hypre_CTAlloc(hypre_ParCSRBlockMatrix*, max_levels - 1, HYPRE_MEMORY_HOST);
   }

   /* RL: if retri_type != 0, R != P^T, allocate R matrices */
   if (restri_type)
   {
      if (R_array == NULL && max_levels > 1)
      {
         R_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels - 1, HYPRE_MEMORY_HOST);
      }
      if (R_block_array == NULL && max_levels > 1)
      {
         R_block_array = hypre_CTAlloc(hypre_ParCSRBlockMatrix*, max_levels - 1, HYPRE_MEMORY_HOST);
      }
   }

   if (CF_marker_array == NULL)
   {
      CF_marker_array = hypre_CTAlloc(hypre_IntArray*, max_levels, HYPRE_MEMORY_HOST);
   }

   if (num_C_points_coarse > 0)
   {
#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
#if defined(HYPRE_USING_SYCL)
         HYPRE_Int *new_end =
            hypreSycl_copy_if( C_points_marker,
                               C_points_marker + num_C_points_coarse,
                               C_points_marker,
                               C_points_local_marker,
                               in_range<HYPRE_BigInt>(first_local_row, first_local_row + local_size - 1) );
         HYPRE_ONEDPL_CALL( std::transform,
                            C_points_local_marker,
                            C_points_local_marker + num_C_points_coarse,
                            C_points_local_marker,
         [first_local_row = first_local_row] (const auto & x) {return x - first_local_row;} );
#else
         HYPRE_Int *new_end =
            HYPRE_THRUST_CALL( copy_if,
                               thrust::make_transform_iterator(C_points_marker,                       _1 - first_local_row),
                               thrust::make_transform_iterator(C_points_marker + num_C_points_coarse, _1 - first_local_row),
                               C_points_marker,
                               C_points_local_marker,
                               in_range<HYPRE_BigInt>(first_local_row, first_local_row + local_size - 1) );
#endif

         num_C_points_coarse = new_end - C_points_local_marker;
      }
      else
#endif /* defined(HYPRE_USING_GPU) */
      {
         k = 0;
         for (j = 0; j < num_C_points_coarse; j++)
         {
            row = (HYPRE_Int) (C_points_marker[j] - first_local_row);
            if ((row >= 0) && (row < local_size))
            {
               C_points_local_marker[k++] = row;
            }
         }
         num_C_points_coarse = k;
      }
   }

   if (dof_func_array == NULL)
   {
      dof_func_array = hypre_CTAlloc(hypre_IntArray*, max_levels, HYPRE_MEMORY_HOST);
   }

   if (num_functions > 1 && dof_func == NULL)
   {
      dof_func = hypre_IntArrayCreate(local_size);
      hypre_IntArrayInitialize_v2(dof_func, memory_location);

      offset = (HYPRE_Int) ( first_local_row % ((HYPRE_BigInt) num_functions) );

#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_BoomerAMGInitDofFuncDevice(hypre_IntArrayData(dof_func), local_size, offset, num_functions);
      }
      else
#endif /* defined(HYPRE_USING_GPU) */
      {
         for (i = 0; i < local_size; i++)
         {
            hypre_IntArrayData(dof_func)[i] = (i + offset) % num_functions;
         }
      }
   }

   A_array[0] = A;

   /* interp vectors setup */
   if (interp_vec_variant == 1)
   {
      num_levels_interp_vectors = interp_vec_first_level + 1;
      hypre_ParAMGNumLevelsInterpVectors(amg_data) = num_levels_interp_vectors;
   }
   if ( interp_vec_variant > 0 &&  num_interp_vectors > 0)
   {
      interp_vectors_array =  hypre_CTAlloc(hypre_ParVector**, num_levels_interp_vectors,
                                            HYPRE_MEMORY_HOST);
      interp_vectors_array[0] = interp_vectors;
      hypre_ParAMGInterpVectorsArray(amg_data) = interp_vectors_array;
   }

   if (block_mode)
   {
      A_block_array[0] = hypre_ParCSRBlockMatrixConvertFromParCSRMatrix(A_array[0],
                                                                        num_functions);
      hypre_ParCSRBlockMatrixSetNumNonzeros(A_block_array[0]);
      hypre_ParCSRBlockMatrixSetDNumNonzeros(A_block_array[0]);
   }

   dof_func_array[0] = dof_func;
   hypre_ParAMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_ParAMGDataNumCPoints(amg_data) = num_C_points_coarse;
   hypre_ParAMGDataDofFunc(amg_data) = dof_func;
   hypre_ParAMGDataDofFuncArray(amg_data) = dof_func_array;
   hypre_ParAMGDataAArray(amg_data) = A_array;
   hypre_ParAMGDataPArray(amg_data) = P_array;

   /* RL: if R != P^T */
   if (restri_type)
   {
      hypre_ParAMGDataRArray(amg_data) = R_array;
   }
   else
   {
      hypre_ParAMGDataRArray(amg_data) = P_array;
   }

   hypre_ParAMGDataABlockArray(amg_data) = A_block_array;
   hypre_ParAMGDataPBlockArray(amg_data) = P_block_array;

   /* RL: if R != P^T */
   if (restri_type)
   {
      hypre_ParAMGDataRBlockArray(amg_data) = R_block_array;
   }
   else
   {
      hypre_ParAMGDataRBlockArray(amg_data) = P_block_array;
   }

   Vtemp = hypre_ParAMGDataVtemp(amg_data);

   if (Vtemp != NULL)
   {
      hypre_ParVectorDestroy(Vtemp);
      Vtemp = NULL;
   }

   Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
   hypre_ParVectorNumVectors(Vtemp) = num_vectors;
   hypre_ParVectorInitialize_v2(Vtemp, memory_location);
   hypre_ParAMGDataVtemp(amg_data) = Vtemp;

   /* If we are doing Cheby relaxation, we also need up two more temp vectors.
    * If cheby_scale is false, only need one, otherwise need two */
   if ((smooth_num_levels > 0 && smooth_type > 9) || relax_weight[0] < 0 || omega[0] < 0 ||
       hypre_ParAMGDataSchwarzRlxWeight(amg_data) < 0 ||
       (grid_relax_type[0] == 16 || grid_relax_type[1] == 16 || grid_relax_type[2] == 16 ||
        grid_relax_type[3] == 16))
   {
      Ptemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                    hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                    hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorNumVectors(Ptemp) = num_vectors;
      hypre_ParVectorInitialize_v2(Ptemp, memory_location);
      hypre_ParAMGDataPtemp(amg_data) = Ptemp;

      /* If not doing chebyshev relaxation, or (doing chebyshev relaxation and scaling) */
      if (!(grid_relax_type[0] == 16 || grid_relax_type[1] == 16 || grid_relax_type[2] == 16 ||
            grid_relax_type[3] == 16) ||
          (hypre_ParAMGDataChebyScale(amg_data)))
      {
         Rtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                       hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                       hypre_ParCSRMatrixRowStarts(A_array[0]));
         hypre_ParVectorNumVectors(Rtemp) = num_vectors;
         hypre_ParVectorInitialize_v2(Rtemp, memory_location);
         hypre_ParAMGDataRtemp(amg_data) = Rtemp;
      }
   }

   /* See if we need the Ztemp vector */
   if ( (smooth_num_levels > 0 && smooth_type > 6) || relax_weight[0] < 0 || omega[0] < 0 ||
        hypre_ParAMGDataSchwarzRlxWeight(amg_data) < 0 )
   {
      needZ = hypre_max(needZ, 1);
   }

   if ( grid_relax_type[0] == 16 || grid_relax_type[1] == 16 || grid_relax_type[2] == 16 ||
        grid_relax_type[3] == 16 )
   {
      /* Chebyshev */
      needZ = hypre_max(needZ, 1);
   }

#if !defined(HYPRE_USING_GPU)
   /* GPU impl. needs Z */
   if (num_threads > 1)
#endif
   {
      /* we need the temp Z vector for relaxation 3 and 6 now if we are using threading */
      for (j = 0; j < 4; j++)
      {
         if (grid_relax_type[j] ==  3 || grid_relax_type[j] ==  4 || grid_relax_type[j] ==  6 ||
             grid_relax_type[j] ==  8 || grid_relax_type[j] == 13 || grid_relax_type[j] == 14 ||
             grid_relax_type[j] == 11 || grid_relax_type[j] == 12 || grid_relax_type[j] == 88 ||
             grid_relax_type[j] == 89)
         {
            needZ = hypre_max(needZ, 1);
            break;
         }
      }
   }

   if (needZ)
   {
      Ztemp = hypre_ParMultiVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                         hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                         hypre_ParCSRMatrixRowStarts(A_array[0]),
                                         num_vectors);
      hypre_ParVectorInitialize_v2(Ztemp, memory_location);
      hypre_ParAMGDataZtemp(amg_data) = Ztemp;
   }

   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);

   if (F_array != NULL || U_array != NULL)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (F_array[j] != NULL)
         {
            hypre_ParVectorDestroy(F_array[j]);
            F_array[j] = NULL;
         }
         if (U_array[j] != NULL)
         {
            hypre_ParVectorDestroy(U_array[j]);
            U_array[j] = NULL;
         }
      }
   }

   if (F_array == NULL)
   {
      F_array = hypre_CTAlloc(hypre_ParVector*, max_levels, HYPRE_MEMORY_HOST);
   }
   if (U_array == NULL)
   {
      U_array = hypre_CTAlloc(hypre_ParVector*, max_levels, HYPRE_MEMORY_HOST);
   }

   F_array[0] = f;
   U_array[0] = u;

   hypre_ParAMGDataFArray(amg_data) = F_array;
   hypre_ParAMGDataUArray(amg_data) = U_array;

   /*----------------------------------------------------------
    * Initialize hypre_ParAMGData
    *----------------------------------------------------------*/

   not_finished_coarsening = 1;
   level = 0;
   HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);

   hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
   hypre_GpuProfilingPushRange(nvtx_name);

   strong_threshold = hypre_ParAMGDataStrongThreshold(amg_data);
   coarsen_cut_factor = hypre_ParAMGDataCoarsenCutFactor(amg_data);
   useSabs = hypre_ParAMGDataSabs(amg_data);
   CR_strong_th = hypre_ParAMGDataCRStrongTh(amg_data);
   max_row_sum = hypre_ParAMGDataMaxRowSum(amg_data);
   trunc_factor = hypre_ParAMGDataTruncFactor(amg_data);
   agg_trunc_factor = hypre_ParAMGDataAggTruncFactor(amg_data);
   agg_P12_trunc_factor = hypre_ParAMGDataAggP12TruncFactor(amg_data);
   P_max_elmts = hypre_ParAMGDataPMaxElmts(amg_data);
   agg_P_max_elmts = hypre_ParAMGDataAggPMaxElmts(amg_data);
   agg_P12_max_elmts = hypre_ParAMGDataAggP12MaxElmts(amg_data);
   jacobi_trunc_threshold = hypre_ParAMGDataJacobiTruncThreshold(amg_data);
   smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   if (smooth_num_levels > level)
   {
      smoother = hypre_CTAlloc(HYPRE_Solver, smooth_num_levels, HYPRE_MEMORY_HOST);
      hypre_ParAMGDataSmoother(amg_data) = smoother;
   }

   /*-----------------------------------------------------
    *  Enter Coarsening Loop
    *-----------------------------------------------------*/

   while (not_finished_coarsening)
   {
      /* only do nodal coarsening on a fixed number of levels */
      if (level >= nodal_levels)
      {
         nodal = 0;
      }

      if (block_mode)
      {
         fine_size = hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[level]);
      }
      else
      {
         fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
      }

      if (level > 0)
      {

         if (block_mode)
         {
            F_array[level] =
               hypre_ParVectorCreateFromBlock(hypre_ParCSRBlockMatrixComm(A_block_array[level]),
                                              hypre_ParCSRMatrixGlobalNumRows(A_block_array[level]),
                                              hypre_ParCSRBlockMatrixRowStarts(A_block_array[level]),
                                              hypre_ParCSRBlockMatrixBlockSize(A_block_array[level]));
            hypre_ParVectorInitialize(F_array[level]);

            U_array[level] =
               hypre_ParVectorCreateFromBlock(hypre_ParCSRBlockMatrixComm(A_block_array[level]),
                                              hypre_ParCSRMatrixGlobalNumRows(A_block_array[level]),
                                              hypre_ParCSRBlockMatrixRowStarts(A_block_array[level]),
                                              hypre_ParCSRBlockMatrixBlockSize(A_block_array[level]));

            hypre_ParVectorInitialize(U_array[level]);
         }
         else
         {
            F_array[level] =
               hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                     hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                     hypre_ParCSRMatrixRowStarts(A_array[level]));
            hypre_ParVectorNumVectors(F_array[level]) = num_vectors;
            hypre_ParVectorInitialize_v2(F_array[level], memory_location);

            U_array[level] =
               hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                     hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                     hypre_ParCSRMatrixRowStarts(A_array[level]));
            hypre_ParVectorNumVectors(U_array[level]) = num_vectors;
            hypre_ParVectorInitialize_v2(U_array[level], memory_location);
         }
      }

      /*-------------------------------------------------------------
       * Select coarse-grid points on 'level' : returns CF_marker
       * for the level.  Returns strength matrix, S
       *--------------------------------------------------------------*/

      dof_func_data = NULL;
      if (dof_func_array[level] != NULL)
      {
         dof_func_data = hypre_IntArrayData(dof_func_array[level]);
      }

      if (debug_flag == 1) { wall_time = time_getWallclockSeconds(); }
      if (debug_flag == 3)
      {
         hypre_printf("\n ===== Proc = %d     Level = %d  =====\n",
                      my_id, level);
         fflush(NULL);
      }

      if (max_levels == 1)
      {
         S = NULL;
         CF_marker_array[level] = hypre_IntArrayCreate(local_size);
         hypre_IntArrayInitialize(CF_marker_array[level]);
         hypre_IntArraySetConstantValues(CF_marker_array[level], 1);
         coarse_size = fine_size;
      }
      else /* max_levels > 1 */
      {
         if (block_mode)
         {
            local_num_vars =
               hypre_CSRBlockMatrixNumRows(hypre_ParCSRBlockMatrixDiag(A_block_array[level]));
         }
         else
         {
            local_num_vars =
               hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
         }
         if (hypre_ParAMGDataGSMG(amg_data) ||
             hypre_ParAMGDataInterpType(amg_data) == 1)
         {
            hypre_BoomerAMGCreateSmoothVecs(amg_data, A_array[level],
                                            hypre_ParAMGDataNumGridSweeps(amg_data)[1],
                                            level, &SmoothVecs);
         }

         /**** Get the Strength Matrix ****/
         if (hypre_ParAMGDataGSMG(amg_data) == 0)
         {
            if (nodal) /* if we are solving systems and
                          not using the unknown approach then we need to
                          convert A to a nodal matrix - values that represent the
                          blocks  - before getting the strength matrix*/
            {

               if (block_mode)
               {
                  hypre_BoomerAMGBlockCreateNodalA(A_block_array[level], hypre_abs(nodal), nodal_diag, &AN);
               }
               else
               {
                  hypre_BoomerAMGCreateNodalA(A_array[level], num_functions,
                                              dof_func_data, hypre_abs(nodal), nodal_diag, &AN);
               }

               /* dof array not needed for creating S because we pass in that
                  the number of functions is 1 */
               /* creat s two different ways - depending on if any entries in AN are negative: */

               /* first: positive and negative entries */
               if (nodal == 3 || nodal == 6 || nodal_diag > 0)
               {
                  hypre_BoomerAMGCreateS(AN, strong_threshold, max_row_sum,
                                         1, NULL, &SN);
               }
               else /* all entries are positive */
               {
                  hypre_BoomerAMGCreateSabs(AN, strong_threshold, max_row_sum,
                                            1, NULL, &SN);
               }
            }
            else /* standard AMG or unknown approach */
            {
               if (!useSabs)
               {
                  hypre_BoomerAMGCreateS(A_array[level], strong_threshold, max_row_sum,
                                         num_functions, dof_func_data, &S);
               }
               else
               {
                  /*
                  hypre_BoomerAMGCreateSabs(A_array[level], strong_threshold, max_row_sum,
                                            num_functions, dof_func_array[level], &S);
                                            */
                  hypre_BoomerAMGCreateSabs(A_array[level], strong_threshold, 1.0,
                                            1, NULL, &S);
               }
            }

            /* for AIR, need absolute value SOC: use a different threshold */
            if (restri_type == 1 || restri_type == 2 || restri_type == 15)
            {
               HYPRE_Real           strong_thresholdR;
               strong_thresholdR = hypre_ParAMGDataStrongThresholdR(amg_data);
               hypre_BoomerAMGCreateSabs(A_array[level], strong_thresholdR, 1.0,
                                         1, NULL, &Sabs);
            }
         }
         else
         {
            hypre_BoomerAMGCreateSmoothDirs(amg_data, A_array[level],
                                            SmoothVecs, strong_threshold,
                                            num_functions, dof_func_data, &S);
         }

         /* Allocate CF_marker for the current level */
         CF_marker_array[level] = hypre_IntArrayCreate(local_num_vars);
         hypre_IntArrayInitialize(CF_marker_array[level]);
         CF_marker = hypre_IntArrayData(CF_marker_array[level]);

         /* Set isolated fine points (SF_PT) given by the user */
         if ((num_isolated_F_points > 0) && (level == 0))
         {
            if (block_mode)
            {
               first_local_row = hypre_ParCSRBlockMatrixFirstRowIndex(A_block_array[level]);
            }
            else
            {
               first_local_row = hypre_ParCSRMatrixFirstRowIndex(A_array[level]);
            }

#if defined(HYPRE_USING_GPU)
            HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IntArrayMemoryLocation(
                                                                  CF_marker_array[level]) );

            if (exec == HYPRE_EXEC_DEVICE)
            {
#if defined(HYPRE_USING_SYCL)
               auto perm_it = oneapi::dpl::make_permutation_iterator(
                                 hypre_IntArrayData(CF_marker_array[level]),
                                 oneapi::dpl::make_transform_iterator( isolated_F_points_marker,
               [first_local_row = first_local_row] (const auto & x) {return x - first_local_row;} ) );
               hypreSycl_transform_if( perm_it,
                                       perm_it + num_isolated_F_points,
                                       isolated_F_points_marker,
                                       perm_it,
               [] (const auto & x) {return -3;},
               in_range<HYPRE_BigInt>(first_local_row, first_local_row + local_size - 1) );
#else
               HYPRE_THRUST_CALL( scatter_if,
                                  thrust::make_constant_iterator(-3),
                                  thrust::make_constant_iterator(-3) + num_isolated_F_points,
                                  thrust::make_transform_iterator(isolated_F_points_marker, _1 - first_local_row),
                                  isolated_F_points_marker,
                                  hypre_IntArrayData(CF_marker_array[level]),
                                  in_range<HYPRE_BigInt>(first_local_row, first_local_row + local_size - 1) );
#endif
            }
            else
#endif
            {
               for (j = 0; j < num_isolated_F_points; j++)
               {
                  row = (HYPRE_Int) (isolated_F_points_marker[j] - first_local_row);
                  if ((row >= 0) && (row < local_size))
                  {
                     hypre_IntArrayData(CF_marker_array[level])[row] = -3; // Assumes SF_PT == -3
                  }
               }
            }
         }

         /**** Do the appropriate coarsening ****/
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Coarsening");

         if (nodal == 0) /* no nodal coarsening */
         {
            if (coarsen_type == 6)
               hypre_BoomerAMGCoarsenFalgout(S, A_array[level], measure_type,
                                             coarsen_cut_factor, debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type == 7)
               hypre_BoomerAMGCoarsen(S, A_array[level], 2,
                                      debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type == 8)
               hypre_BoomerAMGCoarsenPMIS(S, A_array[level], 0,
                                          debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type == 9)
               hypre_BoomerAMGCoarsenPMIS(S, A_array[level], 2,
                                          debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type == 10)
               hypre_BoomerAMGCoarsenHMIS(S, A_array[level], measure_type,
                                          coarsen_cut_factor, debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type == 21 || coarsen_type == 22)
            {
#ifdef HYPRE_MIXEDINT
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "CGC coarsening is not available in mixedint mode!");
               return hypre_error_flag;
#endif
               hypre_BoomerAMGCoarsenCGCb(S, A_array[level], measure_type, coarsen_type,
                                          cgc_its, debug_flag, &(CF_marker_array[level]));
            }
            else if (coarsen_type == 98)
               hypre_BoomerAMGCoarsenCR1(A_array[level], &(CF_marker_array[level]),
                                         &coarse_size, num_CR_relax_steps, IS_type, 0);
            else if (coarsen_type == 99)
            {
               hypre_BoomerAMGCreateS(A_array[level],
                                      CR_strong_th, 1,
                                      num_functions, dof_func_data, &SCR);
               hypre_BoomerAMGCoarsenCR(A_array[level], &(CF_marker_array[level]),
                                        &coarse_size,
                                        num_CR_relax_steps, IS_type, 1, grid_relax_type[0],
                                        relax_weight[level], omega[level], CR_rate,
                                        NULL, NULL, CR_use_CG, SCR);
               hypre_ParCSRMatrixDestroy(SCR);
            }
            else if (coarsen_type)
            {
               hypre_BoomerAMGCoarsenRuge(S, A_array[level], measure_type, coarsen_type,
                                          coarsen_cut_factor, debug_flag, &(CF_marker_array[level]));
               /* DEBUG: SAVE CF the splitting
               HYPRE_Int my_id;
               MPI_Comm comm = hypre_ParCSRMatrixComm(A_array[level]);
               hypre_MPI_Comm_rank(comm, &my_id);
               char CFfile[256];
               hypre_sprintf(CFfile, "hypreCF_%d.txt.%d", level, my_id);
               FILE *fp = fopen(CFfile, "w");
               for (i=0; i<local_size; i++)
               {
                  HYPRE_Int k = CF_marker[i];
                  HYPRE_Real j;
                  if (k == 1) {
                    j = 1.0;
                  } else if (k == -1) {
                    j = 0.0;
                  } else {
                    if (k < 0) {
                      CF_marker[i] = -1;
                    }
                    j = (HYPRE_Real) k;
                  }
                  hypre_fprintf(fp, "%.18e\n", j);
               }
               fclose(fp);
               */
            }
            else
            {
               hypre_BoomerAMGCoarsen(S, A_array[level], 0,
                                      debug_flag, &(CF_marker_array[level]));
            }

            if (level < agg_num_levels)
            {
               hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                          1, dof_func_array[level], CF_marker_array[level],
                                          &coarse_dof_func, coarse_pnts_global1);
               hypre_BoomerAMGCreate2ndS(S, CF_marker, num_paths,
                                         coarse_pnts_global1, &S2);
               if (coarsen_type == 10)
               {
                  hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type + 3, coarsen_cut_factor,
                                             debug_flag, &CFN_marker);
               }
               else if (coarsen_type == 8)
               {
                  hypre_BoomerAMGCoarsenPMIS(S2, S2, 3,
                                             debug_flag, &CFN_marker);
               }
               else if (coarsen_type == 9)
               {
                  hypre_BoomerAMGCoarsenPMIS(S2, S2, 4,
                                             debug_flag, &CFN_marker);
               }
               else if (coarsen_type == 6)
               {
                  hypre_BoomerAMGCoarsenFalgout(S2, S2, measure_type, coarsen_cut_factor,
                                                debug_flag, &CFN_marker);
               }
               else if (coarsen_type == 21 || coarsen_type == 22)
               {
                  hypre_BoomerAMGCoarsenCGCb(S2, S2, measure_type,
                                             coarsen_type, cgc_its, debug_flag, &CFN_marker);
               }
               else if (coarsen_type == 7)
               {
                  hypre_BoomerAMGCoarsen(S2, S2, 2, debug_flag, &CFN_marker);
               }
               else if (coarsen_type)
               {
                  hypre_BoomerAMGCoarsenRuge(S2, S2, measure_type, coarsen_type,
                                             coarsen_cut_factor, debug_flag, &CFN_marker);
               }
               else
               {
                  hypre_BoomerAMGCoarsen(S2, S2, 0, debug_flag, &CFN_marker);
               }

               hypre_ParCSRMatrixDestroy(S2);
            }
         }
         else if (block_mode)
         {
            if (coarsen_type == 6)
               hypre_BoomerAMGCoarsenFalgout(SN, SN, measure_type, coarsen_cut_factor,
                                             debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type == 7)
               hypre_BoomerAMGCoarsen(SN, SN, 2,
                                      debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type == 8)
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 0,
                                          debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type == 9)
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 2,
                                          debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type == 10)
               hypre_BoomerAMGCoarsenHMIS(SN, SN, measure_type, coarsen_cut_factor,
                                          debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type == 21 || coarsen_type == 22)
               hypre_BoomerAMGCoarsenCGCb(SN, SN, measure_type,
                                          coarsen_type, cgc_its, debug_flag, &(CF_marker_array[level]));
            else if (coarsen_type)
               hypre_BoomerAMGCoarsenRuge(SN, SN, measure_type, coarsen_type,
                                          coarsen_cut_factor, debug_flag, &(CF_marker_array[level]));
            else
            {
               hypre_BoomerAMGCoarsen(SN, SN, 0, debug_flag, &(CF_marker_array[level]));
            }
         }
         else if (nodal > 0)
         {
            if (coarsen_type == 6)
               hypre_BoomerAMGCoarsenFalgout(SN, SN, measure_type, coarsen_cut_factor,
                                             debug_flag, &CFN_marker);
            else if (coarsen_type == 7)
            {
               hypre_BoomerAMGCoarsen(SN, SN, 2, debug_flag, &CFN_marker);
            }
            else if (coarsen_type == 8)
            {
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 0, debug_flag, &CFN_marker);
            }
            else if (coarsen_type == 9)
            {
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 2, debug_flag, &CFN_marker);
            }
            else if (coarsen_type == 10)
               hypre_BoomerAMGCoarsenHMIS(SN, SN, measure_type, coarsen_cut_factor,
                                          debug_flag, &CFN_marker);
            else if (coarsen_type == 21 || coarsen_type == 22)
               hypre_BoomerAMGCoarsenCGCb(SN, SN, measure_type,
                                          coarsen_type, cgc_its, debug_flag, &CFN_marker);
            else if (coarsen_type)
               hypre_BoomerAMGCoarsenRuge(SN, SN, measure_type, coarsen_type,
                                          coarsen_cut_factor, debug_flag, &CFN_marker);
            else
               hypre_BoomerAMGCoarsen(SN, SN, 0,
                                      debug_flag, &CFN_marker);
            if (level < agg_num_levels)
            {
               hypre_BoomerAMGCoarseParms(comm, local_num_vars / num_functions,
                                          1, dof_func_array[level], CFN_marker,
                                          &coarse_dof_func, coarse_pnts_global1);
               hypre_BoomerAMGCreate2ndS(SN, hypre_IntArrayData(CFN_marker), num_paths,
                                         coarse_pnts_global1, &S2);
               if (coarsen_type == 10)
               {
                  hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type + 3, coarsen_cut_factor,
                                             debug_flag, &CF2_marker);
               }
               else if (coarsen_type == 8)
               {
                  hypre_BoomerAMGCoarsenPMIS(S2, S2, 3,
                                             debug_flag, &CF2_marker);
               }
               else if (coarsen_type == 9)
               {
                  hypre_BoomerAMGCoarsenPMIS(S2, S2, 4,
                                             debug_flag, &CF2_marker);
               }
               else if (coarsen_type == 6)
               {
                  hypre_BoomerAMGCoarsenFalgout(S2, S2, measure_type, coarsen_cut_factor,
                                                debug_flag, &CF2_marker);
               }
               else if (coarsen_type == 21 || coarsen_type == 22)
               {
                  hypre_BoomerAMGCoarsenCGCb(S2, S2, measure_type,
                                             coarsen_type, cgc_its, debug_flag, &CF2_marker);
               }
               else if (coarsen_type == 7)
               {
                  hypre_BoomerAMGCoarsen(S2, S2, 2, debug_flag, &CF2_marker);
               }
               else if (coarsen_type)
               {
                  hypre_BoomerAMGCoarsenRuge(S2, S2, measure_type, coarsen_type,
                                             coarsen_cut_factor, debug_flag, &CF2_marker);
               }
               else
               {
                  hypre_BoomerAMGCoarsen(S2, S2, 0, debug_flag, &CF2_marker);
               }

               hypre_ParCSRMatrixDestroy(S2);
               S2 = NULL;
            }
            else
            {
               hypre_BoomerAMGCreateScalarCFS(SN, A_array[level], hypre_IntArrayData(CFN_marker),
                                              num_functions, nodal, keep_same_sign,
                                              &dof_func,  &(CF_marker_array[level]),
                                              &S);
               hypre_IntArrayDestroy(CFN_marker);
               CFN_marker = NULL;
               hypre_ParCSRMatrixDestroy(SN);
               SN = NULL;
               hypre_ParCSRMatrixDestroy(AN);
               AN = NULL;
            }
         }

         /**************************************************/
         /*********Set the fixed index to CF_marker*********/
         /* copy CF_marker to the host if needed */
         /* Set fine points (F_PT) given by the user */
         if ( (num_F_points > 0) && (level == 0) )
         {
#if defined(HYPRE_USING_GPU)
            HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IntArrayMemoryLocation(
                                                                  CF_marker_array[level]) );
            if (exec == HYPRE_EXEC_DEVICE)
            {
#if defined(HYPRE_USING_SYCL)
               auto perm_it = oneapi::dpl::make_permutation_iterator(
                                 hypre_IntArrayData(CF_marker_array[level]),
                                 oneapi::dpl::make_transform_iterator( F_points_marker,
               [first_local_row = first_local_row] (const auto & x) {return x - first_local_row;} ) );
               hypreSycl_transform_if( perm_it,
                                       perm_it + num_F_points,
                                       F_points_marker,
                                       perm_it,
               [] (const auto & x) {return -1;},
               in_range<HYPRE_BigInt>(first_local_row, first_local_row + local_size - 1) );
#else
               HYPRE_THRUST_CALL( scatter_if,
                                  thrust::make_constant_iterator(-1),
                                  thrust::make_constant_iterator(-1) + num_F_points,
                                  thrust::make_transform_iterator(F_points_marker, _1 - first_local_row),
                                  F_points_marker,
                                  hypre_IntArrayData(CF_marker_array[level]),
                                  in_range<HYPRE_BigInt>(first_local_row, first_local_row + local_size - 1) );
#endif
            }
            else
#endif
            {
               for (j = 0; j < num_F_points; j++)
               {
                  row = (HYPRE_Int) (F_points_marker[j] - first_local_row);
                  if ((row >= 0) && (row < local_size))
                  {
                     hypre_IntArrayData(CF_marker_array[level])[row] = -1; // Assumes F_PT == -1
                  }
               }
            }
         }


         if (num_C_points_coarse > 0)
         {
            if (block_mode)
            {
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Keeping coarse nodes in block mode is not implemented\n");
            }
            else if (level < hypre_ParAMGDataCPointsLevel(amg_data))
            {
#if defined(HYPRE_USING_GPU)
               HYPRE_MemoryLocation memory_location = hypre_IntArrayMemoryLocation(CF_marker_array[level]);
               HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memory_location);
               if (exec == HYPRE_EXEC_DEVICE)
               {
#if defined(HYPRE_USING_SYCL)
                  auto perm_it = oneapi::dpl::make_permutation_iterator(hypre_IntArrayData(CF_marker_array[level]),
                                                                        C_points_local_marker);
                  HYPRE_ONEDPL_CALL( std::transform,
                                     perm_it,
                                     perm_it + num_C_points_coarse,
                                     perm_it,
                  [] (const auto & x) {return 2;} );
#else
                  HYPRE_THRUST_CALL( scatter,
                                     thrust::make_constant_iterator(2),
                                     thrust::make_constant_iterator(2) + num_C_points_coarse,
                                     C_points_local_marker,
                                     hypre_IntArrayData(CF_marker_array[level]) );
#endif

                  if ( level + 1 < hypre_ParAMGDataCPointsLevel(amg_data) )
                  {
                     HYPRE_Int *tmp = hypre_TAlloc(HYPRE_Int, local_num_vars, memory_location);
#if defined(HYPRE_USING_SYCL)
                     HYPRE_ONEDPL_CALL( std::exclusive_scan,
                                        oneapi::dpl::make_transform_iterator(hypre_IntArrayData(CF_marker_array[level]),
                                                                             in_range<HYPRE_Int>(1, 2)),
                                        oneapi::dpl::make_transform_iterator(hypre_IntArrayData(CF_marker_array[level]) + local_num_vars,
                                                                             in_range<HYPRE_Int>(1, 2)),
                                        tmp,
                                        HYPRE_Int(0) );

                     /* RL: total local_coarse_size is not computed. I don't think it's needed */
                     hypreSycl_copy_if( tmp,
                                        tmp + local_num_vars,
                                        hypre_IntArrayData(CF_marker_array[level]),
                                        C_points_local_marker,
                                        equal<HYPRE_Int>(2) );
#else
                     HYPRE_THRUST_CALL( exclusive_scan,
                                        thrust::make_transform_iterator(hypre_IntArrayData(CF_marker_array[level]),
                                                                        in_range<HYPRE_Int>(1, 2)),
                                        thrust::make_transform_iterator(hypre_IntArrayData(CF_marker_array[level]) + local_num_vars,
                                                                        in_range<HYPRE_Int>(1, 2)),
                                        tmp,
                                        HYPRE_Int(0) );

                     /* RL: total local_coarse_size is not computed. I don't think it's needed */
                     HYPRE_THRUST_CALL( copy_if,
                                        tmp,
                                        tmp + local_num_vars,
                                        hypre_IntArrayData(CF_marker_array[level]),
                                        C_points_local_marker,
                                        equal<HYPRE_Int>(2) );
#endif

                     hypre_TFree(tmp, memory_location);
                  }

#if defined(HYPRE_USING_SYCL)
                  HYPRE_ONEDPL_CALL( std::replace,
                                     hypre_IntArrayData(CF_marker_array[level]),
                                     hypre_IntArrayData(CF_marker_array[level]) + local_num_vars,
                                     2,
                                     1 );
#else
                  HYPRE_THRUST_CALL( replace,
                                     hypre_IntArrayData(CF_marker_array[level]),
                                     hypre_IntArrayData(CF_marker_array[level]) + local_num_vars,
                                     2,
                                     1 );
#endif
               }
               else
#endif
               {
                  for (j = 0; j < num_C_points_coarse; j++)
                  {
                     hypre_IntArrayData(CF_marker_array[level])[C_points_local_marker[j]] = 2;
                  }

                  local_coarse_size = 0;
                  k = 0;
                  for (j = 0; j < local_num_vars; j ++)
                  {
                     if (hypre_IntArrayData(CF_marker_array[level])[j] == 1)
                     {
                        local_coarse_size++;
                     }
                     else if (hypre_IntArrayData(CF_marker_array[level])[j] == 2)
                     {
                        if ((level + 1) < hypre_ParAMGDataCPointsLevel(amg_data))
                        {
                           C_points_local_marker[k++] = local_coarse_size;
                        }
                        local_coarse_size++;
                        hypre_IntArrayData(CF_marker_array[level])[j] = 1;
                     }
                  }
                  // RL: so k is not used after this? update num_C_points_coarse?
               }
            }
         }

         /*****xxxxxxxxxxxxx changes for min_coarse_size */
         /* here we will determine the coarse grid size to be able to
            determine if it is not smaller than requested minimal size */

         hypre_GpuProfilingPushRange("CheckMinSize");

         if (level >= agg_num_levels)
         {
            if (block_mode)
            {
               hypre_BoomerAMGCoarseParms(comm,
                                          hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AN)),
                                          1, NULL, CF_marker_array[level], NULL, coarse_pnts_global);
            }
            else
            {
               hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                          num_functions, dof_func_array[level], CF_marker_array[level],
                                          &coarse_dof_func, coarse_pnts_global);
            }
            if (my_id == num_procs - 1)
            {
               coarse_size = coarse_pnts_global[1];
            }
            hypre_MPI_Bcast(&coarse_size, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

            /* if no coarse-grid, stop coarsening, and set the
             * coarsest solve to be a single sweep of default smoother or smoother set by user */
            if ((coarse_size == 0) || (coarse_size == fine_size))
            {
               HYPRE_Int *num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);
               HYPRE_Int **grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);
               if (grid_relax_type[3] ==  9 || grid_relax_type[3] == 99 ||
                   grid_relax_type[3] == 19 || grid_relax_type[3] == 98)
               {
                  grid_relax_type[3] = grid_relax_type[0];
                  num_grid_sweeps[3] = 1;
                  if (grid_relax_points) { grid_relax_points[3][0] = 0; }
               }
               if (S) { hypre_ParCSRMatrixDestroy(S); }
               if (SN) { hypre_ParCSRMatrixDestroy(SN); }
               if (AN) { hypre_ParCSRMatrixDestroy(AN); }
               //hypre_TFree(CF_marker, HYPRE_MEMORY_HOST);
               if (level > 0)
               {
                  /* note special case treatment of CF_marker is necessary
                   * to do CF relaxation correctly when num_levels = 1 */
                  hypre_IntArrayDestroy(CF_marker_array[level]);
                  CF_marker_array[level] = NULL;
                  hypre_ParVectorDestroy(F_array[level]);
                  hypre_ParVectorDestroy(U_array[level]);
               }
               coarse_size = fine_size;

               if (Sabs)
               {
                  hypre_ParCSRMatrixDestroy(Sabs);
                  Sabs = NULL;
               }

               if (coarse_dof_func)
               {
                  hypre_IntArrayDestroy(coarse_dof_func);
                  coarse_dof_func = NULL;
               }

               HYPRE_ANNOTATE_REGION_END("%s", "Coarsening");
               break;
            }

            if (coarse_size < min_coarse_size)
            {
               if (S) { hypre_ParCSRMatrixDestroy(S); }
               if (SN) { hypre_ParCSRMatrixDestroy(SN); }
               if (AN) { hypre_ParCSRMatrixDestroy(AN); }
               if (num_functions > 1)
               {
                  hypre_IntArrayDestroy(coarse_dof_func);
                  coarse_dof_func = NULL;
               }
               hypre_IntArrayDestroy(CF_marker_array[level]);
               CF_marker_array[level] = NULL;
               if (level > 0)
               {
                  hypre_ParVectorDestroy(F_array[level]);
                  hypre_ParVectorDestroy(U_array[level]);
               }
               coarse_size = fine_size;

               if (Sabs)
               {
                  hypre_ParCSRMatrixDestroy(Sabs);
                  Sabs = NULL;
               }

               HYPRE_ANNOTATE_REGION_END("%s", "Coarsening");
               break;
            }
         }

         hypre_GpuProfilingPopRange();

         /*****xxxxxxxxxxxxx changes for min_coarse_size  end */
         HYPRE_ANNOTATE_REGION_END("%s", "Coarsening");
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Interpolation");

         if (level < agg_num_levels)
         {
            if (nodal == 0)
            {
               if (agg_interp_type == 1)
               {
                  hypre_BoomerAMGBuildExtPIInterp(A_array[level],
                                                  CF_marker, S, coarse_pnts_global1,
                                                  num_functions, dof_func_data, debug_flag,
                                                  agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
               }
               else if (agg_interp_type == 2)
               {
                  hypre_BoomerAMGBuildStdInterp(A_array[level],
                                                CF_marker, S, coarse_pnts_global1,
                                                num_functions, dof_func_data, debug_flag,
                                                agg_P12_trunc_factor, agg_P12_max_elmts, 0, &P1);
               }
               else if (agg_interp_type == 3)
               {
                  hypre_BoomerAMGBuildExtInterp(A_array[level],
                                                CF_marker, S, coarse_pnts_global1,
                                                num_functions, dof_func_data, debug_flag,
                                                agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
               }
               else if (agg_interp_type == 5)
               {
                  hypre_BoomerAMGBuildModExtInterp(A_array[level],
                                                   CF_marker, S, coarse_pnts_global1,
                                                   num_functions, dof_func_data,
                                                   debug_flag,
                                                   agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
               }
               else if (agg_interp_type == 6)
               {
                  hypre_BoomerAMGBuildModExtPIInterp(A_array[level],
                                                     CF_marker, S, coarse_pnts_global1,
                                                     num_functions, dof_func_data,
                                                     debug_flag,
                                                     agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
               }
               else if (agg_interp_type == 7)
               {
                  hypre_BoomerAMGBuildModExtPEInterp(A_array[level],
                                                     CF_marker, S, coarse_pnts_global1,
                                                     num_functions, dof_func_data,
                                                     debug_flag,
                                                     agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
               }

               if (agg_interp_type == 4)
               {
                  hypre_BoomerAMGCorrectCFMarker(CF_marker_array[level], CFN_marker);
                  hypre_IntArrayDestroy(CFN_marker);
                  CFN_marker = NULL;
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             num_functions, dof_func_array[level], CF_marker_array[level],
                                             &coarse_dof_func, coarse_pnts_global);
                  hypre_BoomerAMGBuildMultipass(A_array[level],
                                                CF_marker, S, coarse_pnts_global,
                                                num_functions, dof_func_data, debug_flag,
                                                agg_trunc_factor, agg_P_max_elmts, sep_weight,
                                                &P);
               }
               else if (agg_interp_type == 8)
               {
                  hypre_BoomerAMGCorrectCFMarker(CF_marker_array[level], CFN_marker);
                  hypre_IntArrayDestroy(CFN_marker);
                  CFN_marker = NULL;
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             num_functions, dof_func_array[level], CF_marker_array[level],
                                             &coarse_dof_func, coarse_pnts_global);
                  hypre_BoomerAMGBuildModMultipass(A_array[level],
                                                   CF_marker, S, coarse_pnts_global,
                                                   agg_trunc_factor, agg_P_max_elmts, 8,
                                                   num_functions, dof_func_data, &P);
               }
               else if (agg_interp_type == 9)
               {
                  hypre_BoomerAMGCorrectCFMarker(CF_marker_array[level], CFN_marker);
                  hypre_IntArrayDestroy(CFN_marker);
                  CFN_marker = NULL;
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             num_functions, dof_func_array[level], CF_marker_array[level],
                                             &coarse_dof_func, coarse_pnts_global);
                  hypre_BoomerAMGBuildModMultipass(A_array[level],
                                                   CF_marker, S, coarse_pnts_global,
                                                   agg_trunc_factor, agg_P_max_elmts, 9,
                                                   num_functions, dof_func_data, &P);
               }
               else
               {
                  hypre_BoomerAMGCorrectCFMarker2 (CF_marker_array[level], (CFN_marker));
                  hypre_IntArrayDestroy(CFN_marker);
                  CFN_marker = NULL;
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             num_functions, dof_func_array[level], CF_marker_array[level],
                                             &coarse_dof_func, coarse_pnts_global);
                  if (agg_interp_type == 1 || agg_interp_type == 6 )
                  {
                     hypre_BoomerAMGBuildPartialExtPIInterp(A_array[level],
                                                            CF_marker, S, coarse_pnts_global,
                                                            coarse_pnts_global1, num_functions,
                                                            dof_func_data, debug_flag, agg_P12_trunc_factor,
                                                            agg_P12_max_elmts, &P2);
                  }
                  else if (agg_interp_type == 2)
                  {
                     hypre_BoomerAMGBuildPartialStdInterp(A_array[level],
                                                          CF_marker, S, coarse_pnts_global,
                                                          coarse_pnts_global1, num_functions,
                                                          dof_func_data, debug_flag, agg_P12_trunc_factor,
                                                          agg_P12_max_elmts, sep_weight, &P2);
                  }
                  else if (agg_interp_type == 3)
                  {
                     hypre_BoomerAMGBuildPartialExtInterp(A_array[level],
                                                          CF_marker, S, coarse_pnts_global,
                                                          coarse_pnts_global1, num_functions,
                                                          dof_func_data, debug_flag, agg_P12_trunc_factor,
                                                          agg_P12_max_elmts, &P2);
                  }
                  else if (agg_interp_type == 5)
                  {
                     hypre_BoomerAMGBuildModPartialExtInterp(A_array[level],
                                                             CF_marker, S, coarse_pnts_global,
                                                             coarse_pnts_global1,
                                                             num_functions, dof_func_data,
                                                             debug_flag,
                                                             agg_P12_trunc_factor, agg_P12_max_elmts, &P2);
                  }
                  else if (agg_interp_type == 7)
                  {
                     hypre_BoomerAMGBuildModPartialExtPEInterp(A_array[level],
                                                               CF_marker, S, coarse_pnts_global,
                                                               coarse_pnts_global1,
                                                               num_functions, dof_func_data,
                                                               debug_flag,
                                                               agg_P12_trunc_factor, agg_P12_max_elmts, &P2);
                  }

                  if (hypre_ParAMGDataModularizedMatMat(amg_data))
                  {
                     P = hypre_ParCSRMatMat(P1, P2);
                  }
                  else
                  {
                     P = hypre_ParMatmul(P1, P2);
                  }

                  hypre_BoomerAMGInterpTruncation(P, agg_trunc_factor, agg_P_max_elmts);

                  if (agg_trunc_factor != 0.0 || agg_P_max_elmts > 0 ||
                      agg_P12_trunc_factor != 0.0 || agg_P12_max_elmts > 0)
                  {
                     hypre_ParCSRMatrixCompressOffdMap(P);
                  }

                  hypre_MatvecCommPkgCreate(P);
                  hypre_ParCSRMatrixDestroy(P1);
                  hypre_ParCSRMatrixDestroy(P2);
               }
            }
            else if (nodal > 0)
            {
               if (agg_interp_type == 4)
               {
                  hypre_BoomerAMGCorrectCFMarker(CFN_marker, CF2_marker);
                  hypre_IntArrayDestroy(CF2_marker);
                  CF2_marker = NULL;

                  hypre_BoomerAMGCreateScalarCFS(SN, A_array[level], hypre_IntArrayData(CFN_marker),
                                                 num_functions, nodal, keep_same_sign,
                                                 &dof_func, &(CF_marker_array[level]), &S);
                  hypre_IntArrayDestroy(CFN_marker);
                  CFN_marker = NULL;
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             num_functions, dof_func_array[level], CF_marker_array[level],
                                             &coarse_dof_func, coarse_pnts_global);
                  hypre_BoomerAMGBuildMultipass(A_array[level],
                                                CF_marker, S, coarse_pnts_global,
                                                num_functions, dof_func_data, debug_flag,
                                                agg_trunc_factor, agg_P_max_elmts, sep_weight,
                                                &P);
               }
               else if (agg_interp_type == 8)
               {
                  hypre_BoomerAMGCorrectCFMarker(CFN_marker, CF2_marker);
                  hypre_IntArrayDestroy(CF2_marker);
                  CF2_marker = NULL;

                  hypre_BoomerAMGCreateScalarCFS(SN, A_array[level], hypre_IntArrayData(CFN_marker),
                                                 num_functions, nodal, keep_same_sign,
                                                 &dof_func, &(CF_marker_array[level]), &S);
                  hypre_IntArrayDestroy(CFN_marker);
                  CFN_marker = NULL;
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             num_functions, dof_func_array[level], CF_marker_array[level],
                                             &coarse_dof_func, coarse_pnts_global);
                  hypre_BoomerAMGBuildModMultipass(A_array[level],
                                                   CF_marker, S, coarse_pnts_global,
                                                   agg_trunc_factor, agg_P_max_elmts, 8,
                                                   num_functions, dof_func_data, &P);
               }
               else if (agg_interp_type == 9)
               {
                  hypre_BoomerAMGCorrectCFMarker(CFN_marker, CF2_marker);
                  hypre_IntArrayDestroy(CF2_marker);
                  CF2_marker = NULL;

                  hypre_BoomerAMGCreateScalarCFS(SN, A_array[level], hypre_IntArrayData(CFN_marker),
                                                 num_functions, nodal, keep_same_sign,
                                                 &dof_func, &(CF_marker_array[level]), &S);
                  hypre_IntArrayDestroy(CFN_marker);
                  CFN_marker = NULL;
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             num_functions, dof_func_array[level], CF_marker_array[level],
                                             &coarse_dof_func, coarse_pnts_global);
                  hypre_BoomerAMGBuildModMultipass(A_array[level],
                                                   CF_marker, S, coarse_pnts_global,
                                                   agg_trunc_factor, agg_P_max_elmts, 9,
                                                   num_functions, dof_func_data, &P);
               }
               else
               {
                  hypre_BoomerAMGCreateScalarCFS(SN, A_array[level], hypre_IntArrayData(CFN_marker),
                                                 num_functions, nodal, keep_same_sign,
                                                 &dof_func, &CF3_marker, &S);
                  for (i = 0; i < 2; i++)
                  {
                     coarse_pnts_global1[i] *= num_functions;
                  }
                  if (agg_interp_type == 1)
                  {
                     hypre_BoomerAMGBuildExtPIInterp(A_array[level],
                                                     hypre_IntArrayData(CF3_marker), S, coarse_pnts_global1,
                                                     num_functions, dof_func_data, debug_flag,
                                                     agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
                  }
                  else if (agg_interp_type == 2)
                  {
                     hypre_BoomerAMGBuildStdInterp(A_array[level],
                                                   hypre_IntArrayData(CF3_marker), S, coarse_pnts_global1,
                                                   num_functions, dof_func_data, debug_flag,
                                                   agg_P12_trunc_factor, agg_P12_max_elmts, 0, &P1);
                  }
                  else if (agg_interp_type == 3)
                  {
                     hypre_BoomerAMGBuildExtInterp(A_array[level],
                                                   hypre_IntArrayData(CF3_marker), S, coarse_pnts_global1,
                                                   num_functions, dof_func_data, debug_flag,
                                                   agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
                  }
                  else if (agg_interp_type == 5)
                  {
                     hypre_BoomerAMGBuildModExtInterp(A_array[level],
                                                      hypre_IntArrayData(CF3_marker), S, coarse_pnts_global1,
                                                      num_functions, dof_func_data,
                                                      debug_flag,
                                                      agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
                  }
                  else if (agg_interp_type == 6)
                  {
                     hypre_BoomerAMGBuildModExtPIInterp(A_array[level],
                                                        hypre_IntArrayData(CF3_marker), S, coarse_pnts_global1,
                                                        num_functions, dof_func_data,
                                                        debug_flag,
                                                        agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
                  }
                  else if (agg_interp_type == 7 )
                  {
                     hypre_BoomerAMGBuildModExtPEInterp(A_array[level],
                                                        hypre_IntArrayData(CF3_marker), S, coarse_pnts_global1,
                                                        num_functions, dof_func_data,
                                                        debug_flag,
                                                        agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
                  }

                  hypre_BoomerAMGCorrectCFMarker2 (CFN_marker, CF2_marker);
                  hypre_IntArrayDestroy(CF2_marker);
                  CF2_marker = NULL;
                  hypre_IntArrayDestroy(CF3_marker);
                  CF3_marker = NULL;
                  hypre_ParCSRMatrixDestroy(S);
                  hypre_BoomerAMGCreateScalarCFS(SN, A_array[level], hypre_IntArrayData(CFN_marker),
                                                 num_functions, nodal, keep_same_sign,
                                                 &dof_func, &(CF_marker_array[level]), &S);

                  hypre_IntArrayDestroy(CFN_marker);
                  CFN_marker = NULL;
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             num_functions, dof_func_array[level], CF_marker_array[level],
                                             &coarse_dof_func, coarse_pnts_global);
                  if (agg_interp_type == 1 || agg_interp_type == 6)
                  {
                     hypre_BoomerAMGBuildPartialExtPIInterp(A_array[level],
                                                            CF_marker, S, coarse_pnts_global,
                                                            coarse_pnts_global1, num_functions,
                                                            dof_func_data, debug_flag, agg_P12_trunc_factor,
                                                            agg_P12_max_elmts, &P2);
                  }
                  else if (agg_interp_type == 2)
                  {
                     hypre_BoomerAMGBuildPartialStdInterp(A_array[level],
                                                          CF_marker, S, coarse_pnts_global,
                                                          coarse_pnts_global1, num_functions,
                                                          dof_func_data, debug_flag, agg_P12_trunc_factor,
                                                          agg_P12_max_elmts, sep_weight, &P2);
                  }
                  else if (agg_interp_type == 3)
                  {
                     hypre_BoomerAMGBuildPartialExtInterp(A_array[level],
                                                          CF_marker, S, coarse_pnts_global,
                                                          coarse_pnts_global1, num_functions,
                                                          dof_func_data, debug_flag, agg_P12_trunc_factor,
                                                          agg_P12_max_elmts, &P2);
                  }
                  else if (agg_interp_type == 5)
                  {
                     hypre_BoomerAMGBuildModPartialExtInterp(A_array[level],
                                                             CF_marker, S, coarse_pnts_global, coarse_pnts_global1,
                                                             num_functions, dof_func_data,
                                                             debug_flag,
                                                             agg_P12_trunc_factor, agg_P12_max_elmts, &P2);
                  }
                  else if (agg_interp_type == 7)
                  {
                     hypre_BoomerAMGBuildModPartialExtPEInterp(A_array[level],
                                                               CF_marker, S, coarse_pnts_global, coarse_pnts_global1,
                                                               num_functions, dof_func_data,
                                                               debug_flag,
                                                               agg_P12_trunc_factor, agg_P12_max_elmts, &P2);
                  }

                  if (hypre_ParAMGDataModularizedMatMat(amg_data))
                  {
                     P = hypre_ParCSRMatMat(P1, P2);
                  }
                  else
                  {
                     P = hypre_ParMatmul(P1, P2);
                  }

                  hypre_BoomerAMGInterpTruncation(P, agg_trunc_factor,
                                                  agg_P_max_elmts);

                  if (agg_trunc_factor != 0.0 || agg_P_max_elmts > 0 ||
                      agg_P12_trunc_factor != 0.0 || agg_P12_max_elmts > 0)
                  {
                     hypre_ParCSRMatrixCompressOffdMap(P);
                  }

                  hypre_MatvecCommPkgCreate(P);
                  hypre_ParCSRMatrixDestroy(P1);
                  hypre_ParCSRMatrixDestroy(P2);
               }
               if (SN)
               {
                  hypre_ParCSRMatrixDestroy(SN);
               }
               SN = NULL;
               if (AN)
               {
                  hypre_ParCSRMatrixDestroy(AN);
               }
               AN = NULL;
            }
            if (my_id == (num_procs - 1))
            {
               coarse_size = coarse_pnts_global[1];
            }
            hypre_MPI_Bcast(&coarse_size, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
         }
         else /* no aggressive coarsening */
         {
            /**** Get the coarse parameters ****/
            /* xxxxxxxxxxxxxxxxxxxxxxxxx change for min_coarse_size
                        if (block_mode )
                        {
                           hypre_BoomerAMGCoarseParms(comm,
                                                      hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AN)),
                                                      1, NULL, CF_marker, NULL, coarse_pnts_global);
                        }
                        else
                        {
                           hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                                      num_functions, dof_func_array[level], CF_marker,
                                                      &coarse_dof_func, coarse_pnts_global);
                        }
                        if (my_id == (num_procs -1)) coarse_size = coarse_pnts_global[1];
                        hypre_MPI_Bcast(&coarse_size, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);
             xxxxxxxxxxxxxxxxxxxxxxxxx change for min_coarse_size */
            if (debug_flag == 1)
            {
               wall_time = time_getWallclockSeconds() - wall_time;
               hypre_printf("Proc = %d    Level = %d    Coarsen Time = %f\n",
                            my_id, level, wall_time);
               fflush(NULL);
            }

            /* RL: build restriction */
            if (restri_type)
            {
               HYPRE_Real filter_thresholdR;
               filter_thresholdR = hypre_ParAMGDataFilterThresholdR(amg_data);
               HYPRE_Int is_triangular = hypre_ParAMGDataIsTriangular(amg_data);
               HYPRE_Int gmres_switch = hypre_ParAMGDataGMRESSwitchR(amg_data);
               /* !!! RL: ensure that CF_marker contains -1 or 1 !!! */
#if defined(HYPRE_USING_GPU)
               HYPRE_MemoryLocation memory_location = hypre_IntArrayMemoryLocation(CF_marker_array[level]);
               HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memory_location);
               if (exec == HYPRE_EXEC_DEVICE)
               {
                  hypre_BoomerAMGCFMarkerTo1minus1Device(CF_marker,
                                                         hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level])));
               }
               else
#endif
               {
                  for (i = 0; i < hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level])); i++)
                  {
                     CF_marker[i] = CF_marker[i] > 0 ? 1 : -1;
                  }
               }

               if (restri_type == 1) /* distance-1 AIR */
               {
                  hypre_BoomerAMGBuildRestrAIR(A_array[level], CF_marker,
                                               Sabs, coarse_pnts_global, 1, NULL,
                                               filter_thresholdR, debug_flag,
                                               &R,
                                               is_triangular, gmres_switch );
               }
               else if (restri_type == 2 || restri_type == 15) /* distance-2, 1.5 AIR */
               {
                  hypre_BoomerAMGBuildRestrDist2AIR(A_array[level], CF_marker,
                                                    Sabs, coarse_pnts_global, 1, NULL,
                                                    filter_thresholdR, debug_flag,
                                                    &R, restri_type == 15,
                                                    is_triangular, gmres_switch);
               }
               else
               {
                  HYPRE_Int NeumannAIRDeg = restri_type - 3;
                  hypre_assert(NeumannAIRDeg >= 0);
                  HYPRE_Real strong_thresholdR;
                  strong_thresholdR = hypre_ParAMGDataStrongThresholdR(amg_data);
                  hypre_BoomerAMGBuildRestrNeumannAIR(A_array[level], CF_marker,
                                                      coarse_pnts_global, 1, NULL,
                                                      NeumannAIRDeg, strong_thresholdR,
                                                      filter_thresholdR, debug_flag,
                                                      &R );
               }

#if DEBUG_SAVE_ALL_OPS
               char file[256];
               hypre_sprintf(file, "R_%d.mtx", level);
               hypre_ParCSRMatrixPrintIJ(R, 1, 1, file);
#endif
               if (Sabs)
               {
                  hypre_ParCSRMatrixDestroy(Sabs);
                  Sabs = NULL;
               }
            }

            if (debug_flag == 1) { wall_time = time_getWallclockSeconds(); }

            if (interp_type == 4)
            {
               hypre_BoomerAMGBuildMultipass(A_array[level], CF_marker,
                                             S, coarse_pnts_global, num_functions, dof_func_data,
                                             debug_flag, trunc_factor, P_max_elmts, sep_weight, &P);
            }
            else if (interp_type == 1)
            {
               hypre_BoomerAMGNormalizeVecs(
                  hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level])),
                  hypre_ParAMGDataNumSamples(amg_data), SmoothVecs);

               hypre_BoomerAMGBuildInterpLS(NULL, CF_marker, S,
                                            coarse_pnts_global, num_functions, dof_func_data,
                                            debug_flag, trunc_factor,
                                            hypre_ParAMGDataNumSamples(amg_data), SmoothVecs, &P);
            }
            else if (interp_type == 2)
            {
               hypre_BoomerAMGBuildInterpHE(A_array[level], CF_marker,
                                            S, coarse_pnts_global, num_functions, dof_func_data,
                                            debug_flag, trunc_factor, P_max_elmts, &P);
            }
            else if (interp_type == 3 || interp_type == 15)
            {
               hypre_BoomerAMGBuildDirInterp(A_array[level], CF_marker,
                                             S, coarse_pnts_global, num_functions, dof_func_data,
                                             debug_flag, trunc_factor, P_max_elmts,
                                             interp_type, &P);
            }
            else if (interp_type == 6) /*Extended+i classical interpolation */
            {
               hypre_BoomerAMGBuildExtPIInterp(A_array[level], CF_marker,
                                               S, coarse_pnts_global, num_functions, dof_func_data,
                                               debug_flag, trunc_factor, P_max_elmts, &P);
            }
            else if (interp_type == 14) /*Extended classical interpolation */
            {
               hypre_BoomerAMGBuildExtInterp(A_array[level], CF_marker,
                                             S, coarse_pnts_global, num_functions, dof_func_data,
                                             debug_flag, trunc_factor, P_max_elmts, &P);
            }
            else if (interp_type == 16) /*Extended classical MM interpolation */
            {
               hypre_BoomerAMGBuildModExtInterp(A_array[level], CF_marker,
                                                S, coarse_pnts_global,
                                                num_functions, dof_func_data,
                                                debug_flag,
                                                trunc_factor, P_max_elmts, &P);
            }
            else if (interp_type == 17) /*Extended+i MM interpolation */
            {
               hypre_BoomerAMGBuildModExtPIInterp(A_array[level], CF_marker,
                                                  S, coarse_pnts_global,
                                                  num_functions, dof_func_data,
                                                  debug_flag, trunc_factor, P_max_elmts, &P);
            }
            else if (interp_type == 18) /*Extended+e MM interpolation */
            {
               hypre_BoomerAMGBuildModExtPEInterp(A_array[level], CF_marker,
                                                  S, coarse_pnts_global,
                                                  num_functions, dof_func_data,
                                                  debug_flag, trunc_factor, P_max_elmts, &P);
            }

            else if (interp_type == 7) /*Extended+i (if no common C) interpolation */
            {
               hypre_BoomerAMGBuildExtPICCInterp(A_array[level], CF_marker,
                                                 S, coarse_pnts_global, num_functions, dof_func_data,
                                                 debug_flag, trunc_factor, P_max_elmts, &P);
            }
            else if (interp_type == 12) /*FF interpolation */
            {
               hypre_BoomerAMGBuildFFInterp(A_array[level], CF_marker,
                                            S, coarse_pnts_global, num_functions, dof_func_data,
                                            debug_flag, trunc_factor, P_max_elmts, &P);
            }
            else if (interp_type == 13) /*FF1 interpolation */
            {
               hypre_BoomerAMGBuildFF1Interp(A_array[level], CF_marker,
                                             S, coarse_pnts_global, num_functions, dof_func_data,
                                             debug_flag, trunc_factor, P_max_elmts, &P);
            }
            else if (interp_type == 8) /*Standard interpolation */
            {
               hypre_BoomerAMGBuildStdInterp(A_array[level], CF_marker,
                                             S, coarse_pnts_global, num_functions, dof_func_data,
                                             debug_flag, trunc_factor, P_max_elmts, sep_weight, &P);
            }
            else if (interp_type == 100) /* 1pt interpolation */
            {
               hypre_BoomerAMGBuildInterpOnePnt(A_array[level], CF_marker, S,
                                                coarse_pnts_global, 1, NULL,
                                                debug_flag, &P);

#if DEBUG_SAVE_ALL_OPS
               char file[256];
               hypre_sprintf(file, "P_%d.mtx", level);
               hypre_ParCSRMatrixPrintIJ(P, 1, 1, file);
#endif
            }
            else if (hypre_ParAMGDataGSMG(amg_data) == 0) /* none of above choosen and not GMSMG */
            {
               if (block_mode) /* nodal interpolation */
               {

                  /* convert A to a block matrix if there isn't already a block
                    matrix - there should be one already*/
                  if (!(A_block_array[level]))
                  {
                     A_block_array[level] =  hypre_ParCSRBlockMatrixConvertFromParCSRMatrix(
                                                A_array[level], num_functions);
                  }

                  /* note that the current CF_marker is nodal */
                  if (interp_type == 11)
                  {
                     hypre_BoomerAMGBuildBlockInterpDiag( A_block_array[level], CF_marker,
                                                          SN,
                                                          coarse_pnts_global, 1,
                                                          NULL,
                                                          debug_flag,
                                                          trunc_factor, P_max_elmts, 1,
                                                          &P_block_array[level]);


                  }
                  else if (interp_type == 22)
                  {
                     hypre_BoomerAMGBuildBlockInterpRV( A_block_array[level], CF_marker,
                                                        SN,
                                                        coarse_pnts_global, 1,
                                                        NULL,
                                                        debug_flag,
                                                        trunc_factor, P_max_elmts,
                                                        &P_block_array[level]);
                  }
                  else if (interp_type == 23)
                  {
                     hypre_BoomerAMGBuildBlockInterpRV( A_block_array[level], CF_marker,
                                                        SN,
                                                        coarse_pnts_global, 1,
                                                        NULL,
                                                        debug_flag,
                                                        trunc_factor, P_max_elmts,
                                                        &P_block_array[level]);
                  }
                  else if (interp_type == 20)
                  {
                     hypre_BoomerAMGBuildBlockInterp( A_block_array[level], CF_marker,
                                                      SN,
                                                      coarse_pnts_global, 1,
                                                      NULL,
                                                      debug_flag,
                                                      trunc_factor, P_max_elmts, 0,
                                                      &P_block_array[level]);

                  }
                  else if (interp_type == 21)
                  {
                     hypre_BoomerAMGBuildBlockInterpDiag( A_block_array[level], CF_marker,
                                                          SN,
                                                          coarse_pnts_global, 1,
                                                          NULL,
                                                          debug_flag,
                                                          trunc_factor, P_max_elmts, 0,
                                                          &P_block_array[level]);
                  }
                  else if (interp_type == 24)
                  {
                     hypre_BoomerAMGBuildBlockDirInterp( A_block_array[level], CF_marker,
                                                         SN,
                                                         coarse_pnts_global, 1,
                                                         NULL,
                                                         debug_flag,
                                                         trunc_factor, P_max_elmts,
                                                         &P_block_array[level]);
                  }

                  else /* interp_type ==10 */
                  {

                     hypre_BoomerAMGBuildBlockInterp( A_block_array[level], CF_marker,
                                                      SN,
                                                      coarse_pnts_global, 1,
                                                      NULL,
                                                      debug_flag,
                                                      trunc_factor, P_max_elmts, 1,
                                                      &P_block_array[level]);

                  }

                  /* we need to set the global number of cols in P, as this was
                     not done in the interp
                     (which calls the matrix create) since we didn't
                     have the global partition */
                  /*  this has to be done before converting from block to non-block*/
                  hypre_ParCSRBlockMatrixGlobalNumCols(P_block_array[level]) = coarse_size;

                  /* if we don't do nodal relaxation, we need a CF_array that is
                     not nodal - right now we don't allow this to happen though*/
                  /*
                    if (grid_relax_type[0] < 20  )
                    {
                    hypre_BoomerAMGCreateScalarCF(CFN_marker, num_functions,
                    hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AN)),
                    &dof_func1, &CF_marker);

                    dof_func_array[level+1] = dof_func1;
                    hypre_TFree(CFN_marker, HYPRE_MEMORY_HOST);
                    CF_marker_array[level] = CF_marker;
                    }
                  */

                  /* clean up other things */
                  hypre_ParCSRMatrixDestroy(AN);
                  hypre_ParCSRMatrixDestroy(SN);

               }
               else /* not block mode - use default interp (interp_type = 0) */
               {
                  if (nodal > -1) /* non-systems, or systems with unknown approach interpolation*/
                  {
                     /* if systems, do we want to use an interp. that uses the full strength matrix?*/

                     if ( (num_functions > 1) && (interp_type == 19 || interp_type == 18 || interp_type == 17 ||
                                                  interp_type == 16))
                     {
                        /* so create a second strength matrix and build interp with with num_functions = 1 */
                        hypre_BoomerAMGCreateS(A_array[level],
                                               strong_threshold, max_row_sum,
                                               1, dof_func_data, &S2);
                        switch (interp_type)
                        {

                           case 19:
                              dbg_flg = debug_flag;
                              if (amg_print_level) { dbg_flg = -debug_flag; }
                              hypre_BoomerAMGBuildInterp(A_array[level], CF_marker,
                                                         S2, coarse_pnts_global, 1,
                                                         dof_func_data,
                                                         dbg_flg, trunc_factor, P_max_elmts, &P);
                              break;

                           case 18:
                              hypre_BoomerAMGBuildStdInterp(A_array[level], CF_marker,
                                                            S2, coarse_pnts_global, 1, dof_func_data,
                                                            debug_flag, trunc_factor, P_max_elmts, 0, &P);

                              break;

                           case 17:
                              hypre_BoomerAMGBuildExtPIInterp(A_array[level], CF_marker,
                                                              S2, coarse_pnts_global, 1, dof_func_data,
                                                              debug_flag, trunc_factor, P_max_elmts, &P);
                              break;
                           case 16:
                              dbg_flg = debug_flag;
                              if (amg_print_level) { dbg_flg = -debug_flag; }
                              hypre_BoomerAMGBuildInterpModUnk(A_array[level], CF_marker,
                                                               S2, coarse_pnts_global, num_functions, dof_func_data,
                                                               dbg_flg, trunc_factor, P_max_elmts, &P);
                              break;

                        }


                        hypre_ParCSRMatrixDestroy(S2);

                     }
                     else /* one function only or unknown-based interpolation- */
                     {
                        dbg_flg = debug_flag;
                        if (amg_print_level) { dbg_flg = -debug_flag; }

                        hypre_BoomerAMGBuildInterp(A_array[level], CF_marker,
                                                   S, coarse_pnts_global, num_functions,
                                                   dof_func_data,
                                                   dbg_flg, trunc_factor, P_max_elmts, &P);


                     }
                  }
               }
            }
            else
            {
               hypre_BoomerAMGBuildInterpGSMG(NULL, CF_marker, S,
                                              coarse_pnts_global, num_functions, dof_func_data,
                                              debug_flag, trunc_factor, &P);
            }
         } /* end of no aggressive coarsening */

         dof_func_array[level + 1] = NULL;
         if (num_functions > 1 && nodal > -1 && (!block_mode) )
         {
            dof_func_array[level + 1] = coarse_dof_func;
         }

         HYPRE_ANNOTATE_REGION_END("%s", "Interpolation");
      } /* end of if max_levels > 1 */

      /* if no coarse-grid, stop coarsening, and set the
       * coarsest solve to be a single sweep of Jacobi */
      if ( (coarse_size == 0) || (coarse_size == fine_size) )
      {
         HYPRE_Int     *num_grid_sweeps =
            hypre_ParAMGDataNumGridSweeps(amg_data);
         HYPRE_Int    **grid_relax_points =
            hypre_ParAMGDataGridRelaxPoints(amg_data);
         if (grid_relax_type[3] == 9 || grid_relax_type[3] == 99
             || grid_relax_type[3] == 19 || grid_relax_type[3] == 98)
         {
            grid_relax_type[3] = grid_relax_type[0];
            num_grid_sweeps[3] = 1;
            if (grid_relax_points) { grid_relax_points[3][0] = 0; }
         }
         if (S)
         {
            hypre_ParCSRMatrixDestroy(S);
         }
         if (P)
         {
            hypre_ParCSRMatrixDestroy(P);
         }
         if (level > 0)
         {
            /* note special case treatment of CF_marker is necessary
             * to do CF relaxation correctly when num_levels = 1 */
            hypre_IntArrayDestroy(CF_marker_array[level]);
            CF_marker_array[level] = NULL;
            hypre_ParVectorDestroy(F_array[level]);
            hypre_ParVectorDestroy(U_array[level]);
         }
         if (level + 1 < max_levels)
         {
            hypre_IntArrayDestroy(dof_func_array[level + 1]);
            dof_func_array[level + 1] = NULL;
         }

         break;
      }
      if (level < agg_num_levels && coarse_size < min_coarse_size)
      {
         if (S)
         {
            hypre_ParCSRMatrixDestroy(S);
         }
         if (P)
         {
            hypre_ParCSRMatrixDestroy(P);
         }
         if (level > 0)
         {
            hypre_IntArrayDestroy(CF_marker_array[level]);
            CF_marker_array[level] = NULL;
            hypre_ParVectorDestroy(F_array[level]);
            hypre_ParVectorDestroy(U_array[level]);
         }
         hypre_IntArrayDestroy(dof_func_array[level + 1]);
         dof_func_array[level + 1] = NULL;
         coarse_size = fine_size;

         break;
      }

      /*-------------------------------------------------------------
       * Build prolongation matrix, P, and place in P_array[level]
       *--------------------------------------------------------------*/

      if (interp_refine > 0)
      {
         for (k = 0; k < interp_refine; k++)
            hypre_BoomerAMGRefineInterp(A_array[level],
                                        P,
                                        coarse_pnts_global,
                                        &num_functions,
                                        dof_func_data,
                                        hypre_IntArrayData(CF_marker_array[level]),
                                        level);
      }

      /*  Post processing of interpolation operators to incorporate
          smooth vectors NOTE: must pick nodal coarsening !!!
          (nodal is changed above to 1 if it is 0)  */
      if (interp_vec_variant && nodal && num_interp_vectors)
      {
         /* TO DO: add option of smoothing the vectors at
          * coarser levels?*/

         if (level < interp_vec_first_level)
         {
            /* coarsen the smooth vecs */
            hypre_BoomerAMGCoarsenInterpVectors( P,
                                                 num_interp_vectors,
                                                 interp_vectors_array[level],
                                                 hypre_IntArrayData(CF_marker_array[level]),
                                                 &interp_vectors_array[level + 1],
                                                 0, num_functions);

         }
         /* do  GM 2 and LN (3) at all levels and GM 1 only on first level */
         if (( interp_vec_variant > 1  && level >= interp_vec_first_level) ||
             (interp_vec_variant == 1 && interp_vec_first_level == level))

         {
            /*if (level == 0)
            {
               hypre_ParCSRMatrixPrintIJ(A_array[0], 0, 0, "A");
               hypre_ParVectorPrintIJ(interp_vectors_array[0][0], 0, "rbm");
            }*/
            if (interp_vec_variant < 3) /* GM */
            {
               hypre_BoomerAMG_GMExpandInterp( A_array[level],
                                               &P,
                                               num_interp_vectors,
                                               interp_vectors_array[level],
                                               &num_functions,
                                               dof_func_data,
                                               &dof_func_array[level + 1],
                                               interp_vec_variant, level,
                                               abs_q_trunc,
                                               expandp_weights,
                                               q_max,
                                               hypre_IntArrayData(CF_marker_array[level]),
                                               interp_vec_first_level);
            }
            else /* LN */
            {
               hypre_BoomerAMG_LNExpandInterp( A_array[level],
                                               &P,
                                               coarse_pnts_global,
                                               &num_functions,
                                               dof_func_data,
                                               &dof_func_array[level + 1],
                                               hypre_IntArrayData(CF_marker_array[level]),
                                               level,
                                               expandp_weights,
                                               num_interp_vectors,
                                               interp_vectors_array[level],
                                               abs_q_trunc,
                                               q_max,
                                               interp_vec_first_level);
            }

            if (level == interp_vec_first_level)
            {
               /* check to see if we made A bigger - this can happen
                * in 3D with certain coarsenings   - if so, need to fix vtemp*/

               HYPRE_Int local_sz = hypre_ParVectorActualLocalSize(Vtemp);
               HYPRE_Int local_P_sz = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(P));
               if (local_sz < local_P_sz)
               {
                  hypre_Vector *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
                  hypre_TFree(hypre_VectorData(Vtemp_local), memory_location);
                  hypre_VectorSize(Vtemp_local) = local_P_sz;
                  hypre_VectorData(Vtemp_local) = hypre_CTAlloc(HYPRE_Complex, local_P_sz * num_vectors,
                                                                memory_location);
                  if (Ztemp)
                  {
                     hypre_Vector *Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
                     hypre_TFree(hypre_VectorData(Ztemp_local), memory_location);
                     hypre_VectorSize(Ztemp_local) = local_P_sz;
                     hypre_VectorData(Ztemp_local) = hypre_CTAlloc(HYPRE_Complex, local_P_sz * num_vectors,
                                                                   memory_location);
                  }
                  if (Ptemp)
                  {
                     hypre_Vector *Ptemp_local = hypre_ParVectorLocalVector(Ptemp);
                     hypre_TFree(hypre_VectorData(Ptemp_local), memory_location);
                     hypre_VectorSize(Ptemp_local) = local_P_sz;
                     hypre_VectorData(Ptemp_local) = hypre_CTAlloc(HYPRE_Complex, local_P_sz * num_vectors,
                                                                   memory_location);
                  }
                  if (Rtemp)
                  {
                     hypre_Vector *Rtemp_local = hypre_ParVectorLocalVector(Rtemp);
                     hypre_TFree(hypre_VectorData(Rtemp_local), memory_location);
                     hypre_VectorSize(Rtemp_local) = local_P_sz;
                     hypre_VectorData(Rtemp_local) = hypre_CTAlloc(HYPRE_Complex, local_P_sz * num_vectors,
                                                                   memory_location);
                  }
               }
               /*if (hypre_ParCSRMatrixGlobalNumRows(A_array[0]) < hypre_ParCSRMatrixGlobalNumCols(P))
               {

                  hypre_ParVectorDestroy(Vtemp);
                  Vtemp = NULL;

                  Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(P),
                                                hypre_ParCSRMatrixGlobalNumCols(P),
                                                hypre_ParCSRMatrixColStarts(P));
                  hypre_ParVectorInitialize(Vtemp);
                  hypre_ParAMGDataVtemp(amg_data) = Vtemp;
               }*/
            }
            /* at the first level we have to add space for the new
             * unknowns in the smooth vectors */
            if (interp_vec_variant > 1 && level < max_levels)
            {
               HYPRE_Int expand_level = 0;

               if (level == interp_vec_first_level)
               {
                  expand_level = 1;
               }

               hypre_BoomerAMGCoarsenInterpVectors( P,
                                                    num_interp_vectors,
                                                    interp_vectors_array[level],
                                                    hypre_IntArrayData(CF_marker_array[level]),
                                                    &interp_vectors_array[level + 1],
                                                    expand_level, num_functions);
            }
         } /* end apply variant */
      }/* end interp_vec_variant > 0 */

      /* Improve on P with Jacobi interpolation */
      for (i = 0; i < post_interp_type; i++)
      {
         hypre_BoomerAMGJacobiInterp( A_array[level], &P, S,
                                      num_functions, dof_func_data,
                                      hypre_IntArrayData(CF_marker_array[level]),
                                      level, jacobi_trunc_threshold, 0.5 * jacobi_trunc_threshold );
      }

      dof_func_data = NULL;
      if (dof_func_array[level + 1])
      {
         dof_func_data = hypre_IntArrayData(dof_func_array[level + 1]);

      }

      if (!block_mode)
      {
         if (mult_addlvl > -1 && level >= mult_addlvl && level <= add_end)
         {
            hypre_Vector *d_diag = NULL;

            if (ns == 1)
            {
               d_diag = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(A_array[level]));

               if (add_rlx == 0)
               {
                  hypre_CSRMatrix *lvl_Adiag = hypre_ParCSRMatrixDiag(A_array[level]);
                  HYPRE_Int lvl_nrows = hypre_CSRMatrixNumRows(lvl_Adiag);
                  HYPRE_Int *lvl_i = hypre_CSRMatrixI(lvl_Adiag);
                  HYPRE_Real *lvl_data = hypre_CSRMatrixData(lvl_Adiag);
                  HYPRE_Real w_inv = 1.0 / add_rlx_wt;
                  /*HYPRE_Real w_inv = 1.0/hypre_ParAMGDataRelaxWeight(amg_data)[level];*/
                  hypre_SeqVectorInitialize_v2(d_diag, HYPRE_MEMORY_HOST);
                  for (i = 0; i < lvl_nrows; i++)
                  {
                     hypre_VectorData(d_diag)[i] = lvl_data[lvl_i[i]] * w_inv;
                  }
               }
               else
               {
                  HYPRE_Real *d_diag_data = NULL;

                  hypre_ParCSRComputeL1Norms(A_array[level], 1, NULL, &d_diag_data);

                  hypre_VectorData(d_diag) = d_diag_data;
                  hypre_SeqVectorInitialize_v2(d_diag, hypre_ParCSRMatrixMemoryLocation(A_array[level]));
               }
            }

            HYPRE_ANNOTATE_REGION_BEGIN("%s", "RAP");
            if (ns == 1)
            {
               hypre_ParCSRMatrix *Q = NULL;
               if (hypre_ParAMGDataModularizedMatMat(amg_data))
               {
                  Q = hypre_ParCSRMatMat(A_array[level], P);
                  hypre_ParCSRMatrixAminvDB(P, Q, hypre_VectorData(d_diag), &P_array[level]);
                  A_H = hypre_ParCSRTMatMat(P, Q);
               }
               else
               {
                  Q = hypre_ParMatmul(A_array[level], P);
                  hypre_ParCSRMatrixAminvDB(P, Q, hypre_VectorData(d_diag), &P_array[level]);
                  A_H = hypre_ParTMatmul(P, Q);
               }
               if (num_procs > 1)
               {
                  hypre_MatvecCommPkgCreate(A_H);
               }
               /*hypre_ParCSRMatrixDestroy(P); */
               hypre_SeqVectorDestroy(d_diag);
               /* Set NonGalerkin drop tol on each level */
               if (level < nongalerk_num_tol) { nongalerk_tol_l = nongalerk_tol[level]; }
               if (nongal_tol_array) { nongalerk_tol_l = nongal_tol_array[level]; }
               if (nongalerk_tol_l > 0.0)
               {
                  /* Build Non-Galerkin Coarse Grid */
                  hypre_ParCSRMatrix *Q = NULL;
                  hypre_BoomerAMGBuildNonGalerkinCoarseOperator(&A_H, Q,
                                                                0.333 * strong_threshold, max_row_sum, num_functions,
                                                                dof_func_data, hypre_IntArrayData(CF_marker_array[level]),
                                                                /* nongalerk_tol, sym_collapse, lump_percent, beta );*/
                                                                nongalerk_tol_l,      1,            0.5,    1.0 );

                  hypre_ParCSRMatrixColStarts(P_array[level])[0] = hypre_ParCSRMatrixRowStarts(A_H)[0];
                  hypre_ParCSRMatrixColStarts(P_array[level])[1] = hypre_ParCSRMatrixRowStarts(A_H)[1];
                  if (!hypre_ParCSRMatrixCommPkg(A_H))
                  {
                     hypre_MatvecCommPkgCreate(A_H);
                  }
               }
               hypre_ParCSRMatrixDestroy(Q);
            }
            else
            {
               HYPRE_Int ns_tmp = ns;
               hypre_ParCSRMatrix *C = NULL;
               hypre_ParCSRMatrix *Ptmp = NULL;
               /* Set NonGalerkin drop tol on each level */
               if (level < nongalerk_num_tol)
               {
                  nongalerk_tol_l = nongalerk_tol[level];
               }
               if (nongal_tol_array) { nongalerk_tol_l = nongal_tol_array[level]; }

               if (nongalerk_tol_l > 0.0)
               {
                  /* Construct AP, and then RAP */
                  hypre_ParCSRMatrix *Q = NULL;
                  if (hypre_ParAMGDataModularizedMatMat(amg_data))
                  {
                     Q = hypre_ParCSRMatMat(A_array[level], P);
                     A_H = hypre_ParCSRTMatMatKT(P, Q, keepTranspose);
                  }
                  else
                  {
                     Q = hypre_ParMatmul(A_array[level], P);
                     A_H = hypre_ParTMatmul(P, Q);
                  }
                  if (num_procs > 1)
                  {
                     hypre_MatvecCommPkgCreate(A_H);
                  }

                  /* Build Non-Galerkin Coarse Grid */
                  hypre_BoomerAMGBuildNonGalerkinCoarseOperator(&A_H, Q,
                                                                0.333 * strong_threshold, max_row_sum, num_functions,
                                                                dof_func_data, hypre_IntArrayData(CF_marker_array[level]),
                                                                /* nongalerk_tol, sym_collapse, lump_percent, beta );*/
                                                                nongalerk_tol_l,      1,            0.5,    1.0 );

                  if (!hypre_ParCSRMatrixCommPkg(A_H))
                  {
                     hypre_MatvecCommPkgCreate(A_H);
                  }

                  /* Delete AP */
                  hypre_ParCSRMatrixDestroy(Q);
               }
               else if (rap2)
               {
                  /* Use two matrix products to generate A_H */
                  hypre_ParCSRMatrix *Q = NULL;
                  if (hypre_ParAMGDataModularizedMatMat(amg_data))
                  {
                     Q = hypre_ParCSRMatMat(A_array[level], P);
                     A_H = hypre_ParCSRTMatMatKT(P, Q, keepTranspose);
                  }
                  else
                  {
                     Q = hypre_ParMatmul(A_array[level], P);
                     A_H = hypre_ParTMatmul(P, Q);
                  }

                  if (num_procs > 1)
                  {
                     hypre_MatvecCommPkgCreate(A_H);
                  }
                  /* Delete AP */
                  hypre_ParCSRMatrixDestroy(Q);
               }
               else
               {
                  if (hypre_ParAMGDataModularizedMatMat(amg_data))
                  {
                     A_H = hypre_ParCSRMatrixRAPKT(P, A_array[level],
                                                   P, keepTranspose);
                  }
                  else
                  {
                     hypre_BoomerAMGBuildCoarseOperatorKT(P, A_array[level], P,
                                                          keepTranspose, &A_H);
                  }
               }

               if (add_rlx == 18)
               {
                  C = hypre_CreateC(A_array[level], 0.0);
               }
               else
               {
                  C = hypre_CreateC(A_array[level], add_rlx_wt);
               }

               Ptmp = P;
               while (ns_tmp > 0)
               {
                  Pnew = Ptmp;
                  Ptmp = NULL;
                  if (hypre_ParAMGDataModularizedMatMat(amg_data))
                  {
                     Ptmp = hypre_ParCSRMatMat(C, Pnew);
                  }
                  else
                  {
                     Ptmp = hypre_ParMatmul(C, Pnew);
                  }
                  if (ns_tmp < ns)
                  {
                     hypre_ParCSRMatrixDestroy(Pnew);
                  }
                  ns_tmp--;
               }
               Pnew = Ptmp;
               P_array[level] = Pnew;
               hypre_ParCSRMatrixDestroy(C);
            } /* if (ns == 1) */
            HYPRE_ANNOTATE_REGION_END("%s", "RAP");

            if (add_P_max_elmts || add_trunc_factor)
            {
               hypre_BoomerAMGTruncandBuild(P_array[level], add_trunc_factor, add_P_max_elmts);
            }
            /*else
                hypre_MatvecCommPkgCreate(P_array[level]);  */
            hypre_ParCSRMatrixDestroy(P);
         }
         else
         {
            P_array[level] = P;
            /* RL: save R matrix */
            if (restri_type)
            {
               R_array[level] = R;
            }
         }
      }

      if (S)
      {
         hypre_ParCSRMatrixDestroy(S);
      }
      S = NULL;

      hypre_TFree(SmoothVecs, HYPRE_MEMORY_HOST);

      if (debug_flag == 1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         hypre_printf("Proc = %d    Level = %d    Build Interp Time = %f\n",
                      my_id, level, wall_time);
         fflush(NULL);
      }

      /*-------------------------------------------------------------
       * Build coarse-grid operator, A_array[level+1] by R*A*P
       *--------------------------------------------------------------*/

      HYPRE_ANNOTATE_REGION_BEGIN("%s", "RAP");
      if (debug_flag == 1) { wall_time = time_getWallclockSeconds(); }

      if (block_mode)
      {
         hypre_ParCSRBlockMatrixRAP(P_block_array[level],
                                    A_block_array[level],
                                    P_block_array[level], &A_H_block);

         hypre_ParCSRBlockMatrixSetNumNonzeros(A_H_block);
         hypre_ParCSRBlockMatrixSetDNumNonzeros(A_H_block);
         A_block_array[level + 1] = A_H_block;
      }
      else if (mult_addlvl == -1 || level < mult_addlvl || level > add_end)
      {
         /* Set NonGalerkin drop tol on each level */
         if (level < nongalerk_num_tol)
         {
            nongalerk_tol_l = nongalerk_tol[level];
         }
         if (nongal_tol_array)
         {
            nongalerk_tol_l = nongal_tol_array[level];
         }

         if (nongalerk_tol_l > 0.0)
         {
            /* Construct AP, and then RAP */
            hypre_ParCSRMatrix *Q = NULL;
            if (hypre_ParAMGDataModularizedMatMat(amg_data))
            {
               Q = hypre_ParCSRMatMat(A_array[level], P_array[level]);
               A_H = hypre_ParCSRTMatMatKT(P_array[level], Q, keepTranspose);
            }
            else
            {
               Q = hypre_ParMatmul(A_array[level], P_array[level]);
               A_H = hypre_ParTMatmul(P_array[level], Q);
            }
            if (num_procs > 1) { hypre_MatvecCommPkgCreate(A_H); }

            /* Build Non-Galerkin Coarse Grid */
            hypre_BoomerAMGBuildNonGalerkinCoarseOperator(&A_H, Q,
                                                          0.333 * strong_threshold, max_row_sum, num_functions,
                                                          dof_func_data, hypre_IntArrayData(CF_marker_array[level]),
                                                          /* nongalerk_tol, sym_collapse, lump_percent, beta );*/
                                                          nongalerk_tol_l,      1,            0.5,    1.0 );

            if (!hypre_ParCSRMatrixCommPkg(A_H))
            {
               hypre_MatvecCommPkgCreate(A_H);
            }
            /* Delete AP */
            hypre_ParCSRMatrixDestroy(Q);
         }
         else if (restri_type) /* RL: */
         {
            /* Use two matrix products to generate A_H */
            hypre_ParCSRMatrix *AP = NULL;
            if (hypre_ParAMGDataModularizedMatMat(amg_data))
            {
               AP  = hypre_ParCSRMatMat(A_array[level], P_array[level]);
               A_H = hypre_ParCSRMatMat(R_array[level], AP);
               hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A_H));
            }
            else
            {
               AP  = hypre_ParMatmul(A_array[level], P_array[level]);
               A_H = hypre_ParMatmul(R_array[level], AP);
            }
            if (num_procs > 1)
            {
               hypre_MatvecCommPkgCreate(A_H);
            }
            /* Delete AP */
            hypre_ParCSRMatrixDestroy(AP);
         }
         else if (rap2)
         {
            /* Use two matrix products to generate A_H */
            hypre_ParCSRMatrix *Q = NULL;
            if (hypre_ParAMGDataModularizedMatMat(amg_data))
            {
               Q = hypre_ParCSRMatMat(A_array[level], P_array[level]);
               A_H = hypre_ParCSRTMatMatKT(P_array[level], Q, keepTranspose);
            }
            else
            {
               Q = hypre_ParMatmul(A_array[level], P_array[level]);
               A_H = hypre_ParTMatmul(P_array[level], Q);
            }
            if (num_procs > 1)
            {
               hypre_MatvecCommPkgCreate(A_H);
            }
            /* Delete AP */
            hypre_ParCSRMatrixDestroy(Q);
         }
         else
         {
            /* Compute standard Galerkin coarse-grid product */
            if (hypre_ParAMGDataModularizedMatMat(amg_data))
            {
               A_H = hypre_ParCSRMatrixRAPKT(P_array[level], A_array[level],
                                             P_array[level], keepTranspose);
            }
            else
            {
               hypre_BoomerAMGBuildCoarseOperatorKT(P_array[level], A_array[level],
                                                    P_array[level], keepTranspose, &A_H);
            }

            if (Pnew && ns == 1)
            {
               hypre_ParCSRMatrixDestroy(P);
               P_array[level] = Pnew;
            }
         }
      }

#if DEBUG_SAVE_ALL_OPS
      if (level == 0)
      {
         hypre_ParCSRMatrixPrintIJ(A_array[0], 0, 0, "A_00.IJ.out");
      }
      char file[256];
      hypre_sprintf(file, "A_%02d.IJ.out", level + 1);
      hypre_ParCSRMatrixPrintIJ(A_H, 0, 0, file);

      hypre_sprintf(file, "P_%02d.IJ.out", level);
      hypre_ParCSRMatrixPrintIJ(P_array[level], 0, 0, file);
#endif

      HYPRE_ANNOTATE_REGION_END("%s", "RAP");
      if (debug_flag == 1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         hypre_printf("Proc = %d    Level = %d    Build Coarse Operator Time = %f\n",
                      my_id, level, wall_time);
         fflush(NULL);
      }

      HYPRE_ANNOTATE_MGLEVEL_END(level);
      hypre_GpuProfilingPopRange();
      ++level;
      HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
      hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
      hypre_GpuProfilingPushRange(nvtx_name);

      if (!block_mode)
      {
         /* dropping in A_H */
         hypre_ParCSRMatrixDropSmallEntries(A_H, hypre_ParAMGDataADropTol(amg_data),
                                            hypre_ParAMGDataADropType(amg_data));
         /* if CommPkg for A_H was not built */
         if (num_procs > 1 && hypre_ParCSRMatrixCommPkg(A_H) == NULL)
         {
            hypre_MatvecCommPkgCreate(A_H);
         }
         /* NumNonzeros was set in hypre_ParCSRMatrixDropSmallEntries */
         if (hypre_ParAMGDataADropTol(amg_data) <= 0.0)
         {
            hypre_ParCSRMatrixSetNumNonzeros(A_H);
            hypre_ParCSRMatrixSetDNumNonzeros(A_H);
         }
         A_array[level] = A_H;
      }

#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_HOST)
#endif
      {
         HYPRE_Real size = ((HYPRE_Real)fine_size) * .75;
         if (coarsen_type > 0 && coarse_size >= (HYPRE_BigInt)size)
         {
            coarsen_type = 0;
         }
      }

      {
         HYPRE_Int max_thresh = hypre_max(coarse_threshold, seq_threshold);
#if defined(HYPRE_USING_DSUPERLU)
         max_thresh = hypre_max(max_thresh, dslu_threshold);
#endif
         if ( (level == max_levels - 1) || (coarse_size <= (HYPRE_BigInt) max_thresh) )
         {
            not_finished_coarsening = 0;
         }
      }
   }  /* end of coarsening loop: while (not_finished_coarsening) */

   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Coarse solve");

   /* redundant coarse grid solve */
   if ((seq_threshold >= coarse_threshold) &&
       (coarse_size > (HYPRE_BigInt) coarse_threshold) &&
       (level != max_levels - 1))
   {
      hypre_seqAMGSetup(amg_data, level, coarse_threshold);
   }
#if defined(HYPRE_USING_DSUPERLU)
   else if ((dslu_threshold >= coarse_threshold) &&
            (coarse_size > (HYPRE_BigInt)coarse_threshold) &&
            (level != max_levels - 1))
   {
      HYPRE_Solver dslu_solver;
      hypre_SLUDistSetup(&dslu_solver, A_array[level], amg_print_level);
      hypre_ParAMGDataDSLUSolver(amg_data) = dslu_solver;
   }
#endif
   else if (grid_relax_type[3] == 9   ||
            grid_relax_type[3] == 19  ||
            grid_relax_type[3] == 98  ||
            grid_relax_type[3] == 99  ||
            grid_relax_type[3] == 198 ||
            grid_relax_type[3] == 199)
   {
      /* Gaussian elimination on the coarsest level */
      if (coarse_size <= coarse_threshold)
      {
         hypre_GaussElimSetup(amg_data, level, grid_relax_type[3]);
      }
      else
      {
         grid_relax_type[3] = grid_relax_type[1];
      }
   }
   HYPRE_ANNOTATE_REGION_END("%s", "Coarse solve");
   HYPRE_ANNOTATE_MGLEVEL_END(level);
   hypre_GpuProfilingPopRange();

   if (level > 0)
   {
      if (block_mode)
      {
         F_array[level] =
            hypre_ParVectorCreateFromBlock(hypre_ParCSRBlockMatrixComm(A_block_array[level]),
                                           hypre_ParCSRMatrixGlobalNumRows(A_block_array[level]),
                                           hypre_ParCSRBlockMatrixRowStarts(A_block_array[level]),
                                           hypre_ParCSRBlockMatrixBlockSize(A_block_array[level]));
         hypre_ParVectorInitialize(F_array[level]);

         U_array[level] =
            hypre_ParVectorCreateFromBlock(hypre_ParCSRBlockMatrixComm(A_block_array[level]),
                                           hypre_ParCSRMatrixGlobalNumRows(A_block_array[level]),
                                           hypre_ParCSRBlockMatrixRowStarts(A_block_array[level]),
                                           hypre_ParCSRBlockMatrixBlockSize(A_block_array[level]));

         hypre_ParVectorInitialize(U_array[level]);
      }
      else
      {
         F_array[level] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                  hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                  hypre_ParCSRMatrixRowStarts(A_array[level]));
         hypre_ParVectorNumVectors(F_array[level]) = num_vectors;
         hypre_ParVectorInitialize_v2(F_array[level], memory_location);

         U_array[level] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                  hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                  hypre_ParCSRMatrixRowStarts(A_array[level]));
         hypre_ParVectorNumVectors(U_array[level]) = num_vectors;
         hypre_ParVectorInitialize_v2(U_array[level], memory_location);
      }
   }

   /*-----------------------------------------------------------------------
    * enter all the stuff created, A[level], P[level], CF_marker[level],
    * for levels 1 through coarsest, into amg_data data structure
    *-----------------------------------------------------------------------*/

   num_levels = level + 1;
   hypre_ParAMGDataNumLevels(amg_data) = num_levels;
   if (hypre_ParAMGDataSmoothNumLevels(amg_data) > num_levels - 1)
   {
      hypre_ParAMGDataSmoothNumLevels(amg_data) = num_levels;
   }
   smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);

   /*-----------------------------------------------------------------------
    * Setup of special smoothers when needed
    *-----------------------------------------------------------------------*/

   if (addlvl > -1 ||
       grid_relax_type[1] ==  7 || grid_relax_type[2] ==  7 || grid_relax_type[3] ==  7 ||
       grid_relax_type[1] ==  8 || grid_relax_type[2] ==  8 || grid_relax_type[3] ==  8 ||
       grid_relax_type[1] == 11 || grid_relax_type[2] == 11 || grid_relax_type[3] == 11 ||
       grid_relax_type[1] == 12 || grid_relax_type[2] == 12 || grid_relax_type[3] == 12 ||
       grid_relax_type[1] == 13 || grid_relax_type[2] == 13 || grid_relax_type[3] == 13 ||
       grid_relax_type[1] == 14 || grid_relax_type[2] == 14 || grid_relax_type[3] == 14 ||
       grid_relax_type[1] == 18 || grid_relax_type[2] == 18 || grid_relax_type[3] == 18 ||
       grid_relax_type[1] == 30 || grid_relax_type[2] == 30 || grid_relax_type[3] == 30 ||
       grid_relax_type[1] == 88 || grid_relax_type[2] == 88 || grid_relax_type[3] == 88 ||
       grid_relax_type[1] == 89 || grid_relax_type[2] == 89 || grid_relax_type[3] == 89)
   {
      l1_norms = hypre_CTAlloc(hypre_Vector*, num_levels, HYPRE_MEMORY_HOST);
      hypre_ParAMGDataL1Norms(amg_data) = l1_norms;
   }

   /* Chebyshev */
   if (grid_relax_type[0] == 16 || grid_relax_type[1] == 16 ||
       grid_relax_type[2] == 16 || grid_relax_type[3] == 16)
   {
      max_eig_est = hypre_CTAlloc(HYPRE_Real, num_levels, HYPRE_MEMORY_HOST);
      min_eig_est = hypre_CTAlloc(HYPRE_Real, num_levels, HYPRE_MEMORY_HOST);
      hypre_ParAMGDataMaxEigEst(amg_data) = max_eig_est;
      hypre_ParAMGDataMinEigEst(amg_data) = min_eig_est;
      cheby_ds = hypre_CTAlloc(hypre_Vector *, num_levels, HYPRE_MEMORY_HOST);
      cheby_coefs = hypre_CTAlloc(HYPRE_Real *, num_levels, HYPRE_MEMORY_HOST);
      hypre_ParAMGDataChebyDS(amg_data) = cheby_ds;
      hypre_ParAMGDataChebyCoefs(amg_data) = cheby_coefs;
   }

   /* CG */
   if (grid_relax_type[0] == 15 || grid_relax_type[1] == 15 ||
       grid_relax_type[2] == 15 || grid_relax_type[3] == 15)
   {
      smoother = hypre_CTAlloc(HYPRE_Solver, num_levels, HYPRE_MEMORY_HOST);
      hypre_ParAMGDataSmoother(amg_data) = smoother;
   }

   if (addlvl == -1)
   {
      addlvl = num_levels;
   }

   for (j = 0; j < addlvl; j++)
   {
      HYPRE_Real *l1_norm_data = NULL;

      HYPRE_ANNOTATE_MGLEVEL_BEGIN(j);
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
      hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
      hypre_GpuProfilingPushRange(nvtx_name);

      hypre_sprintf(nvtx_name, "%s-%d", "Relaxation", j);
      hypre_GpuProfilingPushRange(nvtx_name);

      if (j < num_levels - 1 &&
          (grid_relax_type[1] == 8  || grid_relax_type[1] == 89 ||
           grid_relax_type[1] == 13 || grid_relax_type[1] == 14 ||
           grid_relax_type[2] == 8  || grid_relax_type[2] == 89 ||
           grid_relax_type[2] == 13 || grid_relax_type[2] == 14))
      {
         if (relax_order)
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 4, hypre_IntArrayData(CF_marker_array[j]), &l1_norm_data);
         }
         else
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 4, NULL, &l1_norm_data);
         }
      }
      else if (j == num_levels - 1 &&
               (grid_relax_type[3] == 8  || grid_relax_type[3] == 89 ||
                grid_relax_type[3] == 13 || grid_relax_type[3] == 14))
      {
         hypre_ParCSRComputeL1Norms(A_array[j], 4, NULL, &l1_norm_data);
      }

      if (j < num_levels - 1 && (grid_relax_type[1] == 30 || grid_relax_type[2] == 30))
      {
         if (relax_order)
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 3, hypre_IntArrayData(CF_marker_array[j]), &l1_norm_data);
         }
         else
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 3, NULL, &l1_norm_data);
         }
      }
      else if (j == num_levels - 1 && grid_relax_type[3] == 30)
      {
         hypre_ParCSRComputeL1Norms(A_array[j], 3, NULL, &l1_norm_data);
      }

      if (j < num_levels - 1 && (grid_relax_type[1] == 88 || grid_relax_type[2] == 88 ))
      {
         if (relax_order)
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 6, hypre_IntArrayData(CF_marker_array[j]), &l1_norm_data);
         }
         else
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 6, NULL, &l1_norm_data);
         }
      }
      else if (j == num_levels - 1 && (grid_relax_type[3] == 88))
      {
         hypre_ParCSRComputeL1Norms(A_array[j], 6, NULL, &l1_norm_data);
      }

      if (j < num_levels - 1 && (grid_relax_type[1] == 18 || grid_relax_type[2] == 18))
      {
         if (relax_order)
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 1, hypre_IntArrayData(CF_marker_array[j]), &l1_norm_data);
         }
         else
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norm_data);
         }
      }
      else if (j == num_levels - 1 && grid_relax_type[3] == 18)
      {
         hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norm_data);
      }

      if (l1_norm_data)
      {
         l1_norms[j] = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(A_array[j]));
         hypre_VectorData(l1_norms[j]) = l1_norm_data;
         hypre_SeqVectorInitialize_v2(l1_norms[j], hypre_ParCSRMatrixMemoryLocation(A_array[j]));
      }

      HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");
      HYPRE_ANNOTATE_MGLEVEL_END(j);
      hypre_GpuProfilingPopRange();
      hypre_GpuProfilingPopRange();
   }

   for (j = addlvl; j < hypre_min(add_end + 1, num_levels) ; j++)
   {
      if (add_rlx == 18 )
      {
         HYPRE_Real *l1_norm_data = NULL;

         HYPRE_ANNOTATE_MGLEVEL_BEGIN(j);
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
         hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
         hypre_GpuProfilingPushRange(nvtx_name);

         hypre_sprintf(nvtx_name, "%s-%d", "Relaxation", j);
         hypre_GpuProfilingPushRange(nvtx_name);

         hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norm_data);

         l1_norms[j] = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(A_array[j]));
         hypre_VectorData(l1_norms[j]) = l1_norm_data;
         hypre_SeqVectorInitialize_v2(l1_norms[j], hypre_ParCSRMatrixMemoryLocation(A_array[j]));

         HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");
         HYPRE_ANNOTATE_MGLEVEL_END(j);
         hypre_GpuProfilingPopRange();
         hypre_GpuProfilingPopRange();
      }
   }

   for (j = add_end + 1; j < num_levels; j++)
   {
      HYPRE_Real *l1_norm_data = NULL;

      HYPRE_ANNOTATE_MGLEVEL_BEGIN(j);
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
      hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
      hypre_GpuProfilingPushRange(nvtx_name);

      hypre_sprintf(nvtx_name, "%s-%d", "Relaxation", j);
      hypre_GpuProfilingPushRange(nvtx_name);


      if (j < num_levels - 1 &&
          (grid_relax_type[1] == 8 || grid_relax_type[1] == 13 || grid_relax_type[1] == 14 ||
           grid_relax_type[2] == 8 || grid_relax_type[2] == 13 || grid_relax_type[2] == 14))
      {
         if (relax_order)
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 4, hypre_IntArrayData(CF_marker_array[j]),
                                       &l1_norm_data);
         }
         else
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 4, NULL, &l1_norm_data);
         }
      }
      else if ((grid_relax_type[3] == 8 || grid_relax_type[3] == 13 || grid_relax_type[3] == 14) &&
               j == num_levels - 1)
      {
         hypre_ParCSRComputeL1Norms(A_array[j], 4, NULL, &l1_norm_data);
      }
      if ((grid_relax_type[1] == 18 || grid_relax_type[2] == 18) && j < num_levels - 1)
      {
         if (relax_order)
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 1, hypre_IntArrayData(CF_marker_array[j]),
                                       &l1_norm_data);
         }
         else
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norm_data);
         }
      }
      else if (grid_relax_type[3] == 18 && j == num_levels - 1)
      {
         hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norm_data);
      }

      if (l1_norm_data)
      {
         l1_norms[j] = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(A_array[j]));
         hypre_VectorData(l1_norms[j]) = l1_norm_data;
         hypre_SeqVectorInitialize_v2(l1_norms[j], hypre_ParCSRMatrixMemoryLocation(A_array[j]));
      }

      HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");
      HYPRE_ANNOTATE_MGLEVEL_END(j);
      hypre_GpuProfilingPopRange();
      hypre_GpuProfilingPopRange();
   }

   for (j = 0; j < num_levels; j++)
   {
      HYPRE_ANNOTATE_MGLEVEL_BEGIN(j);
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
      hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
      hypre_GpuProfilingPushRange(nvtx_name);

      hypre_sprintf(nvtx_name, "%s-%d", "Relaxation", j);
      hypre_GpuProfilingPushRange(nvtx_name);

      if ( grid_relax_type[1]  == 7 || grid_relax_type[2] == 7   ||
           (grid_relax_type[3] == 7 && j == (num_levels - 1))    ||

           grid_relax_type[1]  == 11 || grid_relax_type[2] == 11 ||
           (grid_relax_type[3] == 11 && j == (num_levels - 1))   ||

           grid_relax_type[1]  == 12 || grid_relax_type[2] == 12 ||
           (grid_relax_type[3] == 12 && j == (num_levels - 1)) )
      {
         HYPRE_Real *l1_norm_data = NULL;

         hypre_ParCSRComputeL1Norms(A_array[j], 5, NULL, &l1_norm_data);

         l1_norms[j] = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(A_array[j]));
         hypre_VectorData(l1_norms[j]) = l1_norm_data;
         hypre_SeqVectorInitialize_v2(l1_norms[j], hypre_ParCSRMatrixMemoryLocation(A_array[j]));
      }
      else if (grid_relax_type[1] == 16 || grid_relax_type[2] == 16 ||
               (grid_relax_type[3] == 16 && j == (num_levels - 1)))
      {
         HYPRE_Int scale = hypre_ParAMGDataChebyScale(amg_data);
         /* If the full array is being considered, create the relevant temp vectors */

         HYPRE_Int variant = hypre_ParAMGDataChebyVariant(amg_data);
         HYPRE_Real max_eig, min_eig = 0;
         HYPRE_Real *coefs = NULL;
         HYPRE_Int cheby_order = hypre_ParAMGDataChebyOrder(amg_data);
         HYPRE_Int cheby_eig_est = hypre_ParAMGDataChebyEigEst(amg_data);
         HYPRE_Real cheby_fraction = hypre_ParAMGDataChebyFraction(amg_data);
         if (cheby_eig_est)
         {
            hypre_ParCSRMaxEigEstimateCG(A_array[j], scale, cheby_eig_est,
                                         &max_eig, &min_eig);
         }
         else
         {
            hypre_ParCSRMaxEigEstimate(A_array[j], scale, &max_eig, &min_eig);
         }
         max_eig_est[j] = max_eig;
         min_eig_est[j] = min_eig;

         cheby_ds[j] = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(A_array[j]));
         hypre_VectorVectorStride(cheby_ds[j])   = hypre_ParCSRMatrixNumRows(A_array[j]);
         hypre_VectorIndexStride(cheby_ds[j])    = 1;
         hypre_VectorMemoryLocation(cheby_ds[j]) = hypre_ParCSRMatrixMemoryLocation(A_array[j]);

         hypre_ParCSRRelax_Cheby_Setup(A_array[j],
                                       max_eig,
                                       min_eig,
                                       cheby_fraction,
                                       cheby_order,
                                       scale,
                                       variant,
                                       &coefs,
                                       &hypre_VectorData(cheby_ds[j]));
         cheby_coefs[j] = coefs;
      }
      else if (grid_relax_type[1] == 15 || (grid_relax_type[3] == 15 && j == (num_levels - 1))  )
      {

         HYPRE_ParCSRPCGCreate(comm, &smoother[j]);
         /*HYPRE_ParCSRPCGSetup(smoother[j],
                             (HYPRE_ParCSRMatrix) A_array[j],
                             (HYPRE_ParVector) F_array[j],
                             (HYPRE_ParVector) U_array[j]);*/

         HYPRE_PCGSetTol(smoother[j], 1e-12); /* make small */
         HYPRE_PCGSetTwoNorm(smoother[j], 1); /* use 2-norm*/

         HYPRE_ParCSRPCGSetup(smoother[j],
                              (HYPRE_ParCSRMatrix) A_array[j],
                              (HYPRE_ParVector) F_array[j],
                              (HYPRE_ParVector) U_array[j]);


      }

      if (relax_weight[j] == 0.0)
      {
         hypre_ParCSRMatrixScaledNorm(A_array[j], &relax_weight[j]);
         if (relax_weight[j] != 0.0)
         {
            relax_weight[j] = 4.0 / 3.0 / relax_weight[j];
         }
         else
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Warning ! Matrix norm is zero !!!");
         }
      }

      if ((smooth_type == 6 || smooth_type == 16) && smooth_num_levels > j)
      {
         /* Sanity check */
         if (num_vectors > 1)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "Schwarz smoothing doesn't support multicomponent vectors");
            return hypre_error_flag;
         }

         schwarz_relax_wt = hypre_ParAMGDataSchwarzRlxWeight(amg_data);

         HYPRE_SchwarzCreate(&smoother[j]);
         HYPRE_SchwarzSetNumFunctions(smoother[j], num_functions);
         HYPRE_SchwarzSetVariant(smoother[j],
                                 hypre_ParAMGDataVariant(amg_data));
         HYPRE_SchwarzSetOverlap(smoother[j],
                                 hypre_ParAMGDataOverlap(amg_data));
         HYPRE_SchwarzSetDomainType(smoother[j],
                                    hypre_ParAMGDataDomainType(amg_data));
         HYPRE_SchwarzSetNonSymm(smoother[j],
                                 hypre_ParAMGDataSchwarzUseNonSymm(amg_data));
         if (schwarz_relax_wt > 0)
         {
            HYPRE_SchwarzSetRelaxWeight(smoother[j], schwarz_relax_wt);
         }
         HYPRE_SchwarzSetup(smoother[j],
                            (HYPRE_ParCSRMatrix) A_array[j],
                            (HYPRE_ParVector) f,
                            (HYPRE_ParVector) u);
         if (schwarz_relax_wt < 0 )
         {
            num_cg_sweeps = (HYPRE_Int) (-schwarz_relax_wt);
            hypre_BoomerAMGCGRelaxWt(amg_data, j, num_cg_sweeps,
                                     &schwarz_relax_wt);
            /*hypre_printf (" schwarz weight %f \n", schwarz_relax_wt);*/
            HYPRE_SchwarzSetRelaxWeight(smoother[j], schwarz_relax_wt);
            if (hypre_ParAMGDataVariant(amg_data) > 0)
            {
               local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[j]));
               hypre_SchwarzReScale(smoother[j], local_size, schwarz_relax_wt);
            }
            schwarz_relax_wt = 1;
         }
      }
      else if ((smooth_type == 9 || smooth_type == 19) && smooth_num_levels > j)
      {
         /* Sanity checks */
#ifdef HYPRE_MIXEDINT
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Euclid smoothing is not available in mixedint mode!");
         return hypre_error_flag;
#endif

         if (num_vectors > 1)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "Euclid smoothing doesn't support multicomponent vectors");
            return hypre_error_flag;
         }

         HYPRE_EuclidCreate(comm, &smoother[j]);
         if (euclidfile)
         {
            HYPRE_EuclidSetParamsFromFile(smoother[j], euclidfile);
         }
         HYPRE_EuclidSetLevel(smoother[j], eu_level);
         if (eu_bj)
         {
            HYPRE_EuclidSetBJ(smoother[j], eu_bj);
         }
         if (eu_sparse_A)
         {
            HYPRE_EuclidSetSparseA(smoother[j], eu_sparse_A);
         }
         HYPRE_EuclidSetup(smoother[j],
                           (HYPRE_ParCSRMatrix) A_array[j],
                           (HYPRE_ParVector) F_array[j],
                           (HYPRE_ParVector) U_array[j]);
      }
      else if ((smooth_type == 4 || smooth_type == 14) && smooth_num_levels > j)
      {
         /* Sanity check */
         if (num_vectors > 1)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "FSAI smoothing doesn't support multicomponent vectors");
            return hypre_error_flag;
         }

         HYPRE_FSAICreate(&smoother[j]);
         HYPRE_FSAISetAlgoType(smoother[j], fsai_algo_type);
         HYPRE_FSAISetLocalSolveType(smoother[j], fsai_local_solve_type);
         HYPRE_FSAISetMaxSteps(smoother[j], fsai_max_steps);
         HYPRE_FSAISetMaxStepSize(smoother[j], fsai_max_step_size);
         HYPRE_FSAISetMaxNnzRow(smoother[j], fsai_max_nnz_row);
         HYPRE_FSAISetNumLevels(smoother[j], fsai_num_levels);
         HYPRE_FSAISetThreshold(smoother[j], fsai_threshold);
         HYPRE_FSAISetKapTolerance(smoother[j], fsai_kap_tolerance);
         HYPRE_FSAISetTolerance(smoother[j], 0.0);
         HYPRE_FSAISetOmega(smoother[j], relax_weight[level]);
         HYPRE_FSAISetEigMaxIters(smoother[j], fsai_eig_max_iters);
         HYPRE_FSAISetPrintLevel(smoother[j], (amg_print_level >= 1) ? 1 : 0);

         HYPRE_FSAISetup(smoother[j],
                         (HYPRE_ParCSRMatrix) A_array[j],
                         (HYPRE_ParVector) F_array[j],
                         (HYPRE_ParVector) U_array[j]);

#if DEBUG_SAVE_ALL_OPS
         {
            char filename[256];
            hypre_sprintf(file, "G_%02d.IJ.out", j);
            hypre_ParCSRMatrixPrintIJ(hypre_ParFSAIDataGmat((hypre_ParFSAIData*) smoother[j]),
                                      0, 0, filename);
         }
#endif
      }
      else if ((smooth_type == 5 || smooth_type == 15) && smooth_num_levels > j)
      {
         /* Sanity check */
         if (num_vectors > 1)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "ILU smoothing doesn't support multicomponent vectors");
            return hypre_error_flag;
         }

         HYPRE_ILUCreate(&smoother[j]);
         HYPRE_ILUSetType(smoother[j], ilu_type);
         HYPRE_ILUSetLocalReordering( smoother[j], ilu_reordering_type);
         HYPRE_ILUSetMaxIter(smoother[j], ilu_max_iter);
         HYPRE_ILUSetTriSolve(smoother[j], ilu_tri_solve);
         HYPRE_ILUSetLowerJacobiIters(smoother[j], ilu_lower_jacobi_iters);
         HYPRE_ILUSetUpperJacobiIters(smoother[j], ilu_upper_jacobi_iters);
         HYPRE_ILUSetTol(smoother[j], 0.);
         HYPRE_ILUSetDropThreshold(smoother[j], ilu_droptol);
         HYPRE_ILUSetLogging(smoother[j], 0);
         HYPRE_ILUSetPrintLevel(smoother[j], 0);
         HYPRE_ILUSetLevelOfFill(smoother[j], ilu_lfil);
         HYPRE_ILUSetMaxNnzPerRow(smoother[j], ilu_max_row_nnz);
         HYPRE_ILUSetup(smoother[j],
                        (HYPRE_ParCSRMatrix) A_array[j],
                        (HYPRE_ParVector) F_array[j],
                        (HYPRE_ParVector) U_array[j]);
      }
      else if ((smooth_type == 8 || smooth_type == 18) && smooth_num_levels > j)
      {
         /* Sanity checks */
#ifdef HYPRE_MIXEDINT
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "ParaSails smoothing is not available in mixedint mode!");
         return hypre_error_flag;
#endif

         if (num_vectors > 1)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "ParaSails smoothing doesn't support multicomponent vectors");
            return hypre_error_flag;
         }

         HYPRE_ParCSRParaSailsCreate(comm, &smoother[j]);
         HYPRE_ParCSRParaSailsSetParams(smoother[j], thresh, nlevel);
         HYPRE_ParCSRParaSailsSetFilter(smoother[j], filter);
         HYPRE_ParCSRParaSailsSetSym(smoother[j], sym);
         HYPRE_ParCSRParaSailsSetup(smoother[j],
                                    (HYPRE_ParCSRMatrix) A_array[j],
                                    (HYPRE_ParVector) F_array[j],
                                    (HYPRE_ParVector) U_array[j]);
      }
      else if ((smooth_type == 7 || smooth_type == 17) && smooth_num_levels > j)
      {
         /* Sanity checks */
#ifdef HYPRE_MIXEDINT
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Pilut smoothing is not available in mixedint mode!");
         return hypre_error_flag;
#endif

         if (num_vectors > 1)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "Pilut smoothing doesn't support multicomponent vectors");
            return hypre_error_flag;
         }

         HYPRE_ParCSRPilutCreate(comm, &smoother[j]);
         HYPRE_ParCSRPilutSetup(smoother[j],
                                (HYPRE_ParCSRMatrix) A_array[j],
                                (HYPRE_ParVector) F_array[j],
                                (HYPRE_ParVector) U_array[j]);
         HYPRE_ParCSRPilutSetDropTolerance(smoother[j], drop_tol);
         HYPRE_ParCSRPilutSetFactorRowSize(smoother[j], max_nz_per_row);
      }
      else if ( (j < num_levels - 1) ||
                ((j == num_levels - 1) &&
                 (grid_relax_type[3] !=  9 && grid_relax_type[3] != 99  &&
                  grid_relax_type[3] != 19 && grid_relax_type[3] != 98) && coarse_size > 9) )
      {
         if (relax_weight[j] < 0)
         {
            num_cg_sweeps = (HYPRE_Int) (-relax_weight[j]);
            hypre_BoomerAMGCGRelaxWt(amg_data, j, num_cg_sweeps, &relax_weight[j]);
         }
         if (omega[j] < 0)
         {
            num_cg_sweeps = (HYPRE_Int) (-omega[j]);
            hypre_BoomerAMGCGRelaxWt(amg_data, j, num_cg_sweeps, &omega[j]);
         }
      }

      HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");
      HYPRE_ANNOTATE_MGLEVEL_END(j);
      hypre_GpuProfilingPopRange();
      hypre_GpuProfilingPopRange();
   } /* end of levels loop */

   if (amg_logging > 1)
   {
      Residual_array = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                             hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                             hypre_ParCSRMatrixRowStarts(A_array[0]) );
      hypre_ParVectorInitialize_v2(Residual_array, memory_location);
      hypre_ParAMGDataResidual(amg_data) = Residual_array;
   }
   else
   {
      hypre_ParAMGDataResidual(amg_data) = NULL;
   }

   if (simple > -1 && simple < num_levels)
   {
      hypre_CreateDinv(amg_data);
   }
   else if ( (mult_additive > -1 && mult_additive < num_levels) ||
             (additive > -1 && additive < num_levels) )
   {
      hypre_CreateLambda(amg_data);
   }

   if (cum_nnz_AP > 0.0)
   {
      cum_nnz_AP = hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
      for (j = 0; j < num_levels - 1; j++)
      {
         hypre_ParCSRMatrixSetDNumNonzeros(P_array[j]);
         cum_nnz_AP += hypre_ParCSRMatrixDNumNonzeros(P_array[j]);
         cum_nnz_AP += hypre_ParCSRMatrixDNumNonzeros(A_array[j + 1]);
      }
      hypre_ParAMGDataCumNnzAP(amg_data) = cum_nnz_AP;
   }

   /*-----------------------------------------------------------------------
    * Print some stuff
    *-----------------------------------------------------------------------*/

   if (amg_print_level == 1 || amg_print_level == 3)
   {
      hypre_BoomerAMGSetupStats(amg_data, A);
   }

   /* print out CF info to plot grids in matlab (see 'tools/AMGgrids.m') */

   if (hypre_ParAMGDataPlotGrids(amg_data))
   {
      HYPRE_Int *CF, *CFc, *itemp;
      FILE* fp;
      char filename[256];
      HYPRE_Int coorddim = hypre_ParAMGDataCoordDim (amg_data);
      float *coordinates = hypre_ParAMGDataCoordinates (amg_data);

      if (!coordinates) { coorddim = 0; }

      if (block_mode)
      {
         local_size = hypre_CSRMatrixNumRows(hypre_ParCSRBlockMatrixDiag(A_block_array[0]));
      }
      else
      {
         local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
      }

      CF  = hypre_CTAlloc(HYPRE_Int, local_size, HYPRE_MEMORY_HOST);
      CFc = hypre_CTAlloc(HYPRE_Int, local_size, HYPRE_MEMORY_HOST);

      for (level = (num_levels - 2); level >= 0; level--)
      {
         /* swap pointers */
         itemp = CFc;
         CFc = CF;
         CF = itemp;
         if (block_mode)
         {
            local_size = hypre_CSRMatrixNumRows(hypre_ParCSRBlockMatrixDiag(A_block_array[level]));
         }
         else
         {
            local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
         }

         /* copy CF_marker to the host if needed */
         hypre_IntArray *CF_marker_host;
         if (hypre_GetActualMemLocation(hypre_IntArrayMemoryLocation(CF_marker_array[level])) ==
             hypre_MEMORY_DEVICE)
         {
            CF_marker_host = hypre_IntArrayCloneDeep_v2(CF_marker_array[level], HYPRE_MEMORY_HOST);
         }
         else
         {
            CF_marker_host = CF_marker_array[level];
         }
         CF_marker = hypre_IntArrayData(CF_marker_host);

         for (i = 0, j = 0; i < local_size; i++)
         {
            /* if a C-point */
            CF[i] = 0;
            if (CF_marker[i] > -1)
            {
               CF[i] = CFc[j] + 1;
               j++;
            }
         }

         /* copy back to device and destroy host copy */
         if (hypre_GetActualMemLocation(hypre_IntArrayMemoryLocation(CF_marker_array[level])) ==
             hypre_MEMORY_DEVICE)
         {
            hypre_IntArrayCopy(CF_marker_host, CF_marker_array[level]);
            hypre_IntArrayDestroy(CF_marker_host);
         }
      }
      if (block_mode)
      {
         local_size = hypre_CSRMatrixNumRows(hypre_ParCSRBlockMatrixDiag(A_block_array[0]));
      }
      else
      {
         local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
      }
      hypre_sprintf (filename, "%s.%05d", hypre_ParAMGDataPlotFileName (amg_data), my_id);
      fp = fopen(filename, "w");

      for (i = 0; i < local_size; i++)
      {
         for (j = 0; j < coorddim; j++)
         {
            hypre_fprintf (fp, "% f ", (HYPRE_Real) coordinates[coorddim * i + j]);
         }
         hypre_fprintf(fp, "%d\n", CF[i]);
      }
      fclose(fp);

      hypre_TFree(CF, HYPRE_MEMORY_HOST);
      hypre_TFree(CFc, HYPRE_MEMORY_HOST);
   }

   /* print out matrices on all levels  */
#if DEBUG
   {
      char  filename[256];

      if (block_mode)
      {
         hypre_ParCSRMatrix *temp_A;

         for (level = 0; level < num_levels; level++)
         {
            hypre_sprintf(filename, "BoomerAMG.out.A_blk.%02d.ij", level);
            temp_A =  hypre_ParCSRBlockMatrixConvertToParCSRMatrix(
                         A_block_array[level]);
            hypre_ParCSRMatrixPrintIJ(temp_A, 0, 0, filename);
            hypre_ParCSRMatrixDestroy(temp_A);
         }

      }
      else
      {
         for (level = 0; level < num_levels; level++)
         {
            hypre_sprintf(filename, "BoomerAMG.out.A.%02d.ij", level);
            hypre_ParCSRMatrixPrintIJ(A_array[level], 0, 0, filename);
         }
         for (level = 0; level < (num_levels - 1); level++)
         {
            hypre_sprintf(filename, "BoomerAMG.out.P.%02d.ij", level);
            hypre_ParCSRMatrixPrintIJ(P_array[level], 0, 0, filename);
         }
      }
   }
#endif

   /* run compatible relaxation on all levels and print results */
#if 0
   {
      hypre_ParVector *u_vec, *f_vec;
      HYPRE_Real      *u, rho0, rho1, rho;
      HYPRE_Int              n;

      for (level = 0; level < (num_levels - 1); level++)
      {
         u_vec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                       hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                       hypre_ParCSRMatrixRowStarts(A_array[level]));
         hypre_ParVectorInitialize(u_vec);
         f_vec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                       hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                       hypre_ParCSRMatrixRowStarts(A_array[level]));
         hypre_ParVectorInitialize(f_vec);

         hypre_ParVectorSetRandomValues(u_vec, 99);
         hypre_ParVectorSetConstantValues(f_vec, 0.0);

         /* set C-pt values to zero */
         n = hypre_VectorSize(hypre_ParVectorLocalVector(u_vec));
         u = hypre_VectorData(hypre_ParVectorLocalVector(u_vec));
         for (i = 0; i < n; i++)
         {
            if (CF_marker_array[level][i] == 1)
            {
               u[i] = 0.0;
            }
         }

         rho1 = hypre_ParVectorInnerProd(u_vec, u_vec);
         for (i = 0; i < 5; i++)
         {
            rho0 = rho1;
            hypre_BoomerAMGRelax(A_array[level], f_vec, CF_marker_array[level],
                                 grid_relax_type[0], -1,
                                 relax_weight[level], omega[level], l1_norms[level],
                                 u_vec, Vtemp, Ztemp);
            rho1 = hypre_ParVectorInnerProd(u_vec, u_vec);
            rho = hypre_sqrt(rho1 / rho0);
            if (rho < 0.01)
            {
               break;
            }
         }

         hypre_ParVectorDestroy(u_vec);
         hypre_ParVectorDestroy(f_vec);

         if (my_id == 0)
         {
            hypre_printf("level = %d, rhocr = %f\n", level, rho);
         }
      }
   }
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return (hypre_error_flag);
}
