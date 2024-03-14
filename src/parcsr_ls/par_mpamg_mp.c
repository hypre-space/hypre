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
#include "hypre_parcsr_ls_mp.h"
#include "hypre_parcsr_mv_mup.h"
#include "hypre_parcsr_ls_mup.h"
#include "hypre_seq_mv_mp.h"
#include "hypre_seq_mv_mup.h"
#include "hypre_utilities_mup.h"

/*--------------------------------------------------------------------------
 * hypre_MPAMGCreate
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_MIXED_PRECISION

void *
hypre_MPAMGCreate_mp( void )
{
   hypre_ParAMGData  *amg_data;

   /* setup params */
   HYPRE_Int    max_levels;
   HYPRE_Int    max_coarse_size;
   HYPRE_Int    min_coarse_size;
   HYPRE_Int    coarsen_cut_factor;
   HYPRE_Real   strong_threshold;
   HYPRE_Int    Sabs;
   HYPRE_Real   max_row_sum;
   HYPRE_Real   trunc_factor;
   HYPRE_Real   agg_trunc_factor;
   HYPRE_Real   agg_P12_trunc_factor;
   HYPRE_Int    interp_type;
   HYPRE_Int    sep_weight;
   HYPRE_Int    coarsen_type;
   HYPRE_Int    measure_type;
   HYPRE_Int    P_max_elmts;
   HYPRE_Int    num_functions;
   HYPRE_Int    nodal, nodal_levels, nodal_diag;
   HYPRE_Int    keep_same_sign;
   HYPRE_Int    num_paths;
   HYPRE_Int    agg_num_levels;
   HYPRE_Int    agg_interp_type;
   HYPRE_Int    agg_P_max_elmts;
   HYPRE_Int    agg_P12_max_elmts;
   HYPRE_Int    rap2;
   HYPRE_Int    keepT;

   /* solve params */
   HYPRE_Int    min_iter;
   HYPRE_Int    max_iter;
   HYPRE_Int    fcycle;
   HYPRE_Int    cycle_type;

   HYPRE_Real   tol;

   HYPRE_Int    num_sweeps;
   HYPRE_Int    relax_down;
   HYPRE_Int    relax_up;
   HYPRE_Int    relax_coarse;
   HYPRE_Int    relax_order;
   HYPRE_Real   relax_wt;
   HYPRE_Real   outer_wt;

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
   HYPRE_Int    one = 1;

   char     plot_file_name[251] = {0};

  // HYPRE_MemoryLocation memory_location = hypre_HandleMemoryLocation_dbl(hypre_handle());

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/

   /* setup params */
   max_levels = 25;
   max_coarse_size = 9;
   min_coarse_size = 0;
   coarsen_cut_factor = 0;
   strong_threshold = 0.25;
   Sabs = 0;
   max_row_sum = 0.9;
   trunc_factor = 0.0;
   agg_trunc_factor = 0.0;
   agg_P12_trunc_factor = 0.0;
   sep_weight = 0;
   coarsen_type = 10;
   interp_type = 6;
   measure_type = 0;
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
   agg_interp_type = 4;

   /* solve params */
   min_iter  = 0;
   max_iter  = 20;
   fcycle = 0;
   cycle_type = 1;
   tol = 1.0e-6;

   num_sweeps = 1;
   relax_down = 13;
   relax_up = 14;
   relax_coarse = 9;
   relax_order = 0;
   relax_wt = 1.0;
   outer_wt = 1.0;

   /* log info */
   num_iterations = 0;
   cum_num_iterations = 0;
   cum_nnz_AP = -1.0;

   /* output params */
   print_level = 0;
   logging = 0;
   hypre_sprintf_dbl(log_file_name, "%s", "amg.out.log");
   /* cycle_op_count = 0; */
   debug_flag = 0;

   rap2 = 0;
   keepT = 0;

   /*-----------------------------------------------------------------------
    * Create the hypre_ParAMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = (hypre_ParAMGData *) hypre_CAlloc_dbl((size_t)(one), (size_t)sizeof(hypre_ParAMGData), HYPRE_MEMORY_HOST);

   /* memory location will be reset at the setup */
   //hypre_ParAMGDataMemoryLocation(amg_data) = memory_location;
   hypre_ParAMGDataMemoryLocation(amg_data) = HYPRE_MEMORY_DEVICE;

   hypre_ParAMGDataMaxLevels(amg_data) =  max_levels;
   hypre_ParAMGDataUserRelaxType(amg_data) = -1;
   hypre_ParAMGDataUserNumSweeps(amg_data) = -1;
   hypre_ParAMGDataUserRelaxWeight(amg_data) = relax_wt;
   hypre_ParAMGDataOuterWt(amg_data) = outer_wt;
   /*hypre_MPAMGSetMaxCoarseSize_mp(amg_data, max_coarse_size);
   hypre_MPAMGSetMinCoarseSize_mp(amg_data, min_coarse_size);
   hypre_MPAMGSetCoarsenCutFactor_mp(amg_data, coarsen_cut_factor);
   hypre_MPAMGSetStrongThreshold_mp(amg_data, strong_threshold);
   hypre_MPAMGSetSabs_mp(amg_data, Sabs);
   hypre_MPAMGSetMaxRowSum_mp(amg_data, max_row_sum);
   hypre_MPAMGSetTruncFactor_mp(amg_data, trunc_factor);
   hypre_MPAMGSetAggTruncFactor_mp(amg_data, agg_trunc_factor);
   hypre_MPAMGSetAggP12TruncFactor_mp(amg_data, agg_P12_trunc_factor);
   hypre_MPAMGSetSepWeight_mp(amg_data, sep_weight);
   hypre_MPAMGSetMeasureType_mp(amg_data, measure_type);
   hypre_MPAMGSetCoarsenType_mp(amg_data, coarsen_type);
   hypre_MPAMGSetInterpType_mp(amg_data, interp_type);
   hypre_MPAMGSetPMaxElmts_mp(amg_data, P_max_elmts);
   hypre_MPAMGSetAggPMaxElmts_mp(amg_data, agg_P_max_elmts);
   hypre_MPAMGSetAggP12MaxElmts_mp(amg_data, agg_P12_max_elmts);
   hypre_MPAMGSetNumFunctions_mp(amg_data, num_functions);
   hypre_MPAMGSetNodal_mp(amg_data, nodal);
   hypre_MPAMGSetNodalLevels_mp(amg_data, nodal_levels);
   hypre_MPAMGSetNodalDiag_mp(amg_data, nodal_diag);
   hypre_MPAMGSetKeepSameSign_mp(amg_data, keep_same_sign);
   hypre_MPAMGSetNumPaths_mp(amg_data, num_paths);
   hypre_MPAMGSetAggNumLevels_mp(amg_data, agg_num_levels);
   hypre_MPAMGSetAggInterpType_mp(amg_data, agg_interp_type);

   hypre_MPAMGSetMinIter_mp(amg_data, min_iter);
   hypre_MPAMGSetMaxIter_mp(amg_data, max_iter);
   hypre_MPAMGSetCycleType_mp(amg_data, cycle_type);
   hypre_MPAMGSetFCycle_mp(amg_data, fcycle);
   hypre_MPAMGSetTol_mp(amg_data, tol);
   hypre_MPAMGSetNumSweeps_mp(amg_data, num_sweeps);
   hypre_MPAMGSetCycleRelaxType_mp(amg_data, relax_down, 1);
   hypre_MPAMGSetCycleRelaxType_mp(amg_data, relax_up, 2);
   hypre_MPAMGSetCycleRelaxType_mp(amg_data, relax_coarse, 3);
   hypre_MPAMGSetRelaxOrder_mp(amg_data, relax_order);
   hypre_MPAMGSetRelaxWt_mp(amg_data, relax_wt);
   hypre_MPAMGSetOuterWt_mp(amg_data, outer_wt);

   hypre_MPAMGSetNumIterations_mp(amg_data, num_iterations);


   hypre_MPAMGSetPrintLevel_mp(amg_data, print_level);
   hypre_MPAMGSetLogging_mp(amg_data, logging);
   hypre_MPAMGSetDebugFlag_mp(amg_data, debug_flag);*/
   //hypre_MPAMGSetPrintFileName(amg_data, log_file_name);

   hypre_ParAMGDataAArray(amg_data) = NULL;
   hypre_ParAMGDataPArray(amg_data) = NULL;
   hypre_ParAMGDataRArray(amg_data) = NULL;
   hypre_ParAMGDataCFMarkerArray(amg_data) = NULL;
   hypre_ParAMGDataFArray(amg_data) = NULL;
   hypre_ParAMGDataUArray(amg_data) = NULL;
   hypre_ParAMGDataDofFunc(amg_data) = NULL;
   hypre_ParAMGDataDofFuncArray(amg_data) = NULL;
   hypre_ParAMGDataL1Norms(amg_data) = NULL;

   hypre_ParAMGDataRAP2(amg_data)              = rap2;
   hypre_ParAMGDataKeepTranspose(amg_data)     = keepT;

   hypre_ParAMGDataCumNnzAP(amg_data) = cum_nnz_AP;

   hypre_ParAMGDataPrecisionArray(amg_data) = NULL;
   hypre_ParAMGDataVtempDBL(amg_data) = NULL;
   hypre_ParAMGDataVtempFLT(amg_data) = NULL;
   hypre_ParAMGDataVtempLONGDBL(amg_data) = NULL;
   hypre_ParAMGDataZtempDBL(amg_data) = NULL;
   hypre_ParAMGDataZtempFLT(amg_data) = NULL;
   hypre_ParAMGDataZtempLONGDBL(amg_data) = NULL;

   HYPRE_ANNOTATE_FUNC_END;

   return (void *) amg_data;
}

/*--------------------------------------------------------------------------
 * hypre_MPAMGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGDestroy_mp( void *data )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   if (amg_data)
   {
      HYPRE_Int     num_levels = hypre_ParAMGDataNumLevels(amg_data);
      MPI_Comm      new_comm = hypre_ParAMGDataNewComm(amg_data);
      HYPRE_Int    *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
      HYPRE_Int     i;
      HYPRE_MemoryLocation memory_location = hypre_ParAMGDataMemoryLocation(amg_data);
      HYPRE_Precision *precision_array = hypre_ParAMGDataPrecisionArray(amg_data);

      if (hypre_ParAMGDataNumGridSweeps(amg_data))
      {
         hypre_Free_dbl(hypre_ParAMGDataNumGridSweeps(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataNumGridSweeps(amg_data) = NULL;
      }
      if (grid_relax_type)
      {
         hypre_Free_dbl(hypre_ParAMGDataGridRelaxType(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataGridRelaxType(amg_data) = NULL;
      }
      if (hypre_ParAMGDataRelaxWeight(amg_data))
      {
         hypre_Free_dbl(hypre_ParAMGDataRelaxWeight(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataRelaxWeight(amg_data) = NULL;
      }
      if (hypre_ParAMGDataOmega(amg_data))
      {
         hypre_Free_dbl(hypre_ParAMGDataOmega(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataOmega(amg_data) = NULL;
      }
      if (hypre_ParAMGDataDofFunc(amg_data))
      {
         hypre_IntArrayDestroy_dbl(hypre_ParAMGDataDofFunc(amg_data));
         hypre_ParAMGDataDofFunc(amg_data) = NULL;
      }
      for (i = 1; i < num_levels; i++)
      {
         if (precision_array[i] == HYPRE_REAL_DOUBLE)
         {
	    hypre_ParVectorDestroy_dbl(hypre_ParAMGDataFArray(amg_data)[i]);
            hypre_ParVectorDestroy_dbl(hypre_ParAMGDataUArray(amg_data)[i]);

            if (hypre_ParAMGDataAArray(amg_data)[i])
            {
               hypre_ParCSRMatrixDestroy_dbl(hypre_ParAMGDataAArray(amg_data)[i]);
            }

            if (hypre_ParAMGDataPArray(amg_data)[i - 1])
            {
               hypre_ParCSRMatrixDestroy_dbl(hypre_ParAMGDataPArray(amg_data)[i - 1]);
            }

            hypre_IntArrayDestroy_dbl(hypre_ParAMGDataCFMarkerArray(amg_data)[i - 1]);
         }
	 else if (precision_array[i] == HYPRE_REAL_SINGLE)
         {
	    hypre_ParVectorDestroy_flt(hypre_ParAMGDataFArray(amg_data)[i]);
            hypre_ParVectorDestroy_flt(hypre_ParAMGDataUArray(amg_data)[i]);

            if (hypre_ParAMGDataAArray(amg_data)[i])
            {
               hypre_ParCSRMatrixDestroy_flt(hypre_ParAMGDataAArray(amg_data)[i]);
            }

            if (hypre_ParAMGDataPArray(amg_data)[i - 1])
            {
               hypre_ParCSRMatrixDestroy_flt(hypre_ParAMGDataPArray(amg_data)[i - 1]);
            }

            hypre_IntArrayDestroy_flt(hypre_ParAMGDataCFMarkerArray(amg_data)[i - 1]);
         }
	 else if (precision_array[i] == HYPRE_REAL_LONGDOUBLE)
         {
	    hypre_ParVectorDestroy_long_dbl(hypre_ParAMGDataFArray(amg_data)[i]);
            hypre_ParVectorDestroy_long_dbl(hypre_ParAMGDataUArray(amg_data)[i]);

            if (hypre_ParAMGDataAArray(amg_data)[i])
            {
               hypre_ParCSRMatrixDestroy_long_dbl(hypre_ParAMGDataAArray(amg_data)[i]);
            }

            if (hypre_ParAMGDataPArray(amg_data)[i - 1])
            {
               hypre_ParCSRMatrixDestroy_long_dbl(hypre_ParAMGDataPArray(amg_data)[i - 1]);
            }

            hypre_IntArrayDestroy_long_dbl(hypre_ParAMGDataCFMarkerArray(amg_data)[i - 1]);
         }
      }
      if (hypre_ParAMGDataGridRelaxPoints(amg_data))
      {
         for (i = 0; i < 4; i++)
         {
            hypre_Free_dbl(hypre_ParAMGDataGridRelaxPoints(amg_data)[i], HYPRE_MEMORY_HOST);
         }
         hypre_Free_dbl(hypre_ParAMGDataGridRelaxPoints(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataGridRelaxPoints(amg_data) = NULL;
      }

      if (hypre_ParAMGDataL1Norms(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
	    if (precision_array[i] == HYPRE_REAL_DOUBLE)
               hypre_SeqVectorDestroy_dbl(hypre_ParAMGDataL1Norms(amg_data)[i]);
	    else if (precision_array[i] == HYPRE_REAL_SINGLE)
               hypre_SeqVectorDestroy_flt(hypre_ParAMGDataL1Norms(amg_data)[i]);
	    else if (precision_array[i] == HYPRE_REAL_DOUBLE)
               hypre_SeqVectorDestroy_long_dbl(hypre_ParAMGDataL1Norms(amg_data)[i]);
         }
         hypre_Free_dbl(hypre_ParAMGDataL1Norms(amg_data), HYPRE_MEMORY_HOST);
      }

      /* see comments in par_coarsen.c regarding special case for CF_marker */
      if (num_levels == 1)
      {
         hypre_IntArrayDestroy_dbl(hypre_ParAMGDataCFMarkerArray(amg_data)[0]);
      }

      hypre_Free_dbl(hypre_ParAMGDataFArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_Free_dbl(hypre_ParAMGDataUArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_Free_dbl(hypre_ParAMGDataAArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_Free_dbl(hypre_ParAMGDataPArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_Free_dbl(hypre_ParAMGDataCFMarkerArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_ParVectorDestroy_dbl(hypre_ParAMGDataVtempDBL(amg_data));
      hypre_ParVectorDestroy_flt(hypre_ParAMGDataVtempFLT(amg_data));
      hypre_ParVectorDestroy_long_dbl(hypre_ParAMGDataVtempLONGDBL(amg_data));
      hypre_ParVectorDestroy_dbl(hypre_ParAMGDataZtempDBL(amg_data));
      hypre_ParVectorDestroy_flt(hypre_ParAMGDataZtempFLT(amg_data));
      hypre_ParVectorDestroy_long_dbl(hypre_ParAMGDataZtempLONGDBL(amg_data));

      if (hypre_ParAMGDataDofFuncArray(amg_data))
      {
         for (i = 1; i < num_levels; i++)
         {
            hypre_IntArrayDestroy_dbl(hypre_ParAMGDataDofFuncArray(amg_data)[i]);
         }
         hypre_Free_dbl(hypre_ParAMGDataDofFuncArray(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataDofFuncArray(amg_data) = NULL;
      }
      hypre_ParVectorDestroy_dbl(hypre_ParAMGDataResidual(amg_data));
      hypre_ParAMGDataResidual(amg_data) = NULL;

      if (new_comm != hypre_MPI_COMM_NULL)
      {
         MPI_Comm_free(&new_comm);
      }

      hypre_Free_dbl(amg_data, HYPRE_MEMORY_HOST);
   }
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
#endif
