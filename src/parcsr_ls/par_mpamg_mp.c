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
#include "HYPRE_parcsr_ls_mp.h"
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
   HYPRE_Int    Sabs;
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

   /* log info */
   HYPRE_Int    num_iterations;
   HYPRE_Int    cum_num_iterations;
   hypre_double cum_nnz_AP;

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
   /* mixed precision paramters */
   ((amg_data) -> strong_threshold_dbl) = 0.25;
   ((amg_data) -> max_row_sum_dbl) = 0.9;
   ((amg_data) -> trunc_factor_dbl) = 0.0;
   ((amg_data) -> agg_trunc_factor_dbl) = 0.0;
   ((amg_data) -> agg_P12_trunc_factor_dbl) = 0.0;
   ((amg_data) -> user_relax_weight_dbl) = 1.0;
   ((amg_data) -> outer_wt_dbl) = 1.0;
   ((amg_data) -> strong_threshold_flt) = 0.25;
   ((amg_data) -> max_row_sum_flt) = 0.9;
   ((amg_data) -> trunc_factor_flt) = 0.0;
   ((amg_data) -> agg_trunc_factor_flt) = 0.0;
   ((amg_data) -> agg_P12_trunc_factor_flt) = 0.0;
   ((amg_data) -> user_relax_weight_flt) = 1.0;
   ((amg_data) -> outer_wt_flt) = 1.0;
   ((amg_data) -> strong_threshold_ldbl) = 0.25;
   ((amg_data) -> max_row_sum_ldbl) = 0.9;
   ((amg_data) -> trunc_factor_ldbl) = 0.0;
   ((amg_data) -> agg_trunc_factor_ldbl) = 0.0;
   ((amg_data) -> agg_P12_trunc_factor_ldbl) = 0.0;
   ((amg_data) -> user_relax_weight_ldbl) = 1.0;
   ((amg_data) -> outer_wt_ldbl) = 1.0;
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
   hypre_MPAMGSetMaxCoarseSize_mp(amg_data, max_coarse_size);
   hypre_MPAMGSetMinCoarseSize_mp(amg_data, min_coarse_size);
   hypre_MPAMGSetCoarsenCutFactor_mp(amg_data, coarsen_cut_factor);
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
   hypre_MPAMGSetNumIterations_mp(amg_data, num_iterations);


   hypre_MPAMGSetPrintLevel_mp(amg_data, print_level);
   hypre_MPAMGSetLogging_mp(amg_data, logging);
   hypre_MPAMGSetDebugFlag_mp(amg_data, debug_flag);
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

      hypre_Free_dbl(amg_data, HYPRE_MEMORY_HOST);
   }
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetMaxLevels_mp( void *data,
                            HYPRE_Int   max_levels )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   //HYPRE_Int old_max_levels;
   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_levels < 1)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   /*old_max_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (old_max_levels < max_levels)
   {
      HYPRE_Real *relax_weight, *omega, *nongal_tol_array;
      HYPRE_Real relax_wt, outer_wt, nongalerkin_tol;
      HYPRE_Int i;
      relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);
      if (relax_weight)
      {
         relax_wt = hypre_ParAMGDataUserRelaxWeight(amg_data);
         relax_weight = (HYPRE_Real *) (hypre_ReAlloc_dbl(relax_weight, (size_t)max_levels, HYPRE_MEMORY_HOST));
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
         omega = (HYPRE_Real *) (hypre_ReAlloc_dbl(omega,  (size_t)max_levels, HYPRE_MEMORY_HOST));
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
         nongal_tol_array = (HYPRE_Real *)(hypre_ReAlloc_dbl(nongal_tol_array,  (size_t)max_levels, HYPRE_MEMORY_HOST));
         for (i = old_max_levels; i < max_levels; i++)
         {
            nongal_tol_array[i] = nongalerkin_tol;
         }
         hypre_ParAMGDataNonGalTolArray(amg_data) = nongal_tol_array;
      }
   }*/
   hypre_ParAMGDataMaxLevels(amg_data) = max_levels;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetMaxLevels_mp( void *data,
                             HYPRE_Int *  max_levels )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_levels = hypre_ParAMGDataMaxLevels(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetMaxCoarseSize_mp( void *data,
                                 HYPRE_Int   max_coarse_size )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_coarse_size < 1)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMaxCoarseSize(amg_data) = max_coarse_size;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetMaxCoarseSize_mp( void *data,
                                 HYPRE_Int *  max_coarse_size )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_coarse_size = hypre_ParAMGDataMaxCoarseSize(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetMinCoarseSize_mp( void *data,
                                 HYPRE_Int   min_coarse_size )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (min_coarse_size < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMinCoarseSize(amg_data) = min_coarse_size;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetMinCoarseSize_mp( void *data,
                                 HYPRE_Int *  min_coarse_size )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *min_coarse_size = hypre_ParAMGDataMinCoarseSize(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetCoarsenCutFactor_mp( void       *data,
                                    HYPRE_Int   coarsen_cut_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (coarsen_cut_factor < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataCoarsenCutFactor(amg_data) = coarsen_cut_factor;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetCoarsenCutFactor_mp( void       *data,
                                   HYPRE_Int  *coarsen_cut_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *coarsen_cut_factor = hypre_ParAMGDataCoarsenCutFactor(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetStrongThreshold_mp( void     *data,
                                  hypre_double strong_threshold )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (strong_threshold < 0 || strong_threshold > 1)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   ((amg_data) -> strong_threshold_dbl) = strong_threshold;
   ((amg_data) -> strong_threshold_flt) = (hypre_float) strong_threshold;
   ((amg_data) -> strong_threshold_ldbl) = (hypre_long_double) strong_threshold;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetMaxRowSum_mp( void     *data,
                            hypre_double  max_row_sum )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_row_sum <= 0 || max_row_sum > 1)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   ((amg_data) -> max_row_sum_dbl) = max_row_sum;
   ((amg_data) -> max_row_sum_flt) = (hypre_float) max_row_sum;
   ((amg_data) -> max_row_sum_ldbl) = (hypre_long_double) max_row_sum;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetTruncFactor_mp( void     *data,
                              hypre_double    trunc_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (trunc_factor < 0 || trunc_factor >= 1)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   ((amg_data) -> trunc_factor_dbl) = trunc_factor;
   ((amg_data) -> trunc_factor_flt) = (hypre_float) trunc_factor;
   ((amg_data) -> trunc_factor_ldbl) = (hypre_long_double) trunc_factor;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetPMaxElmts_mp( void     *data,
                             HYPRE_Int    P_max_elmts )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (P_max_elmts < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataPMaxElmts(amg_data) = P_max_elmts;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetPMaxElmts_mp( void     *data,
                             HYPRE_Int *  P_max_elmts )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *P_max_elmts = hypre_ParAMGDataPMaxElmts(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetInterpType_mp( void     *data,
                              HYPRE_Int       interp_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }


   if ((interp_type < 0 || interp_type > 25) && interp_type != 100)

   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataInterpType(amg_data) = interp_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetInterpType_mp( void     *data,
                              HYPRE_Int *     interp_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *interp_type = hypre_ParAMGDataInterpType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetSepWeight_mp( void     *data,
                             HYPRE_Int       sep_weight )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataSepWeight(amg_data) = sep_weight;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetMinIter_mp( void     *data,
                           HYPRE_Int       min_iter )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMinIter(amg_data) = min_iter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetMinIter_mp( void     *data,
                           HYPRE_Int *     min_iter )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *min_iter = hypre_ParAMGDataMinIter(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetMaxIter_mp( void     *data,
                           HYPRE_Int     max_iter )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_iter < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMaxIter(amg_data) = max_iter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetMaxIter_mp( void     *data,
                           HYPRE_Int *   max_iter )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_iter = hypre_ParAMGDataMaxIter(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetCoarsenType_mp( void  *data,
                               HYPRE_Int    coarsen_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataCoarsenType(amg_data) = coarsen_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetCoarsenType_mp( void  *data,
                               HYPRE_Int *  coarsen_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetMeasureType_mp( void  *data,
                               HYPRE_Int    measure_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMeasureType(amg_data) = measure_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetMeasureType_mp( void  *data,
                               HYPRE_Int *  measure_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *measure_type = hypre_ParAMGDataMeasureType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetCycleType_mp( void  *data,
                             HYPRE_Int    cycle_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (cycle_type < 0 || cycle_type > 2)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataCycleType(amg_data) = cycle_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetCycleType_mp( void  *data,
                             HYPRE_Int *  cycle_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *cycle_type = hypre_ParAMGDataCycleType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetFCycle_mp( void     *data,
                          HYPRE_Int fcycle )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataFCycle(amg_data) = fcycle != 0;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetFCycle_mp( void      *data,
                          HYPRE_Int *fcycle )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *fcycle = hypre_ParAMGDataFCycle(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetTol_mp( void     *data,
                      hypre_double    tol  )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (tol < 0 || tol > 1)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   ((amg_data) -> tol_dbl) = tol;
   ((amg_data) -> tol_flt) = (hypre_float) tol;
   ((amg_data) -> tol_ldbl) = (hypre_long_double) tol;

   return hypre_error_flag;
}

/* The "Get" function for SetNumSweeps is GetCycleNumSweeps. */
HYPRE_Int
hypre_MPAMGSetNumSweeps_mp( void     *data,
                             HYPRE_Int      num_sweeps )
{
   HYPRE_Int i;
   HYPRE_Int *num_grid_sweeps;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (num_sweeps < 1)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      hypre_ParAMGDataNumGridSweeps(amg_data) = (HYPRE_Int *) (hypre_CAlloc_dbl (4, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST));
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
hypre_MPAMGSetCycleNumSweeps_mp( void     *data,
                                  HYPRE_Int      num_sweeps,
                                  HYPRE_Int      k )
{
   HYPRE_Int i;
   HYPRE_Int *num_grid_sweeps;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (num_sweeps < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (k < 1 || k > 3)
   {
      //hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      num_grid_sweeps = (HYPRE_Int *) (hypre_CAlloc_dbl (4, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST));
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
hypre_MPAMGGetCycleNumSweeps_mp( void     *data,
                                  HYPRE_Int *    num_sweeps,
                                  HYPRE_Int      k )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      //hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *num_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data)[k];

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetNumGridSweeps_mp( void     *data,
                                 HYPRE_Int      *num_grid_sweeps )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!num_grid_sweeps)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data))
   {
      hypre_Free_dbl(hypre_ParAMGDataNumGridSweeps(amg_data), HYPRE_MEMORY_HOST);
   }
   hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetNumGridSweeps_mp( void     *data,
                                 HYPRE_Int    ** num_grid_sweeps )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);

   return hypre_error_flag;
}

/* The "Get" function for SetRelaxType is GetCycleRelaxType. */
HYPRE_Int
hypre_MPAMGSetRelaxType_mp( void     *data,
                             HYPRE_Int      relax_type )
{
   HYPRE_Int i;
   HYPRE_Int *grid_relax_type;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (relax_type < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      hypre_ParAMGDataGridRelaxType(amg_data) = (HYPRE_Int *) (hypre_CAlloc_dbl (4, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST));
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
hypre_MPAMGSetCycleRelaxType_mp( void     *data,
                                  HYPRE_Int      relax_type,
                                  HYPRE_Int      k )
{
   HYPRE_Int i;
   HYPRE_Int *grid_relax_type;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      //hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (relax_type < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      grid_relax_type = (HYPRE_Int *) (hypre_CAlloc_dbl (4, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST));
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
hypre_MPAMGGetCycleRelaxType_mp( void     *data,
                                  HYPRE_Int    * relax_type,
                                  HYPRE_Int      k )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      //hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *relax_type = hypre_ParAMGDataGridRelaxType(amg_data)[k];

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetRelaxOrder_mp( void     *data,
                              HYPRE_Int       relax_order)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataRelaxOrder(amg_data) = relax_order;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetRelaxOrder_mp( void     *data,
                              HYPRE_Int     * relax_order)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *relax_order = hypre_ParAMGDataRelaxOrder(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetGridRelaxType_mp( void     *data,
                                 HYPRE_Int      *grid_relax_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!grid_relax_type)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data))
   {
      hypre_Free_dbl(hypre_ParAMGDataGridRelaxType(amg_data), HYPRE_MEMORY_HOST);
   }
   hypre_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = grid_relax_type[3];

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetGridRelaxType_mp( void     *data,
                                 HYPRE_Int    ** grid_relax_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetGridRelaxPoints_mp( void     *data,
                                   HYPRE_Int      **grid_relax_points )
{
   HYPRE_Int i;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!grid_relax_points)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxPoints(amg_data))
   {
      for (i = 0; i < 4; i++)
      {
         hypre_Free_dbl(hypre_ParAMGDataGridRelaxPoints(amg_data)[i], HYPRE_MEMORY_HOST);
      }
      hypre_Free_dbl(hypre_ParAMGDataGridRelaxPoints(amg_data), HYPRE_MEMORY_HOST);
   }
   hypre_ParAMGDataGridRelaxPoints(amg_data) = grid_relax_points;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetGridRelaxPoints_mp( void     *data,
                                   HYPRE_Int    *** grid_relax_points )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetRelaxWt_mp( void     *data,
                          hypre_double    relax_weight )
{
   HYPRE_Int i, num_levels;
   HYPRE_Real *relax_weight_array;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDataUserRelaxWeight(amg_data) = relax_weight;
   ((amg_data) -> user_relax_weight_dbl) = relax_weight;
   ((amg_data) -> user_relax_weight_flt) = (hypre_float) relax_weight;
   ((amg_data) -> user_relax_weight_ldbl) = (hypre_long_double) relax_weight;

   return hypre_error_flag;
}

/*HYPRE_Int
hypre_MPAMGSetLevelRelaxWt_mp( void    *data,
                                HYPRE_Real   relax_weight,
                                HYPRE_Int      level )
{
   HYPRE_Int i, num_levels;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1 || level < 0)
   {
      //hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      hypre_ParAMGDataRelaxWeight(amg_data) = (HYPRE_Real *) (hypre_CAlloc_dbl ((size_t)num_levels, (size_t)sizeof(HYPRE_Real), HYPRE_MEMORY_HOST));
      for (i = 0; i < num_levels; i++)
      {
         hypre_ParAMGDataRelaxWeight(amg_data)[i] = 1.0;
      }
   }

   hypre_ParAMGDataRelaxWeight(amg_data)[level] = relax_weight;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetLevelRelaxWt_mp( void    *data,
                                HYPRE_Real * relax_weight,
                                HYPRE_Int      level )
{
   HYPRE_Int num_levels;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1 || level < 0)
   {
      //hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *relax_weight = hypre_ParAMGDataRelaxWeight(amg_data)[level];

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetOmega_mp( void     *data,
                        HYPRE_Real   *omega )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!omega)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataOmega(amg_data))
   {
      hypre_Free_dbl(hypre_ParAMGDataOmega(amg_data), HYPRE_MEMORY_HOST);
   }
   hypre_ParAMGDataOmega(amg_data) = omega;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetOmega_mp( void     *data,
                         HYPRE_Real ** omega )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *omega = hypre_ParAMGDataOmega(amg_data);

   return hypre_error_flag;
}*/

HYPRE_Int
hypre_MPAMGSetOuterWt_mp( void     *data,
                          hypre_double    omega )
{
   HYPRE_Int i, num_levels;
   HYPRE_Real *omega_array;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   /*num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      hypre_ParAMGDataOmega(amg_data) = (HYPRE_Real *) (hypre_CAlloc_dbl ((size_t)num_levels, (size_t)sizeof(HYPRE_Real), HYPRE_MEMORY_HOST));
   }

   omega_array = hypre_ParAMGDataOmega(amg_data);
   for (i = 0; i < num_levels; i++)
   {
      omega_array[i] = omega;
   }*/
   
   ((amg_data) -> outer_wt_dbl) = omega;
   ((amg_data) -> outer_wt_flt) = (hypre_float) omega;
   ((amg_data) -> outer_wt_ldbl) = (hypre_long_double) omega;

   return hypre_error_flag;
}

/*HYPRE_Int
hypre_MPAMGSetLevelOuterWt_mp( void    *data,
                                HYPRE_Real   omega,
                                HYPRE_Int      level )
{
   HYPRE_Int i, num_levels;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1)
   {
      //hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      hypre_ParAMGDataOmega(amg_data) = (HYPRE_Real *) (hypre_CAlloc_dbl ((size_t)num_levels, (size_t)sizeof(HYPRE_Real), HYPRE_MEMORY_HOST));
      for (i = 0; i < num_levels; i++)
      {
         hypre_ParAMGDataOmega(amg_data)[i] = 1.0;
      }
   }

   hypre_ParAMGDataOmega(amg_data)[level] = omega;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetLevelOuterWt_mp( void    *data,
                                HYPRE_Real * omega,
                                HYPRE_Int      level )
{
   HYPRE_Int num_levels;
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1)
   {
      //hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *omega = hypre_ParAMGDataOmega(amg_data)[level];

   return hypre_error_flag;
}*/

HYPRE_Int
hypre_MPAMGSetLogging_mp( void     *data,
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
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataLogging(amg_data) = logging;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetLogging_mp( void     *data,
                           HYPRE_Int     * logging )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *logging = hypre_ParAMGDataLogging(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetPrintLevel_mp( void     *data,
                              HYPRE_Int print_level )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataPrintLevel(amg_data) = print_level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetPrintLevel_mp( void     *data,
                              HYPRE_Int * print_level )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *print_level =  hypre_ParAMGDataPrintLevel(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetPrintFileName_mp( void       *data,
                                 const char *print_file_name )
{
   hypre_ParAMGData  *amg_data =  (hypre_ParAMGData*)data;
   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if ( strlen(print_file_name) > 256 )
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_sprintf_dbl(hypre_ParAMGDataLogFileName(amg_data), "%s", print_file_name);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetPrintFileName_mp( void       *data,
                                 char ** print_file_name )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_sprintf_dbl( *print_file_name, "%s", hypre_ParAMGDataLogFileName(amg_data) );

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetNumIterations_mp( void    *data,
                                 HYPRE_Int      num_iterations )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNumIterations(amg_data) = num_iterations;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetDebugFlag_mp( void     *data,
                             HYPRE_Int       debug_flag )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataDebugFlag(amg_data) = debug_flag;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetDebugFlag_mp( void     *data,
                             HYPRE_Int     * debug_flag )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *debug_flag = hypre_ParAMGDataDebugFlag(amg_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetNumFunctions_mp( void     *data,
                                HYPRE_Int       num_functions )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (num_functions < 1)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNumFunctions(amg_data) = num_functions;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetNumFunctions_mp( void     *data,
                                HYPRE_Int     * num_functions )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *num_functions = hypre_ParAMGDataNumFunctions(amg_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicate whether to use nodal systems function
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetNodal_mp( void     *data,
                         HYPRE_Int    nodal )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNodal(amg_data) = nodal;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate number of levels for nodal coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetNodalLevels_mp( void     *data,
                               HYPRE_Int    nodal_levels )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNodalLevels(amg_data) = nodal_levels;

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * Indicate how to treat diag for primary matrix with  nodal systems function
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetNodalDiag_mp( void     *data,
                             HYPRE_Int    nodal )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNodalDiag(amg_data) = nodal;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate whether to discard same sign coefficients in S for nodal>0
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetKeepSameSign_mp( void      *data,
                                HYPRE_Int  keep_same_sign )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataKeepSameSign(amg_data) = keep_same_sign;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate the degree of aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetNumPaths_mp( void     *data,
                            HYPRE_Int       num_paths )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (num_paths < 1)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataNumPaths(amg_data) = num_paths;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of levels of aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetAggNumLevels_mp( void     *data,
                                HYPRE_Int       agg_num_levels )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_num_levels < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAggNumLevels(amg_data) = agg_num_levels;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the interpolation used with aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetAggInterpType_mp( void     *data,
                                 HYPRE_Int       agg_interp_type )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_interp_type < 0 || agg_interp_type > 9)
   {
      //hypre_error_in_arg(2);
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
hypre_MPAMGSetAggPMaxElmts_mp( void     *data,
                                HYPRE_Int       agg_P_max_elmts )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_P_max_elmts < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAggPMaxElmts(amg_data) = agg_P_max_elmts;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates max number of elements per row for 1st stage of aggressive
 * coarsening two-stage interpolation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetAggP12MaxElmts_mp( void     *data,
                                  HYPRE_Int       agg_P12_max_elmts )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_P12_max_elmts < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   hypre_ParAMGDataAggP12MaxElmts(amg_data) = agg_P12_max_elmts;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates truncation factor for aggressive coarsening interpolation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetAggTruncFactor_mp( void     *data,
                                 hypre_double  agg_trunc_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_trunc_factor < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   ((amg_data) -> agg_trunc_factor_dbl) = agg_trunc_factor;
   ((amg_data) -> agg_trunc_factor_flt) = (hypre_float) agg_trunc_factor;
   ((amg_data) -> agg_trunc_factor_ldbl) = (hypre_long_double) agg_trunc_factor;
   
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates truncation factor for 1 stage of aggressive coarsening
 * two stage interpolation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGSetAggP12TruncFactor_mp( void     *data,
                                    hypre_double  agg_P12_trunc_factor )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_P12_trunc_factor < 0)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   ((amg_data) -> agg_P12_trunc_factor_dbl) = agg_P12_trunc_factor;
   ((amg_data) -> agg_P12_trunc_factor_flt) = (hypre_float) agg_P12_trunc_factor;
   ((amg_data) -> agg_P12_trunc_factor_ldbl) = (hypre_long_double) agg_P12_trunc_factor;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetDofFunc_mp( void                 *data,
                           HYPRE_Int            *dof_func)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_IntArrayDestroy_dbl(hypre_ParAMGDataDofFunc(amg_data));
   /* NOTE: size and memory location of hypre_IntArray will be set during AMG setup */
   if (dof_func == NULL)
   {
      hypre_ParAMGDataDofFunc(amg_data) = NULL;
   }
   else
   {
      hypre_ParAMGDataDofFunc(amg_data) = hypre_IntArrayCreate_dbl(-1);
      hypre_IntArrayData(hypre_ParAMGDataDofFunc(amg_data)) = dof_func;
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetNumIterations_mp( void     *data,
                                 HYPRE_Int      *num_iterations )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *num_iterations = hypre_ParAMGDataNumIterations(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetCumNumIterations_mp( void     *data,
                                    HYPRE_Int      *cum_num_iterations )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
#ifdef CUMNUMIT
   *cum_num_iterations = hypre_ParAMGDataCumNumIterations(amg_data);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGGetResidual_mp( void * data, hypre_ParVector ** resid )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;
   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *resid = hypre_ParAMGDataResidual( amg_data );
   return hypre_error_flag;
}


HYPRE_Int
hypre_MPAMGGetRelResidualNorm_mp( void     *data,
                                   HYPRE_Real   *rel_resid_norm )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *rel_resid_norm = hypre_ParAMGDataRelativeResidualNorm(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetRAP2_mp( void      *data,
                        HYPRE_Int  rap2 )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataRAP2(amg_data) = rap2;
   return hypre_error_flag;
}


HYPRE_Int
hypre_MPAMGSetKeepTranspose_mp( void       *data,
                                 HYPRE_Int   keepTranspose)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataKeepTranspose(amg_data) = keepTranspose;
   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetCumNnzAP_mp( void         *data,
                           hypre_double  cum_nnz_AP )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParAMGDataCumNnzAP(amg_data) = cum_nnz_AP;

   return hypre_error_flag;
}


HYPRE_Int
hypre_MPAMGGetCumNnzAP_mp( void         *data,
                           hypre_double *cum_nnz_AP )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   if (!amg_data)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *cum_nnz_AP = hypre_ParAMGDataCumNnzAP(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPAMGSetPrecisionArray_mp( void       *data,
                                 HYPRE_Precision *precision_array)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) data;

   hypre_ParAMGDataPrecisionArray(amg_data) = precision_array;
   return hypre_error_flag;
}

#endif
