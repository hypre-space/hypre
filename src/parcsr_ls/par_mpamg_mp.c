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
//#include "hypre_parcsr_ls_mp.h"
#include "_hypre_parcsr_mv_mup.h"
#include "_hypre_parcsr_ls_mup.h"
//#include "HYPRE_seq_mv_mp.h"
#include "_hypre_seq_mv_mup.h"
#include "_hypre_utilities_mup.h"

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

   hypre_long_double   tol;

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
   hypre_sprintf(log_file_name, "%s", "amg.out.log");
   /* cycle_op_count = 0; */
   debug_flag = 0;

   rap2 = 0;
   keepT = 0;
   /*-----------------------------------------------------------------------
    * Create the hypre_ParAMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = hypre_CTAlloc(hypre_ParAMGData, 1, HYPRE_MEMORY_HOST);

   /* memory location will be reset at the setup */
   //hypre_ParAMGDataMemoryLocation(amg_data) = memory_location;
   hypre_ParAMGDataMemoryLocation(amg_data) = HYPRE_MEMORY_DEVICE;

   hypre_ParAMGDataMaxLevels(amg_data) =  max_levels;
   hypre_ParAMGDataUserRelaxType(amg_data) = -1;
   hypre_ParAMGDataUserNumSweeps(amg_data) = -1;
   hypre_BoomerAMGSetMaxCoarseSize(amg_data, max_coarse_size);
   hypre_BoomerAMGSetMinCoarseSize(amg_data, min_coarse_size);
   hypre_BoomerAMGSetCoarsenCutFactor(amg_data, coarsen_cut_factor);
   hypre_BoomerAMGSetSepWeight(amg_data, sep_weight);
   hypre_BoomerAMGSetMeasureType(amg_data, measure_type);
   hypre_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
   hypre_BoomerAMGSetInterpType(amg_data, interp_type);
   hypre_BoomerAMGSetPMaxElmts(amg_data, P_max_elmts);
   hypre_BoomerAMGSetAggPMaxElmts(amg_data, agg_P_max_elmts);
   hypre_BoomerAMGSetAggP12MaxElmts(amg_data, agg_P12_max_elmts);
   hypre_BoomerAMGSetNumFunctions(amg_data, num_functions);
   hypre_BoomerAMGSetNodal(amg_data, nodal);
   hypre_BoomerAMGSetNodalLevels(amg_data, nodal_levels);
   hypre_BoomerAMGSetNodalDiag(amg_data, nodal_diag);
   hypre_BoomerAMGSetKeepSameSign(amg_data, keep_same_sign);
   hypre_BoomerAMGSetNumPaths(amg_data, num_paths);
   hypre_BoomerAMGSetAggNumLevels(amg_data, agg_num_levels);
   hypre_BoomerAMGSetAggInterpType(amg_data, agg_interp_type);

   hypre_BoomerAMGSetMinIter(amg_data, min_iter);
   hypre_BoomerAMGSetMaxIter(amg_data, max_iter);
   hypre_BoomerAMGSetCycleType(amg_data, cycle_type);
   hypre_BoomerAMGSetFCycle(amg_data, fcycle);
   hypre_BoomerAMGSetTol(amg_data, tol);
   hypre_BoomerAMGSetNumSweeps(amg_data, num_sweeps);
   hypre_BoomerAMGSetCycleRelaxType(amg_data, relax_down, 1);
   hypre_BoomerAMGSetCycleRelaxType(amg_data, relax_up, 2);
   hypre_BoomerAMGSetCycleRelaxType(amg_data, relax_coarse, 3);
   hypre_BoomerAMGSetRelaxOrder(amg_data, relax_order);
   hypre_BoomerAMGSetNumIterations(amg_data, num_iterations);

   hypre_BoomerAMGSetStrongThreshold(amg_data,0.25);
   hypre_BoomerAMGSetMaxRowSum(amg_data,0.9);
   hypre_BoomerAMGSetTruncFactor(amg_data,0.0);
   hypre_BoomerAMGSetAggTruncFactor(amg_data,0.0);
   hypre_BoomerAMGSetAggP12TruncFactor(amg_data,0.0);
   hypre_BoomerAMGSetUserRelaxWeight(amg_data,1.0);
   hypre_BoomerAMGSetOuterWt(amg_data,1.0);

   hypre_BoomerAMGSetPrintLevel(amg_data, print_level);
   hypre_BoomerAMGSetLogging(amg_data, logging);
   hypre_BoomerAMGSetDebugFlag(amg_data, debug_flag);

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
         hypre_Free(hypre_ParAMGDataNumGridSweeps(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataNumGridSweeps(amg_data) = NULL;
      }
      if (grid_relax_type)
      {
         hypre_Free(hypre_ParAMGDataGridRelaxType(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataGridRelaxType(amg_data) = NULL;
      }
      if (hypre_ParAMGDataRelaxWeight(amg_data))
      {
         hypre_Free(hypre_ParAMGDataRelaxWeight(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataRelaxWeight(amg_data) = NULL;
      }
      if (hypre_ParAMGDataOmega(amg_data))
      {
         hypre_Free(hypre_ParAMGDataOmega(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataOmega(amg_data) = NULL;
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

         hypre_IntArrayDestroy(hypre_ParAMGDataCFMarkerArray(amg_data)[i - 1]);
      }
      if (hypre_ParAMGDataGridRelaxPoints(amg_data))
      {
         for (i = 0; i < 4; i++)
         {
            hypre_Free(hypre_ParAMGDataGridRelaxPoints(amg_data)[i], HYPRE_MEMORY_HOST);
         }
         hypre_Free(hypre_ParAMGDataGridRelaxPoints(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataGridRelaxPoints(amg_data) = NULL;
      }

      if (hypre_ParAMGDataL1Norms(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            hypre_SeqVectorDestroy(hypre_ParAMGDataL1Norms(amg_data)[i]);
         }
         hypre_Free(hypre_ParAMGDataL1Norms(amg_data), HYPRE_MEMORY_HOST);
      }

      /* see comments in par_coarsen.c regarding special case for CF_marker */
      if (num_levels == 1)
      {
         hypre_IntArrayDestroy(hypre_ParAMGDataCFMarkerArray(amg_data)[0]);
      }

      hypre_Free(hypre_ParAMGDataFArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_Free(hypre_ParAMGDataUArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_Free(hypre_ParAMGDataAArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_Free(hypre_ParAMGDataPArray(amg_data), HYPRE_MEMORY_HOST);
      hypre_Free(hypre_ParAMGDataCFMarkerArray(amg_data), HYPRE_MEMORY_HOST);
      //hypre_Free(hypre_ParAMGDataStrongThreshold(amg_data), HYPRE_MEMORY_HOST);
      //hypre_Free(hypre_ParAMGDataStrongThresholdR(amg_data), HYPRE_MEMORY_HOST);
      //hypre_Free(hypre_ParAMGDataFilterThresholdR(amg_data), HYPRE_MEMORY_HOST);
      //hypre_Free(hypre_ParAMGDataMaxRowSum(amg_data), HYPRE_MEMORY_HOST);
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
            hypre_IntArrayDestroy(hypre_ParAMGDataDofFuncArray(amg_data)[i]);
         }
         hypre_Free(hypre_ParAMGDataDofFuncArray(amg_data), HYPRE_MEMORY_HOST);
         hypre_ParAMGDataDofFuncArray(amg_data) = NULL;
      }
      hypre_ParVectorDestroy(hypre_ParAMGDataResidual(amg_data));
      hypre_ParAMGDataResidual(amg_data) = NULL;

      hypre_Free(amg_data, HYPRE_MEMORY_HOST);
   }
   HYPRE_ANNOTATE_FUNC_END;

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
