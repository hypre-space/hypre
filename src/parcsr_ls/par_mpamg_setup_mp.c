/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "_hypre_parcsr_mv_mup.h"
//#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls_mup.h"
//#include "HYPRE_parcsr_ls_mp.h"
#include "_hypre_utilities_mup.h"
#include "_hypre_utilities.h"

#define DEBUG 0
#define PRINT_CF 0

#define DEBUG_SAVE_ALL_OPS 0
/*****************************************************************************
 *
 * Routine for driving the setup phase of AMG
 *
 *****************************************************************************/

/*****************************************************************************
 * hypre_MPAMGSetup
 *****************************************************************************/
#ifdef HYPRE_MIXED_PRECISION

HYPRE_Int
hypre_MPAMGSetup_mp( void               *amg_vdata,
                     hypre_ParCSRMatrix *A,
                     hypre_ParVector    *f,
                     hypre_ParVector    *u )
{
   MPI_Comm            comm = hypre_ParCSRMatrixComm(A);
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */
   HYPRE_Int            num_vectors = 1;
   hypre_ParCSRMatrix **A_array = NULL;
   hypre_ParVector    **F_array = NULL;
   hypre_ParVector    **U_array = NULL;
   hypre_ParCSRMatrix **P_array = NULL;
   hypre_ParVector     *Residual_array = NULL;
   hypre_ParVector     *Vtemp_dbl = NULL;
   hypre_ParVector     *Vtemp_flt = NULL;
   hypre_ParVector     *Vtemp_long_dbl = NULL;
   hypre_ParVector     *Ztemp_dbl = NULL;
   hypre_ParVector     *Ztemp_flt = NULL;
   hypre_ParVector     *Ztemp_long_dbl = NULL;
   hypre_IntArray     **CF_marker_array = NULL;
   hypre_IntArray     **dof_func_array = NULL;
   hypre_IntArray      *dof_func = NULL;
   HYPRE_Int           *dof_func_data = NULL;
   HYPRE_Int            coarsen_cut_factor;
   HYPRE_Int            useSabs;
   HYPRE_Int            relax_order;
   HYPRE_Int            max_levels;
   HYPRE_Int            amg_logging;
   HYPRE_Int            amg_print_level;
   HYPRE_Int            debug_flag;
   HYPRE_Int            local_num_vars;
   HYPRE_Int            P_max_elmts;
   HYPRE_Int            agg_P_max_elmts;
   HYPRE_Int            agg_P12_max_elmts;
   HYPRE_Int            keep_same_sign = hypre_ParAMGDataKeepSameSign(amg_data);
   HYPRE_Precision     *precision_array = hypre_ParAMGDataPrecisionArray(amg_data);
   HYPRE_Int           *precision_type;
   HYPRE_Precision      level_precision = precision_array[0];

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_ParAMGDataMemoryLocation(amg_data) = memory_location;

   /* Local variables */
   hypre_IntArray      *CF_marker = NULL;
   HYPRE_Int           *CF_marker_data = NULL;
   hypre_IntArray      *CFN_marker = NULL;
   hypre_IntArray      *CF2_marker = NULL;
   hypre_IntArray      *CFN2_marker = NULL;
   hypre_ParCSRMatrix  *S = NULL, *Sabs = NULL;
   hypre_ParCSRMatrix  *S2;
   hypre_ParCSRMatrix  *SN = NULL;
   hypre_ParCSRMatrix  *P = NULL;
   hypre_ParCSRMatrix  *R = NULL;
   hypre_ParCSRMatrix  *A_H;
   hypre_ParCSRMatrix  *AN = NULL;
   hypre_ParCSRMatrix  *P1 = NULL;
   hypre_ParCSRMatrix  *P2 = NULL;
   hypre_Vector    **l1_norms = NULL;

   HYPRE_Int       num_levels;
   HYPRE_Int       level=0;
   HYPRE_Int       local_size, i, row;
   HYPRE_BigInt    coarse_size;
   HYPRE_Int       coarsen_type;
   HYPRE_Int       measure_type;
   HYPRE_BigInt    fine_size;
   HYPRE_Int       offset;
   HYPRE_Real      size;
   HYPRE_Int       not_finished_coarsening = 1;
   HYPRE_Int       coarse_threshold = hypre_ParAMGDataMaxCoarseSize(amg_data);
   HYPRE_Int       min_coarse_size = hypre_ParAMGDataMinCoarseSize(amg_data);
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
   HYPRE_Int       three = 3;

   HYPRE_Int     nlevel;
   HYPRE_Int     max_nz_per_row;

   HYPRE_Int interp_type, restri_type;

   HYPRE_Int       rap2 = hypre_ParAMGDataRAP2(amg_data);
   HYPRE_Int       keepTranspose = hypre_ParAMGDataKeepTranspose(amg_data);

   HYPRE_Int       local_coarse_size;

   HYPRE_Int      *num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);
   HYPRE_Int       ns = num_grid_sweeps[1];
   HYPRE_Real      wall_time;   /* for debugging instrumentation */

   char            nvtx_name[1024];

   hypre_double    cum_nnz_AP = hypre_ParAMGDataCumNnzAP(amg_data);

   HYPRE_Real strong_threshold;
   HYPRE_Real max_row_sum;
   HYPRE_Real trunc_factor, agg_trunc_factor, agg_P12_trunc_factor;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);

   max_levels = hypre_ParAMGDataMaxLevels(amg_data);
   amg_logging = hypre_ParAMGDataLogging(amg_data);
   amg_print_level = hypre_ParAMGDataPrintLevel(amg_data);
   coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);
   measure_type = hypre_ParAMGDataMeasureType(amg_data);
   debug_flag = hypre_ParAMGDataDebugFlag(amg_data);
   interp_type = hypre_ParAMGDataInterpType(amg_data);
   restri_type = hypre_ParAMGDataRestriction(amg_data); /* RL */

   relax_order = hypre_ParAMGDataRelaxOrder(amg_data);
   level_precision = precision_array[0];

   hypre_ParCSRMatrixSetNumNonzeros_pre(level_precision, A);
   hypre_ParCSRMatrixSetDNumNonzeros_pre(level_precision, A);
   hypre_ParAMGDataNumVariables(amg_data) = hypre_ParCSRMatrixNumRows(A);

   S = NULL;

   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);
   dof_func_array = hypre_ParAMGDataDofFuncArray(amg_data);
   dof_func = hypre_ParAMGDataDofFunc(amg_data);
   local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   /* set size of dof_func hypre_IntArray if necessary */
   if (dof_func && hypre_IntArraySize(dof_func) < 0)
   {
      hypre_IntArraySize(dof_func) = local_size;
      hypre_IntArrayMemoryLocation(dof_func) = memory_location;
   }

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

   if (A_array == NULL)
   {
      A_array = (hypre_ParCSRMatrix **) hypre_CAlloc_dbl((size_t)(max_levels), (size_t)sizeof(hypre_ParCSRMatrix *), HYPRE_MEMORY_HOST);
   }
   A_array[0] = A;

   if (P_array == NULL && max_levels > 1)
   {
      P_array = (hypre_ParCSRMatrix **) hypre_CAlloc_dbl((size_t)(max_levels-1), (size_t)sizeof(hypre_ParCSRMatrix *), HYPRE_MEMORY_HOST);
   }

   if (CF_marker_array == NULL)
   {
      CF_marker_array = (hypre_IntArray **) hypre_CAlloc_dbl((size_t)(max_levels), (size_t)sizeof(hypre_IntArray *), HYPRE_MEMORY_HOST);
   }

   if (dof_func_array == NULL)
   {
      dof_func_array = (hypre_IntArray **) hypre_CAlloc_dbl((size_t)(max_levels), (size_t)sizeof(hypre_IntArray *), HYPRE_MEMORY_HOST);
   }

   if (num_functions > 1 && dof_func == NULL)
   {
      dof_func = hypre_IntArrayCreate_pre(level_precision, local_size);
      hypre_IntArrayInitialize_v2_pre( level_precision, dof_func, memory_location);
      offset = (HYPRE_Int) ( hypre_ParCSRMatrixRowStarts(A)[0] % ((HYPRE_BigInt) num_functions) );

      for (i = 0; i < local_size; i++)
      {
         hypre_IntArrayData(dof_func)[i] = (i + offset) % num_functions;
      }
   }

   dof_func_array[0] = dof_func;
   hypre_ParAMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_ParAMGDataDofFunc(amg_data) = dof_func;
   hypre_ParAMGDataDofFuncArray(amg_data) = dof_func_array;
   hypre_ParAMGDataAArray(amg_data) = A_array;
   hypre_ParAMGDataPArray(amg_data) = P_array;

   coarsen_cut_factor = hypre_ParAMGDataCoarsenCutFactor(amg_data);
   useSabs = hypre_ParAMGDataSabs(amg_data);
   trunc_factor = hypre_ParAMGDataTruncFactor(amg_data);
   agg_trunc_factor = hypre_ParAMGDataAggTruncFactor(amg_data);
   agg_P12_trunc_factor = hypre_ParAMGDataAggP12TruncFactor(amg_data);

   P_max_elmts = hypre_ParAMGDataPMaxElmts(amg_data);
   agg_P_max_elmts = hypre_ParAMGDataAggPMaxElmts(amg_data);
   agg_P12_max_elmts = hypre_ParAMGDataAggP12MaxElmts(amg_data);

   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);

   if (F_array == NULL)
   {
      F_array = (hypre_ParVector **) hypre_CAlloc_dbl((size_t)(max_levels), (size_t)sizeof(hypre_ParVector *), HYPRE_MEMORY_HOST);
   }
   if (U_array == NULL)
   {
      U_array = (hypre_ParVector **) hypre_CAlloc_dbl((size_t)(max_levels), (size_t)sizeof(hypre_ParVector *), HYPRE_MEMORY_HOST);
   }

   F_array[0] = f;
   U_array[0] = u;

   hypre_ParAMGDataFArray(amg_data) = F_array;
   hypre_ParAMGDataUArray(amg_data) = U_array;

   precision_type = (HYPRE_Int *) hypre_CAlloc_dbl((size_t)(three), (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
   for (i=0; i< 3; i++) precision_type[i] = 0;

   hypre_ParAMGDataPrecisionType(amg_data) = precision_type;

   for (i=0; i< max_levels; i++)
   {
      if (precision_array[i] == HYPRE_REAL_DOUBLE)
      {
         precision_type[0] = 1; 
      }
      else if (precision_array[i] == HYPRE_REAL_SINGLE)
      {
         precision_type[1] = 1; 
      }
      else if (precision_array[i] == HYPRE_REAL_LONGDOUBLE)
      {
         precision_type[2] = 1; 
      }
   }

   if (precision_type[0])
   {
      Vtemp_dbl = hypre_ParVectorCreate_dbl(hypre_ParCSRMatrixComm(A_array[0]),
                                            hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                            hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize_v2_dbl(Vtemp_dbl, memory_location);
      hypre_ParVectorNumVectors(Vtemp_dbl) = num_vectors;
      hypre_ParAMGDataVtempDBL(amg_data) = Vtemp_dbl;
      Ztemp_dbl = hypre_ParVectorCreate_dbl(hypre_ParCSRMatrixComm(A_array[0]),
                                            hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                            hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize_v2_dbl(Ztemp_dbl, memory_location);
      hypre_ParVectorNumVectors(Ztemp_dbl) = num_vectors;
      hypre_ParAMGDataZtempDBL(amg_data) = Ztemp_dbl;
   }
   if (precision_type[1])
   {
      Vtemp_flt = hypre_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A_array[0]),
                                            hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                            hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize_v2_flt(Vtemp_flt, memory_location);
      hypre_ParVectorNumVectors(Vtemp_flt) = num_vectors;
      hypre_ParAMGDataVtempFLT(amg_data) = Vtemp_flt;
      Ztemp_flt = hypre_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A_array[0]),
                                            hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                            hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize_v2_flt(Ztemp_flt, memory_location);
      hypre_ParVectorNumVectors(Ztemp_flt) = num_vectors;
      hypre_ParAMGDataZtempFLT(amg_data) = Ztemp_flt;
   }
   if (precision_type[2])
   {
      Vtemp_long_dbl = hypre_ParVectorCreate_long_dbl(hypre_ParCSRMatrixComm(A_array[0]),
                                            hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                            hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize_v2_long_dbl(Vtemp_long_dbl, memory_location);
      hypre_ParVectorNumVectors(Vtemp_long_dbl) = num_vectors;
      hypre_ParAMGDataVtempLONGDBL(amg_data) = Vtemp_long_dbl;
      Ztemp_long_dbl = hypre_ParVectorCreate_long_dbl(hypre_ParCSRMatrixComm(A_array[0]),
                                            hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                            hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize_v2_long_dbl(Ztemp_long_dbl, memory_location);
      hypre_ParVectorNumVectors(Ztemp_long_dbl) = num_vectors;
      hypre_ParAMGDataZtempLONGDBL(amg_data) = Ztemp_long_dbl;
   }

   /*-----------------------------------------------------
    *  Enter Coarsening Loop
    *-----------------------------------------------------*/ 
   
   while (not_finished_coarsening)
   {
      level_precision = precision_array[level];
      /* only do nodal coarsening on a fixed number of levels */
      if (level >= nodal_levels)
      {
         nodal = 0;
      }

      {
         fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
      }

      if (level > 0)
      {
         F_array[level] =
               hypre_ParVectorCreate_pre(level_precision, hypre_ParCSRMatrixComm(A_array[level]),
                                         hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                         hypre_ParCSRMatrixRowStarts(A_array[level]));
         hypre_ParVectorNumVectors(F_array[level]) = num_vectors;
         hypre_ParVectorInitialize_v2_pre(level_precision, F_array[level], memory_location);

         U_array[level] =
               hypre_ParVectorCreate_pre(level_precision, hypre_ParCSRMatrixComm(A_array[level]),
                                         hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                         hypre_ParCSRMatrixRowStarts(A_array[level]));
         hypre_ParVectorNumVectors(U_array[level]) = num_vectors;
         hypre_ParVectorInitialize_v2_pre(level_precision, U_array[level], memory_location);
         hypre_ParVectorPrecision(F_array[level]) = level_precision;
         hypre_ParVectorPrecision(U_array[level]) = level_precision;
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

      if (max_levels == 1)
      {
         S = NULL;
         local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
         CF_marker_array[level] = hypre_IntArrayCreate_pre(level_precision, local_size);
         hypre_IntArrayInitialize_pre(level_precision, CF_marker_array[level]);
         hypre_IntArraySetConstantValues_pre(level_precision, CF_marker_array[level], 1);
         coarse_size = fine_size;
      }
      else /* max_levels > 1 */
      {
	 hypre_BoomerAMGGetStrongThreshold_pre(level_precision, amg_data, &strong_threshold);
	 hypre_BoomerAMGGetMaxRowSum_pre(level_precision, amg_data, &max_row_sum);
	 hypre_Strength_Options_pre(level_precision, A_array[level], strong_threshold, 
			            max_row_sum,
	          	            num_functions, nodal, nodal_diag, 
			  	    useSabs, dof_func_data, &S);
	 if (nodal) 
         {
            SN = S;
            S = NULL;
	 }
         local_num_vars =
            hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
 
         if (nodal == 0) /* no nodal coarsening , CF_marker plain, CF2_marker aggressive */
         {
            hypre_Coarsen_Options_pre(level_precision, S, A_array[level], level, debug_flag, coarsen_type, 
   	  	                      measure_type, coarsen_cut_factor, agg_num_levels, 
			              num_paths, local_num_vars, dof_func_array[level],
                                      coarse_pnts_global1, &CF2_marker, &CF_marker);
         }
         else if (nodal > 0) /* CFN_marker plain, CFN2_marker aggressive */
         {
            hypre_Coarsen_Options_pre(level_precision, SN, SN, level, debug_flag, coarsen_type, measure_type,
                                      coarsen_cut_factor, agg_num_levels, num_paths, 
			              (local_num_vars /num_functions), dof_func_array[level],
                                      coarse_pnts_global1, &CFN2_marker, &CFN_marker);
            if (agg_num_levels <= level) /* no aggressive coarsening */
            {
               hypre_BoomerAMGCreateScalarCFS_pre(level_precision, SN, A_array[level], 
			                          hypre_IntArrayData(CFN_marker),
                                                  num_functions, nodal, keep_same_sign,
                                                  &dof_func, &CF_marker, &S);
               hypre_IntArrayDestroy_pre(level_precision, CFN_marker);
               hypre_ParCSRMatrixDestroy_pre(level_precision, SN);
            }
         }

         /*****xxxxxxxxxxxxx changes for min_coarse_size */
         /* here we will determine the coarse grid size to be able to
            determine if it is not smaller than requested minimal size */

         if (level >= agg_num_levels) /* no aggressive coarsening */
         {
	    CF_marker_array[level] = CF_marker; /* done plain for nodal and scalar */
            hypre_BoomerAMGCoarseParms_pre(level_precision, comm, local_num_vars,
                                           num_functions, dof_func_array[level], 
					   CF_marker,
                                           &coarse_dof_func, coarse_pnts_global);
            if (my_id == num_procs - 1)
            {
               coarse_size = coarse_pnts_global[1];
            }
            MPI_Bcast(&coarse_size, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

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
               hypre_ParCSRMatrixDestroy_pre(level_precision, S); 
               hypre_ParCSRMatrixDestroy_pre(level_precision, SN); 
               if (level > 0)
               {
               /* note special case treatment of CF_marker is necessary
                * to do CF relaxation correctly when num_levels = 1 */
                  hypre_IntArrayDestroy_pre(level_precision, CF_marker_array[level]);
                  hypre_ParVectorDestroy_pre(level_precision, F_array[level]);
                  hypre_ParVectorDestroy_pre(level_precision, U_array[level]);
               }
               coarse_size = fine_size;

               hypre_ParCSRMatrixDestroy_pre(level_precision, Sabs);
               hypre_IntArrayDestroy_pre(level_precision, coarse_dof_func);
            }
   
            if (coarse_size < min_coarse_size)
            {
               hypre_ParCSRMatrixDestroy_pre(level_precision, S);
               hypre_ParCSRMatrixDestroy_pre(level_precision, SN);
               if (num_functions > 1)
               {
                  hypre_IntArrayDestroy_pre(level_precision, coarse_dof_func);
               }
               hypre_IntArrayDestroy_pre(level_precision, CF_marker_array[level]);
               if (level > 0)
               {
                  hypre_ParVectorDestroy_pre(level_precision, F_array[level]);
                  hypre_ParVectorDestroy_pre(level_precision, U_array[level]);
               }
               coarse_size = fine_size;
               hypre_ParCSRMatrixDestroy_pre(level_precision, Sabs);
            }
         }
   
         if (level < agg_num_levels) /* aggressive coarsening */
         {
            if (nodal == 0)
            {
               if (agg_interp_type == 4 || agg_interp_type == 8 || agg_interp_type == 9)
               {
                  hypre_BoomerAMGCorrectCFMarker_pre(level_precision, CF_marker, CF2_marker);
                  hypre_IntArrayDestroy_pre(level_precision, CF2_marker);
                  hypre_MPassInterp_Options_pre(level_precision, A_array[level], S, CF_marker, 
         		                        dof_func_array[level], coarse_pnts_global, 
         	                                agg_interp_type, num_functions, 
      				                debug_flag, agg_P_max_elmts, agg_trunc_factor, 
      				                sep_weight, &P);
               }
      	       else
      	       {
                  hypre_StageOneInterp_Options_pre(level_precision, A_array[level], S, CF_marker, 
         		                           coarse_pnts_global1,
         	                                   dof_func_data, agg_interp_type, num_functions, 
      				                   debug_flag, agg_P12_max_elmts, 
      				                   agg_P12_trunc_factor, &P1);
      	          hypre_BoomerAMGCorrectCFMarker2_pre (level_precision, CF_marker, CF2_marker);
                  hypre_IntArrayDestroy_pre(level_precision, CF2_marker);
                  hypre_BoomerAMGCoarseParms_pre(level_precision, comm, local_num_vars, num_functions, 
      			                         dof_func_array[level], CF_marker,
                                                 &coarse_dof_func, coarse_pnts_global);
                  hypre_StageTwoInterp_Options_pre(level_precision, A_array[level], P1, S, CF_marker, 
      			                           coarse_pnts_global, 
      					           coarse_pnts_global1, dof_func_data, 
      					           agg_interp_type, num_functions, 
      				                   debug_flag, sep_weight, agg_P_max_elmts, 
      				                   agg_P12_max_elmts, agg_trunc_factor, 
      				                   agg_P12_trunc_factor, &P);
      	       }
      	    CF_marker_array[level] = CF_marker; /*done aggressive coarsening scalar */
            }
            else if (nodal > 0)
            {
               if (agg_interp_type == 4 || agg_interp_type == 8 || agg_interp_type == 9)
               {
                  hypre_BoomerAMGCorrectCFMarker_pre(level_precision, CFN_marker, CFN2_marker);
                  hypre_IntArrayDestroy_pre(level_precision, CFN2_marker);
                  hypre_BoomerAMGCreateScalarCFS_pre(level_precision, SN, A_array[level], 
      			                             hypre_IntArrayData(CFN_marker),
                                                     num_functions, nodal, keep_same_sign,
                                                     &dof_func, &(CF_marker_array[level]), &S);
                  hypre_IntArrayDestroy_pre(level_precision, CFN_marker);
                  hypre_MPassInterp_Options_pre(level_precision, A_array[level], SN, CF_marker_array[level], 
         		                        dof_func_array[level], coarse_pnts_global, 
         	                                agg_interp_type, num_functions, 
      				                debug_flag, agg_P_max_elmts, agg_trunc_factor, 
      				                sep_weight, &P);
              }
              else 
              {
                  hypre_BoomerAMGCreateScalarCFS_pre(level_precision, SN, A, hypre_IntArrayData(CFN_marker),
                                                     num_functions, nodal, keep_same_sign,
                                                     &dof_func, &CF_marker, &S);
                  for (i = 0; i < 2; i++)
                  {
                     coarse_pnts_global1[i] *= num_functions;
                  }
                  hypre_StageOneInterp_Options_pre(level_precision, A_array[level], S, CF_marker, 
         		                           coarse_pnts_global1, dof_func_data, 
      					           agg_interp_type, num_functions, debug_flag, 
         		                           agg_P12_max_elmts, agg_P12_trunc_factor, 
      					           &P1);
      
                  hypre_BoomerAMGCorrectCFMarker2_pre (level_precision, CFN_marker, CFN2_marker);
                  hypre_IntArrayDestroy_pre(level_precision, CFN2_marker);
                  hypre_ParCSRMatrixDestroy_pre(level_precision, S);
                  hypre_BoomerAMGCreateScalarCFS_pre(level_precision, SN, A_array[level], 
      			                             hypre_IntArrayData(CFN_marker),
                                                     num_functions, nodal, keep_same_sign,
                                                     &dof_func, &(CF_marker_array[level]), &S);
      
                  hypre_IntArrayDestroy_pre(level_precision, CFN_marker);
                  hypre_BoomerAMGCoarseParms_pre(level_precision, comm, local_num_vars,
                                                 num_functions, dof_func_array[level], 
      					         CF_marker_array[level],
      					         &coarse_dof_func, coarse_pnts_global);
      
                  hypre_StageTwoInterp_Options_pre(level_precision, A_array[level], P1, S, CF_marker_array[level], 
         		                           coarse_pnts_global, coarse_pnts_global1,
         	                                   dof_func_data, agg_interp_type, num_functions, 
      				                   debug_flag, sep_weight,
         		                           agg_P_max_elmts, agg_P12_max_elmts,
         		                           agg_trunc_factor, agg_P12_trunc_factor, &P);
      
                  hypre_ParCSRMatrixDestroy_pre(level_precision, SN);
                  hypre_ParCSRMatrixDestroy_pre(level_precision, AN);
                  /* done aggressive coarsening nodal */
               }
            }
            if (my_id == (num_procs - 1))
            {
               coarse_size = coarse_pnts_global[1];
            }
            MPI_Bcast(&coarse_size, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
         }
         else /* no aggressive coarsening */
         {
            hypre_Interp_Options_pre(level_precision, A_array[level], S, CF_marker_array[level], coarse_pnts_global, 
      			                dof_func_data, interp_type, num_functions, debug_flag,
      				        P_max_elmts, trunc_factor, sep_weight, &P);
         } /* end of no aggressive coarsening */

         dof_func_array[level + 1] = NULL;
         if (num_functions > 1 && nodal > -1 )
         {
            dof_func_array[level + 1] = coarse_dof_func;
         }
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

         hypre_ParCSRMatrixDestroy_pre(level_precision, S);
         hypre_ParCSRMatrixDestroy_pre(level_precision, P);
         if (level > 0)
         {
         /* note special case treatment of CF_marker is necessary
          * to do CF relaxation correctly when num_levels = 1 */
            hypre_IntArrayDestroy_pre(level_precision, CF_marker_array[level]);
            CF_marker_array[level] = NULL;
            hypre_ParVectorDestroy_pre(level_precision, F_array[level]);
            hypre_ParVectorDestroy_pre(level_precision, U_array[level]);
         }
         if (level + 1 < max_levels)
         {
            hypre_IntArrayDestroy_pre(level_precision, dof_func_array[level + 1]);
            dof_func_array[level + 1] = NULL;
         }
      }
      if (level < agg_num_levels && coarse_size < min_coarse_size)
      {
         hypre_ParCSRMatrixDestroy_pre(level_precision, S);
         hypre_ParCSRMatrixDestroy_pre(level_precision, P);
         if (level > 0)
         {
            hypre_IntArrayDestroy_pre(level_precision, CF_marker_array[level]);
            CF_marker_array[level] = NULL;
            hypre_ParVectorDestroy_pre(level_precision, F_array[level]);
            hypre_ParVectorDestroy_pre(level_precision, U_array[level]);
         }
         hypre_IntArrayDestroy_pre(level_precision, dof_func_array[level + 1]);
         dof_func_array[level + 1] = NULL;
         coarse_size = fine_size;

         break;
      }

      /*-------------------------------------------------------------
       * Build prolongation matrix, P, and place in P_array[level]
       *--------------------------------------------------------------*/


      dof_func_data = NULL;
      if (dof_func_array[level + 1])
      {
         dof_func_data = hypre_IntArrayData(dof_func_array[level + 1]);
      }

      hypre_ParCSRMatrixPrecision(P) = level_precision;

      hypre_ParCSRMatrixDestroy_pre(level_precision, S);

      P_array[level] = P;

      /*-------------------------------------------------------------
       * Build coarse-grid operator, A_array[level+1] by R*A*P
       *--------------------------------------------------------------*/

      if (rap2)
      {
         hypre_ParCSRMatrix *Q = NULL;
         Q = hypre_ParMatmul_pre(level_precision, A_array[level], P_array[level]);
         A_H = hypre_ParTMatmul_pre(level_precision, P_array[level], Q);
         if (num_procs > 1)
         {
            hypre_MatvecCommPkgCreate_pre(level_precision, A_H);
         }
         /* Delete AP */
         hypre_ParCSRMatrixDestroy_pre(level_precision, Q);
      }
      else
      {
         hypre_BoomerAMGBuildCoarseOperatorKT_pre(level_precision, P_array[level], A_array[level],
                                                  P_array[level], keepTranspose, &A_H);
      }
      if (num_procs > 1 && hypre_ParCSRMatrixCommPkg(A_H) == NULL)
      {
         hypre_MatvecCommPkgCreate_pre(level_precision, A_H);
      }

      ++level;

      hypre_ParCSRMatrixPrecision(A_H) = level_precision;

      if (level_precision != precision_array[level])
      {
	 hypre_ParCSRMatrixConvert_mp(A_H, precision_array[level]);
      } 
      A_array[level] = A_H;

      size = ((HYPRE_Real) fine_size ) * .75;
      if (coarsen_type > 0 && coarse_size >= (HYPRE_BigInt) size)
      {
         coarsen_type = 0;
      }
      if ( (level == max_levels - 1) || (coarse_size <= (HYPRE_BigInt) coarse_threshold ))
      {
         not_finished_coarsening = 0;
      }
   }  /* end of coarsening loop: while (not_finished_coarsening) */


   if (grid_relax_type[3] == 9   ||
       grid_relax_type[3] == 19  ||
       grid_relax_type[3] == 98  ||
       grid_relax_type[3] == 99  ||
       grid_relax_type[3] == 198 ||
       grid_relax_type[3] == 199)
   {
      /* Gaussian elimination on the coarsest level */
      if (coarse_size <= coarse_threshold)
      {
         hypre_GaussElimSetup_pre(precision_array[level], amg_data, level, grid_relax_type[3]);
      }
      else
      {
         grid_relax_type[3] = grid_relax_type[1];
      }
   }

   if (level > 0)
   {
      F_array[level] =
      hypre_ParVectorCreate_pre(level_precision, hypre_ParCSRMatrixComm(A_array[level]),
                                hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                hypre_ParCSRMatrixRowStarts(A_array[level]));
      hypre_ParVectorNumVectors(F_array[level]) = num_vectors;
      hypre_ParVectorInitialize_v2_pre(level_precision, F_array[level], memory_location);

      U_array[level] =
      hypre_ParVectorCreate_pre(level_precision, hypre_ParCSRMatrixComm(A_array[level]),
                                hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                hypre_ParCSRMatrixRowStarts(A_array[level]));
      hypre_ParVectorNumVectors(U_array[level]) = num_vectors;
      hypre_ParVectorInitialize_v2_pre(level_precision, U_array[level], memory_location);
   }

   /*-----------------------------------------------------------------------
    * enter all the stuff created, A[level], P[level], CF_marker[level],
    * for levels 1 through coarsest, into amg_data data structure
    *-----------------------------------------------------------------------*/

   num_levels = level + 1;
   hypre_ParAMGDataNumLevels(amg_data) = num_levels;

   /*-----------------------------------------------------------------------
    * Setup of special smoothers when needed
    *-----------------------------------------------------------------------*/
   if (grid_relax_type[1] ==  7 || grid_relax_type[2] ==  7 || grid_relax_type[3] ==  7 ||
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
      l1_norms = (hypre_Vector **)hypre_CAlloc_dbl((size_t)(num_levels), (size_t)sizeof(hypre_Vector *), HYPRE_MEMORY_HOST);
      hypre_ParAMGDataL1Norms(amg_data) = l1_norms;
   }
   for (j = 0; j < num_levels; j++)
   {
      hypre_Vector *l1_norms_tmp = NULL;
      hypre_Level_L1Norms_pre(precision_array[j], A_array[j], CF_marker_array[j], 
	                      grid_relax_type, j, num_levels, relax_order, &l1_norms_tmp);
      hypre_VectorPrecision(l1_norms_tmp) = precision_array[j];
      l1_norms[j] = l1_norms_tmp;
      l1_norms_tmp = NULL;
   } /* end of levels loop */


   if (amg_logging > 1)
   {
      Residual_array = hypre_ParVectorCreate_pre(precision_array[0], hypre_ParCSRMatrixComm(A_array[0]),
                                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                                 hypre_ParCSRMatrixRowStarts(A_array[0]) );
      hypre_ParVectorInitialize_v2_pre(precision_array[0], Residual_array, memory_location);
      hypre_ParAMGDataResidual(amg_data) = Residual_array;
   }
   else
   {
      hypre_ParAMGDataResidual(amg_data) = NULL;
   }

   if (cum_nnz_AP > 0.0)
   {
      cum_nnz_AP = hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
      for (j = 0; j < num_levels - 1; j++)
      {
         hypre_ParCSRMatrixSetDNumNonzeros_pre(precision_array[j], P_array[j]);
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
      hypre_MPAMGSetupStats_mp(amg_data, A);
   }
   return hypre_error_flag;
}

#endif


