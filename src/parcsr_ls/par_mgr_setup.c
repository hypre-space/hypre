/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_mgr.h"
#include "par_amg.h"

/* Setup MGR data */
HYPRE_Int
hypre_MGRSetup( void               *mgr_vdata,
                hypre_ParCSRMatrix *A,
                hypre_ParVector    *f,
                hypre_ParVector    *u )
{
   MPI_Comm           comm = hypre_ParCSRMatrixComm(A);
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   HYPRE_Int       i, j, final_coarse_size = 0, block_size, idx, **block_cf_marker;
   HYPRE_Int       *block_num_coarse_indexes, *point_marker_array;
   HYPRE_BigInt    row, end_idx;
   HYPRE_Int    lev, num_coarsening_levs, last_level;
   HYPRE_Int    num_c_levels = 0, nc, index_i, cflag;
   HYPRE_Int      set_c_points_method;
   HYPRE_Int    debug_flag = 0;
   HYPRE_Int    block_jacobi_bsize;
   HYPRE_Int    *blk_size = mgr_data -> blk_size;
   HYPRE_Int    level_blk_size;

   hypre_ParCSRMatrix  *RT = NULL;
   hypre_ParCSRMatrix  *R  = NULL;
   hypre_ParCSRMatrix  *P = NULL;
   hypre_ParCSRMatrix  *S = NULL;
   hypre_ParCSRMatrix  *ST = NULL;
   hypre_ParCSRMatrix  *AT = NULL;
   hypre_ParCSRMatrix  *Wp = NULL;
   hypre_ParCSRMatrix  *Wr = NULL;
   hypre_ParCSRMatrix  *AP = NULL;

   HYPRE_Int           *dof_func_buff_data = NULL;
   HYPRE_BigInt         coarse_pnts_global[2]; // TODO: Change to row_starts_cpts
   HYPRE_BigInt         row_starts_fpts[2];
   hypre_Vector       **l1_norms = NULL;
   HYPRE_Real          *l1_norms_data;

   hypre_ParVector     *Ztemp;
   hypre_ParVector     *Vtemp;
   hypre_ParVector     *Utemp;
   hypre_ParVector     *Ftemp;

   /* pointers to mgr data */
   HYPRE_Int  use_default_cgrid_solver = (mgr_data -> use_default_cgrid_solver);
   HYPRE_Int  logging = (mgr_data -> logging);
   HYPRE_Int  print_level = (mgr_data -> print_level);
   HYPRE_Int  relax_type = (mgr_data -> relax_type);
   HYPRE_Int  relax_order = (mgr_data -> relax_order);
   HYPRE_Int  *interp_type = (mgr_data -> interp_type);
   HYPRE_Int  *restrict_type = (mgr_data -> restrict_type);
   HYPRE_Int  *num_relax_sweeps = (mgr_data -> num_relax_sweeps);
   HYPRE_Int num_interp_sweeps = (mgr_data -> num_interp_sweeps);
   //HYPRE_Int num_restrict_sweeps = (mgr_data -> num_interp_sweeps);
   HYPRE_Int *P_max_elmts = (mgr_data -> P_max_elmts);
   HYPRE_Real   max_row_sum = (mgr_data -> max_row_sum);
   HYPRE_Real   strong_threshold = (mgr_data -> strong_threshold);
   HYPRE_Real   trunc_factor = (mgr_data -> trunc_factor);
   HYPRE_Int  old_num_coarse_levels = (mgr_data -> num_coarse_levels);
   HYPRE_Int  max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   HYPRE_Int * reserved_Cpoint_local_indexes = (mgr_data -> reserved_Cpoint_local_indexes);
   hypre_IntArray      **CF_marker_array = (mgr_data -> CF_marker_array);
   HYPRE_Int            *CF_marker;
   hypre_IntArray       *FC_marker;
   hypre_ParCSRMatrix  **A_array = (mgr_data -> A_array);
   hypre_ParCSRMatrix  **B_array = (mgr_data -> B_array);
   hypre_ParCSRMatrix  **B_FF_array = (mgr_data -> B_FF_array);
#if defined(HYPRE_USING_GPU)
   hypre_ParCSRMatrix  **P_FF_array = (mgr_data -> P_FF_array);
#endif
   hypre_ParCSRMatrix  **P_array = (mgr_data -> P_array);
   hypre_ParCSRMatrix  **R_array = (mgr_data -> RT_array);
   hypre_ParCSRMatrix  **RT_array = (mgr_data -> RT_array);
   hypre_ParCSRMatrix   *RAP_ptr = NULL;

   hypre_ParCSRMatrix  *A_FF = NULL;
   hypre_ParCSRMatrix  *A_FC = NULL;
   hypre_ParCSRMatrix  *A_CF = NULL;
   hypre_ParCSRMatrix  *A_CC = NULL;

   hypre_Solver         *aff_base;
   HYPRE_Solver        **aff_solver = (mgr_data -> aff_solver);
   hypre_ParCSRMatrix  **A_ff_array = (mgr_data -> A_ff_array);
   hypre_ParVector     **F_fine_array = (mgr_data -> F_fine_array);
   hypre_ParVector     **U_fine_array = (mgr_data -> U_fine_array);

   HYPRE_Int (*fgrid_solver_setup)(void*, void*, void*, void*);
   HYPRE_Int (*fgrid_solver_solve)(void*, void*, void*, void*);

   hypre_ParVector    **F_array = (mgr_data -> F_array);
   hypre_ParVector    **U_array = (mgr_data -> U_array);
   hypre_ParVector    *residual = (mgr_data -> residual);
   HYPRE_Real    *rel_res_norms = (mgr_data -> rel_res_norms);
   HYPRE_Real    **frelax_diaginv = (mgr_data -> frelax_diaginv);
   HYPRE_Real    **level_diaginv = (mgr_data -> level_diaginv);
   // VPM: Do we need both frelax_diaginv and level_diaginv?

   HYPRE_Solver      default_cg_solver;
   HYPRE_Int (*cgrid_solver_setup)(void*, void*, void*, void*) =
      (HYPRE_Int (*)(void*, void*, void*, void*)) (mgr_data -> coarse_grid_solver_setup);

   HYPRE_Int (*cgrid_solver_solve)(void*, void*, void*, void*) =
      (HYPRE_Int (*)(void*, void*, void*, void*)) (mgr_data -> coarse_grid_solver_solve);

   HYPRE_Int    *level_smooth_type =  (mgr_data -> level_smooth_type);
   HYPRE_Int    *level_smooth_iters = (mgr_data -> level_smooth_iters);
   HYPRE_Solver *level_smoother = (mgr_data -> level_smoother);

   HYPRE_Int    reserved_coarse_size = (mgr_data -> reserved_coarse_size);

   HYPRE_Int      num_procs,  my_id;
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int use_VcycleSmoother = 0;
   HYPRE_Int use_GSElimSmoother = 0;
   //   HYPRE_Int use_ComplexSmoother = 0;
   hypre_ParVector     *VcycleRelaxZtemp;
   hypre_ParVector     *VcycleRelaxVtemp;
   hypre_ParAMGData    **FrelaxVcycleData;
   hypre_ParAMGData    **GSElimData;
   HYPRE_Int *Frelax_method = (mgr_data -> Frelax_method);
   HYPRE_Int *Frelax_num_functions = (mgr_data -> Frelax_num_functions);

   HYPRE_Int *Frelax_type = (mgr_data -> Frelax_type);

   HYPRE_Int *mgr_coarse_grid_method = (mgr_data -> mgr_coarse_grid_method);

   HYPRE_Int use_air = 0;
   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memory_location);
   HYPRE_Real truncate_cg_threshold = (mgr_data -> truncate_coarse_grid_threshold);
   char        region_name[1024];
   char        msg[2048];

   /* ----- begin -----*/
   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("MGRSetup");
   hypre_GpuProfilingPushRange("MGRSetup-Init");

   block_size = (mgr_data -> block_size);
   block_jacobi_bsize = (mgr_data -> block_jacobi_bsize);
   block_cf_marker = (mgr_data -> block_cf_marker);
   block_num_coarse_indexes = (mgr_data -> block_num_coarse_indexes);
   point_marker_array = (mgr_data -> point_marker_array);
   set_c_points_method = (mgr_data -> set_c_points_method);

   HYPRE_Int **level_coarse_indexes = NULL;
   HYPRE_Int *level_coarse_size = NULL;
   HYPRE_Int setNonCpointToF = (mgr_data -> set_non_Cpoints_to_F);
   HYPRE_BigInt *reserved_coarse_indexes = (mgr_data -> reserved_coarse_indexes);
   HYPRE_BigInt *idx_array = (mgr_data -> idx_array);
   HYPRE_Int lvl_to_keep_cpoints = ((mgr_data -> lvl_to_keep_cpoints) >
                                    (mgr_data -> max_num_coarse_levels)) ?
                                   (mgr_data -> max_num_coarse_levels) :
                                   (mgr_data -> lvl_to_keep_cpoints);
   HYPRE_Int    nloc   =  hypre_ParCSRMatrixNumRows(A);
   HYPRE_BigInt ilower =  hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt iupper =  hypre_ParCSRMatrixLastRowIndex(A);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Reset print_level codes. This is useful for printing
      information when solving a sequence of linear systems */
   print_level |= ((print_level & HYPRE_MGR_PRINT_RESERVED_A) == HYPRE_MGR_PRINT_RESERVED_A) ?
                  HYPRE_MGR_PRINT_INFO_PARAMS : 0;
   print_level |= ((print_level & HYPRE_MGR_PRINT_RESERVED_B) == HYPRE_MGR_PRINT_RESERVED_B) ?
                  HYPRE_MGR_PRINT_FINE_MATRIX : 0;
   print_level |= ((print_level & HYPRE_MGR_PRINT_RESERVED_C) == HYPRE_MGR_PRINT_RESERVED_C) ?
                  HYPRE_MGR_PRINT_FINE_RHS : 0;
   (mgr_data -> print_level) = print_level;

   /* Trivial case: simply solve the coarse level problem */
   if (block_size < 2 || (mgr_data -> max_num_coarse_levels) < 1)
   {
      if (my_id == 0 && print_level > 0)
      {
         hypre_printf("Warning: Block size is < 2 or number of coarse levels is < 1. \n");
         hypre_printf("Solving scalar problem on fine grid using coarse level solver \n");
      }

      if (use_default_cgrid_solver)
      {
         if (my_id == 0 && print_level > 0)
         {
            hypre_printf("No coarse grid solver provided. Using default AMG solver ... \n");
         }

         /* create and set default solver parameters here */
         /* create and initialize default_cg_solver */
         default_cg_solver = (HYPRE_Solver) hypre_BoomerAMGCreate();
         hypre_BoomerAMGSetMaxIter(default_cg_solver, (mgr_data -> max_iter));
         hypre_BoomerAMGSetRelaxOrder(default_cg_solver, 1);
         hypre_BoomerAMGSetPrintLevel(default_cg_solver, 3);

         /* set setup and solve functions */
         cgrid_solver_setup = (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSetup;
         cgrid_solver_solve = (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSolve;
         (mgr_data -> coarse_grid_solver_setup) = cgrid_solver_setup;
         (mgr_data -> coarse_grid_solver_solve) = cgrid_solver_solve;
         (mgr_data -> coarse_grid_solver) = default_cg_solver;
      }

      /* keep reserved coarse indexes to coarsest grid */
      if (reserved_coarse_size > 0)
      {
         HYPRE_BoomerAMGSetCPoints((mgr_data ->coarse_grid_solver), 25, reserved_coarse_size,
                                   reserved_coarse_indexes);
      }

      /* setup coarse grid solver */
      cgrid_solver_setup((mgr_data -> coarse_grid_solver), A, f, u);
      (mgr_data -> num_coarse_levels) = 0;

      HYPRE_ANNOTATE_FUNC_END;
      hypre_GpuProfilingPopRange();

      return hypre_error_flag;
   }

   /* If we reduce the reserved C-points, increase one level */
   if (lvl_to_keep_cpoints > 0)
   {
      max_num_coarse_levels++;
   }

   /* Initialize local indexes of coarse sets at different levels */
   level_coarse_indexes = hypre_CTAlloc(HYPRE_Int*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      level_coarse_indexes[i] = hypre_CTAlloc(HYPRE_Int, nloc, HYPRE_MEMORY_HOST);
   }

   level_coarse_size = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int reserved_cpoints_eliminated = 0;

   /* TODO: move this to par_mgr_coarsen.c and port to GPUs (VPM) */
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      // if we want to reduce the reserved Cpoints, set the current level
      // coarse indices the same as the previous level
      if (i == lvl_to_keep_cpoints && i > 0)
      {
         reserved_cpoints_eliminated++;
         for (j = 0; j < final_coarse_size; j++)
         {
            level_coarse_indexes[i][j] = level_coarse_indexes[i - 1][j];
         }
         level_coarse_size[i] = final_coarse_size;
         continue;
      }
      final_coarse_size = 0;
      if (set_c_points_method == 0) // interleaved ordering, i.e. s1,p1,s2,p2,...
      {
         // loop over rows
         for (row = ilower; row <= iupper; row++)
         {
            idx = row % block_size;
            if (block_cf_marker[i - reserved_cpoints_eliminated][idx] == CMRK)
            {
               level_coarse_indexes[i][final_coarse_size++] = (HYPRE_Int)(row - ilower);
            }
         }
      }
      else if (set_c_points_method == 1) // block ordering s1,s2,...,p1,p2,...
      {
         for (j = 0; j < block_size; j++)
         {
            if (block_cf_marker[i - reserved_cpoints_eliminated][j] == CMRK)
            {
               if (j == block_size - 1)
               {
                  end_idx = iupper + 1;
               }
               else
               {
                  end_idx = idx_array[j + 1];
               }
               for (row = idx_array[j]; row < end_idx; row++)
               {
                  level_coarse_indexes[i][final_coarse_size++] = (HYPRE_Int)(row - ilower);
               }
            }
         }
         //hypre_printf("Level %d, # of coarse points %d\n", i, final_coarse_size);
      }
      else if (set_c_points_method == 2)
      {
         HYPRE_Int isCpoint;
         // row start from 0 since point_marker_array is local
         for (row = 0; row < nloc; row++)
         {
            isCpoint = 0;
            for (j = 0; j < block_num_coarse_indexes[i]; j++)
            {
               if (point_marker_array[row] == block_cf_marker[i][j])
               {
                  isCpoint = 1;
                  break;
               }
            }
            if (isCpoint)
            {
               level_coarse_indexes[i][final_coarse_size++] = row;
               //printf("%d\n",row);
            }
         }
      }
      else
      {
         if (my_id == 0)
         {
            hypre_printf("ERROR! Unknown method for setting C points.");
         }
         exit(-1); // TODO: Fix error handling (VPM)
      }
      level_coarse_size[i] = final_coarse_size;
   }

   /* Set reserved coarse indexes to be kept to the coarsest level of the MGR solver */
   hypre_TFree((mgr_data -> reserved_Cpoint_local_indexes), HYPRE_MEMORY_HOST);

   if (reserved_coarse_size > 0)
   {
      (mgr_data -> reserved_Cpoint_local_indexes) = hypre_CTAlloc(HYPRE_Int,
                                                                  reserved_coarse_size,
                                                                  HYPRE_MEMORY_HOST);
      reserved_Cpoint_local_indexes = (mgr_data -> reserved_Cpoint_local_indexes);
      for (i = 0; i < reserved_coarse_size; i++)
      {
         row = reserved_coarse_indexes[i];
         HYPRE_Int local_row = (HYPRE_Int)(row - ilower);
         reserved_Cpoint_local_indexes[i] = local_row;
         HYPRE_Int lvl = lvl_to_keep_cpoints == 0 ? max_num_coarse_levels : lvl_to_keep_cpoints;
         if (set_c_points_method < 2)
         {
            idx = row % block_size;
            for (j = 0; j < lvl; j++)
            {
               if (block_cf_marker[j][idx] != CMRK)
               {
                  level_coarse_indexes[j][level_coarse_size[j]++] = local_row;
               }
            }
         }
         else
         {
            HYPRE_Int isCpoint = 0;
            for (j = 0; j < lvl; j++)
            {
               HYPRE_Int k;
               for (k = 0; k < block_num_coarse_indexes[j]; k++)
               {
                  if (point_marker_array[local_row] == block_cf_marker[j][k])
                  {
                     isCpoint = 1;
                     break;
                  }
               }
               if (!isCpoint)
               {
                  level_coarse_indexes[j][level_coarse_size[j]++] = local_row;
               }
            }
         }
      }
   }

   (mgr_data -> level_coarse_indexes) = level_coarse_indexes;
   (mgr_data -> num_coarse_per_level) = level_coarse_size;

   /* Free Previously allocated data, if any not destroyed */
   if (A_array || B_array || B_FF_array ||
       P_array || R_array || RT_array || CF_marker_array)
   {
      for (j = 1; j < (old_num_coarse_levels); j++)
      {
         if (A_array[j])
         {
            hypre_ParCSRMatrixDestroy(A_array[j]);
            A_array[j] = NULL;
         }
      }

      for (j = 0; j < old_num_coarse_levels; j++)
      {
         if (B_array[j])
         {
            hypre_ParCSRMatrixDestroy(B_array[j]);
            B_array[j] = NULL;
         }

         if (B_FF_array[j])
         {
            hypre_ParCSRMatrixDestroy(B_FF_array[j]);
            B_FF_array[j] = NULL;
         }

         if (P_array[j])
         {
            hypre_ParCSRMatrixDestroy(P_array[j]);
            P_array[j] = NULL;
         }

         if (R_array[j])
         {
            hypre_ParCSRMatrixDestroy(R_array[j]);
            R_array[j] = NULL;
         }

         if (RT_array[j])
         {
            hypre_ParCSRMatrixDestroy(RT_array[j]);
            RT_array[j] = NULL;
         }

         if (CF_marker_array[j])
         {
            hypre_IntArrayDestroy(CF_marker_array[j]);
            CF_marker_array[j] = NULL;
         }
      }
      hypre_TFree(B_array, HYPRE_MEMORY_HOST);
      hypre_TFree(B_FF_array, HYPRE_MEMORY_HOST);
      hypre_TFree(P_array, HYPRE_MEMORY_HOST);
      hypre_TFree(R_array, HYPRE_MEMORY_HOST);
      hypre_TFree(RT_array, HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_array, HYPRE_MEMORY_HOST);
   }

#if defined(HYPRE_USING_GPU)
   if (P_FF_array)
   {
      for (j = 0; j < old_num_coarse_levels; j++)
      {
         if (P_FF_array[j])
         {
            hypre_ParCSRMatrixDestroy(P_FF_array[j]);
            P_FF_array[j] = NULL;
         }
      }
      hypre_TFree(P_FF_array, HYPRE_MEMORY_HOST);
   }
#endif

   /* Free previously allocated FrelaxVcycleData if not destroyed */
   if ((mgr_data -> VcycleRelaxZtemp))
   {
      hypre_ParVectorDestroy((mgr_data -> VcycleRelaxZtemp));
      (mgr_data -> VcycleRelaxZtemp) = NULL;
   }
   if ((mgr_data -> VcycleRelaxVtemp))
   {
      hypre_ParVectorDestroy((mgr_data -> VcycleRelaxVtemp));
      (mgr_data -> VcycleRelaxVtemp) = NULL;
   }
   if ((mgr_data -> FrelaxVcycleData))
   {
      for (j = 0; j < old_num_coarse_levels; j++)
      {
         if ((mgr_data -> FrelaxVcycleData)[j])
         {
            hypre_MGRDestroyFrelaxVcycleData((mgr_data -> FrelaxVcycleData)[j]);
            (mgr_data -> FrelaxVcycleData)[j] = NULL;
         }
      }
      hypre_TFree((mgr_data -> FrelaxVcycleData), HYPRE_MEMORY_HOST);
      (mgr_data -> FrelaxVcycleData) = NULL;
   }

   /* destroy previously allocated Gaussian Elim. data */
   if ((mgr_data -> GSElimData))
   {
      for (j = 0; j < old_num_coarse_levels; j++)
      {
         if ((mgr_data -> GSElimData)[j])
         {
            hypre_MGRDestroyGSElimData((mgr_data -> GSElimData)[j]);
            (mgr_data -> GSElimData)[j] = NULL;
         }
      }
      hypre_TFree((mgr_data -> GSElimData), HYPRE_MEMORY_HOST);
      (mgr_data -> GSElimData) = NULL;
   }

   /* destroy final coarse grid matrix, if not previously destroyed */
   if ((mgr_data -> RAP))
   {
      hypre_ParCSRMatrixDestroy((mgr_data -> RAP));
      (mgr_data -> RAP) = NULL;
   }

   /* Setup for global block smoothers*/
   if (set_c_points_method == 0)
   {
      if (my_id == num_procs)
      {
         mgr_data -> n_block   = (n - reserved_coarse_size) / block_size;
         mgr_data -> left_size = n - block_size * (mgr_data -> n_block);
      }
      else
      {
         mgr_data -> n_block = n / block_size;
         mgr_data -> left_size = n - block_size * (mgr_data -> n_block);
      }
   }
   else
   {
      mgr_data -> n_block = n;
      mgr_data -> left_size = 0;
   }

   /* clear old l1_norm data, if created */
   if ((mgr_data -> l1_norms))
   {
      for (j = 0; j < (old_num_coarse_levels); j++)
      {
         if ((mgr_data -> l1_norms)[j])
         {
            hypre_SeqVectorDestroy((mgr_data -> l1_norms)[i]);
            (mgr_data -> l1_norms)[j] = NULL;
         }
      }
      hypre_TFree((mgr_data -> l1_norms), HYPRE_MEMORY_HOST);
   }

   if ((mgr_data -> frelax_diaginv))
   {
      for (j = 0; j < (old_num_coarse_levels); j++)
      {
         if ((mgr_data -> frelax_diaginv)[j])
         {
            hypre_TFree((mgr_data -> frelax_diaginv)[j], HYPRE_MEMORY_HOST);
            (mgr_data -> frelax_diaginv)[j] = NULL;
         }
      }
      hypre_TFree((mgr_data -> frelax_diaginv), HYPRE_MEMORY_HOST);
   }

   if ((mgr_data -> level_diaginv))
   {
      for (j = 0; j < (old_num_coarse_levels); j++)
      {
         if ((mgr_data -> level_diaginv)[j])
         {
            hypre_TFree((mgr_data -> level_diaginv)[j], HYPRE_MEMORY_HOST);
            (mgr_data -> level_diaginv)[j] = NULL;
         }
      }
      hypre_TFree((mgr_data -> level_diaginv), HYPRE_MEMORY_HOST);
   }

   /* setup temporary storage */
   if ((mgr_data -> Ztemp))
   {
      hypre_ParVectorDestroy((mgr_data -> Ztemp));
      (mgr_data -> Ztemp) = NULL;
   }
   if ((mgr_data -> Vtemp))
   {
      hypre_ParVectorDestroy((mgr_data -> Vtemp));
      (mgr_data -> Vtemp) = NULL;
   }
   if ((mgr_data -> Utemp))
   {
      hypre_ParVectorDestroy((mgr_data -> Utemp));
      (mgr_data -> Utemp) = NULL;
   }
   if ((mgr_data -> Ftemp))
   {
      hypre_ParVectorDestroy((mgr_data -> Ftemp));
      (mgr_data -> Ftemp) = NULL;
   }
   if ((mgr_data -> residual))
   {
      hypre_ParVectorDestroy((mgr_data -> residual));
      (mgr_data -> residual) = NULL;
   }
   hypre_TFree((mgr_data -> rel_res_norms), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> blk_size), HYPRE_MEMORY_HOST);

   Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Vtemp);
   (mgr_data ->Vtemp) = Vtemp;

   Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Ztemp);
   (mgr_data -> Ztemp) = Ztemp;

   Utemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Utemp);
   (mgr_data ->Utemp) = Utemp;

   Ftemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Ftemp);
   (mgr_data ->Ftemp) = Ftemp;

   /* Allocate memory for level structure */
   if (A_array == NULL)
   {
      A_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (B_array == NULL)
   {
      B_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (P_array == NULL && max_num_coarse_levels > 0)
   {
      P_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (R_array == NULL && max_num_coarse_levels > 0)
   {
      R_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
#if defined(HYPRE_USING_GPU)
   if (P_FF_array == NULL && max_num_coarse_levels > 0)
   {
      P_FF_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
#endif
   if (RT_array == NULL && max_num_coarse_levels > 0)
   {
      RT_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (CF_marker_array == NULL)
   {
      CF_marker_array = hypre_CTAlloc(hypre_IntArray*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (l1_norms == NULL)
   {
      l1_norms = hypre_CTAlloc(hypre_Vector*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (P_max_elmts == NULL)
   {
      P_max_elmts = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }

   /* Set default for Frelax_method if not set already -- Supports deprecated function */
   /*
      if (Frelax_method == NULL)
      {
         Frelax_method = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
         for (i = 0; i < max_num_coarse_levels; i++)
         {
            Frelax_method[i] = 0;
         }
         (mgr_data -> Frelax_method) = Frelax_method;
      }
   */
   /* Set default for Frelax_type if not set already.
    * We also consolidate inputs from relax_type and Frelax_method here.
    * This should be simplified once the other options are removed.
   */
   if (Frelax_type == NULL)
   {
      Frelax_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);

      /* Use relax type/ frelax_method data or set default type to use */
      if (Frelax_method)
      {
         for (i = 0; i < max_num_coarse_levels; i++)
         {
            Frelax_type[i] = Frelax_method[i] > 0 ? Frelax_method[i] : relax_type;
         }
      }
      else if (relax_type)
      {
         for (i = 0; i < max_num_coarse_levels; i++)
         {
            Frelax_type[i] = relax_type;
         }
      }
      else /* set default here */
      {
         for (i = 0; i < max_num_coarse_levels; i++)
         {
            Frelax_type[i] = 0;
         }
      }

      (mgr_data -> Frelax_type) = Frelax_type;
   }

   /* When running on the device and using Jacobi relaxation, switch to GPU-supported Jacobi */
#if defined(HYPRE_USING_GPU)
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      if (Frelax_type[i] == 0 && interp_type && interp_type[i] != 12)
      {
         Frelax_type[i] = 7;
         if (print_level)
         {
            hypre_ParPrintf(comm, "Changing F-relaxation type to 7 at MGR level %d\n", i);
         }
      }
   }
#endif

   /* Set default for using non-Galerkin coarse grid */
   if (mgr_coarse_grid_method == NULL)
   {
      mgr_coarse_grid_method = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         mgr_coarse_grid_method[i] = 0;
      }
      (mgr_data -> mgr_coarse_grid_method) = mgr_coarse_grid_method;
   }

   /*
   if (Frelax_num_functions== NULL)
   {
     Frelax_num_functions = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
     for (i = 0; i < max_num_coarse_levels; i++)
     {
       Frelax_num_functions[i] = 1;
     }
     (mgr_data -> Frelax_num_functions) = Frelax_num_functions;
   }
   */
   /* Set default for interp_type and restrict_type if not set already */
   if (interp_type == NULL)
   {
      interp_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         interp_type[i] = 2;
      }
      (mgr_data -> interp_type) = interp_type;
   }
   if (restrict_type == NULL)
   {
      restrict_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         restrict_type[i] = 0;
      }
      (mgr_data -> restrict_type) = restrict_type;
   }
   if (num_relax_sweeps == NULL)
   {
      num_relax_sweeps = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         num_relax_sweeps[i] = 1;
      }
      (mgr_data -> num_relax_sweeps) = num_relax_sweeps;
   }

   /* set interp_type, restrict_type, and Frelax_type if we reduce the reserved C-points */
   reserved_cpoints_eliminated = 0;
   if (lvl_to_keep_cpoints > 0 && reserved_coarse_size > 0)
   {
      HYPRE_Int *level_interp_type   = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels,
                                                     HYPRE_MEMORY_HOST);
      HYPRE_Int *level_restrict_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels,
                                                     HYPRE_MEMORY_HOST);
      HYPRE_Int *level_frelax_type   = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels,
                                                     HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         if (i == lvl_to_keep_cpoints)
         {
            level_interp_type[i] = 2;
            level_restrict_type[i] = 0;
            level_frelax_type[i] = 2; //99;
            reserved_cpoints_eliminated++;
         }
         else
         {
            level_interp_type[i] = interp_type[i - reserved_cpoints_eliminated];
            level_restrict_type[i] = restrict_type[i - reserved_cpoints_eliminated];
            level_frelax_type[i] = Frelax_type[i - reserved_cpoints_eliminated];
         }
      }
      hypre_TFree(interp_type, HYPRE_MEMORY_HOST);
      hypre_TFree(restrict_type, HYPRE_MEMORY_HOST);
      hypre_TFree(Frelax_type, HYPRE_MEMORY_HOST);

      interp_type = level_interp_type;
      restrict_type = level_restrict_type;
      Frelax_type = level_frelax_type;
      (mgr_data -> interp_type) = level_interp_type;
      (mgr_data -> restrict_type) = level_restrict_type;
      (mgr_data -> Frelax_type) = level_frelax_type;
   }

   /* set pointers to mgr data */
   (mgr_data -> A_array)         = A_array;
   (mgr_data -> B_array)         = B_array;
   (mgr_data -> P_array)         = P_array;
   (mgr_data -> R_array)         = R_array;
   (mgr_data -> RT_array)        = RT_array;
   (mgr_data -> CF_marker_array) = CF_marker_array;
   (mgr_data -> l1_norms)        = l1_norms;
   (mgr_data -> P_max_elmts)     = P_max_elmts;
#if defined(HYPRE_USING_GPU)
   (mgr_data -> P_FF_array)      = P_FF_array;
#endif

   /* Set up solution and rhs arrays */
   if (F_array != NULL || U_array != NULL)
   {
      for (j = 1; j < old_num_coarse_levels + 1; j++)
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
      F_array = hypre_CTAlloc(hypre_ParVector*, max_num_coarse_levels + 1, HYPRE_MEMORY_HOST);
   }
   if (U_array == NULL)
   {
      U_array = hypre_CTAlloc(hypre_ParVector*, max_num_coarse_levels + 1, HYPRE_MEMORY_HOST);
   }

   /* TODO: Change to A_FF_array (VPM) */
   if (A_ff_array)
   {
      for (j = 1; j < old_num_coarse_levels; j++)
      {
         if (A_ff_array[j])
         {
            hypre_ParCSRMatrixDestroy(A_ff_array[j]);
            A_ff_array[j] = NULL;
         }
      }
      if (mgr_data -> fsolver_mode != 0)
      {
         if (A_ff_array[0])
         {
            hypre_ParCSRMatrixDestroy(A_ff_array[0]);
            A_ff_array[0] = NULL;
         }
      }
      hypre_TFree(A_ff_array, HYPRE_MEMORY_HOST);
      A_ff_array = NULL;
   }

   if (aff_solver)
   {
      for (j = 1; j < (old_num_coarse_levels); j++)
      {
         if (aff_solver[j])
         {
            aff_base = (hypre_Solver*) aff_solver[j];
            hypre_SolverDestroy(aff_base)((HYPRE_Solver) (aff_base));
            aff_solver[j] = NULL;
         }
      }
      if (mgr_data -> fsolver_mode == 2)
      {
         hypre_BoomerAMGDestroy(aff_solver[0]);
      }
   }

   if ((mgr_data -> fine_grid_solver_setup) != NULL)
   {
      fgrid_solver_setup = (mgr_data -> fine_grid_solver_setup);
   }
   else
   {
      fgrid_solver_setup = (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSetup;
      (mgr_data -> fine_grid_solver_setup) = fgrid_solver_setup;
   }
   if ((mgr_data -> fine_grid_solver_solve) != NULL)
   {
      fgrid_solver_solve = (mgr_data -> fine_grid_solver_solve);
   }
   else
   {
      fgrid_solver_solve = (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSolve;
      (mgr_data -> fine_grid_solver_solve) = fgrid_solver_solve;
   }

   /* Set up solution and rhs arrays for Frelax */
   if (F_fine_array != NULL || U_fine_array != NULL)
   {
      for (j = 1; j < old_num_coarse_levels + 1; j++)
      {
         if (F_fine_array[j] != NULL)
         {
            hypre_ParVectorDestroy(F_fine_array[j]);
            F_fine_array[j] = NULL;
         }
         if (U_fine_array[j] != NULL)
         {
            hypre_ParVectorDestroy(U_fine_array[j]);
            U_fine_array[j] = NULL;
         }
      }
   }

   if (F_fine_array == NULL)
   {
      F_fine_array = hypre_CTAlloc(hypre_ParVector*, max_num_coarse_levels + 1,
                                   HYPRE_MEMORY_HOST);
   }
   if (U_fine_array == NULL)
   {
      U_fine_array = hypre_CTAlloc(hypre_ParVector*, max_num_coarse_levels + 1,
                                   HYPRE_MEMORY_HOST);
   }
   if (aff_solver == NULL)
   {
      aff_solver = hypre_CTAlloc(HYPRE_Solver*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (A_ff_array == NULL)
   {
      A_ff_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (B_FF_array == NULL)
   {
      B_FF_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (frelax_diaginv == NULL)
   {
      frelax_diaginv = hypre_CTAlloc(HYPRE_Real*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (level_diaginv == NULL)
   {
      level_diaginv = hypre_CTAlloc(HYPRE_Real*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (blk_size == NULL)
   {
      blk_size = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (level_smooth_type == NULL)
   {
      level_smooth_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (level_smooth_iters == NULL)
   {
      level_smooth_iters = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }
   if (level_smoother == NULL)
   {
      level_smoother = hypre_CTAlloc(HYPRE_Solver, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }

   /* set solution and rhs pointers */
   F_array[0] = f;
   U_array[0] = u;

   (mgr_data -> F_array) = F_array;
   (mgr_data -> U_array) = U_array;

   (mgr_data -> F_fine_array) = F_fine_array;
   (mgr_data -> U_fine_array) = U_fine_array;
   (mgr_data -> aff_solver) = aff_solver;
   (mgr_data -> A_ff_array) = A_ff_array;
   (mgr_data -> B_FF_array) = B_FF_array;
   (mgr_data -> frelax_diaginv) = frelax_diaginv;
   (mgr_data -> level_diaginv) = level_diaginv;
   (mgr_data -> blk_size) = blk_size;
   (mgr_data -> level_smooth_type) = level_smooth_type;
   (mgr_data -> level_smooth_iters) = level_smooth_iters;
   (mgr_data -> level_smoother) = level_smoother;

   /* begin coarsening loop */
   num_coarsening_levs = max_num_coarse_levels;

   /* initialize level data matrix here */
   RAP_ptr = A;

   /* Close MGRSetup-Init region */
   hypre_GpuProfilingPopRange();

   /* loop over levels of coarsening */
   for (lev = 0; lev < num_coarsening_levs; lev++)
   {
      hypre_sprintf(region_name, "%s-%d", "MGR_Level", lev);
      hypre_GpuProfilingPushRange(region_name);
      HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

      /* Check if this is the last level */
      last_level = (lev == (num_coarsening_levs - 1));

      /* Set level's block size */
      level_blk_size = (lev == 0) ? block_size : block_num_coarse_indexes[lev - 1];

      /* Initialize A_array */
      A_array[lev] = RAP_ptr;
      nloc = hypre_ParCSRMatrixNumRows(A_array[lev]);

      /* Reset pointers */
      l1_norms_data = NULL;

      /* Setup global smoother */
      hypre_sprintf(region_name, "Global-Relax");
      hypre_GpuProfilingPushRange(region_name);
      HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);
      if (level_smooth_iters[lev] > 0)
      {
         /* TODO (VPM): Change option types for block-Jacobi and block-GS to 30 and 31 and
            make them accessible through hypre_BoomerAMGRelax? */
         if (level_smooth_type[lev] == 0 || level_smooth_type[lev] == 1)
         {
            /* TODO (VPM): move this to hypre_MGRBlockRelaxSetup and change its declaration */
#if defined (HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               if (!B_array[lev])
               {
                  hypre_ParCSRMatrixBlockDiagMatrixDevice(A_array[lev], level_blk_size,
                                                          0, NULL, 1, &B_array[lev]);
               }
            }
            else
#endif
            {
               hypre_MGRBlockRelaxSetup(A_array[lev], level_blk_size,
                                        &(mgr_data -> level_diaginv)[lev]);
            }
         }
         else if (level_smooth_type[lev] == 8)
         {
            /* TODO (VPM): Option 8 should be for hybrid L1 Symm. Gauss-Seidel */
            HYPRE_EuclidCreate(comm, &(level_smoother[lev]));
            HYPRE_EuclidSetLevel(level_smoother[lev], 0);
            HYPRE_EuclidSetBJ(level_smoother[lev], 1);
            HYPRE_EuclidSetup(level_smoother[lev], A_array[lev], NULL, NULL);
         }
         else if (level_smooth_type[lev] == 16)
         {
            /* TODO (VPM): Option 16 should be for Chebyshev */
            HYPRE_ILUCreate(&(level_smoother[lev]));
            HYPRE_ILUSetType(level_smoother[lev], 0);
            HYPRE_ILUSetLevelOfFill(level_smoother[lev], 0);
            HYPRE_ILUSetMaxIter(level_smoother[lev], level_smooth_iters[lev]);
            HYPRE_ILUSetTol(level_smoother[lev], 0.0);
            HYPRE_ILUSetLocalReordering(level_smoother[lev], 0);
            HYPRE_ILUSetup(level_smoother[lev], A_array[lev], NULL, NULL);
         }
         else
         {
            /* Compute l1_norms according to relaxation type */
            hypre_BoomerAMGRelaxComputeL1Norms(A_array[lev], level_smooth_type[lev],
                                               0, 0, NULL, &l1_norms_data);
            if (l1_norms_data)
            {
               l1_norms[lev] = hypre_SeqVectorCreate(nloc);
               hypre_VectorData(l1_norms[lev]) = l1_norms_data;
               hypre_VectorMemoryLocation(l1_norms[lev]) = memory_location;

               if (print_level)
               {
                  hypre_ParPrintf(comm, "Setting l1_norms for global relax at MGR level %d\n", i);
               }
            }
         }
      }
      hypre_GpuProfilingPopRange();
      HYPRE_ANNOTATE_REGION_END("%s", region_name);

      /* Compute strength matrix for interpolation operator
         use default parameters, to be modified later */
      hypre_sprintf(region_name, "Coarsen");
      hypre_GpuProfilingPushRange(region_name);
      HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);
      cflag = last_level || setNonCpointToF;
      if (interp_type[lev] == 3 || interp_type[lev] == 5 ||
          interp_type[lev] == 6 || interp_type[lev] == 7 || !cflag)
      {
         hypre_BoomerAMGCreateS(A_array[lev], strong_threshold, max_row_sum, 1, NULL, &S);
      }

      /* Coarsen: Build CF_marker array based on rows of A */
      hypre_MGRCoarsen(S, A_array[lev], level_coarse_size[lev], level_coarse_indexes[lev],
                       debug_flag, &CF_marker_array[lev], cflag);
      CF_marker = hypre_IntArrayData(CF_marker_array[lev]);

      /* Get global fine/coarse partitionings. TODO: generate dof_func */
      hypre_MGRCoarseParms(comm, nloc, CF_marker_array[lev],
                           coarse_pnts_global, row_starts_fpts);
      hypre_GpuProfilingPopRange();
      HYPRE_ANNOTATE_REGION_END("%s", region_name);

      /* Compute Petrov-Galerkin operators */
      num_interp_sweeps = (mgr_data -> num_interp_sweeps);

      if (mgr_data -> block_jacobi_bsize == 0)
      {
         block_jacobi_bsize = level_blk_size - block_num_coarse_indexes[lev];
      }

      if (block_jacobi_bsize == 1 && interp_type[lev] == 12)
      {
         interp_type[lev] = 2;
      }

      /* Compute C/F splitting (needed by RAP computation and other operations) */
      FC_marker = hypre_IntArrayCloneDeep(CF_marker_array[lev]);
      hypre_IntArrayNegate(FC_marker);

      hypre_ParCSRMatrixGenerateFFFC(A_array[lev], CF_marker, coarse_pnts_global,
                                     NULL, &A_FC, &A_FF);
      hypre_ParCSRMatrixGenerateFFFC(A_array[lev], hypre_IntArrayData(FC_marker), row_starts_fpts,
                                     NULL, &A_CF, &A_CC);

      /* Build MGR interpolation */
      hypre_sprintf(region_name, "Interp");
      hypre_GpuProfilingPushRange(region_name);
      HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

      if (interp_type[lev] == 12)
      {
         if (mgr_coarse_grid_method[lev] != 0)
         {
            hypre_MGRBuildBlockJacobiWp(A_FF, A_FC, block_jacobi_bsize, &Wp);
         }
         hypre_MGRBuildInterp(A_array[lev], A_FF, A_FC, CF_marker, Wp,
                              coarse_pnts_global, trunc_factor, P_max_elmts[lev],
                              block_jacobi_bsize, &P, interp_type[lev],
                              num_interp_sweeps);
      }
      else
      {
         hypre_MGRBuildInterp(A_array[lev], A_FF, A_FC, CF_marker, S,
                              coarse_pnts_global, trunc_factor, P_max_elmts[lev],
                              block_jacobi_bsize, &P, interp_type[lev],
                              num_interp_sweeps);
      }

      hypre_GpuProfilingPopRange();
      HYPRE_ANNOTATE_REGION_END("%s", region_name);

      /* Use block Jacobi F-relaxation with block Jacobi interpolation */
      hypre_sprintf(region_name, "F-Relax");
      hypre_GpuProfilingPushRange(region_name);
      HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

      if (interp_type[lev] == 12 && (mgr_data -> num_relax_sweeps)[lev] > 0)
      {
         /* TODO: refactor the following block (VPM) */
#if defined (HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_ParCSRMatrixBlockDiagMatrixDevice(A_FF, block_jacobi_bsize,
                                                    0, NULL, 1, &B_FF_array[lev]);
         }
         else
#endif
         {
            HYPRE_Real *diag_inv = NULL;
            HYPRE_Int   inv_num_rows;
            HYPRE_Int   inv_size;

            /* TODO: replace this with hypre_IntArrayCount (VPM) */
            inv_num_rows = 0;
            for (i = 0; i < nloc; i++)
            {
               inv_num_rows += (CF_marker[i] == -1) ? 1 : 0;
            }

            /* Extract block diagonal inverses */
            inv_size = inv_num_rows * block_jacobi_bsize;
            diag_inv = hypre_CTAlloc(HYPRE_Complex, inv_size, HYPRE_MEMORY_HOST);

            /* TODO: Extend this to device (VPM) */
            hypre_ParCSRMatrixExtractBlockDiagHost(A_array[lev], block_jacobi_bsize, inv_num_rows,
                                                   -1, CF_marker, inv_size, 1, diag_inv);
            frelax_diaginv[lev] = diag_inv;
            blk_size[lev] = block_jacobi_bsize;
            hypre_MGRBuildAff(A_array[lev], CF_marker, debug_flag, &A_FF);
         }

         /* Set A_ff pointer */
         A_ff_array[lev] = A_FF;

         F_fine_array[lev + 1] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_FF),
                                  hypre_ParCSRMatrixGlobalNumRows(A_FF),
                                  hypre_ParCSRMatrixRowStarts(A_FF));
         hypre_ParVectorInitialize(F_fine_array[lev + 1]);

         U_fine_array[lev + 1] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_FF),
                                  hypre_ParCSRMatrixGlobalNumRows(A_FF),
                                  hypre_ParCSRMatrixRowStarts(A_FF));
         hypre_ParVectorInitialize(U_fine_array[lev + 1]);
      }
      hypre_GpuProfilingPopRange();
      HYPRE_ANNOTATE_REGION_END("%s", region_name);

      P_array[lev] = P;

      if (restrict_type[lev] == 4)
      {
         use_air = 1;
      }
      else if (restrict_type[lev] == 5)
      {
         use_air = 2;
      }
      else
      {
         use_air = 0;
      }

      if (use_air)
      {
         HYPRE_Real    filter_thresholdR = 0.0;
         HYPRE_Int     gmres_switch = 64;
         HYPRE_Int     is_triangular = 0;

         hypre_sprintf(region_name, "Restrict");
         hypre_GpuProfilingPushRange(region_name);
         HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

         /* for AIR, need absolute value SOC */
         hypre_BoomerAMGCreateSabs(A_array[lev], strong_threshold, 1.0, 1, NULL, &ST);

         /* !!! Ensure that CF_marker contains -1 or 1 !!! */
         /*
         for (i = 0; i < hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level])); i++)
         {
           CF_marker[i] = CF_marker[i] > 0 ? 1 : -1;
         }
         */
         if (use_air == 1) /* distance-1 AIR */
         {
            hypre_BoomerAMGBuildRestrAIR(A_array[lev], CF_marker,
                                         ST, coarse_pnts_global, 1,
                                         dof_func_buff_data, filter_thresholdR,
                                         debug_flag, &R, is_triangular, gmres_switch);
         }
         else /* distance-1.5 AIR - distance 2 locally and distance 1 across procs. */
         {
            hypre_BoomerAMGBuildRestrDist2AIR(A_array[lev], CF_marker,
                                              ST, coarse_pnts_global, 1,
                                              dof_func_buff_data, filter_thresholdR,
                                              debug_flag, &R, 1, is_triangular, gmres_switch);
         }
         R_array[lev] = R;
         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_REGION_END("%s", region_name);

         /* Use two matrix products to generate A_H */
         hypre_ParCSRMatrix *AP = NULL;

         hypre_sprintf(region_name, "RAP");
         hypre_GpuProfilingPushRange(region_name);
         HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);
         AP      = hypre_ParMatmul(A_array[lev], P_array[lev]);
         RAP_ptr = hypre_ParMatmul(R, AP);
         if (num_procs > 1)
         {
            hypre_MatvecCommPkgCreate(RAP_ptr);
         }

         /* Delete AP */
         hypre_ParCSRMatrixDestroy(AP);
         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_REGION_END("%s", region_name);
      }
      else
      {
         if (mgr_coarse_grid_method[lev] != 0)
         {
            HYPRE_Int block_num_f_points = level_blk_size - block_num_coarse_indexes[lev];

            hypre_sprintf(region_name, "Restrict");
            hypre_GpuProfilingPushRange(region_name);
            HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);
            if (block_num_f_points == 1 && restrict_type[lev] == 12)
            {
               restrict_type[lev] = 2;
            }

            hypre_MGRBuildRestrict(A_array[lev], A_FF, A_FC, A_CF, CF_marker_array[lev],
                                   coarse_pnts_global, trunc_factor, P_max_elmts[lev],
                                   strong_threshold, max_row_sum, block_num_f_points,
                                   restrict_type[lev], &Wr, &R, &RT);
            R_array[lev]  = R;
            RT_array[lev] = RT;

            hypre_GpuProfilingPopRange();
            HYPRE_ANNOTATE_REGION_END("%s", region_name);

            hypre_sprintf(region_name, "RAP");
            hypre_GpuProfilingPushRange(region_name);
            HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

#if defined (HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               hypre_MGRComputeNonGalerkinCGDevice(A_FF, A_FC, A_CF, A_CC,
                                                   Wp, Wr, block_num_f_points,
                                                   mgr_coarse_grid_method[lev],
                                                   truncate_cg_threshold,
                                                   &RAP_ptr);
            }
            else
#endif
            {
               hypre_MGRComputeNonGalerkinCoarseGrid(A_FF, A_FC, A_CF, A_CC, Wp, Wr,
                                                     block_num_f_points, set_c_points_method,
                                                     mgr_coarse_grid_method[lev],
                                                     P_max_elmts[lev], &RAP_ptr);
            }

            if (interp_type[lev] == 12)
            {
               hypre_ParCSRMatrixDeviceColMapOffd(Wp) = NULL;
               hypre_ParCSRMatrixColMapOffd(Wp)       = NULL;
               hypre_ParCSRMatrixDestroy(Wp);
               Wp = NULL;
            }
            hypre_GpuProfilingPopRange();
            HYPRE_ANNOTATE_REGION_END("%s", region_name);
         }
         else
         {
            hypre_sprintf(region_name, "Restrict");
            hypre_GpuProfilingPushRange(region_name);
            HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);
            if (block_jacobi_bsize == 1 && restrict_type[lev] == 12)
            {
               restrict_type[lev] = 2;
            }
            hypre_MGRBuildRestrict(A_array[lev], A_FF, A_FC, A_CF, CF_marker_array[lev],
                                   coarse_pnts_global, trunc_factor, P_max_elmts[lev],
                                   strong_threshold, max_row_sum, block_jacobi_bsize,
                                   restrict_type[lev], &Wr, &R, &RT);
            R_array[lev]  = R;
            RT_array[lev] = RT;
            hypre_GpuProfilingPopRange();
            HYPRE_ANNOTATE_REGION_END("%s", region_name);

            hypre_sprintf(region_name, "RAP");
            hypre_GpuProfilingPushRange(region_name);
            HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);
            if (RT)
            {
               RAP_ptr = hypre_ParCSRMatrixRAPKT(RT, A_array[lev], P, 1);
            }
            else if (R)
            {
               AP      = hypre_ParCSRMatMat(A_array[lev], P);
               RAP_ptr = hypre_ParCSRMatMat(R, AP);
               hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(RAP_ptr));
               hypre_ParCSRMatrixDestroy(AP);
            }
            else
            {
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Expected either R or RT!");
               return hypre_error_flag;
            }
            hypre_GpuProfilingPopRange();
            HYPRE_ANNOTATE_REGION_END("%s", region_name);
         }
      }

      /* TODO (VPM): truncation is also performed in hypre_MGRComputeNonGalerkinCoarseGrid */
      if (truncate_cg_threshold > 0.0)
      {
         /* Truncate the coarse grid */
         if (exec == HYPRE_EXEC_HOST)
         {
            hypre_ParCSRMatrixTruncate(RAP_ptr, truncate_cg_threshold, 0, 0, 0);
         }
#if defined (HYPRE_USING_GPU)
         else
         {
            hypre_ParCSRMatrixDropSmallEntriesDevice(RAP_ptr, truncate_cg_threshold, -1);
         }
#endif
      }

      /* Destroy temporary variables */
      hypre_ParCSRMatrixDestroy(A_FC), A_FC = NULL;
      hypre_ParCSRMatrixDestroy(A_CF), A_CF = NULL;
      hypre_ParCSRMatrixDestroy(A_CC), A_CF = NULL;
      hypre_ParCSRMatrixDestroy(Wr); Wr = NULL;

      /* User-prescribed F-solver */
      if (Frelax_type[lev] == 2  ||
          Frelax_type[lev] == 9  ||
          Frelax_type[lev] == 99 ||
          Frelax_type[lev] == 199)
      {
         if (lev == 0 && (mgr_data -> fsolver_mode) == 0)
         {
            if (Frelax_type[lev] == 2)
            {
               if (((hypre_ParAMGData*)aff_solver[lev])->A_array != NULL)
               {
                  if (((hypre_ParAMGData*)aff_solver[lev])->A_array[0] != NULL)
                  {
                     /* F-solver is already set up, only need to store A_ff_ptr */
                     A_ff_array[lev] = ((hypre_ParAMGData*) aff_solver[lev]) -> A_array[0];
                  }
                  else
                  {
                     hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                                       "F-relaxation solver has not been setup\n");
                     HYPRE_ANNOTATE_FUNC_END;
                     hypre_GpuProfilingPopRange();

                     return hypre_error_flag;
                  }
               }
               else /* F-relaxation solver prescribed but not set up */
               {
                  /* Save A_FF splitting */
                  A_ff_array[lev] = A_FF;

                  /* Setup F-solver */
                  fgrid_solver_setup(aff_solver[lev],
                                     A_ff_array[lev],
                                     F_fine_array[lev + 1],
                                     U_fine_array[lev + 1]);
                  (mgr_data -> fsolver_mode) = 1;
               }
            }
            else if (aff_solver[lev])
            {
               hypre_sprintf(msg, "Warning!! User-prescribed F-solver for the first level\n\
                             reduction (set in HYPRE_MGRSetFSolver()) only supports AMG\n\
                             Ignoring this call and using user prescribed Frelax_type %d",
                             Frelax_type[lev]);
               hypre_error_w_msg(0, msg);
            }
         }
         else if (aff_solver[lev])
         {
            aff_base = (hypre_Solver*) aff_solver[lev];

            /* Save A_FF splitting */
            A_ff_array[lev] = A_FF;

            /* Call setup function */
            hypre_SolverSetup(aff_base)((HYPRE_Solver) aff_solver[lev],
                                        (HYPRE_Matrix) A_ff_array[lev],
                                        (HYPRE_Vector) F_fine_array[lev + 1],
                                        (HYPRE_Vector) U_fine_array[lev + 1]);
         }
         else if (Frelax_type[lev] == 2) /* Construct default AMG solver */
         {
            /* Save A_FF splitting */
            A_ff_array[lev] = A_FF;

            /* Create BoomerAMG solver for A_FF */
            aff_solver[lev] = (HYPRE_Solver*) hypre_BoomerAMGCreate();
            hypre_BoomerAMGSetMaxIter(aff_solver[lev], (mgr_data -> num_relax_sweeps)[lev]);
            hypre_BoomerAMGSetTol(aff_solver[lev], 0.0);
            //hypre_BoomerAMGSetStrongThreshold(aff_solver[lev], 0.6);
#if defined(HYPRE_USING_GPU)
            hypre_BoomerAMGSetRelaxType(aff_solver[lev], 18);
            hypre_BoomerAMGSetCoarsenType(aff_solver[lev], 8);
            hypre_BoomerAMGSetNumSweeps(aff_solver[lev], 3);
#else
            hypre_BoomerAMGSetRelaxOrder(aff_solver[lev], 1);
#endif
            hypre_BoomerAMGSetPrintLevel(aff_solver[lev], mgr_data -> frelax_print_level);

            fgrid_solver_setup(aff_solver[lev],
                               A_ff_array[lev],
                               F_fine_array[lev + 1],
                               U_fine_array[lev + 1]);

            (mgr_data -> fsolver_mode) = 2;
         }
         else
         {
            /* Save A_FF splitting */
            A_ff_array[lev] = A_FF;
         }

         /* TODO: Check use of A_ff_array[lev], vectors at (lev + 1) are correct? (VPM) */
         F_fine_array[lev + 1] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_ff_array[lev]),
                                  hypre_ParCSRMatrixGlobalNumRows(A_ff_array[lev]),
                                  hypre_ParCSRMatrixRowStarts(A_ff_array[lev]));
         hypre_ParVectorInitialize(F_fine_array[lev + 1]);

         U_fine_array[lev + 1] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_ff_array[lev]),
                                  hypre_ParCSRMatrixGlobalNumRows(A_ff_array[lev]),
                                  hypre_ParCSRMatrixRowStarts(A_ff_array[lev]));
         hypre_ParVectorInitialize(U_fine_array[lev + 1]);
      }

      /* TODO: refactor this block (VPM) */
#if defined (HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_MGRBuildPDevice(A_array[lev], hypre_IntArrayData(FC_marker),
                               row_starts_fpts, 0, &P_FF_array[lev]);
      }
#endif

      /* Destroy A_FF if it has not been saved on A_ff_array[lev] */
      if (!A_ff_array[lev])
      {
         hypre_ParCSRMatrixDestroy(A_FF);
      }
      A_FF = NULL;
      hypre_IntArrayDestroy(FC_marker);

      /* TODO: move this to par_mgr_coarsen.c and port to GPUs (VPM) */
      /* Update coarse level indexes for next levels */
      if (lev < num_coarsening_levs - 1)
      {
         for (i = lev + 1; i < max_num_coarse_levels; i++)
         {
            memory_location = hypre_IntArrayMemoryLocation(CF_marker_array[lev]);
            if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE)
            {
               hypre_IntArrayMigrate(CF_marker_array[lev], HYPRE_MEMORY_HOST);
            }
            CF_marker = hypre_IntArrayData(CF_marker_array[lev]);

            /* First mark indexes to be updated */
            for (j = 0; j < level_coarse_size[i]; j++)
            {
               CF_marker[level_coarse_indexes[i][j]] = S_CMRK;
            }

            /* Next: loop over levels to update indexes */
            nc = 0;
            index_i = 0;
            for (j = 0; j < nloc; j++)
            {
               if (CF_marker[j] == CMRK)
               {
                  nc++;
               }
               if (CF_marker[j] == S_CMRK)
               {
                  level_coarse_indexes[i][index_i++] = nc++;
               }
               //if(index_i == level_coarse_size[i]) break;
            }
            hypre_assert(index_i == level_coarse_size[i]);

            // then: reset previously marked indexes
            for (j = 0; j < level_coarse_size[lev]; j++)
            {
               CF_marker[level_coarse_indexes[lev][j]] = CMRK;
            }

            if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE)
            {
               hypre_IntArrayMigrate(CF_marker_array[lev], HYPRE_MEMORY_DEVICE);
            }
         }
      }

      /* Update reserved coarse indexes to be kept to coarsest level
       * first mark indexes to be updated
       * skip if we reduce the reserved C-points before the coarse grid solve */
      if (mgr_data -> lvl_to_keep_cpoints == 0)
      {
         memory_location = hypre_IntArrayMemoryLocation(CF_marker_array[lev]);
         if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE)
         {
            hypre_IntArrayMigrate(CF_marker_array[lev], HYPRE_MEMORY_HOST);
         }
         CF_marker = hypre_IntArrayData(CF_marker_array[lev]);

         for (i = 0; i < reserved_coarse_size; i++)
         {
            CF_marker[reserved_Cpoint_local_indexes[i]] = S_CMRK;
         }

         /* loop to update reserved Cpoints */
         nc = 0;
         index_i = 0;
         for (i = 0; i < nloc; i++)
         {
            if (CF_marker[i] == CMRK)
            {
               nc++;
            }
            if (CF_marker[i] == S_CMRK)
            {
               reserved_Cpoint_local_indexes[index_i++] = nc++;

               /* reset modified CF marker array indexes */
               CF_marker[i] = CMRK;
            }
         }

         if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE)
         {
            hypre_IntArrayMigrate(CF_marker_array[lev], HYPRE_MEMORY_DEVICE);
         }
      }

      /* allocate space for solution and rhs arrays */
      F_array[lev + 1] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(RAP_ptr),
                               hypre_ParCSRMatrixGlobalNumRows(RAP_ptr),
                               hypre_ParCSRMatrixRowStarts(RAP_ptr));
      hypre_ParVectorInitialize(F_array[lev + 1]);

      U_array[lev + 1] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(RAP_ptr),
                               hypre_ParCSRMatrixGlobalNumRows(RAP_ptr),
                               hypre_ParCSRMatrixRowStarts(RAP_ptr));
      hypre_ParVectorInitialize(U_array[lev + 1]);

      /* free memory before starting next level */
      hypre_ParCSRMatrixDestroy(S);
      S = NULL;

      if (!use_air)
      {
         hypre_ParCSRMatrixDestroy(AT);
         AT = NULL;
      }
      hypre_ParCSRMatrixDestroy(ST);
      ST = NULL;

      /* check if Vcycle smoother setup required */
      if ((mgr_data -> max_local_lvls) > 1)
      {
         if (Frelax_type[lev] == 1)
         {
            use_VcycleSmoother = 1;
            // use_ComplexSmoother = 1;
         }
      }
      else
      {
         /* Only check for vcycle smoother option.
          * Currently leaves Frelax_type[lev] = 2 (full amg) option as is */
         if (Frelax_type[lev] == 1)
         {
            Frelax_type[lev] = 0;
         }
      }

      if (Frelax_type[lev] == 9 ||
          Frelax_type[lev] == 99 ||
          Frelax_type[lev] == 199 )
      {
         use_GSElimSmoother = 1;
      }

      hypre_sprintf(region_name, "%s-%d", "MGR_Level", lev);
      hypre_GpuProfilingPopRange();
      HYPRE_ANNOTATE_REGION_END("%s", region_name);

      /* check if last level */
      if (last_level)
      {
         num_c_levels = lev + 1;
         lev = num_coarsening_levs;
      }
   }

   /* set pointer to last level matrix */
   (mgr_data->num_coarse_levels) = num_c_levels;
   (mgr_data->RAP) = RAP_ptr;

   /* setup default coarsest grid solver (BoomerAMG) */
   if (use_default_cgrid_solver)
   {
      if (my_id == 0 && print_level > 0)
      {
         hypre_printf("No coarse grid solver provided. Using default AMG solver ... \n");
      }

      /* create and set default solver parameters here */
      default_cg_solver = (HYPRE_Solver) hypre_BoomerAMGCreate();
      hypre_BoomerAMGSetMaxIter(default_cg_solver, 1);
      hypre_BoomerAMGSetTol(default_cg_solver, 0.0);
      hypre_BoomerAMGSetRelaxOrder(default_cg_solver, 1);
      hypre_BoomerAMGSetPrintLevel(default_cg_solver, mgr_data -> cg_print_level);

      /* set setup and solve functions */
      cgrid_solver_setup = (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSetup;
      cgrid_solver_solve = (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSolve;
      (mgr_data -> coarse_grid_solver_setup) = cgrid_solver_setup;
      (mgr_data -> coarse_grid_solver_solve) = cgrid_solver_solve;
      (mgr_data -> coarse_grid_solver) = default_cg_solver;
   }

   /* keep reserved coarse indexes to coarsest grid */
   if (reserved_coarse_size > 0 && lvl_to_keep_cpoints == 0)
   {
      ilower = hypre_ParCSRMatrixFirstRowIndex(RAP_ptr);
      for (i = 0; i < reserved_coarse_size; i++)
      {
         reserved_coarse_indexes[i] = (HYPRE_BigInt) (reserved_Cpoint_local_indexes[i] + ilower);
      }
      HYPRE_BoomerAMGSetCPoints((mgr_data ->coarse_grid_solver),
                                25, reserved_coarse_size,
                                reserved_coarse_indexes);
   }

   /* setup coarse grid solver */
   hypre_sprintf(region_name, "%s-%d", "MGR_Level", num_c_levels);
   hypre_GpuProfilingPushRange(region_name);
   HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

   cgrid_solver_setup((mgr_data -> coarse_grid_solver),
                      RAP_ptr, F_array[num_c_levels],
                      U_array[num_c_levels]);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_REGION_END("%s", region_name);

   /* Allocate l1_norms when necessary
      TODO (VPM): move this block closer to global smoother setup */
   for (j = 0; j < num_c_levels; j++)
   {
      if (!l1_norms[j])
      {
         /* Compute l1_norms according to relaxation type */
         hypre_BoomerAMGRelaxComputeL1Norms(A_array[j], Frelax_type[j],
                                            relax_order, 0, CF_marker_array[j],
                                            &l1_norms_data);
         if (l1_norms_data)
         {
            l1_norms[j] = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(A_array[j]));
            hypre_VectorData(l1_norms[j]) = l1_norms_data;
            hypre_VectorMemoryLocation(l1_norms[j]) = memory_location;
         }
      }
   }

   /* Setup Vcycle data for Frelax_type == 1 */
   if (use_VcycleSmoother)
   {
      /* allocate memory and set pointer to (mgr_data -> FrelaxVcycleData) */
      FrelaxVcycleData = hypre_TAlloc(hypre_ParAMGData*, max_num_coarse_levels,
                                      HYPRE_MEMORY_HOST);
      (mgr_data -> FrelaxVcycleData) = FrelaxVcycleData;

      /* Setup temporary storage - TODO (VPM): Use hypre_ParVectorInitialize_v2 */
      VcycleRelaxVtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                               hypre_ParCSRMatrixGlobalNumRows(A),
                                               hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(VcycleRelaxVtemp);
      (mgr_data ->VcycleRelaxVtemp) = VcycleRelaxVtemp;

      VcycleRelaxZtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                               hypre_ParCSRMatrixGlobalNumRows(A),
                                               hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(VcycleRelaxZtemp);
      (mgr_data -> VcycleRelaxZtemp) = VcycleRelaxZtemp;

      /* loop over levels */
      for (i = 0; i < (mgr_data->num_coarse_levels); i++)
      {
         if (Frelax_type[i] == 1)
         {
            FrelaxVcycleData[i] = (hypre_ParAMGData*) hypre_MGRCreateFrelaxVcycleData();
            if (Frelax_num_functions != NULL)
            {
               hypre_ParAMGDataNumFunctions(FrelaxVcycleData[i]) = Frelax_num_functions[i];
            }
            (FrelaxVcycleData[i] -> Vtemp) = VcycleRelaxVtemp;
            (FrelaxVcycleData[i] -> Ztemp) = VcycleRelaxZtemp;

            /* Setup variables for the V-cycle in the F-relaxation step */
            hypre_MGRSetupFrelaxVcycleData(mgr_data, A_array[i], F_array[i], U_array[i], i);
         }
      }
   }
   else if (use_GSElimSmoother)
   {
      /* Allocate memory and set pointer to (mgr_data -> GSElimData) */
      GSElimData = hypre_CTAlloc(hypre_ParAMGData*, max_num_coarse_levels,
                                 HYPRE_MEMORY_HOST);
      (mgr_data -> GSElimData) = GSElimData;

      /* loop over levels */
      for (i = 0; i < (mgr_data->num_coarse_levels); i++)
      {
         if (Frelax_type[i] == 9 || Frelax_type[i] == 99 || Frelax_type[i] == 199)
         {
            GSElimData[i] = (hypre_ParAMGData*) hypre_MGRCreateGSElimData();

            // Set pointers to GSElimData. Here, all solvers point to the same array for
            // A_ff_array, F_fine_array, and U_fine_array and will act on the appropriate
            // components during setup and solve. We adjust {F/U}_fine_array to start at index 1
            // by definition of {F/U}_fine_array.
            (GSElimData[i] -> A_array) = A_ff_array;
            (GSElimData[i] -> F_array) = &F_fine_array[1];
            (GSElimData[i] -> U_array) = &U_fine_array[1];

            /* Save current error code to a temporary variable */
            hypre_error_code_save();

            // setup Gaussian Elim. in the F-relaxation step. Here, we apply GSElim at level 0
            // since we have a single matrix (and not an array of matrices).
            // hypre_printf("Setting GSElim Solver %d \n", Frelax_type[i]);
            hypre_GaussElimSetup(GSElimData[i], i, Frelax_type[i]);

            /* Fallback to Jacobi when Gaussian Elim. is not successful */
            if (HYPRE_GetGlobalError(hypre_ParCSRMatrixComm(A_array[i])))
            {
               hypre_MGRDestroyGSElimData((mgr_data -> GSElimData)[i]);
               (mgr_data -> GSElimData)[i] = NULL;

               Frelax_type[i] = 7; /* Jacobi */
               if (print_level)
               {
                  hypre_ParPrintf(comm, "Switching F-relaxation at level %d to Jacobi", i);
               }

               /* Compute l1_norms if needed */
               if (!l1_norms[i])
               {
                  hypre_BoomerAMGRelaxComputeL1Norms(A_array[i], Frelax_type[i], 0, 0, NULL,
                                                     &l1_norms_data);
                  if (l1_norms_data)
                  {
                     l1_norms[i] = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(A_array[i]));
                     hypre_VectorData(l1_norms[i]) = l1_norms_data;
                     hypre_VectorMemoryLocation(l1_norms[i]) = memory_location;
                  }
               }
            }

            /* Restore error code prior to GaussElimSetup call */
            hypre_error_code_restore();
         }
      }
   }

   if (logging > 1)
   {
      residual = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                       hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                       hypre_ParCSRMatrixRowStarts(A_array[0]) );
      hypre_ParVectorInitialize(residual);
      (mgr_data -> residual) = residual;
   }
   else
   {
      (mgr_data -> residual) = NULL;
   }
   rel_res_norms = hypre_CTAlloc(HYPRE_Real, (mgr_data -> max_iter), HYPRE_MEMORY_HOST);
   (mgr_data -> rel_res_norms) = rel_res_norms;

   /* Free level_coarse_indexes data */
   if (level_coarse_indexes != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         hypre_TFree(level_coarse_indexes[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(level_coarse_indexes, HYPRE_MEMORY_HOST);
      (mgr_data -> level_coarse_indexes) = NULL;

      hypre_TFree(level_coarse_size, HYPRE_MEMORY_HOST);
      (mgr_data -> num_coarse_per_level) = NULL;
   }

   /* Print statistics */
   hypre_MGRSetupStats(mgr_vdata);

   /* Print MGR and linear system info according to print level */
   hypre_MGRDataPrint(mgr_vdata);

   HYPRE_ANNOTATE_FUNC_END;
   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRSetupFrelaxVcycleData
 *
 * Setup data for Frelax V-cycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRSetupFrelaxVcycleData( void               *mgr_vdata,
                                hypre_ParCSRMatrix *A,
                                hypre_ParVector    *f,
                                hypre_ParVector    *u,
                                HYPRE_Int           lev )
{
   MPI_Comm           comm = hypre_ParCSRMatrixComm(A);
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   hypre_ParAMGData    **FrelaxVcycleData = mgr_data -> FrelaxVcycleData;

   HYPRE_Int i, j, num_procs, my_id;

   HYPRE_Int max_local_lvls = (mgr_data -> max_local_lvls);
   HYPRE_Int lev_local;
   HYPRE_Int not_finished;
   HYPRE_Int max_local_coarse_size = hypre_ParAMGDataMaxCoarseSize(FrelaxVcycleData[lev]);
   hypre_IntArray       **CF_marker_array = (mgr_data -> CF_marker_array);
   HYPRE_Int local_size;
   HYPRE_BigInt coarse_size;

   HYPRE_BigInt     coarse_pnts_global_lvl[2];
   hypre_IntArray  *coarse_dof_func_lvl    = NULL;
   hypre_IntArray  *dof_func               = NULL;
   HYPRE_Int       *dof_func_data          = NULL;

   hypre_ParCSRMatrix *RAP_local = NULL;
   hypre_ParCSRMatrix *P_local = NULL;
   hypre_ParCSRMatrix *S_local = NULL;

   HYPRE_Int     smrk_local = -1;
   HYPRE_Int       P_max_elmts = 4;
   HYPRE_Real      trunc_factor = 0.0;
   HYPRE_Int       debug_flag = 0;
   HYPRE_Int       measure_type = 0;
   HYPRE_Real      strong_threshold = 0.25;
   HYPRE_Real      max_row_sum = 0.9;
   HYPRE_Int       coarsen_cut_factor = 0;

   HYPRE_Int       old_num_levels = hypre_ParAMGDataNumLevels(FrelaxVcycleData[lev]);
   hypre_IntArray       **CF_marker_array_local = (FrelaxVcycleData[lev] -> CF_marker_array);
   HYPRE_Int            *CF_marker_local = NULL;
   hypre_ParCSRMatrix   **A_array_local = (FrelaxVcycleData[lev] -> A_array);
   hypre_ParCSRMatrix   **P_array_local = (FrelaxVcycleData[lev] -> P_array);
   hypre_ParVector      **F_array_local = (FrelaxVcycleData[lev] -> F_array);
   hypre_ParVector      **U_array_local = (FrelaxVcycleData[lev] -> U_array);
   hypre_IntArray       **dof_func_array = (FrelaxVcycleData[lev] -> dof_func_array);
   HYPRE_Int            relax_type = 3;
   HYPRE_Int            indx, k, tms;
   HYPRE_Int            num_fine_points = 0;
   HYPRE_Int            num_functions = hypre_ParAMGDataNumFunctions(FrelaxVcycleData[lev]);
   HYPRE_Int            relax_order = hypre_ParAMGDataRelaxOrder(FrelaxVcycleData[lev]);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   local_size = hypre_ParCSRMatrixNumRows(A);

   /* Free any local data not previously destroyed */
   if (A_array_local || P_array_local || CF_marker_array_local)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (A_array_local[j])
         {
            hypre_ParCSRMatrixDestroy(A_array_local[j]);
            A_array_local[j] = NULL;
         }
      }

      for (j = 0; j < old_num_levels - 1; j++)
      {
         if (P_array_local[j])
         {
            hypre_ParCSRMatrixDestroy(P_array_local[j]);
            P_array_local[j] = NULL;
         }
      }

      for (j = 0; j < old_num_levels - 1; j++)
      {
         if (CF_marker_array_local[j])
         {
            hypre_IntArrayDestroy(CF_marker_array_local[j]);
            CF_marker_array_local[j] = NULL;
         }
      }
      hypre_TFree(A_array_local, HYPRE_MEMORY_HOST);
      A_array_local = NULL;
      hypre_TFree(P_array_local, HYPRE_MEMORY_HOST);
      P_array_local = NULL;
      hypre_TFree(CF_marker_array_local, HYPRE_MEMORY_HOST);
      CF_marker_array_local = NULL;
   }
   /* free solution arrays not previously destroyed */
   if (F_array_local != NULL || U_array_local != NULL)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (F_array_local[j] != NULL)
         {
            hypre_ParVectorDestroy(F_array_local[j]);
            F_array_local[j] = NULL;
         }
         if (U_array_local[j] != NULL)
         {
            hypre_ParVectorDestroy(U_array_local[j]);
            U_array_local[j] = NULL;
         }
      }
      hypre_TFree(F_array_local, HYPRE_MEMORY_HOST);
      F_array_local = NULL;
      hypre_TFree(U_array_local, HYPRE_MEMORY_HOST);
      U_array_local = NULL;
   }

   /* Initialize some variables and allocate memory */
   not_finished = 1;
   lev_local = 0;
   if (A_array_local == NULL)
   {
      A_array_local = hypre_CTAlloc(hypre_ParCSRMatrix*,  max_local_lvls, HYPRE_MEMORY_HOST);
   }
   if (P_array_local == NULL && max_local_lvls > 1)
   {
      P_array_local = hypre_CTAlloc(hypre_ParCSRMatrix*,  max_local_lvls - 1, HYPRE_MEMORY_HOST);
   }
   if (F_array_local == NULL)
   {
      F_array_local = hypre_CTAlloc(hypre_ParVector*,  max_local_lvls, HYPRE_MEMORY_HOST);
   }
   if (U_array_local == NULL)
   {
      U_array_local = hypre_CTAlloc(hypre_ParVector*,  max_local_lvls, HYPRE_MEMORY_HOST);
   }
   if (CF_marker_array_local == NULL)
   {
      CF_marker_array_local = hypre_CTAlloc(hypre_IntArray*,  max_local_lvls, HYPRE_MEMORY_HOST);
   }
   if (dof_func_array == NULL)
   {
      dof_func_array = hypre_CTAlloc(hypre_IntArray*, max_local_lvls, HYPRE_MEMORY_HOST);
   }

   A_array_local[0] = A;
   F_array_local[0] = f;
   U_array_local[0] = u;

   for (i = 0; i < local_size; i++)
   {
      if (hypre_IntArrayData(CF_marker_array[lev])[i] == smrk_local)
      {
         num_fine_points++;
      }
   }
   //hypre_printf("My_ID = %d, Size of A_FF matrix: %d \n", my_id, num_fine_points);

   if (num_functions > 1 && dof_func == NULL)
   {
      dof_func = hypre_IntArrayCreate(num_fine_points);
      hypre_IntArrayInitialize(dof_func);
      indx = 0;
      tms = num_fine_points / num_functions;
      if (tms * num_functions + indx > num_fine_points) { tms--; }
      for (j = 0; j < tms; j++)
      {
         for (k = 0; k < num_functions; k++)
         {
            hypre_IntArrayData(dof_func)[indx++] = k;
         }
      }
      k = 0;
      while (indx < num_fine_points)
      {
         hypre_IntArrayData(dof_func)[indx++] = k++;
      }
      FrelaxVcycleData[lev] -> dof_func = dof_func;
   }
   dof_func_array[0] = dof_func;
   hypre_ParAMGDataDofFuncArray(FrelaxVcycleData[lev]) = dof_func_array;

   while (not_finished)
   {
      local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_local[lev_local]));
      dof_func_data = NULL;
      if (dof_func_array[lev_local])
      {
         dof_func_data = hypre_IntArrayData(dof_func_array[lev_local]);
      }

      if (lev_local == 0)
      {
         /* use the CF_marker from the outer MGR cycle to create the strength connection matrix */
         hypre_BoomerAMGCreateSFromCFMarker(A_array_local[lev_local], strong_threshold,
                                            max_row_sum,
                                            hypre_IntArrayData(CF_marker_array[lev]),
                                            num_functions, dof_func_data, smrk_local, &S_local);
         //hypre_ParCSRMatrixPrintIJ(S_local, 0, 0, "S_mat");
      }
      else if (lev_local > 0)
      {
         hypre_BoomerAMGCreateS(A_array_local[lev_local], strong_threshold,
                                max_row_sum, num_functions,
                                dof_func_data, &S_local);
      }

      CF_marker_array_local[lev_local] = hypre_IntArrayCreate(local_size);
      hypre_IntArrayInitialize(CF_marker_array_local[lev_local]);
      CF_marker_local = hypre_IntArrayData(CF_marker_array_local[lev_local]);

      hypre_BoomerAMGCoarsenHMIS(S_local, A_array_local[lev_local], measure_type,
                                 coarsen_cut_factor, debug_flag,
                                 &(CF_marker_array_local[lev_local]));

      hypre_BoomerAMGCoarseParms(comm, local_size,
                                 num_functions, dof_func_array[lev_local],
                                 CF_marker_array_local[lev_local],
                                 &coarse_dof_func_lvl, coarse_pnts_global_lvl);

      if (my_id == (num_procs - 1))
      {
         coarse_size = coarse_pnts_global_lvl[1];
      }
      hypre_MPI_Bcast(&coarse_size, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

      if (coarse_size == 0) // stop coarsening
      {
         if (S_local) { hypre_ParCSRMatrixDestroy(S_local); }
         hypre_IntArrayDestroy(coarse_dof_func_lvl);

         if (lev_local == 0)
         {
            // Save the cf_marker from outer MGR level (lev).
            if (relax_order == 1)
            {
               /* We need to mask out C-points from outer CF-marker for
                  C/F relaxation at solve phase --DOK*/
               for (i = 0; i < local_size; i++)
               {
                  if (hypre_IntArrayData(CF_marker_array[lev])[i] == 1)
                  {
                     CF_marker_local[i] = 0;
                  }
               }
            }
            else
            {
               /* Do lexicographic relaxation on F-points from outer CF-marker --DOK*/
               for (i = 0; i < local_size; i++)
               {
                  CF_marker_local[i] = hypre_IntArrayData(CF_marker_array[lev])[i];
               }
            }
         }
         else
         {
            hypre_IntArrayDestroy(CF_marker_array_local[lev_local]);
            CF_marker_array_local[lev_local] = NULL;
         }
         break;
      }

      hypre_BoomerAMGBuildExtPIInterpHost(A_array_local[lev_local], CF_marker_local,
                                          S_local, coarse_pnts_global_lvl, num_functions, dof_func_data,
                                          debug_flag, trunc_factor, P_max_elmts, &P_local);

      //    hypre_BoomerAMGBuildInterp(A_array_local[lev_local], CF_marker_local,
      //                                   S_local, coarse_pnts_global_lvl, 1, NULL,
      //                                   0, 0.0, 0, NULL, &P_local);

      /* Save the CF_marker pointer. For lev_local = 0, save the cf_marker from outer MGR level (lev).
       * This is necessary to enable relaxations over the A_FF matrix during the solve phase. -- DOK
       */
      if (lev_local == 0)
      {
         if (relax_order == 1)
         {
            /* We need to mask out C-points from outer CF-marker for C/F relaxation at solve phase --DOK*/
            for (i = 0; i < local_size; i++)
            {
               if (hypre_IntArrayData(CF_marker_array[lev])[i] == 1)
               {
                  CF_marker_local[i] = 0;
               }
            }
         }
         else
         {
            /* Do lexicographic relaxation on F-points from outer CF-marker --DOK */
            for (i = 0; i < local_size; i++)
            {
               CF_marker_local[i] = hypre_IntArrayData(CF_marker_array[lev])[i];
            }
         }
      }
      /* Save interpolation matrix pointer */
      P_array_local[lev_local] = P_local;

      if (num_functions > 1)
      {
         dof_func_array[lev_local + 1] = coarse_dof_func_lvl;
      }

      /* build the coarse grid */
      hypre_BoomerAMGBuildCoarseOperatorKT(P_local, A_array_local[lev_local],
                                           P_local, 0, &RAP_local);
      /*
          if (my_id == (num_procs -1)) coarse_size = coarse_pnts_global_lvl[1];
          hypre_MPI_Bcast(&coarse_size, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);
      */
      lev_local++;

      if (S_local) { hypre_ParCSRMatrixDestroy(S_local); }
      S_local = NULL;
      if ( (lev_local == max_local_lvls - 1) || (coarse_size <= max_local_coarse_size) )
      {
         not_finished = 0;
      }

      A_array_local[lev_local] = RAP_local;
      F_array_local[lev_local] = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(RAP_local),
                                                       hypre_ParCSRMatrixGlobalNumRows(RAP_local),
                                                       hypre_ParCSRMatrixRowStarts(RAP_local));
      hypre_ParVectorInitialize(F_array_local[lev_local]);

      U_array_local[lev_local] = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(RAP_local),
                                                       hypre_ParCSRMatrixGlobalNumRows(RAP_local),
                                                       hypre_ParCSRMatrixRowStarts(RAP_local));
      hypre_ParVectorInitialize(U_array_local[lev_local]);
   } // end while loop

   // setup Vcycle data
   (FrelaxVcycleData[lev] -> A_array) = A_array_local;
   (FrelaxVcycleData[lev] -> P_array) = P_array_local;
   (FrelaxVcycleData[lev] -> F_array) = F_array_local;
   (FrelaxVcycleData[lev] -> U_array) = U_array_local;
   (FrelaxVcycleData[lev] -> CF_marker_array) = CF_marker_array_local;
   (FrelaxVcycleData[lev] -> num_levels) = lev_local;
   //if(lev == 1)
   //{
   //  for (i = 0; i < local_size; i++)
   //  {
   //    if(CF_marker_array_local[0][i] == 1)
   //    hypre_printf("cfmarker[%d] = %d\n",i, CF_marker_array_local[0][i]);
   //  }
   //}
   /* setup GE for coarsest level (if small enough) */
   if ((lev_local > 0) && (hypre_ParAMGDataUserCoarseRelaxType(FrelaxVcycleData[lev]) == 9))
   {
      if ((coarse_size <= max_local_coarse_size) && coarse_size > 0)
      {
         hypre_GaussElimSetup(FrelaxVcycleData[lev], lev_local, 9);
      }
      else
      {
         /* use relaxation */
         hypre_ParAMGDataUserCoarseRelaxType(FrelaxVcycleData[lev]) = relax_type;
      }
   }

   return hypre_error_flag;
}
