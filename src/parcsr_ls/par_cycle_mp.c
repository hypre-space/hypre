/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "hypre_parcsr_ls_mup.h"
#include "hypre_parcsr_mv_mup.h"
#include "hypre_utilities_mup.h"
#include "par_amg.h"

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCycle
 *--------------------------------------------------------------------------*/
#ifdef HYPRE_MIXED_PRECISION

HYPRE_Int
hypre_MPAMGCycle_mp( void              *amg_vdata,
                     hypre_ParVector  **F_array,
                     hypre_ParVector  **U_array   )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   HYPRE_Solver *smoother;

   /* Data Structure variables */
   hypre_ParCSRMatrix      **A_array;
   hypre_ParCSRMatrix      **P_array;
   hypre_ParVector          *Utemp;
   hypre_ParVector          *Vtemp_dbl;
   hypre_ParVector          *Vtemp_flt;
   hypre_ParVector          *Vtemp_long_dbl;
   hypre_ParVector          *Ztemp_dbl;
   hypre_ParVector          *Ztemp_flt;
   hypre_ParVector          *Ztemp_long_dbl;
   hypre_ParVector          *Aux_U;
   hypre_ParVector          *Aux_F;

   hypre_IntArray **CF_marker_array;
   HYPRE_Int       *CF_marker;

   HYPRE_Precision *precision_array;  
   HYPRE_Precision level_precision;
   hypre_double    cycle_op_count;
   HYPRE_Int       cycle_type;
   HYPRE_Int       fcycle, fcycle_lev;
   HYPRE_Int       num_levels;
   HYPRE_Int       max_levels;
   hypre_double   *num_coeffs;
   HYPRE_Int      *num_grid_sweeps;
   HYPRE_Int      *grid_relax_type;
   HYPRE_Int     **grid_relax_points;

   /* Local variables  */
   HYPRE_Int      *lev_counter;
   HYPRE_Int       Solve_err_flag;
   HYPRE_Int       k;
   HYPRE_Int       i, j, jj;
   HYPRE_Int       level;
   HYPRE_Int       cycle_param;
   HYPRE_Int       coarse_grid;
   HYPRE_Int       fine_grid;
   HYPRE_Int       Not_Finished;
   HYPRE_Int       num_sweep;
   HYPRE_Int       cg_num_sweep = 1;
   HYPRE_Int       relax_type;
   HYPRE_Int       relax_points;
   HYPRE_Int       relax_order;
   HYPRE_Int       relax_local;
   HYPRE_Int       old_version = 0;
   hypre_double    beta;
   hypre_double    alpha;
   HYPRE_Int       local_size;
   hypre_Vector  **l1_norms = NULL;
   hypre_Vector   *l1_norms_level;
   MPI_Comm        comm;

   hypre_double user_relax_weight_dbl = ((amg_data) -> user_relax_weight_dbl);
   hypre_double outer_wt_dbl = ((amg_data) -> outer_wt_dbl);
   hypre_float user_relax_weight_flt = ((amg_data) -> user_relax_weight_flt);
   hypre_float outer_wt_flt = ((amg_data) -> outer_wt_flt);
   hypre_long_double user_relax_weight_ldbl = ((amg_data) -> user_relax_weight_ldbl);
   hypre_long_double outer_wt_ldbl = ((amg_data) -> outer_wt_ldbl);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Acquire data and allocate storage */
   A_array           = hypre_ParAMGDataAArray(amg_data);
   P_array           = hypre_ParAMGDataPArray(amg_data);
   CF_marker_array   = hypre_ParAMGDataCFMarkerArray(amg_data);
   Vtemp_dbl         = hypre_ParAMGDataVtempDBL(amg_data);
   Vtemp_flt         = hypre_ParAMGDataVtempFLT(amg_data);
   Vtemp_long_dbl    = hypre_ParAMGDataVtempLONGDBL(amg_data);
   Ztemp_dbl         = hypre_ParAMGDataZtempDBL(amg_data);
   Ztemp_flt         = hypre_ParAMGDataZtempFLT(amg_data);
   Ztemp_long_dbl    = hypre_ParAMGDataZtempLONGDBL(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   max_levels        = hypre_ParAMGDataMaxLevels(amg_data);
   cycle_type        = hypre_ParAMGDataCycleType(amg_data);
   fcycle            = hypre_ParAMGDataFCycle(amg_data);

   precision_array   = hypre_ParAMGDataPrecisionArray(amg_data);

   num_grid_sweeps     = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order         = hypre_ParAMGDataRelaxOrder(amg_data);
   l1_norms            = hypre_ParAMGDataL1Norms(amg_data);

   cycle_op_count = ((amg_data) -> cycle_op_count_dbl);

   lev_counter = (HYPRE_Int *) (hypre_CAlloc_dbl((size_t)(num_levels), (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST));

   /* Initialize */
   Solve_err_flag = 0;

   if (grid_relax_points)
   {
      old_version = 1;
   }

   num_coeffs = (hypre_double *) (hypre_CAlloc_dbl ((size_t)(num_levels), (size_t)sizeof(hypre_double), HYPRE_MEMORY_HOST));
   num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
   comm = hypre_ParCSRMatrixComm(A_array[0]);

   for (j = 1; j < num_levels; j++)
   {
      num_coeffs[j] = hypre_ParCSRMatrixDNumNonzeros(A_array[j]);
   }

   /*---------------------------------------------------------------------
    *    Initialize cycling control counter
    *
    *     Cycling is controlled using a level counter: lev_counter[k]
    *
    *     Each time relaxation is performed on level k, the
    *     counter is decremented by 1. If the counter is then
    *     negative, we go to the next finer level. If non-
    *     negative, we go to the next coarser level. The
    *     following actions control cycling:
    *
    *     a. lev_counter[0] is initialized to 1.
    *     b. lev_counter[k] is initialized to cycle_type for k>0.
    *
    *     c. During cycling, when going down to level k, lev_counter[k]
    *        is set to the max of (lev_counter[k],cycle_type)
    *---------------------------------------------------------------------*/

   Not_Finished = 1;

   lev_counter[0] = 1;
   for (k = 1; k < num_levels; ++k)
   {
      if (fcycle)
      {
         lev_counter[k] = 1;
      }
      else
      {
         lev_counter[k] = cycle_type;
      }
   }
   fcycle_lev = num_levels - 2;

   level = 0;
   cycle_param = 1;

   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/

   while (Not_Finished)
   {
      level_precision = precision_array[level];
      if (num_levels > 1)
      {
         local_size = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[level]));
         if (level_precision == HYPRE_REAL_DOUBLE)
	 {
            hypre_ParVectorSetLocalSize_dbl(Vtemp_dbl, local_size);
	 }
	 else if (level_precision == HYPRE_REAL_SINGLE)
	 {
            hypre_ParVectorSetLocalSize_flt(Vtemp_flt, local_size);
	 }
	 else if (level_precision == HYPRE_REAL_LONGDOUBLE)
	 {
            hypre_ParVectorSetLocalSize_long_dbl(Vtemp_long_dbl, local_size);
	 }
	 else
         {
            hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
         }

         cg_num_sweep = 1;
         num_sweep = num_grid_sweeps[cycle_param];
         Aux_U = U_array[level];
         Aux_F = F_array[level];
         
         relax_type = grid_relax_type[cycle_param];
      }
      else /* AB: 4/08: removed the max_levels > 1 check - should do this when max-levels = 1 also */
      {
         /* If no coarsening occurred, apply a simple smoother once */
         Aux_U = U_array[level];
         Aux_F = F_array[level];
         num_sweep = num_grid_sweeps[0];
         /* TK: Use the user relax type (instead of 0) to allow for setting a
           convergent smoother (e.g. in the solution of singular problems). */
         relax_type = hypre_ParAMGDataUserRelaxType(amg_data);
         if (relax_type == -1)
         {
            relax_type = 6;
         }
      }

      if (CF_marker_array[level] != NULL)
      {
         CF_marker = hypre_IntArrayData(CF_marker_array[level]);
      }
      else
      {
         CF_marker = NULL;
      }

      if (l1_norms != NULL)
      {
         l1_norms_level = l1_norms[level];
      }
      else
      {
         l1_norms_level = NULL;
      }

      /*------------------------------------------------------------------
       * Do the relaxation num_sweep times
       *-----------------------------------------------------------------*/

      for (j = 0; j < num_sweep; j++)
      {
         if (num_levels == 1 && max_levels > 1)
         {
            relax_points = 0;
            relax_local = 0;
         }
         else
         {
            if (old_version)
            {
               relax_points = grid_relax_points[cycle_param][j];
            }
            relax_local = relax_order;
         }

        /*-----------------------------------------------
         * VERY sloppy approximation to cycle complexity
         *-----------------------------------------------*/
         if (old_version && level < num_levels - 1)
         {
            switch (relax_points)
            {
               case 1:
                  cycle_op_count += num_coeffs[level + 1];
                  break;

               case -1:
                  cycle_op_count += (num_coeffs[level] - num_coeffs[level + 1]);
                  break;
            }
         }
         else
         {
            cycle_op_count += num_coeffs[level];
         }

         /*-----------------------------------------------
            Choose Smoother
          -----------------------------------------------*/
         if (level_precision == HYPRE_REAL_DOUBLE)
	 {
            if (relax_type == 9   || relax_type == 19  ||
                relax_type == 98  || relax_type == 99  ||
                relax_type == 198 || relax_type == 199)
            {
               /* Gaussian elimination */
               hypre_GaussElimSolve_dbl(amg_data, level, relax_type);
            }
            else if (relax_type == 18)
            {
               /* L1 - Jacobi*/
               Solve_err_flag = hypre_BoomerAMGRelaxIF_dbl(A_array[level],
                                                           Aux_F,
                                                           CF_marker,
                                                           relax_type,
                                                           relax_order,
                                                           cycle_param,
                                                           user_relax_weight_dbl,
                                                           outer_wt_dbl,
                                                           l1_norms_level ? (hypre_double *)hypre_VectorData(l1_norms_level) : NULL,
                                                           Aux_U,
                                                           Vtemp_dbl,
                                                           Ztemp_dbl);
            }
            else if (old_version)
            {
               Solve_err_flag = hypre_BoomerAMGRelax_dbl(A_array[level],
                                                         Aux_F,
                                                         CF_marker,
                                                         relax_type,
                                                         relax_points,
                                                         user_relax_weight_dbl,
                                                         outer_wt_dbl,
                                                         l1_norms_level ? (hypre_double *)hypre_VectorData(l1_norms_level) : NULL,
                                                         Aux_U,
                                                         Vtemp_dbl,
                                                         Ztemp_dbl);
            }
            else
            {
               /* smoother than can have CF ordering */
               Solve_err_flag = hypre_BoomerAMGRelaxIF_dbl(A_array[level],
                                                           Aux_F,
                                                           CF_marker,
                                                           relax_type,
                                                           relax_local,
                                                           cycle_param,
                                                           user_relax_weight_dbl,
                                                           outer_wt_dbl,
                                                           l1_norms_level ? (hypre_double *)hypre_VectorData(l1_norms_level) : NULL,
                                                           Aux_U,
                                                           Vtemp_dbl,
                                                           Ztemp_dbl);
            }
         }
	 else if (level_precision == HYPRE_REAL_SINGLE)
	 {
            if (relax_type == 9   || relax_type == 19  ||
                relax_type == 98  || relax_type == 99  ||
                relax_type == 198 || relax_type == 199)
            {
               /* Gaussian elimination */
               hypre_GaussElimSolve_flt(amg_data, level, relax_type);
            }
            else if (relax_type == 18)
            {
               /* L1 - Jacobi*/
               Solve_err_flag = hypre_BoomerAMGRelaxIF_flt(A_array[level],
                                                           Aux_F,
                                                           CF_marker,
                                                           relax_type,
                                                           relax_order,
                                                           cycle_param,
                                                           user_relax_weight_flt,
                                                           outer_wt_flt,
                                                           l1_norms_level ? (hypre_float *)hypre_VectorData(l1_norms_level) : NULL,
                                                           Aux_U,
                                                           Vtemp_flt,
                                                           Ztemp_flt);
            }
            else if (old_version)
            {
               Solve_err_flag = hypre_BoomerAMGRelax_flt(A_array[level],
                                                         Aux_F,
                                                         CF_marker,
                                                         relax_type,
                                                         relax_points,
                                                         user_relax_weight_flt,
                                                         outer_wt_flt,
                                                         l1_norms_level ? (hypre_float *)hypre_VectorData(l1_norms_level) : NULL,
                                                         Aux_U,
                                                         Vtemp_flt,
                                                         Ztemp_flt);
            }
            else
            {
               /* smoother than can have CF ordering */
               Solve_err_flag = hypre_BoomerAMGRelaxIF_flt(A_array[level],
                                                           Aux_F,
                                                           CF_marker,
                                                           relax_type,
                                                           relax_local,
                                                           cycle_param,
                                                           user_relax_weight_flt,
                                                           outer_wt_flt,
                                                           l1_norms_level ? (hypre_float *)hypre_VectorData(l1_norms_level) : NULL,
                                                           Aux_U,
                                                           Vtemp_flt,
                                                           Ztemp_flt);
            }
         }
	 else if (level_precision == HYPRE_REAL_LONGDOUBLE)
	 {
            if (relax_type == 9   || relax_type == 19  ||
                relax_type == 98  || relax_type == 99  ||
                relax_type == 198 || relax_type == 199)
            {
               /* Gaussian elimination */
               hypre_GaussElimSolve_long_dbl(amg_data, level, relax_type);
            }
            else if (relax_type == 18)
            {
               /* L1 - Jacobi*/
               Solve_err_flag = hypre_BoomerAMGRelaxIF_long_dbl(A_array[level],
                                                           Aux_F,
                                                           CF_marker,
                                                           relax_type,
                                                           relax_order,
                                                           cycle_param,
                                                           user_relax_weight_ldbl,
                                                           outer_wt_ldbl,
                                                           l1_norms_level ? (hypre_long_double *)hypre_VectorData(l1_norms_level) : NULL,
                                                           Aux_U,
                                                           Vtemp_long_dbl,
                                                           Ztemp_long_dbl);
            }
            else if (old_version)
            {
               Solve_err_flag = hypre_BoomerAMGRelax_long_dbl(A_array[level],
                                                         Aux_F,
                                                         CF_marker,
                                                         relax_type,
                                                         relax_points,
                                                         user_relax_weight_ldbl,
                                                         outer_wt_ldbl,
                                                         l1_norms_level ? (hypre_long_double *)hypre_VectorData(l1_norms_level) : NULL,
                                                         Aux_U,
                                                         Vtemp_long_dbl,
                                                         Ztemp_long_dbl);
            }
            else
            {
               /* smoother than can have CF ordering */
               Solve_err_flag = hypre_BoomerAMGRelaxIF_long_dbl(A_array[level],
                                                           Aux_F,
                                                           CF_marker,
                                                           relax_type,
                                                           relax_local,
                                                           cycle_param,
                                                           user_relax_weight_ldbl,
                                                           outer_wt_ldbl,
                                                           l1_norms_level ? (hypre_long_double *) hypre_VectorData(l1_norms_level) : NULL,
                                                           Aux_U,
                                                           Vtemp_long_dbl,
                                                           Ztemp_long_dbl);
            }
         }
	 else
         {
            hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
         }

         if (Solve_err_flag != 0)
         {
            return (Solve_err_flag);
         }
      } /* for (j = 0; j < num_sweep; j++) */

      /*------------------------------------------------------------------
       * Decrement the control counter and determine which grid to visit next
       *-----------------------------------------------------------------*/

      --lev_counter[level];

      //if ( level != num_levels-1 && lev_counter[level] >= 0 )
      if (lev_counter[level] >= 0 && level != num_levels - 1)
      {
         /*---------------------------------------------------------------
          * Visit coarser level next.
          * Compute residual using hypre_ParCSRMatrixMatvec.
          * Perform restriction using hypre_ParCSRMatrixMatvecT.
          * Reset counters and cycling parameters for coarse level
          *--------------------------------------------------------------*/

         fine_grid = level;
         coarse_grid = level + 1;

         if (precision_array[coarse_grid] == HYPRE_REAL_DOUBLE)
	 {
            hypre_ParVectorSetZeros_dbl(U_array[coarse_grid]);
	 }
	 else if (precision_array[coarse_grid] == HYPRE_REAL_SINGLE)
	 {
            hypre_ParVectorSetZeros_flt(U_array[coarse_grid]);
	 }
	 else if (precision_array[coarse_grid] == HYPRE_REAL_LONGDOUBLE)
	 {
            hypre_ParVectorSetZeros_long_dbl(U_array[coarse_grid]);
	 }
	 else
         {
            hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
         }

         // JSP: avoid unnecessary copy using out-of-place version of SpMV
         alpha = -1.0;
         beta = 1.0;
         if (precision_array[fine_grid] == HYPRE_REAL_DOUBLE)
	 {
	    hypre_ParCSRMatrixMatvecOutOfPlace_dbl((hypre_double)alpha, A_array[fine_grid], 
			                           U_array[fine_grid], (hypre_double) beta, 
						   F_array[fine_grid], Vtemp_dbl);
	 }
	 else if (precision_array[fine_grid] == HYPRE_REAL_SINGLE)
	 {
	    hypre_ParCSRMatrixMatvecOutOfPlace_flt((hypre_float)alpha, A_array[fine_grid], 
			                           U_array[fine_grid], (hypre_float) beta, 
						   F_array[fine_grid], Vtemp_flt);
	 }
	 else if (precision_array[fine_grid] == HYPRE_REAL_LONGDOUBLE)
	 {
	    hypre_ParCSRMatrixMatvecOutOfPlace_long_dbl((hypre_long_double)alpha, A_array[fine_grid], 
			                           U_array[fine_grid], (hypre_long_double) beta, 
						   F_array[fine_grid], Vtemp_long_dbl);
	 }
	 else
         {
            hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
         }

         alpha = 1.0;
         beta = 0.0;

	 if (precision_array[fine_grid] != precision_array[coarse_grid])
	 {
	    hypre_ParVectorConvert_mp(F_array[coarse_grid],precision_array[fine_grid]);
	 }

         if (precision_array[fine_grid] == HYPRE_REAL_DOUBLE)
	 {
            hypre_ParCSRMatrixMatvecT_dbl((hypre_double) alpha, P_array[fine_grid], 
			                   Vtemp_dbl, (hypre_double) beta, F_array[coarse_grid]);
	 }
	 else if (precision_array[fine_grid] == HYPRE_REAL_SINGLE)
	 {
            hypre_ParCSRMatrixMatvecT_flt((hypre_float) alpha, P_array[fine_grid], 
			                   Vtemp_flt, (hypre_float) beta, F_array[coarse_grid]);
	 }
	 else if (precision_array[fine_grid] == HYPRE_REAL_LONGDOUBLE)
	 {
            hypre_ParCSRMatrixMatvecT_long_dbl((hypre_long_double) alpha, P_array[fine_grid], 
			                   Vtemp_long_dbl, (hypre_long_double) beta, F_array[coarse_grid]);
	 }
	 else
         {
            hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
         }

	 if (precision_array[fine_grid] != precision_array[coarse_grid])
	 {
	    hypre_ParVectorConvert_mp(F_array[coarse_grid],precision_array[coarse_grid]);
	 }

         ++level;
         lev_counter[level] = hypre_max(lev_counter[level], cycle_type);
         cycle_param = 1;

	 if (level == num_levels - 1)
         {
            cycle_param = 3;
         }
      }
      else if (level != 0)
      {
         /*---------------------------------------------------------------
          * Visit finer level next.
          * Interpolate and add correction using hypre_ParCSRMatrixMatvec.
          * Reset counters and cycling parameters for finer level.
          *--------------------------------------------------------------*/
         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;

	 if (precision_array[fine_grid] != precision_array[coarse_grid])
	 {
	    hypre_ParVectorConvert_mp(U_array[coarse_grid],precision_array[fine_grid]);
	 }

	 if (precision_array[fine_grid] == HYPRE_REAL_DOUBLE)
	 {
            hypre_ParCSRMatrixMatvec_dbl((hypre_double)alpha, P_array[fine_grid],
                                         U_array[coarse_grid],
                                         (hypre_double) beta, U_array[fine_grid]);
	 }
	 else if (precision_array[fine_grid] == HYPRE_REAL_SINGLE)
	 {
            hypre_ParCSRMatrixMatvec_flt((hypre_float)alpha, P_array[fine_grid],
                                         U_array[coarse_grid],
                                         (hypre_float) beta, U_array[fine_grid]);
	 }
	 else if (precision_array[fine_grid] == HYPRE_REAL_LONGDOUBLE)
	 {
            hypre_ParCSRMatrixMatvec_long_dbl((hypre_long_double)alpha, P_array[fine_grid],
                                         U_array[coarse_grid],
                                         (hypre_long_double) beta, U_array[fine_grid]);
	 }
	 else
         {
            hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
         }

	 if (precision_array[fine_grid] != precision_array[coarse_grid])
	 {
	    hypre_ParVectorConvert_mp(U_array[coarse_grid],precision_array[coarse_grid]);
	 }

         hypre_ParVectorAllZeros(U_array[fine_grid]) = 0;

         --level;
         cycle_param = 2;
         if (fcycle && fcycle_lev == level)
         {
            lev_counter[level] = hypre_max(lev_counter[level], 1);
            fcycle_lev --;
         }

      }
      else
      {
         Not_Finished = 0;
      }
   } /* main loop: while (Not_Finished) */

   ((amg_data) -> cycle_op_count_dbl) = cycle_op_count;
   ((amg_data) -> cycle_op_count_flt) = (hypre_float) cycle_op_count;
   ((amg_data) -> cycle_op_count_ldbl) = (hypre_long_double) cycle_op_count;

   hypre_Free_dbl(lev_counter, HYPRE_MEMORY_HOST);
   hypre_Free_dbl(num_coeffs, HYPRE_MEMORY_HOST);

   return (Solve_err_flag);
}
#endif

