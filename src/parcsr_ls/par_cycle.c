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
#include "par_amg.h"
#include "../parcsr_block_mv/par_csr_block_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCycle( void              *amg_vdata,
                      hypre_ParVector  **F_array,
                      hypre_ParVector  **U_array   )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   HYPRE_Solver *smoother;

   /* Data Structure variables */
   hypre_ParCSRMatrix      **A_array;
   hypre_ParCSRMatrix      **P_array;
   hypre_ParCSRMatrix      **R_array;
   hypre_ParVector          *Utemp = NULL;
   hypre_ParVector          *Vtemp;
   hypre_ParVector          *Rtemp;
   hypre_ParVector          *Ptemp;
   hypre_ParVector          *Ztemp;
   hypre_ParVector          *Aux_U;
   hypre_ParVector          *Aux_F;
   hypre_ParCSRBlockMatrix **A_block_array;
   hypre_ParCSRBlockMatrix **P_block_array;
   hypre_ParCSRBlockMatrix **R_block_array;

   HYPRE_Real      *Ztemp_data = NULL;
   HYPRE_Real      *Ptemp_data = NULL;
   hypre_IntArray **CF_marker_array;
   HYPRE_Int       *CF_marker;
   /*
   HYPRE_Int     **unknown_map_array;
   HYPRE_Int     **point_map_array;
   HYPRE_Int     **v_at_point_array;
   */
   HYPRE_Real      cycle_op_count;
   HYPRE_Int       cycle_type;
   HYPRE_Int       fcycle, fcycle_lev;
   HYPRE_Int       num_levels;
   HYPRE_Int       max_levels;
   HYPRE_Real     *num_coeffs;
   HYPRE_Int      *num_grid_sweeps;
   HYPRE_Int      *grid_relax_type;
   HYPRE_Int     **grid_relax_points;
   HYPRE_Int       block_mode;
   HYPRE_Int       cheby_order;

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
   HYPRE_Int       relax_points = 0;
   HYPRE_Int       relax_order;
   HYPRE_Int       relax_local;
   HYPRE_Int       old_version = 0;
   HYPRE_Real     *relax_weight;
   HYPRE_Real     *omega;
   HYPRE_Real      alfa, beta, gammaold;
   HYPRE_Real      gamma = 1.0;
   HYPRE_Int       local_size = 0;
   /*   HYPRE_Int      *smooth_option; */
   HYPRE_Int       smooth_type;
   HYPRE_Int       smooth_num_levels;
   HYPRE_Int       my_id;
   HYPRE_Int       restri_type;
   HYPRE_Real      alpha;
   hypre_Vector  **l1_norms = NULL;
   hypre_Vector   *l1_norms_level;
   hypre_Vector  **ds = hypre_ParAMGDataChebyDS(amg_data);
   HYPRE_Real    **coefs = hypre_ParAMGDataChebyCoefs(amg_data);
   HYPRE_Int       seq_cg = 0;
   HYPRE_Int       partial_cycle_coarsest_level;
   HYPRE_Int       partial_cycle_control;
   MPI_Comm        comm;

   char            nvtx_name[1024];

#if 0
   HYPRE_Real   *D_mat;
   HYPRE_Real   *S_vec;
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("AMGCycle");

   /* Acquire data and allocate storage */
   A_array           = hypre_ParAMGDataAArray(amg_data);
   P_array           = hypre_ParAMGDataPArray(amg_data);
   R_array           = hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = hypre_ParAMGDataCFMarkerArray(amg_data);
   Vtemp             = hypre_ParAMGDataVtemp(amg_data);
   Rtemp             = hypre_ParAMGDataRtemp(amg_data);
   Ptemp             = hypre_ParAMGDataPtemp(amg_data);
   Ztemp             = hypre_ParAMGDataZtemp(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   max_levels        = hypre_ParAMGDataMaxLevels(amg_data);
   cycle_type        = hypre_ParAMGDataCycleType(amg_data);
   fcycle            = hypre_ParAMGDataFCycle(amg_data);

   A_block_array     = hypre_ParAMGDataABlockArray(amg_data);
   P_block_array     = hypre_ParAMGDataPBlockArray(amg_data);
   R_block_array     = hypre_ParAMGDataRBlockArray(amg_data);
   block_mode        = hypre_ParAMGDataBlockMode(amg_data);

   num_grid_sweeps     = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order         = hypre_ParAMGDataRelaxOrder(amg_data);
   relax_weight        = hypre_ParAMGDataRelaxWeight(amg_data);
   omega               = hypre_ParAMGDataOmega(amg_data);
   smooth_type         = hypre_ParAMGDataSmoothType(amg_data);
   smooth_num_levels   = hypre_ParAMGDataSmoothNumLevels(amg_data);
   l1_norms            = hypre_ParAMGDataL1Norms(amg_data);
   /* smooth_option       = hypre_ParAMGDataSmoothOption(amg_data); */
   /* RL */
   restri_type = hypre_ParAMGDataRestriction(amg_data);

   partial_cycle_coarsest_level = hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data);
   partial_cycle_control = hypre_ParAMGDataPartialCycleControl(amg_data);

   /*max_eig_est = hypre_ParAMGDataMaxEigEst(amg_data);
   min_eig_est = hypre_ParAMGDataMinEigEst(amg_data);
   cheby_fraction = hypre_ParAMGDataChebyFraction(amg_data);*/
   cheby_order = hypre_ParAMGDataChebyOrder(amg_data);

   cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

   lev_counter = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);

   if (hypre_ParAMGDataParticipate(amg_data))
   {
      seq_cg = 1;
   }

   /* Initialize */
   Solve_err_flag = 0;

   if (grid_relax_points)
   {
      old_version = 1;
   }

   num_coeffs = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
   num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
   comm = hypre_ParCSRMatrixComm(A_array[0]);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (block_mode)
   {
      for (j = 1; j < num_levels; j++)
      {
         num_coeffs[j] = hypre_ParCSRBlockMatrixNumNonzeros(A_block_array[j]);
      }
   }
   else
   {
      for (j = 1; j < num_levels; j++)
      {
         num_coeffs[j] = hypre_ParCSRMatrixDNumNonzeros(A_array[j]);
      }
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

   smoother = hypre_ParAMGDataSmoother(amg_data);

   if (smooth_num_levels > 0)
   {
      if (smooth_type == 7  || smooth_type == 8  || smooth_type == 9 ||
          smooth_type == 17 || smooth_type == 18 || smooth_type == 19)
      {
         HYPRE_Int actual_local_size = hypre_ParVectorActualLocalSize(Vtemp);
         Utemp = hypre_ParVectorCreate(comm, hypre_ParVectorGlobalSize(Vtemp),
                                       hypre_ParVectorPartitioning(Vtemp));
         local_size = hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp));
         if (local_size < actual_local_size)
         {
            hypre_VectorData(hypre_ParVectorLocalVector(Utemp)) = hypre_CTAlloc(HYPRE_Complex,
                                                                                actual_local_size,
                                                                                HYPRE_MEMORY_HOST);
            hypre_ParVectorActualLocalSize(Utemp) = actual_local_size;
         }
         else
         {
            hypre_ParVectorInitialize(Utemp);
         }
      }
   }

   /* Override level control and cycle param in the case of a partial cycle */
   if (partial_cycle_coarsest_level >= 0)
   {
      if (partial_cycle_control == 0)
      {
         level = 0;
         cycle_param = 1;
      }
      else
      {
         level = partial_cycle_coarsest_level;
         if (level == num_levels - 1)
         {
            cycle_param = 3;
         }
         else
         {
            cycle_param = 2;
         }
         for (k = 0; k < num_levels; ++k)
         {
            lev_counter[k] = 0;
         }
      }
   }

   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/

   HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
   hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
   hypre_GpuProfilingPushRange(nvtx_name);
   while (Not_Finished)
   {
      if (num_levels > 1)
      {
         local_size = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[level]));
         hypre_ParVectorSetLocalSize(Vtemp, local_size);

         if (smooth_num_levels <= level)
         {
            cg_num_sweep = 1;
            num_sweep = num_grid_sweeps[cycle_param];
            Aux_U = U_array[level];
            Aux_F = F_array[level];
         }
         else if (smooth_type > 9)
         {
            hypre_ParVectorSetLocalSize(Ztemp, local_size);
            hypre_ParVectorSetLocalSize(Rtemp, local_size);
            hypre_ParVectorSetLocalSize(Ptemp, local_size);

            Ztemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Ztemp));
            Ptemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Ptemp));
            hypre_ParVectorSetConstantValues(Ztemp, 0.0);
            alpha = -1.0;
            beta = 1.0;

            hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[level],
                                               U_array[level], beta, F_array[level], Rtemp);

            cg_num_sweep = hypre_ParAMGDataSmoothNumSweeps(amg_data);
            num_sweep = num_grid_sweeps[cycle_param];
            Aux_U = Ztemp;
            Aux_F = Rtemp;
         }
         else
         {
            cg_num_sweep = 1;
            num_sweep = hypre_ParAMGDataSmoothNumSweeps(amg_data);
            Aux_U = U_array[level];
            Aux_F = F_array[level];
         }
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

      if (cycle_param == 3 && seq_cg)
      {
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Coarse solve");
         hypre_GpuProfilingPushRange("Coarse solve");
         hypre_seqAMGCycle(amg_data, level, F_array, U_array);
         HYPRE_ANNOTATE_REGION_END("%s", "Coarse solve");
         hypre_GpuProfilingPopRange();
      }
#ifdef HYPRE_USING_DSUPERLU
      else if (cycle_param == 3 && hypre_ParAMGDataDSLUSolver(amg_data) != NULL)
      {
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Coarse solve");
         hypre_GpuProfilingPushRange("Coarse solve");
         hypre_SLUDistSolve(hypre_ParAMGDataDSLUSolver(amg_data), Aux_F, Aux_U);
         HYPRE_ANNOTATE_REGION_END("%s", "Coarse solve");
         hypre_GpuProfilingPopRange();
      }
#endif
      else
      {
         /*------------------------------------------------------------------
         * Do the relaxation num_sweep times
         *-----------------------------------------------------------------*/
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
         hypre_GpuProfilingPushRange("Relaxation");

         for (jj = 0; jj < cg_num_sweep; jj++)
         {
            if (smooth_num_levels > level && smooth_type > 9)
            {
               hypre_ParVectorSetConstantValues(Aux_U, 0.0);
            }

            for (j = 0; j < num_sweep; j++)
            {
               if (num_levels == 1 && max_levels > 1)
               {
                  relax_points = 0;
                  relax_local  = 0;
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
               if ( (smooth_num_levels > level) &&
                    (smooth_type == 7  || smooth_type == 8  || smooth_type == 9 ||
                     smooth_type == 17 || smooth_type == 18 || smooth_type == 19) )
               {
                  hypre_ParVectorSetLocalSize(Utemp, local_size);

                  alpha = -1.0;
                  beta = 1.0;
                  hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[level],
                                                     U_array[level], beta, Aux_F, Vtemp);
                  if (smooth_type == 7 || smooth_type == 17)
                  {
                     HYPRE_ParCSRPilutSolve(smoother[level],
                                            (HYPRE_ParCSRMatrix) A_array[level],
                                            (HYPRE_ParVector) Vtemp,
                                            (HYPRE_ParVector) Utemp);
                  }
                  else if (smooth_type == 8 || smooth_type == 18)
                  {
                     HYPRE_ParCSRParaSailsSolve(smoother[level],
                                                (HYPRE_ParCSRMatrix) A_array[level],
                                                (HYPRE_ParVector) Vtemp,
                                                (HYPRE_ParVector) Utemp);
                  }
                  else if (smooth_type == 9 || smooth_type == 19)
                  {
                     HYPRE_EuclidSolve(smoother[level],
                                       (HYPRE_ParCSRMatrix) A_array[level],
                                       (HYPRE_ParVector) Vtemp,
                                       (HYPRE_ParVector) Utemp);
                  }
                  hypre_ParVectorAxpy(relax_weight[level], Utemp, Aux_U);
               }
               else if ( smooth_num_levels > level && (smooth_type == 4) )
               {
                  HYPRE_FSAISetZeroGuess(smoother[level], cycle_param - 2);
                  HYPRE_FSAISetMaxIterations(smoother[level], num_grid_sweeps[cycle_param]);
                  HYPRE_FSAISolve(smoother[level],
                                  (HYPRE_ParCSRMatrix) A_array[level],
                                  (HYPRE_ParVector) Aux_F,
                                  (HYPRE_ParVector) Aux_U);
               }
               else if ( smooth_num_levels > level && (smooth_type == 5 || smooth_type == 15) )
               {
                  HYPRE_ILUSolve(smoother[level],
                                 (HYPRE_ParCSRMatrix) A_array[level],
                                 (HYPRE_ParVector) Aux_F,
                                 (HYPRE_ParVector) Aux_U);
               }
               else if ( smooth_num_levels > level && (smooth_type == 6 || smooth_type == 16) )
               {
                  HYPRE_SchwarzSolve(smoother[level],
                                     (HYPRE_ParCSRMatrix) A_array[level],
                                     (HYPRE_ParVector) Aux_F,
                                     (HYPRE_ParVector) Aux_U);
               }
               else if (relax_type == 9   ||
                        relax_type == 19  ||
                        relax_type == 98  ||
                        relax_type == 99  ||
                        relax_type == 198 ||
                        relax_type == 199)
               {
                  /* Gaussian elimination */
                  hypre_GaussElimSolve(amg_data, level, relax_type);
               }
               else if (relax_type == 18)
               {
                  /* L1 - Jacobi*/
                  Solve_err_flag = hypre_BoomerAMGRelaxIF(A_array[level],
                                                          Aux_F,
                                                          CF_marker,
                                                          relax_type,
                                                          relax_order,
                                                          cycle_param,
                                                          relax_weight[level],
                                                          omega[level],
                                                          l1_norms_level ? hypre_VectorData(l1_norms_level) : NULL,
                                                          Aux_U,
                                                          Vtemp,
                                                          Ztemp);
               }
               else if (relax_type == 15)
               {
                  /* CG */
                  if (j == 0) /* do num sweep iterations of CG */
                  {
                     hypre_ParCSRRelax_CG( smoother[level],
                                           A_array[level],
                                           Aux_F,
                                           Aux_U,
                                           num_sweep);
                  }
               }
               else if (relax_type == 16)
               {
                  /* scaled Chebyshev */
                  HYPRE_Int scale = hypre_ParAMGDataChebyScale(amg_data);
                  HYPRE_Int variant = hypre_ParAMGDataChebyVariant(amg_data);
                  hypre_ParCSRRelax_Cheby_Solve(A_array[level], Aux_F,
                                                hypre_VectorData(ds[level]), coefs[level],
                                                cheby_order, scale,
                                                variant, Aux_U, Vtemp, Ztemp, Ptemp, Rtemp );
               }
               else if (relax_type == 17)
               {
                  if (level == num_levels - 1)
                  {
                     /* if we are on the coarsest level, the cf_marker will be null
                        and we just do one sweep regular Jacobi */
                     hypre_assert(cycle_param == 3);
                     hypre_BoomerAMGRelax(A_array[level], Aux_F, CF_marker, 0, 0, relax_weight[level],
                                          0.0, NULL, Aux_U, Vtemp, NULL);
                  }
                  else
                  {
                     hypre_BoomerAMGRelax_FCFJacobi(A_array[level], Aux_F, CF_marker, relax_weight[level],
                                                    Aux_U, Vtemp);
                  }
               }
               else if (old_version)
               {
                  Solve_err_flag = hypre_BoomerAMGRelax(A_array[level],
                                                        Aux_F,
                                                        CF_marker,
                                                        relax_type,
                                                        relax_points,
                                                        relax_weight[level],
                                                        omega[level],
                                                        l1_norms_level ? hypre_VectorData(l1_norms_level) : NULL,
                                                        Aux_U,
                                                        Vtemp,
                                                        Ztemp);
               }
               else
               {
                  /* smoother than can have CF ordering */
                  if (block_mode)
                  {
                     Solve_err_flag = hypre_BoomerAMGBlockRelaxIF(A_block_array[level],
                                                                  Aux_F,
                                                                  CF_marker,
                                                                  relax_type,
                                                                  relax_local,
                                                                  cycle_param,
                                                                  relax_weight[level],
                                                                  omega[level],
                                                                  Aux_U,
                                                                  Vtemp);
                  }
                  else
                  {
                     Solve_err_flag = hypre_BoomerAMGRelaxIF(A_array[level],
                                                             Aux_F,
                                                             CF_marker,
                                                             relax_type,
                                                             relax_local,
                                                             cycle_param,
                                                             relax_weight[level],
                                                             omega[level],
                                                             l1_norms_level ? hypre_VectorData(l1_norms_level) : NULL,
                                                             Aux_U,
                                                             Vtemp,
                                                             Ztemp);
                  }
               }

               if (Solve_err_flag != 0)
               {
                  HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");
                  HYPRE_ANNOTATE_MGLEVEL_END(level);
                  HYPRE_ANNOTATE_FUNC_END;
                  hypre_GpuProfilingPopRange();
                  hypre_GpuProfilingPopRange();
                  hypre_GpuProfilingPopRange();
                  return (Solve_err_flag);
               }
            } /* for (j = 0; j < num_sweep; j++) */

            if (smooth_num_levels > level && smooth_type > 9)
            {
               gammaold = gamma;
               gamma = hypre_ParVectorInnerProd(Rtemp, Ztemp);
               if (jj == 0)
               {
                  hypre_ParVectorCopy(Ztemp, Ptemp);
               }
               else
               {
                  beta = gamma / gammaold;
                  /* TODO (VPM): Use a ParVector routine to do the following */
                  for (i = 0; i < local_size; i++)
                  {
                     Ptemp_data[i] = Ztemp_data[i] + beta * Ptemp_data[i];
                  }
               }

               hypre_ParCSRMatrixMatvec(1.0, A_array[level], Ptemp, 0.0, Vtemp);
               alfa = gamma / hypre_ParVectorInnerProd(Ptemp, Vtemp);
               hypre_ParVectorAxpy(alfa, Ptemp, U_array[level]);
               hypre_ParVectorAxpy(-alfa, Vtemp, Rtemp);
            }
         } /* for (jj = 0; jj < cg_num_sweep; jj++) */

         HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");
         hypre_GpuProfilingPopRange();
      }

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

         hypre_ParVectorSetZeros(U_array[coarse_grid]);

         alpha = -1.0;
         beta = 1.0;

         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Residual");
         hypre_GpuProfilingPushRange("Residual");
         if (block_mode)
         {
            hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
            hypre_ParCSRBlockMatrixMatvec(alpha, A_block_array[fine_grid], U_array[fine_grid],
                                          beta, Vtemp);
         }
         else
         {
            // JSP: avoid unnecessary copy using out-of-place version of SpMV
            hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[fine_grid], U_array[fine_grid],
                                               beta, F_array[fine_grid], Vtemp);
         }
         HYPRE_ANNOTATE_REGION_END("%s", "Residual");
         hypre_GpuProfilingPopRange();

         alpha = 1.0;
         beta = 0.0;

         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Restriction");
         hypre_GpuProfilingPushRange("Restriction");
         if (block_mode)
         {
            hypre_ParCSRBlockMatrixMatvecT(alpha, R_block_array[fine_grid], Vtemp,
                                           beta, F_array[coarse_grid]);
         }
         else
         {
            if (restri_type)
            {
               /* RL: no transpose for R */
               hypre_ParCSRMatrixMatvec(alpha, R_array[fine_grid], Vtemp,
                                        beta, F_array[coarse_grid]);
            }
            else
            {
               hypre_ParCSRMatrixMatvecT(alpha, R_array[fine_grid], Vtemp,
                                         beta, F_array[coarse_grid]);
            }
         }
         HYPRE_ANNOTATE_REGION_END("%s", "Restriction");
         HYPRE_ANNOTATE_MGLEVEL_END(level);
         hypre_GpuProfilingPopRange();
         hypre_GpuProfilingPopRange();

         ++level;
         lev_counter[level] = hypre_max(lev_counter[level], cycle_type);
         cycle_param = 1;
         if (level == num_levels - 1)
         {
            cycle_param = 3;
         }
         if (partial_cycle_coarsest_level >= 0 && level == partial_cycle_coarsest_level + 1)
         {
            Not_Finished = 0;
         }
         HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
         hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
         hypre_GpuProfilingPushRange(nvtx_name);
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

         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Interpolation");
         hypre_GpuProfilingPushRange("Interpolation");
         if (block_mode)
         {
            hypre_ParCSRBlockMatrixMatvec(alpha, P_block_array[fine_grid],
                                          U_array[coarse_grid],
                                          beta, U_array[fine_grid]);
         }
         else
         {
            /* printf("Proc %d: level %d, n %d, Interpolation\n", my_id, level, local_size); */
            hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid],
                                     U_array[coarse_grid],
                                     beta, U_array[fine_grid]);
            /* printf("Proc %d: level %d, n %d, Interpolation done\n", my_id, level, local_size); */
         }

         hypre_ParVectorAllZeros(U_array[fine_grid]) = 0;

         HYPRE_ANNOTATE_REGION_END("%s", "Interpolation");
         HYPRE_ANNOTATE_MGLEVEL_END(level);
         hypre_GpuProfilingPopRange();
         hypre_GpuProfilingPopRange();

         --level;
         cycle_param = 2;
         if (fcycle && fcycle_lev == level)
         {
            lev_counter[level] = hypre_max(lev_counter[level], 1);
            fcycle_lev --;
         }

         HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
         hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
         hypre_GpuProfilingPushRange(nvtx_name);
      }
      else
      {
         Not_Finished = 0;
      }
   } /* main loop: while (Not_Finished) */

   HYPRE_ANNOTATE_MGLEVEL_END(level);
   hypre_GpuProfilingPopRange();

   hypre_ParAMGDataCycleOpCount(amg_data) = cycle_op_count;

   hypre_TFree(lev_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(num_coeffs, HYPRE_MEMORY_HOST);

   if (smooth_num_levels > 0)
   {
      if (smooth_type ==  7 || smooth_type ==  8 || smooth_type ==  9 ||
          smooth_type == 17 || smooth_type == 18 || smooth_type == 19 )
      {
         hypre_ParVectorDestroy(Utemp);
      }
   }

   HYPRE_ANNOTATE_FUNC_END;
   hypre_GpuProfilingPopRange();

   return (Solve_err_flag);
}
