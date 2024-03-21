/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * MGR solve routine
 *
 *****************************************************************************/
#include "_hypre_parcsr_ls.h"
#include "par_mgr.h"
#include "par_amg.h"

/*--------------------------------------------------------------------
 * hypre_MGRSolve
 *--------------------------------------------------------------------*/
HYPRE_Int
hypre_MGRSolve( void               *mgr_vdata,
                hypre_ParCSRMatrix *A,
                hypre_ParVector    *f,
                hypre_ParVector    *u )
{

   MPI_Comm              comm = hypre_ParCSRMatrixComm(A);
   hypre_ParMGRData     *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   hypre_ParCSRMatrix **A_array = (mgr_data -> A_array);
   hypre_ParVector    **F_array = (mgr_data -> F_array);
   hypre_ParVector    **U_array = (mgr_data -> U_array);

   HYPRE_Real           tol = (mgr_data -> tol);
   HYPRE_Int            logging = (mgr_data -> logging);
   HYPRE_Int            print_level = (mgr_data -> print_level);
   HYPRE_Int            max_iter = (mgr_data -> max_iter);
   HYPRE_Real          *norms = (mgr_data -> rel_res_norms);
   hypre_ParVector     *Vtemp = (mgr_data -> Vtemp);
   //   hypre_ParVector      *Utemp = (mgr_data -> Utemp);
   hypre_ParVector     *residual = NULL;

   HYPRE_Complex        fp_zero = 0.0;
   HYPRE_Complex        fp_one = 1.0;
   HYPRE_Complex        fp_neg_one = - fp_one;
   HYPRE_Real           conv_factor = 0.0;
   HYPRE_Real           resnorm = 1.0;
   HYPRE_Real           init_resnorm = 0.0;
   HYPRE_Real           rel_resnorm;
   HYPRE_Real           rhs_norm = 0.0;
   HYPRE_Real           old_resnorm;
   HYPRE_Real           ieee_check = 0.;

   HYPRE_Int            iter, num_procs, my_id;

   HYPRE_Solver         cg_solver = (mgr_data -> coarse_grid_solver);
   HYPRE_Int            (*coarse_grid_solver_solve)(void*, void*, void*,
                                                    void*) = (mgr_data -> coarse_grid_solver_solve);

   HYPRE_ANNOTATE_FUNC_BEGIN;
   if (logging > 1)
   {
      residual = (mgr_data -> residual);
   }

   (mgr_data -> num_iterations) = 0;

   if ((mgr_data -> num_coarse_levels) == 0)
   {
      /* Do scalar AMG solve when only one level */
      coarse_grid_solver_solve(cg_solver, A, f, u);
      HYPRE_BoomerAMGGetNumIterations(cg_solver, &iter);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(cg_solver, &rel_resnorm);
      (mgr_data -> num_iterations) = iter;
      (mgr_data -> final_rel_residual_norm) = rel_resnorm;
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   U_array[0] = u;
   F_array[0] = f;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/

   /* Print MGR and linear system info according to print level */
   hypre_MGRDataPrint(mgr_vdata);

   /*-----------------------------------------------------------------------
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && (print_level & HYPRE_MGR_PRINT_INFO_SOLVE) && tol > 0.)
   {
      hypre_printf("\n\nMGR SOLVER SOLUTION INFO:\n");
   }

   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print
    *-----------------------------------------------------------------------*/

   if ((print_level & HYPRE_MGR_PRINT_INFO_SOLVE) || logging > 1 || tol > 0.)
   {
      if (logging > 1)
      {
         hypre_ParVectorCopy(F_array[0], residual);
         if (tol > hypre_cabs(fp_zero))
         {
            hypre_ParCSRMatrixMatvec(fp_neg_one, A_array[0], U_array[0], fp_one, residual);
         }
         resnorm = hypre_sqrt(hypre_ParVectorInnerProd(residual, residual));
      }
      else
      {
         hypre_ParVectorCopy(F_array[0], Vtemp);
         if (tol > hypre_cabs(fp_zero))
         {
            hypre_ParCSRMatrixMatvec(fp_neg_one, A_array[0], U_array[0], fp_one, Vtemp);
         }
         resnorm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
      }

      /* Since it does not diminish performance, attempt to return an error flag
       * and notify users when they supply bad input. */
      if (resnorm != 0.)
      {
         ieee_check = resnorm / resnorm; /* INF -> NaN conversion */
      }

      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
          * for ieee_check self-equality works on all IEEE-compliant compilers/
          * machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
          * by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
          * found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (print_level > 0)
         {
            hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            hypre_printf("ERROR -- hypre_MGRSolve: INFs and/or NaNs detected in input.\n");
            hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         hypre_error(HYPRE_ERROR_GENERIC);
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }

      init_resnorm = resnorm;
      rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(f, f));
      if (rhs_norm > HYPRE_REAL_EPSILON)
      {
         rel_resnorm = init_resnorm / rhs_norm;
      }
      else
      {
         /* rhs is zero, return a zero solution */
         hypre_ParVectorSetZeros(U_array[0]);
         if (logging > 0)
         {
            rel_resnorm = fp_zero;
            (mgr_data -> final_rel_residual_norm) = rel_resnorm;
         }
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }
   }
   else
   {
      rel_resnorm = 1.;
   }

   if (my_id == 0 && (print_level & HYPRE_MGR_PRINT_INFO_SOLVE))
   {
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n", init_resnorm,
                   rel_resnorm);
   }

   /************** Main Solver Loop - always do 1 iteration ************/
   iter = 0;
   while ((rel_resnorm >= tol || iter < 1) && iter < max_iter)
   {
      /* Do one cycle of reduction solve on A*e = r */
      hypre_MGRCycle(mgr_data, F_array, U_array);

      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      if ((print_level & HYPRE_MGR_PRINT_INFO_SOLVE) || logging > 1 || tol > 0.)
      {
         old_resnorm = resnorm;

         if (logging > 1)
         {
            hypre_ParVectorCopy(F_array[0], residual);
            hypre_ParCSRMatrixMatvec(fp_neg_one, A_array[0], U_array[0], fp_one, residual);
            resnorm = hypre_sqrt(hypre_ParVectorInnerProd(residual, residual));
         }
         else
         {
            hypre_ParVectorCopy(F_array[0], Vtemp);
            hypre_ParCSRMatrixMatvec(fp_neg_one, A_array[0], U_array[0], fp_one, Vtemp);
            resnorm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
         }

         conv_factor = (old_resnorm > HYPRE_REAL_EPSILON) ? (resnorm / old_resnorm) : resnorm;
         rel_resnorm = (rhs_norm > HYPRE_REAL_EPSILON) ? (resnorm / rhs_norm) : resnorm;
         norms[iter] = rel_resnorm;
      }

      ++iter;
      (mgr_data -> num_iterations) = iter;
      (mgr_data -> final_rel_residual_norm) = rel_resnorm;

      if (my_id == 0 && (print_level & HYPRE_MGR_PRINT_INFO_SOLVE))
      {
         hypre_printf("    MGRCycle %2d   %e    %f     %e \n", iter,
                      resnorm, conv_factor, rel_resnorm);
      }
   }

   /* check convergence within max_iter */
   if (iter == max_iter && tol > 0.)
   {
      hypre_error(HYPRE_ERROR_CONV);

      if (!my_id && (print_level & HYPRE_MGR_PRINT_INFO_SOLVE))
      {
         hypre_printf("\n\n==============================================");
         hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
         hypre_printf("      within the allowed %d iterations\n", max_iter);
         hypre_printf("==============================================");
      }
   }

   if ((my_id == 0) && (print_level & HYPRE_MGR_PRINT_INFO_SOLVE))
   {
      if (iter > 0 && init_resnorm)
      {
         conv_factor = hypre_pow((resnorm / init_resnorm),
                                 (fp_one / (HYPRE_Real) iter));
      }
      else
      {
         conv_factor = fp_one;
      }

      hypre_printf("\n\n Average Convergence Factor = %f \n", conv_factor);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRFrelaxVcycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRFrelaxVcycle ( void            *Frelax_vdata,
                        hypre_ParVector *f,
                        hypre_ParVector *u )
{
   hypre_ParAMGData    *Frelax_data = (hypre_ParAMGData*) Frelax_vdata;

   HYPRE_Int            Not_Finished = 0;
   HYPRE_Int            level = 0;
   HYPRE_Int            cycle_param = 1;
   HYPRE_Int            j, Solve_err_flag, coarse_grid, fine_grid;
   HYPRE_Int            local_size;
   HYPRE_Int            num_sweeps = 1;
   HYPRE_Int            relax_order = hypre_ParAMGDataRelaxOrder(Frelax_data);
   HYPRE_Int            relax_type = 3;
   HYPRE_Real           relax_weight = 1.0;
   HYPRE_Real           omega = 1.0;

   hypre_ParVector    **F_array = (Frelax_data) -> F_array;
   hypre_ParVector    **U_array = (Frelax_data) -> U_array;

   hypre_ParCSRMatrix **A_array = ((Frelax_data) -> A_array);
   hypre_ParCSRMatrix **R_array = ((Frelax_data) -> P_array);
   hypre_ParCSRMatrix **P_array = ((Frelax_data) -> P_array);
   hypre_IntArray     **CF_marker_array = ((Frelax_data) -> CF_marker_array);
   HYPRE_Int           *CF_marker;

   hypre_ParVector     *Vtemp = (Frelax_data) -> Vtemp;
   hypre_ParVector     *Ztemp = (Frelax_data) -> Ztemp;

   HYPRE_Int            num_c_levels = (Frelax_data) -> num_levels;

   hypre_ParVector     *Aux_F = NULL;
   hypre_ParVector     *Aux_U = NULL;

   HYPRE_Complex        fp_zero = 0.0;
   HYPRE_Complex        fp_one = 1.0;
   HYPRE_Complex        fp_neg_one = - fp_one;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   F_array[0] = f;
   U_array[0] = u;

   CF_marker = NULL;
   if (CF_marker_array[0])
   {
      CF_marker = hypre_IntArrayData(CF_marker_array[0]);
   }

   /* (Re)set local_size for Vtemp */
   local_size = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[0]));
   hypre_ParVectorSetLocalSize(Vtemp, local_size);

   /* smoother on finest level:
    * This is separated from subsequent levels since the finest level matrix
    * may be larger than what is needed for the vcycle solve
    */
   if (relax_order == 1) // C/F ordering for smoother
   {
      for (j = 0; j < num_sweeps; j++)
      {
         Solve_err_flag = hypre_BoomerAMGRelaxIF(A_array[0],
                                                 F_array[0],
                                                 CF_marker,
                                                 relax_type,
                                                 relax_order,
                                                 1,
                                                 relax_weight,
                                                 omega,
                                                 NULL,
                                                 U_array[0],
                                                 Vtemp,
                                                 Ztemp);
      }
   }
   else // lexicographic ordering for smoother (on F points in CF marker)
   {
      for (j = 0; j < num_sweeps; j++)
      {
         Solve_err_flag = hypre_BoomerAMGRelax(A_array[0],
                                               F_array[0],
                                               CF_marker,
                                               relax_type,
                                               -1,
                                               relax_weight,
                                               omega,
                                               NULL,
                                               U_array[0],
                                               Vtemp,
                                               Ztemp);
      }
   }

   /* coarse grids exist */
   if (num_c_levels > 0)
   {
      Not_Finished = 1;
   }

   while (Not_Finished)
   {
      if (cycle_param == 1)
      {
         //hypre_printf("Vcycle smoother (down cycle): vtemp size = %d, level = %d \n", hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp)), level);
         /* compute coarse grid vectors */
         fine_grid   = level;
         coarse_grid = level + 1;

         hypre_ParVectorSetZeros(U_array[coarse_grid]);

         /* Avoid unnecessary copy using out-of-place version of SpMV */
         hypre_ParCSRMatrixMatvecOutOfPlace(fp_neg_one, A_array[fine_grid], U_array[fine_grid],
                                            fp_one, F_array[fine_grid], Vtemp);

         hypre_ParCSRMatrixMatvecT(fp_one, R_array[fine_grid], Vtemp,
                                   fp_zero, F_array[coarse_grid]);

         /* update level */
         ++level;

         /* Update scratch vector sizes */
         local_size = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[level]));
         hypre_ParVectorSetLocalSize(Vtemp, local_size);
         hypre_ParVectorSetLocalSize(Ztemp, local_size);

         CF_marker = NULL;
         if (CF_marker_array[level])
         {
            CF_marker = hypre_IntArrayData(CF_marker_array[level]);
         }

         /* next level is coarsest level */
         if (level == num_c_levels)
         {
            /* switch to coarsest level */
            cycle_param = 3;
         }
         else
         {
            Aux_F = F_array[level];
            Aux_U = U_array[level];
            /* relax and visit next coarse grid */
            for (j = 0; j < num_sweeps; j++)
            {
               Solve_err_flag = hypre_BoomerAMGRelaxIF(A_array[level],
                                                       Aux_F,
                                                       CF_marker,
                                                       relax_type,
                                                       relax_order,
                                                       cycle_param,
                                                       relax_weight,
                                                       omega,
                                                       NULL,
                                                       Aux_U,
                                                       Vtemp,
                                                       Ztemp);
            }
            cycle_param = 1;
         }
      }
      else if (cycle_param == 3)
      {
         if (hypre_ParAMGDataUserCoarseRelaxType(Frelax_data) == 9)
         {
            /* solve the coarsest grid with Gaussian elimination */
            hypre_GaussElimSolve(Frelax_data, level, 9);
         }
         else
         {
            /* solve with relaxation */
            Aux_F = F_array[level];
            Aux_U = U_array[level];
            for (j = 0; j < num_sweeps; j++)
            {
               Solve_err_flag = hypre_BoomerAMGRelaxIF(A_array[level],
                                                       Aux_F,
                                                       CF_marker,
                                                       relax_type,
                                                       relax_order,
                                                       cycle_param,
                                                       relax_weight,
                                                       omega,
                                                       NULL,
                                                       Aux_U,
                                                       Vtemp,
                                                       Ztemp);
            }
         }
         //hypre_printf("Vcycle smoother (coarse level): vtemp size = %d, level = %d \n", hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp)), level);
         cycle_param = 2;
      }
      else if (cycle_param == 2)
      {
         /*---------------------------------------------------------------
          * Visit finer level next.
          * Interpolate and add correction using hypre_ParCSRMatrixMatvec.
          * Reset counters and cycling parameters for finer level.
          *--------------------------------------------------------------*/

         fine_grid   = level - 1;
         coarse_grid = level;

         /* Update solution at the fine level */
         hypre_ParCSRMatrixMatvec(fp_one, P_array[fine_grid],
                                  U_array[coarse_grid],
                                  fp_one, U_array[fine_grid]);

         --level;
         cycle_param = 2;
         if (level == 0) { cycle_param = 99; }

         /* Update scratch vector sizes */
         local_size = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[level]));
         hypre_ParVectorSetLocalSize(Vtemp, local_size);
         hypre_ParVectorSetLocalSize(Ztemp, local_size);
         //hypre_printf("Vcycle smoother (up cycle): vtemp size = %d, level = %d \n", hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp)), level);
      }
      else
      {
         Not_Finished = 0;
      }
   }
   HYPRE_ANNOTATE_FUNC_END;

   return Solve_err_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRCycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRCycle( void              *mgr_vdata,
                hypre_ParVector  **F_array,
                hypre_ParVector  **U_array )
{
   MPI_Comm               comm;
   hypre_ParMGRData      *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   hypre_Solver          *aff_base;

   HYPRE_Int              local_size;
   HYPRE_Int              level;
   HYPRE_Int              coarse_grid;
   HYPRE_Int              fine_grid;
   HYPRE_Int              Not_Finished;
   HYPRE_Int              cycle_type;
   HYPRE_Int              print_level = (mgr_data -> print_level);
   HYPRE_Int              frelax_print_level = (mgr_data -> frelax_print_level);

   HYPRE_Complex         *l1_norms;
   HYPRE_Int             *CF_marker_data;

   hypre_ParCSRMatrix   **A_array    = (mgr_data -> A_array);
   hypre_ParCSRMatrix   **RT_array   = (mgr_data -> RT_array);
   hypre_ParCSRMatrix   **P_array    = (mgr_data -> P_array);
   hypre_ParCSRMatrix   **R_array    = (mgr_data -> R_array);
#if defined(HYPRE_USING_GPU)
   hypre_ParCSRMatrix   **B_array    = (mgr_data -> B_array);
   hypre_ParCSRMatrix   **B_FF_array = (mgr_data -> B_FF_array);
   hypre_ParCSRMatrix   **P_FF_array = (mgr_data -> P_FF_array);
#endif
   hypre_ParCSRMatrix    *RAP        = (mgr_data -> RAP);
   HYPRE_Int              use_default_cgrid_solver = (mgr_data -> use_default_cgrid_solver);
   HYPRE_Solver           cg_solver = (mgr_data -> coarse_grid_solver);
   HYPRE_Int            (*coarse_grid_solver_solve)(void*, void*, void*, void*) =
      (mgr_data -> coarse_grid_solver_solve);

   hypre_IntArray       **CF_marker = (mgr_data -> CF_marker_array);
   HYPRE_Int             *nsweeps = (mgr_data -> num_relax_sweeps);
   HYPRE_Int              relax_type = (mgr_data -> relax_type);
   HYPRE_Real             relax_weight = (mgr_data -> relax_weight);
   HYPRE_Real             omega = (mgr_data -> omega);
   hypre_Vector         **l1_norms_array = (mgr_data -> l1_norms);
   hypre_ParVector       *Vtemp = (mgr_data -> Vtemp);
   hypre_ParVector       *Ztemp = (mgr_data -> Ztemp);
   hypre_ParVector       *Utemp = (mgr_data -> Utemp);

   hypre_ParVector      **U_fine_array = (mgr_data -> U_fine_array);
   hypre_ParVector      **F_fine_array = (mgr_data -> F_fine_array);
   HYPRE_Int            (*fine_grid_solver_solve)(void*, void*, void*, void*) =
      (mgr_data -> fine_grid_solver_solve);
   hypre_ParCSRMatrix   **A_ff_array = (mgr_data -> A_ff_array);

   HYPRE_Int              i, relax_points;
   HYPRE_Int              num_coarse_levels = (mgr_data -> num_coarse_levels);

   HYPRE_Complex          fp_zero = 0.0;
   HYPRE_Complex          fp_one = 1.0;
   HYPRE_Complex          fp_neg_one = - fp_one;

   HYPRE_Int             *Frelax_type = (mgr_data -> Frelax_type);
   HYPRE_Int             *interp_type = (mgr_data -> interp_type);
   hypre_ParAMGData     **FrelaxVcycleData = (mgr_data -> FrelaxVcycleData);
   HYPRE_Real           **frelax_diaginv = (mgr_data -> frelax_diaginv);
   HYPRE_Int             *blk_size = (mgr_data -> blk_size);
   HYPRE_Int              block_size = (mgr_data -> block_size);
   HYPRE_Int             *block_num_coarse_indexes = (mgr_data -> block_num_coarse_indexes);
   /* TODO (VPM): refactor names blk_size and block_size */

   HYPRE_Int             *level_smooth_type = (mgr_data -> level_smooth_type);
   HYPRE_Int             *level_smooth_iters = (mgr_data -> level_smooth_iters);

   HYPRE_Int             *restrict_type  = (mgr_data -> restrict_type);
   HYPRE_Int              pre_smoothing  = (mgr_data -> global_smooth_cycle) == 1 ? 1 : 0;
   HYPRE_Int              post_smoothing = (mgr_data -> global_smooth_cycle) == 2 ? 1 : 0;
   HYPRE_Int              my_id;
   char                   region_name[1024];
   char                   msg[1024];

#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation   memory_location;
   HYPRE_ExecutionPolicy  exec;
#endif

   /* Initialize */
   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("MGRCycle");

   comm = hypre_ParCSRMatrixComm(A_array[0]);
   hypre_MPI_Comm_rank(comm, &my_id);

   Not_Finished = 1;
   cycle_type = 1;
   level = 0;

   /***** Main loop ******/
   while (Not_Finished)
   {
      /* Update scratch vector sizes */
      local_size = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[level]));
      hypre_ParVectorSetLocalSize(Vtemp, local_size);
      hypre_ParVectorSetLocalSize(Ztemp, local_size);
      hypre_ParVectorSetLocalSize(Utemp, local_size);

      /* Do coarse grid correction solve */
      if (cycle_type == 3)
      {
         /* call coarse grid solver here (default is BoomerAMG) */
         hypre_sprintf(region_name, "%s-%d", "MGR_Level", level);
         hypre_GpuProfilingPushRange(region_name);
         HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

         coarse_grid_solver_solve(cg_solver, RAP, F_array[level], U_array[level]);
         if (use_default_cgrid_solver)
         {
            HYPRE_Real convergence_factor_cg;
            hypre_BoomerAMGGetRelResidualNorm(cg_solver, &convergence_factor_cg);
            (mgr_data -> cg_convergence_factor) = convergence_factor_cg;
            if ((print_level) > 1 && my_id == 0 && convergence_factor_cg > hypre_cabs(fp_one))
            {
               hypre_printf("Warning!!! Coarse grid solve diverges. Factor = %1.2e\n",
                            convergence_factor_cg);
            }
         }

         /* Error checking */
         if (HYPRE_GetError())
         {
            hypre_sprintf(msg, "[%d]: Error from MGR's coarsest level solver (level %d)\n",
                          my_id, level);
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
            HYPRE_ClearAllErrors();
         }

         /* DEBUG: print the coarse system indicated by mgr_data->print_coarse_system */
         if (mgr_data -> print_coarse_system)
         {
            hypre_ParCSRMatrixPrintIJ(RAP, 1, 1, "RAP_mat");
            hypre_ParVectorPrintIJ(F_array[level], 1, "RAP_rhs");
            hypre_ParVectorPrintIJ(U_array[level], 1, "RAP_sol");
            mgr_data -> print_coarse_system--;
         }

         /**** cycle up ***/
         cycle_type = 2;

         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_REGION_END("%s", region_name);
      }
      /* Down cycle */
      else if (cycle_type == 1)
      {
         /* Set fine/coarse grid level indices */
         fine_grid       = level;
         coarse_grid     = level + 1;
         l1_norms        = l1_norms_array[fine_grid] ?
                           hypre_VectorData(l1_norms_array[fine_grid]) : NULL;
         CF_marker_data  = hypre_IntArrayData(CF_marker[fine_grid]);

#if defined(HYPRE_USING_GPU)
         memory_location = hypre_ParCSRMatrixMemoryLocation(A_array[fine_grid]);
         exec            = hypre_GetExecPolicy1(memory_location);
#endif

         hypre_sprintf(region_name, "%s-%d", "MGR_Level", fine_grid);
         hypre_GpuProfilingPushRange(region_name);
         HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

         /* Global pre smoothing sweeps */
         if (pre_smoothing && (level_smooth_iters[fine_grid] > 0))
         {
            hypre_sprintf(region_name, "Global-Relax");
            hypre_GpuProfilingPushRange(region_name);
            HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

            if ((level_smooth_type[fine_grid]) == 0 ||
                (level_smooth_type[fine_grid]) == 1)
            {
               /* Block Jacobi/Gauss-Seidel smoother */
#if defined(HYPRE_USING_GPU)
               if (exec == HYPRE_EXEC_DEVICE)
               {
                  for (i = 0; i < level_smooth_iters[fine_grid]; i++)
                  {
                     hypre_MGRBlockRelaxSolveDevice(B_array[fine_grid], A_array[fine_grid],
                                                    F_array[fine_grid], U_array[fine_grid],
                                                    Vtemp, fp_one);
                  }
               }
               else
#endif
               {
                  HYPRE_Real *level_diaginv  = (mgr_data -> level_diaginv)[fine_grid];
                  HYPRE_Int   level_blk_size = (level == 0) ? block_size :
                                               block_num_coarse_indexes[level - 1];
                  HYPRE_Int   nrows          = hypre_ParCSRMatrixNumRows(A_array[fine_grid]);
                  HYPRE_Int   n_block        = nrows / level_blk_size;
                  HYPRE_Int   left_size      = nrows - n_block * level_blk_size;
                  for (i = 0; i < level_smooth_iters[fine_grid]; i++)
                  {
                     hypre_MGRBlockRelaxSolve(A_array[fine_grid], F_array[fine_grid],
                                              U_array[fine_grid], level_blk_size,
                                              n_block, left_size, level_smooth_type[fine_grid],
                                              level_diaginv, Vtemp);
                  }
               }
               hypre_ParVectorAllZeros(U_array[fine_grid]) = 0;
            }
            else if ((level_smooth_type[fine_grid] > 1) &&
                     (level_smooth_type[fine_grid] < 7))
            {
               for (i = 0; i < level_smooth_iters[fine_grid]; i ++)
               {
                  hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid], NULL,
                                       level_smooth_type[fine_grid] - 1, 0, fp_one,
                                       fp_zero, NULL, U_array[fine_grid], Vtemp, NULL);
               }
            }
            else if (level_smooth_type[fine_grid] == 8)
            {
               /* Euclid ILU smoother */
               for (i = 0; i < level_smooth_iters[fine_grid]; i++)
               {
                  /* Compute residual */
                  hypre_ParCSRMatrixMatvecOutOfPlace(fp_neg_one, A_array[fine_grid],
                                                     U_array[fine_grid], fp_one,
                                                     F_array[fine_grid], Vtemp);

                  /* Solve */
                  HYPRE_EuclidSolve((mgr_data -> level_smoother)[fine_grid],
                                    A_array[fine_grid], Vtemp, Utemp);

                  /* Update solution */
                  hypre_ParVectorAxpy(fp_one, Utemp, U_array[fine_grid]);
                  hypre_ParVectorAllZeros(U_array[fine_grid]) = 0;
               }
            }
            else if (level_smooth_type[fine_grid] == 16)
            {
               /* hypre_ILU smoother */
               HYPRE_ILUSolve((mgr_data -> level_smoother)[fine_grid],
                              A_array[fine_grid], F_array[fine_grid],
                              U_array[fine_grid]);
               hypre_ParVectorAllZeros(U_array[fine_grid]) = 0;
            }
            else
            {
               /* Generic relaxation interface */
               for (i = 0; i < level_smooth_iters[fine_grid]; i++)
               {
                  hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid],
                                       NULL, level_smooth_type[fine_grid],
                                       0, fp_one, fp_one, l1_norms,
                                       U_array[fine_grid], Vtemp, Ztemp);
               }
            }

            /* Error checking */
            if (HYPRE_GetError())
            {
               hypre_sprintf(msg, "[%d]: Error from global pre-relaxation %d at level %d \n",
                             my_id, level_smooth_type[fine_grid], fine_grid);
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
               HYPRE_ClearAllErrors();
            }

            hypre_GpuProfilingPopRange();
            HYPRE_ANNOTATE_REGION_END("%s", region_name);
         } /* End global pre-smoothing */

         /* F-relaxation */
         relax_points = -1;
         hypre_sprintf(region_name, "F-Relax");
         hypre_GpuProfilingPushRange(region_name);
         HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

         if (Frelax_type[fine_grid] == 0)
         {
            /* (single level) Block-relaxation for A_ff */
            if (interp_type[fine_grid] == 12)
            {
               HYPRE_Int  nrows     = hypre_ParCSRMatrixNumRows(A_ff_array[fine_grid]);
               HYPRE_Int  n_block   = nrows / blk_size[fine_grid];
               HYPRE_Int  left_size = nrows - n_block * blk_size[fine_grid];

               for (i = 0; i < nsweeps[fine_grid]; i++)
               {
                  /* F-relaxation is reducing the global residual, thus recompute it */
                  hypre_ParCSRMatrixMatvecOutOfPlace(fp_neg_one, A_array[fine_grid],
                                                     U_array[fine_grid], fp_one,
                                                     F_array[fine_grid], Vtemp);

                  /* Restrict to F points */
#if defined(HYPRE_USING_GPU)
                  if (exec == HYPRE_EXEC_DEVICE)
                  {
                     hypre_ParCSRMatrixMatvecT(fp_one, P_FF_array[fine_grid], Vtemp,
                                               fp_zero, F_fine_array[coarse_grid]);
                  }
                  else
#endif
                  {
                     hypre_MGRAddVectorR(CF_marker[fine_grid], FMRK, fp_one, Vtemp,
                                         fp_zero, &(F_fine_array[coarse_grid]));
                  }

                  /* Set initial guess to zero */
                  hypre_ParVectorSetZeros(U_fine_array[coarse_grid]);

#if defined(HYPRE_USING_GPU)
                  if (exec == HYPRE_EXEC_DEVICE)
                  {
                     hypre_MGRBlockRelaxSolveDevice(B_FF_array[fine_grid],
                                                    A_ff_array[fine_grid],
                                                    F_fine_array[fine_grid],
                                                    U_fine_array[fine_grid],
                                                    Vtemp, fp_one);
                  }
                  else
#endif
                  {
                     hypre_MGRBlockRelaxSolve(A_ff_array[fine_grid], F_fine_array[coarse_grid],
                                              U_fine_array[coarse_grid], blk_size[fine_grid],
                                              n_block, left_size, 0, frelax_diaginv[fine_grid],
                                              Vtemp);
                  }

                  /* Interpolate the solution back to the fine grid level */
#if defined(HYPRE_USING_GPU)
                  if (exec == HYPRE_EXEC_DEVICE)
                  {
                     hypre_ParCSRMatrixMatvec(fp_one, P_FF_array[fine_grid],
                                              U_fine_array[coarse_grid], fp_one,
                                              U_fine_array[fine_grid]);
                  }
                  else
#endif
                  {
                     hypre_MGRAddVectorP(CF_marker[fine_grid], FMRK, fp_one,
                                         U_fine_array[coarse_grid], fp_one,
                                         &(U_array[fine_grid]));
                  }
               }
            }
            else
            {
               if (relax_type == 18)
               {
#if defined(HYPRE_USING_GPU)
                  for (i = 0; i < nsweeps[fine_grid]; i++)
                  {
                     hypre_MGRRelaxL1JacobiDevice(A_array[fine_grid], F_array[fine_grid],
                                                  CF_marker_data, relax_points, relax_weight,
                                                  l1_norms, U_array[fine_grid], Vtemp);
                  }
#else
                  for (i = 0; i < nsweeps[fine_grid]; i++)
                  {
                     hypre_ParCSRRelax_L1_Jacobi(A_array[fine_grid], F_array[fine_grid],
                                                 CF_marker_data, relax_points, relax_weight,
                                                 l1_norms, U_array[fine_grid], Vtemp);
                  }
#endif
               }
               else
               {
                  for (i = 0; i < nsweeps[fine_grid]; i++)
                  {
                     hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid],
                                          CF_marker_data, relax_type, relax_points,
                                          relax_weight, omega, l1_norms,
                                          U_array[fine_grid], Vtemp, Ztemp);
                  }
               }
            }
         }
         else if (Frelax_type[fine_grid] == 1)
         {
            /* V-cycle smoother for A_ff */
            //HYPRE_Real convergence_factor_frelax;
            // compute residual before solve
            // hypre_ParCSRMatrixMatvecOutOfPlace(-fp_one, A_array[fine_grid],
            //                                    U_array[fine_grid], fp_one,
            //                                    F_array[fine_grid], Vtemp);
            //  convergence_factor_frelax = hypre_ParVectorInnerProd(Vtemp, Vtemp);

            HYPRE_Real resnorm, init_resnorm;
            HYPRE_Real rhs_norm, old_resnorm;
            HYPRE_Real rel_resnorm = fp_one;
            HYPRE_Real conv_factor = fp_one;
            if (frelax_print_level > 1)
            {
               hypre_ParCSRMatrixMatvecOutOfPlace(fp_neg_one, A_array[fine_grid],
                                                  U_array[fine_grid], fp_one,
                                                  F_array[fine_grid], Vtemp);

               resnorm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
               init_resnorm = resnorm;
               rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(F_array[fine_grid], F_array[fine_grid]));

               if (rhs_norm > HYPRE_REAL_EPSILON)
               {
                  rel_resnorm = init_resnorm / rhs_norm;
               }
               else
               {
                  /* rhs is zero, return a zero solution */
                  hypre_ParVectorSetZeros(U_array[0]);

                  HYPRE_ANNOTATE_FUNC_END;
                  hypre_GpuProfilingPopRange();

                  return hypre_error_flag;
               }
               if (my_id == 0 && frelax_print_level > 1)
               {
                  hypre_printf("\nBegin F-relaxation: V-Cycle Smoother \n");
                  hypre_printf("                                            relative\n");
                  hypre_printf("               residual        factor       residual\n");
                  hypre_printf("               --------        ------       --------\n");
                  hypre_printf("    Initial    %e                 %e\n", init_resnorm,
                               rel_resnorm);
               }
            }

            for (i = 0; i < nsweeps[fine_grid]; i++)
            {
               hypre_MGRFrelaxVcycle(FrelaxVcycleData[fine_grid],
                                     F_array[fine_grid],
                                     U_array[fine_grid]);

               if (frelax_print_level > 1)
               {
                  old_resnorm = resnorm;
                  hypre_ParCSRMatrixMatvecOutOfPlace(fp_neg_one, A_array[fine_grid],
                                                     U_array[fine_grid], fp_one,
                                                     F_array[fine_grid], Vtemp);
                  resnorm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
                  conv_factor = (old_resnorm > HYPRE_REAL_EPSILON) ?
                                (resnorm / old_resnorm) : resnorm;
                  rel_resnorm = (rhs_norm > HYPRE_REAL_EPSILON) ? (resnorm / rhs_norm) : resnorm;

                  if (my_id == 0)
                  {
                     hypre_printf("\n    V-Cycle %2d   %e    %f     %e \n", i,
                                  resnorm, conv_factor, rel_resnorm);
                  }
               }
            }
            if (my_id == 0 && frelax_print_level > 1)
            {
               hypre_printf("End F-relaxation: V-Cycle Smoother \n\n");
            }
            // compute residual after solve
            //hypre_ParCSRMatrixMatvecOutOfPlace(fp_neg_one, A_array[fine_grid],
            //                                   U_array[fine_grid], fp_one,
            //                                   F_array[fine_grid], Vtemp);
            //convergence_factor_frelax = hypre_ParVectorInnerProd(Vtemp, Vtemp)/convergence_factor_frelax;
            //hypre_printf("F-relaxation V-cycle convergence factor: %5f\n", convergence_factor_frelax);
         }
         else if (Frelax_type[level] == 2  ||
                  Frelax_type[level] == 9  ||
                  Frelax_type[level] == 99 ||
                  Frelax_type[level] == 199)
         {
            /* We need to compute the residual first to ensure that
               F-relaxation is reducing the global residual */
            hypre_ParCSRMatrixMatvecOutOfPlace(fp_neg_one, A_array[fine_grid],
                                               U_array[fine_grid], fp_one,
                                               F_array[fine_grid], Vtemp);

            /* Restrict to F points */
#if defined (HYPRE_USING_GPU)
            hypre_ParCSRMatrixMatvecT(fp_one, P_FF_array[fine_grid], Vtemp,
                                      fp_zero, F_fine_array[coarse_grid]);
#else
            hypre_MGRAddVectorR(CF_marker[fine_grid], FMRK, fp_one, Vtemp,
                                fp_zero, &(F_fine_array[coarse_grid]));
#endif

            /* Set initial guess to zeros */
            hypre_ParVectorSetZeros(U_fine_array[coarse_grid]);

            if (Frelax_type[level] == 2)
            {
               /* Do F-relaxation using AMG */
               if (level == 0)
               {
                  /* TODO (VPM): unify with the next block */
                  fine_grid_solver_solve((mgr_data -> aff_solver)[fine_grid],
                                         A_ff_array[fine_grid],
                                         F_fine_array[coarse_grid],
                                         U_fine_array[coarse_grid]);
               }
               else
               {
                  aff_base = (hypre_Solver*) (mgr_data -> aff_solver)[level];

                  hypre_SolverSolve(aff_base)((HYPRE_Solver) (mgr_data -> aff_solver)[level],
                                              (HYPRE_Matrix) A_ff_array[level],
                                              (HYPRE_Vector) F_fine_array[level + 1],
                                              (HYPRE_Vector) U_fine_array[level + 1]);
               }
            }
            else
            {
               /* Do F-relaxation using Gaussian Elimination */
               hypre_GaussElimSolve((mgr_data -> GSElimData)[fine_grid],
                                    level, Frelax_type[level]);
            }

            /* Interpolate the solution back to the fine grid level */
#if defined (HYPRE_USING_GPU)
            hypre_ParCSRMatrixMatvec(fp_one, P_FF_array[fine_grid],
                                     U_fine_array[coarse_grid], fp_one,
                                     U_array[fine_grid]);
#else
            hypre_MGRAddVectorP(CF_marker[fine_grid], FMRK, fp_one,
                                U_fine_array[coarse_grid], fp_one,
                                &(U_array[fine_grid]));
#endif
         }
         else
         {
            for (i = 0; i < nsweeps[fine_grid]; i++)
            {
               hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid],
                                    CF_marker_data, Frelax_type[fine_grid],
                                    relax_points, relax_weight, omega, l1_norms,
                                    U_array[fine_grid], Vtemp, Ztemp);
            }
         }

         /* Error checking */
         if (HYPRE_GetError())
         {
            hypre_sprintf(msg, "[%d]: Error from F-relaxation %d at MGR level %d\n",
                          my_id, Frelax_type[fine_grid], fine_grid);
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
            HYPRE_ClearAllErrors();
         }

         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_REGION_END("%s", region_name);

         /* Update residual and compute coarse-grid rhs */
         hypre_sprintf(region_name, "Residual");
         hypre_GpuProfilingPushRange(region_name);
         HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

         hypre_ParCSRMatrixMatvecOutOfPlace(fp_neg_one, A_array[fine_grid],
                                            U_array[fine_grid], fp_one,
                                            F_array[fine_grid], Vtemp);

         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_REGION_END("%s", region_name);

         hypre_sprintf(region_name, "Restrict");
         hypre_GpuProfilingPushRange(region_name);
         HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);
         if (R_array[fine_grid])
         {
            /* no transpose necessary for R */
            hypre_ParCSRMatrixMatvec(fp_one, R_array[fine_grid], Vtemp,
                                     fp_zero, F_array[coarse_grid]);
         }
         else
         {
#if defined(HYPRE_USING_GPU)
            if (restrict_type[fine_grid] > 0 || (exec == HYPRE_EXEC_DEVICE))
#else
            if (restrict_type[fine_grid] > 0)
#endif
            {
               hypre_ParCSRMatrixMatvecT(fp_one, RT_array[fine_grid], Vtemp,
                                         fp_zero, F_array[coarse_grid]);
            }
            else
            {
               hypre_MGRAddVectorR(CF_marker[fine_grid], CMRK, fp_one,
                                   Vtemp, fp_zero, &(F_array[coarse_grid]));
            }
         }
         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_REGION_END("%s", region_name);

         hypre_sprintf(region_name, "%s-%d", "MGR_Level", fine_grid);
         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_REGION_END("%s", region_name);

         /* Initialize coarse grid solution array (VPM: double-check this for multiple cycles)*/
         hypre_ParVectorSetZeros(U_array[coarse_grid]);

         ++level;
         if (level == num_coarse_levels)
         {
            cycle_type = 3;
         }
      }
      /* Up cycle */
      else if (level != 0)
      {
         /* Set fine/coarse grid level indices */
         fine_grid       = level - 1;
         coarse_grid     = level;
         l1_norms        = l1_norms_array[fine_grid] ?
                           hypre_VectorData(l1_norms_array[fine_grid]) : NULL;
         CF_marker_data  = hypre_IntArrayData(CF_marker[fine_grid]);

#if defined(HYPRE_USING_GPU)
         memory_location = hypre_ParCSRMatrixMemoryLocation(A_array[fine_grid]);
         exec            = hypre_GetExecPolicy1(memory_location);
#endif

         hypre_sprintf(region_name, "%s-%d", "MGR_Level", fine_grid);
         hypre_GpuProfilingPushRange(region_name);
         HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

         /* Interpolate */
         hypre_sprintf(region_name, "Prolongate");
         hypre_GpuProfilingPushRange(region_name);
         HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

#if defined(HYPRE_USING_GPU)
         if (interp_type[fine_grid] > 0 || (exec == HYPRE_EXEC_DEVICE))
#else
         if (interp_type[fine_grid] > 0)
#endif
         {
            hypre_ParCSRMatrixMatvec(fp_one, P_array[fine_grid],
                                     U_array[coarse_grid],
                                     fp_one, U_array[fine_grid]);
         }
         else
         {
            hypre_MGRAddVectorP(CF_marker[fine_grid], CMRK, fp_one,
                                U_array[coarse_grid], fp_one,
                                &(U_array[fine_grid]));
         }

         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_REGION_END("%s", region_name);

         /* Global post smoothing sweeps */
         if (post_smoothing & (level_smooth_iters[fine_grid] > 0))
         {
            hypre_sprintf(region_name, "Global-Relax");
            hypre_GpuProfilingPushRange(region_name);
            HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);

            /* Block Jacobi smoother */
            if ((level_smooth_type[fine_grid] == 0) ||
                (level_smooth_type[fine_grid] == 1))
            {
#if defined(HYPRE_USING_GPU)
               if (exec == HYPRE_EXEC_DEVICE)
               {
                  for (i = 0; i < level_smooth_iters[fine_grid]; i++)
                  {
                     hypre_MGRBlockRelaxSolveDevice(B_array[fine_grid], A_array[fine_grid],
                                                    F_array[fine_grid], U_array[fine_grid],
                                                    Vtemp, fp_one);
                  }
               }
               else
#endif
               {
                  HYPRE_Real *level_diaginv  = (mgr_data -> level_diaginv)[fine_grid];
                  HYPRE_Int   level_blk_size = (fine_grid == 0) ? block_size :
                                               block_num_coarse_indexes[fine_grid - 1];
                  HYPRE_Int   nrows          = hypre_ParCSRMatrixNumRows(A_array[fine_grid]);
                  HYPRE_Int   n_block        = nrows / level_blk_size;
                  HYPRE_Int   left_size      = nrows - n_block * level_blk_size;
                  for (i = 0; i < level_smooth_iters[fine_grid]; i++)
                  {
                     hypre_MGRBlockRelaxSolve(A_array[fine_grid], F_array[fine_grid],
                                              U_array[fine_grid], level_blk_size, n_block,
                                              left_size, level_smooth_type[fine_grid],
                                              level_diaginv, Vtemp);
                  }
               }
            }
            else if ((level_smooth_type[fine_grid] > 1) && (level_smooth_type[fine_grid] < 7))
            {
               for (i = 0; i < level_smooth_iters[fine_grid]; i++)
               {
                  hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid], NULL,
                                       level_smooth_type[fine_grid] - 1, 0, fp_one,
                                       fp_zero, l1_norms, U_array[fine_grid], Vtemp, NULL);
               }
            }
            else if (level_smooth_type[fine_grid] == 8)
            {
               /* Euclid ILU */
               for (i = 0; i < level_smooth_iters[fine_grid]; i++)
               {
                  /* Compute residual */
                  hypre_ParCSRMatrixMatvecOutOfPlace(fp_neg_one, A_array[fine_grid],
                                                     U_array[fine_grid], fp_one,
                                                     F_array[fine_grid], Vtemp);
                  /* Solve */
                  HYPRE_EuclidSolve((mgr_data -> level_smoother)[fine_grid],
                                    A_array[fine_grid], Vtemp, Utemp);

                  /* Update solution */
                  hypre_ParVectorAxpy(fp_one, Utemp, U_array[fine_grid]);
               }
            }
            else if (level_smooth_type[fine_grid] == 16)
            {
               /* HYPRE ILU */
               HYPRE_ILUSolve((mgr_data -> level_smoother)[fine_grid],
                              A_array[fine_grid], F_array[fine_grid],
                              U_array[fine_grid]);
            }
            else
            {
               /* Generic relaxation interface */
               for (i = 0; i < level_smooth_iters[level]; i++)
               {
                  hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid],
                                       NULL, level_smooth_type[fine_grid], 0,
                                       fp_one, fp_one, l1_norms,
                                       U_array[fine_grid], Vtemp, Ztemp);
               }
            }

            /* Error checking */
            if (HYPRE_GetError())
            {
               hypre_sprintf(msg, "[%d]: Error from global post-relaxation %d at MGR level %d\n",
                             my_id, level_smooth_type[fine_grid], fine_grid);
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
               HYPRE_ClearAllErrors();
            }

            hypre_GpuProfilingPopRange();
            HYPRE_ANNOTATE_REGION_END("%s", region_name);
         } /* End post-smoothing */

         hypre_sprintf(region_name, "%s-%d", "MGR_Level", fine_grid);
         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_REGION_END("%s", region_name);

         --level;
      } /* End interpolate */
      else
      {
         Not_Finished = 0;
      }
   }
   HYPRE_ANNOTATE_FUNC_END;
   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}
