/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * AMG transpose solve routines
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"

/*--------------------------------------------------------------------
 * hypre_BoomerAMGSolveT
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSolveT( void               *amg_vdata,
                       hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u         )
{

   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);

   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */

   HYPRE_Int      amg_print_level;
   HYPRE_Int      amg_logging;
   HYPRE_Real  *num_coeffs;
   HYPRE_Int     *num_variables;
   HYPRE_Real   cycle_op_count;
   HYPRE_Int      num_levels;
   /* HYPRE_Int      num_unknowns; */
   HYPRE_Real   tol;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   /*  Local variables  */

   /*FILE    *fp;*/

   HYPRE_Int      j;
   HYPRE_Int      Solve_err_flag;
   HYPRE_Int      min_iter;
   HYPRE_Int      max_iter;
   HYPRE_Int      cycle_count;
   HYPRE_Real   total_coeffs;
   HYPRE_Int      total_variables;
   HYPRE_Int      num_procs, my_id;

   HYPRE_Real   alpha = 1.0;
   HYPRE_Real   beta = -1.0;
   HYPRE_Real   cycle_cmplxty = 0.0;
   HYPRE_Real   operat_cmplxty;
   HYPRE_Real   grid_cmplxty;
   HYPRE_Real   conv_factor;
   HYPRE_Real   resid_nrm;
   HYPRE_Real   resid_nrm_init;
   HYPRE_Real   relative_resid;
   HYPRE_Real   rhs_norm;
   HYPRE_Real   old_resid;

   hypre_ParVector  *Vtemp;
   hypre_ParVector  *Residual;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   amg_print_level = hypre_ParAMGDataPrintLevel(amg_data);
   amg_logging   = hypre_ParAMGDataLogging(amg_data);
   if ( amg_logging > 1 )
   {
      Residual = hypre_ParAMGDataResidual(amg_data);
   }
   /* num_unknowns  = hypre_ParAMGDataNumUnknowns(amg_data); */
   num_levels    = hypre_ParAMGDataNumLevels(amg_data);
   A_array       = hypre_ParAMGDataAArray(amg_data);
   F_array       = hypre_ParAMGDataFArray(amg_data);
   U_array       = hypre_ParAMGDataUArray(amg_data);

   tol           = hypre_ParAMGDataTol(amg_data);
   min_iter      = hypre_ParAMGDataMinIter(amg_data);
   max_iter      = hypre_ParAMGDataMaxIter(amg_data);

   num_coeffs = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
   num_variables = hypre_CTAlloc(HYPRE_Int,  num_levels, HYPRE_MEMORY_HOST);
   num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
   num_variables[0] = hypre_ParCSRMatrixGlobalNumRows(A_array[0]);

   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*   Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                    hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                    hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize(Vtemp);
      hypre_ParAMGDataVtemp(amg_data) = Vtemp;
   */
   Vtemp = hypre_ParAMGDataVtemp(amg_data);
   for (j = 1; j < num_levels; j++)
   {
      num_coeffs[j]    = hypre_ParCSRMatrixDNumNonzeros(A_array[j]);
      num_variables[j] = hypre_ParCSRMatrixGlobalNumRows(A_array[j]);
   }

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/


   if (my_id == 0 && amg_print_level > 1)
   {
      hypre_BoomerAMGWriteSolverParams(amg_data);
   }



   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag and assorted bookkeeping variables
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;

   total_coeffs = 0;
   total_variables = 0;
   cycle_count = 0;
   operat_cmplxty = 0;
   grid_cmplxty = 0;

   /*-----------------------------------------------------------------------
    *     open the log file and write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && amg_print_level > 1)
   {
      /*fp = fopen(file_name, "a");*/

      hypre_printf("\n\nAMG SOLUTION INFO:\n");

   }

   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print to logfile
    *-----------------------------------------------------------------------*/

   if ( amg_logging > 1 )
   {
      hypre_ParVectorCopy(F_array[0], Residual );
      hypre_ParCSRMatrixMatvecT(alpha, A_array[0], U_array[0], beta, Residual );
      resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd( Residual, Residual ));
   }
   else
   {
      hypre_ParVectorCopy(F_array[0], Vtemp);
      hypre_ParCSRMatrixMatvecT(alpha, A_array[0], U_array[0], beta, Vtemp);
      resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
   }


   resid_nrm_init = resid_nrm;
   rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(f, f));
   relative_resid = 9999;
   if (rhs_norm)
   {
      relative_resid = resid_nrm_init / rhs_norm;
   }

   if (my_id == 0 && (amg_print_level > 1))
   {
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n", resid_nrm_init,
                   relative_resid);
   }

   /*-----------------------------------------------------------------------
    *    Main V-cycle loop
    *-----------------------------------------------------------------------*/

   while ((relative_resid >= tol || cycle_count < min_iter)
          && cycle_count < max_iter
          && Solve_err_flag == 0)
   {
      hypre_ParAMGDataCycleOpCount(amg_data) = 0;
      /* Op count only needed for one cycle */

      Solve_err_flag = hypre_BoomerAMGCycleT(amg_data, F_array, U_array);

      old_resid = resid_nrm;

      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      if ( amg_logging > 1 )
      {
         hypre_ParVectorCopy(F_array[0], Residual );
         hypre_ParCSRMatrixMatvecT(alpha, A_array[0], U_array[0], beta, Residual );
         resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd( Residual, Residual ));
      }
      else
      {
         hypre_ParVectorCopy(F_array[0], Vtemp);
         hypre_ParCSRMatrixMatvecT(alpha, A_array[0], U_array[0], beta, Vtemp);
         resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
      }

      conv_factor = resid_nrm / old_resid;
      relative_resid = 9999;
      if (rhs_norm)
      {
         relative_resid = resid_nrm / rhs_norm;
      }

      ++cycle_count;



      hypre_ParAMGDataRelativeResidualNorm(amg_data) = relative_resid;
      hypre_ParAMGDataNumIterations(amg_data) = cycle_count;

      if (my_id == 0 && (amg_print_level > 1))
      {
         hypre_printf("    Cycle %2d   %e    %f     %e \n", cycle_count,
                      resid_nrm, conv_factor, relative_resid);
      }
   }

   if (cycle_count == max_iter) { Solve_err_flag = 1; }

   /*-----------------------------------------------------------------------
    *    Compute closing statistics
    *-----------------------------------------------------------------------*/

   conv_factor = hypre_pow((resid_nrm / resid_nrm_init), (1.0 / ((HYPRE_Real) cycle_count)));


   for (j = 0; j < hypre_ParAMGDataNumLevels(amg_data); j++)
   {
      total_coeffs += num_coeffs[j];
      total_variables += num_variables[j];
   }

   cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

   if (num_variables[0])
   {
      grid_cmplxty = ((HYPRE_Real) total_variables) / ((HYPRE_Real) num_variables[0]);
   }
   if (num_coeffs[0])
   {
      operat_cmplxty = total_coeffs / num_coeffs[0];
      cycle_cmplxty = cycle_op_count / num_coeffs[0];
   }

   if (my_id == 0 && amg_print_level > 1)
   {
      if (Solve_err_flag == 1)
      {
         hypre_printf("\n\n==============================================");
         hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
         hypre_printf("      within the allowed %d V-cycles\n", max_iter);
         hypre_printf("==============================================");
      }
      hypre_printf("\n\n Average Convergence Factor = %f", conv_factor);
      hypre_printf("\n\n     Complexity:    grid = %f\n", grid_cmplxty);
      hypre_printf("                operator = %f\n", operat_cmplxty);
      hypre_printf("                   cycle = %f\n\n", cycle_cmplxty);
   }

   /*----------------------------------------------------------
    * Close the output file (if open)
    *----------------------------------------------------------*/

   /*if (my_id == 0 && amg_print_level >= 1)
   {
      fclose(fp);
   }*/

   hypre_TFree(num_coeffs, HYPRE_MEMORY_HOST);
   hypre_TFree(num_variables, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return (Solve_err_flag);
}

/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCycleT
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCycleT( void              *amg_vdata,
                       hypre_ParVector  **F_array,
                       hypre_ParVector  **U_array   )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */

   hypre_ParCSRMatrix    **A_array;
   hypre_ParCSRMatrix    **P_array;
   hypre_ParCSRMatrix    **R_array;
   hypre_ParVector    *Vtemp;

   hypre_IntArray   **CF_marker_array;
   HYPRE_Int         *CF_marker;
   /* HYPRE_Int     **unknown_map_array; */
   /* HYPRE_Int     **point_map_array; */
   /* HYPRE_Int     **v_at_point_array; */

   HYPRE_Real    cycle_op_count;
   HYPRE_Int       cycle_type;
   HYPRE_Int       num_levels;
   HYPRE_Int       max_levels;

   HYPRE_Real   *num_coeffs;
   HYPRE_Int      *num_grid_sweeps;
   HYPRE_Int      *grid_relax_type;
   HYPRE_Int     **grid_relax_points;

   /* Local variables  */

   HYPRE_Int      *lev_counter;
   HYPRE_Int       Solve_err_flag;
   HYPRE_Int       k;
   HYPRE_Int       j;
   HYPRE_Int       level;
   HYPRE_Int       cycle_param;
   HYPRE_Int       coarse_grid;
   HYPRE_Int       fine_grid;
   HYPRE_Int       Not_Finished;
   HYPRE_Int       num_sweep;
   HYPRE_Int       relax_type;
   HYPRE_Int       relax_points = 0;
   HYPRE_Real     *relax_weight;

   HYPRE_Int       old_version = 0;


   HYPRE_Real    alpha;
   HYPRE_Real    beta;
#if 0
   HYPRE_Real   *D_mat;
   HYPRE_Real   *S_vec;
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Acquire data and allocate storage */

   A_array           = hypre_ParAMGDataAArray(amg_data);
   P_array           = hypre_ParAMGDataPArray(amg_data);
   R_array           = hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = hypre_ParAMGDataCFMarkerArray(amg_data);
   /* unknown_map_array = hypre_ParAMGDataUnknownMapArray(amg_data); */
   /* point_map_array   = hypre_ParAMGDataPointMapArray(amg_data); */
   /* v_at_point_array  = hypre_ParAMGDataVatPointArray(amg_data); */
   Vtemp             = hypre_ParAMGDataVtemp(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   max_levels        = hypre_ParAMGDataMaxLevels(amg_data);
   cycle_type        = hypre_ParAMGDataCycleType(amg_data);
   /* num_unknowns      =  hypre_ParCSRMatrixNumRows(A_array[0]); */

   num_grid_sweeps     = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_weight        = hypre_ParAMGDataRelaxWeight(amg_data);

   cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

   lev_counter = hypre_CTAlloc(HYPRE_Int,  num_levels, HYPRE_MEMORY_HOST);

   /* Initialize */

   Solve_err_flag = 0;

   if (grid_relax_points) { old_version = 1; }

   num_coeffs = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
   num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A_array[0]);

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
      lev_counter[k] = cycle_type;
   }

   level = 0;
   cycle_param = 0;

   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/

   HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
   while (Not_Finished)
   {
      num_sweep = num_grid_sweeps[cycle_param];
      relax_type = grid_relax_type[cycle_param];
      if (relax_type != 7 && relax_type != 9)
      {
         relax_type = 7;
      }

      /*------------------------------------------------------------------
       * Do the relaxation num_sweep times
       *-----------------------------------------------------------------*/

      for (j = 0; j < num_sweep; j++)
      {

         if (num_levels == 1 && max_levels > 1)
         {
            relax_points = 0;
         }
         else
         {
            if (old_version)
            {
               relax_points = grid_relax_points[cycle_param][j];
            }
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

         /* note: this does not use relax_points, so it doesn't matter if
            its the "old version" */

         if (CF_marker_array[level] == NULL)
         {
            CF_marker = NULL;
         }
         else
         {
            CF_marker = hypre_IntArrayData(CF_marker_array[level]);
         }
         Solve_err_flag = hypre_BoomerAMGRelaxT(A_array[level],
                                                F_array[level],
                                                CF_marker,
                                                relax_type,
                                                relax_points,
                                                relax_weight[level],
                                                U_array[level],
                                                Vtemp);


         if (Solve_err_flag != 0)
         {
            hypre_TFree(lev_counter, HYPRE_MEMORY_HOST);
            hypre_TFree(num_coeffs, HYPRE_MEMORY_HOST);
            HYPRE_ANNOTATE_MGLEVEL_END(level);
            HYPRE_ANNOTATE_FUNC_END;

            return (Solve_err_flag);
         }
      }


      /*------------------------------------------------------------------
       * Decrement the control counter and determine which grid to visit next
       *-----------------------------------------------------------------*/

      --lev_counter[level];

      if (lev_counter[level] >= 0 && level != num_levels - 1)
      {

         /*---------------------------------------------------------------
          * Visit coarser level next.  Compute residual using hypre_ParCSRMatrixMatvec.
          * Use interpolation (since transpose i.e. P^TATR instead of
          * RAP) using hypre_ParCSRMatrixMatvecT.
          * Reset counters and cycling parameters for coarse level
          *--------------------------------------------------------------*/

         fine_grid = level;
         coarse_grid = level + 1;

         hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0);

         hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
         alpha = -1.0;
         beta = 1.0;
         hypre_ParCSRMatrixMatvecT(alpha, A_array[fine_grid], U_array[fine_grid],
                                   beta, Vtemp);

         alpha = 1.0;
         beta = 0.0;

         hypre_ParCSRMatrixMatvecT(alpha, P_array[fine_grid], Vtemp,
                                   beta, F_array[coarse_grid]);

         HYPRE_ANNOTATE_MGLEVEL_END(level);

         ++level;
         lev_counter[level] = hypre_max(lev_counter[level], cycle_type);
         cycle_param = 1;
         if (level == num_levels - 1) { cycle_param = 3; }

         HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
      }

      else if (level != 0)
      {

         /*---------------------------------------------------------------
          * Visit finer level next.
          * Use restriction (since transpose i.e. P^TA^TR instead of RAP)
          * and add correction using hypre_ParCSRMatrixMatvec.
          * Reset counters and cycling parameters for finer level.
          *--------------------------------------------------------------*/

         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;

         hypre_ParCSRMatrixMatvec(alpha, R_array[fine_grid], U_array[coarse_grid],
                                  beta, U_array[fine_grid]);

         HYPRE_ANNOTATE_MGLEVEL_END(level);

         --level;
         cycle_param = 2;
         if (level == 0) { cycle_param = 0; }

         HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
      }
      else
      {
         Not_Finished = 0;
      }
   }

   HYPRE_ANNOTATE_MGLEVEL_END(level);

   hypre_ParAMGDataCycleOpCount(amg_data) = cycle_op_count;
   hypre_TFree(lev_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(num_coeffs, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return (Solve_err_flag);
}

/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGRelaxT
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGRelaxT( hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       HYPRE_Int          *cf_marker,
                       HYPRE_Int           relax_type,
                       HYPRE_Int           relax_points,
                       HYPRE_Real          relax_weight,
                       hypre_ParVector    *u,
                       hypre_ParVector    *Vtemp )
{
   HYPRE_UNUSED_VAR(cf_marker);
   HYPRE_UNUSED_VAR(relax_points);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i     = hypre_CSRMatrixI(A_diag);

   HYPRE_BigInt     global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int        n       = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     first_index = hypre_ParVectorFirstIndex(u);

   hypre_Vector   *u_local = hypre_ParVectorLocalVector(u);
   HYPRE_Real     *u_data  = hypre_VectorData(u_local);

   hypre_Vector   *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   HYPRE_Real     *Vtemp_data = hypre_VectorData(Vtemp_local);

   hypre_CSRMatrix *A_CSR;
   HYPRE_Int      *A_CSR_i;
   HYPRE_Int      *A_CSR_j;
   HYPRE_Real     *A_CSR_data;

   hypre_Vector    *f_vector;
   HYPRE_Real     *f_vector_data;

   HYPRE_Int        i;
   HYPRE_Int        jj;
   HYPRE_Int        column;
   HYPRE_Int        relax_error = 0;

   HYPRE_Real      *A_mat;
   HYPRE_Real      *b_vec;

   HYPRE_Real       zero = 0.0;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*-----------------------------------------------------------------------
    * Switch statement to direct control based on relax_type:
    *     relax_type = 7 -> Jacobi (uses ParMatvec)
    *     relax_type = 9 -> Direct Solve
    *-----------------------------------------------------------------------*/

   switch (relax_type)
   {

      case 7: /* Jacobi (uses ParMatvec) */
      {

         /*-----------------------------------------------------------------
          * Copy f into temporary vector.
          *-----------------------------------------------------------------*/

         hypre_ParVectorCopy(f, Vtemp);

         /*-----------------------------------------------------------------
          * Perform MatvecT Vtemp=f-A^Tu
          *-----------------------------------------------------------------*/

         hypre_ParCSRMatrixMatvecT(-1.0, A, u, 1.0, Vtemp);
         for (i = 0; i < n; i++)
         {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero)
            {
               u_data[i] += relax_weight * Vtemp_data[i]
                            / A_diag_data[A_diag_i[i]];
            }
         }
      }
      break;


      case 9: /* Direct solve: use gaussian elimination */
      {

         HYPRE_Int n_global = (HYPRE_Int) global_num_rows;
         /*-----------------------------------------------------------------
          *  Generate CSR matrix from ParCSRMatrix A
          *-----------------------------------------------------------------*/

         A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
         f_vector = hypre_ParVectorToVectorAll(f);
         if (n)
         {
            A_CSR_i = hypre_CSRMatrixI(A_CSR);
            A_CSR_j = hypre_CSRMatrixJ(A_CSR);
            A_CSR_data = hypre_CSRMatrixData(A_CSR);
            f_vector_data = hypre_VectorData(f_vector);

            A_mat = hypre_CTAlloc(HYPRE_Real,  n_global * n_global, HYPRE_MEMORY_HOST);
            b_vec = hypre_CTAlloc(HYPRE_Real,  n_global, HYPRE_MEMORY_HOST);

            /*---------------------------------------------------------------
             *  Load transpose of CSR matrix into A_mat.
             *---------------------------------------------------------------*/

            for (i = 0; i < n_global; i++)
            {
               for (jj = A_CSR_i[i]; jj < A_CSR_i[i + 1]; jj++)
               {
                  column = A_CSR_j[jj];
                  A_mat[column * n_global + i] = A_CSR_data[jj];
               }
               b_vec[i] = f_vector_data[i];
            }

            hypre_gselim(A_mat, b_vec, n_global, relax_error);

            for (i = 0; i < n; i++)
            {
               u_data[i] = b_vec[first_index + i];
            }

            hypre_TFree(A_mat, HYPRE_MEMORY_HOST);
            hypre_TFree(b_vec, HYPRE_MEMORY_HOST);
            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;

         }
      }
      break;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return (relax_error);
}
