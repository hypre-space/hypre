/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * AMG solve routine
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"

/*--------------------------------------------------------------------
 * hypre_BoomerAMGSolve
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSolve( void               *amg_vdata,
                      hypre_ParCSRMatrix *A,
                      hypre_ParVector    *f,
                      hypre_ParVector    *u         )
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   hypre_ParAMGData    *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */
   HYPRE_Int            amg_print_level;
   HYPRE_Int            amg_logging;
   HYPRE_Int            cycle_count;
   HYPRE_Int            num_levels;
   HYPRE_Int            converge_type;
   HYPRE_Int            block_mode;
   HYPRE_Int            additive;
   HYPRE_Int            mult_additive;
   HYPRE_Int            simple;
   HYPRE_Int            min_iter;
   HYPRE_Int            max_iter;
   HYPRE_Real           tol;

   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   hypre_ParCSRBlockMatrix **A_block_array;

   /*  Local variables  */
   HYPRE_Int           j;
   HYPRE_Int           Solve_err_flag;
   HYPRE_Int           num_procs, my_id;
   HYPRE_Int           num_vectors;
   HYPRE_Real          alpha = 1.0;
   HYPRE_Real          beta = -1.0;
   HYPRE_Real          cycle_op_count;
   HYPRE_Real          total_coeffs;
   HYPRE_Real          total_variables;
   HYPRE_Real         *num_coeffs;
   HYPRE_Real         *num_variables;
   HYPRE_Real          cycle_cmplxty = 0.0;
   HYPRE_Real          operat_cmplxty;
   HYPRE_Real          grid_cmplxty;
   HYPRE_Real          conv_factor = 0.0;
   HYPRE_Real          resid_nrm = 1.0;
   HYPRE_Real          resid_nrm_init = 0.0;
   HYPRE_Real          relative_resid;
   HYPRE_Real          rhs_norm = 0.0;
   HYPRE_Real          old_resid;
   HYPRE_Real          ieee_check = 0.;

   hypre_ParVector    *Vtemp;
   hypre_ParVector    *Rtemp;
   hypre_ParVector    *Ptemp;
   hypre_ParVector    *Ztemp;
   hypre_ParVector    *Residual = NULL;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   amg_print_level  = hypre_ParAMGDataPrintLevel(amg_data);
   amg_logging      = hypre_ParAMGDataLogging(amg_data);
   if (amg_logging > 1)
   {
      Residual = hypre_ParAMGDataResidual(amg_data);
   }
   num_levels       = hypre_ParAMGDataNumLevels(amg_data);
   A_array          = hypre_ParAMGDataAArray(amg_data);
   F_array          = hypre_ParAMGDataFArray(amg_data);
   U_array          = hypre_ParAMGDataUArray(amg_data);

   converge_type    = hypre_ParAMGDataConvergeType(amg_data);
   tol              = hypre_ParAMGDataTol(amg_data);
   min_iter         = hypre_ParAMGDataMinIter(amg_data);
   max_iter         = hypre_ParAMGDataMaxIter(amg_data);
   additive         = hypre_ParAMGDataAdditive(amg_data);
   simple           = hypre_ParAMGDataSimple(amg_data);
   mult_additive    = hypre_ParAMGDataMultAdditive(amg_data);
   block_mode       = hypre_ParAMGDataBlockMode(amg_data);
   A_block_array    = hypre_ParAMGDataABlockArray(amg_data);
   Vtemp            = hypre_ParAMGDataVtemp(amg_data);
   Rtemp            = hypre_ParAMGDataRtemp(amg_data);
   Ptemp            = hypre_ParAMGDataPtemp(amg_data);
   Ztemp            = hypre_ParAMGDataZtemp(amg_data);
   num_vectors      = hypre_ParVectorNumVectors(f);

   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;

   /* Verify that the number of vectors held by f and u match */
   if (hypre_ParVectorNumVectors(f) !=
       hypre_ParVectorNumVectors(u))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: num_vectors for RHS and LHS do not match!\n");
      return hypre_error_flag;
   }

   /* Update work vectors */
   hypre_ParVectorResize(Vtemp, num_vectors);
   hypre_ParVectorResize(Rtemp, num_vectors);
   hypre_ParVectorResize(Ptemp, num_vectors);
   hypre_ParVectorResize(Ztemp, num_vectors);
   if (amg_logging > 1)
   {
      hypre_ParVectorResize(Residual, num_vectors);
   }
   for (j = 1; j < num_levels; j++)
   {
      hypre_ParVectorResize(F_array[j], num_vectors);
      hypre_ParVectorResize(U_array[j], num_vectors);
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
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && amg_print_level > 1 && tol > 0.)
   {
      hypre_printf("\n\nAMG SOLUTION INFO:\n");
   }

   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print
    *-----------------------------------------------------------------------*/

   if (amg_print_level > 1 || amg_logging > 1 || tol > 0.)
   {
      if ( amg_logging > 1 )
      {
         hypre_ParVectorCopy(F_array[0], Residual);
         if (tol > 0)
         {
            hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Residual);
         }
         resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd( Residual, Residual ));
      }
      else
      {
         hypre_ParVectorCopy(F_array[0], Vtemp);
         if (tol > 0)
         {
            hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
         }
         resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
      }

      /* Since it does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (resid_nrm != 0.)
      {
         ieee_check = resid_nrm / resid_nrm; /* INF -> NaN conversion */
      }

      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (amg_print_level > 0)
         {
            hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            hypre_printf("ERROR -- hypre_BoomerAMGSolve: INFs and/or NaNs detected in input.\n");
            hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         hypre_error(HYPRE_ERROR_GENERIC);
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }

      /* r0 */
      resid_nrm_init = resid_nrm;

      if (0 == converge_type)
      {
         rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(f, f));
         if (rhs_norm)
         {
            relative_resid = resid_nrm_init / rhs_norm;
         }
         else
         {
            relative_resid = resid_nrm_init;
         }
      }
      else
      {
         /* converge_type != 0, test convergence with ||r|| / ||r0|| */
         relative_resid = 1.0;
      }
   }
   else
   {
      relative_resid = 1.;
   }

   if (my_id == 0 && amg_print_level > 1)
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

   while ( (relative_resid >= tol || cycle_count < min_iter) && cycle_count < max_iter )
   {
      hypre_ParAMGDataCycleOpCount(amg_data) = 0;
      /* Op count only needed for one cycle */
      if ( (additive      < 0 || additive      >= num_levels) &&
           (mult_additive < 0 || mult_additive >= num_levels) &&
           (simple        < 0 || simple        >= num_levels) )
      {
         hypre_BoomerAMGCycle(amg_data, F_array, U_array);
      }
      else
      {
         /* RL TODO: for now, force u's all-zero flag to be FALSE */
         hypre_ParVectorAllZeros(u) = 0;

         hypre_BoomerAMGAdditiveCycle(amg_data);
      }

      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      if (amg_print_level > 1 || amg_logging > 1 || tol > 0.)
      {
         old_resid = resid_nrm;

         if (amg_logging > 1)
         {
            hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[0], U_array[0], beta, F_array[0],
                                               Residual);
            resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Residual, Residual));
         }
         else
         {
            hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[0], U_array[0], beta, F_array[0],
                                               Vtemp);
            resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
         }

         if (old_resid)
         {
            conv_factor = resid_nrm / old_resid;
         }
         else
         {
            conv_factor = resid_nrm;
         }

         if (0 == converge_type)
         {
            if (rhs_norm)
            {
               relative_resid = resid_nrm / rhs_norm;
            }
            else
            {
               relative_resid = resid_nrm;
            }
         }
         else
         {
            relative_resid = resid_nrm / resid_nrm_init;
         }

         hypre_ParAMGDataRelativeResidualNorm(amg_data) = relative_resid;
      }

      ++cycle_count;

      hypre_ParAMGDataNumIterations(amg_data) = cycle_count;
#ifdef CUMNUMIT
      ++hypre_ParAMGDataCumNumIterations(amg_data);
#endif

      if (my_id == 0 && amg_print_level > 1)
      {
         hypre_printf("    Cycle %2d   %e    %f     %e \n", cycle_count,
                      resid_nrm, conv_factor, relative_resid);
      }
   }

   if (cycle_count == max_iter && tol > 0.)
   {
      Solve_err_flag = 1;
      hypre_error(HYPRE_ERROR_CONV);
   }

   /*-----------------------------------------------------------------------
    *    Compute closing statistics
    *-----------------------------------------------------------------------*/

   if (cycle_count > 0 && resid_nrm_init)
   {
      conv_factor = hypre_pow((resid_nrm / resid_nrm_init), (1.0 / (HYPRE_Real) cycle_count));
   }
   else
   {
      conv_factor = 1.;
   }

   if (amg_print_level > 1)
   {
      num_coeffs       = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
      num_variables    = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
      num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A);
      num_variables[0] = hypre_ParCSRMatrixGlobalNumRows(A);

      if (block_mode)
      {
         for (j = 1; j < num_levels; j++)
         {
            num_coeffs[j]    = (HYPRE_Real) hypre_ParCSRBlockMatrixNumNonzeros(A_block_array[j]);
            num_variables[j] = (HYPRE_Real) hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[j]);
         }
         num_coeffs[0]    = hypre_ParCSRBlockMatrixDNumNonzeros(A_block_array[0]);
         num_variables[0] = hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[0]);

      }
      else
      {
         for (j = 1; j < num_levels; j++)
         {
            num_coeffs[j]    = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A_array[j]);
            num_variables[j] = (HYPRE_Real) hypre_ParCSRMatrixGlobalNumRows(A_array[j]);
         }
      }


      for (j = 0; j < hypre_ParAMGDataNumLevels(amg_data); j++)
      {
         total_coeffs += num_coeffs[j];
         total_variables += num_variables[j];
      }

      cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

      if (num_variables[0])
      {
         grid_cmplxty = total_variables / num_variables[0];
      }
      if (num_coeffs[0])
      {
         operat_cmplxty = total_coeffs / num_coeffs[0];
         cycle_cmplxty = cycle_op_count / num_coeffs[0];
      }

      if (my_id == 0)
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
         hypre_printf("                   cycle = %f\n\n\n\n", cycle_cmplxty);
      }

      hypre_TFree(num_coeffs, HYPRE_MEMORY_HOST);
      hypre_TFree(num_variables, HYPRE_MEMORY_HOST);
   }
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
