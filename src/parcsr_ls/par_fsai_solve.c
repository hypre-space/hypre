/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

/******************************************************************************
 *
 * FSAI solve routine
 *
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------
 * hypre_FSAISolve
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISolve( void               *fsai_vdata,
                 hypre_ParCSRMatrix *A,
                 hypre_ParVector    *b,
                 hypre_ParVector    *x )
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   hypre_ParFSAIData   *fsai_data   = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   hypre_ParVector     *r_work      = hypre_ParFSAIDataRWork(fsai_data);
   HYPRE_Real           tol         = hypre_ParFSAIDataTolerance(fsai_data);
   HYPRE_Int            zero_guess  = hypre_ParFSAIDataZeroGuess(fsai_data);
   HYPRE_Int            max_iter    = hypre_ParFSAIDataMaxIterations(fsai_data);
   HYPRE_Int            print_level = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int            logging     = hypre_ParFSAIDataLogging(fsai_data);

   /* Local variables */
   HYPRE_Int            iter, my_id;
   HYPRE_Real           old_resnorm, resnorm = 0.0, rel_resnorm;
   HYPRE_Complex        one = 1.0;
   HYPRE_Complex        neg_one = -1.0;
   HYPRE_Complex        zero = 0.0;

   /* Sanity check */
   if (hypre_ParVectorNumVectors(b) > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "FSAI doesn't support multicomponent vectors");
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("FSAISolve");

   hypre_MPI_Comm_rank(comm, &my_id);

   /*-----------------------------------------------------------------
    * Preconditioned Richardson - Main solver loop
    * x(k+1) = x(k) + omega * (G^T*G) * (b - A*x(k))
    * ----------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1)
   {
      hypre_printf("\n\n FSAI SOLVER SOLUTION INFO:\n");
   }

   iter        = 0;
   rel_resnorm = resnorm = 1.0;

   if (my_id == 0 && print_level > 1)
   {
      hypre_printf("                new         relative\n");
      hypre_printf("    iter #      res norm    res norm\n");
      hypre_printf("    --------    --------    --------\n");
   }

   if (max_iter > 0)
   {
      /* First iteration */
      if (zero_guess)
      {
         /* Compute: x(k+1) = omega*G^T*G*b */
         hypre_FSAIApply(fsai_vdata, zero, b, x);
      }
      else
      {
         /* Compute: x(k+1) = x(k) + omega*G^T*G*(b - A*x(k)) */
         hypre_ParCSRMatrixMatvecOutOfPlace(neg_one, A, x, one, b, r_work);
         hypre_FSAIApply(fsai_vdata, one, r_work, x);
      }

      /* Update iteration count */
      iter++;
   }
   else
   {
      hypre_ParVectorCopy(b, x);
   }

   /* Apply remaining iterations */
   for (; iter < max_iter; iter++)
   {
      /* Update residual */
      hypre_ParCSRMatrixMatvecOutOfPlace(neg_one, A, x, one, b, r_work);

      if (tol > 0.0)
      {
         old_resnorm = resnorm;
         resnorm = hypre_ParVectorInnerProd(r_work, r_work);

         /* Compute rel_resnorm */
         rel_resnorm = resnorm / old_resnorm;

         if (my_id == 0 && print_level > 1)
         {
            hypre_printf("    %e          %e          %e\n", iter, resnorm, rel_resnorm);
         }

         /* Exit if convergence tolerance has been achieved */
         if (rel_resnorm >= tol)
         {
            break;
         }
      }

      /* Compute: x(k+1) = x(k) + omega*inv(M)*r */
      hypre_FSAIApply(fsai_vdata, one, r_work, x);
   }

   if (logging > 1)
   {
      hypre_ParFSAIDataNumIterations(fsai_data) = iter;
      hypre_ParFSAIDataRelResNorm(fsai_data)    = rel_resnorm;
   }
   else
   {
      hypre_ParFSAIDataNumIterations(fsai_data) = 0;
      hypre_ParFSAIDataRelResNorm(fsai_data)    = 0.0;
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_FSAIApply
 *
 * Computes x(k+1) = alpha*x(k) + omega*G^T*G*b
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAIApply( void               *fsai_vdata,
                 HYPRE_Complex       alpha,
                 hypre_ParVector    *b,
                 hypre_ParVector    *x )
{
   hypre_ParFSAIData   *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   hypre_ParCSRMatrix  *G         = hypre_ParFSAIDataGmat(fsai_data);
   hypre_ParCSRMatrix  *GT        = hypre_ParFSAIDataGTmat(fsai_data);
   hypre_ParVector     *z_work    = hypre_ParFSAIDataZWork(fsai_data);
   HYPRE_Real           omega     = hypre_ParFSAIDataOmega(fsai_data);

   HYPRE_Complex        one  = 1.0;
   HYPRE_Complex        zero = 0.0;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("FSAIApply");

   hypre_ParCSRMatrixMatvec(one, G, b, zero, z_work);
   hypre_ParCSRMatrixMatvec(omega, GT, z_work, alpha, x);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
