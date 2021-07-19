/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

   hypre_ParFSAIData    *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   hypre_ParCSRMatrix  *G                    = hypre_ParFSAIDataGmat(fsai_data);
   hypre_ParCSRMatrix  *GT                   = hypre_ParFSAIDataGTmat(fsai_data);
   HYPRE_Int            tol                  = hypre_ParFSAIDataTolerance(fsai_data);
   HYPRE_Int            max_iter             = hypre_ParFSAIDataMaxIterations(fsai_data);
   HYPRE_Int            print_level          = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int            logging              = hypre_ParFSAIDataLogging(fsai_data);
   HYPRE_Real           omega                = hypre_ParFSAIDataOmega(fsai_data);

   /* Initilaize residual */
   hypre_ParVector      *r;
   hypre_ParVectorCreate(comm, hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(r);
   hypre_ParFSAIDataResidual(fsai_data)      = r;

   /* Local variables */

   HYPRE_Int            iter, num_procs, my_id;
   HYPRE_Real           old_rn, new_rn, rel_resnorm;

   //HYPRE_Int            n = hypre_CSRMatrixNumRows(A_diag);

   hypre_ParVector         *x_work            = hypre_ParVectorCreate(comm, hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVector         *r_work            = hypre_ParVectorCreate(comm, hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVector         *r_old             = hypre_ParVectorCreate(comm, hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixRowStarts(A));

   hypre_ParVectorInitialize(x_work);
   hypre_ParVectorInitialize(r_work);
   hypre_ParVectorInitialize(r_old);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_ParFSAIDataNumIterations(fsai_data) = 0;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);


   /*-----------------------------------------------------------------
    * Preconditioned Richardson - Main solver loop
    * x(k+1) = x(k) + omega * (G^T*G) * (b - A*x(k))
    * ----------------------------------------------------------------*/

   if(my_id == 0 && print_level > 1)
      hypre_printf("\n\n FSAI SOLVER SOLUTION INFO:\n");

   iter               = 0;
   rel_resnorm        = 1.0;
   hypre_ParVectorSetConstantValues(x, 0.0);        /* Set initial guess x_0 */
   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, x, 1.0, b, r); /* r_0 = b - Ax_0 */
   if(my_id == 0 && print_level > 1)
   {
      hypre_printf("                old         new         relative\n");
      hypre_printf("    iter #      res norm    res norm    res norm\n");
      hypre_printf("    --------    --------    --------    --------\n");
   }
   while(rel_resnorm >= tol && iter < max_iter)
   {

      /* Update solution vector */
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, x, 1.0, b, x_work); /* x_work = b - A*x(k) */

      /* VPM: In y = A*x + b, y cannot be the same vector as x */
      //hypre_ParCSRMatrixMatvec(1.0, G, x_work, 0.0, NULL, x_work);    /* x_work = G*x_work */
      //hypre_ParCSRMatrixMatvec(1.0, GT, x_work, 0.0, NULL, x_work);   /* x_work = G^T*x_work */

      hypre_ParVectorAxpy(omega, x_work, x);                          /* x(k+1) = x(k) = omega*x_work */

      /* Compute residual */
      old_rn             = hypre_ParVectorInnerProd(r, r);
      hypre_ParCSRMatrixMatvecOutOfPlace(1.0, G, r, 0.0, NULL, r_work);

      /* VPM: In y = A*x + b, y cannot be the same vector as x */
      //hypre_ParCSRMatrixMatvec(1.0, GT, r_work, 0.0, NULL, r_work);
      //hypre_ParCSRMatrixMatvec(1.0, A, r_work, 0.0, NULL, r_work);

      hypre_ParVectorAxpy(-1.0, r_work, r);
      new_rn             = hypre_ParVectorInnerProd(r, r);

      /* Compute rel_resnorm */
      rel_resnorm = new_rn/old_rn;

      if(my_id == 0 && print_level > 1)
         hypre_printf("    %e          %e          %e          %e\n", iter, old_rn, new_rn, rel_resnorm);

      iter++;

   }

   if(logging > 1)
   {
      hypre_ParFSAIDataNumIterations(fsai_data) = iter;
      hypre_ParFSAIDataRelResNorm(fsai_data)    = rel_resnorm;
   }

   HYPRE_ANNOTATE_FUNC_END;

   hypre_ParVectorDestroy(x_work);
   hypre_ParVectorDestroy(r_work);
   hypre_ParVectorDestroy(r_old);

   return hypre_error_flag;
}
