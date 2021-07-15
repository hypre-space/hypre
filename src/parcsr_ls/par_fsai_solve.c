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
#include "par_fsai.h"

/*--------------------------------------------------------------------
 * hypre_FSAISolve
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISolve( void   *fsai_vdata,
                        hypre_ParCSRMatrix *A,
                        hypre_ParVector    *b,
                        hypre_ParVector    *x )
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);

   hypre_ParFSAIData    *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   hypre_ParCSRMatrix   *G                   = hypre_ParFSAIDataGmat(fsai_data);
   /* XXX: Only want the values of r for this process - how? */
   hypre_ParVector      *r                   = hypre_ParFSAIResidual(fsai_data);
   HYPRE_Int            tol                  = hypre_ParFSAITolerance(fsai_data);
   HYPRE_Int            max_iter             = hypre_ParFSAIMaxIterations(fsai_data);
   HYPRE_Int            print_level          = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int            logging              = hypre_ParFSAIDataLogging(fsai_data);

   /* Local variables */

   HYPRE_Int            iter, num_procs, my_id;
   HYPRE_Real           old_rn, new_rn, rel_resnorm;
   HYPRE_CSRMatrix      *A_diag     = hypre_ParCSRMatrixDiag(A);
   HYPRE_CSRMatrix      *G_diag     = hypre_ParCSRMatrixDiag(G);
   HYPRE_CSRMatrix      *G_diag_T;

   HYPRE_Int            n = hypre_CSRMatrixNumRows(A_diag);

   hypre_CSRMatrixTranspose(G_diag, &G_diag_T, 1);
   HYPRE_Vector         *x_work            = hypre_SeqVectorCreate(n);
   HYPRE_Vector         *r_work            = hypre_SeqVectorCreate(n);
   HYPRE_Vector         *r_old             = hypre_SeqVectorCreate(n);

   hypre_SeqVectorInitialize(x_work);
   hypre_SeqVectorInitialize(r_work);
   hypre_SeqVectorInitialize(r_old);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_ParFSAINumIterations(fsai_data) = 0;

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
   hypre_SeqVectorSetConstantValues(x, 0.0);             /* Set initial guess x_0 */
   hypre_ParCSRMatvec(-1.0, A_diag, x, 1.0, b, r);       /* r_0 = b - Ax_0 */
   if(my_id == 0 && print_level > 1)
   {
      hypre_printf("                old         new         relative\n");
      hypre_printf("    iter #      res norm    res norm    res norm\n");
      hypre_printf("    --------    --------    --------    --------\n");
   }
   while(rel_resnorm >= tol && iter < max_iter)
   {
    
      /* Update solution vector */  
      hypre_ParCSRMatvec(-1.0, A_diag, x, 1.0, b, x_work);           /* x_work = b - A*x(k) */     
      hypre_ParCSRMatvec(1.0, G_diag, x_work, 0.0, NULL, x_work);    /* x_work = G*x_work */
      hypre_ParCSRMatvec(1.0, G_diag_T, x_work, 0.0, NULL, x_work);  /* x_work = G^T*x_work */
      hypre_SeqVectorAxpy(omega, x_work, x);                         /* x(k+1) = x(k) = omega*x_work */

      /* Compute residual */
      old_rn             = hypre_SeqVectorInnerProd(r, r);
      hypre_ParCSRMatVec(1.0, G_diag, r, 0.0, NULL, r_work);
      hypre_ParCSRMatVec(1.0, G_diag_T, r_work, 0.0, NULL, r_work);
      hypre_ParCSRMatVec(1.0, A_diag, r_work, 0.0, NULL, r_work);
      hypre_SeqVectorAxpy(-1.0, r_work, r);
      new_rn             = hypre_SeqVectorInnerProd(r, r);

      /* Compute rel_resnorm */
      rel_resnorm = new_rn/old_rn;

      if(my_id == 0 && print_level > 1)
         hypre_printf("    %e          %e          %e          %e\n", iter, old_rn, new_rn, rel_resnorm);

      iter++;

   }

   hypre_ParFSAIDataNumIterations(fsai_data) = iter;
   hypre_ParFSAIDataRelResNorm(fsai_data)    = rel_resnorm;
 
   HYPRE_ANNOTATE_FUNC_END;

   hypre_SeqVectorDestroy(x_work);
   hypre_SeqVectorDestroy(r_work);
   hypre_SeqVectorDestroy(r_old);

}
