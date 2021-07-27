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
   hypre_ParVector     *x_work               = hypre_ParFSAIDataXWork(fsai_data);
   hypre_ParVector     *r_work               = hypre_ParFSAIDataRWork(fsai_data);
   hypre_ParVector     *z_work               = hypre_ParFSAIDataZWork(fsai_data);
   hypre_ParVector     *r                    = hypre_ParFSAIDataResidual(fsai_data);
   HYPRE_Int            tol                  = hypre_ParFSAIDataTolerance(fsai_data);
   HYPRE_Int            max_iter             = hypre_ParFSAIDataMaxIterations(fsai_data);
   HYPRE_Int            print_level          = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int            logging              = hypre_ParFSAIDataLogging(fsai_data);
   HYPRE_Real           omega                = hypre_ParFSAIDataOmega(fsai_data);

   /* Local variables */

   HYPRE_Int            iter, num_procs, my_id;
   HYPRE_Real           old_rn, new_rn, rel_resnorm;

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

   if(my_id == 0 && print_level > 1)
   {
      hypre_printf("                old         new         relative\n");
      hypre_printf("    iter #      res norm    res norm    res norm\n");
      hypre_printf("    --------    --------    --------    --------\n");
   }

   /* Compute initial reisdual. r(0) = b - Ax */
   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, x, 1.0, b, r); /* residual */

   while(rel_resnorm >= tol && iter < max_iter)
   {

      /* Compute Preconditoned Residual. z_temp = G^T*G*r(k) */      
      hypre_ParCSRMatrixMatvecOutOfPlace(1.0, G, r, 0.0, x_work, r_work);         /* r_work = G*r */
      hypre_ParCSRMatrixMatvecOutOfPlace(1.0, GT, r_work, 0.0, x_work, z_work);   /* z_work = G^T*r_work */

      /* Compute updated solution vector. x(k+1) = x(k) + omega*z(k) */
      hypre_ParVectorAxpy(omega, z_work, x); 

      /* Compute residual norm */
      old_rn             = hypre_ParVectorInnerProd(r, r);
      
      /* Update residual: r(k+1) = r(k) - Az_work(k) */
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, z_work, 1.0, r, r_work);

      new_rn             = hypre_ParVectorInnerProd(r_work, r_work);

      /* Compute rel_resnorm */
      rel_resnorm = new_rn/old_rn;

      if(my_id == 0 && print_level > 1)
         hypre_printf("    %e          %e          %e          %e\n", iter, old_rn, new_rn, rel_resnorm);

      iter++;

   }

   if(logging > 1)
   {
      hypre_ParVectorCopy(r_work, r);
      hypre_ParFSAIDataNumIterations(fsai_data) = iter;
      hypre_ParFSAIDataRelResNorm(fsai_data)    = rel_resnorm;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
