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
   HYPRE_Real           rel_resnorm          = hypre_ParFSAIDataRelResNorm(fsai_data);

   /* Local variables */

   HYPRE_Int            iter, num_procs, my_id;
   HYPRE_CSRMatrix      *A_diag     = hypre_ParCSRMatrixDiag(A);
   HYPRE_CSRMatrix      *G_diag     = hypre_ParCSRMatrixDiag(G);
   HYPRE_CSRMatrix      *G_diag_T;

   HYPRE_Int            n = hypre_CSRMatrixNumRows(A_diag);

   hypre_CSRMatrixTranspose(G_diag, &G_diag_T, 1);
   HYPRE_Vector         *z                 = hypre_SeqVectorCreate(n);
   HYPRE_Vector         *p                 = hypre_SeqVectorCreate(n);
   HYPRE_Vector         *Ap                = hypre_SeqVectorCreate(n);
   HYPRE_Vector         *LinvAp            = hypre_SeqVectorCreate(n);
   HYPRE_Real           alpha, beta, IP_z_old, IP_z_new;

   hypre_SeqVectorInitialize(z);
   hypre_SeqVectorInitialize(p);
   hypre_SeqVectorInitialize(Ap);
   hypre_SeqVectorInitialize(LinvAp);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_ParFSAINumIterations(fsai_data) = 0;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);


   /*----------------------------------------------------------------- 
    * Split PCG Main solver loop - Do 1 iteration at least 
    * Reference: "Interative Methods for Sparse Linear Systems"
    * By Yousef Saad. Page 298, Algorithm 9.2 
    * ----------------------------------------------------------------*/
   iter = 0;
   rel_resnorm        = 1.0;
   hypre_SeqVectorSetConstantValues(x, 0.0);             /* Set initial guess x_0 */
   hypre_ParCSRMatvec(-1.0, A_diag, x, 1.0, b, r);       /* r_0 = b - Ax_0 */
   hypre_ParCSRMatvec(1.0, G_diag, r, 0.0, NULL, z);     /* z_0 = Gr_0 */
   hypre_ParCSRMatvec(1.0, G_diag_T, z, 0.0, NULL, p);   /* p_0 = G^T*z_0 */
   IP_z_old = hypre_SeqVectorInnerProd(z, z);

   while(rel_resnorm >= tol && iter < max_iter)
   {
   
      /* alpha(j) = (z, z)/(Ap, p) */
      hypre_ParCSRMatvec(1.0, A_diag, p, 0.0, NULL, Ap);
      alpha = IP_z_old/hypre_SeqVectorInnerProd(Ap, p);

      /* x(j+1) = x(j) + alpha(j)*p(j) */
      hypre_SeqVectorAxpy(alpha, x, p);

      /* z(j+1) = z(j) - alpha(j)*G*A*p(j) */
      hypre_ParCSRMatvec(alpha, G_diag, Ap, 0.0, NULL, LinvAp);
      hypre_SeqVectorAxpy(-1.0, z, LinvAp);

      /* rel_resnorm(j) = (z(j+1), z(j+1))/(z(j), z(j)) */
      IP_z_new = hypre_SeqVectorInnerProd(z, z);
      rel_resnorm = IP_z_new/IP_z_old;
      IP_z_old = IP_z_new; 

      /* p(j+1) = G^T*z(j+1) + rel_resnorm(j)*p(j) */
      hypre_ParCSRMatvec(1.0, G_diag_T, z, rel_resnorm, p, p);

      iter++;

   }

   hypre_ParFSAIDataNumIterations(fsai_data) = iter;
   hypre_SeqVectorCopy(z, r);   
 
   HYPRE_ANNOTATE_FUNC_END;

   hypre_SeqVectorDestroy(z);
   hypre_SeqVectorDestroy(p);
   hypre_SeqVectorDestroy(Ap);
   hypre_SeqVectorDestroy(LinvAp);

}
