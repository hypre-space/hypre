/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * computes |D^-1/2 A D^-1/2 |_sup where D diagonal matrix
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixScaledNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixScaledNorm( hypre_ParCSRMatrix *A, HYPRE_Real *scnorm)
{
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   MPI_Comm     comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix      *diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int      *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Int      *diag_j = hypre_CSRMatrixJ(diag);
   HYPRE_Real     *diag_data = hypre_CSRMatrixData(diag);
   hypre_CSRMatrix      *offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int      *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Int      *offd_j = hypre_CSRMatrixJ(offd);
   HYPRE_Real     *offd_data = hypre_CSRMatrixData(offd);
   HYPRE_BigInt       global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt           *row_starts = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int       num_rows = hypre_CSRMatrixNumRows(diag);

   hypre_ParVector      *dinvsqrt;
   HYPRE_Real     *dis_data;
   hypre_Vector         *dis_ext;
   HYPRE_Real     *dis_ext_data;
   hypre_Vector         *sum;
   HYPRE_Real     *sum_data;

   HYPRE_Int         num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int         num_sends, i, j, index, start;

   HYPRE_Real *d_buf_data;
   HYPRE_Real  mat_norm, max_row_sum;

   dinvsqrt = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   hypre_ParVectorInitialize(dinvsqrt);
   dis_data = hypre_VectorData(hypre_ParVectorLocalVector(dinvsqrt));
   dis_ext = hypre_SeqVectorCreate(num_cols_offd);
   hypre_SeqVectorInitialize(dis_ext);
   dis_ext_data = hypre_VectorData(dis_ext);
   sum = hypre_SeqVectorCreate(num_rows);
   hypre_SeqVectorInitialize(sum);
   sum_data = hypre_VectorData(sum);

   /* generate dinvsqrt */
   for (i = 0; i < num_rows; i++)
   {
      dis_data[i] = 1.0 / hypre_sqrt(hypre_abs(diag_data[diag_i[i]]));
   }

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   d_buf_data = hypre_CTAlloc(HYPRE_Real,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                           num_sends), HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         d_buf_data[index++]
            = dis_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, d_buf_data,
                                               dis_ext_data);

   for (i = 0; i < num_rows; i++)
   {
      for (j = diag_i[i]; j < diag_i[i + 1]; j++)
      {
         sum_data[i] += hypre_abs(diag_data[j]) * dis_data[i] * dis_data[diag_j[j]];
      }
   }
   hypre_ParCSRCommHandleDestroy(comm_handle);

   for (i = 0; i < num_rows; i++)
   {
      for (j = offd_i[i]; j < offd_i[i + 1]; j++)
      {
         sum_data[i] += hypre_abs(offd_data[j]) * dis_data[i] * dis_ext_data[offd_j[j]];
      }
   }

   max_row_sum = 0;
   for (i = 0; i < num_rows; i++)
   {
      if (max_row_sum < sum_data[i])
      {
         max_row_sum = sum_data[i];
      }
   }

   hypre_MPI_Allreduce(&max_row_sum, &mat_norm, 1, HYPRE_MPI_REAL, hypre_MPI_MAX, comm);

   hypre_ParVectorDestroy(dinvsqrt);
   hypre_SeqVectorDestroy(sum);
   hypre_SeqVectorDestroy(dis_ext);
   hypre_TFree(d_buf_data, HYPRE_MEMORY_HOST);

   *scnorm = mat_norm;
   return 0;
}
