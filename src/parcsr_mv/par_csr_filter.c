/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_ParCSRMatrix class.
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixBlkFilterHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixBlkFilterHost( hypre_ParCSRMatrix  *A,
                                 HYPRE_Int            block_size,
                                 hypre_ParCSRMatrix **B_ptr )
{
   MPI_Comm             comm              = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt         global_num_rows   = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt         global_num_cols   = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt        *row_starts        = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt        *col_starts        = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt        *col_map_offd_A    = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_MemoryLocation memory_location   = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_CSRMatrix     *A_diag            = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int            num_rows          = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int           *A_diag_i          = hypre_CSRMatrixI(A_diag);
   HYPRE_Int           *A_diag_j          = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex       *A_diag_a          = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix     *A_offd            = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int           *A_offd_i          = hypre_CSRMatrixI(A_offd);
   HYPRE_Int           *A_offd_j          = hypre_CSRMatrixJ(A_offd);
   HYPRE_Complex       *A_offd_a          = hypre_CSRMatrixData(A_offd);
   HYPRE_Int            num_cols_offd_A   = hypre_CSRMatrixNumCols(A_offd);

   /* Output matrix variables */
   hypre_ParCSRMatrix  *B;
   hypre_CSRMatrix     *B_diag, *B_offd;
   HYPRE_Int           *B_diag_i, *B_offd_i;
   HYPRE_Int           *B_diag_j, *B_offd_j;
   HYPRE_Complex       *B_diag_a, *B_offd_a;
   HYPRE_BigInt        *col_map_offd_B;
   HYPRE_Int            num_cols_offd_B;
   HYPRE_Int            B_diag_nnz, B_offd_nnz;

   /* Local variables */
   HYPRE_BigInt         big_block_size    = (HYPRE_BigInt) block_size;
   HYPRE_Int            i, j, c;
   HYPRE_Int           *marker;

   /*-----------------------------------------------------------------------
    *  Sanity checks
    *-----------------------------------------------------------------------*/

   if (block_size < 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "block size must be greater than one!\n");
      return hypre_error_flag;
   }

   if (global_num_rows % big_block_size)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "block size is not a divisor of the number of rows!\n");
      return hypre_error_flag;
   }

   if (row_starts[0] % big_block_size)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "block size is not a divisor of the first global row!\n");
      return hypre_error_flag;
   }

   if (global_num_rows != global_num_cols)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Function not implemented for rectangular matrices!\n");
      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------------
    *  First pass: compute nonzero counts of B
    *-----------------------------------------------------------------------*/

   B_diag_nnz = B_offd_nnz = 0;
   for (i = 0; i < num_rows; i++)
   {
      c = i % block_size;

      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         if (c == (A_diag_j[j] % block_size))
         {
            B_diag_nnz++;
         }
      }

      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         if (c == (HYPRE_Int) (col_map_offd_A[A_offd_j[j]] % big_block_size))
         {
            B_offd_nnz++;
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Create and initialize output matrix
    *-----------------------------------------------------------------------*/

   B = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts, num_cols_offd_A,
                                B_diag_nnz, B_offd_nnz);

   hypre_ParCSRMatrixInitialize_v2(B, memory_location);

   B_diag   = hypre_ParCSRMatrixDiag(B);
   B_diag_i = hypre_CSRMatrixI(B_diag);
   B_diag_j = hypre_CSRMatrixJ(B_diag);
   B_diag_a = hypre_CSRMatrixData(B_diag);

   B_offd   = hypre_ParCSRMatrixOffd(B);
   B_offd_i = hypre_CSRMatrixI(B_offd);
   B_offd_j = hypre_CSRMatrixJ(B_offd);
   B_offd_a = hypre_CSRMatrixData(B_offd);

   col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);

   /*-----------------------------------------------------------------------
    *  Second pass: Fill entries of B
    *-----------------------------------------------------------------------*/

   marker = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_rows; i++)
   {
      c = i % block_size;

      B_diag_i[i + 1] = B_diag_i[i];
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         if (c == (A_diag_j[j] % block_size))
         {
            B_diag_j[B_diag_i[i + 1]] = A_diag_j[j];
            B_diag_a[B_diag_i[i + 1]] = A_diag_a[j];
            B_diag_i[i + 1]++;
         }
      }

      B_offd_i[i + 1] = B_offd_i[i];
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         if (c == (HYPRE_Int) (col_map_offd_A[A_offd_j[j]] % big_block_size))
         {
            B_offd_j[B_offd_i[i + 1]] = A_offd_j[j];
            B_offd_a[B_offd_i[i + 1]] = A_offd_a[j];
            B_offd_i[i + 1]++;
            marker[A_offd_j[j]] = 1;
         }
      }
   }

   /* Update col_map array */
   num_cols_offd_B = 0;
   for (i = 0; i < num_cols_offd_A; i++)
   {
      if (marker[i])
      {
         col_map_offd_B[num_cols_offd_B++] = col_map_offd_A[i];
      }
   }
   hypre_CSRMatrixNumCols(B_offd) = num_cols_offd_B;
   hypre_TFree(marker, HYPRE_MEMORY_HOST);

   /* Update global nonzeros */
   hypre_ParCSRMatrixSetDNumNonzeros(B);
   hypre_ParCSRMatrixNumNonzeros(B) = (HYPRE_BigInt) hypre_ParCSRMatrixDNumNonzeros(B);
   hypre_MatvecCommPkgCreate(B);

   /* Set output pointer */
   *B_ptr = B;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixBlkFilter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixBlkFilter( hypre_ParCSRMatrix  *A,
                             HYPRE_Int            block_size,
                             hypre_ParCSRMatrix **B_ptr )
{
   HYPRE_ANNOTATE_FUNC_BEGIN;

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A)) == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixBlkFilterDevice(A, block_size, B_ptr);
   }
   else
#endif
   {
      hypre_ParCSRMatrixBlkFilterHost(A, block_size, B_ptr);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
