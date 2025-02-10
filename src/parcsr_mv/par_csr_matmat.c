/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatMatDiagHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatMatDiagHost(hypre_ParCSRMatrix  *A,
                           hypre_ParCSRMatrix  *BT,
                           hypre_ParCSRMatrix  *C)
{
   HYPRE_Int             num_rows        = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix      *A_diag          = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *A_offd          = hypre_ParCSRMatrixOffd(A);

   hypre_CSRMatrix      *BT_diag, *BT_offd, *C_diag;
   HYPRE_Int            *A_diag_i, *BT_diag_i, *C_diag_i;
   HYPRE_Int            *A_offd_i, *BT_offd_i;
   HYPRE_Int            *A_diag_j, *BT_diag_j, *C_diag_j;
   HYPRE_Int            *A_offd_j, *BT_offd_j;
   HYPRE_Complex        *A_diag_a, *BT_diag_a, *C_diag_a;
   HYPRE_Complex        *A_offd_a, *BT_offd_a;
   HYPRE_BigInt         *A_col_map_offd;
   HYPRE_BigInt         *BT_col_map_offd;

   HYPRE_Int             i, kA, kB;
   HYPRE_Complex         diag;

   /* Load pointers */
   BT_diag   = hypre_ParCSRMatrixDiag(BT);
   BT_offd   = hypre_ParCSRMatrixOffd(BT);
   BT_diag_i = hypre_CSRMatrixI(BT_diag);
   BT_offd_i = hypre_CSRMatrixI(BT_offd);
   BT_diag_j = hypre_CSRMatrixJ(BT_diag);
   BT_offd_j = hypre_CSRMatrixJ(BT_offd);
   BT_diag_a = hypre_CSRMatrixData(BT_diag);
   BT_offd_a = hypre_CSRMatrixData(BT_offd);
   A_diag_i  = hypre_CSRMatrixI(A_diag);
   A_offd_i  = hypre_CSRMatrixI(A_offd);
   A_diag_j  = hypre_CSRMatrixJ(A_diag);
   A_offd_j  = hypre_CSRMatrixJ(A_offd);
   A_diag_a  = hypre_CSRMatrixData(A_diag);
   A_offd_a  = hypre_CSRMatrixData(A_offd);
   C_diag    = hypre_ParCSRMatrixDiag(C);
   C_diag_i  = hypre_CSRMatrixI(C_diag);
   C_diag_j  = hypre_CSRMatrixJ(C_diag);
   C_diag_a  = hypre_CSRMatrixData(C_diag);

   BT_col_map_offd = hypre_ParCSRMatrixColMapOffd(BT);
   A_col_map_offd  = hypre_ParCSRMatrixColMapOffd(A);

   /* Compute C = diag(A .* BT) */
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i, kA, kB, diag)
#endif
   for (i = 0; i < num_rows; i++)
   {
      /* Compute diagonal matrix contributions */
      diag = 0.0;
      for (kA = A_diag_i[i], kB = BT_diag_i[i];
           kA < A_diag_i[i + 1] && kB < BT_diag_i[i + 1];)
      {
         if (A_diag_j[kA] < BT_diag_j[kB])
         {
            kA++;
         }
         else if (A_diag_j[kA] > BT_diag_j[kB])
         {
            kB++;
         }
         else
         {
            diag += A_diag_a[kA] * BT_diag_a[kB];
            kA++; kB++;
         }
      }

      /* Compute off-diagonal matrix contributions */
      for (kA = A_offd_i[i], kB = BT_offd_i[i];
           kA < A_offd_i[i + 1] && kB < BT_offd_i[i + 1];)
      {
         if (A_col_map_offd[A_offd_j[kA]] < BT_col_map_offd[BT_offd_j[kB]])
         {
            kA++;
         }
         else if (A_col_map_offd[A_offd_j[kA]] > BT_col_map_offd[BT_offd_j[kB]])
         {
            kB++;
         }
         else
         {
            diag += A_offd_a[kA] * BT_offd_a[kB];
            kA++; kB++;
         }
      }

      C_diag_a[i] = diag;
      C_diag_j[i] = i;
      C_diag_i[i + 1] = i + 1;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatMatDiag
 *
 * Computes C = diag(A * B)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatMatDiag(hypre_ParCSRMatrix  *A,
                       hypre_ParCSRMatrix  *B,
                       hypre_ParCSRMatrix **C_ptr)
{
   MPI_Comm              comm            = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt          global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt          global_num_cols = hypre_ParCSRMatrixGlobalNumCols(B);
   HYPRE_Int             num_rows        = hypre_ParCSRMatrixNumRows(A);
   HYPRE_BigInt         *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_MemoryLocation  memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_ParCSRMatrix   *C, *BT;

   /* Create and initialize output matrix C */
   C = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, row_starts, 0, num_rows, 0);
   hypre_ParCSRMatrixInitialize_v2(C, memory_location);

   /* Transpose B for easier multiplication with A */
   hypre_ParCSRMatrixTranspose(B, &BT, 1);

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2(hypre_ParCSRMatrixMemoryLocation(A),
                                                     hypre_ParCSRMatrixMemoryLocation(BT));
   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatMatDiagDevice(A, BT, C);
   }
   else
#endif
   {
      hypre_ParCSRMatMatDiagHost(A, BT, C);
   }

   /* Output pointer */
   *C_ptr = C;

   /* Free memory */
   hypre_ParCSRMatrixDestroy(BT);

   return hypre_error_flag;
}
