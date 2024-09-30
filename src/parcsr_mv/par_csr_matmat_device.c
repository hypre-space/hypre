/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypreGPUKernel_ParCSRMatMatDiag
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_ParCSRMatMatDiag(hypre_DeviceItem &item,
                                HYPRE_Int         num_rows,
                                HYPRE_Int        *A_diag_i,
                                HYPRE_Int        *A_diag_j,
                                HYPRE_Complex    *A_diag_data,
                                HYPRE_Int        *A_offd_i,
                                HYPRE_Int        *A_offd_j,
                                HYPRE_Complex    *A_offd_data,
                                HYPRE_Int        *BT_diag_i,
                                HYPRE_Int        *BT_diag_j,
                                HYPRE_Complex    *BT_diag_data,
                                HYPRE_Int        *BT_offd_i,
                                HYPRE_Int        *BT_offd_j,
                                HYPRE_Complex    *BT_offd_data,
                                HYPRE_BigInt     *A_col_map_offd,
                                HYPRE_BigInt     *BT_col_map_offd,
                                HYPRE_Complex    *C_diag_data)
{
   const HYPRE_Int row = hypre_gpu_get_thread_id<1>(item);

   if (row < num_rows)
   {
      HYPRE_Complex sum = 0.0;

      /* Process diagonal part of A */
      HYPRE_Int kA = A_diag_i[row];
      HYPRE_Int kB = BT_diag_i[row];
      while (kA < A_diag_i[row + 1] && kB < BT_diag_i[row + 1])
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
            sum += A_diag_data[kA] * BT_diag_data[kB];
            kA++;
            kB++;
         }
      }

      /* Process off-diagonal part of A */
      kA = A_offd_i[row];
      kB = BT_offd_i[row];
      while (kA < A_offd_i[row + 1] && kB < BT_offd_i[row + 1])
      {
         HYPRE_BigInt col_A = A_col_map_offd[A_offd_j[kA]];
         HYPRE_BigInt col_B = BT_col_map_offd[BT_offd_j[kB]];
         if (col_A < col_B)
         {
            kA++;
         }
         else if (col_A > col_B)
         {
            kB++;
         }
         else
         {
            sum += A_offd_data[kA] * BT_offd_data[kB];
            kA++;
            kB++;
         }
      }

      C_diag_data[row] = sum;
   }
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatMatDiagDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatMatDiagDevice(hypre_ParCSRMatrix  *A,
                             hypre_ParCSRMatrix  *BT,
                             hypre_ParCSRMatrix  *C)
{
   HYPRE_Int             num_rows        = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix      *A_diag          = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *A_offd          = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix      *BT_diag         = hypre_ParCSRMatrixDiag(BT);
   hypre_CSRMatrix      *BT_offd         = hypre_ParCSRMatrixOffd(BT);
   hypre_CSRMatrix      *C_diag          = hypre_ParCSRMatrixDiag(C);

   HYPRE_Int            *C_diag_i        = hypre_CSRMatrixI(C_diag);
   HYPRE_Int            *C_diag_j        = hypre_CSRMatrixJ(C_diag);

   hypre_GpuProfilingPushRange("ParCSRMatMatDiag");

   /* Set up C_diag_i and C_diag_j */
   HYPRE_THRUST_CALL(sequence, C_diag_i, C_diag_i + num_rows + 1, 0);
   HYPRE_THRUST_CALL(sequence, C_diag_j, C_diag_j + num_rows, 0);

   /* Update device column maps if needed */
   hypre_ParCSRMatrixCopyColMapOffdToDevice(A);
   hypre_ParCSRMatrixCopyColMapOffdToDevice(BT);

   /* Launch GPU kernel */
   const dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   const dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "threads", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_ParCSRMatMatDiag, gDim, bDim,
                     num_rows,
                     hypre_CSRMatrixI(A_diag),
                     hypre_CSRMatrixJ(A_diag),
                     hypre_CSRMatrixData(A_diag),
                     hypre_CSRMatrixI(A_offd),
                     hypre_CSRMatrixJ(A_offd),
                     hypre_CSRMatrixData(A_offd),
                     hypre_CSRMatrixI(BT_diag),
                     hypre_CSRMatrixJ(BT_diag),
                     hypre_CSRMatrixData(BT_diag),
                     hypre_CSRMatrixI(BT_offd),
                     hypre_CSRMatrixJ(BT_offd),
                     hypre_CSRMatrixData(BT_offd),
                     hypre_ParCSRMatrixDeviceColMapOffd(A),
                     hypre_ParCSRMatrixDeviceColMapOffd(BT),
                     hypre_CSRMatrixData(C_diag) );

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

#endif /* if defined(HYPRE_USING_GPU) */
