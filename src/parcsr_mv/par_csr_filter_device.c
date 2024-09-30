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
 * hypreGPUKernel_ParCSRMatrixBlkFilterCount
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_ParCSRMatrixBlkFilterCount(hypre_DeviceItem  &item,
                                          HYPRE_Int          num_rows,
                                          HYPRE_Int          block_size,
                                          HYPRE_Int         *A_diag_i,
                                          HYPRE_Int         *A_diag_j,
                                          HYPRE_Int         *A_offd_i,
                                          HYPRE_Int         *A_offd_j,
                                          HYPRE_BigInt      *A_col_map_offd,
                                          HYPRE_Int         *B_diag_i,
                                          HYPRE_Int         *B_offd_i)
{
   HYPRE_Int    row  = hypre_gpu_get_grid_warp_id<1, 1>(item);
   HYPRE_Int    lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_BigInt big_block_size = (HYPRE_BigInt) block_size;

   if (row < num_rows)
   {
      HYPRE_Int p = 0, q = 0, pA, qA;

      if (lane < 2)
      {
         p = read_only_load(A_diag_i + row + lane);
         q = read_only_load(A_offd_i + row + lane);
      }
      pA = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
      p  = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);
      qA = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, q, 1);
      q  = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, q, 0);

      HYPRE_Int diag_count = 0;
      HYPRE_Int offd_count = 0;

      for (HYPRE_Int j = p + lane;
           warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < pA);
           j += HYPRE_WARP_SIZE)
      {
         if (j < pA)
         {
            const HYPRE_Int col = read_only_load(A_diag_j + j);
            if ((col % block_size) == (row % block_size))
            {
               diag_count++;
            }
         }
      }

      for (HYPRE_Int j = q + lane;
           warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < qA);
           j += HYPRE_WARP_SIZE)
      {
         if (j < qA)
         {
            const HYPRE_Int col = read_only_load(A_offd_j + j);
            const HYPRE_BigInt global_col = read_only_load(A_col_map_offd + col);
            if ((HYPRE_Int) (global_col % big_block_size) == (row % block_size))
            {
               offd_count++;
            }
         }
      }

      diag_count = warp_reduce_sum(item, diag_count);
      offd_count = warp_reduce_sum(item, offd_count);

      if (lane == 0)
      {
         B_diag_i[row] = diag_count;
         B_offd_i[row] = offd_count;
      }
   }
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_ParCSRMatrixBlkFilterFill
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_ParCSRMatrixBlkFilterFill(hypre_DeviceItem &item,
                                         HYPRE_Int         num_rows,
                                         HYPRE_Int         block_size,
                                         HYPRE_Int         A_num_cols_offd,
                                         HYPRE_Int        *A_diag_i,
                                         HYPRE_Int        *A_diag_j,
                                         HYPRE_Complex    *A_diag_a,
                                         HYPRE_Int        *A_offd_i,
                                         HYPRE_Int        *A_offd_j,
                                         HYPRE_Complex    *A_offd_a,
                                         HYPRE_BigInt     *A_col_map_offd,
                                         HYPRE_Int        *B_diag_i,
                                         HYPRE_Int        *B_diag_j,
                                         HYPRE_Complex    *B_diag_a,
                                         HYPRE_Int        *B_offd_i,
                                         HYPRE_Int        *B_offd_j,
                                         HYPRE_Complex    *B_offd_a,
                                         HYPRE_Int        *col_map_marker)
{
   const HYPRE_Int  row = hypre_gpu_get_grid_warp_id<1, 1>(item);
   const HYPRE_Int  lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int        p = 0, q = 0, pA, qA;
   HYPRE_BigInt     big_block_size = (HYPRE_BigInt) block_size;

   if (row >= num_rows)
   {
      return;
   }

   if (lane < 2)
   {
      p = read_only_load(A_diag_i + row + lane);
      q = read_only_load(A_offd_i + row + lane);
   }
   pA = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p  = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);
   qA = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, q, 1);
   q  = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, q, 0);

   HYPRE_Int diag_offset = B_diag_i[row];
   for (HYPRE_Int j = p + lane;
        warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < pA);
        j += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int col     = (j < pA) ? read_only_load(A_diag_j + j) : 0;
      HYPRE_Int       write   = (j < pA && (col % block_size) == (row % block_size));
      hypre_mask      ballot  = hypre_ballot_sync(HYPRE_WARP_FULL_MASK, write);
      HYPRE_Int       laneoff = hypre_popc(ballot & ((hypre_mask_one << lane) - 1));

      if (write)
      {
         HYPRE_Int idx = diag_offset + laneoff;
         B_diag_j[idx] = col;
         B_diag_a[idx] = A_diag_a[j];
      }

      diag_offset += hypre_popc(ballot);
   }

   if (col_map_marker)
   {
      HYPRE_Int offd_offset = B_offd_i[row];
      for (HYPRE_Int j = q + lane;
           warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < qA);
           j += HYPRE_WARP_SIZE)
      {
         const HYPRE_Int    col        = (j < qA) ? read_only_load(A_offd_j + j) : 0;
         const HYPRE_BigInt global_col = (j < qA) ? read_only_load(A_col_map_offd + col) : 0;
         HYPRE_Int          write      = (j < qA) &&
                                         (HYPRE_Int) (global_col % big_block_size) == (row % block_size);
         hypre_mask         ballot     = hypre_ballot_sync(HYPRE_WARP_FULL_MASK, write);
         HYPRE_Int          laneoff    = hypre_popc(ballot & ((hypre_mask_one << lane) - 1));

         if (write)
         {
            HYPRE_Int idx = offd_offset + laneoff;
            B_offd_j[idx] = col;
            B_offd_a[idx] = A_offd_a[j];

#ifndef HYPRE_USING_SYCL
            if (col < A_num_cols_offd)
            {
               atomicOr(col_map_marker + col, 1);
            }
#endif
         }

         offd_offset += hypre_popc(ballot);
      }
   }
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixBlkFilterDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixBlkFilterDevice(hypre_ParCSRMatrix  *A,
                                  HYPRE_Int            block_size,
                                  hypre_ParCSRMatrix **B_ptr)
{
   MPI_Comm             comm            = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt         global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt         global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt        *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt        *col_starts      = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt        *A_col_map_offd  = hypre_ParCSRMatrixDeviceColMapOffd(A);
   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_CSRMatrix     *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int            num_rows        = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int           *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Int           *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex       *A_diag_a        = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix     *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int           *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Int           *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_Complex       *A_offd_a        = hypre_CSRMatrixData(A_offd);
   HYPRE_Int            num_cols_offd   = hypre_CSRMatrixNumCols(A_offd);

   hypre_ParCSRMatrix  *B;
   hypre_CSRMatrix     *B_diag;
   hypre_CSRMatrix     *B_offd;
   HYPRE_Int           *B_diag_i;
   HYPRE_Int           *B_diag_j;
   HYPRE_Complex       *B_diag_a;
   HYPRE_Int           *B_offd_i;
   HYPRE_Int           *B_offd_j;
   HYPRE_Complex       *B_offd_a;

   HYPRE_Int            B_diag_nnz, B_offd_nnz;
   HYPRE_BigInt        *B_col_map_offd;
   HYPRE_Int           *col_map_marker;
   HYPRE_BigInt        *col_map_end;

   const dim3           bDim = hypre_GetDefaultDeviceBlockDimension();
   const dim3           gDim = hypre_GetDefaultDeviceGridDimension(num_rows,
                                                                   "w", bDim);

   hypre_GpuProfilingPushRange("ParCSRMatrixBlkFilter");

   /* Create A's device column map */
   if (!hypre_ParCSRMatrixDeviceColMapOffd(A) &&
       hypre_ParCSRMatrixColMapOffd(A))
   {
      hypre_ParCSRMatrixCopyColMapOffdToDevice(A);
      A_col_map_offd = hypre_ParCSRMatrixDeviceColMapOffd(A);
   }

   /* Create and initialize output matrix B */
   B = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts, num_cols_offd,
                                0, 0);
   hypre_ParCSRMatrixInitialize_v2(B, memory_location);

   B_diag = hypre_ParCSRMatrixDiag(B);
   B_offd = hypre_ParCSRMatrixOffd(B);

   B_diag_i = hypre_CSRMatrixI(B_diag);
   B_offd_i = hypre_CSRMatrixI(B_offd);

   /* First pass: count nonzeros */
   HYPRE_GPU_LAUNCH( hypreGPUKernel_ParCSRMatrixBlkFilterCount, gDim, bDim,
                     num_rows, block_size,
                     A_diag_i, A_diag_j,
                     A_offd_i, A_offd_j,
                     A_col_map_offd,
                     B_diag_i, B_offd_i );

   /* Compute row pointers and get total number of nonzeros */
   hypreDevice_IntegerExclusiveScan(num_rows + 1, B_diag_i);
   hypreDevice_IntegerExclusiveScan(num_rows + 1, B_offd_i);
   hypre_TMemcpy(&B_diag_nnz, B_diag_i + num_rows, HYPRE_Int, 1,
                 HYPRE_MEMORY_HOST, memory_location);
   hypre_TMemcpy(&B_offd_nnz, B_offd_i + num_rows, HYPRE_Int, 1,
                 HYPRE_MEMORY_HOST, memory_location);

   /* Allocate memory for B */
   B_diag_j = hypre_TAlloc(HYPRE_Int, B_diag_nnz, memory_location);
   B_offd_j = hypre_TAlloc(HYPRE_Int, B_offd_nnz, memory_location);
   B_diag_a = hypre_TAlloc(HYPRE_Complex, B_diag_nnz, memory_location);
   B_offd_a = hypre_TAlloc(HYPRE_Complex, B_offd_nnz, memory_location);

   /* Create a marker for used columns */
   if (num_cols_offd > 0)
   {
      col_map_marker = hypre_CTAlloc(HYPRE_Int, num_cols_offd, memory_location);
   }
   else
   {
      col_map_marker = NULL;
   }

   /* Second pass: fill B */
   HYPRE_GPU_LAUNCH( hypreGPUKernel_ParCSRMatrixBlkFilterFill, gDim, bDim,
                     num_rows, block_size, num_cols_offd,
                     A_diag_i, A_diag_j, A_diag_a,
                     A_offd_i, A_offd_j, A_offd_a,
                     A_col_map_offd,
                     B_diag_i, B_diag_j, B_diag_a,
                     B_offd_i, B_offd_j, B_offd_a,
                     col_map_marker );

   /* Update CSR matrix structures */
   hypre_CSRMatrixJ(B_diag)           = B_diag_j;
   hypre_CSRMatrixData(B_diag)        = B_diag_a;
   hypre_CSRMatrixNumNonzeros(B_diag) = B_diag_nnz;
   hypre_CSRMatrixJ(B_offd)           = B_offd_j;
   hypre_CSRMatrixData(B_offd)        = B_offd_a;
   hypre_CSRMatrixNumNonzeros(B_offd) = B_offd_nnz;

   /* Set up B's col_map_offd */
   if (B_offd_nnz > 0)
   {
      /* Create B's device column map */
      hypre_ParCSRMatrixDeviceColMapOffd(B) = hypre_CTAlloc(HYPRE_BigInt,
                                                            num_cols_offd,
                                                            HYPRE_MEMORY_DEVICE);
      B_col_map_offd = hypre_ParCSRMatrixDeviceColMapOffd(B);

#ifndef HYPRE_USING_SYCL
      /* Copy used columns to B's col_map_offd */
      col_map_end = HYPRE_THRUST_CALL(copy_if,
                                      A_col_map_offd,
                                      A_col_map_offd + num_cols_offd,
                                      col_map_marker,
                                      B_col_map_offd,
                                      thrust::identity<HYPRE_Int>());

      hypre_CSRMatrixNumCols(B_offd) = (HYPRE_Int) (col_map_end - B_col_map_offd);

      /* Copy B's column map to host */
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(B),
                    hypre_ParCSRMatrixDeviceColMapOffd(B),
                    HYPRE_BigInt,
                    hypre_CSRMatrixNumCols(B_offd),
                    HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
#else
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "SYCL path not implemented!");
      hypre_GpuProfilingPopRange();
      return hypre_error_flag;
#endif
   }

   /* Update global nonzeros */
   hypre_ParCSRMatrixSetDNumNonzeros(B);
   hypre_ParCSRMatrixNumNonzeros(B) = (HYPRE_BigInt) hypre_ParCSRMatrixDNumNonzeros(B);

   /* TODO (VPM): compute B's commpkg directly from A's commpkg */
   hypre_MatvecCommPkgCreate(B);

   /* Set output pointer */
   *B_ptr = B;

   hypre_TFree(col_map_marker, memory_location);
   hypre_GpuProfilingPopRange();
   return hypre_error_flag;
}

#endif /* if defined(HYPRE_USING_GPU) */
