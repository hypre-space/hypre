/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#define SPMV_BLOCKDIM 512
#define VERSION 1

#define SPMV_ADD_SUM(p)                                              \
{                                                                    \
   const HYPRE_Int col = read_only_load(&d_ja[p]);                   \
   if (F == 0)                                                       \
   {                                                                 \
      const T val = d_a ? read_only_load(&d_a[p]) : T(1);            \
      sum += val * read_only_load(&d_x[col]);                        \
   }                                                                 \
   else if (F == -1)                                                 \
   {                                                                 \
      if (col <= grid_group_id)                                      \
      {                                                              \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);         \
         sum += val * read_only_load(&d_x[col]);                     \
      }                                                              \
   }                                                                 \
   else if (F == 1)                                                  \
   {                                                                 \
      if (col >= grid_group_id)                                      \
      {                                                              \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);         \
         sum += val * read_only_load(&d_x[col]);                     \
      }                                                              \
   }                                                                 \
   else if (F == -2)                                                 \
   {                                                                 \
      if (col < grid_group_id)                                       \
      {                                                              \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);         \
         sum += val * read_only_load(&d_x[col]);                     \
      }                                                              \
   }                                                                 \
   else if (F == 2)                                                  \
   {                                                                 \
      if (col > grid_group_id)                                       \
      {                                                              \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);         \
         sum += val * read_only_load(&d_x[col]);                     \
      }                                                              \
   }                                                                 \
}

/* K is the number of threads working on a single row. K = 2, 4, 8, 16, 32 */
template <HYPRE_Int F, HYPRE_Int K, typename T>
__global__ void
hypre_csr_v_k_shuffle(hypre_DeviceItem &item,
                      HYPRE_Int  n,
                      HYPRE_Int *row_id,
                      T          alpha,
                      HYPRE_Int *d_ia,
                      HYPRE_Int *d_ja,
                      T         *d_a,
                      T         *d_x,
                      T          beta,
                      T         *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *               warp-shuffle reduction
    *           (1 group of K threads) per row
    *------------------------------------------------------------*/
   const HYPRE_Int grid_ngroups = gridDim.x * (SPMV_BLOCKDIM / K);
   HYPRE_Int grid_group_id = (blockIdx.x * SPMV_BLOCKDIM + threadIdx.x) / K;
   const HYPRE_Int group_lane = threadIdx.x & (K - 1);

   for (; warp_any_sync(item, HYPRE_WARP_FULL_MASK, grid_group_id < n); grid_group_id += grid_ngroups)
   {
      HYPRE_Int grid_row_id = -1, p = 0, q = 0;

      if (row_id)
      {
         if (grid_group_id < n && group_lane == 0)
         {
            grid_row_id = read_only_load(&row_id[grid_group_id]);
         }
         grid_row_id = __shfl_sync(HYPRE_WARP_FULL_MASK, grid_row_id, 0, K);
      }
      else
      {
         grid_row_id = grid_group_id;
      }

      if (grid_group_id < n && group_lane < 2)
      {
         p = read_only_load(&d_ia[grid_row_id + group_lane]);
      }
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1, K);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0, K);

      T sum = 0.0;
#if VERSION == 1
#pragma unroll 1
      for (p += group_lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, p < q); p += K * 2)
      {
         if (p < q)
         {
            SPMV_ADD_SUM(p)
            if (p + K < q)
            {
               SPMV_ADD_SUM((p + K))
            }
         }
      }
#elif VERSION == 2
#pragma unroll 1
      for (p += group_lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, p < q); p += K)
      {
         if (p < q)
         {
            SPMV_ADD_SUM(p)
         }
      }
#else
#pragma unroll 1
      for (p += group_lane;  p < q; p += K)
      {
         SPMV_ADD_SUM(p)
      }
#endif
      // parallel reduction
#pragma unroll
      for (HYPRE_Int d = K / 2; d > 0; d >>= 1)
      {
         sum += __shfl_down_sync(HYPRE_WARP_FULL_MASK, sum, d);
      }
      if (grid_group_id < n && group_lane == 0)
      {
         if (beta)
         {
            d_y[grid_row_id] = alpha * sum + beta * d_y[grid_row_id];
         }
         else
         {
            d_y[grid_row_id] = alpha * sum;
         }
      }
   }
}

/* F is fill-mode
 *    0: whole matrix
 *   -1: lower
 *    1: upper
 *   -2: strict lower
 *    2: strict upper
 */
template <HYPRE_Int F, typename T>
HYPRE_Int
hypreDevice_CSRMatrixMatvec( HYPRE_Int  nrows,
                             HYPRE_Int *rowid,
                             HYPRE_Int  nnz,
                             T          alpha,
                             HYPRE_Int *d_ia,
                             HYPRE_Int *d_ja,
                             T         *d_a,
                             T         *d_x,
                             T          beta,
                             T         *d_y )
{
   const HYPRE_Int rownnz = (nnz + nrows - 1) / nrows;
   const dim3 bDim(SPMV_BLOCKDIM);

   if (rownnz >= 64)
   {
      const HYPRE_Int group_size = 32;
      const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
      HYPRE_GPU_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, T>), gDim, bDim,
                        nrows, rowid, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
   }
   else if (rownnz >= 32)
   {
      const HYPRE_Int group_size = 16;
      const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
      HYPRE_GPU_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, T>), gDim, bDim,
                        nrows, rowid, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
   }
   else if (rownnz >= 16)
   {
      const HYPRE_Int group_size = 8;
      const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
      HYPRE_GPU_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, T>), gDim, bDim,
                        nrows, rowid, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
   }
   else if (rownnz >= 8)
   {
      const HYPRE_Int group_size = 4;
      const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
      HYPRE_GPU_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, T>), gDim, bDim,
                        nrows, rowid, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
   }
   else
   {
      const HYPRE_Int group_size = 4;
      const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
      HYPRE_GPU_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, T>), gDim, bDim,
                        nrows, rowid, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
   }

   return hypre_error_flag;
}

/* y = alpha*op(B)*x + beta*y, op(B) = B or B^T
 * Note: if B has no numrical values, assume the values are all ones
 */
HYPRE_Int
hypre_CSRMatrixSpMVDevice( HYPRE_Int        trans,
                           HYPRE_Complex    alpha,
                           hypre_CSRMatrix *B,
                           hypre_Vector    *x,
                           HYPRE_Complex    beta,
                           hypre_Vector    *y,
                           HYPRE_Int        fill )
{
   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(B);

   if (nnz <= 0 || alpha == 0.0)
   {
      hypre_SeqVectorScale(beta, y);
      return hypre_error_flag;
   }

   hypre_CSRMatrix *A = NULL;

   if (trans)
   {
      hypre_CSRMatrixTransposeDevice(B, &A, hypre_CSRMatrixData(B) != NULL);
   }
   else
   {
      A = B;
   }

   HYPRE_Int nrows, *rowid;

   if (hypre_CSRMatrixRownnz(A))
   {
      nrows = hypre_CSRMatrixNumRownnz(A);
      rowid = hypre_CSRMatrixRownnz(A);
   }
   else
   {
      nrows = hypre_CSRMatrixNumRows(A);
      rowid = NULL;
   }

   hypre_assert(nrows > 0);

   HYPRE_Int     *d_ia = hypre_CSRMatrixI(A);
   HYPRE_Int     *d_ja = hypre_CSRMatrixJ(A);
   HYPRE_Complex *d_a  = hypre_CSRMatrixData(A);
   HYPRE_Complex *d_x  = hypre_VectorData(x);
   HYPRE_Complex *d_y  = hypre_VectorData(y);

   if (fill == 0)
   {
      return hypreDevice_CSRMatrixMatvec<0>(nrows, rowid, nnz, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == 1)
   {
      return hypreDevice_CSRMatrixMatvec<1>(nrows, rowid, nnz, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == -1)
   {
      return hypreDevice_CSRMatrixMatvec < -1 > (nrows, rowid, nnz, alpha, d_ia, d_ja, d_a, d_x, beta,
                                                 d_y);
   }
   else if (fill == 2)
   {
      return hypreDevice_CSRMatrixMatvec<2>(nrows, rowid, nnz, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == -2)
   {
      return hypreDevice_CSRMatrixMatvec < -2 > (nrows, rowid, nnz, alpha, d_ia, d_ja, d_a, d_x, beta,
                                                 d_y);
   }

   if (trans)
   {
      hypre_CSRMatrixDestroy(A);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_CSRMatrixIntSpMVDevice( HYPRE_Int  nrows,
                              HYPRE_Int  nnz,
                              HYPRE_Int  alpha,
                              HYPRE_Int *d_ia,
                              HYPRE_Int *d_ja,
                              HYPRE_Int *d_a,
                              HYPRE_Int *d_x,
                              HYPRE_Int  beta,
                              HYPRE_Int *d_y )
{
   hypreDevice_CSRMatrixMatvec<0, HYPRE_Int>(nrows, NULL, nnz, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);

   return hypre_error_flag;
}
#endif /*#if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP) */
