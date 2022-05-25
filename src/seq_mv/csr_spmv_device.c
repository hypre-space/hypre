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

/* #define SPMM_ADD_SUM(p)                                              \ */
/* {                                                                    \ */
/*    const HYPRE_Int col = read_only_load(&d_ja[p]);                   \ */
/*    const T aij = read_only_load(&d_a[p]);                            \ */
/*    if (F == 0)                                                       \ */
/*    {                                                                 \ */
/*       sum0 += aij * read_only_load(&d_x[col]);                       \ */
/*       sum1 += aij * read_only_load(&d_x[col + n]);                   \ */
/*    }                                                                 \ */
/*    else if (F == -1)                                                 \ */
/*    {                                                                 \ */
/*       if (col <= grid_group_id)                                      \ */
/*       {                                                              \ */
/*          sum0 += aij * read_only_load(&d_x[col]);                    \ */
/*          sum1 += aij * read_only_load(&d_x[col + n]);                \ */
/*       }                                                              \ */
/*    }                                                                 \ */
/*    else if (F == 1)                                                  \ */
/*    {                                                                 \ */
/*       if (col >= grid_group_id)                                      \ */
/*       {                                                              \ */
/*          sum0 += aij * read_only_load(&d_x[col]);                    \ */
/*          sum1 += aij * read_only_load(&d_x[col + n]);                \ */
/*       }                                                              \ */
/*    }                                                                 \ */
/*    else if (F == -2)                                                 \ */
/*    {                                                                 \ */
/*       if (col < grid_group_id)                                       \ */
/*       {                                                              \ */
/*          sum0 += aij * read_only_load(&d_x[col]);                    \ */
/*          sum1 += aij * read_only_load(&d_x[col + n]);                \ */
/*       }                                                              \ */
/*    }                                                                 \ */
/*    else if (F == 2)                                                  \ */
/*    {                                                                 \ */
/*       if (col > grid_group_id)                                       \ */
/*       {                                                              \ */
/*          sum0 += aij * read_only_load(&d_x[col]);                    \ */
/*          sum1 += aij * read_only_load(&d_x[col + n]);                \ */
/*       }                                                              \ */
/*    }                                                                 \ */
/* } */

#define HYPRE_SPMM_KERNEL_LAUNCH(num_vectors)                                             \
   if (rownnz >= 64)                                                                      \
   {                                                                                      \
      const HYPRE_Int group_size = 32;                                                    \
      const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;                  \
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);         \
      HYPRE_CUDA_LAUNCH( (hypre_spmm_shuffle<F, group_size, num_vectors, HYPRE_Real>),    \
                         gDim, bDim, nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );     \
   }                                                                                      \
   else if (rownnz >= 32)                                                                 \
   {                                                                                      \
      const HYPRE_Int group_size = 16;                                                    \
      const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;                  \
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);         \
      HYPRE_CUDA_LAUNCH( (hypre_spmm_shuffle<F, group_size, num_vectors, HYPRE_Real>),    \
                         gDim, bDim, nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );     \
   }                                                                                      \
   else if (rownnz >= 16)                                                                 \
   {                                                                                      \
      const HYPRE_Int group_size = 8;                                                     \
      const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;                  \
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);         \
      HYPRE_CUDA_LAUNCH( (hypre_spmm_shuffle<F, group_size, num_vectors, HYPRE_Real>),    \
                         gDim, bDim, nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );     \
   }                                                                                      \
   else if (rownnz >= 8)                                                                  \
   {                                                                                      \
      const HYPRE_Int group_size = 4;                                                     \
      const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;                  \
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);         \
      HYPRE_CUDA_LAUNCH( (hypre_spmm_shuffle<F, group_size, num_vectors, HYPRE_Real>),    \
                         gDim, bDim, nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );     \
   }                                                                                      \
   else                                                                                   \
   {                                                                                      \
      const HYPRE_Int group_size = 4;                                                     \
      const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;                  \
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);         \
      HYPRE_CUDA_LAUNCH( (hypre_spmm_shuffle<F, group_size, num_vectors, HYPRE_Real>),    \
                         gDim, bDim, nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );     \
   }

#define SPMV_ADD_SUM(p)                                              \
{                                                                    \
   const HYPRE_Int col = read_only_load(&d_ja[p]);                   \
   if (F == 0)                                                       \
   {                                                                 \
      sum += read_only_load(&d_a[p]) * read_only_load(&d_x[col]);    \
   }                                                                 \
   else if (F == -1)                                                 \
   {                                                                 \
      if (col <= grid_group_id)                                      \
      {                                                              \
         sum += read_only_load(&d_a[p]) * read_only_load(&d_x[col]); \
      }                                                              \
   }                                                                 \
   else if (F == 1)                                                  \
   {                                                                 \
      if (col >= grid_group_id)                                      \
      {                                                              \
         sum += read_only_load(&d_a[p]) * read_only_load(&d_x[col]); \
      }                                                              \
   }                                                                 \
   else if (F == -2)                                                 \
   {                                                                 \
      if (col < grid_group_id)                                       \
      {                                                              \
         sum += read_only_load(&d_a[p]) * read_only_load(&d_x[col]); \
      }                                                              \
   }                                                                 \
   else if (F == 2)                                                  \
   {                                                                 \
      if (col > grid_group_id)                                       \
      {                                                              \
         sum += read_only_load(&d_a[p]) * read_only_load(&d_x[col]); \
      }                                                              \
   }                                                                 \
}

/* K is the number of threads working on a single row. K = 2, 4, 8, 16, 32 */
template <HYPRE_Int F, HYPRE_Int K, typename T>
__global__ void
hypre_csr_v_k_shuffle(HYPRE_Int     n,
                      T             alpha,
                      HYPRE_Int * __restrict__     d_ia,
                      HYPRE_Int * __restrict__     d_ja,
                      T * __restrict__             d_a,
                      T * __restrict__             d_x,
                      T             beta,
                      T * __restrict__             d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *               warp-shuffle reduction
    *            (Group of K threads) per row
    *------------------------------------------------------------*/
   const HYPRE_Int grid_ngroups = gridDim.x * (SPMV_BLOCKDIM / K);
   HYPRE_Int grid_group_id = (blockIdx.x * SPMV_BLOCKDIM + threadIdx.x) / K;
   const HYPRE_Int group_lane = threadIdx.x & (K - 1);
   const HYPRE_Int warp_lane = threadIdx.x & (HYPRE_WARP_SIZE - 1);
   const HYPRE_Int warp_group_id = warp_lane / K;
   const HYPRE_Int warp_ngroups = HYPRE_WARP_SIZE / K;

   for (; __any_sync(HYPRE_WARP_FULL_MASK, grid_group_id < n); grid_group_id += grid_ngroups)
   {
#if 0
      HYPRE_Int p = 0, q = 0;
      if (grid_group_id < n && group_lane < 2)
      {
         p = read_only_load(&d_ia[grid_group_id + group_lane]);
      }
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1, K);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0, K);
#else
      const HYPRE_Int s = grid_group_id - warp_group_id + warp_lane;
      HYPRE_Int p = 0, q = 0;
      if (s <= n && warp_lane <= warp_ngroups)
      {
         p = read_only_load(&d_ia[s]);
      }
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, warp_group_id + 1);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, warp_group_id);
#endif
      T sum = 0.0;
#if VERSION == 1
#pragma unroll 1
      for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K * 2)
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
      for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K)
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
         d_y[grid_group_id] = alpha * sum + beta * d_y[grid_group_id];
      }
   }
}

/* Assumes only two vectors for now */
/* K is the number of threads working on a single row. K = 2, 4, 8, 16, 32 */
template <HYPRE_Int F, HYPRE_Int K, HYPRE_Int NV, typename T>
__global__ void
hypre_spmm_shuffle(HYPRE_Int     n,
                   T             alpha,
                   HYPRE_Int * __restrict__   d_ia,
                   HYPRE_Int * __restrict__   d_ja,
                   T * __restrict__           d_a,
                   T * __restrict__           d_x,
                   T             beta,
                   T * __restrict__           d_y)
{
   /*------------------------------------------------------------*
    *             CSR spmv-multivector kernel
    *               warp-shuffle reduction
    *            (Group of K threads) per row
    *------------------------------------------------------------*/
   const HYPRE_Int grid_ngroups = gridDim.x * (SPMV_BLOCKDIM / K);
   HYPRE_Int grid_group_id = (blockIdx.x * SPMV_BLOCKDIM + threadIdx.x) / K;
   const HYPRE_Int group_lane = threadIdx.x & (K - 1);
   const HYPRE_Int warp_lane = threadIdx.x & (HYPRE_WARP_SIZE - 1);
   const HYPRE_Int warp_group_id = warp_lane / K;
   const HYPRE_Int warp_ngroups = HYPRE_WARP_SIZE / K;

   for (; __any_sync(HYPRE_WARP_FULL_MASK, grid_group_id < n); grid_group_id += grid_ngroups)
   {
      const HYPRE_Int s = grid_group_id - warp_group_id + warp_lane;
      HYPRE_Int p = 0, q = 0;
      if (s <= n && warp_lane <= warp_ngroups)
      {
         p = read_only_load(&d_ia[s]);
      }
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, warp_group_id + 1);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, warp_group_id);

      T sum[NV] = {0.0};
#pragma unroll 1
      for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K * 2)
      {
         if (p < q)
         {
            const HYPRE_Int col = read_only_load(&d_ja[p]);
            const T aij = read_only_load(&d_a[p]);

            if (F == 0)
            {
#pragma unroll
               for (HYPRE_Int i = 0; i < NV; i++)
               {
                  sum[i] += aij * read_only_load(&d_x[col + i*n]);
               }
            }

            if (p + K < q)
            {
               const HYPRE_Int col = read_only_load(&d_ja[p + K]);
               const T aij = read_only_load(&d_a[p + K]);

               if (F == 0)
               {
#pragma unroll
                  for (HYPRE_Int i = 0; i < NV; i++)
                  {
                     sum[i] += aij * read_only_load(&d_x[col + i*n]);
                  }
               }
            }
         }
      }

      // parallel reduction
#pragma unroll
      for (HYPRE_Int d = K / 2; d > 0; d >>= 1)
      {
#pragma unroll
         for (HYPRE_Int i = 0; i < NV; i++)
         {
            sum[i] += __shfl_down_sync(HYPRE_WARP_FULL_MASK, sum[i], d);
         }
      }

      if (grid_group_id < n && group_lane == 0)
      {
#pragma unroll
         for (HYPRE_Int i = 0; i < NV; i++)
         {
            d_y[grid_group_id + i*n] = alpha * sum[i] + beta * d_y[grid_group_id + i*n];
         }
      }
   }
}

/* Assumes only two vectors for now */
/* K is the number of threads working on a single row. K = 2, 4, 8, 16, 32 */
template <HYPRE_Int F, HYPRE_Int K, typename T>
__global__ void
hypre_spmm_shuffle4(HYPRE_Int     n,
                   T             alpha,
                   const HYPRE_Int * __restrict__    d_ia,
                   const HYPRE_Int * __restrict__    d_ja,
                   const T * __restrict__            d_a,
                   const T * __restrict__            d_x,
                   T             beta,
                   T * __restrict__            d_y)
{
   /*------------------------------------------------------------*
    *             CSR spmv-multivector kernel
    *               warp-shuffle reduction
    *            (Group of K threads) per row
    *------------------------------------------------------------*/
   const HYPRE_Int grid_ngroups = gridDim.x * (SPMV_BLOCKDIM / K);
   HYPRE_Int grid_group_id = (blockIdx.x * SPMV_BLOCKDIM + threadIdx.x) / K;
   const HYPRE_Int group_lane = threadIdx.x & (K - 1);
   const HYPRE_Int warp_lane = threadIdx.x & (HYPRE_WARP_SIZE - 1);
   const HYPRE_Int warp_group_id = warp_lane / K;
   const HYPRE_Int warp_ngroups = HYPRE_WARP_SIZE / K;
   const HYPRE_Int nn = n + n;
   const HYPRE_Int nnn = n + n + n;

   for (; __any_sync(HYPRE_WARP_FULL_MASK, grid_group_id < n); grid_group_id += grid_ngroups)
   {
      const HYPRE_Int s = grid_group_id - warp_group_id + warp_lane;
      HYPRE_Int p = 0, q = 0;
      if (s <= n && warp_lane <= warp_ngroups)
      {
         p = read_only_load(&d_ia[s]);
      }
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, warp_group_id + 1);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, warp_group_id);

      //T sum[4] = {0.0};
      T sum0 = 0.0f;
      T sum1 = 0.0f;
      T sum2 = 0.0f;
      T sum3 = 0.0f;
#pragma unroll 1
      for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K * 2)
      {
         if (p < q)
         {
            const HYPRE_Int col = read_only_load(&d_ja[p]);
            const T aij = read_only_load(&d_a[p]);

            if (F == 0)
            {
               sum0 += aij * read_only_load(&d_x[col]);
               sum1 += aij * read_only_load(&d_x[col + n]);
               sum2 += aij * read_only_load(&d_x[col + nn]);
               sum3 += aij * read_only_load(&d_x[col + nnn]);
            }

            if (p + K < q)
            {
               const HYPRE_Int col = read_only_load(&d_ja[p + K]);
               const T aij = read_only_load(&d_a[p + K]);

               if (F == 0)
               {
                  sum0 += aij * read_only_load(&d_x[col]);
                  sum1 += aij * read_only_load(&d_x[col + n]);
                  sum2 += aij * read_only_load(&d_x[col + nn]);
                  sum3 += aij * read_only_load(&d_x[col + nnn]);
               }
            }
         }
      }

      // parallel reduction
#pragma unroll
      for (HYPRE_Int d = K / 2; d > 0; d >>= 1)
      {
         sum0 += __shfl_down_sync(HYPRE_WARP_FULL_MASK, sum0, d);
         sum1 += __shfl_down_sync(HYPRE_WARP_FULL_MASK, sum1, d);
         sum2 += __shfl_down_sync(HYPRE_WARP_FULL_MASK, sum2, d);
         sum3 += __shfl_down_sync(HYPRE_WARP_FULL_MASK, sum3, d);
      }

      if (grid_group_id < n && group_lane == 0)
      {
         d_y[grid_group_id]       = alpha * sum0 + beta * d_y[grid_group_id];
         d_y[grid_group_id + n]   = alpha * sum1 + beta * d_y[grid_group_id + n];
         d_y[grid_group_id + nn]  = alpha * sum2 + beta * d_y[grid_group_id + nn];
         d_y[grid_group_id + nnn] = alpha * sum3 + beta * d_y[grid_group_id + nnn];
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
template <HYPRE_Int F>
HYPRE_Int
hypreDevice_CSRMatrixMatvec( HYPRE_Int      nrows,
                             HYPRE_Int      nnz,
                             HYPRE_Int      num_vectors,
                             HYPRE_Complex  alpha,
                             HYPRE_Int * __restrict__     d_ia,
                             HYPRE_Int * __restrict__     d_ja,
                             HYPRE_Complex * __restrict__ d_a,
                             HYPRE_Complex * __restrict__ d_x,
                             HYPRE_Complex  beta,
                             HYPRE_Complex * __restrict__ d_y )
{
   const HYPRE_Int rownnz = (nnz + nrows - 1) / nrows;
   const dim3 bDim(SPMV_BLOCKDIM);

   if (num_vectors == 1)
   {
      if (rownnz >= 64)
      {
         const HYPRE_Int group_size = 32;
         const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
         const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
         HYPRE_CUDA_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, HYPRE_Real>), gDim, bDim,
                            nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
      }
      else if (rownnz >= 32)
      {
         const HYPRE_Int group_size = 16;
         const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
         const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
         HYPRE_CUDA_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, HYPRE_Real>), gDim, bDim,
                            nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
      }
      else if (rownnz >= 16)
      {
         const HYPRE_Int group_size = 8;
         const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
         const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
         HYPRE_CUDA_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, HYPRE_Real>), gDim, bDim,
                            nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
      }
      else if (rownnz >= 8)
      {
         const HYPRE_Int group_size = 4;
         const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
         const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
         HYPRE_CUDA_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, HYPRE_Real>), gDim, bDim,
                            nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
      }
      else
      {
         const HYPRE_Int group_size = 4;
         const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
         const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
         HYPRE_CUDA_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, HYPRE_Real>), gDim, bDim,
                            nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
      }
   }
   else
   {
      if (num_vectors == 2)
      {
         HYPRE_SPMM_KERNEL_LAUNCH(2);
      }
      else if (num_vectors == 3)
      {
         HYPRE_SPMM_KERNEL_LAUNCH(3);
      }
      else if (num_vectors == 4)
      {
#if 0
         HYPRE_SPMM_KERNEL_LAUNCH(4);
#else
         if (rownnz >= 64)
         {
            const HYPRE_Int group_size = 32;
            const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
            const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
            HYPRE_CUDA_LAUNCH( (hypre_spmm_shuffle4<F, group_size, HYPRE_Real>), gDim, bDim,
                               nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
         }
         else if (rownnz >= 32)
         {
            const HYPRE_Int group_size = 16;
            const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
            const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
            HYPRE_CUDA_LAUNCH( (hypre_spmm_shuffle4<F, group_size, HYPRE_Real>), gDim, bDim,
                               nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
         }
         else if (rownnz >= 16)
         {
            const HYPRE_Int group_size = 8;
            const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
            const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
            HYPRE_CUDA_LAUNCH( (hypre_spmm_shuffle4<F, group_size, HYPRE_Real>), gDim, bDim,
                               nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
         }
         else if (rownnz >= 8)
         {
            const HYPRE_Int group_size = 4;
            const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
            const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
            HYPRE_CUDA_LAUNCH( (hypre_spmm_shuffle4<F, group_size, HYPRE_Real>), gDim, bDim,
                               nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
         }
         else
         {
            const HYPRE_Int group_size = 4;
            const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
            const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
            HYPRE_CUDA_LAUNCH( (hypre_spmm_shuffle4<F, group_size, HYPRE_Real>), gDim, bDim,
                               nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
         }
#endif
      }
      else if (num_vectors == 8)
      {
         HYPRE_SPMM_KERNEL_LAUNCH(8);
      }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "num_vectors > 8 not implemented yet!");
         return hypre_error_flag;
      }
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_CSRMatrixSpMVDevice( HYPRE_Complex    alpha,
                           hypre_CSRMatrix *A,
                           hypre_Vector    *x,
                           HYPRE_Complex    beta,
                           hypre_Vector    *y,
                           HYPRE_Int        fill )
{
   HYPRE_Int      nrows       = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      nnz         = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int     *d_ia        = hypre_CSRMatrixI(A);
   HYPRE_Int     *d_ja        = hypre_CSRMatrixJ(A);
   HYPRE_Complex *d_a         = hypre_CSRMatrixData(A);
   HYPRE_Complex *d_x         = hypre_VectorData(x);
   HYPRE_Complex *d_y         = hypre_VectorData(y);

   if (nnz <= 0 || alpha == 0.0)
   {
      hypre_SeqVectorScale(beta, y);

      return hypre_error_flag;
   }

   hypre_assert(nrows > 0);

   if (fill == 0)
   {
      return hypreDevice_CSRMatrixMatvec<0>(nrows, nnz, num_vectors, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == 1)
   {
      return hypreDevice_CSRMatrixMatvec<1>(nrows, nnz, num_vectors, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == -1)
   {
      return hypreDevice_CSRMatrixMatvec<-1>(nrows, nnz, num_vectors, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == 2)
   {
      return hypreDevice_CSRMatrixMatvec<2>(nrows, nnz, num_vectors, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == -2)
   {
      return hypreDevice_CSRMatrixMatvec<-2>(nrows, nnz, num_vectors, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Fill mode for SpMV unavailable!");
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

#endif /*#if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP) */
