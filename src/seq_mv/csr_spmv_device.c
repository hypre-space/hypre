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

#if defined(HYPRE_USING_CUDA) ||\
    defined(HYPRE_USING_HIP)  ||\
    defined(HYPRE_USING_SYCL)

#include "csr_spmv_device.h"

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatvecShuffleGT8
 *
 * Templated SpMV device kernel based of warp-shuffle reduction.
 * Uses groups of K threads per row.
 * Specialized function for num_vectors > 8
 *
 * Template parameters:
 *   1) K:  number of threads working on a single row. K = 2, 4, 8, 16, 32
 *   2) F:  fill-mode. See hypreDevice_CSRMatrixMatvec for supported values
 *   3) NV: number of vectors (> 1 for multi-component vectors)
 *   4) T:  data type of matrix/vector coefficients
 *--------------------------------------------------------------------------*/

template <HYPRE_Int F, HYPRE_Int K, HYPRE_Int NV, typename T>
__global__ void
hypreGPUKernel_CSRMatvecShuffleGT8(hypre_DeviceItem &item,
                                   HYPRE_Int         num_rows,
                                   HYPRE_Int         num_vectors,
                                   HYPRE_Int        *row_id,
                                   HYPRE_Int         idxstride_x,
                                   HYPRE_Int         idxstride_y,
                                   HYPRE_Int         vecstride_x,
                                   HYPRE_Int         vecstride_y,
                                   T                 alpha,
                                   HYPRE_Int        *d_ia,
                                   HYPRE_Int        *d_ja,
                                   T                *d_a,
                                   T                *d_x,
                                   T                 beta,
                                   T                *d_y )
{
#if defined (HYPRE_USING_SYCL)
   const HYPRE_Int  grid_ngroups  = item.get_group_range(2) * (HYPRE_SPMV_BLOCKDIM / K);
   HYPRE_Int        grid_group_id = (item.get_group(2) * HYPRE_SPMV_BLOCKDIM + item.get_local_id(
                                        2)) / K;
   const HYPRE_Int  group_lane    = item.get_local_id(2) & (K - 1);
#else
   const HYPRE_Int  grid_ngroups  = gridDim.x * (HYPRE_SPMV_BLOCKDIM / K);
   HYPRE_Int        grid_group_id = (blockIdx.x * HYPRE_SPMV_BLOCKDIM + threadIdx.x) / K;
   const HYPRE_Int  group_lane    = threadIdx.x & (K - 1);
#endif
   T sum[64];

   for (; warp_any_sync(item, HYPRE_WARP_FULL_MASK, grid_group_id < num_rows);
        grid_group_id += grid_ngroups)
   {
      HYPRE_Int grid_row_id = -1, p = 0, q = 0;

      if (row_id)
      {
         if (grid_group_id < num_rows && group_lane == 0)
         {
            grid_row_id = read_only_load(&row_id[grid_group_id]);
         }
         grid_row_id = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, grid_row_id, 0, K);
      }
      else
      {
         grid_row_id = grid_group_id;
      }

      if (grid_group_id < num_rows && group_lane < 2)
      {
         p = read_only_load(&d_ia[grid_row_id + group_lane]);
      }
      q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1, K);
      p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0, K);

      for (HYPRE_Int i = 0; i < num_vectors; i++)
      {
         sum[i] = T(0.0);
      }

#pragma unroll 1
      for (p += group_lane; p < q; p += K * 2)
      {
         HYPRE_SPMV_ADD_SUM(p, num_vectors)
         if (p + K < q)
         {
            HYPRE_SPMV_ADD_SUM((p + K), num_vectors)
         }
      }

      // parallel reduction
      for (HYPRE_Int i = 0; i < num_vectors; i++)
      {
         for (HYPRE_Int d = K / 2; d > 0; d >>= 1)
         {
            sum[i] += warp_shuffle_down_sync(item, HYPRE_WARP_FULL_MASK, sum[i], d);
         }
      }

      if (grid_group_id < num_rows && group_lane == 0)
      {
         if (beta)
         {
            for (HYPRE_Int i = 0; i < num_vectors; i++)
            {
               d_y[grid_row_id * idxstride_y + i * vecstride_y] =
                  alpha * sum[i] +
                  beta * d_y[grid_row_id * idxstride_y + i * vecstride_y];
            }
         }
         else
         {
            for (HYPRE_Int i = 0; i < num_vectors; i++)
            {
               d_y[grid_row_id * idxstride_y + i * vecstride_y] = alpha * sum[i];
            }
         }
      }
   }
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatvecShuffle
 *
 * Templated SpMV device kernel based of warp-shuffle reduction.
 * Uses groups of K threads per row
 *
 * Template parameters:
 *   1) K:  number of threads working on a single row. K = 2, 4, 8, 16, 32
 *   2) F:  fill-mode. See hypreDevice_CSRMatrixMatvec for supported values
 *   3) NV: number of vectors (> 1 for multi-component vectors)
 *   4) T:  data type of matrix/vector coefficients
 *--------------------------------------------------------------------------*/

template <HYPRE_Int F, HYPRE_Int K, HYPRE_Int NV, typename T>
__global__ void
//__launch_bounds__(512, 1)
hypreGPUKernel_CSRMatvecShuffle(hypre_DeviceItem &item,
                                HYPRE_Int         num_rows,
                                HYPRE_Int         num_vectors,
                                HYPRE_Int        *row_id,
                                HYPRE_Int         idxstride_x,
                                HYPRE_Int         idxstride_y,
                                HYPRE_Int         vecstride_x,
                                HYPRE_Int         vecstride_y,
                                T                 alpha,
                                HYPRE_Int        *d_ia,
                                HYPRE_Int        *d_ja,
                                T                *d_a,
                                T                *d_x,
                                T                 beta,
                                T                *d_y )
{
#if defined(HYPRE_USING_SYCL)
   HYPRE_Int grid_ngroups  = item.get_group_range(2) * (HYPRE_SPMV_BLOCKDIM / K);
   HYPRE_Int grid_group_id = (item.get_group(2) * HYPRE_SPMV_BLOCKDIM + item.get_local_id(2)) / K;
   HYPRE_Int group_lane    = item.get_local_id(2) & (K - 1);
#else
   const HYPRE_Int  grid_ngroups  = gridDim.x * (HYPRE_SPMV_BLOCKDIM / K);
   HYPRE_Int        grid_group_id = (blockIdx.x * HYPRE_SPMV_BLOCKDIM + threadIdx.x) / K;
   const HYPRE_Int  group_lane    = threadIdx.x & (K - 1);
#endif

   for (; warp_any_sync(item, HYPRE_WARP_FULL_MASK, grid_group_id < num_rows);
        grid_group_id += grid_ngroups)
   {
      HYPRE_Int grid_row_id = -1, p = 0, q = 0;

      if (row_id)
      {
         if (grid_group_id < num_rows && group_lane == 0)
         {
            grid_row_id = read_only_load(&row_id[grid_group_id]);
         }
         grid_row_id = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, grid_row_id, 0, K);
      }
      else
      {
         grid_row_id = grid_group_id;
      }

      if (grid_group_id < num_rows && group_lane < 2)
      {
         p = read_only_load(&d_ia[grid_row_id + group_lane]);
      }
      q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1, K);
      p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0, K);

      T sum[NV] = {T(0)};
#if HYPRE_SPMV_VERSION == 1
#pragma unroll 1
      for (p += group_lane; p < q; p += K * 2)
      {
         HYPRE_SPMV_ADD_SUM(p, NV)
         if (p + K < q)
         {
            HYPRE_SPMV_ADD_SUM((p + K), NV)
         }
      }
#elif HYPRE_SPMV_VERSION == 2
#pragma unroll 1
      for (p += group_lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, p < q); p += K)
      {
         if (p < q)
         {
            HYPRE_SPMV_ADD_SUM(p, NV)
         }
      }
#else
#pragma unroll 1
      for (p += group_lane;  p < q; p += K)
      {
         HYPRE_SPMV_ADD_SUM(p, NV)
      }
#endif

      // parallel reduction
      for (HYPRE_Int i = 0; i < NV; i++)
      {
         for (HYPRE_Int d = K / 2; d > 0; d >>= 1)
         {
            sum[i] += warp_shuffle_down_sync(item, HYPRE_WARP_FULL_MASK, sum[i], d);
         }
      }

      if (grid_group_id < num_rows && group_lane == 0)
      {
         if (beta)
         {
            for (HYPRE_Int i = 0; i < NV; i++)
            {
               d_y[grid_row_id * idxstride_y + i * vecstride_y] =
                  alpha * sum[i] +
                  beta * d_y[grid_row_id * idxstride_y + i * vecstride_y];
            }
         }
         else
         {
            for (HYPRE_Int i = 0; i < NV; i++)
            {
               d_y[grid_row_id * idxstride_y + i * vecstride_y] = alpha * sum[i];
            }
         }
      }
   }
}

/*--------------------------------------------------------------------------
 * hypreDevice_CSRMatrixMatvec
 *
 * Templated host function for launching the device kernels for SpMV.
 *
 * The template parameter F is the fill-mode. Supported values:
 *    0: whole matrix
 *   -1: lower
 *    1: upper
 *   -2: strict lower
 *    2: strict upper
 * The template parameter T is the matrix/vector coefficient data type
 *--------------------------------------------------------------------------*/

template <HYPRE_Int F, typename T>
HYPRE_Int
hypreDevice_CSRMatrixMatvec( HYPRE_Int  num_vectors,
                             HYPRE_Int  num_rows,
                             HYPRE_Int *rowid,
                             HYPRE_Int  num_nonzeros,
                             HYPRE_Int  idxstride_x,
                             HYPRE_Int  idxstride_y,
                             HYPRE_Int  vecstride_x,
                             HYPRE_Int  vecstride_y,
                             T          alpha,
                             HYPRE_Int *d_ia,
                             HYPRE_Int *d_ja,
                             T         *d_a,
                             T         *d_x,
                             T          beta,
                             T         *d_y )
{
   if (num_vectors > 64)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "hypre's SpMV: (num_vectors > 64) not implemented");
      return hypre_error_flag;
   }

   const HYPRE_Int avg_rownnz = (num_nonzeros + num_rows - 1) / num_rows;

   static constexpr HYPRE_Int group_sizes[5] = {32, 16, 8, 4, 4};

   static constexpr HYPRE_Int unroll_depth[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

   static HYPRE_Int avg_rownnz_lower_bounds[5] = {64, 32, 16, 8, 0};

   static HYPRE_Int num_groups_per_block[5] = { HYPRE_SPMV_BLOCKDIM / group_sizes[0],
                                                HYPRE_SPMV_BLOCKDIM / group_sizes[1],
                                                HYPRE_SPMV_BLOCKDIM / group_sizes[2],
                                                HYPRE_SPMV_BLOCKDIM / group_sizes[3],
                                                HYPRE_SPMV_BLOCKDIM / group_sizes[4]
                                              };

   const dim3 bDim = hypre_dim3(HYPRE_SPMV_BLOCKDIM);

   /* Select execution path */
   switch (num_vectors)
   {
      case unroll_depth[1]:
         HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[1]);
         break;

      case unroll_depth[2]:
         HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[2]);
         break;

      case unroll_depth[3]:
         HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[3]);
         break;

      case unroll_depth[4]:
         HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[4]);
         break;

      case unroll_depth[5]:
         HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[5]);
         break;

      case unroll_depth[6]:
         HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[6]);
         break;

      case unroll_depth[7]:
         HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[7]);
         break;

      case unroll_depth[8]:
         HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[8]);
         break;

      default:
         HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffleGT8, unroll_depth[8]);
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSpMVDevice
 *
 * hypre's internal implementation of sparse matrix/vector multiplication
 * (SpMV) on GPUs.
 *
 * Computes:  y = alpha*op(B)*x + beta*y
 *
 * Supported cases:
 *   1) rownnz_B != NULL: y(rownnz_B) = alpha*op(B)*x + beta*y(rownnz_B)
 *
 *   2) op(B) = B (trans = 0) or B^T (trans = 1)
 *      op(B) = B^T: not recommended since it computes B^T at every call
 *
 *   3) multi-component vectors up to 64 components (1 <= num_vectors <= 64)
 *
 * Notes:
 *   1) if B has no numerical values, assume the values are all ones
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSpMVDevice( HYPRE_Int        trans,
                           HYPRE_Complex    alpha,
                           hypre_CSRMatrix *B,
                           hypre_Vector    *x,
                           HYPRE_Complex    beta,
                           hypre_Vector    *y,
                           HYPRE_Int        fill )
{
   /* Input data variables */
   HYPRE_Int        num_rows      = trans ? hypre_CSRMatrixNumCols(B) : hypre_CSRMatrixNumRows(B);
   HYPRE_Int        num_nonzeros  = hypre_CSRMatrixNumNonzeros(B);
   HYPRE_Int        num_vectors_x = hypre_VectorNumVectors(x);
   HYPRE_Int        num_vectors_y = hypre_VectorNumVectors(y);
   HYPRE_Complex   *d_x           = hypre_VectorData(x);
   HYPRE_Complex   *d_y           = hypre_VectorData(y);
   HYPRE_Int        idxstride_x   = hypre_VectorIndexStride(x);
   HYPRE_Int        vecstride_x   = hypre_VectorVectorStride(x);
   HYPRE_Int        idxstride_y   = hypre_VectorIndexStride(y);
   HYPRE_Int        vecstride_y   = hypre_VectorVectorStride(y);

   /* Matrix A variables */
   hypre_CSRMatrix *A = NULL;
   HYPRE_Int       *d_ia;
   HYPRE_Int       *d_ja;
   HYPRE_Complex   *d_a;
   HYPRE_Int       *d_rownnz_A = NULL;

   /* Sanity checks */
   if (num_vectors_x != num_vectors_y)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "num_vectors_x != num_vectors_y");
      return hypre_error_flag;
   }
   hypre_assert(num_rows > 0);

   /* Trivial case when alpha * op(B) * x = 0 */
   if (num_nonzeros <= 0 || alpha == 0.0)
   {
      hypre_SeqVectorScale(beta, y);
      return hypre_error_flag;
   }

   /* Select op(B) */
   if (trans)
   {
      hypre_CSRMatrixTransposeDevice(B, &A, hypre_CSRMatrixData(B) != NULL);
   }
   else
   {
      A = B;
   }

   /* Get matrix A info */
   d_ia = hypre_CSRMatrixI(A);
   d_ja = hypre_CSRMatrixJ(A);
   d_a  = hypre_CSRMatrixData(A);

   if (hypre_CSRMatrixRownnz(A))
   {
      num_rows   = hypre_CSRMatrixNumRownnz(A);
      d_rownnz_A = hypre_CSRMatrixRownnz(A);

      hypre_SeqVectorScale(beta, y);
      beta = beta ? 1.0 : 0.0;
   }

   /* Choose matrix fill mode */
   switch (fill)
   {
      case HYPRE_SPMV_FILL_STRICT_LOWER:
         /* Strict lower matrix */
         hypreDevice_CSRMatrixMatvec<HYPRE_SPMV_FILL_STRICT_LOWER>(num_vectors_x,
                                                                   num_rows,
                                                                   d_rownnz_A,
                                                                   num_nonzeros,
                                                                   idxstride_x,
                                                                   idxstride_y,
                                                                   vecstride_x,
                                                                   vecstride_y,
                                                                   alpha,
                                                                   d_ia,
                                                                   d_ja,
                                                                   d_a,
                                                                   d_x,
                                                                   beta,
                                                                   d_y);
         break;

      case HYPRE_SPMV_FILL_LOWER:
         /* Lower matrix */
         hypreDevice_CSRMatrixMatvec<HYPRE_SPMV_FILL_LOWER>(num_vectors_x,
                                                            num_rows,
                                                            d_rownnz_A,
                                                            num_nonzeros,
                                                            idxstride_x,
                                                            idxstride_y,
                                                            vecstride_x,
                                                            vecstride_y,
                                                            alpha,
                                                            d_ia,
                                                            d_ja,
                                                            d_a,
                                                            d_x,
                                                            beta,
                                                            d_y);
         break;

      case HYPRE_SPMV_FILL_WHOLE:
         /* Full matrix */
         hypreDevice_CSRMatrixMatvec<HYPRE_SPMV_FILL_WHOLE>(num_vectors_x,
                                                            num_rows,
                                                            d_rownnz_A,
                                                            num_nonzeros,
                                                            idxstride_x,
                                                            idxstride_y,
                                                            vecstride_x,
                                                            vecstride_y,
                                                            alpha,
                                                            d_ia,
                                                            d_ja,
                                                            d_a,
                                                            d_x,
                                                            beta,
                                                            d_y);
         break;

      case HYPRE_SPMV_FILL_UPPER:
         /* Upper matrix */
         hypreDevice_CSRMatrixMatvec<HYPRE_SPMV_FILL_UPPER>(num_vectors_x,
                                                            num_rows,
                                                            d_rownnz_A,
                                                            num_nonzeros,
                                                            idxstride_x,
                                                            idxstride_y,
                                                            vecstride_x,
                                                            vecstride_y,
                                                            alpha,
                                                            d_ia,
                                                            d_ja,
                                                            d_a,
                                                            d_x,
                                                            beta,
                                                            d_y);
         break;

      case HYPRE_SPMV_FILL_STRICT_UPPER:
         /* Strict upper matrix */
         hypreDevice_CSRMatrixMatvec<HYPRE_SPMV_FILL_STRICT_UPPER>(num_vectors_x,
                                                                   num_rows,
                                                                   d_rownnz_A,
                                                                   num_nonzeros,
                                                                   idxstride_x,
                                                                   idxstride_y,
                                                                   vecstride_x,
                                                                   vecstride_y,
                                                                   alpha,
                                                                   d_ia,
                                                                   d_ja,
                                                                   d_a,
                                                                   d_x,
                                                                   beta,
                                                                   d_y);
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Fill mode for SpMV unavailable!");
         return hypre_error_flag;
   }

   /* Free memory */
   if (trans)
   {
      hypre_CSRMatrixDestroy(A);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixIntSpMVDevice
 *
 * Sparse matrix/vector multiplication with integer data on GPUs
 *
 * Note: This function does not support multi-component vectors
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixIntSpMVDevice( HYPRE_Int  num_rows,
                              HYPRE_Int  num_nonzeros,
                              HYPRE_Int  alpha,
                              HYPRE_Int *d_ia,
                              HYPRE_Int *d_ja,
                              HYPRE_Int *d_a,
                              HYPRE_Int *d_x,
                              HYPRE_Int  beta,
                              HYPRE_Int *d_y )
{
   /* Additional input variables */
   HYPRE_Int        num_vectors = 1;
   HYPRE_Int        idxstride_x = 1;
   HYPRE_Int        vecstride_x = 1;
   HYPRE_Int        idxstride_y = 1;
   HYPRE_Int        vecstride_y = 1;
   HYPRE_Int       *d_rownnz    = NULL;

   hypreDevice_CSRMatrixMatvec<HYPRE_SPMV_FILL_WHOLE, HYPRE_Int>(num_vectors,
                                                                 num_rows,
                                                                 d_rownnz,
                                                                 num_nonzeros,
                                                                 idxstride_x,
                                                                 idxstride_y,
                                                                 vecstride_x,
                                                                 vecstride_y,
                                                                 alpha,
                                                                 d_ia,
                                                                 d_ja,
                                                                 d_a,
                                                                 d_x,
                                                                 beta,
                                                                 d_y);

   return hypre_error_flag;
}
#endif /* #if defined(HYPRE_USING_GPU) */
