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

#include "csr_spmv_device.h"

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatvecShuffle
 *
 * Templated SpMV device kernel based of warp-shuffle reduction.
 * Uses groups of K threads per row
 *
 * Template parameters:
 *   1) K:  number of threads working on a single row. K = 2, 4, 8, 16, 32
 *   2) F:  fill-mode. See hypreDevice_CSRMatrixMatvec for supported values
 *   3) NV: number of vectors (> 1 for multivectors)
 *   4) T:  data type of matrix/vector coefficients
 *--------------------------------------------------------------------------*/

template <HYPRE_Int F, HYPRE_Int K, HYPRE_Int NV, typename T>
__global__ void
hypreGPUKernel_CSRMatvecShuffle(hypre_DeviceItem &item,
                                HYPRE_Int         nrows,
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
   const HYPRE_Int  grid_ngroups  = gridDim.x * (HYPRE_SPMV_BLOCKDIM / K);
   HYPRE_Int        grid_group_id = (blockIdx.x * HYPRE_SPMV_BLOCKDIM + threadIdx.x) / K;
   const HYPRE_Int  group_lane    = threadIdx.x & (K - 1);

   for (; warp_any_sync(item, HYPRE_WARP_FULL_MASK, grid_group_id < nrows);
          grid_group_id += grid_ngroups)
   {
      HYPRE_Int grid_row_id = -1, p = 0, q = 0;

      if (row_id)
      {
         if (grid_group_id < nrows && group_lane == 0)
         {
            grid_row_id = read_only_load(&row_id[grid_group_id]);
         }
         grid_row_id = __shfl_sync(HYPRE_WARP_FULL_MASK, grid_row_id, 0, K);
      }
      else
      {
         grid_row_id = grid_group_id;
      }

      if (grid_group_id < nrows && group_lane < 2)
      {
         p = read_only_load(&d_ia[grid_row_id + group_lane]);
      }
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1, K);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0, K);

      T sum[NV] = {T(0)};
#if HYPRE_SPMV_VERSION == 1
#pragma unroll 1
      for (p += group_lane; p < q; p += K * 2)
      {
         HYPRE_SPMV_ADD_SUM(p)
         if (p + K < q)
         {
            HYPRE_SPMV_ADD_SUM((p + K))
         }
      }
#elif HYPRE_SPMV_VERSION == 2
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
         HYPRE_SPMV_ADD_SUM(p)
      }
#endif

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

      if (grid_group_id < nrows && group_lane == 0)
      {
         if (beta)
         {
#pragma unroll
            for (HYPRE_Int i = 0; i < NV; i++)
            {
               d_y[grid_row_id * idxstride_y + i * vecstride_y] =
                  alpha * sum[i] +
                  beta * d_y[grid_row_id * idxstride_y + i * vecstride_y];
            }
         }
         else
         {
#pragma unroll
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
                             HYPRE_Int  nrows,
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
   const HYPRE_Int avg_rownnz = (num_nonzeros + nrows - 1) / nrows;
   const dim3 bDim(HYPRE_SPMV_BLOCKDIM);

   /* Note: cannot transform this into a loop because num_vectors is a template argument */
   switch (num_vectors)
   {
      case 1:
         HYPRE_SPMV_GPU_LAUNCH(1);
         break;

      case 2:
         HYPRE_SPMV_GPU_LAUNCH(2);
         break;

      case 3:
         HYPRE_SPMV_GPU_LAUNCH(3);
         break;

      case 4:
         HYPRE_SPMV_GPU_LAUNCH(4);
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "hypre's SpMV: (num_vectors > 4) not implemented");
         return hypre_error_flag;
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
 *   3) multivectors up to 4 components (1 <= num_vectors <= 4)
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
   HYPRE_Int        num_rows     = trans ? hypre_CSRMatrixNumCols(B) : hypre_CSRMatrixNumRows(B);
   HYPRE_Int        num_nonzeros = hypre_CSRMatrixNumNonzeros(B);
   HYPRE_Int        num_vectors  = hypre_VectorNumVectors(x);
   HYPRE_Complex   *d_x          = hypre_VectorData(x);
   HYPRE_Complex   *d_y          = hypre_VectorData(y);
   HYPRE_Int        idxstride_x  = hypre_VectorIndexStride(x);
   HYPRE_Int        vecstride_x  = hypre_VectorVectorStride(x);
   HYPRE_Int        idxstride_y  = hypre_VectorIndexStride(y);
   HYPRE_Int        vecstride_y  = hypre_VectorVectorStride(y);

   /* Matrix A variables */
   hypre_CSRMatrix *A = NULL;
   HYPRE_Int       *d_ia;
   HYPRE_Int       *d_ja;
   HYPRE_Complex   *d_a;
   HYPRE_Int       *d_rownnz_A = NULL;

   /* Trivial case when alpha*op(B)*x = 0 */
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
   hypre_assert(num_rows > 0);

   /* Get matrix A info */
   d_ia = hypre_CSRMatrixI(A);
   d_ja = hypre_CSRMatrixJ(A);
   d_a  = hypre_CSRMatrixData(A);
   if (hypre_CSRMatrixRownnz(A))
   {
      num_rows   = hypre_CSRMatrixNumRownnz(A);
      d_rownnz_A = hypre_CSRMatrixRownnz(A);
   }

   /* Choose matrix fill mode */
   switch (fill)
   {
      case -2:
         /* Strict lower matrix */
         hypreDevice_CSRMatrixMatvec<-2>(num_vectors, num_rows, d_rownnz_A, num_nonzeros,
                                         idxstride_x, idxstride_y, vecstride_x, vecstride_y,
                                         alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
         break;

      case -1:
         /* Lower matrix */
         hypreDevice_CSRMatrixMatvec<-1>(num_vectors, num_rows, d_rownnz_A, num_nonzeros,
                                         idxstride_x, idxstride_y, vecstride_x, vecstride_y,
                                         alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
         break;

      case 0:
         /* Whole matrix */
         hypreDevice_CSRMatrixMatvec<0>(num_vectors, num_rows, d_rownnz_A, num_nonzeros,
                                        idxstride_x, idxstride_y, vecstride_x, vecstride_y,
                                        alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
         break;

      case 1:
         /* Upper matrix */
         hypreDevice_CSRMatrixMatvec<1>(num_vectors, num_rows, d_rownnz_A, num_nonzeros,
                                        idxstride_x, idxstride_y, vecstride_x, vecstride_y,
                                        alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
         break;

      case 2:
         /* Strict upper matrix */
         hypreDevice_CSRMatrixMatvec<2>(num_vectors, num_rows, d_rownnz_A, num_nonzeros,
                                        idxstride_x, idxstride_y, vecstride_x, vecstride_y,
                                        alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
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
 * Note: This function does not support multivectors
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

   hypreDevice_CSRMatrixMatvec<0, HYPRE_Int>(num_vectors, num_rows, d_rownnz, num_nonzeros,
                                             idxstride_x, idxstride_y, vecstride_x, vecstride_y,
                                             alpha, d_ia, d_ja, d_a,
                                             d_x, beta, d_y);

   return hypre_error_flag;
}
#endif /*#if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP) */
