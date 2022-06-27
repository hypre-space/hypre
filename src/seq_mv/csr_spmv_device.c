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
 *   1) K:  the number of threads working on a single row. K = 2, 4, 8, 16, 32
 *   2) F:  fill-mode. See hypreDevice_CSRMatrixMatvec for supported values
 *   3) NV: number of vectors (> 1 for multivectors)
 *   4) T:  data type of matrix/vector coefficients
 *--------------------------------------------------------------------------*/

template <HYPRE_Int F, HYPRE_Int K, HYPRE_Int NV, typename T>
__global__ void
hypreGPUKernel_CSRMatvecShuffle(HYPRE_Int     nrows,
                                HYPRE_Int     ncols,
                                T             alpha,
                                HYPRE_Int    *d_ia,
                                HYPRE_Int    *d_ja,
                                T            *d_a,
                                T            *d_x,
                                T             beta,
                                T            *d_y,
                                HYPRE_Int    *d_yind)
{
   const HYPRE_Int  grid_ngroups  = gridDim.x * (HYPRE_SPMV_BLOCKDIM / K);
   HYPRE_Int        grid_group_id = (blockIdx.x * HYPRE_SPMV_BLOCKDIM + threadIdx.x) / K;
   const HYPRE_Int  group_lane    = threadIdx.x & (K - 1);
   const HYPRE_Int  warp_lane     = threadIdx.x & (HYPRE_WARP_SIZE - 1);
   const HYPRE_Int  warp_group_id = warp_lane / K;
   const HYPRE_Int  warp_ngroups  = HYPRE_WARP_SIZE / K;

   for (; __any_sync(HYPRE_WARP_FULL_MASK, grid_group_id < nrows); grid_group_id += grid_ngroups)
   {
#if 0
      HYPRE_Int p = 0, q = 0;
      if (grid_group_id < nrows && group_lane < 2)
      {
         p = read_only_load(&d_ia[grid_group_id + group_lane]);
      }
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1, K);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0, K);
#else
      const HYPRE_Int s = grid_group_id - warp_group_id + warp_lane;
      HYPRE_Int p = 0, q = 0;
      if (s <= nrows && warp_lane <= warp_ngroups)
      {
         p = read_only_load(&d_ia[s]);
      }
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, warp_group_id + 1);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, warp_group_id);
#endif

      T sum[NV] = {0.0};
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
      for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K)
      {
         if (p < q)
         {
            HYPRE_SPMV_ADD_SUM(p)
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
         HYPRE_Int row = d_yind ? read_only_load(&d_yind[grid_group_id]) : grid_group_id;
         if (beta)
         {
#pragma unroll
            for (HYPRE_Int i = 0; i < NV; i++)
            {
               d_y[row + i * nrows] = alpha * sum[i] + beta * d_y[row + i * nrows];
            }
         }
         else
         {
#pragma unroll
            for (HYPRE_Int i = 0; i < NV; i++)
            {
               d_y[row + i * nrows] = alpha * sum[i];
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
 *--------------------------------------------------------------------------*/

template <HYPRE_Int F>
HYPRE_Int
hypreDevice_CSRMatrixMatvec( HYPRE_Int      num_vectors,
                             HYPRE_Int      nrows,
                             HYPRE_Int      ncols,
                             HYPRE_Int      num_nonzeros,
                             HYPRE_Complex  alpha,
                             HYPRE_Int     *d_ia,
                             HYPRE_Int     *d_ja,
                             HYPRE_Complex *d_a,
                             HYPRE_Complex *d_x,
                             HYPRE_Complex  beta,
                             HYPRE_Complex *d_y,
                             HYPRE_Int     *d_yind )
{
   const HYPRE_Int avg_rownnz = (num_nonzeros + nrows - 1) / nrows;
   const dim3 bDim(HYPRE_SPMV_BLOCKDIM);

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
 * Supported cases:
 *   1) ind != NULL, y(ind) = alpha*op(B)*x + beta*y(ind)
 *      ind == NULL, y      = alpha*op(B)*x + beta*y
 *      y_ind has size equal to the number of rows of op(B)
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
                           HYPRE_Int       *y_ind,
                           HYPRE_Int        fill )
{
   /* Input data variables */
   HYPRE_Int        nrows        = trans ? hypre_CSRMatrixNumCols(B) : hypre_CSRMatrixNumRows(B);
   HYPRE_Int        ncols        = trans ? hypre_CSRMatrixNumRows(B) : hypre_CSRMatrixNumCols(B);
   HYPRE_Int        num_nonzeros = hypre_CSRMatrixNumNonzeros(B);
   HYPRE_Int        num_vectors  = hypre_VectorNumVectors(x);
   HYPRE_Complex   *d_x          = hypre_VectorData(x);
   HYPRE_Complex   *d_y          = hypre_VectorData(y);

   /* Matrix A variables */
   hypre_CSRMatrix *A = NULL;
   HYPRE_Int       *d_ia;
   HYPRE_Int       *d_ja;
   HYPRE_Complex   *d_a;

   /* Trivial case when alpha*op(B)*x = 0 */
   if (num_nonzeros <= 0 || alpha == 0.0)
   {
      if (y_ind)
      {
         HYPRE_THRUST_CALL( transform,
                            thrust::make_permutation_iterator(d_y, y_ind),
                            thrust::make_permutation_iterator(d_y, y_ind) + nrows,
                            thrust::make_permutation_iterator(d_y, y_ind),
                            beta * _1 );
      }
      else
      {
         hypre_SeqVectorScale(beta, y);
      }

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
   hypre_assert(nrows == hypre_CSRMatrixNumRows(A));
   hypre_assert(nrows > 0);

   /* Set pointers */
   d_ia = hypre_CSRMatrixI(A);
   d_ja = hypre_CSRMatrixJ(A);
   d_a  = hypre_CSRMatrixData(A);

   /* Choose matrix fill mode */
   switch (fill)
   {
      case -2:
         /* Strict lower matrix */
         hypreDevice_CSRMatrixMatvec<-2>(num_vectors, nrows, ncols, num_nonzeros, alpha,
                                         d_ia, d_ja, d_a, d_x, beta, d_y, y_ind);
         break;

      case -1:
         /* Lower matrix */
         hypreDevice_CSRMatrixMatvec<-1>(num_vectors, nrows, ncols, num_nonzeros, alpha,
                                         d_ia, d_ja, d_a, d_x, beta, d_y, y_ind);
         break;

      case 0:
         /* Whole matrix */
         hypreDevice_CSRMatrixMatvec<0>(num_vectors, nrows, ncols, num_nonzeros, alpha,
                                        d_ia, d_ja, d_a, d_x, beta, d_y, y_ind);
         break;

      case 1:
         /* Upper matrix */
         hypreDevice_CSRMatrixMatvec<1>(num_vectors, nrows, ncols, num_nonzeros, alpha,
                                        d_ia, d_ja, d_a, d_x, beta, d_y, y_ind);
         break;

      case 2:
         /* Strict upper matrix */
         hypreDevice_CSRMatrixMatvec<2>(num_vectors, nrows, ncols, num_nonzeros, alpha,
                                        d_ia, d_ja, d_a, d_x, beta, d_y, y_ind);
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

#endif /*#if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP) */
