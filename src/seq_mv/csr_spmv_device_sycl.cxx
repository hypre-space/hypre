/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

#if defined(HYPRE_USING_SYCL)

#define SPMV_BLOCKDIM 512
#define VERSION 1

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
void
hypre_csr_v_k_shuffle(sycl::nd_item<1>& item,
		      HYPRE_Int     n,
                      T             alpha,
                      HYPRE_Int    *d_ia,
                      HYPRE_Int    *d_ja,
                      T            *d_a,
                      T            *d_x,
                      T             beta,
                      T            *d_y    )
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *               warp-shuffle reduction
    *            (Group of K threads) per row
    *------------------------------------------------------------*/

   sycl::group<1> grp = item.get_group();
   sycl::ext::oneapi::sub_group SG = item.get_sub_group();
   HYPRE_Int sub_group_size = SG.get_local_range().get(0);

   const HYPRE_Int grid_ngroups = item.get_group_range(0) * (SPMV_BLOCKDIM / K);
   HYPRE_Int grid_group_id = (item.get_group(0) * SPMV_BLOCKDIM + item.get_local_id(0)) / K;
   const HYPRE_Int group_lane = item.get_local_id(0) & (K - 1);
   const HYPRE_Int warp_lane = item.get_local_id(0) & (sub_group_size - 1);
   const HYPRE_Int warp_group_id = warp_lane / K;
   const HYPRE_Int warp_ngroups = sub_group_size / K;


   for (; sycl::ext::oneapi::any_of(grp, grid_group_id < n); grid_group_id += grid_ngroups)
   {
#if 0
      HYPRE_Int p = 0, q = 0;
      if (grid_group_id < n && group_lane < 2)
      {
         p = read_only_load(&d_ia[grid_group_id+group_lane]);
      }
      q = SG.shuffle(p, 1, K);
      p = SG.shuffle(p, 0, K);
#else
      const HYPRE_Int s = grid_group_id - warp_group_id + warp_lane;
      HYPRE_Int p = 0, q = 0;
      if (s <= n && warp_lane <= warp_ngroups)
      {
         p = read_only_load(&d_ia[s]);
      }

      q = SG.shuffle(p, warp_group_id + 1);
      p = SG.shuffle(p, warp_group_id);
#endif
      T sum = 0.0;
#if VERSION == 1
#pragma unroll(1)

      for (p += group_lane; sycl::ext::oneapi::any_of(grp, p < q); p += K * 2)
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
#pragma unroll(1)
      for (p += group_lane; sycl::ext::oneapi::any_of(grp, p < q); p += K)
      {
         if (p < q)
         {
            SPMV_ADD_SUM(p)
         }
      }
#else
#pragma unroll(1)
      for (p += group_lane;  p < q; p += K)
      {
         SPMV_ADD_SUM(p)
      }
#endif
      // parallel reduction
#pragma unroll
      for (HYPRE_Int d = K/2; d > 0; d >>= 1)
      {
         sum += SG.shuffle_down(sum, d);
      }
      if (grid_group_id < n && group_lane == 0)
      {
         d_y[grid_group_id] = alpha * sum + beta * d_y[grid_group_id];
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
HYPRE_Int hypreDevice_CSRMatrixMatvec(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Complex alpha,
				      HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
				      HYPRE_Complex *d_x, HYPRE_Complex beta,
				      HYPRE_Complex *d_y)
{
  const HYPRE_Int rownnz = (nnz + nrows - 1) / nrows;
  const sycl::range<1> bDim(SPMV_BLOCKDIM);

  if (rownnz >= 64)
  {
    const HYPRE_Int group_size = 32;
    const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
    const sycl::range<1> gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
    HYPRE_SYCL_1D_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, HYPRE_Int>), gDim, bDim,
                          nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
  }
  else if (rownnz >= 32)
  {
    const HYPRE_Int group_size = 16;
    const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
    const sycl::range<1> gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
    HYPRE_SYCL_1D_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, HYPRE_Int>), gDim, bDim,
                          nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
  }
  else if (rownnz >= 16)
  {
    const HYPRE_Int group_size = 8;
    const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
    const sycl::range<1> gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
    HYPRE_SYCL_1D_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, HYPRE_Int>), gDim, bDim,
                          nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
  }
  else if (rownnz >= 8)
  {
    const HYPRE_Int group_size = 4;
    const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
    const sycl::range<1> gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
    HYPRE_SYCL_1D_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, HYPRE_Int>), gDim, bDim,
                          nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
  }
  else
  {
    const HYPRE_Int group_size = 4;
    const HYPRE_Int num_groups_per_block = SPMV_BLOCKDIM / group_size;
    const sycl::range<1> gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);
    HYPRE_SYCL_1D_LAUNCH( (hypre_csr_v_k_shuffle<F, group_size, HYPRE_Int>), gDim, bDim,
                          nrows, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );
  }

  return hypre_error_flag;
}

HYPRE_Int
hypre_CSRMatrixSpMVDevice(HYPRE_Complex alpha, hypre_CSRMatrix *A,
			  hypre_Vector *x, HYPRE_Complex beta,
			  hypre_Vector *y, HYPRE_Int fill)
{
   HYPRE_Int      nrows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      nnz   = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int     *d_ia  = hypre_CSRMatrixI(A);
   HYPRE_Int     *d_ja  = hypre_CSRMatrixJ(A);
   HYPRE_Complex *d_a = hypre_CSRMatrixData(A);
   HYPRE_Complex *d_x = hypre_VectorData(x);
   HYPRE_Complex *d_y = hypre_VectorData(y);

   if (nnz <= 0 || alpha == 0.0)
   {
      hypre_SeqVectorScale(beta, y);

      return hypre_error_flag;
   }

   hypre_assert(nrows > 0);

   if (fill == 0)
   {
      return hypreDevice_CSRMatrixMatvec<0>(nrows, nnz, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == 1)
   {
      return hypreDevice_CSRMatrixMatvec<1>(nrows, nnz, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == -1)
   {
      return hypreDevice_CSRMatrixMatvec<-1>(nrows, nnz, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == 2)
   {
      return hypreDevice_CSRMatrixMatvec<2>(nrows, nnz, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }
   else if (fill == -2)
   {
      return hypreDevice_CSRMatrixMatvec<-2>(nrows, nnz, alpha, d_ia, d_ja, d_a, d_x, beta, d_y);
   }

   return hypre_error_flag;
}

#endif /*#if defined(HYPRE_USING_SYCL)
