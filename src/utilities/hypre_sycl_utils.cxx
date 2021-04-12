/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <CL/sycl.hpp>
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_SYCL)

sycl::range<1> hypre_GetDefaultSYCLWorkgroupDimension()
{
  // 256 - max work group size for Gen9
  // 512 - max work group size for ATS
  sycl::range<1> wgDim(64);
  return wgDim;
}

sycl::range<1> hypre_GetDefaultSYCLGridDimension(HYPRE_Int n,
                                                 const char *granularity,
                                                 sycl::range<1> wgDim)
{
   HYPRE_Int num_WGs = 0;
   HYPRE_Int num_workitems_per_WG = wgDim[0];

   if (granularity[0] == 't')
   {
      num_WGs = (n + num_workitems_per_WG - 1) / num_workitems_per_WG;
   }
   else if (granularity[0] == 'w')
   {
      HYPRE_Int num_subgroups_per_block = num_workitems_per_WG >> HYPRE_SUBGROUP_BITSHIFT;
      hypre_assert(num_subgroups_per_block * HYPRE_WARP_SIZE == num_workitems_per_WG);
      num_WGs = (n + num_subgroups_per_block - 1) / num_subgroups_per_block;
   }
   else
   {
      hypre_printf("Error %s %d: Unknown granularity !\n", __FILE__, __LINE__);
      hypre_assert(0);
   }

   sycl::range<1> gDim(num_WGs);

   return gDim;
}

/**
 * Get NNZ of each row in d_row_indices and stored the results in d_rownnz
 * All pointers are device pointers.
 * d_rownnz can be the same as d_row_indices
 */
void
hypreSYCLKernel_GetRowNnz(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int *d_diag_ia, HYPRE_Int *d_offd_ia,
                          HYPRE_Int *d_rownnz, sycl::nd_item<3>& item)
{
   const HYPRE_Int global_thread_id = hypre_Sycl_get_grid_thread_id<1, 1>(item);

   if (global_thread_id < nrows)
   {
      HYPRE_Int i;

      if (d_row_indices)
      {
         i = read_only_load(&d_row_indices[global_thread_id]);
      }
      else
      {
         i = global_thread_id;
      }

      d_rownnz[global_thread_id] = read_only_load(&d_diag_ia[i+1]) - read_only_load(&d_diag_ia[i]) +
                                   read_only_load(&d_offd_ia[i+1]) - read_only_load(&d_offd_ia[i]);
   }
}

/* special case: if d_row_indices == nullptr, it means d_row_indices=[0,1,...,nrows-1] */
HYPRE_Int
hypreDevice_GetRowNnz(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int *d_diag_ia, HYPRE_Int *d_offd_ia,
                      HYPRE_Int *d_rownnz)
{
   const sycl::range<1> wgDim = hypre_GetDefaultSYCLWorkgroupDimension();
   const sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(nrows, "thread", wgDim);

   /* trivial case */
   if (nrows <= 0)
   {
      return hypre_error_flag;
   }

   HYPRE_SYCL_LAUNCH( hypreSYCLKernel_GetRowNnz, gDim, wgDim, nrows, d_row_indices, d_diag_ia, d_offd_ia, d_rownnz );

   return hypre_error_flag;
}

void
hypreSYCLKernel_CopyParCSRRows(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int has_offd,
                               HYPRE_BigInt first_col, HYPRE_Int *d_col_map_offd_A,
                               HYPRE_Int *d_diag_i, HYPRE_Int *d_diag_j, HYPRE_Complex *d_diag_a,
                               HYPRE_Int *d_offd_i, HYPRE_Int *d_offd_j, HYPRE_Complex *d_offd_a,
                               HYPRE_Int *d_ib, HYPRE_BigInt *d_jb, HYPRE_Complex *d_ab,
                               sycl::nd_item<3>& item)
{
   const HYPRE_Int global_warp_id = hypre_Sycl_get_grid_warp_id<1, 1>(item);

   if (global_warp_id >= nrows)
   {
      return;
   }

   /* lane id inside the warp */
   const HYPRE_Int lane_id = hypre_Sycl_get_lane_id<1>(item);
   HYPRE_Int i, j, k, p, row, istart, iend, bstart;

   /* diag part */
   if (lane_id < 2)
   {
      /* row index to work on */
      if (d_row_indices)
      {
         row = read_only_load(d_row_indices + global_warp_id);
      }
      else
      {
         row = global_warp_id;
      }
      /* start/end position of the row */
      j = read_only_load(d_diag_i + row + lane_id);
      /* start position of b */
      k = d_ib ? read_only_load(d_ib + global_warp_id) : 0;
   }

   istart = item.get_sub_group().shuffle(j, 0);
   iend = item.get_sub_group().shuffle(j, 1);
   bstart = item.get_sub_group().shuffle(k, 0);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += HYPRE_WARP_SIZE)
   {
      d_jb[p+i] = read_only_load(d_diag_j + i) + first_col;
      if (d_ab)
      {
         d_ab[p+i] = read_only_load(d_diag_a + i);
      }
   }

   if (!has_offd)
   {
      return;
   }

   /* offd part */
   if (lane_id < 2)
   {
      j = read_only_load(d_offd_i + row + lane_id);
   }
   bstart += iend - istart;

   istart = item.get_sub_group().shuffle(j, 0);
   iend = item.get_sub_group().shuffle(j, 1);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += HYPRE_WARP_SIZE)
   {
      if (d_col_map_offd_A)
      {
         d_jb[p+i] = d_col_map_offd_A[read_only_load(d_offd_j + i)];
      }
      else
      {
         d_jb[p+i] = -1 - read_only_load(d_offd_j + i);
      }

      if (d_ab)
      {
         d_ab[p+i] = read_only_load(d_offd_a + i);
      }
   }
}

/* B = A(row_indices, :) */
/* Note: d_ib is an input vector that contains row ptrs,
 *       i.e., start positions where to put the rows in d_jb and d_ab.
 *       The col indices in B are global indices, i.e., BigJ
 *       of length (nrows + 1) or nrow (without the last entry, nnz) */
/* Special cases:
 *    if d_row_indices == nullptr, it means d_row_indices=[0,1,...,nrows-1]
 *    If col_map_offd_A == nullptr, use (-1 - d_offd_j) as column id
 *    If nrows == 1 and d_ib == nullptr, it means d_ib[0] = 0 */
HYPRE_Int
hypreDevice_CopyParCSRRows(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int job, HYPRE_Int has_offd,
                           HYPRE_BigInt first_col, HYPRE_BigInt *d_col_map_offd_A,
                           HYPRE_Int *d_diag_i, HYPRE_Int *d_diag_j, HYPRE_Complex *d_diag_a,
                           HYPRE_Int *d_offd_i, HYPRE_Int *d_offd_j, HYPRE_Complex *d_offd_a,
                           HYPRE_Int *d_ib, HYPRE_BigInt *d_jb, HYPRE_Complex *d_ab)
{
   /* trivial case */
   if (nrows <= 0)
   {
      return hypre_error_flag;
   }

   hypre_assert(!(nrows > 1 && d_ib == nullptr));

   const sycl::range<1> wgDim = hypre_GetDefaultSYCLWorkgroupDimension();
   const sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(nrows, "warp", wgDim);

   /*
   if (job == 2)
   {
   }
   */

   HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_CopyParCSRRows, gDim, wgDim,
                      nrows, d_row_indices, has_offd, first_col, d_col_map_offd_A,
                      d_diag_i, d_diag_j, d_diag_a,
                      d_offd_i, d_offd_j, d_offd_a,
                      d_ib, d_jb, d_ab );

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_IntegerReduceSum(HYPRE_Int n, HYPRE_Int *d_i)
{
   return HYPRE_ONEDPL_CALL(std::reduce, d_i, d_i + n);
}

HYPRE_Int
hypreDevice_IntegerInclusiveScan(HYPRE_Int n, HYPRE_Int *d_i)
{
   HYPRE_ONEDPL_CALL(std::inclusive_scan, d_i, d_i + n, d_i);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_IntegerExclusiveScan(HYPRE_Int n, HYPRE_Int *d_i)
{
   HYPRE_ONEDPL_CALL(std::exclusive_scan, d_i, d_i + n, d_i);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_BigIntFilln(HYPRE_BigInt *d_x, size_t n, HYPRE_BigInt v)
{
   HYPRE_ONEDPL_CALL( std::fill_n, d_x, n, v);

   return hypre_error_flag;
}

struct hypre_empty_row_functor
{
   // This is needed for clang
   typedef bool result_type;

   bool operator()(const std::tuple<HYPRE_Int, HYPRE_Int> &t) const
   {
      const HYPRE_Int a = std::get<0>(t);
      const HYPRE_Int b = std::get<1>(t);

      return a != b;
   }
};

HYPRE_Int*
hypreDevice_CsrRowPtrsToIndices(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr)
{
   /* trivial case */
   if (nrows <= 0 || nnz <= 0)
   {
      return nullptr;
   }

   HYPRE_Int *d_row_ind = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnz, d_row_ptr, d_row_ind);

   return d_row_ind;
}

HYPRE_Int
hypreDevice_CsrRowPtrsToIndices_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr, HYPRE_Int *d_row_ind)
{
  /* trivial case */
  if (nrows <= 0 || nnz <= 0)
  {
    return hypre_error_flag;
  }

  HYPRE_ONEDPL_CALL( std::fill, d_row_ind, d_row_ind + nnz, 0 );

  // todo: check if this is correct for scatter_if
  d_row_ind = HYPRE_ONEDPL_CALL( dpct::copy_if,
                                 oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                                 oneapi::dpl::counting_iterator<HYPRE_Int>(nrows),
                                 d_row_ptr,
                                 oneapi::dpl::make_transform_iterator(oneapi::dpl::make_zip_iterator(d_row_ptr, d_row_ptr + 1), hypre_empty_row_functor()) );

   // HYPRE_THRUST_CALL( scatter_if,
   //                    thrust::counting_iterator<HYPRE_Int>(0),
   //                    thrust::counting_iterator<HYPRE_Int>(nrows),
   //                    d_row_ptr,
   //                    thrust::make_transform_iterator( thrust::make_zip_iterator(thrust::make_tuple(d_row_ptr, d_row_ptr+1)), hypre_empty_row_functor() ),
   //                 d_row_ind );

  HYPRE_ONEDPL_CALL( std::inclusive_scan, d_row_ind, d_row_ind + nnz, d_row_ind,
                     oneapi::dpl::maximum<HYPRE_Int>() );

  return hypre_error_flag;
}

template <typename T>
HYPRE_Int
hypreDevice_CsrRowPtrsToIndicesWithRowNum(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr, T *d_row_num, T *d_row_ind)
{
   /* trivial case */
   if (nrows <= 0)
   {
      return hypre_error_flag;
   }

   HYPRE_Int *map = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnz, d_row_ptr, map);

   HYPRE_ONEDPL_CALL(gather, map, map + nnz, d_row_num, d_row_ind);

   hypre_TFree(map, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr, HYPRE_Int *d_row_num, HYPRE_Int *d_row_ind);
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
template HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr, HYPRE_BigInt *d_row_num, HYPRE_BigInt *d_row_ind);
#endif

HYPRE_Int*
hypreDevice_CsrRowIndicesToPtrs(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind)
{
   HYPRE_Int *d_row_ptr = hypre_TAlloc(HYPRE_Int, nrows+1, HYPRE_MEMORY_DEVICE);

   //NOTE: dpl::lower_bound uses std::lower_bound(C++20)
   HYPRE_ONEDPL_CALL(oneapi::dpl::lower_bound,
		     d_row_ind, d_row_ind + nnz,
                     oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                     oneapi::dpl::counting_iterator<HYPRE_Int>(nrows + 1),
                     d_row_ptr);

   return d_row_ptr;
}

HYPRE_Int
hypreDevice_CsrRowIndicesToPtrs_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind, HYPRE_Int *d_row_ptr)
{
   HYPRE_ONEDPL_CALL(oneapi::dpl::lower_bound,
		     d_row_ind, d_row_ind + nnz,
                     oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                     oneapi::dpl::counting_iterator<HYPRE_Int>(nrows + 1),
                     d_row_ptr);

   return hypre_error_flag;
}

void
hypreSYCLKernel_ScatterAddTrivial(HYPRE_Int n, HYPRE_Real *x, HYPRE_Int *map, HYPRE_Real *y)
{
   for (HYPRE_Int i = 0; i < n; i++)
   {
      x[map[i]] += y[i];
   }
}

/* x[map[i]] += y[i], same index cannot appear more than once in map */
void
hypreSYCLKernel_ScatterAdd(HYPRE_Int n, HYPRE_Real *x, HYPRE_Int *map, HYPRE_Real *y,
                           sycl::nd_item<3>& item)
{
   HYPRE_Int global_thread_id = hypre_Sycl_get_grid_thread_id<1, 1>(item);

   if (global_thread_id < n)
   {
      x[map[global_thread_id]] += y[global_thread_id];
   }
}

/* Generalized Scatter-and-Add
 * for i = 0 : ny-1, x[map[i]] += y[i];
 * Note: An index is allowed to appear more than once in map
 *       Content in y will be destroyed
 *       When work != nullptr, work is at least of size [2*sizeof(HYPRE_Int)+sizeof(HYPRE_Complex)]*ny
 */
HYPRE_Int
hypreDevice_GenScatterAdd(HYPRE_Real *x, HYPRE_Int ny, HYPRE_Int *map, HYPRE_Real *y, char *work)
{
   if (ny <= 0)
   {
      return hypre_error_flag;
   }

   if (ny <= 2)
   {
      /* trivial cases, n = 1, 2 */
      sycl::range<3> wgDim = sycl::range<3>(1, 1, 1);
      sycl::range<3> gDim = sycl::range<3>(1, 1, 1);
      HYPRE_SYCL_LAUNCH( hypreSYCLKernel_ScatterAddTrivial, wgDim, gDim, ny, x, map, y );

      // todo: launch SYCL singletask
   }
   else
   {
      /* general cases */
      HYPRE_Int *map2, *reduced_map, reduced_n;
      HYPRE_Real *reduced_y;

      if (work)
      {
         map2 = (HYPRE_Int *) work;
         reduced_map = map2 + ny;
         reduced_y = (HYPRE_Real *) (reduced_map + ny);
      }
      else
      {
         map2        = hypre_TAlloc(HYPRE_Int,  ny, HYPRE_MEMORY_DEVICE);
         reduced_map = hypre_TAlloc(HYPRE_Int,  ny, HYPRE_MEMORY_DEVICE);
         reduced_y   = hypre_TAlloc(HYPRE_Real, ny, HYPRE_MEMORY_DEVICE);
      }

      hypre_TMemcpy(map2, map, HYPRE_Int, ny, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

      // Next two lines for: HYPRE_THRUST_CALL(sort_by_key, map2, map2 + ny, y);
      auto zipped_begin = oneapi::dpl::make_zip_iterator(map2, y);
      HYPRE_ONEDPL_CALL(std::sort, zipped_begin, zipped_begin + ny,
                        [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });

      std::pair<HYPRE_Int*, HYPRE_Real*> new_end = HYPRE_ONEDPL_CALL( oneapi::dpl::reduce_by_segment,
                                                                      map2,
                                                                      map2 + ny,
                                                                      y,
                                                                      reduced_map,
                                                                      reduced_y );

      reduced_n = new_end.first - reduced_map;

      hypre_assert(reduced_n == new_end.second - reduced_y);

      sycl::range<1> wgDim = hypre_GetDefaultSYCLWorkgroupDimension();
      sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(reduced_n, "thread", wgDim);

      HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_ScatterAdd, gDim, wgDim,
                            reduced_n, x, reduced_map, reduced_y );

      if (!work)
      {
         hypre_TFree(map2, HYPRE_MEMORY_DEVICE);
         hypre_TFree(reduced_map, HYPRE_MEMORY_DEVICE);
         hypre_TFree(reduced_y, HYPRE_MEMORY_DEVICE);
      }
   }

   return hypre_error_flag;
}

/* x[map[i]] = v */
template <typename T>
void
hypreSYCLKernel_ScatterConstant(T *x, HYPRE_Int n, HYPRE_Int *map, T v,
                                sycl::nd_item<3>& item)
{
   HYPRE_Int global_thread_id = hypre_Sycl_get_grid_thread_id<1, 1>(item);

   if (global_thread_id < n)
   {
      x[map[global_thread_id]] = v;
   }
}

/* x[map[i]] = v
 * n is length of map
 * TODO: thrust? */
template <typename T>
HYPRE_Int
hypreDevice_ScatterConstant(T *x, HYPRE_Int n, HYPRE_Int *map, T v)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   sycl::range<1> wgDim = hypre_GetDefaultSYCLWorkgroupDimension();
   sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(n, "thread", wgDim);

   HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_ScatterConstant, gDim, wgDim, x, n, map, v );

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_ScatterConstant(HYPRE_Int     *x, HYPRE_Int n, HYPRE_Int *map, HYPRE_Int     v);
template HYPRE_Int hypreDevice_ScatterConstant(HYPRE_Complex *x, HYPRE_Int n, HYPRE_Int *map, HYPRE_Complex v);

void
hypreSYCLKernel_IVAXPY(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x, HYPRE_Complex *y,
                       sycl::nd_item<3>& item)
{
   HYPRE_Int i = hypre_Sycl_get_grid_thread_id<1, 1>(item);

   if (i < n)
   {
      y[i] += x[i] / a[i];
   }
}

/* Inverse Vector AXPY: y[i] = x[i] / a[i] + y[i] */
HYPRE_Int
hypreDevice_IVAXPY(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x, HYPRE_Complex *y)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   sycl::range<1> wgDim = hypre_GetDefaultSYCLWorkgroupDimension();
   sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(n, "thread", wgDim);

   HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_IVAXPY, gDim, wgDim, n, a, x, y );

   return hypre_error_flag;
}

void
hypreSYCLKernel_MaskedIVAXPY(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x, HYPRE_Complex *y, HYPRE_Int *mask,
                             sycl::nd_item<3>& item)
{
   HYPRE_Int i = hypre_Sycl_get_grid_thread_id<1, 1>(item);

   if (i < n)
   {
      y[mask[i]] += x[mask[i]] / a[mask[i]];
   }
}

/* Inverse Vector AXPY: y[i] = x[i] / a[i] + y[i] */
HYPRE_Int
hypreDevice_MaskedIVAXPY(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x, HYPRE_Complex *y, HYPRE_Int *mask)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   sycl::range<1> wgDim = hypre_GetDefaultSYCLWorkgroupDimension();
   sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(n, "thread", wgDim);

   HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_MaskedIVAXPY, gDim, wgDim, n, a, x, y, mask );

   return hypre_error_flag;
}

void
hypreSYCLKernel_DiagScaleVector(HYPRE_Int n, HYPRE_Int *A_i, HYPRE_Complex *A_data, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y,
                                sycl::nd_item<3>& item)
{
   HYPRE_Int i = hypre_Sycl_get_grid_thread_id<1, 1>(item);

   if (i < n)
   {
      if (beta != 0.0)
      {
         y[i] = x[i] / A_data[A_i[i]] + beta * y[i];
      }
      else
      {
         y[i] = x[i] / A_data[A_i[i]];
      }
   }
}

/* y = diag(A) \ x + beta y
 * Note: Assume A_i[i] points to the ith diagonal entry of A */
HYPRE_Int
hypreDevice_DiagScaleVector(HYPRE_Int n, HYPRE_Int *A_i, HYPRE_Complex *A_data, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   sycl::range<1> wgDim = hypre_GetDefaultSYCLWorkgroupDimension();
   sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(n, "thread", wgDim);

   HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_DiagScaleVector, gDim, wgDim, n, A_i, A_data, x, beta, y );

   return hypre_error_flag;
}

void
hypreSYCLKernel_BigToSmallCopy(      HYPRE_Int*    __restrict__ tgt,
                               const HYPRE_BigInt* __restrict__ src,
                                     HYPRE_Int                  size,
                                     sycl::nd_item<3>& item)
{
   HYPRE_Int i = hypre_Sycl_get_grid_thread_id<1, 1>(item);

   if (i < size)
   {
      tgt[i] = src[i];
   }
}

HYPRE_Int
hypreDevice_BigToSmallCopy(HYPRE_Int *tgt, const HYPRE_BigInt *src, HYPRE_Int size)
{
   sycl::range<1> wgDim = hypre_GetDefaultSYCLWorkgroupDimension();
   sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(size, "thread", wgDim);

   HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_BigToSmallCopy, gDim, wgDim, tgt, src, size);

   return hypre_error_flag;
}


/* https://github.com/OrangeOwlSolutions/Thrust/blob/master/Sort_by_key_with_tuple_key.cu */
/* opt: 0, (a,b) < (a',b') iff a < a' or (a = a' and  b  <  b')  [normal tupe comp]
 *      1, (a,b) < (a',b') iff a < a' or (a = a' and |b| > |b'|) [used in dropping small entries]
 *      2, (a,b) < (a',b') iff a < a' or (a = a' and (b == a or b < b') and b' != a') [used in putting diagonal first]
 */
template <typename T1, typename T2, typename T3>
HYPRE_Int
hypreDevice_StableSortByTupleKey(HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals, HYPRE_Int opt)
{
   auto begin_keys = oneapi::dpl::make_zip_iterator(keys1,     keys2,     vals    );
   auto end_keys   = oneapi::dpl::make_zip_iterator(keys1 + N, keys2 + N, vals + N);

   if (opt == 0)
   {
      HYPRE_ONEDPL_CALL(std::stable_sort, begin_keys, end_keys, std::less<T1, T2>());
   }
   else if (opt == 1)
   {
      HYPRE_ONEDPL_CALL(std::stable_sort, begin_keys, end_keys, [](auto lhs, auto rhs) {
          if (std::get<0>(lhs) < std::get<0>(rhs)) { return true; }
          if (std::get<0>(lhs) > std::get<0>(rhs)) { return false; }
          return hypre_abs(std::get<1>(lhs)) > hypre_abs(std::get<1>(rhs)); } );
   }
   else if (opt == 2)
   {
      HYPRE_ONEDPL_CALL(std::stable_sort, begin_keys, end_keys, [](auto lhs, auto rhs) {
	  if (std::get<0>(lhs) < std::get<0>(rhs)) { return true; }
	  if (std::get<0>(lhs) > std::get<0>(rhs)) { return false; }
	  if (std::get<0>(rhs) == std::get<1>(rhs)) { return false; }
	  return std::get<0>(lhs) == std::get<1>(lhs) || std::get<1>(lhs) < std::get<1>(rhs); } );
   }

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N, HYPRE_Int *keys1, HYPRE_Int  *keys2, HYPRE_Int *vals, HYPRE_Int opt);
template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N, HYPRE_Int *keys1, HYPRE_Real *keys2, HYPRE_Int *vals, HYPRE_Int opt);
template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N, HYPRE_BigInt *keys1, HYPRE_BigInt *keys2, HYPRE_Complex *vals, HYPRE_Int opt);

/* opt:
 *      0, (a,b) < (a',b') iff a < a' or (a = a' and  b  <  b')                       [normal tupe comp]
 *      2, (a,b) < (a',b') iff a < a' or (a = a' and (b == a or b < b') and b' != a') [used in assembly to put diagonal first]
 */
template <typename T1, typename T2, typename T3, typename T4>
HYPRE_Int
hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals1, T4 *vals2, HYPRE_Int opt)
{
   auto begin_keys = oneapi::dpl::make_zip_iterator(keys1,     keys2,     vals1,     vals2    );
   auto end_keys   = oneapi::dpl::make_zip_iterator(keys1 + N, keys2 + N, vals1 + N, vals2 + N);

   if (opt == 0)
   {
      HYPRE_ONEDPL_CALL(std::stable_sort, begin_keys, end_keys, std::less<T1, T2>());
   }
   else if (opt == 2)
   {
      HYPRE_ONEDPL_CALL(std::stable_sort, begin_keys, end_keys, [](auto lhs, auto rhs) {
          if (std::get<0>(lhs) < std::get<0>(rhs)) { return true; }
          if (std::get<0>(lhs) > std::get<0>(rhs)) { return false; }
          if (std::get<0>(rhs) == std::get<1>(rhs)) { return false; }
          return std::get<0>(lhs) == std::get<1>(lhs) || std::get<1>(lhs) < std::get<1>(rhs); } );
   }

   return hypre_error_flag;
}

#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
template HYPRE_Int hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, HYPRE_BigInt *keys1, HYPRE_BigInt *keys2, char *vals1, HYPRE_Complex *vals2, HYPRE_Int opt);
#endif
template HYPRE_Int hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, HYPRE_Int *keys1, HYPRE_Int *keys2, char *vals1, HYPRE_Complex *vals2, HYPRE_Int opt);

template <typename T1, typename T2, typename T3>
HYPRE_Int
hypreDevice_ReduceByTupleKey(HYPRE_Int N, T1 *keys1_in,  T2 *keys2_in,  T3 *vals_in,
                                          T1 *keys1_out, T2 *keys2_out, T3 *vals_out)
{
   auto begin_keys_in  = oneapi::dpl::make_zip_iterator(keys1_in,     keys2_in    );
   auto end_keys_in    = oneapi::dpl::make_zip_iterator(keys1_in + N, keys2_in + N);
   auto begin_keys_out = oneapi::dpl::make_zip_iterator(keys1_out,    keys2_out   );
   std::equal_to<T1, T2> binary_pred;
   std::plus<T3> binary_op;

   auto new_end = HYPRE_ONEDPL_CALL(oneapi::dpl::reduce_by_segment,
                                    begin_keys_in, end_keys_in, vals_in, begin_keys_out, vals_out,
                                    binary_pred, binary_op);

   return new_end.second - vals_out;
}

template HYPRE_Int hypreDevice_ReduceByTupleKey(HYPRE_Int N, HYPRE_Int *keys1_in, HYPRE_Int *keys2_in, HYPRE_Complex *vals_in, HYPRE_Int *keys1_out, HYPRE_Int *keys2_out, HYPRE_Complex *vals_out);

#endif // #if defined(HYPRE_USING_SYCL)

//todo: HYPRE_USING_CUSPARSE
#if defined(HYPRE_USING_CUSPARSE)
/*
 * @brief Determines the associated SyclDataType for the HYPRE_Complex typedef
 * @return Returns cuda data type corresponding with HYPRE_Complex
 *
 * @todo Should be known compile time
 * @todo Support more sizes
 * @todo Support complex
 * @warning Only works for Single and Double precision
 * @note Perhaps some typedefs should be added where HYPRE_Complex is typedef'd
 */
int hypre_HYPREComplexToSyclDataType()
{
   /*
   if (sizeof(char)*CHAR_BIT != 8)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ERROR:  Unsupported char size");
      hypre_assert(false);
   }
   */
#if defined(HYPRE_COMPLEX)
   return SYCL_C_64F;
#else
#if defined(HYPRE_SINGLE)
   hypre_assert(sizeof(HYPRE_Complex) == 4);
   return SYCL_R_32F;
#elif defined(HYPRE_LONG_DOUBLE)
#error "Long Double is not supported on GPUs"
#else
   hypre_assert(sizeof(HYPRE_Complex) == 8);
   return 1;
#endif
#endif // #if defined(HYPRE_COMPLEX)
}

/*
 * @brief Determines the associated cusparseIndexType_t for the HYPRE_Int typedef
 */
cusparseIndexType_t
hypre_HYPREIntToCusparseIndexType()
{
   /*
   if(sizeof(char)*CHAR_BIT!=8)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ERROR:  Unsupported char size");
      hypre_assert(false);
   }
   */

#if defined(HYPRE_BIGINT)
   hypre_assert(sizeof(HYPRE_Int) == 8);
   return CUSPARSE_INDEX_64I;
#else
   hypre_assert(sizeof(HYPRE_Int) == 4);
   return CUSPARSE_INDEX_32I;
#endif
}
#endif // #if defined(HYPRE_USING_CUSPARSE)

#if defined(HYPRE_USING_SYCL)

sycl::queue*
hypre_SyclDataSyclQueue(hypre_SyclData *data, HYPRE_Int i)
{
  auto sycl_asynchandler = [] (sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const& ex) {
            std::cout << "Caught asynchronous SYCL exception:" << std::endl
            << ex.what() << ", OpenCL code: " << ex.get_cl_code() << std::endl;
        }
    }
  };

  sycl::queue *stream = nullptr;
  sycl::device syclDev = data->sycl_device;
  sycl::context syclctxt = sycl::context(syclDev, sycl_asynchandler);

#if defined(HYPRE_USING_SYCL_STREAMS)
  if (i >= HYPRE_MAX_NUM_STREAMS)
  {
    /* there is no default queue, so raise an error! */
    hypre_printf("Error %s %d: SYCL queues %d exceeds the max number of queues %d\n",
                 __FILE__, __LINE__, i, HYPRE_MAX_NUM_STREAMS);
    assert(0); exit(1);
  }

  if (data->sycl_streams[i])
  {
    return data->sycl_streams[i];
  }

  stream = new sycl::queue(syclctxt, syclDev, sycl::property_list{sycl::property::queue::in_order{}});

  data->sycl_streams[i] = stream;
#endif

  return stream;
}

sycl::queue *hypre_SyclDataSyclComputeQueue(hypre_SyclData *data)
{
  return hypre_SyclDataSyclQueue(data, hypre_SyclDataSyclComputeQueueNum(data));
}

oneapi::mkl::rng::philox4x32x10*
hypre_SyclDataonemklrngGenerator(hypre_SyclData *data) try {
   if (data->onemklrng_generator)
   {
      return data->onemklrng_generator;
   }

   sycl::queue* q = hypre_SyclDataSyclComputeQueue(data);
   data->onemklrng_generator = new oneapi::mkl::rng::philox4x32x10(*q, 1234ULL);

   return data->onemklrng_generator;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#endif

hypre_SyclData*
hypre_SyclDataCreate()
{
   hypre_SyclData *data = hypre_CTAlloc(hypre_SyclData, 1, HYPRE_MEMORY_HOST);

   hypre_SyclDataSyclDevice(data)           = sycl::device(sycl::gpu_selector{});
   hypre_SyclDataSyclComputeQueueNum(data)  = 0;

   /* SpGeMM */
#ifdef HYPRE_USING_CUSPARSE
   hypre_SyclDataSpgemmUseCusparse(data) = 1;
#else
   hypre_SyclDataSpgemmUseCusparse(data) = 0;
#endif
   hypre_SyclDataSpgemmNumPasses(data) = 3;
   /* 1: naive overestimate, 2: naive underestimate, 3: Cohen's algorithm */
   hypre_SyclDataSpgemmRownnzEstimateMethod(data) = 3;
   hypre_SyclDataSpgemmRownnzEstimateNsamples(data) = 32;
   hypre_SyclDataSpgemmRownnzEstimateMultFactor(data) = 1.5;
   hypre_SyclDataSpgemmHashType(data) = 'L';

   return data;
}

void hypre_SyclDataDestroy(hypre_SyclData *data) try {
   if (!data)
   {
      return;
   }

   hypre_TFree(hypre_SyclDataSyclReduceBuffer(data),     HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_SyclDataStructCommRecvBuffer(data), HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_SyclDataStructCommSendBuffer(data), HYPRE_MEMORY_DEVICE);

   if (data->onemklrng_generator)
   {
     delete data->onemklrng_generator;
   }

   for (HYPRE_Int i = 0; i < HYPRE_MAX_NUM_STREAMS; i++)
   {
      if (data->sycl_streams[i])
      {
         /*
         DPCT1003:45: Migrated API does not return error code. (*, 0) is
         inserted. You may need to rewrite this code.
         */
         HYPRE_SYCL_CALL(
             (dpct::get_current_device().destroy_queue(data->sycl_streams[i]),
              0));
      }
   }

   hypre_TFree(data, HYPRE_MEMORY_HOST);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

HYPRE_Int hypre_SyncSyclDevice(hypre_Handle *hypre_handle) try {
#if defined(HYPRE_USING_SYCL)
   HYPRE_SYCL_CALL( dpct::get_current_device().queues_wait_and_throw() );
#endif
   return hypre_error_flag;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/* synchronize the Hypre compute stream
 * action: 0: set sync stream to false
 *         1: set sync stream to true
 *         2: restore sync stream to default
 *         3: return the current value of sycl_compute_queue_sync
 *         4: sync stream based on sycl_compute_queue_sync
 */
HYPRE_Int
hypre_SyncSyclComputeQueue_core(HYPRE_Int action, hypre_Handle *hypre_handle,
                                 HYPRE_Int *sycl_compute_queue_sync_ptr) try {
   /* with UVM the default is to sync at kernel completions, since host is also able to
    * touch GPU memory */
#if defined(HYPRE_USING_UNIFIED_MEMORY)
   static const HYPRE_Int sycl_compute_queue_sync_default = 1;
#else
   static const HYPRE_Int sycl_compute_queue_sync_default = 0;
#endif

   /* this controls if synchronize the stream after computations */
   static HYPRE_Int sycl_compute_queue_sync = sycl_compute_queue_sync_default;

   switch (action)
   {
      case 0:
         sycl_compute_queue_sync = 0;
         break;
      case 1:
         sycl_compute_queue_sync = 1;
         break;
      case 2:
         sycl_compute_queue_sync = sycl_compute_queue_sync_default;
         break;
      case 3:
         *sycl_compute_queue_sync_ptr = sycl_compute_queue_sync;
         break;
      case 4:
#if defined(HYPRE_USING_SYCL)
         if (sycl_compute_queue_sync)
         {
           HYPRE_SYCL_CALL( hypre_HandleSyclComputeQueue(hypre_handle)->wait() );
         }
#endif
         break;
      default:
         hypre_printf("hypre_SyncSyclComputeQueue_core invalid action\n");
         hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

HYPRE_Int
hypre_SetSyncSyclCompute(HYPRE_Int action)
{
   /* convert to 1/0 */
   action = action != 0;
   hypre_SyncSyclComputeQueue_core(action, nullptr, nullptr);

   return hypre_error_flag;
}

HYPRE_Int
hypre_RestoreSyncSyclCompute()
{
   hypre_SyncSyclComputeQueue_core(2, nullptr, nullptr);

   return hypre_error_flag;
}

HYPRE_Int
hypre_GetSyncSyclCompute(HYPRE_Int *sycl_compute_queue_sync_ptr)
{
   hypre_SyncSyclComputeQueue_core(3, nullptr, sycl_compute_queue_sync_ptr);

   return hypre_error_flag;
}

HYPRE_Int
hypre_SyncSyclComputeQueue(hypre_Handle *hypre_handle)
{
   hypre_SyncSyclComputeQueue_core(4, hypre_handle, nullptr);

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_SYCL)
