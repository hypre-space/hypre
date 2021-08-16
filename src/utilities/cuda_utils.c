/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/*
 * The architecture identification macro __CUDA_ARCH__ is assigned a three-digit value string xy0
 * (ending in a literal 0) during each nvcc compilation stage 1 that compiles for compute_xy.
 * This macro can be used in the implementation of GPU functions for determining the virtual architecture
 * for which it is currently being compiled. The host code (the non-GPU code) must not depend on it.
 * Note that compute_XX refers to a PTX version and sm_XX refers to a cubin version.
*/
__global__ void
hypreCUDAKernel_CompileFlagSafetyCheck(hypre_int *cuda_arch_compile)
{
#if defined(__CUDA_ARCH__)
   cuda_arch_compile[0] = __CUDA_ARCH__;
#endif
}

/*
 * Assume this function is called inside HYPRE_Init(), at a place where we do not want to
 * activate memory pooling, so we do not use hypre's memory model to Alloc and Free.
 * See commented out code below (and do not delete)
*/
void hypre_CudaCompileFlagCheck()
{
  // This is really only defined for CUDA and not for HIP
#if defined(HYPRE_USING_CUDA)

   HYPRE_Int device = hypre_HandleCudaDevice(hypre_handle());

   struct cudaDeviceProp props;
   cudaGetDeviceProperties(&props, device);
   hypre_int cuda_arch_actual = props.major*100 + props.minor*10;
   hypre_int cuda_arch_compile = -1;
   dim3 gDim(1,1,1), bDim(1,1,1);

   hypre_int *cuda_arch_compile_d = NULL;
   //cuda_arch_compile_d = hypre_TAlloc(hypre_int, 1, HYPRE_MEMORY_DEVICE);
   HYPRE_CUDA_CALL( cudaMalloc(&cuda_arch_compile_d, sizeof(hypre_int)) );
   hypre_TMemcpy(cuda_arch_compile_d, &cuda_arch_compile, hypre_int, 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CompileFlagSafetyCheck, gDim, bDim, cuda_arch_compile_d );
   hypre_TMemcpy(&cuda_arch_compile, cuda_arch_compile_d, hypre_int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   //hypre_TFree(cuda_arch_compile_d, HYPRE_MEMORY_DEVICE);
   HYPRE_CUDA_CALL( cudaFree(cuda_arch_compile_d) );

   /* HYPRE_CUDA_CALL(cudaDeviceSynchronize()); */

   if (-1 == cuda_arch_compile)
   {
      hypre_error_w_msg(1, "hypre error: no proper cuda_arch found");
   }
   else if (cuda_arch_actual != cuda_arch_compile)
   {
      char msg[256];
      hypre_sprintf(msg, "hypre warning: Compile 'arch=compute_' does not match device arch %d", cuda_arch_actual);
      hypre_error_w_msg(1, msg);
      /*
      hypre_printf("%s\n", msg);
      hypre_MPI_Abort(hypre_MPI_COMM_WORLD, -1);
      */
   }

#endif // defined(HYPRE_USING_CUDA)
}

dim3
hypre_GetDefaultCUDABlockDimension()
{
   dim3 bDim(512, 1, 1);

   return bDim;
}

dim3
hypre_GetDefaultCUDAGridDimension( HYPRE_Int n,
                                   const char *granularity,
                                   dim3 bDim )
{
   HYPRE_Int num_blocks = 0;
   HYPRE_Int num_threads_per_block = bDim.x * bDim.y * bDim.z;

   if (granularity[0] == 't')
   {
      num_blocks = (n + num_threads_per_block - 1) / num_threads_per_block;
   }
   else if (granularity[0] == 'w')
   {
      HYPRE_Int num_warps_per_block = num_threads_per_block >> HYPRE_WARP_BITSHIFT;

      hypre_assert(num_warps_per_block * HYPRE_WARP_SIZE == num_threads_per_block);

      num_blocks = (n + num_warps_per_block - 1) / num_warps_per_block;
   }
   else
   {
      hypre_printf("Error %s %d: Unknown granularity !\n", __FILE__, __LINE__);
      hypre_assert(0);
   }

   dim3 gDim(num_blocks, 1, 1);

   return gDim;
}

/**
 * Get NNZ of each row in d_row_indices and stored the results in d_rownnz
 * All pointers are device pointers.
 * d_rownnz can be the same as d_row_indices
 */
__global__ void
hypreCUDAKernel_GetRowNnz(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int *d_diag_ia, HYPRE_Int *d_offd_ia,
                          HYPRE_Int *d_rownnz)
{
   const HYPRE_Int global_thread_id = hypre_cuda_get_grid_thread_id<1,1>();

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

/* special case: if d_row_indices == NULL, it means d_row_indices=[0,1,...,nrows-1] */
HYPRE_Int
hypreDevice_GetRowNnz(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int *d_diag_ia, HYPRE_Int *d_offd_ia,
                      HYPRE_Int *d_rownnz)
{
   const dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   const dim3 gDim = hypre_GetDefaultCUDAGridDimension(nrows, "thread", bDim);

   /* trivial case */
   if (nrows <= 0)
   {
      return hypre_error_flag;
   }

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_GetRowNnz, gDim, bDim, nrows, d_row_indices, d_diag_ia, d_offd_ia, d_rownnz );

   return hypre_error_flag;
}

__global__ void
hypreCUDAKernel_CopyParCSRRows(HYPRE_Int      nrows,
                               HYPRE_Int     *d_row_indices,
                               HYPRE_Int      has_offd,
                               HYPRE_BigInt   first_col,
                               HYPRE_BigInt  *d_col_map_offd_A,
                               HYPRE_Int     *d_diag_i,
                               HYPRE_Int     *d_diag_j,
                               HYPRE_Complex *d_diag_a,
                               HYPRE_Int     *d_offd_i,
                               HYPRE_Int     *d_offd_j,
                               HYPRE_Complex *d_offd_a,
                               HYPRE_Int     *d_ib,
                               HYPRE_BigInt  *d_jb,
                               HYPRE_Complex *d_ab)
{
   const HYPRE_Int global_warp_id = hypre_cuda_get_grid_warp_id<1,1>();

   if (global_warp_id >= nrows)
   {
      return;
   }

   /* lane id inside the warp */
   const HYPRE_Int lane_id = hypre_cuda_get_lane_id<1>();
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
   istart = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
   iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);
   bstart = __shfl_sync(HYPRE_WARP_FULL_MASK, k, 0);

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
   istart = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
   iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);

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
 *    if d_row_indices == NULL, it means d_row_indices=[0,1,...,nrows-1]
 *    If col_map_offd_A == NULL, use (-1 - d_offd_j) as column id
 *    If nrows == 1 and d_ib == NULL, it means d_ib[0] = 0 */
HYPRE_Int
hypreDevice_CopyParCSRRows(HYPRE_Int      nrows,
                           HYPRE_Int     *d_row_indices,
                           HYPRE_Int      job,
                           HYPRE_Int      has_offd,
                           HYPRE_BigInt   first_col,
                           HYPRE_BigInt  *d_col_map_offd_A,
                           HYPRE_Int     *d_diag_i,
                           HYPRE_Int     *d_diag_j,
                           HYPRE_Complex *d_diag_a,
                           HYPRE_Int     *d_offd_i,
                           HYPRE_Int     *d_offd_j,
                           HYPRE_Complex *d_offd_a,
                           HYPRE_Int     *d_ib,
                           HYPRE_BigInt  *d_jb,
                           HYPRE_Complex *d_ab)
{
   /* trivial case */
   if (nrows <= 0)
   {
      return hypre_error_flag;
   }

   hypre_assert(!(nrows > 1 && d_ib == NULL));

   const dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   const dim3 gDim = hypre_GetDefaultCUDAGridDimension(nrows, "warp", bDim);

   /*
   if (job == 2)
   {
   }
   */

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CopyParCSRRows, gDim, bDim,
                      nrows, d_row_indices, has_offd, first_col, d_col_map_offd_A,
                      d_diag_i, d_diag_j, d_diag_a,
                      d_offd_i, d_offd_j, d_offd_a,
                      d_ib, d_jb, d_ab );

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_IntegerReduceSum(HYPRE_Int n, HYPRE_Int *d_i)
{
   return HYPRE_THRUST_CALL(reduce, d_i, d_i + n);
}

HYPRE_Int
hypreDevice_IntegerInclusiveScan(HYPRE_Int n, HYPRE_Int *d_i)
{
   HYPRE_THRUST_CALL(inclusive_scan, d_i, d_i + n, d_i);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_IntegerExclusiveScan(HYPRE_Int n, HYPRE_Int *d_i)
{
   HYPRE_THRUST_CALL(exclusive_scan, d_i, d_i + n, d_i);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_Scalen(HYPRE_Complex *d_x, size_t n, HYPRE_Complex v)
{
   HYPRE_THRUST_CALL( transform, d_x, d_x + n, d_x, v * _1 );

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_Filln(HYPRE_Complex *d_x, size_t n, HYPRE_Complex v)
{
   HYPRE_THRUST_CALL( fill_n, d_x, n, v);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_BigIntFilln(HYPRE_BigInt *d_x, size_t n, HYPRE_BigInt v)
{
   HYPRE_THRUST_CALL( fill_n, d_x, n, v);

   return hypre_error_flag;
}

struct hypre_empty_row_functor
{
   // This is needed for clang
   typedef bool result_type;

   __device__
   bool operator()(const thrust::tuple<HYPRE_Int, HYPRE_Int>& t) const
   {
      const HYPRE_Int a = thrust::get<0>(t);
      const HYPRE_Int b = thrust::get<1>(t);

      return a != b;
   }
};

HYPRE_Int*
hypreDevice_CsrRowPtrsToIndices(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr)
{
   /* trivial case */
   if (nrows <= 0 || nnz <= 0)
   {
      return NULL;
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

   HYPRE_THRUST_CALL( fill, d_row_ind, d_row_ind + nnz, 0 );

   HYPRE_THRUST_CALL( scatter_if,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(nrows),
                      d_row_ptr,
                      thrust::make_transform_iterator( thrust::make_zip_iterator(thrust::make_tuple(d_row_ptr, d_row_ptr+1)),
                                                       hypre_empty_row_functor() ),
                      d_row_ind );

   HYPRE_THRUST_CALL( inclusive_scan, d_row_ind, d_row_ind + nnz, d_row_ind, thrust::maximum<HYPRE_Int>());

   return hypre_error_flag;
}

/* Input: d_row_num, of size nrows, contains the rows indices that can be BigInt or Int
 * Output: d_row_ind */
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

   HYPRE_THRUST_CALL(gather, map, map + nnz, d_row_num, d_row_ind);

   hypre_TFree(map, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr, HYPRE_Int *d_row_num, HYPRE_Int *d_row_ind);
#if defined(HYPRE_MIXEDINT)
template HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr, HYPRE_BigInt *d_row_num, HYPRE_BigInt *d_row_ind);
#endif

HYPRE_Int*
hypreDevice_CsrRowIndicesToPtrs(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind)
{
   HYPRE_Int *d_row_ptr = hypre_TAlloc(HYPRE_Int, nrows+1, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( lower_bound,
                      d_row_ind, d_row_ind + nnz,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(nrows+1),
                      d_row_ptr);

   return d_row_ptr;
}

HYPRE_Int
hypreDevice_CsrRowIndicesToPtrs_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind, HYPRE_Int *d_row_ptr)
{
   HYPRE_THRUST_CALL( lower_bound,
                      d_row_ind, d_row_ind + nnz,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(nrows+1),
                      d_row_ptr);

   return hypre_error_flag;
}

__global__ void
hypreCUDAKernel_ScatterAddTrivial(HYPRE_Int n, HYPRE_Real *x, HYPRE_Int *map, HYPRE_Real *y)
{
   for (HYPRE_Int i = 0; i < n; i++)
   {
      x[map[i]] += y[i];
   }
}

/* x[map[i]] += y[i], same index cannot appear more than once in map */
__global__ void
hypreCUDAKernel_ScatterAdd(HYPRE_Int n, HYPRE_Real *x, HYPRE_Int *map, HYPRE_Real *y)
{
   HYPRE_Int global_thread_id = hypre_cuda_get_grid_thread_id<1,1>();

   if (global_thread_id < n)
   {
      x[map[global_thread_id]] += y[global_thread_id];
   }
}

/* Generalized Scatter-and-Add
 * for i = 0 : ny-1, x[map[i]] += y[i];
 * Note: An index is allowed to appear more than once in map
 *       Content in y will be destroyed
 *       When work != NULL, work is at least of size [2*sizeof(HYPRE_Int)+sizeof(HYPRE_Complex)]*ny
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
      dim3 bDim = 1;
      dim3 gDim = 1;
      HYPRE_CUDA_LAUNCH( hypreCUDAKernel_ScatterAddTrivial, gDim, bDim, ny, x, map, y );
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

      HYPRE_THRUST_CALL(sort_by_key, map2, map2 + ny, y);

      thrust::pair<HYPRE_Int*, HYPRE_Real*> new_end = HYPRE_THRUST_CALL( reduce_by_key,
                                                                         map2,
                                                                         map2 + ny,
                                                                         y,
                                                                         reduced_map,
                                                                         reduced_y );

      reduced_n = new_end.first - reduced_map;

      hypre_assert(reduced_n == new_end.second - reduced_y);

      dim3 bDim = hypre_GetDefaultCUDABlockDimension();
      dim3 gDim = hypre_GetDefaultCUDAGridDimension(reduced_n, "thread", bDim);

      HYPRE_CUDA_LAUNCH( hypreCUDAKernel_ScatterAdd, gDim, bDim,
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
__global__ void
hypreCUDAKernel_ScatterConstant(T *x, HYPRE_Int n, HYPRE_Int *map, T v)
{
   HYPRE_Int global_thread_id = hypre_cuda_get_grid_thread_id<1,1>();

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

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(n, "thread", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_ScatterConstant, gDim, bDim, x, n, map, v );

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_ScatterConstant(HYPRE_Int     *x, HYPRE_Int n, HYPRE_Int *map, HYPRE_Int     v);
template HYPRE_Int hypreDevice_ScatterConstant(HYPRE_Complex *x, HYPRE_Int n, HYPRE_Int *map, HYPRE_Complex v);

__global__ void
hypreCUDAKernel_IVAXPY(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x, HYPRE_Complex *y)
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>();

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

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(n, "thread", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_IVAXPY, gDim, bDim, n, a, x, y );

   return hypre_error_flag;
}

__global__ void
hypreCUDAKernel_IVAXPYMarked(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x, HYPRE_Complex *y, HYPRE_Int *marker, HYPRE_Int marker_val)
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>();

   if (i < n)
   {
      if (marker[i] == marker_val)
      {
         y[i] += x[i] / a[i];
      }         
   }
}

/* Inverse Vector AXPY: y[i] = x[i] / a[i] + y[i] */
HYPRE_Int
hypreDevice_IVAXPYMarked(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x, HYPRE_Complex *y, HYPRE_Int *marker, HYPRE_Int marker_val)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(n, "thread", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_IVAXPYMarked, gDim, bDim, n, a, x, y, marker, marker_val );

   return hypre_error_flag;
}

__global__ void
hypreCUDAKernel_DiagScaleVector(HYPRE_Int n, HYPRE_Int *A_i, HYPRE_Complex *A_data, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y)
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>();

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

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(n, "thread", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_DiagScaleVector, gDim, bDim, n, A_i, A_data, x, beta, y );

   return hypre_error_flag;
}

__global__ void
hypreCUDAKernel_DiagScaleVector2(HYPRE_Int n, HYPRE_Int *A_i, HYPRE_Complex *A_data, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y, HYPRE_Complex *z)
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>();

   if (i < n)
   {
      HYPRE_Complex t = x[i] / A_data[A_i[i]];
      y[i] = t;
      z[i] += beta*t;
   }
}

/* y = diag(A) \ x
 * z = beta * (diag(A) \ x) + z
 * Note: Assume A_i[i] points to the ith diagonal entry of A */
HYPRE_Int
hypreDevice_DiagScaleVector2(HYPRE_Int n, HYPRE_Int *A_i, HYPRE_Complex *A_data, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y, HYPRE_Complex *z)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(n, "thread", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_DiagScaleVector2, gDim, bDim, n, A_i, A_data, x, beta, y, z );

   return hypre_error_flag;
}

__global__ void
hypreCUDAKernel_BigToSmallCopy(      HYPRE_Int*    __restrict__ tgt,
                               const HYPRE_BigInt* __restrict__ src,
                                     HYPRE_Int                  size)
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>();

   if (i < size)
   {
      tgt[i] = src[i];
   }
}

HYPRE_Int
hypreDevice_BigToSmallCopy(HYPRE_Int *tgt, const HYPRE_BigInt *src, HYPRE_Int size)
{
   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(size, "thread", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_BigToSmallCopy, gDim, bDim, tgt, src, size);

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
   auto begin_keys = thrust::make_zip_iterator(thrust::make_tuple(keys1,     keys2));
   auto end_keys   = thrust::make_zip_iterator(thrust::make_tuple(keys1 + N, keys2 + N));

   if (opt == 0)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key, begin_keys, end_keys, vals, thrust::less< thrust::tuple<T1, T2> >());
   }
   else if (opt == 1)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key, begin_keys, end_keys, vals, TupleComp2<T1,T2>());
   }
   else if (opt == 2)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key, begin_keys, end_keys, vals, TupleComp3<T1,T2>());
   }

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N, HYPRE_Int *keys1, HYPRE_Int  *keys2, HYPRE_Int     *vals, HYPRE_Int opt);
template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N, HYPRE_Int *keys1, HYPRE_Real *keys2, HYPRE_Int     *vals, HYPRE_Int opt);
template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N, HYPRE_Int *keys1, HYPRE_Int  *keys2, HYPRE_Complex *vals, HYPRE_Int opt);

/* opt:
 *      0, (a,b) < (a',b') iff a < a' or (a = a' and  b  <  b')                       [normal tupe comp]
 *      2, (a,b) < (a',b') iff a < a' or (a = a' and (b == a or b < b') and b' != a') [used in assembly to put diagonal first]
 */
template <typename T1, typename T2, typename T3, typename T4>
HYPRE_Int
hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals1, T4 *vals2, HYPRE_Int opt)
{
   auto begin_keys = thrust::make_zip_iterator(thrust::make_tuple(keys1,     keys2));
   auto end_keys   = thrust::make_zip_iterator(thrust::make_tuple(keys1 + N, keys2 + N));
   auto begin_vals = thrust::make_zip_iterator(thrust::make_tuple(vals1,     vals2));

   if (opt == 0)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key, begin_keys, end_keys, begin_vals, thrust::less< thrust::tuple<T1, T2> >());
   }
   else if (opt == 2)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key, begin_keys, end_keys, begin_vals, TupleComp3<T1,T2>());
   }

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, HYPRE_Int *keys1, HYPRE_Int *keys2, char *vals1, HYPRE_Complex *vals2, HYPRE_Int opt);
#if defined(HYPRE_MIXEDINT)
template HYPRE_Int hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, HYPRE_BigInt *keys1, HYPRE_BigInt *keys2, char *vals1, HYPRE_Complex *vals2, HYPRE_Int opt);
#endif

template <typename T1, typename T2, typename T3>
HYPRE_Int
hypreDevice_ReduceByTupleKey(HYPRE_Int N, T1 *keys1_in,  T2 *keys2_in,  T3 *vals_in,
                                          T1 *keys1_out, T2 *keys2_out, T3 *vals_out)
{
   auto begin_keys_in  = thrust::make_zip_iterator(thrust::make_tuple(keys1_in,     keys2_in    ));
   auto end_keys_in    = thrust::make_zip_iterator(thrust::make_tuple(keys1_in + N, keys2_in + N));
   auto begin_keys_out = thrust::make_zip_iterator(thrust::make_tuple(keys1_out,    keys2_out   ));
   thrust::equal_to< thrust::tuple<T1, T2> > pred;
   thrust::plus<T3> func;

   auto new_end = HYPRE_THRUST_CALL(reduce_by_key, begin_keys_in, end_keys_in, vals_in, begin_keys_out, vals_out, pred, func);

   return new_end.second - vals_out;
}

template HYPRE_Int hypreDevice_ReduceByTupleKey(HYPRE_Int N, HYPRE_Int *keys1_in, HYPRE_Int *keys2_in, HYPRE_Complex *vals_in, HYPRE_Int *keys1_out, HYPRE_Int *keys2_out, HYPRE_Complex *vals_out);

#endif // #if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP)

#if defined(HYPRE_USING_CUSPARSE)
/*
 * @brief Determines the associated CudaDataType for the HYPRE_Complex typedef
 * @return Returns cuda data type corresponding with HYPRE_Complex
 *
 * @todo Should be known compile time
 * @todo Support more sizes
 * @todo Support complex
 * @warning Only works for Single and Double precision
 * @note Perhaps some typedefs should be added where HYPRE_Complex is typedef'd
 */
cudaDataType
hypre_HYPREComplexToCudaDataType()
{
   /*
   if (sizeof(char)*CHAR_BIT != 8)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ERROR:  Unsupported char size");
      hypre_assert(false);
   }
   */
#if defined(HYPRE_COMPLEX)
   return CUDA_C_64F;
#else
#if defined(HYPRE_SINGLE)
   hypre_assert(sizeof(HYPRE_Complex) == 4);
   return CUDA_R_32F;
#elif defined(HYPRE_LONG_DOUBLE)
#error "Long Double is not supported on GPUs"
#else
   hypre_assert(sizeof(HYPRE_Complex) == 8);
   return CUDA_R_64F;
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

#if defined(HYPRE_USING_GPU)

#if defined(HYPRE_USING_DEVICE_OPENMP)
cudaStream_t
#elif defined(HYPRE_USING_CUDA)
cudaStream_t
#elif defined(HYPRE_USING_HIP)
hipStream_t
#endif
hypre_CudaDataCudaStream(hypre_CudaData *data, HYPRE_Int i)
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
   cudaStream_t stream = 0;
#elif defined(HYPRE_USING_CUDA)
   cudaStream_t stream = 0;
#elif defined(HYPRE_USING_HIP)
   hipStream_t stream = 0;
#endif

#if defined(HYPRE_USING_CUDA_STREAMS)
   if (i >= HYPRE_MAX_NUM_STREAMS)
   {
      /* return the default stream, i.e., the NULL stream */
      /*
      hypre_printf("CUDA stream %d exceeds the max number %d\n",
                   i, HYPRE_MAX_NUM_STREAMS);
      */
      return NULL;
   }

   if (data->cuda_streams[i])
   {
      return data->cuda_streams[i];
   }

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
#elif defined(HYPRE_USING_CUDA)
   //HYPRE_CUDA_CALL(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
   HYPRE_CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamDefault));
#endif

   data->cuda_streams[i] = stream;
#endif

   return stream;
}

#if defined(HYPRE_USING_DEVICE_OPENMP)
cudaStream_t
#elif defined(HYPRE_USING_CUDA)
cudaStream_t
#elif defined(HYPRE_USING_HIP)
hipStream_t
#endif
hypre_CudaDataCudaComputeStream(hypre_CudaData *data)
{
   return hypre_CudaDataCudaStream(data,
                                   hypre_CudaDataCudaComputeStreamNum(data));
}

#if defined(HYPRE_USING_CURAND)
curandGenerator_t
hypre_CudaDataCurandGenerator(hypre_CudaData *data)
{
   if (data->curand_generator)
   {
      return data->curand_generator;
   }

   curandGenerator_t gen;
   HYPRE_CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
   HYPRE_CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
   HYPRE_CURAND_CALL( curandSetStream(gen, hypre_CudaDataCudaComputeStream(data)) );

   data->curand_generator = gen;

   return gen;
}

/* T = float or hypre_double */
template <typename T>
HYPRE_Int
hypre_CurandUniform_core( HYPRE_Int          n,
                          T                 *urand,
                          HYPRE_Int          set_seed,
                          hypre_ulonglongint seed,
                          HYPRE_Int          set_offset,
                          hypre_ulonglongint offset)
{
   curandGenerator_t gen = hypre_HandleCurandGenerator(hypre_handle());

   if (set_seed)
   {
      HYPRE_CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, seed) );
   }

   if (set_offset)
   {
      HYPRE_CURAND_CALL( curandSetGeneratorOffset(gen, offset) );
   }

   if (sizeof(T) == sizeof(hypre_double))
   {
      HYPRE_CURAND_CALL( curandGenerateUniformDouble(gen, (hypre_double *) urand, n) );
   }
   else if (sizeof(T) == sizeof(float))
   {
      HYPRE_CURAND_CALL( curandGenerateUniform(gen, (float *) urand, n) );
   }

   return hypre_error_flag;
}
#endif /* #if defined(HYPRE_USING_CURAND) */

#if defined(HYPRE_USING_ROCRAND)
rocrand_generator
hypre_CudaDataCurandGenerator(hypre_CudaData *data)
{
   if (data->curand_generator)
   {
      return data->curand_generator;
   }

   rocrand_generator gen;
   HYPRE_ROCRAND_CALL( rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT) );
   HYPRE_ROCRAND_CALL( rocrand_set_seed(gen, 1234ULL) );
   HYPRE_ROCRAND_CALL( rocrand_set_stream(gen, hypre_CudaDataCudaComputeStream(data)) );

   data->curand_generator = gen;

   return gen;
}

template <typename T>
HYPRE_Int
hypre_CurandUniform_core( HYPRE_Int          n,
                          T                 *urand,
                          HYPRE_Int          set_seed,
                          hypre_ulonglongint seed,
                          HYPRE_Int          set_offset,
                          hypre_ulonglongint offset)
{
  hypre_GpuProfilingPushRange("hypre_CurandUniform_core");

   rocrand_generator gen = hypre_HandleCurandGenerator(hypre_handle());

   if (set_seed)
   {
      HYPRE_ROCRAND_CALL( rocrand_set_seed(gen, seed) );
   }

   if (set_offset)
   {
      HYPRE_ROCRAND_CALL( rocrand_set_offset(gen, offset) );
   }

   if (sizeof(T) == sizeof(hypre_double))
   {
      HYPRE_ROCRAND_CALL( rocrand_generate_uniform_double(gen, (hypre_double *) urand, n) );
   }
   else if (sizeof(T) == sizeof(float))
   {
      HYPRE_ROCRAND_CALL( rocrand_generate_uniform(gen, (float *) urand, n) );
   }

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}
#endif /* #if defined(HYPRE_USING_ROCRAND) */

#if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND)

HYPRE_Int
hypre_CurandUniform( HYPRE_Int          n,
                     HYPRE_Real        *urand,
                     HYPRE_Int          set_seed,
                     hypre_ulonglongint seed,
                     HYPRE_Int          set_offset,
                     hypre_ulonglongint offset)
{
   return hypre_CurandUniform_core(n, urand, set_seed, seed, set_offset, offset);
}

HYPRE_Int
hypre_CurandUniformSingle( HYPRE_Int          n,
                           float             *urand,
                           HYPRE_Int          set_seed,
                           hypre_ulonglongint seed,
                           HYPRE_Int          set_offset,
                           hypre_ulonglongint offset)
{
   return hypre_CurandUniform_core(n, urand, set_seed, seed, set_offset, offset);
}

#endif /* #if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND) */

#if defined(HYPRE_USING_CUBLAS)
cublasHandle_t
hypre_CudaDataCublasHandle(hypre_CudaData *data)
{
   if (data->cublas_handle)
   {
      return data->cublas_handle;
   }

   cublasHandle_t handle;
   HYPRE_CUBLAS_CALL( cublasCreate(&handle) );

   HYPRE_CUBLAS_CALL( cublasSetStream(handle, hypre_CudaDataCudaComputeStream(data)) );

   data->cublas_handle = handle;

   return handle;
}
#endif

#if defined(HYPRE_USING_CUSPARSE)
cusparseHandle_t
hypre_CudaDataCusparseHandle(hypre_CudaData *data)
{
   if (data->cusparse_handle)
   {
      return data->cusparse_handle;
   }

   cusparseHandle_t handle;
   HYPRE_CUSPARSE_CALL( cusparseCreate(&handle) );

   HYPRE_CUSPARSE_CALL( cusparseSetStream(handle, hypre_CudaDataCudaComputeStream(data)) );

   data->cusparse_handle = handle;

   return handle;
}
#endif // defined(HYPRE_USING_CUSPARSE)


#if defined(HYPRE_USING_ROCSPARSE)
rocsparse_handle
hypre_CudaDataCusparseHandle(hypre_CudaData *data)
{
   if (data->cusparse_handle)
   {
      return data->cusparse_handle;
   }

   rocsparse_handle handle;
   HYPRE_ROCSPARSE_CALL( rocsparse_create_handle(&handle) );

   HYPRE_ROCSPARSE_CALL( rocsparse_set_stream(handle, hypre_CudaDataCudaComputeStream(data)) );

   data->cusparse_handle = handle;

   return handle;
}
#endif // defined(HYPRE_USING_ROCSPARSE)



hypre_CudaData*
hypre_CudaDataCreate()
{
   hypre_CudaData *data = hypre_CTAlloc(hypre_CudaData, 1, HYPRE_MEMORY_HOST);

   hypre_CudaDataCudaDevice(data)            = 0;
   hypre_CudaDataCudaComputeStreamNum(data)  = 0;

   /* SpGeMM */
#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)
   hypre_CudaDataSpgemmUseCusparse(data) = 1;
#else
   hypre_CudaDataSpgemmUseCusparse(data) = 0;
#endif
   hypre_CudaDataSpgemmNumPasses(data) = 3;
   /* 1: naive overestimate, 2: naive underestimate, 3: Cohen's algorithm */
   hypre_CudaDataSpgemmRownnzEstimateMethod(data) = 3;
   hypre_CudaDataSpgemmRownnzEstimateNsamples(data) = 32;
   hypre_CudaDataSpgemmRownnzEstimateMultFactor(data) = 1.5;
   hypre_CudaDataSpgemmHashType(data) = 'L';

   /* pmis */
#if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND)
   hypre_CudaDataUseGpuRand(data) = 1;
#else
   hypre_CudaDataUseGpuRand(data) = 0;
#endif

   /* device pool */
#ifdef HYPRE_USING_DEVICE_POOL
   hypre_CudaDataCubBinGrowth(data)      = 8u;
   hypre_CudaDataCubMinBin(data)         = 1u;
   hypre_CudaDataCubMaxBin(data)         = (hypre_uint) -1;
   hypre_CudaDataCubMaxCachedBytes(data) = (size_t) -1;
   hypre_CudaDataCubDevAllocator(data)   = NULL;
   hypre_CudaDataCubUvmAllocator(data)   = NULL;
#endif

   return data;
}

void
hypre_CudaDataDestroy(hypre_CudaData *data)
{
   if (!data)
   {
      return;
   }

   hypre_TFree(hypre_CudaDataCudaReduceBuffer(data),     HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_CudaDataStructCommRecvBuffer(data), HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_CudaDataStructCommSendBuffer(data), HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_CURAND)
   if (data->curand_generator)
   {
      HYPRE_CURAND_CALL( curandDestroyGenerator(data->curand_generator) );
   }
#endif

#if defined(HYPRE_USING_ROCRAND)
   if (data->curand_generator)
   {
      HYPRE_ROCRAND_CALL( rocrand_destroy_generator(data->curand_generator) );
   }
#endif

#if defined(HYPRE_USING_CUBLAS)
   if (data->cublas_handle)
   {
      HYPRE_CUBLAS_CALL( cublasDestroy(data->cublas_handle) );
   }
#endif

#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)
   if (data->cusparse_handle)
   {
#if defined(HYPRE_USING_CUSPARSE)
      HYPRE_CUSPARSE_CALL( cusparseDestroy(data->cusparse_handle) );
#elif defined(HYPRE_USING_ROCSPARSE)
      HYPRE_ROCSPARSE_CALL( rocsparse_destroy_handle(data->cusparse_handle) );
#endif
   }
#endif // #if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)

   for (HYPRE_Int i = 0; i < HYPRE_MAX_NUM_STREAMS; i++)
   {
      if (data->cuda_streams[i])
      {
#if defined(HYPRE_USING_DEVICE_OPENMP)
         HYPRE_CUDA_CALL( cudaStreamDestroy(data->cuda_streams[i]) );
#elif defined(HYPRE_USING_CUDA)
         HYPRE_CUDA_CALL( cudaStreamDestroy(data->cuda_streams[i]) );
#elif defined(HYPRE_USING_HIP)
         HYPRE_HIP_CALL( hipStreamDestroy(data->cuda_streams[i]) );
#endif
      }
   }

#ifdef HYPRE_USING_DEVICE_POOL
   hypre_CudaDataCubCachingAllocatorDestroy(data);
#endif

   hypre_TFree(data, HYPRE_MEMORY_HOST);
}

HYPRE_Int
hypre_SyncCudaDevice(hypre_Handle *hypre_handle)
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaDeviceSynchronize() );
#elif defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaDeviceSynchronize() );
#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipDeviceSynchronize() );
#endif
   return hypre_error_flag;
}

/* synchronize the Hypre compute stream
 * action: 0: set sync stream to false
 *         1: set sync stream to true
 *         2: restore sync stream to default
 *         3: return the current value of cuda_compute_stream_sync
 *         4: sync stream based on cuda_compute_stream_sync
 */
HYPRE_Int
hypre_SyncCudaComputeStream_core(HYPRE_Int     action,
                                 hypre_Handle *hypre_handle,
                                 HYPRE_Int    *cuda_compute_stream_sync_ptr)
{
   /* with UVM the default is to sync at kernel completions, since host is also able to
    * touch GPU memory */
#if defined(HYPRE_USING_UNIFIED_MEMORY)
   static const HYPRE_Int cuda_compute_stream_sync_default = 1;
#else
   static const HYPRE_Int cuda_compute_stream_sync_default = 0;
#endif

   /* this controls if synchronize the stream after computations */
   static HYPRE_Int cuda_compute_stream_sync = cuda_compute_stream_sync_default;

   switch (action)
   {
      case 0:
         cuda_compute_stream_sync = 0;
         break;
      case 1:
         cuda_compute_stream_sync = 1;
         break;
      case 2:
         cuda_compute_stream_sync = cuda_compute_stream_sync_default;
         break;
      case 3:
         *cuda_compute_stream_sync_ptr = cuda_compute_stream_sync;
         break;
      case 4:
#if defined(HYPRE_USING_DEVICE_OPENMP)
         HYPRE_CUDA_CALL( cudaDeviceSynchronize() );
#else
         if (cuda_compute_stream_sync)
         {
#if defined(HYPRE_USING_CUDA)
            HYPRE_CUDA_CALL( cudaStreamSynchronize(hypre_HandleCudaComputeStream(hypre_handle)) );
#elif defined(HYPRE_USING_HIP)
            HYPRE_HIP_CALL( hipStreamSynchronize(hypre_HandleCudaComputeStream(hypre_handle)) );
#endif
         }
#endif
         break;
      default:
         hypre_printf("hypre_SyncCudaComputeStream_core invalid action\n");
         hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetSyncCudaCompute(HYPRE_Int action)
{
   /* convert to 1/0 */
   action = action != 0;
   hypre_SyncCudaComputeStream_core(action, NULL, NULL);

   return hypre_error_flag;
}

HYPRE_Int
hypre_RestoreSyncCudaCompute()
{
   hypre_SyncCudaComputeStream_core(2, NULL, NULL);

   return hypre_error_flag;
}

HYPRE_Int
hypre_GetSyncCudaCompute(HYPRE_Int *cuda_compute_stream_sync_ptr)
{
   hypre_SyncCudaComputeStream_core(3, NULL, cuda_compute_stream_sync_ptr);

   return hypre_error_flag;
}

HYPRE_Int
hypre_SyncCudaComputeStream(hypre_Handle *hypre_handle)
{
   hypre_SyncCudaComputeStream_core(4, hypre_handle, NULL);

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU)


/* This function is supposed to be used in the test drivers to mimic
 * users' GPU binding approaches
 * It is supposed to be called before HYPRE_Init,
 * so that HYPRE_Init can get the wanted device id
 */
HYPRE_Int
hypre_bind_device( HYPRE_Int myid,
                   HYPRE_Int nproc,
                   MPI_Comm  comm )
{
#ifdef HYPRE_USING_GPU
   /* proc id (rank) on the running node */
   HYPRE_Int myNodeid;
   /* num of procs (size) on the node */
   HYPRE_Int NodeSize;
   /* num of devices seen */
   hypre_int nDevices;
   /* device id that want to bind */
   hypre_int device_id;

   hypre_MPI_Comm node_comm;
   hypre_MPI_Comm_split_type( comm, hypre_MPI_COMM_TYPE_SHARED,
                              myid, hypre_MPI_INFO_NULL, &node_comm );
   hypre_MPI_Comm_rank(node_comm, &myNodeid);
   hypre_MPI_Comm_size(node_comm, &NodeSize);
   hypre_MPI_Comm_free(&node_comm);

   /* get number of devices on this node */
   hypre_GetDeviceCount(&nDevices);

   /* set device */
   device_id = myNodeid % nDevices;
   hypre_SetDevice(device_id, NULL);

#if defined(HYPRE_DEBUG) && defined(HYPRE_PRINT_ERRORS)
   hypre_printf("Proc [global %d/%d, local %d/%d] can see %d GPUs and is running on %d\n",
                myid, nproc, myNodeid, NodeSize, nDevices, device_id);
#endif

#endif /* #ifdef HYPRE_USING_GPU */

   return hypre_error_flag;
}

