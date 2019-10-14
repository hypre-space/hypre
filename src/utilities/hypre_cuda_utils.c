/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

#if defined(HYPRE_USING_CUDA)

/* for struct solvers */
HYPRE_Int hypre_exec_policy = HYPRE_MEMORY_DEVICE;

__global__ void
hypreCUDAKernel_CompileFlagSafetyCheck(HYPRE_Int *cuda_arch)
{
#ifdef __CUDA_ARCH__
    cuda_arch[0] = __CUDA_ARCH__;
#endif
}

void hypre_CudaCompileFlagCheck()
{
   HYPRE_Int device = hypre_HandleCudaDevice(hypre_handle);

   struct cudaDeviceProp props;
   cudaGetDeviceProperties(&props, device);
   HYPRE_Int cuda_arch_actual = props.major*100 + props.minor*10;

   HYPRE_Int *cuda_arch = hypre_TAlloc(HYPRE_Int, 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int h_cuda_arch;

   dim3 gDim(1,1,1), bDim(1,1,1);
   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CompileFlagSafetyCheck, gDim, bDim, cuda_arch );

   hypre_TMemcpy(&h_cuda_arch, cuda_arch, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   if (h_cuda_arch != cuda_arch_actual)
   {
      hypre_printf("ERROR: Compile arch flags %d does not match actual device arch = sm_%d\n", h_cuda_arch, cuda_arch_actual);
   }

   HYPRE_CUDA_CALL(cudaDeviceSynchronize());

   hypre_TFree(cuda_arch, HYPRE_MEMORY_DEVICE);
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
      HYPRE_Int num_warps_per_block = num_threads_per_block >> 5;

      assert(num_warps_per_block * 32 == num_threads_per_block);

      num_blocks = (n + num_warps_per_block - 1) / num_warps_per_block;
   }
   else
   {
      hypre_printf("Error %s %d: Unknown granularity !\n", __FILE__, __LINE__);
      assert(0);
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
   const HYPRE_Int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
   //const HYPRE_Int total_num_threads = gridDim.x  * blockDim.x;

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
hypreCUDAKernel_CopyParCSRRows(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int has_offd,
                               HYPRE_Int first_col, HYPRE_Int *d_col_map_offd_A,
                               HYPRE_Int *d_diag_i, HYPRE_Int *d_diag_j, HYPRE_Complex *d_diag_a,
                               HYPRE_Int *d_offd_i, HYPRE_Int *d_offd_j, HYPRE_Complex *d_offd_a,
                               HYPRE_Int *d_ib, HYPRE_BigInt *d_jb, HYPRE_Complex *d_ab)
{
   HYPRE_Int global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;

   if (global_warp_id >= nrows)
   {
      return;
   }
   /* lane id inside the warp */
   HYPRE_Int lane_id = threadIdx.x;
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
      k = read_only_load(d_ib + global_warp_id);
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
/* d_ib is input that contains row ptrs, of length (nrows + 1) or nrow (without the last entry, nnz) */
/* special case: if d_row_indices == NULL, it means d_row_indices=[0,1,...,nrows-1]
 * if col_map_offd_A == NULL, use (-1 - d_offd_j) as column id*/
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

   HYPRE_Int num_warps_per_block = 16;
   const dim3 bDim(HYPRE_WARP_SIZE, num_warps_per_block);
   const dim3 gDim((nrows + num_warps_per_block - 1) / num_warps_per_block);

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

__global__ void
hypreCUDAKernel_CsrRowPtrsToIndices(HYPRE_Int n, HYPRE_Int *ptr, HYPRE_Int *num, HYPRE_Int *idx)
{
   /* warp id in the grid */
   HYPRE_Int global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
   /* lane id inside the warp */
   HYPRE_Int lane_id = threadIdx.x;

   if (global_warp_id < n)
   {
      HYPRE_Int istart, iend, k;

      if (lane_id < 2)
      {
         k = read_only_load(ptr + global_warp_id + lane_id);
      }
      istart = __shfl_sync(HYPRE_WARP_FULL_MASK, k, 0);
      iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, k, 1);

      HYPRE_Int i;
      for (i = istart + lane_id; i < iend; i += HYPRE_WARP_SIZE)
      {
         HYPRE_Int j;
         if (num == NULL)
         {
            j = global_warp_id;
         }
         else
         {
            j = read_only_load(num + global_warp_id);
         }
         idx[i] = j;
      }
   }
}

HYPRE_Int*
hypreDevice_CsrRowPtrsToIndices(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr)
{
   /* trivial case */
   if (nrows <= 0)
   {
      return NULL;
   }

   HYPRE_Int *d_row_ind = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);

   HYPRE_Int num_warps_per_block = 16;
   const dim3 bDim(HYPRE_WARP_SIZE, num_warps_per_block);
   const dim3 gDim((nrows + num_warps_per_block - 1) / num_warps_per_block);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CsrRowPtrsToIndices, gDim, bDim,
                      nrows, d_row_ptr, NULL, d_row_ind );

   return d_row_ind;
}

HYPRE_Int
hypreDevice_CsrRowPtrsToIndices_v2(HYPRE_Int nrows, HYPRE_Int *d_row_ptr, HYPRE_Int *d_row_ind)
{
   /* trivial case */
   if (nrows <= 0)
   {
      return hypre_error_flag;
   }

   HYPRE_Int num_warps_per_block = 16;
   const dim3 bDim(HYPRE_WARP_SIZE, num_warps_per_block);
   const dim3 gDim((nrows + num_warps_per_block - 1) / num_warps_per_block);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CsrRowPtrsToIndices, gDim, bDim,
                      nrows, d_row_ptr, NULL, d_row_ind );

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CsrRowPtrsToIndicesWithRowNum(HYPRE_Int nrows, HYPRE_Int *d_row_ptr, HYPRE_Int *d_row_num, HYPRE_Int *d_row_ind)
{
   /* trivial case */
   if (nrows <= 0)
   {
      return hypre_error_flag;
   }

   HYPRE_Int num_warps_per_block = 16;
   const dim3 bDim(HYPRE_WARP_SIZE, num_warps_per_block);
   const dim3 gDim((nrows + num_warps_per_block - 1) / num_warps_per_block);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CsrRowPtrsToIndices, gDim, bDim,
                      nrows, d_row_ptr, d_row_num, d_row_ind );

   return hypre_error_flag;
}

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

/* x[map[i]] += y[i] */
__global__ void
hypreCUDAKernel_ScatterAdd(HYPRE_Int n, HYPRE_Real *x, HYPRE_Int *map, HYPRE_Real *y)
{
   HYPRE_Int global_thread_id = hypre_cuda_get_grid_thread_id<1,1>();

   if (global_thread_id < n)
   {
      x[map[global_thread_id]] += y[global_thread_id];
   }
}

/* Generalized x[map[i]] += y[i] where the same index may appear more
 * than once in map
 * Note: content in y will be destroyed */
HYPRE_Int
hypreDevice_GenScatterAdd(HYPRE_Real *x, HYPRE_Int ny, HYPRE_Int *map, HYPRE_Real *y)
{
   HYPRE_Int *map2 = hypre_TAlloc(HYPRE_Int, ny, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *reduced_map = hypre_TAlloc(HYPRE_Int, ny, HYPRE_MEMORY_DEVICE);
   HYPRE_Real *reduced_y = hypre_TAlloc(HYPRE_Real, ny, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(map2, map, HYPRE_Int, ny, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL(sort_by_key, map2, map2+ny, y);

   thrust::pair<HYPRE_Int*, HYPRE_Real*> new_end =
      HYPRE_THRUST_CALL(reduce_by_key, map2, map2+ny, y, reduced_map, reduced_y);

   HYPRE_Int reduced_n = new_end.first - reduced_map;

   hypre_assert(reduced_n == new_end.second - reduced_y);

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(reduced_n, "thread", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_ScatterAdd, gDim, bDim,
                      reduced_n, x, reduced_map, reduced_y );

   hypre_TFree(map2, HYPRE_MEMORY_DEVICE);
   hypre_TFree(reduced_map, HYPRE_MEMORY_DEVICE);
   hypre_TFree(reduced_y, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/* x[map[i]] = v */
__global__ void
hypreCUDAKernel_ScatterConstant(HYPRE_Int *x, HYPRE_Int n, HYPRE_Int *map, HYPRE_Int v)
{
   HYPRE_Int global_thread_id = hypre_cuda_get_grid_thread_id<1,1>();

   if (global_thread_id < n)
   {
      x[map[global_thread_id]] = v;
   }
}

/* x[map[i]] = v
 * TODO: thrust? */
HYPRE_Int
hypreDevice_ScatterConstant(HYPRE_Int *x, HYPRE_Int n, HYPRE_Int *map, HYPRE_Int v)
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
hypreCUDAKernel_DiagScaleVector(HYPRE_Int n, HYPRE_Int *A_i, HYPRE_Complex *A_data, HYPRE_Complex *x, HYPRE_Complex *y)
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>();

   if (i < n)
   {
      y[i] = x[i] / A_data[A_i[i]];
   }
}

/* y = diag(A) \ x. A_i[i] points to the ith diagonal entry of A */
HYPRE_Int
hypreDevice_DiagScaleVector(HYPRE_Int n, HYPRE_Int *A_i, HYPRE_Complex *A_data, HYPRE_Complex *x, HYPRE_Complex *y)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(n, "thread", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_DiagScaleVector, gDim, bDim, n, A_i, A_data, x, y );

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
/* opt: 0, (a,b) <= (a',b') if and only if a < a' or (a = a' and  b  <=  b')
 *      1, (a,b) <= (a',b') if and only if a < a' or (a = a' and |b| >= |b'|)
 */
template <typename T1, typename T2, typename T3>
HYPRE_Int
hypreDevice_StableSortByTupleKey(HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals, HYPRE_Int opt)
{
   auto begin_keys = thrust::make_zip_iterator(thrust::make_tuple(keys1,     keys2));
   auto end_keys   = thrust::make_zip_iterator(thrust::make_tuple(keys1 + N, keys2 + N));

   if (opt == 0)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key, begin_keys, end_keys, vals, TupleComp1<T1,T2>());
   }
   else if (opt == 1)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key, begin_keys, end_keys, vals, TupleComp2<T1,T2>());
   }

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N, HYPRE_Int *keys1, HYPRE_Int  *keys2, HYPRE_Int *vals, HYPRE_Int opt);
template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N, HYPRE_Int *keys1, HYPRE_Real *keys2, HYPRE_Int *vals, HYPRE_Int opt);

#endif // #if defined(HYPRE_USING_CUDA)

