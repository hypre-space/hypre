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

#include "_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetConstantValuesDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetConstantValuesDevice( hypre_Vector *v,
                                        HYPRE_Complex value )
{
   HYPRE_Complex *vector_data = hypre_VectorData(v);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(v);
   HYPRE_Int      size        = hypre_VectorSize(v);
   HYPRE_Int      total_size  = size * num_vectors;

   //hypre_SeqVectorPrefetch(v, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_GPU)
   hypreDevice_ComplexFilln( vector_data, total_size, value );

   hypre_SyncComputeStream(hypre_handle());

#elif defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(vector_data)
   for (i = 0; i < total_size; i++)
   {
      vector_data[i] = value;
   }
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorScaleDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorScaleDevice( HYPRE_Complex alpha,
                            hypre_Vector *y )
{
   HYPRE_Complex *y_data      = hypre_VectorData(y);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(y);
   HYPRE_Int      size        = hypre_VectorSize(y);
   HYPRE_Int      total_size  = size * num_vectors;

   hypre_GpuProfilingPushRange("SeqVectorScale");
   //hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_GPU)

#if ( defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) ) && defined(HYPRE_USING_CUBLAS)
   HYPRE_CUBLAS_CALL( hypre_cublas_scal(hypre_HandleCublasHandle(hypre_handle()),
                                        total_size, &alpha, y_data, 1) );
#elif defined(HYPRE_USING_SYCL) && defined(HYPRE_USING_ONEMKLBLAS)
   HYPRE_ONEMKL_CALL( oneapi::mkl::blas::scal(*hypre_HandleComputeStream(hypre_handle()),
                                              total_size, alpha,
                                              y_data, 1).wait() );
#else
   hypreDevice_ComplexScalen( y_data, total_size, y_data, alpha );
#endif

   hypre_SyncComputeStream(hypre_handle());

#elif defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data)
   for (i = 0; i < total_size; i++)
   {
      y_data[i] *= alpha;
   }
#endif

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpyDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorAxpyDevice( HYPRE_Complex alpha,
                           hypre_Vector *x,
                           hypre_Vector *y )
{
   HYPRE_Complex *x_data      = hypre_VectorData(x);
   HYPRE_Complex *y_data      = hypre_VectorData(y);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int      size        = hypre_VectorSize(x);
   HYPRE_Int      total_size  = size * num_vectors;

#if defined(HYPRE_USING_GPU)

#if ( defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) ) && defined(HYPRE_USING_CUBLAS)
   HYPRE_CUBLAS_CALL( hypre_cublas_axpy(hypre_HandleCublasHandle(hypre_handle()),
                                        total_size, &alpha, x_data, 1,
                                        y_data, 1) );
#elif defined(HYPRE_USING_SYCL) && defined(HYPRE_USING_ONEMKLBLAS)
   HYPRE_ONEMKL_CALL( oneapi::mkl::blas::axpy(*hypre_HandleComputeStream(hypre_handle()),
                                              total_size, alpha,
                                              x_data, 1, y_data, 1).wait() );
#else
   hypreDevice_ComplexAxpyn(x_data, total_size, y_data, y_data, alpha);
#endif

   hypre_SyncComputeStream(hypre_handle());

#elif defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data, x_data)
   for (i = 0; i < total_size; i++)
   {
      y_data[i] += alpha * x_data[i];
   }
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpyzDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorAxpyzDevice( HYPRE_Complex  alpha,
                            hypre_Vector  *x,
                            HYPRE_Complex  beta,
                            hypre_Vector  *y,
                            hypre_Vector  *z )
{
   HYPRE_Complex  *x_data      = hypre_VectorData(x);
   HYPRE_Complex  *y_data      = hypre_VectorData(y);
   HYPRE_Complex  *z_data      = hypre_VectorData(z);

   HYPRE_Int       num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int       size        = hypre_VectorSize(x);
   HYPRE_Int       total_size  = size * num_vectors;

#if defined(HYPRE_USING_GPU)
   hypreDevice_ComplexAxpyzn(total_size, x_data, y_data, z_data, alpha, beta);

   hypre_SyncComputeStream(hypre_handle());

#elif defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(z_data, y_data, x_data)
   for (i = 0; i < total_size; i++)
   {
      z_data[i] = alpha * x_data[i] + beta * y_data[i];
   }
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorElmdivpyDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorElmdivpyDevice( hypre_Vector *x,
                               hypre_Vector *b,
                               hypre_Vector *y,
                               HYPRE_Int    *marker,
                               HYPRE_Int     marker_val )
{
#if defined(HYPRE_USING_GPU)
   HYPRE_Complex  *x_data        = hypre_VectorData(x);
   HYPRE_Complex  *b_data        = hypre_VectorData(b);
   HYPRE_Complex  *y_data        = hypre_VectorData(y);
   HYPRE_Int       num_vectors_x = hypre_VectorNumVectors(x);
   HYPRE_Int       num_vectors_y = hypre_VectorNumVectors(y);
   HYPRE_Int       num_vectors_b = hypre_VectorNumVectors(b);
   HYPRE_Int       size          = hypre_VectorSize(b);

   hypre_GpuProfilingPushRange("SeqVectorElmdivpyDevice");
   if (num_vectors_b == 1)
   {
      if (num_vectors_x == 1)
      {
         if (marker)
         {
            hypreDevice_IVAXPYMarked(size, b_data, x_data, y_data, marker, marker_val);
         }
         else
         {
            hypreDevice_IVAXPY(size, b_data, x_data, y_data);
         }
      }
#if !defined(HYPRE_USING_SYCL)
      else if (num_vectors_x == num_vectors_y)
      {
         if (!marker)
         {
            hypreDevice_IVAMXPMY(num_vectors_x, size, b_data, x_data, y_data);
         }
         else
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "marker != NULL not supported!\n");
         }
      }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported combination of num_vectors!\n");
      }

#else
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "num_vectors_x != 1 not supported for SYCL!\n");
      }
#endif
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "num_vectors_b != 1 not supported!\n");
   }

   hypre_SyncComputeStream(hypre_handle());
   hypre_GpuProfilingPopRange();

#elif defined(HYPRE_USING_OPENMP)
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented for device OpenMP!\n");
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProdDevice
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_SeqVectorInnerProdDevice( hypre_Vector *x,
                                hypre_Vector *y )
{
   HYPRE_Complex *x_data      = hypre_VectorData(x);
   HYPRE_Complex *y_data      = hypre_VectorData(y);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int      size        = hypre_VectorSize(x);
   HYPRE_Int      total_size  = size * num_vectors;

   HYPRE_Real     result = 0.0;

   //hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_GPU)

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#if defined(HYPRE_USING_CUBLAS)
   HYPRE_CUBLAS_CALL( hypre_cublas_dot(hypre_HandleCublasHandle(hypre_handle()), total_size,
                                       x_data, 1, y_data, 1, &result) );
#else
   result = HYPRE_THRUST_CALL( inner_product, x_data, x_data + total_size, y_data, 0.0 );
#endif

#elif defined(HYPRE_USING_SYCL)
#if defined(HYPRE_USING_ONEMKLBLAS)
   HYPRE_Real *result_dev = hypre_CTAlloc(HYPRE_Real, 1, HYPRE_MEMORY_DEVICE);
   HYPRE_ONEMKL_CALL( oneapi::mkl::blas::dot(*hypre_HandleComputeStream(hypre_handle()),
                                             total_size, x_data, 1,
                                             y_data, 1, result_dev).wait() );
   hypre_TMemcpy(&result, result_dev, HYPRE_Real, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TFree(result_dev, HYPRE_MEMORY_DEVICE);
#else
   result = HYPRE_ONEDPL_CALL( std::transform_reduce, x_data, x_data + total_size, y_data, 0.0 );
#endif
#endif

   hypre_SyncComputeStream(hypre_handle());

#elif defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) reduction(+:result) is_device_ptr(y_data, x_data) map(result)
   for (i = 0; i < total_size; i++)
   {
      result += hypre_conj(y_data[i]) * x_data[i];
   }
#endif

   return result;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSumEltsDevice
 *--------------------------------------------------------------------------*/

HYPRE_Complex
hypre_SeqVectorSumEltsDevice( hypre_Vector *vector )
{
   HYPRE_Complex  *data        = hypre_VectorData(vector);
   HYPRE_Int       num_vectors = hypre_VectorNumVectors(vector);
   HYPRE_Int       size        = hypre_VectorSize(vector);
   HYPRE_Int       total_size  = size * num_vectors;
   HYPRE_Complex   sum = 0.0;

#if defined(HYPRE_USING_GPU)
   sum = hypreDevice_ComplexReduceSum(total_size, data);

   hypre_SyncComputeStream(hypre_handle());

#elif HYPRE_USING_DEVICE_OPENMP
   HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) reduction(+:sum) is_device_ptr(data) map(sum)
   for (i = 0; i < total_size; i++)
   {
      sum += data[i];
   }
#endif

   return sum;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorPrefetch
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPrefetch( hypre_Vector        *x,
                         HYPRE_MemoryLocation memory_location )
{
#if defined(HYPRE_USING_UNIFIED_MEMORY)
   if (hypre_VectorMemoryLocation(x) != HYPRE_MEMORY_DEVICE)
   {
      /* hypre_error_w_msg(HYPRE_ERROR_GENERIC," Error! CUDA Prefetch with non-unified momory\n"); */
      return hypre_error_flag;
   }

   HYPRE_Complex  *x_data      = hypre_VectorData(x);
   HYPRE_Int       num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int       size        = hypre_VectorSize(x);
   HYPRE_Int       total_size  = size * num_vectors;

   if (total_size == 0)
   {
      return hypre_error_flag;
   }

   hypre_MemPrefetch(x_data, sizeof(HYPRE_Complex) * total_size, memory_location);
#endif

   return hypre_error_flag;
}

#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

__global__ void
hypreGPUKernel_MassAxpy(hypre_DeviceItem &item, const HYPRE_Int k, const HYPRE_Int n, const HYPRE_Complex * __restrict__ alpha, const HYPRE_Complex * __restrict__ x, HYPRE_Complex * __restrict__ y)
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1,1>(item);
   HYPRE_Int j = 0;
   HYPRE_Complex sum = 0.0, xx=0.0;

   if (i < n)
   {
      for (j = 0; j < k; ++j)
      {
         xx = x[j*n+i];
         sum += alpha[j] * xx;
      }
      y[i] += sum;
   }
}

HYPRE_Int
hypreDevice_MassAxpy(HYPRE_Int k, HYPRE_Int n, HYPRE_Complex *alpha, HYPRE_Complex *x, HYPRE_Complex *y)
{
   /* trivial case */
   if (n <= 0 || k<=0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n, "thread", bDim);
   HYPRE_GPU_LAUNCH( hypreGPUKernel_MassAxpy, gDim, bDim, k, n, alpha, x, y);
   return hypre_error_flag;
}

template<HYPRE_Int B, HYPRE_Int WS>
__global__ void
hypreGPUKernel_MassInnerProd_kernel1(const HYPRE_Int k, const HYPRE_Int n, const HYPRE_Complex * __restrict__ x, const HYPRE_Complex * __restrict__ y, HYPRE_Complex * __restrict__ z)
{
   constexpr HYPRE_Int size=B/WS;
   extern volatile __shared__ HYPRE_Complex shmem[];
   HYPRE_Int i = 0, j = 0, ii=0;
   HYPRE_Int tidx = threadIdx.x+blockDim.x*blockIdx.x;
   HYPRE_Int warp_lane = threadIdx.x & (WS - 1);
   HYPRE_Int warp_id = threadIdx.x / WS;
   HYPRE_Complex xx = 0.0, sum = 0.0;

   i=threadIdx.x;
   while(i<size*k)
   {
      shmem[i] = 0.0;
	  i+=blockDim.x;
   }
   __syncthreads();

   for (ii=tidx; ii<n; ii+=blockDim.x*gridDim.x)
   {
	  xx = x[ii];

      for (j=0; j<k; ++j)
      {
         sum = xx*y[ii+j*n];

         // Warp shuffle reduce
#pragma unroll
         for (i = WS>>1; i > 0; i >>= 1)
         {
            sum += __shfl_down_sync(WS, sum, i);
         }

		 // Save in shmem for each k-value
		 if (warp_lane==0)
		 {
		    shmem[warp_id+j*size] += sum;
		 }
	  }
   }
   __syncthreads();

   // Complete the reduction for each k-value
   HYPRE_Int reduce_id =   threadIdx.x / size;
   HYPRE_Int reduce_lane = threadIdx.x % size;
   for (j=reduce_id; j<k; j+=WS)
   {
      sum = shmem[reduce_lane+j*size];

#pragma unroll
      for (i = size>>1; i > 0; i >>= 1)
      {
         sum += __shfl_down_sync(size, sum, i);
	  }

      /* Write to global memory */
      if (reduce_lane==0)
      {
         z[blockIdx.x+j*gridDim.x] = sum;
      }
   }
}


template<HYPRE_Int B, HYPRE_Int WS>
__global__ void
hypreGPUKernel_MassInnerProd_kernel2(const HYPRE_Int n, const HYPRE_Complex * __restrict__ z, HYPRE_Complex * __restrict__ w)
{
   const HYPRE_Int size=B/WS;
   volatile __shared__ HYPRE_Complex shmem[size];
   HYPRE_Int i = 0;
   HYPRE_Int warp_lane = threadIdx.x & (WS - 1);
   HYPRE_Int warp_id = threadIdx.x / WS;
   HYPRE_Complex sum = 0.0, zz = 0.0;

   for (i=threadIdx.x; i<n; i+=blockDim.x)
   {
      zz = z[blockIdx.x*n + i];
      sum += zz;
   }

   // Warp shuffle reduce
#pragma unroll
   for (i = WS>>1; i > 0; i >>= 1)
   {
      sum += __shfl_down_sync(WS, sum, i);
   }

   // Combine across warps through shmem
   __syncthreads();
   if (warp_lane==0)
   {
      shmem[warp_id] = sum;
   }
   __syncthreads();

   // Put it back in a register to finish the reduction
   if (threadIdx.x<size)
   {
      sum = shmem[threadIdx.x];
   }

   // Warp shuffle reduce
#pragma unroll
   for (i = size>>1; i > 0; i >>= 1)
   {
      sum += __shfl_down_sync(size, sum, i);
   }

   /* Write to global memory */
   if (threadIdx.x==0)
   {
      w[blockIdx.x] = sum;
   }
}


template<HYPRE_Int B, HYPRE_Int WS>
__global__ void
hypreGPUKernel_MassDotpTwo_kernel1(const HYPRE_Int k, const HYPRE_Int n, const HYPRE_Complex * __restrict__ x, const HYPRE_Complex * __restrict__ y, const HYPRE_Complex * __restrict__ z, HYPRE_Complex * __restrict__ res_x, HYPRE_Complex * __restrict__ res_y)
{
   constexpr HYPRE_Int size=B/WS;
   extern volatile __shared__ HYPRE_Complex shmem[];
   HYPRE_Int i = 0, j = 0, ii=0;
   HYPRE_Int tidx = threadIdx.x+blockDim.x*blockIdx.x;
   HYPRE_Int warp_lane = threadIdx.x & (WS - 1);
   HYPRE_Int warp_id = threadIdx.x / WS;
   HYPRE_Complex xx = 0.0, yy = 0.0, zz = 0.0;
   HYPRE_Complex sum_x = 0.0, sum_y = 0.0;

   i=threadIdx.x;
   while(i<2*size*k)
   {
      shmem[i] = 0.0;
	  i+=blockDim.x;
   }
   __syncthreads();

   for (ii=tidx; ii<n; ii+=blockDim.x*gridDim.x)
   {
	  xx = x[ii];
	  yy = y[ii];

      for (j=0; j<k; ++j)
      {
         zz = z[ii+j*n];
         sum_x = xx*zz;
         sum_y = yy*zz;

         // Warp shuffle reduce
#pragma unroll
         for (i = WS>>1; i > 0; i >>= 1)
         {
            sum_x += __shfl_down_sync(WS, sum_x, i);
            sum_y += __shfl_down_sync(WS, sum_y, i);
         }

		 // Save in shmem for each k-value
		 if (warp_lane==0)
		 {
		    shmem[warp_id+j*size]        += sum_x;
		    shmem[warp_id+j*size+k*size] += sum_y;
		 }
	  }
   }
   __syncthreads();

   // Complete the reduction for each k-value
   HYPRE_Int reduce_id =   threadIdx.x / size;
   HYPRE_Int reduce_lane = threadIdx.x % size;
   for (j=reduce_id; j<k; j+=WS)
   {
      sum_x = shmem[reduce_lane+j*size];
      sum_y = shmem[reduce_lane+j*size+k*size];

#pragma unroll
      for (i = size>>1; i > 0; i >>= 1)
      {
         sum_x += __shfl_down_sync(size, sum_x, i);
         sum_y += __shfl_down_sync(size, sum_y, i);
	  }

      /* Write to global memory */
      if (reduce_lane==0)
      {
		 res_x[blockIdx.x+j*gridDim.x] = sum_x;
		 res_y[blockIdx.x+j*gridDim.x] = sum_y;
      }
   }
}

template<HYPRE_Int B, HYPRE_Int WS>
__global__ void
hypreGPUKernel_MassDotpTwo_kernel2(const HYPRE_Int n, const HYPRE_Complex * __restrict__ z1, HYPRE_Complex *w1, const HYPRE_Complex * __restrict__ z2, HYPRE_Complex * __restrict__ w2)
{
   const HYPRE_Int size=B/WS;
   volatile __shared__ HYPRE_Complex shmem[size*2];
   HYPRE_Int i = 0;
   HYPRE_Int warp_lane = threadIdx.x & (WS - 1);
   HYPRE_Int warp_id = threadIdx.x / WS;
   HYPRE_Complex sum1 = 0.0, sum2 = 0.0;
   HYPRE_Complex zz1 = 0.0, zz2 = 0.0;

   for (i=threadIdx.x; i<n; i+=blockDim.x)
   {
      zz1 = z1[blockIdx.x*n + i];
      zz2 = z2[blockIdx.x*n + i];
      sum1 += zz1;
      sum2 += zz2;
   }

   // Warp shuffle reduce
#pragma unroll
   for (i = WS>>1; i > 0; i >>= 1)
   {
      sum1 += __shfl_down_sync(WS, sum1, i);
      sum2 += __shfl_down_sync(WS, sum2, i);
   }

   // Combine across warps through shmem
   __syncthreads();
   if (warp_lane==0)
   {
      shmem[warp_id]      = sum1;
      shmem[warp_id+size] = sum2;
   }
   __syncthreads();

   // Put it back in a register to finish the reduction
   if (threadIdx.x<size)
   {
      sum1 = shmem[threadIdx.x];
      sum2 = shmem[threadIdx.x+size];
   }

   // Warp shuffle reduce
#pragma unroll
   for (i = size>>1; i > 0; i >>= 1)
   {
      sum1 += __shfl_down_sync(size, sum1, i);
      sum2 += __shfl_down_sync(size, sum2, i);
   }

   /* Write to global memory */
   if (threadIdx.x==0)
   {
      w1[blockIdx.x] = sum1;
      w2[blockIdx.x] = sum2;
   }
}


struct DivOp : public thrust::unary_function<HYPRE_Int,HYPRE_Int>
{
  HYPRE_Int _n;
  DivOp(HYPRE_Int n) : _n(n)
  {
  }

   __host__ __device__
   HYPRE_Int operator()(HYPRE_Int x) const
   {
     return x/_n;
   }
};

struct ModOp : public thrust::unary_function<HYPRE_Int,HYPRE_Int>
{
  HYPRE_Int _n;
  ModOp(HYPRE_Int n) : _n(n)
  {
  }

   __host__ __device__
   HYPRE_Int operator()(HYPRE_Int x) const
   {
     return x%_n;
   }
};

struct multTupleComponents : public thrust::unary_function<thrust::tuple<HYPRE_Complex, HYPRE_Complex> ,HYPRE_Complex>
{
  __host__ __device__
  HYPRE_Complex operator()(thrust::tuple<HYPRE_Complex, HYPRE_Complex> x) const
  {
    return thrust::get<0>(x)*thrust::get<1>(x);
  }
};


HYPRE_Int
hypreDevice_MassInnerProd(HYPRE_Int k, HYPRE_Int n, HYPRE_Complex *x, HYPRE_Complex *y, HYPRE_Complex *result)
{
   /* trivial case */
   if (n <= 0 || k<=0)
   {
      return hypre_error_flag;
   }

#if 0

   HYPRE_Int * d_keys_out = hypre_CTAlloc(HYPRE_Int,k,HYPRE_MEMORY_DEVICE);
   HYPRE_Complex * d_result = hypre_CTAlloc(HYPRE_Complex,k,HYPRE_MEMORY_DEVICE);

   DivOp divOp(n);
   ModOp modOp(n);
   multTupleComponents mult;

   typedef thrust::counting_iterator<HYPRE_Int> countIter;
   typedef thrust::transform_iterator<ModOp, countIter> transIter;
   typedef thrust::permutation_iterator<HYPRE_Complex *, transIter> permIter;
   typedef thrust::tuple<HYPRE_Complex *, permIter> IterTuple;
   typedef thrust::zip_iterator<IterTuple> zipIter;

   // thrust reduce by key
   HYPRE_THRUST_CALL(reduce_by_key,
                     thrust::make_transform_iterator<DivOp, countIter>(thrust::make_counting_iterator(0),   divOp),
                     thrust::make_transform_iterator<DivOp, countIter>(thrust::make_counting_iterator(k*n), divOp),
                     thrust::make_transform_iterator<multTupleComponents, zipIter>(
                        thrust::make_tuple(y, thrust::make_permutation_iterator(x, thrust::make_transform_iterator(thrust::make_counting_iterator(0), modOp))),
                        mult),
                     d_keys_out,
                     d_result);

   hypre_TMemcpy(result, d_result, HYPRE_Complex, k, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_keys_out,HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_result,HYPRE_MEMORY_DEVICE);

#else

   HYPRE_Int warpSize, dev, numSMs, maxThreads;
#if defined(HYPRE_USING_CUDA)
   struct cudaDeviceProp deviceProp;
   HYPRE_CUDA_CALL( cudaGetDevice(&dev) );
   HYPRE_CUDA_CALL( cudaGetDeviceProperties(&deviceProp, dev) );
#endif

#if defined(HYPRE_USING_HIP)
   hipDeviceProp_t deviceProp;
   HYPRE_HIP_CALL( hipGetDevice(&dev) );
   HYPRE_HIP_CALL( hipGetDeviceProperties(&deviceProp, dev) );
#endif

   // Get the warpSize
   warpSize = deviceProp.warpSize;
   numSMs = deviceProp.multiProcessorCount;
   maxThreads = deviceProp.maxThreadsPerMultiProcessor;

   const HYPRE_Int numThreads = 256;
   const int numBlocks = min(8*(maxThreads/numThreads)*numSMs, (n+numThreads-1)/numThreads);

   HYPRE_Complex * d_result;
   HYPRE_Int totalMem = k*numBlocks+k;

   d_result = hypre_CTAlloc(HYPRE_Complex,totalMem,HYPRE_MEMORY_DEVICE);

   /* Kernel 1 : Initial Reduction */
   const HYPRE_Int shmemSize = sizeof(HYPRE_Complex)*k*(numThreads/warpSize);
   if (warpSize==64)
   {
	   hypreGPUKernel_MassInnerProd_kernel1<numThreads,64><<<numBlocks,numThreads,shmemSize>>>(k, n, x, y, d_result);
	   hypreGPUKernel_MassInnerProd_kernel2<numThreads,64><<<k,numThreads>>>(numBlocks, d_result, d_result+k*numBlocks);
   }
   else
   {
      hypreGPUKernel_MassInnerProd_kernel1<numThreads,32><<<numBlocks,numThreads,shmemSize>>>(k, n, x, y, d_result);
	  hypreGPUKernel_MassInnerProd_kernel2<numThreads,32><<<k,numThreads>>>(numBlocks, d_result, d_result+k*numBlocks);
   }
   hypre_TMemcpy(result, d_result+k*numBlocks, HYPRE_Complex, k, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_result, HYPRE_MEMORY_DEVICE);

#endif
   return hypre_error_flag;
}




HYPRE_Int
hypreDevice_MassDotpTwo(HYPRE_Int k, HYPRE_Int n, HYPRE_Complex *x, HYPRE_Complex *y, HYPRE_Complex *z, HYPRE_Complex *result_x, HYPRE_Complex *result_y)
{
   /* trivial case */
   if (n <= 0 || k<=0)
   {
      return hypre_error_flag;
   }

   HYPRE_Int warpSize, dev, numSMs, maxThreads;

#if defined(HYPRE_USING_CUDA)
   struct cudaDeviceProp deviceProp;
   HYPRE_CUDA_CALL( cudaGetDevice(&dev) );
   HYPRE_CUDA_CALL( cudaGetDeviceProperties(&deviceProp, dev) );
#endif

#if defined(HYPRE_USING_HIP)
   hipDeviceProp_t deviceProp;
   HYPRE_HIP_CALL( hipGetDevice(&dev) );
   HYPRE_HIP_CALL( hipGetDeviceProperties(&deviceProp, dev) );
#endif

   // Get the warpSize
   warpSize = deviceProp.warpSize;
   numSMs = deviceProp.multiProcessorCount;
   maxThreads = deviceProp.maxThreadsPerMultiProcessor;

   const HYPRE_Int numThreads = 256;
   const int numBlocks = min(8*(maxThreads/numThreads)*numSMs, (n+numThreads-1)/numThreads);

   HYPRE_Complex * d_result;
   HYPRE_Int totalMem = 2*(k*numBlocks+k);

   d_result = hypre_CTAlloc(HYPRE_Complex,totalMem,HYPRE_MEMORY_DEVICE);

   HYPRE_Complex * d_result_x = d_result;
   HYPRE_Complex * d_result_y = d_result + totalMem/2;

   /* Kernel 1 : Initial Reduction */
   const HYPRE_Int shmemSize = sizeof(HYPRE_Complex)*k*(numThreads/warpSize)*2;
   if (warpSize==64)
   {
      hypreGPUKernel_MassDotpTwo_kernel1<numThreads,64><<<numBlocks,numThreads,shmemSize>>>(k, n, x, y, z, d_result_x, d_result_y);
	  hypreGPUKernel_MassDotpTwo_kernel2<numThreads,64><<<k,numThreads>>>(numBlocks, d_result_x, d_result_x+k*numBlocks, d_result_y, d_result_y+k*numBlocks);
   }
   else
   {
      hypreGPUKernel_MassDotpTwo_kernel1<numThreads,32><<<numBlocks,numThreads,shmemSize>>>(k, n, x, y, z, d_result_x, d_result_y);
	  hypreGPUKernel_MassDotpTwo_kernel2<numThreads,32><<<k,numThreads>>>(numBlocks, d_result_x, d_result_x+k*numBlocks, d_result_y, d_result_y+k*numBlocks);
   }
   hypre_TMemcpy(result_x, d_result_x + k*numBlocks, HYPRE_Complex, k, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(result_y, d_result_y + k*numBlocks, HYPRE_Complex, k, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_result, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP)
