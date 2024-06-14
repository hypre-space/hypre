/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"
#include <math.h>

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      generic device functions (HYPRE_USING_GPU)
 *      NOTE: This includes device openmp for now
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------
 * hypre_DeviceDataCreate
 *--------------------------------------------------------------------*/

hypre_DeviceData*
hypre_DeviceDataCreate()
{
   hypre_DeviceData *data = hypre_CTAlloc(hypre_DeviceData, 1, HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_SYCL)
   hypre_DeviceDataDevice(data)           = nullptr;
#else
   hypre_DeviceDataDevice(data)           = 0;
#endif
   hypre_DeviceDataComputeStreamNum(data) = 0;

   /* SpMV, SpGeMM, SpTrans: use vendor's lib by default */
#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE) || defined(HYPRE_USING_ONEMKLSPARSE)
   hypre_DeviceDataSpgemmUseVendor(data)  = 1;
   hypre_DeviceDataSpMVUseVendor(data)    = 1;
   hypre_DeviceDataSpTransUseVendor(data) = 1;
#else
   hypre_DeviceDataSpgemmUseVendor(data)  = 0;
   hypre_DeviceDataSpMVUseVendor(data)    = 0;
   hypre_DeviceDataSpTransUseVendor(data) = 0;
#endif
   /* for CUDA, it seems cusparse is slow due to memory allocation inside the transposition */
#if defined(HYPRE_USING_CUDA)
   hypre_DeviceDataSpTransUseVendor(data) = 0;
#endif

   /* hypre SpGEMM parameters */
   const HYPRE_Int  Nsamples   = 64;
   const HYPRE_Real sigma      = 1.0 / hypre_sqrt((HYPRE_Real)(Nsamples - 2.0));
   const HYPRE_Real multfactor = 1.0 / (1.0 - 3.0 * sigma);

   hypre_DeviceDataSpgemmAlgorithm(data)                = 1;
   hypre_DeviceDataSpgemmBinned(data)                   = 0;
   hypre_DeviceDataSpgemmNumBin(data)                   = 0;
   hypre_DeviceDataSpgemmHighestBin(data)[0]            = 0;
   hypre_DeviceDataSpgemmHighestBin(data)[1]            = 0;
   /* 1: naive overestimate, 2: naive underestimate, 3: Cohen's algorithm */
   hypre_DeviceDataSpgemmRownnzEstimateMethod(data)     = 3;
   hypre_DeviceDataSpgemmRownnzEstimateNsamples(data)   = Nsamples;
   hypre_DeviceDataSpgemmRownnzEstimateMultFactor(data) = multfactor;

   /* pmis */
#if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND) || defined(HYPRE_USING_ONEMKLRAND)
   hypre_DeviceDataUseGpuRand(data) = 1;
#else
   hypre_DeviceDataUseGpuRand(data) = 0;
#endif

   /* device pool */
#ifdef HYPRE_USING_DEVICE_POOL
   hypre_DeviceDataCubBinGrowth(data)      = 8u;
   hypre_DeviceDataCubMinBin(data)         = 1u;
   hypre_DeviceDataCubMaxBin(data)         = (hypre_uint) - 1;
   hypre_DeviceDataCubMaxCachedBytes(data) = (size_t) -1;
   hypre_DeviceDataCubDevAllocator(data)   = NULL;
   hypre_DeviceDataCubUvmAllocator(data)   = NULL;
#endif

   return data;
}

/*--------------------------------------------------------------------
 * hypre_DeviceDataDestroy
 *--------------------------------------------------------------------*/

void
hypre_DeviceDataDestroy(hypre_DeviceData *data)
{
   if (!data)
   {
      return;
   }

   hypre_TFree(hypre_DeviceDataReduceBuffer(data),         HYPRE_MEMORY_DEVICE);

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

#if defined(HYPRE_USING_CUSOLVER) || defined(HYPRE_USING_ROCSOLVER)
   if (data->vendor_solver_handle)
   {
#if defined(HYPRE_USING_CUSOLVER)
      HYPRE_CUSOLVER_CALL(cusolverDnDestroy(data->vendor_solver_handle));
#else
      HYPRE_ROCBLAS_CALL(rocblas_destroy_handle(data->vendor_solver_handle));
#endif
   }
#endif // #if defined(HYPRE_USING_CUSOLVER) || defined(HYPRE_USING_ROCSOLVER)

#if defined(HYPRE_USING_CUDA_STREAMS)
   for (HYPRE_Int i = 0; i < HYPRE_MAX_NUM_STREAMS; i++)
   {
      if (data->streams[i])
      {
#if defined(HYPRE_USING_CUDA)
         HYPRE_CUDA_CALL( cudaStreamDestroy(data->streams[i]) );
#elif defined(HYPRE_USING_HIP)
         HYPRE_HIP_CALL( hipStreamDestroy(data->streams[i]) );
#elif defined(HYPRE_USING_SYCL)
         delete data->streams[i];
         data->streams[i] = nullptr;
#endif
      }
   }
#endif

#ifdef HYPRE_USING_DEVICE_POOL
   hypre_DeviceDataCubCachingAllocatorDestroy(data);
#endif

#if defined(HYPRE_USING_SYCL)
   delete data->device;
   data->device = nullptr;
#endif

   hypre_TFree(data, HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------
 * hypre_SyncCudaDevice
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_SyncCudaDevice(hypre_Handle *hypre_handle)
{
#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaDeviceSynchronize() );
#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipDeviceSynchronize() );
#elif defined(HYPRE_USING_SYCL)
   try
   {
      HYPRE_SYCL_CALL( hypre_HandleComputeStream(hypre_handle)->wait_and_throw() );
   }
   catch (sycl::exception const &exc)
   {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
   }
#endif
   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ResetCudaDevice
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ResetCudaDevice(hypre_Handle *hypre_handle)
{
#if defined(HYPRE_USING_CUDA)
   cudaDeviceReset();
#elif defined(HYPRE_USING_HIP)
   hipDeviceReset();
#endif
   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_SyncComputeStream_core
 *
 * Synchronize the Hypre compute stream
 *
 * action: 0: set sync stream to false
 *         1: set sync stream to true
 *         2: restore sync stream to default
 *         3: return the current value of cuda_compute_stream_sync
 *         4: sync stream based on cuda_compute_stream_sync
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_SyncComputeStream_core(HYPRE_Int     action,
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
         if (hypre_HandleDefaultExecPolicy(hypre_handle) == HYPRE_EXEC_DEVICE && cuda_compute_stream_sync)
         {
#if defined(HYPRE_USING_CUDA)
            HYPRE_CUDA_CALL( cudaStreamSynchronize(hypre_HandleComputeStream(hypre_handle)) );
#elif defined(HYPRE_USING_HIP)
            HYPRE_HIP_CALL( hipStreamSynchronize(hypre_HandleComputeStream(hypre_handle)) );
#elif defined(HYPRE_USING_SYCL)
            HYPRE_SYCL_CALL( hypre_HandleComputeStream(hypre_handle)->ext_oneapi_submit_barrier() );
#endif
         }
         break;
      default:
         hypre_printf("hypre_SyncComputeStream_core invalid action\n");
         hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_SetSyncCudaCompute
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_SetSyncCudaCompute(HYPRE_Int action)
{
   /* convert to 1/0 */
   action = action != 0;
   hypre_SyncComputeStream_core(action, NULL, NULL);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_RestoreSyncCudaCompute
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_RestoreSyncCudaCompute()
{
   hypre_SyncComputeStream_core(2, NULL, NULL);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_GetSyncCudaCompute
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_GetSyncCudaCompute(HYPRE_Int *cuda_compute_stream_sync_ptr)
{
   hypre_SyncComputeStream_core(3, NULL, cuda_compute_stream_sync_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_SyncComputeStream
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_SyncComputeStream(hypre_Handle *hypre_handle)
{
   hypre_SyncComputeStream_core(4, hypre_handle, NULL);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ForceSyncComputeStream
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ForceSyncComputeStream(hypre_Handle *hypre_handle)
{
   HYPRE_Int sync_stream;
   hypre_GetSyncCudaCompute(&sync_stream);
   hypre_SetSyncCudaCompute(1);
   hypre_SyncComputeStream_core(4, hypre_handle, NULL);
   hypre_SetSyncCudaCompute(sync_stream);

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      generic device functions (cuda/hip/sycl)
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------
 * hypre_DeviceDataComputeStream
 *--------------------------------------------------------------------*/

/* CUDA/HIP stream */
#if defined(HYPRE_USING_CUDA)
cudaStream_t
#elif defined(HYPRE_USING_HIP)
hipStream_t
#elif defined(HYPRE_USING_SYCL)
sycl::queue*
#endif
hypre_DeviceDataComputeStream(hypre_DeviceData *data)
{
   return hypre_DeviceDataStream(data, hypre_DeviceDataComputeStreamNum(data));
}

/*--------------------------------------------------------------------
 * hypre_DeviceDataStream
 *--------------------------------------------------------------------*/

#if defined(HYPRE_USING_CUDA)
cudaStream_t
#elif defined(HYPRE_USING_HIP)
hipStream_t
#elif defined(HYPRE_USING_SYCL)
sycl::queue*
#endif
hypre_DeviceDataStream(hypre_DeviceData *data, HYPRE_Int i)
{
#if defined(HYPRE_USING_CUDA)
   cudaStream_t stream = 0;
#elif defined(HYPRE_USING_HIP)
   hipStream_t stream = 0;
#elif defined(HYPRE_USING_SYCL)
   sycl::queue *stream = NULL;
#endif

#if defined(HYPRE_USING_CUDA_STREAMS)
   if (i >= HYPRE_MAX_NUM_STREAMS)
   {
      /* return the default stream, i.e., the NULL stream */
      /*
      hypre_printf("device stream %d exceeds the max number %d\n",
                   i, HYPRE_MAX_NUM_STREAMS);
      */
      return NULL;
   }

   if (data->streams[i])
   {
      return data->streams[i];
   }

#if defined(HYPRE_USING_CUDA)
   //HYPRE_CUDA_CALL(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
   HYPRE_CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamDefault));
#elif defined(HYPRE_USING_SYCL)
   auto sycl_asynchandler = [] (sycl::exception_list exceptions)
   {
      for (std::exception_ptr const& e : exceptions)
      {
         try
         {
            std::rethrow_exception(e);
         }
         catch (sycl::exception const& ex)
         {
            std::cout << "Caught asynchronous SYCL exception:" << std::endl
                      << ex.what() << ", SYCL code: " << ex.code() << std::endl;
         }
      }
   };

   if (!data->device)
   {
      HYPRE_DeviceInitialize();
   }
   sycl::device* sycl_device = data->device;
   sycl::context sycl_ctxt   = sycl::context(*sycl_device, sycl_asynchandler);
   stream = new sycl::queue(sycl_ctxt, *sycl_device, sycl::property_list{sycl::property::queue::in_order{}});
#endif

   data->streams[i] = stream;
#endif

   return stream;
}

/*--------------------------------------------------------------------
 * hypre_GetDefaultDeviceBlockDimension
 *--------------------------------------------------------------------*/

dim3
hypre_GetDefaultDeviceBlockDimension()
{
#if defined(HYPRE_USING_SYCL)
   dim3 bDim(1, 1, hypre_HandleDeviceMaxWorkGroupSize(hypre_handle()));
#else
   dim3 bDim(HYPRE_1D_BLOCK_SIZE, 1, 1);
#endif

   return bDim;
}

/*--------------------------------------------------------------------
 * hypre_GetDefaultDeviceGridDimension
 *--------------------------------------------------------------------*/

dim3
hypre_GetDefaultDeviceGridDimension( HYPRE_Int   n,
                                     const char *granularity,
                                     dim3        bDim )
{
   HYPRE_Int num_blocks = 0;
#if defined(HYPRE_USING_SYCL)
   HYPRE_Int num_threads_per_block = bDim.get(0) * bDim.get(1) * bDim.get(2);
#else
   HYPRE_Int num_threads_per_block = bDim.x * bDim.y * bDim.z;
#endif

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

   dim3 gDim = hypre_dim3(num_blocks);

   return gDim;
}

/*--------------------------------------------------------------------
 * hypre_dim3
 * NOTE: these functions are necessary due to different linearization
 * procedures between cuda/hip and sycl
 *--------------------------------------------------------------------*/

dim3
hypre_dim3(HYPRE_Int x)
{
#if defined(HYPRE_USING_SYCL)
   dim3 d(1, 1, x);
#else
   dim3 d(x);
#endif
   return d;
}

dim3
hypre_dim3(HYPRE_Int x, HYPRE_Int y)
{
#if defined(HYPRE_USING_SYCL)
   dim3 d(1, y, x);
#else
   dim3 d(x, y);
#endif
   return d;
}

dim3
hypre_dim3(HYPRE_Int x, HYPRE_Int y, HYPRE_Int z)
{
#if defined(HYPRE_USING_SYCL)
   dim3 d(z, y, x);
#else
   dim3 d(x, y, z);
#endif
   return d;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_ArrayToArrayOfPtrs
 *--------------------------------------------------------------------------*/

template <typename T>
__global__ void
hypreGPUKernel_ArrayToArrayOfPtrs( hypre_DeviceItem  &item,
                                   HYPRE_Int          n,
                                   HYPRE_Int          m,
                                   T                 *data,
                                   T                **data_aop )
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < n)
   {
      data_aop[i] = &data[i * m];
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_ArrayToArrayOfPtrs
 *--------------------------------------------------------------------*/

template <typename T>
HYPRE_Int
hypreDevice_ArrayToArrayOfPtrs(HYPRE_Int n, HYPRE_Int m, T *data, T **data_aop)
{
   /* Trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_ArrayToArrayOfPtrs, gDim, bDim, n, m, data, data_aop);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreDevice_ComplexArrayToArrayOfPtrs
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_ComplexArrayToArrayOfPtrs(HYPRE_Int       n,
                                      HYPRE_Int       m,
                                      HYPRE_Complex  *data,
                                      HYPRE_Complex **data_aop)
{
   return hypreDevice_ArrayToArrayOfPtrs(n, m, data, data_aop);
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_IVAXPY
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_IVAXPY( hypre_DeviceItem &item, HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x,
                       HYPRE_Complex *y)
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   if (i < n)
   {
      y[i] += x[i] / a[i];
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_IVAXPY
 *
 * Inverse Vector AXPY: y[i] = x[i] / a[i] + y[i]
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_IVAXPY(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x, HYPRE_Complex *y)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_IVAXPY, gDim, bDim, n, a, x, y );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_IVAXPYMarked
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_IVAXPYMarked( hypre_DeviceItem &item,
                             HYPRE_Int         n,
                             HYPRE_Complex    *a,
                             HYPRE_Complex    *x,
                             HYPRE_Complex    *y,
                             HYPRE_Int        *marker,
                             HYPRE_Int         marker_val)
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   if (i < n)
   {
      if (marker[i] == marker_val)
      {
         y[i] += x[i] / a[i];
      }
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_IVAXPYMarked
 *
 * Inverse Vector AXPY: y[i] = x[i] / a[i] + y[i]
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_IVAXPYMarked( HYPRE_Int      n,
                          HYPRE_Complex *a,
                          HYPRE_Complex *x,
                          HYPRE_Complex *y,
                          HYPRE_Int     *marker,
                          HYPRE_Int      marker_val )
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_IVAXPYMarked, gDim, bDim, n, a, x, y, marker, marker_val );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_IVAMXPMY
 *
 * Device kernel for hypreDevice_IVAMXPMY. The template argument MM tells
 * the maximum number of vectors in the unrolled loop
 *--------------------------------------------------------------------------*/

template <HYPRE_Int MM>
__global__ void
hypreGPUKernel_IVAMXPMY( hypre_DeviceItem &item,
                         HYPRE_Int         m,
                         HYPRE_Int         n,
                         HYPRE_Complex    *a,
                         HYPRE_Complex    *x,
                         HYPRE_Complex    *y)
{
   HYPRE_Int     i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   HYPRE_Int     j;
   HYPRE_Complex val;

   if (i < n)
   {
      val = 1.0 / a[i];

      if (MM > 0)
      {
#pragma unroll
         for (j = 0; j < MM; j++)
         {
            y[i + j * n] += x[i + j * n] * val;
         }
      }
      else
      {
         /* Generic case */
         for (j = 0; j < m; j++)
         {
            y[i + j * n] += x[i + j * n] * val;
         }
      }
   }
}

/*--------------------------------------------------------------------------
 * hypreDevice_IVAMXPMY
 *
 * Inverse Vector AXPY for m vectors x and y of size n stored column-wise:
 *
 *   y[i +       0] += x[i +       0] / a[i]
 *   y[i +       n] += x[i +       n] / a[i]
 *     ...           ...
 *   y[i + (m-1)*n] += x[i + (m-1)*n] / a[i]
 *
 * Note: does not work for row-wise multivectors
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_IVAMXPMY( HYPRE_Int       m,
                      HYPRE_Int       n,
                      HYPRE_Complex  *a,
                      HYPRE_Complex  *x,
                      HYPRE_Complex  *y)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n, "thread", bDim);

   switch (m)
   {
      case 1:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_IVAXPY, gDim, bDim, n, a, x, y );
         break;

      case 2:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_IVAMXPMY<2>, gDim, bDim, m, n, a, x, y );
         break;

      case 3:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_IVAMXPMY<3>, gDim, bDim, m, n, a, x, y );
         break;

      case 4:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_IVAMXPMY<4>, gDim, bDim, m, n, a, x, y );
         break;

      default:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_IVAMXPMY<0>, gDim, bDim, m, n, a, x, y );
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreDevice_CsrRowPtrsToIndices
 *--------------------------------------------------------------------*/

HYPRE_Int*
hypreDevice_CsrRowPtrsToIndices( HYPRE_Int  nrows,
                                 HYPRE_Int  nnz,
                                 HYPRE_Int *d_row_ptr )
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

#if defined(HYPRE_USING_SYCL)

/*--------------------------------------------------------------------
 * hypreSYCLKernel_ScatterRowPtr
 *--------------------------------------------------------------------*/

void
hypreSYCLKernel_ScatterRowPtr( hypre_DeviceItem &item,
                               HYPRE_Int         nrows,
                               HYPRE_Int        *d_row_ptr,
                               HYPRE_Int        *d_row_ind )
{
   HYPRE_Int i = (HYPRE_Int) item.get_global_linear_id();

   if (i < nrows)
   {
      HYPRE_Int row_start = d_row_ptr[i];
      HYPRE_Int row_end = d_row_ptr[i + 1];
      if (row_start != row_end)
      {
         d_row_ind[row_start] = i;
      }
   }
}
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
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
#endif

/*--------------------------------------------------------------------
 * hypreDevice_CsrRowPtrsToIndices_v2
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_CsrRowPtrsToIndices_v2( HYPRE_Int  nrows,
                                    HYPRE_Int  nnz,
                                    HYPRE_Int *d_row_ptr,
                                    HYPRE_Int *d_row_ind )
{
   /* trivial case */
   if (nrows <= 0 || nnz <= 0)
   {
      return hypre_error_flag;
   }
#if defined(HYPRE_USING_SYCL)
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "thread", bDim);
   HYPRE_ONEDPL_CALL( std::fill, d_row_ind, d_row_ind + nnz, 0 );
   HYPRE_GPU_LAUNCH( hypreSYCLKernel_ScatterRowPtr, gDim, bDim, nrows, d_row_ptr, d_row_ind );
   HYPRE_ONEDPL_CALL( std::inclusive_scan, d_row_ind, d_row_ind + nnz, d_row_ind,
                      oneapi::dpl::maximum<HYPRE_Int>());
#else

   hypre_GpuProfilingPushRange("CsrRowPtrsToIndices");
   HYPRE_THRUST_CALL( fill, d_row_ind, d_row_ind + nnz, 0 );
   HYPRE_THRUST_CALL( scatter_if,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(nrows),
                      d_row_ptr,
                      thrust::make_transform_iterator( thrust::make_zip_iterator(thrust::make_tuple(d_row_ptr,
                                                                                                    d_row_ptr + 1)),
                                                       hypre_empty_row_functor() ),
                      d_row_ind );
   HYPRE_THRUST_CALL( inclusive_scan, d_row_ind, d_row_ind + nnz, d_row_ind,
                      thrust::maximum<HYPRE_Int>());
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreDevice_CsrRowIndicesToPtrs
 *--------------------------------------------------------------------*/

HYPRE_Int*
hypreDevice_CsrRowIndicesToPtrs( HYPRE_Int  nrows,
                                 HYPRE_Int  nnz,
                                 HYPRE_Int *d_row_ind )
{
   HYPRE_Int *d_row_ptr = hypre_TAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowIndicesToPtrs_v2(nrows, nnz, d_row_ind, d_row_ptr);

   return d_row_ptr;
}

/*--------------------------------------------------------------------
 * hypreDevice_CsrRowIndicesToPtrs_v2
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_CsrRowIndicesToPtrs_v2( HYPRE_Int  nrows,
                                    HYPRE_Int  nnz,
                                    HYPRE_Int *d_row_ind,
                                    HYPRE_Int *d_row_ptr )
{
#if defined(HYPRE_USING_SYCL)
   /* WM: if nnz <= 0, then dpl::lower_bound is a no-op, which means we still need to zero out the row pointer */
   /* Note that this is different from thrust's behavior, where lower_bound zeros out the row pointer when nnz = 0 */
   if (nnz <= 0)
   {
      hypre_Memset(d_row_ptr, 0, (nrows + 1) * sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
      return hypre_error_flag;
   }
   oneapi::dpl::counting_iterator<HYPRE_Int> count(0);
   HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                      d_row_ind, d_row_ind + nnz,
                      count,
                      count + nrows + 1,
                      d_row_ptr);
#else
   hypre_GpuProfilingPushRange("CSRIndicesToPtrs");
   HYPRE_THRUST_CALL( lower_bound,
                      d_row_ind, d_row_ind + nnz,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(nrows + 1),
                      d_row_ptr);
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_GetRowNnz
 *
 * Get NNZ of each row in d_row_indices and store the results in d_rownnz
 * All pointers are device pointers.
 * d_rownnz can be the same as d_row_indices.
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_GetRowNnz( hypre_DeviceItem &item,
                          HYPRE_Int         nrows,
                          HYPRE_Int        *d_row_indices,
                          HYPRE_Int        *d_diag_ia,
                          HYPRE_Int        *d_offd_ia,
                          HYPRE_Int        *d_rownnz )
{
   const HYPRE_Int global_thread_id = hypre_gpu_get_grid_thread_id<1, 1>(item);

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

      d_rownnz[global_thread_id] =
         read_only_load(&d_diag_ia[i + 1]) - read_only_load(&d_diag_ia[i]) +
         read_only_load(&d_offd_ia[i + 1]) - read_only_load(&d_offd_ia[i]);
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_GetRowNnz
 *
 * Note: (d_row_indices == NULL) means d_row_indices = [0,1,...,nrows-1]
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_GetRowNnz( HYPRE_Int  nrows,
                       HYPRE_Int *d_row_indices,
                       HYPRE_Int *d_diag_ia,
                       HYPRE_Int *d_offd_ia,
                       HYPRE_Int *d_rownnz )
{
   const dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   const dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "thread", bDim);

   /* trivial case */
   if (nrows <= 0)
   {
      return hypre_error_flag;
   }

   HYPRE_GPU_LAUNCH( hypreGPUKernel_GetRowNnz, gDim, bDim, nrows, d_row_indices,
                     d_diag_ia, d_offd_ia, d_rownnz );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreDevice_IntegerInclusiveScan
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_IntegerInclusiveScan( HYPRE_Int  n,
                                  HYPRE_Int *d_i )
{
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL(std::inclusive_scan, d_i, d_i + n, d_i);
#else
   HYPRE_THRUST_CALL(inclusive_scan, d_i, d_i + n, d_i);
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_CopyParCSRRows
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CopyParCSRRows( hypre_DeviceItem  &item,
                               HYPRE_Int          nrows,
                               HYPRE_Int         *d_row_indices,
                               HYPRE_Int          has_offd,
                               HYPRE_BigInt       first_col,
                               HYPRE_BigInt      *d_col_map_offd_A,
                               HYPRE_Int         *d_diag_i,
                               HYPRE_Int         *d_diag_j,
                               HYPRE_Complex     *d_diag_a,
                               HYPRE_Int         *d_offd_i,
                               HYPRE_Int         *d_offd_j,
                               HYPRE_Complex     *d_offd_a,
                               HYPRE_Int         *d_ib,
                               HYPRE_BigInt      *d_jb,
                               HYPRE_Complex     *d_ab )
{
   const HYPRE_Int global_warp_id = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (global_warp_id >= nrows)
   {
      return;
   }

   /* lane id inside the warp */
   const HYPRE_Int lane_id = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int i, j = 0, k = 0, p, row, istart, iend, bstart;

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
   istart = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 1);
   bstart = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, k, 0);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += HYPRE_WARP_SIZE)
   {
      d_jb[p + i] = read_only_load(d_diag_j + i) + first_col;
      if (d_ab)
      {
         d_ab[p + i] = read_only_load(d_diag_a + i);
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
   istart = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 1);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += HYPRE_WARP_SIZE)
   {
      if (d_col_map_offd_A)
      {
         d_jb[p + i] = d_col_map_offd_A[read_only_load(d_offd_j + i)];
      }
      else
      {
         d_jb[p + i] = -1 - read_only_load(d_offd_j + i);
      }

      if (d_ab)
      {
         d_ab[p + i] = read_only_load(d_offd_a + i);
      }
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_CopyParCSRRows
 *
 * B = A(row_indices, :)
 * Note: d_ib is an input vector that contains row ptrs,
 *       i.e., start positions where to put the rows in d_jb and d_ab.
 *       The col indices in B are global indices, i.e., BigJ
 *       of length (nrows + 1) or nrow (without the last entry, nnz)
 * Special cases:
 *    if d_row_indices == NULL, it means d_row_indices=[0,1,...,nrows-1]
 *    If col_map_offd_A == NULL, use (-1 - d_offd_j) as column id
 *    If nrows == 1 and d_ib == NULL, it means d_ib[0] = 0
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_CopyParCSRRows( HYPRE_Int      nrows,
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
                            HYPRE_Complex *d_ab )
{
   /* trivial case */
   if (nrows <= 0)
   {
      return hypre_error_flag;
   }

   hypre_assert(!(nrows > 1 && d_ib == NULL));

   const dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   const dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   /*
   if (job == 2)
   {
   }
   */

   HYPRE_GPU_LAUNCH( hypreGPUKernel_CopyParCSRRows, gDim, bDim,
                     nrows, d_row_indices, has_offd, first_col, d_col_map_offd_A,
                     d_diag_i, d_diag_j, d_diag_a,
                     d_offd_i, d_offd_j, d_offd_a,
                     d_ib, d_jb, d_ab );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreDevice_IntegerExclusiveScan
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_IntegerExclusiveScan( HYPRE_Int  n,
                                  HYPRE_Int *d_i )
{
#if defined(HYPRE_USING_SYCL)
   /* WM: todo - this is a workaround since oneDPL's exclusive_scan gives incorrect results when doing the scan in place */
   HYPRE_Int *tmp = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
   /* HYPRE_ONEDPL_CALL(std::exclusive_scan, d_i, d_i + n, d_i, 0); */
   HYPRE_ONEDPL_CALL(std::exclusive_scan, d_i, d_i + n, tmp, 0);
   hypre_TMemcpy(d_i, tmp, HYPRE_Int, n, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TFree(tmp, HYPRE_MEMORY_DEVICE);
#else
   HYPRE_THRUST_CALL(exclusive_scan, d_i, d_i + n, d_i);
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreDevice_StableSortByTupleKey
 *
 * https://github.com/OrangeOwlSolutions/Thrust/blob/master/Sort_by_key_with_tuple_key.cu
 *
 * opt: 0, (a,b) < (a',b') iff a < a' or (a = a' and  b  <  b') [normal tupe comp]
 *      1, (a,b) < (a',b') iff a < a' or (a = a' and |b| > |b'|) [used in dropping small entries]
 *      2, (a,b) < (a',b') iff a < a' or (a = a' and (b == a or b < b') and b' != a') [used in putting diagonal first]
 *--------------------------------------------------------------------*/

template <typename T1, typename T2, typename T3>
HYPRE_Int
hypreDevice_StableSortByTupleKey( HYPRE_Int N,
                                  T1 *keys1, T2 *keys2, T3 *vals,
                                  HYPRE_Int opt )
{
#if defined(HYPRE_USING_SYCL)
   auto zipped_begin = oneapi::dpl::make_zip_iterator(keys1, keys2, vals);

   if (opt == 0)
   {
      HYPRE_ONEDPL_CALL(std::stable_sort,
                        zipped_begin,
                        zipped_begin + N,
                        std::less< std::tuple<T1, T2, T3> >());
   }
   else if (opt == 1)
   {
      HYPRE_ONEDPL_CALL(std::stable_sort,
                        zipped_begin,
                        zipped_begin + N,
                        TupleComp2<T1, T2, T3>());
   }
   else if (opt == 2)
   {
      HYPRE_ONEDPL_CALL(std::stable_sort,
                        zipped_begin,
                        zipped_begin + N,
                        TupleComp3<T1, T2, T3>());
   }
#else
   hypre_GpuProfilingPushRange("StableSortByTupleKey");
   auto begin_keys = thrust::make_zip_iterator(thrust::make_tuple(keys1,     keys2));
   auto end_keys   = thrust::make_zip_iterator(thrust::make_tuple(keys1 + N, keys2 + N));

   if (opt == 0)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key,
                        begin_keys,
                        end_keys,
                        vals,
                        thrust::less< thrust::tuple<T1, T2> >());
   }
   else if (opt == 1)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key,
                        begin_keys,
                        end_keys,
                        vals,
                        TupleComp2<T1, T2>());
   }
   else if (opt == 2)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key,
                        begin_keys,
                        end_keys,
                        vals,
                        TupleComp3<T1, T2>());
   }
   hypre_GpuProfilingPopRange();
#endif
   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N,
                                                    HYPRE_Int *keys1, HYPRE_Int *keys2,
                                                    HYPRE_Int *vals, HYPRE_Int opt);
template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N,
                                                    HYPRE_Int *keys1, HYPRE_Real *keys2,
                                                    HYPRE_Int *vals, HYPRE_Int opt);
template HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N,
                                                    HYPRE_Int *keys1, HYPRE_Int *keys2,
                                                    HYPRE_Complex *vals, HYPRE_Int opt);

/*--------------------------------------------------------------------
 * hypreDevice_ReduceByTupleKey
 *--------------------------------------------------------------------*/

template <typename T1, typename T2, typename T3>
HYPRE_Int
hypreDevice_ReduceByTupleKey( HYPRE_Int N,
                              T1 *keys1_in,  T2 *keys2_in,  T3 *vals_in,
                              T1 *keys1_out, T2 *keys2_out, T3 *vals_out )
{
#if defined(HYPRE_USING_SYCL)
   auto begin_keys_in  = oneapi::dpl::make_zip_iterator(keys1_in,  keys2_in );
   auto begin_keys_out = oneapi::dpl::make_zip_iterator(keys1_out, keys2_out);
   std::equal_to< std::tuple<T1, T2> > pred;
   std::plus<T3> func;

   auto new_end = HYPRE_ONEDPL_CALL(oneapi::dpl::reduce_by_segment,
                                    begin_keys_in,
                                    begin_keys_in + N,
                                    vals_in,
                                    begin_keys_out,
                                    vals_out,
                                    pred,
                                    func);
#else
   auto begin_keys_in  = thrust::make_zip_iterator(thrust::make_tuple(keys1_in,     keys2_in    ));
   auto end_keys_in    = thrust::make_zip_iterator(thrust::make_tuple(keys1_in + N, keys2_in + N));
   auto begin_keys_out = thrust::make_zip_iterator(thrust::make_tuple(keys1_out,    keys2_out   ));
   thrust::equal_to< thrust::tuple<T1, T2> > pred;
   thrust::plus<T3> func;

   auto new_end = HYPRE_THRUST_CALL(reduce_by_key,
                                    begin_keys_in,
                                    end_keys_in,
                                    vals_in,
                                    begin_keys_out,
                                    vals_out,
                                    pred,
                                    func);
#endif

   return new_end.second - vals_out;
}

template HYPRE_Int hypreDevice_ReduceByTupleKey(HYPRE_Int      N,
                                                HYPRE_Int     *keys1_in,
                                                HYPRE_Int     *keys2_in,
                                                HYPRE_Complex *vals_in,
                                                HYPRE_Int     *keys1_out,
                                                HYPRE_Int     *keys2_out,
                                                HYPRE_Complex *vals_out);

/*--------------------------------------------------------------------
 * hypreGPUKernel_ScatterConstant
 *--------------------------------------------------------------------*/

template <typename T>
__global__ void
hypreGPUKernel_ScatterConstant(hypre_DeviceItem &item,
                               T                *x,
                               HYPRE_Int         n,
                               HYPRE_Int        *map,
                               T                 v)
{
   HYPRE_Int global_thread_id = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (global_thread_id < n)
   {
      x[map[global_thread_id]] = v;
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_ScatterConstant
 *
 * x[map[i]] = v
 * n is length of map
 * TODO: thrust?
 *--------------------------------------------------------------------*/

template <typename T>
HYPRE_Int
hypreDevice_ScatterConstant(T *x, HYPRE_Int n, HYPRE_Int *map, T v)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_ScatterConstant, gDim, bDim, x, n, map, v );

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_ScatterConstant(HYPRE_Int     *x, HYPRE_Int n, HYPRE_Int *map,
                                               HYPRE_Int     v);
template HYPRE_Int hypreDevice_ScatterConstant(HYPRE_Complex *x, HYPRE_Int n, HYPRE_Int *map,
                                               HYPRE_Complex v);

/*--------------------------------------------------------------------
 * hypreGPUKernel_ScatterAddTrivial
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_ScatterAddTrivial(hypre_DeviceItem &item,
                                 HYPRE_Int         n,
                                 HYPRE_Real       *x,
                                 HYPRE_Int        *map,
                                 HYPRE_Real       *y)
{
   for (HYPRE_Int i = 0; i < n; i++)
   {
      x[map[i]] += y[i];
   }
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_ScatterAdd
 *
 * x[map[i]] += y[i], same index cannot appear more than once in map
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_ScatterAdd(hypre_DeviceItem &item,
                          HYPRE_Int         n,
                          HYPRE_Real       *x,
                          HYPRE_Int        *map,
                          HYPRE_Real       *y)
{
   HYPRE_Int global_thread_id = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (global_thread_id < n)
   {
      x[map[global_thread_id]] += y[global_thread_id];
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_GenScatterAdd
 *
 * Generalized Scatter-and-Add
 *
 * for i = 0 : ny-1, x[map[i]] += y[i];
 *
 * Note: An index is allowed to appear more than once in map
 *       Content in y will be destroyed
 *       When work != NULL, work is at least of size
 *          [2 * sizeof(HYPRE_Int) + sizeof(HYPRE_Complex)] * ny
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_GenScatterAdd( HYPRE_Real  *x,
                           HYPRE_Int    ny,
                           HYPRE_Int   *map,
                           HYPRE_Real  *y,
                           char        *work)
{
   if (ny <= 0)
   {
      return hypre_error_flag;
   }

   if (ny <= 2)
   {
      /* trivial cases, n = 1, 2 */
      dim3 bDim = hypre_dim3(1);
      dim3 gDim = hypre_dim3(1);
      HYPRE_GPU_LAUNCH( hypreGPUKernel_ScatterAddTrivial, gDim, bDim, ny, x, map, y );
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

#if defined(HYPRE_USING_SYCL)
      auto zipped_begin = oneapi::dpl::make_zip_iterator(map2, y);
      HYPRE_ONEDPL_CALL(std::sort, zipped_begin, zipped_begin + ny,
      [](auto lhs, auto rhs) {return std::get<0>(lhs) < std::get<0>(rhs);});

      // WM: todo - ABB: The below code has issues because of name mangling issues,
      //       similar to https://github.com/oneapi-src/oneDPL/pull/166
      //       https://github.com/oneapi-src/oneDPL/issues/507
      //       should be fixed by now?
      /* auto new_end = HYPRE_ONEDPL_CALL( oneapi::dpl::reduce_by_segment, */
      /*                                   map2, */
      /*                                   map2 + ny, */
      /*                                   y, */
      /*                                   reduced_map, */
      /*                                   reduced_y ); */
      std::pair<HYPRE_Int*, HYPRE_Real*> new_end = oneapi::dpl::reduce_by_segment(
                                                      oneapi::dpl::execution::make_device_policy<class devutils>(*hypre_HandleComputeStream(
                                                               hypre_handle())), map2, map2 + ny, y, reduced_map, reduced_y );
#else
      HYPRE_THRUST_CALL(sort_by_key, map2, map2 + ny, y);

      thrust::pair<HYPRE_Int*, HYPRE_Real*> new_end = HYPRE_THRUST_CALL( reduce_by_key,
                                                                         map2,
                                                                         map2 + ny,
                                                                         y,
                                                                         reduced_map,
                                                                         reduced_y );
#endif

      reduced_n = new_end.first - reduced_map;

      hypre_assert(reduced_n == new_end.second - reduced_y);

      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(reduced_n, "thread", bDim);

      HYPRE_GPU_LAUNCH( hypreGPUKernel_ScatterAdd, gDim, bDim,
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

/*--------------------------------------------------------------------
 * hypreGPUKernel_Axpyzn
 *--------------------------------------------------------------------*/

template<typename T>
__global__ void
hypreGPUKernel_Axpyzn( hypre_DeviceItem &item,
                       HYPRE_Int         n,
                       T                *x,
                       T                *y,
                       T                *z,
                       T                 a,
                       T                 b )
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < n)
   {
      z[i] = a * x[i] + b * y[i];
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_Axpyzn
 *--------------------------------------------------------------------*/

template<typename T>
HYPRE_Int
hypreDevice_Axpyzn(HYPRE_Int n, T *d_x, T *d_y, T *d_z, T a, T b)
{
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_Axpyzn, gDim, bDim, n, d_x, d_y, d_z, a, b );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreDevice_ComplexAxpyn
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_ComplexAxpyn( HYPRE_Complex  *d_x,
                          size_t          n,
                          HYPRE_Complex  *d_y,
                          HYPRE_Complex  *d_z,
                          HYPRE_Complex   a )
{
   return hypreDevice_Axpyzn((HYPRE_Int) n, d_x, d_y, d_z, a, (HYPRE_Complex) 1.0);
}

/*--------------------------------------------------------------------
 * hypreDevice_IntAxpyn
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_IntAxpyn( HYPRE_Int *d_x,
                      size_t     n,
                      HYPRE_Int *d_y,
                      HYPRE_Int *d_z,
                      HYPRE_Int  a )
{
   return hypreDevice_Axpyzn((HYPRE_Int) n, d_x, d_y, d_z, a, (HYPRE_Int) 1);
}

/*--------------------------------------------------------------------
 * hypreDevice_BigIntAxpyn
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_BigIntAxpyn( HYPRE_BigInt *d_x,
                         size_t        n,
                         HYPRE_BigInt *d_y,
                         HYPRE_BigInt *d_z,
                         HYPRE_BigInt  a )
{
   return hypreDevice_Axpyzn((HYPRE_Int) n, d_x, d_y, d_z, a, (HYPRE_BigInt) 1);
}

/*--------------------------------------------------------------------
 * hypreDevice_ComplexAxpyzn
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_ComplexAxpyzn( HYPRE_Int       n,
                           HYPRE_Complex  *d_x,
                           HYPRE_Complex  *d_y,
                           HYPRE_Complex  *d_z,
                           HYPRE_Complex   a,
                           HYPRE_Complex   b )
{
   return hypreDevice_Axpyzn(n, d_x, d_y, d_z, a, b);
}

#if defined(HYPRE_USING_CURAND)

/*--------------------------------------------------------------------
 * hypre_DeviceDataCurandGenerator
 *--------------------------------------------------------------------*/

curandGenerator_t
hypre_DeviceDataCurandGenerator(hypre_DeviceData *data)
{
   if (data->curand_generator)
   {
      return data->curand_generator;
   }

   curandGenerator_t gen;
   HYPRE_CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
   HYPRE_CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
   HYPRE_CURAND_CALL( curandSetGeneratorOffset(gen, 0) );
   HYPRE_CURAND_CALL( curandSetStream(gen, hypre_DeviceDataComputeStream(data)) );

   data->curand_generator = gen;

   return gen;
}

/*--------------------------------------------------------------------
 * hypre_CurandUniform_core
 *
 * T = float or hypre_double
 *--------------------------------------------------------------------*/

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

   hypre_GpuProfilingPushRange("RandGen");

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

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}
#endif /* #if defined(HYPRE_USING_CURAND) */

#if defined(HYPRE_USING_ROCRAND)

/*--------------------------------------------------------------------
 * hypre_DeviceDataCurandGenerator
 *--------------------------------------------------------------------*/

rocrand_generator
hypre_DeviceDataCurandGenerator(hypre_DeviceData *data)
{
   if (data->curand_generator)
   {
      return data->curand_generator;
   }

   rocrand_generator gen;
   HYPRE_ROCRAND_CALL( rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT) );
   HYPRE_ROCRAND_CALL( rocrand_set_seed(gen, 1234ULL) );
   HYPRE_ROCRAND_CALL( rocrand_set_offset(gen, 0) );
   HYPRE_ROCRAND_CALL( rocrand_set_stream(gen, hypre_DeviceDataComputeStream(data)) );

   data->curand_generator = gen;

   return gen;
}

/*--------------------------------------------------------------------
 * hypre_CurandUniform_core
 *--------------------------------------------------------------------*/

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

#if defined(HYPRE_USING_ONEMKLRAND)

/*--------------------------------------------------------------------
 * hypre_CurandUniform_core
 *
 * T = float or hypre_double
 *--------------------------------------------------------------------*/

template <typename T>
HYPRE_Int
hypre_CurandUniform_core( HYPRE_Int          n,
                          T                 *urand,
                          HYPRE_Int          set_seed,
                          hypre_ulonglongint seed,
                          HYPRE_Int          set_offset,
                          hypre_ulonglongint offset)
{
   /* WM: if n is zero, onemkl rand throws an error */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   static_assert(std::is_same_v<T, float> || std::is_same_v<T, hypre_double>,
                 "oneMKL: rng/uniform: T is not supported");

   oneapi::mkl::rng::default_engine engine(*hypre_HandleComputeStream(hypre_handle()), seed);
   oneapi::mkl::rng::uniform<T> distribution(0.0 + offset, 1.0 + offset);
   oneapi::mkl::rng::generate(distribution, engine, n, urand).wait_and_throw();

   return hypre_error_flag;
}
#endif /* #if defined(HYPRE_USING_ONEMKLRAND) */

#if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND) || defined(HYPRE_USING_ONEMKLRAND)

/*--------------------------------------------------------------------
 * hypre_CurandUniform
 *--------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------
 * hypre_CurandUniformSingle
 *--------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------
 * hypre_ResetDeviceRandGenerator
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ResetDeviceRandGenerator( hypre_ulonglongint seed,
                                hypre_ulonglongint offset )
{
#if defined(HYPRE_USING_CURAND)
   curandGenerator_t gen = hypre_HandleCurandGenerator(hypre_handle());
   HYPRE_CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, seed) );
   HYPRE_CURAND_CALL( curandSetGeneratorOffset(gen, offset) );

#elif defined(HYPRE_USING_ROCRAND)
   rocrand_generator gen = hypre_HandleCurandGenerator(hypre_handle());
   HYPRE_ROCRAND_CALL( rocrand_set_seed(gen, seed) );
   HYPRE_ROCRAND_CALL( rocrand_set_offset(gen, offset) );
#endif

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND) || defined(HYPRE_USING_ONEMKLRAND) */

/*--------------------------------------------------------------------
 * hypreGPUKernel_filln
 *--------------------------------------------------------------------*/

template<typename T>
__global__ void
hypreGPUKernel_filln(hypre_DeviceItem &item, T *x, size_t n, T v)
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < n)
   {
      x[i] = v;
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_Filln
 *--------------------------------------------------------------------*/

template<typename T>
HYPRE_Int
hypreDevice_Filln(T *d_x, size_t n, T v)
{
#if 0
   HYPRE_THRUST_CALL( fill_n, d_x, n, v);
#else
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_filln, gDim, bDim, d_x, n, v );
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreDevice_ComplexFilln
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_ComplexFilln( HYPRE_Complex *d_x,
                          size_t         n,
                          HYPRE_Complex  v )
{
   return hypreDevice_Filln(d_x, n, v);
}

/*--------------------------------------------------------------------
 * hypreDevice_CharFilln
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_CharFilln( char   *d_x,
                       size_t  n,
                       char    v )
{
   return hypreDevice_Filln(d_x, n, v);
}

/*--------------------------------------------------------------------
 * hypreDevice_IntFilln
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_IntFilln( HYPRE_Int *d_x,
                      size_t     n,
                      HYPRE_Int  v )
{
   return hypreDevice_Filln(d_x, n, v);
}

/*--------------------------------------------------------------------
 * hypreDevice_BigIntFilln
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_BigIntFilln( HYPRE_BigInt *d_x,
                         size_t        n,
                         HYPRE_BigInt  v)
{
   return hypreDevice_Filln(d_x, n, v);
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_StridedCopy
 *--------------------------------------------------------------------*/

template<typename T>
__global__ void
hypreGPUKernel_StridedCopy(hypre_DeviceItem &item,
                           HYPRE_Int         size,
                           HYPRE_Int         stride,
                           T                *in,
                           T                *out )
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < size)
   {
      out[i] = in[i * stride];
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_StridedCopy
 *--------------------------------------------------------------------*/

template<typename T>
HYPRE_Int
hypreDevice_StridedCopy( HYPRE_Int  size,
                         HYPRE_Int  stride,
                         T         *in,
                         T         *out )
{
   if (size < 1 || stride < 1)
   {
      return hypre_error_flag;
   }

   if (in == out)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Cannot perform in-place strided copy");
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(size, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_StridedCopy, gDim, bDim, size, stride, in, out );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreDevice_IntStridedCopy
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_IntStridedCopy( HYPRE_Int  size,
                            HYPRE_Int  stride,
                            HYPRE_Int *in,
                            HYPRE_Int *out )
{
   return hypreDevice_StridedCopy(size, stride, in, out);
}

/*--------------------------------------------------------------------
 * hypreDevice_ComplexStridedCopy
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_ComplexStridedCopy( HYPRE_Int      size,
                                HYPRE_Int      stride,
                                HYPRE_Complex *in,
                                HYPRE_Complex *out )
{
   return hypreDevice_StridedCopy(size, stride, in, out);
}

/*--------------------------------------------------------------------
 * hypreDevice_CsrRowPtrsToIndicesWithRowNum
 *
 * Input:  d_row_num, of size nrows, contains the rows indices that
 *         can be HYPRE_BigInt or HYPRE_Int
 * Output: d_row_ind
 *--------------------------------------------------------------------*/

template <typename T>
HYPRE_Int
hypreDevice_CsrRowPtrsToIndicesWithRowNum( HYPRE_Int  nrows,
                                           HYPRE_Int  nnz,
                                           HYPRE_Int *d_row_ptr,
                                           T         *d_row_num,
                                           T         *d_row_ind )
{
   /* trivial case */
   if (nrows <= 0)
   {
      return hypre_error_flag;
   }

   HYPRE_Int *map = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnz, d_row_ptr, map);

#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather(map, map + nnz, d_row_num, d_row_ind);
#else
   HYPRE_THRUST_CALL(gather, map, map + nnz, d_row_num, d_row_ind);
#endif

   hypre_TFree(map, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum( HYPRE_Int  nrows,
                                                              HYPRE_Int  nnz,
                                                              HYPRE_Int *d_row_ptr,
                                                              HYPRE_Int *d_row_num,
                                                              HYPRE_Int *d_row_ind );
#if defined(HYPRE_MIXEDINT)
template HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum( HYPRE_Int     nrows,
                                                              HYPRE_Int     nnz,
                                                              HYPRE_Int    *d_row_ptr,
                                                              HYPRE_BigInt *d_row_num,
                                                              HYPRE_BigInt *d_row_ind );
#endif

/*--------------------------------------------------------------------
 * hypreDevice_IntegerReduceSum
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_IntegerReduceSum( HYPRE_Int  n,
                              HYPRE_Int *d_i )
{
#if defined(HYPRE_USING_SYCL)
   return HYPRE_ONEDPL_CALL(std::reduce, d_i, d_i + n);
#else
   return HYPRE_THRUST_CALL(reduce, d_i, d_i + n);
#endif
}

/*--------------------------------------------------------------------
 * hypreDevice_ComplexReduceSum
 *--------------------------------------------------------------------*/

HYPRE_Complex
hypreDevice_ComplexReduceSum(HYPRE_Int n, HYPRE_Complex *d_x)
{
#if defined(HYPRE_USING_SYCL)
   return HYPRE_ONEDPL_CALL(std::reduce, d_x, d_x + n);
#else
   return HYPRE_THRUST_CALL(reduce, d_x, d_x + n);
#endif
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_scalen
 *--------------------------------------------------------------------*/

template<typename T>
__global__ void
hypreGPUKernel_scalen( hypre_DeviceItem &item,
                       T                *x,
                       size_t            n,
                       T                *y,
                       T                 v )
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < n)
   {
      y[i] = x[i] * v;
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_Scalen
 *--------------------------------------------------------------------*/

template<typename T>
HYPRE_Int
hypreDevice_Scalen( T *d_x, size_t n, T *d_y, T v )
{
#if 0
   HYPRE_THRUST_CALL( transform, d_x, d_x + n, d_y, v * _1 );
#else
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_scalen, gDim, bDim, d_x, n, d_y, v );
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreDevice_IntScalen
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_IntScalen( HYPRE_Int *d_x,
                       size_t     n,
                       HYPRE_Int *d_y,
                       HYPRE_Int  v )
{
   return hypreDevice_Scalen(d_x, n, d_y, v);
}

/*--------------------------------------------------------------------
 * hypreDevice_ComplexScalen
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_ComplexScalen( HYPRE_Complex *d_x,
                           size_t         n,
                           HYPRE_Complex *d_y,
                           HYPRE_Complex  v )
{
   return hypreDevice_Scalen(d_x, n, d_y, v);
}

/*--------------------------------------------------------------------
 * hypreDevice_StableSortTupleByTupleKey
 *
 * opt:
 *      0, (a,b) < (a',b') iff a < a' or (a = a' and  b  <  b')
 *                         [normal tupe comp]
 *
 *      2, (a,b) < (a',b') iff a < a' or (a = a' and (b == a or b < b') and b' != a')
 *                         [used in assembly to put diagonal first]
 *--------------------------------------------------------------------*/

template <typename T1, typename T2, typename T3, typename T4>
HYPRE_Int
hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N,
                                      T1 *keys1, T2 *keys2, T3 *vals1, T4 *vals2,
                                      HYPRE_Int opt)
{
#if defined(HYPRE_USING_SYCL)
   auto zipped_begin = oneapi::dpl::make_zip_iterator(keys1, keys2, vals1, vals2);

   if (opt == 0)
   {
      HYPRE_ONEDPL_CALL(std::stable_sort,
                        zipped_begin,
                        zipped_begin + N,
                        std::less< std::tuple<T1, T2, T3, T4> >());
   }
   else if (opt == 2)
   {
      HYPRE_ONEDPL_CALL(std::stable_sort,
                        zipped_begin,
                        zipped_begin + N,
                        TupleComp3<T1, T2, T3, T4>());
   }
#else
   auto begin_keys = thrust::make_zip_iterator(thrust::make_tuple(keys1,     keys2));
   auto end_keys   = thrust::make_zip_iterator(thrust::make_tuple(keys1 + N, keys2 + N));
   auto begin_vals = thrust::make_zip_iterator(thrust::make_tuple(vals1,     vals2));

   if (opt == 0)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key,
                        begin_keys,
                        end_keys,
                        begin_vals,
                        thrust::less< thrust::tuple<T1, T2> >());
   }
   else if (opt == 2)
   {
      HYPRE_THRUST_CALL(stable_sort_by_key,
                        begin_keys,
                        end_keys,
                        begin_vals,
                        TupleComp3<T1, T2>());
   }
#endif

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, HYPRE_Int *keys1,
                                                         HYPRE_Int *keys2, char *vals1, HYPRE_Complex *vals2, HYPRE_Int opt);
#if defined(HYPRE_MIXEDINT)
template HYPRE_Int hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, HYPRE_BigInt *keys1,
                                                         HYPRE_BigInt *keys2, char *vals1, HYPRE_Complex *vals2, HYPRE_Int opt);
#endif

/*--------------------------------------------------------------------
 * hypreGPUKernel_DiagScaleVector
 *--------------------------------------------------------------------*/

template <HYPRE_Int NV>
__global__ void
hypreGPUKernel_DiagScaleVector( hypre_DeviceItem &item,
                                HYPRE_Int         num_vectors,
                                HYPRE_Int         num_rows,
                                HYPRE_Int        *A_i,
                                HYPRE_Complex    *A_data,
                                HYPRE_Complex    *x,
                                HYPRE_Complex     beta,
                                HYPRE_Complex    *y )
{
   HYPRE_Int     i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   HYPRE_Int     j;
   HYPRE_Complex val;

   if (i < num_rows)
   {
      val = 1.0 / A_data[A_i[i]];

      if (beta != 0.0)
      {
         if (NV > 0)
         {
#pragma unroll
            for (j = 0; j < NV; j++)
            {
               y[i + j * num_rows] = val  * x[i + j * num_rows] +
                                     beta * y[i + j * num_rows];
            }
         }
         else
         {
#pragma unroll 8
            for (j = 0; j < num_vectors; j++)
            {
               y[i + j * num_rows] = val  * x[i + j * num_rows] +
                                     beta * y[i + j * num_rows];
            }
         }
      }
      else
      {
         if (NV > 0)
         {
#pragma unroll
            for (j = 0; j < NV; j++)
            {
               y[i + j * num_rows] = val  * x[i + j * num_rows];
            }
         }
         else
         {
#pragma unroll 8
            for (j = 0; j < num_vectors; j++)
            {
               y[i + j * num_rows] = val  * x[i + j * num_rows];
            }
         }
      }
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_DiagScaleVector
 *
 * y = diag(A) \ x + beta y
 * Note: Assume A_i[i] points to the ith diagonal entry of A
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_DiagScaleVector( HYPRE_Int       num_vectors,
                             HYPRE_Int       num_rows,
                             HYPRE_Int      *A_i,
                             HYPRE_Complex  *A_data,
                             HYPRE_Complex  *x,
                             HYPRE_Complex   beta,
                             HYPRE_Complex  *y )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return hypre_error_flag;
   }
   hypre_assert(num_vectors > 0);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

   switch (num_vectors)
   {
      case 1:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_DiagScaleVector<1>, gDim, bDim,
                           num_vectors, num_rows, A_i, A_data, x, beta, y );
         break;

      case 2:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_DiagScaleVector<2>, gDim, bDim,
                           num_vectors, num_rows, A_i, A_data, x, beta, y );
         break;

      case 3:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_DiagScaleVector<3>, gDim, bDim,
                           num_vectors, num_rows, A_i, A_data, x, beta, y );
         break;

      case 4:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_DiagScaleVector<4>, gDim, bDim,
                           num_vectors, num_rows, A_i, A_data, x, beta, y );
         break;

      case 5:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_DiagScaleVector<5>, gDim, bDim,
                           num_vectors, num_rows, A_i, A_data, x, beta, y );
         break;

      case 6:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_DiagScaleVector<6>, gDim, bDim,
                           num_vectors, num_rows, A_i, A_data, x, beta, y );
         break;

      case 7:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_DiagScaleVector<7>, gDim, bDim,
                           num_vectors, num_rows, A_i, A_data, x, beta, y );
         break;

      case 8:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_DiagScaleVector<8>, gDim, bDim,
                           num_vectors, num_rows, A_i, A_data, x, beta, y );
         break;

      default:
         HYPRE_GPU_LAUNCH( hypreGPUKernel_DiagScaleVector<0>, gDim, bDim,
                           num_vectors, num_rows, A_i, A_data, x, beta, y );
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_DiagScaleVector2
 *--------------------------------------------------------------------*/

template <HYPRE_Int NV, HYPRE_Int CY>
__global__ void
hypreGPUKernel_DiagScaleVector2( hypre_DeviceItem &item,
                                 HYPRE_Int         num_vectors,
                                 HYPRE_Int         num_rows,
                                 HYPRE_Complex    *diag,
                                 HYPRE_Complex    *x,
                                 HYPRE_Complex     beta,
                                 HYPRE_Complex    *y,
                                 HYPRE_Complex    *z )
{
   HYPRE_Int      i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   HYPRE_Int      j;
   HYPRE_Complex  inv_diag;
   HYPRE_Complex  x_over_diag;

   if (i < num_rows)
   {
      inv_diag = 1.0 / diag[i];

      if (NV > 0)
      {
#pragma unroll
         for (j = 0; j < NV; j++)
         {
            x_over_diag = x[i + j * num_rows] * inv_diag;

            if (CY)
            {
               y[i + j * num_rows] = x_over_diag;
            }
            z[i + j * num_rows] += beta * x_over_diag;
         }
      }
      else
      {
#pragma unroll 8
         for (j = 0; j < num_vectors; j++)
         {
            x_over_diag = x[i + j * num_rows] * inv_diag;

            if (CY)
            {
               y[i + j * num_rows] = x_over_diag;
            }
            z[i + j * num_rows] += beta * x_over_diag;
         }
      }
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_DiagScaleVector2
 *
 * y = x ./ diag
 * z = z + beta * (x ./ diag)
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_DiagScaleVector2( HYPRE_Int       num_vectors,
                              HYPRE_Int       num_rows,
                              HYPRE_Complex  *diag,
                              HYPRE_Complex  *x,
                              HYPRE_Complex   beta,
                              HYPRE_Complex  *y,
                              HYPRE_Complex  *z,
                              HYPRE_Int       computeY )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return hypre_error_flag;
   }
   hypre_assert(num_vectors > 0);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

   switch (num_vectors)
   {
      case 1:
         if (computeY > 0)
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<1, 1>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         else
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<1, 0>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         break;

      case 2:
         if (computeY > 0)
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<2, 1>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         else
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<2, 0>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         break;

      case 3:
         if (computeY > 0)
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<3, 1>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         else
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<3, 0>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         break;

      case 4:
         if (computeY > 0)
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<4, 1>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         else
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<4, 0>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         break;

      case 5:
         if (computeY > 0)
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<5, 1>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         else
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<5, 0>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         break;

      case 6:
         if (computeY > 0)
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<6, 1>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         else
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<6, 0>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         break;

      case 7:
         if (computeY > 0)
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<7, 1>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         else
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<7, 0>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         break;

      case 8:
         if (computeY > 0)
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<8, 1>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         else
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<8, 0>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         break;

      default:
         if (computeY > 0)
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<0, 1>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         else
         {
            HYPRE_GPU_LAUNCH( (hypreGPUKernel_DiagScaleVector2<0, 0>), gDim, bDim,
                              num_vectors, num_rows, diag, x, beta, y, z );
         }
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_zeqxmydd
 *
 * z[i] = (x[i] + alpha*y[i])*d[i]
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_zeqxmydd(hypre_DeviceItem             &item,
                        HYPRE_Int                    n,
                        HYPRE_Complex* __restrict__  x,
                        HYPRE_Complex                alpha,
                        HYPRE_Complex* __restrict__  y,
                        HYPRE_Complex* __restrict__  z,
                        HYPRE_Complex* __restrict__  d)
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < n)
   {
      z[i] = (x[i] + alpha * y[i]) * d[i];
   }
}

/*--------------------------------------------------------------------
 * hypreDevice_zeqxmydd
 *--------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_zeqxmydd(HYPRE_Int       n,
                     HYPRE_Complex  *x,
                     HYPRE_Complex   alpha,
                     HYPRE_Complex  *y,
                     HYPRE_Complex  *z,
                     HYPRE_Complex  *d)
{
   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_zeqxmydd, gDim, bDim, n, x, alpha, y, z, d);

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      cuda/hip functions
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/*--------------------------------------------------------------------
 * hypreGPUKernel_CompileFlagSafetyCheck
 *
 * The architecture identification macro __CUDA_ARCH__ is assigned a
 * three-digit value string xy0 (ending in a literal 0) during each
 * nvcc compilation stage 1 that compiles for compute_xy.
 *
 * This macro can be used in the implementation of GPU functions for
 * determining the virtual architecture for which it is currently being
 * compiled. The host code (the non-GPU code) must not depend on it.
 *
 * Note that compute_XX refers to a PTX version and sm_XX refers to
 * a cubin version.
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CompileFlagSafetyCheck( hypre_DeviceItem &item,
                                       hypre_int        *cuda_arch_compile )
{
#if defined(__CUDA_ARCH__)
   cuda_arch_compile[0] = __CUDA_ARCH__;
#endif
}

/*--------------------------------------------------------------------
 * hypre_CudaCompileFlagCheck
 *
 * Assume this function is called inside HYPRE_Init(), at a place
 * where we do not want to activate memory pooling, so we do not use
 * hypre's memory model to Alloc and Free.
 *
 * See commented out code below (and do not delete)
 *
 * This is really only defined for CUDA and not for HIP
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_CudaCompileFlagCheck()
{
#if defined(HYPRE_USING_CUDA)
   HYPRE_Int device;
   hypre_GetDevice(&device);

   struct cudaDeviceProp props;
   cudaGetDeviceProperties(&props, device);
   hypre_int cuda_arch_actual = props.major * 100 + props.minor * 10;
   hypre_int cuda_arch_compile = -1;
   dim3 gDim(1, 1, 1), bDim(1, 1, 1);

   hypre_int *cuda_arch_compile_d = NULL;
   //cuda_arch_compile_d = hypre_TAlloc(hypre_int, 1, HYPRE_MEMORY_DEVICE);
   HYPRE_CUDA_CALL( cudaMalloc(&cuda_arch_compile_d, sizeof(hypre_int)) );
   HYPRE_CUDA_CALL( cudaMemcpy(cuda_arch_compile_d, &cuda_arch_compile, sizeof(hypre_int),
                               cudaMemcpyHostToDevice) );
   HYPRE_GPU_LAUNCH( hypreGPUKernel_CompileFlagSafetyCheck, gDim, bDim, cuda_arch_compile_d );
   HYPRE_CUDA_CALL( cudaMemcpy(&cuda_arch_compile, cuda_arch_compile_d, sizeof(hypre_int),
                               cudaMemcpyDeviceToHost) );
   //hypre_TFree(cuda_arch_compile_d, HYPRE_MEMORY_DEVICE);
   HYPRE_CUDA_CALL( cudaFree(cuda_arch_compile_d) );

   /* HYPRE_CUDA_CALL(cudaDeviceSynchronize()); */

   const hypre_int cuda_arch_actual_major  = cuda_arch_actual  / 100;
   const hypre_int cuda_arch_compile_major = cuda_arch_compile / 100;
   const hypre_int cuda_arch_actual_minor  = cuda_arch_actual  % 100;
   const hypre_int cuda_arch_compile_minor = cuda_arch_compile % 100;

   if (cuda_arch_actual_major != cuda_arch_compile_major ||
       cuda_arch_actual_minor < cuda_arch_compile_minor)
   {
      char msg[256];

      if (-1 == cuda_arch_compile)
      {
         hypre_sprintf(msg, "hypre error: no proper cuda_arch found");
      }
      else
      {
         hypre_sprintf(msg,
                       "hypre error: Compile arch %d ('--generate-code arch=compute_%d') does not match device arch %d",
                       cuda_arch_compile, cuda_arch_compile / 10, cuda_arch_actual);
      }

      hypre_error_w_msg(1, msg);
#if defined(HYPRE_DEBUG)
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "%s\n", msg);
#endif
      hypre_assert(0);
   }
#endif // defined(HYPRE_USING_CUDA)

   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)

/*--------------------------------------------------------------------
 * hypre_HYPREComplexToCudaDataType
 *
 * Determines the associated CudaDataType for HYPRE_Complex
 *
 * TODO: Should be known at compile time.
 *       Support more sizes.
 *       Support complex.
 *
 * Note: Only works for Single and Double precision.
 *--------------------------------------------------------------------*/

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

#if CUSPARSE_VERSION >= 10300
/*--------------------------------------------------------------------
 * hypre_HYPREIntToCusparseIndexType
 *
 * Determines the associated cusparseIndexType_t for HYPRE_Int
 *--------------------------------------------------------------------*/

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
#endif

#endif // #if defined(HYPRE_USING_CUSPARSE)

#if defined(HYPRE_USING_CUBLAS)

/*--------------------------------------------------------------------
 * hypre_DeviceDataCublasHandle
 *--------------------------------------------------------------------*/

cublasHandle_t
hypre_DeviceDataCublasHandle(hypre_DeviceData *data)
{
   if (data->cublas_handle)
   {
      return data->cublas_handle;
   }

   cublasHandle_t handle;
   HYPRE_CUBLAS_CALL( cublasCreate(&handle) );

   HYPRE_CUBLAS_CALL( cublasSetStream(handle, hypre_DeviceDataComputeStream(data)) );

   data->cublas_handle = handle;

   return handle;
}
#endif

#if defined(HYPRE_USING_CUSPARSE)

/*--------------------------------------------------------------------
 * hypre_DeviceDataCusparseHandle
 *--------------------------------------------------------------------*/

cusparseHandle_t
hypre_DeviceDataCusparseHandle(hypre_DeviceData *data)
{
   if (data->cusparse_handle)
   {
      return data->cusparse_handle;
   }

   cusparseHandle_t handle;
   HYPRE_CUSPARSE_CALL( cusparseCreate(&handle) );

   HYPRE_CUSPARSE_CALL( cusparseSetStream(handle, hypre_DeviceDataComputeStream(data)) );

   data->cusparse_handle = handle;

   return handle;
}
#endif // defined(HYPRE_USING_CUSPARSE)

#if defined(HYPRE_USING_ROCSPARSE)

/*--------------------------------------------------------------------
 * hypre_DeviceDataCusparseHandle
 *--------------------------------------------------------------------*/

rocsparse_handle
hypre_DeviceDataCusparseHandle(hypre_DeviceData *data)
{
   if (data->cusparse_handle)
   {
      return data->cusparse_handle;
   }

   rocsparse_handle handle;
   HYPRE_ROCSPARSE_CALL( rocsparse_create_handle(&handle) );

   HYPRE_ROCSPARSE_CALL( rocsparse_set_stream(handle, hypre_DeviceDataComputeStream(data)) );

   data->cusparse_handle = handle;

   return handle;
}
#endif // defined(HYPRE_USING_ROCSPARSE)

#if defined(HYPRE_USING_CUSOLVER) || defined(HYPRE_USING_ROCSOLVER)

/*--------------------------------------------------------------------
 * hypre_DeviceDataVendorSolverHandle
 *--------------------------------------------------------------------*/

vendorSolverHandle_t
hypre_DeviceDataVendorSolverHandle(hypre_DeviceData *data)
{
   if (data->vendor_solver_handle)
   {
      return data->vendor_solver_handle;
   }

#if defined(HYPRE_USING_CUSOLVER)
   cusolverDnHandle_t handle;

   HYPRE_CUSOLVER_CALL( cusolverDnCreate(&handle) );
   HYPRE_CUSOLVER_CALL( cusolverDnSetStream(handle, hypre_DeviceDataComputeStream(data)) );
#else
   rocblas_handle handle;

   HYPRE_ROCBLAS_CALL( rocblas_create_handle(&handle) );
   HYPRE_ROCBLAS_CALL( rocblas_set_stream(handle, hypre_DeviceDataComputeStream(data)) );
#endif

   data->vendor_solver_handle = handle;

   return handle;
}
#endif // defined(HYPRE_USING_CUSOLVER) || defined(HYPRE_USING_ROCSOLVER)

#endif // #if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      sycl functions
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(HYPRE_USING_SYCL)

/*--------------------------------------------------------------------
 * HYPRE_SetSYCLDevice
 *--------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetSYCLDevice(sycl::device user_device)
{
   hypre_DeviceData *data = hypre_HandleDeviceData(hypre_handle());

   /* Cleanup default device and queues */
   if (data->device)
   {
      delete data->device;
   }
   for (HYPRE_Int i = 0; i < HYPRE_MAX_NUM_STREAMS; i++)
   {
      if (data->streams[i])
      {
         delete data->streams[i];
         data->streams[i] = nullptr;
      }
   }

   /* Setup new device and compute stream */
   data->device = new sycl::device(user_device);
   hypre_HandleComputeStream(hypre_handle());

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_SYCL)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      additional functions
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

/*--------------------------------------------------------------------
 * hypre_bind_device
 *
 * This function is supposed to be used in the test drivers to mimic
 * users' GPU binding approaches
 * It is supposed to be called before HYPRE_Init,
 * so that HYPRE_Init can get the wanted device id
 * WM: note - sycl has no analogue to cudaSetDevice(),
 * so this has no effect on the sycl implementation.
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_bind_device_id( HYPRE_Int device_id_in,
                      HYPRE_Int myid,
                      HYPRE_Int nproc,
                      MPI_Comm  comm )
{
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
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
   hypre_GetDeviceCount(&nDevices);

   if (-1 == device_id_in)
   {
      /* get number of devices on this node */
      device_id = myNodeid % nDevices;
   }
   else
   {
      device_id = (hypre_int) device_id_in;
   }

   /* set device */
#if defined(HYPRE_USING_DEVICE_OPENMP)
   omp_set_default_device(device_id);
#endif

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaSetDevice(device_id) );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipSetDevice(device_id) );
#endif

#if defined(HYPRE_DEBUG) && defined(HYPRE_PRINT_ERRORS)
   hypre_printf("Proc [global %d/%d, local %d/%d] can see %d GPUs and is running on %d\n",
                myid, nproc, myNodeid, NodeSize, nDevices, device_id);
#endif

#else
   HYPRE_UNUSED_VAR(device_id_in);
   HYPRE_UNUSED_VAR(myid);
   HYPRE_UNUSED_VAR(nproc);
   HYPRE_UNUSED_VAR(comm);

#endif // #if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)

   return hypre_error_flag;
}

HYPRE_Int
hypre_bind_device( HYPRE_Int myid,
                   HYPRE_Int nproc,
                   MPI_Comm  comm )
{
   return hypre_bind_device_id(-1, myid, nproc, comm);
}

