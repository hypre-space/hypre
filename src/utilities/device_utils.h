/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_DEVICE_UTILS_H
#define HYPRE_DEVICE_UTILS_H

#if defined(HYPRE_USING_GPU)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                          cuda includes
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cusparse.h>

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

#ifndef CUDA_VERSION
#error CUDA_VERSION Undefined!
#endif

#if CUDA_VERSION >= 11000
#define THRUST_IGNORE_DEPRECATED_CPP11
#define CUB_IGNORE_DEPRECATED_CPP11
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT
#endif

#define CUSPARSE_NEWAPI_VERSION 11000

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                          hip includes
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#elif defined(HYPRE_USING_HIP)

#include <hip/hip_runtime.h>

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                          sycl includes
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#elif defined(HYPRE_USING_SYCL)

#if defined(HYPRE_USING_ONEMKLRNG)
#include <oneapi/mkl/rng/device.hpp>
#endif

using dim3 = sycl::range<1>;
#define __global__
#define __host__
#define __device__
#define __forceinline__ __inline__ __attribute__((always_inline))

using sycl::rint;
using sycl::min;
using sycl::max;

/* WM: problems with this being inside extern C++ {} */
/* #include <CL/sycl.hpp> */

#endif // defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

#if defined(HYPRE_USING_ROCSPARSE)
#include <rocsparse.h>
#endif

#if defined(HYPRE_USING_ROCRAND)
#include <rocrand.h>
#endif

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      macros for wrapping cuda/hip/sycl calls for error reporting
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#define HYPRE_CUDA_CALL(call) do {                                                           \
   cudaError_t err = call;                                                                   \
   if (cudaSuccess != err) {                                                                 \
      printf("CUDA ERROR (code = %d, %s) at %s:%d\n", err, cudaGetErrorString(err),          \
                   __FILE__, __LINE__);                                                      \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

#elif defined(HYPRE_USING_HIP)
#define HYPRE_HIP_CALL(call) do {                                                            \
   hipError_t err = call;                                                                    \
   if (hipSuccess != err) {                                                                  \
      printf("HIP ERROR (code = %d, %s) at %s:%d\n", err, hipGetErrorString(err),            \
                   __FILE__, __LINE__);                                                      \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

#elif defined(HYPRE_USING_SYCL)
#define HYPRE_SYCL_CALL(call)                                                                \
   try                                                                                       \
   {                                                                                         \
      call;                                                                                  \
   }                                                                                         \
   catch (sycl::exception const &ex)                                                         \
   {                                                                                         \
      hypre_printf("SYCL ERROR (code = %s) at %s:%d\n", ex.what(),                           \
                     __FILE__, __LINE__);                                                    \
      assert(0); exit(1);                                                                    \
   }                                                                                         \
   catch(std::runtime_error const& ex)                                                       \
   {                                                                                         \
      hypre_printf("STD ERROR (code = %s) at %s:%d\n", ex.what(),                            \
                   __FILE__, __LINE__);                                                      \
      assert(0); exit(1);                                                                    \
   }

#define HYPRE_ONEMKL_CALL(call)                                                              \
   try                                                                                       \
   {                                                                                         \
      call;                                                                                  \
   }                                                                                         \
   catch (oneapi::mkl::exception const &ex)                                                  \
   {                                                                                         \
      hypre_printf("ONEMKL ERROR (code = %s) at %s:%d\n", ex.what(),                         \
                   __FILE__, __LINE__);                                                      \
      assert(0); exit(1);                                                                    \
   }

#endif // defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

#define HYPRE_CUBLAS_CALL(call) do {                                                         \
   cublasStatus_t err = call;                                                                \
   if (CUBLAS_STATUS_SUCCESS != err) {                                                       \
      printf("CUBLAS ERROR (code = %d, %d) at %s:%d\n",                                      \
            err, err == CUBLAS_STATUS_EXECUTION_FAILED, __FILE__, __LINE__);                 \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define HYPRE_CUSPARSE_CALL(call) do {                                                       \
   cusparseStatus_t err = call;                                                              \
   if (CUSPARSE_STATUS_SUCCESS != err) {                                                     \
      printf("CUSPARSE ERROR (code = %d, %s) at %s:%d\n",                                    \
            err, cusparseGetErrorString(err), __FILE__, __LINE__);                           \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define HYPRE_ROCSPARSE_CALL(call) do {                                                      \
   rocsparse_status err = call;                                                              \
   if (rocsparse_status_success != err) {                                                    \
      printf("rocSPARSE ERROR (code = %d) at %s:%d\n",                                       \
            err, __FILE__, __LINE__);                                                        \
      assert(0); exit(1);                                                                    \
   } } while(0)

#define HYPRE_CURAND_CALL(call) do {                                                         \
   curandStatus_t err = call;                                                                \
   if (CURAND_STATUS_SUCCESS != err) {                                                       \
      printf("CURAND ERROR (code = %d) at %s:%d\n", err, __FILE__, __LINE__);                \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define HYPRE_ROCRAND_CALL(call) do {                                                        \
   rocrand_status err = call;                                                                \
   if (ROCRAND_STATUS_SUCCESS != err) {                                                      \
      printf("ROCRAND ERROR (code = %d) at %s:%d\n", err, __FILE__, __LINE__);               \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      device defined values
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

// HYPRE_WARP_BITSHIFT is just log2 of HYPRE_WARP_SIZE
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP) || defined(HYPRE_USING_SYCL)
#define HYPRE_WARP_SIZE       32
#define HYPRE_WARP_BITSHIFT   5
#elif defined(HYPRE_USING_HIP)
#define HYPRE_WARP_SIZE       64
#define HYPRE_WARP_BITSHIFT   6
#endif

#define HYPRE_WARP_FULL_MASK  0xFFFFFFFF
#define HYPRE_MAX_NUM_WARPS   (64 * 64 * 32)
#define HYPRE_FLT_LARGE       1e30
#define HYPRE_1D_BLOCK_SIZE   512
#define HYPRE_MAX_NUM_STREAMS 10

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      device info data structures
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

struct hypre_cub_CachingDeviceAllocator;
typedef struct hypre_cub_CachingDeviceAllocator hypre_cub_CachingDeviceAllocator;

struct hypre_DeviceData
{
#if defined(HYPRE_USING_CURAND)
   curandGenerator_t                 curand_generator;
#elif defined(HYPRE_USING_ROCRAND)
   rocrand_generator                 curand_generator;
#elif defined(HYPRE_USING_ONEMKLRNG)
   oneapi::mkl::rng::device::philox4x32x10<>* curand_generator = nullptr;
#endif

#if defined(HYPRE_USING_CUBLAS)
   cublasHandle_t                    cublas_handle;
#endif

#if defined(HYPRE_USING_CUSPARSE)
   cusparseHandle_t                  cusparse_handle;
#elif defined(HYPRE_USING_ROCSPARSE)
   rocsparse_handle                  cusparse_handle;
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   cudaStream_t                      streams[HYPRE_MAX_NUM_STREAMS];
#elif defined(HYPRE_USING_HIP)
   hipStream_t                       streams[HYPRE_MAX_NUM_STREAMS];
#elif defined(HYPRE_USING_SYCL)
   sycl::queue*                      streams[HYPRE_MAX_NUM_STREAMS] = {NULL};
#endif

#ifdef HYPRE_USING_DEVICE_POOL
   hypre_uint                        cub_bin_growth;
   hypre_uint                        cub_min_bin;
   hypre_uint                        cub_max_bin;
   size_t                            cub_max_cached_bytes;
   hypre_cub_CachingDeviceAllocator *cub_dev_allocator;
   hypre_cub_CachingDeviceAllocator *cub_uvm_allocator;
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_device_allocator            device_allocator;
#endif
#if defined(HYPRE_USING_SYCL)
   sycl::device                     *device;
   HYPRE_Int                         device_max_work_group_size;
#else
   HYPRE_Int                         device;
#endif
   /* by default, hypre puts GPU computations in this stream
    * Do not be confused with the default (null) stream */
   HYPRE_Int                         compute_stream_num;
   /* work space for hypre's device reducer */
   void                             *reduce_buffer;
   /* the device buffers needed to do MPI communication for struct comm */
   HYPRE_Complex*                    struct_comm_recv_buffer;
   HYPRE_Complex*                    struct_comm_send_buffer;
   HYPRE_Int                         struct_comm_recv_buffer_size;
   HYPRE_Int                         struct_comm_send_buffer_size;
   /* device spgemm options */
   HYPRE_Int                         spgemm_use_cusparse;
   HYPRE_Int                         spgemm_algorithm;
   HYPRE_Int                         spgemm_rownnz_estimate_method;
   HYPRE_Int                         spgemm_rownnz_estimate_nsamples;
   float                             spgemm_rownnz_estimate_mult_factor;
   char                              spgemm_hash_type;
   /* PMIS */
   HYPRE_Int                         use_gpu_rand;
};

#define hypre_DeviceDataCubBinGrowth(data)                   ((data) -> cub_bin_growth)
#define hypre_DeviceDataCubMinBin(data)                      ((data) -> cub_min_bin)
#define hypre_DeviceDataCubMaxBin(data)                      ((data) -> cub_max_bin)
#define hypre_DeviceDataCubMaxCachedBytes(data)              ((data) -> cub_max_cached_bytes)
#define hypre_DeviceDataCubDevAllocator(data)                ((data) -> cub_dev_allocator)
#define hypre_DeviceDataCubUvmAllocator(data)                ((data) -> cub_uvm_allocator)
#define hypre_DeviceDataDevice(data)                         ((data) -> device)
#define hypre_DeviceDataDeviceMaxWorkGroupSize(data)         ((data) -> device_max_work_group_size)
#define hypre_DeviceDataComputeStreamNum(data)               ((data) -> compute_stream_num)
#define hypre_DeviceDataReduceBuffer(data)                   ((data) -> reduce_buffer)
#define hypre_DeviceDataStructCommRecvBuffer(data)           ((data) -> struct_comm_recv_buffer)
#define hypre_DeviceDataStructCommSendBuffer(data)           ((data) -> struct_comm_send_buffer)
#define hypre_DeviceDataStructCommRecvBufferSize(data)       ((data) -> struct_comm_recv_buffer_size)
#define hypre_DeviceDataStructCommSendBufferSize(data)       ((data) -> struct_comm_send_buffer_size)
#define hypre_DeviceDataSpgemmUseCusparse(data)              ((data) -> spgemm_use_cusparse)
#define hypre_DeviceDataSpgemmAlgorithm(data)                ((data) -> spgemm_algorithm)
#define hypre_DeviceDataSpgemmRownnzEstimateMethod(data)     ((data) -> spgemm_rownnz_estimate_method)
#define hypre_DeviceDataSpgemmRownnzEstimateNsamples(data)   ((data) -> spgemm_rownnz_estimate_nsamples)
#define hypre_DeviceDataSpgemmRownnzEstimateMultFactor(data) ((data) -> spgemm_rownnz_estimate_mult_factor)
#define hypre_DeviceDataSpgemmHashType(data)                 ((data) -> spgemm_hash_type)
#define hypre_DeviceDataDeviceAllocator(data)                ((data) -> device_allocator)
#define hypre_DeviceDataUseGpuRand(data)                     ((data) -> use_gpu_rand)

hypre_DeviceData*     hypre_DeviceDataCreate();
void                hypre_DeviceDataDestroy(hypre_DeviceData* data);

#if defined(HYPRE_USING_CURAND)
curandGenerator_t   hypre_DeviceDataCurandGenerator(hypre_DeviceData *data);
#elif defined(HYPRE_USING_ROCRAND)
rocrand_generator   hypre_DeviceDataCurandGenerator(hypre_DeviceData *data);
#elif defined(HYPRE_USING_ONEMKLRNG)
oneapi::mkl::rng::device::philox4x32x10<>* hypre_DeviceDataCurandGenerator(hypre_DeviceData *data);
#endif

#if defined(HYPRE_USING_CUBLAS)
cublasHandle_t      hypre_DeviceDataCublasHandle(hypre_DeviceData *data);
#endif

#if defined(HYPRE_USING_CUSPARSE)
cusparseHandle_t    hypre_DeviceDataCusparseHandle(hypre_DeviceData *data);
#elif defined(HYPRE_USING_ROCSPARSE)
rocsparse_handle    hypre_DeviceDataCusparseHandle(hypre_DeviceData *data);
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
cudaStream_t        hypre_DeviceDataStream(hypre_DeviceData *data, HYPRE_Int i);
cudaStream_t        hypre_DeviceDataComputeStream(hypre_DeviceData *data);
#elif defined(HYPRE_USING_HIP)
hipStream_t         hypre_DeviceDataStream(hypre_DeviceData *data, HYPRE_Int i);
hipStream_t         hypre_DeviceDataComputeStream(hypre_DeviceData *data);
#elif defined(HYPRE_USING_SYCL)
sycl::queue*        hypre_DeviceDataStream(hypre_DeviceData *data, HYPRE_Int i);
sycl::queue*        hypre_DeviceDataComputeStream(hypre_DeviceData *data);
#endif

// Data structure and accessor routines for Cuda Sparse Triangular Matrices
struct hypre_CsrsvData
{
#if defined(HYPRE_USING_CUSPARSE)
   csrsv2Info_t info_L;
   csrsv2Info_t info_U;
#elif defined(HYPRE_USING_ROCSPARSE)
   rocsparse_mat_info info_L;
   rocsparse_mat_info info_U;
#endif
   hypre_int    BufferSize;
   char        *Buffer;
};

#define hypre_CsrsvDataInfoL(data)      ((data) -> info_L)
#define hypre_CsrsvDataInfoU(data)      ((data) -> info_U)
#define hypre_CsrsvDataBufferSize(data) ((data) -> BufferSize)
#define hypre_CsrsvDataBuffer(data)     ((data) -> Buffer)

struct hypre_GpuMatData
{
#if defined(HYPRE_USING_CUSPARSE)
   cusparseMatDescr_t    mat_descr;
   char                 *spmv_buffer;
#endif

#if defined(HYPRE_USING_ROCSPARSE)
   rocsparse_mat_descr   mat_descr;
   rocsparse_mat_info    mat_info;
#endif
};

#define hypre_GpuMatDataMatDecsr(data)    ((data) -> mat_descr)
#define hypre_GpuMatDataMatInfo(data)     ((data) -> mat_info)
#define hypre_GpuMatDataSpMVBuffer(data)  ((data) -> spmv_buffer)

/* device_utils.c, some common functions for CUDA, SYCL, HIP */

dim3 hypre_GetDefaultDeviceBlockDimension();

dim3 hypre_GetDefaultDeviceGridDimension( HYPRE_Int n, const char *granularity,
					  dim3 bDim );

HYPRE_Int hypreDevice_CsrRowPtrsToIndices_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr,
                                             HYPRE_Int *d_row_ind);

HYPRE_Int* hypreDevice_CsrRowPtrsToIndices(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr);

HYPRE_Int* hypreDevice_CsrRowIndicesToPtrs(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind);

HYPRE_Int hypreDevice_CsrRowIndicesToPtrs_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind,
                                             HYPRE_Int *d_row_ptr);

HYPRE_Int hypreDevice_IntegerReduceSum(HYPRE_Int m, HYPRE_Int *d_i);

HYPRE_Int hypreDevice_IntegerInclusiveScan(HYPRE_Int n, HYPRE_Int *d_i);

HYPRE_Int hypreDevice_IntegerExclusiveScan(HYPRE_Int n, HYPRE_Int *d_i);

template <typename T>
HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum(HYPRE_Int nrows, HYPRE_Int nnz,
                                                    HYPRE_Int *d_row_ptr, T *d_row_num, T *d_row_ind);

static __device__ __forceinline__
hypre_int next_power_of_2(hypre_int n)
{
   if (n <= 0)
   {
      return 0;
   }

   /* if n is power of 2, return itself */
   if ( (n & (n - 1)) == 0 )
   {
      return n;
   }

   n |= (n >>  1);
   n |= (n >>  2);
   n |= (n >>  4);
   n |= (n >>  8);
   n |= (n >> 16);
   n ^= (n >>  1);
   n  = (n <<  1);

   return n;
}

#endif //#if defined(HYPRE_USING_GPU)

#if defined(HYPRE_USING_SYCL)

/* device_utils.c */
HYPRE_Int HYPRE_SetSYCLDevice(sycl::device user_device);
#endif // #if defined(HYPRE_USING_SYCL)

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#include <thrust/execution_policy.h>
#if defined(HYPRE_USING_CUDA)
#include <thrust/system/cuda/execution_policy.h>
#elif defined(HYPRE_USING_HIP)
#include <thrust/system/hip/execution_policy.h>
#endif
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/adjacent_difference.h>
#include <thrust/inner_product.h>
#include <thrust/logical.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>

using namespace thrust::placeholders;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 * macro for launching CUDA kernels, CUDA, Thrust, Cusparse, Curand calls
 *                    NOTE: IN HYPRE'S DEFAULT STREAM
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 */

#if defined(HYPRE_DEBUG)
#if defined(HYPRE_USING_CUDA)
#define GPU_LAUNCH_SYNC { hypre_SyncDeviceComputeStream(hypre_handle()); HYPRE_CUDA_CALL( cudaGetLastError() ); }
#elif defined(HYPRE_USING_HIP)
#define GPU_LAUNCH_SYNC { hypre_SyncDeviceComputeStream(hypre_handle()); HYPRE_HIP_CALL( hipGetLastError() );  }
#endif
#else // #if defined(HYPRE_DEBUG)
#define GPU_LAUNCH_SYNC
#endif // defined(HYPRE_DEBUG)

#define HYPRE_GPU_LAUNCH2(kernel_name, gridsize, blocksize, shmem_size, ...)                                                 \
{                                                                                                                             \
   if ( gridsize.x  == 0 || gridsize.y  == 0 || gridsize.z  == 0 ||                                                           \
        blocksize.x == 0 || blocksize.y == 0 || blocksize.z == 0 )                                                            \
   {                                                                                                                          \
      /* printf("Warning %s %d: Zero CUDA grid/block (%d %d %d) (%d %d %d)\n",                                                \
                 __FILE__, __LINE__,                                                                                          \
                 gridsize.x, gridsize.y, gridsize.z, blocksize.x, blocksize.y, blocksize.z); */                               \
   }                                                                                                                          \
   else                                                                                                                       \
   {                                                                                                                          \
      (kernel_name) <<< (gridsize), (blocksize), shmem_size, hypre_HandleComputeStream(hypre_handle()) >>> (__VA_ARGS__); \
      GPU_LAUNCH_SYNC;                                                                                                        \
   }                                                                                                                          \
}

#define HYPRE_GPU_LAUNCH(kernel_name, gridsize, blocksize, ...) HYPRE_GPU_LAUNCH2(kernel_name, gridsize, blocksize, 0, __VA_ARGS__)

/* RL: TODO Want macro HYPRE_THRUST_CALL to return value but I don't know how to do it right
 * The following one works OK for now */

#if defined(HYPRE_USING_CUDA)
#define HYPRE_THRUST_CALL(func_name, ...) \
   thrust::func_name(thrust::cuda::par(hypre_HandleDeviceAllocator(hypre_handle())).on(hypre_HandleComputeStream(hypre_handle())), __VA_ARGS__);
#elif defined(HYPRE_USING_HIP)
#define HYPRE_THRUST_CALL(func_name, ...) \
   thrust::func_name(thrust::hip::par(hypre_HandleDeviceAllocator(hypre_handle())).on(hypre_HandleComputeStream(hypre_handle())), __VA_ARGS__);
#endif

/* return the number of threads in block */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_num_threads()
{
   switch (dim)
   {
      case 1:
         return (blockDim.x);
      case 2:
         return (blockDim.x * blockDim.y);
      case 3:
         return (blockDim.x * blockDim.y * blockDim.z);
   }

   return -1;
}

/* return the flattened thread id in block */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_thread_id()
{
   switch (dim)
   {
      case 1:
         return (threadIdx.x);
      case 2:
         return (threadIdx.y * blockDim.x + threadIdx.x);
      case 3:
         return (threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
                 threadIdx.x);
   }

   return -1;
}

/* return the number of warps in block */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_num_warps()
{
   return hypre_cuda_get_num_threads<dim>() >> HYPRE_WARP_BITSHIFT;
}

/* return the warp id in block */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_warp_id()
{
   return hypre_cuda_get_thread_id<dim>() >> HYPRE_WARP_BITSHIFT;
}

/* return the thread lane id in warp */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_lane_id()
{
   return hypre_cuda_get_thread_id<dim>() & (HYPRE_WARP_SIZE - 1);
}

/* return the num of blocks in grid */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_num_blocks()
{
   switch (dim)
   {
      case 1:
         return (gridDim.x);
      case 2:
         return (gridDim.x * gridDim.y);
      case 3:
         return (gridDim.x * gridDim.y * gridDim.z);
   }

   return -1;
}

/* return the flattened block id in grid */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_block_id()
{
   switch (dim)
   {
      case 1:
         return (blockIdx.x);
      case 2:
         return (blockIdx.y * gridDim.x + blockIdx.x);
      case 3:
         return (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x +
                 blockIdx.x);
   }

   return -1;
}

/* return the number of threads in grid */
template <hypre_int bdim, hypre_int gdim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_grid_num_threads()
{
   return hypre_cuda_get_num_blocks<gdim>() * hypre_cuda_get_num_threads<bdim>();
}

/* return the flattened thread id in grid */
template <hypre_int bdim, hypre_int gdim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_grid_thread_id()
{
   return hypre_cuda_get_block_id<gdim>() * hypre_cuda_get_num_threads<bdim>() +
          hypre_cuda_get_thread_id<bdim>();
}

/* return the number of warps in grid */
template <hypre_int bdim, hypre_int gdim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_grid_num_warps()
{
   return hypre_cuda_get_num_blocks<gdim>() * hypre_cuda_get_num_warps<bdim>();
}

/* return the flattened warp id in grid */
template <hypre_int bdim, hypre_int gdim>
static __device__ __forceinline__
hypre_int hypre_cuda_get_grid_warp_id()
{
   return hypre_cuda_get_block_id<gdim>() * hypre_cuda_get_num_warps<bdim>() +
          hypre_cuda_get_warp_id<bdim>();
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __device__ __forceinline__
hypre_double atomicAdd(hypre_double* address, hypre_double val)
{
   hypre_ulonglongint* address_as_ull = (hypre_ulonglongint*) address;
   hypre_ulonglongint old = *address_as_ull, assumed;

   do
   {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(val +
                                           __longlong_as_double(assumed)));

      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
   }
   while (assumed != old);

   return __longlong_as_double(old);
}
#endif

// There are no *_sync functions in HIP
#if defined(HYPRE_USING_HIP) || (CUDA_VERSION < 9000)

template <typename T>
static __device__ __forceinline__
T __shfl_sync(unsigned mask, T val, hypre_int src_line, hypre_int width = HYPRE_WARP_SIZE)
{
   return __shfl(val, src_line, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_down_sync(unsigned mask, T val, unsigned delta, hypre_int width = HYPRE_WARP_SIZE)
{
   return __shfl_down(val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_xor_sync(unsigned mask, T val, unsigned lanemask, hypre_int width = HYPRE_WARP_SIZE)
{
   return __shfl_xor(val, lanemask, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_up_sync(unsigned mask, T val, unsigned delta, hypre_int width = HYPRE_WARP_SIZE)
{
   return __shfl_up(val, delta, width);
}

static __device__ __forceinline__
void __syncwarp()
{
}

#endif // #if defined(HYPRE_USING_HIP) || (CUDA_VERSION < 9000)


// __any and __ballot were technically deprecated in CUDA 7 so we don't bother
// with these overloads for CUDA, just for HIP.
#if defined(HYPRE_USING_HIP)
static __device__ __forceinline__
hypre_int __any_sync(unsigned mask, hypre_int predicate)
{
   return __any(predicate);
}

static __device__ __forceinline__
hypre_int __ballot_sync(unsigned mask, hypre_int predicate)
{
   return __ballot(predicate);
}
#endif


template <typename T>
static __device__ __forceinline__
T read_only_load( const T *ptr )
{
   return __ldg( ptr );
}

/* exclusive prefix scan */
template <typename T>
static __device__ __forceinline__
T warp_prefix_sum(hypre_int lane_id, T in, T &all_sum)
{
#pragma unroll
   for (hypre_int d = 2; d <= HYPRE_WARP_SIZE; d <<= 1)
   {
      T t = __shfl_up_sync(HYPRE_WARP_FULL_MASK, in, d >> 1);
      if ( (lane_id & (d - 1)) == (d - 1) )
      {
         in += t;
      }
   }

   all_sum = __shfl_sync(HYPRE_WARP_FULL_MASK, in, HYPRE_WARP_SIZE - 1);

   if (lane_id == HYPRE_WARP_SIZE - 1)
   {
      in = 0;
   }

#pragma unroll
   for (hypre_int d = HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      T t = __shfl_xor_sync(HYPRE_WARP_FULL_MASK, in, d);

      if ( (lane_id & (d - 1)) == (d - 1))
      {
         if ( (lane_id & ((d << 1) - 1)) == ((d << 1) - 1) )
         {
            in += t;
         }
         else
         {
            in = t;
         }
      }
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_reduce_sum(T in)
{
#pragma unroll
   for (hypre_int d = HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in += __shfl_down_sync(HYPRE_WARP_FULL_MASK, in, d);
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_allreduce_sum(T in)
{
#pragma unroll
   for (hypre_int d = HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in += __shfl_xor_sync(HYPRE_WARP_FULL_MASK, in, d);
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_reduce_max(T in)
{
#pragma unroll
   for (hypre_int d = HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = max(in, __shfl_down_sync(HYPRE_WARP_FULL_MASK, in, d));
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_allreduce_max(T in)
{
#pragma unroll
   for (hypre_int d = HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = max(in, __shfl_xor_sync(HYPRE_WARP_FULL_MASK, in, d));
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_reduce_min(T in)
{
#pragma unroll
   for (hypre_int d = HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = min(in, __shfl_down_sync(HYPRE_WARP_FULL_MASK, in, d));
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_allreduce_min(T in)
{
#pragma unroll
   for (hypre_int d = HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = min(in, __shfl_xor_sync(HYPRE_WARP_FULL_MASK, in, d));
   }
   return in;
}

template<typename T>
struct absolute_value : public thrust::unary_function<T, T>
{
   __host__ __device__ T operator()(const T &x) const
   {
      return x < T(0) ? -x : x;
   }
};

template<typename T1, typename T2>
struct TupleComp2
{
   typedef thrust::tuple<T1, T2> Tuple;

   __host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
   {
      if (thrust::get<0>(t1) < thrust::get<0>(t2))
      {
         return true;
      }
      if (thrust::get<0>(t1) > thrust::get<0>(t2))
      {
         return false;
      }
      return hypre_abs(thrust::get<1>(t1)) > hypre_abs(thrust::get<1>(t2));
   }
};

template<typename T1, typename T2>
struct TupleComp3
{
   typedef thrust::tuple<T1, T2> Tuple;

   __host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
   {
      if (thrust::get<0>(t1) < thrust::get<0>(t2))
      {
         return true;
      }
      if (thrust::get<0>(t1) > thrust::get<0>(t2))
      {
         return false;
      }
      if (thrust::get<0>(t2) == thrust::get<1>(t2))
      {
         return false;
      }
      return thrust::get<0>(t1) == thrust::get<1>(t1) || thrust::get<1>(t1) < thrust::get<1>(t2);
   }
};

template<typename T>
struct is_negative : public thrust::unary_function<T, bool>
{
   __host__ __device__ bool operator()(const T &x)
   {
      return (x < 0);
   }
};

template<typename T>
struct is_positive : public thrust::unary_function<T, bool>
{
   __host__ __device__ bool operator()(const T &x)
   {
      return (x > 0);
   }
};

template<typename T>
struct is_nonnegative : public thrust::unary_function<T, bool>
{
   __host__ __device__ bool operator()(const T &x)
   {
      return (x >= 0);
   }
};

template<typename T>
struct in_range : public thrust::unary_function<T, bool>
{
   T low, up;

   in_range(T low_, T up_) { low = low_; up = up_; }

   __host__ __device__ bool operator()(const T &x)
   {
      return (x >= low && x <= up);
   }
};

template<typename T>
struct out_of_range : public thrust::unary_function<T, bool>
{
   T low, up;

   out_of_range(T low_, T up_) { low = low_; up = up_; }

   __host__ __device__ bool operator()(const T &x)
   {
      return (x < low || x > up);
   }
};

template<typename T>
struct less_than : public thrust::unary_function<T, bool>
{
   T val;

   less_than(T val_) { val = val_; }

   __host__ __device__ bool operator()(const T &x)
   {
      return (x < val);
   }
};

template<typename T>
struct modulo : public thrust::unary_function<T, T>
{
   T val;

   modulo(T val_) { val = val_; }

   __host__ __device__ T operator()(const T &x)
   {
      return (x % val);
   }
};

template<typename T>
struct equal : public thrust::unary_function<T, bool>
{
   T val;

   equal(T val_) { val = val_; }

   __host__ __device__ bool operator()(const T &x)
   {
      return (x == val);
   }
};

struct print_functor
{
   __host__ __device__ void operator()(HYPRE_Real val)
   {
      printf("%f\n", val);
   }
};

/* device_utils.c */
template <typename T1, typename T2, typename T3> HYPRE_Int hypreDevice_StableSortByTupleKey(
   HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals, HYPRE_Int opt);

template <typename T1, typename T2, typename T3, typename T4> HYPRE_Int
hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals1, T4 *vals2,
                                      HYPRE_Int opt);

template <typename T1, typename T2, typename T3> HYPRE_Int hypreDevice_ReduceByTupleKey(HYPRE_Int N,
                                                                                        T1 *keys1_in,  T2 *keys2_in,  T3 *vals_in, T1 *keys1_out, T2 *keys2_out, T3 *vals_out);

template <typename T>
HYPRE_Int hypreDevice_ScatterConstant(T *x, HYPRE_Int n, HYPRE_Int *map, T v);

HYPRE_Int hypreDevice_GetRowNnz(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int *d_diag_ia,
                                HYPRE_Int *d_offd_ia, HYPRE_Int *d_rownnz);

HYPRE_Int hypreDevice_CopyParCSRRows(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int job,
                                     HYPRE_Int has_offd, HYPRE_BigInt first_col, HYPRE_BigInt *d_col_map_offd_A, HYPRE_Int *d_diag_i,
                                     HYPRE_Int *d_diag_j, HYPRE_Complex *d_diag_a, HYPRE_Int *d_offd_i, HYPRE_Int *d_offd_j,
                                     HYPRE_Complex *d_offd_a, HYPRE_Int *d_ib, HYPRE_BigInt *d_jb, HYPRE_Complex *d_ab);

HYPRE_Int hypreDevice_GenScatterAdd(HYPRE_Real *x, HYPRE_Int ny, HYPRE_Int *map, HYPRE_Real *y,
                                    char *work);

HYPRE_Int hypreDevice_BigToSmallCopy(HYPRE_Int *tgt, const HYPRE_BigInt *src, HYPRE_Int size);

void hypre_CudaCompileFlagCheck();

#if defined(HYPRE_USING_CUDA)
cudaError_t hypre_CachingMallocDevice(void **ptr, size_t nbytes);

cudaError_t hypre_CachingMallocManaged(void **ptr, size_t nbytes);

cudaError_t hypre_CachingFreeDevice(void *ptr);

cudaError_t hypre_CachingFreeManaged(void *ptr);
#endif

hypre_cub_CachingDeviceAllocator * hypre_DeviceDataCubCachingAllocatorCreate(hypre_uint bin_growth,
                                                                             hypre_uint min_bin, hypre_uint max_bin, size_t max_cached_bytes, bool skip_cleanup, bool debug,
                                                                             bool use_managed_memory);

void hypre_DeviceDataCubCachingAllocatorDestroy(hypre_DeviceData *data);

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(HYPRE_USING_SYCL)

#pragma once

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>

#include <dpct/dpl_extras/algorithm.h> // dpct::remove_if, remove_copy_if, copy_if, scatter_if

#include <algorithm>
#include <numeric>
#include <functional>
#include <iterator>

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 * macro for launching SYCL kernels, SYCL, oneDPL, oneMKL calls
 *                    NOTE: IN HYPRE'S DEFAULT STREAM
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 */

template <typename InputIter1, typename InputIter2,
          typename OutputIter>
OutputIter hypreSycl_gather(InputIter1 map_first, InputIter1 map_last,
                  InputIter2 input_first, OutputIter result) {
  static_assert(
      std::is_same<typename std::iterator_traits<InputIter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<InputIter2>::iterator_category,
              std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<OutputIter>::iterator_category,
              std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto perm_begin =
      oneapi::dpl::make_permutation_iterator(input_first, map_first);
  const int n = ::std::distance(map_first, map_last);

  return oneapi::dpl::copy(oneapi::dpl::execution::make_device_policy(*hypre_HandleComputeStream(hypre_handle())),
			   perm_begin, perm_begin + n, result);
}

#if defined(HYPRE_DEBUG)
#if defined(HYPRE_USING_CUDA)
#define GPU_LAUNCH_SYNC { hypre_SyncDeviceComputeStream(hypre_handle()); HYPRE_CUDA_CALL( cudaGetLastError() ); }
#endif
#else // #if defined(HYPRE_DEBUG)
#define GPU_LAUNCH_SYNC
#endif // defined(HYPRE_DEBUG)

#define HYPRE_GPU_LAUNCH(kernel_name, gridsize, blocksize, ...) \
  ({                                                            \
    hypre_HandleComputeStream(hypre_handle())->parallel_for(    \
      sycl::nd_range(((gridsize) * (blocksize)), (blocksize))   \
      , [=] (auto item)                                         \
      [[intel::reqd_sub_group_size(HYPRE_WARP_SIZE)]] {         \
        (kernel_name)(item, __VA_ARGS__);                       \
      });                                                       \
  })

/* RL: TODO Want macro HYPRE_ONEDPL_CALL to return value but I don't know how to do it right
 * The following one works OK for now */

#define HYPRE_ONEDPL_CALL(func_name, ...) \
  func_name(oneapi::dpl::execution::make_device_policy(*hypre_HandleComputeStream(hypre_handle())), __VA_ARGS__);

// /* return the number of threads in block */
// template <hypre_int dim>
// static __forceinline__
// hypre_int hypre_gpu_get_num_threads()
// {
//    switch (dim)
//    {
//       case 1:
//          return (blockDim.x);
//       case 2:
//          return (blockDim.x * blockDim.y);
//       case 3:
//          return (blockDim.x * blockDim.y * blockDim.z);
//    }

//    return -1;
// }

/* return the number of (sub_groups) warps in (work-group) block */
template <hypre_int dim>
static __forceinline__
hypre_int hypre_gpu_get_num_warps(sycl::nd_item<dim>& item)
{
   return item.get_sub_group().get_group_range().get(0);
}

/* return the thread lane id in warp */
template <hypre_int dim>
static __forceinline__
hypre_int hypre_gpu_get_lane_id(sycl::nd_item<dim>& item)
{
   return item.get_sub_group().get_local_linear_id();
}

// /* return the number of threads in grid */
// template <hypre_int bdim, hypre_int gdim>
// static __forceinline__
// hypre_int hypre_gpu_get_grid_num_threads()
// {
//    return hypre_gpu_get_num_blocks<gdim>() * hypre_gpu_get_num_threads<bdim>();
// }

/* return the flattened work-item/thread id in global work space,
 * Note: Since the use-cases always involved bdim = gdim = 1, the
 * sycl::nd_item<1> is only being used. SFINAE is used to prevent
 * other dimensions (i.e., bdim != gdim != 1) */
template < hypre_int bdim, hypre_int gdim >
static __forceinline__
hypre_int hypre_gpu_get_grid_thread_id(sycl::nd_item<1>& item)
{
   static_assert(bdim == 1 && gdim == 1);
   return item.get_global_id(0);
}

// /* return the number of warps in grid */
// template <hypre_int bdim, hypre_int gdim>
// static __forceinline__
// hypre_int hypre_gpu_get_grid_num_warps()
// {
//    return hypre_gpu_get_num_blocks<gdim>() * hypre_gpu_get_num_warps<bdim>();
// }

/* return the flattened warp id in grid */
template <hypre_int bdim, hypre_int gdim>
static __forceinline__
hypre_int hypre_gpu_get_grid_warp_id(sycl::nd_item<1>& item)
{
  return item.get_group(0) * hypre_gpu_get_num_warps<bdim>(item) +
     item.get_sub_group().get_group_linear_id();
}

template <typename T>
static __forceinline__
T atomicCAS(T* address, T expected, T val)
{
    sycl::ext::oneapi::atomic_ref< T,
                                   sycl::ext::oneapi::memory_order::relaxed,
                                   sycl::ext::oneapi::memory_scope::device,
                                   sycl::access::address_space::local_space >
      atomicCAS_ref( *address );
    return static_cast<T>( atomicCAS_ref.compare_exchange_strong(expected, val) );
}

template <typename T>
static __forceinline__
T atomicAdd(T* address, T val)
{
    sycl::ext::oneapi::atomic_ref< T,
                                   sycl::ext::oneapi::memory_order::relaxed,
                                   sycl::ext::oneapi::memory_scope::device,
                                   sycl::access::address_space::local_space >
      atomicAdd_ref( *address );
    return atomicAdd_ref.fetch_add(val);
}

template <typename T>
static __forceinline__
T read_only_load( const T *ptr )
{
   return *ptr;
}

/* exclusive prefix scan */
template <typename T, int DIM>
static __forceinline__
T warp_prefix_sum(hypre_int lane_id, T in, T &all_sum, sycl::nd_item<DIM>& item)
{
  //sycl::exclusive_scan_over_group(item.get_sub_group(), in, std::plus<>());

   sycl::sub_group SG = item.get_sub_group();
   HYPRE_Int warp_size = SG.get_local_range().get(0);

#pragma unroll
   for (hypre_int d = 2; d <=warp_size; d <<= 1)
   {
      T t = SG.shuffle(in, d >> 1);
      if ( (lane_id & (d - 1)) == (d - 1) )
      {
         in += t;
      }
   }

   all_sum = SG.shuffle(in, warp_size-1);

   if (lane_id == warp_size-1)
   {
      in = 0;
   }

#pragma unroll
   for (hypre_int d = warp_size/2; d > 0; d >>= 1)
   {
      T t = SG.shuffle_xor(in, d);

      if ( (lane_id & (d - 1)) == (d - 1))
      {
        if ( (lane_id & ((d << 1) - 1)) == ((d << 1) - 1) )
         {
            in += t;
         }
         else
         {
            in = t;
         }
      }
   }
   return in;
}

template <typename T, int DIM>
static __forceinline__
T warp_reduce_sum(T in, sycl::nd_item<DIM>& item)
{
  sycl::sub_group SG = item.get_sub_group();
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in += SG.shuffle_down(in, d);
  }
  return in;
}

template <typename T, int DIM>
static __forceinline__
T warp_allreduce_sum(T in, sycl::nd_item<DIM>& item)
{
  sycl::sub_group SG = item.get_sub_group();
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in += SG.shuffle_xor(in, d);
  }
  return in;
}

template <typename T, int DIM>
static __forceinline__
T warp_reduce_max(T in, sycl::nd_item<DIM>& item)
{
  sycl::sub_group SG = item.get_sub_group();
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in = std::max(in, SG.shuffle_down(in, d));
  }
  return in;
}

template <typename T, int DIM>
static __forceinline__
T warp_allreduce_max(T in, sycl::nd_item<DIM>& item)
{
  sycl::sub_group SG = item.get_sub_group();
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in = std::max(in, SG.shuffle_xor(in, d));
  }
  return in;
}

template <typename T, int DIM>
static __forceinline__
T warp_reduce_min(T in, sycl::nd_item<DIM>& item)
{
  sycl::sub_group SG = item.get_sub_group();
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in = std::min(in, SG.shuffle_down(in, d));
  }
  return in;
}

template <typename T, int DIM>
static __forceinline__
T warp_allreduce_min(T in, sycl::nd_item<DIM>& item)
{
  sycl::sub_group SG = item.get_sub_group();
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in = std::min(in, SG.shuffle_xor(in, d));
  }
  return in;
}

// static __forceinline__
// hypre_int next_power_of_2(hypre_int n)
// {
//    if (n <= 0)
//    {
//       return 0;
//    }

//    /* if n is power of 2, return itself */
//    if ( (n & (n - 1)) == 0 )
//    {
//       return n;
//    }

//    n |= (n >>  1);
//    n |= (n >>  2);
//    n |= (n >>  4);
//    n |= (n >>  8);
//    n |= (n >> 16);
//    n ^= (n >>  1);
//    n  = (n <<  1);

//    return n;
// }

// template<typename T>
// struct absolute_value : public thrust::unary_function<T,T>
// {
//   T operator()(const T &x) const
//   {
//     return x < T(0) ? -x : x;
//   }
// };

// template<typename T1, typename T2>
// struct TupleComp2
// {
//    typedef thrust::tuple<T1, T2> Tuple;

//    bool operator()(const Tuple& t1, const Tuple& t2)
//    {
//       if (thrust::get<0>(t1) < thrust::get<0>(t2))
//       {
//          return true;
//       }
//       if (thrust::get<0>(t1) > thrust::get<0>(t2))
//       {
//          return false;
//       }
//       return hypre_abs(thrust::get<1>(t1)) > hypre_abs(thrust::get<1>(t2));
//    }
// };

// template<typename T1, typename T2>
// struct TupleComp3
// {
//    typedef thrust::tuple<T1, T2> Tuple;

//    bool operator()(const Tuple& t1, const Tuple& t2)
//    {
//       if (thrust::get<0>(t1) < thrust::get<0>(t2))
//       {
//          return true;
//       }
//       if (thrust::get<0>(t1) > thrust::get<0>(t2))
//       {
//          return false;
//       }
//       if (thrust::get<0>(t2) == thrust::get<1>(t2))
//       {
//          return false;
//       }
//       return thrust::get<0>(t1) == thrust::get<1>(t1) || thrust::get<1>(t1) < thrust::get<1>(t2);
//    }
// };

// template<typename T>
// struct is_negative : public thrust::unary_function<T,bool>
// {
//    bool operator()(const T &x)
//    {
//       return (x < 0);
//    }
// };

// template<typename T>
// struct is_positive : public thrust::unary_function<T,bool>
// {
//    bool operator()(const T &x)
//    {
//       return (x > 0);
//    }
// };

// template<typename T>
// struct is_nonnegative : public thrust::unary_function<T,bool>
// {
//    bool operator()(const T &x)
//    {
//       return (x >= 0);
//    }
// };

template<typename T>
struct in_range
{
   T low, up;
   in_range(T low_, T up_) { low = low_; up = up_; }

   bool operator()(const T &x) const { return (x >= low && x <= up); }
};

// template<typename T>
// struct out_of_range : public thrust::unary_function<T,bool>
// {
//    T low, up;

//    out_of_range(T low_, T up_) { low = low_; up = up_; }

//    bool operator()(const T &x)
//    {
//       return (x < low || x > up);
//    }
// };

#ifdef HYPRE_COMPLEX
template<typename T,
	 typename = typename std::enable_if<std::is_same<T, HYPRE_Complex>::value>::type>
struct less_than
{
  T val;
  less_than(T val_) { val = val_; }
  bool operator()(const T &x) const { return (hypre_abs(x) < hypre_abs(val)); }
};
#else
template<typename T,
	 typename = typename std::enable_if<std::is_same<T, HYPRE_Real>::value>::type>
struct less_than
{
  T val;
  less_than(T val_) { val = val_; }
  bool operator()(const T &x) const { return (x < val); }
};
#endif
// template<typename T>
// struct modulo : public thrust::unary_function<T,T>
// {
//    T val;

//    modulo(T val_) { val = val_; }

//    T operator()(const T &x)
//    {
//       return (x % val);
//    }
// };

// template<typename T>
// struct equal : public thrust::unary_function<T,bool>
// {
//    T val;

//    equal(T val_) { val = val_; }

//    bool operator()(const T &x)
//    {
//       return (x == val);
//    }
// };

// struct print_functor
// {
//    void operator()(HYPRE_Real val)
//    {
//       printf("%f\n", val);
//    }
// };

#endif // #if defined(HYPRE_USING_SYCL)

////////////////////////////////////////////////////////////////////////////////////////

#if defined(HYPRE_USING_CUSPARSE)

cudaDataType hypre_HYPREComplexToCudaDataType();

cusparseIndexType_t hypre_HYPREIntToCusparseIndexType();

#endif // #if defined(HYPRE_USING_CUSPARSE)

#endif /* #ifndef HYPRE_CUDA_UTILS_H */
