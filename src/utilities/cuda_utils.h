/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_CUDA_UTILS_H
#define HYPRE_CUDA_UTILS_H

#if defined(HYPRE_USING_GPU)

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

#elif defined(HYPRE_USING_HIP)

#include <hip/hip_runtime.h>

#endif // defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

#if defined(HYPRE_USING_ROCSPARSE)
#include <rocsparse.h>
#endif

#if defined(HYPRE_USING_ROCRAND)
#include <rocrand.h>
#endif


#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#define HYPRE_CUDA_CALL(call) do {                                                           \
   cudaError_t err = call;                                                                   \
   if (cudaSuccess != err) {                                                                 \
      hypre_printf("CUDA ERROR (code = %d, %s) at %s:%d\n", err, cudaGetErrorString(err),    \
                   __FILE__, __LINE__);                                                      \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

#elif defined(HYPRE_USING_HIP)
#define HYPRE_HIP_CALL(call) do {                                                            \
   hipError_t err = call;                                                                    \
   if (hipSuccess != err) {                                                                  \
      hypre_printf("HIP ERROR (code = %d, %s) at %s:%d\n", err, hipGetErrorString(err),      \
                   __FILE__, __LINE__);                                                      \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

#endif // defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

#define HYPRE_CUBLAS_CALL(call) do {                                                         \
   cublasStatus_t err = call;                                                                \
   if (CUBLAS_STATUS_SUCCESS != err) {                                                       \
      hypre_printf("CUBLAS ERROR (code = %d, %d) at %s:%d\n",                                \
            err, err == CUBLAS_STATUS_EXECUTION_FAILED, __FILE__, __LINE__);                 \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define HYPRE_CUSPARSE_CALL(call) do {                                                       \
   cusparseStatus_t err = call;                                                              \
   if (CUSPARSE_STATUS_SUCCESS != err) {                                                     \
      hypre_printf("CUSPARSE ERROR (code = %d, %s) at %s:%d\n",                              \
            err, cusparseGetErrorString(err), __FILE__, __LINE__);                           \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define HYPRE_ROCSPARSE_CALL(call) do {                                                      \
   rocsparse_status err = call;                                                              \
   if (rocsparse_status_success != err) {                                                    \
      hypre_printf("rocSPARSE ERROR (code = %d) at %s:%d\n",                                 \
            err, __FILE__, __LINE__);                                                        \
      assert(0); exit(1);                                                                    \
   } } while(0)


#define HYPRE_CURAND_CALL(call) do {                                                         \
   curandStatus_t err = call;                                                                \
   if (CURAND_STATUS_SUCCESS != err) {                                                       \
      hypre_printf("CURAND ERROR (code = %d) at %s:%d\n", err, __FILE__, __LINE__);          \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define HYPRE_ROCRAND_CALL(call) do {                                                        \
   rocrand_status err = call;                                                                \
   if (ROCRAND_STATUS_SUCCESS != err) {                                                      \
      hypre_printf("ROCRAND ERROR (code = %d) at %s:%d\n", err, __FILE__, __LINE__);         \
      hypre_assert(0); exit(1);                                                              \
   } } while(0)

struct hypre_cub_CachingDeviceAllocator;
typedef struct hypre_cub_CachingDeviceAllocator hypre_cub_CachingDeviceAllocator;

// HYPRE_WARP_BITSHIFT is just log2 of HYPRE_WARP_SIZE
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
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

struct hypre_CudaData
{
#if defined(HYPRE_USING_CURAND)
   curandGenerator_t                 curand_generator;
#endif

#if defined(HYPRE_USING_ROCRAND)
   rocrand_generator                 curand_generator;
#endif

#if defined(HYPRE_USING_CUBLAS)
   cublasHandle_t                    cublas_handle;
#endif

#if defined(HYPRE_USING_CUSPARSE)
   cusparseHandle_t                  cusparse_handle;
#endif

#if defined(HYPRE_USING_ROCSPARSE)
   rocsparse_handle                  cusparse_handle;
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   cudaStream_t                      cuda_streams[HYPRE_MAX_NUM_STREAMS];
#elif defined(HYPRE_USING_HIP)
   hipStream_t                       cuda_streams[HYPRE_MAX_NUM_STREAMS];
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

   HYPRE_Int                         cuda_device;
   /* by default, hypre puts GPU computations in this stream
    * Do not be confused with the default (null) CUDA stream */
   HYPRE_Int                         cuda_compute_stream_num;
   /* work space for hypre's CUDA reducer */
   void                             *cuda_reduce_buffer;
   /* the device buffers needed to do MPI communication for struct comm */
   HYPRE_Complex*                    struct_comm_recv_buffer;
   HYPRE_Complex*                    struct_comm_send_buffer;
   HYPRE_Int                         struct_comm_recv_buffer_size;
   HYPRE_Int                         struct_comm_send_buffer_size;
   /* device spgemm options */
   HYPRE_Int                         spgemm_use_cusparse;
   HYPRE_Int                         spgemm_num_passes;
   HYPRE_Int                         spgemm_rownnz_estimate_method;
   HYPRE_Int                         spgemm_rownnz_estimate_nsamples;
   float                             spgemm_rownnz_estimate_mult_factor;
   char                              spgemm_hash_type;
   /* PMIS */
   HYPRE_Int                         use_gpu_rand;
};

#define hypre_CudaDataCubBinGrowth(data)                   ((data) -> cub_bin_growth)
#define hypre_CudaDataCubMinBin(data)                      ((data) -> cub_min_bin)
#define hypre_CudaDataCubMaxBin(data)                      ((data) -> cub_max_bin)
#define hypre_CudaDataCubMaxCachedBytes(data)              ((data) -> cub_max_cached_bytes)
#define hypre_CudaDataCubDevAllocator(data)                ((data) -> cub_dev_allocator)
#define hypre_CudaDataCubUvmAllocator(data)                ((data) -> cub_uvm_allocator)
#define hypre_CudaDataCudaDevice(data)                     ((data) -> cuda_device)
#define hypre_CudaDataCudaComputeStreamNum(data)           ((data) -> cuda_compute_stream_num)
#define hypre_CudaDataCudaReduceBuffer(data)               ((data) -> cuda_reduce_buffer)
#define hypre_CudaDataStructCommRecvBuffer(data)           ((data) -> struct_comm_recv_buffer)
#define hypre_CudaDataStructCommSendBuffer(data)           ((data) -> struct_comm_send_buffer)
#define hypre_CudaDataStructCommRecvBufferSize(data)       ((data) -> struct_comm_recv_buffer_size)
#define hypre_CudaDataStructCommSendBufferSize(data)       ((data) -> struct_comm_send_buffer_size)
#define hypre_CudaDataSpgemmUseCusparse(data)              ((data) -> spgemm_use_cusparse)
#define hypre_CudaDataSpgemmNumPasses(data)                ((data) -> spgemm_num_passes)
#define hypre_CudaDataSpgemmRownnzEstimateMethod(data)     ((data) -> spgemm_rownnz_estimate_method)
#define hypre_CudaDataSpgemmRownnzEstimateNsamples(data)   ((data) -> spgemm_rownnz_estimate_nsamples)
#define hypre_CudaDataSpgemmRownnzEstimateMultFactor(data) ((data) -> spgemm_rownnz_estimate_mult_factor)
#define hypre_CudaDataSpgemmHashType(data)                 ((data) -> spgemm_hash_type)
#define hypre_CudaDataDeviceAllocator(data)                ((data) -> device_allocator)
#define hypre_CudaDataUseGpuRand(data)                     ((data) -> use_gpu_rand)

hypre_CudaData*     hypre_CudaDataCreate();
void                hypre_CudaDataDestroy(hypre_CudaData* data);

#if defined(HYPRE_USING_CURAND)
curandGenerator_t   hypre_CudaDataCurandGenerator(hypre_CudaData *data);
#endif

#if defined(HYPRE_USING_ROCRAND)
rocrand_generator   hypre_CudaDataCurandGenerator(hypre_CudaData *data);
#endif

#if defined(HYPRE_USING_CUBLAS)
cublasHandle_t      hypre_CudaDataCublasHandle(hypre_CudaData *data);
#endif

#if defined(HYPRE_USING_CUSPARSE)
cusparseHandle_t    hypre_CudaDataCusparseHandle(hypre_CudaData *data);
#endif

#if defined(HYPRE_USING_ROCSPARSE)
rocsparse_handle    hypre_CudaDataCusparseHandle(hypre_CudaData *data);
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
cudaStream_t        hypre_CudaDataCudaStream(hypre_CudaData *data, HYPRE_Int i);
cudaStream_t        hypre_CudaDataCudaComputeStream(hypre_CudaData *data);
#elif defined(HYPRE_USING_HIP)
hipStream_t         hypre_CudaDataCudaStream(hypre_CudaData *data, HYPRE_Int i);
hipStream_t         hypre_CudaDataCudaComputeStream(hypre_CudaData *data);
#endif

// Data structure and accessor routines for Cuda Sparse Triangular Matrices
struct hypre_CsrsvData
{
#if defined(HYPRE_USING_CUSPARSE)
   csrsv2Info_t info_L;
   csrsv2Info_t info_U;
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
#endif

#if defined(HYPRE_USING_ROCSPARSE)
   rocsparse_mat_descr   mat_descr;
   rocsparse_mat_info    mat_info;
#endif
};

#define hypre_GpuMatDataMatDecsr(data) ((data) -> mat_descr)
#define hypre_GpuMatDataMatInfo(data)  ((data) -> mat_info)

#endif //#if defined(HYPRE_USING_GPU)

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
#define HYPRE_CUDA_LAUNCH(kernel_name, gridsize, blocksize, ...)                                                     \
{                                                                                                                    \
   if ( gridsize.x  == 0 || gridsize.y  == 0 || gridsize.z  == 0 ||                                                  \
        blocksize.x == 0 || blocksize.y == 0 || blocksize.z == 0 )                                                   \
   {                                                                                                                 \
      /* hypre_printf("Warning %s %d: Zero CUDA grid/block (%d %d %d) (%d %d %d)\n",                                 \
                 __FILE__, __LINE__,                                                                                 \
                 gridsize.x, gridsize.y, gridsize.z, blocksize.x, blocksize.y, blocksize.z); */                      \
   }                                                                                                                 \
   else                                                                                                              \
   {                                                                                                                 \
      (kernel_name) <<< (gridsize), (blocksize), 0, hypre_HandleCudaComputeStream(hypre_handle()) >>> (__VA_ARGS__); \
   }                                                                                                                 \
   hypre_SyncCudaComputeStream(hypre_handle());                                                                      \
   HYPRE_CUDA_CALL( cudaGetLastError() );                                                                            \
}
#elif defined(HYPRE_USING_HIP)
#define HYPRE_CUDA_LAUNCH(kernel_name, gridsize, blocksize, ...)                                                     \
{                                                                                                                    \
   if ( gridsize.x  == 0 || gridsize.y  == 0 || gridsize.z  == 0 ||                                                  \
        blocksize.x == 0 || blocksize.y == 0 || blocksize.z == 0 )                                                   \
   {                                                                                                                 \
      /* hypre_printf("Warning %s %d: Zero CUDA grid/block (%d %d %d) (%d %d %d)\n",                                 \
                 __FILE__, __LINE__,                                                                                 \
                 gridsize.x, gridsize.y, gridsize.z, blocksize.x, blocksize.y, blocksize.z); */                      \
   }                                                                                                                 \
   else                                                                                                              \
   {                                                                                                                 \
      (kernel_name) <<< (gridsize), (blocksize), 0, hypre_HandleCudaComputeStream(hypre_handle()) >>> (__VA_ARGS__); \
   }                                                                                                                 \
   hypre_SyncCudaComputeStream(hypre_handle());                                                                      \
   HYPRE_HIP_CALL( hipGetLastError() );                                                                            \
}
#endif //HYPRE_USING_CUDA

#else // #if defined(HYPRE_DEBUG)

#define HYPRE_CUDA_LAUNCH(kernel_name, gridsize, blocksize, ...)                                                     \
{                                                                                                                    \
   if ( gridsize.x  == 0 || gridsize.y  == 0 || gridsize.z  == 0 ||                                                  \
        blocksize.x == 0 || blocksize.y == 0 || blocksize.z == 0 )                                                   \
   {                                                                                                                 \
      /* hypre_printf("Warning %s %d: Zero CUDA grid/block (%d %d %d) (%d %d %d)\n",                                 \
                 __FILE__, __LINE__,                                                                                 \
                 gridsize.x, gridsize.y, gridsize.z, blocksize.x, blocksize.y, blocksize.z); */                      \
   }                                                                                                                 \
   else                                                                                                              \
   {                                                                                                                 \
      (kernel_name) <<< (gridsize), (blocksize), 0, hypre_HandleCudaComputeStream(hypre_handle()) >>> (__VA_ARGS__); \
   }                                                                                                                 \
}

#endif // defined(HYPRE_DEBUG)

/* RL: TODO Want macro HYPRE_THRUST_CALL to return value but I don't know how to do it right
 * The following one works OK for now */

#if defined(HYPRE_USING_CUDA)
#define HYPRE_THRUST_CALL(func_name, ...) \
   thrust::func_name(thrust::cuda::par(hypre_HandleDeviceAllocator(hypre_handle())).on(hypre_HandleCudaComputeStream(hypre_handle())), __VA_ARGS__);
#elif defined(HYPRE_USING_HIP)
#define HYPRE_THRUST_CALL(func_name, ...) \
   thrust::func_name(thrust::hip::par(hypre_HandleDeviceAllocator(hypre_handle())).on(hypre_HandleCudaComputeStream(hypre_handle())), __VA_ARGS__);
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
   return hypre_cuda_get_thread_id<dim>() & (HYPRE_WARP_SIZE-1);
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

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

// There are no *_sync functions in HIP
#if defined(HYPRE_USING_HIP) || (CUDA_VERSION < 9000)

template <typename T>
static __device__ __forceinline__
T __shfl_sync(unsigned mask, T val, hypre_int src_line, hypre_int width=HYPRE_WARP_SIZE)
{
   return __shfl(val, src_line, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_down_sync(unsigned mask, T val, unsigned delta, hypre_int width=HYPRE_WARP_SIZE)
{
   return __shfl_down(val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_xor_sync(unsigned mask, T val, unsigned lanemask, hypre_int width=HYPRE_WARP_SIZE)
{
   return __shfl_xor(val, lanemask, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_up_sync(unsigned mask, T val, unsigned delta, hypre_int width=HYPRE_WARP_SIZE)
{
   return __shfl_up(val, delta, width);
}

static __device__ __forceinline__
void __syncwarp()
{
}

#endif // #if defined(HYPRE_USING_HIP) || (CUDA_VERSION < 9000)


// __any was technically deprecated in CUDA 7 so we don't bother
// with this overload for CUDA, just for HIP.
#if defined(HYPRE_USING_HIP)
static __device__ __forceinline__
hypre_int __any_sync(unsigned mask, hypre_int predicate)
{
  return __any(predicate);
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
   for (hypre_int d = 2; d <=HYPRE_WARP_SIZE; d <<= 1)
   {
      T t = __shfl_up_sync(HYPRE_WARP_FULL_MASK, in, d >> 1);
      if ( (lane_id & (d - 1)) == (d - 1) )
      {
         in += t;
      }
   }

   all_sum = __shfl_sync(HYPRE_WARP_FULL_MASK, in, HYPRE_WARP_SIZE-1);

   if (lane_id == HYPRE_WARP_SIZE-1)
   {
      in = 0;
   }

#pragma unroll
   for (hypre_int d = HYPRE_WARP_SIZE/2; d > 0; d >>= 1)
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
  for (hypre_int d = HYPRE_WARP_SIZE/2; d > 0; d >>= 1)
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
  for (hypre_int d = HYPRE_WARP_SIZE/2; d > 0; d >>= 1)
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
  for (hypre_int d = HYPRE_WARP_SIZE/2; d > 0; d >>= 1)
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
  for (hypre_int d = HYPRE_WARP_SIZE/2; d > 0; d >>= 1)
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
  for (hypre_int d = HYPRE_WARP_SIZE/2; d > 0; d >>= 1)
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
  for (hypre_int d = HYPRE_WARP_SIZE/2; d > 0; d >>= 1)
  {
    in = min(in, __shfl_xor_sync(HYPRE_WARP_FULL_MASK, in, d));
  }
  return in;
}

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

template<typename T>
struct absolute_value : public thrust::unary_function<T,T>
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
struct is_negative : public thrust::unary_function<T,bool>
{
   __host__ __device__ bool operator()(const T &x)
   {
      return (x < 0);
   }
};

template<typename T>
struct is_positive : public thrust::unary_function<T,bool>
{
   __host__ __device__ bool operator()(const T &x)
   {
      return (x > 0);
   }
};

template<typename T>
struct is_nonnegative : public thrust::unary_function<T,bool>
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
struct out_of_range : public thrust::unary_function<T,bool>
{
   T low, up;

   out_of_range(T low_, T up_) { low = low_; up = up_; }

   __host__ __device__ bool operator()(const T &x)
   {
      return (x < low || x > up);
   }
};

template<typename T>
struct less_than : public thrust::unary_function<T,bool>
{
   T val;

   less_than(T val_) { val = val_; }

   __host__ __device__ bool operator()(const T &x)
   {
      return (x < val);
   }
};

template<typename T>
struct equal : public thrust::unary_function<T,bool>
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

/* cuda_utils.c */
dim3 hypre_GetDefaultCUDABlockDimension();

dim3 hypre_GetDefaultCUDAGridDimension( HYPRE_Int n, const char *granularity, dim3 bDim );

template <typename T1, typename T2, typename T3> HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals, HYPRE_Int opt);

template <typename T1, typename T2, typename T3, typename T4> HYPRE_Int hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals1, T4 *vals2, HYPRE_Int opt);

template <typename T1, typename T2, typename T3> HYPRE_Int hypreDevice_ReduceByTupleKey(HYPRE_Int N, T1 *keys1_in,  T2 *keys2_in,  T3 *vals_in, T1 *keys1_out, T2 *keys2_out, T3 *vals_out);

template <typename T>
HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr, T *d_row_num, T *d_row_ind);

template <typename T>
HYPRE_Int hypreDevice_ScatterConstant(T *x, HYPRE_Int n, HYPRE_Int *map, T v);

HYPRE_Int hypreDevice_GetRowNnz(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int *d_diag_ia, HYPRE_Int *d_offd_ia, HYPRE_Int *d_rownnz);

HYPRE_Int hypreDevice_CopyParCSRRows(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int job, HYPRE_Int has_offd, HYPRE_BigInt first_col, HYPRE_BigInt *d_col_map_offd_A, HYPRE_Int *d_diag_i, HYPRE_Int *d_diag_j, HYPRE_Complex *d_diag_a, HYPRE_Int *d_offd_i, HYPRE_Int *d_offd_j, HYPRE_Complex *d_offd_a, HYPRE_Int *d_ib, HYPRE_BigInt *d_jb, HYPRE_Complex *d_ab);

HYPRE_Int hypreDevice_IntegerReduceSum(HYPRE_Int m, HYPRE_Int *d_i);

HYPRE_Int hypreDevice_IntegerInclusiveScan(HYPRE_Int n, HYPRE_Int *d_i);

HYPRE_Int hypreDevice_IntegerExclusiveScan(HYPRE_Int n, HYPRE_Int *d_i);

HYPRE_Int* hypreDevice_CsrRowPtrsToIndices(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr);

HYPRE_Int hypreDevice_CsrRowPtrsToIndices_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr, HYPRE_Int *d_row_ind);

HYPRE_Int* hypreDevice_CsrRowIndicesToPtrs(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind);

HYPRE_Int hypreDevice_CsrRowIndicesToPtrs_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind, HYPRE_Int *d_row_ptr);

HYPRE_Int hypreDevice_GenScatterAdd(HYPRE_Real *x, HYPRE_Int ny, HYPRE_Int *map, HYPRE_Real *y, char *work);

HYPRE_Int hypreDevice_BigToSmallCopy(HYPRE_Int *tgt, const HYPRE_BigInt *src, HYPRE_Int size);

void hypre_CudaCompileFlagCheck();

#if defined(HYPRE_USING_CUDA)
cudaError_t hypre_CachingMallocDevice(void **ptr, size_t nbytes);

cudaError_t hypre_CachingMallocManaged(void **ptr, size_t nbytes);

cudaError_t hypre_CachingFreeDevice(void *ptr);

cudaError_t hypre_CachingFreeManaged(void *ptr);
#endif

hypre_cub_CachingDeviceAllocator * hypre_CudaDataCubCachingAllocatorCreate(hypre_uint bin_growth, hypre_uint min_bin, hypre_uint max_bin, size_t max_cached_bytes, bool skip_cleanup, bool debug, bool use_managed_memory);

void hypre_CudaDataCubCachingAllocatorDestroy(hypre_CudaData *data);

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#if defined(HYPRE_USING_CUSPARSE)

cudaDataType hypre_HYPREComplexToCudaDataType();

cusparseIndexType_t hypre_HYPREIntToCusparseIndexType();

#endif // #if defined(HYPRE_USING_CUSPARSE)

#endif /* #ifndef HYPRE_CUDA_UTILS_H */
