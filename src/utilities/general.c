/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#ifdef HYPRE_USING_MEMORY_TRACKER
hypre_MemoryTracker *_hypre_memory_tracker = NULL;

/* accessor to the global ``_hypre_memory_tracker'' */
hypre_MemoryTracker*
hypre_memory_tracker()
{
   if (!_hypre_memory_tracker)
   {
      _hypre_memory_tracker = hypre_MemoryTrackerCreate();
   }

   return _hypre_memory_tracker;
}
#endif

/* global variable _hypre_handle:
 * Outside this file, do NOT access it directly,
 * but use hypre_handle() instead (see handle.h) */
hypre_Handle *_hypre_handle = NULL;

/* accessor to the global ``_hypre_handle'' */
hypre_Handle*
hypre_handle()
{
   if (!_hypre_handle)
   {
      _hypre_handle = hypre_HandleCreate();
   }

   return _hypre_handle;
}

hypre_Handle*
hypre_HandleCreate()
{
   hypre_Handle *hypre_handle_ = hypre_CTAlloc(hypre_Handle, 1, HYPRE_MEMORY_HOST);

   hypre_HandleMemoryLocation(hypre_handle_) = HYPRE_MEMORY_DEVICE;

#if defined(HYPRE_USING_GPU)
   hypre_HandleDefaultExecPolicy(hypre_handle_) = HYPRE_EXEC_DEVICE;
   hypre_HandleStructExecPolicy(hypre_handle_) = HYPRE_EXEC_DEVICE;
   hypre_HandleCudaData(hypre_handle_) = hypre_CudaDataCreate();
   /* Gauss-Seidel: SpTrSV */
   hypre_HandleDeviceGSMethod(hypre_handle_) = 1; /* CPU: 0; Cusparse: 1 */
#endif

   return hypre_handle_;
}

HYPRE_Int
hypre_HandleDestroy(hypre_Handle *hypre_handle_)
{
   if (!hypre_handle_)
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   hypre_CudaDataDestroy(hypre_HandleCudaData(hypre_handle_));
#endif

   hypre_TFree(hypre_handle_, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetDevice(hypre_int device_id, hypre_Handle *hypre_handle_)
{

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaSetDevice(device_id) );
   omp_set_default_device(device_id);
#endif

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaSetDevice(device_id) );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipSetDevice(device_id) );
#endif

#if defined(HYPRE_USING_GPU)
   if (hypre_handle_)
   {
      hypre_HandleCudaDevice(hypre_handle_) = device_id;
   }
#endif

   return hypre_error_flag;
}

/* Note: it doesn't return device_id in hypre_Handle->hypre_CudaData,
 *       calls API instead. But these two should match at all times
 */
HYPRE_Int
hypre_GetDevice(hypre_int *device_id)
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
   *device_id = omp_get_default_device();
#endif

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetDevice(device_id) );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipGetDevice(device_id) );
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_GetDeviceCount(hypre_int *device_count)
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
   *device_count = omp_get_num_devices();
#endif

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetDeviceCount(device_count) );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipGetDeviceCount(device_count) );
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_GetDeviceLastError()
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaGetLastError() );
#endif

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetLastError() );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipGetLastError() );
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre initialization
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_Init()
{
#ifdef HYPRE_USING_MEMORY_TRACKER
   if (!_hypre_memory_tracker)
   {
      _hypre_memory_tracker = hypre_MemoryTrackerCreate();
   }
#endif

   if (!_hypre_handle)
   {
      _hypre_handle = hypre_HandleCreate();
   }

#if defined(HYPRE_USING_GPU)
   hypre_GetDeviceLastError();

   /* Notice: the cudaStream created is specific to the device
    * that was in effect when you created the stream.
    * So, we should first set the device and create the streams
    */
   hypre_int device_id;
   hypre_GetDevice(&device_id);
   hypre_SetDevice(device_id, _hypre_handle);

   /* To include the cost of creating streams/cudahandles in HYPRE_Init */
   /* If not here, will be done at the first use */
   hypre_HandleCudaComputeStream(_hypre_handle);

   /* A separate stream for prefetching */
   //hypre_HandleCudaPrefetchStream(_hypre_handle);
#endif // HYPRE_USING_GPU

#if defined(HYPRE_USING_CUBLAS)
   hypre_HandleCublasHandle(_hypre_handle);
#endif

#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)
   hypre_HandleCusparseHandle(_hypre_handle);
#endif

#if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND)
   hypre_HandleCurandGenerator(_hypre_handle);
#endif

   /* Check if cuda arch flags in compiling match the device */
#if defined(HYPRE_USING_CUDA)
   hypre_CudaCompileFlagCheck();
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_OMPOffloadOn();
#endif

#ifdef HYPRE_USING_DEVICE_POOL
   /* Keep this check here at the end of HYPRE_Init()
    * Make sure that device pool allocator has not been setup in HYPRE_Init,
    * otherwise users are not able to set all the parameters
    */
   if ( hypre_HandleCubDevAllocator(_hypre_handle) ||
        hypre_HandleCubUvmAllocator(_hypre_handle) )
   {
      char msg[256];
      hypre_sprintf(msg, "%s %s", "ERROR: device pool allocators have been created in", __func__);
      hypre_fprintf(stderr, "%s\n", msg);
      hypre_error_w_msg(-1, msg);
   }
#endif

#if defined(HYPRE_USING_UMPIRE)
   hypre_UmpireInit(_hypre_handle);
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre finalization
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_Finalize()
{
#if defined(HYPRE_USING_UMPIRE)
   hypre_UmpireFinalize(_hypre_handle);
#endif

   hypre_HandleDestroy(_hypre_handle);

   _hypre_handle = NULL;

   hypre_GetDeviceLastError();

#ifdef HYPRE_USING_MEMORY_TRACKER
   hypre_PrintMemoryTracker();
   hypre_MemoryTrackerDestroy(_hypre_memory_tracker);
#endif

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_PrintDeviceInfo()
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
  hypre_int dev;
  struct cudaDeviceProp deviceProp;

  HYPRE_CUDA_CALL( cudaGetDevice(&dev) );
  HYPRE_CUDA_CALL( cudaGetDeviceProperties(&deviceProp, dev) );
  hypre_printf("Running on \"%s\", major %d, minor %d, total memory %.2f GB\n", deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem/1e9);
#endif

#if defined(HYPRE_USING_CUDA)
  hypre_int dev;
  struct cudaDeviceProp deviceProp;

  HYPRE_CUDA_CALL( cudaGetDevice(&dev) );
  HYPRE_CUDA_CALL( cudaGetDeviceProperties(&deviceProp, dev) );
  hypre_printf("Running on \"%s\", major %d, minor %d, total memory %.2f GB\n", deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem/1e9);
#endif

#if defined(HYPRE_USING_HIP)
  hypre_int dev;
  hipDeviceProp_t deviceProp;

  HYPRE_HIP_CALL( hipGetDevice(&dev) );
  HYPRE_HIP_CALL( hipGetDeviceProperties(&deviceProp, dev) );
  hypre_printf("Running on \"%s\", major %d, minor %d, total memory %.2f GB\n", deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem/1e9);
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre Umpire
 *
 *****************************************************************************/

#if defined(HYPRE_USING_UMPIRE)
HYPRE_Int
hypre_UmpireInit(hypre_Handle *hypre_handle_)
{
   umpire_resourcemanager_get_instance(&hypre_HandleUmpireResourceMan(hypre_handle_));

   hypre_HandleUmpireDevicePoolSize(hypre_handle_) = 4LL * 1024 * 1024 * 1024;
   hypre_HandleUmpireUMPoolSize(hypre_handle_)     = 4LL * 1024 * 1024 * 1024;
   hypre_HandleUmpireHostPoolSize(hypre_handle_)   = 4LL * 1024 * 1024 * 1024;
   hypre_HandleUmpirePinnedPoolSize(hypre_handle_) = 4LL * 1024 * 1024 * 1024;

   hypre_HandleUmpireBlockSize(hypre_handle_) = 512;

   strcpy(hypre_HandleUmpireDevicePoolName(hypre_handle_), "HYPRE_DEVICE_POOL");
   strcpy(hypre_HandleUmpireUMPoolName(hypre_handle_),     "HYPRE_UM_POOL");
   strcpy(hypre_HandleUmpireHostPoolName(hypre_handle_),   "HYPRE_HOST_POOL");
   strcpy(hypre_HandleUmpirePinnedPoolName(hypre_handle_), "HYPRE_PINNED_POOL");

   hypre_HandleOwnUmpireDevicePool(hypre_handle_) = 0;
   hypre_HandleOwnUmpireUMPool(hypre_handle_)     = 0;
   hypre_HandleOwnUmpireHostPool(hypre_handle_)   = 0;
   hypre_HandleOwnUmpirePinnedPool(hypre_handle_) = 0;

   return hypre_error_flag;
}

HYPRE_Int
hypre_UmpireFinalize(hypre_Handle *hypre_handle_)
{
   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(hypre_handle_);
   umpire_allocator allocator;

#if defined(HYPRE_USING_UMPIRE_HOST)
   if (hypre_HandleOwnUmpireHostPool(hypre_handle_))
   {
      const char *pool_name = hypre_HandleUmpireHostPoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_DEVICE)
   if (hypre_HandleOwnUmpireDevicePool(hypre_handle_))
   {
      const char *pool_name = hypre_HandleUmpireDevicePoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_UM)
   if (hypre_HandleOwnUmpireUMPool(hypre_handle_))
   {
      const char *pool_name = hypre_HandleUmpireUMPoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_PINNED)
   if (hypre_HandleOwnUmpirePinnedPool(hypre_handle_))
   {
      const char *pool_name = hypre_HandleUmpirePinnedPoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireDevicePoolSize(size_t nbytes)
{
   hypre_HandleUmpireDevicePoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireUMPoolSize(size_t nbytes)
{
   hypre_HandleUmpireUMPoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireHostPoolSize(size_t nbytes)
{
   hypre_HandleUmpireHostPoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpirePinnedPoolSize(size_t nbytes)
{
   hypre_HandleUmpirePinnedPoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireDevicePoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpireDevicePoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireUMPoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpireUMPoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireHostPoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpireHostPoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpirePinnedPoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpirePinnedPoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_UMPIRE) */

/******************************************************************************
 *
 * HYPRE memory location
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_SetMemoryLocation(HYPRE_MemoryLocation memory_location)
{
   hypre_HandleMemoryLocation(hypre_handle()) = memory_location;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_GetMemoryLocation(HYPRE_MemoryLocation *memory_location)
{
   *memory_location = hypre_HandleMemoryLocation(hypre_handle());

   return hypre_error_flag;
}

/******************************************************************************
 *
 * HYPRE execution policy
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_SetExecutionPolicy(HYPRE_ExecutionPolicy exec_policy)
{
   hypre_HandleDefaultExecPolicy(hypre_handle()) = exec_policy;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetStructExecutionPolicy(HYPRE_ExecutionPolicy exec_policy)
{
   hypre_HandleStructExecPolicy(hypre_handle()) = exec_policy;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_GetExecutionPolicy(HYPRE_ExecutionPolicy *exec_policy)
{
   *exec_policy = hypre_HandleDefaultExecPolicy(hypre_handle());

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_GetStructExecutionPolicy(HYPRE_ExecutionPolicy *exec_policy)
{
   *exec_policy = hypre_HandleStructExecPolicy(hypre_handle());

   return hypre_error_flag;
}

