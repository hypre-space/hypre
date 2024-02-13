/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

/* global variable _hypre_handle:
 * Outside this file, do NOT access it directly,
 * but use hypre_handle() instead (see handle.h) */
hypre_Handle *_hypre_handle = NULL;

/* accessor to the global ``_hypre_handle'' */
hypre_Handle*
hypre_handle(void)
{
   if (!_hypre_handle)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "ERROR - _hypre_handle is not initialized. Calling HYPRE_Initialize(). All HYPRE_* or hypre_* function calls should occur between HYPRE_Initialize() and HYPRE_Finalize().\n");
      HYPRE_Initialize();
   }

   return _hypre_handle;
}

hypre_Handle*
hypre_HandleCreate(void)
{
   hypre_Handle *hypre_handle_ = hypre_CTAlloc(hypre_Handle, 1, HYPRE_MEMORY_HOST);

   hypre_HandleMemoryLocation(hypre_handle_) = HYPRE_MEMORY_DEVICE;

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_HandleDefaultExecPolicy(hypre_handle_) = HYPRE_EXEC_DEVICE;
#endif

#if defined(HYPRE_USING_GPU)
   hypre_HandleDeviceData(hypre_handle_) = hypre_DeviceDataCreate();
   /* Gauss-Seidel: SpTrSV */
   hypre_HandleDeviceGSMethod(hypre_handle_) = 1; /* CPU: 0; Cusparse: 1 */
#endif

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_WITH_GPU_AWARE_MPI)
   hypre_HandleUseGpuAwareMPI(hypre_handle_) = 1;
#else
   hypre_HandleUseGpuAwareMPI(hypre_handle_) = 0;
#endif
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

   hypre_TFree(hypre_HandleStructCommRecvBuffer(hypre_handle_), HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_HandleStructCommSendBuffer(hypre_handle_), HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_GPU)
   hypre_DeviceDataDestroy(hypre_HandleDeviceData(hypre_handle_));
   hypre_HandleDeviceData(hypre_handle_) = NULL;
#endif

   hypre_TFree(hypre_handle_, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetDevice(hypre_int device_id, hypre_Handle *hypre_handle_)
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
   omp_set_default_device(device_id);

#elif defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaSetDevice(device_id) );
   hypre_HandleDevice(hypre_handle_) = device_id;

#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipSetDevice(device_id) );
   hypre_HandleDevice(hypre_handle_) = device_id;

#elif defined(HYPRE_USING_SYCL)
   if (hypre_handle_)
   {
      if (!hypre_HandleDevice(hypre_handle_))
      {
         /* Note: this enforces "explicit scaling," i.e. we treat each tile of a multi-tile GPU as a separate device */
         sycl::platform platform(sycl::gpu_selector{});
         auto gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
         hypre_int n_devices = 0;
         hypre_GetDeviceCount(&n_devices);
         if (device_id >= n_devices)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "ERROR: SYCL device-ID exceed the number of devices on-node\n");
         }

         hypre_int local_n_devices = 0;
         hypre_int i;
         for (i = 0; i < gpu_devices.size(); i++)
         {
            if (local_n_devices == device_id)
            {
               hypre_HandleDevice(hypre_handle_) = new sycl::device(gpu_devices[i]);
            }
            local_n_devices++;
         }
      }
      hypre_DeviceDataDeviceMaxWorkGroupSize(hypre_HandleDeviceData(hypre_handle_)) =
         hypre_DeviceDataDevice(hypre_HandleDeviceData(
                                   hypre_handle_))->get_info<sycl::info::device::max_work_group_size>();
   }
#else
   HYPRE_UNUSED_VAR(device_id);
   HYPRE_UNUSED_VAR(hypre_handle_);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_GetDeviceMaxShmemSize(hypre_int  device_id,
                            hypre_int *max_size_ptr,
                            hypre_int *max_size_optin_ptr)
{
   hypre_int max_size = 0, max_size_optin = 0;

#if defined(HYPRE_USING_GPU)
   hypre_Handle *handle = hypre_handle();

   if (!hypre_HandleDeviceMaxShmemPerBlockInited(handle))
   {
      if (device_id == -1)
      {
         hypre_GetDevice(&device_id);
      }

#if defined(HYPRE_USING_CUDA)
      cudaDeviceGetAttribute(&max_size, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
      cudaDeviceGetAttribute(&max_size_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);

#elif defined(HYPRE_USING_HIP)
      hipDeviceGetAttribute(&max_size, hipDeviceAttributeMaxSharedMemoryPerBlock, device_id);

#elif defined(HYPRE_USING_SYCL)
      auto device = *hypre_HandleDevice(hypre_handle());
      max_size = device.get_info<sycl::info::device::local_mem_size>();
#endif
      hypre_HandleDeviceMaxShmemPerBlock(handle)[0] = max_size;
      hypre_HandleDeviceMaxShmemPerBlock(handle)[1] = max_size_optin;

      hypre_HandleDeviceMaxShmemPerBlockInited(handle) = 1;
   }

   if (max_size_ptr)
   {
      *max_size_ptr = hypre_HandleDeviceMaxShmemPerBlock(handle)[0];
   }

   if (max_size_optin_ptr)
   {
      *max_size_optin_ptr = hypre_HandleDeviceMaxShmemPerBlock(handle)[1];
   }
#else /* not HYPRE_USING_GPU */
   HYPRE_UNUSED_VAR(device_id);

   if (max_size_ptr)
   {
      *max_size_ptr = max_size;
   }

   if (max_size_optin_ptr)
   {
      *max_size_optin_ptr = max_size_optin;
   }
#endif

   return hypre_error_flag;
}

/* Note: it doesn't return device_id in hypre_Handle->hypre_DeviceData,
 *       calls API instead. But these two should match at all times
 */
HYPRE_Int
hypre_GetDevice(hypre_int *device_id)
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
   *device_id = omp_get_default_device();

#elif defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetDevice(device_id) );

#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipGetDevice(device_id) );

#elif defined(HYPRE_USING_SYCL)
   /* WM: note - no sycl call to get which device is setup for use (if the user has already setup a device at all)
    * Assume the rank/device binding below */
   HYPRE_Int my_id;
   hypre_int n_devices;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_id);
   hypre_GetDeviceCount(&n_devices);
   (*device_id) = my_id % n_devices;

#else
   *device_id = 0;
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_GetDeviceCount(hypre_int *device_count)
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
   *device_count = omp_get_num_devices();

#elif defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetDeviceCount(device_count) );

#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipGetDeviceCount(device_count) );

#elif defined(HYPRE_USING_SYCL)
   (*device_count) = 0;
   sycl::platform platform(sycl::gpu_selector{});
   auto const& gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
   HYPRE_Int i;
   for (i = 0; i < gpu_devices.size(); i++)
   {
      (*device_count)++;
   }

#else
   *device_count = 0;
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_GetDeviceLastError(void)
{
#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetLastError() );

#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipGetLastError() );

#elif defined(HYPRE_USING_SYCL)
   try
   {
      hypre_HandleComputeStream(hypre_handle())->wait_and_throw();
   }
   catch (sycl::exception const& e)
   {
      std::cout << "Caught synchronous SYCL exception:\n"
                << e.what() << std::endl;
   }
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre initialization
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_DeviceInitialize(void)
{
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_Handle *handle = hypre_handle();

#if !defined(HYPRE_USING_SYCL)
   /* With sycl, cannot call hypre_GetDeviceLastError() until after device and queue setup */
   hypre_GetDeviceLastError();
#endif

   /* Notice: the cudaStream created is specific to the device
    * that was in effect when you created the stream.
    * So, we should first set the device and create the streams
    */
   hypre_int device_id;
   hypre_GetDevice(&device_id);
   hypre_SetDevice(device_id, handle);

   hypre_GetDeviceMaxShmemSize(device_id, NULL, NULL);

#if defined(HYPRE_USING_DEVICE_MALLOC_ASYNC)
   cudaMemPool_t mempool;
   cudaDeviceGetDefaultMemPool(&mempool, device_id);
   uint64_t threshold = UINT64_MAX;
   cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
#endif

   /* To include the cost of creating streams/cudahandles in HYPRE_Init */
   /* If not here, will be done at the first use */
#if defined(HYPRE_USING_CUDA_STREAMS)
   hypre_HandleComputeStream(handle);
#endif

   /* A separate stream for prefetching */
   //hypre_HandleCudaPrefetchStream(handle);

#if defined(HYPRE_USING_CUBLAS)
   hypre_HandleCublasHandle(handle);
#endif

#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)
   hypre_HandleCusparseHandle(handle);
#endif

#if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND)
   hypre_HandleCurandGenerator(handle);
#endif

#if defined(HYPRE_USING_CUSOLVER) || defined(HYPRE_USING_ROCSOLVER)
   hypre_HandleVendorSolverHandle(handle);
#endif

   /* Check if cuda arch flags in compiling match the device */
#if defined(HYPRE_USING_CUDA) && defined(HYPRE_DEBUG)
   hypre_CudaCompileFlagCheck();
#endif

#if defined(HYPRE_USING_DEVICE_POOL)
   /* Keep this check here at the end of HYPRE_Initialize()
    * Make sure that device pool allocator has not been setup in HYPRE_Initialize,
    * otherwise users are not able to set all the parameters
    */
   if ( hypre_HandleCubDevAllocator(handle) ||
        hypre_HandleCubUvmAllocator(handle) )
   {
      char msg[256];
      hypre_sprintf(msg, "%s %s", "ERROR: device pool allocators have been created in", __func__);
      hypre_error_w_msg(-1, msg);
   }
#endif

#endif /* if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP) */

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_Initialize(void)
{
   /* Return if the hypre library is in initialized state */
   if (hypre_Initialized())
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_MEMORY_TRACKER)
   if (!_hypre_memory_tracker)
   {
      _hypre_memory_tracker = hypre_MemoryTrackerCreate();
   }
#endif

   if (!_hypre_handle)
   {
      _hypre_handle = hypre_HandleCreate();
   }

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_OMPOffloadOn();
#endif

#if defined(HYPRE_USING_UMPIRE)
   hypre_UmpireInit(_hypre_handle);
#endif

#if defined(HYPRE_USING_MAGMA)
   hypre_MagmaInitialize();
#endif

   /* Update library state */
   hypre_SetInitialized();

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre finalization
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_Finalize(void)
{
   /* Return if the hypre library has already been finalized */
   if (hypre_Finalized())
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_UMPIRE)
   hypre_UmpireFinalize(_hypre_handle);
#endif

#if defined(HYPRE_USING_MAGMA)
   hypre_MagmaFinalize();
#endif

#if defined(HYPRE_USING_SYCL)
   /* With sycl, cannot call hypre_GetDeviceLastError() after destroying the handle, so do it here */
   hypre_GetDeviceLastError();
#endif

   hypre_HandleDestroy(_hypre_handle);
   _hypre_handle = NULL;

#if !defined(HYPRE_USING_SYCL)
   hypre_GetDeviceLastError();
#endif

#if defined(HYPRE_USING_MEMORY_TRACKER)
   hypre_PrintMemoryTracker(hypre_total_bytes, hypre_peak_bytes, hypre_current_bytes,
                            hypre_memory_tracker_print, hypre_memory_tracker_filename);

   hypre_MemoryTrackerDestroy(_hypre_memory_tracker);
   _hypre_memory_tracker = NULL;
#endif

   /* Update library state */
   hypre_SetFinalized();

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_PrintDeviceInfo(void)
{
#if defined(HYPRE_USING_GPU)
   hypre_int dev = 0;
#endif

#if defined(HYPRE_USING_CUDA)
   struct cudaDeviceProp deviceProp;

   HYPRE_CUDA_CALL( cudaGetDevice(&dev) );
   HYPRE_CUDA_CALL( cudaGetDeviceProperties(&deviceProp, dev) );
   hypre_printf("Running on \"%s\", major %d, minor %d, total memory %.2f GB\n", deviceProp.name,
                deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem / 1e9);

#elif defined(HYPRE_USING_HIP)
   hipDeviceProp_t deviceProp;

   HYPRE_HIP_CALL( hipGetDevice(&dev) );
   HYPRE_HIP_CALL( hipGetDeviceProperties(&deviceProp, dev) );
   hypre_printf("Running on \"%s\", major %d, minor %d, total memory %.2f GB\n", deviceProp.name,
                deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem / 1e9);

#elif defined(HYPRE_USING_SYCL)
   auto device = *hypre_HandleDevice(hypre_handle());
   auto p_name = device.get_platform().get_info<sycl::info::platform::name>();
   hypre_printf("Platform Name: %s\n", p_name.c_str());
   auto p_version = device.get_platform().get_info<sycl::info::platform::version>();
   hypre_printf("Platform Version: %s\n", p_version.c_str());
   auto d_name = device.get_info<sycl::info::device::name>();
   hypre_printf("Device Name: %s\n", d_name.c_str());
   auto max_work_group = device.get_info<sycl::info::device::max_work_group_size>();
   hypre_printf("Max Work Groups: %d\n", max_work_group);
   auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
   hypre_printf("Max Compute Units: %d\n", max_compute_units);
#endif

#if defined(HYPRE_USING_GPU)
   hypre_int max_size = 0, max_size_optin = 0;
   hypre_GetDeviceMaxShmemSize(dev, &max_size, &max_size_optin);
   hypre_printf("MaxSharedMemoryPerBlock %d, MaxSharedMemoryPerBlockOptin %d\n",
                max_size, max_size_optin);
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
HYPRE_GetExecutionPolicy(HYPRE_ExecutionPolicy *exec_policy)
{
   *exec_policy = hypre_HandleDefaultExecPolicy(hypre_handle());

   return hypre_error_flag;
}

const char*
HYPRE_GetExecutionPolicyName(HYPRE_ExecutionPolicy exec_policy)
{
   switch (exec_policy)
   {
      case HYPRE_EXEC_HOST:
         return "Host";

      case HYPRE_EXEC_DEVICE:
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_USING_CUDA)
         return "Device (CUDA)";
#elif defined(HYPRE_USING_HIP)
         return "Device (HIP)";
#elif defined(HYPRE_USING_SYCL)
         return "Device (SYCL)";
#else
         return "Device (OpenMP)";
#endif
#else
         return "Host";
#endif
      case HYPRE_EXEC_UNDEFINED:
      default:
         return "Undefined";
   }
}
