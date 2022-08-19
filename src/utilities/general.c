/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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
   hypre_HandleDeviceData(hypre_handle_) = hypre_DeviceDataCreate();
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
#if defined(HYPRE_USING_SYCL)
      if (!hypre_HandleDevice(hypre_handle_))
      {
         /* Note: this enforces "explicit scaling," i.e. we treat each tile of a multi-tile GPU as a separate device */
         sycl::platform platform(sycl::gpu_selector{});
         auto gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
         HYPRE_Int n_devices = 0;
         hypre_GetDeviceCount(&n_devices);
         if (device_id >= n_devices)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "ERROR: SYCL device-ID exceed the number of devices on-node\n");
         }

         HYPRE_Int local_n_devices = 0;
         HYPRE_Int i;
         for (i = 0; i < gpu_devices.size(); i++)
         {
            /* WM: commenting out multi-tile GPU stuff for now as it is not yet working */
            // multi-tile GPUs
            /* if (gpu_devices[i].get_info<sycl::info::device::partition_max_sub_devices>() > 0) */
            /* { */
            /*    auto subDevicesDomainNuma = */
            /*       gpu_devices[i].create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain> */
            /*       (sycl::info::partition_affinity_domain::numa); */
            /*    for (auto &tile : subDevicesDomainNuma) */
            /*    { */
            /*       if (local_n_devices == device_id) */
            /*       { */
            /*          hypre_HandleDevice(hypre_handle_) = new sycl::device(tile); */
            /*       } */
            /*       local_n_devices++; */
            /*    } */
            /* } */
            /* // single-tile GPUs */
            /* else */
            {
               if (local_n_devices == device_id)
               {
                  hypre_HandleDevice(hypre_handle_) = new sycl::device(gpu_devices[i]);
               }
               local_n_devices++;
            }
         }
      }
      hypre_DeviceDataDeviceMaxWorkGroupSize(hypre_HandleDeviceData(hypre_handle_)) =
         hypre_DeviceDataDevice(hypre_HandleDeviceData(
                                   hypre_handle_))->get_info<sycl::info::device::max_work_group_size>();
#else
      hypre_HandleDevice(hypre_handle_) = device_id;
#endif // #if defined(HYPRE_USING_SYCL)
   }
#endif // # if defined(HYPRE_USING_GPU)

   return hypre_error_flag;
}

HYPRE_Int
hypre_GetDeviceMaxShmemSize(hypre_int device_id, hypre_Handle *hypre_handle_)
{
#if defined(HYPRE_USING_GPU)
   hypre_int max_size = 0, max_size_optin = 0;
#endif

#if defined(HYPRE_USING_CUDA)
   cudaDeviceGetAttribute(&max_size, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
   cudaDeviceGetAttribute(&max_size_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
#endif

#if defined(HYPRE_USING_HIP)
   hipDeviceGetAttribute(&max_size, hipDeviceAttributeMaxSharedMemoryPerBlock, device_id);
#endif

#if defined(HYPRE_USING_GPU)
   hypre_HandleDeviceMaxShmemPerBlock(hypre_handle_)[0] = max_size;
   hypre_HandleDeviceMaxShmemPerBlock(hypre_handle_)[1] = max_size_optin;
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
#endif

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetDevice(device_id) );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipGetDevice(device_id) );
#endif

#if defined(HYPRE_USING_SYCL)
   /* WM: note - no sycl call to get which device is setup for use (if the user has already setup a device at all)
    * Assume the rank/device binding below */
   HYPRE_Int n_devices, my_id;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_id);
   hypre_GetDeviceCount(&n_devices);
   (*device_id) = my_id % n_devices;
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

#if defined(HYPRE_USING_SYCL)
   (*device_count) = 0;
   sycl::platform platform(sycl::gpu_selector{});
   auto const& gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
   HYPRE_Int i;
   for (i = 0; i < gpu_devices.size(); i++)
   {
      /* WM: commenting out multi-tile GPU stuff for now as it is not yet working */
      /* if (gpu_devices[i].get_info<sycl::info::device::partition_max_sub_devices>() > 0) */
      /* { */
      /*    auto subDevicesDomainNuma = */
      /*       gpu_devices[i].create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain> */
      /*       (sycl::info::partition_affinity_domain::numa); */
      /*    (*device_count) += subDevicesDomainNuma.size(); */
      /* } */
      /* else */
      {
         (*device_count)++;
      }
   }
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_GetDeviceLastError()
{
#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetLastError() );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipGetLastError() );
#endif

#if defined(HYPRE_USING_SYCL)
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
   hypre_SetDevice(device_id, _hypre_handle);
   hypre_GetDeviceMaxShmemSize(device_id, _hypre_handle);

#if defined(HYPRE_USING_DEVICE_MALLOC_ASYNC)
   cudaMemPool_t mempool;
   cudaDeviceGetDefaultMemPool(&mempool, device_id);
   uint64_t threshold = UINT64_MAX;
   cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
#endif

   /* To include the cost of creating streams/cudahandles in HYPRE_Init */
   /* If not here, will be done at the first use */
#if defined(HYPRE_USING_CUDA_STREAMS)
   hypre_HandleComputeStream(_hypre_handle);
#endif

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

#if defined(HYPRE_USING_DEVICE_POOL)
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

#if !defined(HYPRE_USING_SYCL)
   /* With sycl, cannot call hypre_GetDeviceLastError() after destroying the handle */
   hypre_GetDeviceLastError();
#endif

#ifdef HYPRE_USING_MEMORY_TRACKER
   hypre_PrintMemoryTracker();
   hypre_MemoryTrackerDestroy(_hypre_memory_tracker);
#endif

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_PrintDeviceInfo()
{
#if defined(HYPRE_USING_CUDA)
   hypre_int dev;
   struct cudaDeviceProp deviceProp;

   HYPRE_CUDA_CALL( cudaGetDevice(&dev) );
   HYPRE_CUDA_CALL( cudaGetDeviceProperties(&deviceProp, dev) );
   hypre_printf("Running on \"%s\", major %d, minor %d, total memory %.2f GB\n", deviceProp.name,
                deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem / 1e9);
#endif

#if defined(HYPRE_USING_HIP)
   hypre_int dev;
   hipDeviceProp_t deviceProp;

   HYPRE_HIP_CALL( hipGetDevice(&dev) );
   HYPRE_HIP_CALL( hipGetDeviceProperties(&deviceProp, dev) );
   hypre_printf("Running on \"%s\", major %d, minor %d, total memory %.2f GB\n", deviceProp.name,
                deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem / 1e9);
#endif

#if defined(HYPRE_USING_SYCL)
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
   hypre_printf("MaxSharedMemoryPerBlock %d, MaxSharedMemoryPerBlockOptin %d\n",
                hypre_HandleDeviceMaxShmemPerBlock(hypre_handle())[0],
                hypre_HandleDeviceMaxShmemPerBlock(hypre_handle())[1]);
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

