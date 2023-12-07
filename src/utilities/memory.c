/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Memory management utilities
 *
 *****************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USE_UMALLOC)
#undef HYPRE_USE_UMALLOC
#endif

/******************************************************************************
 *
 * Helper routines
 *
 *****************************************************************************/

HYPRE_Int
hypre_GetMemoryLocationName(hypre_MemoryLocation  memory_location,
                            char                 *memory_location_name)
{
   if (memory_location == hypre_MEMORY_HOST)
   {
      sprintf(memory_location_name, "%s", "HOST");
   }
   else if (memory_location == hypre_MEMORY_HOST_PINNED)
   {
      sprintf(memory_location_name, "%s", "HOST PINNED");
   }
   else if (memory_location == hypre_MEMORY_DEVICE)
   {
      sprintf(memory_location_name, "%s", "DEVICE");
   }
   else if (memory_location == hypre_MEMORY_UNIFIED)
   {
      sprintf(memory_location_name, "%s", "UNIFIED");
   }
   else
   {
      sprintf(memory_location_name, "%s", "");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_OutOfMemory
 *--------------------------------------------------------------------------*/

static inline void
hypre_OutOfMemory(size_t size)
{
   char msg[1024];

   hypre_sprintf(msg, "Out of memory trying to allocate %zu bytes\n", size);
   hypre_error_w_msg(HYPRE_ERROR_MEMORY, msg);
   hypre_assert(0);
   fflush(stdout);
}

static inline void
hypre_WrongMemoryLocation(void)
{
   hypre_error_w_msg(HYPRE_ERROR_MEMORY, "Unrecognized hypre_MemoryLocation\n");
   hypre_assert(0);
   fflush(stdout);
}

void
hypre_CheckMemoryLocation(void *ptr, hypre_MemoryLocation location)
{
#if defined(HYPRE_DEBUG) && defined(HYPRE_USING_GPU)
   if (!ptr)
   {
      return;
   }

   hypre_MemoryLocation location_ptr;
   hypre_GetPointerLocation(ptr, &location_ptr);
   /* do not use hypre_assert, which has alloc and free;
    * will create an endless loop otherwise */
   assert(location == location_ptr);
#else
   HYPRE_UNUSED_VAR(ptr);
   HYPRE_UNUSED_VAR(location);
#endif
}

/*==========================================================================
 * Physical memory location (hypre_MemoryLocation) interface
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * Memset
 *--------------------------------------------------------------------------*/
static inline void
hypre_HostMemset(void *ptr, HYPRE_Int value, size_t num)
{
   memset(ptr, value, num);
}

static inline void
hypre_DeviceMemset(void *ptr, HYPRE_Int value, size_t num)
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
   #pragma omp target teams distribute parallel for is_device_ptr(ptr)
   for (size_t i = 0; i < num; i++)
   {
      ((unsigned char *) ptr)[i] = (unsigned char) value;
   }
#else
   memset(ptr, value, num);
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, num, "update", "to");
#endif
   /* HYPRE_CUDA_CALL( cudaDeviceSynchronize() ); */

#elif defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );

#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipMemset(ptr, value, num) );

#elif defined(HYPRE_USING_SYCL)
   HYPRE_SYCL_CALL( (hypre_HandleComputeStream(hypre_handle()))->memset(ptr, value, num).wait() );

#else
   HYPRE_UNUSED_VAR(ptr);
   HYPRE_UNUSED_VAR(value);
   HYPRE_UNUSED_VAR(num);
#endif
}

static inline void
hypre_UnifiedMemset(void *ptr, HYPRE_Int value, size_t num)
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
   #pragma omp target teams distribute parallel for is_device_ptr(ptr)
   for (size_t i = 0; i < num; i++)
   {
      ((unsigned char *) ptr)[i] = (unsigned char) value;
   }
#else
   memset(ptr, value, num);
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, num, "update", "to");
#endif
   /* HYPRE_CUDA_CALL( cudaDeviceSynchronize() ); */

#elif defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );

#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipMemset(ptr, value, num) );

#elif defined(HYPRE_USING_SYCL)
   HYPRE_SYCL_CALL( (hypre_HandleComputeStream(hypre_handle()))->memset(ptr, value, num).wait() );

#else
   HYPRE_UNUSED_VAR(ptr);
   HYPRE_UNUSED_VAR(value);
   HYPRE_UNUSED_VAR(num);
#endif
}

/*--------------------------------------------------------------------------
 * Memprefetch
 *--------------------------------------------------------------------------*/
static inline void
hypre_UnifiedMemPrefetch(void *ptr, size_t size, hypre_MemoryLocation location)
{
   if (!size)
   {
      return;
   }

   hypre_CheckMemoryLocation(ptr, hypre_MEMORY_UNIFIED);

#if defined(HYPRE_USING_CUDA)
   if (location == hypre_MEMORY_DEVICE)
   {
      HYPRE_CUDA_CALL( cudaMemPrefetchAsync(ptr, size, hypre_HandleDevice(hypre_handle()),
                                            hypre_HandleComputeStream(hypre_handle())) );
   }
   else if (location == hypre_MEMORY_HOST)
   {
      HYPRE_CUDA_CALL( cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId,
                                            hypre_HandleComputeStream(hypre_handle())) );
   }

#elif defined(HYPRE_USING_HIP)
   HYPRE_UNUSED_VAR(ptr);
   HYPRE_UNUSED_VAR(size);
   HYPRE_UNUSED_VAR(location);
   // Not currently implemented for HIP, but leaving place holder
   /*
    *if (location == hypre_MEMORY_DEVICE)
    *{
    *  HYPRE_HIP_CALL( hipMemPrefetchAsync(ptr, size, hypre_HandleDevice(hypre_handle()),
    *                   hypre_HandleComputeStream(hypre_handle())) );
    *}
    *else if (location == hypre_MEMORY_HOST)
    *{
    *   HYPRE_CUDA_CALL( hipMemPrefetchAsync(ptr, size, cudaCpuDeviceId,
    *                    hypre_HandleComputeStream(hypre_handle())) );
    *}
    */

#elif defined(HYPRE_USING_SYCL)
   HYPRE_UNUSED_VAR(ptr);
   HYPRE_UNUSED_VAR(size);
   HYPRE_UNUSED_VAR(location);
   if (location == hypre_MEMORY_DEVICE)
   {
      /* WM: todo - the call below seems like it may occasionally result in an error: */
      /*     Native API returns: -997 (The plugin has emitted a backend specific error) */
      /*     or a seg fault. On the other hand, removing this line can also cause the code
       *     to hang (or run excessively slow?). */
      /* HYPRE_SYCL_CALL( hypre_HandleComputeStream(hypre_handle())->prefetch(ptr, size).wait() ); */
   }
#else
   HYPRE_UNUSED_VAR(ptr);
   HYPRE_UNUSED_VAR(size);
   HYPRE_UNUSED_VAR(location);
#endif
}

/*--------------------------------------------------------------------------
 * Malloc
 *--------------------------------------------------------------------------*/
static inline void *
hypre_HostMalloc(size_t size, HYPRE_Int zeroinit)
{
   void *ptr = NULL;

#if defined(HYPRE_USING_UMPIRE_HOST)
   hypre_umpire_host_pooled_allocate(&ptr, size);
   if (zeroinit)
   {
      memset(ptr, 0, size);
   }
#else
   if (zeroinit)
   {
      ptr = calloc(size, 1);
   }
   else
   {
      ptr = malloc(size);
   }
#endif

   return ptr;
}

static inline void *
hypre_DeviceMalloc(size_t size, HYPRE_Int zeroinit)
{
   void *ptr = NULL;

   if ( hypre_HandleUserDeviceMalloc(hypre_handle()) )
   {
      hypre_HandleUserDeviceMalloc(hypre_handle())(&ptr, size);
   }
   else
   {
#if defined(HYPRE_USING_UMPIRE_DEVICE)
      hypre_umpire_device_pooled_allocate(&ptr, size);
#else

#if defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
      ptr = omp_target_alloc(size, hypre__offload_device_num);
#else
      ptr = malloc(size + sizeof(size_t));
      size_t *sp = (size_t*) ptr;
      sp[0] = size;
      ptr = (void *) (&sp[1]);
      HYPRE_OMPOffload(hypre__offload_device_num, ptr, size, "enter", "alloc");
#endif
#endif

#if defined(HYPRE_USING_CUDA)
#if defined(HYPRE_USING_DEVICE_POOL)
      HYPRE_CUDA_CALL( hypre_CachingMallocDevice(&ptr, size) );
#elif defined(HYPRE_USING_DEVICE_MALLOC_ASYNC)
      HYPRE_CUDA_CALL( cudaMallocAsync(&ptr, size, NULL) );
#else
      HYPRE_CUDA_CALL( cudaMalloc(&ptr, size) );
#endif
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipMalloc(&ptr, size) );
#endif

#if defined(HYPRE_USING_SYCL)
      ptr = (void *)sycl::malloc_device(size, *(hypre_HandleComputeStream(hypre_handle())));
#endif

#endif /* #if defined(HYPRE_USING_UMPIRE_DEVICE) */
   }

   if (ptr && zeroinit)
   {
      hypre_DeviceMemset(ptr, 0, size);
   }

   return ptr;
}

static inline void *
hypre_UnifiedMalloc(size_t size, HYPRE_Int zeroinit)
{
   void *ptr = NULL;

#if defined(HYPRE_USING_UMPIRE_UM)
   hypre_umpire_um_pooled_allocate(&ptr, size);
#else

#if defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
   ptr = omp_target_alloc(size, hypre__offload_device_num);
#else
   ptr = malloc(size + sizeof(size_t));
   size_t *sp = (size_t*) ptr;
   sp[0] = size;
   ptr = (void *) (&sp[1]);
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, size, "enter", "alloc");
#endif
#endif

#if defined(HYPRE_USING_CUDA)
#if defined(HYPRE_USING_DEVICE_POOL)
   HYPRE_CUDA_CALL( hypre_CachingMallocManaged(&ptr, size) );
#else
   HYPRE_CUDA_CALL( cudaMallocManaged(&ptr, size, cudaMemAttachGlobal) );
#endif
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipMallocManaged(&ptr, size, hipMemAttachGlobal) );
#endif

#if defined(HYPRE_USING_SYCL)
   HYPRE_SYCL_CALL( ptr = (void *)sycl::malloc_shared(size,
                                                      *(hypre_HandleComputeStream(hypre_handle()))) );
#endif

#endif /* #if defined(HYPRE_USING_UMPIRE_UM) */

   /* prefecth to device */
   if (ptr)
   {
      hypre_UnifiedMemPrefetch(ptr, size, hypre_MEMORY_DEVICE);
   }

   if (ptr && zeroinit)
   {
      hypre_UnifiedMemset(ptr, 0, size);
   }

   return ptr;
}

static inline void *
hypre_HostPinnedMalloc(size_t size, HYPRE_Int zeroinit)
{
   void *ptr = NULL;

#if defined(HYPRE_USING_UMPIRE_PINNED)
   hypre_umpire_pinned_pooled_allocate(&ptr, size);
#else

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaMallocHost(&ptr, size) );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipHostMalloc(&ptr, size) );
#endif

#if defined(HYPRE_USING_SYCL)
   HYPRE_SYCL_CALL( ptr = (void *)sycl::malloc_host(size,
                                                    *(hypre_HandleComputeStream(hypre_handle()))) );
#endif

#endif /* #if defined(HYPRE_USING_UMPIRE_PINNED) */

   if (ptr && zeroinit)
   {
      hypre_HostMemset(ptr, 0, size);
   }

   return ptr;
}

static inline void *
hypre_MAlloc_core(size_t size, HYPRE_Int zeroinit, hypre_MemoryLocation location)
{
   if (size == 0)
   {
      return NULL;
   }

   void *ptr = NULL;

   switch (location)
   {
      case hypre_MEMORY_HOST :
         ptr = hypre_HostMalloc(size, zeroinit);
         break;
      case hypre_MEMORY_DEVICE :
         ptr = hypre_DeviceMalloc(size, zeroinit);
         break;
      case hypre_MEMORY_UNIFIED :
         ptr = hypre_UnifiedMalloc(size, zeroinit);
         break;
      case hypre_MEMORY_HOST_PINNED :
         ptr = hypre_HostPinnedMalloc(size, zeroinit);
         break;
      default :
         hypre_WrongMemoryLocation();
   }

   if (!ptr)
   {
      hypre_OutOfMemory(size);
      hypre_MPI_Abort(hypre_MPI_COMM_WORLD, -1);
   }

   return ptr;
}

void *
_hypre_MAlloc(size_t size, hypre_MemoryLocation location)
{
   return hypre_MAlloc_core(size, 0, location);
}

/*--------------------------------------------------------------------------
 * Free
 *--------------------------------------------------------------------------*/
static inline void
hypre_HostFree(void *ptr)
{
#if defined(HYPRE_USING_UMPIRE_HOST)
   hypre_umpire_host_pooled_free(ptr);
#else
   free(ptr);
#endif
}

static inline void
hypre_DeviceFree(void *ptr)
{
   if ( hypre_HandleUserDeviceMfree(hypre_handle()) )
   {
      hypre_HandleUserDeviceMfree(hypre_handle())(ptr);
   }
   else
   {
#if defined(HYPRE_USING_UMPIRE_DEVICE)
      hypre_umpire_device_pooled_free(ptr);
#else

#if defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_free(ptr, hypre__offload_device_num);
#else
      HYPRE_OMPOffload(hypre__offload_device_num, ptr, ((size_t *) ptr)[-1], "exit", "delete");
#endif
#endif

#if defined(HYPRE_USING_CUDA)
#if defined(HYPRE_USING_DEVICE_POOL)
      HYPRE_CUDA_CALL( hypre_CachingFreeDevice(ptr) );
#elif defined(HYPRE_USING_DEVICE_MALLOC_ASYNC)
      HYPRE_CUDA_CALL( cudaFreeAsync(ptr, NULL) );
#else
      HYPRE_CUDA_CALL( cudaFree(ptr) );
#endif
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipFree(ptr) );
#endif

#if defined(HYPRE_USING_SYCL)
      HYPRE_SYCL_CALL( sycl::free(ptr, *(hypre_HandleComputeStream(hypre_handle()))) );
#endif

#endif /* #if defined(HYPRE_USING_UMPIRE_DEVICE) */
   }
}

static inline void
hypre_UnifiedFree(void *ptr)
{
#if defined(HYPRE_USING_UMPIRE_UM)
   hypre_umpire_um_pooled_free(ptr);

#elif defined(HYPRE_USING_DEVICE_OPENMP) && defined(HYPRE_DEVICE_OPENMP_ALLOC)
   omp_target_free(ptr, hypre__offload_device_num);

#elif defined(HYPRE_USING_DEVICE_OPENMP) && !defined(HYPRE_DEVICE_OPENMP_ALLOC)
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, ((size_t *) ptr)[-1], "exit", "delete");

#elif defined(HYPRE_USING_CUDA) && defined(HYPRE_USING_DEVICE_POOL)
   HYPRE_CUDA_CALL( hypre_CachingFreeManaged(ptr) );

#elif defined(HYPRE_USING_CUDA) && !defined(HYPRE_USING_DEVICE_POOL)
   HYPRE_CUDA_CALL( cudaFree(ptr) );

#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipFree(ptr) );

#elif defined(HYPRE_USING_SYCL)
   HYPRE_SYCL_CALL( sycl::free(ptr, *(hypre_HandleComputeStream(hypre_handle()))) );

#else
   HYPRE_UNUSED_VAR(ptr);

#endif /* #if defined(HYPRE_USING_UMPIRE_UM) */
}

static inline void
hypre_HostPinnedFree(void *ptr)
{
#if defined(HYPRE_USING_UMPIRE_PINNED)
   hypre_umpire_pinned_pooled_free(ptr);

#elif defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaFreeHost(ptr) );

#elif defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipHostFree(ptr) );

#elif defined(HYPRE_USING_SYCL)
   HYPRE_SYCL_CALL( sycl::free(ptr, *(hypre_HandleComputeStream(hypre_handle()))) );

#else
   HYPRE_UNUSED_VAR(ptr);

#endif /* #if defined(HYPRE_USING_UMPIRE_PINNED) */
}

static inline void
hypre_Free_core(void *ptr, hypre_MemoryLocation location)
{
   if (!ptr)
   {
      return;
   }

   hypre_CheckMemoryLocation(ptr, location);

   switch (location)
   {
      case hypre_MEMORY_HOST :
         hypre_HostFree(ptr);
         break;
      case hypre_MEMORY_DEVICE :
         hypre_DeviceFree(ptr);
         break;
      case hypre_MEMORY_UNIFIED :
         hypre_UnifiedFree(ptr);
         break;
      case hypre_MEMORY_HOST_PINNED :
         hypre_HostPinnedFree(ptr);
         break;
      default :
         hypre_WrongMemoryLocation();
   }
}

void
_hypre_Free(void *ptr, hypre_MemoryLocation location)
{
   hypre_Free_core(ptr, location);
}


/*--------------------------------------------------------------------------
 * Memcpy
 *--------------------------------------------------------------------------*/
static inline void
hypre_Memcpy_core(void *dst, void *src, size_t size, hypre_MemoryLocation loc_dst,
                  hypre_MemoryLocation loc_src)
{
#if defined(HYPRE_USING_SYCL)
   sycl::queue* q = hypre_HandleComputeStream(hypre_handle());
#endif

   if (dst == NULL || src == NULL)
   {
      if (size)
      {
         hypre_printf("hypre_Memcpy warning: copy %ld bytes from %p to %p !\n", size, src, dst);
         hypre_assert(0);
      }

      return;
   }

   if (dst == src)
   {
      return;
   }

   if (size > 0)
   {
      hypre_CheckMemoryLocation(dst, loc_dst);
      hypre_CheckMemoryLocation(src, loc_src);
   }

   /* Totally 4 x 4 = 16 cases */

   /* 4: Host   <-- Host, Host   <-- Pinned,
    *    Pinned <-- Host, Pinned <-- Pinned.
    */
   if ( loc_dst != hypre_MEMORY_DEVICE && loc_dst != hypre_MEMORY_UNIFIED &&
        loc_src != hypre_MEMORY_DEVICE && loc_src != hypre_MEMORY_UNIFIED )
   {
      memcpy(dst, src, size);
      return;
   }


   /* 3: UVM <-- Device, Device <-- UVM, UVM <-- UVM */
   if ( (loc_dst == hypre_MEMORY_UNIFIED && loc_src == hypre_MEMORY_DEVICE)  ||
        (loc_dst == hypre_MEMORY_DEVICE  && loc_src == hypre_MEMORY_UNIFIED) ||
        (loc_dst == hypre_MEMORY_UNIFIED && loc_src == hypre_MEMORY_UNIFIED) )
   {
#if defined(HYPRE_USING_DEVICE_OPENMP)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_device_num, hypre__offload_device_num);
#endif

#if defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) );
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice) );
#endif

#if defined(HYPRE_USING_SYCL)
      HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }


   /* 2: UVM <-- Host, UVM <-- Pinned */
   if (loc_dst == hypre_MEMORY_UNIFIED)
   {
#if defined(HYPRE_USING_DEVICE_OPENMP)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_device_num, hypre__offload_host_num);
#endif

#if defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyHostToDevice) );
#endif

#if defined(HYPRE_USING_SYCL)
      HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }


   /* 2: Host <-- UVM, Pinned <-- UVM */
   if (loc_src == hypre_MEMORY_UNIFIED)
   {
#if defined(HYPRE_USING_DEVICE_OPENMP)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_host_num, hypre__offload_device_num);
#endif

#if defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) );
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToHost) );
#endif

#if defined(HYPRE_USING_SYCL)
      HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }


   /* 2: Device <-- Host, Device <-- Pinned */
   if ( loc_dst == hypre_MEMORY_DEVICE && (loc_src == hypre_MEMORY_HOST ||
                                           loc_src == hypre_MEMORY_HOST_PINNED) )
   {
#if defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_device_num, hypre__offload_host_num);
#else
      memcpy(dst, src, size);
      HYPRE_OMPOffload(hypre__offload_device_num, dst, size, "update", "to");
#endif
#endif

#if defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyHostToDevice) );
#endif

#if defined(HYPRE_USING_SYCL)
      HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }


   /* 2: Host <-- Device, Pinned <-- Device */
   if ( (loc_dst == hypre_MEMORY_HOST || loc_dst == hypre_MEMORY_HOST_PINNED) &&
        loc_src == hypre_MEMORY_DEVICE )
   {
#if defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_host_num, hypre__offload_device_num);
#else
      HYPRE_OMPOffload(hypre__offload_device_num, src, size, "update", "from");
      memcpy(dst, src, size);
#endif
#endif

#if defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost) );
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToHost) );
#endif

#if defined(HYPRE_USING_SYCL)
      HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }


   /* 1: Device <-- Device */
   if (loc_dst == hypre_MEMORY_DEVICE && loc_src == hypre_MEMORY_DEVICE)
   {
#if defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_device_num, hypre__offload_device_num);
#else
      HYPRE_OMPOffload(hypre__offload_device_num, src, size, "update", "from");
      memcpy(dst, src, size);
      HYPRE_OMPOffload(hypre__offload_device_num, dst, size, "update", "to");
#endif
#endif

#if defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) );
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice) );
#endif

#if defined(HYPRE_USING_SYCL)
      HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }

   hypre_WrongMemoryLocation();
}

/*--------------------------------------------------------------------------*
 * ExecPolicy
 *--------------------------------------------------------------------------*/
static inline HYPRE_ExecutionPolicy
hypre_GetExecPolicy1_core(hypre_MemoryLocation location)
{
   HYPRE_ExecutionPolicy exec = HYPRE_EXEC_UNDEFINED;

   switch (location)
   {
      case hypre_MEMORY_HOST :
      case hypre_MEMORY_HOST_PINNED :
         exec = HYPRE_EXEC_HOST;
         break;
      case hypre_MEMORY_DEVICE :
         exec = HYPRE_EXEC_DEVICE;
         break;
      case hypre_MEMORY_UNIFIED :
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
         exec = hypre_HandleDefaultExecPolicy(hypre_handle());
#endif
         break;
      default :
         hypre_WrongMemoryLocation();
   }

   hypre_assert(exec != HYPRE_EXEC_UNDEFINED);

   return exec;
}

/* for binary operation */
static inline HYPRE_ExecutionPolicy
hypre_GetExecPolicy2_core(hypre_MemoryLocation location1,
                          hypre_MemoryLocation location2)
{
   HYPRE_ExecutionPolicy exec = HYPRE_EXEC_UNDEFINED;

   /* HOST_PINNED has the same exec policy as HOST */
   if (location1 == hypre_MEMORY_HOST_PINNED)
   {
      location1 = hypre_MEMORY_HOST;
   }

   if (location2 == hypre_MEMORY_HOST_PINNED)
   {
      location2 = hypre_MEMORY_HOST;
   }

   /* no policy for these combinations */
   if ( (location1 == hypre_MEMORY_HOST && location2 == hypre_MEMORY_DEVICE) ||
        (location2 == hypre_MEMORY_HOST && location1 == hypre_MEMORY_DEVICE) )
   {
      exec = HYPRE_EXEC_UNDEFINED;
   }

   /* this should never happen */
   if ( (location1 == hypre_MEMORY_UNIFIED && location2 == hypre_MEMORY_DEVICE) ||
        (location2 == hypre_MEMORY_UNIFIED && location1 == hypre_MEMORY_DEVICE) )
   {
      exec = HYPRE_EXEC_UNDEFINED;
   }

   if (location1 == hypre_MEMORY_UNIFIED && location2 == hypre_MEMORY_UNIFIED)
   {
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
      exec = hypre_HandleDefaultExecPolicy(hypre_handle());
#endif
   }

   if (location1 == hypre_MEMORY_HOST || location2 == hypre_MEMORY_HOST)
   {
      exec = HYPRE_EXEC_HOST;
   }

   if (location1 == hypre_MEMORY_DEVICE || location2 == hypre_MEMORY_DEVICE)
   {
      exec = HYPRE_EXEC_DEVICE;
   }

   hypre_assert(exec != HYPRE_EXEC_UNDEFINED);

   return exec;
}

/*==========================================================================
 * Conceptual memory location (HYPRE_MemoryLocation) interface
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_Memset
 * "Sets the first num bytes of the block of memory pointed by ptr to the specified value
 * (*** value is interpreted as an unsigned char ***)"
 * http://www.cplusplus.com/reference/cstring/memset/
 *--------------------------------------------------------------------------*/
void *
hypre_Memset(void *ptr, HYPRE_Int value, size_t num, HYPRE_MemoryLocation location)
{
   if (num == 0)
   {
      return ptr;
   }

   if (ptr == NULL)
   {
      if (num)
      {
         hypre_printf("hypre_Memset warning: set values for %ld bytes at %p !\n", num, ptr);
      }
      return ptr;
   }

   hypre_CheckMemoryLocation(ptr, hypre_GetActualMemLocation(location));

   switch (hypre_GetActualMemLocation(location))
   {
      case hypre_MEMORY_HOST :
      case hypre_MEMORY_HOST_PINNED :
         hypre_HostMemset(ptr, value, num);
         break;
      case hypre_MEMORY_DEVICE :
         hypre_DeviceMemset(ptr, value, num);
         break;
      case hypre_MEMORY_UNIFIED :
         hypre_UnifiedMemset(ptr, value, num);
         break;
      default :
         hypre_WrongMemoryLocation();
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * Memprefetch
 *--------------------------------------------------------------------------*/
void
hypre_MemPrefetch(void *ptr, size_t size, HYPRE_MemoryLocation location)
{
   hypre_UnifiedMemPrefetch( ptr, size, hypre_GetActualMemLocation(location) );
}

/*--------------------------------------------------------------------------*
 * hypre_MAlloc, hypre_CAlloc
 *--------------------------------------------------------------------------*/

void *
hypre_MAlloc(size_t size, HYPRE_MemoryLocation location)
{
   return hypre_MAlloc_core(size, 0, hypre_GetActualMemLocation(location));
}

void *
hypre_CAlloc( size_t count, size_t elt_size, HYPRE_MemoryLocation location)
{
   return hypre_MAlloc_core(count * elt_size, 1, hypre_GetActualMemLocation(location));
}

/*--------------------------------------------------------------------------
 * hypre_Free
 *--------------------------------------------------------------------------*/

void
hypre_Free(void *ptr, HYPRE_MemoryLocation location)
{
   hypre_Free_core(ptr, hypre_GetActualMemLocation(location));
}

/*--------------------------------------------------------------------------
 * hypre_Memcpy
 *--------------------------------------------------------------------------*/

void
hypre_Memcpy(void *dst, void *src, size_t size, HYPRE_MemoryLocation loc_dst,
             HYPRE_MemoryLocation loc_src)
{
   hypre_Memcpy_core( dst, src, size, hypre_GetActualMemLocation(loc_dst),
                      hypre_GetActualMemLocation(loc_src) );
}

/*--------------------------------------------------------------------------
 * hypre_ReAlloc
 *--------------------------------------------------------------------------*/
void *
hypre_ReAlloc(void *ptr, size_t size, HYPRE_MemoryLocation location)
{
   if (size == 0)
   {
      hypre_Free(ptr, location);
      return NULL;
   }

   if (ptr == NULL)
   {
      return hypre_MAlloc(size, location);
   }

   if (hypre_GetActualMemLocation(location) != hypre_MEMORY_HOST)
   {
      hypre_printf("hypre_TReAlloc only works with HYPRE_MEMORY_HOST; Use hypre_TReAlloc_v2 instead!\n");
      hypre_assert(0);
      hypre_MPI_Abort(hypre_MPI_COMM_WORLD, -1);
      return NULL;
   }

#if defined(HYPRE_USING_UMPIRE_HOST)
   ptr = hypre_umpire_host_pooled_realloc(ptr, size);
#else
   ptr = realloc(ptr, size);
#endif

   if (!ptr)
   {
      hypre_OutOfMemory(size);
   }

   return ptr;
}

void *
hypre_ReAlloc_v2(void *ptr, size_t old_size, size_t new_size, HYPRE_MemoryLocation location)
{
   if (new_size == 0)
   {
      hypre_Free(ptr, location);
      return NULL;
   }

   if (ptr == NULL)
   {
      return hypre_MAlloc(new_size, location);
   }

   if (old_size == new_size)
   {
      return ptr;
   }

   void *new_ptr = hypre_MAlloc(new_size, location);
   size_t smaller_size = new_size > old_size ? old_size : new_size;
   hypre_Memcpy(new_ptr, ptr, smaller_size, location, location);
   hypre_Free(ptr, location);
   ptr = new_ptr;

   if (!ptr)
   {
      hypre_OutOfMemory(new_size);
   }

   return ptr;
}

/*--------------------------------------------------------------------------*
 * hypre_GetExecPolicy: return execution policy based on memory locations
 *--------------------------------------------------------------------------*/
/* for unary operation */
HYPRE_ExecutionPolicy
hypre_GetExecPolicy1(HYPRE_MemoryLocation location)
{

   return hypre_GetExecPolicy1_core(hypre_GetActualMemLocation(location));
}

/* for binary operation */
HYPRE_ExecutionPolicy
hypre_GetExecPolicy2(HYPRE_MemoryLocation location1,
                     HYPRE_MemoryLocation location2)
{
   return hypre_GetExecPolicy2_core(hypre_GetActualMemLocation(location1),
                                    hypre_GetActualMemLocation(location2));
}

/*--------------------------------------------------------------------------
 * Query the actual memory location pointed by ptr
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_GetPointerLocation(const void *ptr, hypre_MemoryLocation *memory_location)
{
   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   *memory_location = hypre_MEMORY_UNDEFINED;

#if defined(HYPRE_USING_CUDA)
   struct cudaPointerAttributes attr;

#if (CUDART_VERSION >= 10000)
#if (CUDART_VERSION >= 11000)
   HYPRE_CUDA_CALL( cudaPointerGetAttributes(&attr, ptr) );
#else
   cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
   if (err != cudaSuccess)
   {
      ierr = 1;
      /* clear the error */
      cudaGetLastError();
   }
#endif
   if (attr.type == cudaMemoryTypeUnregistered)
   {
      *memory_location = hypre_MEMORY_HOST;
   }
   else if (attr.type == cudaMemoryTypeHost)
   {
      *memory_location = hypre_MEMORY_HOST_PINNED;
   }
   else if (attr.type == cudaMemoryTypeDevice)
   {
      *memory_location = hypre_MEMORY_DEVICE;
   }
   else if (attr.type == cudaMemoryTypeManaged)
   {
      *memory_location = hypre_MEMORY_UNIFIED;
   }
#else
   cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
   if (err != cudaSuccess)
   {
      ierr = 1;

      /* clear the error */
      cudaGetLastError();

      if (err == cudaErrorInvalidValue)
      {
         *memory_location = hypre_MEMORY_HOST;
      }
   }
   else if (attr.isManaged)
   {
      *memory_location = hypre_MEMORY_UNIFIED;
   }
   else if (attr.memoryType == cudaMemoryTypeDevice)
   {
      *memory_location = hypre_MEMORY_DEVICE;
   }
   else if (attr.memoryType == cudaMemoryTypeHost)
   {
      *memory_location = hypre_MEMORY_HOST_PINNED;
   }
#endif // CUDART_VERSION >= 10000
#endif // defined(HYPRE_USING_CUDA)

#if defined(HYPRE_USING_HIP)

   struct hipPointerAttribute_t attr;
   *memory_location = hypre_MEMORY_UNDEFINED;

   hipError_t err = hipPointerGetAttributes(&attr, ptr);
   if (err != hipSuccess)
   {
      ierr = 1;

      /* clear the error */
      hipGetLastError();

      if (err == hipErrorInvalidValue)
      {
         *memory_location = hypre_MEMORY_HOST;
      }
   }
   else if (attr.isManaged)
   {
      *memory_location = hypre_MEMORY_UNIFIED;
   }
#if (HIP_VERSION_MAJOR >= 6)
   else if (attr.type == hipMemoryTypeDevice)
#else // (HIP_VERSION_MAJOR < 6)
   else if (attr.memoryType == hipMemoryTypeDevice)
#endif // (HIP_VERSION_MAJOR >= 6)
   {
      *memory_location = hypre_MEMORY_DEVICE;
   }
#if (HIP_VERSION_MAJOR >= 6)
   else if (attr.type == hipMemoryTypeHost)
#else // (HIP_VERSION_MAJOR < 6)
   else if (attr.memoryType == hipMemoryTypeHost)
#endif // (HIP_VERSION_MAJOR >= 6)
   {
      *memory_location = hypre_MEMORY_HOST_PINNED;
   }
#endif // defined(HYPRE_USING_HIP)

#if defined(HYPRE_USING_SYCL)
   /* If the device is not setup, then all allocations are assumed to be on the host */
   *memory_location = hypre_MEMORY_HOST;
   if (hypre_HandleDeviceData(hypre_handle()))
   {
      if (hypre_HandleDevice(hypre_handle()))
      {
         sycl::usm::alloc allocType;
         allocType = sycl::get_pointer_type(ptr, (hypre_HandleComputeStream(hypre_handle()))->get_context());

         if (allocType == sycl::usm::alloc::unknown)
         {
            *memory_location = hypre_MEMORY_HOST;
         }
         else if (allocType == sycl::usm::alloc::host)
         {
            *memory_location = hypre_MEMORY_HOST_PINNED;
         }
         else if (allocType == sycl::usm::alloc::device)
         {
            *memory_location = hypre_MEMORY_DEVICE;
         }
         else if (allocType == sycl::usm::alloc::shared)
         {
            *memory_location = hypre_MEMORY_UNIFIED;
         }
      }
   }
#endif //HYPRE_USING_SYCL

#else /* #if defined(HYPRE_USING_GPU) */
   *memory_location = hypre_MEMORY_HOST;
   HYPRE_UNUSED_VAR(ptr);
#endif

   return ierr;
}

/*--------------------------------------------------------------------------*
 * Memory Pool
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetCubMemPoolSize(hypre_uint cub_bin_growth,
                        hypre_uint cub_min_bin,
                        hypre_uint cub_max_bin,
                        size_t     cub_max_cached_bytes)
{
#if defined(HYPRE_USING_CUDA) && defined(HYPRE_USING_DEVICE_POOL)
   hypre_HandleCubBinGrowth(hypre_handle())      = cub_bin_growth;
   hypre_HandleCubMinBin(hypre_handle())         = cub_min_bin;
   hypre_HandleCubMaxBin(hypre_handle())         = cub_max_bin;
   hypre_HandleCubMaxCachedBytes(hypre_handle()) = cub_max_cached_bytes;

   //TODO XXX RL: cub_min_bin, cub_max_bin are not (re)set
   if (hypre_HandleCubDevAllocator(hypre_handle()))
   {
      hypre_HandleCubDevAllocator(hypre_handle()) -> SetMaxCachedBytes(cub_max_cached_bytes);
   }

   if (hypre_HandleCubUvmAllocator(hypre_handle()))
   {
      hypre_HandleCubUvmAllocator(hypre_handle()) -> SetMaxCachedBytes(cub_max_cached_bytes);
   }
#else
   HYPRE_UNUSED_VAR(cub_bin_growth);
   HYPRE_UNUSED_VAR(cub_min_bin);
   HYPRE_UNUSED_VAR(cub_max_bin);
   HYPRE_UNUSED_VAR(cub_max_cached_bytes);
#endif

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetGPUMemoryPoolSize(HYPRE_Int bin_growth,
                           HYPRE_Int min_bin,
                           HYPRE_Int max_bin,
                           size_t    max_cached_bytes)
{
   return hypre_SetCubMemPoolSize(bin_growth, min_bin, max_bin, max_cached_bytes);
}

#if defined(HYPRE_USING_DEVICE_POOL)
cudaError_t
hypre_CachingMallocDevice(void **ptr, size_t nbytes)
{
   if (!hypre_HandleCubDevAllocator(hypre_handle()))
   {
      hypre_HandleCubDevAllocator(hypre_handle()) =
         hypre_DeviceDataCubCachingAllocatorCreate( hypre_HandleCubBinGrowth(hypre_handle()),
                                                    hypre_HandleCubMinBin(hypre_handle()),
                                                    hypre_HandleCubMaxBin(hypre_handle()),
                                                    hypre_HandleCubMaxCachedBytes(hypre_handle()),
                                                    false,
                                                    false,
                                                    false );
   }

   return hypre_HandleCubDevAllocator(hypre_handle()) -> DeviceAllocate(ptr, nbytes);
}

cudaError_t
hypre_CachingFreeDevice(void *ptr)
{
   return hypre_HandleCubDevAllocator(hypre_handle()) -> DeviceFree(ptr);
}

cudaError_t
hypre_CachingMallocManaged(void **ptr, size_t nbytes)
{
   if (!hypre_HandleCubUvmAllocator(hypre_handle()))
   {
      hypre_HandleCubUvmAllocator(hypre_handle()) =
         hypre_DeviceDataCubCachingAllocatorCreate( hypre_HandleCubBinGrowth(hypre_handle()),
                                                    hypre_HandleCubMinBin(hypre_handle()),
                                                    hypre_HandleCubMaxBin(hypre_handle()),
                                                    hypre_HandleCubMaxCachedBytes(hypre_handle()),
                                                    false,
                                                    false,
                                                    true );
   }

   return hypre_HandleCubUvmAllocator(hypre_handle()) -> DeviceAllocate(ptr, nbytes);
}

cudaError_t
hypre_CachingFreeManaged(void *ptr)
{
   return hypre_HandleCubUvmAllocator(hypre_handle()) -> DeviceFree(ptr);
}

hypre_cub_CachingDeviceAllocator *
hypre_DeviceDataCubCachingAllocatorCreate(hypre_uint bin_growth,
                                          hypre_uint min_bin,
                                          hypre_uint max_bin,
                                          size_t     max_cached_bytes,
                                          bool       skip_cleanup,
                                          bool       debug,
                                          bool       use_managed_memory)
{
   hypre_cub_CachingDeviceAllocator *allocator =
      new hypre_cub_CachingDeviceAllocator( bin_growth,
                                            min_bin,
                                            max_bin,
                                            max_cached_bytes,
                                            skip_cleanup,
                                            debug,
                                            use_managed_memory );

   return allocator;
}

void
hypre_DeviceDataCubCachingAllocatorDestroy(hypre_DeviceData *data)
{
   delete hypre_DeviceDataCubDevAllocator(data);
   delete hypre_DeviceDataCubUvmAllocator(data);
}

#endif // #if defined(HYPRE_USING_DEVICE_POOL)

#if defined(HYPRE_USING_UMPIRE_HOST)
HYPRE_Int
hypre_umpire_host_pooled_allocate(void **ptr, size_t nbytes)
{
   hypre_Handle *handle = hypre_handle();
   const char *resource_name = "HOST";
   const char *pool_name = hypre_HandleUmpireHostPoolName(handle);

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       hypre_HandleUmpireHostPoolSize(handle),
                                                       hypre_HandleUmpireBlockSize(handle), &pooled_allocator);
      hypre_HandleOwnUmpireHostPool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return hypre_error_flag;
}

HYPRE_Int
hypre_umpire_host_pooled_free(void *ptr)
{
   hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpireHostPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return hypre_error_flag;
}

void *
hypre_umpire_host_pooled_realloc(void *ptr, size_t size)
{
   hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpireHostPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   ptr = umpire_resourcemanager_reallocate_with_allocator(rm_ptr, ptr, size, pooled_allocator);

   return ptr;
}
#endif

#if defined(HYPRE_USING_UMPIRE_DEVICE)
HYPRE_Int
hypre_umpire_device_pooled_allocate(void **ptr, size_t nbytes)
{
   hypre_Handle *handle = hypre_handle();
   const hypre_int device_id = hypre_HandleDevice(handle);
   char resource_name[16];
   const char *pool_name = hypre_HandleUmpireDevicePoolName(handle);

   hypre_sprintf(resource_name, "%s::%d", "DEVICE", device_id);

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       hypre_HandleUmpireDevicePoolSize(handle),
                                                       hypre_HandleUmpireBlockSize(handle), &pooled_allocator);

      hypre_HandleOwnUmpireDevicePool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return hypre_error_flag;
}

HYPRE_Int
hypre_umpire_device_pooled_free(void *ptr)
{
   hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpireDevicePoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return hypre_error_flag;
}
#endif

#if defined(HYPRE_USING_UMPIRE_UM)
HYPRE_Int
hypre_umpire_um_pooled_allocate(void **ptr, size_t nbytes)
{
   hypre_Handle *handle = hypre_handle();
   const char *resource_name = "UM";
   const char *pool_name = hypre_HandleUmpireUMPoolName(handle);

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       hypre_HandleUmpireUMPoolSize(handle),
                                                       hypre_HandleUmpireBlockSize(handle), &pooled_allocator);

      hypre_HandleOwnUmpireUMPool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return hypre_error_flag;
}

HYPRE_Int
hypre_umpire_um_pooled_free(void *ptr)
{
   hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpireUMPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return hypre_error_flag;
}
#endif

#if defined(HYPRE_USING_UMPIRE_PINNED)
HYPRE_Int
hypre_umpire_pinned_pooled_allocate(void **ptr, size_t nbytes)
{
   hypre_Handle *handle = hypre_handle();
   const char *resource_name = "PINNED";
   const char *pool_name = hypre_HandleUmpirePinnedPoolName(handle);

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       hypre_HandleUmpirePinnedPoolSize(handle),
                                                       hypre_HandleUmpireBlockSize(handle), &pooled_allocator);

      hypre_HandleOwnUmpirePinnedPool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return hypre_error_flag;
}

HYPRE_Int
hypre_umpire_pinned_pooled_free(void *ptr)
{
   hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpirePinnedPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return hypre_error_flag;
}
#endif
