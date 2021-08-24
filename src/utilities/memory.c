/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

#ifdef HYPRE_USE_UMALLOC
#undef HYPRE_USE_UMALLOC
#endif

/******************************************************************************
 *
 * Helper routines
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_OutOfMemory
 *--------------------------------------------------------------------------*/
static inline void
hypre_OutOfMemory(size_t size)
{
   hypre_error_w_msg(HYPRE_ERROR_MEMORY,"Out of memory trying to allocate too many bytes\n");
   hypre_assert(0);
   fflush(stdout);
}

static inline void
hypre_WrongMemoryLocation()
{
   hypre_error_w_msg(HYPRE_ERROR_MEMORY, "Wrong HYPRE MEMORY location: Only HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE and HYPRE_MEMORY_HOST_PINNED are supported!\n");
   hypre_assert(0);
   fflush(stdout);
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
   HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );
#else
   memset(ptr, value, num);
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, num, "update", "to");
#endif
   HYPRE_CUDA_CALL( cudaDeviceSynchronize() );
#endif

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipMemset(ptr, value, num) );
#endif
}

static inline void
hypre_UnifiedMemset(void *ptr, HYPRE_Int value, size_t num)
{
#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );
   HYPRE_CUDA_CALL( cudaDeviceSynchronize() );
#endif

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipMemset(ptr, value, num) );
#endif
}

/*--------------------------------------------------------------------------
 * Memprefetch
 *--------------------------------------------------------------------------*/
static inline void
hypre_UnifiedMemPrefetch(void *ptr, size_t size, hypre_MemoryLocation location)
{
#if defined(HYPRE_USING_GPU)
#ifdef HYPRE_DEBUG
   hypre_MemoryLocation tmp;
   hypre_GetPointerLocation(ptr, &tmp);
   /* do not use hypre_assert, which has alloc and free;
    * will create an endless loop otherwise */
   assert(hypre_MEMORY_UNIFIED == tmp);
#endif
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
   if (location == hypre_MEMORY_DEVICE)
   {
      HYPRE_CUDA_CALL( cudaMemPrefetchAsync(ptr, size, hypre_HandleCudaDevice(hypre_handle()),
                       hypre_HandleCudaComputeStream(hypre_handle())) );
   }
   else if (location == hypre_MEMORY_HOST)
   {
      HYPRE_CUDA_CALL( cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId,
                       hypre_HandleCudaComputeStream(hypre_handle())) );
   }
#endif

#if defined(HYPRE_USING_CUDA)
   if (location == hypre_MEMORY_DEVICE)
   {
      HYPRE_CUDA_CALL( cudaMemPrefetchAsync(ptr, size, hypre_HandleCudaDevice(hypre_handle()),
                       hypre_HandleCudaComputeStream(hypre_handle())) );
   }
   else if (location == hypre_MEMORY_HOST)
   {
      HYPRE_CUDA_CALL( cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId,
                       hypre_HandleCudaComputeStream(hypre_handle())) );
   }
#endif

#if defined(HYPRE_USING_HIP)
   // Not currently implemented for HIP, but leaving place holder
   /*
    *if (location == hypre_MEMORY_DEVICE)
    *{
    *  HYPRE_HIP_CALL( hipMemPrefetchAsync(ptr, size, hypre_HandleCudaDevice(hypre_handle()),
    *                   hypre_HandleCudaComputeStream(hypre_handle())) );
    *}
    *else if (location == hypre_MEMORY_HOST)
    *{
    *   HYPRE_CUDA_CALL( hipMemPrefetchAsync(ptr, size, cudaCpuDeviceId,
    *                    hypre_HandleCudaComputeStream(hypre_handle())) );
    *}
    */
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
#else
   HYPRE_CUDA_CALL( cudaMalloc(&ptr, size) );
#endif
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipMalloc(&ptr, size) );
#endif

#endif /* #if defined(HYPRE_USING_UMPIRE_DEVICE) */

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
   HYPRE_CUDA_CALL( cudaMallocManaged(&ptr, size, cudaMemAttachGlobal) );
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

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaMallocHost(&ptr, size) );
#endif

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaMallocHost(&ptr, size) );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipHostMalloc(&ptr, size) );
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
#else
   HYPRE_CUDA_CALL( cudaFree(ptr) );
#endif
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipFree(ptr) );
#endif

#endif /* #if defined(HYPRE_USING_UMPIRE_DEVICE) */
}

static inline void
hypre_UnifiedFree(void *ptr)
{
#if defined(HYPRE_USING_UMPIRE_UM)
   hypre_umpire_um_pooled_free(ptr);
#else

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaFree(ptr) );
#endif

#if defined(HYPRE_USING_CUDA)
#if defined(HYPRE_USING_DEVICE_POOL)
   HYPRE_CUDA_CALL( hypre_CachingFreeManaged(ptr) );
#else
   HYPRE_CUDA_CALL( cudaFree(ptr) );
#endif
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipFree(ptr) );
#endif

#endif /* #if defined(HYPRE_USING_UMPIRE_UM) */
}

static inline void
hypre_HostPinnedFree(void *ptr)
{
#if defined(HYPRE_USING_UMPIRE_PINNED)
   hypre_umpire_pinned_pooled_free(ptr);
#else

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaFreeHost(ptr) );
#endif

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaFreeHost(ptr) );
#endif

#if defined(HYPRE_USING_HIP)
   HYPRE_HIP_CALL( hipHostFree(ptr) );
#endif

#endif /* #if defined(HYPRE_USING_UMPIRE_PINNED) */
}

static inline void
hypre_Free_core(void *ptr, hypre_MemoryLocation location)
{
   if (!ptr)
   {
      return;
   }

#ifdef HYPRE_DEBUG
   hypre_MemoryLocation tmp;
   hypre_GetPointerLocation(ptr, &tmp);
   /* do not use hypre_assert, which has alloc and free;
    * will create an endless loop otherwise */
   assert(location == tmp);
#endif

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
hypre_Memcpy_core(void *dst, void *src, size_t size, hypre_MemoryLocation loc_dst, hypre_MemoryLocation loc_src)
{
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
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) );
#endif

#if defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) );
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice) );
#endif
      return;
   }


   /* 2: UVM <-- Host, UVM <-- Pinned */
   if (loc_dst == hypre_MEMORY_UNIFIED)
   {
#if defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
#endif

#if defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyHostToDevice) );
#endif
      return;
   }


   /* 2: Host <-- UVM, Pinned <-- UVM */
   if (loc_src == hypre_MEMORY_UNIFIED)
   {
#if defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) );
#endif

#if defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) );
#endif

#if defined(HYPRE_USING_HIP)
      HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToHost) );
#endif
      return;
   }


   /* 2: Device <-- Host, Device <-- Pinned */
   if ( loc_dst == hypre_MEMORY_DEVICE && (loc_src == hypre_MEMORY_HOST || loc_src == hypre_MEMORY_HOST_PINNED) )
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
      return;
   }


   /* 2: Host <-- Device, Pinned <-- Device */
   if ( (loc_dst == hypre_MEMORY_HOST || loc_dst == hypre_MEMORY_HOST_PINNED) && loc_src == hypre_MEMORY_DEVICE )
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
#if defined(HYPRE_USING_GPU)
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
#if defined(HYPRE_USING_GPU)
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
hypre_Memcpy(void *dst, void *src, size_t size, HYPRE_MemoryLocation loc_dst, HYPRE_MemoryLocation loc_src)
{
   hypre_Memcpy_core( dst, src, size, hypre_GetActualMemLocation(loc_dst), hypre_GetActualMemLocation(loc_src) );
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

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
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
#endif // defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

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
   else if (attr.memoryType == hipMemoryTypeDevice)
   {
      *memory_location = hypre_MEMORY_DEVICE;
   }
   else if (attr.memoryType == hipMemoryTypeHost)
   {
      *memory_location = hypre_MEMORY_HOST_PINNED;
   }
#endif // defined(HYPRE_USING_HIP)

#else /* #if defined(HYPRE_USING_GPU) */
   *memory_location = hypre_MEMORY_HOST;
#endif

   return ierr;
}

#ifdef HYPRE_USING_MEMORY_TRACKER

/*--------------------------------------------------------------------------
 * Memory tracker
 * do not use hypre_T* in the following since we don't want to track them   *
 *--------------------------------------------------------------------------*/
hypre_MemoryTracker *
hypre_MemoryTrackerCreate()
{
   hypre_MemoryTracker *ptr = (hypre_MemoryTracker *) calloc(1, sizeof(hypre_MemoryTracker));
   return ptr;
}

void
hypre_MemoryTrackerDestroy(hypre_MemoryTracker *tracker)
{
   if (tracker)
   {
      free(tracker->data);
      free(tracker);
   }
}

void
hypre_MemoryTrackerInsert(const char           *action,
                          void                 *ptr,
                          size_t                nbytes,
                          hypre_MemoryLocation  memory_location,
                          const char           *filename,
                          const char           *function,
                          HYPRE_Int             line)
{

   if (ptr == NULL)
   {
      return;
   }

   hypre_MemoryTracker *tracker = hypre_memory_tracker();

   if (tracker->alloced_size <= tracker->actual_size)
   {
      tracker->alloced_size = 2 * tracker->alloced_size + 1;
      tracker->data = (hypre_MemoryTrackerEntry *) realloc(tracker->data, tracker->alloced_size * sizeof(hypre_MemoryTrackerEntry));
   }

   hypre_assert(tracker->actual_size < tracker->alloced_size);

   hypre_MemoryTrackerEntry *entry = tracker->data + tracker->actual_size;

   sprintf(entry->_action, "%s", action);
   entry->_ptr = ptr;
   entry->_nbytes = nbytes;
   entry->_memory_location = memory_location;
   sprintf(entry->_filename, "%s", filename);
   sprintf(entry->_function, "%s", function);
   entry->_line = line;
   /* -1 is the initial value */
   entry->_pair = (size_t) -1;

   tracker->actual_size ++;
}


/* do not use hypre_printf, hypre_fprintf, which have TAlloc
 * endless loop "for (i = 0; i < tracker->actual_size; i++)" otherwise */
HYPRE_Int
hypre_PrintMemoryTracker()
{
   HYPRE_Int myid, ierr = 0;
   char filename[256];
   FILE *file;
   size_t i, j;

   hypre_MemoryTracker *tracker = hypre_memory_tracker();

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_sprintf(filename,"HypreMemoryTrack.log.%05d", myid);
   if ((file = fopen(filename, "a")) == NULL)
   {
      fprintf(stderr, "Error: can't open output file %s\n", filename);
      return hypre_error_flag;
   }

   fprintf(file, "==== Operations:\n");
   fprintf(file, "     ID        EVENT                 ADDRESS       BYTE         LOCATION                                       FILE(LINE)                                           FUNCTION  |  Memory (   H            P            D            U )\n");

   size_t totl_bytes[hypre_MEMORY_UNIFIED+1] = {0};
   size_t peak_bytes[hypre_MEMORY_UNIFIED+1] = {0};
   size_t curr_bytes[hypre_MEMORY_UNIFIED+1] = {0};

   for (i = 0; i < tracker->actual_size; i++)
   {
      if (strstr(tracker->data[i]._action, "alloc") != NULL)
      {
         totl_bytes[tracker->data[i]._memory_location] += tracker->data[i]._nbytes;
         curr_bytes[tracker->data[i]._memory_location] += tracker->data[i]._nbytes;
         peak_bytes[tracker->data[i]._memory_location] =
            hypre_max( curr_bytes[tracker->data[i]._memory_location],
                       peak_bytes[tracker->data[i]._memory_location] );

         /* for each unpaired "alloc", find its "free" */
         if (tracker->data[i]._pair != (size_t) -1)
         {
            if ( tracker->data[i]._pair >= tracker->actual_size ||
                 tracker->data[tracker->data[i]._pair]._pair != i)
            {
               fprintf(stderr, "hypre memory tracker internal error!\n");
               hypre_MPI_Abort(hypre_MPI_COMM_WORLD, 1);
            }

            continue;
         }

         for (j = i+1; j < tracker->actual_size; j++)
         {
            if ( strstr(tracker->data[j]._action, "free") != NULL &&
                 tracker->data[j]._pair == (size_t) -1 &&
                 tracker->data[i]._ptr == tracker->data[j]._ptr &&
                 tracker->data[i]._memory_location == tracker->data[j]._memory_location )
            {
               tracker->data[i]._pair = j;
               tracker->data[j]._pair = i;
               tracker->data[j]._nbytes = tracker->data[i]._nbytes;
               break;
            }
         }

         if (tracker->data[i]._pair == (size_t) -1)
         {
            fprintf(stderr, "%6zu: %16p may not freed\n", i, tracker->data[i]._ptr );
         }
      }
      else if (strstr(tracker->data[i]._action, "free") != NULL)
      {
         size_t pair = tracker->data[i]._pair;

         if (pair == (size_t) -1)
         {
            fprintf(stderr, "%6zu: unpaired free at %16p\n", i, tracker->data[i]._ptr );
         }
         else
         {
            curr_bytes[tracker->data[i]._memory_location] -= tracker->data[pair]._nbytes;
         }
      }

      if (i < tracker->prev_end)
      {
         continue;
      }

      char memory_location[256];
      char nbytes[32];

      if (tracker->data[i]._memory_location == hypre_MEMORY_HOST)
      {
         sprintf(memory_location, "%s", "HOST");
      }
      else if (tracker->data[i]._memory_location == hypre_MEMORY_HOST_PINNED)
      {
         sprintf(memory_location, "%s", "HOST_PINNED");
      }
      else if (tracker->data[i]._memory_location == hypre_MEMORY_DEVICE)
      {
         sprintf(memory_location, "%s", "DEVICE");
      }
      else if (tracker->data[i]._memory_location == hypre_MEMORY_UNIFIED)
      {
         sprintf(memory_location, "%s", "UNIFIED");
      }
      else
      {
         sprintf(memory_location, "%s", "UNDEFINED");
      }

      if (tracker->data[i]._nbytes != (size_t) -1)
      {
         sprintf(nbytes, "%zu", tracker->data[i]._nbytes);
      }
      else
      {
         sprintf(nbytes, "%s", "");
      }

      fprintf(file, " %6zu %12s        %16p %10s %16s %40s (%5d) %50s  |  %12zu %12zu %12zu %12zu\n",
              i,
              tracker->data[i]._action,
              tracker->data[i]._ptr,
              nbytes,
              memory_location,
              tracker->data[i]._filename,
              tracker->data[i]._line,
              tracker->data[i]._function,
              curr_bytes[hypre_MEMORY_HOST],
              curr_bytes[hypre_MEMORY_HOST_PINNED],
              curr_bytes[hypre_MEMORY_DEVICE],
              curr_bytes[hypre_MEMORY_UNIFIED]
              );
   }

   fprintf(file, "\n==== Total allocated (byte):\n");
   fprintf(file, "HOST: %16zu, HOST_PINNED %16zu, DEVICE %16zu, UNIFIED %16zu\n",
           totl_bytes[hypre_MEMORY_HOST],
           totl_bytes[hypre_MEMORY_HOST_PINNED],
           totl_bytes[hypre_MEMORY_DEVICE],
           totl_bytes[hypre_MEMORY_UNIFIED]);

   fprintf(file, "\n==== Peak (byte):\n");
   fprintf(file, "HOST: %16zu, HOST_PINNED %16zu, DEVICE %16zu, UNIFIED %16zu\n",
           peak_bytes[hypre_MEMORY_HOST],
           peak_bytes[hypre_MEMORY_HOST_PINNED],
           peak_bytes[hypre_MEMORY_DEVICE],
           peak_bytes[hypre_MEMORY_UNIFIED]);

   fprintf(file, "\n==== Reachable (byte):\n");
   fprintf(file, "HOST: %16zu, HOST_PINNED %16zu, DEVICE %16zu, UNIFIED %16zu\n",
           curr_bytes[hypre_MEMORY_HOST],
           curr_bytes[hypre_MEMORY_HOST_PINNED],
           curr_bytes[hypre_MEMORY_DEVICE],
           curr_bytes[hypre_MEMORY_UNIFIED]);

   fprintf(file, "\n==== Warnings:\n");
   for (i = 0; i < tracker->actual_size; i++)
   {
      if (tracker->data[i]._pair == (size_t) -1)
      {
         if (strstr(tracker->data[i]._action, "alloc") != NULL)
         {
            fprintf(file, "%6zu: %p may have not been freed\n", i, tracker->data[i]._ptr );
         }
         else if (strstr(tracker->data[i]._action, "free") != NULL)
         {
            fprintf(file, "%6zu: unpaired free at %16p\n", i, tracker->data[i]._ptr );
         }
      }
   }

   fclose(file);

   tracker->prev_end = tracker->actual_size;

   return ierr;
}
#endif

/*--------------------------------------------------------------------------*
 * Memory Pool
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetCubMemPoolSize(hypre_uint cub_bin_growth,
                        hypre_uint cub_min_bin,
                        hypre_uint cub_max_bin,
                        size_t     cub_max_cached_bytes)
{
#if defined(HYPRE_USING_CUDA)
#ifdef HYPRE_USING_DEVICE_POOL
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
#endif
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

#ifdef HYPRE_USING_DEVICE_POOL
cudaError_t
hypre_CachingMallocDevice(void **ptr, size_t nbytes)
{
   if (!hypre_HandleCubDevAllocator(hypre_handle()))
   {
      hypre_HandleCubDevAllocator(hypre_handle()) =
         hypre_CudaDataCubCachingAllocatorCreate( hypre_HandleCubBinGrowth(hypre_handle()),
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
         hypre_CudaDataCubCachingAllocatorCreate( hypre_HandleCubBinGrowth(hypre_handle()),
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
hypre_CudaDataCubCachingAllocatorCreate(hypre_uint bin_growth,
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
hypre_CudaDataCubCachingAllocatorDestroy(hypre_CudaData *data)
{
   delete hypre_CudaDataCubDevAllocator(data);
   delete hypre_CudaDataCubUvmAllocator(data);
}

#endif // #ifdef HYPRE_USING_DEVICE_POOL

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
      umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
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
   const hypre_int device_id = hypre_HandleCudaDevice(handle);
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
      umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
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
      umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
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
      umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
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
   const hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpirePinnedPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return hypre_error_flag;
}
#endif

