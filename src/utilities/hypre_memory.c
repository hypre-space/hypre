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
#include "../struct_mv/_hypre_struct_mv.h"

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
   fflush(stdout);
}

static inline void
hypre_WrongMemoryLocation()
{
   hypre_error_w_msg(HYPRE_ERROR_MEMORY,"Wrong HYPRE MEMORY location: \n Only HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_SHARED,\n and HYPRE_MEMORY_HOST_PINNED are supported!\n");
   fflush(stdout);
}

/******************************************************************************
 *
 * Standard routines
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_MAlloc
 *--------------------------------------------------------------------------*/

static inline void *
hypre_HostMalloc(size_t size, HYPRE_Int zeroinit)
{
   void *ptr = NULL;

   if (zeroinit)
   {
      ptr = calloc(size, 1);
   }
   else
   {
      ptr = malloc(size);
   }

   return ptr;
}

static inline void *
hypre_DeviceMalloc(size_t size, HYPRE_Int zeroinit)
{
   void *ptr = NULL;

#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
   ptr = omp_target_alloc(size, hypre__offload_device_num);
#elif defined(HYPRE_USING_DEVICE_OPENMP)
   ptr = malloc(size + sizeof(size_t));
   size_t *sp = (size_t*) ptr;
   sp[0] = size;
   ptr = (void *) (&sp[1]);
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, size, "enter", "alloc");
#elif defined(HYPRE_USING_CUDA)
#if defined(HYPRE_USING_CUB_ALLOCATOR)
   HYPRE_CUDA_CALL( hypre_HandleCubCachingDeviceAllocator(hypre_handle)->DeviceAllocate( (void**)&ptr, size ) );
#else
   HYPRE_CUDA_CALL( cudaMalloc(&ptr, size) );
#endif
   /* HYPRE_CUDA_CALL( cudaDeviceSynchronize() ); */
#endif

   if (ptr && zeroinit)
   {
      hypre_Memset(ptr, 0, size, HYPRE_MEMORY_DEVICE);
   }

   return ptr;
}

static inline void *
hypre_UnifiedMalloc(size_t size, HYPRE_Int zeroinit)
{
   void *ptr = NULL;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_USING_CUB_ALLOCATOR)
   HYPRE_CUDA_CALL( hypre_HandleCubCachingManagedAllocator(hypre_handle)->DeviceAllocate( (void**)&ptr, size ) );
#else
   HYPRE_CUDA_CALL( cudaMallocManaged(&ptr, size, cudaMemAttachGlobal) );
#endif
   HYPRE_CUDA_CALL( cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation,
                                  hypre_HandleCudaDevice(hypre_handle)) );
   /* prefecth to device */
   hypre_Memcpy(ptr, ptr, size, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_SHARED);

   if (zeroinit)
   {
      hypre_Memset(ptr, 0, size, HYPRE_MEMORY_SHARED);
   }

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaDeviceSynchronize() );
#endif

#endif

   return ptr;
}

static inline void *
hypre_HostPinnedMalloc(size_t size, HYPRE_Int zeroinit)
{
   void *ptr = NULL;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaMallocHost(&ptr, size) );

   if (zeroinit)
   {
      hypre_Memset(ptr, 0, size, HYPRE_MEMORY_HOST_PINNED);
   }
#endif

   return ptr;
}

static inline void *
hypre_MAlloc_core(size_t size, HYPRE_Int zeroinit, HYPRE_Int location)
{
   if (size == 0)
   {
      return NULL;
   }

   void *ptr = NULL;

   location = hypre_GetActualMemLocation(location);

   switch (location)
   {
      case HYPRE_MEMORY_HOST :
         /* ask for cpu memory */
         ptr = hypre_HostMalloc(size, zeroinit);
         break;
      case HYPRE_MEMORY_DEVICE :
         /* ask for device memory */
         ptr = hypre_DeviceMalloc(size, zeroinit);
         break;
      case HYPRE_MEMORY_SHARED :
         /* ask for unified memory */
         ptr = hypre_UnifiedMalloc(size, zeroinit);
         break;
      case HYPRE_MEMORY_HOST_PINNED :
         /* ask for page-locked memory on the host */
         ptr = hypre_HostPinnedMalloc(size, zeroinit);
         break;
      default :
         /* unrecognized location */
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
hypre_MAlloc(size_t size, HYPRE_Int location)
{
   return hypre_MAlloc_core(size, 0, location);
}

void *
hypre_CAlloc( size_t count, size_t elt_size, HYPRE_Int location)
{
   return hypre_MAlloc_core(count * elt_size, 1, location);
}


/*--------------------------------------------------------------------------
 * hypre_Free
 *--------------------------------------------------------------------------*/

static inline void
hypre_HostFree(void *ptr)
{
   free(ptr);
}

static inline void
hypre_DeviceFree(void *ptr)
{
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
   omp_target_free(ptr, hypre__offload_device_num);
#elif defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, ((size_t *) ptr)[-1], "exit", "delete");
#elif defined(HYPRE_USING_CUDA)
#ifdef HYPRE_USING_CUB_ALLOCATOR
   HYPRE_CUDA_CALL( hypre_HandleCubCachingDeviceAllocator(hypre_handle)->DeviceFree(ptr) );
#else
   HYPRE_CUDA_CALL( cudaFree(ptr) );
#endif
#endif
}

static inline void
hypre_UnifiedFree(void *ptr)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#ifdef HYPRE_USING_CUB_ALLOCATOR
   HYPRE_CUDA_CALL( hypre_HandleCubCachingManagedAllocator(hypre_handle)->DeviceFree(ptr) );
#else
   HYPRE_CUDA_CALL( cudaFree(ptr) );
#endif
#endif
}

static inline void
hypre_HostPinnedFree(void *ptr)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaFreeHost(ptr) );
#endif
}

void
hypre_Free(void *ptr, HYPRE_Int location)
{
   if (!ptr)
   {
      return;
   }

   location = hypre_GetActualMemLocation(location);

#ifdef HYPRE_DEBUG
   HYPRE_Int tmp;
   hypre_GetMemoryLocation(ptr, &tmp);
   /* do not use hypre_assert, which has alloc and free;
    * will create an endless loop otherwise */
   assert(location == tmp);
#endif

   switch (location)
   {
      case HYPRE_MEMORY_HOST :
         /* free cpu memory */
         hypre_HostFree(ptr);
         break;
      case HYPRE_MEMORY_DEVICE :
         /* free device memory */
         hypre_DeviceFree(ptr);
         break;
      case HYPRE_MEMORY_SHARED :
         /* free unified memory */
         hypre_UnifiedFree(ptr);
         break;
      case HYPRE_MEMORY_HOST_PINNED :
         /* free host page-locked memory */
         hypre_HostPinnedFree(ptr);
         break;
      default :
         /* unrecognized location */
         hypre_WrongMemoryLocation();
   }
}

/*--------------------------------------------------------------------------
 * hypre_ReAlloc
 *--------------------------------------------------------------------------*/
static inline void *
hypre_HostReAlloc(void *ptr, size_t size)
{
   return realloc(ptr, size);
}

static inline void *
hypre_Device_Unified_HostPinned_ReAlloc(void *ptr, size_t old_size, size_t new_size, HYPRE_Int location)
{
   /* device/unified/hostpinned memory realloc: malloc+copy+free */
   void *new_ptr = hypre_MAlloc(new_size, location);
   size_t smaller_size = new_size > old_size ? old_size : new_size;
   hypre_Memcpy(new_ptr, ptr, smaller_size, location, location);
   hypre_Free(ptr, location);

   return new_ptr;
}

void *
hypre_ReAlloc(void *ptr, size_t size, HYPRE_Int location)
{
   location = hypre_GetActualMemLocation(location);

   if (size == 0)
   {
      hypre_Free(ptr, location);
      return NULL;
   }

   if (ptr == NULL)
   {
      return hypre_MAlloc(size, location);
   }

   if (location != HYPRE_MEMORY_HOST)
   {
      hypre_printf("hypre_TReAlloc only works with HYPRE_MEMORY_HOST; Use hypre_TReAlloc_v2 instead!\n");
      hypre_MPI_Abort(hypre_MPI_COMM_WORLD, -1);
      return NULL;
   }

   ptr = hypre_HostReAlloc(ptr, size);

   if (!ptr)
   {
      hypre_OutOfMemory(size);
   }

   return ptr;
}

void *
hypre_ReAlloc_v2(void *ptr, size_t old_size, size_t new_size, HYPRE_Int location)
{
   location = hypre_GetActualMemLocation(location);

   if (new_size == 0)
   {
      hypre_Free(ptr, location);
      return NULL;
   }

   if (ptr == NULL)
   {
      return hypre_MAlloc(new_size, location);
   }

   switch (location)
   {
      case HYPRE_MEMORY_HOST :
         /* realloc cpu memory */
         ptr = hypre_HostReAlloc(ptr, new_size);
         break;
      case HYPRE_MEMORY_DEVICE :
         /* realloc device memory */
      case HYPRE_MEMORY_SHARED :
         /* realloc unified memory */
      case HYPRE_MEMORY_HOST_PINNED :
         /* realloc host pinned memory */
         ptr = hypre_Device_Unified_HostPinned_ReAlloc(ptr, old_size, new_size, location);
         break;
      default :
         /* unrecognized location */
         hypre_WrongMemoryLocation();
   }

   if (!ptr)
   {
      hypre_OutOfMemory(new_size);
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_Memcpy
 *--------------------------------------------------------------------------*/
void
hypre_Memcpy(void *dst, void *src, size_t size, HYPRE_Int loc_dst, HYPRE_Int loc_src)
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

   loc_dst = hypre_GetActualMemLocation(loc_dst);
   loc_src = hypre_GetActualMemLocation(loc_src);

   /* special uses for GPU shared memory prefetch */
#if defined(HYPRE_USING_UNIFIED_MEMORY)
   if ( dst == src && loc_src == HYPRE_MEMORY_SHARED && (loc_dst == HYPRE_MEMORY_DEVICE || loc_dst == HYPRE_MEMORY_HOST) )
   {
      /* src (== dst) must point to cuda unified memory */
      if (loc_dst == HYPRE_MEMORY_DEVICE)
      {
         HYPRE_CUDA_CALL(
         cudaMemPrefetchAsync(src, size, hypre_HandleCudaDevice(hypre_handle),
                              hypre_HandleCudaComputeStream(hypre_handle))
         );
      }
      else if (loc_dst == HYPRE_MEMORY_HOST)
      {
         HYPRE_CUDA_CALL(
         cudaMemPrefetchAsync(src, size, cudaCpuDeviceId,
                              hypre_HandleCudaComputeStream(hypre_handle))
         );
      }

      return;
   }
#endif

   if (dst == src)
   {
      return;
   }

   /* Totally 4 x 4 = 16 cases */

   /* 4: Host   <-- Host, Host   <-- Pinned,
    *    Pinned <-- Host, Pinned <-- Pinned.
    */
   if ( loc_dst != HYPRE_MEMORY_DEVICE && loc_dst != HYPRE_MEMORY_SHARED &&
        loc_src != HYPRE_MEMORY_DEVICE && loc_src != HYPRE_MEMORY_SHARED )
   {
      memcpy(dst, src, size);
      return;
   }


   /* 3: Shared <-- Device, Device <-- Shared, Shared <-- Shared */
   if ( (loc_dst == HYPRE_MEMORY_SHARED && loc_src == HYPRE_MEMORY_DEVICE) ||
        (loc_dst == HYPRE_MEMORY_DEVICE && loc_src == HYPRE_MEMORY_SHARED) ||
        (loc_dst == HYPRE_MEMORY_SHARED && loc_src == HYPRE_MEMORY_SHARED) )
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) );
#endif
      return;
   }


   /* 2: Shared <-- Host, Shared <-- Pinned */
   if (loc_dst == HYPRE_MEMORY_SHARED)
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
#endif
      return;
   }


   /* 2: Host <-- Shared, Pinned <-- Shared */
   if (loc_src == HYPRE_MEMORY_SHARED)
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) );
#endif
      return;
   }


   /* 2: Device <-- Host, Device <-- Pinned */
   if ( loc_dst == HYPRE_MEMORY_DEVICE && (loc_src == HYPRE_MEMORY_HOST || loc_src == HYPRE_MEMORY_HOST_PINNED) )
   {
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_device_num, hypre__offload_host_num);
#elif defined(HYPRE_USING_DEVICE_OPENMP)
      memcpy(dst, src, size);
      HYPRE_OMPOffload(hypre__offload_device_num, dst, size, "update", "to");
#elif defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
#endif
      return;
   }


   /* 2: Host <-- Device, Pinned <-- Device */
   if ( (loc_dst == HYPRE_MEMORY_HOST || loc_dst == HYPRE_MEMORY_HOST_PINNED) && loc_src == HYPRE_MEMORY_DEVICE )
   {
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_host_num, hypre__offload_device_num);
#elif defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_OMPOffload(hypre__offload_device_num, src, size, "update", "from");
      memcpy(dst, src, size);
#elif defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost) );
#endif
      return;
   }


   /* 1: Device <-- Device */
   if (loc_dst == HYPRE_MEMORY_DEVICE && loc_src == HYPRE_MEMORY_DEVICE)
   {
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_device_num, hypre__offload_device_num);
#elif defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_OMPOffload(hypre__offload_device_num, src, size, "update", "from");
      memcpy(dst, src, size);
      HYPRE_OMPOffload(hypre__offload_device_num, dst, size, "update", "to");
#elif defined(HYPRE_USING_CUDA)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) );
#endif
      return;
   }

   hypre_WrongMemoryLocation();
}

/*--------------------------------------------------------------------------
 * hypre_Memset
 * "Sets the first num bytes of the block of memory pointed by ptr to the specified value
 * (*** interpreted as an unsigned char ***)"
 * http://www.cplusplus.com/reference/cstring/memset/
 *--------------------------------------------------------------------------*/
void *
hypre_Memset(void *ptr, HYPRE_Int value, size_t num, HYPRE_Int location)
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

   location = hypre_GetActualMemLocation(location);

#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
   unsigned char *ucptr = (unsigned char *) ptr;
   unsigned char ucvalue = (unsigned char) value;
#endif

   switch (location)
   {
      case HYPRE_MEMORY_HOST :
         /* memset cpu memory */
      case HYPRE_MEMORY_HOST_PINNED :
         /* memset host pinned memory */
         memset(ptr, value, num);
         break;
      case HYPRE_MEMORY_DEVICE :
         /* memset device memory */
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
#define DEVICE_VAR is_device_ptr(ucptr)
         hypre_LoopBegin(num, k)
         {
            ucptr[k] = ucvalue;
         }
         hypre_LoopEnd()
#undef DEVICE_VAR
#elif defined(HYPRE_USING_DEVICE_OPENMP)
         memset(ptr, value, num);
         HYPRE_OMPOffload(hypre__offload_device_num, ptr, num, "update", "to");
#elif defined(HYPRE_USING_CUDA)
         HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );
#endif
         break;
      case HYPRE_MEMORY_SHARED :
         /* memset unified memory */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
         HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );
#endif
         break;
      default :
         /* unrecognized location */
         hypre_WrongMemoryLocation();
   }

   return ptr;
}

HYPRE_Int
hypre_GetMemoryLocation(const void *ptr, HYPRE_Int *memory_location)
{
   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   struct cudaPointerAttributes attr;
   *memory_location = HYPRE_MEMORY_UNSET;

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
      *memory_location = HYPRE_MEMORY_HOST;
   }
   else if (attr.type == cudaMemoryTypeHost)
   {
      *memory_location = HYPRE_MEMORY_HOST_PINNED;
   }
   else if (attr.type == cudaMemoryTypeDevice)
   {
      *memory_location = HYPRE_MEMORY_DEVICE;
   }
   else if (attr.type == cudaMemoryTypeManaged)
   {
      *memory_location = HYPRE_MEMORY_SHARED;
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
         *memory_location = HYPRE_MEMORY_HOST;
      }
   }
   else if (attr.isManaged)
   {
      *memory_location = HYPRE_MEMORY_SHARED;
   }
   else if (attr.memoryType == cudaMemoryTypeDevice)
   {
      *memory_location = HYPRE_MEMORY_DEVICE;
   }
   else if (attr.memoryType == cudaMemoryTypeHost)
   {
      *memory_location = HYPRE_MEMORY_HOST_PINNED;
   }
#endif

#else /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP) */
   *memory_location = HYPRE_MEMORY_HOST;
#endif

   return ierr;
}

#ifdef HYPRE_USING_MEMORY_TRACKER
std::vector<hypre_memory_tracker_t> hypre_memory_tracker;
#endif

HYPRE_Int
hypre_PrintMemoryTracker()
{
   HYPRE_Int ierr = 0;
#ifdef HYPRE_USING_MEMORY_TRACKER
   size_t i;
   HYPRE_Int myid;
   char filename[256];
   FILE *file;

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_sprintf(filename,"HypreMemoryTrack.log.%05d", myid);
   if ((file = fopen(filename, "w")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Error: can't open output file %s\n");
      return hypre_error_flag;
   }

   char *mark = hypre_CTAlloc(char, hypre_memory_tracker.size(), HYPRE_MEMORY_HOST);

   for (i = 0; i < hypre_memory_tracker.size(); i++)
   {
      if (hypre_memory_tracker[i]._ptr == NULL && hypre_memory_tracker[i]._nbytes == 0)
      {
         continue;
      }

      hypre_fprintf(file, "%6ld: %8s  %16p  %10ld  %d  %32s  %64s      %d\n", i,
            hypre_memory_tracker[i]._action,
            hypre_memory_tracker[i]._ptr,
            hypre_memory_tracker[i]._nbytes,
            hypre_memory_tracker[i]._memory_location,
            hypre_memory_tracker[i]._filename,
            hypre_memory_tracker[i]._function,
            hypre_memory_tracker[i]._line);

      if ( strstr(hypre_memory_tracker[i]._action, "alloc") != NULL)
      {
         size_t j;
         HYPRE_Int found = 0;
         for (j = i+1; j < hypre_memory_tracker.size(); j++)
         {
            if ( mark[j] == 0 &&
                 strstr(hypre_memory_tracker[j]._action, "free") != NULL &&
                 hypre_memory_tracker[i]._ptr == hypre_memory_tracker[j]._ptr &&
                 hypre_memory_tracker[i]._memory_location == hypre_memory_tracker[j]._memory_location )
            {
               mark[j] = 1;
               found = 1;
               break;
            }
         }

         if (!found)
         {
            hypre_printf("Proc %3d: [%6d], %16p may have not been freed\n",
                  myid, i, hypre_memory_tracker[i]._ptr );
         }
      }
   }

   hypre_TFree(mark, HYPRE_MEMORY_HOST);

   fclose(file);
#endif
   return ierr;
}

/******************************************************************************
 *
 * Memory Pool
 *
 *****************************************************************************/
HYPRE_Int
hypre_SetCubMemPoolSize(hypre_uint cub_bin_growth,
                        hypre_uint cub_min_bin,
                        hypre_uint cub_max_bin,
                        size_t     cub_max_cached_bytes)
{
   HYPRE_Int ierr = 0;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#ifdef HYPRE_USING_CUB_ALLOCATOR
   if (!hypre_handle)
   {
      return -1;
   }

   hypre_handle->cub_bin_growth       = cub_bin_growth;
   hypre_handle->cub_min_bin          = cub_min_bin;
   hypre_handle->cub_max_bin          = cub_max_bin;
   hypre_handle->cub_max_cached_bytes = cub_max_cached_bytes;

   // RL: TODO
   if (hypre_handle->cub_dev_allocator)
   {
      hypre_handle->cub_dev_allocator->SetMaxCachedBytes(cub_max_cached_bytes);
   }

   if (hypre_handle->cub_um_allocator)
   {
      hypre_handle->cub_um_allocator->SetMaxCachedBytes(cub_max_cached_bytes);
   }
#endif
#endif

   return ierr;
}
