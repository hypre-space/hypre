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
#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

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
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
   unsigned char *ucptr   = (unsigned char *) ptr;
   unsigned char  ucvalue = (unsigned char)   value;
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
}

static inline void
hypre_UnifiedMemset(void *ptr, HYPRE_Int value, size_t num)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );
#endif
}

/*--------------------------------------------------------------------------
 * Memprefetch
 *--------------------------------------------------------------------------*/
static inline void
hypre_UnifiedMemPrefetch(void *ptr, size_t size, hypre_MemoryLocation location)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#ifdef HYPRE_DEBUG
   hypre_MemoryLocation tmp;
   hypre_GetPointerLocation(ptr, &tmp);
   /* do not use hypre_assert, which has alloc and free;
    * will create an endless loop otherwise */
   assert(hypre_MEMORY_UNIFIED == tmp);
#endif

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
}

/*--------------------------------------------------------------------------
 * Malloc
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
   HYPRE_CUDA_CALL( hypre_CachingMallocDevice(&ptr, size) );
#else
   HYPRE_CUDA_CALL( cudaMalloc(&ptr, size) );
#endif
   /* HYPRE_CUDA_CALL( cudaDeviceSynchronize() ); */
#endif

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

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#if defined(HYPRE_USING_CUB_ALLOCATOR)
   HYPRE_CUDA_CALL( hypre_CachingMallocManaged(&ptr, size) );
#else
   HYPRE_CUDA_CALL( cudaMallocManaged(&ptr, size, cudaMemAttachGlobal) );
#endif
   //HYPRE_CUDA_CALL( cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation,
   //                               hypre_HandleCudaDevice(hypre_handle())) );
   /* prefecth to device */
   hypre_UnifiedMemPrefetch(ptr, size, hypre_MEMORY_DEVICE);

   if (zeroinit)
   {
      hypre_UnifiedMemset(ptr, 0, size);
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
      hypre_HostMemset(ptr, 0, size);
   }
#endif

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
   HYPRE_CUDA_CALL( hypre_CachingFreeDevice(ptr) );
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
   HYPRE_CUDA_CALL( hypre_CachingFreeManaged(ptr) );
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
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) );
#endif
      return;
   }


   /* 2: UVM <-- Host, UVM <-- Pinned */
   if (loc_dst == hypre_MEMORY_UNIFIED)
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
#endif
      return;
   }


   /* 2: Host <-- UVM, Pinned <-- UVM */
   if (loc_src == hypre_MEMORY_UNIFIED)
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) );
#endif
      return;
   }


   /* 2: Device <-- Host, Device <-- Pinned */
   if ( loc_dst == hypre_MEMORY_DEVICE && (loc_src == hypre_MEMORY_HOST || loc_src == hypre_MEMORY_HOST_PINNED) )
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
   if ( (loc_dst == hypre_MEMORY_HOST || loc_dst == hypre_MEMORY_HOST_PINNED) && loc_src == hypre_MEMORY_DEVICE )
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
   if (loc_dst == hypre_MEMORY_DEVICE && loc_src == hypre_MEMORY_DEVICE)
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
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
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
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
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

   ptr = realloc(ptr, size);

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

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   struct cudaPointerAttributes attr;
   *memory_location = hypre_MEMORY_UNDEFINED;

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
#endif

#else /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP) */
   *memory_location = hypre_MEMORY_HOST;
#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * Memory tracker
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------*
 * Memory Pool
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetCubMemPoolSize(hypre_uint cub_bin_growth,
                        hypre_uint cub_min_bin,
                        hypre_uint cub_max_bin,
                        size_t     cub_max_cached_bytes)
{
   HYPRE_Int ierr = 0;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#ifdef HYPRE_USING_CUB_ALLOCATOR
   hypre_HandleCubBinGrowth(hypre_handle())      = cub_bin_growth;
   hypre_HandleCubMinBin(hypre_handle())         = cub_min_bin;
   hypre_HandleCubMaxBin(hypre_handle())         = cub_max_bin;
   hypre_HandleCubMaxCachedBytes(hypre_handle()) = cub_max_cached_bytes;

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

   return ierr;
}

#ifdef HYPRE_USING_CUB_ALLOCATOR
cudaError_t
hypre_CachingMallocDevice(void **ptr, size_t nbytes)
{
   if (!hypre_HandleCubDevAllocator(hypre_handle()))
   {
      hypre_HandleCubDevAllocator(hypre_handle()) =
         new hypre_cub_CachingDeviceAllocator( hypre_HandleCubBinGrowth(hypre_handle()),
                                               hypre_HandleCubMinBin(hypre_handle()),
                                               hypre_HandleCubMaxBin(hypre_handle()),
                                               hypre_HandleCubMaxCachedBytes(hypre_handle()),
                                               false,
                                               false,
                                               false );
   }

   return hypre_HandleCubDevAllocator(hypre_handle())->DeviceAllocate(ptr, nbytes);
}

cudaError_t
hypre_CachingFreeDevice(void *ptr)
{
   return hypre_HandleCubDevAllocator(hypre_handle())->DeviceFree(ptr);
}

cudaError_t
hypre_CachingMallocManaged(void **ptr, size_t nbytes)
{
   if (!hypre_HandleCubUvmAllocator(hypre_handle()))
   {
      hypre_HandleCubUvmAllocator(hypre_handle()) =
         new hypre_cub_CachingDeviceAllocator( hypre_HandleCubBinGrowth(hypre_handle()),
                                               hypre_HandleCubMinBin(hypre_handle()),
                                               hypre_HandleCubMaxBin(hypre_handle()),
                                               hypre_HandleCubMaxCachedBytes(hypre_handle()),
                                               false,
                                               false,
                                               true );
   }

   return hypre_HandleCubUvmAllocator(hypre_handle())->DeviceAllocate(ptr, nbytes);
}

cudaError_t
hypre_CachingFreeManaged(void *ptr)
{
   return hypre_HandleCubUvmAllocator(hypre_handle())->DeviceFree(ptr);
}

void
hypre_CudaDataCubCachingAllocatorDestroy(hypre_CudaData *data)
{
   delete hypre_CudaDataCubDevAllocator(data);
   delete hypre_CudaDataCubUvmAllocator(data);
}

#endif // #ifdef HYPRE_USING_CUB_ALLOCATOR

