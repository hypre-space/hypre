/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

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

/* global variables for device OpenMP */
#if defined(HYPRE_USING_DEVICE_OPENMP)
HYPRE_Int hypre__global_offload = 0;
HYPRE_Int hypre__offload_device_num;
HYPRE_Int hypre__offload_host_num;
/* stats */
size_t hypre__target_allc_count = 0;
size_t hypre__target_free_count = 0;
size_t hypre__target_allc_bytes = 0;
size_t hypre__target_free_bytes = 0;

size_t hypre__target_htod_count = 0;
size_t hypre__target_dtoh_count = 0;
size_t hypre__target_htod_bytes = 0;
size_t hypre__target_dtoh_bytes = 0;
#endif

/******************************************************************************
 *
 * Helper routines
 *
 *****************************************************************************/
/*--------------------------------------------------------------------------
 * hypre_RedefMemLocation
 *   Redefine location based on the selected memory model in hypre_memory.h
 *--------------------------------------------------------------------------*/
static inline HYPRE_Int hypre_RedefMemLocation(HYPRE_Int location)
{
   if (location == HYPRE_MEMORY_HOST)
   {
      return HYPRE_MEMORY_HOST_ACT;
   }

   if (location == HYPRE_MEMORY_DEVICE)
   {
      return HYPRE_MEMORY_DEVICE_ACT;
   }

   if (location == HYPRE_MEMORY_SHARED)
   {
      return HYPRE_MEMORY_SHARED_ACT;
   }

   if (location == HYPRE_MEMORY_HOST_PINNED)
   {
      return HYPRE_MEMORY_HOST_PINNED_ACT;
   }

   return HYPRE_MEMORY_UNSET;
}

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

/*--------------------------------------------------------------------------
 * hypre_GetPadMemsize:
 * Device/HostPinned malloc stores the size in bytes at the beginning size_t
 *--------------------------------------------------------------------------*/
static inline size_t
hypre_GetPadMemsize(void *ptr, HYPRE_Int location)
{
   location = hypre_RedefMemLocation(location);

   /* no stored size for host memory */
   if (location == HYPRE_MEMORY_HOST)
   {
      return 0;
   }

   size_t *sp = (size_t*) ptr - HYPRE_MEM_PAD_LEN;

   if (location == HYPRE_MEMORY_DEVICE)
   {
      /* special case for mapped device openmp; size available on host memory */
#if defined(HYPRE_DEVICE_OPENMP_MAPPED)
      return *sp;
#else
      /* copy size from device memory */
      size_t size;
      hypre_Memcpy(&size, sp, sizeof(size_t), HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      return size;
#endif
   }

   if (location == HYPRE_MEMORY_SHARED)
   {
      return *sp;
   }

   if (location == HYPRE_MEMORY_HOST_PINNED)
   {
      return *sp;
   }

   hypre_WrongMemoryLocation();

   /* no stored size for host memory */
   return 0;
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

   /* without UM, device alloc */
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
   /* omp target alloc */
   ptr = omp_target_alloc(size + sizeof(size_t)*HYPRE_MEM_PAD_LEN, hypre__offload_device_num);
   size_t *sp = (size_t*) ptr;
#pragma omp target is_device_ptr(sp)
   {
      sp[0] = size;
   }
   ptr = (void*) (&sp[HYPRE_MEM_PAD_LEN]);
#elif defined(HYPRE_USING_DEVICE_OPENMP)
   /* omp target map */
   ptr = malloc(size + sizeof(size_t)*HYPRE_MEM_PAD_LEN);
   size_t *sp = (size_t*) ptr;
   sp[0] = size;
   ptr = (void *) (&sp[HYPRE_MEM_PAD_LEN]);
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, size, "enter", "alloc");
#elif defined(HYPRE_USING_CUDA)
   /* cudaMalloc */
   hypre_CheckErrorDevice( cudaMalloc(&ptr, size + sizeof(size_t)*HYPRE_MEM_PAD_LEN) );
   hypre_CheckErrorDevice( cudaDeviceSynchronize() );
   hypre_Memcpy(ptr, &size, sizeof(size_t), HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   size_t *sp = (size_t*) ptr;
   ptr = (void*) (&sp[HYPRE_MEM_PAD_LEN]);
#endif

   /* after device alloc, memset to 0 */
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
   size_t count = size + sizeof(size_t)*HYPRE_MEM_PAD_LEN;
   /* with UM, managed memory alloc */
   hypre_CheckErrorDevice( cudaMallocManaged(&ptr, count, CUDAMEMATTACHTYPE) );
   hypre_CheckErrorDevice( cudaMemAdvise(ptr, count, cudaMemAdviseSetPreferredLocation, HYPRE_DEVICE) );
   size_t *sp = (size_t*) ptr;
   sp[0] = size;
   ptr = (void*) (&sp[HYPRE_MEM_PAD_LEN]);

   /* after UM alloc, memset to 0 */
   if (zeroinit)
   {
      hypre_Memset(ptr, 0, size, HYPRE_MEMORY_SHARED);
   }
#endif

   return ptr;
}

static inline void *
hypre_HostPinnedMalloc(size_t size, HYPRE_Int zeroinit)
{
   void *ptr = NULL;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   /* TODO which one of the following two? */
   /* hypre_CheckErrorDevice( cudaHostAlloc(&ptr,size + sizeof(size_t)*HYPRE_MEM_PAD_LEN,
                                            cudaHostAllocMapped)); */
   hypre_CheckErrorDevice( cudaMallocHost(&ptr, size + sizeof(size_t)*HYPRE_MEM_PAD_LEN) );
   size_t *sp = (size_t*) ptr;
   sp[0] = size;
   ptr = (void*) (&sp[HYPRE_MEM_PAD_LEN]);

   /* after host alloc, memset to 0 */
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

   location = hypre_RedefMemLocation(location);

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
      exit(0);
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
   /* without UM, device free */
#if defined(HYPRE_DEVICE_OPENMP_ALLOC)
   size_t *sp = (size_t *) ptr;
   ptr = (void *) (&sp[-HYPRE_MEM_PAD_LEN]);
   omp_target_free(ptr, hypre__offload_device_num);
#elif defined(HYPRE_USING_DEVICE_OPENMP)
   size_t size = ((size_t *) ptr)[-HYPRE_MEM_PAD_LEN];
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, size, "exit", "delete");
#elif defined(HYPRE_USING_CUDA)
   /* cudaFree((size_t *) ptr - HYPRE_MEM_PAD_LEN); */
   cudaSafeFree(ptr, HYPRE_MEM_PAD_LEN);
#endif
}

static inline void
hypre_UnifiedFree(void *ptr)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   /* with UM, managed memory free */
   /* cudaFree((size_t *) ptr - HYPRE_MEM_PAD_LEN); */
   cudaSafeFree(ptr, HYPRE_MEM_PAD_LEN);
#endif
}

static inline void
hypre_HostPinnedFree(void *ptr)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   /* page-locked memory on the host */
   /* cudaFreeHost((size_t *) ptr - HYPRE_MEM_PAD_LEN); */
   cudaSafeFree(ptr, HYPRE_MEM_PAD_LEN);
#endif
}

void
hypre_Free(void *ptr, HYPRE_Int location)
{
   if (!ptr)
   {
      return;
   }

   location = hypre_RedefMemLocation(location);

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
hypre_Device_Unified_HostPinned_ReAlloc(void *ptr, size_t size, HYPRE_Int location)
{
   /* device/unified/hostpinned memory realloc: malloc+copy+free */
   void *new_ptr = hypre_MAlloc(size, location);
   size_t old_size = hypre_GetPadMemsize(ptr, location);
   size_t smaller_size = size > old_size ? old_size : size;
   hypre_Memcpy(new_ptr, ptr, smaller_size, location, location);
   hypre_Free(ptr, location);

   return new_ptr;
}

void *
hypre_ReAlloc(void *ptr, size_t size, HYPRE_Int location)
{
   location = hypre_RedefMemLocation(location);

   if (size == 0)
   {
      hypre_Free(ptr, location);
      return NULL;
   }

   if (ptr == NULL)
   {
      return hypre_MAlloc(size, location);
   }

   switch (location)
   {
      case HYPRE_MEMORY_HOST :
         /* realloc cpu memory */
         ptr = hypre_HostReAlloc(ptr, size);
         break;
      case HYPRE_MEMORY_DEVICE :
         /* realloc device memory */
      case HYPRE_MEMORY_SHARED :
         /* realloc unified memory */
      case HYPRE_MEMORY_HOST_PINNED :
         /* realloc host pinned memory */
         ptr = hypre_Device_Unified_HostPinned_ReAlloc(ptr, size, location);
         break;
      default :
         /* unrecognized location */
         hypre_WrongMemoryLocation();
   }

   if (!ptr)
   {
      hypre_OutOfMemory(size);
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
      return;
   }

   loc_dst = hypre_RedefMemLocation(loc_dst);
   loc_src = hypre_RedefMemLocation(loc_src);

   /* 4 x 4 = 16 cases = 9 + 2 + 2 + 2 + 1 */
   /* 9: Host   <-- Host, Host   <-- Shared, Host   <-- Pinned,
    *    Shared <-- Host, Shared <-- Shared, Shared <-- Pinned,
    *    Pinned <-- Host, Pinned <-- Shared, Pinned <-- Pinned.
    *              (i.e, without Device involved)
    */
   if (loc_dst != HYPRE_MEMORY_DEVICE && loc_src != HYPRE_MEMORY_DEVICE)
   {
      memcpy(dst, src, size);
      return;
   }

   /* 2: Shared <-- Device, Device <-- Shared */
   if (loc_dst == HYPRE_MEMORY_SHARED || loc_src == HYPRE_MEMORY_SHARED)
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
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
      cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
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
      cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost);
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
      cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
#endif
      return;
   }

   hypre_WrongMemoryLocation();
}

/*--------------------------------------------------------------------------
 * hypre_Memset
 * "Sets the first num bytes of the block of memory pointed by ptr to the specified value
 * (*** interpreted as an unsigned char ***)"
 *--------------------------------------------------------------------------*/
void *
hypre_Memset(void *ptr, HYPRE_Int value, size_t num, HYPRE_Int location)
{
   if (ptr == NULL || num == 0)
   {
      return ptr;
   }

   location = hypre_RedefMemLocation(location);

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
         cudaMemset(ptr, value, num);
#endif
         break;
      case HYPRE_MEMORY_SHARED :
         /* memset unified memory */
         memset(ptr, value, num);
         break;
      default :
         /* unrecognized location */
         hypre_WrongMemoryLocation();
   }

   return ptr;
}

