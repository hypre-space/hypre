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

/* global variables for OMP 45 */
#if defined(HYPRE_USE_OMP45)
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

/* Memory Environment */
/* Host memory only */
#define HOST_MEM_ONLY 0
/* Device memory without unified memory */
#define DEVC_MEM_WOUM 1
/* Device memory with    unified memory */
#define DEVC_MEM_WTUM 2

#if defined(HYPRE_USE_MANAGED)
#define HYPRE_MEMORY_ENV DEVC_MEM_WTUM
#elif defined(HYPRE_USE_CUDA) || defined(HYPRE_USE_OMP45)
#define HYPRE_MEMORY_ENV DEVC_MEM_WOUM
#else
#define HYPRE_MEMORY_ENV HOST_MEM_ONLY
#endif

/* if  true, DeviceMalloc is always device-only malloc no matter what UM is
 * if false, DeviceMalloc becomes UM malloc when with UM */
#if defined(HYPRE_USE_MANAGED) && !defined(HYPRE_USE_CUDA) && !defined(HYPRE_USE_OMP45)
#define DEVICE_ALWARYS_DEVICE 0
#else
#define DEVICE_ALWARYS_DEVICE 1
#endif

/******************************************************************************
 *
 * Helper routines
 *
 *****************************************************************************/
/*--------------------------------------------------------------------------
 * hypre_RedefMemLocation
 *--------------------------------------------------------------------------*/
static inline HYPRE_Int hypre_RedefMemLocation(HYPRE_Int location)
{
#if HYPRE_MEMORY_ENV == HOST_MEM_ONLY
   return HYPRE_MEMORY_HOST;
#else
   if (location == HYPRE_MEMORY_HOST)
   {
      return HYPRE_MEMORY_HOST;
   }

   if (location == HYPRE_MEMORY_DEVICE)
   {
#if !DEVICE_ALWARYS_DEVICE && HYPRE_MEMORY_ENV == DEVC_MEM_WTUM
      return HYPRE_MEMORY_SHARED;
#else
      return HYPRE_MEMORY_DEVICE;
#endif
   }

   if (location == HYPRE_MEMORY_SHARED)
   {
#if HYPRE_MEMORY_ENV == DEVC_MEM_WTUM
      return HYPRE_MEMORY_SHARED;
#else
      return HYPRE_MEMORY_DEVICE;
#endif
   }

   if (location == HYPRE_MEMORY_HOST_PINNED)
   {
      return HYPRE_MEMORY_HOST_PINNED;
   }

   return HYPRE_MEMORY_UNSET;
#endif
}

/*--------------------------------------------------------------------------
 * hypre_OutOfMemory
 *--------------------------------------------------------------------------*/
static inline void
hypre_OutOfMemory(size_t size)
{
   hypre_printf("Out of memory trying to allocate %ld bytes\n", size);
   fflush(stdout);
   hypre_error(HYPRE_ERROR_MEMORY);
}
   
static inline void
hypre_WrongMemoryLocation()
{
   hypre_printf("Wrong memory location! ", 
                "Only HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE and HYPRE_MEMORY_SHARED are avaible\n");
   fflush(stdout);
   hypre_error(HYPRE_ERROR_MEMORY);
}

/*--------------------------------------------------------------------------
 * hypre_GetPadMemsize: 
 * Device/HostPinned malloc stores the size in bytes at beginning size_t
 *--------------------------------------------------------------------------*/
static inline size_t 
hypre_GetPadMemsize(void *ptr, HYPRE_Int location)
{
#if HYPRE_MEMORY_ENV != HOST_MEM_ONLY
   /* no stored size for host memory */
   if (location == HYPRE_MEMORY_HOST)
   {
      return 0;
   }

   size_t *sp = (size_t*) ptr - MEM_PAD_LEN;

   if (location == HYPRE_MEMORY_DEVICE)
   {
#if !defined(HYPRE_USE_OMP45_TARGET_ALLOC) && defined(HYPRE_USE_OMP45)
      return *sp;
#else
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

#endif
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
#if HYPRE_MEMORY_ENV != HOST_MEM_ONLY
   /* without UM, device alloc */
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC)
   /* omp target alloc */
   ptr = omp_target_alloc(size + sizeof(size_t)*MEM_PAD_LEN, hypre__offload_device_num);
   size_t *sp = (size_t*) ptr;
#pragma omp target is_device_ptr(sp)
   {
      sp[0] = size;
   }
   ptr = (void*) (&sp[MEM_PAD_LEN]);
#elif defined(HYPRE_USE_OMP45)
   /* omp target map */
   ptr = malloc(size + sizeof(size_t)*MEM_PAD_LEN);
   size_t *sp = (size_t*) ptr;
   sp[0] = size;
   ptr = (void *) (&sp[MEM_PAD_LEN]);
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, size, "enter", "alloc");
#else
   /* cudaMalloc */
   hypre_CheckErrorDevice( cudaMalloc(&ptr, size + sizeof(size_t)*MEM_PAD_LEN) );
   hypre_CheckErrorDevice( cudaDeviceSynchronize() );
   hypre_Memcpy(ptr, &size, sizeof(size_t), HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   size_t *sp = (size_t*) ptr;
   ptr = (void*) (&sp[MEM_PAD_LEN]);
#endif

   /* after device alloc, memset to 0 */
   if (zeroinit)
   {
      hypre_Memset(ptr, 0, size, HYPRE_MEMORY_DEVICE);
   }
#endif
   return ptr;
}

static inline void *
hypre_UnifiedMalloc(size_t size, HYPRE_Int zeroinit)
{
   void *ptr = NULL;
#if HYPRE_MEMORY_ENV != HOST_MEM_ONLY
   /* with UM, managed memory alloc */
   hypre_CheckErrorDevice( cudaMallocManaged(&ptr, size + sizeof(size_t)*MEM_PAD_LEN, CUDAMEMATTACHTYPE) );
   size_t *sp = (size_t*) ptr;
   sp[0] = size;
   ptr = (void*) (&sp[MEM_PAD_LEN]);

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
#if HYPRE_MEMORY_ENV != HOST_MEM_ONLY
   /* TODO which one of the following two? */ 
   /* hypre_CheckErrorDevice( cudaHostAlloc(&ptr,size + sizeof(size_t)*MEM_PAD_LEN,
                                            cudaHostAllocMapped)); */
   hypre_CheckErrorDevice( cudaMallocHost(&ptr, size + sizeof(size_t)*MEM_PAD_LEN) );
   size_t *sp = (size_t*) ptr;
   sp[0] = size;
   ptr = (void*) (&sp[MEM_PAD_LEN]);

   /* after host alloc, memset to 0 */
   if (zeroinit)
   {
      hypre_Memset(ptr, 0, size, HYPRE_MEMORY_HOST_PINNED);
   }

#endif
   return ptr;
}

static inline void *
hypre_MAllocWithInit(size_t size, HYPRE_Int zeroinit, HYPRE_Int location)
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
   }

   return ptr;
}

void *
hypre_MAlloc(size_t size, HYPRE_Int location)
{
   return hypre_MAllocWithInit(size, 0, location);
}

void *
hypre_CAlloc( size_t count, size_t elt_size, HYPRE_Int location)
{
   size_t size = count * elt_size;
   return hypre_MAllocWithInit(size, 1, location);
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
#if HYPRE_MEMORY_ENV != HOST_MEM_ONLY
   /* without UM, device free */
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC)
   size_t *sp = (size_t *) ptr;
   ptr = (void *) (&sp[-MEM_PAD_LEN]);
   omp_target_free(ptr, hypre__offload_device_num);
#elif defined(HYPRE_USE_OMP45)
   size_t size = ((size_t *) ptr)[-MEM_PAD_LEN];
   HYPRE_OMPOffload(hypre__offload_device_num, ptr, size, "exit", "delete");
#else
   /* cudaFree((size_t *) ptr - MEM_PAD_LEN); */
   cudaSafeFree(ptr, MEM_PAD_LEN);
#endif
#endif
}

static inline void 
hypre_UnifiedFree(void *ptr)
{
#if HYPRE_MEMORY_ENV != HOST_MEM_ONLY
   /* with UM, managed memory free */
   /* cudaFree((size_t *) ptr - MEM_PAD_LEN); */
   cudaSafeFree(ptr, MEM_PAD_LEN);
#endif
}

static inline void 
hypre_HostPinnedFree(void *ptr)
{
#if HYPRE_MEMORY_ENV != HOST_MEM_ONLY
   /* page-locked memory on the host */
   /* cudaFreeHost((size_t *) ptr - MEM_PAD_LEN); */
   cudaSafeFree(ptr, MEM_PAD_LEN);
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
    *              (i.e, with Device involved)
    */
   if (loc_dst != HYPRE_MEMORY_DEVICE && loc_src != HYPRE_MEMORY_DEVICE)
   {
      memcpy(dst, src, size);
      return;
   }

#if HYPRE_MEMORY_ENV != HOST_MEM_ONLY
   /* 2: Shared <-- Device, Device <-- Shared */
   if (loc_dst == HYPRE_MEMORY_SHARED || loc_src == HYPRE_MEMORY_SHARED)
   {
      cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
      return;
   }

   /* 2: Device <-- Host, Device <-- Pinned */
   if ( loc_dst == HYPRE_MEMORY_DEVICE && (loc_src == HYPRE_MEMORY_HOST || loc_src == HYPRE_MEMORY_HOST_PINNED) )
   {
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_device_num, hypre__offload_host_num);
#elif defined(HYPRE_USE_OMP45)
      memcpy(dst, src, size);
      HYPRE_OMPOffload(hypre__offload_device_num, dst, size, "update", "to");
#else
      cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
#endif
      return;
   }


   /* 2: Host <-- Device, Pinned <-- Device */
   if ( (loc_dst == HYPRE_MEMORY_HOST || loc_dst == HYPRE_MEMORY_HOST_PINNED) && loc_src == HYPRE_MEMORY_DEVICE )
   {
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_host_num, hypre__offload_device_num);
#elif defined(HYPRE_USE_OMP45)
      HYPRE_OMPOffload(hypre__offload_device_num, src, size, "update", "from");
      memcpy(dst, src, size);
#else
      cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost);
#endif
      return;
   }


   /* 1: Device <-- Device */
   if (loc_dst == HYPRE_MEMORY_DEVICE && loc_src == HYPRE_MEMORY_DEVICE)
   {
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, hypre__offload_device_num, hypre__offload_device_num);
#elif defined(HYPRE_USE_OMP45)
      HYPRE_OMPOffload(hypre__offload_device_num, src, size, "update", "from");
      memcpy(dst, src, size);
      HYPRE_OMPOffload(hypre__offload_device_num, dst, size, "update", "to");
#else
      cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
#endif
      return;
   }
#endif

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

#if defined(HYPRE_USE_OMP45_TARGET_ALLOC)
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
#if HYPRE_MEMORY_ENV != HOST_MEM_ONLY
      case HYPRE_MEMORY_DEVICE :
         /* memset device memory */
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC)
#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(ucptr)
         hypre_LoopBegin(num, k)
         {
            ucptr[k] = ucvalue;
         }
         hypre_LoopEnd()
#undef DEVICE_VAR
#define DEVICE_VAR
#elif defined(HYPRE_USE_OMP45)
         memset(ptr, value, num);
         HYPRE_OMPOffload(hypre__offload_device_num, ptr, num, "enter", "to");
#else
         cudaMemset(ptr, value, num);
#endif
         break;
      case HYPRE_MEMORY_SHARED :
         /* memset unified memory */
         memset(ptr, value, num);
         break;
#endif
      default :
         /* unrecognized location */         
         hypre_WrongMemoryLocation();
   }

   return ptr;
}













#if 0

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
 * OLD CODE 
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

/*--------------------------------------------------------------------------
 * hypre_MAllocPinned: TODO where to put it
 *--------------------------------------------------------------------------*/
char *
hypre_MAllocPinned( size_t size )
{
   void *ptr;

   if (size > 0)
   {
#if defined(HYPRE_USE_GPU)
     PUSH_RANGE_PAYLOAD("MALLOC",2,size);
#endif /* HYPRE_USE_GPU */
#ifdef HYPRE_USE_UMALLOC
      HYPRE_Int threadid = hypre_GetThreadID();
#ifdef HYPRE_USE_MANAGED
      printf("ERROR HYPRE_USE_UMALLOC AND HYPRE_USE_MANAGED are mutually exclusive\n");
#endif /* HYPRE_USE_MANAGED */
      ptr = _umalloc_(size);
#elif HYPRE_USE_MANAGED /*else HYPRE_USE_UMALLOC*/
#ifdef HYPRE_USE_MANAGED_SCALABLE
#ifdef HYPRE_GPU_USE_PINNED
      hypre_CheckErrorDevice( cudaHostAlloc(&ptr,size+sizeof(size_t)*MEM_PAD_LEN,cudaHostAllocMapped));
#else /* else HYPRE_GPU_USE_PINNED */
      hypre_CheckErrorDevice( cudaMallocManaged(&ptr,size+sizeof(size_t)*MEM_PAD_LEN,CUDAMEMATTACHTYPE) );
#endif /* end HYPRE_GPU_USE_PINNED */
      size_t *sp=(size_t*)ptr;
      *sp=size;
      ptr=(void*)(&sp[MEM_PAD_LEN]);
#else /* else HYPRE_USE_MANAGED_SCALABLE */
      hypre_CheckErrorDevice( cudaMallocManaged(&ptr,size,CUDAMEMATTACHTYPE) );
      mempush(ptr,size,0);
#endif /* end HYPRE_USE_MANAGED_SCALABLE */
#else /*else HYPRE_USE_UMALLOC */
      ptr = malloc(size);
#endif

#if 1
      if (ptr == NULL)
      {
        hypre_OutOfMemory(size);
      }
#endif
#if defined(HYPRE_USE_GPU)      
      POP_RANGE;
#endif
   }
   else
   {
      ptr = NULL;
   }
   return (char*)ptr;
}




void *
hypre_MAlloc(size_t size, HYPRE_Int location)
{
   void *ptr;

   if (size > 0)
   {
#if defined(HYPRE_USE_GPU)
      PUSH_RANGE_PAYLOAD("MALLOC",2,size);
#endif
      if (location == HYPRE_MEMORY_DEVICE)
      {
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC)
         /* ptr = omp_target_alloc(size+sizeof(size_t)*MEM_PAD_LEN, hypre__offload_device_num); */
         hypre_CheckErrorDevice( cudaMalloc(&ptr, size + sizeof(size_t)*MEM_PAD_LEN) );
         size_t *sp = (size_t*) ptr;
#pragma omp target is_device_ptr(sp)
         {
            sp[0] = size;
         }
         ptr = (void*) (&sp[MEM_PAD_LEN]);
#elif defined(HYPRE_USE_OMP45) /*else HYPRE_USE_OMP45_TARGET_ALLOC */
         void *ptr_alloc = malloc(size + HYPRE_OMP45_SZE_PAD);
         char *ptr_inuse = (char *) ptr_alloc + HYPRE_OMP45_SZE_PAD;
         size_t size_inuse = size;
         ((size_t *) ptr_alloc)[0] = size_inuse;
         //printf("Malloc: try to map %ld bytes\n", size_inuse);
         hypre_omp45_offload(hypre__offload_device_num, ptr_inuse, char, 0, size_inuse, "enter", "alloc");
         ptr = (void *) ptr_inuse;
#elif defined(HYPRE_MEMORY_GPU) /*else HYPRE_USE_OMP45_TARGET_ALLOC */
         hypre_CheckErrorDevice( cudaMalloc(&ptr,size+sizeof(size_t)*MEM_PAD_LEN) );
	 hypre_CheckErrorDevice(cudaDeviceSynchronize());
         size_t *sp=(size_t*)ptr;
         cudaMemset(ptr,size,sizeof(size_t)*MEM_PAD_LEN);
         ptr=(void*)(&sp[MEM_PAD_LEN]);
#elif defined(HYPRE_USE_MANAGED)
#ifdef HYPRE_USE_UMALLOC
         HYPRE_Int threadid = hypre_GetThreadID();
#ifdef HYPRE_USE_MANAGED
         printf("ERROR HYPRE_USE_UMALLOC AND HYPRE_USE_MANAGED are mutually exclusive\n");
#endif/* HYPRE_USE_MANAGED */
         ptr = _umalloc_(size);	 
#elif HYPRE_USE_MANAGED /*else HYPRE_USE_UMALLOC */
#ifdef HYPRE_USE_MANAGED_SCALABLE
         hypre_CheckErrorDevice( cudaMallocManaged(&ptr,size+sizeof(size_t)*MEM_PAD_LEN,CUDAMEMATTACHTYPE) );
         size_t *sp=(size_t*)ptr;
         *sp=size;
         ptr=(void*)(&sp[MEM_PAD_LEN]);
#else /* HYPRE_USE_MANAGED_SCALABLE */
         hypre_CheckErrorDevice( cudaMallocManaged(&ptr,size,CUDAMEMATTACHTYPE) );
         mempush(ptr,size,0);
#endif/* HYPRE_USE_MANAGED_SCALABLE */
#endif /* end HYPRE_USE_UMALLOC*/
#else /*else HYPRE_USE_OMP45_TARGET_ALLOC */
         ptr = malloc(size);
#endif /*end HYPRE_USE_OMP45_TARGET_ALLOC */
      }
      else if (location==HYPRE_MEMORY_HOST)
      {
         ptr = malloc(size);
      }
      else if (location==HYPRE_MEMORY_SHARED)
      {
#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_MANAGED) || defined(HYPRE_USE_OMP45)
	 hypre_CheckErrorDevice( cudaMallocManaged(&ptr,size+sizeof(size_t)*MEM_PAD_LEN,CUDAMEMATTACHTYPE));
         size_t *sp=(size_t*)ptr;
         *sp=size;
         ptr=(void*)(&sp[MEM_PAD_LEN]);
#else
         ptr = malloc(size);
#endif
      }
      else
      {
         hypre_printf("Wrong memory location. Only HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST, and HYPRE_MEMORY_SHARED are avaible\n");
         fflush(stdout);
         hypre_error(HYPRE_ERROR_MEMORY);
      }
#if 1
      if (ptr == NULL)
      {
         hypre_OutOfMemory(size);
      }
#endif
#if defined(HYPRE_USE_GPU)
      POP_RANGE;
#endif
   }
   else
   {
      ptr = NULL;
   }

   return (void*)ptr;
}

/*--------------------------------------------------------------------------
 * hypre_CAlloc
 *--------------------------------------------------------------------------*/
void *
hypre_CAlloc( size_t count, size_t elt_size, HYPRE_Int location)
{
   void   *ptr;
   size_t  size = count*elt_size;

   if (size > 0)
   {
#if defined(HYPRE_USE_GPU)
      PUSH_RANGE_PAYLOAD("MALLOC",4,size);
#endif
      if (location==HYPRE_MEMORY_DEVICE)
      {
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC) 
         ptr = (void*) hypre_MAlloc(size, location);
         char *char_ptr = (char *) ptr;
#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(char_ptr)
         hypre_LoopBegin(size, k)
         {
            char_ptr[k] = 0;
         }
         hypre_LoopEnd()
#undef DEVICE_VAR
#define DEVICE_VAR
         /* cudaDeviceSynchronize(); */
#elif defined(HYPRE_USE_OMP45) /* else HYPRE_USE_OMP45_TARGET_ALLOC */
         //void *ptr_alloc = calloc(count + HYPRE_OMP45_CNT_PAD(elt_size), elt_size);
         void *ptr_alloc = malloc(size + HYPRE_OMP45_SZE_PAD);
         char *ptr_inuse = (char *) ptr_alloc + HYPRE_OMP45_SZE_PAD;
         size_t size_inuse = size;
         ((size_t *) ptr_alloc)[0] = size_inuse;
         memset(ptr_inuse, 0, size_inuse);
         //printf("Calloc: try to map %ld bytes\n", size_inuse);
         hypre_omp45_offload(hypre__offload_device_num, ptr_inuse, char, 0, size_inuse, "enter", "to");
         ptr = (void*) ptr_inuse;
#elif defined(HYPRE_MEMORY_GPU)
         ptr = (void*) hypre_MAlloc(size, location);
         cudaMemset(ptr, 0, size);
#elif defined(HYPRE_USE_MANAGED)/* else HYPRE_USE_OMP45_TARGET_ALLOC */
#ifdef HYPRE_USE_UMALLOC
#ifdef HYPRE_USE_MANAGED
         printf("ERROR HYPRE_USE_UMALLOC AND HYPRE_USE_MANAGED are mutually exclusive\n");
#endif /* end HYPRE_USE_MANAGED */
         HYPRE_Int threadid = hypre_GetThreadID();
         ptr = _ucalloc_(count, elt_size);     
#elif HYPRE_USE_MANAGED /*else HYPRE_USE_UMALLOC */
#ifdef HYPRE_USE_MANAGED_SCALABLE
         ptr=(void*)hypre_MAlloc(size, location);
         memset(ptr,0,count*elt_size);
#else /* else HYPRE_USE_MANAGED_SCALABLE */
         hypre_CheckErrorDevice( cudaMallocManaged(&ptr,size,CUDAMEMATTACHTYPE) );
         memset(ptr,0,count*elt_size);
         mempush(ptr,size,0);
#endif /* end HYPRE_USE_MANAGED_SCALABLE */
#endif/*end HYPRE_USE_UMALLOC */
#else /* else HYPRE_USE_OMP45_TARGET_ALLOC */
         ptr = calloc(count, elt_size);
#endif /* end HYPRE_USE_OMP45_TARGET_ALLOC */
      }
      else if (location==HYPRE_MEMORY_HOST)
      {
         ptr = calloc(count, elt_size);
      }
      else if (location==HYPRE_MEMORY_SHARED)
      {
#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_MANAGED) || defined(HYPRE_USE_OMP45)
         ptr = (void*) hypre_MAlloc(size, location);
         memset(ptr,0,count*elt_size);
#else
         ptr = calloc(count, elt_size);
#endif
      }
      else
      {
         hypre_printf("Wrong memory location. Only HYPRE_LOCATION_DEVICE and HYPRE_LOCATION_HOST are avaible\n");
         fflush(stdout);
         hypre_error(HYPRE_ERROR_MEMORY);
      }

#if 1
      if (ptr == NULL)
      {
         hypre_OutOfMemory(size);
      }
#endif
#if defined(HYPRE_USE_GPU)
      POP_RANGE;
#endif
   }
   else
   {
      ptr = NULL;
   }

   return(void*) ptr;
}

/*--------------------------------------------------------------------------
 * hypre_ReAlloc
 *--------------------------------------------------------------------------*/
void *
hypre_ReAlloc( void *ptr, size_t size, HYPRE_Int location)
{
   if (size == 0)
   {
      hypre_Free(ptr, location);
      ptr = NULL;
   } 
   else if (ptr == NULL)
   {
      ptr = hypre_MAlloc(size, location);
   }
   else if (location == HYPRE_MEMORY_DEVICE)
   {
      // TODO: How to do it for NONUNIFIED DEVICE memory
#if 0
#if defined(HYPRE_USE_MANAGED)
      void *new_ptr = hypre_MAlloc(size, location);
#ifdef HYPRE_USE_MANAGED_SCALABLE
      size_t old_size = memsize((void*)ptr);
#else
      size_t old_size = mempush((void*)ptr, 0, 0);
#endif
      size_t smaller_size = size > old_size ? old_size : size;
      hypre_Memcpy(new_ptr, ptr, smaller_size, location, location);
#else
      ptr = (char*) realloc(ptr, size);
#endif
#else
      hypre_printf("hypre error: ReAlloc for device memory has not been implemented\n");
      exit(0);
#endif
   }
   else if (location == HYPRE_MEMORY_HOST)
   {
      ptr = (char*)realloc(ptr, size);
   }
   else if (location == HYPRE_MEMORY_SHARED)
   {
#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_MANAGED) || defined(HYPRE_USE_OMP45)
      void *new_ptr = hypre_MAlloc(size, location);
#ifdef HYPRE_USE_MANAGED_SCALABLE
#if defined(TRACK_MEMORY_ALLOCATIONS)
      ASSERT_MANAGED(ptr);
#endif
      size_t old_size = memsize((void*)ptr);
#else
      size_t old_size = mempush((void*)ptr, 0, 0);
#endif
      size_t smaller_size = size > old_size ? old_size : size;
      hypre_Memcpy((char*)new_ptr, ptr, smaller_size, location, location);
      hypre_Free(ptr,location);
      ptr=(char*)new_ptr;
#else
      ptr = (char*)realloc(ptr, size);
#endif
   }
   else
   {
      hypre_printf("Wrong memory location. Only HYPRE_LOCATION_DEVICE and HYPRE_LOCATION_HOST are avaible\n");
      fflush(stdout);
      hypre_error(HYPRE_ERROR_MEMORY);
   }

   if ((ptr == NULL) && (size > 0))
   {
      hypre_OutOfMemory(size);
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_Free
 *--------------------------------------------------------------------------*/

void
hypre_Free( void *ptr ,
            HYPRE_Int location)
{
   if (ptr)
   {
     if (location==HYPRE_MEMORY_DEVICE)
     {
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC)
      //omp_target_free(ptr, hypre__offload_device_num);
      cudaSafeFree(ptr,MEM_PAD_LEN);
#elif defined(HYPRE_USE_OMP45)/*else HYPRE_USE_OMP45_TARGET_ALLOC */
      char *ptr_alloc = ((char*) ptr) - HYPRE_OMP45_SZE_PAD;
      size_t size_inuse = ((size_t *) ptr_alloc)[0];
      hypre_omp45_offload(hypre__offload_device_num, ptr, char, 0, size_inuse, "exit", "delete");
      free(ptr_alloc);
#elif defined(HYPRE_MEMORY_GPU) /*else HYPRE_USE_OMP45_TARGET_ALLOC */
      cudaSafeFree(ptr,MEM_PAD_LEN);
#elif defined(HYPRE_USE_MANAGED)/*else HYPRE_USE_OMP45_TARGET_ALLOC */
#ifdef HYPRE_USE_UMALLOC
      HYPRE_Int threadid = hypre_GetThreadID();

      _ufree_(ptr);
#elif HYPRE_USE_MANAGED /* else HYPRE_USE_UMALLOC */
      //size_t size=mempush(ptr,0,0);
#ifdef HYPRE_USE_MANAGED_SCALABLE
      cudaSafeFree(ptr,MEM_PAD_LEN);
#else/* else HYPRE_USE_MANAGED_SCALABLE */
      mempush(ptr,0,1);
      cudaSafeFree(ptr,0);
#endif /* end HYPRE_USE_MANAGED_SCALABLE */
#endif /* end HYPRE_USE_UMALLOC */
#else /*else HYPRE_USE_OMP45_TARGET_ALLOC */
#ifdef TRACK_MEMORY_ALLOCATIONS
      ASSERT_HOST(ptr);
#endif
      free(ptr);
#endif /*end HYPRE_USE_OMP45_TARGET_ALLOC */
     }
     else if (location==HYPRE_MEMORY_SHARED)
     {
#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_MANAGED) || defined(HYPRE_USE_OMP45)
        cudaSafeFree(ptr,MEM_PAD_LEN);
#else
        //ASSERT_HOST(ptr);
        free(ptr);
#endif
     }
     else
     {
#if defined(TRACK_MEMORY_ALLOCATIONS)
       ASSERT_HOST(ptr);
#endif
        free(ptr);
     }
   }
}

/*--------------------------------------------------------------------------
 * hypre_Memcpy
 *--------------------------------------------------------------------------*/
void
hypre_Memcpy( void *dst,
              void *src,
              size_t size,
              HYPRE_Int locdst,
              HYPRE_Int locsrc )
{
   if (src)
   {
      if ( locdst==HYPRE_MEMORY_DEVICE && locsrc==HYPRE_MEMORY_DEVICE )
      {
         if (dst != src)
         {
#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_OMP45_TARGET_ALLOC)
            cudaMemcpy( dst, src, size, cudaMemcpyDeviceToDevice);
#elif defined(HYPRE_USE_MANAGED)
            memcpy( dst, src, size);
#elif defined(HYPRE_USE_OMP45)
            hypre_omp45_offload(hypre__offload_device_num, src, char, 0, size, "update", "from");
            memcpy(dst, src, size);
            hypre_omp45_offload(hypre__offload_device_num, dst, char, 0, size, "update", "to");
#else
            memcpy( dst, src, size);
#endif
         }
         else
         {
            dst = src;
         }
      }
      else if ( locdst==HYPRE_MEMORY_DEVICE && locsrc==HYPRE_MEMORY_HOST )
      {
#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_OMP45_TARGET_ALLOC)
         cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice);
#elif defined(HYPRE_USE_MANAGED)
         memcpy( dst, src, size);
#elif defined(HYPRE_USE_OMP45)
         memcpy(dst, src, size);
         hypre_omp45_offload(hypre__offload_device_num, dst, char, 0, size, "update", "to");
#else
         memcpy( dst, src, size);
#endif        
      }
      else if ( locdst==HYPRE_MEMORY_HOST && locsrc==HYPRE_MEMORY_DEVICE )
      {
#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_OMP45_TARGET_ALLOC)
         cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost);
#elif defined(HYPRE_USE_MANAGED)
         memcpy( dst, src, size);
#elif defined(HYPRE_USE_OMP45)
         hypre_omp45_offload(hypre__offload_device_num, src, char, 0, size, "update", "from");
         memcpy( dst, src, size);
#else
         memcpy( dst, src, size);
#endif
      }
      else if ( locdst==HYPRE_MEMORY_HOST && locsrc==HYPRE_MEMORY_HOST )
      {
         if (dst != src)
         {
            memcpy( dst, src, size);
         }
         else
         {
            dst = src;
         }
      }
      else
      {	
	/* This needs to be fixes for speed */
	if ( locdst==HYPRE_MEMORY_SHARED && locsrc==HYPRE_MEMORY_SHARED ) memcpy( dst, src, size);
	else {
         hypre_printf("Wrong memory location.\n");
         fflush(stdout);
         hypre_error(HYPRE_ERROR_MEMORY);
	}
      }
   }
}


#if defined(TRACK_MEMORY_ALLOCATIONS)
char *
hypre_MAllocIns( size_t size , HYPRE_Int location,char *file, HYPRE_Int line)
{
  char *ret = hypre_MAlloc(size,location);
  //printf("%s %d %d %p\n",file,line,location,ret);
  pattr_t *ss=(pattr_t*)hypre_MAlloc(sizeof(pattr_t),HYPRE_MEMORY_HOST);
  ss->file=file;
  ss->line=line;
  ss->type=location;
  ss->size=size;
  ss->end=(void*)(ret+size);
  patpush(ret,ss);
  return ret;
}

char *
hypre_CAllocIns( size_t count, 
              size_t elt_size,
		 HYPRE_Int location,char *file, HYPRE_Int line){
  char *ret=hypre_CAlloc(count,elt_size,location);
  //printf("%s %d %d %p\n",file,line,location,ret);
  pattr_t *ss=(pattr_t*)hypre_MAlloc(sizeof(pattr_t),HYPRE_MEMORY_HOST);
  ss->file=file;
  ss->line=line;
  ss->type=location;
  ss->size=count*elt_size;
  ss->end=(void*)(ret+ss->size);
  patpush(ret,ss);
  return ret;
}

char *
hypre_ReAllocIns( char *ptr, size_t size, HYPRE_Int location, char *file, HYPRE_Int line)
{
  char *ret = hypre_ReAlloc(ptr,size,location);
  //printf("%s %d %d %p\n",file,line,location,ret);
  pattr_t *ss=(pattr_t*)hypre_MAlloc(sizeof(pattr_t),HYPRE_MEMORY_HOST);
  ss->file=file;
  ss->line=line;
  ss->type=location;
  ss->size=size;
  ss->end=(void*)(ret+size);
  patpush(ret,ss);
  return ret;
}

#endif

/*--------------------------------------------------------------------------
 * hypre_MemcpyAsync
 *--------------------------------------------------------------------------*/
void
hypre_MemcpyAsync( char *dst,
		   char *src,
		   size_t size,
		   HYPRE_Int locdst,
		   HYPRE_Int locsrc )
{
   if (src)
   {
     if ( locdst==HYPRE_MEMORY_DEVICE && locsrc==HYPRE_MEMORY_DEVICE )
     {
        if (dst != src)
        {
#if defined(HYPRE_USE_MANAGED)
	   cudaMemcpyAsync( dst, src, size, cudaMemcpyDefault, HYPRE_STREAM(0)); 
#elif defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_OMP45_TARGET_ALLOC)
	   //cudaMemcpyAsync( dst, src, size, cudaMemcpyDeviceDevice,0);
#elif defined(HYPRE_USE_OMP45)
	   //TODO
#else
	   memcpy( dst, src, size);
#endif
	}
	else
        {
	  /* When src == dst, Prefetch the data to GPU */
#if defined(HYPRE_USE_MANAGED)
	   HYPRE_Int device = -1;
	   cudaGetDevice(&device);
	   cudaMemPrefetchAsync(src, size, device, HYPRE_STREAM(0));
#endif	   
	}
     }
     else if ( locdst==HYPRE_MEMORY_DEVICE && locsrc==HYPRE_MEMORY_HOST )
     {
#if defined(HYPRE_USE_MANAGED)
        cudaMemcpyAsync( dst, src, size, cudaMemcpyDefault, HYPRE_STREAM(0)); 
#elif defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_OMP45_TARGET_ALLOC)
	//cudaMemcpyAsync( dst, src, size, cudaMemcpyHostToDevice);
#elif defined(HYPRE_USE_OMP45)
	memcpy(dst, src, size);
	hypre_omp45_offload(hypre__offload_device_num, dst, char, 0, size, "update", "to");
#else
	memcpy( dst, src, size);
#endif
     }
     else if ( locdst==HYPRE_MEMORY_HOST && locsrc==HYPRE_MEMORY_DEVICE )
     {
#if defined(HYPRE_USE_MANAGED)
        cudaMemcpyAsync( dst, src, size, cudaMemcpyDefault, HYPRE_STREAM(0)); 
#elif defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_OMP45_TARGET_ALLOC)
	//cudaMemcpyAsync( dst, src, size, cudaMemcpyDeviceToHost);
#elif defined(HYPRE_USE_OMP45)
	hypre_omp45_offload(hypre__offload_device_num, src, char, 0, size, "update", "from");
	memcpy( dst, src, size);
#else
	memcpy( dst, src, size);
#endif
     }
     else if ( locdst==HYPRE_MEMORY_HOST && locsrc==HYPRE_MEMORY_HOST )
     {
        if (dst != src)
        {
	   memcpy( dst, src, size);
	}
	else
	{
	   dst = src;
	}
     }
     else
     {
         hypre_printf("Wrong memory location. Only HYPRE_LOCATION_DEVICE and HYPRE_LOCATION_HOST are avaible\n");
	 fflush(stdout);
	 hypre_error(HYPRE_ERROR_MEMORY);
     }
   }
}

/*--------------------------------------------------------------------------
 * hypre_MAllocHost
 *--------------------------------------------------------------------------*/
char *
hypre_MAllocHost( size_t size )
{
   void *ptr;

   if (size > 0)
   {
     ptr = malloc(size);
#if 1
      if (ptr == NULL)
      {
        hypre_OutOfMemory(size);
      }
#endif
#if defined(HYPRE_USE_GPU)    
      POP_RANGE;
#endif
   }
   else
   {
      ptr = NULL;
   }

   return (char*)ptr;
}

#endif
