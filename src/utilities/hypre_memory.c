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

#define HYPRE_USE_MANAGED_SCALABLE 1
#include "_hypre_utilities.h"
#ifdef HYPRE_USE_UMALLOC
#undef HYPRE_USE_UMALLOC
#endif

/* global variables for OMP 45 */
#if defined(HYPRE_USE_OMP45)
HYPRE_Int hypre__global_offload = 0;
HYPRE_Int hypre__offload_device_num;

/* stats */
HYPRE_Long hypre__target_allc_count = 0;
HYPRE_Long hypre__target_free_count = 0;
HYPRE_Long hypre__target_allc_bytes = 0;
HYPRE_Long hypre__target_free_bytes = 0;

HYPRE_Long hypre__target_htod_count = 0;
HYPRE_Long hypre__target_dtoh_count = 0;
HYPRE_Long hypre__target_htod_bytes = 0;
HYPRE_Long hypre__target_dtoh_bytes = 0;

#endif

/******************************************************************************
 *
 * Standard routines
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_OutOfMemory
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_OutOfMemory( size_t size )
{
   hypre_printf("Out of memory trying to allocate %d bytes\n", (HYPRE_Int) size);
   fflush(stdout);

   hypre_error(HYPRE_ERROR_MEMORY);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_MAlloc
 *--------------------------------------------------------------------------*/

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
  patpush(ret,ss);
  return ret;
}
#endif

char *
hypre_MAlloc( size_t size , HYPRE_Int location)
{
   void *ptr;

   if (size > 0)
   {
#if defined(HYPRE_USE_GPU)
      PUSH_RANGE_PAYLOAD("MALLOC",2,size);
#endif
      if (location==HYPRE_MEMORY_DEVICE)
      {
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC)
	/*ptr = omp_target_alloc(size+sizeof(size_t)*MEM_PAD_LEN, hypre__offload_device_num);*/
         hypre_CheckErrorDevice( cudaMalloc(&ptr,size+sizeof(size_t)*MEM_PAD_LEN) );
         size_t *sp=(size_t*)ptr;
         cudaMemset(ptr,size,sizeof(size_t)*MEM_PAD_LEN);
         ptr=(void*)(&sp[MEM_PAD_LEN]);
#elif defined(HYPRE_USE_OMP45) /*else HYPRE_USE_OMP45_TARGET_ALLOC */
         void *ptr_alloc = malloc(size + HYPRE_OMP45_SZE_PAD);
         char *ptr_inuse = (char *) ptr_alloc + HYPRE_OMP45_SZE_PAD;
         size_t size_inuse = size;
         ((size_t *) ptr_alloc)[0] = size_inuse;
         hypre_omp45_offload(hypre__offload_device_num, ptr_inuse, char, 0, size_inuse, "enter", "alloc");
         ptr = (void *) ptr_inuse;
#elif defined(HYPRE_MEMORY_GPU) /*else HYPRE_USE_OMP45_TARGET_ALLOC */
         hypre_CheckErrorDevice( cudaMalloc(&ptr,size+sizeof(size_t)*MEM_PAD_LEN) );
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

   return (char*)ptr;
}

/*--------------------------------------------------------------------------
 * hypre_CAlloc
 *--------------------------------------------------------------------------*/
#if defined(TRACK_MEMORY_ALLOCATIONS)
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
  patpush(ret,ss);
  return ret;
}
#endif

char *
hypre_CAlloc( size_t count, 
              size_t elt_size,
              HYPRE_Int location)
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
#if defined(HYPRE_USE_OMP45_TARGET_ALLOC) || defined(HYPRE_MEMORY_GPU)
         ptr=(void*)hypre_MAlloc(size,location);
         cudaMemset(ptr,0,size);
#elif defined(HYPRE_USE_OMP45) /* else HYPRE_USE_OMP45_TARGET_ALLOC */
         void *ptr_alloc = calloc(count + HYPRE_OMP45_CNT_PAD(elt_size), elt_size);
         char *ptr_inuse = (char *) ptr_alloc + HYPRE_OMP45_SZE_PAD;
         size_t size_inuse = elt_size * count;
         ((size_t *) ptr_alloc)[0] = size_inuse;
         hypre_omp45_offload(hypre__offload_device_num, ptr_inuse, char, 0, size_inuse, "enter", "to");
         ptr = (void*) ptr_inuse;
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
         ptr=(void*)hypre_MAlloc(size, location);
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

   return(char*) ptr;
}

#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_MANAGED) 
size_t memsize(const void *ptr){
   return ((size_t*)ptr)[-MEM_PAD_LEN];
}
#else
size_t memsize(const void *ptr){
  return 0;
}
#endif
/*--------------------------------------------------------------------------
 * hypre_ReAlloc
 *--------------------------------------------------------------------------*/

#if defined(TRACK_MEMORY_ALLOCATIONS)
char *
hypre_ReAllocIns( char *ptr, size_t size , HYPRE_Int location,char *file, HYPRE_Int line)
{
  char *ret = hypre_ReAlloc(ptr,size,location);
  //printf("%s %d %d %p\n",file,line,location,ret);
  pattr_t *ss=(pattr_t*)hypre_MAlloc(sizeof(pattr_t),HYPRE_MEMORY_HOST);
  ss->file=file;
  ss->line=line;
  ss->type=location;
  patpush(ret,ss);
  return ret;
}
#endif

char *
hypre_ReAlloc( char   *ptr, 
               size_t  size,
               HYPRE_Int location)
{
#ifdef HYPRE_USE_UMALLOC
   if (ptr == NULL)
   {
      ptr = hypre_MAlloc(size, location);
   }
   else if (size == 0)
   {
      hypre_Free(ptr, location);
   }
   else
   {
      HYPRE_Int threadid = hypre_GetThreadID();
      ptr = (char*)_urealloc_(ptr, size);
   }
#elif HYPRE_USE_MANAGED
   if (ptr == NULL)
   {
      ptr = hypre_MAlloc(size, location);
   }
   else if (size == 0)
   {
      hypre_Free(ptr, location);
      return NULL;
   }
   else
   {
      void *nptr = hypre_MAlloc(size, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USE_MANAGED_SCALABLE
      size_t old_size=memsize((void*)ptr);
#else
      size_t old_size=mempush((void*)ptr,0,0);
#endif
      if (size > old_size)
      {
         hypre_Memcpy(nptr,ptr,old_size,location,location);
      }
      else
      {
         hypre_Memcpy(nptr,ptr,size,location,location);
      }
      hypre_Free(ptr, location);
      ptr=(char*) nptr;
   }
#else
   if (ptr == NULL)
   {
      ptr = (char*)malloc(size);
   }
   else
   {
      ptr = (char*)realloc(ptr, size);
   }
#endif

#if 1
   if ((ptr == NULL) && (size > 0))
   {
      hypre_OutOfMemory(size);
   }
#endif

   return ptr;
}


/*--------------------------------------------------------------------------
 * hypre_Free
 *--------------------------------------------------------------------------*/

void
hypre_Free( char *ptr ,
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
      //ASSERT_HOST(ptr);
      free(ptr);
#endif /*end HYPRE_USE_OMP45_TARGET_ALLOC */
     }
     else if (location==HYPRE_MEMORY_SHARED)
     {
#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_MANAGED) || defined(HYPRE_USE_OMP45)
        cudaSafeFree(ptr,MEM_PAD_LEN);
#if defined(TRACK_MEMORY_ALLOCATIONS)
	ASSERT_HOST(ptr);
#endif

#else
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
hypre_Memcpy( char *dst,
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
            memcpy( dst, src, size);
#elif defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_OMP45_TARGET_ALLOC)
            cudaMemcpy( dst, src, size, cudaMemcpyDeviceToDevice);
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
         hypre_printf("Wrong memory location.\n");
         fflush(stdout);
         hypre_error(HYPRE_ERROR_MEMORY);
      }
   }
}

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
 * hypre_MAllocPinned
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


