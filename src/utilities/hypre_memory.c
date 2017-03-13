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
#include "gpgpu.h"
#include "hypre_nvtx.h"
#include "gpuMem.h"
#ifdef HYPRE_USE_UMALLOC
#undef HYPRE_USE_UMALLOC
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

char *
hypre_MAlloc( size_t size )
{
   void *ptr;

   if (size > 0)
   {
#ifdef HYPRE_USE_UMALLOC
      HYPRE_Int threadid = hypre_GetThreadID();
#ifdef HYPRE_USE_MANAGED
      printf("ERROR HYPRE_USE_UMALLOC AND HYPRE_USE_MANAGED are mutually exclusive\n");
#endif
      ptr = _umalloc_(size);
#elif HYPRE_USE_MANAGED
      gpuErrchk( cudaMallocManaged(&ptr,size,CUDAMEMATTACHTYPE) );
      //gpuErrchk(cudaStreamAttachMemAsync(getstream(4), ptr,0,cudaMemAttachHost));
      mempush(ptr,size,0);
#else
      ptr = malloc(size);
#endif

#if 1
      if (ptr == NULL)
      {
        hypre_OutOfMemory(size);
      }
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

char *
hypre_CAlloc( size_t count,
              size_t elt_size )
{
   void   *ptr;
   size_t  size = count*elt_size;

   if (size > 0)
   {
#ifdef HYPRE_USE_UMALLOC
#ifdef HYPRE_USE_MANAGED
      printf("ERROR HYPRE_USE_UMALLOC AND HYPRE_USE_MANAGED are mutually exclusive\n");
#endif
      HYPRE_Int threadid = hypre_GetThreadID();

      ptr = _ucalloc_(count, elt_size);
#elif HYPRE_USE_MANAGED
      gpuErrchk( cudaMallocManaged(&ptr,size,CUDAMEMATTACHTYPE) );
      //gpuErrchk(cudaStreamAttachMemAsync(getstream(4), ptr,0,cudaMemAttachHost));
      memset(ptr,0,count*elt_size);
      mempush(ptr,size,0);
#else
      ptr = calloc(count, elt_size);
#endif

#if 1
      if (ptr == NULL)
      {
        hypre_OutOfMemory(size);
      }
#endif
   }
   else
   {
      ptr = NULL;
   }

   return(char*) ptr;
}

/*--------------------------------------------------------------------------
 * hypre_ReAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_ReAlloc( char   *ptr,
               size_t  size )
{
#ifdef HYPRE_USE_UMALLOC
   if (ptr == NULL)
   {
      ptr = hypre_MAlloc(size);
   }
   else if (size == 0)
   {
      hypre_Free(ptr);
   }
   else
   {
      HYPRE_Int threadid = hypre_GetThreadID();
      ptr = (char*)_urealloc_(ptr, size);
   }
#elif HYPRE_USE_MANAGED
   if (ptr == NULL)
   {

      ptr = hypre_MAlloc(size);
      //fprintf(stderr,"REalloc doing amalloc = %p\n",ptr);
   }
   else if (size == 0)
   {
     //printf("BOOM\n");
     hypre_Free(ptr);
     return NULL;
   }
   else
   {
     void *nptr = hypre_MAlloc(size);
     size_t old_size=mempush((void*)ptr,0,0);
     if (size>old_size)
       memcpy(nptr,ptr,old_size);
     else
       memcpy(nptr,ptr,size);
     //printf("Realloced %zu -> %zu(%zu), old ptr = %p, new ptr = %p \n",old_size,size,mempush(nptr,0,0),ptr,nptr);
     hypre_Free(ptr);
     ptr=nptr;
   }
#else
   if (ptr == NULL)
   {
     //printf("BOOM\n");
	   ptr = (char*)malloc(size);
   }
   else
   {
     //printf("BOOM\n");
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
hypre_Free( char *ptr )
{
   if (ptr)
   {
#ifdef HYPRE_USE_UMALLOC
      HYPRE_Int threadid = hypre_GetThreadID();

      _ufree_(ptr);
#elif HYPRE_USE_MANAGED
      //size_t size=mempush(ptr,0,0);
      mempush(ptr,0,1);
      cudaSafeFree(ptr);
      //gpuErrchk(cudaFree((void*)ptr));
#else
      free(ptr);
#endif
   }
}
