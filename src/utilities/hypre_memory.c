/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Memory management utilities
 *
 *****************************************************************************/

#include "_hypre_utilities.h"

#ifdef HYPRE_USE_PTHREADS
#include "threading.h"

#ifdef HYPRE_USE_UMALLOC
#include "umalloc_local.h"

#define _umalloc_(size) (threadid == hypre_NumThreads) ? \
                        (char *) malloc(size) : \
                        (char *) _umalloc(_uparam[threadid].myheap, size)
#define _ucalloc_(count, size) (threadid == hypre_NumThreads) ? \
                               (char *) calloc(count, size) : \
                               (char *) _ucalloc(_uparam[threadid].myheap,\
                                                 count, size)
#define _urealloc_(ptr, size) (threadid == hypre_NumThreads) ? \
                              (char *) realloc(ptr, size) : \
                              (char *) _urealloc(ptr, size)
#define _ufree_(ptr)          (threadid == hypre_NumThreads) ? \
                              free(ptr) : _ufree(ptr)
#endif
#else
#ifdef HYPRE_USE_UMALLOC
#undef HYPRE_USE_UMALLOC
#endif
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
   char *ptr;

   if (size > 0)
   {
#ifdef HYPRE_USE_UMALLOC
      HYPRE_Int threadid = hypre_GetThreadID();

      ptr = _umalloc_(size);
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

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_CAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_CAlloc( size_t count,
              size_t elt_size )
{
   char   *ptr;
   size_t  size = count*elt_size;

   if (size > 0)
   {
#ifdef HYPRE_USE_UMALLOC
      HYPRE_Int threadid = hypre_GetThreadID();

      ptr = _ucalloc_(count, elt_size);
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

   return ptr;
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
      ptr = _urealloc_(ptr, size);
   }
#else
   if (ptr == NULL)
   {
      ptr = malloc(size);
   }
   else
   {
      ptr = realloc(ptr, size);
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
#else
      free(ptr);
#endif
   }
}


/*--------------------------------------------------------------------------
 * These Shared routines are for one thread to allocate memory for data
 * will be visible to all threads.  The file-scope pointer
 * global_alloc_ptr is used in these routines.
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_USE_PTHREADS

char *global_alloc_ptr;
double *global_data_ptr;

/*--------------------------------------------------------------------------
 * hypre_SharedMAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_SharedMAlloc( size_t size )
{
   char *ptr;
   HYPRE_Int unthreaded = pthread_equal(initial_thread, pthread_self());
   HYPRE_Int I_call_malloc = unthreaded ||
                       pthread_equal(hypre_thread[0],pthread_self());

   if (I_call_malloc) {
      global_alloc_ptr = hypre_MAlloc( size );
   }

   hypre_barrier(&talloc_mtx, unthreaded);
   ptr = global_alloc_ptr;
   hypre_barrier(&talloc_mtx, unthreaded);

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_SharedCAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_SharedCAlloc( size_t count,
                    size_t elt_size )
{
   char *ptr;
   HYPRE_Int unthreaded = pthread_equal(initial_thread, pthread_self());
   HYPRE_Int I_call_calloc = unthreaded ||
                       pthread_equal(hypre_thread[0],pthread_self());

   if (I_call_calloc) {
      global_alloc_ptr = hypre_CAlloc( count, elt_size );
   }

   hypre_barrier(&talloc_mtx, unthreaded);
   ptr = global_alloc_ptr;
   hypre_barrier(&talloc_mtx, unthreaded);

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_SharedReAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_SharedReAlloc( char   *ptr,
                     size_t  size )
{
   HYPRE_Int unthreaded = pthread_equal(initial_thread, pthread_self());
   HYPRE_Int I_call_realloc = unthreaded ||
                       pthread_equal(hypre_thread[0],pthread_self());

   if (I_call_realloc) {
      global_alloc_ptr = hypre_ReAlloc( ptr, size );
   }

   hypre_barrier(&talloc_mtx, unthreaded);
   ptr = global_alloc_ptr;
   hypre_barrier(&talloc_mtx, unthreaded);

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_SharedFree
 *--------------------------------------------------------------------------*/

void
hypre_SharedFree( char *ptr )
{
   HYPRE_Int unthreaded = pthread_equal(initial_thread, pthread_self());
   HYPRE_Int I_call_free = unthreaded ||
                     pthread_equal(hypre_thread[0],pthread_self());

   hypre_barrier(&talloc_mtx, unthreaded);
   if (I_call_free) {
      hypre_Free(ptr);
   }
   hypre_barrier(&talloc_mtx, unthreaded);
}

/*--------------------------------------------------------------------------
 * hypre_IncrementSharedDataPtr
 *--------------------------------------------------------------------------*/

double *
hypre_IncrementSharedDataPtr( double *ptr, size_t size )
{
   HYPRE_Int unthreaded = pthread_equal(initial_thread, pthread_self());
   HYPRE_Int I_increment = unthreaded ||
                     pthread_equal(hypre_thread[0],pthread_self());

   if (I_increment) {
      global_data_ptr = ptr + size;
   }

   hypre_barrier(&talloc_mtx, unthreaded);
   ptr = global_data_ptr;
   hypre_barrier(&talloc_mtx, unthreaded);

   return ptr;
}

#endif

