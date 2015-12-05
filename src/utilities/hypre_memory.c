/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Memory management utilities
 *
 *****************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include "utilities.h"

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

int
hypre_OutOfMemory( int size )
{
   printf("Out of memory trying to allocate %d bytes\n", size);
   fflush(stdout);

   hypre_error(HYPRE_ERROR_MEMORY);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_MAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_MAlloc( int size )
{
   char *ptr;

   if (size > 0)
   {
#ifdef HYPRE_USE_UMALLOC
      int threadid = hypre_GetThreadID();

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
hypre_CAlloc( int count,
              int elt_size )
{
   char *ptr;
   int   size = count*elt_size;

   if (size > 0)
   {
#ifdef HYPRE_USE_UMALLOC
      int threadid = hypre_GetThreadID();

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
hypre_ReAlloc( char *ptr,
               int   size )
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
      int threadid = hypre_GetThreadID();
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
      int threadid = hypre_GetThreadID();

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
hypre_SharedMAlloc( int size )
{
   char *ptr;
   int unthreaded = pthread_equal(initial_thread, pthread_self());
   int I_call_malloc = unthreaded ||
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
hypre_SharedCAlloc( int count,
              int elt_size )
{
   char *ptr;
   int unthreaded = pthread_equal(initial_thread, pthread_self());
   int I_call_calloc = unthreaded ||
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
hypre_SharedReAlloc( char *ptr,
                     int   size )
{
   int unthreaded = pthread_equal(initial_thread, pthread_self());
   int I_call_realloc = unthreaded ||
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
   int unthreaded = pthread_equal(initial_thread, pthread_self());
   int I_call_free = unthreaded ||
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
hypre_IncrementSharedDataPtr( double *ptr, int size )
{
   int unthreaded = pthread_equal(initial_thread, pthread_self());
   int I_increment = unthreaded ||
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

