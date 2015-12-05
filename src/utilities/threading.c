/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/



#include <stdlib.h>
#include <stdio.h>
#include "_hypre_utilities.h"

#if defined(HYPRE_USING_OPENMP) || defined (HYPRE_USING_PGCC_SMP)

HYPRE_Int
hypre_NumThreads( )
{
   HYPRE_Int num_threads;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel
   num_threads = omp_get_num_threads();
#endif
#ifdef HYPRE_USING_PGCC_SMP
   num_threads = 2;
#endif

   return num_threads;
}



/* This next function must be called from within a 
parallel region! */

HYPRE_Int
hypre_GetThreadNum( )
{
   HYPRE_Int my_thread_num;

#ifdef HYPRE_USING_OPENMP
   my_thread_num = omp_get_thread_num();
#endif
#ifdef HYPRE_USING_PGCC_SMP
   /* THIS NEEDS TO BE FIXED */
   my_thread_num = 0;
#endif

   return my_thread_num;
}


#endif

/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/* The pthreads stuff needs to be reworked */

#define HYPRE_THREAD_GLOBALS

#ifdef HYPRE_USE_PTHREADS

#ifdef HYPRE_USE_UMALLOC
#include "umalloc_local.h"
#endif

HYPRE_Int iteration_counter = 0;
volatile HYPRE_Int hypre_thread_counter;
volatile HYPRE_Int work_continue = 1;


HYPRE_Int HYPRE_InitPthreads( HYPRE_Int num_threads )
{
   HYPRE_Int err;
   HYPRE_Int i;
   hypre_qptr =
          (hypre_workqueue_t) malloc(sizeof(struct hypre_workqueue_struct));

   hypre_NumThreads = num_threads;
   initial_thread = pthread_self();

   if (hypre_qptr != NULL) {
      pthread_mutex_init(&hypre_qptr->lock, NULL);
      pthread_cond_init(&hypre_qptr->work_wait, NULL);
      pthread_cond_init(&hypre_qptr->finish_wait, NULL);
      hypre_qptr->n_working = hypre_qptr->n_waiting = hypre_qptr->n_queue = 0;
      hypre_qptr->inp = hypre_qptr->outp = 0;
      for (i=0; i < hypre_NumThreads; i++) {
#ifdef HYPRE_USE_UMALLOC
         /* Get initial area to start heap */
         hypre_assert ((_uinitial_block[i] = malloc(INITIAL_HEAP_SIZE))!=NULL);
 
         /* Create a user heap */
         hypre_assert ((_uparam[i].myheap = _ucreate(initial_block[i],
                                    INITIAL_HEAP_SIZE,
                                    _BLOCK_CLEAN,
                                    _HEAP_REGULAR,
                                    _uget_fn,
                                    _urelease_fn)) != NULL);
#endif
         err=pthread_create(&hypre_thread[i], NULL, 
                            (void *(*)(void *))hypre_pthread_worker,
                            (void *)i);
         hypre_assert(err == 0);
      }
   }

   pthread_mutex_init(&hypre_mutex_boxloops, NULL);
   pthread_mutex_init(&mpi_mtx, NULL);
   pthread_mutex_init(&talloc_mtx, NULL);
   pthread_mutex_init(&time_mtx, NULL);
   pthread_mutex_init(&worker_mtx, NULL);
   hypre_thread_counter = 0;
   hypre_thread_release = 0;

   return (err);
}   

void hypre_StopWorker(void *i)
{
   work_continue = 0;
}

void HYPRE_DestroyPthreads( void )
{
   HYPRE_Int i;
   void *status;

   for (i=0; i < hypre_NumThreads; i++) {
      hypre_work_put(hypre_StopWorker, (void *) &i);
   }

#ifdef HYPRE_USE_UMALLOC
   for (i=0; i<hypre_NumThreads; i++)
   {
     _udestroy (_uparam[i].myheap, _FORCE);
   }
#endif

   for (i=0; i<hypre_NumThreads; i++)
      pthread_join(hypre_thread[i], &status);
   pthread_mutex_destroy(&hypre_qptr->lock);
   pthread_mutex_destroy(&hypre_mutex_boxloops);
   pthread_mutex_destroy(&mpi_mtx);
   pthread_mutex_destroy(&talloc_mtx);
   pthread_mutex_destroy(&time_mtx);
   pthread_mutex_destroy(&worker_mtx);
   pthread_cond_destroy(&hypre_qptr->work_wait);
   pthread_cond_destroy(&hypre_qptr->finish_wait);
   free (hypre_qptr);
}


void hypre_pthread_worker( HYPRE_Int threadid )
{
   void *argptr;
   hypre_work_proc_t funcptr;

   pthread_mutex_lock(&hypre_qptr->lock);

   hypre_qptr->n_working++;

   while(work_continue) {
      while (hypre_qptr->n_queue == 0) {
         if (--hypre_qptr->n_working == 0)
            pthread_cond_signal(&hypre_qptr->finish_wait);         
         hypre_qptr->n_waiting++;
         pthread_cond_wait(&hypre_qptr->work_wait, &hypre_qptr->lock);
         hypre_qptr->n_waiting--;
         hypre_qptr->n_working++;
      }
      hypre_qptr->n_queue--;
      funcptr = hypre_qptr->worker_proc_queue[hypre_qptr->outp];
      argptr = hypre_qptr->argqueue[hypre_qptr->outp];
      
      hypre_qptr->outp = (hypre_qptr->outp + 1) % MAX_QUEUE;

      pthread_mutex_unlock(&hypre_qptr->lock);

      (*funcptr)(argptr);

      hypre_barrier(&worker_mtx, 0);

      if (work_continue)
         pthread_mutex_lock(&hypre_qptr->lock);
   }
}

void
hypre_work_put( hypre_work_proc_t funcptr, void *argptr )
{
   pthread_mutex_lock(&hypre_qptr->lock);
   if (hypre_qptr->n_waiting) {
      /* idle workers to be awakened */
      pthread_cond_signal(&hypre_qptr->work_wait);
   }
   hypre_assert(hypre_qptr->n_queue != MAX_QUEUE);

   hypre_qptr->n_queue++;
   hypre_qptr->worker_proc_queue[hypre_qptr->inp] = funcptr;
   hypre_qptr->argqueue[hypre_qptr->inp] = argptr;
   hypre_qptr->inp = (hypre_qptr->inp + 1) % MAX_QUEUE;
   pthread_mutex_unlock(&hypre_qptr->lock);
}


/* Wait until all work is done and workers quiesce. */
void
hypre_work_wait( void )
{       
   pthread_mutex_lock(&hypre_qptr->lock);
   while(hypre_qptr->n_queue !=0 || hypre_qptr->n_working != 0)
      pthread_cond_wait(&hypre_qptr->finish_wait, &hypre_qptr->lock);
   pthread_mutex_unlock(&hypre_qptr->lock);
}                               


HYPRE_Int
hypre_fetch_and_add( HYPRE_Int *w )
{
   HYPRE_Int temp;

   temp = *w;
   *w += 1;
   
   return temp;
}
   
HYPRE_Int
ifetchadd( HYPRE_Int *w, pthread_mutex_t *mutex_fetchadd )
{
   HYPRE_Int n;
   
   pthread_mutex_lock(mutex_fetchadd);
   n = *w;
   *w += 1;                   
   pthread_mutex_unlock(mutex_fetchadd);
 
   return n;
}

static volatile HYPRE_Int thb_count = 0;
static volatile HYPRE_Int thb_release = 0;

void hypre_barrier(pthread_mutex_t *mtx, HYPRE_Int unthreaded)
{
   if (!unthreaded) {
      pthread_mutex_lock(mtx);
      thb_count++;

      if (thb_count < hypre_NumThreads) {
         pthread_mutex_unlock(mtx);
         while (!thb_release);
         pthread_mutex_lock(mtx);
         thb_count--;
         pthread_mutex_unlock(mtx);
         while (thb_release);
      }
      else if (thb_count == hypre_NumThreads) {
         thb_count--;
         pthread_mutex_unlock(mtx);
         thb_release++;
         while (thb_count);
         thb_release = 0;
      }
   }
}

HYPRE_Int
hypre_GetThreadID( void )
{
   HYPRE_Int i;

   if (pthread_equal(pthread_self(), initial_thread)) 
      return hypre_NumThreads;

   for (i = 0; i < hypre_NumThreads; i++)
   {
      if (pthread_equal(pthread_self(), hypre_thread[i]))
         return i;
   }

   return -1;
}

#endif
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
