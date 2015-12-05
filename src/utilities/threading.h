/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



#ifndef hypre_THREADING_HEADER
#define hypre_THREADING_HEADER

#if defined(HYPRE_USING_OPENMP) || defined (HYPRE_USING_PGCC_SMP)

HYPRE_Int hypre_NumThreads( void );
HYPRE_Int hypre_GetThreadNum( void );

#else

#define hypre_NumThreads() 1
#define hypre_GetThreadNum() 0

#endif


/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/* The pthreads stuff needs to be reworked */

#ifdef HYPRE_USE_PTHREADS

#ifndef MAX_QUEUE
#define MAX_QUEUE 256
#endif

#include <pthread.h>

/* hypre_work_proc_t typedef'd to be a pointer to a function with a void*
   argument and a void return type */
typedef void (*hypre_work_proc_t)(void *);

typedef struct hypre_workqueue_struct {
   pthread_mutex_t lock;
   pthread_cond_t work_wait;
   pthread_cond_t finish_wait;
   hypre_work_proc_t worker_proc_queue[MAX_QUEUE];
   HYPRE_Int n_working;
   HYPRE_Int n_waiting;
   HYPRE_Int n_queue;
   HYPRE_Int inp;
   HYPRE_Int outp;
   void *argqueue[MAX_QUEUE];
} *hypre_workqueue_t;

void hypre_work_put( hypre_work_proc_t funcptr, void *argptr );
void hypre_work_wait( void );
HYPRE_Int HYPRE_InitPthreads( HYPRE_Int num_threads );
void HYPRE_DestroyPthreads( void );
void hypre_pthread_worker( HYPRE_Int threadid );
HYPRE_Int ifetchadd( HYPRE_Int *w, pthread_mutex_t *mutex_fetchadd );
HYPRE_Int hypre_fetch_and_add( HYPRE_Int *w );
void hypre_barrier(pthread_mutex_t *mpi_mtx, HYPRE_Int unthreaded);
HYPRE_Int hypre_GetThreadID( void );

pthread_t initial_thread;
pthread_t hypre_thread[hypre_MAX_THREADS];
pthread_mutex_t hypre_mutex_boxloops;
pthread_mutex_t talloc_mtx;
pthread_mutex_t worker_mtx;
hypre_workqueue_t hypre_qptr;
pthread_mutex_t mpi_mtx;
pthread_mutex_t time_mtx;
volatile HYPRE_Int hypre_thread_release;

#ifdef HYPRE_THREAD_GLOBALS
HYPRE_Int hypre_NumThreads = 4;
#else
extern HYPRE_Int hypre_NumThreads;
#endif

#endif
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#endif
