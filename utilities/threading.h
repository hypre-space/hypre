/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

#ifndef hypre_THREADING_HEADER
#define hypre_THREADING_HEADER

#ifdef HYPRE_USE_PTHREADS

#ifndef hypre_MAX_THREADS
#define hypre_MAX_THREADS 128
#endif
#ifndef MAX_QUEUE
#define MAX_QUEUE 256
#endif


#include<pthread.h>
#include "mpi.h"


#define hypre_BarrierThreadWrapper(body) \
   body;\
   hypre_barrier(&talloc_mtx, pthread_equal(pthread_self(), initial_thread))


/* hypre_work_proc_t typedef'd to be a pointer to a function with a void*
   argument and a void return type */
typedef void (*hypre_work_proc_t)(void *);

typedef struct hypre_workqueue_struct {
   pthread_mutex_t lock;
   pthread_cond_t work_wait;
   pthread_cond_t finish_wait;
   hypre_work_proc_t worker_proc_queue[MAX_QUEUE];
   int n_working;
   int n_waiting;
   int n_queue;
   int inp;
   int outp;
   void *argqueue[MAX_QUEUE];
} *hypre_workqueue_t;

void hypre_work_put( hypre_work_proc_t funcptr, void *argptr );
void hypre_work_wait( void );
int HYPRE_InitPthreads( int num_threads );
void HYPRE_DestroyPthreads( void );
void hypre_pthread_worker( int threadid );
int ifetchadd( int *w, pthread_mutex_t *mutex_fetchadd );
int hypre_fetch_and_add( int *w );
void hypre_barrier(pthread_mutex_t *mpi_mtx, int unthreaded);
int hypre_GetThreadID( void );


pthread_t initial_thread;
pthread_t hypre_thread[hypre_MAX_THREADS];
pthread_mutex_t hypre_mutex_boxloops;
pthread_mutex_t talloc_mtx;
hypre_workqueue_t hypre_qptr;
pthread_mutex_t mpi_mtx;
pthread_mutex_t time_mtx;
int hypre_thread_release;

#ifdef HYPRE_THREAD_GLOBALS
int hypre_NumThreads = 4;
#else
extern int hypre_NumThreads;
#endif

#endif

#endif
