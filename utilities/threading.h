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

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif
#ifndef MAX_QUEUE
#define MAX_QUEUE 256
#endif


#include<pthread.h>
#include "mpi.h"

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
int HYPRE_InitPthreads( MPI_Comm comm );
void HYPRE_DestroyPthreads( void );
void hypre_pthread_worker( int threadid );
int ifetchadd( int *w, pthread_mutex_t *mutex_fetchadd );
int hypre_fetch_and_add( int *w );
void hypre_barrier(mutex *mpi_mtx,cond *mpi_cnd,int *th_sem);

pthread_t hypre_thread[NUM_THREADS];
pthread_cond_t hypre_cond_boxloops;
pthread_mutex_t hypre_mutex_boxloops;
hypre_workqueue_t hypre_qptr;
pthread_mutex_t mpi_mtx;
pthread_cond_t mpi_cnd;

#endif
