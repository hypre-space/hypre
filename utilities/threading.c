/*BHEADER********************************************************************** 
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

#ifdef HYPRE_USE_PTHREADS

#include <malloc.h>
#include <assert.h>
#include <stdio.h>
#include <signal.h>
#include "mpi.h"
#include "threading.h"

int iteration_counter[3] = {0,0,0};
int hypre_thread_counter;

int HYPRE_InitPthreads( MPI_Comm comm )
{
   int err;
   int i;
   hypre_qptr =
          (hypre_workqueue_t) malloc(sizeof(struct hypre_workqueue_struct) +
                                      (MAX_QUEUE * sizeof(void *)));


   initial_thread = pthread_self();

   if (hypre_qptr != NULL) {
      pthread_mutex_init(&hypre_qptr->lock, NULL);
      pthread_cond_init(&hypre_qptr->work_wait, NULL);
      pthread_cond_init(&hypre_qptr->finish_wait, NULL);
      hypre_qptr->n_working = hypre_qptr->n_waiting = hypre_qptr->n_queue = 0;
      hypre_qptr->inp = hypre_qptr->outp = 0;
      for (i=0; i < NUM_THREADS; i++) {
         err=pthread_create(&hypre_thread[i], NULL, 
                            (void *(*)(void *))hypre_pthread_worker,
                            (void *)i);
         assert(err == 0);
      }
   }

   pthread_mutex_init(&hypre_mutex_boxloops, NULL);
   pthread_mutex_init(&mpi_mtx, NULL);
   pthread_cond_init(&hypre_cond_boxloops, NULL);
   pthread_cond_init(&mpi_cnd, NULL);
   hypre_thread_counter = 0;
   hypre_thread_release = 0;

   return (err);
}   

void HYPRE_DestroyPthreads( void )
{
   int i,x;

   for (i=0; i < NUM_THREADS; i++) {
      x= pthread_cancel(hypre_thread[i]);
   }

   pthread_mutex_destroy(&hypre_qptr->lock);
   pthread_mutex_destroy(&hypre_mutex_boxloops);
   pthread_mutex_destroy(&mpi_mtx);
   pthread_cond_destroy(&hypre_qptr->work_wait);
   pthread_cond_destroy(&hypre_qptr->finish_wait);
   pthread_cond_destroy(&hypre_cond_boxloops);
   pthread_cond_destroy(&mpi_cnd); 
   free (hypre_qptr);
}

void hypre_pthread_worker( int threadid )
{
   void *argptr;
   hypre_work_proc_t funcptr;

   pthread_mutex_lock(&hypre_qptr->lock);

   hypre_qptr->n_working++;

   for(;;) {
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
   assert(hypre_qptr->n_queue != MAX_QUEUE);

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


int
hypre_fetch_and_add( int *w )
{
   int temp;

   temp = *w;
   *w += 1;
   
   return temp;
}
   
int
ifetchadd( int *w, pthread_mutex_t *mutex_fetchadd )
{
   int n;
    
   
   pthread_mutex_lock(mutex_fetchadd);
   n = hypre_fetch_and_add(w);
   pthread_mutex_unlock(mutex_fetchadd);
 
   return n;
}

static int thb_count = 0;
static int thb_release = 0;

void hypre_barrier(pthread_mutex_t *mtx)
{
  pthread_mutex_lock(mtx);
  thb_count++;

  if (thb_count < NUM_THREADS) {
    pthread_mutex_unlock(mtx);
    while (!thb_release);
    pthread_mutex_lock(mtx);
    (thb_count)--;
    pthread_mutex_unlock(mtx);
    while (thb_release);
  }
  else if (thb_count == NUM_THREADS) {
    (thb_count)--;
    pthread_mutex_unlock(mtx);
    (thb_release)++;
    while (thb_count);
    thb_release = 0;
  }
}


#endif
