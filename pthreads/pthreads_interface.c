#include <pthread.h>
#if 0
#include <sys/atomic_op.h>
#endif
#include <unistd.h>


#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>


#define ABS(x)  (((x)<0)?(-(x)):(x))
#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

#define MXTHREADS  33  /*   Maximum no. compute threads     */
#define MXFADDLOCKS  65
#define debug   0

#if F2C == 1
#define F_T_INIT f_t_init_
#define F_T_CREATE f_t_create_
#define F_T_ID f_t_id_
#define FBARRIER fbarrier_
#define VBARRIER vbarrier_
#define F_SEM_DELETE f_sem_delete_
#define IFETCHADD ifetchadd_
#define ABARRIER abarrier_
#define LOOPEND loopend_
#define IWAITER iwaiter_
#endif
#if 1
#define F_T_INIT f_t_init
#define F_T_CREATE f_t_create
#define F_T_ID f_t_id
#define FBARRIER fbarrier
#define VBARRIER vbarrier
#define F_SEM_DELETE f_sem_delete
#define IFETCHADD ifetchadd
#define ABARRIER abarrier
#define LOOPEND loopend
#define IWAITER iwaiter
#endif
#if F2C == 3
#define F_T_INIT .f_t_init
#define F_T_CREATE .f_t_create
#define F_T_ID .f_t_id
#define FBARRIER .fbarrier
#define VBARRIER .vbarrier
#define F_SEM_DELETE .f_sem_delete
#define IFETCHADD .ifetchadd
#define ABARRIER .abarrier
#define LOOPEND .loopend
#define IWAITER .iwaiter
#endif



/***************************************************************************/
/* global data, all mutexes created with default attributes are fast ones,
   i.e. can be locked only once before being unlocked *****/
/***************************************************************************/

pthread_attr_t   pthread_attributes; /* attributes object for threads */
pthread_t  threads  [ MXTHREADS   ]; /* array of thread handles       */

pthread_mutex_t     lock_tid; /* a lock for f_t_create + f_t_id */
pthread_mutex_t     lock_bar; /* a lock for fbarrier            */
pthread_mutex_t     lock_zap; /* a lock for fbarrier            */
pthread_mutex_t     lock_bars;/* a lock for abarrier            */
pthread_mutex_t     lock_fadd [ MXFADDLOCKS+1 ]; /* locks for ifetchadd */
pthread_cond_t      barrier_cv;   /* condition var for abarrier         */
static int          barrier_phase; /* parity of abarrier                */
static int          barrier_count; /* thread count for abarrier         */
static int          barrier_nwaiting; /* thread count for fbarrier      */
static volatile int barrier_parity; /* parity of fbarrier               */

static int Semid;            /* Semaphore id used in abarrier */
#define MXSEM 8
/***************************************************************************/
/* global data, all mutexes created with default attributes are fast ones,
   i.e. can be locked only once before being unlocked *****/
/***************************************************************************/

void F_T_INIT( int *stacksize, int *istatus)
{
   u_long stack_size;
   ushort zeros[MXSEM]= {0,0,0,0,0,0,0,0};
   int i, semid_base;
   pthread_mutexattr_t mutex_fast_attr;

   stack_size  =        (u_long)*stacksize;

   *istatus    = pthread_attr_init       (&pthread_attributes            );
   *istatus   += pthread_attr_setstacksize (&pthread_attributes, stack_size);
   pthread_mutexattr_init(&mutex_fast_attr);
/* This function not found by SPARC
   pthread_mutexattr_setkind_np(&mutex_fast_attr, MUTEX_FAST_NP); */

   *istatus   += pthread_mutex_init        (&lock_tid,    &mutex_fast_attr );
   *istatus   += pthread_mutex_init        (&lock_bar,    &mutex_fast_attr );
   *istatus   += pthread_mutex_init        (&lock_zap,    &mutex_fast_attr );
   *istatus   += pthread_mutex_init        (&lock_bars,   &mutex_fast_attr );
   *istatus   += pthread_cond_init         (&barrier_cv, NULL);
   barrier_phase = 0;
   barrier_count = 0;
   barrier_nwaiting = 0;
   barrier_parity = 0;
   for (i=1;i<=MXFADDLOCKS;i++)
   {
      *istatus   += pthread_mutex_init    (&lock_fadd[i],&mutex_fast_attr);
   }

   /* Make semaphores for abarrier function. */

   semid_base = ((('p'<<24) + ('p'<<16) + ('m'<<8) ))+ getpid();
   i=0;

#if 0
   do
   {
      Semid = semget( semid_base + i, MXSEM, IPC_CREAT|0600 );
   }
   while(Semid == -1 && i < 10);
#endif


   /* 10 attempts at unique ids, then bail */
#if 0
   if ( Semid < 0                             )
   {
      perror("ipcinit");
      exit(1);
   }
   if ( semctl( Semid, MXSEM, SETALL, zeros ) )
   {
      perror("ipcinit");
      exit(1);
   }
#endif

   return;
}

/***************************************************************************/
/* Create thread for routine f_routine with thread handle ithread, used by
   fortran calling routine to refer to this thread ********/
/***************************************************************************/

void F_T_CREATE( int *ithread, void *f_routine(), int *istatus )
{
#if 0
   typedef void *_addr_t;
   typedef _addr_t (*pthread_func)(void *);
#endif
   int i  = *ithread;
   *istatus  =  pthread_mutex_lock   (&lock_tid);
   *istatus +=  pthread_create       (&threads[i], &pthread_attributes,
                                      f_routine, (void*)i);
   *istatus +=  pthread_mutex_unlock (&lock_tid);
#if 0
   *istatus +=  pthread_mutex_unlock (&lock_tid);
#endif

   return;
}

/***************************************************************************/
/* Find id of calling thread */
/***************************************************************************/

void F_T_ID( int *ithread, int *istatus )
{
   pthread_t thread_me = pthread_self(); int my_id = 0; int i;

   *istatus  = pthread_mutex_lock   (&lock_tid);
   for (i = 1; i <= MXTHREADS; i++)
   {
      if (pthread_equal(threads[i],thread_me))
         my_id = i;
   }
   *istatus += pthread_mutex_unlock (&lock_tid);
   *ithread  = my_id;

   return;
}

/***************************************************************************/
/*ifetchadd function for SPPM */
/*pthread version of SPPM routine */
/***************************************************************************/

int IFETCHADD( int *w)
{
   int temp;
   
   temp = *w;
   *w += 1;
   return temp;
}

/***************************************************************************/
/* This spinning barrier sychronization is a pthread implementation of the
   sPPM routine fbarrier (spinning barrier) ***************/
/***************************************************************************/

void FBARRIER( int num_threads)
{
   int my_parity;

   my_parity = barrier_parity;
   IFETCHADD(&barrier_nwaiting);
   if (barrier_nwaiting == num_threads)
   {
      barrier_nwaiting = 0;
      barrier_parity = 1 - my_parity;
   }
   else
   {
      while (barrier_parity == my_parity);
   }

   return;
}

/**************************************************************/
/* Delete semaphores for SPPM                                 */
/**************************************************************/

void F_SEM_DELETE()
{
   int i;
#if 0
   if ( semctl( Semid, MXSEM, IPC_RMID ) )
   {
      perror("semdestroy");
      exit(1);
   }
#endif
}


/***************************************************************************/
/*This is a sleeping barrier sychronization using semaphores */
/*pthread version of SPPM routine abarrier (sleeping barrier)  */
/***************************************************************************/

/*  THIS ROUTINE COMMENTED OUT */
#if 0
void ABARRIER(unsigned *num)
{
   struct sembuf sb;
   static volatile u_long bars       = 0;
   volatile u_long bars_local = 0;
   int nm1;
   int status;

   nm1 = *num - 1;
   if ( !nm1 ) return;
   sb.sem_flg = 0;
   sb.sem_num = 0;

   /*SWH -note nm1 is the last man in. Every proc before  */
   /*SWH -increments counter and sleeps on  Semid.        */
   status   = pthread_mutex_lock     (&lock_bars);
   bars_local = bars;
   bars++;
   status   = pthread_mutex_unlock   (&lock_bars);
   if ( bars_local == nm1 )
   {
      bars = 0;
      sb.sem_op = nm1;  /* notify */
   }
   else
   {
      sb.sem_op = -1;   /* wait */
   }
   if ( semop( Semid, &sb, 1 ) ) perror("swait");
}
#endif

/*********************************************************************/
/*Twiddle (spin)  until ivar changes.                                */
/*volatile insures ivar fetched from memory for every while check    */
/*********************************************************************/

void LOOPEND( volatile int ivar )
{
   while(ivar!=0);        /* spin */
}

void IWAITER( volatile int ivar )
{
   while(ivar==0);        /* spin */
}

void ABARRIER( int num_threads)
{
   int n = num_threads;
   int my_phase;
   int istatus;

   if (n<=1) return;

   istatus  = pthread_mutex_lock   (&lock_bars);
   my_phase = barrier_phase;
   barrier_count++;
   if (barrier_count == n)
   {
      barrier_count = 0;
      barrier_phase = 1 - my_phase;
      pthread_cond_broadcast(&barrier_cv);
   }
   while (barrier_phase == my_phase)
   {
      pthread_cond_wait(&barrier_cv, &lock_bars);
   }
   istatus += pthread_mutex_unlock(&lock_bars);
   return;
}


