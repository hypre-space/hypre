changequote(<<,>>)
include(pthreads_c_definitions.m4)

#include<stdio.h>
#include<pthread.h>
#define NUM_THREADS 4

int indx[3]={0,0,0};
pthread_cond_t threads_sync_cv;
pthread_mutex_t mutex_threads_sync;
pthread_mutex_t mutex_hello;
   int loopdone=0;

void *hello_world (void*);

int main()
{

   pthread_t threads[NUM_THREADS];
   int t,rv;

   pthread_mutex_init(&mutex_threads_sync,NULL);
   pthread_cond_init(&threads_sync_cv,NULL);

   for(t=0; t<NUM_THREADS;t++) {
      rv=pthread_create(&threads[t], NULL, hello_world, (void*)t);
   } 
   pthread_exit(NULL);

}


void *hello_world (void *threadid)
{
   int i;




   printf("hello world %d\n",  (int)threadid);

   pthread_mutex_lock(&mutex_threads_sync);

   

   PLOOP(i,0,1111111,indx,0,<<
      if ((int)threadid < NUM_THREADS - 1)
         pthread_cond_wait(&threads_sync_cv, &mutex_threads_sync);
      else if ((int)threadid == NUM_THREADS - 1)
         pthread_cond_broadcast(&threads_sync_cv);
      else
         exit(-1);
      printf("%d %d\n",i, (int)threadid);
      >>,mutex_hello)

   pthread_mutex_unlock(&mutex_threads_sync);  
   PLOOPEND(indx,0)



   pthread_mutex_destroy(&mutex_threads_sync);
   pthread_cond_destroy(&threads_sync_cv);
}
