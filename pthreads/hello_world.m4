changequote(<<,>>)
include(pthreads_c_definitions.m4)

#include<stdio.h>
#include<pthread.h>
#define NUM_THREADS 4

int thread_counter=0;
int loop_level[3]={0,0,0};
pthread_cond_t threads_sync_cv;
pthread_mutex_t mutex_hello;

void *hello_world (void*);

int main()
{

   pthread_t threads[NUM_THREADS];
   int t,rv;

   pthread_mutex_init(&mutex_hello,NULL);
   pthread_cond_init(&threads_sync_cv,NULL);

   for(t=0; t<NUM_THREADS;t++) {
      rv=pthread_create(&threads[t], NULL, hello_world, (void*)t);
   } 

   for(t=0; t<NUM_THREADS;t++) {
      rv=pthread_create(&threads[t], NULL, hello_world, (void*)t);
   } 

   
   pthread_exit(NULL);

}


void *hello_world (void *threadid)
{
   int i;


   printf("hello world %d\n",  (int)threadid);

   PLOOP(i,0,2,loop_level,0,thread_counter,mutex_hello,threads_sync_cv,
         <<printf("%d %d\n",i, (int)threadid);>>)
}
