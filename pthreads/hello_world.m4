changequote(<<,>>)
include(pthreads_c_definitions.m4)

#include<stdio.h>
#include<pthread.h>
#define NUM_THREADS 4

int indx[3]={0,0,0};

void *hello_world (void*);

int main()
{

   pthread_t threads[NUM_THREADS];
   int t,rv;

   for(t=0; t<NUM_THREADS;t++) {
      rv=pthread_create(&threads[t], NULL, hello_world, (void*)t);
   } 
   pthread_exit(NULL);

}


void *hello_world (void *threadid)
{
   int i;
   pthread_mutex_t mutex_hello;
   printf("hello world %d\n",  (int)threadid);

   PLOOP(i,0,8,indx,0,<<printf("%d\n",i);>>,mutex_hello)
   PLOOPEND(indx,0)

}
