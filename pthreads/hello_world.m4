changequote(<<,>>)
include(pthreads_c_definitions.m4)

#include<pthread.h>
#define NUM_THREADS 4

void *hello_world (void*);

int main()
{

   pthread_t threads[NUM_THREADS];
   int t;

   for(t=0; t<NUM_THREADS;t++)
      pthread_create(&threads[t], NULL, hello_world, (void *)t);

   pthread_exit(NULL);
}

void *hello_world (void *threadid)
{
   int indx[3]={0,0,0};
   int i;

   printf("hello world %d\n", (int)threadid);
   PLOOP(i,0,8,indx,0,<<printf("%d\n",i);>>)
   PLOOPEND(indx,0)
}
