
/* cliches.h

 m4 definitions.
*/



/* ================================================================
 Self-scheduled parallelism macros.
*/

/*
Usage:  BARRIER(number_of_threads)
Argument:  $1 -- integer containing the number of threads being used
*/



/*
Usage:  FBARRIER(number_of_threads)
Argument:  $1 -- integer containing the number of threads being used
*/



/* PLOOPEND waits until the last proc. resets the loop index to zero.
Usage:  PLOOPEND(index_array_name, array_index)
Arguments:  $1 -- the name of the array of indices
            $2 -- an index of the given array
To wait until element n of array "indx" is set to zero, enter 
   PLOOPEND(indx,n)
*/



/* IWAIT is used to examine values of the global flag ipencil in calchyd
 If there is only one thread, this is a noop.
Usage:  IWAIT(flag_array_name, array_index)
Arguments:  $1 -- the name of an array of flags
            $2 -- an index of the given array
*/



/* PLOOP parallel loop macro.
 Example:
     PLOOP(z,lz,mz,indx,3,body)
 The indx used ($5) must not be reused for a loop
 until a synch. point. guarantees all threads have exited.

Usage: PLOOP(increment_variable, loop_initial_value, loop_stopping value,
             index_array_name, array_index, thread_counter, mutex_object,
             cond_object loop_body)

NUM_THREADS must either be #defined as the number of threads being used or
be an integer variable containing the number of threads.

Arguments:  $1 -- the name of the increment variable for the loop
            $2 -- the initial value for the increment variable
            $3 -- the stopping value for the increment variable
                  (loop will not be entered when increment reaches this value)
            $4 -- the name of an array of indices
            $5 -- an index of the given array
            $6 -- an integer counter to count each thread as it passes
            $7 -- a pthread_mutex_t object (must be initialized
                  externally)
            $8 -- a pthread_cond_t object (must be initialized externally)
            $9 -- The body of the loop (enclose between  and  )
*/







/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the Box structures
 *
 *****************************************************************************/

#ifndef hypre_BOX_PTHREADS_HEADER
#define hypre_BOX_PTHREADS_HEADER
#include <pthread.h>
#include "threading.h"

/*--------------------------------------------------------------------------
 * Threaded Looping macros:
 *--------------------------------------------------------------------------*/

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

int hypre_thread_counter;
int iteration_counter[3]={0,0,0};

#define hypre_BoxLoop0_pthread(i, j, k, loop_size,\
                       body)\
{\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   \
\
   for (k = ifetchadd(&iteration_counter[0], &hypre_mutex_boxloops) + 0;\
        k <  hypre__nz;\
        k = ifetchadd(&iteration_counter[0], &hypre_mutex_boxloops) + 0) {\
      for (j = 0; j < hypre__ny; j++ )\
        {\
           for (i = 0; i < hypre__nx; i++ )\
           {\
               body;\
           }\
        }\
   \
   }\
\
   ifetchadd(&hypre_thread_counter, &hypre_mutex_boxloops);\
\
   pthread_mutex_lock(&hypre_mutex_boxloops);\
\
   if (hypre_thread_counter < NUM_THREADS)\
      pthread_cond_wait(&hypre_cond_boxloops, &hypre_mutex_boxloops);\
   else if (hypre_thread_counter == NUM_THREADS) {\
      pthread_cond_broadcast(&hypre_cond_boxloops);\
      iteration_counter[0] = 0;\
      hypre_thread_counter = 0;\
   }\
\
   pthread_mutex_unlock(&hypre_mutex_boxloops);\
}

#define hypre_BoxLoop1_pthread(i, j, k, loop_size,\
                       data_box1, start1, stride1, i1,\
                       body)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__iinc1, hypre__jinc1, hypre__kinc1);\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   i1 = hypre_BoxIndexRank(data_box1, start1);\
   \
\
   for (k = ifetchadd(&iteration_counter[0], &hypre_mutex_boxloops) + 0;\
        k <  hypre__nz;\
        k = ifetchadd(&iteration_counter[0], &hypre_mutex_boxloops) + 0) {\
      for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\
         {\
            body;\
            i1 += hypre__iinc1;\
         }\
         i1 += hypre__jinc1;\
      }\
      i1 += hypre__kinc1;\
   \
   }\
\
   ifetchadd(&hypre_thread_counter, &hypre_mutex_boxloops);\
\
   pthread_mutex_lock(&hypre_mutex_boxloops);\
\
   if (hypre_thread_counter < NUM_THREADS)\
      pthread_cond_wait(&hypre_cond_boxloops, &hypre_mutex_boxloops);\
   else if (hypre_thread_counter == NUM_THREADS) {\
      pthread_cond_broadcast(&hypre_cond_boxloops);\
      iteration_counter[0] = 0;\
      hypre_thread_counter = 0;\
   }\
\
   pthread_mutex_unlock(&hypre_mutex_boxloops);\
}

#define hypre_BoxLoop2_pthread(i, j, k, loop_size,\
                       data_box1, start1, stride1, i1,\
                       data_box2, start2, stride2, i2,\
                       body)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__iinc1, hypre__jinc1, hypre__kinc1);\
   hypre_BoxLoopDeclare(loop_size, data_box2, stride2,\
                        hypre__iinc2, hypre__jinc2, hypre__kinc2);\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   i1 = hypre_BoxIndexRank(data_box1, start1);\
   i2 = hypre_BoxIndexRank(data_box2, start2);\
   \
\
   for (k = ifetchadd(&iteration_counter[0], &hypre_mutex_boxloops) + 0;\
        k <  hypre__nz;\
        k = ifetchadd(&iteration_counter[0], &hypre_mutex_boxloops) + 0) {\
      for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\  
         {\
            body;\
            i1 += hypre__iinc1;\
            i2 += hypre__iinc2;\
         }\
         i1 += hypre__jinc1;\
         i2 += hypre__jinc2;\
      }\
      i1 += hypre__kinc1;\
      i2 += hypre__kinc2;\
   \
   }\
\
   ifetchadd(&hypre_thread_counter, &hypre_mutex_boxloops);\
\
   pthread_mutex_lock(&hypre_mutex_boxloops);\
\
   if (hypre_thread_counter < NUM_THREADS)\
      pthread_cond_wait(&hypre_cond_boxloops, &hypre_mutex_boxloops);\
   else if (hypre_thread_counter == NUM_THREADS) {\
      pthread_cond_broadcast(&hypre_cond_boxloops);\
      iteration_counter[0] = 0;\
      hypre_thread_counter = 0;\
   }\
\
   pthread_mutex_unlock(&hypre_mutex_boxloops);\
}

#define hypre_BoxLoop3_pthread(i, j, k, loop_size,\
                       data_box1, start1, stride1, i1,\
                       data_box2, start2, stride2, i2,\
                       data_box3, start3, stride3, i3,\
                       body)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__iinc1, hypre__jinc1, hypre__kinc1);\
   hypre_BoxLoopDeclare(loop_size, data_box2, stride2,\
                        hypre__iinc2, hypre__jinc2, hypre__kinc2);\
   hypre_BoxLoopDeclare(loop_size, data_box3, stride3,\
                        hypre__iinc3, hypre__jinc3, hypre__kinc3);\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   i1 = hypre_BoxIndexRank(data_box1, start1);\
   i2 = hypre_BoxIndexRank(data_box2, start2);\
   i3 = hypre_BoxIndexRank(data_box3, start3);\
   \
\
   for (k = ifetchadd(&iteration_counter[0], &hypre_mutex_boxloops) + 0;\
        k <  hypre__nz;\
        k = ifetchadd(&iteration_counter[0], &hypre_mutex_boxloops) + 0) {\
      for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\
         {\
            body;\
            i1 += hypre__iinc1;\
            i2 += hypre__iinc2;\
            i3 += hypre__iinc3;\
         }\
         i1 += hypre__jinc1;\
         i2 += hypre__jinc2;\
         i3 += hypre__jinc3;\
      }\
      i1 += hypre__kinc1;\
      i2 += hypre__kinc2;\
      i3 += hypre__kinc3;\
   \
   }\
\
   ifetchadd(&hypre_thread_counter, &hypre_mutex_boxloops);\
\
   pthread_mutex_lock(&hypre_mutex_boxloops);\
\
   if (hypre_thread_counter < NUM_THREADS)\
      pthread_cond_wait(&hypre_cond_boxloops, &hypre_mutex_boxloops);\
   else if (hypre_thread_counter == NUM_THREADS) {\
      pthread_cond_broadcast(&hypre_cond_boxloops);\
      iteration_counter[0] = 0;\
      hypre_thread_counter = 0;\
   }\
\
   pthread_mutex_unlock(&hypre_mutex_boxloops);\
}

#define hypre_BoxLoop4_pthread(i, j, k, loop_size,\
                       data_box1, start1, stride1, i1,\
                       data_box2, start2, stride2, i2,\
                       data_box3, start3, stride3, i3,\
                       data_box4, start4, stride4, i4,\
                       body)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__iinc1, hypre__jinc1, hypre__kinc1);\
   hypre_BoxLoopDeclare(loop_size, data_box2, stride2,\
                        hypre__iinc2, hypre__jinc2, hypre__kinc2);\
   hypre_BoxLoopDeclare(loop_size, data_box3, stride3,\
                        hypre__iinc3, hypre__jinc3, hypre__kinc3);\
   hypre_BoxLoopDeclare(loop_size, data_box4, stride4,\
                        hypre__iinc4, hypre__jinc4, hypre__kinc4);\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   i1 = hypre_BoxIndexRank(data_box1, start1);\
   i2 = hypre_BoxIndexRank(data_box2, start2);\
   i3 = hypre_BoxIndexRank(data_box3, start3);\
   i4 = hypre_BoxIndexRank(data_box4, start4);\
   \
\
   for (k = ifetchadd(&iteration_counter[0], &hypre_mutex_boxloops) + 0;\
        k <  hypre__nz;\
        k = ifetchadd(&iteration_counter[0], &hypre_mutex_boxloops) + 0) {\
      for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\
         {\
            body;\
            i1 += hypre__iinc1;\
            i2 += hypre__iinc2;\
            i3 += hypre__iinc3;\
            i4 += hypre__iinc4;\
         }\
         i1 += hypre__jinc1;\   
         i2 += hypre__jinc2;\   
         i3 += hypre__jinc3;\   
         i4 += hypre__jinc4;\
      }\
      i1 += hypre__kinc1;\   
      i2 += hypre__kinc2;\   
      i3 += hypre__kinc3;\
      i4 += hypre__kinc4;\
   \
   }\
\
   ifetchadd(&hypre_thread_counter, &hypre_mutex_boxloops);\
\
   pthread_mutex_lock(&hypre_mutex_boxloops);\
\
   if (hypre_thread_counter < NUM_THREADS)\
      pthread_cond_wait(&hypre_cond_boxloops, &hypre_mutex_boxloops);\
   else if (hypre_thread_counter == NUM_THREADS) {\
      pthread_cond_broadcast(&hypre_cond_boxloops);\
      iteration_counter[0] = 0;\
      hypre_thread_counter = 0;\
   }\
\
   pthread_mutex_unlock(&hypre_mutex_boxloops);\
}

#endif

