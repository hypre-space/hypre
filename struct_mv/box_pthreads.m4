changequote(<<,>>)
sinclude(pthreads_c_definitions.m4)

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
   PLOOP(k, 0, hypre__nz, iteration_counter, 0, hypre_thread_counter,
         hypre_mutex_boxloops, hypre_cond_boxloops,
      <<for (j = 0; j < hypre__ny; j++ )\
        {\
           for (i = 0; i < hypre__nx; i++ )\
           {\
               body;\
           }\
        }\
   >>)
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
   PLOOP(k, 0, hypre__nz, iteration_counter, 0, hypre_thread_counter,
         hypre_mutex_boxloops, hypre_cond_boxloops,
      <<for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\
         {\
            body;\
            i1 += hypre__iinc1;\
         }\
         i1 += hypre__jinc1;\
      }\
      i1 += hypre__kinc1;\
   >>)
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
   PLOOP(k, 0, hypre__nz, iteration_counter, 0, hypre_thread_counter,
         hypre_mutex_boxloops, hypre_cond_boxloops,
      <<for (j = 0; j < hypre__ny; j++ )\
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
   >>)
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
   PLOOP(k, 0, hypre__nz, iteration_counter, 0, hypre_thread_counter,
         hypre_mutex_boxloops, hypre_cond_boxloops,
      <<for (j = 0; j < hypre__ny; j++ )\
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
   >>)
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
   PLOOP(k, 0, hypre__nz, iteration_counter, 0, hypre_thread_counter,
         hypre_mutex_boxloops, hypre_cond_boxloops,
      <<for (j = 0; j < hypre__ny; j++ )\
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
   >>)
}

#endif

