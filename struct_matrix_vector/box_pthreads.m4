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
#include <semaphore.h>
#include "threading.h"

/*--------------------------------------------------------------------------
 * Threaded Looping macros:
 *--------------------------------------------------------------------------*/

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

int hypre_thread_counter;
int iteration_counter[3]={0,0,0};

#define hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz)\
   int hypre__cx = hypre__nx / 4 + !!(hypre__nx % 4);\
   int hypre__cy = hypre__ny / 4 + !!(hypre__ny % 4);\
   int hypre__cz = hypre__nz / 4 + !!(hypre__nz % 4);\
   int znumchunk = hypre__nz / hypre__cz + !!(hypre__nz % hypre__cz);\
   int ynumchunk = hypre__ny / hypre__cy + !!(hypre__ny % hypre__cy);\
   int xnumchunk = hypre__nx / hypre__cx + !!(hypre__nx % hypre__cx);\
   int numchunks = znumchunk * ynumchunk * xnumchunk;\
   int freq[3], reset[3];\
   int start[3];\
   int finish[3];\
   int chunkcount;\
   freq[0] = 1;\
   reset[0] = xnumchunk;\
   freq[1] = reset[0];\
   reset[1] = ynumchunk * znumchunk;\
   freq[2] = reset[1];\
   reset[2] = znumchunk * ynumchunk * xnumchunk
 
#define hypre_ChunkLoopInternalSetup(start, finish, reset, freq,\
                                     hypre__nx, hypre__ny, hypre__nz,\
                                     hypre__cx, hypre__cy, hypre__cz,\
                                     chunkcount)\
      start[0] = ((chunkcount % reset[0]) / freq[0]) * hypre__cx;\
      if (start[0] < hypre__nx - hypre__cx)\
         finish[0] = start[0] + hypre__cx;\
      else\
         finish[0] = hypre__nx;\
      start[1] = ((chunkcount % reset[1]) / freq[1]) * hypre__cy;\
      if (start[1] < hypre__ny - hypre__cy)\
         finish[1] = start[1] + hypre__cy;\
      else\
         finish[1] = hypre__ny;\
      start[2] = ((chunkcount % reset[2]) / freq[2]) * hypre__cz;\
      if (start[2] < hypre__nz - hypre__cz)\
         finish[2] = start[2] + hypre__cz;\
      else\
         finish[2] = hypre__nz

#define hypre_BoxLoop0_pthread(i, j, k,loop_size,\
                       body)\
{\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
   PLOOP(chunkcount, 0, numchunks, iteration_counter, 0,
         hypre_thread_counter, hypre_mutex_boxloops, hypre_cond_boxloops,
    <<hypre_ChunkLoopInternalSetup(start, finish, reset, freq,\
                                   hypre__nx, hypre__ny, hypre__nz,\
                                   hypre__cx, hypre__cy, hypre__cz,\
                                   chunkcount);\
      for (k = start[2]; k < finish[2]; k++) {\
         for (j = start[1]; j < finish[1]; j++)\
         {\
            for (i = start[0]; i < finish[0]; i++)\
            {\
                body;\
            }\
         }\
      }\
    >>)
}

ifelse(<<
#define hypre_BoxLoop0_pthread_old(i, j, k, loop_size,\
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
>>)

#define hypre_BoxLoop1_pthread(i, j, k, loop_size,\
                       data_box1, start1, stride1, i1,\
                       body)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__iinc1, hypre__jinc1, hypre__kinc1);\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
   hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
   PLOOP(chunkcount, 0, numchunks, iteration_counter, 0,
         hypre_thread_counter, hypre_mutex_boxloops, hypre_cond_boxloops,
    <<hypre_ChunkLoopInternalSetup(start, finish, reset, freq,\
                                   hypre__nx, hypre__ny, hypre__nz,\
                                   hypre__cx, hypre__cy, hypre__cz,\
                                   chunkcount);\
      i1 = orig_i1 + start[2]*(hypre__kinc1 + start[1]*(hypre__jinc1 +\
                                                    start[0]*hypre__iinc1)) +\
                     start[1]*(hypre__jinc1 + start[0]*hypre__iinc1) +\
                     start[0]*hypre__iinc1;\
      for (k = start[2]; k < finish[2]; k++) {\
         for (j = start[1]; j < finish[1]; j++)\
         {\
            for (i = start[0]; i < finish[0]; i++)\
            {\
               body;\
               i1 += hypre__iinc1;\
            }\
            i1 += hypre__jinc1;\
         }\
         i1 += hypre__kinc1;\
      }>>)
}

ifelse(<<
#define hypre_BoxLoop1_pthread_old(i, j, k, loop_size,\
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
>>)

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
   int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
   int orig_i2 = hypre_BoxIndexRank(data_box2, start2);\
   hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
   PLOOP(chunkcount, 0, numchunks, iteration_counter, 0,
         hypre_thread_counter, hypre_mutex_boxloops, hypre_cond_boxloops,
    <<hypre_ChunkLoopInternalSetup(start, finish, reset, freq,\
                                   hypre__nx, hypre__ny, hypre__nz,\
                                   hypre__cx, hypre__cy, hypre__cz,\
                                   chunkcount);\
      i1 = orig_i1 + start[2]*(hypre__kinc1 + start[1]*(hypre__jinc1 +\
                                                    start[0]*hypre__iinc1)) +\
                     start[1]*(hypre__jinc1 + start[0]*hypre__iinc1) +\
                     start[0]*hypre__iinc1;\
      i2 = orig_i2 + start[2]*(hypre__kinc2 + start[1]*(hypre__jinc2 +\
                                                    start[0]*hypre__iinc2)) +\
                     start[1]*(hypre__jinc2 + start[0]*hypre__iinc2) +\
                     start[0]*hypre__iinc2;\
      for (k = start[2]; k < finish[2]; k++) {\
         for (j = start[1]; j < finish[1]; j++)\
         {\
            for (i = start[0]; i < finish[0]; i++)\
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
      }\
    >>)
}

ifelse(<<
#define hypre_BoxLoop2_pthread_old(i, j, k, loop_size,\
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
>>)

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
   int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
   int orig_i2 = hypre_BoxIndexRank(data_box2, start2);\
   int orig_i3 = hypre_BoxIndexRank(data_box3, start3);\
   hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
   PLOOP(chunkcount, 0, numchunks, iteration_counter, 0,
         hypre_thread_counter, hypre_mutex_boxloops, hypre_cond_boxloops,
    <<hypre_ChunkLoopInternalSetup(start, finish, reset, freq,\
                                   hypre__nx, hypre__ny, hypre__nz,\
                                   hypre__cx, hypre__cy, hypre__cz,\
                                   chunkcount);\
      i1 = orig_i1 + start[2]*(hypre__kinc1 + start[1]*(hypre__jinc1 +\
                                                    start[0]*hypre__iinc1)) +\
                     start[1]*(hypre__jinc1 + start[0]*hypre__iinc1) +\
                     start[0]*hypre__iinc1;\
      i2 = orig_i2 + start[2]*(hypre__kinc2 + start[1]*(hypre__jinc2 +\
                                                    start[0]*hypre__iinc2)) +\
                     start[1]*(hypre__jinc2 + start[0]*hypre__iinc2) +\
                     start[0]*hypre__iinc2;\
      i3 = orig_i3 + start[2]*(hypre__kinc3 + start[1]*(hypre__jinc3 +\
                                                    start[0]*hypre__iinc3)) +\
                     start[1]*(hypre__jinc3 + start[0]*hypre__iinc3) +\
                     start[0]*hypre__iinc3;\
      for (k = start[2]; k < finish[2]; k++) {\
         for (j = start[1]; j < finish[1]; j++)\
         {\
            for (i = start[0]; i < finish[0]; i++)\
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
      }\
    >>)
}

ifelse(<<
#define hypre_BoxLoop3_pthread_old(i, j, k, loop_size,\
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
>>)

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
   int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
   int orig_i2 = hypre_BoxIndexRank(data_box2, start2);\
   int orig_i3 = hypre_BoxIndexRank(data_box3, start3);\
   int orig_i4 = hypre_BoxIndexRank(data_box4, start4);\
   hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
   PLOOP(chunkcount, 0, numchunks, iteration_counter, 0,
         hypre_thread_counter, hypre_mutex_boxloops, hypre_cond_boxloops,
    <<hypre_ChunkLoopInternalSetup(start, finish, reset, freq,\
                                   hypre__nx, hypre__ny, hypre__nz,\
                                   hypre__cx, hypre__cy, hypre__cz,\
                                   chunkcount);\
      i1 = orig_i1 + start[2]*(hypre__kinc1 + start[1]*(hypre__jinc1 +\
                                                    start[0]*hypre__iinc1)) +\
                     start[1]*(hypre__jinc1 + start[0]*hypre__iinc1) +\
                     start[0]*hypre__iinc1;\
      i2 = orig_i2 + start[2]*(hypre__kinc2 + start[1]*(hypre__jinc2 +\
                                                    start[0]*hypre__iinc2)) +\
                     start[1]*(hypre__jinc2 + start[0]*hypre__iinc2) +\
                     start[0]*hypre__iinc2;\
      i3 = orig_i3 + start[2]*(hypre__kinc3 + start[1]*(hypre__jinc3 +\
                                                    start[0]*hypre__iinc3)) +\
                     start[1]*(hypre__jinc3 + start[0]*hypre__iinc3) +\
                     start[0]*hypre__iinc3;\
      i4 = orig_i4 + start[2]*(hypre__kinc4 + start[1]*(hypre__jinc4 +\
                                                    start[0]*hypre__iinc4)) +\
                     start[1]*(hypre__jinc4 + start[0]*hypre__iinc4) +\
                     start[0]*hypre__iinc4;\
      for (k = start[2]; k < finish[2]; k++) {\
         for (j = start[1]; j < finish[1]; j++)\
         {\
            for (i = start[0]; i < finish[0]; i++)\
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
      }\
    >>)
}

ifelse(<<
#define hypre_BoxLoop4_pthread_old(i, j, k, loop_size,\
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
>>)


#endif

