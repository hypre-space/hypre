/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
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

#ifdef HYPRE_USE_PTHREADS

#include <pthread.h>
#include "threading.h"


extern int hypre_thread_counter;
extern int iteration_counter;

/*--------------------------------------------------------------------------
 * Threaded Looping macros:
 *--------------------------------------------------------------------------*/

#ifndef MAX_ISIZE
#define MAX_ISIZE 5
#endif
#ifndef MAX_JSIZE
#define MAX_JSIZE 5
#endif
#ifndef MAX_KSIZE
#define MAX_KSIZE 5
#endif

#define hypre_ThreadLoop(local_counter, init_val, stop_val, tl_index,\
                         tl_count, tl_release, tl_mtx, tl_body)\
{\
   for (local_counter = ifetchadd(&tl_index, &tl_mtx) + init_val;\
        local_counter < stop_val;\
        local_counter = ifetchadd(&tl_index, &tl_mtx) + init_val)\
   {\
      tl_body;\
   }\
   if (pthread_equal(initial_thread, pthread_self()) == 0)\
   {\
      pthread_mutex_lock(&tl_mtx);\
      tl_count++;\
      if (tl_count < hypre_NumThreads)\
      {\
         pthread_mutex_unlock(&tl_mtx);\
         while (!tl_release);\
         pthread_mutex_lock(&tl_mtx);\
         tl_count--;\
         pthread_mutex_unlock(&tl_mtx);\
         while (tl_release);\
      }\
      else\
      {\
         tl_count--;\
         tl_index = 0;\
         pthread_mutex_unlock(&tl_mtx);\
         tl_release = 1;\
         while (tl_count);\
         tl_release = 0;\
      }\
   }\
   else\
      tl_index = 0;\
}

#define hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz)\
   int hypre__cx = min(hypre__nx / 4 + !!(hypre__nx % 4), MAX_ISIZE);\
   int hypre__cy = min(hypre__ny / 4 + !!(hypre__ny % 4), MAX_JSIZE);\
   int hypre__cz = min(hypre__nz / 4 + !!(hypre__nz % 4), MAX_KSIZE);\
   int znumchunk = hypre__nz / hypre__cz + !!(hypre__nz % hypre__cz);\
   int ynumchunk = hypre__ny / hypre__cy + !!(hypre__ny % hypre__cy);\
   int xnumchunk = hypre__nx / hypre__cx + !!(hypre__nx % hypre__cx);\
   int numchunks = znumchunk * ynumchunk * xnumchunk;\
   int clfreq[3], clreset[3];\
   int clstart[3];\
   int clfinish[3];\
   int chunkcount;\
   clfreq[0] = 1;\
   clreset[0] = xnumchunk;\
   clfreq[1] = clreset[0];\
   clreset[1] = ynumchunk * xnumchunk;\
   clfreq[2] = clreset[1];\
   clreset[2] = znumchunk * ynumchunk * xnumchunk
 
#define hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                     hypre__nx, hypre__ny, hypre__nz,\
                                     hypre__cx, hypre__cy, hypre__cz,\
                                     chunkcount)\
      clstart[0] = ((chunkcount % clreset[0]) / clfreq[0]) * hypre__cx;\
      if (clstart[0] < hypre__nx - hypre__cx)\
         clfinish[0] = clstart[0] + hypre__cx;\
      else\
         clfinish[0] = hypre__nx;\
      clstart[1] = ((chunkcount % clreset[1]) / clfreq[1]) * hypre__cy;\
      if (clstart[1] < hypre__ny - hypre__cy)\
         clfinish[1] = clstart[1] + hypre__cy;\
      else\
         clfinish[1] = hypre__ny;\
      clstart[2] = ((chunkcount % clreset[2]) / clfreq[2]) * hypre__cz;\
      if (clstart[2] < hypre__nz - hypre__cz)\
         clfinish[2] = clstart[2] + hypre__cz;\
      else\
         clfinish[2] = hypre__nz



#define hypre_BoxLoop0(i, j, k,loop_size,\
                       body)\
{\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
   hypre_ThreadLoop(chunkcount, 0, numchunks, iteration_counter,\
                    hypre_thread_counter, hypre_thread_release,\
                    hypre_mutex_boxloops,\
   {\
      hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                   hypre__nx, hypre__ny, hypre__nz,\
                                   hypre__cx, hypre__cy, hypre__cz,\
                                   chunkcount);\
      for (k = clstart[2]; k < clfinish[2]; k++) {\
         for (j = clstart[1]; j < clfinish[1]; j++)\
         {\
            for (i = clstart[0]; i < clfinish[0]; i++)\
            {\
                body;\
            }\
         }\
      }\
   });\
}



#define hypre_BoxLoop1(i, j, k, loop_size,\
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
   hypre_ThreadLoop(chunkcount, 0, numchunks, iteration_counter,\
                    hypre_thread_counter, hypre_thread_release,\
                    hypre_mutex_boxloops,\
   {\
      hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                   hypre__nx, hypre__ny, hypre__nz,\
                                   hypre__cx, hypre__cy, hypre__cz,\
                                   chunkcount);\
      for (k = clstart[2]; k < clfinish[2]; k++) {\
         for (j = clstart[1]; j < clfinish[1]; j++)\
         {\
            for (i = clstart[0]; i < clfinish[0]; i++)\
            {\
               i1 = orig_i1 +\
                    (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc1 +\
                    (j + hypre__ny*k)*hypre__jinc1 + k*hypre__kinc1;\
               body;\
            }\
         }\
      }\
   });\
}



#define hypre_BoxLoop2(i, j, k, loop_size,\
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
   hypre_ThreadLoop(chunkcount, 0, numchunks, iteration_counter,\
                    hypre_thread_counter, hypre_thread_release,\
                    hypre_mutex_boxloops,\
   {\
      hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                   hypre__nx, hypre__ny, hypre__nz,\
                                   hypre__cx, hypre__cy, hypre__cz,\
                                   chunkcount);\
      for (k = clstart[2]; k < clfinish[2]; k++) {\
         for (j = clstart[1]; j < clfinish[1]; j++)\
         {\
            for (i = clstart[0]; i < clfinish[0]; i++)\
            {\
               i1 = orig_i1 +\
                    (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc1 +\
                    (j + hypre__ny*k)*hypre__jinc1 + k*hypre__kinc1;\
               i2 = orig_i2 +\
                    (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc2 +\
                    (j + hypre__ny*k)*hypre__jinc2 + k*hypre__kinc2;\
               body;\
            }\
         }\
      }\
   });\
}


#define hypre_BoxLoop3(i, j, k, loop_size,\
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
   hypre_ThreadLoop(chunkcount, 0, numchunks, iteration_counter,\
                    hypre_thread_counter, hypre_thread_release,\
                    hypre_mutex_boxloops,\
   {\
      hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                   hypre__nx, hypre__ny, hypre__nz,\
                                   hypre__cx, hypre__cy, hypre__cz,\
                                   chunkcount);\
      for (k = clstart[2]; k < clfinish[2]; k++)\
      {\
         for (j = clstart[1]; j < clfinish[1]; j++)\
         {\
            for (i = clstart[0]; i < clfinish[0]; i++)\
            {\
               i1 = orig_i1 +\
                    (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc1 +\
                    (j + hypre__ny*k)*hypre__jinc1 + k*hypre__kinc1;\
               i2 = orig_i2 +\
                    (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc2 +\
                    (j + hypre__ny*k)*hypre__jinc2 + k*hypre__kinc2;\
               i3 = orig_i3 +\
                    (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc3 +\
                    (j + hypre__ny*k)*hypre__jinc3 + k*hypre__kinc3;\
               body;\
            }\
         }\
      }\
   })\
}



#define hypre_BoxLoop4(i, j, k, loop_size,\
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
   hypre_ThreadLoop(chunkcount, 0, numchunks, iteration_counter,\
                    hypre_thread_counter, hypre_thread_release,\
                    hypre_mutex_boxloops,\
   {\
      hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                   hypre__nx, hypre__ny, hypre__nz,\
                                   hypre__cx, hypre__cy, hypre__cz,\
                                   chunkcount);\
      for (k = clstart[2]; k < clfinish[2]; k++) {\
         for (j = clstart[1]; j < clfinish[1]; j++)\
         {\
            for (i = clstart[0]; i < clfinish[0]; i++)\
            {\
               i1 = orig_i1 +\
                    (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc1 +\
                    (j + hypre__ny*k)*hypre__jinc1 + k*hypre__kinc1;\
               i2 = orig_i2 +\
                    (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc2 +\
                    (j + hypre__ny*k)*hypre__jinc2 + k*hypre__kinc2;\
               i3 = orig_i3 +\
                    (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc3 +\
                    (j + hypre__ny*k)*hypre__jinc3 + k*hypre__kinc3;\
               i4 = orig_i4 +\
                    (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc4 +\
                    (j + hypre__ny*k)*hypre__jinc4 + k*hypre__kinc4;\
               body;\
            }\
         }\
      }\
   });\
}



#endif

#endif

