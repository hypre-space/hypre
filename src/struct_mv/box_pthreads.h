/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Header info for the Box structures
 *
 *****************************************************************************/

#ifdef HYPRE_USE_PTHREADS

#ifndef hypre_BOX_PTHREADS_HEADER
#define hypre_BOX_PTHREADS_HEADER

#include <pthread.h>
#include "threading.h"


extern volatile HYPRE_Int hypre_thread_counter;
extern HYPRE_Int iteration_counter;

/*--------------------------------------------------------------------------
 * Threaded Looping macros:
 *--------------------------------------------------------------------------*/

#ifndef CHUNK_GOAL
#define CHUNK_GOAL (hypre_NumThreads*1)
#endif
#ifndef MIN_VOL
#define MIN_VOL 125
#endif
#ifndef MAX_VOL
#define MAX_VOL 64000
#endif

#define hypre_BoxLoopDeclare(loop_size, data_box, stride, iinc, jinc, kinc) \
HYPRE_Int  iinc = (hypre_IndexX(stride));\
HYPRE_Int  jinc = (hypre_IndexY(stride)*hypre_BoxSizeX(data_box) -\
             hypre_IndexX(loop_size)*hypre_IndexX(stride));\
HYPRE_Int  kinc = (hypre_IndexZ(stride)*\
             hypre_BoxSizeX(data_box)*hypre_BoxSizeY(data_box) -\
             hypre_IndexY(loop_size)*\
             hypre_IndexY(stride)*hypre_BoxSizeX(data_box))

#define vol_cbrt(vol) (HYPRE_Int) pow((double)(vol), 1. / 3.) 

#define hypre_ThreadLoopBegin(local_counter, init_val, stop_val, tl_index,\
			      tl_mtx, tl_body)\
   for (local_counter = ifetchadd(&tl_index, &tl_mtx) + init_val;\
        local_counter < stop_val;\
        local_counter = ifetchadd(&tl_index, &tl_mtx) + init_val)\
     {\
	tl_body;

#define hypre_ThreadLoop(tl_index,\
                         tl_count, tl_release, tl_mtx)\
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
      tl_index = 0

#define hypre_ThreadLoopOld(local_counter, init_val, stop_val, tl_index,\
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
   HYPRE_Int target_vol, target_area, target_len;\
   HYPRE_Int cbrt_tar_vol, sqrt_tar_area;\
   HYPRE_Int edge_divisor;\
   HYPRE_Int znumchunk, ynumchunk, xnumchunk;\
   HYPRE_Int hypre__cz, hypre__cy, hypre__cx;\
   HYPRE_Int numchunks;\
   HYPRE_Int clfreq[3], clreset[3];\
   HYPRE_Int clstart[3];\
   HYPRE_Int clfinish[3];\
   HYPRE_Int chunkcount;\
   target_vol    = hypre_min(hypre_max((hypre__nx * hypre__ny * hypre__nz) / CHUNK_GOAL,\
                           MIN_VOL), MAX_VOL);\
   cbrt_tar_vol  = (HYPRE_Int) (pow ((double)target_vol, 1./3.));\
   edge_divisor  = hypre__nz / cbrt_tar_vol + !!(hypre__nz % cbrt_tar_vol);\
   hypre__cz     = hypre__nz / edge_divisor + !!(hypre__nz % edge_divisor);\
   znumchunk     = hypre__nz / hypre__cz + !!(hypre__nz % hypre__cz);\
   target_area   = target_vol / hypre__cz;\
   sqrt_tar_area = (HYPRE_Int) (sqrt((double)target_area));\
   edge_divisor  = hypre__ny / sqrt_tar_area + !!(hypre__ny % sqrt_tar_area);\
   hypre__cy     = hypre__ny / edge_divisor + !!(hypre__ny % edge_divisor);\
   ynumchunk     = hypre__ny / hypre__cy + !!(hypre__ny % hypre__cy);\
   target_len    = target_area / hypre__cy;\
   edge_divisor  = hypre__nx / target_len + !!(hypre__nx % target_len);\
   hypre__cx     = hypre__nx / edge_divisor + !!(hypre__nx % edge_divisor);\
   xnumchunk     = hypre__nx / hypre__cx + !!(hypre__nx % hypre__cx);\
   numchunks     = znumchunk * ynumchunk * xnumchunk;\
   clfreq[0]     = 1;\
   clreset[0]    = xnumchunk;\
   clfreq[1]     = clreset[0];\
   clreset[1]    = ynumchunk * xnumchunk;\
   clfreq[2]     = clreset[1];\
   clreset[2]    = znumchunk * ynumchunk * xnumchunk
 
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

#define hypre_BoxLoop0Begin(loop_size)\
{\
   HYPRE_Int hypre__nx = hypre_IndexX(loop_size);\
   HYPRE_Int hypre__ny = hypre_IndexY(loop_size);\
   HYPRE_Int hypre__nz = hypre_IndexZ(loop_size);\
   if (hypre__nx && hypre__ny && hypre__nz )\
   {\
      hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
      hypre_ThreadLoopBegin(chunkcount, 0, numchunks, iteration_counter,\
                       hypre_mutex_boxloops,\
         hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                      hypre__nx, hypre__ny, hypre__nz,\
                                      hypre__cx, hypre__cy, hypre__cz,\
                                      chunkcount));

#define hypre_BoxLoop0For(i, j, k)\
         for (k = clstart[2]; k < clfinish[2]; k++ )\
	 {\
            for (j = clstart[1]; j < clfinish[1]; j++ )\
            {\
               for (i = clstart[0]; i < clfinish[0]; i++ )\
               {

#define hypre_BoxLoop0End() }}}hypre_ThreadLoop(iteration_counter,\
			     hypre_thread_counter, hypre_thread_release,\
					      hypre_mutex_boxloops);}}}


#define hypre_BoxLoop1Begin(loop_size,\
			    data_box1, start1, stride1, i1)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__iinc1, hypre__jinc1, hypre__kinc1);\
   HYPRE_Int hypre__nx = hypre_IndexX(loop_size);\
   HYPRE_Int hypre__ny = hypre_IndexY(loop_size);\
   HYPRE_Int hypre__nz = hypre_IndexZ(loop_size);\
   HYPRE_Int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
   if (hypre__nx && hypre__ny && hypre__nz )\
   {\
      hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
      hypre_ThreadLoopBegin(chunkcount, 0, numchunks, iteration_counter,\
                       hypre_mutex_boxloops,\
         hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                      hypre__nx, hypre__ny, hypre__nz,\
                                      hypre__cx, hypre__cy, hypre__cz,\
                                      chunkcount));

#define hypre_BoxLoop1For(i, j, k, i1)\
         for (k = clstart[2]; k < clfinish[2]; k++)\
	   {\
            for (j = clstart[1]; j < clfinish[1]; j++)\
            {\
               for (i = clstart[0]; i < clfinish[0]; i++)\
               {\
                  i1 = orig_i1 +\
                      (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc1 +\
                      (j + hypre__ny*k)*hypre__jinc1 + k*hypre__kinc1;

#define hypre_BoxLoop1End(i1) }}}hypre_ThreadLoop(iteration_counter,\
			     hypre_thread_counter, hypre_thread_release,\
					      hypre_mutex_boxloops);}}}

#define hypre_BoxLoop2Begin(loop_size,\
			    data_box1, start1, stride1, i1,\
                            data_box2, start2, stride2, i2)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__iinc1, hypre__jinc1, hypre__kinc1);\
   hypre_BoxLoopDeclare(loop_size, data_box2, stride2,\
                        hypre__iinc2, hypre__jinc2, hypre__kinc2);\
   HYPRE_Int hypre__nx = hypre_IndexX(loop_size);\
   HYPRE_Int hypre__ny = hypre_IndexY(loop_size);\
   HYPRE_Int hypre__nz = hypre_IndexZ(loop_size);\
   HYPRE_Int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
   HYPRE_Int orig_i2 = hypre_BoxIndexRank(data_box2, start2);\
   if (hypre__nx && hypre__ny && hypre__nz )\
   {\
      hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
      hypre_ThreadLoopBegin(chunkcount, 0, numchunks, iteration_counter,\
                       hypre_mutex_boxloops,\
         hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                      hypre__nx, hypre__ny, hypre__nz,\
                                      hypre__cx, hypre__cy, hypre__cz,\
                                      chunkcount))

#define hypre_BoxLoop2For(i, j, k, i1, i2)\
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
                      (j + hypre__ny*k)*hypre__jinc2 + k*hypre__kinc2;

#define hypre_BoxLoop2End(i1, i2) }}}hypre_ThreadLoop(iteration_counter,\
                       hypre_thread_counter, hypre_thread_release,\
                       hypre_mutex_boxloops);}}}
					       
					      


#define hypre_BoxLoop3Begin(loop_size,\
			    data_box1, start1, stride1, i1,\
                            data_box2, start2, stride2, i2,\
                            data_box3, start3, stride3, i3)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__iinc1, hypre__jinc1, hypre__kinc1);\
   hypre_BoxLoopDeclare(loop_size, data_box2, stride2,\
                        hypre__iinc2, hypre__jinc2, hypre__kinc2);\
   hypre_BoxLoopDeclare(loop_size, data_box3, stride3,\
                        hypre__iinc3, hypre__jinc3, hypre__kinc3);\
   HYPRE_Int hypre__nx = hypre_IndexX(loop_size);\
   HYPRE_Int hypre__ny = hypre_IndexY(loop_size);\
   HYPRE_Int hypre__nz = hypre_IndexZ(loop_size);\
   HYPRE_Int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
   HYPRE_Int orig_i2 = hypre_BoxIndexRank(data_box2, start2);\
   HYPRE_Int orig_i3 = hypre_BoxIndexRank(data_box3, start3);\
   if (hypre__nx && hypre__ny && hypre__nz )\
   {\
      hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
      hypre_ThreadLoopBegin(chunkcount, 0, numchunks, iteration_counter,\
                       hypre_mutex_boxloops,\
         hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                      hypre__nx, hypre__ny, hypre__nz,\
                                      hypre__cx, hypre__cy, hypre__cz,\
                                      chunkcount))

#define hypre_BoxLoop3For(i, j, k, i1, i2, i3)\
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

#define hypre_BoxLoop3End(i1, i2, i3) }}}hypre_ThreadLoop(iteration_counter,\
			     hypre_thread_counter, hypre_thread_release,\
					      hypre_mutex_boxloops);}}}


#define hypre_BoxLoop4Begin(loop_size,\
			    data_box1, start1, stride1, i1,\
                            data_box2, start2, stride2, i2,\
                            data_box3, start3, stride3, i3,\
                            data_box4, start4, stride4, i4)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__iinc1, hypre__jinc1, hypre__kinc1);\
   hypre_BoxLoopDeclare(loop_size, data_box2, stride2,\
                        hypre__iinc2, hypre__jinc2, hypre__kinc2);\
   hypre_BoxLoopDeclare(loop_size, data_box3, stride3,\
                        hypre__iinc3, hypre__jinc3, hypre__kinc3);\
   hypre_BoxLoopDeclare(loop_size, data_box4, stride4,\
                        hypre__iinc4, hypre__jinc4, hypre__kinc4);\
   HYPRE_Int hypre__nx = hypre_IndexX(loop_size);\
   HYPRE_Int hypre__ny = hypre_IndexY(loop_size);\
   HYPRE_Int hypre__nz = hypre_IndexZ(loop_size);\
   HYPRE_Int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
   HYPRE_Int orig_i2 = hypre_BoxIndexRank(data_box2, start2);\
   HYPRE_Int orig_i3 = hypre_BoxIndexRank(data_box3, start3);\
   HYPRE_Int orig_i4 = hypre_BoxIndexRank(data_box4, start4);\
   if (hypre__nx && hypre__ny && hypre__nz )\
   {\
      hypre_ChunkLoopExternalSetup(hypre__nx, hypre__ny, hypre__nz);\
      hypre_ThreadLoopBegin(chunkcount, 0, numchunks, iteration_counter,\
                       hypre_mutex_boxloops,\
         hypre_ChunkLoopInternalSetup(clstart, clfinish, clreset, clfreq,\
                                      hypre__nx, hypre__ny, hypre__nz,\
                                      hypre__cx, hypre__cy, hypre__cz,\
                                      chunkcount))

#define hypre_BoxLoop4For(i, j, k, i1, i2, i3, i4)\
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
                  i4 = orig_i4 +\
                      (i + hypre__nx*j + hypre__nx*hypre__ny*k)*hypre__iinc4 +\
                      (j + hypre__ny*k)*hypre__jinc4 + k*hypre__kinc4;\


#define hypre_BoxLoop4End(i1, i2, i3, i4) }}}hypre_ThreadLoop(iteration_counter,\
			     hypre_thread_counter, hypre_thread_release,\
					      hypre_mutex_boxloops);}}}


#endif

#endif

