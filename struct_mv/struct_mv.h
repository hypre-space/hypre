
#include "HYPRE_mv.h"

#ifndef hypre_MV_HEADER
#define hypre_MV_HEADER

#include "utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifndef hypre_BOX_HEADER
#define hypre_BOX_HEADER

/*--------------------------------------------------------------------------
 * hypre_Index:
 *   This is used to define indices in index space, or dimension
 *   sizes of boxes.
 *
 *   The spatial dimensions x, y, and z may be specified by the
 *   integers 0, 1, and 2, respectively (see the hypre_IndexD macro below).
 *   This simplifies the code in the hypre_Box class by reducing code
 *   replication.
 *--------------------------------------------------------------------------*/

typedef int  hypre_Index[3];
typedef int *hypre_IndexRef;

/*--------------------------------------------------------------------------
 * hypre_Box:
 *   Structure describing a cartesian region of some index space.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Index imin;           /* min bounding indices */
   hypre_Index imax;           /* max bounding indices */

} hypre_Box;

/*--------------------------------------------------------------------------
 * hypre_BoxArray:
 *   An array of boxes.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Box  *boxes;         /* Array of boxes */
   int         size;          /* Size of box array */
   int         alloc_size;    /* Size of currently alloced space */

} hypre_BoxArray;

#define hypre_BoxArrayExcess 10

/*--------------------------------------------------------------------------
 * hypre_BoxArrayArray:
 *   An array of box arrays.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_BoxArray  **box_arrays;    /* Array of pointers to box arrays */
   int               size;          /* Size of box array array */

} hypre_BoxArrayArray;


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_Index
 *--------------------------------------------------------------------------*/

#define hypre_IndexD(index, d)  (index[d])

#define hypre_IndexX(index)     hypre_IndexD(index, 0)
#define hypre_IndexY(index)     hypre_IndexD(index, 1)
#define hypre_IndexZ(index)     hypre_IndexD(index, 2)

/*--------------------------------------------------------------------------
 * Member functions: hypre_Index
 *--------------------------------------------------------------------------*/

#define hypre_SetIndex(index, ix, iy, iz) \
( hypre_IndexX(index) = ix,\
  hypre_IndexY(index) = iy,\
  hypre_IndexZ(index) = iz )

#define hypre_ClearIndex(index)  hypre_SetIndex(index, 0, 0, 0)

#define hypre_CopyIndex(index1, index2) \
( hypre_IndexX(index2) = hypre_IndexX(index1),\
  hypre_IndexY(index2) = hypre_IndexY(index1),\
  hypre_IndexZ(index2) = hypre_IndexZ(index1) )

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_Box
 *--------------------------------------------------------------------------*/

#define hypre_BoxIMin(box)     ((box) -> imin)
#define hypre_BoxIMax(box)     ((box) -> imax)

#define hypre_BoxIMinD(box, d) (hypre_IndexD(hypre_BoxIMin(box), d))
#define hypre_BoxIMaxD(box, d) (hypre_IndexD(hypre_BoxIMax(box), d))
#define hypre_BoxSizeD(box, d) \
hypre_max(0, (hypre_BoxIMaxD(box, d) - hypre_BoxIMinD(box, d) + 1))

#define hypre_BoxIMinX(box)    hypre_BoxIMinD(box, 0)
#define hypre_BoxIMinY(box)    hypre_BoxIMinD(box, 1)
#define hypre_BoxIMinZ(box)    hypre_BoxIMinD(box, 2)

#define hypre_BoxIMaxX(box)    hypre_BoxIMaxD(box, 0)
#define hypre_BoxIMaxY(box)    hypre_BoxIMaxD(box, 1)
#define hypre_BoxIMaxZ(box)    hypre_BoxIMaxD(box, 2)

#define hypre_BoxSizeX(box)    hypre_BoxSizeD(box, 0)
#define hypre_BoxSizeY(box)    hypre_BoxSizeD(box, 1)
#define hypre_BoxSizeZ(box)    hypre_BoxSizeD(box, 2)

#define hypre_CopyBox(box1, box2) \
( hypre_CopyIndex(hypre_BoxIMin(box1), hypre_BoxIMin(box2)),\
  hypre_CopyIndex(hypre_BoxIMax(box1), hypre_BoxIMax(box2)) )

#define hypre_BoxVolume(box) \
(hypre_BoxSizeX(box) * hypre_BoxSizeY(box) * hypre_BoxSizeZ(box))

#define hypre_BoxIndexRank(box, index) \
((hypre_IndexX(index) - hypre_BoxIMinX(box)) + \
 ((hypre_IndexY(index) - hypre_BoxIMinY(box)) + \
   ((hypre_IndexZ(index) - hypre_BoxIMinZ(box)) * \
    hypre_BoxSizeY(box))) * \
  hypre_BoxSizeX(box))

#define hypre_BoxOffsetDistance(box, index) \
(hypre_IndexX(index) + \
 (hypre_IndexY(index) + \
  (hypre_IndexZ(index) * \
   hypre_BoxSizeY(box))) * \
 hypre_BoxSizeX(box))
  
/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxArray
 *--------------------------------------------------------------------------*/

#define hypre_BoxArrayBoxes(box_array)     ((box_array) -> boxes)
#define hypre_BoxArrayBox(box_array, i)    &((box_array) -> boxes[(i)])
#define hypre_BoxArraySize(box_array)      ((box_array) -> size)
#define hypre_BoxArrayAllocSize(box_array) ((box_array) -> alloc_size)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxArrayArray
 *--------------------------------------------------------------------------*/

#define hypre_BoxArrayArrayBoxArrays(box_array_array) \
((box_array_array) -> box_arrays)
#define hypre_BoxArrayArrayBoxArray(box_array_array, i) \
((box_array_array) -> box_arrays[(i)])
#define hypre_BoxArrayArraySize(box_array_array) \
((box_array_array) -> size)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define hypre_ForBoxI(i, box_array) \
for (i = 0; i < hypre_BoxArraySize(box_array); i++)

#define hypre_ForBoxArrayI(i, box_array_array) \
for (i = 0; i < hypre_BoxArrayArraySize(box_array_array); i++)

#define hypre_BoxLoopDeclare(loop_size, data_box, stride, iinc, jinc, kinc) \
int  iinc = (hypre_IndexX(stride));\
int  jinc = (hypre_IndexY(stride)*hypre_BoxSizeX(data_box) -\
             hypre_IndexX(loop_size)*hypre_IndexX(stride));\
int  kinc = (hypre_IndexZ(stride)*\
             hypre_BoxSizeX(data_box)*hypre_BoxSizeY(data_box) -\
             hypre_IndexY(loop_size)*\
             hypre_IndexY(stride)*hypre_BoxSizeX(data_box))

/*-------------------------------------------------------------------------
 * Threaded versions of looping macros are in box_pthreads.h.
 *-------------------------------------------------------------------------*/

#ifndef HYPRE_USE_PTHREADS

#define hypre_BoxLoop0(i, j, k, loop_size,\
                       body)\
{\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   for (k = 0; k < hypre__nz; k++ )\
   {\
      for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\
         {\
            body;\
         }\
      }\
   }\
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
   i1 = hypre_BoxIndexRank(data_box1, start1);\
   for (k = 0; k < hypre__nz; k++ )\
   {\
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
   }\
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
   i1 = hypre_BoxIndexRank(data_box1, start1);\
   i2 = hypre_BoxIndexRank(data_box2, start2);\
   for (k = 0; k < hypre__nz; k++ )\
   {\
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
   }\
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
   i1 = hypre_BoxIndexRank(data_box1, start1);\
   i2 = hypre_BoxIndexRank(data_box2, start2);\
   i3 = hypre_BoxIndexRank(data_box3, start3);\
   for (k = 0; k < hypre__nz; k++ )\
   {\
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
   }\
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
   i1 = hypre_BoxIndexRank(data_box1, start1);\
   i2 = hypre_BoxIndexRank(data_box2, start2);\
   i3 = hypre_BoxIndexRank(data_box3, start3);\
   i4 = hypre_BoxIndexRank(data_box4, start4);\
   for (k = 0; k < hypre__nz; k++ )\
   {\
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
   }\
}

#endif  /* ifndef HYPRE_USE_PTHREADS */

#endif
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

#ifdef HYPRE_USE_PTHREADS

#ifndef hypre_BOX_PTHREADS_HEADER
#define hypre_BOX_PTHREADS_HEADER

#include <pthread.h>
#include "threading.h"


extern volatile int hypre_thread_counter;
extern int iteration_counter;

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

#define vol_cbrt(vol) (int) pow((double)(vol), 1. / 3.) 

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
   int target_vol, target_area, target_len;\
   int cbrt_tar_vol, sqrt_tar_area;\
   int edge_divisor;\
   int znumchunk, ynumchunk, xnumchunk;\
   int hypre__cz, hypre__cy, hypre__cx;\
   int numchunks;\
   int clfreq[3], clreset[3];\
   int clstart[3];\
   int clfinish[3];\
   int chunkcount;\
   target_vol    = hypre_min(hypre_max((hypre__nx * hypre__ny * hypre__nz) / CHUNK_GOAL,\
                           MIN_VOL), MAX_VOL);\
   cbrt_tar_vol  = (int) (pow ((double)target_vol, 1./3.));\
   edge_divisor  = hypre__nz / cbrt_tar_vol + !!(hypre__nz % cbrt_tar_vol);\
   hypre__cz     = hypre__nz / edge_divisor + !!(hypre__nz % edge_divisor);\
   znumchunk     = hypre__nz / hypre__cz + !!(hypre__nz % hypre__cz);\
   target_area   = target_vol / hypre__cz;\
   sqrt_tar_area = (int) (sqrt((double)target_area));\
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

#define hypre_BoxLoop0(i, j, k,loop_size,\
                       body)\
{\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   if (hypre__nx && hypre__ny && hypre__nz )\
   {\
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
   }\
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
   if (hypre__nx && hypre__ny && hypre__nz )\
   {\
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
   }\
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
   if (hypre__nx && hypre__ny && hypre__nz )\
   {\
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
   }\
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
   if (hypre__nx && hypre__ny && hypre__nz )\
   {\
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
   }\
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
   if (hypre__nx && hypre__ny && hypre__nz )\
   {\
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
   }\
}

#endif

#endif

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
 * Header info for the hypre_BoxNeighbors structures
 *
 *****************************************************************************/

#ifndef hypre_BOX_NEIGHBORS_HEADER
#define hypre_BOX_NEIGHBORS_HEADER

/*--------------------------------------------------------------------------
 * hypre_RankLink:
 *--------------------------------------------------------------------------*/

typedef struct rank_link
{
   int               rank;
   struct rank_link *next;

} hypre_RankLink;

typedef hypre_RankLink *hypre_RankLinkArray[3][3][3];

/*--------------------------------------------------------------------------
 * hypre_BoxNeighbors:
 *--------------------------------------------------------------------------*/

typedef struct
{
   int                  num_local;      /* number of local boxes */
   hypre_BoxArray      *boxes;          /* array of boxes */
   int                 *processes;      /* processes of `boxes' */
   int                  max_distance;   /* in infinity norm */

   hypre_RankLinkArray *rank_links;     /* neighbors of `box_ranks' boxes */

} hypre_BoxNeighbors;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_RankLink
 *--------------------------------------------------------------------------*/

#define hypre_RankLinkRank(link)      ((link) -> rank)
#define hypre_RankLinkDistance(link)  ((link) -> distance)
#define hypre_RankLinkNext(link)      ((link) -> next)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxNeighbors
 *--------------------------------------------------------------------------*/

#define hypre_BoxNeighborsNumLocal(neighbors)    ((neighbors) -> num_local)
#define hypre_BoxNeighborsBoxes(neighbors)       ((neighbors) -> boxes)
#define hypre_BoxNeighborsBox(neighbors, n)      ((neighbors) -> boxes[n])
#define hypre_BoxNeighborsProcesses(neighbors)   ((neighbors) -> processes)
#define hypre_BoxNeighborsProcess(neighbors, n)  ((neighbors) -> processes[n])
#define hypre_BoxNeighborsMaxDistance(neighbors) ((neighbors) -> max_distance)
#define hypre_BoxNeighborsRankLinks(neighbors)   ((neighbors) -> rank_links)

#define hypre_BoxNeighborsNumBoxes(neighbors) \
(hypre_BoxArraySize(hypre_BoxNeighborsBoxes(neighbors)))
#define hypre_BoxNeighborsRankLink(neighbors, b, i, j, k) \
(hypre_BoxNeighborsRankLinks(neighbors)[b][i+1][j+1][k+1])

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/
 
#define hypre_BeginBoxNeighborsLoop(n, neighbors, b, distance_index)\
{\
   int             hypre__istart = 0;\
   int             hypre__jstart = 0;\
   int             hypre__kstart = 0;\
   int             hypre__istop  = 0;\
   int             hypre__jstop  = 0;\
   int             hypre__kstop  = 0;\
   hypre_RankLink *hypre__rank_link;\
   int             hypre__i, hypre__j, hypre__k;\
\
   hypre__i = hypre_IndexX(distance_index);\
   if (hypre__i < 0)\
      hypre__istart = -1;\
   else if (hypre__i > 0)\
      hypre__istop = 1;\
\
   hypre__j = hypre_IndexY(distance_index);\
   if (hypre__j < 0)\
      hypre__jstart = -1;\
   else if (hypre__j > 0)\
      hypre__jstop = 1;\
\
   hypre__k = hypre_IndexZ(distance_index);\
   if (hypre__k < 0)\
      hypre__kstart = -1;\
   else if (hypre__k > 0)\
      hypre__kstop = 1;\
\
   for (hypre__k = hypre__kstart; hypre__k <= hypre__kstop; hypre__k++)\
   {\
      for (hypre__j = hypre__jstart; hypre__j <= hypre__jstop; hypre__j++)\
      {\
         for (hypre__i = hypre__istart; hypre__i <= hypre__istop; hypre__i++)\
         {\
            hypre__rank_link = \
               hypre_BoxNeighborsRankLink(neighbors, b,\
                                          hypre__i, hypre__j, hypre__k);\
            while (hypre__rank_link)\
            {\
               n = hypre_RankLinkRank(hypre__rank_link);

#define hypre_EndBoxNeighborsLoop\
               hypre__rank_link = hypre_RankLinkNext(hypre__rank_link);\
            }\
         }\
      }\
   }\
}

#endif
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
 * Header info for hypre_StructStencil data structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_STENCIL_HEADER
#define hypre_STRUCT_STENCIL_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructStencil
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Index   *shape;   /* Description of a stencil's shape */
   int            size;    /* Number of stencil coefficients */
   int            max_offset;
                
   int            dim;     /* Number of dimensions */

   int            ref_count;

} hypre_StructStencil;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_StructStencil structure
 *--------------------------------------------------------------------------*/

#define hypre_StructStencilShape(stencil)      ((stencil) -> shape)
#define hypre_StructStencilSize(stencil)       ((stencil) -> size)
#define hypre_StructStencilMaxOffset(stencil)  ((stencil) -> max_offset)
#define hypre_StructStencilDim(stencil)        ((stencil) -> dim)
#define hypre_StructStencilRefCount(stencil)   ((stencil) -> ref_count)

#define hypre_StructStencilElement(stencil, i) \
hypre_StructStencilShape(stencil)[i]

#endif
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
 * Header info for the hypre_StructGrid structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_GRID_HEADER
#define hypre_STRUCT_GRID_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructGrid:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm             comm;
                      
   hypre_BoxArray      *boxes;        /* Array of boxes in this process */
                      
   int                  dim;          /* Number of grid dimensions */
                      
   int                  global_size;  /* Total number of grid points */
   int                  local_size;   /* Number of grid points locally */

   hypre_BoxNeighbors  *neighbors;    /* neighbors of boxes */
   int                  max_distance;

   hypre_Index          periodic;     /* indicates if grid is periodic */

   /* keep this for now to (hopefully) improve SMG setup performance */
   hypre_BoxArray      *all_boxes;      /* valid only before Assemble */
   int                 *processes;
   int                 *box_ranks;
   hypre_BoxArray      *base_all_boxes;
   hypre_Index          pindex;         /* description of index-space to */
   hypre_Index          pstride;        /* project base_all_boxes onto   */
   int                  alloced;        /* boolean used to free up */

   int                  ref_count;

} hypre_StructGrid;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructGrid
 *--------------------------------------------------------------------------*/

#define hypre_StructGridComm(grid)          ((grid) -> comm)
#define hypre_StructGridBoxes(grid)         ((grid) -> boxes)
#define hypre_StructGridDim(grid)           ((grid) -> dim)
#define hypre_StructGridGlobalSize(grid)    ((grid) -> global_size)
#define hypre_StructGridLocalSize(grid)     ((grid) -> local_size)
#define hypre_StructGridNeighbors(grid)     ((grid) -> neighbors)
#define hypre_StructGridMaxDistance(grid)   ((grid) -> max_distance)
#define hypre_StructGridPeriodic(grid)      ((grid) -> periodic)
#define hypre_StructGridAllBoxes(grid)      ((grid) -> all_boxes)
#define hypre_StructGridProcesses(grid)     ((grid) -> processes)
#define hypre_StructGridBoxRanks(grid)      ((grid) -> box_ranks)
#define hypre_StructGridBaseAllBoxes(grid)  ((grid) -> base_all_boxes)
#define hypre_StructGridPIndex(grid)        ((grid) -> pindex)
#define hypre_StructGridPStride(grid)       ((grid) -> pstride)
#define hypre_StructGridAlloced(grid)       ((grid) -> alloced)
#define hypre_StructGridRefCount(grid)      ((grid) -> ref_count)

#define hypre_StructGridBox(grid, i) \
(hypre_BoxArrayBox(hypre_StructGridBoxes(grid), i))
#define hypre_StructGridNumBoxes(grid) \
(hypre_BoxArraySize(hypre_StructGridBoxes(grid)))

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/
 
#define hypre_ForStructGridBoxI(i, grid) \
hypre_ForBoxI(i, hypre_StructGridBoxes(grid))

#endif

/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef hypre_COMMUNICATION_HEADER
#define hypre_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_CommTypeEntry:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Index  imin;             /* global imin for the data */
   hypre_Index  imax;             /* global imin for the data */
   int          offset;           /* offset for the data */

   int          dim;              /* dimension of the communication */
   int          length_array[4];
   int          stride_array[4];

} hypre_CommTypeEntry;

/*--------------------------------------------------------------------------
 * hypre_CommType:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_CommTypeEntry  **comm_entries;
   int                    num_entries;

} hypre_CommType;

/*--------------------------------------------------------------------------
 * hypre_CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct
{
   int                    num_values;
   MPI_Comm               comm;

   int                    num_sends;
   int                    num_recvs;
   int                   *send_procs;
   int                   *recv_procs;

   /* remote communication information */
   hypre_CommType       **send_types;
   hypre_CommType       **recv_types;
   MPI_Datatype          *send_mpi_types;
   MPI_Datatype          *recv_mpi_types;

   /* local copy information */
   hypre_CommType        *copy_from_type;
   hypre_CommType        *copy_to_type;

} hypre_CommPkg;

/*--------------------------------------------------------------------------
 * CommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_CommPkg  *comm_pkg;
   double         *send_data;
   double         *recv_data;

   int             num_requests;
   MPI_Request    *requests;
   MPI_Status     *status;

#if defined(HYPRE_COMM_SIMPLE)
   double        **send_buffers;
   double        **recv_buffers;
   int            *send_sizes;
   int            *recv_sizes;
#endif

} hypre_CommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommTypeEntry
 *--------------------------------------------------------------------------*/
 
#define hypre_CommTypeEntryIMin(entry)          (entry -> imin)
#define hypre_CommTypeEntryIMax(entry)          (entry -> imax)
#define hypre_CommTypeEntryOffset(entry)        (entry -> offset)
#define hypre_CommTypeEntryDim(entry)           (entry -> dim)
#define hypre_CommTypeEntryLengthArray(entry)   (entry -> length_array)
#define hypre_CommTypeEntryStrideArray(entry)   (entry -> stride_array)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommType
 *--------------------------------------------------------------------------*/
 
#define hypre_CommTypeCommEntries(type)   (type -> comm_entries)
#define hypre_CommTypeCommEntry(type, i)  (type -> comm_entries[i])
#define hypre_CommTypeNumEntries(type)    (type -> num_entries)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_CommPkgNumValues(comm_pkg)       (comm_pkg -> num_values)
#define hypre_CommPkgComm(comm_pkg)            (comm_pkg -> comm)
                                               
#define hypre_CommPkgNumSends(comm_pkg)        (comm_pkg -> num_sends)
#define hypre_CommPkgNumRecvs(comm_pkg)        (comm_pkg -> num_recvs)
#define hypre_CommPkgSendProcs(comm_pkg)       (comm_pkg -> send_procs)
#define hypre_CommPkgSendProc(comm_pkg, i)     (comm_pkg -> send_procs[i])
#define hypre_CommPkgRecvProcs(comm_pkg)       (comm_pkg -> recv_procs)
#define hypre_CommPkgRecvProc(comm_pkg, i)     (comm_pkg -> recv_procs[i])

#define hypre_CommPkgSendTypes(comm_pkg)       (comm_pkg -> send_types)
#define hypre_CommPkgSendType(comm_pkg, i)     (comm_pkg -> send_types[i])
#define hypre_CommPkgRecvTypes(comm_pkg)       (comm_pkg -> recv_types)
#define hypre_CommPkgRecvType(comm_pkg, i)     (comm_pkg -> recv_types[i])
#define hypre_CommPkgSendMPITypes(comm_pkg)    (comm_pkg -> send_mpi_types)
#define hypre_CommPkgSendMPIType(comm_pkg, i)  (comm_pkg -> send_mpi_types[i])
#define hypre_CommPkgRecvMPITypes(comm_pkg)    (comm_pkg -> recv_mpi_types)
#define hypre_CommPkgRecvMPIType(comm_pkg, i)  (comm_pkg -> recv_mpi_types[i])

#define hypre_CommPkgCopyFromType(comm_pkg)    (comm_pkg -> copy_from_type)
#define hypre_CommPkgCopyToType(comm_pkg)      (comm_pkg -> copy_to_type)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_CommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_CommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_CommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_CommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_CommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_CommHandleStatus(comm_handle)      (comm_handle -> status)
#if defined(HYPRE_COMM_SIMPLE)
#define hypre_CommHandleSendBuffers(comm_handle) (comm_handle -> send_buffers)
#define hypre_CommHandleRecvBuffers(comm_handle) (comm_handle -> recv_buffers)
#define hypre_CommHandleSendSizes(comm_handle)   (comm_handle -> send_sizes)
#define hypre_CommHandleRecvSizes(comm_handle)   (comm_handle -> recv_sizes)
#endif

#endif
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
 * Header info for computation
 *
 *****************************************************************************/

#ifndef hypre_COMPUTATION_HEADER
#define hypre_COMPUTATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_ComputePkg:
 *   Structure containing information for doing computations.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_CommPkg         *comm_pkg;

   hypre_BoxArrayArray   *indt_boxes;
   hypre_BoxArrayArray   *dept_boxes;
   hypre_Index            stride;

   hypre_StructGrid      *grid;
   hypre_BoxArray        *data_space;
   int                    num_values;

} hypre_ComputePkg;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ComputePkg
 *--------------------------------------------------------------------------*/
 
#define hypre_ComputePkgCommPkg(compute_pkg)      (compute_pkg -> comm_pkg)

#define hypre_ComputePkgIndtBoxes(compute_pkg)    (compute_pkg -> indt_boxes)
#define hypre_ComputePkgDeptBoxes(compute_pkg)    (compute_pkg -> dept_boxes)
#define hypre_ComputePkgStride(compute_pkg)       (compute_pkg -> stride)

#define hypre_ComputePkgGrid(compute_pkg)         (compute_pkg -> grid)
#define hypre_ComputePkgDataSpace(compute_pkg)    (compute_pkg -> data_space)
#define hypre_ComputePkgNumValues(compute_pkg)    (compute_pkg -> num_values)

#endif
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
 * Header info for the hypre_StructMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_MATRIX_HEADER
#define hypre_STRUCT_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   hypre_StructGrid     *grid;
   hypre_StructStencil  *user_stencil;
   hypre_StructStencil  *stencil;
   int                   num_values;   /* Number of "stored" coefficients */

   hypre_BoxArray       *data_space;

   double               *data;         /* Pointer to matrix data */
   int                   data_alloced; /* Boolean used for freeing data */
   int                   data_size;    /* Size of matrix data */
   int                 **data_indices; /* num-boxes by stencil-size array
                                          of indices into the data array.
                                          data_indices[b][s] is the starting
                                          index of matrix data corresponding
                                          to box b and stencil coefficient s */
                      
   int                   symmetric;    /* Is the matrix symmetric */
   int                  *symm_elements;/* Which elements are "symmetric" */
   int                   num_ghost[6]; /* Num ghost layers in each direction */
                      
   int                   global_size;  /* Total number of nonzero coeffs */

   hypre_CommPkg        *comm_pkg;     /* Info on how to update ghost data */

   int                   ref_count;

} hypre_StructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_StructMatrixComm(matrix)          ((matrix) -> comm)
#define hypre_StructMatrixGrid(matrix)          ((matrix) -> grid)
#define hypre_StructMatrixUserStencil(matrix)   ((matrix) -> user_stencil)
#define hypre_StructMatrixStencil(matrix)       ((matrix) -> stencil)
#define hypre_StructMatrixNumValues(matrix)     ((matrix) -> num_values)
#define hypre_StructMatrixDataSpace(matrix)     ((matrix) -> data_space)
#define hypre_StructMatrixData(matrix)          ((matrix) -> data)
#define hypre_StructMatrixDataAlloced(matrix)   ((matrix) -> data_alloced)
#define hypre_StructMatrixDataSize(matrix)      ((matrix) -> data_size)
#define hypre_StructMatrixDataIndices(matrix)   ((matrix) -> data_indices)
#define hypre_StructMatrixSymmetric(matrix)     ((matrix) -> symmetric)
#define hypre_StructMatrixSymmElements(matrix)  ((matrix) -> symm_elements)
#define hypre_StructMatrixNumGhost(matrix)      ((matrix) -> num_ghost)
#define hypre_StructMatrixGlobalSize(matrix)    ((matrix) -> global_size)
#define hypre_StructMatrixCommPkg(matrix)       ((matrix) -> comm_pkg)
#define hypre_StructMatrixRefCount(matrix)      ((matrix) -> ref_count)

#define hypre_StructMatrixBox(matrix, b) \
hypre_BoxArrayBox(hypre_StructMatrixDataSpace(matrix), b)

#define hypre_StructMatrixBoxData(matrix, b, s) \
(hypre_StructMatrixData(matrix) + hypre_StructMatrixDataIndices(matrix)[b][s])

#define hypre_StructMatrixBoxDataValue(matrix, b, s, index) \
(hypre_StructMatrixBoxData(matrix, b, s) + \
 hypre_BoxIndexRank(hypre_StructMatrixBox(matrix, b), index))

#endif
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
 * Header info for the hypre_StructVector structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_VECTOR_HEADER
#define hypre_STRUCT_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   hypre_StructGrid     *grid;

   hypre_BoxArray       *data_space;

   double               *data;         /* Pointer to vector data */
   int                   data_alloced; /* Boolean used for freeing data */
   int                   data_size;    /* Size of vector data */
   int                  *data_indices; /* num-boxes array of indices into
                                          the data array.  data_indices[b]
                                          is the starting index of vector
                                          data corresponding to box b. */
                      
   int                   num_ghost[6]; /* Num ghost layers in each direction */
                      
   int                   global_size;  /* Total number coefficients */

   int                   ref_count;

} hypre_StructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructVector
 *--------------------------------------------------------------------------*/

#define hypre_StructVectorComm(vector)          ((vector) -> comm)
#define hypre_StructVectorGrid(vector)          ((vector) -> grid)
#define hypre_StructVectorDataSpace(vector)     ((vector) -> data_space)
#define hypre_StructVectorData(vector)          ((vector) -> data)
#define hypre_StructVectorDataAlloced(vector)   ((vector) -> data_alloced)
#define hypre_StructVectorDataSize(vector)      ((vector) -> data_size)
#define hypre_StructVectorDataIndices(vector)   ((vector) -> data_indices)
#define hypre_StructVectorNumGhost(vector)      ((vector) -> num_ghost)
#define hypre_StructVectorGlobalSize(vector)    ((vector) -> global_size)
#define hypre_StructVectorRefCount(vector)      ((vector) -> ref_count)
 
#define hypre_StructVectorBox(vector, b) \
hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), b)
 
#define hypre_StructVectorBoxData(vector, b) \
(hypre_StructVectorData(vector) + hypre_StructVectorDataIndices(vector)[b])
 
#define hypre_StructVectorBoxDataValue(vector, b, index) \
(hypre_StructVectorBoxData(vector, b) + \
 hypre_BoxIndexRank(hypre_StructVectorBox(vector, b), index))

#endif
# define	P(s) s

/* F90_HYPRE_struct_grid.c */
void hypre_F90_IFACE P((int hypre_newstructgrid ));
void hypre_F90_IFACE P((int hypre_freestructgrid ));
void hypre_F90_IFACE P((int hypre_setstructgridextents ));
void hypre_F90_IFACE P((int hypre_setstructgridperiodic ));
void hypre_F90_IFACE P((int hypre_assemblestructgrid ));

/* F90_HYPRE_struct_matrix.c */
void hypre_F90_IFACE P((int hypre_newstructmatrix ));
void hypre_F90_IFACE P((int hypre_freestructmatrix ));
void hypre_F90_IFACE P((int hypre_initializestructmatrix ));
void hypre_F90_IFACE P((int hypre_setstructmatrixvalues ));
void hypre_F90_IFACE P((int hypre_setstructmatrixboxvalues ));
void hypre_F90_IFACE P((int hypre_assemblestructmatrix ));
void hypre_F90_IFACE P((int hypre_setstructmatrixnumghost ));
void hypre_F90_IFACE P((int hypre_structmatrixgrid ));
void hypre_F90_IFACE P((int hypre_setstructmatrixsymmetric ));

/* F90_HYPRE_struct_stencil.c */
void hypre_F90_IFACE P((int hypre_newstructstencil ));
void hypre_F90_IFACE P((int hypre_setstructstencilelement ));
void hypre_F90_IFACE P((int hypre_freestructstencil ));

/* F90_HYPRE_struct_vector.c */
void hypre_F90_IFACE P((int hypre_newstructvector ));
void hypre_F90_IFACE P((int hypre_freestructvector ));
void hypre_F90_IFACE P((int hypre_initializestructvector ));
void hypre_F90_IFACE P((int hypre_setstructvectorvalues ));
void hypre_F90_IFACE P((int hypre_getstructvectorvalues ));
void hypre_F90_IFACE P((int hypre_setstructvectorboxvalues ));
void hypre_F90_IFACE P((int hypre_getstructvectorboxvalues ));
void hypre_F90_IFACE P((int hypre_assemblestructvector ));
void hypre_F90_IFACE P((int hypre_setstructvectornumghost ));
void hypre_F90_IFACE P((int hypre_setstructvectorconstantva ));
void hypre_F90_IFACE P((int hypre_getmigratestructvectorcom ));
void hypre_F90_IFACE P((int hypre_migratestructvector ));
void hypre_F90_IFACE P((int hypre_freecommpkg ));

/* HYPRE_struct_grid.c */
int HYPRE_NewStructGrid P((MPI_Comm comm , int dim , HYPRE_StructGrid *grid ));
int HYPRE_FreeStructGrid P((HYPRE_StructGrid grid ));
int HYPRE_SetStructGridExtents P((HYPRE_StructGrid grid , int *ilower , int *iupper ));
int HYPRE_SetStructGridPeriodic P((HYPRE_StructGrid grid , int *periodic ));
int HYPRE_AssembleStructGrid P((HYPRE_StructGrid grid ));

/* HYPRE_struct_matrix.c */
int HYPRE_NewStructMatrix P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructMatrix *matrix ));
int HYPRE_FreeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_InitializeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixValues P((HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_SetStructMatrixBoxValues P((HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_AssembleStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixNumGhost P((HYPRE_StructMatrix matrix , int *num_ghost ));
int HYPRE_StructMatrixGrid P((HYPRE_StructMatrix matrix , HYPRE_StructGrid *grid ));
int HYPRE_SetStructMatrixSymmetric P((HYPRE_StructMatrix matrix , int symmetric ));
int HYPRE_PrintStructMatrix P((char *filename , HYPRE_StructMatrix matrix , int all ));

/* HYPRE_struct_stencil.c */
int HYPRE_NewStructStencil P((int dim , int size , HYPRE_StructStencil *stencil ));
int HYPRE_SetStructStencilElement P((HYPRE_StructStencil stencil , int element_index , int *offset ));
int HYPRE_FreeStructStencil P((HYPRE_StructStencil stencil ));

/* HYPRE_struct_vector.c */
int HYPRE_NewStructVector P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructVector *vector ));
int HYPRE_FreeStructVector P((HYPRE_StructVector struct_vector ));
int HYPRE_InitializeStructVector P((HYPRE_StructVector vector ));
int HYPRE_SetStructVectorValues P((HYPRE_StructVector vector , int *grid_index , double values ));
int HYPRE_GetStructVectorValues P((HYPRE_StructVector vector , int *grid_index , double *values_ptr ));
int HYPRE_SetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , double *values ));
int HYPRE_GetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , double *values ));
int HYPRE_AssembleStructVector P((HYPRE_StructVector vector ));
int HYPRE_PrintStructVector P((char *filename , HYPRE_StructVector vector , int all ));
int HYPRE_SetStructVectorNumGhost P((HYPRE_StructVector vector , int *num_ghost ));
int HYPRE_SetStructVectorConstantValues P((HYPRE_StructVector vector , double values ));
int HYPRE_GetMigrateStructVectorCommPkg P((HYPRE_StructVector from_vector , HYPRE_StructVector to_vector , HYPRE_CommPkg *comm_pkg ));
int HYPRE_MigrateStructVector P((HYPRE_CommPkg comm_pkg , HYPRE_StructVector from_vector , HYPRE_StructVector to_vector ));
int HYPRE_FreeCommPkg P((HYPRE_CommPkg comm_pkg ));

/* box.c */
hypre_Box *hypre_NewBox P((void ));
int hypre_SetBoxExtents P((hypre_Box *box , hypre_Index imin , hypre_Index imax ));
hypre_BoxArray *hypre_NewBoxArray P((int size ));
int hypre_SetBoxArraySize P((hypre_BoxArray *box_array , int size ));
hypre_BoxArrayArray *hypre_NewBoxArrayArray P((int size ));
int hypre_FreeBox P((hypre_Box *box ));
int hypre_FreeBoxArray P((hypre_BoxArray *box_array ));
int hypre_FreeBoxArrayArray P((hypre_BoxArrayArray *box_array_array ));
hypre_Box *hypre_DuplicateBox P((hypre_Box *box ));
hypre_BoxArray *hypre_DuplicateBoxArray P((hypre_BoxArray *box_array ));
hypre_BoxArrayArray *hypre_DuplicateBoxArrayArray P((hypre_BoxArrayArray *box_array_array ));
int hypre_AppendBox P((hypre_Box *box , hypre_BoxArray *box_array ));
int hypre_DeleteBox P((hypre_BoxArray *box_array , int index ));
int hypre_AppendBoxArray P((hypre_BoxArray *box_array_0 , hypre_BoxArray *box_array_1 ));
int hypre_GetBoxSize P((hypre_Box *box , hypre_Index size ));
int hypre_GetStrideBoxSize P((hypre_Box *box , hypre_Index stride , hypre_Index size ));

/* box_algebra.c */
int hypre_IntersectBoxes P((hypre_Box *box1 , hypre_Box *box2 , hypre_Box *ibox ));
int hypre_SubtractBoxes P((hypre_Box *box1 , hypre_Box *box2 , hypre_BoxArray *box_array ));
int hypre_UnionBoxArray P((hypre_BoxArray *boxes ));

/* box_alloc.c */
int hypre_InitializeBoxMemory P((const int at_a_time ));
int hypre_FinalizeBoxMemory P((void ));
hypre_Box *hypre_BoxAlloc P((void ));
int hypre_BoxFree P((hypre_Box *box ));

/* box_data.c */
void hypre_CopyBoxArrayData P((hypre_BoxArray *box_array_in , hypre_BoxArray *data_space_in , int num_values_in , double *data_in , hypre_BoxArray *box_array_out , hypre_BoxArray *data_space_out , int num_values_out , double *data_out ));

/* box_neighbors.c */
hypre_RankLink *hypre_NewRankLink P((int rank ));
int hypre_FreeRankLink P((hypre_RankLink *rank_link ));
hypre_BoxNeighbors *hypre_NewBoxNeighbors P((int *local_ranks , int num_local , hypre_BoxArray *boxes , int *processes , int max_distance ));
int hypre_FreeBoxNeighbors P((hypre_BoxNeighbors *neighbors ));

/* communication.c */
hypre_CommPkg *hypre_NewCommPkg P((hypre_BoxArrayArray *send_boxes , hypre_BoxArrayArray *recv_boxes , hypre_Index send_stride , hypre_Index recv_stride , hypre_BoxArray *send_data_space , hypre_BoxArray *recv_data_space , int **send_processes , int **recv_processes , int num_values , MPI_Comm comm ));
int hypre_FreeCommPkg P((hypre_CommPkg *comm_pkg ));
int hypre_InitializeCommunication P((hypre_CommPkg *comm_pkg , double *send_data , double *recv_data , hypre_CommHandle **comm_handle_ptr ));
int hypre_InitializeCommunication P((hypre_CommPkg *comm_pkg , double *send_data , double *recv_data , hypre_CommHandle **comm_handle_ptr ));
int hypre_FinalizeCommunication P((hypre_CommHandle *comm_handle ));
int hypre_FinalizeCommunication P((hypre_CommHandle *comm_handle ));
int hypre_ExchangeLocalData P((hypre_CommPkg *comm_pkg , double *send_data , double *recv_data ));
hypre_CommType *hypre_NewCommType P((hypre_CommTypeEntry **comm_entries , int num_entries ));
int hypre_FreeCommType P((hypre_CommType *comm_type ));
hypre_CommTypeEntry *hypre_NewCommTypeEntry P((hypre_Box *box , hypre_Index stride , hypre_Box *data_box , int num_values , int data_box_offset ));
int hypre_FreeCommTypeEntry P((hypre_CommTypeEntry *comm_entry ));
int hypre_NewCommPkgInfo P((hypre_BoxArrayArray *boxes , hypre_Index stride , hypre_BoxArray *data_space , int **processes , int num_values , MPI_Comm comm , int *num_comms_ptr , int **comm_processes_ptr , hypre_CommType ***comm_types_ptr , hypre_CommType **copy_type_ptr ));
int hypre_SortCommType P((hypre_CommType *comm_type ));
int hypre_CommitCommPkg P((hypre_CommPkg *comm_pkg ));
int hypre_UnCommitCommPkg P((hypre_CommPkg *comm_pkg ));
int hypre_BuildCommMPITypes P((int num_comms , int *comm_procs , hypre_CommType **comm_types , MPI_Datatype *comm_mpi_types ));
int hypre_BuildCommEntryMPIType P((hypre_CommTypeEntry *comm_entry , MPI_Datatype *comm_entry_mpi_type ));

/* communication_info.c */
int hypre_NewCommInfoFromStencil P((hypre_StructGrid *grid , hypre_StructStencil *stencil , hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr ));
int hypre_NewCommInfoFromNumGhost P((hypre_StructGrid *grid , int *num_ghost , hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr ));
int hypre_NewCommInfoFromGrids P((hypre_StructGrid *from_grid , hypre_StructGrid *to_grid , hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr ));

/* computation.c */
int hypre_GetComputeInfo P((hypre_StructGrid *grid , hypre_StructStencil *stencil , hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr , hypre_BoxArrayArray **indt_boxes_ptr , hypre_BoxArrayArray **dept_boxes_ptr ));
int hypre_NewComputePkg P((hypre_BoxArrayArray *send_boxes , hypre_BoxArrayArray *recv_boxes , hypre_Index send_stride , hypre_Index recv_stride , int **send_processes , int **recv_processes , hypre_BoxArrayArray *indt_boxes , hypre_BoxArrayArray *dept_boxes , hypre_Index stride , hypre_StructGrid *grid , hypre_BoxArray *data_space , int num_values , hypre_ComputePkg **compute_pkg_ptr ));
int hypre_FreeComputePkg P((hypre_ComputePkg *compute_pkg ));
int hypre_InitializeIndtComputations P((hypre_ComputePkg *compute_pkg , double *data , hypre_CommHandle **comm_handle_ptr ));
int hypre_FinalizeIndtComputations P((hypre_CommHandle *comm_handle ));

/* grow.c */
hypre_BoxArray *hypre_GrowBoxByStencil P((hypre_Box *box , hypre_StructStencil *stencil , int transpose ));
hypre_BoxArrayArray *hypre_GrowBoxArrayByStencil P((hypre_BoxArray *box_array , hypre_StructStencil *stencil , int transpose ));

/* project.c */
int hypre_ProjectBox P((hypre_Box *box , hypre_Index index , hypre_Index stride ));
int hypre_ProjectBoxArray P((hypre_BoxArray *box_array , hypre_Index index , hypre_Index stride ));
int hypre_ProjectBoxArrayArray P((hypre_BoxArrayArray *box_array_array , hypre_Index index , hypre_Index stride ));

/* struct_axpy.c */
int hypre_StructAxpy P((double alpha , hypre_StructVector *x , hypre_StructVector *y ));

/* struct_copy.c */
int hypre_StructCopy P((hypre_StructVector *x , hypre_StructVector *y ));

/* struct_grid.c */
hypre_StructGrid *hypre_NewStructGrid P((MPI_Comm comm , int dim ));
hypre_StructGrid *hypre_RefStructGrid P((hypre_StructGrid *grid ));
int hypre_FreeStructGrid P((hypre_StructGrid *grid ));
int hypre_SetStructGridPeriodic P((hypre_StructGrid *grid , hypre_Index periodic ));
int hypre_SetStructGridExtents P((hypre_StructGrid *grid , hypre_Index ilower , hypre_Index iupper ));
int hypre_SetStructGridBoxes P((hypre_StructGrid *grid , hypre_BoxArray *boxes ));
int hypre_SetStructGridGlobalInfo P((hypre_StructGrid *grid , hypre_BoxArray *all_boxes , int *processes , int *box_ranks , hypre_BoxArray *base_all_boxes , hypre_Index pindex , hypre_Index pstride ));
int hypre_AssembleStructGrid P((hypre_StructGrid *grid ));
int hypre_GatherAllBoxes P((MPI_Comm comm , hypre_BoxArray *boxes , hypre_BoxArray **all_boxes_ptr , int **processes_ptr , int **box_ranks_ptr ));
int hypre_PrintStructGrid P((FILE *file , hypre_StructGrid *grid ));
hypre_StructGrid *hypre_ReadStructGrid P((MPI_Comm comm , FILE *file ));

/* struct_innerprod.c */
double hypre_StructInnerProd P((hypre_StructVector *x , hypre_StructVector *y ));

/* struct_io.c */
int hypre_PrintBoxArrayData P((FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , int num_values , double *data ));
int hypre_ReadBoxArrayData P((FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , int num_values , double *data ));

/* struct_matrix.c */
double *hypre_StructMatrixExtractPointerByIndex P((hypre_StructMatrix *matrix , int b , hypre_Index index ));
hypre_StructMatrix *hypre_NewStructMatrix P((MPI_Comm comm , hypre_StructGrid *grid , hypre_StructStencil *user_stencil ));
hypre_StructMatrix *hypre_RefStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_FreeStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_InitializeStructMatrixShell P((hypre_StructMatrix *matrix ));
int hypre_InitializeStructMatrixData P((hypre_StructMatrix *matrix , double *data ));
int hypre_InitializeStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_SetStructMatrixValues P((hypre_StructMatrix *matrix , hypre_Index grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int hypre_SetStructMatrixBoxValues P((hypre_StructMatrix *matrix , hypre_Box *value_box , int num_stencil_indices , int *stencil_indices , double *values ));
int hypre_AssembleStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_SetStructMatrixNumGhost P((hypre_StructMatrix *matrix , int *num_ghost ));
int hypre_PrintStructMatrix P((char *filename , hypre_StructMatrix *matrix , int all ));
int hypre_MigrateStructMatrix P((hypre_StructMatrix *from_matrix , hypre_StructMatrix *to_matrix ));
hypre_StructMatrix *hypre_ReadStructMatrix P((MPI_Comm comm , char *filename , int *num_ghost ));

/* struct_matrix_mask.c */
hypre_StructMatrix *hypre_NewStructMatrixMask P((hypre_StructMatrix *matrix , int num_stencil_indices , int *stencil_indices ));

/* struct_matvec.c */
void *hypre_StructMatvecInitialize P((void ));
int hypre_StructMatvecSetup P((void *matvec_vdata , hypre_StructMatrix *A , hypre_StructVector *x ));
int hypre_StructMatvecCompute P((void *matvec_vdata , double alpha , hypre_StructMatrix *A , hypre_StructVector *x , double beta , hypre_StructVector *y ));
int hypre_StructMatvecFinalize P((void *matvec_vdata ));
int hypre_StructMatvec P((double alpha , hypre_StructMatrix *A , hypre_StructVector *x , double beta , hypre_StructVector *y ));

/* struct_scale.c */
int hypre_StructScale P((double alpha , hypre_StructVector *y ));

/* struct_stencil.c */
hypre_StructStencil *hypre_NewStructStencil P((int dim , int size , hypre_Index *shape ));
hypre_StructStencil *hypre_RefStructStencil P((hypre_StructStencil *stencil ));
int hypre_FreeStructStencil P((hypre_StructStencil *stencil ));
int hypre_StructStencilElementRank P((hypre_StructStencil *stencil , hypre_Index stencil_element ));
int hypre_SymmetrizeStructStencil P((hypre_StructStencil *stencil , hypre_StructStencil **symm_stencil_ptr , int **symm_elements_ptr ));

/* struct_vector.c */
hypre_StructVector *hypre_NewStructVector P((MPI_Comm comm , hypre_StructGrid *grid ));
hypre_StructVector *hypre_RefStructVector P((hypre_StructVector *vector ));
int hypre_FreeStructVector P((hypre_StructVector *vector ));
int hypre_InitializeStructVectorShell P((hypre_StructVector *vector ));
int hypre_InitializeStructVectorData P((hypre_StructVector *vector , double *data ));
int hypre_InitializeStructVector P((hypre_StructVector *vector ));
int hypre_SetStructVectorValues P((hypre_StructVector *vector , hypre_Index grid_index , double values ));
int hypre_GetStructVectorValues P((hypre_StructVector *vector , hypre_Index grid_index , double *values_ptr ));
int hypre_SetStructVectorBoxValues P((hypre_StructVector *vector , hypre_Box *value_box , double *values ));
int hypre_GetStructVectorBoxValues P((hypre_StructVector *vector , hypre_Box *value_box , double *values ));
int hypre_SetStructVectorNumGhost P((hypre_StructVector *vector , int *num_ghost ));
int hypre_AssembleStructVector P((hypre_StructVector *vector ));
int hypre_SetStructVectorConstantValues P((hypre_StructVector *vector , double values ));
int hypre_ClearStructVectorGhostValues P((hypre_StructVector *vector ));
int hypre_ClearStructVectorAllValues P((hypre_StructVector *vector ));
hypre_CommPkg *hypre_GetMigrateStructVectorCommPkg P((hypre_StructVector *from_vector , hypre_StructVector *to_vector ));
int hypre_MigrateStructVector P((hypre_CommPkg *comm_pkg , hypre_StructVector *from_vector , hypre_StructVector *to_vector ));
int hypre_PrintStructVector P((char *filename , hypre_StructVector *vector , int all ));
hypre_StructVector *hypre_ReadStructVector P((MPI_Comm comm , char *filename , int *num_ghost ));

/* thread_wrappers.c */
void HYPRE_NewStructGridVoidPtr P((void *argptr ));
int HYPRE_NewStructGridPush P((MPI_Comm comm , int dim , HYPRE_StructGridArray *grid ));
void HYPRE_FreeStructGridVoidPtr P((void *argptr ));
int HYPRE_FreeStructGridPush P((HYPRE_StructGridArray grid ));
void HYPRE_SetStructGridExtentsVoidPtr P((void *argptr ));
int HYPRE_SetStructGridExtentsPush P((HYPRE_StructGridArray grid , int *ilower , int *iupper ));
void HYPRE_SetStructGridPeriodicVoidPtr P((void *argptr ));
int HYPRE_SetStructGridPeriodicPush P((HYPRE_StructGridArray grid , int *periodic ));
void HYPRE_AssembleStructGridVoidPtr P((void *argptr ));
int HYPRE_AssembleStructGridPush P((HYPRE_StructGridArray grid ));
void HYPRE_NewStructMatrixVoidPtr P((void *argptr ));
int HYPRE_NewStructMatrixPush P((MPI_Comm comm , HYPRE_StructGridArray grid , HYPRE_StructStencilArray stencil , HYPRE_StructMatrixArray *matrix ));
void HYPRE_FreeStructMatrixVoidPtr P((void *argptr ));
int HYPRE_FreeStructMatrixPush P((HYPRE_StructMatrixArray matrix ));
void HYPRE_InitializeStructMatrixVoidPtr P((void *argptr ));
int HYPRE_InitializeStructMatrixPush P((HYPRE_StructMatrixArray matrix ));
void HYPRE_SetStructMatrixValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructMatrixValuesPush P((HYPRE_StructMatrixArray matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
void HYPRE_SetStructMatrixBoxValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructMatrixBoxValuesPush P((HYPRE_StructMatrixArray matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
void HYPRE_AssembleStructMatrixVoidPtr P((void *argptr ));
int HYPRE_AssembleStructMatrixPush P((HYPRE_StructMatrixArray matrix ));
void HYPRE_SetStructMatrixNumGhostVoidPtr P((void *argptr ));
int HYPRE_SetStructMatrixNumGhostPush P((HYPRE_StructMatrixArray matrix , int *num_ghost ));
void HYPRE_StructMatrixGridVoidPtr P((void *argptr ));
int HYPRE_StructMatrixGridPush P((HYPRE_StructMatrixArray matrix , HYPRE_StructGridArray *grid ));
void HYPRE_SetStructMatrixSymmetricVoidPtr P((void *argptr ));
int HYPRE_SetStructMatrixSymmetricPush P((HYPRE_StructMatrixArray matrix , int symmetric ));
void HYPRE_PrintStructMatrixVoidPtr P((void *argptr ));
int HYPRE_PrintStructMatrixPush P((char *filename , HYPRE_StructMatrixArray matrix , int all ));
void HYPRE_NewStructStencilVoidPtr P((void *argptr ));
int HYPRE_NewStructStencilPush P((int dim , int size , HYPRE_StructStencilArray *stencil ));
void HYPRE_SetStructStencilElementVoidPtr P((void *argptr ));
int HYPRE_SetStructStencilElementPush P((HYPRE_StructStencilArray stencil , int element_index , int *offset ));
void HYPRE_FreeStructStencilVoidPtr P((void *argptr ));
int HYPRE_FreeStructStencilPush P((HYPRE_StructStencilArray stencil ));
void HYPRE_NewStructVectorVoidPtr P((void *argptr ));
int HYPRE_NewStructVectorPush P((MPI_Comm comm , HYPRE_StructGridArray grid , HYPRE_StructStencilArray stencil , HYPRE_StructVectorArray *vector ));
void HYPRE_FreeStructVectorVoidPtr P((void *argptr ));
int HYPRE_FreeStructVectorPush P((HYPRE_StructVectorArray struct_vector ));
void HYPRE_InitializeStructVectorVoidPtr P((void *argptr ));
int HYPRE_InitializeStructVectorPush P((HYPRE_StructVectorArray vector ));
void HYPRE_SetStructVectorValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructVectorValuesPush P((HYPRE_StructVectorArray vector , int *grid_index , double values ));
void HYPRE_GetStructVectorValuesVoidPtr P((void *argptr ));
int HYPRE_GetStructVectorValuesPush P((HYPRE_StructVectorArray vector , int *grid_index , double *values_ptr ));
void HYPRE_SetStructVectorBoxValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructVectorBoxValuesPush P((HYPRE_StructVectorArray vector , int *ilower , int *iupper , double *values ));
void HYPRE_GetStructVectorBoxValuesVoidPtr P((void *argptr ));
int HYPRE_GetStructVectorBoxValuesPush P((HYPRE_StructVectorArray vector , int *ilower , int *iupper , double *values ));
void HYPRE_AssembleStructVectorVoidPtr P((void *argptr ));
int HYPRE_AssembleStructVectorPush P((HYPRE_StructVectorArray vector ));
void HYPRE_PrintStructVectorVoidPtr P((void *argptr ));
int HYPRE_PrintStructVectorPush P((char *filename , HYPRE_StructVectorArray vector , int all ));
void HYPRE_SetStructVectorNumGhostVoidPtr P((void *argptr ));
int HYPRE_SetStructVectorNumGhostPush P((HYPRE_StructVectorArray vector , int *num_ghost ));
void HYPRE_SetStructVectorConstantValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructVectorConstantValuesPush P((HYPRE_StructVectorArray vector , double values ));
void HYPRE_GetMigrateStructVectorCommPkgVoidPtr P((void *argptr ));
int HYPRE_GetMigrateStructVectorCommPkgPush P((HYPRE_StructVectorArray from_vector , HYPRE_StructVectorArray to_vector , HYPRE_CommPkgArray *comm_pkg ));
void HYPRE_MigrateStructVectorVoidPtr P((void *argptr ));
int HYPRE_MigrateStructVectorPush P((HYPRE_CommPkgArray comm_pkg , HYPRE_StructVectorArray from_vector , HYPRE_StructVectorArray to_vector ));
void HYPRE_FreeCommPkgVoidPtr P((void *argptr ));
int HYPRE_FreeCommPkgPush P((HYPRE_CommPkgArray comm_pkg ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

