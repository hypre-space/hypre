
#include <HYPRE_config.h>

#include "HYPRE_struct_mv.h"

#ifndef hypre_STRUCT_MV_HEADER
#define hypre_STRUCT_MV_HEADER

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

typedef struct hypre_Box_struct
{
   hypre_Index imin;           /* min bounding indices */
   hypre_Index imax;           /* max bounding indices */

} hypre_Box;

/*--------------------------------------------------------------------------
 * hypre_BoxArray:
 *   An array of boxes.
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxArray_struct
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

typedef struct hypre_BoxArrayArray_struct
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

#define hypre_CopyToCleanIndex(in_index, ndim, out_index) \
{\
   int d;\
   for (d = 0; d < ndim; d++)\
   {\
      hypre_IndexD(out_index, d) = hypre_IndexD(in_index, d);\
   }\
   for (d = ndim; d < 3; d++)\
   {\
      hypre_IndexD(out_index, d) = 0;\
   }\
}

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

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *
 * NOTE: PThreads version of BoxLoop looping macros are in `box_pthreads.h'.
 *
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_USE_PTHREADS

#define hypre_BoxLoopDeclareS(dbox, stride, sx, sy, sz) \
int  sx = (hypre_IndexX(stride));\
int  sy = (hypre_IndexY(stride)*hypre_BoxSizeX(dbox));\
int  sz = (hypre_IndexZ(stride)*\
           hypre_BoxSizeX(dbox)*hypre_BoxSizeY(dbox))

#define hypre_BoxLoopDeclareN(loop_size) \
int  hypre__nx = hypre_IndexX(loop_size);\
int  hypre__ny = hypre_IndexY(loop_size);\
int  hypre__nz = hypre_IndexZ(loop_size);\
int  hypre__mx = hypre__nx;\
int  hypre__my = hypre__ny;\
int  hypre__mz = hypre__nz;\
int  hypre__dir, hypre__max;\
int  hypre__div, hypre__mod;\
int  hypre__block, hypre__num_blocks;\
hypre__dir = 0;\
hypre__max = hypre__nx;\
if (hypre__ny > hypre__max)\
{\
   hypre__dir = 1;\
   hypre__max = hypre__ny;\
}\
if (hypre__nz > hypre__max)\
{\
   hypre__dir = 2;\
   hypre__max = hypre__nz;\
}\
hypre__num_blocks = hypre_NumThreads();\
if (hypre__max < hypre__num_blocks)\
{\
   hypre__num_blocks = hypre__max;\
}\
if (hypre__num_blocks > 0)\
{\
   hypre__div = hypre__max / hypre__num_blocks;\
   hypre__mod = hypre__max % hypre__num_blocks;\
}

#define hypre_BoxLoopSet(i, j, k) \
i = 0;\
j = 0;\
k = 0;\
hypre__nx = hypre__mx;\
hypre__ny = hypre__my;\
hypre__nz = hypre__mz;\
if (hypre__num_blocks > 1)\
{\
   if (hypre__dir == 0)\
   {\
      i = hypre__block * hypre__div + hypre_min(hypre__mod, hypre__block);\
      hypre__nx = hypre__div + ((hypre__mod > hypre__block) ? 1 : 0);\
   }\
   else if (hypre__dir == 1)\
   {\
      j = hypre__block * hypre__div + hypre_min(hypre__mod, hypre__block);\
      hypre__ny = hypre__div + ((hypre__mod > hypre__block) ? 1 : 0);\
   }\
   else if (hypre__dir == 2)\
   {\
      k = hypre__block * hypre__div + hypre_min(hypre__mod, hypre__block);\
      hypre__nz = hypre__div + ((hypre__mod > hypre__block) ? 1 : 0);\
   }\
}

/*-----------------------------------*/

#define hypre_BoxLoop0Begin(loop_size)\
{\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop0For(i, j, k)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
   hypre_BoxLoopSet(i, j, k);\
   for (k = 0; k < hypre__nz; k++)\
   {\
      for (j = 0; j < hypre__ny; j++)\
      {\
         for (i = 0; i < hypre__nx; i++)\
         {

#define hypre_BoxLoop0End()\
         }\
      }\
   }\
   }\
}
  
/*-----------------------------------*/

#define hypre_BoxLoop1Begin(loop_size,\
			    dbox1, start1, stride1, i1)\
{\
   int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop1For(i, j, k, i1)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
   hypre_BoxLoopSet(i, j, k);\
   i1 = hypre__i1start + i*hypre__sx1 + j*hypre__sy1 + k*hypre__sz1;\
   for (k = 0; k < hypre__nz; k++)\
   {\
      for (j = 0; j < hypre__ny; j++)\
      {\
         for (i = 0; i < hypre__nx; i++)\
         {

#define hypre_BoxLoop1End(i1)\
            i1 += hypre__sx1;\
         }\
         i1 += hypre__sy1 - hypre__nx*hypre__sx1;\
      }\
      i1 += hypre__sz1 - hypre__ny*hypre__sy1;\
   }\
   }\
}
  
/*-----------------------------------*/

#define hypre_BoxLoop2Begin(loop_size,\
			    dbox1, start1, stride1, i1,\
			    dbox2, start2, stride2, i2)\
{\
   int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop2For(i, j, k, i1, i2)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
   hypre_BoxLoopSet(i, j, k);\
   i1 = hypre__i1start + i*hypre__sx1 + j*hypre__sy1 + k*hypre__sz1;\
   i2 = hypre__i2start + i*hypre__sx2 + j*hypre__sy2 + k*hypre__sz2;\
   for (k = 0; k < hypre__nz; k++)\
   {\
      for (j = 0; j < hypre__ny; j++)\
      {\
         for (i = 0; i < hypre__nx; i++)\
         {

#define hypre_BoxLoop2End(i1, i2)\
            i1 += hypre__sx1;\
            i2 += hypre__sx2;\
         }\
         i1 += hypre__sy1 - hypre__nx*hypre__sx1;\
         i2 += hypre__sy2 - hypre__nx*hypre__sx2;\
      }\
      i1 += hypre__sz1 - hypre__ny*hypre__sy1;\
      i2 += hypre__sz2 - hypre__ny*hypre__sy2;\
   }\
   }\
}

/*-----------------------------------*/

#define hypre_BoxLoop3Begin(loop_size,\
			    dbox1, start1, stride1, i1,\
			    dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3)\
{\
   int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);\
   int  hypre__i3start = hypre_BoxIndexRank(dbox3, start3);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);\
   hypre_BoxLoopDeclareS(dbox3, stride3, hypre__sx3, hypre__sy3, hypre__sz3);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop3For(i, j, k, i1, i2, i3)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
   hypre_BoxLoopSet(i, j, k);\
   i1 = hypre__i1start + i*hypre__sx1 + j*hypre__sy1 + k*hypre__sz1;\
   i2 = hypre__i2start + i*hypre__sx2 + j*hypre__sy2 + k*hypre__sz2;\
   i3 = hypre__i3start + i*hypre__sx3 + j*hypre__sy3 + k*hypre__sz3;\
   for (k = 0; k < hypre__nz; k++)\
   {\
      for (j = 0; j < hypre__ny; j++)\
      {\
         for (i = 0; i < hypre__nx; i++)\
         {

#define hypre_BoxLoop3End(i1, i2, i3)\
            i1 += hypre__sx1;\
            i2 += hypre__sx2;\
            i3 += hypre__sx3;\
         }\
         i1 += hypre__sy1 - hypre__nx*hypre__sx1;\
         i2 += hypre__sy2 - hypre__nx*hypre__sx2;\
         i3 += hypre__sy3 - hypre__nx*hypre__sx3;\
      }\
      i1 += hypre__sz1 - hypre__ny*hypre__sy1;\
      i2 += hypre__sz2 - hypre__ny*hypre__sy2;\
      i3 += hypre__sz3 - hypre__ny*hypre__sy3;\
   }\
   }\
}

/*-----------------------------------*/

#define hypre_BoxLoop4Begin(loop_size,\
			    dbox1, start1, stride1, i1,\
			    dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3,\
                            dbox4, start4, stride4, i4)\
{\
   int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);\
   int  hypre__i3start = hypre_BoxIndexRank(dbox3, start3);\
   int  hypre__i4start = hypre_BoxIndexRank(dbox4, start4);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);\
   hypre_BoxLoopDeclareS(dbox3, stride3, hypre__sx3, hypre__sy3, hypre__sz3);\
   hypre_BoxLoopDeclareS(dbox4, stride4, hypre__sx4, hypre__sy4, hypre__sz4);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop4For(i, j, k, i1, i2, i3, i4)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
   hypre_BoxLoopSet(i, j, k);\
   i1 = hypre__i1start + i*hypre__sx1 + j*hypre__sy1 + k*hypre__sz1;\
   i2 = hypre__i2start + i*hypre__sx2 + j*hypre__sy2 + k*hypre__sz2;\
   i3 = hypre__i3start + i*hypre__sx3 + j*hypre__sy3 + k*hypre__sz3;\
   i4 = hypre__i4start + i*hypre__sx4 + j*hypre__sy4 + k*hypre__sz4;\
   for (k = 0; k < hypre__nz; k++)\
   {\
      for (j = 0; j < hypre__ny; j++)\
      {\
         for (i = 0; i < hypre__nx; i++)\
         {

#define hypre_BoxLoop4End(i1, i2, i3, i4)\
            i1 += hypre__sx1;\
            i2 += hypre__sx2;\
            i3 += hypre__sx3;\
            i4 += hypre__sx4;\
         }\
         i1 += hypre__sy1 - hypre__nx*hypre__sx1;\
         i2 += hypre__sy2 - hypre__nx*hypre__sx2;\
         i3 += hypre__sy3 - hypre__nx*hypre__sx3;\
         i4 += hypre__sy4 - hypre__nx*hypre__sx4;\
      }\
      i1 += hypre__sz1 - hypre__ny*hypre__sy1;\
      i2 += hypre__sz2 - hypre__ny*hypre__sy2;\
      i3 += hypre__sz3 - hypre__ny*hypre__sy3;\
      i4 += hypre__sz4 - hypre__ny*hypre__sy4;\
   }\
   }\
}

/*-----------------------------------*/

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

#define hypre_BoxLoopDeclare(loop_size, data_box, stride, iinc, jinc, kinc) \
int  iinc = (hypre_IndexX(stride));\
int  jinc = (hypre_IndexY(stride)*hypre_BoxSizeX(data_box) -\
             hypre_IndexX(loop_size)*hypre_IndexX(stride));\
int  kinc = (hypre_IndexZ(stride)*\
             hypre_BoxSizeX(data_box)*hypre_BoxSizeY(data_box) -\
             hypre_IndexY(loop_size)*\
             hypre_IndexY(stride)*hypre_BoxSizeX(data_box))

#define vol_cbrt(vol) (int) pow((double)(vol), 1. / 3.) 

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

#define hypre_BoxLoop0Begin(loop_size)\
{\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
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
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
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
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
   int orig_i2 = hypre_BoxIndexRank(data_box2, start2);\
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
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size);\
   int orig_i1 = hypre_BoxIndexRank(data_box1, start1);\
   int orig_i2 = hypre_BoxIndexRank(data_box2, start2);\
   int orig_i3 = hypre_BoxIndexRank(data_box3, start3);\
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

typedef struct hypre_RankLink_struct
{
   int                           rank;
   struct hypre_RankLink_struct *next;

} hypre_RankLink;

typedef hypre_RankLink *hypre_RankLinkArray[3][3][3];

/*--------------------------------------------------------------------------
 * hypre_BoxNeighbors:
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxNeighbors_struct
{
   hypre_BoxArray      *boxes;            /* boxes in the neighborhood */
   int                 *procs;            /* procs for 'boxes' */
   int                 *ids;              /* ids for 'boxes' */
   int                  first_local;      /* first local box address */
   int                  num_local;        /* number of local boxes */
   int                  num_periodic;     /* number of periodic boxes */

   hypre_RankLinkArray *rank_links;      /* neighbors of local boxes */

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

#define hypre_BoxNeighborsBoxes(neighbors)       ((neighbors) -> boxes)
#define hypre_BoxNeighborsProcs(neighbors)       ((neighbors) -> procs)
#define hypre_BoxNeighborsIDs(neighbors)         ((neighbors) -> ids)
#define hypre_BoxNeighborsFirstLocal(neighbors)  ((neighbors) -> first_local)
#define hypre_BoxNeighborsNumLocal(neighbors)    ((neighbors) -> num_local)
#define hypre_BoxNeighborsNumPeriodic(neighbors) ((neighbors) -> num_periodic)
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
 * Header info for the hypre_StructGrid structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_GRID_HEADER
#define hypre_STRUCT_GRID_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructGrid:
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructGrid_struct
{
   MPI_Comm             comm;
                      
   int                  dim;          /* Number of grid dimensions */
                      
   hypre_BoxArray      *boxes;        /* Array of boxes in this process */
   int                 *ids;          /* Unique IDs for boxes */
                      
   hypre_BoxNeighbors  *neighbors;    /* Neighbors of boxes */
   int                  max_distance; /* Neighborhood size */

   hypre_Box           *bounding_box; /* Bounding box around grid */

   int                  local_size;   /* Number of grid points locally */
   int                  global_size;  /* Total number of grid points */

   hypre_Index          periodic;     /* Indicates if grid is periodic */

   int                  ref_count;

} hypre_StructGrid;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructGrid
 *--------------------------------------------------------------------------*/

#define hypre_StructGridComm(grid)          ((grid) -> comm)
#define hypre_StructGridDim(grid)           ((grid) -> dim)
#define hypre_StructGridBoxes(grid)         ((grid) -> boxes)
#define hypre_StructGridIDs(grid)           ((grid) -> ids)
#define hypre_StructGridNeighbors(grid)     ((grid) -> neighbors)
#define hypre_StructGridMaxDistance(grid)   ((grid) -> max_distance)
#define hypre_StructGridBoundingBox(grid)   ((grid) -> bounding_box)
#define hypre_StructGridLocalSize(grid)     ((grid) -> local_size)
#define hypre_StructGridGlobalSize(grid)    ((grid) -> global_size)
#define hypre_StructGridPeriodic(grid)      ((grid) -> periodic)
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

typedef struct hypre_StructStencil_struct
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

#ifndef hypre_COMMUNICATION_HEADER
#define hypre_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_CommTypeEntry:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommTypeEntry_struct
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

typedef struct hypre_CommType_struct
{
   hypre_CommTypeEntry  **comm_entries;
   int                    num_entries;

} hypre_CommType;

/*--------------------------------------------------------------------------
 * hypre_CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommPkg_struct
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

typedef struct hypre_CommHandle_struct
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

typedef struct hypre_ComputePkg_struct
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

typedef struct hypre_StructMatrix_struct
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

typedef struct hypre_StructVector_struct
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

/* HYPRE_struct_grid.c */
int HYPRE_StructGridCreate( MPI_Comm comm , int dim , HYPRE_StructGrid *grid );
int HYPRE_StructGridDestroy( HYPRE_StructGrid grid );
int HYPRE_StructGridSetExtents( HYPRE_StructGrid grid , int *ilower , int *iupper );
int HYPRE_StructGridSetPeriodic( HYPRE_StructGrid grid , int *periodic );
int HYPRE_StructGridAssemble( HYPRE_StructGrid grid );

/* HYPRE_struct_matrix.c */
int HYPRE_StructMatrixCreate( MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructMatrix *matrix );
int HYPRE_StructMatrixDestroy( HYPRE_StructMatrix matrix );
int HYPRE_StructMatrixInitialize( HYPRE_StructMatrix matrix );
int HYPRE_StructMatrixSetValues( HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values );
int HYPRE_StructMatrixSetBoxValues( HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values );
int HYPRE_StructMatrixAddToValues( HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values );
int HYPRE_StructMatrixAddToBoxValues( HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values );
int HYPRE_StructMatrixAssemble( HYPRE_StructMatrix matrix );
int HYPRE_StructMatrixSetNumGhost( HYPRE_StructMatrix matrix , int *num_ghost );
int HYPRE_StructMatrixGetGrid( HYPRE_StructMatrix matrix , HYPRE_StructGrid *grid );
int HYPRE_StructMatrixSetSymmetric( HYPRE_StructMatrix matrix , int symmetric );
int HYPRE_StructMatrixPrint( char *filename , HYPRE_StructMatrix matrix , int all );

/* HYPRE_struct_stencil.c */
int HYPRE_StructStencilCreate( int dim , int size , HYPRE_StructStencil *stencil );
int HYPRE_StructStencilSetElement( HYPRE_StructStencil stencil , int element_index , int *offset );
int HYPRE_StructStencilDestroy( HYPRE_StructStencil stencil );

/* HYPRE_struct_vector.c */
int HYPRE_StructVectorCreate( MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructVector *vector );
int HYPRE_StructVectorDestroy( HYPRE_StructVector struct_vector );
int HYPRE_StructVectorInitialize( HYPRE_StructVector vector );
int HYPRE_StructVectorSetValues( HYPRE_StructVector vector , int *grid_index , double values );
int HYPRE_StructVectorSetBoxValues( HYPRE_StructVector vector , int *ilower , int *iupper , double *values );
int HYPRE_StructVectorAddToValues( HYPRE_StructVector vector , int *grid_index , double values );
int HYPRE_StructVectorAddToBoxValues( HYPRE_StructVector vector , int *ilower , int *iupper , double *values );
int HYPRE_StructVectorGetValues( HYPRE_StructVector vector , int *grid_index , double *values_ptr );
int HYPRE_StructVectorGetBoxValues( HYPRE_StructVector vector , int *ilower , int *iupper , double *values );
int HYPRE_StructVectorAssemble( HYPRE_StructVector vector );
int HYPRE_StructVectorPrint( char *filename , HYPRE_StructVector vector , int all );
int HYPRE_StructVectorSetNumGhost( HYPRE_StructVector vector , int *num_ghost );
int HYPRE_StructVectorSetConstantValues( HYPRE_StructVector vector , double values );
int HYPRE_StructVectorGetMigrateCommPkg( HYPRE_StructVector from_vector , HYPRE_StructVector to_vector , HYPRE_CommPkg *comm_pkg );
int HYPRE_StructVectorMigrate( HYPRE_CommPkg comm_pkg , HYPRE_StructVector from_vector , HYPRE_StructVector to_vector );
int HYPRE_CommPkgDestroy( HYPRE_CommPkg comm_pkg );

/* box.c */
hypre_Box *hypre_BoxCreate( void );
int hypre_BoxSetExtents( hypre_Box *box , hypre_Index imin , hypre_Index imax );
hypre_BoxArray *hypre_BoxArrayCreate( int size );
int hypre_BoxArraySetSize( hypre_BoxArray *box_array , int size );
hypre_BoxArrayArray *hypre_BoxArrayArrayCreate( int size );
int hypre_BoxDestroy( hypre_Box *box );
int hypre_BoxArrayDestroy( hypre_BoxArray *box_array );
int hypre_BoxArrayArrayDestroy( hypre_BoxArrayArray *box_array_array );
hypre_Box *hypre_BoxDuplicate( hypre_Box *box );
hypre_BoxArray *hypre_BoxArrayDuplicate( hypre_BoxArray *box_array );
hypre_BoxArrayArray *hypre_BoxArrayArrayDuplicate( hypre_BoxArrayArray *box_array_array );
int hypre_AppendBox( hypre_Box *box , hypre_BoxArray *box_array );
int hypre_DeleteBox( hypre_BoxArray *box_array , int index );
int hypre_AppendBoxArray( hypre_BoxArray *box_array_0 , hypre_BoxArray *box_array_1 );
int hypre_BoxGetSize( hypre_Box *box , hypre_Index size );
int hypre_BoxGetStrideSize( hypre_Box *box , hypre_Index stride , hypre_Index size );
int hypre_IModPeriod( int i , int period );
int hypre_IModPeriodX( hypre_Index index , hypre_Index periodic );
int hypre_IModPeriodY( hypre_Index index , hypre_Index periodic );
int hypre_IModPeriodZ( hypre_Index index , hypre_Index periodic );

/* box_algebra.c */
int hypre_IntersectBoxes( hypre_Box *box1 , hypre_Box *box2 , hypre_Box *ibox );
int hypre_SubtractBoxes( hypre_Box *box1 , hypre_Box *box2 , hypre_BoxArray *box_array );
int hypre_UnionBoxes( hypre_BoxArray *boxes );

/* box_alloc.c */
int hypre_BoxInitializeMemory( const int at_a_time );
int hypre_BoxFinalizeMemory( void );
hypre_Box *hypre_BoxAlloc( void );
int hypre_BoxFree( hypre_Box *box );

/* box_neighbors.c */
int hypre_RankLinkCreate( int rank , hypre_RankLink **rank_link_ptr );
int hypre_RankLinkDestroy( hypre_RankLink *rank_link );
int hypre_BoxNeighborsCreate( hypre_BoxArray *boxes , int *procs , int *ids , int first_local , int num_local , int num_periodic , hypre_BoxNeighbors **neighbors_ptr );
int hypre_BoxNeighborsAssemble( hypre_BoxNeighbors *neighbors , int max_distance , int prune );
int hypre_BoxNeighborsDestroy( hypre_BoxNeighbors *neighbors );

/* communication.c */
hypre_CommPkg *hypre_CommPkgCreate( hypre_BoxArrayArray *send_boxes , hypre_BoxArrayArray *recv_boxes , hypre_Index send_stride , hypre_Index recv_stride , hypre_BoxArray *send_data_space , hypre_BoxArray *recv_data_space , int **send_processes , int **recv_processes , int num_values , MPI_Comm comm , hypre_Index periodic );
int hypre_CommPkgDestroy( hypre_CommPkg *comm_pkg );
int hypre_InitializeCommunication( hypre_CommPkg *comm_pkg , double *send_data , double *recv_data , hypre_CommHandle **comm_handle_ptr );
int hypre_InitializeCommunication( hypre_CommPkg *comm_pkg , double *send_data , double *recv_data , hypre_CommHandle **comm_handle_ptr );
int hypre_FinalizeCommunication( hypre_CommHandle *comm_handle );
int hypre_FinalizeCommunication( hypre_CommHandle *comm_handle );
int hypre_ExchangeLocalData( hypre_CommPkg *comm_pkg , double *send_data , double *recv_data );
hypre_CommType *hypre_CommTypeCreate( hypre_CommTypeEntry **comm_entries , int num_entries );
int hypre_CommTypeDestroy( hypre_CommType *comm_type );
hypre_CommTypeEntry *hypre_CommTypeEntryCreate( hypre_Box *box , hypre_Index stride , hypre_Box *data_box , int num_values , int data_box_offset );
int hypre_CommTypeEntryDestroy( hypre_CommTypeEntry *comm_entry );
int hypre_CommPkgCreateInfo( hypre_BoxArrayArray *boxes , hypre_Index stride , hypre_BoxArray *data_space , int **processes , int num_values , MPI_Comm comm , hypre_Index periodic , int *num_comms_ptr , int **comm_processes_ptr , hypre_CommType ***comm_types_ptr , hypre_CommType **copy_type_ptr );
int hypre_CommTypeSort( hypre_CommType *comm_type , hypre_Index periodic );
int hypre_CommPkgCommit( hypre_CommPkg *comm_pkg );
int hypre_CommPkgUnCommit( hypre_CommPkg *comm_pkg );
int hypre_CommTypeBuildMPI( int num_comms , int *comm_procs , hypre_CommType **comm_types , MPI_Datatype *comm_mpi_types );
int hypre_CommTypeEntryBuildMPI( hypre_CommTypeEntry *comm_entry , MPI_Datatype *comm_entry_mpi_type );

/* communication_info.c */
int hypre_CreateCommInfoFromStencil( hypre_StructGrid *grid , hypre_StructStencil *stencil , hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_procs_ptr , int ***recv_procs_ptr );
int hypre_CreateCommInfoFromNumGhost( hypre_StructGrid *grid , int *num_ghost , hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_procs_ptr , int ***recv_procs_ptr );
int hypre_CreateCommInfoFromGrids( hypre_StructGrid *from_grid , hypre_StructGrid *to_grid , hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_procs_ptr , int ***recv_procs_ptr );

/* computation.c */
int hypre_CreateComputeInfo( hypre_StructGrid *grid , hypre_StructStencil *stencil , hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr , hypre_BoxArrayArray **indt_boxes_ptr , hypre_BoxArrayArray **dept_boxes_ptr );
int hypre_ComputePkgCreate( hypre_BoxArrayArray *send_boxes , hypre_BoxArrayArray *recv_boxes , hypre_Index send_stride , hypre_Index recv_stride , int **send_processes , int **recv_processes , hypre_BoxArrayArray *indt_boxes , hypre_BoxArrayArray *dept_boxes , hypre_Index stride , hypre_StructGrid *grid , hypre_BoxArray *data_space , int num_values , hypre_ComputePkg **compute_pkg_ptr );
int hypre_ComputePkgDestroy( hypre_ComputePkg *compute_pkg );
int hypre_InitializeIndtComputations( hypre_ComputePkg *compute_pkg , double *data , hypre_CommHandle **comm_handle_ptr );
int hypre_FinalizeIndtComputations( hypre_CommHandle *comm_handle );

/* grow.c */
hypre_BoxArray *hypre_GrowBoxByStencil( hypre_Box *box , hypre_StructStencil *stencil , int transpose );
hypre_BoxArrayArray *hypre_GrowBoxArrayByStencil( hypre_BoxArray *box_array , hypre_StructStencil *stencil , int transpose );

/* project.c */
int hypre_ProjectBox( hypre_Box *box , hypre_Index index , hypre_Index stride );
int hypre_ProjectBoxArray( hypre_BoxArray *box_array , hypre_Index index , hypre_Index stride );
int hypre_ProjectBoxArrayArray( hypre_BoxArrayArray *box_array_array , hypre_Index index , hypre_Index stride );

/* struct_axpy.c */
int hypre_StructAxpy( double alpha , hypre_StructVector *x , hypre_StructVector *y );

/* struct_copy.c */
int hypre_StructCopy( hypre_StructVector *x , hypre_StructVector *y );

/* struct_grid.c */
int hypre_StructGridCreate( MPI_Comm comm , int dim , hypre_StructGrid **grid_ptr );
int hypre_StructGridRef( hypre_StructGrid *grid , hypre_StructGrid **grid_ref );
int hypre_StructGridDestroy( hypre_StructGrid *grid );
int hypre_StructGridSetHoodInfo( hypre_StructGrid *grid , int max_distance );
int hypre_StructGridSetPeriodic( hypre_StructGrid *grid , hypre_Index periodic );
int hypre_StructGridSetExtents( hypre_StructGrid *grid , hypre_Index ilower , hypre_Index iupper );
int hypre_StructGridSetBoxes( hypre_StructGrid *grid , hypre_BoxArray *boxes );
int hypre_StructGridSetHood( hypre_StructGrid *grid , hypre_BoxArray *hood_boxes , int *hood_procs , int *hood_ids , int first_local , int num_local , int num_periodic , hypre_Box *bounding_box );
int hypre_StructGridAssemble( hypre_StructGrid *grid );
int hypre_GatherAllBoxes( MPI_Comm comm , hypre_BoxArray *boxes , hypre_BoxArray **all_boxes_ptr , int **all_procs_ptr , int *first_local_ptr );
int hypre_StructGridPrint( FILE *file , hypre_StructGrid *grid );
int hypre_StructGridRead( MPI_Comm comm , FILE *file , hypre_StructGrid **grid_ptr );
int hypre_StructGridPeriodicAllBoxes( hypre_StructGrid *grid , hypre_BoxArray **all_boxes_ptr , int **all_procs_ptr , int *first_local_ptr , int *num_periodic_ptr );

/* struct_innerprod.c */
double hypre_StructInnerProd( hypre_StructVector *x , hypre_StructVector *y );

/* struct_io.c */
int hypre_PrintBoxArrayData( FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , int num_values , double *data );
int hypre_ReadBoxArrayData( FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , int num_values , double *data );

/* struct_matrix.c */
double *hypre_StructMatrixExtractPointerByIndex( hypre_StructMatrix *matrix , int b , hypre_Index index );
hypre_StructMatrix *hypre_StructMatrixCreate( MPI_Comm comm , hypre_StructGrid *grid , hypre_StructStencil *user_stencil );
hypre_StructMatrix *hypre_StructMatrixRef( hypre_StructMatrix *matrix );
int hypre_StructMatrixDestroy( hypre_StructMatrix *matrix );
int hypre_StructMatrixInitializeShell( hypre_StructMatrix *matrix );
int hypre_StructMatrixInitializeData( hypre_StructMatrix *matrix , double *data );
int hypre_StructMatrixInitialize( hypre_StructMatrix *matrix );
int hypre_StructMatrixSetValues( hypre_StructMatrix *matrix , hypre_Index grid_index , int num_stencil_indices , int *stencil_indices , double *values , int add_to );
int hypre_StructMatrixSetBoxValues( hypre_StructMatrix *matrix , hypre_Box *value_box , int num_stencil_indices , int *stencil_indices , double *values , int add_to );
int hypre_StructMatrixAssemble( hypre_StructMatrix *matrix );
int hypre_StructMatrixSetNumGhost( hypre_StructMatrix *matrix , int *num_ghost );
int hypre_StructMatrixPrint( char *filename , hypre_StructMatrix *matrix , int all );
int hypre_StructMatrixMigrate( hypre_StructMatrix *from_matrix , hypre_StructMatrix *to_matrix );
hypre_StructMatrix *hypre_StructMatrixRead( MPI_Comm comm , char *filename , int *num_ghost );

/* struct_matrix_mask.c */
hypre_StructMatrix *hypre_StructMatrixCreateMask( hypre_StructMatrix *matrix , int num_stencil_indices , int *stencil_indices );

/* struct_matvec.c */
void *hypre_StructMatvecCreate( void );
int hypre_StructMatvecSetup( void *matvec_vdata , hypre_StructMatrix *A , hypre_StructVector *x );
int hypre_StructMatvecCompute( void *matvec_vdata , double alpha , hypre_StructMatrix *A , hypre_StructVector *x , double beta , hypre_StructVector *y );
int hypre_StructMatvecDestroy( void *matvec_vdata );
int hypre_StructMatvec( double alpha , hypre_StructMatrix *A , hypre_StructVector *x , double beta , hypre_StructVector *y );

/* struct_scale.c */
int hypre_StructScale( double alpha , hypre_StructVector *y );

/* struct_stencil.c */
hypre_StructStencil *hypre_StructStencilCreate( int dim , int size , hypre_Index *shape );
hypre_StructStencil *hypre_StructStencilRef( hypre_StructStencil *stencil );
int hypre_StructStencilDestroy( hypre_StructStencil *stencil );
int hypre_StructStencilElementRank( hypre_StructStencil *stencil , hypre_Index stencil_element );
int hypre_StructStencilSymmetrize( hypre_StructStencil *stencil , hypre_StructStencil **symm_stencil_ptr , int **symm_elements_ptr );

/* struct_vector.c */
hypre_StructVector *hypre_StructVectorCreate( MPI_Comm comm , hypre_StructGrid *grid );
hypre_StructVector *hypre_StructVectorRef( hypre_StructVector *vector );
int hypre_StructVectorDestroy( hypre_StructVector *vector );
int hypre_StructVectorInitializeShell( hypre_StructVector *vector );
int hypre_StructVectorInitializeData( hypre_StructVector *vector , double *data );
int hypre_StructVectorInitialize( hypre_StructVector *vector );
int hypre_StructVectorSetValues( hypre_StructVector *vector , hypre_Index grid_index , double values , int add_to );
int hypre_StructVectorSetBoxValues( hypre_StructVector *vector , hypre_Box *value_box , double *values , int add_to );
int hypre_StructVectorGetValues( hypre_StructVector *vector , hypre_Index grid_index , double *values_ptr );
int hypre_StructVectorGetBoxValues( hypre_StructVector *vector , hypre_Box *value_box , double *values );
int hypre_StructVectorSetNumGhost( hypre_StructVector *vector , int *num_ghost );
int hypre_StructVectorAssemble( hypre_StructVector *vector );
int hypre_StructVectorSetConstantValues( hypre_StructVector *vector , double values );
int hypre_StructVectorClearGhostValues( hypre_StructVector *vector );
int hypre_StructVectorClearAllValues( hypre_StructVector *vector );
hypre_CommPkg *hypre_StructVectorGetMigrateCommPkg( hypre_StructVector *from_vector , hypre_StructVector *to_vector );
int hypre_StructVectorMigrate( hypre_CommPkg *comm_pkg , hypre_StructVector *from_vector , hypre_StructVector *to_vector );
int hypre_StructVectorPrint( char *filename , hypre_StructVector *vector , int all );
hypre_StructVector *hypre_StructVectorRead( MPI_Comm comm , char *filename , int *num_ghost );


#ifdef __cplusplus
}
#endif

#endif

