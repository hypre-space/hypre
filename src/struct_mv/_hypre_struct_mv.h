/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/

#ifndef hypre_STRUCT_MV_HEADER
#define hypre_STRUCT_MV_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "HYPRE_struct_mv.h"
#include "_hypre_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the Box structures
 *
 *****************************************************************************/

#ifndef hypre_BOX_HEADER
#define hypre_BOX_HEADER

#ifndef HYPRE_MAXDIM
#define HYPRE_MAXDIM 3
#endif

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

typedef HYPRE_Int  hypre_Index[3];
typedef HYPRE_Int *hypre_IndexRef;

/*--------------------------------------------------------------------------
 * hypre_Box:
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
   HYPRE_Int   size;          /* Size of box array */
   HYPRE_Int   alloc_size;    /* Size of currently alloced space */

} hypre_BoxArray;

#define hypre_BoxArrayExcess 10

/*--------------------------------------------------------------------------
 * hypre_BoxArrayArray:
 *   An array of box arrays.
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxArrayArray_struct
{
   hypre_BoxArray  **box_arrays;    /* Array of pointers to box arrays */
   HYPRE_Int         size;          /* Size of box array array */

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

#define hypre_IndexZero(index)\
   (hypre_IndexX(index) == 0 &&  hypre_IndexY(index) == 0 \
    && hypre_IndexZ(index) == 0)

#define hypre_IndexGTESize(index, size) \
(hypre_IndexX(index) >= size &&  hypre_IndexY(index) >= size \
    && hypre_IndexZ(index) >= size)

#define hypre_CopyIndex(index1, index2) \
( hypre_IndexX(index2) = hypre_IndexX(index1),\
  hypre_IndexY(index2) = hypre_IndexY(index1),\
  hypre_IndexZ(index2) = hypre_IndexZ(index1) )

#define hypre_CopyToCleanIndex(in_index, ndim, out_index) \
{\
   HYPRE_Int d;\
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

#define hypre_AddIndex(index1, index2, index3) \
( hypre_IndexX(index3) = hypre_IndexX(index2) + hypre_IndexX(index1),\
  hypre_IndexY(index3) = hypre_IndexY(index2) + hypre_IndexY(index1),\
  hypre_IndexZ(index3) = hypre_IndexZ(index2) + hypre_IndexZ(index1) )

#define hypre_SubtractIndex(index1, index2, index3) \
( hypre_IndexX(index3) = hypre_IndexX(index1) - hypre_IndexX(index2),\
  hypre_IndexY(index3) = hypre_IndexY(index1) - hypre_IndexY(index2),\
  hypre_IndexZ(index3) = hypre_IndexZ(index1) - hypre_IndexZ(index2) )

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

#define hypre_BoxEqualP( box1, box2 ) (\
 hypre_BoxIMinX(box1)==hypre_BoxIMinX(box2) &&\
 hypre_BoxIMaxX(box1)==hypre_BoxIMaxX(box2) &&\
 hypre_BoxIMinY(box1)==hypre_BoxIMinY(box2) &&\
 hypre_BoxIMaxY(box1)==hypre_BoxIMaxY(box2) &&\
 hypre_BoxIMinZ(box1)==hypre_BoxIMinZ(box2) &&\
 hypre_BoxIMaxZ(box1)==hypre_BoxIMaxZ(box2) )

#define hypre_IndexInBoxP( index, box ) (\
 hypre_IndexX(index)>=hypre_BoxIMinX(box) &&\
 hypre_IndexX(index)<=hypre_BoxIMaxX(box) &&\
 hypre_IndexY(index)>=hypre_BoxIMinY(box) &&\
 hypre_IndexY(index)<=hypre_BoxIMaxY(box) &&\
 hypre_IndexZ(index)>=hypre_BoxIMinZ(box) &&\
 hypre_IndexZ(index)<=hypre_BoxIMaxZ(box) )


#define hypre_IndexDInBoxP( index, d, box ) (\
 hypre_IndexD(index, d)>=hypre_BoxIMinD(box, d) &&\
 hypre_IndexD(index, d)<=hypre_BoxIMaxD(box, d) )

#define hypre_CopyBox(box1, box2) \
( hypre_CopyIndex(hypre_BoxIMin(box1), hypre_BoxIMin(box2)),\
  hypre_CopyIndex(hypre_BoxIMax(box1), hypre_BoxIMax(box2)) )

#define hypre_BoxVolume(box) \
(hypre_BoxSizeX(box) * hypre_BoxSizeY(box) * hypre_BoxSizeZ(box))

#define hypre_BoxShiftPos(box, shift) \
{\
   hypre_BoxIMinX(box) += hypre_IndexX(shift);\
   hypre_BoxIMinY(box) += hypre_IndexY(shift);\
   hypre_BoxIMinZ(box) += hypre_IndexZ(shift);\
   hypre_BoxIMaxX(box) += hypre_IndexX(shift);\
   hypre_BoxIMaxY(box) += hypre_IndexY(shift);\
   hypre_BoxIMaxZ(box) += hypre_IndexZ(shift);\
}

#define hypre_BoxShiftNeg(box, shift) \
{\
   hypre_BoxIMinX(box) -= hypre_IndexX(shift);\
   hypre_BoxIMinY(box) -= hypre_IndexY(shift);\
   hypre_BoxIMinZ(box) -= hypre_IndexZ(shift);\
   hypre_BoxIMaxX(box) -= hypre_IndexX(shift);\
   hypre_BoxIMaxY(box) -= hypre_IndexY(shift);\
   hypre_BoxIMaxZ(box) -= hypre_IndexZ(shift);\
}

#define hypre_BoxIndexRank(box, index) \
((hypre_IndexX(index) - hypre_BoxIMinX(box)) + \
 ((hypre_IndexY(index) - hypre_BoxIMinY(box)) + \
   ((hypre_IndexZ(index) - hypre_BoxIMinZ(box)) * \
    hypre_BoxSizeY(box))) * \
  hypre_BoxSizeX(box))

/* The first hypre_CCBoxIndexRank is better style because it keeps
   its similarity to the variable coefficient hypre_BoxIndexRank.
   The second one sometimes avoids compiler warnings...*/
#define hypre_CCBoxIndexRank(box, index) 0
#define hypre_CCBoxIndexRank_noargs() 0

#define hypre_BoxOffsetDistance(box, index) \
(hypre_IndexX(index) + \
 (hypre_IndexY(index) + \
  (hypre_IndexZ(index) * \
   hypre_BoxSizeY(box))) * \
 hypre_BoxSizeX(box))

#define hypre_CCBoxOffsetDistance(box, index) 0
  
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
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_USE_PTHREADS

#define HYPRE_BOX_PRIVATE hypre__nx,hypre__ny,hypre__nz,hypre__i,hypre__j,hypre__k

#define hypre_BoxLoopDeclareS(dbox, stride, sx, sy, sz) \
HYPRE_Int  sx = (hypre_IndexX(stride));\
HYPRE_Int  sy = (hypre_IndexY(stride)*hypre_BoxSizeX(dbox));\
HYPRE_Int  sz = (hypre_IndexZ(stride)*\
           hypre_BoxSizeX(dbox)*hypre_BoxSizeY(dbox))

#define hypre_BoxLoopDeclareN(loop_size) \
HYPRE_Int  hypre__i, hypre__j, hypre__k;\
HYPRE_Int  hypre__nx = hypre_IndexX(loop_size);\
HYPRE_Int  hypre__ny = hypre_IndexY(loop_size);\
HYPRE_Int  hypre__nz = hypre_IndexZ(loop_size);\
HYPRE_Int  hypre__mx = hypre__nx;\
HYPRE_Int  hypre__my = hypre__ny;\
HYPRE_Int  hypre__mz = hypre__nz;\
HYPRE_Int  hypre__dir, hypre__max;\
HYPRE_Int  hypre__div, hypre__mod;\
HYPRE_Int  hypre__block, hypre__num_blocks;\
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

#define hypre_BoxLoopGetIndex(index) \
index[0] = hypre__i; index[1] = hypre__j; index[2] = hypre__k

/* Use this before the For macros below to force only one block */
#define hypre_BoxLoopSetOneBlock() hypre__num_blocks = 1

/* Use this to get the block iteration inside a BoxLoop */
#define hypre_BoxLoopBlock() hypre__block

/*-----------------------------------*/

#define hypre_BoxLoop0Begin(ndim, loop_size)\
{\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop0For()\
   hypre__BoxLoop0For(hypre__i, hypre__j, hypre__k)
#define hypre__BoxLoop0For(i, j, k)\
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

#define hypre_BoxLoop1Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1)\
{\
   HYPRE_Int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop1For(i1)\
   hypre__BoxLoop1For(hypre__i, hypre__j, hypre__k, i1)
#define hypre__BoxLoop1For(i, j, k, i1)\
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

#define hypre_BoxLoop2Begin(ndim,loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2)\
{\
   HYPRE_Int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   HYPRE_Int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop2For(i1, i2)\
   hypre__BoxLoop2For(hypre__i, hypre__j, hypre__k, i1, i2)
#define hypre__BoxLoop2For(i, j, k, i1, i2)\
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

#define hypre_BoxLoop3Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3)\
{\
   HYPRE_Int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   HYPRE_Int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);\
   HYPRE_Int  hypre__i3start = hypre_BoxIndexRank(dbox3, start3);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);\
   hypre_BoxLoopDeclareS(dbox3, stride3, hypre__sx3, hypre__sy3, hypre__sz3);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop3For(i1, i2, i3)\
   hypre__BoxLoop3For(hypre__i, hypre__j, hypre__k, i1, i2, i3)
#define hypre__BoxLoop3For(i, j, k, i1, i2, i3)\
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

#define hypre_BoxLoop4Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3,\
                            dbox4, start4, stride4, i4)\
{\
   HYPRE_Int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   HYPRE_Int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);\
   HYPRE_Int  hypre__i3start = hypre_BoxIndexRank(dbox3, start3);\
   HYPRE_Int  hypre__i4start = hypre_BoxIndexRank(dbox4, start4);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);\
   hypre_BoxLoopDeclareS(dbox3, stride3, hypre__sx3, hypre__sy3, hypre__sz3);\
   hypre_BoxLoopDeclareS(dbox4, stride4, hypre__sx4, hypre__sy4, hypre__sz4);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop4For(i1, i2, i3, i4)\
   hypre__BoxLoop4For(hypre__i, hypre__j, hypre__k, i1, i2, i3, i4)
#define hypre__BoxLoop4For(i, j, k, i1, i2, i3, i4)\
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

/******************************************************************************
 *
 * NEW BoxLoop STUFF
 *
 *****************************************************************************/

#ifndef hypre_ZBOX_HEADER
#define hypre_ZBOX_HEADER

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *--------------------------------------------------------------------------*/

#define zypre_BoxLoopDeclare() \
HYPRE_Int  hypre__i[HYPRE_MAXDIM+1];\
HYPRE_Int  hypre__n[HYPRE_MAXDIM+1];\
HYPRE_Int  hypre__m[HYPRE_MAXDIM];\
HYPRE_Int  hypre__d, hypre__ndim;\
HYPRE_Int  hypre__dir, hypre__max;\
HYPRE_Int  hypre__div, hypre__mod;\
HYPRE_Int  hypre__block, hypre__num_blocks;\
HYPRE_Int  hypre__I, hypre__J, hypre__IN, hypre__JN

#define zypre_BoxLoopDeclareK(k) \
HYPRE_Int  hypre__sk[k][HYPRE_MAXDIM];\
HYPRE_Int  hypre__ikstart[k], hypre__ikinc[k][HYPRE_MAXDIM+1], hypre__i0inc[k]

#define zypre_BoxLoopInit(ndim, loop_size) \
hypre__ndim = ndim;\
for (hypre__d = 0; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__n[hypre__d] = hypre__m[hypre__d] = loop_size[hypre__d];\
}\
hypre__dir = 0;\
hypre__max = hypre__n[0];\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   if (hypre__n[hypre__d] > hypre__max)\
   {\
      hypre__dir = hypre__d;\
      hypre__max = hypre__n[hypre__d];\
   }\
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

#define zypre_BoxLoopInitK(k, dboxk, startk, stridek, ik) \
hypre__sk[k][0] = 1;\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__sk[k][hypre__d] =\
      hypre__sk[k][hypre__d-1]*hypre_BoxSizeD(dboxk, hypre__d-1);\
}\
for (hypre__d = 0; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__sk[k][hypre__d] = hypre__sk[k][hypre__d]*stridek[hypre__d];\
}\
hypre__ikstart[k] = hypre_BoxIndexRank(dboxk, startk)

#define zypre_BoxLoopSet() \
for (hypre__d = 0; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__n[hypre__d] = hypre__m[hypre__d];\
}\
if (hypre__num_blocks > 1)\
{\
   hypre__i[hypre__dir] =\
      hypre__block * hypre__div + hypre_min(hypre__mod, hypre__block);\
   hypre__n[hypre__dir] =\
      hypre__div + ((hypre__mod > hypre__block) ? 1 : 0);\
}

#define zypre_BoxLoopSetK(k, ik) \
ik = hypre__ikstart[k];\
if (hypre__num_blocks > 1)\
{\
   ik += hypre__i[hypre__dir]*hypre__sk[k][hypre__dir];\
}\
hypre__ikinc[k][0] = hypre__sk[k][0];\
hypre__i0inc[k] = hypre__ikinc[k][0];\
if (hypre__ndim > 1)\
{\
   hypre__ikinc[k][1] = hypre__sk[k][1] - hypre__n[0]*hypre__sk[k][0];\
}\
for (hypre__d = 2; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__ikinc[k][hypre__d] = hypre__ikinc[k][hypre__d-1] +\
      hypre__sk[k][hypre__d] - hypre__n[hypre__d-1]*hypre__sk[k][hypre__d-1];\
}\
hypre__ikinc[k][hypre__ndim] = 0

#define zypre_BoxLoopSetLoop() \
hypre__IN = hypre__n[0];\
hypre__JN = 1;\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__JN *= hypre__n[hypre__d];\
   hypre__i[hypre__d] = 0;\
   hypre__n[hypre__d] -= 2; /* this produces a simpler comparison below */\
}\
hypre__i[hypre__ndim] = 0;\
hypre__n[hypre__ndim] = 0

#define zypre_BoxLoopIncLoop() \
for (hypre__d = 1; hypre__i[hypre__d] > hypre__n[hypre__d]; hypre__d++)\
{\
   hypre__i[hypre__d] = 0;\
}\
hypre__i[hypre__d]++

#define zypre_BoxLoopGetIndex(index) \
index[0] = hypre__I;\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   index[hypre__d] = hypre__i[hypre__d];\
}

/*-----------------------------------*/

#define zypre_BoxLoop0Begin(ndim, loop_size)\
{\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopInit(ndim, loop_size);

#define zypre_BoxLoop0For()\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetLoop();\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_BoxLoop0End()\
         }\
         zypre_BoxLoopIncLoop();\
      }\
   }\
}

/*-----------------------------------*/

#define zypre_BoxLoop1Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1)\
{\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(1);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(0, dbox1, start1, stride1, i1);

#define zypre_BoxLoop1For(i1)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(0, i1);\
      zypre_BoxLoopSetLoop();\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_BoxLoop1End(i1)\
            i1 += hypre__i0inc[0];\
         }\
         zypre_BoxLoopIncLoop();\
         i1 += hypre__ikinc[0][hypre__d];\
      }\
   }\
}

/*-----------------------------------*/

#define zypre_BoxLoop2Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2)\
{\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(2);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(0, dbox1, start1, stride1, i1);\
   zypre_BoxLoopInitK(1, dbox2, start2, stride2, i2);

#define zypre_BoxLoop2For(i1, i2)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(0, i1);\
      zypre_BoxLoopSetK(1, i2);\
      zypre_BoxLoopSetLoop();\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_BoxLoop2End(i1, i2)\
            i1 += hypre__i0inc[0];\
            i2 += hypre__i0inc[1];\
         }\
         zypre_BoxLoopIncLoop();\
         i1 += hypre__ikinc[0][hypre__d];\
         i2 += hypre__ikinc[1][hypre__d];\
      }\
   }\
}

/*-----------------------------------*/

#define zypre_BoxLoop3Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3)\
{\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(3);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(0, dbox1, start1, stride1, i1);\
   zypre_BoxLoopInitK(1, dbox2, start2, stride2, i2);\
   zypre_BoxLoopInitK(2, dbox3, start3, stride3, i3);

#define zypre_BoxLoop3For(i1, i2, i3)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(0, i1);\
      zypre_BoxLoopSetK(1, i2);\
      zypre_BoxLoopSetK(2, i3);\
      zypre_BoxLoopSetLoop();\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_BoxLoop3End(i1, i2, i3)\
            i1 += hypre__i0inc[0];\
            i2 += hypre__i0inc[1];\
            i3 += hypre__i0inc[2];\
/*            i1 ++;*/\
/*            i2 ++;*/\
/*            i3 ++;*/\
         }\
         zypre_BoxLoopIncLoop();\
         i1 += hypre__ikinc[0][hypre__d];\
         i2 += hypre__ikinc[1][hypre__d];\
         i3 += hypre__ikinc[2][hypre__d];\
      }\
   }\
}

/*-----------------------------------*/

#define zypre_BoxLoop4Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3,\
                            dbox4, start4, stride4, i4)\
{\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(4);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(0, dbox1, start1, stride1, i1);\
   zypre_BoxLoopInitK(1, dbox2, start2, stride2, i2);\
   zypre_BoxLoopInitK(2, dbox3, start3, stride3, i3);\
   zypre_BoxLoopInitK(3, dbox4, start4, stride4, i4);

#define zypre_BoxLoop4For(i1, i2, i3, i4)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(0, i1);\
      zypre_BoxLoopSetK(1, i2);\
      zypre_BoxLoopSetK(2, i3);\
      zypre_BoxLoopSetK(3, i4);\
      zypre_BoxLoopSetLoop();\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_BoxLoop4End(i1, i2, i3, i4)\
            i1 += hypre__i0inc[0];\
            i2 += hypre__i0inc[1];\
            i3 += hypre__i0inc[2];\
            i4 += hypre__i0inc[3];\
         }\
         zypre_BoxLoopIncLoop();\
         i1 += hypre__ikinc[0][hypre__d];\
         i2 += hypre__ikinc[1][hypre__d];\
         i3 += hypre__ikinc[2][hypre__d];\
         i4 += hypre__ikinc[3][hypre__d];\
      }\
   }\
}

/*-----------------------------------*/

#endif




#if 0

/*- NOTES ------------------------------*/

#define hypre_BoxLoop2Begin(loop_size,
                            dbox1, start1, stride1, i1,
                            dbox2, start2, stride2, i2)
{
   /* init hypre__i1start */
   HYPRE_Int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);
   HYPRE_Int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);
   /* declare and set hypre__s1 */
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);
   /* declare and set hypre__n, hypre__m, hypre__dir, hypre__max,
    *                 hypre__div, hypre__mod, hypre__block, hypre__num_blocks */
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop2For(i, j, k, i1, i2)
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)
   {
   /* set i and hypre__n */
   hypre_BoxLoopSet(i, j, k);
   /* set i1 */
   i1 = hypre__i1start + i*hypre__sx1 + j*hypre__sy1 + k*hypre__sz1;
   i2 = hypre__i2start + i*hypre__sx2 + j*hypre__sy2 + k*hypre__sz2;
   for (k = 0; k < hypre__nz; k++)
   {
      for (j = 0; j < hypre__ny; j++)
      {
         for (i = 0; i < hypre__nx; i++)
         {

#define hypre_BoxLoop2End(i1, i2)
            i1 += hypre__sx1;
            i2 += hypre__sx2;
         }
         i1 += hypre__sy1 - hypre__nx*hypre__sx1;
         i2 += hypre__sy2 - hypre__nx*hypre__sx2;
      }
      i1 += hypre__sz1 - hypre__ny*hypre__sy1;
      i2 += hypre__sz2 - hypre__ny*hypre__sy2;
   }
   }
}

/*-----------------------------------*/

/* Idea 2 */

N = 1;
for (d = 0; d < ndim; d++)
{
   N *= n[d];
   i[d] = 0;
   n[d] -= 2; /* this produces a simpler comparison below */
}
i[ndim] = 0;
n[ndim] = 0;
for (I = 0; I < N; I++)
{
   /* loop body */

   for (d = 0; i[d] > n[d]; d++)
   {
      i[d] = 0;
   }
   i[d]++;
   i1 += s1[d]; /* NOTE: These are different from hypre__sx1, etc. above */
   i2 += s2[d]; /* The lengths of i, n, and s must be (ndim+1) */
}

/*-----------------------------------*/

/* Idea 3 */

N = 1;
for (d = 1; d < ndim; d++)
{
   N *= n[d];
   i[d] = 0;
   n[d] -= 2; /* this produces a simpler comparison below */
}
i[ndim] = 0;
n[ndim] = 0;
for (J = 0; J < N; J++)
{
   for (I = 0; I < n[0]; I++)
   {
      /* loop body */

      i1 += s1[0];
      i2 += s2[0];
   }
   for (d = 1; i[d] > n[d]; d++)
   {
      i[d] = 0;
   }
   i[d]++;
   i1 += s1[d]; /* NOTE: These are different from hypre__sx1, etc. above */
   i2 += s2[d]; /* The lengths of i, n, and s must be (ndim+1) */
}

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
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

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Header info for the struct assumed partition
 *
 *****************************************************************************/

#ifndef hypre_ASSUMED_PART_HEADER
#define hypre_ASSUMED_PART_HEADER


/* to prevent overflow */

#define hypre_doubleBoxVolume(box) \
   ((double) hypre_BoxSizeX(box) * (double) hypre_BoxSizeY(box) * (double) hypre_BoxSizeZ(box))


typedef struct 
{
   /* the entries will be the same for all procs */  
   hypre_BoxArray      *regions;  /* areas of the grid with boxes */
   HYPRE_Int           num_regions;  /* how many regions */    
   HYPRE_Int           *proc_partitions;  /* proc ids assigned to each region  
                                             - this is size num_regions +1*/
   hypre_Index         *divisions;        /* number of proc divisions in x y z 
                                             direction
                                             for each region */
   /* these entries are specific to each proc */
   hypre_BoxArray      *my_partition;  /*the portion of grid that I own - at most 2 */
   hypre_BoxArray      *my_partition_boxes;  /* boxes in my portion */
   HYPRE_Int           *my_partition_proc_ids;
   HYPRE_Int           *my_partition_boxnums;
   HYPRE_Int           my_partition_ids_size;   
   HYPRE_Int           my_partition_ids_alloc;
   HYPRE_Int           my_partition_num_distinct_procs;
    
} hypre_StructAssumedPart;


/*Accessor macros */

#define hypre_StructAssumedPartRegions(apart) ((apart)->regions) 
#define hypre_StructAssumedPartNumRegions(apart) ((apart)->num_regions) 
#define hypre_StructAssumedPartDivisions(apart) ((apart)->divisions) 
#define hypre_StructAssumedPartDivision(apart, i) ((apart)->divisions[i]) 
#define hypre_StructAssumedPartProcPartitions(apart) ((apart)->proc_partitions) 
#define hypre_StructAssumedPartProcPartition(apart, i) ((apart)->proc_partitions[i]) 
#define hypre_StructAssumedPartMyPartition(apart) ((apart)->my_partition)
#define hypre_StructAssumedPartMyPartitionBoxes(apart) ((apart)->my_partition_boxes)
#define hypre_StructAssumedPartMyPartitionProcIds(apart) ((apart)->my_partition_proc_ids)
#define hypre_StructAssumedPartMyPartitionIdsSize(apart) ((apart)->my_partition_ids_size)
#define hypre_StructAssumedPartMyPartitionIdsAlloc(apart) ((apart)->my_partition_ids_alloc)
#define hypre_StructAssumedPartMyPartitionNumDistinctProcs(apart) ((apart)->my_partition_num_distinct_procs)
#define hypre_StructAssumedPartMyPartitionBoxnums(apart) ((apart)->my_partition_boxnums)

#define hypre_StructAssumedPartMyPartitionProcId(apart, i) ((apart)->my_partition_proc_ids[i])
#define hypre_StructAssumedPartMyPartitionBoxnum(apart, i) ((apart)->my_partition_boxnums[i])
#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/

#ifndef hypre_BOX_MANAGER_HEADER
#define hypre_BOX_MANAGER_HEADER



/*--------------------------------------------------------------------------
 * BoxManEntry
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxManEntry_struct
{
   hypre_Index  imin; /*extents of box */
   hypre_Index  imax;

   HYPRE_Int proc; /*this is a two-part unique id: (proc, id) */
   HYPRE_Int id;
   HYPRE_Int num_ghost[6];

   HYPRE_Int position; /* this indicates the location of the entry in the
                  * the box manager entries array and is used for
                  * pairing with the info object (populated in addentry) */
   
   void *boxman; /* the owning manager (populated in addentry)*/
   
   struct hypre_BoxManEntry_struct  *next;

} hypre_BoxManEntry;


/*---------------------------------------------------------------------------
 *
 * Box Manager: organizes arbitrary information in a spatial way
 *
 *----------------------------------------------------------------------------*/


typedef struct
{

   MPI_Comm            comm;

   HYPRE_Int           max_nentries;  /* storage in entries allocated to this 
                                         amount */

    
   HYPRE_Int           is_gather_called; /* boolean to indicate  whether GatherEntries
                                            function has been called  (prior to 
                                            assemble) - may not want this (can tell
                                            by the size of gather_regions array) */
   
   hypre_BoxArray     *gather_regions;  /*this is where we collect boxes input 
                                          by calls to BoxManGatherEntries - to be 
                                          gathered in the assemble.  These are then 
                                          deleted after the assemble */
   

   HYPRE_Int           all_global_known; /* Boolean to say that every
                                            processor already has all
                                            of the global data for
                                            this manager (this could be
                                            acessed by a coarsening routine, 
                                            for example) */
   
   HYPRE_Int           is_entries_sort;     /* Boolean to say that entries were 
                                            added in sorted order (id, proc)
                                            (this could be
                                            acessed by a coarsening routine, 
                                            for example) */


   HYPRE_Int           entry_info_size;  /* in bytes, the (max) size of the info 
                                            object for the entries */ 

   HYPRE_Int           is_assembled;        /* flag to indicate if the box manager has been 
                                            assembled (use to control whether or not
                                            functions can be used prior to assemble)*/
   

   /* storing the entries */
   HYPRE_Int           nentries;     /* number of entries stored */
   hypre_BoxManEntry  *entries;      /* These are the actual box manager entries - these
                                      are sorted by (proc, id) at the end of the assemble)*/  

   HYPRE_Int          *procs_sort;    /* the sorted procs corresponding to entries */
   HYPRE_Int          *ids_sort;      /* sorted ids corresponding to the entries */
 
   HYPRE_Int          num_procs_sort; /* number of distinct procs in *entries */
   HYPRE_Int          *procs_sort_offsets;  /* offsets for procs into the 
                                             *entry_sort array */
   HYPRE_Int          first_local;      /* position of local infomation in entries*/  
   HYPRE_Int          local_proc_offset;  /*position of local information in offsets */

   /* here is the table  that organizes the entries spatially (by index)*/
   hypre_BoxManEntry **index_table; /* this points into 'entries' array  
                                            and corresponds to the index arays*/

   HYPRE_Int          *indexes[3]; /* here we have the x,y,z indexes (ordered) 
                                      for the imin and imax
                                      of each box in the entries array*/
   HYPRE_Int           size[3];    /* how many indexes we have in each direction 
                                      - x,y,z */ 

   HYPRE_Int           last_index[3]; /* the last index used in the indexes map */

   HYPRE_Int           num_my_entries; /* number of entries with proc_id = myid */
   HYPRE_Int           *my_ids;        /* an array of ids corresponding to my entries */ 
   hypre_BoxManEntry   **my_entries;   /* points into *entries that are mine & corresponds to
                                          my_ids array.  This is destroyed in the assemble */
   
   void               *info_objects;    /* this is an array of info objects (of each is of 
                                         size entry_info_size) -this is managed byte-wise */ 
   

   hypre_StructAssumedPart *assumed_partition; /* the assumed partition object  - for now this is only
                                                  used during the assemble (where it is created)*/
   HYPRE_Int           dim;           /* problem dimension (known in the grid) */

   hypre_Box           *bounding_box;  /* bounding box - from associated grid */
   

   HYPRE_Int           next_id; /* counter to indicate the next id 
                                   that would be unique (regardless of proc id) */  

   /* ghost stuff  */

   HYPRE_Int          num_ghost[6]; 



} hypre_BoxManager;




/*--------------------------------------------------------------------------
 * Accessor macros:  hypre_BoxMan
 *--------------------------------------------------------------------------*/

#define hypre_BoxManComm(manager)               ((manager) -> comm)

#define hypre_BoxManMaxNEntries(manager)        ((manager) -> max_nentries)

#define hypre_BoxManIsGatherCalled(manager)     ((manager) -> is_gather_called)
#define hypre_BoxManIsEntriesSort(manager)      ((manager) -> is_entries_sort)
#define hypre_BoxManGatherRegions(manager)      ((manager) -> gather_regions)
#define hypre_BoxManAllGlobalKnown(manager)     ((manager) -> all_global_known)
#define hypre_BoxManEntryInfoSize(manager)      ((manager) -> entry_info_size)
#define hypre_BoxManNEntries(manager)           ((manager) -> nentries)
#define hypre_BoxManEntries(manager)            ((manager) -> entries)
#define hypre_BoxManInfoObjects(manager)        ((manager) -> info_objects)
#define hypre_BoxManIsAssembled(manager)        ((manager) -> is_assembled) 

#define hypre_BoxManProcsSort(manager)          ((manager) -> procs_sort)
#define hypre_BoxManIdsSort(manager)            ((manager) -> ids_sort)
#define hypre_BoxManNumProcsSort(manager)       ((manager) -> num_procs_sort)
#define hypre_BoxManProcsSortOffsets(manager)   ((manager) -> procs_sort_offsets)
#define hypre_BoxManLocalProcOffset(manager)    ((manager) -> local_proc_offset)

#define hypre_BoxManFirstLocal(manager)         ((manager) -> first_local)

#define hypre_BoxManIndexTable(manager)         ((manager) -> index_table)
#define hypre_BoxManIndexes(manager)            ((manager) -> indexes)
#define hypre_BoxManSize(manager)               ((manager) -> size)
#define hypre_BoxManLastIndex(manager)          ((manager) -> last_index)

#define hypre_BoxManNumMyEntries(manager)       ((manager) -> num_my_entries)
#define hypre_BoxManMyIds(manager)              ((manager) -> my_ids)
#define hypre_BoxManMyEntries(manager)          ((manager) -> my_entries)
#define hypre_BoxManAssumedPartition(manager)   ((manager) -> assumed_partition)
#define hypre_BoxManDim(manager)                ((manager) -> dim)
#define hypre_BoxManBoundingBox(manager)        ((manager) -> bounding_box)

#define hypre_BoxManNextId(manager)             ((manager) -> next_id)

#define hypre_BoxManNumGhost(manager)           ((manager) -> num_ghost)

#define hypre_BoxManIndexesD(manager, d)    hypre_BoxManIndexes(manager)[d]
#define hypre_BoxManSizeD(manager, d)       hypre_BoxManSize(manager)[d]
#define hypre_BoxManLastIndexD(manager, d)  hypre_BoxManLastIndex(manager)[d]
#define hypre_BoxManIndexTableEntry(manager, i, j, k) \
hypre_BoxManIndexTable(manager)[((k*hypre_BoxManSizeD(manager, 1) + j)*\
                           hypre_BoxManSizeD(manager, 0) + i)]

#define hypre_BoxManInfoObject(manager, i) \
(void *) ((char *)hypre_BoxManInfoObjects(manager) + i* hypre_BoxManEntryInfoSize(manager))



/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxManEntry
 *--------------------------------------------------------------------------*/

#define hypre_BoxManEntryIMin(entry)     ((entry) -> imin)
#define hypre_BoxManEntryIMax(entry)     ((entry) -> imax)
#define hypre_BoxManEntryProc(entry)     ((entry) -> proc)
#define hypre_BoxManEntryId(entry)       ((entry) -> id)
#define hypre_BoxManEntryPosition(entry) ((entry) -> position)
#define hypre_BoxManEntryNumGhost(entry) ((entry) -> num_ghost)
#define hypre_BoxManEntryNext(entry)     ((entry) -> next)
#define hypre_BoxManEntryBoxMan(entry)   ((entry) -> boxman)
#

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/



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
                      
   HYPRE_Int            dim;          /* Number of grid dimensions */
                      
   hypre_BoxArray      *boxes;        /* Array of boxes in this process */
   HYPRE_Int           *ids;          /* Unique IDs for boxes */
   hypre_Index          max_distance; /* Neighborhood size - in each dimension*/

   hypre_Box           *bounding_box; /* Bounding box around grid */

   HYPRE_Int            local_size;   /* Number of grid points locally */
   HYPRE_Int            global_size;  /* Total number of grid points */

   hypre_Index          periodic;     /* Indicates if grid is periodic */
   HYPRE_Int            num_periods;  /* number of box set periods */
   
   hypre_Index         *pshifts;      /* shifts of periodicity */


   HYPRE_Int            ref_count;


   HYPRE_Int           ghlocal_size;   /* Number of vars in box including ghosts */
   HYPRE_Int           num_ghost[6];   /* ghost layer size for each box  */  

   hypre_BoxManager   *box_man;
   

} hypre_StructGrid;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructGrid
 *--------------------------------------------------------------------------*/

#define hypre_StructGridComm(grid)          ((grid) -> comm)
#define hypre_StructGridDim(grid)           ((grid) -> dim)
#define hypre_StructGridBoxes(grid)         ((grid) -> boxes)
#define hypre_StructGridIDs(grid)           ((grid) -> ids)
#define hypre_StructGridMaxDistance(grid)   ((grid) -> max_distance)
#define hypre_StructGridBoundingBox(grid)   ((grid) -> bounding_box)
#define hypre_StructGridLocalSize(grid)     ((grid) -> local_size)
#define hypre_StructGridGlobalSize(grid)    ((grid) -> global_size)
#define hypre_StructGridPeriodic(grid)      ((grid) -> periodic)
#define hypre_StructGridNumPeriods(grid)    ((grid) -> num_periods)
#define hypre_StructGridPShifts(grid)       ((grid) -> pshifts)
#define hypre_StructGridPShift(grid, i)     ((grid) -> pshifts[i])
#define hypre_StructGridRefCount(grid)      ((grid) -> ref_count)
#define hypre_StructGridGhlocalSize(grid)   ((grid) -> ghlocal_size)
#define hypre_StructGridNumGhost(grid)      ((grid) -> num_ghost)
#define hypre_StructGridBoxMan(grid)        ((grid) -> box_man) 

#define hypre_StructGridBox(grid, i) \
(hypre_BoxArrayBox(hypre_StructGridBoxes(grid), i))
#define hypre_StructGridNumBoxes(grid) \
(hypre_BoxArraySize(hypre_StructGridBoxes(grid)))

#define hypre_StructGridIDPeriod(grid) \
hypre_BoxNeighborsIDPeriod(hypre_StructGridNeighbors(grid))

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/
 
#define hypre_ForStructGridBoxI(i, grid) \
hypre_ForBoxI(i, hypre_StructGridBoxes(grid))

#endif

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/



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
   HYPRE_Int      size;    /* Number of stencil coefficients */
                
   HYPRE_Int      dim;     /* Number of dimensions */

   HYPRE_Int      ref_count;

} hypre_StructStencil;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_StructStencil structure
 *--------------------------------------------------------------------------*/

#define hypre_StructStencilShape(stencil)      ((stencil) -> shape)
#define hypre_StructStencilSize(stencil)       ((stencil) -> size)
#define hypre_StructStencilDim(stencil)        ((stencil) -> dim)
#define hypre_StructStencilRefCount(stencil)   ((stencil) -> ref_count)

#define hypre_StructStencilElement(stencil, i) \
hypre_StructStencilShape(stencil)[i]

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/




#ifndef hypre_COMMUNICATION_HEADER
#define hypre_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_CommInfo:
 *
 * For "reverse" communication, the following are not needed (may be NULL)
 *    send_rboxnums, send_rboxes, send_transforms
 *
 * For "forward" communication, the following are not needed (may be NULL)
 *    recv_rboxnums, recv_rboxes, recv_transforms
 *
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommInfo_struct
{
   hypre_BoxArrayArray   *send_boxes;
   hypre_Index            send_stride;
   HYPRE_Int            **send_processes;
   HYPRE_Int            **send_rboxnums;
   hypre_BoxArrayArray   *send_rboxes;  /* send_boxes, some with periodic shift */

   hypre_BoxArrayArray   *recv_boxes;
   hypre_Index            recv_stride;
   HYPRE_Int            **recv_processes;
   HYPRE_Int            **recv_rboxnums;
   hypre_BoxArrayArray   *recv_rboxes;  /* recv_boxes, some with periodic shift */

   HYPRE_Int              num_transforms;  /* may be 0    = identity transform */
   hypre_Index           *coords;          /* may be NULL = identity transform */
   hypre_Index           *dirs;            /* may be NULL = identity transform */
   HYPRE_Int            **send_transforms; /* may be NULL = identity transform */
   HYPRE_Int            **recv_transforms; /* may be NULL = identity transform */

   HYPRE_Int              boxes_match;  /* true (>0) if each send box has a
                                         * matching box on the recv processor */

} hypre_CommInfo;

/*--------------------------------------------------------------------------
 * hypre_CommEntryType:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommEntryType_struct
{
   HYPRE_Int  offset;           /* offset for the data */
   HYPRE_Int  dim;              /* dimension of the communication */
   HYPRE_Int  length_array[3];  /* 4th dimension has length 'num_values' */
   HYPRE_Int  stride_array[4];
   HYPRE_Int *order;            /* order of 4th dimension values */

} hypre_CommEntryType;

/*--------------------------------------------------------------------------
 * hypre_CommType:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommType_struct
{
   HYPRE_Int             proc;
   HYPRE_Int             bufsize;     /* message buffer size (in doubles) */
   HYPRE_Int             num_entries;
   hypre_CommEntryType  *entries;

   /* this is only needed until first send buffer prefix is packed */
   HYPRE_Int            *rem_boxnums; /* entry remote box numbers */
   hypre_Box            *rem_boxes;   /* entry remote boxes */

} hypre_CommType;

/*--------------------------------------------------------------------------
 * hypre_CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommPkg_struct
{
   MPI_Comm          comm;

   HYPRE_Int         first_comm; /* is this the first communication? */
                   
   HYPRE_Int         num_values;
   hypre_Index       send_stride;
   hypre_Index       recv_stride;
   HYPRE_Int         send_bufsize; /* total send buffer size (in doubles) */
   HYPRE_Int         recv_bufsize; /* total recv buffer size (in doubles) */

   HYPRE_Int         num_sends;
   HYPRE_Int         num_recvs;
   hypre_CommType   *send_types;
   hypre_CommType   *recv_types;

   hypre_CommType   *copy_from_type;
   hypre_CommType   *copy_to_type;

   /* these pointers are just to help free up memory for send/from types */
   hypre_CommEntryType *entries;
   HYPRE_Int           *rem_boxnums;
   hypre_Box           *rem_boxes;

   HYPRE_Int         num_orders;
   HYPRE_Int       **orders;            /* num_orders x num_values */

   HYPRE_Int        *recv_data_offsets; /* offsets into recv data (by box) */
   hypre_BoxArray   *recv_data_space;   /* recv data dimensions (by box) */

   hypre_Index       identity_coord;
   hypre_Index       identity_dir;
   HYPRE_Int        *identity_order;

} hypre_CommPkg;

/*--------------------------------------------------------------------------
 * CommHandle:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommHandle_struct
{
   hypre_CommPkg  *comm_pkg;
   double         *send_data;
   double         *recv_data;

   HYPRE_Int       num_requests;
   hypre_MPI_Request    *requests;
   hypre_MPI_Status     *status;

   double        **send_buffers;
   double        **recv_buffers;

   /* set = 0, add = 1 */
   HYPRE_Int       action;

} hypre_CommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommInto
 *--------------------------------------------------------------------------*/
 
#define hypre_CommInfoSendBoxes(info)      (info -> send_boxes)
#define hypre_CommInfoSendStride(info)     (info -> send_stride)
#define hypre_CommInfoSendProcesses(info)  (info -> send_processes)
#define hypre_CommInfoSendRBoxnums(info)   (info -> send_rboxnums)
#define hypre_CommInfoSendRBoxes(info)     (info -> send_rboxes)
                                           
#define hypre_CommInfoRecvBoxes(info)      (info -> recv_boxes)
#define hypre_CommInfoRecvStride(info)     (info -> recv_stride)
#define hypre_CommInfoRecvProcesses(info)  (info -> recv_processes)
#define hypre_CommInfoRecvRBoxnums(info)   (info -> recv_rboxnums)
#define hypre_CommInfoRecvRBoxes(info)     (info -> recv_rboxes)
                                           
#define hypre_CommInfoNumTransforms(info)  (info -> num_transforms)
#define hypre_CommInfoCoords(info)         (info -> coords)
#define hypre_CommInfoDirs(info)           (info -> dirs)
#define hypre_CommInfoSendTransforms(info) (info -> send_transforms)
#define hypre_CommInfoRecvTransforms(info) (info -> recv_transforms)
                                           
#define hypre_CommInfoBoxesMatch(info)     (info -> boxes_match)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommEntryType
 *--------------------------------------------------------------------------*/
 
#define hypre_CommEntryTypeOffset(entry)       (entry -> offset)
#define hypre_CommEntryTypeDim(entry)          (entry -> dim)
#define hypre_CommEntryTypeLengthArray(entry)  (entry -> length_array)
#define hypre_CommEntryTypeStrideArray(entry)  (entry -> stride_array)
#define hypre_CommEntryTypeOrder(entry)        (entry -> order)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommType
 *--------------------------------------------------------------------------*/
 
#define hypre_CommTypeProc(type)          (type -> proc)
#define hypre_CommTypeBufsize(type)       (type -> bufsize)
#define hypre_CommTypeNumEntries(type)    (type -> num_entries)
#define hypre_CommTypeEntries(type)       (type -> entries)
#define hypre_CommTypeEntry(type, i)     &(type -> entries[i])

#define hypre_CommTypeRemBoxnums(type)    (type -> rem_boxnums)
#define hypre_CommTypeRemBoxnum(type, i)  (type -> rem_boxnums[i])
#define hypre_CommTypeRemBoxes(type)      (type -> rem_boxes)
#define hypre_CommTypeRemBox(type, i)    &(type -> rem_boxes[i])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_CommPkgComm(comm_pkg)            (comm_pkg -> comm)

#define hypre_CommPkgFirstComm(comm_pkg)       (comm_pkg -> first_comm)

#define hypre_CommPkgNumValues(comm_pkg)       (comm_pkg -> num_values)
#define hypre_CommPkgSendStride(comm_pkg)      (comm_pkg -> send_stride)
#define hypre_CommPkgRecvStride(comm_pkg)      (comm_pkg -> recv_stride)
#define hypre_CommPkgSendBufsize(comm_pkg)     (comm_pkg -> send_bufsize)
#define hypre_CommPkgRecvBufsize(comm_pkg)     (comm_pkg -> recv_bufsize)
                                               
#define hypre_CommPkgNumSends(comm_pkg)        (comm_pkg -> num_sends)
#define hypre_CommPkgNumRecvs(comm_pkg)        (comm_pkg -> num_recvs)
#define hypre_CommPkgSendTypes(comm_pkg)       (comm_pkg -> send_types)
#define hypre_CommPkgSendType(comm_pkg, i)    &(comm_pkg -> send_types[i])
#define hypre_CommPkgRecvTypes(comm_pkg)       (comm_pkg -> recv_types)
#define hypre_CommPkgRecvType(comm_pkg, i)    &(comm_pkg -> recv_types[i])

#define hypre_CommPkgCopyFromType(comm_pkg)    (comm_pkg -> copy_from_type)
#define hypre_CommPkgCopyToType(comm_pkg)      (comm_pkg -> copy_to_type)

#define hypre_CommPkgEntries(comm_pkg)         (comm_pkg -> entries)
#define hypre_CommPkgRemBoxnums(comm_pkg)      (comm_pkg -> rem_boxnums)
#define hypre_CommPkgRemBoxes(comm_pkg)        (comm_pkg -> rem_boxes)

#define hypre_CommPkgNumOrders(comm_pkg)       (comm_pkg -> num_orders)
#define hypre_CommPkgOrders(comm_pkg)          (comm_pkg -> orders)

#define hypre_CommPkgRecvDataOffsets(comm_pkg) (comm_pkg -> recv_data_offsets)
#define hypre_CommPkgRecvDataSpace(comm_pkg)   (comm_pkg -> recv_data_space)

#define hypre_CommPkgIdentityCoord(comm_pkg)   (comm_pkg -> identity_coord)
#define hypre_CommPkgIdentityDir(comm_pkg)     (comm_pkg -> identity_dir)
#define hypre_CommPkgIdentityOrder(comm_pkg)   (comm_pkg -> identity_order)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_CommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_CommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_CommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_CommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_CommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_CommHandleStatus(comm_handle)      (comm_handle -> status)
#define hypre_CommHandleSendBuffers(comm_handle) (comm_handle -> send_buffers)
#define hypre_CommHandleRecvBuffers(comm_handle) (comm_handle -> recv_buffers)
#define hypre_CommHandleAction(comm_handle)      (comm_handle -> action)

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Header info for computation
 *
 *****************************************************************************/

#ifndef hypre_COMPUTATION_HEADER
#define hypre_COMPUTATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_ComputeInfo:
 *--------------------------------------------------------------------------*/

typedef struct hypre_ComputeInfo_struct
{
   hypre_CommInfo        *comm_info;

   hypre_BoxArrayArray   *indt_boxes;
   hypre_BoxArrayArray   *dept_boxes;
   hypre_Index            stride;

} hypre_ComputeInfo;

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
   HYPRE_Int              num_values;

} hypre_ComputePkg;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ComputeInfo
 *--------------------------------------------------------------------------*/
 
#define hypre_ComputeInfoCommInfo(info)     (info -> comm_info)
#define hypre_ComputeInfoIndtBoxes(info)    (info -> indt_boxes)
#define hypre_ComputeInfoDeptBoxes(info)    (info -> dept_boxes)
#define hypre_ComputeInfoStride(info)       (info -> stride)

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
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the hypre_StructMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_MATRIX_HEADER
#define hypre_STRUCT_MATRIX_HEADER

#include <assert.h>

/*--------------------------------------------------------------------------
 * hypre_StructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructMatrix_struct
{
   MPI_Comm              comm;

   hypre_StructGrid     *grid;
   hypre_StructStencil  *user_stencil;
   hypre_StructStencil  *stencil;
   HYPRE_Int             num_values;   /* Number of "stored" coefficients */

   hypre_BoxArray       *data_space;

   double               *data;         /* Pointer to matrix data */
   HYPRE_Int             data_alloced; /* Boolean used for freeing data */
   HYPRE_Int             data_size;    /* Size of matrix data */
   HYPRE_Int           **data_indices; /* num-boxes by stencil-size array
                                          of indices into the data array.
                                          data_indices[b][s] is the starting
                                          index of matrix data corresponding
                                          to box b and stencil coefficient s */
   HYPRE_Int             constant_coefficient;  /* normally 0; set to 1 for
                                                   constant coefficient matrices
                                                   or 2 for constant coefficient
                                                   with variable diagonal */
                      
   HYPRE_Int             symmetric;    /* Is the matrix symmetric */
   HYPRE_Int            *symm_elements;/* Which elements are "symmetric" */
   HYPRE_Int             num_ghost[6]; /* Num ghost layers in each direction */
                      
   HYPRE_Int             global_size;  /* Total number of nonzero coeffs */

   hypre_CommPkg        *comm_pkg;     /* Info on how to update ghost data */

   HYPRE_Int             ref_count;

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
#define hypre_StructMatrixConstantCoefficient(matrix) ((matrix) -> constant_coefficient)
#define hypre_StructMatrixSymmetric(matrix)     ((matrix) -> symmetric)
#define hypre_StructMatrixSymmElements(matrix)  ((matrix) -> symm_elements)
#define hypre_StructMatrixNumGhost(matrix)      ((matrix) -> num_ghost)
#define hypre_StructMatrixGlobalSize(matrix)    ((matrix) -> global_size)
#define hypre_StructMatrixCommPkg(matrix)       ((matrix) -> comm_pkg)
#define hypre_StructMatrixRefCount(matrix)      ((matrix) -> ref_count)

#define hypre_StructMatrixDim(matrix) \
hypre_StructGridDim(hypre_StructMatrixGrid(matrix))

#define hypre_StructMatrixBox(matrix, b) \
hypre_BoxArrayBox(hypre_StructMatrixDataSpace(matrix), b)

#define hypre_StructMatrixBoxData(matrix, b, s) \
(hypre_StructMatrixData(matrix) + hypre_StructMatrixDataIndices(matrix)[b][s])

#define hypre_StructMatrixBoxDataValue(matrix, b, s, index) \
(hypre_StructMatrixBoxData(matrix, b, s) + \
 hypre_BoxIndexRank(hypre_StructMatrixBox(matrix, b), index))

#define hypre_CCStructMatrixBoxDataValue(matrix, b, s, index) \
(hypre_StructMatrixBoxData(matrix, b, s) + \
 hypre_CCBoxIndexRank(hypre_StructMatrixBox(matrix, b), index))

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/

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
   HYPRE_Int             data_alloced; /* Boolean used for freeing data */
   HYPRE_Int             data_size;    /* Size of vector data */
   HYPRE_Int            *data_indices; /* num-boxes array of indices into
                                          the data array.  data_indices[b]
                                          is the starting index of vector
                                          data corresponding to box b. */
                      
   HYPRE_Int             num_ghost[6]; /* Num ghost layers in each direction */
   HYPRE_Int             bghost_not_clear; /* Are boundary ghosts clear? */
                      
   HYPRE_Int             global_size;  /* Total number coefficients */

   HYPRE_Int             ref_count;

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
#define hypre_StructVectorBGhostNotClear(vector)((vector) -> bghost_not_clear)
#define hypre_StructVectorGlobalSize(vector)    ((vector) -> global_size)
#define hypre_StructVectorRefCount(vector)      ((vector) -> ref_count)
 
#define hypre_StructVectorDim(vector) \
hypre_StructGridDim(hypre_StructVectorGrid(vector))

#define hypre_StructVectorBox(vector, b) \
hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), b)
 
#define hypre_StructVectorBoxData(vector, b) \
(hypre_StructVectorData(vector) + hypre_StructVectorDataIndices(vector)[b])
 
#define hypre_StructVectorBoxDataValue(vector, b, index) \
(hypre_StructVectorBoxData(vector, b) + \
 hypre_BoxIndexRank(hypre_StructVectorBox(vector, b), index))

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.28 $
 ***********************************************************************EHEADER*/

/* assumed_part.c */
HYPRE_Int hypre_APSubdivideRegion ( hypre_Box *region , HYPRE_Int dim , HYPRE_Int level , hypre_BoxArray *box_array , HYPRE_Int *num_new_boxes );
HYPRE_Int hypre_APFindMyBoxesInRegions ( hypre_BoxArray *region_array , hypre_BoxArray *my_box_array , HYPRE_Int **p_count_array , double **p_vol_array );
HYPRE_Int hypre_APGetAllBoxesInRegions ( hypre_BoxArray *region_array , hypre_BoxArray *my_box_array , HYPRE_Int **p_count_array , double **p_vol_array , MPI_Comm comm );
HYPRE_Int hypre_APShrinkRegions ( hypre_BoxArray *region_array , hypre_BoxArray *my_box_array , MPI_Comm comm );
HYPRE_Int hypre_APPruneRegions ( hypre_BoxArray *region_array , HYPRE_Int **p_count_array , double **p_vol_array );
HYPRE_Int hypre_APRefineRegionsByVol ( hypre_BoxArray *region_array , double *vol_array , HYPRE_Int max_regions , double gamma , HYPRE_Int dim , HYPRE_Int *return_code , MPI_Comm comm );
HYPRE_Int hypre_StructAssumedPartitionCreate ( HYPRE_Int dim , hypre_Box *bounding_box , double global_boxes_size , HYPRE_Int global_num_boxes , hypre_BoxArray *local_boxes , HYPRE_Int *local_boxnums , HYPRE_Int max_regions , HYPRE_Int max_refinements , double gamma , MPI_Comm comm , hypre_StructAssumedPart **p_assumed_partition );
HYPRE_Int hypre_StructAssumedPartitionDestroy ( hypre_StructAssumedPart *assumed_part );
HYPRE_Int hypre_APFillResponseStructAssumedPart ( void *p_recv_contact_buf , HYPRE_Int contact_size , HYPRE_Int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , HYPRE_Int *response_message_size );
HYPRE_Int hypre_StructAssumedPartitionGetRegionsFromProc ( hypre_StructAssumedPart *assumed_part , HYPRE_Int proc_id , hypre_BoxArray *assumed_regions );
HYPRE_Int hypre_StructAssumedPartitionGetProcsFromBox ( hypre_StructAssumedPart *assumed_part , hypre_Box *box , HYPRE_Int *num_proc_array , HYPRE_Int *size_alloc_proc_array , HYPRE_Int **p_proc_array );

/* box_algebra.c */
HYPRE_Int hypre_IntersectBoxes ( hypre_Box *box1 , hypre_Box *box2 , hypre_Box *ibox );
HYPRE_Int hypre_SubtractBoxes ( hypre_Box *box1 , hypre_Box *box2 , hypre_BoxArray *box_array );
HYPRE_Int hypre_SubtractBoxArrays ( hypre_BoxArray *box_array1 , hypre_BoxArray *box_array2 , hypre_BoxArray *tmp_box_array );
HYPRE_Int hypre_UnionBoxes ( hypre_BoxArray *boxes );
HYPRE_Int hypre_MinUnionBoxes ( hypre_BoxArray *boxes );

/* box_alloc.c */
HYPRE_Int hypre_BoxInitializeMemory ( const HYPRE_Int at_a_time );
HYPRE_Int hypre_BoxFinalizeMemory ( void );
hypre_Box *hypre_BoxAlloc ( void );
HYPRE_Int hypre_BoxFree ( hypre_Box *box );

/* box_boundary.c */
HYPRE_Int hypre_BoxBoundaryIntersect ( hypre_Box *box , hypre_StructGrid *grid , HYPRE_Int d , HYPRE_Int dir , hypre_BoxArray *boundary );
HYPRE_Int hypre_BoxBoundaryG ( hypre_Box *box , hypre_StructGrid *g , hypre_BoxArray *boundary );
HYPRE_Int hypre_BoxBoundaryDG ( hypre_Box *box , hypre_StructGrid *g , hypre_BoxArray *boundarym , hypre_BoxArray *boundaryp , HYPRE_Int d );

/* box.c */
hypre_Box *hypre_BoxCreate ( void );
HYPRE_Int hypre_BoxSetExtents ( hypre_Box *box , hypre_Index imin , hypre_Index imax );
hypre_BoxArray *hypre_BoxArrayCreate ( HYPRE_Int size );
HYPRE_Int hypre_BoxArraySetSize ( hypre_BoxArray *box_array , HYPRE_Int size );
hypre_BoxArrayArray *hypre_BoxArrayArrayCreate ( HYPRE_Int size );
HYPRE_Int hypre_BoxDestroy ( hypre_Box *box );
HYPRE_Int hypre_BoxArrayDestroy ( hypre_BoxArray *box_array );
HYPRE_Int hypre_BoxArrayArrayDestroy ( hypre_BoxArrayArray *box_array_array );
hypre_Box *hypre_BoxDuplicate ( hypre_Box *box );
hypre_BoxArray *hypre_BoxArrayDuplicate ( hypre_BoxArray *box_array );
hypre_BoxArrayArray *hypre_BoxArrayArrayDuplicate ( hypre_BoxArrayArray *box_array_array );
HYPRE_Int hypre_AppendBox ( hypre_Box *box , hypre_BoxArray *box_array );
HYPRE_Int hypre_DeleteBox ( hypre_BoxArray *box_array , HYPRE_Int index );
HYPRE_Int hypre_AppendBoxArray ( hypre_BoxArray *box_array_0 , hypre_BoxArray *box_array_1 );
HYPRE_Int hypre_BoxGetSize ( hypre_Box *box , hypre_Index size );
HYPRE_Int hypre_BoxGetStrideSize ( hypre_Box *box , hypre_Index stride , hypre_Index size );
HYPRE_Int hypre_BoxGetStrideVolume ( hypre_Box *box , hypre_Index stride , HYPRE_Int *volume_ptr );
HYPRE_Int hypre_BoxExpand ( hypre_Box *box , HYPRE_Int *numexp );
HYPRE_Int hypre_DeleteMultipleBoxes ( hypre_BoxArray *box_array , HYPRE_Int *indices , HYPRE_Int num );
HYPRE_Int hypre_MaxIndexPosition ( hypre_Index index , HYPRE_Int *position );
HYPRE_Int hypre_MinIndexPosition ( hypre_Index index , HYPRE_Int *position );
HYPRE_Int hypre_BoxExpandConstant ( hypre_Box *box , HYPRE_Int expand );
HYPRE_Int hypre_BoxExpandConstantDim ( hypre_Box *box , HYPRE_Int *expand );

/* box_manager.c */
HYPRE_Int hypre_BoxManEntrySetInfo ( hypre_BoxManEntry *entry , void *info );
HYPRE_Int hypre_BoxManEntryGetInfo ( hypre_BoxManEntry *entry , void **info_ptr );
HYPRE_Int hypre_BoxManEntryGetExtents ( hypre_BoxManEntry *entry , hypre_Index imin , hypre_Index imax );
HYPRE_Int hypre_BoxManEntryCopy ( hypre_BoxManEntry *fromentry , hypre_BoxManEntry *toentry );
HYPRE_Int hypre_BoxManSetAllGlobalKnown ( hypre_BoxManager *manager , HYPRE_Int known );
HYPRE_Int hypre_BoxManGetAllGlobalKnown ( hypre_BoxManager *manager , HYPRE_Int *known );
HYPRE_Int hypre_BoxManSetIsEntriesSort ( hypre_BoxManager *manager , HYPRE_Int is_sort );
HYPRE_Int hypre_BoxManGetIsEntriesSort ( hypre_BoxManager *manager , HYPRE_Int *is_sort );
HYPRE_Int hypre_BoxManGetGlobalIsGatherCalled ( hypre_BoxManager *manager , MPI_Comm comm , HYPRE_Int *is_gather );
HYPRE_Int hypre_BoxManGetAssumedPartition ( hypre_BoxManager *manager , hypre_StructAssumedPart **assumed_partition );
HYPRE_Int hypre_BoxManSetAssumedPartition ( hypre_BoxManager *manager , hypre_StructAssumedPart *assumed_partition );
HYPRE_Int hypre_BoxManSetBoundingBox ( hypre_BoxManager *manager , hypre_Box *bounding_box );
HYPRE_Int hypre_BoxManSetNumGhost ( hypre_BoxManager *manager , HYPRE_Int *num_ghost );
HYPRE_Int hypre_BoxManDeleteMultipleEntriesAndInfo ( hypre_BoxManager *manager , HYPRE_Int *indices , HYPRE_Int num );
HYPRE_Int hypre_BoxManCreate ( HYPRE_Int max_nentries , HYPRE_Int info_size , HYPRE_Int dim , hypre_Box *bounding_box , MPI_Comm comm , hypre_BoxManager **manager_ptr );
HYPRE_Int hypre_BoxManIncSize ( hypre_BoxManager *manager , HYPRE_Int inc_size );
HYPRE_Int hypre_BoxManDestroy ( hypre_BoxManager *manager );
HYPRE_Int hypre_BoxManAddEntry ( hypre_BoxManager *manager , hypre_Index imin , hypre_Index imax , HYPRE_Int proc_id , HYPRE_Int box_id , void *info );
HYPRE_Int hypre_BoxManGetEntry ( hypre_BoxManager *manager , HYPRE_Int proc , HYPRE_Int id , hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_BoxManGetAllEntries ( hypre_BoxManager *manager , HYPRE_Int *num_entries , hypre_BoxManEntry **entries );
HYPRE_Int hypre_BoxManGetAllEntriesBoxes ( hypre_BoxManager *manager , hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxManGetLocalEntriesBoxes ( hypre_BoxManager *manager , hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxManGetAllEntriesBoxesProc ( hypre_BoxManager *manager , hypre_BoxArray *boxes , HYPRE_Int **procs_ptr );
HYPRE_Int hypre_BoxManGatherEntries ( hypre_BoxManager *manager , hypre_Index imin , hypre_Index imax );
HYPRE_Int hypre_BoxManAssemble ( hypre_BoxManager *manager );
HYPRE_Int hypre_BoxManIntersect ( hypre_BoxManager *manager , hypre_Index ilower , hypre_Index iupper , hypre_BoxManEntry ***entries_ptr , HYPRE_Int *nentries_ptr );
HYPRE_Int hypre_FillResponseBoxManAssemble1 ( void *p_recv_contact_buf , HYPRE_Int contact_size , HYPRE_Int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseBoxManAssemble2 ( void *p_recv_contact_buf , HYPRE_Int contact_size , HYPRE_Int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , HYPRE_Int *response_message_size );
void hypre_entryqsort2 ( HYPRE_Int *v , hypre_BoxManEntry **ent , HYPRE_Int left , HYPRE_Int right );
void hypre_entryswap2 ( HYPRE_Int *v , hypre_BoxManEntry **ent , HYPRE_Int i , HYPRE_Int j );

/* communication_info.c */
HYPRE_Int hypre_CommInfoCreate ( hypre_BoxArrayArray *send_boxes , hypre_BoxArrayArray *recv_boxes , HYPRE_Int **send_procs , HYPRE_Int **recv_procs , HYPRE_Int **send_rboxnums , HYPRE_Int **recv_rboxnums , hypre_BoxArrayArray *send_rboxes , hypre_BoxArrayArray *recv_rboxes , HYPRE_Int boxes_match , hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CommInfoSetTransforms ( hypre_CommInfo *comm_info , HYPRE_Int num_transforms , hypre_Index *coords , hypre_Index *dirs , HYPRE_Int **send_transforms , HYPRE_Int **recv_transforms );
HYPRE_Int hypre_CommInfoGetTransforms ( hypre_CommInfo *comm_info , HYPRE_Int *num_transforms , hypre_Index **coords , hypre_Index **dirs );
HYPRE_Int hypre_CommInfoProjectSend ( hypre_CommInfo *comm_info , hypre_Index index , hypre_Index stride );
HYPRE_Int hypre_CommInfoProjectRecv ( hypre_CommInfo *comm_info , hypre_Index index , hypre_Index stride );
HYPRE_Int hypre_CommInfoDestroy ( hypre_CommInfo *comm_info );
HYPRE_Int hypre_CreateCommInfoFromStencil ( hypre_StructGrid *grid , hypre_StructStencil *stencil , hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromNumGhost ( hypre_StructGrid *grid , HYPRE_Int *num_ghost , hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromGrids ( hypre_StructGrid *from_grid , hypre_StructGrid *to_grid , hypre_CommInfo **comm_info_ptr );

/* computation.c */
HYPRE_Int hypre_ComputeInfoCreate ( hypre_CommInfo *comm_info , hypre_BoxArrayArray *indt_boxes , hypre_BoxArrayArray *dept_boxes , hypre_ComputeInfo **compute_info_ptr );
HYPRE_Int hypre_ComputeInfoProjectSend ( hypre_ComputeInfo *compute_info , hypre_Index index , hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectRecv ( hypre_ComputeInfo *compute_info , hypre_Index index , hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectComp ( hypre_ComputeInfo *compute_info , hypre_Index index , hypre_Index stride );
HYPRE_Int hypre_ComputeInfoDestroy ( hypre_ComputeInfo *compute_info );
HYPRE_Int hypre_CreateComputeInfo ( hypre_StructGrid *grid , hypre_StructStencil *stencil , hypre_ComputeInfo **compute_info_ptr );
HYPRE_Int hypre_ComputePkgCreate ( hypre_ComputeInfo *compute_info , hypre_BoxArray *data_space , HYPRE_Int num_values , hypre_StructGrid *grid , hypre_ComputePkg **compute_pkg_ptr );
HYPRE_Int hypre_ComputePkgDestroy ( hypre_ComputePkg *compute_pkg );
HYPRE_Int hypre_InitializeIndtComputations ( hypre_ComputePkg *compute_pkg , double *data , hypre_CommHandle **comm_handle_ptr );
HYPRE_Int hypre_FinalizeIndtComputations ( hypre_CommHandle *comm_handle );

/* grow.c */
hypre_BoxArray *hypre_GrowBoxByStencil ( hypre_Box *box , hypre_StructStencil *stencil , HYPRE_Int transpose );
hypre_BoxArrayArray *hypre_GrowBoxArrayByStencil ( hypre_BoxArray *box_array , hypre_StructStencil *stencil , HYPRE_Int transpose );

/* HYPRE_struct_grid.c */
HYPRE_Int HYPRE_StructGridCreate ( MPI_Comm comm , HYPRE_Int dim , HYPRE_StructGrid *grid );
HYPRE_Int HYPRE_StructGridDestroy ( HYPRE_StructGrid grid );
HYPRE_Int HYPRE_StructGridSetExtents ( HYPRE_StructGrid grid , HYPRE_Int *ilower , HYPRE_Int *iupper );
HYPRE_Int HYPRE_StructGridSetPeriodic ( HYPRE_StructGrid grid , HYPRE_Int *periodic );
HYPRE_Int HYPRE_StructGridAssemble ( HYPRE_StructGrid grid );
HYPRE_Int HYPRE_StructGridSetNumGhost ( HYPRE_StructGrid grid , HYPRE_Int *num_ghost );

/* HYPRE_struct_matrix.c */
HYPRE_Int HYPRE_StructMatrixCreate ( MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructMatrix *matrix );
HYPRE_Int HYPRE_StructMatrixDestroy ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixInitialize ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixSetValues ( HYPRE_StructMatrix matrix , HYPRE_Int *grid_index , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values );
HYPRE_Int HYPRE_StructMatrixGetValues ( HYPRE_StructMatrix matrix , HYPRE_Int *grid_index , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values );
HYPRE_Int HYPRE_StructMatrixSetBoxValues ( HYPRE_StructMatrix matrix , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values );
HYPRE_Int HYPRE_StructMatrixGetBoxValues ( HYPRE_StructMatrix matrix , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values );
HYPRE_Int HYPRE_StructMatrixSetConstantValues ( HYPRE_StructMatrix matrix , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values );
HYPRE_Int HYPRE_StructMatrixAddToValues ( HYPRE_StructMatrix matrix , HYPRE_Int *grid_index , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values );
HYPRE_Int HYPRE_StructMatrixAddToBoxValues ( HYPRE_StructMatrix matrix , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values );
HYPRE_Int HYPRE_StructMatrixAddToConstantValues ( HYPRE_StructMatrix matrix , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values );
HYPRE_Int HYPRE_StructMatrixAssemble ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixSetNumGhost ( HYPRE_StructMatrix matrix , HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructMatrixGetGrid ( HYPRE_StructMatrix matrix , HYPRE_StructGrid *grid );
HYPRE_Int HYPRE_StructMatrixSetSymmetric ( HYPRE_StructMatrix matrix , HYPRE_Int symmetric );
HYPRE_Int HYPRE_StructMatrixSetConstantEntries ( HYPRE_StructMatrix matrix , HYPRE_Int nentries , HYPRE_Int *entries );
HYPRE_Int HYPRE_StructMatrixPrint ( const char *filename , HYPRE_StructMatrix matrix , HYPRE_Int all );
HYPRE_Int HYPRE_StructMatrixMatvec ( double alpha , HYPRE_StructMatrix A , HYPRE_StructVector x , double beta , HYPRE_StructVector y );

/* HYPRE_struct_stencil.c */
HYPRE_Int HYPRE_StructStencilCreate ( HYPRE_Int dim , HYPRE_Int size , HYPRE_StructStencil *stencil );
HYPRE_Int HYPRE_StructStencilSetElement ( HYPRE_StructStencil stencil , HYPRE_Int element_index , HYPRE_Int *offset );
HYPRE_Int HYPRE_StructStencilDestroy ( HYPRE_StructStencil stencil );

/* HYPRE_struct_vector.c */
HYPRE_Int HYPRE_StructVectorCreate ( MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructVector *vector );
HYPRE_Int HYPRE_StructVectorDestroy ( HYPRE_StructVector struct_vector );
HYPRE_Int HYPRE_StructVectorInitialize ( HYPRE_StructVector vector );
HYPRE_Int HYPRE_StructVectorSetValues ( HYPRE_StructVector vector , HYPRE_Int *grid_index , double values );
HYPRE_Int HYPRE_StructVectorSetBoxValues ( HYPRE_StructVector vector , HYPRE_Int *ilower , HYPRE_Int *iupper , double *values );
HYPRE_Int HYPRE_StructVectorAddToValues ( HYPRE_StructVector vector , HYPRE_Int *grid_index , double values );
HYPRE_Int HYPRE_StructVectorAddToBoxValues ( HYPRE_StructVector vector , HYPRE_Int *ilower , HYPRE_Int *iupper , double *values );
HYPRE_Int HYPRE_StructVectorScaleValues ( HYPRE_StructVector vector , double factor );
HYPRE_Int HYPRE_StructVectorGetValues ( HYPRE_StructVector vector , HYPRE_Int *grid_index , double *values );
HYPRE_Int HYPRE_StructVectorGetBoxValues ( HYPRE_StructVector vector , HYPRE_Int *ilower , HYPRE_Int *iupper , double *values );
HYPRE_Int HYPRE_StructVectorAssemble ( HYPRE_StructVector vector );
HYPRE_Int HYPRE_StructVectorPrint ( const char *filename , HYPRE_StructVector vector , HYPRE_Int all );
HYPRE_Int HYPRE_StructVectorSetNumGhost ( HYPRE_StructVector vector , HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructVectorCopy ( HYPRE_StructVector x , HYPRE_StructVector y );
HYPRE_Int HYPRE_StructVectorSetConstantValues ( HYPRE_StructVector vector , double values );
HYPRE_Int HYPRE_StructVectorGetMigrateCommPkg ( HYPRE_StructVector from_vector , HYPRE_StructVector to_vector , HYPRE_CommPkg *comm_pkg );
HYPRE_Int HYPRE_StructVectorMigrate ( HYPRE_CommPkg comm_pkg , HYPRE_StructVector from_vector , HYPRE_StructVector to_vector );
HYPRE_Int HYPRE_CommPkgDestroy ( HYPRE_CommPkg comm_pkg );

/* project.c */
HYPRE_Int hypre_ProjectBox ( hypre_Box *box , hypre_Index index , hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArray ( hypre_BoxArray *box_array , hypre_Index index , hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArrayArray ( hypre_BoxArrayArray *box_array_array , hypre_Index index , hypre_Index stride );

/* struct_axpy.c */
HYPRE_Int hypre_StructAxpy ( double alpha , hypre_StructVector *x , hypre_StructVector *y );

/* struct_communication.c */
HYPRE_Int hypre_CommPkgCreate ( hypre_CommInfo *comm_info , hypre_BoxArray *send_data_space , hypre_BoxArray *recv_data_space , HYPRE_Int num_values , HYPRE_Int **orders , HYPRE_Int reverse , MPI_Comm comm , hypre_CommPkg **comm_pkg_ptr );
HYPRE_Int hypre_CommTypeSetEntries ( hypre_CommType *comm_type , HYPRE_Int *boxnums , hypre_Box *boxes , hypre_Index stride , hypre_Index coord , hypre_Index dir , HYPRE_Int *order , hypre_BoxArray *data_space , HYPRE_Int *data_offsets );
HYPRE_Int hypre_CommTypeSetEntry ( hypre_Box *box , hypre_Index stride , hypre_Index coord , hypre_Index dir , HYPRE_Int *order , hypre_Box *data_box , HYPRE_Int data_box_offset , hypre_CommEntryType *comm_entry );
HYPRE_Int hypre_InitializeCommunication ( hypre_CommPkg *comm_pkg , double *send_data , double *recv_data , HYPRE_Int action , HYPRE_Int tag , hypre_CommHandle **comm_handle_ptr );
HYPRE_Int hypre_FinalizeCommunication ( hypre_CommHandle *comm_handle );
HYPRE_Int hypre_ExchangeLocalData ( hypre_CommPkg *comm_pkg , double *send_data , double *recv_data , HYPRE_Int action );
HYPRE_Int hypre_CommPkgDestroy ( hypre_CommPkg *comm_pkg );

/* struct_copy.c */
HYPRE_Int hypre_StructCopy ( hypre_StructVector *x , hypre_StructVector *y );
HYPRE_Int hypre_StructPartialCopy ( hypre_StructVector *x , hypre_StructVector *y , hypre_BoxArrayArray *array_boxes );

/* struct_grid.c */
HYPRE_Int hypre_StructGridCreate ( MPI_Comm comm , HYPRE_Int dim , hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridRef ( hypre_StructGrid *grid , hypre_StructGrid **grid_ref );
HYPRE_Int hypre_StructGridDestroy ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridSetPeriodic ( hypre_StructGrid *grid , hypre_Index periodic );
HYPRE_Int hypre_StructGridSetExtents ( hypre_StructGrid *grid , hypre_Index ilower , hypre_Index iupper );
HYPRE_Int hypre_StructGridSetBoxes ( hypre_StructGrid *grid , hypre_BoxArray *boxes );
HYPRE_Int hypre_StructGridSetBoundingBox ( hypre_StructGrid *grid , hypre_Box *new_bb );
HYPRE_Int hypre_StructGridSetIDs ( hypre_StructGrid *grid , HYPRE_Int *ids );
HYPRE_Int hypre_StructGridSetBoxManager ( hypre_StructGrid *grid , hypre_BoxManager *boxman );
HYPRE_Int hypre_StructGridSetMaxDistance ( hypre_StructGrid *grid , hypre_Index dist );
HYPRE_Int hypre_StructGridAssemble ( hypre_StructGrid *grid );
HYPRE_Int hypre_GatherAllBoxes ( MPI_Comm comm , hypre_BoxArray *boxes , hypre_BoxArray **all_boxes_ptr , HYPRE_Int **all_procs_ptr , HYPRE_Int *first_local_ptr );
HYPRE_Int hypre_ComputeBoxnums ( hypre_BoxArray *boxes , HYPRE_Int *procs , HYPRE_Int **boxnums_ptr );
HYPRE_Int hypre_StructGridPrint ( FILE *file , hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridRead ( MPI_Comm comm , FILE *file , hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridSetNumGhost ( hypre_StructGrid *grid , HYPRE_Int *num_ghost );

/* struct_innerprod.c */
double hypre_StructInnerProd ( hypre_StructVector *x , hypre_StructVector *y );

/* struct_io.c */
HYPRE_Int hypre_PrintBoxArrayData ( FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , HYPRE_Int num_values , double *data );
HYPRE_Int hypre_PrintCCVDBoxArrayData ( FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , HYPRE_Int num_values , HYPRE_Int center_rank , HYPRE_Int stencil_size , HYPRE_Int *symm_elements , double *data );
HYPRE_Int hypre_PrintCCBoxArrayData ( FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , HYPRE_Int num_values , double *data );
HYPRE_Int hypre_ReadBoxArrayData ( FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , HYPRE_Int num_values , double *data );
HYPRE_Int hypre_ReadBoxArrayData_CC ( FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , HYPRE_Int stencil_size , HYPRE_Int real_stencil_size , HYPRE_Int constant_coefficient , double *data );

/* struct_matrix.c */
double *hypre_StructMatrixExtractPointerByIndex ( hypre_StructMatrix *matrix , HYPRE_Int b , hypre_Index index );
hypre_StructMatrix *hypre_StructMatrixCreate ( MPI_Comm comm , hypre_StructGrid *grid , hypre_StructStencil *user_stencil );
hypre_StructMatrix *hypre_StructMatrixRef ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixDestroy ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitializeShell ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitializeData ( hypre_StructMatrix *matrix , double *data );
HYPRE_Int hypre_StructMatrixInitialize ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixSetValues ( hypre_StructMatrix *matrix , hypre_Index grid_index , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values , HYPRE_Int action , HYPRE_Int boxnum , HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixSetBoxValues ( hypre_StructMatrix *matrix , hypre_Box *set_box , hypre_Box *value_box , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values , HYPRE_Int action , HYPRE_Int boxnum , HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixSetConstantValues ( hypre_StructMatrix *matrix , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , double *values , HYPRE_Int action );
HYPRE_Int hypre_StructMatrixClearValues ( hypre_StructMatrix *matrix , hypre_Index grid_index , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , HYPRE_Int boxnum , HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixClearBoxValues ( hypre_StructMatrix *matrix , hypre_Box *clear_box , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices , HYPRE_Int boxnum , HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixAssemble ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixSetNumGhost ( hypre_StructMatrix *matrix , HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructMatrixSetConstantCoefficient ( hypre_StructMatrix *matrix , HYPRE_Int constant_coefficient );
HYPRE_Int hypre_StructMatrixSetConstantEntries ( hypre_StructMatrix *matrix , HYPRE_Int nentries , HYPRE_Int *entries );
HYPRE_Int hypre_StructMatrixClearGhostValues ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixPrint ( const char *filename , hypre_StructMatrix *matrix , HYPRE_Int all );
HYPRE_Int hypre_StructMatrixMigrate ( hypre_StructMatrix *from_matrix , hypre_StructMatrix *to_matrix );
hypre_StructMatrix *hypre_StructMatrixRead ( MPI_Comm comm , const char *filename , HYPRE_Int *num_ghost );

/* struct_matrix_mask.c */
hypre_StructMatrix *hypre_StructMatrixCreateMask ( hypre_StructMatrix *matrix , HYPRE_Int num_stencil_indices , HYPRE_Int *stencil_indices );

/* struct_matvec.c */
void *hypre_StructMatvecCreate ( void );
HYPRE_Int hypre_StructMatvecSetup ( void *matvec_vdata , hypre_StructMatrix *A , hypre_StructVector *x );
HYPRE_Int hypre_StructMatvecCompute ( void *matvec_vdata , double alpha , hypre_StructMatrix *A , hypre_StructVector *x , double beta , hypre_StructVector *y );
HYPRE_Int hypre_StructMatvecCC0 ( double alpha , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *y , hypre_BoxArrayArray *compute_box_aa , hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC1 ( double alpha , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *y , hypre_BoxArrayArray *compute_box_aa , hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC2 ( double alpha , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *y , hypre_BoxArrayArray *compute_box_aa , hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecDestroy ( void *matvec_vdata );
HYPRE_Int hypre_StructMatvec ( double alpha , hypre_StructMatrix *A , hypre_StructVector *x , double beta , hypre_StructVector *y );

/* struct_overlap_innerprod.c */
double hypre_StructOverlapInnerProd ( hypre_StructVector *x , hypre_StructVector *y );

/* struct_scale.c */
HYPRE_Int hypre_StructScale ( double alpha , hypre_StructVector *y );

/* struct_stencil.c */
hypre_StructStencil *hypre_StructStencilCreate ( HYPRE_Int dim , HYPRE_Int size , hypre_Index *shape );
hypre_StructStencil *hypre_StructStencilRef ( hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilDestroy ( hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilElementRank ( hypre_StructStencil *stencil , hypre_Index stencil_element );
HYPRE_Int hypre_StructStencilSymmetrize ( hypre_StructStencil *stencil , hypre_StructStencil **symm_stencil_ptr , HYPRE_Int **symm_elements_ptr );

/* struct_vector.c */
hypre_StructVector *hypre_StructVectorCreate ( MPI_Comm comm , hypre_StructGrid *grid );
hypre_StructVector *hypre_StructVectorRef ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorDestroy ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitializeShell ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitializeData ( hypre_StructVector *vector , double *data );
HYPRE_Int hypre_StructVectorInitialize ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorSetValues ( hypre_StructVector *vector , hypre_Index grid_index , double *values , HYPRE_Int action , HYPRE_Int boxnum , HYPRE_Int outside );
HYPRE_Int hypre_StructVectorSetBoxValues ( hypre_StructVector *vector , hypre_Box *set_box , hypre_Box *value_box , double *values , HYPRE_Int action , HYPRE_Int boxnum , HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearValues ( hypre_StructVector *vector , hypre_Index grid_index , HYPRE_Int boxnum , HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearBoxValues ( hypre_StructVector *vector , hypre_Box *clear_box , HYPRE_Int boxnum , HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearAllValues ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorSetNumGhost ( hypre_StructVector *vector , HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructVectorAssemble ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorCopy ( hypre_StructVector *x , hypre_StructVector *y );
HYPRE_Int hypre_StructVectorSetConstantValues ( hypre_StructVector *vector , double values );
HYPRE_Int hypre_StructVectorSetFunctionValues ( hypre_StructVector *vector , double (*fcn )());
HYPRE_Int hypre_StructVectorClearGhostValues ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorClearBoundGhostValues ( hypre_StructVector *vector , HYPRE_Int force );
HYPRE_Int hypre_StructVectorScaleValues ( hypre_StructVector *vector , double factor );
hypre_CommPkg *hypre_StructVectorGetMigrateCommPkg ( hypre_StructVector *from_vector , hypre_StructVector *to_vector );
HYPRE_Int hypre_StructVectorMigrate ( hypre_CommPkg *comm_pkg , hypre_StructVector *from_vector , hypre_StructVector *to_vector );
HYPRE_Int hypre_StructVectorPrint ( const char *filename , hypre_StructVector *vector , HYPRE_Int all );
hypre_StructVector *hypre_StructVectorRead ( MPI_Comm comm , const char *filename , HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructVectorMaxValue ( hypre_StructVector *vector , double *max_value , HYPRE_Int *max_index , hypre_Index max_xyz_index );


#ifdef __cplusplus
}
#endif

#endif

