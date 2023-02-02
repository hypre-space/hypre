/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

typedef HYPRE_Int  hypre_Index[HYPRE_MAXDIM];
typedef HYPRE_Int *hypre_IndexRef;

/*--------------------------------------------------------------------------
 * hypre_Box:
 *--------------------------------------------------------------------------*/

typedef struct hypre_Box_struct
{
   hypre_Index imin;           /* min bounding indices */
   hypre_Index imax;           /* max bounding indices */
   HYPRE_Int   ndim;           /* number of dimensions */

} hypre_Box;

/*--------------------------------------------------------------------------
 * hypre_BoxArray:
 *   An array of boxes.
 *   Since size can be zero, need to store ndim separately.
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxArray_struct
{
   hypre_Box  *boxes;         /* Array of boxes */
   HYPRE_Int   size;          /* Size of box array */
   HYPRE_Int   alloc_size;    /* Size of currently alloced space */
   HYPRE_Int   ndim;          /* number of dimensions */

} hypre_BoxArray;

#define hypre_BoxArrayExcess 10

/*--------------------------------------------------------------------------
 * hypre_BoxArrayArray:
 *   An array of box arrays.
 *   Since size can be zero, need to store ndim separately.
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxArrayArray_struct
{
   hypre_BoxArray  **box_arrays;    /* Array of pointers to box arrays */
   HYPRE_Int         size;          /* Size of box array array */
   HYPRE_Int         ndim;          /* number of dimensions */

} hypre_BoxArrayArray;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_Index
 *--------------------------------------------------------------------------*/

#define hypre_IndexD(index, d)  (index[d])

/* Avoid using these macros */
#define hypre_IndexX(index)     hypre_IndexD(index, 0)
#define hypre_IndexY(index)     hypre_IndexD(index, 1)
#define hypre_IndexZ(index)     hypre_IndexD(index, 2)

/*--------------------------------------------------------------------------
 * Member functions: hypre_Index
 *--------------------------------------------------------------------------*/

/*----- Avoid using these Index macros -----*/

#define hypre_SetIndex3(index, ix, iy, iz) \
( hypre_IndexD(index, 0) = ix,\
  hypre_IndexD(index, 1) = iy,\
  hypre_IndexD(index, 2) = iz )

#define hypre_ClearIndex(index)  hypre_SetIndex(index, 0)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_Box
 *--------------------------------------------------------------------------*/

#define hypre_BoxIMin(box)     ((box) -> imin)
#define hypre_BoxIMax(box)     ((box) -> imax)
#define hypre_BoxNDim(box)     ((box) -> ndim)

#define hypre_BoxIMinD(box, d) (hypre_IndexD(hypre_BoxIMin(box), d))
#define hypre_BoxIMaxD(box, d) (hypre_IndexD(hypre_BoxIMax(box), d))
#define hypre_BoxSizeD(box, d) \
hypre_max(0, (hypre_BoxIMaxD(box, d) - hypre_BoxIMinD(box, d) + 1))

#define hypre_IndexDInBox(index, d, box) \
( hypre_IndexD(index, d) >= hypre_BoxIMinD(box, d) && \
  hypre_IndexD(index, d) <= hypre_BoxIMaxD(box, d) )

/* The first hypre_CCBoxIndexRank is better style because it is similar to
   hypre_BoxIndexRank.  The second one sometimes avoids compiler warnings. */
#define hypre_CCBoxIndexRank(box, index) 0
#define hypre_CCBoxIndexRank_noargs() 0
#define hypre_CCBoxOffsetDistance(box, index) 0

/*----- Avoid using these Box macros -----*/

#define hypre_BoxSizeX(box)    hypre_BoxSizeD(box, 0)
#define hypre_BoxSizeY(box)    hypre_BoxSizeD(box, 1)
#define hypre_BoxSizeZ(box)    hypre_BoxSizeD(box, 2)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxArray
 *--------------------------------------------------------------------------*/

#define hypre_BoxArrayBoxes(box_array)     ((box_array) -> boxes)
#define hypre_BoxArrayBox(box_array, i)    &((box_array) -> boxes[(i)])
#define hypre_BoxArraySize(box_array)      ((box_array) -> size)
#define hypre_BoxArrayAllocSize(box_array) ((box_array) -> alloc_size)
#define hypre_BoxArrayNDim(box_array)      ((box_array) -> ndim)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxArrayArray
 *--------------------------------------------------------------------------*/

#define hypre_BoxArrayArrayBoxArrays(box_array_array) \
((box_array_array) -> box_arrays)
#define hypre_BoxArrayArrayBoxArray(box_array_array, i) \
((box_array_array) -> box_arrays[(i)])
#define hypre_BoxArrayArraySize(box_array_array) \
((box_array_array) -> size)
#define hypre_BoxArrayArrayNDim(box_array_array) \
((box_array_array) -> ndim)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define hypre_ForBoxI(i, box_array) \
for (i = 0; i < hypre_BoxArraySize(box_array); i++)

#define hypre_ForBoxArrayI(i, box_array_array) \
for (i = 0; i < hypre_BoxArrayArraySize(box_array_array); i++)

#define ZYPRE_BOX_PRIVATE hypre__IN,hypre__JN,hypre__I,hypre__J,hypre__d,hypre__i
#define HYPRE_BOX_PRIVATE ZYPRE_BOX_PRIVATE

#define zypre_BoxLoopDeclare() \
HYPRE_Int  hypre__tot, hypre__div, hypre__mod;\
HYPRE_Int  hypre__block, hypre__num_blocks;\
HYPRE_Int  hypre__d, hypre__ndim;\
HYPRE_Int  hypre__I, hypre__J, hypre__IN, hypre__JN;\
HYPRE_Int  hypre__i[HYPRE_MAXDIM+1], hypre__n[HYPRE_MAXDIM+1]

#define zypre_BoxLoopDeclareK(k) \
HYPRE_Int  hypre__ikstart##k, hypre__i0inc##k;\
HYPRE_Int  hypre__sk##k[HYPRE_MAXDIM], hypre__ikinc##k[HYPRE_MAXDIM+1]

#define zypre_BoxLoopInit(ndim, loop_size) \
hypre__ndim = ndim;\
hypre__n[0] = loop_size[0];\
hypre__tot = 1;\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__n[hypre__d] = loop_size[hypre__d];\
   hypre__tot *= hypre__n[hypre__d];\
}\
hypre__n[hypre__ndim] = 2;\
hypre__num_blocks = hypre_NumThreads();\
if (hypre__tot < hypre__num_blocks)\
{\
   hypre__num_blocks = hypre__tot;\
}\
if (hypre__num_blocks > 0)\
{\
   hypre__div = hypre__tot / hypre__num_blocks;\
   hypre__mod = hypre__tot % hypre__num_blocks;\
}

#define zypre_BoxLoopInitK(k, dboxk, startk, stridek, ik) \
hypre__sk##k[0] = stridek[0];\
hypre__ikinc##k[0] = 0;\
ik = hypre_BoxSizeD(dboxk, 0); /* temporarily use ik */\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__sk##k[hypre__d] = ik*stridek[hypre__d];\
   hypre__ikinc##k[hypre__d] = hypre__ikinc##k[hypre__d-1] +\
      hypre__sk##k[hypre__d] - hypre__n[hypre__d-1]*hypre__sk##k[hypre__d-1];\
   ik *= hypre_BoxSizeD(dboxk, hypre__d);\
}\
hypre__i0inc##k = hypre__sk##k[0];\
hypre__ikinc##k[hypre__ndim] = 0;\
hypre__ikstart##k = hypre_BoxIndexRank(dboxk, startk)

#define zypre_BoxLoopSet() \
hypre__IN = hypre__n[0];\
if (hypre__num_blocks > 1)/* in case user sets num_blocks to 1 */\
{\
   hypre__JN = hypre__div + ((hypre__mod > hypre__block) ? 1 : 0);\
   hypre__J = hypre__block * hypre__div + hypre_min(hypre__mod, hypre__block);\
   for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
   {\
      hypre__i[hypre__d] = hypre__J % hypre__n[hypre__d];\
      hypre__J /= hypre__n[hypre__d];\
   }\
}\
else\
{\
   hypre__JN = hypre__tot;\
   for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
   {\
      hypre__i[hypre__d] = 0;\
   }\
}\
hypre__i[hypre__ndim] = 0

#define zypre_BoxLoopSetK(k, ik) \
ik = hypre__ikstart##k;\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   ik += hypre__i[hypre__d]*hypre__sk##k[hypre__d];\
}

#define zypre_BoxLoopInc1() \
hypre__d = 1;\
while ((hypre__i[hypre__d]+2) > hypre__n[hypre__d])\
{\
   hypre__d++;\
}

#define zypre_BoxLoopInc2() \
hypre__i[hypre__d]++;\
while (hypre__d > 1)\
{\
   hypre__d--;\
   hypre__i[hypre__d] = 0;\
}

/* This returns the loop index (of type hypre_Index) for the current iteration,
 * where the numbering starts at 0.  It works even when threading is turned on,
 * as long as 'index' is declared to be private. */
#define zypre_BoxLoopGetIndex(index) \
index[0] = hypre__I;\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   index[hypre__d] = hypre__i[hypre__d];\
}

/* Use this before the For macros below to force only one block */
#define zypre_BoxLoopSetOneBlock() hypre__num_blocks = 1

/* Use this to get the block iteration inside a BoxLoop */
#define zypre_BoxLoopBlock() hypre__block

#define zypre_BasicBoxLoopInitK(k, stridek) \
hypre__sk##k[0] = stridek[0];\
hypre__ikinc##k[0] = 0;\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__sk##k[hypre__d] = stridek[hypre__d];\
   hypre__ikinc##k[hypre__d] = hypre__ikinc##k[hypre__d-1] +\
      hypre__sk##k[hypre__d] - hypre__n[hypre__d-1]*hypre__sk##k[hypre__d-1];\
}\
hypre__i0inc##k = hypre__sk##k[0];\
hypre__ikinc##k[hypre__ndim] = 0;\
hypre__ikstart##k = 0

/*--------------------------------------------------------------------------
 * NOTES - Keep these for reference here and elsewhere in the code
 *--------------------------------------------------------------------------*/

#if 0

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
      i1 = hypre__i1start + i * hypre__sx1 + j * hypre__sy1 + k * hypre__sz1;
      i2 = hypre__i2start + i * hypre__sx2 + j * hypre__sy2 + k * hypre__sz2;
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
            i1 += hypre__sy1 - hypre__nx * hypre__sx1;
            i2 += hypre__sy2 - hypre__nx * hypre__sx2;
         }
         i1 += hypre__sz1 - hypre__ny * hypre__sy1;
         i2 += hypre__sz2 - hypre__ny * hypre__sy2;
      }
   }
}

/*----------------------------------------
 * Idea 2: Simple version of Idea 3 below
 *----------------------------------------*/

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

/*----------------------------------------
 * Idea 3: Approach used in the box loops
 *----------------------------------------*/

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
#endif /* #ifndef hypre_BOX_HEADER */

