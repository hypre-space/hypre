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

#define hypre_BoxLoopDeclare(loop_size, data_box, stride, is, js, ks) \
int  is = (hypre_IndexX(stride));\
int  js = (hypre_IndexY(stride)*hypre_BoxSizeX(data_box));\
int  ks = (hypre_IndexZ(stride)*\
           hypre_BoxSizeX(data_box)*hypre_BoxSizeY(data_box))

/*-------------------------------------------------------------------------
 * Threaded versions of looping macros are in box_pthreads.h.
 *-------------------------------------------------------------------------*/

#ifndef HYPRE_USE_PTHREADS

#define hypre_BoxLoop0Begin(loop_size)\
{\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size)

#define hypre_BoxLoop0For(i, j, k)\
   for (k = 0; k < hypre__nz; k++ )\
   {\
      for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\
         {\


#define hypre_BoxLoop0End }}}}
  



#define hypre_BoxLoop1Begin(loop_size,\
			    data_box1, start1, stride1, i1)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__is1, hypre__js1, hypre__ks1);\
   int hypre__i1start = hypre_BoxIndexRank(data_box1, start1);\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size)

#define hypre_BoxLoop1For(i, j, k, i1)\
   for (k = 0; k < hypre__nz; k++ )\
   {\
      for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\
         {\
            i1 = hypre__i1start + i*hypre__is1 + j*hypre__js1 + k*hypre__ks1;


#define hypre_BoxLoopEnd }}}}
  


#define hypre_BoxLoop2Begin(loop_size,\
			    data_box1, start1, stride1, i1,\
			    data_box2, start2, stride2, i2)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__is1, hypre__js1, hypre__ks1);\
   hypre_BoxLoopDeclare(loop_size, data_box2, stride2,\
                        hypre__is2, hypre__js2, hypre__ks2);\
   int hypre__i1start = hypre_BoxIndexRank(data_box1, start1);\
   int hypre__i2start = hypre_BoxIndexRank(data_box2, start2);\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size)

#define hypre_BoxLoop2For(i, j, k, i1, i2)\
   for (k = 0; k < hypre__nz; k++ )\
   {\
      for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\
         {\
            i1 = hypre__i1start + i*hypre__is1 + j*hypre__js1 + k*hypre__ks1;\
            i2 = hypre__i2start + i*hypre__is2 + j*hypre__js2 + k*hypre__ks2;

#define hypre_BoxLoop2End }}}}


#define hypre_BoxLoop3Begin(loop_size,\
			    data_box1, start1, stride1, i1,\
			    data_box2, start2, stride2, i2,\
                            data_box3, start3, stride3, i3)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__is1, hypre__js1, hypre__ks1);\
   hypre_BoxLoopDeclare(loop_size, data_box2, stride2,\
                        hypre__is2, hypre__js2, hypre__ks2);\
   hypre_BoxLoopDeclare(loop_size, data_box3, stride3,\
                        hypre__is3, hypre__js3, hypre__ks3);\
   int hypre__i1start = hypre_BoxIndexRank(data_box1, start1);\
   int hypre__i2start = hypre_BoxIndexRank(data_box2, start2);\
   int hypre__i3start = hypre_BoxIndexRank(data_box3, start3);\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size)

#define hypre_BoxLoop3For(i, j, k, i1, i2, i3)\
   for (k = 0; k < hypre__nz; k++ )\
   {\
      for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\
         {\
            i1 = hypre__i1start + i*hypre__is1 + j*hypre__js1 + k*hypre__ks1;\
            i2 = hypre__i2start + i*hypre__is2 + j*hypre__js2 + k*hypre__ks2;\
            i3 = hypre__i3start + i*hypre__is3 + j*hypre__js3 + k*hypre__ks3;

#define hypre_BoxLoop3End }}}}


#define hypre_BoxLoop4Begin(loop_size,\
			    data_box1, start1, stride1, i1,\
			    data_box2, start2, stride2, i2,\
                            data_box3, start3, stride3, i3,\
                            data_box4, start4, stride4, i4)\
{\
   hypre_BoxLoopDeclare(loop_size, data_box1, stride1,\
                        hypre__is1, hypre__js1, hypre__ks1);\
   hypre_BoxLoopDeclare(loop_size, data_box2, stride2,\
                        hypre__is2, hypre__js2, hypre__ks2);\
   hypre_BoxLoopDeclare(loop_size, data_box3, stride3,\
                        hypre__is3, hypre__js3, hypre__ks3);\
   hypre_BoxLoopDeclare(loop_size, data_box4, stride4,\
                        hypre__is4, hypre__js4, hypre__ks4);\
   int hypre__i1start = hypre_BoxIndexRank(data_box1, start1);\
   int hypre__i2start = hypre_BoxIndexRank(data_box2, start2);\
   int hypre__i3start = hypre_BoxIndexRank(data_box3, start3);\
   int hypre__i4start = hypre_BoxIndexRank(data_box4, start4);\
   int hypre__nx = hypre_IndexX(loop_size);\
   int hypre__ny = hypre_IndexY(loop_size);\
   int hypre__nz = hypre_IndexZ(loop_size)

#define hypre_BoxLoop4For(i, j, k, i1, i2, i3, i4)\
   for (k = 0; k < hypre__nz; k++ )\
   {\
      for (j = 0; j < hypre__ny; j++ )\
      {\
         for (i = 0; i < hypre__nx; i++ )\
         {\
            i1 = hypre__i1start + i*hypre__is1 + j*hypre__js1 + k*hypre__ks1;\
            i2 = hypre__i2start + i*hypre__is2 + j*hypre__js2 + k*hypre__ks2;\
            i3 = hypre__i3start + i*hypre__is3 + j*hypre__js3 + k*hypre__ks3;\
            i4 = hypre__i4start + i*hypre__is4 + j*hypre__js4 + k*hypre__ks4;

#define hypre_BoxLoop4End }}}}

#endif  /* ifndef HYPRE_USE_PTHREADS */


#endif
