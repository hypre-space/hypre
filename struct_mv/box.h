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

#ifndef zzz_BOX_HEADER
#define zzz_BOX_HEADER

/*--------------------------------------------------------------------------
 * zzz_Index:
 *   This is used to define indices in index space, or dimension
 *   sizes of boxes.
 *
 *   The spatial dimensions x, y, and z may be specified by the
 *   integers 0, 1, and 2, respectively (see the zzz_IndexD macro below).
 *   This simplifies the code in the zzz_Box class by reducing code
 *   replication.
 *--------------------------------------------------------------------------*/

typedef int zzz_Index;

/*--------------------------------------------------------------------------
 * zzz_Box:
 *   Structure describing a cartesian region of some index space.
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_Index *imin;           /* min bounding indices */
   zzz_Index *imax;           /* max bounding indices */

} zzz_Box;

/*--------------------------------------------------------------------------
 * zzz_BoxArray:
 *   An array of boxes.
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_Box  **boxes;         /* Array of pointers to boxes */
   int        size;          /* Size of box array */

} zzz_BoxArray;

#define zzz_BoxArrayBlocksize 10

/*--------------------------------------------------------------------------
 * zzz_BoxArrayArray:
 *   An array of box arrays.
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_BoxArray  **box_arrays;    /* Array of pointers to box arrays */
   int             size;          /* Size of box array array */

} zzz_BoxArrayArray;


/*--------------------------------------------------------------------------
 * Member macros: zzz_Index
 *--------------------------------------------------------------------------*/

#define zzz_NewIndex()       ctalloc(zzz_Index, 3)
#define zzz_FreeIndex(index) tfree(index)

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_Index
 *--------------------------------------------------------------------------*/

#define zzz_IndexD(index, d)  (index[d])

#define zzz_IndexX(index)     zzz_IndexD(index, 0)
#define zzz_IndexY(index)     zzz_IndexD(index, 1)
#define zzz_IndexZ(index)     zzz_IndexD(index, 2)

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_Box
 *--------------------------------------------------------------------------*/

#define zzz_BoxIMin(box)     ((box) -> imin)
#define zzz_BoxIMax(box)     ((box) -> imax)

#define zzz_BoxIMinD(box, d) (zzz_IndexD(zzz_BoxIMin(box), d))
#define zzz_BoxIMaxD(box, d) (zzz_IndexD(zzz_BoxIMax(box), d))
#define zzz_BoxSizeD(box, d) (zzz_BoxIMaxD(box, d) - zzz_BoxIMinD(box, d) + 1)

#define zzz_BoxIMinX(box)    zzz_BoxIMinD(box, 0)
#define zzz_BoxIMinY(box)    zzz_BoxIMinD(box, 1)
#define zzz_BoxIMinZ(box)    zzz_BoxIMinD(box, 2)

#define zzz_BoxIMaxX(box)    zzz_BoxIMaxD(box, 0)
#define zzz_BoxIMaxY(box)    zzz_BoxIMaxD(box, 1)
#define zzz_BoxIMaxZ(box)    zzz_BoxIMaxD(box, 2)

#define zzz_BoxSizeX(box)    zzz_BoxSizeD(box, 0)
#define zzz_BoxSizeY(box)    zzz_BoxSizeD(box, 1)
#define zzz_BoxSizeZ(box)    zzz_BoxSizeD(box, 2)

#define zzz_BoxVolume(box) \
(zzz_BoxSizeX(box) * zzz_BoxSizeY(box) * zzz_BoxSizeZ(box))


#define zzz_BoxIndexRank(box, index) \
((zzz_IndexX(index) - zzz_BoxIMinX(box)) + \
 ((zzz_IndexY(index) - zzz_BoxIMinY(box)) + \
  ((zzz_IndexZ(index) - zzz_BoxIMinZ(box)) * \
   zzz_BoxSizeY(box)) * \
  zzz_BoxSizeX(box)))

#define zzz_BoxOffsetDistance(box, index) \
(zzz_IndexX(index) + \
 (zzz_IndexY(index) + \
  (zzz_IndexZ(index) * \
   zzz_BoxSizeY(box)) * \
  zzz_BoxSizeX(box)))
  
/*--------------------------------------------------------------------------
 * Accessor macros: zzz_BoxArray
 *--------------------------------------------------------------------------*/

#define zzz_BoxArrayBoxes(box_array)  ((box_array) -> boxes)
#define zzz_BoxArrayBox(box_array, i) ((box_array) -> boxes[(i)])
#define zzz_BoxArraySize(box_array)   ((box_array) -> size)

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_BoxArrayArray
 *--------------------------------------------------------------------------*/

#define zzz_BoxArrayArrayBoxArrays(box_array_array) \
((box_array_array) -> box_arrays)
#define zzz_BoxArrayArrayBoxArray(box_array_array, i) \
((box_array_array) -> box_arrays[(i)])
#define zzz_BoxArrayArraySize(box_array_array) \
((box_array_array) -> size)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define zzz_ForBoxI(i, box_array) \
for (i = 0; i < zzz_BoxArraySize(box_array); i++)

#define zzz_ForBoxArrayI(i, box_array_array) \
for (i = 0; i < zzz_BoxArrayArraySize(box_array_array); i++)

#define zzz_BoxLoopDeclare(box, data_box, stride, iinc, jinc, kinc) \
int  iinc = (zzz_IndexX(stride));\
int  jinc = (zzz_IndexY(stride)*zzz_BoxSizeX(data_box)\
             - zzz_BoxSizeX(box)*zzz_IndexX(stride));\
int  kinc = (zzz_IndexZ(stride)*zzz_BoxSizeX(data_box)*zzz_BoxSizeY(data_box)\
             - zzz_BoxSizeY(box)*zzz_IndexY(stride)*zzz_BoxSizeX(data_box))

#define zzz_BoxLoop0(box, index,\
		     body)\
{\
   for (zzz_IndexZ(index) = zzz_BoxIMinZ(box);\
	zzz_IndexZ(index) <= zzz_BoxIMaxZ(box);\
	zzz_IndexZ(index)++)\
   {\
      for (zzz_IndexY(index) = zzz_BoxIMinY(box);\
	   zzz_IndexY(index) <= zzz_BoxIMaxY(box);\
	   zzz_IndexY(index)++)\
      {\
	 for (zzz_IndexX(index) = zzz_BoxIMinX(box);\
	      zzz_IndexX(index) <= zzz_BoxIMaxX(box);\
	      zzz_IndexX(index)++)\
	 {\
	    body;\
	 }\
      }\
   }\
}

#define zzz_BoxLoop1(box, index,\
		     data_box1, start1, stride1, i1,\
		     body)\
{\
   zzz_BoxLoopDeclare(box, data_box1, stride1,\
                      zzz__iinc1, zzz__jinc1, zzz__kinc1);\
   i1 = zzz_BoxIndexRank(data_box1, start1);\
   for (zzz_IndexZ(index) = zzz_BoxIMinZ(box);\
	zzz_IndexZ(index) <= zzz_BoxIMaxZ(box);\
	zzz_IndexZ(index)++)\
   {\
      for (zzz_IndexY(index) = zzz_BoxIMinY(box);\
	   zzz_IndexY(index) <= zzz_BoxIMaxY(box);\
	   zzz_IndexY(index)++)\
      {\
	 for (zzz_IndexX(index) = zzz_BoxIMinX(box);\
	      zzz_IndexX(index) <= zzz_BoxIMaxX(box);\
	      zzz_IndexX(index)++)\
	 {\
	    body;\
	    i1 += zzz__iinc1;\
	 }\
	 i1 += zzz__jinc1;\
      }\
      i1 += zzz__kinc1;\
   }\
}

#define zzz_BoxLoop2(box, index,\
		     data_box1, start1, stride1, i1,\
		     data_box2, start2, stride2, i2,\
		     body)\
{\
   zzz_BoxLoopDeclare(box, data_box1, stride1,\
                      zzz__iinc1, zzz__jinc1, zzz__kinc1);\
   zzz_BoxLoopDeclare(box, data_box2, stride2,\
                      zzz__iinc2, zzz__jinc2, zzz__kinc2);\
   i1 = zzz_BoxIndexRank(data_box1, start1);\
   i2 = zzz_BoxIndexRank(data_box2, start2);\
   for (zzz_IndexZ(index) = zzz_BoxIMinZ(box);\
	zzz_IndexZ(index) <= zzz_BoxIMaxZ(box);\
	zzz_IndexZ(index)++)\
   {\
      for (zzz_IndexY(index) = zzz_BoxIMinY(box);\
	   zzz_IndexY(index) <= zzz_BoxIMaxY(box);\
	   zzz_IndexY(index)++)\
      {\
	 for (zzz_IndexX(index) = zzz_BoxIMinX(box);\
	      zzz_IndexX(index) <= zzz_BoxIMaxX(box);\
	      zzz_IndexX(index)++)\
	 {\
	    body;\
	    i1 += zzz__iinc1;\
	    i2 += zzz__iinc2;\
	 }\
	 i1 += zzz__jinc1;\
	 i2 += zzz__jinc2;\
      }\
      i1 += zzz__kinc1;\
      i2 += zzz__kinc2;\
   }\
}

#define zzz_BoxLoop3(box, index,\
		     data_box1, start1, stride1, i1,\
		     data_box2, start2, stride2, i2,\
		     data_box3, start3, stride3, i3,\
		     body)\
{\
   zzz_BoxLoopDeclare(box, data_box1, stride1,\
                      zzz__iinc1, zzz__jinc1, zzz__kinc1);\
   zzz_BoxLoopDeclare(box, data_box2, stride2,\
                      zzz__iinc2, zzz__jinc2, zzz__kinc2);\
   zzz_BoxLoopDeclare(box, data_box3, stride3,\
                      zzz__iinc3, zzz__jinc3, zzz__kinc3);\
   i1 = zzz_BoxIndexRank(data_box1, start1);\
   i2 = zzz_BoxIndexRank(data_box2, start2);\
   i3 = zzz_BoxIndexRank(data_box3, start3);\
   for (zzz_IndexZ(index) = zzz_BoxIMinZ(box);\
	zzz_IndexZ(index) <= zzz_BoxIMaxZ(box);\
	zzz_IndexZ(index)++)\
   {\
      for (zzz_IndexY(index) = zzz_BoxIMinY(box);\
	   zzz_IndexY(index) <= zzz_BoxIMaxY(box);\
	   zzz_IndexY(index)++)\
      {\
	 for (zzz_IndexX(index) = zzz_BoxIMinX(box);\
	      zzz_IndexX(index) <= zzz_BoxIMaxX(box);\
	      zzz_IndexX(index)++)\
	    body;\
	    i1 += zzz__iinc1;\
	    i2 += zzz__iinc2;\
	    i3 += zzz__iinc3;\
	 }\
	 i1 += zzz__jinc1;\
	 i2 += zzz__jinc2;\
	 i3 += zzz__jinc3;\
      }\
      i1 += zzz__kinc1;\
      i2 += zzz__kinc2;\
      i3 += zzz__kinc3;\
   }\
}

#define zzz_BoxLoop4(box, index,\
		     data_box1, start1, stride1, i1,\
		     data_box2, start2, stride2, i2,\
		     data_box3, start3, stride3, i3,\
		     data_box4, start4, stride4, i4,\
		     body)\
{\
   zzz_BoxLoopDeclare(box, data_box1, stride1,\
                      zzz__iinc1, zzz__jinc1, zzz__kinc1);\
   zzz_BoxLoopDeclare(box, data_box2, stride2,\
                      zzz__iinc2, zzz__jinc2, zzz__kinc2);\
   zzz_BoxLoopDeclare(box, data_box3, stride3,\
                      zzz__iinc3, zzz__jinc3, zzz__kinc3);\
   zzz_BoxLoopDeclare(box, data_box4, stride4,\
                      zzz__iinc4, zzz__jinc4, zzz__kinc4);\
   i1 = zzz_BoxIndexRank(data_box1, start1);\
   i2 = zzz_BoxIndexRank(data_box2, start2);\
   i3 = zzz_BoxIndexRank(data_box3, start3);\
   i4 = zzz_BoxIndexRank(data_box4, start4);\
   for (zzz_IndexZ(index) = zzz_BoxIMinZ(box);\
	zzz_IndexZ(index) <= zzz_BoxIMaxZ(box);\
	zzz_IndexZ(index)++)\
   {\
      for (zzz_IndexY(index) = zzz_BoxIMinY(box);\
	   zzz_IndexY(index) <= zzz_BoxIMaxY(box);\
	   zzz_IndexY(index)++)\
      {\
	 for (zzz_IndexX(index) = zzz_BoxIMinX(box);\
	      zzz_IndexX(index) <= zzz_BoxIMaxX(box);\
	      zzz_IndexX(index)++)\
	    body;\
	    i1 += zzz__iinc1;\
	    i2 += zzz__iinc2;\
	    i3 += zzz__iinc3;\
	    i4 += zzz__iinc4;\
	 }\
	 i1 += zzz__jinc1;\
	 i2 += zzz__jinc2;\
	 i3 += zzz__jinc3;\
	 i4 += zzz__jinc4;\
      }\
      i1 += zzz__kinc1;\
      i2 += zzz__kinc2;\
      i3 += zzz__kinc3;\
      i4 += zzz__kinc4;\
   }\
}

#endif
