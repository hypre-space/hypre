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

#define zzz_BoxTotalSize(box) \
(zzz_BoxSizeX(box)*zzz_BoxSizeY(box)*zzz_BoxSizeZ(box))
  
/*--------------------------------------------------------------------------
 * Accessor macros: zzz_BoxArray
 *--------------------------------------------------------------------------*/

#define zzz_BoxArrayBoxes(box_array)  ((box_array) -> boxes)
#define zzz_BoxArrayBox(box_array, i) ((box_array) -> boxes[(i)])
#define zzz_BoxArraySize(box_array)   ((box_array) -> size)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define zzz_ForBoxI(i, box_array) \
for (i = 0; i < zzz_BoxArraySize(box_array); i++)

#define zzz_BoxDeclare(jinc, kinc, size, data_size, stride)\
int  iinc = (zzz_IndexX(stride));\
int  jinc = (zzz_IndexY(stride)*zzz_IndexX(data_size) -\
             zzz_IndexX(size)*zzz_IndexX(stride));\
int  kinc = (zzz_IndexZ(stride)*zzz_IndexX(data_size)*zzz_IndexY(data_size) -\
             zzz_IndexY(size)*zzz_IndexY(stride)*zzz_IndexX(data_size))

#define zzz_BoxLoop0(index, box,\
		     body)\
{\
   for (zzz_IndexZ(i) = zzz_BoxIMinZ(box);\
	zzz_IndexZ(i) <= zzz_BoxIMaxZ(box);\
	zzz_IndexZ(i)++)\
   {\
      for (zzz_IndexY(i) = zzz_BoxIMinY(box);\
	   zzz_IndexY(i) <= zzz_BoxIMaxY(box);\
	   zzz_IndexY(i)++)\
      {\
	 for (zzz_IndexX(i) = zzz_BoxIMinX(box);\
	      zzz_IndexX(i) <= zzz_BoxIMaxX(box);\
	      zzz_IndexX(i)++)\
	 {\
	    body;\
	 }\
      }\
   }\
}

#define zzz_BoxLoop1(index, box,\
		     i1, data_size1, stride1,\
		     body)\
{\
   zzz_BoxDeclare(PV_iinc_1, PV_jinc_1, PV_kinc_1, size, data_size1, stride1);\
   for (zzz_IndexZ(i) = zzz_BoxIMinZ(box);\
	zzz_IndexZ(i) <= zzz_BoxIMaxZ(box);\
	zzz_IndexZ(i)++)\
   {\
      for (zzz_IndexY(i) = zzz_BoxIMinY(box);\
	   zzz_IndexY(i) <= zzz_BoxIMaxY(box);\
	   zzz_IndexY(i)++)\
      {\
	 for (zzz_IndexX(i) = zzz_BoxIMinX(box);\
	      zzz_IndexX(i) <= zzz_BoxIMaxX(box);\
	      zzz_IndexX(i)++)\
	 {\
	    body;\
	    i1 += PV_iinc_1;\
	 }\
	 i1 += PV_jinc_1;\
      }\
      i1 += PV_kinc_1;\
   }\
}

#define zzz_BoxLoop2(index, box,\
		     i1, data_size1, stride1,\
		     i2, data_size2, stride2,\
		     body)\
{\
   zzz_BoxDeclare(PV_iinc_1, PV_jinc_1, PV_kinc_1, size, data_size1, stride1);\
   zzz_BoxDeclare(PV_iinc_2, PV_jinc_2, PV_kinc_2, size, data_size2, stride2);\
   for (zzz_IndexZ(i) = zzz_BoxIMinZ(box);\
	zzz_IndexZ(i) <= zzz_BoxIMaxZ(box);\
	zzz_IndexZ(i)++)\
   {\
      for (zzz_IndexY(i) = zzz_BoxIMinY(box);\
	   zzz_IndexY(i) <= zzz_BoxIMaxY(box);\
	   zzz_IndexY(i)++)\
      {\
	 for (zzz_IndexX(i) = zzz_BoxIMinX(box);\
	      zzz_IndexX(i) <= zzz_BoxIMaxX(box);\
	      zzz_IndexX(i)++)\
	 {\
	    body;\
	    i1 += PV_iinc_1;\
	    i2 += PV_iinc_2;\
	 }\
	 i1 += PV_jinc_1;\
	 i2 += PV_jinc_2;\
      }\
      i1 += PV_kinc_1;\
      i2 += PV_kinc_2;\
   }\
}

#define zzz_BoxLoop3(index, box,\
		     i1, data_size1, stride1,\
		     i2, data_size2, stride2,\
		     i3, data_size3, stride3,\
		     body)\
{\
   zzz_BoxDeclare(PV_iinc_1, PV_jinc_1, PV_kinc_1, size, data_size1, stride1);\
   zzz_BoxDeclare(PV_iinc_2, PV_jinc_2, PV_kinc_2, size, data_size2, stride2);\
   zzz_BoxDeclare(PV_iinc_3, PV_jinc_3, PV_kinc_3, size, data_size3, stride3);\
   for (zzz_IndexZ(i) = zzz_BoxIMinZ(box);\
	zzz_IndexZ(i) <= zzz_BoxIMaxZ(box);\
	zzz_IndexZ(i)++)\
   {\
      for (zzz_IndexY(i) = zzz_BoxIMinY(box);\
	   zzz_IndexY(i) <= zzz_BoxIMaxY(box);\
	   zzz_IndexY(i)++)\
      {\
	 for (zzz_IndexX(i) = zzz_BoxIMinX(box);\
	      zzz_IndexX(i) <= zzz_BoxIMaxX(box);\
	      zzz_IndexX(i)++)\
	    body;\
	    i1 += PV_iinc_1;\
	    i2 += PV_iinc_2;\
	    i3 += PV_iinc_3;\
	 }\
	 i1 += PV_jinc_1;\
	 i2 += PV_jinc_2;\
	 i3 += PV_jinc_3;\
      }\
      i1 += PV_kinc_1;\
      i2 += PV_kinc_2;\
      i3 += PV_kinc_3;\
   }\
}

#define zzz_BoxLoop4(index, box,\
		     i1, data_size1, stride1,\
		     i2, data_size2, stride2,\
		     i3, data_size3, stride3,\
		     i4, data_size4, stride4,\
		     body)\
{\
   zzz_BoxDeclare(PV_iinc_1, PV_jinc_1, PV_kinc_1, size, data_size1, stride1);\
   zzz_BoxDeclare(PV_iinc_2, PV_jinc_2, PV_kinc_2, size, data_size2, stride2);\
   zzz_BoxDeclare(PV_iinc_3, PV_jinc_3, PV_kinc_3, size, data_size3, stride3);\
   zzz_BoxDeclare(PV_iinc_4, PV_jinc_4, PV_kinc_4, size, data_size4, stride4);\
   for (zzz_IndexZ(i) = zzz_BoxIMinZ(box);\
	zzz_IndexZ(i) <= zzz_BoxIMaxZ(box);\
	zzz_IndexZ(i)++)\
   {\
      for (zzz_IndexY(i) = zzz_BoxIMinY(box);\
	   zzz_IndexY(i) <= zzz_BoxIMaxY(box);\
	   zzz_IndexY(i)++)\
      {\
	 for (zzz_IndexX(i) = zzz_BoxIMinX(box);\
	      zzz_IndexX(i) <= zzz_BoxIMaxX(box);\
	      zzz_IndexX(i)++)\
	    body;\
	    i1 += PV_iinc_1;\
	    i2 += PV_iinc_2;\
	    i3 += PV_iinc_3;\
	    i4 += PV_iinc_4;\
	 }\
	 i1 += PV_jinc_1;\
	 i2 += PV_jinc_2;\
	 i3 += PV_jinc_3;\
	 i4 += PV_jinc_4;\
      }\
      i1 += PV_kinc_1;\
      i2 += PV_kinc_2;\
      i3 += PV_kinc_3;\
      i4 += PV_kinc_4;\
   }\
}


#endif
