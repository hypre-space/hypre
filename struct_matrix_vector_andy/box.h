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

typedef int hypre_Index;

/*--------------------------------------------------------------------------
 * hypre_Box:
 *   Structure describing a cartesian region of some index space.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Index *imin;           /* min bounding indices */
   hypre_Index *imax;           /* max bounding indices */

} hypre_Box;

/*--------------------------------------------------------------------------
 * hypre_BoxArray:
 *   An array of boxes.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Box  **boxes;         /* Array of pointers to boxes */
   int        size;          /* Size of box array */

} hypre_BoxArray;

#define hypre_BoxArrayBlocksize 10


/*--------------------------------------------------------------------------
 * Member macros: hypre_Index
 *--------------------------------------------------------------------------*/

#define hypre_NewIndex()       hypre_CTAlloc(hypre_Index, 3)
#define hypre_FreeIndex(index) hypre_TFree(index)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_Index
 *--------------------------------------------------------------------------*/

#define hypre_IndexD(index, d)  (index[d])

#define hypre_IndexX(index)     hypre_IndexD(index, 0)
#define hypre_IndexY(index)     hypre_IndexD(index, 1)
#define hypre_IndexZ(index)     hypre_IndexD(index, 2)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_Box
 *--------------------------------------------------------------------------*/

#define hypre_BoxIMin(box)     ((box) -> imin)
#define hypre_BoxIMax(box)     ((box) -> imax)

#define hypre_BoxIMinD(box, d) (hypre_IndexD(hypre_BoxIMin(box), d))
#define hypre_BoxIMaxD(box, d) (hypre_IndexD(hypre_BoxIMax(box), d))
#define hypre_BoxSizeD(box, d) (hypre_BoxIMaxD(box, d) - hypre_BoxIMinD(box, d) + 1)

#define hypre_BoxIMinX(box)    hypre_BoxIMinD(box, 0)
#define hypre_BoxIMinY(box)    hypre_BoxIMinD(box, 1)
#define hypre_BoxIMinZ(box)    hypre_BoxIMinD(box, 2)

#define hypre_BoxIMaxX(box)    hypre_BoxIMaxD(box, 0)
#define hypre_BoxIMaxY(box)    hypre_BoxIMaxD(box, 1)
#define hypre_BoxIMaxZ(box)    hypre_BoxIMaxD(box, 2)

#define hypre_BoxSizeX(box)    hypre_BoxSizeD(box, 0)
#define hypre_BoxSizeY(box)    hypre_BoxSizeD(box, 1)
#define hypre_BoxSizeZ(box)    hypre_BoxSizeD(box, 2)

#define hypre_BoxTotalSize(box)\
(hypre_BoxSizeX(box)*hypre_BoxSizeY(box)*hypre_BoxSizeZ(box))
  
/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxArray
 *--------------------------------------------------------------------------*/

#define hypre_BoxArrayBoxes(box_array)  ((box_array) -> boxes)
#define hypre_BoxArrayBox(box_array, i) ((box_array) -> boxes[(i)])
#define hypre_BoxArraySize(box_array)   ((box_array) -> size)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define hypre_ForBoxI(i, box_array) \
for (i = 0; i < hypre_BoxArraySize(box_array); i++)

#define hypre_BoxDeclare(jinc, kinc, size, data_size, stride)\
int  iinc = (hypre_IndexX(stride));\
int  jinc = (hypre_IndexY(stride)*hypre_IndexX(data_size) -\
             hypre_IndexX(size)*hypre_IndexX(stride));\
int  kinc = (hypre_IndexZ(stride)*hypre_IndexX(data_size)*hypre_IndexY(data_size) -\
             hypre_IndexY(size)*hypre_IndexY(stride)*hypre_IndexX(data_size))

#define hypre_BoxLoop0(index, box,\
		     body)\
{\
   for (hypre_IndexZ(i) = hypre_BoxIMinZ(box);\
	hypre_IndexZ(i) <= hypre_BoxIMaxZ(box);\
	hypre_IndexZ(i)++)\
   {\
      for (hypre_IndexY(i) = hypre_BoxIMinY(box);\
	   hypre_IndexY(i) <= hypre_BoxIMaxY(box);\
	   hypre_IndexY(i)++)\
      {\
	 for (hypre_IndexX(i) = hypre_BoxIMinX(box);\
	      hypre_IndexX(i) <= hypre_BoxIMaxX(box);\
	      hypre_IndexX(i)++)\
	 {\
	    body;\
	 }\
      }\
   }\
}

#define hypre_BoxLoop1(index, box,\
		     i1, data_size1, stride1,\
		     body)\
{\
   hypre_BoxDeclare(PV_iinc_1, PV_jinc_1, PV_kinc_1, size, data_size1, stride1);\
   for (hypre_IndexZ(i) = hypre_BoxIMinZ(box);\
	hypre_IndexZ(i) <= hypre_BoxIMaxZ(box);\
	hypre_IndexZ(i)++)\
   {\
      for (hypre_IndexY(i) = hypre_BoxIMinY(box);\
	   hypre_IndexY(i) <= hypre_BoxIMaxY(box);\
	   hypre_IndexY(i)++)\
      {\
	 for (hypre_IndexX(i) = hypre_BoxIMinX(box);\
	      hypre_IndexX(i) <= hypre_BoxIMaxX(box);\
	      hypre_IndexX(i)++)\
	 {\
	    body;\
	    i1 += PV_iinc_1;\
	 }\
	 i1 += PV_jinc_1;\
      }\
      i1 += PV_kinc_1;\
   }\
}

#define hypre_BoxLoop2(index, box,\
		     i1, data_size1, stride1,\
		     i2, data_size2, stride2,\
		     body)\
{\
   hypre_BoxDeclare(PV_iinc_1, PV_jinc_1, PV_kinc_1, size, data_size1, stride1);\
   hypre_BoxDeclare(PV_iinc_2, PV_jinc_2, PV_kinc_2, size, data_size2, stride2);\
   for (hypre_IndexZ(i) = hypre_BoxIMinZ(box);\
	hypre_IndexZ(i) <= hypre_BoxIMaxZ(box);\
	hypre_IndexZ(i)++)\
   {\
      for (hypre_IndexY(i) = hypre_BoxIMinY(box);\
	   hypre_IndexY(i) <= hypre_BoxIMaxY(box);\
	   hypre_IndexY(i)++)\
      {\
	 for (hypre_IndexX(i) = hypre_BoxIMinX(box);\
	      hypre_IndexX(i) <= hypre_BoxIMaxX(box);\
	      hypre_IndexX(i)++)\
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

#define hypre_BoxLoop3(index, box,\
		     i1, data_size1, stride1,\
		     i2, data_size2, stride2,\
		     i3, data_size3, stride3,\
		     body)\
{\
   hypre_BoxDeclare(PV_iinc_1, PV_jinc_1, PV_kinc_1, size, data_size1, stride1);\
   hypre_BoxDeclare(PV_iinc_2, PV_jinc_2, PV_kinc_2, size, data_size2, stride2);\
   hypre_BoxDeclare(PV_iinc_3, PV_jinc_3, PV_kinc_3, size, data_size3, stride3);\
   for (hypre_IndexZ(i) = hypre_BoxIMinZ(box);\
	hypre_IndexZ(i) <= hypre_BoxIMaxZ(box);\
	hypre_IndexZ(i)++)\
   {\
      for (hypre_IndexY(i) = hypre_BoxIMinY(box);\
	   hypre_IndexY(i) <= hypre_BoxIMaxY(box);\
	   hypre_IndexY(i)++)\
      {\
	 for (hypre_IndexX(i) = hypre_BoxIMinX(box);\
	      hypre_IndexX(i) <= hypre_BoxIMaxX(box);\
	      hypre_IndexX(i)++)\
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

#define hypre_BoxLoop4(index, box,\
		     i1, data_size1, stride1,\
		     i2, data_size2, stride2,\
		     i3, data_size3, stride3,\
		     i4, data_size4, stride4,\
		     body)\
{\
   hypre_BoxDeclare(PV_iinc_1, PV_jinc_1, PV_kinc_1, size, data_size1, stride1);\
   hypre_BoxDeclare(PV_iinc_2, PV_jinc_2, PV_kinc_2, size, data_size2, stride2);\
   hypre_BoxDeclare(PV_iinc_3, PV_jinc_3, PV_kinc_3, size, data_size3, stride3);\
   hypre_BoxDeclare(PV_iinc_4, PV_jinc_4, PV_kinc_4, size, data_size4, stride4);\
   for (hypre_IndexZ(i) = hypre_BoxIMinZ(box);\
	hypre_IndexZ(i) <= hypre_BoxIMaxZ(box);\
	hypre_IndexZ(i)++)\
   {\
      for (hypre_IndexY(i) = hypre_BoxIMinY(box);\
	   hypre_IndexY(i) <= hypre_BoxIMaxY(box);\
	   hypre_IndexY(i)++)\
      {\
	 for (hypre_IndexX(i) = hypre_BoxIMinX(box);\
	      hypre_IndexX(i) <= hypre_BoxIMaxX(box);\
	      hypre_IndexX(i)++)\
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
