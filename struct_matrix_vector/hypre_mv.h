
#include "HYPRE_mv.h"

#ifndef hypre_MV_HEADER
#define hypre_MV_HEADER

#include "hypre_utilities.h"

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
   hypre_Box  **boxes;         /* Array of pointers to boxes */
   int          size;          /* Size of box array */
   int          alloc_size;

} hypre_BoxArray;

#define hypre_BoxArrayBlocksize 5

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
(hypre_BoxIMaxD(box, d) - hypre_BoxIMinD(box, d) + 1)

#define hypre_BoxIMinX(box)    hypre_BoxIMinD(box, 0)
#define hypre_BoxIMinY(box)    hypre_BoxIMinD(box, 1)
#define hypre_BoxIMinZ(box)    hypre_BoxIMinD(box, 2)

#define hypre_BoxIMaxX(box)    hypre_BoxIMaxD(box, 0)
#define hypre_BoxIMaxY(box)    hypre_BoxIMaxD(box, 1)
#define hypre_BoxIMaxZ(box)    hypre_BoxIMaxD(box, 2)

#define hypre_BoxSizeX(box)    hypre_BoxSizeD(box, 0)
#define hypre_BoxSizeY(box)    hypre_BoxSizeD(box, 1)
#define hypre_BoxSizeZ(box)    hypre_BoxSizeD(box, 2)

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
#define hypre_BoxArrayBox(box_array, i)    ((box_array) -> boxes[(i)])
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
 
#define hypre_BeginBoxNeighborsLoop(n, b, neighbors, distance_index)\
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
 * Header info for the hypre_SBox ("Stride Box") structures
 *
 *****************************************************************************/

#ifndef hypre_SBOX_HEADER
#define hypre_SBOX_HEADER

/*--------------------------------------------------------------------------
 * hypre_SBox:
 *   Structure describing a strided cartesian region of some index space.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Box    *box;
   hypre_Index   stride;       /* Striding factors */

} hypre_SBox;

/*--------------------------------------------------------------------------
 * hypre_SBoxArray:
 *   An array of sboxes.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_SBox  **sboxes;       /* Array of pointers to sboxes */
   int           size;         /* Size of sbox array */
   int           alloc_size;

} hypre_SBoxArray;

#define hypre_SBoxArrayBlocksize hypre_BoxArrayBlocksize

/*--------------------------------------------------------------------------
 * hypre_SBoxArrayArray:
 *   An array of sbox arrays.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_SBoxArray  **sbox_arrays;   /* Array of pointers to sbox arrays */
   int                size;          /* Size of sbox array array */

} hypre_SBoxArrayArray;


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SBox
 *--------------------------------------------------------------------------*/

#define hypre_SBoxBox(sbox)         ((sbox) -> box)
#define hypre_SBoxStride(sbox)      ((sbox) -> stride)
				        
#define hypre_SBoxIMin(sbox)        hypre_BoxIMin(hypre_SBoxBox(sbox))
#define hypre_SBoxIMax(sbox)        hypre_BoxIMax(hypre_SBoxBox(sbox))
				        
#define hypre_SBoxIMinD(sbox, d)    hypre_BoxIMinD(hypre_SBoxBox(sbox), d)
#define hypre_SBoxIMaxD(sbox, d)    hypre_BoxIMaxD(hypre_SBoxBox(sbox), d)
#define hypre_SBoxStrideD(sbox, d)  hypre_IndexD(hypre_SBoxStride(sbox), d)
#define hypre_SBoxSizeD(sbox, d) \
((hypre_BoxSizeD(hypre_SBoxBox(sbox), d) - 1) / hypre_SBoxStrideD(sbox, d) + 1)
				        
#define hypre_SBoxIMinX(sbox)       hypre_SBoxIMinD(sbox, 0)
#define hypre_SBoxIMinY(sbox)       hypre_SBoxIMinD(sbox, 1)
#define hypre_SBoxIMinZ(sbox)       hypre_SBoxIMinD(sbox, 2)

#define hypre_SBoxIMaxX(sbox)       hypre_SBoxIMaxD(sbox, 0)
#define hypre_SBoxIMaxY(sbox)       hypre_SBoxIMaxD(sbox, 1)
#define hypre_SBoxIMaxZ(sbox)       hypre_SBoxIMaxD(sbox, 2)

#define hypre_SBoxStrideX(sbox)     hypre_SBoxStrideD(sbox, 0)
#define hypre_SBoxStrideY(sbox)     hypre_SBoxStrideD(sbox, 1)
#define hypre_SBoxStrideZ(sbox)     hypre_SBoxStrideD(sbox, 2)

#define hypre_SBoxSizeX(sbox)       hypre_SBoxSizeD(sbox, 0)
#define hypre_SBoxSizeY(sbox)       hypre_SBoxSizeD(sbox, 1)
#define hypre_SBoxSizeZ(sbox)       hypre_SBoxSizeD(sbox, 2)

#define hypre_SBoxVolume(sbox) \
(hypre_SBoxSizeX(sbox) * hypre_SBoxSizeY(sbox) * hypre_SBoxSizeZ(sbox))

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SBoxArray
 *--------------------------------------------------------------------------*/

#define hypre_SBoxArraySBoxes(sbox_array)    ((sbox_array) -> sboxes)
#define hypre_SBoxArraySBox(sbox_array, i)   ((sbox_array) -> sboxes[(i)])
#define hypre_SBoxArraySize(sbox_array)      ((sbox_array) -> size)
#define hypre_SBoxArrayAllocSize(sbox_array) ((sbox_array) -> alloc_size)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SBoxArrayArray
 *--------------------------------------------------------------------------*/

#define hypre_SBoxArrayArraySBoxArrays(sbox_array_array) \
((sbox_array_array) -> sbox_arrays)
#define hypre_SBoxArrayArraySBoxArray(sbox_array_array, i) \
((sbox_array_array) -> sbox_arrays[(i)])
#define hypre_SBoxArrayArraySize(sbox_array_array) \
((sbox_array_array) -> size)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define hypre_ForSBoxI(i, sbox_array) \
for (i = 0; i < hypre_SBoxArraySize(sbox_array); i++)

#define hypre_ForSBoxArrayI(i, sbox_array_array) \
for (i = 0; i < hypre_SBoxArrayArraySize(sbox_array_array); i++)


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

} hypre_StructStencil;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_StructStencil structure
 *--------------------------------------------------------------------------*/

#define hypre_StructStencilShape(stencil)      ((stencil) -> shape)
#define hypre_StructStencilSize(stencil)       ((stencil) -> size)
#define hypre_StructStencilMaxOffset(stencil)  ((stencil) -> max_offset)

#define hypre_StructStencilDim(stencil)   ((stencil) -> dim)

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

#if defined(HYPRE_COMM_SIMPLE)
   double        **send_buffers;
   double        **recv_buffers;
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
#define hypre_CommHandleRequest(comm_handle, i)  (comm_handle -> requests[(i)])
#if defined(HYPRE_COMM_SIMPLE)
#define hypre_CommHandleSendBuffers(comm_handle) (comm_handle -> send_buffers)
#define hypre_CommHandleRecvBuffers(comm_handle) (comm_handle -> recv_buffers)
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

   hypre_SBoxArrayArray  *indt_sboxes;
   hypre_SBoxArrayArray  *dept_sboxes;

   hypre_StructGrid      *grid;
   hypre_BoxArray        *data_space;
   int                    num_values;

} hypre_ComputePkg;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ComputePkg
 *--------------------------------------------------------------------------*/
 
#define hypre_ComputePkgCommPkg(compute_pkg)      (compute_pkg -> comm_pkg)

#define hypre_ComputePkgIndtSBoxes(compute_pkg)   (compute_pkg -> indt_sboxes)
#define hypre_ComputePkgDeptSBoxes(compute_pkg)   (compute_pkg -> dept_sboxes)

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
#define hypre_StructMatrixDataSize(matrix)      ((matrix) -> data_size)
#define hypre_StructMatrixDataIndices(matrix)   ((matrix) -> data_indices)
#define hypre_StructMatrixSymmetric(matrix)     ((matrix) -> symmetric)
#define hypre_StructMatrixSymmElements(matrix)  ((matrix) -> symm_elements)
#define hypre_StructMatrixNumGhost(matrix)      ((matrix) -> num_ghost)
#define hypre_StructMatrixGlobalSize(matrix)    ((matrix) -> global_size)
#define hypre_StructMatrixCommPkg(matrix)       ((matrix) -> comm_pkg)

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
   int                   data_size;    /* Size of vector data */
   int                  *data_indices; /* num-boxes array of indices into
                                          the data array.  data_indices[b]
                                          is the starting index of vector
                                          data corresponding to box b. */
                      
   int                   num_ghost[6]; /* Num ghost layers in each direction */
                      
   int                   global_size;  /* Total number coefficients */

} hypre_StructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructVector
 *--------------------------------------------------------------------------*/

#define hypre_StructVectorComm(vector)          ((vector) -> comm)
#define hypre_StructVectorGrid(vector)          ((vector) -> grid)
#define hypre_StructVectorDataSpace(vector)     ((vector) -> data_space)
#define hypre_StructVectorData(vector)          ((vector) -> data)
#define hypre_StructVectorDataSize(vector)      ((vector) -> data_size)
#define hypre_StructVectorDataIndices(vector)   ((vector) -> data_indices)
#define hypre_StructVectorNumGhost(vector)      ((vector) -> num_ghost)
#define hypre_StructVectorGlobalSize(vector)    ((vector) -> global_size)
 
#define hypre_StructVectorBox(vector, b) \
hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), b)
 
#define hypre_StructVectorBoxData(vector, b) \
(hypre_StructVectorData(vector) + hypre_StructVectorDataIndices(vector)[b])
 
#define hypre_StructVectorBoxDataValue(vector, b, index) \
(hypre_StructVectorBoxData(vector, b) + \
 hypre_BoxIndexRank(hypre_StructVectorBox(vector, b), index))

#endif
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_struct_grid.c */
HYPRE_StructGrid HYPRE_NewStructGrid P((MPI_Comm comm , int dim ));
void HYPRE_FreeStructGrid P((HYPRE_StructGrid grid ));
void HYPRE_SetStructGridExtents P((HYPRE_StructGrid grid , int *ilower , int *iupper ));
void HYPRE_AssembleStructGrid P((HYPRE_StructGrid grid ));

/* HYPRE_struct_matrix.c */
HYPRE_StructMatrix HYPRE_NewStructMatrix P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_InitializeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixValues P((HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_SetStructMatrixBoxValues P((HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_AssembleStructMatrix P((HYPRE_StructMatrix matrix ));
void HYPRE_SetStructMatrixNumGhost P((HYPRE_StructMatrix matrix , int *num_ghost ));
HYPRE_StructGrid HYPRE_StructMatrixGrid P((HYPRE_StructMatrix matrix ));
void HYPRE_SetStructMatrixSymmetric P((HYPRE_StructMatrix matrix , int symmetric ));
void HYPRE_PrintStructMatrix P((char *filename , HYPRE_StructMatrix matrix , int all ));

/* HYPRE_struct_stencil.c */
HYPRE_StructStencil HYPRE_NewStructStencil P((int dim , int size ));
void HYPRE_SetStructStencilElement P((HYPRE_StructStencil stencil , int element_index , int *offset ));
void HYPRE_FreeStructStencil P((HYPRE_StructStencil stencil ));

/* HYPRE_struct_vector.c */
HYPRE_StructVector HYPRE_NewStructVector P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructVector P((HYPRE_StructVector struct_vector ));
int HYPRE_InitializeStructVector P((HYPRE_StructVector vector ));
int HYPRE_SetStructVectorValues P((HYPRE_StructVector vector , int *grid_index , double values ));
int HYPRE_GetStructVectorValues P((HYPRE_StructVector vector , int *grid_index , double *values_ptr ));
int HYPRE_SetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , double *values ));
int HYPRE_GetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , double **values_ptr ));
int HYPRE_AssembleStructVector P((HYPRE_StructVector vector ));
void HYPRE_PrintStructVector P((char *filename , HYPRE_StructVector vector , int all ));
void HYPRE_SetStructVectorNumGhost P((HYPRE_StructVector vector , int *num_ghost ));
int HYPRE_SetStructVectorConstantValues P((HYPRE_StructVector vector , double values ));
HYPRE_CommPkg HYPRE_GetMigrateStructVectorCommPkg P((HYPRE_StructVector from_vector , HYPRE_StructVector to_vector ));
int HYPRE_MigrateStructVector P((HYPRE_CommPkg comm_pkg , HYPRE_StructVector from_vector , HYPRE_StructVector to_vector ));
void HYPRE_FreeCommPkg P((HYPRE_CommPkg comm_pkg ));

/* box.c */
hypre_Box *hypre_NewBox P((hypre_Index imin , hypre_Index imax ));
hypre_BoxArray *hypre_NewBoxArray P((int alloc_size ));
hypre_BoxArrayArray *hypre_NewBoxArrayArray P((int size ));
void hypre_FreeBox P((hypre_Box *box ));
void hypre_FreeBoxArrayShell P((hypre_BoxArray *box_array ));
void hypre_FreeBoxArray P((hypre_BoxArray *box_array ));
void hypre_FreeBoxArrayArrayShell P((hypre_BoxArrayArray *box_array_array ));
void hypre_FreeBoxArrayArray P((hypre_BoxArrayArray *box_array_array ));
hypre_Box *hypre_DuplicateBox P((hypre_Box *box ));
hypre_BoxArray *hypre_DuplicateBoxArray P((hypre_BoxArray *box_array ));
hypre_BoxArrayArray *hypre_DuplicateBoxArrayArray P((hypre_BoxArrayArray *box_array_array ));
void hypre_AppendBox P((hypre_Box *box , hypre_BoxArray *box_array ));
void hypre_DeleteBox P((hypre_BoxArray *box_array , int index ));
void hypre_AppendBoxArray P((hypre_BoxArray *box_array_0 , hypre_BoxArray *box_array_1 ));
void hypre_AppendBoxArrayArray P((hypre_BoxArrayArray *box_array_array_0 , hypre_BoxArrayArray *box_array_array_1 ));
int hypre_GetBoxSize P((hypre_Box *box , hypre_Index size ));
void hypre_CopyBoxArrayData P((hypre_BoxArray *box_array_in , hypre_BoxArray *data_space_in , int num_values_in , double *data_in , hypre_BoxArray *box_array_out , hypre_BoxArray *data_space_out , int num_values_out , double *data_out ));

/* box_algebra.c */
hypre_Box *hypre_IntersectBoxes P((hypre_Box *box1 , hypre_Box *box2 ));
hypre_BoxArray *hypre_IntersectBoxArrays P((hypre_BoxArray *box_array1 , hypre_BoxArray *box_array2 ));
hypre_BoxArray *hypre_SubtractBoxes P((hypre_Box *box1 , hypre_Box *box2 ));
hypre_BoxArray *hypre_UnionBoxArray P((hypre_BoxArray *boxes ));

/* box_alloc.c */
void hypre_InitializeBoxMemory P((const int at_a_time ));
void hypre_FinalizeBoxMemory P((void ));
hypre_Box *hypre_BoxAlloc P((void ));
void hypre_BoxFree P((hypre_Box *box ));

/* box_neighbors.c */
hypre_RankLink *hypre_NewRankLink P((int rank ));
int hypre_FreeRankLink P((hypre_RankLink *rank_link ));
hypre_BoxNeighbors *hypre_NewBoxNeighbors P((int *local_ranks , int num_local , hypre_BoxArray *boxes , int *processes , int max_distance ));
int hypre_FreeBoxNeighbors P((hypre_BoxNeighbors *neighbors ));

/* communication.c */
hypre_CommPkg *hypre_NewCommPkg P((hypre_SBoxArrayArray *send_sboxes , hypre_SBoxArrayArray *recv_sboxes , hypre_BoxArray *send_data_space , hypre_BoxArray *recv_data_space , int **send_processes , int **recv_processes , int num_values , MPI_Comm comm ));
void hypre_FreeCommPkg P((hypre_CommPkg *comm_pkg ));
hypre_CommHandle *hypre_InitializeCommunication P((hypre_CommPkg *comm_pkg , double *send_data , double *recv_data ));
hypre_CommHandle *hypre_InitializeCommunication P((hypre_CommPkg *comm_pkg , double *send_data , double *recv_data ));
int hypre_FinalizeCommunication P((hypre_CommHandle *comm_handle ));
int hypre_FinalizeCommunication P((hypre_CommHandle *comm_handle ));
int hypre_ExchangeLocalData P((hypre_CommPkg *comm_pkg , double *send_data , double *recv_data ));
hypre_CommType *hypre_NewCommType P((hypre_CommTypeEntry **comm_entries , int num_entries ));
void hypre_FreeCommType P((hypre_CommType *comm_type ));
hypre_CommTypeEntry *hypre_NewCommTypeEntry P((hypre_SBox *sbox , hypre_Box *data_box , int num_values , int data_box_offset ));
void hypre_FreeCommTypeEntry P((hypre_CommTypeEntry *comm_entry ));
int hypre_NewCommPkgInfo P((hypre_SBoxArrayArray *sboxes , hypre_BoxArray *data_space , int **processes , int num_values , MPI_Comm comm , int *num_comms_ptr , int **comm_processes_ptr , hypre_CommType ***comm_types_ptr , hypre_CommType **copy_type_ptr ));
int hypre_SortCommType P((hypre_CommType *comm_type ));
int hypre_CommitCommPkg P((hypre_CommPkg *comm_pkg ));
int hypre_UnCommitCommPkg P((hypre_CommPkg *comm_pkg ));
int hypre_BuildCommMPITypes P((int num_comms , int *comm_procs , hypre_CommType **comm_types , MPI_Datatype *comm_mpi_types ));
int hypre_BuildCommEntryMPIType P((hypre_CommTypeEntry *comm_entry , MPI_Datatype *comm_entry_mpi_type ));

/* communication_info.c */
void hypre_NewCommInfoFromStencil P((hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
void hypre_NewCommInfoFromGrids P((hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr , hypre_StructGrid *from_grid , hypre_StructGrid *to_grid ));

/* computation.c */
void hypre_GetComputeInfo P((hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr , hypre_BoxArrayArray **indt_boxes_ptr , hypre_BoxArrayArray **dept_boxes_ptr , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
hypre_ComputePkg *hypre_NewComputePkg P((hypre_SBoxArrayArray *send_sboxes , hypre_SBoxArrayArray *recv_sboxes , int **send_processes , int **recv_processes , hypre_SBoxArrayArray *indt_sboxes , hypre_SBoxArrayArray *dept_sboxes , hypre_StructGrid *grid , hypre_BoxArray *data_space , int num_values ));
void hypre_FreeComputePkg P((hypre_ComputePkg *compute_pkg ));
hypre_CommHandle *hypre_InitializeIndtComputations P((hypre_ComputePkg *compute_pkg , double *data ));
void hypre_FinalizeIndtComputations P((hypre_CommHandle *comm_handle ));

/* create_2d_laplacian.c */
int main P((int argc , char *argv []));

/* create_3d_laplacian.c */
int main P((int argc , char *argv []));

/* driver_internal.c */
int main P((int argc , char *argv []));

/* grow.c */
hypre_BoxArray *hypre_GrowBoxByStencil P((hypre_Box *box , hypre_StructStencil *stencil , int transpose ));
hypre_BoxArrayArray *hypre_GrowBoxArrayByStencil P((hypre_BoxArray *box_array , hypre_StructStencil *stencil , int transpose ));

/* one_to_many.c */
int main P((int argc , char *argv []));

/* one_to_many_vector.c */
int main P((int argc , char *argv []));

/* project.c */
hypre_SBox *hypre_ProjectBox P((hypre_Box *box , hypre_Index index , hypre_Index stride ));
hypre_SBoxArray *hypre_ProjectBoxArray P((hypre_BoxArray *box_array , hypre_Index index , hypre_Index stride ));
hypre_SBoxArrayArray *hypre_ProjectBoxArrayArray P((hypre_BoxArrayArray *box_array_array , hypre_Index index , hypre_Index stride ));
hypre_SBoxArrayArray *hypre_ProjectRBPoint P((hypre_BoxArrayArray *box_array_array , hypre_Index rb [4 ]));

/* sbox.c */
hypre_SBox *hypre_NewSBox P((hypre_Box *box , hypre_Index stride ));
hypre_SBoxArray *hypre_NewSBoxArray P((int alloc_size ));
hypre_SBoxArrayArray *hypre_NewSBoxArrayArray P((int size ));
void hypre_FreeSBox P((hypre_SBox *sbox ));
void hypre_FreeSBoxArrayShell P((hypre_SBoxArray *sbox_array ));
void hypre_FreeSBoxArray P((hypre_SBoxArray *sbox_array ));
void hypre_FreeSBoxArrayArrayShell P((hypre_SBoxArrayArray *sbox_array_array ));
void hypre_FreeSBoxArrayArray P((hypre_SBoxArrayArray *sbox_array_array ));
hypre_SBox *hypre_DuplicateSBox P((hypre_SBox *sbox ));
hypre_SBoxArray *hypre_DuplicateSBoxArray P((hypre_SBoxArray *sbox_array ));
hypre_SBoxArrayArray *hypre_DuplicateSBoxArrayArray P((hypre_SBoxArrayArray *sbox_array_array ));
hypre_SBox *hypre_ConvertToSBox P((hypre_Box *box ));
hypre_SBoxArray *hypre_ConvertToSBoxArray P((hypre_BoxArray *box_array ));
hypre_SBoxArrayArray *hypre_ConvertToSBoxArrayArray P((hypre_BoxArrayArray *box_array_array ));
void hypre_AppendSBox P((hypre_SBox *sbox , hypre_SBoxArray *sbox_array ));
void hypre_DeleteSBox P((hypre_SBoxArray *sbox_array , int index ));
void hypre_AppendSBoxArray P((hypre_SBoxArray *sbox_array_0 , hypre_SBoxArray *sbox_array_1 ));
void hypre_AppendSBoxArrayArray P((hypre_SBoxArrayArray *sbox_array_array_0 , hypre_SBoxArrayArray *sbox_array_array_1 ));
int hypre_GetSBoxSize P((hypre_SBox *sbox , hypre_Index size ));

/* struct_axpy.c */
int hypre_StructAxpy P((double alpha , hypre_StructVector *x , hypre_StructVector *y ));

/* struct_copy.c */
int hypre_StructCopy P((hypre_StructVector *x , hypre_StructVector *y ));

/* struct_grid.c */
hypre_StructGrid *hypre_NewStructGrid P((MPI_Comm comm , int dim ));
void hypre_FreeStructGrid P((hypre_StructGrid *grid ));
void hypre_SetStructGridExtents P((hypre_StructGrid *grid , hypre_Index ilower , hypre_Index iupper ));
void hypre_SetStructGridBoxes P((hypre_StructGrid *grid , hypre_BoxArray *boxes ));
void hypre_AssembleStructGrid P((hypre_StructGrid *grid , hypre_BoxArray *all_boxes , int *processes , int *box_ranks ));
int hypre_GatherAllBoxes P((MPI_Comm comm , hypre_BoxArray *boxes , hypre_BoxArray **all_boxes_ptr , int **processes_ptr , int **box_ranks_ptr ));
void hypre_PrintStructGrid P((FILE *file , hypre_StructGrid *grid ));
hypre_StructGrid *hypre_ReadStructGrid P((MPI_Comm comm , FILE *file ));

/* struct_innerprod.c */
double hypre_StructInnerProd P((hypre_StructVector *x , hypre_StructVector *y ));

/* struct_io.c */
void hypre_PrintBoxArrayData P((FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , int num_values , double *data ));
void hypre_ReadBoxArrayData P((FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , int num_values , double *data ));

/* struct_matrix.c */
double *hypre_StructMatrixExtractPointerByIndex P((hypre_StructMatrix *matrix , int b , hypre_Index index ));
hypre_StructMatrix *hypre_NewStructMatrix P((MPI_Comm comm , hypre_StructGrid *grid , hypre_StructStencil *user_stencil ));
int hypre_FreeStructMatrixShell P((hypre_StructMatrix *matrix ));
int hypre_FreeStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_InitializeStructMatrixShell P((hypre_StructMatrix *matrix ));
void hypre_InitializeStructMatrixData P((hypre_StructMatrix *matrix , double *data ));
int hypre_InitializeStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_SetStructMatrixValues P((hypre_StructMatrix *matrix , hypre_Index grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int hypre_SetStructMatrixBoxValues P((hypre_StructMatrix *matrix , hypre_Box *value_box , int num_stencil_indices , int *stencil_indices , double *values ));
int hypre_AssembleStructMatrix P((hypre_StructMatrix *matrix ));
void hypre_SetStructMatrixNumGhost P((hypre_StructMatrix *matrix , int *num_ghost ));
void hypre_PrintStructMatrix P((char *filename , hypre_StructMatrix *matrix , int all ));
int hypre_MigrateStructMatrix P((hypre_StructMatrix *from_matrix , hypre_StructMatrix *to_matrix ));
hypre_StructMatrix *hypre_ReadStructMatrix P((MPI_Comm comm , char *filename , int *num_ghost ));

/* struct_matrix_mask.c */
hypre_StructMatrix *hypre_NewStructMatrixMask P((hypre_StructMatrix *matrix , int num_stencil_indices , int *stencil_indices ));
int hypre_FreeStructMatrixMask P((hypre_StructMatrix *mask ));

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
void hypre_FreeStructStencil P((hypre_StructStencil *stencil ));
int hypre_StructStencilElementRank P((hypre_StructStencil *stencil , hypre_Index stencil_element ));
int hypre_SymmetrizeStructStencil P((hypre_StructStencil *stencil , hypre_StructStencil **symm_stencil_ptr , int **symm_elements_ptr ));

/* struct_vector.c */
hypre_StructVector *hypre_NewStructVector P((MPI_Comm comm , hypre_StructGrid *grid ));
int hypre_FreeStructVectorShell P((hypre_StructVector *vector ));
int hypre_FreeStructVector P((hypre_StructVector *vector ));
int hypre_InitializeStructVectorShell P((hypre_StructVector *vector ));
void hypre_InitializeStructVectorData P((hypre_StructVector *vector , double *data ));
int hypre_InitializeStructVector P((hypre_StructVector *vector ));
int hypre_SetStructVectorValues P((hypre_StructVector *vector , hypre_Index grid_index , double values ));
int hypre_GetStructVectorValues P((hypre_StructVector *vector , hypre_Index grid_index , double *values_ptr ));
int hypre_SetStructVectorBoxValues P((hypre_StructVector *vector , hypre_Box *value_box , double *values ));
int hypre_GetStructVectorBoxValues P((hypre_StructVector *vector , hypre_Box *value_box , double **values_ptr ));
void hypre_SetStructVectorNumGhost P((hypre_StructVector *vector , int *num_ghost ));
int hypre_AssembleStructVector P((hypre_StructVector *vector ));
int hypre_SetStructVectorConstantValues P((hypre_StructVector *vector , double values ));
int hypre_ClearStructVectorGhostValues P((hypre_StructVector *vector ));
int hypre_ClearStructVectorAllValues P((hypre_StructVector *vector ));
hypre_CommPkg *hypre_GetMigrateStructVectorCommPkg P((hypre_StructVector *from_vector , hypre_StructVector *to_vector ));
int hypre_MigrateStructVector P((hypre_CommPkg *comm_pkg , hypre_StructVector *from_vector , hypre_StructVector *to_vector ));
void hypre_PrintStructVector P((char *filename , hypre_StructVector *vector , int all ));
hypre_StructVector *hypre_ReadStructVector P((MPI_Comm comm , char *filename , int *num_ghost ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

