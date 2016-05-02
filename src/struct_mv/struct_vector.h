/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
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
 *
 * Most of the routines currently only work when the base grid and grid are the
 * same (i.e., when nboxes equals the number of boxes in the grid and stride is
 * the unit stride).  The number of boxes in data_space and data_indices is the
 * same as in the base grid, even though nboxes may be smaller.
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructVector_struct
{
   MPI_Comm              comm;

   hypre_StructGrid     *grid;         /* Base grid */
   HYPRE_Int             nboxes;       /* Grid number of boxes */
   HYPRE_Int            *boxnums;      /* Grid boxnums in base grid */
   hypre_Index           stride;       /* Grid coarsening stride */

   HYPRE_Complex        *data;         /* Pointer to vector data */
   hypre_BoxArray       *data_space;   /* Layout of vector data */
   HYPRE_Int             data_alloced; /* Boolean used for freeing data */
   HYPRE_Int             data_size;    /* Size of vector data */
   HYPRE_Int            *data_indices; /* Array of indices into the data array.
                                          data_indices[b] is the starting index
                                          of data corresponding to boxnum b. */
                      
   HYPRE_Int             num_ghost[2*HYPRE_MAXDIM]; /* Num ghost layers in each
                                                     * direction */
   HYPRE_Int             bghost_not_clear; /* Are boundary ghosts clear? */
                      
   HYPRE_Int             global_size;  /* Total number coefficients */

   HYPRE_Int             ref_count;

   /* Information needed to Restore() after Reindex() and Resize() */
   hypre_StructGrid     *save_grid;
   hypre_Index           save_stride;
   HYPRE_Complex        *save_data;
   hypre_BoxArray       *save_data_space;
   HYPRE_Int             save_data_size;

} hypre_StructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructVector
 *--------------------------------------------------------------------------*/

#define hypre_StructVectorComm(vector)          ((vector) -> comm)
#define hypre_StructVectorGrid(vector)          ((vector) -> grid)
#define hypre_StructVectorNBoxes(vector)        ((vector) -> nboxes)
#define hypre_StructVectorBoxnums(vector)       ((vector) -> boxnums)
#define hypre_StructVectorBoxnum(vector, i)     ((vector) -> boxnums[i])
#define hypre_StructVectorStride(vector)        ((vector) -> stride)
#define hypre_StructVectorData(vector)          ((vector) -> data)
#define hypre_StructVectorDataSpace(vector)     ((vector) -> data_space)
#define hypre_StructVectorDataAlloced(vector)   ((vector) -> data_alloced)
#define hypre_StructVectorDataSize(vector)      ((vector) -> data_size)
#define hypre_StructVectorDataIndices(vector)   ((vector) -> data_indices)
#define hypre_StructVectorNumGhost(vector)      ((vector) -> num_ghost)
#define hypre_StructVectorBGhostNotClear(vector)((vector) -> bghost_not_clear)
#define hypre_StructVectorGlobalSize(vector)    ((vector) -> global_size)
#define hypre_StructVectorRefCount(vector)      ((vector) -> ref_count)
 
#define hypre_StructVectorSaveGrid(vector)      ((vector) -> save_grid)
#define hypre_StructVectorSaveStride(vector)    ((vector) -> save_stride)
#define hypre_StructVectorSaveData(vector)      ((vector) -> save_data)
#define hypre_StructVectorSaveDataSpace(vector) ((vector) -> save_data_space)
#define hypre_StructVectorSaveDataSize(vector)  ((vector) -> save_data_size)

#define hypre_StructVectorNDim(vector) \
hypre_StructGridNDim(hypre_StructVectorGrid(vector))

#define hypre_StructVectorBox(vector, b) \
hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), b)
 
#define hypre_StructVectorBoxData(vector, b) \
(hypre_StructVectorData(vector) + hypre_StructVectorDataIndices(vector)[b])
 
#define hypre_StructVectorBoxDataValue(vector, b, index) \
(hypre_StructVectorBoxData(vector, b) + \
 hypre_BoxIndexRank(hypre_StructVectorBox(vector, b), index))

#endif
