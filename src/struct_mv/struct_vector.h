/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the hypre_StructVector structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_VECTOR_HEADER
#define hypre_STRUCT_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructVector: RDF BASE - update this comment
 *
 * Most of the routines currently only work when the base grid and grid are the
 * same (i.e., when nboxes equals the number of boxes in the grid and stride is
 * the unit stride).  The number of boxes in data_space and data_indices is the
 * same as in the base grid, even though nboxes may be smaller.
 *
 * The following illustrates the relationship between the grid, vector grid, and
 * data layout.  The index space for each is denoted by I(s), where s is a
 * stride (anchor/origin points are omitted for brevity).  The boxes and box
 * numbering for each are shown (e.g., gb2 is a grid box with box number 2).
 *
 * ---------------------------------------  grid (origin, stride, baseboxes):
 * | bb0 bb1 bb2 bb3 bb4 bb5 bb6 bb7 bb8 |    baseboxes - I(1)
 * |     gb0 gb1         gb2 gb3     gb4 |    gridboxes - I(stride)
 * ---------------------------------------
 *                       vg0         vg1    vector grid - I(stride*'vec_stride')
 *   dl0 dl1 dl2 dl3 dl4 dl5 dl6 dl7 dl8    data layout - I(stride*'vec_stride')
 *
 * Notes:
 * - The entry 'vec_stride' is a stride over the points in the grid, i.e., over
 *   points in I(stride), not I(1).
 * - The index space for the data layout is the same as the vector grid.
 * - The number of boxes in the data layout is always the same as in baseboxes.
 *
 * RDF BASE TODO: Eliminate the separate vector grid in stages - make it the
 * same as the grid.
 *
 * NOTE/TODO: The 'data_alloced=2' and 'save_data' aspects of the vector are
 * only needed to support InitializeData(). Consider removing this feature,
 * especially since it creates an issue with Forget().
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructVector_struct
{
   MPI_Comm              comm;

   /* Note: nboxes and boxnums are computed from (grid, stride) */
   hypre_StructGrid     *grid;                        /* Base grid */
   HYPRE_Int             vec_nboxes;                  /* Vector grid number of boxes */
   HYPRE_Int            *vec_boxnums;                 /* Vector grid boxnums in grid */
   hypre_Index           vec_stride;                  /* Vector grid stride on grid */

   HYPRE_MemoryLocation  memory_location;             /* memory location of data */
   HYPRE_Int             memory_mode;                 /* memory management mode */
   HYPRE_Complex        *data;                        /* Pointer to vector data on device*/
   hypre_BoxArray       *data_space;                  /* Boxes describing the data layout */
   HYPRE_Int            *data_indices;                /* Array of indices into the data array -
                                                         data_indices[b] is the starting index of
                                                         data for base boxnum b. */
   HYPRE_Int             data_alloced;                /* TODO (VPM): change this to owns_data */
   HYPRE_Int             data_size;                   /* Size of vector data */
   HYPRE_Int             num_ghost[2 * HYPRE_MAXDIM]; /* Num ghost layers in each direction */
   HYPRE_Int             bghost_not_clear;            /* Are boundary ghosts clear? */
   HYPRE_BigInt          global_size;                 /* Total number coefficients */
   HYPRE_Int             ref_count;

   /* Information needed to Restore() after Rebase() and Resize() */
   hypre_StructGrid     *save_grid;
   hypre_Index           save_stride;
   HYPRE_Complex        *save_data;                   /* Only needed to support InitializeData() */
   hypre_BoxArray       *save_data_space;
   HYPRE_Int             save_data_size;

#if defined(HYPRE_MIXED_PRECISION)
   HYPRE_Precision vector_precision;
#endif

} hypre_StructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructVector
 *
 * Notation: 'i' is a grid box index and 'b' is a base-grid box index
 *--------------------------------------------------------------------------*/

#define hypre_StructVectorComm(vector)           ((vector) -> comm)
#define hypre_StructVectorGrid(vector)           ((vector) -> grid)
#define hypre_StructVectorNBoxes(vector)         ((vector) -> vec_nboxes)
#define hypre_StructVectorBoxnums(vector)        ((vector) -> vec_boxnums)
#define hypre_StructVectorBoxnum(vector, i)      ((vector) -> vec_boxnums[i])
#define hypre_StructVectorStride(vector)         ((vector) -> vec_stride)
#define hypre_StructVectorMemoryLocation(vector) ((vector) -> memory_location)
#define hypre_StructVectorMemoryMode(vector)     ((vector) -> memory_mode)
#define hypre_StructVectorData(vector)           ((vector) -> data)
#define hypre_StructVectorDataSpace(vector)      ((vector) -> data_space)
#define hypre_StructVectorDataIndices(vector)    ((vector) -> data_indices)
#define hypre_StructVectorDataAlloced(vector)    ((vector) -> data_alloced)
#define hypre_StructVectorDataSize(vector)       ((vector) -> data_size)
#define hypre_StructVectorNumGhost(vector)       ((vector) -> num_ghost)
#define hypre_StructVectorBGhostNotClear(vector) ((vector) -> bghost_not_clear)
#define hypre_StructVectorGlobalSize(vector)     ((vector) -> global_size)
#define hypre_StructVectorRefCount(vector)       ((vector) -> ref_count)
#define hypre_StructVectorSaveGrid(vector)       ((vector) -> save_grid)
#define hypre_StructVectorSaveStride(vector)     ((vector) -> save_stride)
#define hypre_StructVectorSaveData(vector)       ((vector) -> save_data)
#define hypre_StructVectorSaveDataSpace(vector)  ((vector) -> save_data_space)
#define hypre_StructVectorSaveDataSize(vector)   ((vector) -> save_data_size)

#define hypre_StructVectorBaseBoxIDs(vector) \
hypre_BoxArrayIDs(hypre_StructGridBaseBoxes(hypre_StructVectorGrid(vector)))
#define hypre_StructVectorNDim(vector) \
hypre_StructGridNDim(hypre_StructVectorGrid(vector))
#define hypre_StructVectorTmpStride(vector) \
hypre_StructGridStride(hypre_StructVectorGrid(vector))

/* The following use a base-grid box index */
#define hypre_StructVectorBaseDataBox(vector, b) \
hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), b)
#define hypre_StructVectorBaseData(vector, b) \
(hypre_StructVectorData(vector) + hypre_StructVectorDataIndices(vector)[b])
#define hypre_StructVectorBaseDataValue(vector, b, index) \
(hypre_StructVectorBaseData(vector, b) + \
 hypre_BoxIndexRank(hypre_StructVectorBaseDataBox(vector, b), index))

// RDF BASE TODO: Delete 'Tmp' and '(tmp)' in these macro definitions
/*  The following use a (tmp) grid box index */
#define hypre_StructVectorTmpBaseBoxnum(vector, i) \
hypre_StructGridBaseBoxnum(hypre_StructVectorGrid(vector), i)
#define hypre_StructVectorTmpBox(vector, i) \
hypre_StructGridBox(hypre_StructVectorGrid(vector), i)
#define hypre_StructVectorTmpBoxCopy(vector, i, box) \
hypre_CopyBox(hypre_StructVectorTmpBox(vector, i), box); /* on (tmp) grid index space */ \
hypre_StructVectorMapDataBox(vector, box);               /* maps to data index space */
#define hypre_StructVectorTmpBoxDataBox(vector, i) \
hypre_StructVectorBaseDataBox(vector, hypre_StructVectorTmpBaseBoxnum(vector, i))
#define hypre_StructVectorTmpBoxData(vector, i) \
hypre_StructVectorBaseData(vector, hypre_StructVectorTmpBaseBoxnum(vector, i))
#define hypre_StructVectorTmpBoxDataValue(vector, i, index) \
hypre_StructVectorBaseDataValue(vector, hypre_StructVectorTmpBaseBoxnum(vector, i), index)

// RDF BASE TODO: Delete these macros since they will replicate the renamed 'Tmp' ones above
/* The following use a grid box index */
#define hypre_StructVectorBaseBoxnum(vector, i) \
hypre_StructVectorTmpBaseBoxnum(vector, hypre_StructVectorBoxnum(vector, i))
#define hypre_StructVectorBox(vector, i) \
hypre_StructVectorTmpBox(vector, hypre_StructVectorBoxnum(vector, i))
#define hypre_StructVectorBoxCopy(vector, i, box) \
hypre_StructVectorTmpBoxCopy(vector, hypre_StructVectorBoxnum(vector, i), box)
#define hypre_StructVectorBoxDataBox(vector, i) \
hypre_StructVectorTmpBoxDataBox(vector, hypre_StructVectorBoxnum(vector, i))
#define hypre_StructVectorBoxData(vector, i) \
hypre_StructVectorTmpBoxData(vector, hypre_StructVectorBoxnum(vector, i))
#define hypre_StructVectorBoxDataValue(vector, i, index) \
hypre_StructVectorTmpBoxDataValue(vector, hypre_StructVectorBoxnum(vector, i), index)

#if defined(HYPRE_MIXED_PRECISION)
#define hypre_StructVectorPrecision(vector)       ((vector) -> vector_precision)
#endif

#endif
