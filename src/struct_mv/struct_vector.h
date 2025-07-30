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
 * hypre_StructVector:
 *
 * Most of the routines currently only work when the base grid and grid are the
 * same (i.e., when nboxes equals the number of boxes in the grid and stride is
 * the unit stride).  The number of boxes in data_space and data_indices is the
 * same as in the base grid, even though nboxes may be smaller.
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
   hypre_Index           stride;                      /* Grid coarsening stride */
   HYPRE_Int             nboxes;                      /* Grid number of boxes */
   HYPRE_Int            *boxnums;                     /* Grid boxnums in base grid */

   HYPRE_MemoryLocation  memory_location;             /* memory location of data */
   HYPRE_Int             memory_mode;                 /* memory management mode */
   hypre_BoxArray       *data_space;
   HYPRE_Complex        *data;                        /* Pointer to vector data on device*/
   HYPRE_Int             data_alloced;                /* TODO (VPM): change this to owns_data */
   HYPRE_Int             data_size;                   /* Size of vector data */
   HYPRE_Int            *data_indices;                /* Array of indices into the data array.
                                                         data_indices[b] is the starting index of
                                                         data for boxnum b. */
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

typedef struct
{
   MPI_Comm              comm;

   hypre_StructGrid     *grid;

   hypre_BoxArray       *data_space;

   HYPRE_MemoryLocation  memory_location;             /* memory location of data */
   void                 *data;                        /* Pointer to vector data on device*/
   HYPRE_Int             data_alloced;                /* Boolean used for freeing data */
   HYPRE_Int             data_size;                   /* Size of vector data */
   HYPRE_Int            *data_indices;                /* num-boxes array of indices into
                                                         the data array.  data_indices[b]
                                                         is the starting index of vector
                                                         data corresponding to box b. */

   HYPRE_Int             num_ghost[2 * HYPRE_MAXDIM]; /* Num ghost layers in each
                                                       * direction */
   HYPRE_Int             bghost_not_clear;            /* Are boundary ghosts clear? */

   HYPRE_BigInt          global_size;                 /* Total number coefficients */

   HYPRE_Int             ref_count;

#if defined(HYPRE_MIXED_PRECISION)
   HYPRE_Precision vector_precision;
#endif

} hypre_StructVector_mp;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructVector
 *
 * Notation: 'i' is a grid box index and 'b' is a base-grid box index
 *--------------------------------------------------------------------------*/

#define hypre_StructVectorComm(vector)           ((vector) -> comm)
#define hypre_StructVectorGrid(vector)           ((vector) -> grid)
#define hypre_StructVectorStride(vector)         ((vector) -> stride)
#define hypre_StructVectorNBoxes(vector)         ((vector) -> nboxes)
#define hypre_StructVectorBoxnums(vector)        ((vector) -> boxnums)
#define hypre_StructVectorBoxnum(vector, i)      ((vector) -> boxnums[i])
#define hypre_StructVectorMemoryLocation(vector) ((vector) -> memory_location)
#define hypre_StructVectorMemoryMode(vector)     ((vector) -> memory_mode)
#define hypre_StructVectorDataSpace(vector)      ((vector) -> data_space)
#define hypre_StructVectorData(vector)           ((vector) -> data)
#define hypre_StructVectorDataAlloced(vector)    ((vector) -> data_alloced)
#define hypre_StructVectorDataSize(vector)       ((vector) -> data_size)
#define hypre_StructVectorDataIndices(vector)    ((vector) -> data_indices)
#define hypre_StructVectorNumGhost(vector)       ((vector) -> num_ghost)
#define hypre_StructVectorBGhostNotClear(vector) ((vector) -> bghost_not_clear)
#define hypre_StructVectorGlobalSize(vector)     ((vector) -> global_size)
#define hypre_StructVectorRefCount(vector)       ((vector) -> ref_count)
#define hypre_StructVectorSaveGrid(vector)       ((vector) -> save_grid)
#define hypre_StructVectorSaveStride(vector)     ((vector) -> save_stride)
#define hypre_StructVectorSaveData(vector)       ((vector) -> save_data)
#define hypre_StructVectorSaveDataSpace(vector)  ((vector) -> save_data_space)
#define hypre_StructVectorSaveDataSize(vector)   ((vector) -> save_data_size)

#define hypre_StructVectorBoxIDs(vector) \
hypre_BoxArrayIDs(hypre_StructVectorDataSpace(vector))

#define hypre_StructVectorNDim(vector) \
hypre_StructGridNDim(hypre_StructVectorGrid(vector))

/* The following use a base-grid box index */

#define hypre_StructVectorDataSpaceBox(vector, b) \
hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), b)

#define hypre_StructVectorBoxData(vector, b) \
(hypre_StructVectorData(vector) + hypre_StructVectorDataIndices(vector)[b])

#define hypre_StructVectorBoxDataValue(vector, b, index) \
(hypre_StructVectorBoxData(vector, b) + \
 hypre_BoxIndexRank(hypre_StructVectorDataSpaceBox(vector, b), index))

/* The following "Grid" macros use a grid box index */

#define hypre_StructVectorGridBaseBox(vector, i) \
hypre_StructGridBox(hypre_StructVectorGrid(vector), hypre_StructVectorBoxnum(vector, i))

#define hypre_StructVectorGridBoxCopy(vector, i, box) \
hypre_CopyBox(hypre_StructVectorGridBaseBox(vector, i), box); /* on base-grid index space */ \
hypre_StructVectorMapDataBox(vector, box);                    /* maps to data index space */

#define hypre_StructVectorGridDataBox(vector, i) \
hypre_StructVectorDataSpaceBox(vector, hypre_StructVectorBoxnum(vector, i))

#define hypre_StructVectorGridData(vector, i) \
hypre_StructVectorBoxData(vector, hypre_StructVectorBoxnum(vector, i))

#define hypre_StructVectorGridDataValue(vector, i, index) \
hypre_StructVectorBoxDataValue(vector, hypre_StructVectorGridDataBox(vector, i), index)

#if defined(HYPRE_MIXED_PRECISION)
#define hypre_StructVectorPrecision(vector)       ((vector) -> vector_precision)
#endif

#endif
