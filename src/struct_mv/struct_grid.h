/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the hypre_StructGrid structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_GRID_HEADER
#define hypre_STRUCT_GRID_HEADER

/*--------------------------------------------------------------------------
 * A grid is defined by an index space I('origin', 'stride') intersected with a
 * set of 'baseboxes'.  The grid maps (coarsens) to form a set of 'boxes' with a
 * uniform index numbering.
 *
 * The following illustrates the relationship between the baseboxes and boxes in
 * the grid.  The index space for each is denoted by I(s), where s is a stride
 * (omitting the origin/anchor point for brevity).  The boxes and box numbering
 * for each are shown (e.g., gb2 is a grid box with box number 2).
 *
 * ---------------------------------------  grid (origin, stride, baseboxes):
 * | bb0 bb1 bb2 bb3 bb4 bb5 bb6 bb7 bb8 |    baseboxes - I(1)
 * |     gb0 gb1         gb2 gb3     gb4 |    boxes     - I(stride)
 * ---------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructGrid_struct
{
   MPI_Comm             comm;

   HYPRE_Int            ndim;         /* Number of grid dimensions */

   hypre_BoxArray      *baseboxes;    /* Array of base boxes in this process */
   hypre_Index          origin;       /* Origin index for coarsening baseboxes */
   hypre_Index          stride;       /* Stride index for coarsening baseboxes */

   hypre_BoxArray      *boxes;        /* Array of nonempty coarsened baseboxes */

   hypre_Index          max_distance; /* Neighborhood size - in each dimension*/

   hypre_Box           *bounding_box; /* Bounding box around grid */

   HYPRE_Int            local_size;   /* Number of grid points locally */
   HYPRE_BigInt         global_size;  /* Total number of grid points */

   hypre_Index          periodic;     /* Indicates if grid is periodic */
   HYPRE_Int            num_periods;  /* number of box set periods */

   hypre_Index         *pshifts;      /* shifts of periodicity */

   HYPRE_Int            ref_count;

   HYPRE_Int            ghlocal_size; /* Number of vars in box including ghosts */
   HYPRE_Int            num_ghost[2 * HYPRE_MAXDIM]; /* ghost layer size */

   hypre_BoxManager    *boxman;

} hypre_StructGrid;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructGrid
 *--------------------------------------------------------------------------*/

#define hypre_StructGridComm(grid)          ((grid) -> comm)
#define hypre_StructGridNDim(grid)          ((grid) -> ndim)
#define hypre_StructGridBaseBoxes(grid)     ((grid) -> baseboxes)
#define hypre_StructGridOrigin(grid)        ((grid) -> origin)
#define hypre_StructGridStride(grid)        ((grid) -> stride)
#define hypre_StructGridBoxes(grid)         ((grid) -> boxes)
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
#define hypre_StructGridBoxMan(grid)        ((grid) -> boxman)

#define hypre_StructGridBox(grid, i)        (hypre_BoxArrayBox(hypre_StructGridBoxes(grid), i))
#define hypre_StructGridNumBaseBoxes(grid)  (hypre_BoxArraySize(hypre_StructGridBaseBoxes(grid)))
#define hypre_StructGridNumBoxes(grid)      (hypre_BoxArraySize(hypre_StructGridBoxes(grid)))
#define hypre_StructGridIDs(grid)           (hypre_BoxArrayIDs(hypre_StructGridBoxes(grid)))
#define hypre_StructGridID(grid, i)         (hypre_BoxArrayID(hypre_StructGridBoxes(grid), i))
#define hypre_StructGridBaseBoxnum(grid, i) (hypre_StructGridID(grid, i))

#define hypre_StructGridIDPeriod(grid)      hypre_BoxNeighborsIDPeriod(hypre_StructGridNeighbors(grid))
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#define hypre_StructGridDataLocation(grid)  ((grid) -> data_location)
#endif

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define hypre_ForStructGridBoxI(i, grid)    hypre_ForBoxI(i, hypre_StructGridBoxes(grid))

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#define HYPRE_MIN_GPU_SIZE                  (131072)
#define hypre_SetDeviceOn()                 hypre_HandleStructExecPolicy(hypre_handle()) = HYPRE_EXEC_DEVICE
#define hypre_SetDeviceOff()                hypre_HandleStructExecPolicy(hypre_handle()) = HYPRE_EXEC_HOST
#endif

#endif

