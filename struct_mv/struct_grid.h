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
 * Header info for the zzz_StructGrid structures
 *
 *****************************************************************************/

#ifndef zzz_STRUCT_GRID_HEADER
#define zzz_STRUCT_GRID_HEADER


/*--------------------------------------------------------------------------
 * zzz_StructGrid:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      *comm;

   zzz_BoxArray  *all_boxes;    /* Array of all grid boxes in the grid */
   int           *processes;    /* Processes corresponding to grid boxes */

   zzz_BoxArray  *boxes;        /* Array of grid boxes in this process */
   int           *box_ranks;    /* Ranks of grid boxes in this process */

   int            dim;          /* Number of grid dimensions */

   int            global_size;  /* Total number of grid points */
   int            local_size;   /* Total number of points locally */

} zzz_StructGrid;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_StructGrid
 *--------------------------------------------------------------------------*/

#define zzz_StructGridComm(grid)          ((grid) -> comm)
#define zzz_StructGridAllBoxes(grid)      ((grid) -> all_boxes)
#define zzz_StructGridProcesses(grid)     ((grid) -> processes)
#define zzz_StructGridBoxes(grid)         ((grid) -> boxes)
#define zzz_StructGridBoxRanks(grid)      ((grid) -> box_ranks)
#define zzz_StructGridDim(grid)           ((grid) -> dim)
#define zzz_StructGridGlobalSize(grid)    ((grid) -> global_size)
#define zzz_StructGridLocalSize(grid)     ((grid) -> local_size)

#define zzz_StructGridProcess(grid, i) \
(zzz_StructGridProcesses(grid)[i])
#define zzz_StructGridBox(grid, i) \
(zzz_BoxArrayBox(zzz_StructGridBoxes(grid), i))
#define zzz_StructGridNumBoxes(grid) \
(zzz_BoxArraySize(zzz_StructGridBoxes(grid)))
#define zzz_StructGridBoxRank(grid, i) \
(zzz_StructGridBoxRanks(grid)[i])

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/
 
#define zzz_ForStructGridBoxI(i, grid) \
zzz_ForBoxI(i, zzz_StructGridBoxes(grid))


#endif
