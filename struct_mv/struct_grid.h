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
