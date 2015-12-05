/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/



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

typedef struct hypre_StructGrid_struct
{
   MPI_Comm             comm;
                      
   HYPRE_Int            dim;          /* Number of grid dimensions */
                      
   hypre_BoxArray      *boxes;        /* Array of boxes in this process */
   HYPRE_Int           *ids;          /* Unique IDs for boxes */
   hypre_Index          max_distance; /* Neighborhood size - in each dimension*/

   hypre_Box           *bounding_box; /* Bounding box around grid */

   HYPRE_Int            local_size;   /* Number of grid points locally */
   HYPRE_Int            global_size;  /* Total number of grid points */

   hypre_Index          periodic;     /* Indicates if grid is periodic */
   HYPRE_Int            num_periods;  /* number of box set periods */
   
   hypre_Index         *pshifts;      /* shifts of periodicity */


   HYPRE_Int            ref_count;


   HYPRE_Int           ghlocal_size;   /* Number of vars in box including ghosts */
   HYPRE_Int           num_ghost[6];   /* ghost layer size for each box  */  

   hypre_BoxManager   *box_man;
   

} hypre_StructGrid;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructGrid
 *--------------------------------------------------------------------------*/

#define hypre_StructGridComm(grid)          ((grid) -> comm)
#define hypre_StructGridDim(grid)           ((grid) -> dim)
#define hypre_StructGridBoxes(grid)         ((grid) -> boxes)
#define hypre_StructGridIDs(grid)           ((grid) -> ids)
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
#define hypre_StructGridBoxMan(grid)        ((grid) -> box_man) 

#define hypre_StructGridBox(grid, i) \
(hypre_BoxArrayBox(hypre_StructGridBoxes(grid), i))
#define hypre_StructGridNumBoxes(grid) \
(hypre_BoxArraySize(hypre_StructGridBoxes(grid)))

#define hypre_StructGridIDPeriod(grid) \
hypre_BoxNeighborsIDPeriod(hypre_StructGridNeighbors(grid))

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/
 
#define hypre_ForStructGridBoxI(i, grid) \
hypre_ForBoxI(i, hypre_StructGridBoxes(grid))

#endif

