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

   hypre_BoxArrayArray   *indt_boxes;
   hypre_BoxArrayArray   *dept_boxes;
   hypre_Index            stride;

   hypre_StructGrid      *grid;
   hypre_BoxArray        *data_space;
   int                    num_values;

} hypre_ComputePkg;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ComputePkg
 *--------------------------------------------------------------------------*/
 
#define hypre_ComputePkgCommPkg(compute_pkg)      (compute_pkg -> comm_pkg)

#define hypre_ComputePkgIndtBoxes(compute_pkg)    (compute_pkg -> indt_boxes)
#define hypre_ComputePkgDeptBoxes(compute_pkg)    (compute_pkg -> dept_boxes)
#define hypre_ComputePkgStride(compute_pkg)       (compute_pkg -> stride)

#define hypre_ComputePkgGrid(compute_pkg)         (compute_pkg -> grid)
#define hypre_ComputePkgDataSpace(compute_pkg)    (compute_pkg -> data_space)
#define hypre_ComputePkgNumValues(compute_pkg)    (compute_pkg -> num_values)

#endif
