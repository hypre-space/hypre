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
   int                  num_values;

} hypre_ComputePkg;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ComputePkg
 *--------------------------------------------------------------------------*/
 
#define hypre_ComputePkgCommPkg(compute_pkg)      (compute_pkg -> comm_pkg)

#define hypre_ComputePkgSendSBoxes(compute_pkg) \
hypre_CommPkgSendSBoxes(hypre_ComputePkgCommPkg(compute_pkg))
#define hypre_ComputePkgRecvSBoxes(compute_pkg) \
hypre_CommPkgRecvSBoxes(hypre_ComputePkgCommPkg(compute_pkg))

#define hypre_ComputePkgSendBoxRanks(compute_pkg) \
hypre_CommPkgSendBoxRanks(hypre_ComputePkgCommPkg(compute_pkg))
#define hypre_ComputePkgRecvBoxRanks(compute_pkg) \
hypre_CommPkgRecvBoxRanks(hypre_ComputePkgCommPkg(compute_pkg))

#define hypre_ComputePkgIndtSBoxes(compute_pkg)   (compute_pkg -> indt_sboxes)
#define hypre_ComputePkgDeptSBoxes(compute_pkg)   (compute_pkg -> dept_sboxes)

#define hypre_ComputePkgGrid(compute_pkg)         (compute_pkg -> grid)
#define hypre_ComputePkgDataSpace(compute_pkg)    (compute_pkg -> data_space)
#define hypre_ComputePkgNumValues(compute_pkg)    (compute_pkg -> num_values)

#endif
