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

#ifndef zzz_COMPUTATION_HEADER
#define zzz_COMPUTATION_HEADER

/*--------------------------------------------------------------------------
 * zzz_ComputePkg:
 *   Structure containing information for doing computations.
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_CommPkg         *comm_pkg;

   zzz_SBoxArrayArray  *indt_sboxes;
   zzz_SBoxArrayArray  *dept_sboxes;

   zzz_StructGrid      *grid;
   zzz_BoxArray        *data_space;
   int                  num_values;

} zzz_ComputePkg;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_ComputePkg
 *--------------------------------------------------------------------------*/
 
#define zzz_ComputePkgCommPkg(compute_pkg)      (compute_pkg -> comm_pkg)

#define zzz_ComputePkgSendSBoxes(compute_pkg) \
zzz_CommPkgSendSBoxes(zzz_ComputePkgCommPkg(compute_pkg))
#define zzz_ComputePkgRecvSBoxes(compute_pkg) \
zzz_CommPkgRecvSBoxes(zzz_ComputePkgCommPkg(compute_pkg))

#define zzz_ComputePkgSendBoxRanks(compute_pkg) \
zzz_CommPkgSendBoxRanks(zzz_ComputePkgCommPkg(compute_pkg))
#define zzz_ComputePkgRecvBoxRanks(compute_pkg) \
zzz_CommPkgRecvBoxRanks(zzz_ComputePkgCommPkg(compute_pkg))

#define zzz_ComputePkgIndtSBoxes(compute_pkg)   (compute_pkg -> indt_sboxes)
#define zzz_ComputePkgDeptSBoxes(compute_pkg)   (compute_pkg -> dept_sboxes)

#define zzz_ComputePkgGrid(compute_pkg)         (compute_pkg -> grid)
#define zzz_ComputePkgDataSpace(compute_pkg)    (compute_pkg -> data_space)
#define zzz_ComputePkgNumValues(compute_pkg)    (compute_pkg -> num_values)

#endif
