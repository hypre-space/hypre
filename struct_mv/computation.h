/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 * Header info for computation
 *
 *****************************************************************************/

#ifndef zzz_COMPUTATION_HEADER
#define zzz_COMPUTATION_HEADER

/*--------------------------------------------------------------------------
 * zzz_ComputeInfo:
 *   Structure containing information for doing computations.
 *   This structure depends only on a grid and stencil.
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_SBoxArrayArray  *send_sboxes;
   zzz_SBoxArrayArray  *recv_sboxes;

   int                **send_processes;
   int                **recv_processes;

   zzz_SBoxArrayArray  *indt_sboxes;
   zzz_SBoxArrayArray  *dept_sboxes;

} zzz_ComputeInfo;

/*--------------------------------------------------------------------------
 * zzz_ComputePkg:
 *   Structure containing information for doing computations.
 *   This structure depends on both the grid and stencil (as above),
 *   and the data-space.
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_ComputeInfo *compute_info;
                   
   zzz_BoxArray    *data_space;
   int              num_values;
                   
   zzz_CommPkg     *comm_pkg;

} zzz_ComputePkg;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_ComputeInfo
 *--------------------------------------------------------------------------*/
 
#define zzz_ComputeInfoSendSBoxes(compute_info)  (compute_info -> send_sboxes)
#define zzz_ComputeInfoRecvSBoxes(compute_info)  (compute_info -> recv_sboxes)

#define zzz_ComputeInfoSendProcesses(compute_info) \
(compute_info -> send_processes)
#define zzz_ComputeInfoRecvProcesses(compute_info) \
(compute_info -> recv_processes)

#define zzz_ComputeInfoIndtSBoxes(compute_info)  (compute_info -> indt_sboxes)
#define zzz_ComputeInfoDeptSBoxes(compute_info)  (compute_info -> dept_sboxes)

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_ComputePkg
 *--------------------------------------------------------------------------*/
 
#define zzz_ComputePkgComputeInfo(compute_pkg)  (compute_pkg -> compute_info)
#define zzz_ComputePkgDataSpace(compute_pkg)    (compute_pkg -> data_space)
#define zzz_ComputePkgNumValues(compute_pkg)    (compute_pkg -> num_values)
#define zzz_ComputePkgCommPkg(compute_pkg)      (compute_pkg -> comm_pkg)

#define zzz_ComputePkgSendSBoxes(compute_pkg) \
zzz_ComputeInfoSendSBoxes(zzz_ComputePkgComputeInfo(compute_info))
#define zzz_ComputePkgRecvSBoxes(compute_pkg) \
zzz_ComputeInfoRecvSBoxes(zzz_ComputePkgComputeInfo(compute_info))

#define zzz_ComputePkgSendProcesses(compute_pkg) \
zzz_ComputeInfoSendProcesses(zzz_ComputePkgComputeInfo(compute_info))
#define zzz_ComputePkgRecvProcesses(compute_pkg) \
zzz_ComputeInfoRecvProcesses(zzz_ComputePkgComputeInfo(compute_info))

#define zzz_ComputePkgIndtSBoxes(compute_pkg) \
zzz_ComputeInfoIndtSBoxes(zzz_ComputePkgComputeInfo(compute_info))
#define zzz_ComputePkgDeptSBoxes(compute_pkg) \
zzz_ComputeInfoDeptSBoxes(zzz_ComputePkgComputeInfo(compute_info))

#endif
