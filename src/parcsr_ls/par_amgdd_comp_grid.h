/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Header info for Parallel AMGDD composite grid structure (on a single level)
 *
 *****************************************************************************/

#ifndef hypre_PAR_AMGDD_COMP_GRID_HEADER
#define hypre_PAR_AMGDD_COMP_GRID_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParCompGridCommPkg
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_PAR_AMGDD_COMP_GRID_COMM_PKG
#define HYPRE_PAR_AMGDD_COMP_GRID_COMM_PKG
#endif

typedef struct
{
   // Info needed for subsequent psi_c residual communication
	HYPRE_Int 			num_levels; // levels in the amg hierarchy
	HYPRE_Int 			*num_send_procs; // number of send procs to communicate with
   HYPRE_Int         *num_recv_procs; // number of recv procs to communicate with

   HYPRE_Int         **send_procs; // list of send procs
   HYPRE_Int         **recv_procs; // list of recv procs
   HYPRE_Int         **send_map_starts; // send map starts from comm pkg of A^eta on each level later used as send map starts for full residual communication
   HYPRE_Int         **send_map_elmts; // send map elmts from comm pkg of A^eta on each level later used as send map elmts for full residual communication
   HYPRE_Int         **recv_map_starts; // recv map starts for full residual communication
   HYPRE_Int         **recv_map_elmts; // recv map elmts for full residual communication
   HYPRE_Int         **ghost_marker; // marks send elmts as ghost or real dofs for the associated processor

	HYPRE_Int 			**send_buffer_size; // size of send buffer on each level for each proc
	HYPRE_Int 			**recv_buffer_size; // size of recv buffer on each level for each proc

	HYPRE_Int 			***num_send_nodes; // number of nodes to send on each composite level
   HYPRE_Int         ***num_recv_nodes; // number of nodes to recv on each composite level
	HYPRE_Int 			****send_flag; // flags which nodes to send after composite grid is built
	HYPRE_Int 			****recv_map; // mapping from recv buffer to appropriate local indices on each comp grid

} hypre_ParCompGridCommPkg;

/*--------------------------------------------------------------------------
 * Accessor functions for the Comp Grid Comm Pkg structure
 *--------------------------------------------------------------------------*/

 #define hypre_ParCompGridCommPkgNumLevels(compGridCommPkg)				((compGridCommPkg) -> num_levels)
 #define hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)				((compGridCommPkg) -> num_send_procs)
 #define hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)           ((compGridCommPkg) -> num_recv_procs)
 #define hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)           ((compGridCommPkg) -> send_procs)
 #define hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)           ((compGridCommPkg) -> recv_procs)
 #define hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)           ((compGridCommPkg) -> send_map_starts)
 #define hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)            ((compGridCommPkg) -> send_map_elmts)
 #define hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)           ((compGridCommPkg) -> recv_map_starts)
 #define hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)           ((compGridCommPkg) -> recv_map_elmts)
 #define hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)           ((compGridCommPkg) -> ghost_marker)
 #define hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg)		((compGridCommPkg) -> send_buffer_size)
 #define hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg)		((compGridCommPkg) -> recv_buffer_size)
 #define hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)			((compGridCommPkg) -> num_send_nodes)
 #define hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)       ((compGridCommPkg) -> num_recv_nodes)
 #define hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)				((compGridCommPkg) -> send_flag)
 #define hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)				((compGridCommPkg) -> recv_map)




/*--------------------------------------------------------------------------
 * hypre_ParCompGrid
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_PAR_AMGDD_COMP_GRID_STRUCT
#define HYPRE_PAR_AMGDD_COMP_GRID_STRUCT
#endif

/*--------------------------------------------------------------------------
 * CompGridMatrix (basically a coupled collection of CSR matrices)
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_PAR_CSR_MATRIX_STRUCT
#define HYPRE_PAR_CSR_MATRIX_STRUCT
#endif

typedef struct
{
   hypre_CSRMatrix      *owned_diag; // Domain: owned domain of mat. Range: owned range of mat.
   hypre_CSRMatrix      *owned_offd; // Domain: nonowned domain of mat. Range: owned range of mat.
   hypre_CSRMatrix      *nonowned_diag; // Domain: nonowned domain of mat. Range: nonowned range of mat.
   hypre_CSRMatrix      *nonowned_offd; // Domain: owned domain of mat. Range: nonowned range of mat.

   HYPRE_Int            owns_owned_matrices;
   HYPRE_Int            owns_offd_col_indices;

} hypre_ParCompGridMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CompGridMatrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCompGridMatrixOwnedDiag(matrix)            ((matrix) -> owned_diag)
#define hypre_ParCompGridMatrixOwnedOffd(matrix)            ((matrix) -> owned_offd)
#define hypre_ParCompGridMatrixNonOwnedDiag(matrix)            ((matrix) -> nonowned_diag)
#define hypre_ParCompGridMatrixNonOwnedOffd(matrix)            ((matrix) -> nonowned_offd)
#define hypre_ParCompGridMatrixOwnsOwnedMatrices(matrix)       ((matrix) -> owns_owned_matrices)
#define hypre_ParCompGridMatrixOwnsOffdColIndices(matrix)         ((matrix) -> owns_offd_col_indices)

/*--------------------------------------------------------------------------
 * CompGridVector
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_PAR_VECTOR_STRUCT
#define HYPRE_PAR_VECTOR_STRUCT
#endif

typedef struct
{
   hypre_Vector         *owned_vector; // Original on-processor points (should be ordered)
   hypre_Vector         *nonowned_vector; // Off-processor points (not ordered)

   HYPRE_Int            owns_owned_vector;

} hypre_ParCompGridVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the CompGridVector structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCompGridVectorOwned(matrix)            ((matrix) -> owned_vector)
#define hypre_ParCompGridVectorNonOwned(matrix)            ((matrix) -> nonowned_vector)
#define hypre_ParCompGridVectorOwnsOwnedVector(matrix)       ((matrix) -> owns_owned_vector)


typedef struct
{
   HYPRE_Int        first_global_index;
   HYPRE_Int        last_global_index;
   HYPRE_Int        num_owned_nodes;
   HYPRE_Int        num_nonowned_nodes;
   HYPRE_Int        num_nonowned_real_nodes;
   HYPRE_Int        num_owned_c_points;
   HYPRE_Int        num_nonowned_c_points;
   HYPRE_Int        num_missing_col_indices;

   HYPRE_Int        *nonowned_global_indices;
   HYPRE_Int        *nonowned_coarse_indices;
   HYPRE_Int        *nonowned_real_marker;
   HYPRE_Int        *nonowned_sort;
   HYPRE_Int        *nonowned_invsort;
   HYPRE_Int        *nonowned_diag_missing_col_indices;

   HYPRE_Int        *owned_coarse_indices;

   hypre_ParCompGridMatrix *A_new;
   hypre_ParCompGridMatrix *P_new;
   hypre_ParCompGridMatrix *R_new;

   hypre_ParCompGridVector     *u_new;
   hypre_ParCompGridVector     *f_new;
   hypre_ParCompGridVector     *t_new;
   hypre_ParCompGridVector     *s_new;
   hypre_ParCompGridVector     *q_new;
   hypre_ParCompGridVector     *temp_new;
   hypre_ParCompGridVector     *temp2_new;
   hypre_ParCompGridVector     *temp3_new;

   // TODO
   HYPRE_Real       *l1_norms;
   // HYPRE_Int        *cf_marker_array;
   // int              *c_mask;
   // int              *f_mask;

   // HYPRE_Real       *cheby_coeffs;

} hypre_ParCompGrid;

/*--------------------------------------------------------------------------
 * Accessor functions for the Comp Grid structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCompGridFirstGlobalIndex(compGrid)               ((compGrid) -> first_global_index)
#define hypre_ParCompGridLastGlobalIndex(compGrid)               ((compGrid) -> last_global_index)
#define hypre_ParCompGridNumOwnedNodes(compGrid)               ((compGrid) -> num_owned_nodes)
#define hypre_ParCompGridNumNonOwnedNodes(compGrid)               ((compGrid) -> num_nonowned_nodes)
#define hypre_ParCompGridNumNonOwnedRealNodes(compGrid)               ((compGrid) -> num_nonowned_real_nodes)
#define hypre_ParCompGridNumOwnedCPoints(compGrid)               ((compGrid) -> num_owned_c_points)
#define hypre_ParCompGridNumNonOwnedCPoints(compGrid)               ((compGrid) -> num_nonowned_c_points)
#define hypre_ParCompGridNumMissingColIndices(compGrid)               ((compGrid) -> num_missing_col_indices)

#define hypre_ParCompGridNonOwnedGlobalIndices(compGrid)               ((compGrid) -> nonowned_global_indices)
#define hypre_ParCompGridNonOwnedCoarseIndices(compGrid)               ((compGrid) -> nonowned_coarse_indices)
#define hypre_ParCompGridNonOwnedRealMarker(compGrid)               ((compGrid) -> nonowned_real_marker)
#define hypre_ParCompGridNonOwnedSort(compGrid)               ((compGrid) -> nonowned_sort)
#define hypre_ParCompGridNonOwnedInvSort(compGrid)               ((compGrid) -> nonowned_invsort)
#define hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid)               ((compGrid) -> nonowned_diag_missing_col_indices)

#define hypre_ParCompGridOwnedCoarseIndices(compGrid)               ((compGrid) -> owned_coarse_indices)

#define hypre_ParCompGridANew(compGrid)               ((compGrid) -> A_new)
#define hypre_ParCompGridPNew(compGrid)               ((compGrid) -> P_new)
#define hypre_ParCompGridRNew(compGrid)               ((compGrid) -> R_new)

#define hypre_ParCompGridUNew(compGrid)           ((compGrid) -> u_new)
#define hypre_ParCompGridFNew(compGrid)           ((compGrid) -> f_new)
#define hypre_ParCompGridTNew(compGrid)           ((compGrid) -> t_new)
#define hypre_ParCompGridSNew(compGrid)           ((compGrid) -> s_new)
#define hypre_ParCompGridQNew(compGrid)           ((compGrid) -> q_new)
#define hypre_ParCompGridTempNew(compGrid)        ((compGrid) -> temp_new)
#define hypre_ParCompGridTemp2New(compGrid)        ((compGrid) -> temp2_new)
#define hypre_ParCompGridTemp3New(compGrid)        ((compGrid) -> temp3_new)

// TODO
#define hypre_ParCompGridL1Norms(compGrid)         ((compGrid) -> l1_norms)
// #define hypre_ParCompGridCFMarkerArray(compGrid)         ((compGrid) -> cf_marker_array)
// #define hypre_ParCompGridCMask(compGrid)         ((compGrid) -> c_mask)
// #define hypre_ParCompGridFMask(compGrid)         ((compGrid) -> f_mask)

// #define hypre_ParCompGridChebyCoeffs(compGrid)         ((compGrid) -> cheby_coeffs)

#endif
