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
 * hypre_ParCompMatrixRow
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_PAR_AMGDD_COMP_MATRIX_ROW_STRUCT
#define HYPRE_PAR_AMGDD_COMP_MATRIX_ROW_STRUCT
#endif

typedef struct
{
   HYPRE_Int 			size;
   HYPRE_Complex 		*data;
   HYPRE_Int 			*global_indices;
   HYPRE_Int 			*local_indices;



} hypre_ParCompMatrixRow;

/*--------------------------------------------------------------------------
 * Accessor functions for the Comp Matrix Row structure
 *--------------------------------------------------------------------------*/

 #define hypre_ParCompMatrixRowSize(row)				((row) -> size)
 #define hypre_ParCompMatrixRowData(row)				((row) -> data)
 #define hypre_ParCompMatrixRowGlobalIndices(row)		((row) -> global_indices)
 #define hypre_ParCompMatrixRowLocalIndices(row)		((row) -> local_indices)




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
	HYPRE_Int 			*num_sends; // num procs to send to on each level
	HYPRE_Int 			*num_recvs; // num procs to recv from on each level

   HYPRE_Int         **send_procs; // procs to send to on each level
   HYPRE_Int         **recv_procs; // procs to recv from on each level
   HYPRE_Int         **send_map_starts; // send map starts from comm pkg of A^eta on each level
   HYPRE_Int         **send_map_elmts; // send map elmts from comm pkg of A^eta on each level

	HYPRE_Int 			**send_buffer_size; // size of send buffer on each level for each proc
	HYPRE_Int 			**recv_buffer_size; // size of recv buffer on each level for each proc

	HYPRE_Int 			***num_send_nodes; // number of nodes to send on each composite level
	HYPRE_Int 			****send_flag; // flags which nodes to send after composite grid is built
	HYPRE_Int 			****recv_map; // mapping from recv buffer to appropriate local indices on each comp grid

   // CommPkg needed for subsequent ghost node residual communication
   HYPRE_Int         **numGhostFromProc; // numGhostFromProc[proc][level]
   HYPRE_Int         ***ghostGlobalIndex; // ghostGlobalIndex[proc][level][index]
   HYPRE_Int         ***ghostUnpackIndex; // ghostUnpackIndex[proc][level][index]

} hypre_ParCompGridCommPkg;

/*--------------------------------------------------------------------------
 * Accessor functions for the Comp Grid Comm Pkg structure
 *--------------------------------------------------------------------------*/

 #define hypre_ParCompGridCommPkgNumLevels(compGridCommPkg)				((compGridCommPkg) -> num_levels)
 #define hypre_ParCompGridCommPkgNumSends(compGridCommPkg)				((compGridCommPkg) -> num_sends)
 #define hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)				((compGridCommPkg) -> num_recvs)
 #define hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)           ((compGridCommPkg) -> send_procs)
 #define hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)           ((compGridCommPkg) -> recv_procs)
 #define hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)           ((compGridCommPkg) -> send_map_starts)
 #define hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)           ((compGridCommPkg) -> send_map_elmts)
 #define hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg)		((compGridCommPkg) -> send_buffer_size)
 #define hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg)		((compGridCommPkg) -> recv_buffer_size)
 #define hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)			((compGridCommPkg) -> num_send_nodes)
 #define hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)				((compGridCommPkg) -> send_flag)
 #define hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)				((compGridCommPkg) -> recv_map)
 #define hypre_ParCompGridCommPkgNumGhostFromProc(compGridCommPkg)            ((compGridCommPkg) -> numGhostFromProc)
 #define hypre_ParCompGridCommPkgGhostGlobalIndex(compGridCommPkg)            ((compGridCommPkg) -> ghostGlobalIndex)
 #define hypre_ParCompGridCommPkgGhostUnpackIndex(compGridCommPkg)            ((compGridCommPkg) -> ghostUnpackIndex)




/*--------------------------------------------------------------------------
 * hypre_ParCompGrid
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_PAR_AMGDD_COMP_GRID_STRUCT
#define HYPRE_PAR_AMGDD_COMP_GRID_STRUCT
#endif

typedef struct
{
   HYPRE_Int       num_nodes; // total number of nodes including real and ghost nodes
   HYPRE_Int		 num_owned_nodes; // number of nodes owned by this proc in the original partition
   HYPRE_Int       num_real_nodes; // number of real nodes (num_nodes - num_ghost_nodes)
   HYPRE_Int		 mem_size;

   HYPRE_Complex     *u;
   HYPRE_Complex     *f;

   HYPRE_Int        *global_indices;
   HYPRE_Int        *coarse_global_indices; 
   HYPRE_Int        *coarse_local_indices; 

   hypre_ParCompMatrixRow 	**A_rows;
   hypre_ParCompMatrixRow  **P_rows;



} hypre_ParCompGrid;

/*--------------------------------------------------------------------------
 * Accessor functions for the Comp Grid structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCompGridNumNodes(compGrid)           ((compGrid) -> num_nodes)
#define hypre_ParCompGridNumOwnedNodes(compGrid)           ((compGrid) -> num_owned_nodes)
#define hypre_ParCompGridNumRealNodes(compGrid)           ((compGrid) -> num_real_nodes)
#define hypre_ParCompGridMemSize(compGrid)           ((compGrid) -> mem_size)
#define hypre_ParCompGridU(compGrid)           ((compGrid) -> u)
#define hypre_ParCompGridF(compGrid)           ((compGrid) -> f)
#define hypre_ParCompGridGlobalIndices(compGrid)           ((compGrid) -> global_indices)
#define hypre_ParCompGridCoarseGlobalIndices(compGrid)           ((compGrid) -> coarse_global_indices)
#define hypre_ParCompGridCoarseLocalIndices(compGrid)           ((compGrid) -> coarse_local_indices)
#define hypre_ParCompGridARows(compGrid)           ((compGrid) -> A_rows)
#define hypre_ParCompGridPRows(compGrid)           ((compGrid) -> P_rows)





#endif
