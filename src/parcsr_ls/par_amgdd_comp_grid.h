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
   HYPRE_Int         transition_level; // transition level used for adaptive AMG-RD (at this level and below, each proc owns the global grids)
   HYPRE_Int         *transition_res_recv_sizes; // if using the transition level, these are the recv sizes for the Allgatherv
   HYPRE_Int         *transition_res_recv_disps; // if useing the transition level, these are the recv displacements for the Allgatherv
	HYPRE_Int 			*num_send_procs; // number of send procs to communicate with
   HYPRE_Int         *num_recv_procs; // number of recv procs to communicate with
	HYPRE_Int 			*num_send_partitions; // number of neighbor paritions to communicate with
   MPI_Comm          *agg_local_comms; // local communicators for processor agglomeration on different levels
   MPI_Comm          *agg_global_comms; // global communicators between agglomerated partitions on different levels

   HYPRE_Int         **send_procs; // list of send procs
   HYPRE_Int         **recv_procs; // list of recv procs
   HYPRE_Int         **send_partitions; // list of neighbor partitions
   HYPRE_Int         **send_proc_partitions; // list of which partition each send proc belongs to
   HYPRE_Int         ***send_partition_ranks; // list of ranks that belong to each partition in partitions
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
 #define hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg)          ((compGridCommPkg) -> transition_level)
 #define hypre_ParCompGridCommPkgTransitionResRecvSizes(compGridCommPkg)          ((compGridCommPkg) -> transition_res_recv_sizes)
 #define hypre_ParCompGridCommPkgTransitionResRecvDisps(compGridCommPkg)          ((compGridCommPkg) -> transition_res_recv_disps)
 #define hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)				((compGridCommPkg) -> num_send_procs)
 #define hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)           ((compGridCommPkg) -> num_recv_procs)
 #define hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)				((compGridCommPkg) -> num_send_partitions)
 #define hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)       ((compGridCommPkg) -> agg_local_comms)
 #define hypre_ParCompGridCommPkgAggGlobalComms(compGridCommPkg)       ((compGridCommPkg) -> agg_global_comms)
 #define hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)           ((compGridCommPkg) -> send_procs)
 #define hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)           ((compGridCommPkg) -> recv_procs)
 #define hypre_ParCompGridCommPkgSendPartitions(compGridCommPkg)           ((compGridCommPkg) -> send_partitions)
 #define hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)           ((compGridCommPkg) -> send_proc_partitions)
 #define hypre_ParCompGridCommPkgSendPartitionRanks(compGridCommPkg)               ((compGridCommPkg) -> send_partition_ranks)
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

typedef struct
{
   HYPRE_Int       num_nodes; // total number of nodes including real and ghost nodes
   HYPRE_Int       num_owned_blocks; // number of blocks of owned nodes
   HYPRE_Int       *owned_block_starts; // start positions for the blocks of owned nodes
   HYPRE_Int       num_real_nodes; // number of real nodes
   HYPRE_Int       num_c_points; // number of C points
   HYPRE_Int		 mem_size;
   HYPRE_Int       A_mem_size;
   HYPRE_Int       P_mem_size;
   HYPRE_Int       R_mem_size;

   HYPRE_Int        *global_indices;
   HYPRE_Int        *coarse_global_indices; 
   HYPRE_Int        *coarse_local_indices;
   HYPRE_Int        *real_dof_marker;

   HYPRE_Int        *A_rowptr;
   HYPRE_Int        *A_colind;
   HYPRE_Int        *A_global_colind;
   HYPRE_Complex    *A_data;

   HYPRE_Int        *P_rowptr;
   HYPRE_Int        *P_colind;
   HYPRE_Complex    *P_data;

   HYPRE_Int        *R_rowptr;
   HYPRE_Int        *R_colind;
   HYPRE_Complex    *R_data;
   
   hypre_CSRMatrix  *A;
   hypre_CSRMatrix  *A_real;
   hypre_CSRMatrix  *P;
   hypre_CSRMatrix  *R;

   hypre_Vector     *u;
   hypre_Vector     *f;
   hypre_Vector     *t;
   hypre_Vector     *s;
   hypre_Vector     *temp;
   hypre_Vector     *temp2;
   hypre_Vector     *temp3;

   HYPRE_Real       *l1_norms;
   HYPRE_Int        *cf_marker_array;
   int              *c_mask;
   int              *f_mask;

   HYPRE_Real       *cheby_coeffs;

} hypre_ParCompGrid;

/*--------------------------------------------------------------------------
 * Accessor functions for the Comp Grid structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCompGridNumNodes(compGrid)           ((compGrid) -> num_nodes)
#define hypre_ParCompGridNumOwnedBlocks(compGrid)           ((compGrid) -> num_owned_blocks)
#define hypre_ParCompGridOwnedBlockStarts(compGrid)           ((compGrid) -> owned_block_starts)
#define hypre_ParCompGridNumRealNodes(compGrid)           ((compGrid) -> num_real_nodes)
#define hypre_ParCompGridNumCPoints(compGrid)           ((compGrid) -> num_c_points)
#define hypre_ParCompGridMemSize(compGrid)           ((compGrid) -> mem_size)
#define hypre_ParCompGridAMemSize(compGrid)           ((compGrid) -> A_mem_size)
#define hypre_ParCompGridPMemSize(compGrid)           ((compGrid) -> P_mem_size)
#define hypre_ParCompGridRMemSize(compGrid)           ((compGrid) -> R_mem_size)
#define hypre_ParCompGridU(compGrid)           ((compGrid) -> u)
#define hypre_ParCompGridF(compGrid)           ((compGrid) -> f)
#define hypre_ParCompGridT(compGrid)           ((compGrid) -> t)
#define hypre_ParCompGridS(compGrid)           ((compGrid) -> s)
#define hypre_ParCompGridTemp(compGrid)        ((compGrid) -> temp)
#define hypre_ParCompGridTemp2(compGrid)        ((compGrid) -> temp2)
#define hypre_ParCompGridTemp3(compGrid)        ((compGrid) -> temp3)
#define hypre_ParCompGridGlobalIndices(compGrid)           ((compGrid) -> global_indices)
#define hypre_ParCompGridCoarseGlobalIndices(compGrid)           ((compGrid) -> coarse_global_indices)
#define hypre_ParCompGridCoarseLocalIndices(compGrid)           ((compGrid) -> coarse_local_indices)
#define hypre_ParCompGridRealDofMarker(compGrid) ((compGrid) -> real_dof_marker)
#define hypre_ParCompGridARowPtr(compGrid)         ((compGrid) -> A_rowptr)
#define hypre_ParCompGridAColInd(compGrid)         ((compGrid) -> A_colind)
#define hypre_ParCompGridAGlobalColInd(compGrid)         ((compGrid) -> A_global_colind)
#define hypre_ParCompGridAData(compGrid)           ((compGrid) -> A_data)
#define hypre_ParCompGridPRowPtr(compGrid)         ((compGrid) -> P_rowptr)
#define hypre_ParCompGridPColInd(compGrid)         ((compGrid) -> P_colind)
#define hypre_ParCompGridPData(compGrid)           ((compGrid) -> P_data)
#define hypre_ParCompGridRRowPtr(compGrid)         ((compGrid) -> R_rowptr)
#define hypre_ParCompGridRColInd(compGrid)         ((compGrid) -> R_colind)
#define hypre_ParCompGridRData(compGrid)           ((compGrid) -> R_data)

#define hypre_ParCompGridA(compGrid)               ((compGrid) -> A)
#define hypre_ParCompGridAReal(compGrid)               ((compGrid) -> A_real)
#define hypre_ParCompGridP(compGrid)               ((compGrid) -> P)
#define hypre_ParCompGridR(compGrid)               ((compGrid) -> R)

#define hypre_ParCompGridL1Norms(compGrid)         ((compGrid) -> l1_norms)
#define hypre_ParCompGridCFMarkerArray(compGrid)         ((compGrid) -> cf_marker_array)
#define hypre_ParCompGridCMask(compGrid)         ((compGrid) -> c_mask)
#define hypre_ParCompGridFMask(compGrid)         ((compGrid) -> f_mask)

#define hypre_ParCompGridChebyCoeffs(compGrid)         ((compGrid) -> cheby_coeffs)

#endif
