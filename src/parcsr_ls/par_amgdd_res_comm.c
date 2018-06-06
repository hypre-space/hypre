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

#define HYPRE_TIMING

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"	

#define DEBUG_COMP_GRID 1 // if true, prints out what is stored in the comp grids for each processor to a file
#define DEBUGGING_MESSAGES 1 // if true, prints a bunch of messages to the screen to let you know where in the algorithm you are
#define USE_BARRIERS 1 // if true, puts MPI barriers between major tasks in setup phase (for timing purposes)

HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGrid *compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int padding );

HYPRE_Int
RecursivelyFindNeighborProcessors( hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, HYPRE_Int **add_flag,
   HYPRE_Int *num_starting_nodes, HYPRE_Int **starting_nodes,
   HYPRE_Int num_searching_procs, HYPRE_Int *searching_procs,
   HYPRE_Int *num_request_nodes, HYPRE_Int **request_nodes,
   HYPRE_Int *num_send_procs, HYPRE_Int *send_procs, 
   HYPRE_Int num_neighboring_procs );

HYPRE_Int
RecursivelyFindNeighborNodes(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, 
   HYPRE_Int *request_nodes, HYPRE_Int *num_request_nodes );

HYPRE_Complex*
PackSendBufferNew( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, HYPRE_Int *send_flag_buffer_size, 
   HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int padding );

HYPRE_Int
RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int *add_flag_coarse, 
   HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding);

HYPRE_Int 
GeneratePsiComposite( hypre_ParCompGrid **psiComposite, hypre_ParCompGrid **compGrid, hypre_ParCSRCommPkg *commPkg, HYPRE_Int *send_flag_buffer_size, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
GetBufferSize( hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels );

HYPRE_Complex*
PackSendBuffer( hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels, HYPRE_Int buffer_size );

HYPRE_Int
UnpackRecvBufferNew( HYPRE_Complex *recv_buffer, hypre_ParCompGrid **compGrid, HYPRE_Int **recv_map_send, 
      HYPRE_Int *recv_map_size, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, 
      HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index, HYPRE_Int *num_added_nodes, HYPRE_Int *num_incoming_nodes );

HYPRE_Int
UnpackRecvBuffer( HYPRE_Complex *recv_buffer, hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
AddToCompGrid(hypre_ParCompGrid **compGrid, hypre_ParCompGrid **psiComposite, HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_size, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels, HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index, HYPRE_Int *num_added_nodes );

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_send_buffer, HYPRE_Int *num_incoming_nodes, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
UnpackSendFlagBuffer(HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *send_buffer_size, HYPRE_Int *num_send_nodes, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
PackResidualBuffer( HYPRE_Complex *send_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
UnpackResidualBuffer( HYPRE_Complex *recv_buffer, HYPRE_Int **recv_map, hypre_ParCompGrid **compGrid, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
CommunicateGhostNodes(hypre_ParCompGrid **compGrid, HYPRE_Int **numGhostFromProc, HYPRE_Int **ghostInfoOffset, HYPRE_Int ***ghostGlobalIndex, HYPRE_Int *numNewGhostNodes, HYPRE_Int ***ghostUnpackIndex, HYPRE_Int num_levels, HYPRE_Int *global_nodes );

HYPRE_Int
FillGhostNodeResponse(void* recv_buf, int contact_size, int contact_proc, void* response_obj, MPI_Comm comm, void** response_buf, int* response_message_size);

HYPRE_Int
CommunicateGhostNodesResidualOnly(hypre_ParCompGrid **compGrid, HYPRE_Int **numGhostFromProc, HYPRE_Int ***ghostGlobalIndex, HYPRE_Int ***ghostUnpackIndex, HYPRE_Int num_levels, HYPRE_Int *global_nodes );

HYPRE_Int
FillGhostNodeResponseResidualOnly(void* recv_buf, int contact_size, int contact_proc, void* response_obj, MPI_Comm comm, void** response_buf, int* response_message_size);

HYPRE_Int
PackGhostNodeContact( HYPRE_Int num_levels, HYPRE_Int num_contacts, HYPRE_Int *contact_proc_list, HYPRE_Int **numGhostFromProc, HYPRE_Int **ghostInfoOffset, HYPRE_Int ***ghostGlobalIndex, HYPRE_Int *contact_send_buf, HYPRE_Int *contact_send_buf_starts );

HYPRE_Int
UnpackGhostNodeResponse( hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int num_contacts, HYPRE_Int *contact_proc_list, HYPRE_Complex *response_buf, HYPRE_Int *numNewGhostNodes, HYPRE_Int **ghostInfoOffset, HYPRE_Int ***ghostUnpackIndex);

HYPRE_Int
UnpackGhostNodeResponseResidualOnly( hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int num_contacts, HYPRE_Int *contact_proc_list, HYPRE_Complex *response_buf, HYPRE_Int ***ghostUnpackIndex);

HYPRE_Int
FindGhostNodes( hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index, HYPRE_Int **numGhostFromProc, HYPRE_Int **ghostInfoOffset, HYPRE_Int ***ghostGlobalIndex, HYPRE_Int *numNewGhostNodes, HYPRE_Int ***ghostUnpackIndex, HYPRE_Int *global_nodes, hypre_IJAssumedPart **apart );

HYPRE_Int 
LocateGhostNodes(HYPRE_Int **numGhostFromProc, HYPRE_Int ***ghostGlobalIndex, HYPRE_Int ***ghostUnpackIndex, HYPRE_Int **ghostInfoOffset, hypre_IJAssumedPart **apart, HYPRE_Int num_levels, HYPRE_Int *global_nodes);

HYPRE_Int
FillResponseForLocateGhostNodes(void *p_recv_contact_buf, HYPRE_Int contact_size, HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf, HYPRE_Int *response_message_size);

/*****************************************************************************
 *
 * Routine for communicating the composite grid residuals in AMG-DD
 *
 *****************************************************************************/

/*****************************************************************************
 * hypre_AMGDD_res_comm
 *****************************************************************************/

HYPRE_Int
hypre_BoomerAMGDDCompGridSetup( void *amg_vdata, HYPRE_Int *timers, HYPRE_Int padding )
{
   #if DEBUG_COMP_GRID
   char filename[256];
   #endif

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   #if DEBUGGING_MESSAGES
   hypre_printf("Began comp grid setup on rank %d\n", myid);
   #endif

   MPI_Comm 	      comm;
   hypre_ParAMGData   *amg_data = amg_vdata;

   /* Data Structure variables */
 
   // level counters, indices, and parameters
   HYPRE_Int                  num_levels;
   HYPRE_Real                 alpha, beta;
   HYPRE_Int                  level,i,j,k,cnt;
   HYPRE_Int                  *num_psi_levels_send;
   HYPRE_Int                  *num_psi_levels_recv;
   HYPRE_Int                  *num_added_nodes;

   // info from amg setup
   hypre_ParCSRMatrix         **A_array;
   hypre_ParVector            **F_array;
   hypre_ParVector            **U_array;
   hypre_ParCSRMatrix         **P_array;
   hypre_ParVector            *Vtemp;
   HYPRE_Int                  **CF_marker_array;
   HYPRE_Int                  *proc_first_index, *proc_last_index, *global_nodes;
   hypre_IJAssumedPart        **apart;

   // composite grids and temporary psiComposite grids used for communication
   hypre_ParCompGrid          **compGrid;
   // hypre_ParCompGrid          ***psiComposite_send;
   // hypre_ParCompGrid          ***psiComposite_recv;

   // info needed for later composite grid communication
   hypre_ParCompGridCommPkg   *compGridCommPkg;
   HYPRE_Int                  num_sends, num_recvs;
   HYPRE_Int                  **send_buffer_size;
   HYPRE_Int                  **recv_buffer_size;
   HYPRE_Int                  ***num_send_nodes;
   HYPRE_Int                  ****send_flag;
   HYPRE_Int                  ****recv_map;
   HYPRE_Int                  *numNewGhostNodes; // numNewGhostNodes[level]
   HYPRE_Int                  **numGhostFromProc; // numGhostFromProc[proc][level]
   HYPRE_Int                  **ghostInfoOffset; // ghostInfoOffset[proc][level]
   HYPRE_Int                  ***ghostGlobalIndex; // ghostGlobalIndex[proc][level][index]
   HYPRE_Int                  ***ghostUnpackIndex; // ghostUnpackIndex[proc][level][index]

   // temporary arrays used for communication during comp grid setup
   HYPRE_Complex              **send_buffer;
   HYPRE_Complex              **recv_buffer;
   HYPRE_Int                  ***recv_map_send;
   HYPRE_Int                  **num_incoming_nodes;
   HYPRE_Int                  **recv_map_size;
   HYPRE_Int                  **send_flag_buffer;
   HYPRE_Int                  **recv_map_send_buffer;
   HYPRE_Int                  *send_flag_buffer_size;
   HYPRE_Int                  *recv_map_send_buffer_size;

   // mpi stuff
   hypre_MPI_Request          *requests;
   hypre_MPI_Status           *status;
   HYPRE_Int                  request_counter = 0;

   // get info from amg
   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);
   Vtemp = hypre_ParAMGDataVtemp(amg_data);
   CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);
   num_levels = hypre_ParAMGDataNumLevels(amg_data);

   // get first and last global indices on each level for this proc
   proc_first_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   proc_last_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   global_nodes = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   num_added_nodes = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   apart = hypre_CTAlloc(hypre_IJAssumedPart*, num_levels, HYPRE_MEMORY_HOST);
   for (level = 0; level < num_levels; level++)
   {
      proc_first_index[level] = hypre_ParVectorFirstIndex(F_array[level]);
      proc_last_index[level] = hypre_ParVectorLastIndex(F_array[level]);
      global_nodes[level] = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
      apart[level] = hypre_ParCSRMatrixAssumedPartition(A_array[level]);
   }

   // Allocate space for some variables that store info on each level
   compGrid = hypre_CTAlloc(hypre_ParCompGrid*, num_levels, HYPRE_MEMORY_HOST);
   compGridCommPkg = hypre_ParCompGridCommPkgCreate();
   hypre_ParCompGridCommPkgNumLevels(compGridCommPkg) = num_levels;
   hypre_ParCompGridCommPkgNumSends(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   send_buffer_size = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   recv_buffer_size = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   send_flag = hypre_CTAlloc(HYPRE_Int***, num_levels, HYPRE_MEMORY_HOST);
   num_send_nodes = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);
   recv_map = hypre_CTAlloc(HYPRE_Int***, num_levels, HYPRE_MEMORY_HOST);
   numGhostFromProc =  hypre_CTAlloc(HYPRE_Int*, num_procs, HYPRE_MEMORY_HOST);
   ghostInfoOffset =  hypre_CTAlloc(HYPRE_Int*, num_procs, HYPRE_MEMORY_HOST);
   ghostGlobalIndex = hypre_CTAlloc(HYPRE_Int**, num_procs, HYPRE_MEMORY_HOST);
   numNewGhostNodes = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   ghostUnpackIndex = hypre_CTAlloc(HYPRE_Int**, num_procs, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_procs; i++)
   {
      numGhostFromProc[i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
      ghostInfoOffset[i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
      ghostGlobalIndex[i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
      ghostUnpackIndex[i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
      for (j = 0; j < num_levels; j++) ghostUnpackIndex[i][j] = NULL;
      for (j = 0; j < num_levels; j++) ghostGlobalIndex[i][j] = NULL;
   }

   /* Form residual and restrict down to all levels and initialize composite grids 
      Note that residuals will be stored in F_array and the fine grid RHS will be stored in Vtemp */
   hypre_ParVectorCopy(F_array[0],Vtemp);
   alpha = -1.0;
   beta = 1.0;
   hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0],
                        beta, F_array[0]);

   compGrid[0] = hypre_ParCompGridCreate();
   hypre_ParCompGridInitialize( compGrid[0], F_array[0], CF_marker_array[0], proc_first_index[1], A_array[0], P_array[0] );

   for (level = 0; level < num_levels-1; level++)
   {
      alpha = 1.0;
      beta = 0.0;
      hypre_ParCSRMatrixMatvecT(alpha,P_array[level],F_array[level],
                            beta,F_array[level+1]);

      compGrid[level+1] = hypre_ParCompGridCreate();
      // if (myid == 3) hypre_printf("Initialize level %d\n", level+1);
      if (level != num_levels-2) hypre_ParCompGridInitialize( compGrid[level+1], F_array[level+1], CF_marker_array[level+1], proc_first_index[level+2], A_array[level+1], P_array[level+1] );
      else hypre_ParCompGridInitialize( compGrid[level+1], F_array[level+1], CF_marker_array[level+1], 0, A_array[level+1], NULL );
   }

   // Now that the comp grids are initialized, store RHS back in F_array[0]
   hypre_ParVectorCopy(Vtemp,F_array[0]);







   // Debugging:
   for (i = 0; i < num_levels; i++)
   {
      hypre_sprintf(filename, "outputs/CompGrids/initCompGridRank%dLevel%d.txt", myid, i);
      hypre_ParCompGridDebugPrint( compGrid[i], filename );
   }



   // On each level, setup a long distance commPkg that has communication info for distance (eta + numGhostLayers)
   for (level = 0; level < num_levels; level++)
   {
      // hypre_printf("      Rank %d: SetupNearestProcessorNeighbors() Level %d\n", myid, level);
      SetupNearestProcessorNeighbors(A_array[level], compGrid[level], compGridCommPkg, level, padding);
   }
   

   // Debugging: dump info on the compGridCommPkg
   for (i = 0; i < num_levels; i++)
   {
      hypre_sprintf(filename, "outputs/CompGrids/initCompGridCommPkgRank%dLevel%d.txt",myid,i);
      FILE *file = fopen(filename,"w");
      num_sends = hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[i];
      fprintf(file, "num_sends = %d, send_procs = ", num_sends);
      for (j = 0; j < num_sends; j++) fprintf(file, "%d ", hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[i][j]);
      fprintf(file,"\n");
      fprintf(file, "send_map_starts = ");
      for (j = 0; j < num_sends+1; j++) fprintf(file, "%d ", hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[i][j]);
      fprintf(file,"\n");
      fprintf(file, "send_map_elmts = ");
      for (j = 0; j < hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[i][num_sends]; j++) fprintf(file, "%d ", hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[i][j]);
      fprintf(file,"\n");
      fclose(file);
   }
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);


   // For the given padding, eta, need to compute A^eta to determine processor neighbors of degree eta !!! Need a more efficient way of doing this !!!
   // hypre_ParCSRMatrix **A_eta_array;
   // A_eta_array = hypre_CTAlloc(hypre_ParCSRMatrix*, num_levels, HYPRE_MEMORY_HOST);
   // for (level = 0; level < num_levels; level++)
   // {

   //    A_eta_array[level] = hypre_ParMatmul(A_array[level], A_array[level]);
   //    for (i = 0; i < padding + 1; i++)
   //    {
   //       #if DEBUGGING_MESSAGES
   //       printf("Rank %d: Expanding stencil for padding %d\n", myid, i);
   //       #endif
   //       hypre_ParCSRMatrix *old_matrix = A_eta_array[level];
   //       A_eta_array[level] = hypre_ParMatmul(A_array[level], old_matrix);
   //       hypre_ParCSRMatrixDestroy(old_matrix); // Cleanup old matrices to prevent memory leak
   //    }
   //    // Create the commpkg for A^eta on this level
   //    hypre_MatvecCommPkgCreate ( A_eta_array[level] );
   // }

   // // On the coarsest level, need to make sure eta is large enough such that all procs send all nodes to other procs with nodes
   // HYPRE_Int need_to_expand_stencil_local = 1;
   // HYPRE_Int need_to_expand_stencil_global = 0;
   // while (need_to_expand_stencil_local)
   // {
   //    need_to_expand_stencil_local = 0;
   //    commPkg =  hypre_ParCSRMatrixCommPkg( A_eta_array[num_levels - 1] );
   //    HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends( commPkg );
   //    for (i = 0; i < num_sends; i++)
   //    {
   //       HYPRE_Int start = hypre_ParCSRCommPkgSendMapStart(commPkg, i);
   //       HYPRE_Int finish = hypre_ParCSRCommPkgSendMapStart(commPkg, i+1);
   //       if ( (finish - start) != hypre_ParCSRMatrixNumRows( A_eta_array[num_levels - 1] ) ) need_to_expand_stencil_local = 1;
   //    }
   //    hypre_MPI_Allreduce(&need_to_expand_stencil_global, &need_to_expand_stencil_local, 1, HYPRE_MPI_INT, hypre_MPI_MAX, hypre_MPI_COMM_WORLD);
   //    // hypre_printf("Rank %d: need_to_expand_stencil_local = %d, need_to_expand_stencil_global = %d\n", myid, need_to_expand_stencil_local, need_to_expand_stencil_global);
   //    if (need_to_expand_stencil_global)
   //    {
   //       #if DEBUGGING_MESSAGES
   //       printf("Rank %d: Expanding stencil on coarsest level\n", myid);
   //       #endif
   //       hypre_ParCSRMatrix *old_matrix = A_eta_array[num_levels - 1];
   //       A_eta_array[num_levels - 1] = hypre_ParMatmul(A_array[num_levels - 1], old_matrix);
   //       if (old_matrix != A_array[num_levels - 1]) hypre_ParCSRMatrixDestroy(old_matrix);
   //       hypre_MatvecCommPkgCreate ( A_eta_array[num_levels - 1] );              
   //    }
   // }














   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   #if DEBUGGING_MESSAGES
   if (myid == 0) printf("  Looping over levels\n");
   #endif
   for (level = num_levels-1; level > -1; level--)
   {
      #if DEBUGGING_MESSAGES
      if (myid == 0) printf("    Level %d:\n", level);
      #endif
      if ( proc_last_index[level] >= proc_first_index[level] ) // If there are any owned nodes on this level
      {


         // Get the commPkg for this level


         // Get the commPkg of matrix A^eta on this level
         // commPkg = hypre_ParCSRMatrixCommPkg(A_eta_array[level]);
         // comm = hypre_ParCSRCommPkgComm(commPkg);
         // num_sends = hypre_ParCSRCommPkgNumSends(commPkg);
         // num_recvs = hypre_ParCSRCommPkgNumRecvs(commPkg);


         comm = hypre_MPI_COMM_WORLD;
         num_sends = hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level];
         num_recvs = hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level];




         // Copy over some info for comp grid comm pkg
         // hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level] = num_sends;
         // hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level] = num_recvs;
         // hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
         // hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, num_recvs, HYPRE_MEMORY_HOST);
         // hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);
         // hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStarts(commPkg)[num_sends], HYPRE_MEMORY_HOST );
         // memcpy( hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level], hypre_ParCSRCommPkgSendProcs(commPkg), num_sends * sizeof(HYPRE_Int) );
         // memcpy( hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level], hypre_ParCSRCommPkgRecvProcs(commPkg), num_sends * sizeof(HYPRE_Int) );
         // memcpy( hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level], hypre_ParCSRCommPkgSendMapStarts(commPkg), (num_sends + 1)*sizeof(HYPRE_Int) );
         // memcpy( hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level], hypre_ParCSRCommPkgSendMapElmts(commPkg), hypre_ParCSRCommPkgSendMapStarts(commPkg)[num_sends]*sizeof(HYPRE_Int) );

         // allocate space for the buffers, buffer sizes, requests and status, psiComposite_send, psiComposite_recv, send and recv maps
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         request_counter = 0;
         send_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
         recv_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_recvs, HYPRE_MEMORY_HOST);
         send_buffer = hypre_CTAlloc(HYPRE_Complex*, num_sends, HYPRE_MEMORY_HOST);
         recv_buffer = hypre_CTAlloc(HYPRE_Complex*, num_recvs, HYPRE_MEMORY_HOST);
         // psiComposite_send = hypre_CTAlloc(hypre_ParCompGrid**, num_sends, HYPRE_MEMORY_HOST);
         // psiComposite_recv = hypre_CTAlloc(hypre_ParCompGrid**, num_recvs, HYPRE_MEMORY_HOST);
         num_psi_levels_send = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
         num_psi_levels_recv = hypre_CTAlloc(HYPRE_Int, num_recvs, HYPRE_MEMORY_HOST);

         send_flag[level] = hypre_CTAlloc(HYPRE_Int**, num_sends, HYPRE_MEMORY_HOST);
         num_send_nodes[level] = hypre_CTAlloc(HYPRE_Int*, num_sends, HYPRE_MEMORY_HOST);
         recv_map[level] = hypre_CTAlloc(HYPRE_Int**, num_recvs, HYPRE_MEMORY_HOST);
         recv_map_send = hypre_CTAlloc(HYPRE_Int**, num_recvs, HYPRE_MEMORY_HOST);
         recv_map_size = hypre_CTAlloc(HYPRE_Int*, num_recvs, HYPRE_MEMORY_HOST);
         send_flag_buffer = hypre_CTAlloc(HYPRE_Int*, num_sends, HYPRE_MEMORY_HOST);
         send_flag_buffer_size = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
         recv_map_send_buffer = hypre_CTAlloc(HYPRE_Int*, num_recvs, HYPRE_MEMORY_HOST);
         recv_map_send_buffer_size = hypre_CTAlloc(HYPRE_Int, num_recvs, HYPRE_MEMORY_HOST);
         num_incoming_nodes = hypre_CTAlloc(HYPRE_Int*, num_recvs, HYPRE_MEMORY_HOST);

         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

         if (timers) hypre_BeginTiming(timers[1]);

         // loop over send procs
         #if DEBUGGING_MESSAGES
         if (myid == 0) printf("      Loop over send procs:\n");
         #endif

         // for (i = 0; i < num_sends; i++)
         // {
         //    // allocate space for psiComposite_send
         //    psiComposite_send[i] = hypre_CTAlloc(hypre_ParCompGrid*, num_levels, HYPRE_MEMORY_HOST);

         //    // generate psiComposite
         //    #if DEBUGGING_MESSAGES
         //    if (myid == 0) printf("        GeneratePsiComposite() for proc %d\n", hypre_ParCSRCommPkgSendProcs(commPkg)[i]);
         //    #endif
         //    num_psi_levels_send[i] = GeneratePsiComposite( psiComposite_send[i], compGrid, commPkg, &(send_flag_buffer_size[i]), i, level, num_levels );
         // }

         for (i = 0; i < num_sends; i++)
         {
            // pack send buffers
            #if DEBUGGING_MESSAGES
            // if (myid == 0) printf("        PackSendBuffer() for proc %d\n", hypre_ParCSRCommPkgSendProcs(commPkg)[i]);
            #endif
            send_buffer[i] = PackSendBufferNew( compGrid, compGridCommPkg, &(send_buffer_size[level][i]), &(send_flag_buffer_size[i]), i, level, num_levels, padding );
         }

         if (timers) hypre_EndTiming(timers[1]);

         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

         if (timers) hypre_BeginTiming(timers[2]);

         // post the receives for the buffer size
         for (i = 0; i < num_recvs; i++)
         {
            hypre_MPI_Irecv( &(recv_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 0, comm, &requests[request_counter++] );
         }

         // send the buffer sizes
         for (i = 0; i < num_sends; i++)
         {
            // send_buffer_size[level][i] = GetBufferSize( psiComposite_send[i], level, num_levels, num_psi_levels_send[i] );
            hypre_MPI_Isend(&(send_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 0, comm, &requests[request_counter++]);
         }
         
         // wait for all buffer sizes to be received
         hypre_printf("Rank %d: wait on buffer sizes\n", myid);
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );
         hypre_printf("Rank %d: done waiting on buffer sizes\n", myid);

         if (timers) hypre_EndTiming(timers[2]);


         // free and reallocate space for the requests and status
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

         if (timers) hypre_BeginTiming(timers[3]);

         // allocate space for the receive buffers and post the receives
         for (i = 0; i < num_recvs; i++)
         {
            recv_buffer[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size[level][i], HYPRE_MEMORY_HOST );
            hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_COMPLEX, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 1, comm, &requests[request_counter++]);
         }

         // pack and send the buffers
         #if DEBUGGING_MESSAGES
         if (myid == 0) printf("      Loop over send procs:\n");
         #endif
         for (i = 0; i < num_sends; i++)
         {
            #if DEBUGGING_MESSAGES
            if (myid == 0) printf("        Post send for proc %d\n", hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i]);
            #endif
            // send_buffer[i] = PackSendBuffer( psiComposite_send[i], level, num_levels, num_psi_levels_send[i], send_buffer_size[level][i] );
            hypre_MPI_Isend(send_buffer[i], send_buffer_size[level][i], HYPRE_MPI_COMPLEX, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 1, comm, &requests[request_counter++]);
         }

         // wait for buffers to be received
         hypre_printf("Rank %d: wait on buffers\n", myid);
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );
         hypre_printf("Rank %d: done waiting on buffers\n", myid);




         // for (i = 0; i < num_recvs; i++)
         // {
         //    FILE *file;
         //    hypre_sprintf(filename, "outputs/recv_buffer_level%d_rank%d.txt", level, myid);
         //    file = fopen(filename, "w");
         //    for (j = 0; j < recv_buffer_size[level][i]; j++)
         //       fprintf(file, "%f\n", recv_buffer[i][j]);
         //    fclose(file);
         // }





         if (timers) hypre_EndTiming(timers[3]);

         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

         if (timers) hypre_BeginTiming(timers[4]);

         // loop over received buffers
         #if DEBUGGING_MESSAGES
         if (myid == 0) printf("      Loop over recv procs:\n");
         #endif
         for (i = 0; i < num_recvs; i++)
         {
            // unpack the buffers
            // psiComposite_recv[i] = hypre_CTAlloc(hypre_ParCompGrid*, num_levels, HYPRE_MEMORY_HOST);
            // num_psi_levels_recv[i] = UnpackRecvBuffer( recv_buffer[i], psiComposite_recv[i], level, num_levels );

            // allocate space for the recv map info
            recv_map_send[i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            recv_map_size[i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
            num_incoming_nodes[i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);

            // and add information to this composite grid
            #if DEBUGGING_MESSAGES
            if (myid == 0) printf("        AddToCompGrid() for proc %d\n", hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i]);
            #endif
            // AddToCompGrid(compGrid, psiComposite_recv[i], recv_map_send[i], recv_map_size[i], &(recv_map_send_buffer_size[i]), level, num_levels, num_psi_levels_recv[i], proc_first_index, proc_last_index, num_added_nodes );
         
            UnpackRecvBufferNew(recv_buffer[i], compGrid, recv_map_send[i], recv_map_size[i], &(recv_map_send_buffer_size[i]), level, num_levels, proc_first_index, proc_last_index, num_added_nodes, num_incoming_nodes[i]);
         }




         // Debugging:
         // for (i = 0; i < num_levels; i++)
         // {
         //    hypre_sprintf(filename, "outputs/CompGrids/afterUnpack%dCompGridRank%dLevel%d.txt", level, myid, i);
         //    hypre_ParCompGridDebugPrint( compGrid[i], filename );
         // }



         if (timers) hypre_EndTiming(timers[4]);

         // Setup local indices for the composite grid
         #if DEBUGGING_MESSAGES
         if (myid == 0) printf("      Setup local indices\n");
         #endif
         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif
         if (timers) hypre_BeginTiming(timers[9]);
         hypre_ParCompGridSetupLocalIndices(compGrid, num_added_nodes, num_levels, proc_first_index, proc_last_index);
         if (timers) hypre_EndTiming(timers[9]);








         // Debugging:
         // for (i = 0; i < num_levels; i++)
         // {
         //    hypre_sprintf(filename, "outputs/CompGrids/afterSetupLocal%dCompGridRank%dLevel%d.txt", level, myid, i);
         //    hypre_ParCompGridDebugPrint( compGrid[i], filename );
         // }






         // Zero out num_added_nodes
         for (i = level; i < num_levels; i++) num_added_nodes[i] = 0;

         // free and reallocate space for the requests and status
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

         if (timers) hypre_BeginTiming(timers[5]);         

         // post receives for send maps - NOTE: we want to receive this info from procs we sent to
         for (i = 0; i < num_sends; i++)
         {
            send_flag_buffer[i] = hypre_CTAlloc(HYPRE_Int, send_flag_buffer_size[i], HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv( send_flag_buffer[i], send_flag_buffer_size[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 2, comm, &requests[request_counter++]);
         }

         // send recv_map_send to procs received from to become their send maps - NOTE: we want to send this info from procs we received from
         for (i = 0; i < num_recvs; i++)
         {
            // pack up the recv_map_send's and send them
            recv_map_send_buffer[i] = hypre_CTAlloc(HYPRE_Int, recv_map_send_buffer_size[i], HYPRE_MEMORY_HOST);
            PackRecvMapSendBuffer(recv_map_send[i], recv_map_send_buffer[i], num_incoming_nodes[i], level, num_levels);



            char filename[256];
            FILE *file;
            sprintf(filename,"outputs/recv_map_send_buffer_level%d_%d_to_%d.txt", level, myid, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i]);
            file = fopen(filename,"w");
            fprintf(file, "send size = %d, recv size = %d\n", recv_map_send_buffer_size[i], send_flag_buffer_size[i]);
            for (j = 0; j < recv_map_send_buffer_size[i]; j++)
               fprintf(file, "%d\n", recv_map_send_buffer[i][j] );
            fclose(file);



            hypre_MPI_Isend( recv_map_send_buffer[i], recv_map_send_buffer_size[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 2, comm, &requests[request_counter++]);
         }

         // wait for maps to be received
         hypre_printf("Rank %d: wait on send map buffer\n", myid);
         // HYPRE_Int count;
         // hypre_MPI_Get_count(status, HYPRE_MPI_COMPLEX, &count);
         // hypre_printf("Rank %d: count = %d\n", myid, count);
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );
         // hypre_MPI_Get_count(status, HYPRE_MPI_COMPLEX, &count);
         // hypre_printf("Rank %d: count = %d\n", myid, count);
         hypre_printf("Rank %d: done waiting on send map buffers\n", myid);

         // unpack and setup the send flag arrays
         for (i = 0; i < num_sends; i++)
         {
            send_flag[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_send_nodes[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
            UnpackSendFlagBuffer(send_flag_buffer[i], send_flag[level][i], &(send_buffer_size[level][i]), num_send_nodes[level][i], level, num_levels);
         }

         if (timers) hypre_EndTiming(timers[5]);

         // finalize the recv maps and get final recv buffer size
         for (i = 0; i < num_recvs; i++)
         {
            // buffers will store number of nodes on each level
            recv_buffer_size[level][i] = num_levels - level;

            // allocate space for each level of the receive map for this proc
            recv_map[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);

            // for each level
            for (j = level; j < num_levels; j++)
            {
               // if there is info for this proc on this level
               if (recv_map_send[i][j])
               {
                  // allocate the appropriate amount of space for the map
                  recv_map[level][i][j] = hypre_CTAlloc(HYPRE_Int, recv_map_size[i][j], HYPRE_MEMORY_HOST);
                  cnt = 0;

                  for (k = 0; k < num_incoming_nodes[i][j]; k++)
                  {
                     if ( recv_map_send[i][j][k] != -1 )
                     {
                        recv_map[level][i][j][cnt++] = recv_map_send[i][j][k];
                        recv_buffer_size[level][i]++;
                     }
                  }
               }
            }
         }

         // clean up memory for this level
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; i++)
         {
            hypre_TFree(send_buffer[i], HYPRE_MEMORY_HOST);
            hypre_TFree(send_flag_buffer[i], HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < num_recvs; i++)
         {
            hypre_TFree(recv_buffer[i], HYPRE_MEMORY_HOST);
            hypre_TFree(recv_map_send_buffer[i], HYPRE_MEMORY_HOST);
            hypre_TFree(recv_map_size[i], HYPRE_MEMORY_HOST);
            hypre_TFree(num_incoming_nodes[i], HYPRE_MEMORY_HOST);
            for (j = 0; j < num_levels; j++)
            {
               if (recv_map_send[i][j]) hypre_TFree(recv_map_send[i][j], HYPRE_MEMORY_HOST);
            }
            hypre_TFree(recv_map_send[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(send_buffer, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_buffer, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_map_send, HYPRE_MEMORY_HOST);
         hypre_TFree(send_flag_buffer, HYPRE_MEMORY_HOST);
         hypre_TFree(send_flag_buffer_size, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_map_send_buffer, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_map_send_buffer_size, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_map_size, HYPRE_MEMORY_HOST);
         hypre_TFree(num_incoming_nodes, HYPRE_MEMORY_HOST);
         hypre_TFree(num_psi_levels_send, HYPRE_MEMORY_HOST);
         hypre_TFree(num_psi_levels_recv, HYPRE_MEMORY_HOST);
      }
      else 
      {
         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif
      }
      hypre_printf("Rank %d: done with level %d\n", myid, level);
   }


   #if DEBUG_COMP_GRID
   for (level = 0; level < num_levels; level++)
   {
      // hypre_sprintf(filename, "/p/lscratchd/wbm/CompGrids/setupCompGridRank%dLevel%d.txt", myid, level);
      hypre_sprintf(filename, "outputs/CompGrids/setupCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrint( compGrid[level], filename );
      if (myid == 0)
      {
         FILE             *file;
         // hypre_sprintf(filename,"/p/lscratchd/wbm/CompGrids/global_num_nodes.txt");
         hypre_sprintf(filename,"outputs/CompGrids/global_num_nodes.txt");
         file = fopen(filename,"w");
         hypre_fprintf(file, "%d\n", hypre_ParCSRMatrixGlobalNumRows(A_array[0]) );
         fclose(file);
         // Print info on how to read files
         // hypre_sprintf(filename,"/p/lscratchd/wbm/CompGrids/info.txt");
         hypre_sprintf(filename,"outputs/CompGrids/info.txt");
         file = fopen(filename,"w");
         hypre_fprintf(file, "num_nodes\nmem_size\nnum_owned_nodes\nsolution values\nresidual values\nglobal indices\ncoarse global indices\ncoarse local indices\nrows of matrix A: size, data, global indices, local indices\nrows of matrix P: size, data, global indices, local indices\nghost P rows: size, data, global indices, local indices\n");
         fclose(file);
      }
   }
   #endif

   // store communication info in compGridCommPkg
   hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg) = send_buffer_size;
   hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg) = recv_buffer_size;
   hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg) = num_send_nodes;
   hypre_ParCompGridCommPkgSendFlag(compGridCommPkg) = send_flag;
   hypre_ParCompGridCommPkgRecvMap(compGridCommPkg) = recv_map;
   hypre_ParCompGridCommPkgNumGhostFromProc(compGridCommPkg) = ghostInfoOffset;
   hypre_ParCompGridCommPkgGhostGlobalIndex(compGridCommPkg) = ghostGlobalIndex;
   hypre_ParCompGridCommPkgGhostUnpackIndex(compGridCommPkg) = ghostUnpackIndex;

   // assign compGrid and compGridCommPkg info to the amg structure
   hypre_ParAMGDataCompGrid(amg_data) = compGrid;
   hypre_ParAMGDataCompGridCommPkg(amg_data) = compGridCommPkg;

   #if DEBUGGING_MESSAGES
   hypre_printf("Finished comp grid setup on rank %d\n", myid);
   #endif

   // Cleanup memory
   for (i = 0; i < num_procs; i++) hypre_TFree(numGhostFromProc[i], HYPRE_MEMORY_HOST);
   hypre_TFree(numGhostFromProc, HYPRE_MEMORY_HOST);
   hypre_TFree(numNewGhostNodes, HYPRE_MEMORY_HOST);
   hypre_TFree(num_added_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(global_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_first_index, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_last_index, HYPRE_MEMORY_HOST);
   hypre_TFree(apart, HYPRE_MEMORY_HOST);
   // if (padding > 1)
   // {
   //    for (level = 0; level < num_levels; level++)
   //    {
   //       hypre_ParCSRMatrixDestroy( A_eta_array[level] );
   //    }
   //    hypre_TFree( A_eta_array, HYPRE_MEMORY_HOST );
   // }
   
   return 0;
}  

HYPRE_Int 
hypre_BoomerAMGDDResidualCommunication( void *amg_vdata )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   #if DEBUGGING_MESSAGES
   hypre_printf("Began residual communication on rank %d\n", myid);
   #endif

   MPI_Comm          comm;
   hypre_ParAMGData   *amg_data = amg_vdata;
   
   /* Data Structure variables */

   // level counters, indices, and parameters
   HYPRE_Int                  num_levels;
   HYPRE_Real                 alpha, beta;
   HYPRE_Int                  level,i;

   // info from amg
   hypre_ParCSRMatrix         **A_array;
   hypre_ParVector            **F_array;
   hypre_ParVector            **U_array;
   hypre_ParCSRMatrix         **P_array;
   hypre_ParVector            *Vtemp;
   HYPRE_Int                  *proc_first_index, *proc_last_index;
   HYPRE_Int                  *global_nodes;
   hypre_ParCompGrid          **compGrid;

   // info from comp grid comm pkg
   hypre_ParCompGridCommPkg   *compGridCommPkg;
   HYPRE_Int                  num_sends, num_recvs;
   HYPRE_Int                  **send_procs;
   HYPRE_Int                  **recv_procs;
   HYPRE_Int                  **send_buffer_size;
   HYPRE_Int                  **recv_buffer_size;
   HYPRE_Int                  ***num_send_nodes;
   HYPRE_Int                  ****send_flag;
   HYPRE_Int                  ****recv_map;

   // temporary arrays used for communication during comp grid setup
   HYPRE_Complex              **send_buffer;
   HYPRE_Complex              **recv_buffer;

   // temporary vectors used to copy data into composite grid structures
   hypre_Vector      *residual_local;
   HYPRE_Complex     *residual_data;

   // mpi stuff
   hypre_MPI_Request          *requests;
   hypre_MPI_Status           *status;
   HYPRE_Int                  request_counter = 0;

   // get info from amg
   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);
   Vtemp = hypre_ParAMGDataVtemp(amg_data);
   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   compGrid = hypre_ParAMGDataCompGrid(amg_data);
   compGridCommPkg = hypre_ParAMGDataCompGridCommPkg(amg_data);

   // get info from comp grid comm pkg
   send_procs = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg);
   recv_procs = hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg);
   send_buffer_size = hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg);
   recv_buffer_size = hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg);
   num_send_nodes = hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg);
   send_flag = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg);
   recv_map = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg);

   // get first and last global indices on each level for this proc
   proc_first_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   proc_last_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   global_nodes = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   for (level = 0; level < num_levels; level++)
   {
      proc_first_index[level] = hypre_ParVectorFirstIndex(F_array[level]);
      proc_last_index[level] = hypre_ParVectorLastIndex(F_array[level]);
      global_nodes[level] = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
   }

   /* Form residual and restrict down to all levels and initialize composite grids 
      Note that from here on, residuals will be stored in F_array and the fine grid RHS will be stored in Vtemp */
   hypre_ParVectorCopy(F_array[0],Vtemp);
   alpha = -1.0;
   beta = 1.0;
   hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0],
                        beta, F_array[0]);

   for (level = 0; level < num_levels-1; level++)
   {
      alpha = 1.0;
      beta = 0.0;
      hypre_ParCSRMatrixMatvecT(alpha,P_array[level],F_array[level],
                            beta,F_array[level+1]);
   }

   // copy new restricted residual into comp grid structure
   for (level = 0; level < num_levels; level++)
   {
      // Access the residual data
      residual_local = hypre_ParVectorLocalVector(F_array[level]);
      residual_data = hypre_VectorData(residual_local);
      for (i = 0; i < hypre_VectorSize(residual_local); i++)
      {
         hypre_ParCompGridF(compGrid[level])[i] = residual_data[i];
      }
   }

   // Copy RHS back into F_array[0]
   hypre_ParVectorCopy(Vtemp,F_array[0]);

   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   for (level = num_levels-1; level > -1; level--)
   {      
      if ( proc_last_index[level] >= proc_first_index[level] ) // If there are any owned nodes on this level
      {
         // Get some communication info
         comm = hypre_ParCSRMatrixComm(A_array[level]);
         num_sends = hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level];
         num_recvs = hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level];

         // allocate space for the buffers, buffer sizes, requests and status, psiComposite_send, psiComposite_recv, send and recv maps
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         request_counter = 0;
         send_buffer = hypre_CTAlloc(HYPRE_Complex*, num_sends, HYPRE_MEMORY_HOST);
         recv_buffer = hypre_CTAlloc(HYPRE_Complex*, num_recvs, HYPRE_MEMORY_HOST);


         // allocate space for the receive buffers and post the receives
         for (i = 0; i < num_recvs; i++)
         {
            recv_buffer[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size[level][i], HYPRE_MEMORY_HOST );
            hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_COMPLEX, recv_procs[level][i], 0, comm, &requests[request_counter++]);
         }

         // pack and send the buffers
         for (i = 0; i < num_sends; i++)
         {
            send_buffer[i] = hypre_CTAlloc(HYPRE_Complex, send_buffer_size[level][i], HYPRE_MEMORY_HOST);
            PackResidualBuffer(send_buffer[i], send_flag[level][i], num_send_nodes[level][i], compGrid, compGridCommPkg, i, level, num_levels);
            hypre_MPI_Isend(send_buffer[i], send_buffer_size[level][i], HYPRE_MPI_COMPLEX, send_procs[level][i], 0, comm, &requests[request_counter++]);
         }

         // wait for buffers to be received
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );

         // loop over received buffers
         for (i = 0; i < num_recvs; i++)
         {
            // unpack the buffers
            UnpackResidualBuffer(recv_buffer[i], recv_map[level][i], compGrid, level, num_levels);
         }

         // clean up memory for this level
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; i++)
         {
            hypre_TFree(send_buffer[i], HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < num_recvs; i++)
         {
            hypre_TFree(recv_buffer[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(send_buffer, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_buffer, HYPRE_MEMORY_HOST);
      }
   }

   // Communicate ghost residuals
   HYPRE_Int **numGhostFromProc = hypre_ParCompGridCommPkgNumGhostFromProc(compGridCommPkg);
   HYPRE_Int ***ghostGlobalIndex = hypre_ParCompGridCommPkgGhostGlobalIndex(compGridCommPkg);
   HYPRE_Int ***ghostUnpackIndex = hypre_ParCompGridCommPkgGhostUnpackIndex(compGridCommPkg);
   // CommunicateGhostNodesResidualOnly(compGrid, numGhostFromProc, ghostGlobalIndex, ghostUnpackIndex, num_levels, global_nodes);

   // #if DEBUG_COMP_GRID
   // char filename[256];
   // for (level = 0; level < num_levels; level++)
   // {
   //    hypre_sprintf(filename, "outputs/CompGrids/communicateCompGridRank%dLevel%d.txt", myid, level);
   //    hypre_ParCompGridDebugPrint( compGrid[level], filename );
   // }
   // #endif

   // store communication info in compGridCommPkg
   hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg) = send_buffer_size;
   hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg) = recv_buffer_size;
   hypre_ParCompGridCommPkgSendFlag(compGridCommPkg) = send_flag;
   hypre_ParCompGridCommPkgRecvMap(compGridCommPkg) = recv_map;

   // assign compGrid and compGridCommPkg info to the amg structure
   hypre_ParAMGDataCompGrid(amg_data) = compGrid;
   hypre_ParAMGDataCompGridCommPkg(amg_data) = compGridCommPkg;

   #if DEBUGGING_MESSAGES
   hypre_printf("Finished residual communication on rank %d\n", myid);
   #endif

   // Cleanup memory
   hypre_TFree(proc_first_index, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_last_index, HYPRE_MEMORY_HOST);
   hypre_TFree(global_nodes, HYPRE_MEMORY_HOST);
   
   return 0;
}








HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGrid *compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int padding )
{
   HYPRE_Int               i,j,cnt;
   HYPRE_Int               num_nodes = hypre_ParCSRMatrixNumRows(A);
   hypre_ParCSRCommPkg     *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int               start,finish;

   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   // hypre_printf("Rank %d: Inside SetupNearestProcessorNeighbors()\n", myid);

   // Initialize add_flag (this is how we will track nodes to send to each proc until routine finishes)
   HYPRE_Int      num_sends = hypre_ParCSRCommPkgNumSends(commPkg);
   HYPRE_Int      *send_procs = hypre_CTAlloc( HYPRE_Int, (padding+4)*num_sends, HYPRE_MEMORY_HOST); // Note: in the worst case, we extend the number of procs to send to by padding+4 times
   HYPRE_Int      **add_flag = hypre_CTAlloc( HYPRE_Int*, (padding+4)*num_sends, HYPRE_MEMORY_HOST);
   HYPRE_Int      *searching_procs = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_sends; i++)
   {
      send_procs[i] = hypre_ParCSRCommPkgSendProc(commPkg,i);
      add_flag[i] = hypre_CTAlloc( HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
      start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
      finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
      for (j = start; j < finish; j++) add_flag[i][ hypre_ParCSRCommPkgSendMapElmt(commPkg,j) ] = padding + 4; // must be set to padding + numGhostLayers (note that the starting nodes are already distance 1 from their neighbors on the adjacent processor)
   }

   // Setup initial num_starting_nodes and starting_nodes (these are the starting nodes when searching for long distance neighbors)
   HYPRE_Int *num_starting_nodes = hypre_CTAlloc( HYPRE_Int, (padding+4)*num_sends, HYPRE_MEMORY_HOST );
   HYPRE_Int **starting_nodes = hypre_CTAlloc( HYPRE_Int*, (padding+4)*num_sends, HYPRE_MEMORY_HOST );
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
      finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
      searching_procs[i] = i;
      num_starting_nodes[i] = finish - start;
      starting_nodes[i] = hypre_CTAlloc( HYPRE_Int, num_starting_nodes[i], HYPRE_MEMORY_HOST );
      for (j = 0; j < num_starting_nodes[i]; j++)
      {
         starting_nodes[i][j] = hypre_ParCSRCommPkgSendMapElmt(commPkg, j + start );
      }
   }

   // Recursively find my own send nodes and communicate with neighbors to find off-processor long-range connections
   HYPRE_Int *num_request_nodes = hypre_CTAlloc(HYPRE_Int, (padding+4)*num_sends, HYPRE_MEMORY_HOST);
   HYPRE_Int **request_nodes = hypre_CTAlloc(HYPRE_Int*, (padding+4)*num_sends, HYPRE_MEMORY_HOST);

   // hypre_printf("Rank %d: first call of RecursivelyFindNeighborProcessors()\n", myid);

   RecursivelyFindNeighborProcessors(compGrid, A, add_flag, 
      num_starting_nodes, starting_nodes, 
      num_sends, searching_procs,
      num_request_nodes, request_nodes,
      &num_sends, send_procs, 
      num_sends); // Note that num_sends may change here

   // hypre_printf("Rank %d: Done with RecursivelyFindNeighborProcessors()\n", myid);


   // Use add_flag to generate relevant info for CompGridCommPkg
   cnt = 0;
   HYPRE_Int *send_map_starts = hypre_CTAlloc( HYPRE_Int, num_sends+1, HYPRE_MEMORY_HOST );
   for (i = 0; i < num_sends; i++)
   {
      send_map_starts[i] = cnt;
      for (j = 0; j < num_nodes; j++) if (add_flag[i][j] > 0) cnt++;
   }
   send_map_starts[num_sends] = cnt;
   HYPRE_Int *send_map_elmts = hypre_CTAlloc(HYPRE_Int, cnt, HYPRE_MEMORY_HOST);
   HYPRE_Int *ghost_marker = hypre_CTAlloc(HYPRE_Int, cnt, HYPRE_MEMORY_HOST);
   cnt = 0;
   for (i = 0; i < num_sends; i++) for (j = 0; j < num_nodes; j++) 
   {
      if (add_flag[i][j] > 0) 
      {
         send_map_elmts[cnt] = j;
         if (add_flag[i][j] > 4) ghost_marker[cnt] = 0;
         else ghost_marker[cnt] = 1;
         cnt++;
      }
   }
   send_procs = hypre_TReAlloc(send_procs, HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
   HYPRE_Int *recv_procs = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_sends; i++) recv_procs[i] = send_procs[i];

   hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level] = num_sends;
   hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level] = num_sends;
   hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level] = send_procs;
   hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level] = recv_procs;
   hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level] = send_map_starts;
   hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level] = send_map_elmts;
   hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level] = ghost_marker;






   // Debugging: dump info on the compGridCommPkg
   // char filename[256];
   // hypre_sprintf(filename, "outputs/CompGrids/initCompGridCommPkgRank%dLevel%d.txt",myid,level);
   // FILE *file = fopen(filename,"w");
   // num_sends = hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level];
   // fprintf(file, "num_sends = %d, send_procs = ", num_sends);
   // for (j = 0; j < num_sends; j++) fprintf(file, "%d ", hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][j]);
   // fprintf(file,"\n");
   // fprintf(file, "send_map_starts = ");
   // for (j = 0; j < num_sends+1; j++) fprintf(file, "%d ", hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][j]);
   // fprintf(file,"\n");
   // fprintf(file, "send_map_elmts = ");
   // for (j = 0; j < hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][num_sends]; j++) fprintf(file, "%d ", hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level][j]);
   // fprintf(file,"\n");
   // fclose(file);
   // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);





   // Clean up memory
   for (i = 0; i < num_sends; i++)
   {
      hypre_TFree(add_flag[i], HYPRE_MEMORY_HOST);
      hypre_TFree(starting_nodes[i], HYPRE_MEMORY_HOST);
      hypre_TFree(request_nodes[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);
   hypre_TFree(starting_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(searching_procs, HYPRE_MEMORY_HOST);
   hypre_TFree(request_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(num_starting_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(num_request_nodes, HYPRE_MEMORY_HOST);


   // hypre_printf("Rank %d: Leaving SetupNearestProcessorNeighbors()\n", myid);

   return 0;
}

HYPRE_Int
RecursivelyFindNeighborProcessors( hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, HYPRE_Int **add_flag,
   HYPRE_Int *num_starting_nodes, HYPRE_Int **starting_nodes,
   HYPRE_Int num_searching_procs, HYPRE_Int *searching_procs,
   HYPRE_Int *num_request_nodes, HYPRE_Int **request_nodes,
   HYPRE_Int *num_send_procs, HYPRE_Int *send_procs, 
   HYPRE_Int num_neighboring_procs )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int      i,j,k,cnt;

   // Update add_flag by recursively adding neighbors
   // hypre_printf("Rank %d: About to call RecursivelyFindNeighborNodes()\n", myid);
   if (myid == 1)
   {
      hypre_printf("num_searching_procs = %d, searching procs = ", num_searching_procs);
      for (i = 0; i < num_searching_procs; i++) hypre_printf("%d ", searching_procs[i]);
      hypre_printf("\n");
   }
   for (i = 0; i < num_searching_procs; i++)
   {
      HYPRE_Int proc_index = searching_procs[i];
      num_request_nodes[proc_index] = 0;
      if (!request_nodes[proc_index])
      {
         request_nodes[proc_index] = hypre_CTAlloc( HYPRE_Int, 2*hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)), HYPRE_MEMORY_HOST );
         if (myid == 1)
         {
            hypre_printf("num cols offd = %d, global indices = ", hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)));
            for (j = 0; j < hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)); j++) hypre_printf("%d ", hypre_ParCSRMatrixColMapOffd(A)[j]);
            hypre_printf("\n");
         }
         // hypre_printf("Rank %d: request_nodes[%d] allocated to size %d\n", myid, proc_index, 2*hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)));
      }
      if (myid == 1)
      {
         hypre_printf("num_starting_nodes[%d] = %d, starting_nodes = ", proc_index, num_starting_nodes[proc_index]);
         for (j = 0; j < num_starting_nodes[proc_index]; j++) hypre_printf("%d ", starting_nodes[proc_index][j]);
         hypre_printf("\n");
         hypre_printf("add_flag[%d] before finding neighbor nodes: ", proc_index);
         for (j = 0; j < hypre_ParCSRMatrixNumRows(A); j++) hypre_printf("%d ", add_flag[proc_index][j]);
         hypre_printf("\n");
      }
      for (j = 0; j < num_starting_nodes[proc_index]; j++)
      {
         RecursivelyFindNeighborNodes( starting_nodes[proc_index][j], add_flag[proc_index][ starting_nodes[proc_index][j] ] - 1, compGrid, add_flag[proc_index], request_nodes[proc_index], &(num_request_nodes[proc_index]) );
      }

      if (myid == 1)
      {
         hypre_printf("add_flag[%d] after finding neighbor nodes: ", proc_index);
         for (j = 0; j < hypre_ParCSRMatrixNumRows(A); j++) hypre_printf("%d ", add_flag[proc_index][j]);
         hypre_printf("\n");
      }

      num_starting_nodes[proc_index] = 0;
   }

   // Debugging:
   // hypre_printf("Rank %d: Done with RecursivelyFindNeighborNodes()\n", myid);
   if (myid == 1)
   {
      for (i = 0; i < num_searching_procs; i++)
      {
         HYPRE_Int proc_index = searching_procs[i];
         hypre_printf("num_request_nodes[%d] = %d, request nodes = ", proc_index, num_request_nodes[0]);
         for (i = 0; i < num_request_nodes[proc_index]; i++) hypre_printf("%d ", request_nodes[proc_index][2*i]);
         hypre_printf("\n");
      }
   }

   // Exchange message sizes
   HYPRE_Int send_size = 1;
   for (i = 0; i < num_searching_procs; i++)
   {
      HYPRE_Int proc_index = searching_procs[i];
      if (num_request_nodes[proc_index])
      {
         send_size += 2*num_request_nodes[proc_index] + 2;
      }
   }
   HYPRE_Int *recv_sizes = hypre_CTAlloc(HYPRE_Int, num_neighboring_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Request *requests = hypre_CTAlloc(hypre_MPI_Request, 4*num_neighboring_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Status *statuses = hypre_CTAlloc(hypre_MPI_Status, 4*num_neighboring_procs, HYPRE_MEMORY_HOST);
   HYPRE_Int request_cnt = 0;
   // hypre_printf("Rank %d: About post sends and recvs for message sizes, size of allocated requests is %d\n", myid, 4*num_neighboring_procs);
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Irecv(&(recv_sizes[i]), 1, HYPRE_MPI_INT, send_procs[i], 0, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      hypre_MPI_Isend(&send_size, 1, HYPRE_MPI_INT, send_procs[i], 0, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }

   // Wait on the recv sizes
   hypre_MPI_Waitall(2*num_neighboring_procs, requests, statuses);

   // hypre_printf("Rank %d: Done waiting on message sizes\n", myid);

   // Check whether we are done
   if (send_size == 1)
   {
      HYPRE_Int finished = 1;
      for (i = 0; i < num_neighboring_procs; i++)
      {
         if (recv_sizes[i] > 1)
         {
            finished = 0;
            break;
         }
      }
      if (finished) return 0;
   }

   // Allocate recv buffers
   HYPRE_Int **recv_buffers = hypre_CTAlloc(HYPRE_Int*, num_neighboring_procs, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_neighboring_procs; i++) recv_buffers[i] = hypre_CTAlloc(HYPRE_Int, recv_sizes[i], HYPRE_MEMORY_HOST);

   // Post the recvs
   // hypre_printf("Rank %d: About to post recvs, request_cnt = %d, num_neighboring_procs = %d\n", myid, request_cnt, num_neighboring_procs);
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Irecv(recv_buffers[i], recv_sizes[i], HYPRE_MPI_INT, send_procs[i], 1, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   // Setup the send buffer and post the sends
   HYPRE_Int *send_buffer = hypre_CTAlloc(HYPRE_Int*, send_size, HYPRE_MEMORY_HOST);
   cnt = 1;
   HYPRE_Int num_request_procs = 0;
   for (i = 0; i < num_searching_procs; i++)
   {
      HYPRE_Int proc_index = searching_procs[i];
      if (num_request_nodes[proc_index])
      {
         num_request_procs++;
         HYPRE_Int proc_index = searching_procs[i];
         send_buffer[cnt++] = send_procs[proc_index];
         send_buffer[cnt++] = num_request_nodes[proc_index];
         for (j = 0; j < num_request_nodes[proc_index]; j++)
         {
            send_buffer[cnt++] = request_nodes[proc_index][2*j];
            send_buffer[cnt++] = request_nodes[proc_index][2*j+1];
         }
      }
   }
   send_buffer[0] = num_request_procs;
   // hypre_printf("Rank %d: posting sends\n", myid);
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Isend(send_buffer, send_size, HYPRE_MPI_INT, send_procs[i], 1, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }

   // Wait 
   hypre_MPI_Waitall(2*num_neighboring_procs, &(requests[2*num_neighboring_procs]), &(statuses[2*num_neighboring_procs]));


   // Debugging:
   if (myid == 1)
   {
      for (i = 0; i < num_neighboring_procs; i++)
      {
         hypre_printf("recv_buffer[%d] = ",i);
         for (j = 0; j < recv_sizes[i]; j++)
         {
            hypre_printf("%d ", recv_buffers[i][j]);
         }
         hypre_printf("\n");
      }
   }


   // Unpack recieved messages and update add_flag where appropriate
   // hypre_printf("Rank %d: About to unpack recvs\n", myid);
   HYPRE_Int num_new_searching_procs = 0;
   HYPRE_Int max_num_new_searching_procs = num_neighboring_procs*num_neighboring_procs;
   HYPRE_Int *new_searching_procs = hypre_CTAlloc(HYPRE_Int, max_num_new_searching_procs, HYPRE_MEMORY_HOST); // !!! I'm guessing on how big to allocate this... !!!
   for (i = 0; i < num_neighboring_procs; i++)
   {
      cnt = 0;
      HYPRE_Int num_incoming_procs = recv_buffers[i][cnt++];
      if (myid == 1) hypre_printf("unpacking buffer %d\n", i);
      for (j = 0; j < num_incoming_procs; j++)
      {
         HYPRE_Int incoming_proc = recv_buffers[i][cnt++];
         HYPRE_Int num_incoming_nodes = recv_buffers[i][cnt++];
         // See whether we already have an add_flag array setup for this proc (and check to avoid info for our rank)
         if (incoming_proc != myid)
         {
            if (myid == 1) 
            {
               hypre_printf("incoming_proc = %d\n", incoming_proc);
               hypre_printf("send_procs = ");
               for (k = 0; k < (*num_send_procs); k++) hypre_printf("%d ", send_procs[k]);
               hypre_printf("\n");
            }

            HYPRE_Int local_proc_index = -1;
            HYPRE_Int new_procs_to_search = 0;
            // Look for an add_flag already set up for this processor
            for (k = 0; k < (*num_send_procs); k++) if (send_procs[k] == incoming_proc) local_proc_index = k;
            // If we didn't find one, set up a new add_flag and new starting_nodes
            if (local_proc_index < 0)
            {
               local_proc_index = (*num_send_procs);
               (*num_send_procs)++;
               send_procs[local_proc_index] = incoming_proc;
               add_flag[local_proc_index] = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRMatrixNumRows(A), HYPRE_MEMORY_HOST);
               starting_nodes[local_proc_index] = hypre_CTAlloc(HYPRE_Int, num_incoming_nodes, HYPRE_MEMORY_HOST);
               new_procs_to_search = 1;
            }
            if (myid == 1) hypre_printf("local_proc_index = %d\n", local_proc_index);
            for (k = 0; k < num_incoming_nodes; k++)
            {
               HYPRE_Int local_index = recv_buffers[i][cnt++] - hypre_ParCompGridGlobalIndices(compGrid)[0]; 
               HYPRE_Int incoming_flag = recv_buffers[i][cnt++];
               if (local_index >=0 && local_index < hypre_ParCSRMatrixNumRows(A))
               {
                  if (myid == 1) hypre_printf("local_index = %d\n", local_index);
                  if (incoming_flag > add_flag[local_proc_index][local_index])
                  {
                     add_flag[local_proc_index][local_index] = incoming_flag;
                     starting_nodes[local_proc_index][num_starting_nodes[ local_proc_index]++ ] = local_index;
                     new_procs_to_search = 1;
                  }
               }
            }
            if (new_procs_to_search)
            {
               if (num_new_searching_procs >= max_num_new_searching_procs)
               {
                  max_num_new_searching_procs *= 2;
                  new_searching_procs = hypre_TReAlloc(new_searching_procs, HYPRE_Int, max_num_new_searching_procs, HYPRE_MEMORY_HOST);
               }
               new_searching_procs[num_new_searching_procs++] = local_proc_index;
            }
         }
         else
         {
            if (myid == 1) hypre_printf("skipping add flag for my self\n");
            cnt += 2*num_incoming_nodes;
         }
      }
   }
  
   // Clean up memory
   for (i = 0; i < num_neighboring_procs; i++) hypre_TFree(recv_buffers[i], HYPRE_MEMORY_HOST);
   hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);

   // Recurse
   RecursivelyFindNeighborProcessors( compGrid, A, add_flag, 
      num_starting_nodes, starting_nodes, 
      num_new_searching_procs, new_searching_procs,
      num_request_nodes, request_nodes,
      num_send_procs, send_procs, 
      num_neighboring_procs );

   // Finish cleaning up memory
   hypre_TFree(new_searching_procs, HYPRE_MEMORY_HOST);
   // hypre_printf("Rank %d: Leaving RecursivelyFindNeighborProcessors()\n", myid);

   return 0;
}

HYPRE_Int
RecursivelyFindNeighborNodes(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, 
   HYPRE_Int *request_nodes, HYPRE_Int *num_request_nodes )
{
   HYPRE_Int         i,j,index,coarse_grid_index;
   hypre_ParCompMatrixRow *A_row = hypre_ParCompGridARows(compGrid)[node];

   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Look at neighbors
   for (i = 0; i < hypre_ParCompMatrixRowSize(A_row); i++)
   {
      // Get the index of the neighbor
      index = hypre_ParCompMatrixRowLocalIndices(A_row)[i];

      // If the neighbor info is available on this proc
      if (index >= 0)
      {
         // And if we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[index] < m)
         {
            add_flag[index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) RecursivelyFindNeighborNodes(index, m-1, compGrid, add_flag, request_nodes, num_request_nodes);
         }
      }
      // otherwise note this as a starting node to request from neighboring procs
      else
      {
         // Check whether we have already requested this node (!!! linear search !!!)
         HYPRE_Int global_index = hypre_ParCompMatrixRowGlobalIndices(A_row)[i];
         // if (myid == 1) hypre_printf("looking for global_index = %d\n", global_index);
         HYPRE_Int add_requeset = 1;
         for (j = 0; j < (*num_request_nodes); j++)
         {
            // hypre_printf("Rank %d: reading request_nodes[%d]\n", myid, 2*j);
            if (request_nodes[2*j] == global_index) add_requeset = 0;
         }
         if (add_requeset)
         {
            // hypre_printf("Rank %d: writing to request_nodes[%d,%d]\n",myid,2*(*num_request_nodes),2*(*num_request_nodes)+1);
            request_nodes[2*(*num_request_nodes)] = global_index;
            request_nodes[2*(*num_request_nodes)+1] = m-1; //double check!!!
            (*num_request_nodes)++;
            if (myid == 0) 
            {
               // hypre_printf("added node, num_request_nodes = %d, request_nodes = ", *num_request_nodes);
               // for (j = 0; j < 2*(*num_request_nodes); j++) hypre_printf("%d ", request_nodes[j]);
               // hypre_printf("\n");
            }
         }
      }
   }

   return 0;
}

HYPRE_Complex*
PackSendBufferNew( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, HYPRE_Int *send_flag_buffer_size, 
   HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int padding )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_printf("Rank %d: inside PackSendBufferNew() for processor %d\n", myid, processor);






   // FILE *file;
   // char filename[256];
   // hypre_sprintf(filename,"outputs/count_send_buffer_level%d_rank_%d_to_%d.txt",current_level,myid,processor);
   // file = fopen(filename,"w");







   HYPRE_Int            level,i,j,cnt,row_length,send_elmt,coarse_grid_index;
   HYPRE_Int            need_coarse_info, nodes_to_add = 0;
   HYPRE_Int            **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            num_psi_levels = 1;
   HYPRE_Int            *num_send_nodes = hypre_CTAlloc( HYPRE_Int, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            **send_node_indices = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            **ghost_marker = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );


   // Get where to look in commPkgSendMapElmts
   HYPRE_Int            start = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][processor];
   HYPRE_Int            finish = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][processor+1];

   // initialize send map buffer size
   (*send_flag_buffer_size) = num_levels - current_level;

   // see whether we need coarse info and allcoate the add_flag array on next level if appropriate
   if (current_level != num_levels-1) need_coarse_info = 1;
   else need_coarse_info = 0;
   if (need_coarse_info) add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[current_level+1]), HYPRE_MEMORY_HOST );

   // Mark the nodes to send (including Psi_c grid plus ghost nodes)

   // Start by adding the nodes listed by the compGridCommPkg on this level and their coarse grid counterparts if applicable
   // Note that the compGridCommPkg is set up to list all nodes within the padding plus 4 ghost layers
   (*send_flag_buffer_size) += finish - start;
   if (need_coarse_info)
   {
      for (i = start; i < finish; i++)
      {
         // flag nodes that are repeated on the next coarse grid
         send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];
         if (!hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[current_level][i])
         {
            coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid[current_level])[send_elmt];
            if ( coarse_grid_index != -1 ) 
            {
               add_flag[current_level+1][ coarse_grid_index ] = padding+1;
               nodes_to_add = 1;
            }
         }
      }
   }
   // Count up the buffer size for the starting nodes
   num_send_nodes[current_level] = finish - start;
   send_node_indices[current_level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level], HYPRE_MEMORY_HOST );
   ghost_marker[current_level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level], HYPRE_MEMORY_HOST );
   (*buffer_size) += 2;
   if (need_coarse_info) (*buffer_size) += 4*num_send_nodes[current_level];
   else (*buffer_size) += 2*num_send_nodes[current_level];
   // fprintf(file, "after level %d nodes, buffer_size = %d\n",current_level, *buffer_size);
   for (i = start; i < finish; i++)
   {
      send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];
      send_node_indices[current_level][i - start] = send_elmt;
      ghost_marker[current_level][i - start] = hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[current_level][i];
      (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(compGrid[current_level])[send_elmt]) + 1;
      if (need_coarse_info) (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridPRows(compGrid[current_level])[send_elmt]) + 1;
   }
   // fprintf(file, "after level %d matrix rows, buffer_size = %d\n",current_level, *buffer_size);

   // Now build out the psi_c composite grid (along with required ghost nodes) on coarser levels
   for (level = current_level + 1; level < num_levels; level++)
   {
      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         num_psi_levels++;
         (*buffer_size)++;
         nodes_to_add = 0;

         // see whether we need coarse info on this level
         if (level != num_levels-1) need_coarse_info = 1;
         else need_coarse_info = 0;

         // if we need coarse info, allocate space for the add flag on the next level
         if (need_coarse_info) add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]), HYPRE_MEMORY_HOST );

         // Expand by the padding on this level and add coarse grid counterparts if applicable
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i])
            {
               // Recursively add the region of padding (flagging coarse nodes on the next level if applicable)
               if (need_coarse_info) RecursivelyBuildPsiComposite(i, padding, compGrid[level], add_flag[level], add_flag[level+1], need_coarse_info, &nodes_to_add, padding);
               else RecursivelyBuildPsiComposite(i, padding, compGrid[level], add_flag[level], NULL, need_coarse_info, &nodes_to_add, padding);
            }
         }

         // Expand by the number of ghost layers (4)
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 1) add_flag[level][i] = 6;
            else if (add_flag[level][i] == 1) add_flag[level][i] = 5;
         }
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            // Recursively add the region of ghost nodes (do not add any coarse nodes underneath)
            RecursivelyBuildPsiComposite(i, 4, compGrid[level], add_flag[level], NULL, 0, NULL, 0);
         }

         // Count up the buffer size
         // fprintf(file,"add_flag level %d:\n", level);
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 0)
            {
               num_send_nodes[level]++;
               (*send_flag_buffer_size)++;
               if (need_coarse_info) (*buffer_size) += 4;
               else (*buffer_size) += 2;
               // (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(compGrid[level])[i]) + 1;
               // if (need_coarse_info) (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridPRows(compGrid[level])[i]) + 1;
            }
         }
         // fprintf(file, "after level %d nodes, buffer_size = %d\n", level, *buffer_size);
         // Count up the buffer size
         // fprintf(file, "level %d matrix rows counted:\n", level);
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 0)
            {
               // fprintf(file, "row %d, size %d\n", i, hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(compGrid[level])[i]));
               (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(compGrid[level])[i]) + 1;
               if (need_coarse_info) (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridPRows(compGrid[level])[i]) + 1;
            }
         }
         // fprintf(file, "after level %d matrix rows, buffer_size = %d\n", level, *buffer_size);

         // Save the indices so I don't have to keep looping over all nodes in compGrid when I pack the buffer
         send_node_indices[level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[level], HYPRE_MEMORY_HOST );
         ghost_marker[level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[level], HYPRE_MEMORY_HOST );
         // fprintf(file,"level %d send_node_indices:\n");
         cnt =  0;
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 0) 
            {
               // fprintf(file, "%d\n", i);
               send_node_indices[level][cnt] = i;
               if (add_flag[level][i] < 5) ghost_marker[level][cnt] = 1;
               cnt++;
            }
         }
      }
      else break;
   }

   hypre_printf("Rank %d: done finding psi_c for proc %d\n", myid, processor);

   // Allocate the buffer
   HYPRE_Complex *send_buffer = hypre_CTAlloc(HYPRE_Complex, (*buffer_size), HYPRE_MEMORY_HOST);


   // FILE *file;
   // char filename[256];
   // hypre_sprintf(filename,"outputs/send_nodes_level%d_rank_%d_to_%d.txt",current_level,myid,processor);
   // file = fopen(filename,"w");
   // fprintf(file,"buffer_size = %d\n", *buffer_size);
   // fprintf(file,"num_send_nodes =\n");
   // for (level = current_level; level < current_level + num_psi_levels; level++) fprintf(file,"%d\n", num_send_nodes[level]);
   // fclose(file);






   hypre_printf("Rank %d: done allocating buffer of size %d for proc %d\n", myid, (*buffer_size), processor);

   // Pack the buffer
   cnt = 0;
   send_buffer[cnt++] = (HYPRE_Complex) num_psi_levels;
   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      // store the number of nodes on this level
      send_buffer[cnt++] = (HYPRE_Complex) num_send_nodes[level];

      // copy all global indices
      for (i = 0; i < num_send_nodes[level]; i++)
      {
         send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompGridGlobalIndices(compGrid[level])[ send_node_indices[level][i] ];
      }
      // copy all residual values
      for (i = 0; i < num_send_nodes[level]; i++)
      {
         send_buffer[cnt++] = hypre_ParCompGridF(compGrid[level])[ send_node_indices[level][i] ];
      }

      // if not on last level, copy these
      if (level != num_levels-1)
      {
         // copy ghost marker
         for (i = 0; i < num_send_nodes[level]; i++)
         {
            send_buffer[cnt++] = (HYPRE_Complex) ghost_marker[level][i];
         }
         for (i = 0; i < num_send_nodes[level]; i++)
         {
            send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ send_node_indices[level][i] ];
         }
      }
      // fprintf(file, "after level %d nodes, cnt = %d\n", level, cnt);
      // now loop over matrix rows
      // fprintf(file, "level %d matrix rows packed:\n", level);
      for (i = 0; i < num_send_nodes[level]; i++)
      {
         // store the row length for matrix A
         if (cnt > (*buffer_size)) hypre_printf("\n\n\nHEY! Rank %d, current_level %d, cnt = %d exceeded buffer size = %d!\n\n\n\n", myid, current_level, cnt, *buffer_size);
         row_length = hypre_ParCompMatrixRowSize( hypre_ParCompGridARows( compGrid[level] )[ send_node_indices[level][i] ] );
         send_buffer[cnt++] = (HYPRE_Complex) row_length;
         // fprintf(file, "row %d, size %d\n", send_node_indices[level][i], row_length);

         // copy matrix entries for matrix A
         for (j = 0; j < row_length; j++)
         {
            send_buffer[cnt++] = hypre_ParCompMatrixRowData( hypre_ParCompGridARows( compGrid[level] )[ send_node_indices[level][i] ] )[j];
         }
         // copy global indices for matrix A
         for (j = 0; j < row_length; j++)
         {
            send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( compGrid[level] )[ send_node_indices[level][i] ] )[j];
         }

         if (level != num_levels-1)
            {
            // store the row length for matrix P
            row_length = hypre_ParCompMatrixRowSize( hypre_ParCompGridPRows( compGrid[level] )[ send_node_indices[level][i] ] );
            send_buffer[cnt++] = (HYPRE_Complex) row_length;

            // copy matrix entries for matrix P
            for (j = 0; j < row_length; j++)
            {
               send_buffer[cnt++] = hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( compGrid[level] )[ send_node_indices[level][i] ] )[j];
            }
            // copy global indices for matrix P
            for (j = 0; j < row_length; j++)
            {
               send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( compGrid[level] )[ send_node_indices[level][i] ] )[j];
            }
         }
      }
      // fprintf(file, "after level %d matrix rows, cnt = %d\n", level, cnt);
   }

   // fclose(file);
   hypre_printf("Rank %d: done packing buffer for proc %d\n", myid, processor);

   // Clean up memory
   for (level = 0; level < num_levels; level++)
   {
      if (add_flag[level]) hypre_TFree(add_flag[level], HYPRE_MEMORY_HOST);
      if (send_node_indices[level]) hypre_TFree(send_node_indices[level], HYPRE_MEMORY_HOST);
      if (ghost_marker[level]) hypre_TFree(ghost_marker[level], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);
   hypre_TFree(send_node_indices, HYPRE_MEMORY_HOST);
   hypre_TFree(ghost_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(num_send_nodes, HYPRE_MEMORY_HOST);

   // Return the send buffer
   hypre_printf("Rank %d: done cleaning up memory for proc %d\n", myid, processor);
   return send_buffer;
}

HYPRE_Int
RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int *add_flag_coarse, 
   HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding)
{
   HYPRE_Int         i,index,coarse_grid_index;
   hypre_ParCompMatrixRow *A_row = hypre_ParCompGridARows(compGrid)[node];

   // Look at neighbors
   for (i = 0; i < hypre_ParCompMatrixRowSize(A_row); i++)
   {
      // Get the index of the neighbor
      index = hypre_ParCompMatrixRowLocalIndices(A_row)[i];

      // If the neighbor info is available on this proc
      if (index > 0)
      {
         // And if we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[index] < m)
         {
            add_flag[index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) RecursivelyBuildPsiComposite(index, m-1, compGrid, add_flag, add_flag_coarse, need_coarse_info, nodes_to_add, padding);
         }
      }
   }

   // Flag this node on the next coarsest level if applicable
   if (need_coarse_info)
   {
      coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid)[node];
      if ( coarse_grid_index != -1 ) 
      {
         // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
         add_flag_coarse[ coarse_grid_index ] = padding+1;
         *nodes_to_add = 1;
      }
   }

   return 0;
}

HYPRE_Int
GeneratePsiComposite( hypre_ParCompGrid **psiComposite, hypre_ParCompGrid **compGrid, hypre_ParCSRCommPkg *commPkg, HYPRE_Int *send_flag_buffer_size, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels )
{
   HYPRE_Int                  level,i,j,cnt = 0;
   HYPRE_Int                  send_elmt;
   HYPRE_Int                  row_size;
   HYPRE_Int                  nodes_to_add = 0, coarse_grid_index, need_coarse_info;
   HYPRE_Int                  **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   hypre_ParCompMatrixRow     *row;
   HYPRE_Int                  num_psi_levels;

   // Get where to look in commPkgSendMapElmts
   HYPRE_Int            start = hypre_ParCSRCommPkgSendMapStart(commPkg, processor);
   HYPRE_Int            finish = hypre_ParCSRCommPkgSendMapStart(commPkg, processor+1);

   // initialize send map buffer size
   *send_flag_buffer_size = num_levels - current_level;

   // see whether we need coarse info
   if (current_level != num_levels-1) need_coarse_info = 1;
   else need_coarse_info = 0;

   // create psiComposite on this level and allocate space
   psiComposite[current_level] = hypre_ParCompGridCreate(0);
   hypre_ParCompGridSetSize(psiComposite[current_level], finish - start, need_coarse_info);
   if (need_coarse_info) add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[current_level+1]), HYPRE_MEMORY_HOST );

   // copy correct data into psiComposite from compGrid using sendMapElmts from commPkg
   // !!! CHECK ORDERING OF COMP GRIDS VS ORDERING OF SENDMAPELMTS... SHOULD BE OK I THINK... BUT CHECK !!!
   for (i = start; i < finish; i++)
   {
      // see whether we need coarse info
      if (current_level != num_levels-1) need_coarse_info = 1;
      else need_coarse_info = 0;

      // get index of element to send
      send_elmt = hypre_ParCSRCommPkgSendMapElmt(commPkg, i);

      // copy data and global indices into psiComposite
      hypre_ParCompGridCopyNode( compGrid[current_level], psiComposite[current_level], send_elmt, cnt );

      // count send flag buffer size
      (*send_flag_buffer_size)++;

      // flag nodes that will be on the next coarse grid
      if (need_coarse_info)
      {
         coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(psiComposite[current_level])[cnt];
         if ( coarse_grid_index != -1 )
         {
            // look at the matrix row associated with the coarse node at coarse_grid_index
            row = hypre_ParCompGridARows(compGrid[current_level+1])[coarse_grid_index];
            row_size = hypre_ParCompMatrixRowSize(row);
            // loop over neighbors and flag them to add to next coarse psiComposite grid
            for (j = 0; j < row_size; j++)
            {
               if ( hypre_ParCompMatrixRowLocalIndices(row)[j] != -1 )
               {
                  if ( ! add_flag[current_level+1][ hypre_ParCompMatrixRowLocalIndices(row)[j] ] )
                  {
                     add_flag[current_level+1][ hypre_ParCompMatrixRowLocalIndices(row)[j] ] = 1;
                     nodes_to_add++;
                  }
               }
            }
         }
      }
      cnt++;
   }

   // get composite grid generated by psi
   num_psi_levels = 1;
   for (level = current_level + 1; level < num_levels; level++)
   {
      // see whether we need coarse info on this level
      if (level != num_levels-1) need_coarse_info = 1;
      else need_coarse_info = 0;

      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         // allocate space for psiComposite on this level
         psiComposite[level] = hypre_ParCompGridCreate(0);
         hypre_ParCompGridSetSize(psiComposite[level], nodes_to_add, need_coarse_info);

         // if we need coarse info, allocate space for the add flag on the next level
         if (need_coarse_info)
         {
            add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]), HYPRE_MEMORY_HOST );
            nodes_to_add = 0;
         }

         // loop over nodes and add those flagged by add nodes
         cnt = 0;
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i])
            {
               hypre_ParCompGridCopyNode( compGrid[level], psiComposite[level], i, cnt);

               // count send flag buffer size
               (*send_flag_buffer_size)++;

               if (need_coarse_info)
               {
                  coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(psiComposite[level])[cnt];
                  if ( coarse_grid_index != -1 )
                  {
                     // look at the matrix row associated with the coarse node at coarse_grid_index
                     row = hypre_ParCompGridARows(compGrid[level+1])[coarse_grid_index];
                     row_size = hypre_ParCompMatrixRowSize(row);
                     // loop over neighbors and flag them to add to next coarse psiComposite grid
                     for (j = 0; j < row_size; j++)
                     {
                        if ( hypre_ParCompMatrixRowLocalIndices(row)[j] != -1 )
                        {
                           if ( ! add_flag[level+1][ hypre_ParCompMatrixRowLocalIndices(row)[j] ] )
                           {
                              add_flag[level+1][ hypre_ParCompMatrixRowLocalIndices(row)[j] ] = 1;
                              nodes_to_add++;
                           }
                        }
                     }
                  }                  
               }
               // increment counter for psiComposite[level] index
               cnt++;
            }
         }
      
         // increment num_psi_levels
         num_psi_levels++;
      }
      else break;
   }

   // Cleanup memory
   for (level = 0; level < num_levels; level++)
   {
      hypre_TFree(add_flag[level], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);

   return num_psi_levels;

}

HYPRE_Int
GetBufferSize( hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels )
{
   HYPRE_Int            level, i;

   // get size of buffer
   HYPRE_Int            buffer_size = 1;

   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      buffer_size += 3*hypre_ParCompGridNumNodes(psiComposite[level]) + 1;
      for (i = 0; i < hypre_ParCompGridNumNodes(psiComposite[level]); i++)
      {
         buffer_size += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(psiComposite[level])[i]) + 1;
         if (level != num_levels-1) buffer_size += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridPRows(psiComposite[level])[i]) + 1;
      }
      if (level == num_levels-1) buffer_size -= hypre_ParCompGridNumNodes(psiComposite[level]);
   }
   return buffer_size;
}

HYPRE_Complex*
PackSendBuffer( hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels, HYPRE_Int buffer_size )
{
   HYPRE_Int         level,i,j;
   HYPRE_Int         num_nodes, row_length;


 

   // allocate space for buffer
   HYPRE_Complex     *send_buffer = hypre_CTAlloc(HYPRE_Complex, buffer_size, HYPRE_MEMORY_HOST);

   // Initialize the counter and store num_psi_levels as first entry in buffer
   HYPRE_Int cnt = 0;
   send_buffer[cnt++] = (HYPRE_Complex) num_psi_levels;

   // loop over psi levels
   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      // store the number of nodes on this level
      num_nodes = hypre_ParCompGridNumNodes(psiComposite[level]);
      send_buffer[cnt++] = (HYPRE_Complex) num_nodes;

      // copy all residual values
      for (i = 0; i < num_nodes; i++)
      {
         send_buffer[cnt++] = hypre_ParCompGridF(psiComposite[level])[i];
      }
      // copy all global indices
      for (i = 0; i < num_nodes; i++)
      {
         send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompGridGlobalIndices(psiComposite[level])[i];
      }
      // if there are coarse indices (i.e. not on last level), copy these
      if (hypre_ParCompGridCoarseGlobalIndices(psiComposite[level]))
      {
         for (i = 0; i < num_nodes; i++)
         {
            send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompGridCoarseGlobalIndices(psiComposite[level])[i];
         }
      }
      // now loop over matrix rows
      for (i = 0; i < num_nodes; i++)
      {
         // store the row length for matrix A
         row_length = hypre_ParCompMatrixRowSize( hypre_ParCompGridARows( psiComposite[level] )[i] );
         send_buffer[cnt++] = (HYPRE_Complex) row_length;

         // copy matrix entries for matrix A
         for (j = 0; j < row_length; j++)
         {
            send_buffer[cnt++] = hypre_ParCompMatrixRowData( hypre_ParCompGridARows( psiComposite[level] )[i] )[j];
         }
         // copy global indices for matrix A
         for (j = 0; j < row_length; j++)
         {
            send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( psiComposite[level] )[i] )[j];
         }

         if (hypre_ParCompGridPRows(psiComposite[level]))
            {
            // store the row length for matrix P
            row_length = hypre_ParCompMatrixRowSize( hypre_ParCompGridPRows( psiComposite[level] )[i] );
            send_buffer[cnt++] = (HYPRE_Complex) row_length;

            // copy matrix entries for matrix P
            for (j = 0; j < row_length; j++)
            {
               send_buffer[cnt++] = hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( psiComposite[level] )[i] )[j];
            }
            // copy global indices for matrix P
            for (j = 0; j < row_length; j++)
            {
               send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( psiComposite[level] )[i] )[j];
            }
         }
      }
   }

   return send_buffer;
}

HYPRE_Int
UnpackRecvBufferNew( HYPRE_Complex *recv_buffer, hypre_ParCompGrid **compGrid, HYPRE_Int **recv_map_send, 
      HYPRE_Int *recv_map_size, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, 
      HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index, HYPRE_Int *num_added_nodes, HYPRE_Int *num_incoming_nodes )
{

   HYPRE_Int            level, i, j;
   HYPRE_Int            num_psi_levels, num_nodes, row_size, offset, level_start, global_index, add_node_cnt;
   HYPRE_Int            *add_flag;

   // initialize the counter
   HYPRE_Int            cnt = 0;

   // get the number of levels received
   num_psi_levels = (HYPRE_Int) recv_buffer[cnt++];

   // Init the recv_map_send_buffer_size !!! I think this can just be set a priori instead of counting it up in this function... !!!
   *recv_map_send_buffer_size = num_levels - current_level;

   // On the first level, all info will be new, so copy in all nodes
   num_nodes = (HYPRE_Int) recv_buffer[cnt++];
   num_incoming_nodes[current_level] = num_nodes;
   offset = hypre_ParCompGridNumNodes(compGrid[current_level]);
   if (num_nodes + offset > hypre_ParCompGridMemSize(compGrid[current_level])) 
      hypre_ParCompGridResize(compGrid[current_level], num_nodes + offset); // !!! Is there a better way to manage memory? !!!
   for (i = 0; i < num_nodes; i++)
   {
      hypre_ParCompGridGlobalIndices(compGrid[current_level])[i+offset] = (HYPRE_Int) recv_buffer[cnt++];
   }
   for (i = 0; i < num_nodes; i++)
   {
      hypre_ParCompGridF(compGrid[current_level])[i+offset] = recv_buffer[cnt++];
   }
   recv_map_send[current_level] = hypre_CTAlloc( HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST );
   for (i = 0; i < num_nodes; i++)
   {      
      recv_map_send[current_level][i] = i+offset;
      recv_map_size[current_level]++;
      (*recv_map_send_buffer_size)++;
   }
   if (current_level != num_levels-1)
   {
      for (i = 0; i < num_nodes; i++)
      {
         hypre_ParCompGridGhostMarker(compGrid[current_level])[i+offset] = (HYPRE_Int) recv_buffer[cnt++];
      }
      for (i = 0; i < num_nodes; i++)
      {
         hypre_ParCompGridCoarseGlobalIndices(compGrid[current_level])[i+offset] = (HYPRE_Int) recv_buffer[cnt++];
      }
   }
   for (i = 0; i < num_nodes; i++)
   {
      // get the row length of matrix A
      row_size = (HYPRE_Int) recv_buffer[cnt++];
      // Create row and allocate space
      hypre_ParCompGridARows(compGrid[current_level])[i+offset] = hypre_ParCompMatrixRowCreate();
      hypre_ParCompMatrixRowSize( hypre_ParCompGridARows( compGrid[current_level] )[i+offset] ) = row_size;
      hypre_ParCompMatrixRowData( hypre_ParCompGridARows( compGrid[current_level] )[i+offset] ) = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
      hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( compGrid[current_level] )[i+offset] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
      hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridARows( compGrid[current_level] )[i+offset] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);

      // copy matrix entries of matrix A
      for (j = 0; j < row_size; j++)
      {
         hypre_ParCompMatrixRowData( hypre_ParCompGridARows( compGrid[current_level] )[i+offset] )[j] = recv_buffer[cnt++];
      }
      // copy global indices of matrix A
      for (j = 0; j < row_size; j++)
      {
         hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( compGrid[current_level] )[i+offset] )[j] = (HYPRE_Int) recv_buffer[cnt++];
      }

      if (current_level != num_levels-1)
      {
         // get the row length of matrix P
         row_size = (HYPRE_Int) recv_buffer[cnt++];
         // Create row and allocate space
         hypre_ParCompGridPRows(compGrid[current_level])[i+offset] = hypre_ParCompMatrixRowCreate();
         hypre_ParCompMatrixRowSize( hypre_ParCompGridPRows( compGrid[current_level] )[i+offset] ) = row_size;
         hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( compGrid[current_level] )[i+offset] ) = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
         hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( compGrid[current_level] )[i+offset] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
         hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridPRows( compGrid[current_level] )[i+offset] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);

         // copy matrix entries of matrix P
         for (j = 0; j < row_size; j++)
         {
            hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( compGrid[current_level] )[i+offset] )[j] = recv_buffer[cnt++];
         }
         // copy global indices of matrix P
         for (j = 0; j < row_size; j++)
         {
            hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( compGrid[current_level] )[i+offset] )[j] = (HYPRE_Int) recv_buffer[cnt++];
         }
      }
   }
   hypre_ParCompGridNumNodes(compGrid[current_level]) = num_nodes + offset;
   num_added_nodes[current_level] += num_nodes;

   // loop over coarser psi levels
   for (level = current_level + 1; level < current_level + num_psi_levels; level++)
   {
      // get the number of nodes on this level
      num_nodes = (HYPRE_Int) recv_buffer[cnt++];
      num_incoming_nodes[level] = num_nodes;
      level_start = cnt;

      // loop over incoming nodes and check for redundant info
      add_flag = hypre_CTAlloc( HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
      add_node_cnt = 0;
      for (i = 0; i < num_nodes; i++)
      {
         global_index = recv_buffer[cnt++];
         if (global_index < proc_first_index[level] || global_index > proc_last_index[level])
         {
            add_flag[i] = 1;
            // !!! Backward linear search: needs improvement !!!
            for (j = hypre_ParCompGridNumNodes(compGrid[level]) - 1; j >= hypre_ParCompGridNumOwnedNodes(compGrid[level]); j--)
            {
               if (hypre_ParCompGridGlobalIndices(compGrid[level])[j] == global_index)
               {
                  add_flag[i] = 0;
                  // Check if the incoming node is a real node. If so, update the existing dof to be a real node.
                  if (level != num_levels-1) if (!((HYPRE_Int) recv_buffer[cnt + 2*num_nodes])) hypre_ParCompGridGhostMarker(compGrid[level])[j] = 0;
               }
            }
            if (add_flag[i]) add_node_cnt++;
         }
      }
      num_added_nodes[level] += add_node_cnt;

      // if necessary, reallocate more space for compGrid
      offset = hypre_ParCompGridNumNodes(compGrid[level]);
      if (add_node_cnt + offset > hypre_ParCompGridMemSize(compGrid[level])) 
         hypre_ParCompGridResize(compGrid[level], add_node_cnt + offset); // !!! Is there a better way to manage memory? !!!

      // add non-redundant nodes to the compGrid
      cnt = level_start;
      add_node_cnt = 0;
      for (i = 0; i < num_nodes; i++)
      {
         if (add_flag[i]) hypre_ParCompGridGlobalIndices(compGrid[level])[offset + add_node_cnt++] = (HYPRE_Int) recv_buffer[cnt];
         cnt++;
      }
      add_node_cnt = 0;
      for (i = 0; i < num_nodes; i++)
      {
         if (add_flag[i]) hypre_ParCompGridF(compGrid[level])[offset + add_node_cnt++] = recv_buffer[cnt];
         cnt++;
      }
      add_node_cnt = 0;
      recv_map_send[level] = hypre_CTAlloc( HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST );
      for (i = 0; i < num_nodes; i++)
      {
         if (add_flag[i])
         {
            recv_map_send[level][i] = offset + add_node_cnt++;
            recv_map_size[level]++;
            (*recv_map_send_buffer_size)++;
         }
         else
         {
            // flag node as repeated info which doesn't need to be sent later
            recv_map_send[level][i] = -1;
            (*recv_map_send_buffer_size)++;
         }
      }
      if (level != num_levels-1)
      {
         add_node_cnt = 0;
         for (i = 0; i < num_nodes; i++)
         {
            if (add_flag[i]) hypre_ParCompGridGhostMarker(compGrid[level])[offset + add_node_cnt++] = (HYPRE_Int) recv_buffer[cnt];
            cnt++;
         }
         add_node_cnt = 0;
         for (i = 0; i < num_nodes; i++)
         {
            if (add_flag[i]) hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[offset + add_node_cnt++] = (HYPRE_Int) recv_buffer[cnt];
            cnt++;
         }
      }
      add_node_cnt = 0;
      for (i = 0; i < num_nodes; i++)
      {
         if (add_flag[i])
         {
            // get the row length of matrix A
            row_size = (HYPRE_Int) recv_buffer[cnt++];

            // Create row and allocate space
            hypre_ParCompGridARows(compGrid[level])[offset + add_node_cnt] = hypre_ParCompMatrixRowCreate();
            hypre_ParCompMatrixRowSize( hypre_ParCompGridARows( compGrid[level] )[offset + add_node_cnt] ) = row_size;
            hypre_ParCompMatrixRowData( hypre_ParCompGridARows( compGrid[level] )[offset + add_node_cnt] ) = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
            hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( compGrid[level] )[offset + add_node_cnt] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
            hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridARows( compGrid[level] )[offset + add_node_cnt] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);

            // copy matrix entries of matrix A
            for (j = 0; j < row_size; j++)
            {
               hypre_ParCompMatrixRowData( hypre_ParCompGridARows( compGrid[level] )[offset + add_node_cnt] )[j] = recv_buffer[cnt++];
            }
            // copy global indices of matrix A
            for (j = 0; j < row_size; j++)
            {
               hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( compGrid[level] )[offset + add_node_cnt] )[j] = (HYPRE_Int) recv_buffer[cnt++];
            }

            if (level != num_levels-1)
            {
               // get the row length of matrix P
               row_size = (HYPRE_Int) recv_buffer[cnt++];
               // Create row and allocate space
               hypre_ParCompGridPRows(compGrid[level])[offset + add_node_cnt] = hypre_ParCompMatrixRowCreate();
               hypre_ParCompMatrixRowSize( hypre_ParCompGridPRows( compGrid[level] )[offset + add_node_cnt] ) = row_size;
               hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( compGrid[level] )[offset + add_node_cnt] ) = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
               hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( compGrid[level] )[offset + add_node_cnt] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
               hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridPRows( compGrid[level] )[offset + add_node_cnt] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);

               // copy matrix entries of matrix P
               for (j = 0; j < row_size; j++)
               {
                  hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( compGrid[level] )[i] )[j] = recv_buffer[cnt++];
               }
               // copy global indices of matrix P
               for (j = 0; j < row_size; j++)
               {
                  hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( compGrid[level] )[i] )[j] = (HYPRE_Int) recv_buffer[cnt++];
               }
            }
            add_node_cnt++;
         }
         else
         {
            row_size = (HYPRE_Int) recv_buffer[cnt++];
            cnt += 2*row_size;
            if (level != num_levels-1)
            {
               row_size = (HYPRE_Int) recv_buffer[cnt++];
               cnt += 2*row_size;
            }
         }
      }
      hypre_ParCompGridNumNodes(compGrid[level]) = offset + add_node_cnt;

      // free up add flag
      hypre_TFree(add_flag, HYPRE_MEMORY_HOST);
   }




   return 0;
}

HYPRE_Int
UnpackRecvBuffer( HYPRE_Complex *recv_buffer, hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels )
{
   HYPRE_Int            level, i, j;
   HYPRE_Int            num_psi_levels, num_nodes, row_size, need_coarse_info;

   // initialize the counter
   HYPRE_Int            cnt = 0;

   // get the number of levels received
   num_psi_levels = (HYPRE_Int) recv_buffer[cnt++];

   // loop over psi levels
   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      // see whether we need coarse info
      if (level != num_levels-1) need_coarse_info = 1;
      else need_coarse_info = 0;

      // create psiComposite on this level
      psiComposite[level] = hypre_ParCompGridCreate(0);

      // get the number of nodes on this level and allocate space in psiComposite
      num_nodes = (HYPRE_Int) recv_buffer[cnt++];
      hypre_ParCompGridSetSize(psiComposite[level], num_nodes, need_coarse_info);

      // copy all residual values
      for (i = 0; i < num_nodes; i++)
      {
         hypre_ParCompGridF(psiComposite[level])[i] = recv_buffer[cnt++];
      }
      // copy all global indices
      for (i = 0; i < num_nodes; i++)
      {
         hypre_ParCompGridGlobalIndices(psiComposite[level])[i] = (HYPRE_Int) recv_buffer[cnt++];
      }
      // if not on last level, get coarse indices
      if (level != num_levels-1)
      {
         for (i = 0; i < num_nodes; i++)
         {
            hypre_ParCompGridCoarseGlobalIndices(psiComposite[level])[i] = (HYPRE_Int) recv_buffer[cnt++];
         }
      }
      // now loop over matrix rows
      for (i = 0; i < num_nodes; i++)
      {
         // get the row length of matrix A
         row_size = (HYPRE_Int) recv_buffer[cnt++];
         // Create row and allocate space
         hypre_ParCompGridARows(psiComposite[level])[i] = hypre_ParCompMatrixRowCreate();
         hypre_ParCompMatrixRowSize( hypre_ParCompGridARows( psiComposite[level] )[i] ) = row_size;
         hypre_ParCompMatrixRowData( hypre_ParCompGridARows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
         hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
         hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridARows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);

         // copy matrix entries of matrix A
         for (j = 0; j < row_size; j++)
         {
            hypre_ParCompMatrixRowData( hypre_ParCompGridARows( psiComposite[level] )[i] )[j] = recv_buffer[cnt++];
         }
         // copy global indices of matrix A
         for (j = 0; j < row_size; j++)
         {
            hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( psiComposite[level] )[i] )[j] = (HYPRE_Int) recv_buffer[cnt++];
         }

         if (level != num_levels-1)
         {
            // get the row length of matrix P
            row_size = (HYPRE_Int) recv_buffer[cnt++];
            // Create row and allocate space
            hypre_ParCompGridPRows(psiComposite[level])[i] = hypre_ParCompMatrixRowCreate();
            hypre_ParCompMatrixRowSize( hypre_ParCompGridPRows( psiComposite[level] )[i] ) = row_size;
            hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
            hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
            hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridPRows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);

            // copy matrix entries of matrix P
            for (j = 0; j < row_size; j++)
            {
               hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( psiComposite[level] )[i] )[j] = recv_buffer[cnt++];
            }
            // copy global indices of matrix P
            for (j = 0; j < row_size; j++)
            {
               hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( psiComposite[level] )[i] )[j] = (HYPRE_Int) recv_buffer[cnt++];
            }
         }
      }
   }

   return num_psi_levels;
}

HYPRE_Int
AddToCompGrid( hypre_ParCompGrid **compGrid, hypre_ParCompGrid **psiComposite, HYPRE_Int **recv_map_send, 
      HYPRE_Int *recv_map_size, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, 
      HYPRE_Int num_psi_levels, HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index, HYPRE_Int *num_added_nodes )
{
   // HYPRE_Int   myid;
   // hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int         level,i,j;
   HYPRE_Int         need_coarse_info;
   HYPRE_Int         num_nodes;
   HYPRE_Int         add_flag;

   num_nodes = hypre_ParCompGridNumNodes(compGrid[current_level]);
   *recv_map_send_buffer_size = num_levels - current_level;

   // copy all info on this level (there will not be redundant info)
   for (i = 0; i < hypre_ParCompGridNumNodes(psiComposite[current_level]); i++) 
   {
      // check whether we need to allocate more space in order to add to compGrid on this level
      hypre_ParCompGridDynamicResize(compGrid[current_level]);

      // copy data into compGrid
      hypre_ParCompGridCopyNode( psiComposite[current_level], compGrid[current_level], i, num_nodes );

      // generate the receive map for this proc on this level
      recv_map_send[current_level][i] = num_nodes;
      recv_map_size[current_level]++;
      (*recv_map_send_buffer_size)++;

      // increment num_nodes
      hypre_ParCompGridNumNodes(compGrid[current_level]) = ++num_nodes;

      // count the number of added nodes on this level
      num_added_nodes[current_level]++;
   }

   // loop over coarser levels
   for (level = current_level+1; level < current_level + num_psi_levels; level++)
   {
      // get the number of nodes in compGrid on this level
      num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);

      // if this level of compGrid was empty, then copy over everything from psiComposite
      if ( num_nodes == 0 )
      {
         // check whether we need coarse info
         if ( level != num_levels-1 ) need_coarse_info = 1;
         else need_coarse_info = 0;

         // set an initial size for compGrid on this level equal to the number of psiComposite nodes you are about to add
         hypre_ParCompGridSetSize(compGrid[level], hypre_ParCompGridNumNodes(psiComposite[level]), need_coarse_info);

         // count the number of added nodes on this level
         num_added_nodes[level] = hypre_ParCompGridNumNodes(psiComposite[level]);

         // copy over the data
         for (i = 0; i < hypre_ParCompGridNumNodes(psiComposite[level]); i++) 
         {
            // copy data into compGrid
            hypre_ParCompGridCopyNode( psiComposite[level], compGrid[level], i, i );

            // generate the receive map for this proc on this level
            recv_map_send[level][i] = i;
            recv_map_size[level]++;
            (*recv_map_send_buffer_size)++;
         }
      }
      // otherwise, loop over nodes in psiComposite
      else
      {
         for (i = 0; i < hypre_ParCompGridNumNodes(psiComposite[level]); i++) 
         {
            // check whether node is already in the compGrid
            // if not owned by this processor, have to check added nodes
            add_flag = 0;
            if ( hypre_ParCompGridGlobalIndices(psiComposite[level])[i] < proc_first_index[level] || hypre_ParCompGridGlobalIndices(psiComposite[level])[i] > proc_last_index[level] )
            {
               add_flag = 1;
               // search over nodes added to this comp grid (i.e. those with local index greater than num_owned_nodes) !!! linear search !!!
               for (j = hypre_ParCompGridNumNodes(compGrid[level]) - 1; j >= hypre_ParCompGridNumOwnedNodes(compGrid[level]); j--) // Note: doing the search backward (hopefully shorter)
               {
                  if ( hypre_ParCompGridGlobalIndices(psiComposite[level])[i] == hypre_ParCompGridGlobalIndices(compGrid[level])[j] )
                  {
                     add_flag = 0;
                     break;
                  }
               }
            }

            // if node is not present, add the node
            if (add_flag)
            {
               // check whether we need to allocate more space in order to add to compGrid on this level
               hypre_ParCompGridDynamicResize(compGrid[level]);

               // copy data into compGrid
               hypre_ParCompGridCopyNode( psiComposite[level], compGrid[level], i, num_nodes );

               // generate the receive map for this proc on this level
               recv_map_send[level][i] = num_nodes;
               recv_map_size[level]++;
               (*recv_map_send_buffer_size)++;

               // increment num_nodes
               hypre_ParCompGridNumNodes(compGrid[level]) = ++num_nodes;

               // count the number of added nodes on this level
               num_added_nodes[level]++;
            }
            else
            {
               // flag node as repeated info which doesn't need to be sent later
               recv_map_send[level][i] = -1;
               (*recv_map_send_buffer_size)++;
            }
         }
      }
   }

   return 0;
}

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_send_buffer, HYPRE_Int *num_incoming_nodes, HYPRE_Int current_level, HYPRE_Int num_levels)
{
   HYPRE_Int      level, i, cnt;

   cnt = 0;
   for (level = current_level; level < num_levels; level++)
   {
      // if there were nodes in psiComposite on this level
      if (recv_map_send[level])
      {
         // store the number of nodes on this level
         recv_map_send_buffer[cnt++] = num_incoming_nodes[level];

         for (i = 0; i < num_incoming_nodes[level]; i++)
         {
            // store the map values for each node
            recv_map_send_buffer[cnt++] = recv_map_send[level][i];
         }
      }
      // otherwise record that there were zero nodes on this level
      else recv_map_send_buffer[cnt++] = 0;
   }

   return 0;
}

HYPRE_Int
UnpackSendFlagBuffer(HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *send_buffer_size, HYPRE_Int *num_send_nodes, HYPRE_Int current_level, HYPRE_Int num_levels)
{
   HYPRE_Int      level, i, cnt;
   HYPRE_Int      num_nodes;

   cnt = 0;
   *send_buffer_size = num_levels - current_level;
   for (level = current_level; level < num_levels; level++)
   {
      num_nodes = send_flag_buffer[cnt++];
      num_send_nodes[level] = 0;

      if (num_nodes) send_flag[level] = hypre_CTAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);

      for (i = 0; i < num_nodes; i++)
      {
         if (send_flag_buffer[cnt++] != -1) 
         {
            // flag the node to be sent on later communications and count the send buffer size
            send_flag[level][i] = 1;
            (*send_buffer_size)++;
            num_send_nodes[level]++;
         }
         else send_flag[level][i] = 0;
      }
   }

   return 0;
}

HYPRE_Int
PackResidualBuffer( HYPRE_Complex *send_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels )
{
   HYPRE_Int                  level,i,j,cnt = 0, flag_cnt;
   HYPRE_Int                  send_elmt;
   HYPRE_Int                  row_size;
   HYPRE_Int                  nodes_to_add = 0, coarse_grid_index, need_coarse_info;
   HYPRE_Int                  **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   hypre_ParCompMatrixRow     *row;

   // Get where to look in commPkgSendMapElmts
   HYPRE_Int            start = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][processor];
   HYPRE_Int            finish = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][processor+1];

   // see whether we need coarse info and if so, set up add_flag
   if (current_level != num_levels-1) need_coarse_info = 1;
   else need_coarse_info = 0;
   if (need_coarse_info) add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[current_level+1]), HYPRE_MEMORY_HOST );

   // pack the number of nodes sent on this level
   send_buffer[cnt++] = finish - start;

   // copy correct data into psiComposite from compGrid using sendMapElmts from compGridCommPkg
   // !!! CHECK ORDERING OF COMP GRIDS VS ORDERING OF SENDMAPELMTS... SHOULD BE OK I THINK... BUT CHECK !!!
   for (i = start; i < finish; i++)
   {
      // see whether we need coarse info
      if (current_level != num_levels-1) need_coarse_info = 1;
      else need_coarse_info = 0;

      // get index of element to send
      send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];

      // copy the residual at that index into the send buffer
      send_buffer[cnt++] = hypre_ParCompGridF(compGrid[current_level])[send_elmt];

      // flag nodes that will be on the next coarse grid
      if (need_coarse_info)
      {
         coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid[current_level])[send_elmt];
         if ( coarse_grid_index != -1 )
         {
            // look at the matrix row associated with the coarse node at coarse_grid_index
            row = hypre_ParCompGridARows(compGrid[current_level+1])[coarse_grid_index];
            row_size = hypre_ParCompMatrixRowSize(row);
            // loop over neighbors and flag them to add to next coarse psiComposite grid
            for (j = 0; j < row_size; j++)
            {
               if ( hypre_ParCompMatrixRowLocalIndices(row)[j] != -1 )
               {
                  if ( ! add_flag[current_level+1][ hypre_ParCompMatrixRowLocalIndices(row)[j] ] )
                  {
                     add_flag[current_level+1][ hypre_ParCompMatrixRowLocalIndices(row)[j] ] = 1;
                     nodes_to_add++;
                  }
               }
            }
         }
      }
   }

   // get composite grid generated by psi
   for (level = current_level + 1; level < num_levels; level++)
   {
      // store number of nodes to send on this level
      send_buffer[cnt++] = num_send_nodes[level];

      // see whether we need coarse info on this level
      if (level != num_levels-1) need_coarse_info = 1;
      else need_coarse_info = 0;

      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         // reset the flag_cnt
         flag_cnt = 0;

         // if we need coarse info, allocate space for the add flag on the next level
         if (need_coarse_info)
         {
            add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]), HYPRE_MEMORY_HOST );
            nodes_to_add = 0;
         }

         // loop over nodes and add those flagged by add nodes
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i])
            {
               // if this info not redundant, pack it into the send buffer
               if ( send_flag[level][flag_cnt++] ) send_buffer[cnt++] = hypre_ParCompGridF(compGrid[level])[i];

               if (need_coarse_info)
               {
                  coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i];
                  if ( coarse_grid_index != -1 )
                  {
                     // look at the matrix row associated with the coarse node at coarse_grid_index
                     row = hypre_ParCompGridARows(compGrid[level+1])[coarse_grid_index];
                     row_size = hypre_ParCompMatrixRowSize(row);
                     // loop over neighbors and flag them to add to next coarse psiComposite grid
                     for (j = 0; j < row_size; j++)
                     {
                        if ( hypre_ParCompMatrixRowLocalIndices(row)[j] != -1 )
                        {
                           if ( ! add_flag[level+1][ hypre_ParCompMatrixRowLocalIndices(row)[j] ] )
                           {
                              add_flag[level+1][ hypre_ParCompMatrixRowLocalIndices(row)[j] ] = 1;
                              nodes_to_add++;
                           }
                        }
                     }
                  }                  
               }
            }
         }
      }
   }

   // Cleanup memory
   for (level = 0; level < num_levels; level++)
   {
      hypre_TFree(add_flag[level], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);

   return 0;

}

HYPRE_Int
UnpackResidualBuffer( HYPRE_Complex *recv_buffer, HYPRE_Int **recv_map, hypre_ParCompGrid **compGrid, HYPRE_Int current_level, HYPRE_Int num_levels)
{
   HYPRE_Int                  level,i,cnt = 0, map_cnt, num_nodes;

   // loop over levels
   for (level = current_level; level < num_levels; level++)
   {
      // get number of nodes to unpack on this level
      num_nodes = recv_buffer[cnt++];

      // reset the map counter
      map_cnt = 0;

      for (i = 0; i < num_nodes; i++)
      {
         hypre_ParCompGridF(compGrid[level])[ recv_map[level][map_cnt++] ] = recv_buffer[cnt++];
      }
   }

   return 0;
}
