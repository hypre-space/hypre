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

#define DEBUG_COMP_GRID 1 // if true, runs some tests, prints out what is stored in the comp grids for each processor to a file
#define DEBUG_PROC_NEIGHBORS 0 // if true, dumps info on the add flag structures that determine nearest processor neighbors 
#define DEBUGGING_MESSAGES 0 // if true, prints a bunch of messages to the screen to let you know where in the algorithm you are

HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGrid *compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int padding, HYPRE_Int num_ghost_layers, HYPRE_Int *bandwidth_cost );

HYPRE_Int
FindNeighborProcessors( hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, HYPRE_Int ***add_flag,
   HYPRE_Int **num_starting_nodes, HYPRE_Int ***starting_nodes,
   HYPRE_Int **search_proc_marker,
   HYPRE_Int **num_request_nodes, HYPRE_Int ***request_nodes,
   HYPRE_Int *num_send_procs, HYPRE_Int **send_procs, 
   HYPRE_Int num_neighboring_procs,
   HYPRE_Int *send_proc_array_size
   , HYPRE_Int level, HYPRE_Int iteration, HYPRE_Int *bandwidth_cost );

HYPRE_Int
RecursivelyFindNeighborNodes(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, 
   HYPRE_Int *request_nodes, HYPRE_Int *num_request_nodes
   , HYPRE_Int level, HYPRE_Int iteration, HYPRE_Int proc );

HYPRE_Int FindTransitionLevel(hypre_ParAMGData *amg_data);
HYPRE_Int AllgatherCoarseLevels(hypre_ParAMGData *amg_data, HYPRE_Int transition_level);
HYPRE_Int PackCoarseLevels(hypre_ParAMGData *amg_data, HYPRE_Int transition_level, HYPRE_Int **int_buffer, HYPRE_Complex **complex_buffer, HYPRE_Int *buffer_sizes);
HYPRE_Int UnpackCoarseLevels(hypre_ParAMGData *amg_data, HYPRE_Int *recv_int_buffer, HYPRE_Complex *recv_complex_buffer, HYPRE_Int *max_buffer_sizes, HYPRE_Int transition_level);

HYPRE_Complex*
PackSendBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, HYPRE_Int *send_flag_buffer_size, 
   HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int padding, 
   HYPRE_Int num_ghost_layers, HYPRE_Int send_rank, HYPRE_Int *bandwidth_cost );

HYPRE_Int
RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int *add_flag_coarse, 
   HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding);

HYPRE_Int
UnpackRecvBuffer( HYPRE_Complex *recv_buffer, hypre_ParCompGrid **compGrid, 
      HYPRE_Int *num_sends, HYPRE_Int *num_recvs,
      HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
      HYPRE_Int ****recv_map, HYPRE_Int ***recv_map_send, 
      HYPRE_Int ***num_recv_nodes, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int transition_level,
      HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index, HYPRE_Int *num_added_nodes, HYPRE_Int **num_incoming_nodes, HYPRE_Int buffer_number, HYPRE_Int recv_rank );

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_send_buffer, HYPRE_Int *num_incoming_nodes, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
UnpackSendFlagBuffer(HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
FinalizeSendFlag(HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes, HYPRE_Int *num_sends, HYPRE_Int num_levels);

HYPRE_Int
TestCompGrids1(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int transition_level, HYPRE_Int padding, HYPRE_Int num_ghost_layers, HYPRE_Int current_level, HYPRE_Int check_ghost_info);

HYPRE_Int
TestCompGrids2(hypre_ParCompGrid **compGrid, HYPRE_Int transition_level);

HYPRE_Int
TestCompGrids3(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, hypre_ParCSRMatrix **A, hypre_ParCSRMatrix **P, hypre_ParVector **F);

/*****************************************************************************
 *
 * Routine for setting up the composite grids in AMG-DD
 
 *****************************************************************************/

/*****************************************************************************
 * hypre_BoomerAMGDDSetup
 *****************************************************************************/

HYPRE_Int
hypre_BoomerAMGDDSetup( void *amg_vdata, 
                        hypre_ParCSRMatrix *A, 
                        hypre_ParVector *b, 
                        hypre_ParVector *x, 
                        HYPRE_Int *timers, 
                        HYPRE_Int use_barriers,
                        HYPRE_Int *bandwidth_cost )
{
   char filename[256];

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   MPI_Comm 	      comm;
   hypre_ParAMGData   *amg_data = amg_vdata;

   // If the underlying AMG data structure has not yet been set up, call BoomerAMGSetup()
   if (!hypre_ParAMGDataAArray(amg_data))
      hypre_BoomerAMGSetup(amg_vdata, A, b, x);



   #if DEBUGGING_MESSAGES
   hypre_printf("Began comp grid setup on rank %d\n", myid);
   #endif

   /* Data Structure variables */
 
   // level counters, indices, and parameters
   HYPRE_Int                  num_levels;
   HYPRE_Int                  padding;
   HYPRE_Int                  num_ghost_layers;
   HYPRE_Int                  use_transition_level;
   HYPRE_Int                  transition_level;
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

   // composite grids
   hypre_ParCompGrid          **compGrid;

   // info needed for later composite grid communication
   hypre_ParCompGridCommPkg   *compGridCommPkg;
   HYPRE_Int                  num_sends, num_recvs;
   HYPRE_Int                  **send_buffer_size;
   HYPRE_Int                  **recv_buffer_size;
   HYPRE_Int                  ***num_send_nodes;
   HYPRE_Int                  ***num_recv_nodes;
   HYPRE_Int                  ****send_flag;
   HYPRE_Int                  ****recv_map;

   // temporary arrays used for communication during comp grid setup
   HYPRE_Complex              **send_buffer;
   HYPRE_Complex              **recv_buffer;
   HYPRE_Int                  ***recv_map_send;
   HYPRE_Int                  **num_incoming_nodes;
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
   padding = hypre_ParAMGDataAMGDDPadding(amg_data);
   num_ghost_layers = hypre_ParAMGDataAMGDDNumGhostLayers(amg_data);
   use_transition_level = hypre_ParAMGDataAMGDDUseTransitionLevel(amg_data);

   // get first and last global indices on each level for this proc
   proc_first_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   proc_last_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   global_nodes = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   num_added_nodes = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   for (level = 0; level < num_levels; level++)
   {
      proc_first_index[level] = hypre_ParVectorFirstIndex(F_array[level]);
      proc_last_index[level] = hypre_ParVectorLastIndex(F_array[level]);
      global_nodes[level] = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
   }

   // Allocate space for some variables that store info on each level
   compGrid = hypre_CTAlloc(hypre_ParCompGrid*, num_levels, HYPRE_MEMORY_HOST);
   compGridCommPkg = hypre_ParCompGridCommPkgCreate();
   hypre_ParCompGridCommPkgNumSends(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgNumLevels(compGridCommPkg) = num_levels;
   send_buffer_size = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   recv_buffer_size = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   send_flag = hypre_CTAlloc(HYPRE_Int***, num_levels, HYPRE_MEMORY_HOST);
   num_send_nodes = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);
   recv_map = hypre_CTAlloc(HYPRE_Int***, num_levels, HYPRE_MEMORY_HOST);
   num_recv_nodes = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);






   // assign compGrid and compGridCommPkg info to the amg structure
   hypre_ParAMGDataCompGrid(amg_data) = compGrid;
   hypre_ParAMGDataCompGridCommPkg(amg_data) = compGridCommPkg;





   // If needed, find the transition level and initialize comp grids above and below the transition as appropriate
   if (use_transition_level)
   {
      transition_level = FindTransitionLevel(amg_data);
      hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg) = transition_level;

      // Do all gather so that all processors own the coarsest levels of the AMG hierarchy
      AllgatherCoarseLevels(amg_data, transition_level);

      // Initialize composite grids above the transition level
      for (level = 0; level < transition_level; level++)
      {
         compGrid[level] = hypre_ParCompGridCreate();
         if (level != num_levels-1) hypre_ParCompGridInitialize( compGrid[level], F_array[level], CF_marker_array[level], proc_first_index[level+1], A_array[level], P_array[level] );
         else hypre_ParCompGridInitialize( compGrid[level], F_array[level], CF_marker_array[level], 0, A_array[level], NULL );
      }
   }
   // Otherwise just initialize comp grid on all levels
   else
   {
      transition_level = num_levels;
      for (level = 0; level < num_levels; level++)
      {
         compGrid[level] = hypre_ParCompGridCreate();
         if (level != num_levels-1) hypre_ParCompGridInitialize( compGrid[level], F_array[level], CF_marker_array[level], proc_first_index[level+1], A_array[level], P_array[level] );
         else hypre_ParCompGridInitialize( compGrid[level], F_array[level], CF_marker_array[level], 0, A_array[level], NULL );
      }   
   }

   #if DEBUG_COMP_GRID == 2
   for (level = 0; level < num_levels; level++)
   {
      sprintf(filename, "outputs/AMG_hierarchy/A_rank%d_level%d.txt", myid, level);
      hypre_ParCompGridMatlabAMatrixDump( compGrid[level], filename);
      sprintf(filename, "outputs/AMG_hierarchy/coarse_global_indices_rank%d_level%d.txt", myid, level);
      hypre_ParCompGridCoarseGlobalIndicesDump( compGrid[level], filename);
      // sprintf(filename, "outputs/AMG_hierarchy/P_rank%d_level%d.txt", myid, level);
      // hypre_ParCompGridMatlabPMatrixDump( compGrid[level], filename);
      // hypre_sprintf(filename, "outputs/CompGrids/initCompGridRank%dLevel%d.txt", myid, level);
      // hypre_ParCompGridDebugPrint( compGrid[level], filename );
   }
   #endif

   // On each level, setup a long distance commPkg that has communication info for distance (eta + numGhostLayers)
   if (timers) hypre_BeginTiming(timers[0]);
   for (level = 0; level < transition_level; level++)
   {
      // !!! For now, not counting bandwidth cost here !!!
      // SetupNearestProcessorNeighbors(A_array[level], compGrid[level], compGridCommPkg, level, padding, num_ghost_layers, bandwidth_cost);
      SetupNearestProcessorNeighbors(A_array[level], compGrid[level], compGridCommPkg, level, padding, num_ghost_layers, NULL);
   }
   if (timers) hypre_EndTiming(timers[0]);
   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   
   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   #if DEBUGGING_MESSAGES
   if (myid == 0) printf("  Looping over levels\n");
   #endif

   for (level = transition_level - 1; level >= 0; level--)
   {
      comm = hypre_ParCSRMatrixComm(A_array[level]);
      num_sends = hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level];
      num_recvs = hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level];
      #if DEBUGGING_MESSAGES
      printf("    Rank %d: Level %d:\n", myid, level);
      #endif
      if ( proc_last_index[level] >= proc_first_index[level] && num_sends ) // If there are any owned nodes on this level
      {
         // allocate space for the buffers, buffer sizes, requests and status, psiComposite_send, psiComposite_recv, send and recv maps
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         request_counter = 0;
         send_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
         recv_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_recvs, HYPRE_MEMORY_HOST);
         send_buffer = hypre_CTAlloc(HYPRE_Complex*, num_sends, HYPRE_MEMORY_HOST);
         recv_buffer = hypre_CTAlloc(HYPRE_Complex*, num_recvs, HYPRE_MEMORY_HOST);
         num_psi_levels_send = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
         num_psi_levels_recv = hypre_CTAlloc(HYPRE_Int, num_recvs, HYPRE_MEMORY_HOST);

         send_flag[level] = hypre_CTAlloc(HYPRE_Int**, num_sends, HYPRE_MEMORY_HOST);
         num_send_nodes[level] = hypre_CTAlloc(HYPRE_Int*, num_sends, HYPRE_MEMORY_HOST);
         recv_map[level] = hypre_CTAlloc(HYPRE_Int**, num_recvs, HYPRE_MEMORY_HOST);
         num_recv_nodes[level] = hypre_CTAlloc(HYPRE_Int*, num_recvs, HYPRE_MEMORY_HOST);
         recv_map_send = hypre_CTAlloc(HYPRE_Int**, num_recvs, HYPRE_MEMORY_HOST);
         send_flag_buffer = hypre_CTAlloc(HYPRE_Int*, num_sends, HYPRE_MEMORY_HOST);
         send_flag_buffer_size = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
         recv_map_send_buffer = hypre_CTAlloc(HYPRE_Int*, num_recvs, HYPRE_MEMORY_HOST);
         recv_map_send_buffer_size = hypre_CTAlloc(HYPRE_Int, num_recvs, HYPRE_MEMORY_HOST);
         num_incoming_nodes = hypre_CTAlloc(HYPRE_Int*, num_recvs, HYPRE_MEMORY_HOST);

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

         if (timers) hypre_BeginTiming(timers[1]);

         // loop over send procs
         #if DEBUGGING_MESSAGES
         printf("      Rank %d: Loop over send procs:\n", myid);
         #endif

         for (i = 0; i < num_sends; i++)
         {
            // pack send buffers
            send_flag[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_send_nodes[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
            send_buffer[i] = PackSendBuffer( compGrid, compGridCommPkg, &(send_buffer_size[level][i]), 
                                             &(send_flag_buffer_size[i]), send_flag, num_send_nodes, i, level, num_levels, padding, 
                                             num_ghost_layers, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], bandwidth_cost );
            
            if (bandwidth_cost) for (j = 0; j < num_levels; j++) bandwidth_cost[5*level + 1] += num_send_nodes[level][i][j];
         }
         #if DEBUGGING_MESSAGES
         printf("      Rank %d: Done packing send buffers\n", myid);
         #endif

         if (timers) hypre_EndTiming(timers[1]);

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

         if (timers) hypre_BeginTiming(timers[2]);

         // post the receives for the buffer size
         for (i = 0; i < num_recvs; i++)
         {
            hypre_MPI_Irecv( &(recv_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 0, comm, &requests[request_counter++] );
         }

         // send the buffer sizes
         for (i = 0; i < num_sends; i++)
         {
            hypre_MPI_Isend(&(send_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 0, comm, &requests[request_counter++]);
            if (bandwidth_cost) bandwidth_cost[5*level] += sizeof(HYPRE_Int);
         }
      
         // wait for all buffer sizes to be received
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );

         if (timers) hypre_EndTiming(timers[2]);

         // free and reallocate space for the requests and status
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

         if (timers) hypre_BeginTiming(timers[3]);

         // allocate space for the receive buffers and post the receives
         for (i = 0; i < num_recvs; i++)
         {
            recv_buffer[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size[level][i], HYPRE_MEMORY_HOST );
            hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_COMPLEX, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 1, comm, &requests[request_counter++]);
         }

         // pack and send the buffers
         for (i = 0; i < num_sends; i++)
         {
            #if DEBUGGING_MESSAGES
            if (myid == 0) printf("        Post send for proc %d\n", hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i]);
            #endif

            hypre_MPI_Isend(send_buffer[i], send_buffer_size[level][i], HYPRE_MPI_COMPLEX, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 1, comm, &requests[request_counter++]);
            if (bandwidth_cost) bandwidth_cost[5*level] += send_buffer_size[level][i]*sizeof(HYPRE_Complex);
         }

         // wait for buffers to be received
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );

         #if DEBUGGING_MESSAGES
         hypre_printf("      Rank %d: done waiting on buffers\n", myid);
         #endif

         if (timers) hypre_EndTiming(timers[3]);

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

         if (timers) hypre_BeginTiming(timers[4]);

         // loop over received buffers
         #if DEBUGGING_MESSAGES
         printf("      Rank %d: Loop over recv procs:\n",myid);
         #endif
         for (i = 0; i < num_recvs; i++)
         {
            // allocate space for the recv map info
            recv_map_send[i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_recv_nodes[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
            num_incoming_nodes[i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);

            // and add information to this composite grid
            UnpackRecvBuffer(recv_buffer[i], compGrid, hypre_ParCompGridCommPkgNumSends(compGridCommPkg), hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg), 
               send_flag, num_send_nodes, 
               recv_map, recv_map_send, 
               num_recv_nodes, &(recv_map_send_buffer_size[i]), level, num_levels, transition_level, proc_first_index, proc_last_index, num_added_nodes, num_incoming_nodes, i, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i]);
         }

         if (timers) hypre_EndTiming(timers[4]);

         // Setup local indices for the composite grid
         #if DEBUGGING_MESSAGES
         printf("      Rank %d: Setup local indices\n",myid);
         #endif

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         
         if (timers) hypre_BeginTiming(timers[5]);
         
         hypre_ParCompGridSetupLocalIndices(compGrid, num_added_nodes, transition_level, proc_first_index, proc_last_index);
         if (level == 0) hypre_ParCompGridSetupLocalIndicesP(compGrid, num_levels, transition_level);
         
         if (timers) hypre_EndTiming(timers[5]);

         #if DEBUGGING_MESSAGES
         printf("      Rank %d: Done with setup local indices\n",myid);
         #endif

         // Zero out num_added_nodes
         for (i = level; i < num_levels; i++) num_added_nodes[i] = 0;

         // free and reallocate space for the requests and status
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

         if (timers) hypre_BeginTiming(timers[6]);         

         // post receives for send maps - NOTE: we want to receive this info from procs we sent to
         for (i = 0; i < num_sends; i++)
         {
            send_flag_buffer[i] = hypre_CTAlloc(HYPRE_Int, send_flag_buffer_size[i], HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv( send_flag_buffer[i], send_flag_buffer_size[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 2, comm, &requests[request_counter++]);
            if (bandwidth_cost) bandwidth_cost[5*level] += send_flag_buffer_size[i]*sizeof(HYPRE_Int);
         }

         // send recv_map_send to procs received from to become their send maps - NOTE: we want to send this info from procs we received from
         for (i = 0; i < num_recvs; i++)
         {
            // pack up the recv_map_send's and send them
            recv_map_send_buffer[i] = hypre_CTAlloc(HYPRE_Int, recv_map_send_buffer_size[i], HYPRE_MEMORY_HOST);
            PackRecvMapSendBuffer(recv_map_send[i], recv_map_send_buffer[i], num_incoming_nodes[i], level, num_levels);
            hypre_MPI_Isend( recv_map_send_buffer[i], recv_map_send_buffer_size[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 2, comm, &requests[request_counter++]);
         }

         // wait for maps to be received
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );

         #if DEBUGGING_MESSAGES
         hypre_printf("      Rank %d: done waiting on send map buffers\n", myid);
         #endif

         // unpack and setup the send flag arrays
         for (i = 0; i < num_sends; i++)
         {
            UnpackSendFlagBuffer(send_flag_buffer[i], send_flag[level][i], &(send_buffer_size[level][i]), level, num_levels);
         }

         // finalize the recv maps and get final recv buffer size
         for (i = 0; i < num_recvs; i++)
         {
            // allocate space for each level of the receive map for this proc
            recv_map[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);

            // for each level
            for (j = level; j < num_levels; j++)
            {
               // if there is info for this proc on this level
               if (recv_map_send[i][j])
               {
                  // allocate the appropriate amount of space for the map
                  recv_map[level][i][j] = hypre_CTAlloc(HYPRE_Int, num_recv_nodes[level][i][j], HYPRE_MEMORY_HOST);
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
         
         if (timers) hypre_EndTiming(timers[6]);

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
         hypre_TFree(num_incoming_nodes, HYPRE_MEMORY_HOST);
         hypre_TFree(num_psi_levels_send, HYPRE_MEMORY_HOST);
         hypre_TFree(num_psi_levels_recv, HYPRE_MEMORY_HOST);
      }
      else 
      {
         if (use_barriers)
         {
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         }
      }
      #if DEBUGGING_MESSAGES
      hypre_printf("      Rank %d: done with level %d\n", myid, level);
      #endif 


      #if DEBUG_COMP_GRID
      HYPRE_Int error_code;
      error_code = TestCompGrids1(compGrid, num_levels, transition_level, padding, num_ghost_layers, level, 0);
      if (error_code)
      {
         printf("TestCompGrids1 failed! Rank %d, level %d\n", myid, level);
      }
      #endif

   }


   // Setup the ghost/real dof markers for the comp grids
   if (use_transition_level) hypre_ParCompGridSetupRealDofMarker(compGrid, transition_level+1, padding);
   else hypre_ParCompGridSetupRealDofMarker(compGrid, num_levels, padding);

   #if DEBUG_COMP_GRID
   // Test whether comp grids have correct shape
   HYPRE_Int test_failed = 0;
   HYPRE_Int error_code;
   error_code = TestCompGrids1(compGrid, num_levels, transition_level, padding, num_ghost_layers, 0, 1);
   if (error_code)
   {
      printf("TestCompGrids1 failed!\n");
      test_failed = 1;
   }
   #endif

   // Finalize the send flag and the compGrids
   FinalizeSendFlag(send_flag, num_send_nodes, hypre_ParCompGridCommPkgNumSends(compGridCommPkg), transition_level);
   hypre_ParCompGridFinalize(compGrid, num_levels, transition_level);

   // Count up the bandwidth cost for subsequent residual communications
   if (bandwidth_cost)
   {
      for (level = 0; level < num_levels; level++)
      {
         for (i = 0; i < hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level]; i++)
         {
            bandwidth_cost[5*level+2] += send_buffer_size[level][i]*sizeof(HYPRE_Complex);
            for (j = 0; j < num_levels; j++)
            {
               bandwidth_cost[5*level+3] += num_send_nodes[level][i][j];
            }
         }
      }
   }

   #if DEBUG_COMP_GRID
   if (use_transition_level) error_code = TestCompGrids2(compGrid, transition_level+1);
   else error_code = TestCompGrids2(compGrid, num_levels);
   if (error_code)
   {
      printf("TestCompGrids2 failed!\n");
      test_failed = 1;
   }
   error_code = TestCompGrids3(compGrid, transition_level, hypre_ParAMGDataAArray(amg_data), hypre_ParAMGDataPArray(amg_data), hypre_ParAMGDataFArray(amg_data));
   if (error_code)
   {
      printf("TestCompGrids3 failed!\n");
      test_failed = 1;
   }
   #endif

   #if DEBUG_COMP_GRID == 2
   for (level = 0; level < num_levels; level++)
   {
      hypre_sprintf(filename, "outputs/CompGrids/setupCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrint( compGrid[level], filename );
      // hypre_ParCompGridDumpSorted( compGrid[level], filename );
      // hypre_sprintf(filename, "outputs/CompGrids/setupACompRank%dLevel%d.txt", myid, level);
      // hypre_ParCompGridMatlabAMatrixDump( compGrid[level], filename );
      // if (level != num_levels-1)
      // {
      //    hypre_sprintf(filename, "outputs/CompGrids/setupPCompRank%dLevel%d.txt", myid, level);
      //    hypre_ParCompGridMatlabPMatrixDump( compGrid[level], filename );
      // }
   }
   #endif

   // store communication info in compGridCommPkg
   hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg) = send_buffer_size;
   hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg) = recv_buffer_size;
   hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg) = num_send_nodes;
   hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg) = num_recv_nodes;
   hypre_ParCompGridCommPkgSendFlag(compGridCommPkg) = send_flag;
   hypre_ParCompGridCommPkgRecvMap(compGridCommPkg) = recv_map;

   #if DEBUGGING_MESSAGES
   hypre_printf("Finished comp grid setup on rank %d\n", myid);
   #endif

   // Cleanup memory
   hypre_TFree(num_added_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(global_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_first_index, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_last_index, HYPRE_MEMORY_HOST);
   
   #if DEBUG_COMP_GRID
   return test_failed;
   #else
   return 0;
   #endif
}  


HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGrid *compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int padding, HYPRE_Int num_ghost_layers, HYPRE_Int *bandwidth_cost )
{
   HYPRE_Int               i,j,cnt;
   HYPRE_Int               num_nodes = hypre_ParCSRMatrixNumRows(A);
   hypre_ParCSRCommPkg     *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int               start,finish;

   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Get the default (distance 1) number of send procs
   HYPRE_Int      num_sends = hypre_ParCSRCommPkgNumSends(commPkg);

   #if DEBUG_PROC_NEIGHBORS
   // Check to make sure original matrix has symmetric send/recv relationship !!! I should really have a check to make sure the matrix is actually symmetric !!!
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(commPkg);
   if (num_sends == num_recvs)
   {
      for (i = 0; i < num_sends; i++)
      {
         HYPRE_Int send_found = 0;
         for (j = 0; j < num_recvs; j++)
         {
            if (hypre_ParCSRCommPkgSendProc(commPkg,i) == hypre_ParCSRCommPkgRecvProc(commPkg,j))
            {
               send_found = 1;
               break;
            }
         }
         if (!send_found) printf("Error: initial commPkg send and recv ranks differ on level %d, rank %d\n", level, myid);
      }
   }
   else printf("Error: num_sends doesn't equal num_recvs for original commPkg on  level %d, rank %d\n", level, myid);
   #endif

   // If num_sends is zero, then simply note that in compGridCommPkg and we are done
   if (num_sends == 0)
   {
      hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level] = num_sends;
      hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level] = num_sends;
   }
   else
   {
      // Initialize add_flag (this is how we will track nodes to send to each proc until routine finishes)
      // Note: several allocations occur below for arrays that are meant to store objects/info for each proc that we end up sending to
      // This number is unknown a priori, so we start with double the number of initial send procs and reallocate as necessary inside FindNeighborProcessors()
      HYPRE_Int      send_proc_array_size = 2*num_sends;
      HYPRE_Int      *send_procs = hypre_CTAlloc( HYPRE_Int, send_proc_array_size, HYPRE_MEMORY_HOST);
      HYPRE_Int      **add_flag = hypre_CTAlloc( HYPRE_Int*, send_proc_array_size, HYPRE_MEMORY_HOST);
      HYPRE_Int      *search_proc_marker = hypre_CTAlloc(HYPRE_Int, send_proc_array_size, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         send_procs[i] = hypre_ParCSRCommPkgSendProc(commPkg,i);
         add_flag[i] = hypre_CTAlloc( HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
         for (j = start; j < finish; j++) add_flag[i][ hypre_ParCSRCommPkgSendMapElmt(commPkg,j) ] = padding + num_ghost_layers; // must be set to padding + numGhostLayers (note that the starting nodes are already distance 1 from their neighbors on the adjacent processor)
      }

      // Setup initial num_starting_nodes and starting_nodes (these are the starting nodes when searching for long distance neighbors) !!! I don't think I actually have a good upper bound on sizes here... how to properly allocate/reallocate these? !!!
      HYPRE_Int *num_starting_nodes = hypre_CTAlloc( HYPRE_Int, send_proc_array_size, HYPRE_MEMORY_HOST );
      HYPRE_Int **starting_nodes = hypre_CTAlloc( HYPRE_Int*, send_proc_array_size, HYPRE_MEMORY_HOST );
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
         search_proc_marker[i] = 1;
         num_starting_nodes[i] = finish - start;
      }
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         starting_nodes[i] = hypre_CTAlloc( HYPRE_Int, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)), HYPRE_MEMORY_HOST );
         for (j = 0; j < num_starting_nodes[i]; j++)
         {
            starting_nodes[i][j] = hypre_ParCSRCommPkgSendMapElmt(commPkg, j + start );
         }
      }

      // Find my own send nodes and communicate with neighbors to find off-processor long-range connections
      HYPRE_Int *num_request_nodes = hypre_CTAlloc(HYPRE_Int, send_proc_array_size, HYPRE_MEMORY_HOST);
      HYPRE_Int **request_nodes = hypre_CTAlloc(HYPRE_Int*, send_proc_array_size, HYPRE_MEMORY_HOST);

      FILE *file;
      char filename[256];
      for (i = 0; i < padding + num_ghost_layers - 1; i++)
      {
         FindNeighborProcessors(compGrid, A, &(add_flag), 
            &(num_starting_nodes), &(starting_nodes), 
            &(search_proc_marker),
            &(num_request_nodes), &(request_nodes),
            &num_sends, &(send_procs), 
            hypre_ParCSRCommPkgNumSends(commPkg),
            &(send_proc_array_size)
            , level, i, bandwidth_cost); // Note that num_sends may change here
      }

      #if DEBUG_PROC_NEIGHBORS
      for (j = 0; j < num_sends; j++)
      {
         sprintf(filename,"outputs/add_flag_level%d_proc%d_rank%d.txt", level, send_procs[j], myid);
         file = fopen(filename,"w");
         HYPRE_Int k;
         for (k = 0; k < num_nodes; k++)
         {
            fprintf(file, "%d ", add_flag[j][k]);
         }
         fprintf(file, "\n");
         for (k = 0; k < num_nodes; k++)
         {
            fprintf(file, "%d ", hypre_ParCompGridGlobalIndices(compGrid)[k]);
         }
         fclose(file);
         // if (num_request_nodes[j])
         // {
         //    sprintf(filename,"outputs/request_nodes_level%d_proc%d_rank%d_i%d.txt", level, send_procs[j], myid, padding + num_ghost_layers - 1);
         //    file = fopen(filename,"w");
         //    for (k = 0; k < num_request_nodes[j]; k++)
         //    {
         //       fprintf(file, "%d ", request_nodes[j][2*k]);
         //    }
         //    fprintf(file, "\n");
         //    for (k = 0; k < num_request_nodes[j]; k++)
         //    {
         //       fprintf(file, "%d ", request_nodes[j][2*k+1]);
         //    }
         //    fclose(file);
         // }
      }
      #endif

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
            if (add_flag[i][j] > num_ghost_layers) ghost_marker[cnt] = 0;
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

      // Clean up memory
      for (i = 0; i < num_sends; i++)
      {
         hypre_TFree(add_flag[i], HYPRE_MEMORY_HOST);
         hypre_TFree(starting_nodes[i], HYPRE_MEMORY_HOST);
         hypre_TFree(request_nodes[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(add_flag, HYPRE_MEMORY_HOST);
      hypre_TFree(starting_nodes, HYPRE_MEMORY_HOST);
      hypre_TFree(search_proc_marker, HYPRE_MEMORY_HOST);
      hypre_TFree(request_nodes, HYPRE_MEMORY_HOST);
      hypre_TFree(num_starting_nodes, HYPRE_MEMORY_HOST);
      hypre_TFree(num_request_nodes, HYPRE_MEMORY_HOST);
   }


   #if DEBUG_PROC_NEIGHBORS
   // Check to make sure what we end up with has symmetric send/recv relationship 
   HYPRE_Int max_size;
   HYPRE_Int   num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Allreduce(&num_sends, &max_size, 1, HYPRE_MPI_INT, MPI_MAX, hypre_MPI_COMM_WORLD);
   HYPRE_Int *send_send_procs = hypre_CTAlloc(HYPRE_Int, max_size, HYPRE_MEMORY_HOST);
   HYPRE_Int *recv_send_procs = hypre_CTAlloc(HYPRE_Int, max_size*num_procs, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_sends; i++) send_send_procs[i] = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i];
   for (i = num_sends; i < max_size; i++) send_send_procs[i] = -1;
   hypre_MPI_Allgather(send_send_procs, max_size, HYPRE_MPI_INT, recv_send_procs, max_size, HYPRE_MPI_INT, hypre_MPI_COMM_WORLD);
   for (i = 0; i < num_sends; i++)
   {
      HYPRE_Int send_found = 0;
      for (j = 0; j < max_size; j++)
      {
         if (recv_send_procs[ hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i]*max_size + j ] == myid)
         {
            send_found = 1;
            break;
         }
      }
      if (!send_found) printf("Error: send and recv ranks differ on level %d, rank %d sends to proc %d, but not the reverse\n", level, myid, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i]);
   }
   #endif

   return 0;
}

HYPRE_Int
FindNeighborProcessors( hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, HYPRE_Int ***add_flag,
   HYPRE_Int **num_starting_nodes, HYPRE_Int ***starting_nodes,
   HYPRE_Int **search_proc_marker,
   HYPRE_Int **num_request_nodes, HYPRE_Int ***request_nodes,
   HYPRE_Int *num_send_procs, HYPRE_Int **send_procs, 
   HYPRE_Int num_neighboring_procs,
   HYPRE_Int *send_proc_array_size
   , HYPRE_Int level, HYPRE_Int iteration, HYPRE_Int *bandwidth_cost )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int      i,j,k,cnt;

   // Update add_flag by recursively adding neighbors
   for (i = 0; i < (*num_send_procs); i++)
   {
      if ((*search_proc_marker)[i])
      {
         (*num_request_nodes)[i] = 0;
         if (!(*request_nodes)[i]) (*request_nodes)[i] = hypre_CTAlloc( HYPRE_Int, 2*hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)), HYPRE_MEMORY_HOST );

         for (j = 0; j < (*num_starting_nodes)[i]; j++)
         {
            if ((*add_flag)[i][ (*starting_nodes)[i][j] ] - 1 > 0)
               RecursivelyFindNeighborNodes( (*starting_nodes)[i][j], (*add_flag)[i][ (*starting_nodes)[i][j] ] - 1, compGrid, (*add_flag)[i], (*request_nodes)[i], &((*num_request_nodes)[i])
               , level, iteration, (*send_procs)[i] );
         }

         (*num_starting_nodes)[i] = 0;
      }
   }

   // Exchange message sizes
   HYPRE_Int send_size = 1;
   for (i = 0; i < (*num_send_procs); i++)
   {
      if ((*search_proc_marker)[i])
      {
         if ((*num_request_nodes)[i])
         {
            send_size += 2*(*num_request_nodes)[i] + 2;
         }
      }
   }
   HYPRE_Int *recv_sizes = hypre_CTAlloc(HYPRE_Int, num_neighboring_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Request *requests = hypre_CTAlloc(hypre_MPI_Request, 4*num_neighboring_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Status *statuses = hypre_CTAlloc(hypre_MPI_Status, 4*num_neighboring_procs, HYPRE_MEMORY_HOST);
   HYPRE_Int request_cnt = 0;
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Irecv(&(recv_sizes[i]), 1, HYPRE_MPI_INT, (*send_procs)[i], 4, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      hypre_MPI_Isend(&send_size, 1, HYPRE_MPI_INT, (*send_procs)[i], 4, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (bandwidth_cost) (*bandwidth_cost) += sizeof(HYPRE_Int);
   }

   // Wait on the recv sizes
   hypre_MPI_Waitall(2*num_neighboring_procs, requests, statuses);

   // Allocate recv buffers
   HYPRE_Int **recv_buffers = hypre_CTAlloc(HYPRE_Int*, num_neighboring_procs, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_neighboring_procs; i++) recv_buffers[i] = hypre_CTAlloc(HYPRE_Int, recv_sizes[i], HYPRE_MEMORY_HOST);

   // Post the recvs
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Irecv(recv_buffers[i], recv_sizes[i], HYPRE_MPI_INT, (*send_procs)[i], 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   
   // Setup the send buffer and post the sends
   HYPRE_Int *send_buffer = hypre_CTAlloc(HYPRE_Int, send_size, HYPRE_MEMORY_HOST);
   cnt = 1;
   HYPRE_Int num_request_procs = 0;
   for (i = 0; i < (*num_send_procs); i++)
   {
      if ((*search_proc_marker)[i])
      {
         if ((*num_request_nodes)[i])
         {
            num_request_procs++;
            send_buffer[cnt++] = (*send_procs)[i];
            send_buffer[cnt++] = (*num_request_nodes)[i];
            for (j = 0; j < (*num_request_nodes)[i]; j++)
            {
               send_buffer[cnt++] = (*request_nodes)[i][2*j];
               send_buffer[cnt++] = (*request_nodes)[i][2*j+1];
            }
         }
      }
   }
   send_buffer[0] = num_request_procs;
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Isend(send_buffer, send_size, HYPRE_MPI_INT, (*send_procs)[i], 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (bandwidth_cost) (*bandwidth_cost) += send_size*sizeof(HYPRE_Int);
   }

   // Wait 
   hypre_MPI_Waitall(2*num_neighboring_procs, &(requests[2*num_neighboring_procs]), &(statuses[2*num_neighboring_procs]));

   // Reset search_proc_marker
   for (i = 0; i < (*num_send_procs); i++) (*search_proc_marker)[i] = 0;

   // Unpack recieved messages and update add_flag where appropriate
   for (i = 0; i < num_neighboring_procs; i++)
   {
      cnt = 0;
      HYPRE_Int num_incoming_procs = recv_buffers[i][cnt++];
      for (j = 0; j < num_incoming_procs; j++)
      {
         HYPRE_Int incoming_proc = recv_buffers[i][cnt++];
         HYPRE_Int num_incoming_nodes = recv_buffers[i][cnt++];

         // See whether we already have an add_flag array setup for this proc (and check to avoid info for our rank)
         if (incoming_proc != myid)
         {
            // Look for an add_flag already set up for this processor
            HYPRE_Int local_proc_index = -1;
            HYPRE_Int add_proc = 0;
            for (k = 0; k < (*num_send_procs); k++) if ((*send_procs)[k] == incoming_proc) local_proc_index = k;
            // If we didn't find one, this is a new proc that we may or may not need to account for
            if (local_proc_index < 0)
            {
               // Loop over incoming nodes and see whether we need to setup a new send_proc
               HYPRE_Int subCnt = cnt;
               for (k = 0; k < num_incoming_nodes; k++)
               {
                  HYPRE_Int local_index = recv_buffers[i][subCnt] - hypre_ParCompGridGlobalIndices(compGrid)[0]; 
                  subCnt += 2;
                  if (local_index >=0 && local_index < hypre_ParCSRMatrixNumRows(A))
                  {
                     add_proc = 1;
                     break;
                  }
               }
               if (add_proc)
               {
                  local_proc_index = (*num_send_procs);
                  (*num_send_procs)++;

                  // Note: we may need to grow several arrays here
                  if ((*num_send_procs) > (*send_proc_array_size))
                  {
                     (*send_proc_array_size) = 2*(*send_proc_array_size);
                     (*send_procs) = hypre_TReAlloc((*send_procs), HYPRE_Int, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*add_flag) = hypre_TReAlloc((*add_flag), HYPRE_Int*, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*search_proc_marker) = hypre_TReAlloc((*search_proc_marker), HYPRE_Int, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*num_starting_nodes) = hypre_TReAlloc((*num_starting_nodes), HYPRE_Int, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*starting_nodes) = hypre_TReAlloc((*starting_nodes), HYPRE_Int*, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*num_request_nodes) = hypre_TReAlloc((*num_request_nodes), HYPRE_Int, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*request_nodes) = hypre_TReAlloc((*request_nodes), HYPRE_Int*, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     for (k = local_proc_index; k < (*send_proc_array_size); k++)
                     {
                        (*send_procs)[k] = 0;
                        (*add_flag)[k] = NULL;
                        (*search_proc_marker)[k] = 0;
                        (*num_starting_nodes)[k] = 0;
                        (*starting_nodes)[k] = NULL;
                        (*num_request_nodes)[k] = 0;
                        (*request_nodes)[k] = NULL;
                     }
                  }

                  // Now
                  (*send_procs)[local_proc_index] = incoming_proc;
                  (*add_flag)[local_proc_index] = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRMatrixNumRows(A), HYPRE_MEMORY_HOST);
                  (*starting_nodes)[local_proc_index] = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)), HYPRE_MEMORY_HOST);
                  (*search_proc_marker)[local_proc_index] = 1;
               }
            }
            else add_proc = 1;
            // If we must account for the add_flag of this proc, process the incoming request nodes and update add_flag as appropriate
            if (add_proc)
            {
               for (k = 0; k < num_incoming_nodes; k++)
               {
                  HYPRE_Int local_index = recv_buffers[i][cnt++] - hypre_ParCompGridGlobalIndices(compGrid)[0]; 
                  HYPRE_Int incoming_flag = recv_buffers[i][cnt++];
                  if (local_index >=0 && local_index < hypre_ParCSRMatrixNumRows(A))
                  {
                     if (incoming_flag > (*add_flag)[local_proc_index][local_index])
                     {
                        (*add_flag)[local_proc_index][local_index] = incoming_flag;
                        (*starting_nodes)[local_proc_index][ (*num_starting_nodes)[local_proc_index]++ ] = local_index;
                        (*search_proc_marker)[local_proc_index] = 1;
                     }
                  }
               }
            }
            else
            {
               cnt += 2*num_incoming_nodes;
            }
         }
         else
         {
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

   return 0;
}

HYPRE_Int
RecursivelyFindNeighborNodes(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, 
   HYPRE_Int *request_nodes, HYPRE_Int *num_request_nodes
   , HYPRE_Int level, HYPRE_Int iteration, HYPRE_Int proc )
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
            if (m-1 > 0) RecursivelyFindNeighborNodes(index, m-1, compGrid, add_flag, request_nodes, num_request_nodes
               , level, iteration, proc);
         }
      }
      // otherwise note this as a starting node to request from neighboring procs
      else
      {
         // Check whether we have already requested this node (!!! linear search, but should be over small set !!!)
         HYPRE_Int global_index = hypre_ParCompMatrixRowGlobalIndices(A_row)[i];
         HYPRE_Int add_requeset = 1;
         for (j = 0; j < (*num_request_nodes); j++)
         {
            if (request_nodes[2*j] == global_index)
            {
               add_requeset = 0;
               if (m > request_nodes[2*j+1])
               {
                  request_nodes[2*j+1] = m;
               }
            }
         }
         if (add_requeset)
         {
            request_nodes[2*(*num_request_nodes)] = global_index;
            request_nodes[2*(*num_request_nodes)+1] = m; 
            (*num_request_nodes)++;
         }
      }
   }

   return 0;
}

HYPRE_Int
FindTransitionLevel(hypre_ParAMGData *amg_data)
{
   HYPRE_Int i;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int local_transition, global_transition;

   // Loop over levels, starting at the coarsest
   for (i = num_levels - 1; i >= 0; i--)
   {
      // If this proc owns nodes on this level, note this level and leave the loop
      if (hypre_ParCSRMatrixNumRows( hypre_ParAMGDataAArray(amg_data)[i]) )
      {
         local_transition = i;
         break;
      }
   }

   // Communicate to find the coarsest level on which all procs own at least one dof
   hypre_MPI_Allreduce(&local_transition, &global_transition, 1, HYPRE_MPI_INT, MPI_MIN, hypre_MPI_COMM_WORLD);

   return global_transition;
}

HYPRE_Int
AllgatherCoarseLevels(hypre_ParAMGData *amg_data, HYPRE_Int transition_level)
{
   // Get MPI comm size
   HYPRE_Int   num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   // Pack up the buffer containing comp grid info that will be gathered onto all procs and get its size
   HYPRE_Int *buffer_sizes = hypre_CTAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
   HYPRE_Int *send_int_buffer;
   HYPRE_Complex *send_complex_buffer;
   PackCoarseLevels(amg_data, transition_level, &send_int_buffer, &send_complex_buffer, buffer_sizes);

   // Do an allreduce to find the maximum buffer size
   HYPRE_Int *max_buffer_sizes = hypre_CTAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
   hypre_MPI_Allreduce(buffer_sizes, max_buffer_sizes, 2, HYPRE_MPI_INT, MPI_MAX, hypre_MPI_COMM_WORLD);

   // !!! Remove: !!!
   // FILE *file;
   // char filename[256];
   // HYPRE_Int myid,i;
   // hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   // sprintf(filename, "outputs/allgatherCoarse_send_buffer_rank%d.txt", myid);
   // file = fopen(filename,"w");
   // for (i = 0; i < buffer_size; i++) fprintf(file, "%f ", send_buffer[i]);
   // fclose(file);

   // Allocate a buffer to recieve comp grid info from all procs and do the allgather
   HYPRE_Int *recv_int_buffer = hypre_CTAlloc(HYPRE_Int, num_procs*max_buffer_sizes[0], HYPRE_MEMORY_HOST);
   HYPRE_Complex *recv_complex_buffer = hypre_CTAlloc(HYPRE_Complex, num_procs*max_buffer_sizes[1], HYPRE_MEMORY_HOST);

   hypre_MPI_Request *request = hypre_CTAlloc(hypre_MPI_Request, 2, HYPRE_MEMORY_HOST);
   hypre_MPI_Status *status = hypre_CTAlloc(hypre_MPI_Status, 2, HYPRE_MEMORY_HOST);
   
   MPI_Iallgather(send_int_buffer, buffer_sizes[0], HYPRE_MPI_INT, recv_int_buffer, max_buffer_sizes[0], HYPRE_MPI_INT, hypre_MPI_COMM_WORLD, &(request[0]));
   MPI_Iallgather(send_complex_buffer, buffer_sizes[1], HYPRE_MPI_COMPLEX, recv_complex_buffer, max_buffer_sizes[1], HYPRE_MPI_COMPLEX, hypre_MPI_COMM_WORLD, &(request[1]));
   
   hypre_MPI_Waitall(2, request, status);

   // !!! Remove: !!!
   // sprintf(filename, "outputs/allgatherCoarse_recv_buffer_rank%d.txt", myid);
   // file = fopen(filename,"w");
   // for (i = 0; i < num_procs*max_buffer_size; i++) fprintf(file, "%f ", recv_buffer[i]);
   // fclose(file);

   // Unpack the recv buffer and generate the comp grid structures for the coarse levels
   UnpackCoarseLevels(amg_data, recv_int_buffer, recv_complex_buffer, max_buffer_sizes, transition_level);

   // Clean up memory
   hypre_TFree(request, HYPRE_MEMORY_HOST);
   hypre_TFree(status, HYPRE_MEMORY_HOST);
   hypre_TFree(buffer_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(max_buffer_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(send_int_buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(send_complex_buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_int_buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_complex_buffer, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Int
PackCoarseLevels(hypre_ParAMGData *amg_data, HYPRE_Int transition_level, HYPRE_Int **int_buffer, HYPRE_Complex **complex_buffer, HYPRE_Int *buffer_sizes)
{
   // The buffers will have the following forms:
   // int_buffer = [ [level], (transition_level)     level = [ num_nodes,
   //                [level],                                  A nnz,
   //                ...    ,                                  P nnz,
   //                [level] ] (num_levels - 1)                [A row sizes]
   //                                                          [A col indices]
   //                                                          [P row sizes]
   //                                                          [P col indices] ]
   //
   // complex_buffer = [ [level], (transition_level)     level = [ [A data],
   //                    [level],                                  [P data] ]
   //                    ...    ,
   //                    [level] ] (num_levels - 1)
   

   // Count up how large the buffers will be
   HYPRE_Int level,i,j;
   buffer_sizes[0] = 0;
   buffer_sizes[1] = 0;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   for (level = transition_level; level < num_levels; level++)
   {
      // Get num nodes and num non zeros for matrices on this level
      hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(amg_data)[level];
      hypre_ParCSRMatrix *P = NULL;
      if (level != num_levels-1) P = hypre_ParAMGDataPArray(amg_data)[level];
      HYPRE_Int num_nodes = hypre_ParCSRMatrixNumRows(A);
      HYPRE_Int A_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(A) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(A) );
      HYPRE_Int P_nnz = 0;
      if (level != num_levels-1) P_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(P) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(P) ); 
      // Increment the buffer size appropriately
      buffer_sizes[0] += 3;
      buffer_sizes[0] += 2*num_nodes;
      buffer_sizes[0] += A_nnz + P_nnz;
      buffer_sizes[1] += A_nnz + P_nnz;
   }

   // Allocate and pack the buffer
   (*int_buffer) = hypre_CTAlloc(HYPRE_Int, buffer_sizes[0], HYPRE_MEMORY_HOST);
   (*complex_buffer) = hypre_CTAlloc(HYPRE_Complex, buffer_sizes[1], HYPRE_MEMORY_HOST);
   HYPRE_Int int_cnt = 0;
   HYPRE_Int complex_cnt = 0;
   for (level = transition_level; level < num_levels; level++)
   {
      // Get num nodes and num non zeros for matrices on this level
      hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(amg_data)[level];
      hypre_ParCSRMatrix *P;
      if (level != num_levels-1) P = hypre_ParAMGDataPArray(amg_data)[level];
      HYPRE_Int num_nodes = hypre_ParCSRMatrixNumRows(A);
      HYPRE_Int A_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(A) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(A) );
      HYPRE_Int P_nnz = 0;
      if (level != num_levels-1) P_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(P) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(P) ); 
      HYPRE_Int first_index = hypre_ParVectorFirstIndex( hypre_ParAMGDataUArray(amg_data)[level] );
      // Save the header data for this level
      (*int_buffer)[int_cnt++] = num_nodes;
      (*int_buffer)[int_cnt++] = A_nnz;
      if (level != num_levels-1) (*int_buffer)[int_cnt++] = P_nnz;
      // Save the matrix data for this level
      // !!! NOTE: This calls the get and restore row routinges a lot, which might be kind of inefficient... Is it ok or bad? !!!
      HYPRE_Int row_size;
      HYPRE_Int *row_col_ind;
      HYPRE_Complex *row_values;
      // Save the row ptr for A
      for (i = 0; i < num_nodes; i++)
      {
         hypre_ParCSRMatrixGetRow( A, first_index + i, &row_size, &row_col_ind, &row_values );
         (*int_buffer)[int_cnt++] = row_size;
         hypre_ParCSRMatrixRestoreRow( A, first_index + i, &row_size, &row_col_ind, &row_values );
      }
      // Save the col indices and values for A
      for (i = 0; i < num_nodes; i++)
      {
         hypre_ParCSRMatrixGetRow( A, first_index + i, &row_size, &row_col_ind, &row_values );
         for (j = 0; j < row_size; j++) (*int_buffer)[int_cnt++] = row_col_ind[j];
         for (j = 0; j < row_size; j++) (*complex_buffer)[complex_cnt++] = row_values[j];
         hypre_ParCSRMatrixRestoreRow( A, first_index + i, &row_size, &row_col_ind, &row_values );
      }
      if (level != num_levels-1)
      {
         // Save the row ptr for P
         for (i = 0; i < num_nodes; i++)
         {
            hypre_ParCSRMatrixGetRow( P, first_index + i, &row_size, &row_col_ind, &row_values );
            (*int_buffer)[int_cnt++] = row_size;
            hypre_ParCSRMatrixRestoreRow( P, first_index + i, &row_size, &row_col_ind, &row_values );
         }
         // Save the col indices and values for P
         for (i = 0; i < num_nodes; i++)
         {
            hypre_ParCSRMatrixGetRow( P, first_index + i, &row_size, &row_col_ind, &row_values );
            for (j = 0; j < row_size; j++) (*int_buffer)[int_cnt++] = row_col_ind[j];
            for (j = 0; j < row_size; j++) (*complex_buffer)[complex_cnt++] = row_values[j];
            hypre_ParCSRMatrixRestoreRow( P, first_index + i, &row_size, &row_col_ind, &row_values );
         }
      }
   }


   return 0;
}

HYPRE_Int
UnpackCoarseLevels(hypre_ParAMGData *amg_data, HYPRE_Int *recv_int_buffer, HYPRE_Complex *recv_complex_buffer, HYPRE_Int *max_buffer_sizes, HYPRE_Int transition_level)
{
   // Get MPI comm size
   HYPRE_Int   num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   // Loop over levels and generate new global comp grids 
   HYPRE_Int level,proc,i,j;
   HYPRE_Int max_res_buffer_size = 0;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

   // Get the global number of nodes and number of nonzeros for the matrices (read in from buffer)
   HYPRE_Int *res_num_recv_nodes = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
   HYPRE_Int *global_num_nodes = hypre_CTAlloc(HYPRE_Int, num_levels - transition_level, HYPRE_MEMORY_HOST);
   HYPRE_Int *global_A_nnz = hypre_CTAlloc(HYPRE_Int, num_levels - transition_level, HYPRE_MEMORY_HOST);
   HYPRE_Int *global_P_nnz = hypre_CTAlloc(HYPRE_Int, num_levels - transition_level, HYPRE_MEMORY_HOST);
   for (proc = 0; proc < num_procs; proc++)
   {
      HYPRE_Int int_cnt = max_buffer_sizes[0]*proc;
      for (level = transition_level; level < num_levels; level++)
      {
         // Read header info for this proc
         HYPRE_Int num_nodes = recv_int_buffer[int_cnt++];
         if (level == transition_level)
         {
            if (num_nodes > max_res_buffer_size) max_res_buffer_size = num_nodes;
            res_num_recv_nodes[proc] = num_nodes;
         }
         HYPRE_Int A_nnz = recv_int_buffer[int_cnt++];
         HYPRE_Int P_nnz = 0;
         if (level != num_levels-1) P_nnz = recv_int_buffer[int_cnt++];

         // Add the global totals
         global_num_nodes[level - transition_level] += num_nodes;
         global_A_nnz[level - transition_level] += A_nnz;
         if (level != num_levels-1) global_P_nnz[level - transition_level] += P_nnz;

         // Increment counter appropriately
         int_cnt += num_nodes + A_nnz;
         if (level != num_levels-1) int_cnt += num_nodes + P_nnz;
      }
   }

   // Create and allocate the comp grids
   for (level = transition_level; level < num_levels; level++)
   {
      hypre_ParCompGrid *compGrid = hypre_ParCompGridCreate();
      hypre_ParAMGDataCompGrid(amg_data)[level] = compGrid;
      hypre_ParCompGridSetSizeMatricesOnly(compGrid, global_num_nodes[level - transition_level], global_A_nnz[level - transition_level], global_P_nnz[level - transition_level]);
      hypre_ParCompGridNumOwnedNodes(compGrid) = hypre_ParCSRMatrixNumRows(hypre_ParAMGDataAArray(amg_data)[level]);
      hypre_ParCompGridNumRealNodes(compGrid) = global_num_nodes[level - transition_level];
   }

   // Now get matrix info from all processors from recv_buffer
   HYPRE_Int *globalRowCnt_start = hypre_CTAlloc(HYPRE_Int, num_levels - transition_level, HYPRE_MEMORY_HOST);
   HYPRE_Int *globalANnzCnt_start = hypre_CTAlloc(HYPRE_Int, num_levels - transition_level, HYPRE_MEMORY_HOST);
   HYPRE_Int *globalPNnzCnt_start = hypre_CTAlloc(HYPRE_Int, num_levels - transition_level, HYPRE_MEMORY_HOST);
   for (proc = 0; proc < num_procs; proc++)
   {
      HYPRE_Int int_cnt = max_buffer_sizes[0]*proc;
      HYPRE_Int complex_cnt = max_buffer_sizes[1]*proc;
      for (level = transition_level; level < num_levels; level++)
      {
         // Set the counters appropriately
         HYPRE_Int globalRowCnt = globalRowCnt_start[level - transition_level];
         HYPRE_Int globalANnzCnt = globalANnzCnt_start[level - transition_level];
         HYPRE_Int globalPNnzCnt = globalPNnzCnt_start[level - transition_level];


         // Get the comp grid
         hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];


         // Read header info for this proc
         HYPRE_Int num_nodes = recv_int_buffer[int_cnt++];
         HYPRE_Int A_nnz = recv_int_buffer[int_cnt++];
         HYPRE_Int P_nnz = 0;
         if (level != num_levels-1) P_nnz = recv_int_buffer[int_cnt++];



         // Read in A row sizes and update row ptr
         if (proc == 0) hypre_ParCompGridARowPtr(compGrid)[globalRowCnt++] = 0;
         for (i = 0; i < num_nodes; i++)
         {
            hypre_ParCompGridARowPtr(compGrid)[globalRowCnt] = hypre_ParCompGridARowPtr(compGrid)[globalRowCnt-1] + recv_int_buffer[int_cnt++];
            globalRowCnt++;
         }
         // Read in A col indices
         for (i = 0; i < A_nnz; i++)
         {
            hypre_ParCompGridAColInd(compGrid)[globalANnzCnt++] = recv_int_buffer[int_cnt++];
         }
         globalANnzCnt = globalANnzCnt_start[level - transition_level];
         // Read in A values
         for (i = 0; i < A_nnz; i++)
         {
            hypre_ParCompGridAData(compGrid)[globalANnzCnt++] = recv_complex_buffer[complex_cnt++];
         }

         if (level != num_levels-1)
         {
            // Read in P row sizes and update row ptr
            globalRowCnt = globalRowCnt_start[level - transition_level];
            if (proc == 0) hypre_ParCompGridPRowPtr(compGrid)[globalRowCnt++] = 0;
            for (i = 0; i < num_nodes; i++)
            {
               hypre_ParCompGridPRowPtr(compGrid)[globalRowCnt] = hypre_ParCompGridPRowPtr(compGrid)[globalRowCnt-1] + recv_int_buffer[int_cnt++];
               globalRowCnt++;
            }
            // Read in P col indices
            for (i = 0; i < P_nnz; i++)
            {
               hypre_ParCompGridPColInd(compGrid)[globalPNnzCnt++] = recv_int_buffer[int_cnt++];
            }
            globalPNnzCnt = globalPNnzCnt_start[level - transition_level];
            // Read in A values
            for (i = 0; i < P_nnz; i++)
            {
               hypre_ParCompGridPData(compGrid)[globalPNnzCnt++] = recv_complex_buffer[complex_cnt++];
            }
         }

         // Update counters 
         globalRowCnt_start[level - transition_level] = globalRowCnt;
         globalANnzCnt_start[level - transition_level] = globalANnzCnt;
         globalPNnzCnt_start[level - transition_level] = globalPNnzCnt;
      }
   }

   hypre_ParCompGridCommPkgMaxResidualBufferSize(hypre_ParAMGDataCompGridCommPkg(amg_data)) = max_res_buffer_size;
   hypre_ParCompGridCommPkgResNumRecvNodes(hypre_ParAMGDataCompGridCommPkg(amg_data)) = res_num_recv_nodes;

   hypre_TFree(global_num_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(global_A_nnz, HYPRE_MEMORY_HOST);
   hypre_TFree(global_P_nnz, HYPRE_MEMORY_HOST);
   hypre_TFree(globalRowCnt_start, HYPRE_MEMORY_HOST);
   hypre_TFree(globalANnzCnt_start, HYPRE_MEMORY_HOST);
   hypre_TFree(globalPNnzCnt_start, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Complex*
PackSendBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, 
   HYPRE_Int *send_flag_buffer_size, HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
   HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int padding, HYPRE_Int num_ghost_layers, HYPRE_Int send_rank, HYPRE_Int *bandwidth_cost )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   
   HYPRE_Int            level,i,j,cnt,row_length,send_elmt,coarse_grid_index;
   HYPRE_Int            nodes_to_add = 0;
   HYPRE_Int            **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            num_psi_levels = 1;

   // Get where to look in commPkgSendMapElmts
   HYPRE_Int            start = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][processor];
   HYPRE_Int            finish = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][processor+1];

   // Get the transition level
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg);
   if (transition_level < 0) transition_level = num_levels;

   // initialize send map buffer size
   (*send_flag_buffer_size) = num_levels - current_level;

   // see whether we need coarse info and allcoate the add_flag array on next level if appropriate
   if (current_level != transition_level-1) add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[current_level+1]), HYPRE_MEMORY_HOST );

   // Mark the nodes to send (including Psi_c grid plus ghost nodes)

   // Start by adding the nodes listed by the compGridCommPkg on this level and their coarse grid counterparts if applicable
   // Note that the compGridCommPkg is set up to list all nodes within the padding plus ghost layers
   (*send_flag_buffer_size) += finish - start;
   if (current_level != transition_level-1)
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
   num_send_nodes[current_level][processor][current_level] = finish - start;
   send_flag[current_level][processor][current_level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level][processor][current_level], HYPRE_MEMORY_HOST );
   if (bandwidth_cost) bandwidth_cost[5*current_level + 4] += num_send_nodes[current_level][processor][current_level];

   (*buffer_size) += 2;
   if (current_level != transition_level-1) (*buffer_size) += 2*num_send_nodes[current_level][processor][current_level];
   else (*buffer_size) += num_send_nodes[current_level][processor][current_level];

   for (i = start; i < finish; i++)
   {
      send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];
      send_flag[current_level][processor][current_level][i - start] = send_elmt;
      (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(compGrid[current_level])[send_elmt]) + 1;
      if (current_level != num_levels-1) (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridPRows(compGrid[current_level])[send_elmt]) + 1;
   }

   // Now build out the psi_c composite grid (along with required ghost nodes) on coarser levels
   for (level = current_level + 1; level < transition_level; level++)
   {
      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         num_psi_levels++;
         (*buffer_size)++;
         nodes_to_add = 0;

         // if we need coarse info, allocate space for the add flag on the next level
         if (level != transition_level-1) add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]), HYPRE_MEMORY_HOST );

         // Expand by the padding on this level and add coarse grid counterparts if applicable
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] == padding + 1)
            {
               // Recursively add the region of padding (flagging coarse nodes on the next level if applicable)
               if (level != transition_level-1) RecursivelyBuildPsiComposite(i, padding, compGrid[level], add_flag[level], add_flag[level+1], 1, &nodes_to_add, padding);
               else RecursivelyBuildPsiComposite(i, padding, compGrid[level], add_flag[level], NULL, 0, &nodes_to_add, padding);
            }
         }

         // Expand by the number of ghost layers 
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 1) add_flag[level][i] = num_ghost_layers + 2;
            else if (add_flag[level][i] == 1) add_flag[level][i] = num_ghost_layers + 1;
         }
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            // Recursively add the region of ghost nodes (do not add any coarse nodes underneath)
            if (add_flag[level][i] == num_ghost_layers + 1) RecursivelyBuildPsiComposite(i, num_ghost_layers, compGrid[level], add_flag[level], NULL, 0, NULL, 0);
         }

         // Count up the total number of send nodes (note: this will change after we check for redundancy)
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 0)
            {
               num_send_nodes[current_level][processor][level]++;
            }
         }
         if (bandwidth_cost) bandwidth_cost[5*current_level + 4] += num_send_nodes[current_level][processor][level];

         // Save the indices (in global index ordering) so I don't have to keep looping over all nodes in compGrid when I pack the buffer
         send_flag[current_level][processor][level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level][processor][level], HYPRE_MEMORY_HOST );
         cnt =  0;
         HYPRE_Int insert_owned_position;
         if (hypre_ParCompGridNumOwnedNodes(compGrid[level]))
         {
            HYPRE_Int first_owned = hypre_ParCompGridGlobalIndices(compGrid[level])[0];
            HYPRE_Int last_owned = hypre_ParCompGridGlobalIndices(compGrid[level])[ hypre_ParCompGridNumOwnedNodes(compGrid[level]) - 1 ];
            HYPRE_Int first_nonowned = hypre_ParCompGridGlobalIndices(compGrid[level])[ hypre_ParCompGridNumOwnedNodes(compGrid[level]) ];
            HYPRE_Int last_nonowned = hypre_ParCompGridGlobalIndices(compGrid[level])[ hypre_ParCompGridNumNodes(compGrid[level]) - 1 ];

            // Find where to insert owned nodes in the list of all comp grid nodes (such that they are ordered according to global index)
            if (last_owned < first_nonowned) insert_owned_position = hypre_ParCompGridNumOwnedNodes(compGrid[level]);
            else if (first_owned > last_nonowned) insert_owned_position = hypre_ParCompGridNumNodes(compGrid[level]);
            else
            {
               // Binary search to find where to insert
               insert_owned_position = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level], first_owned, 1);
            }
         }
         else insert_owned_position = 0;

         // Generate the send_flag in global index ordering
         for (i = hypre_ParCompGridNumOwnedNodes(compGrid[level]); i < insert_owned_position; i++)
         {
            if (add_flag[level][i] > 0) 
            {
               send_flag[current_level][processor][level][cnt++] = i;
            }
         }
         for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 0) 
            {
               send_flag[current_level][processor][level][cnt++] = i;
            }
         }
         for (i = insert_owned_position; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 0) 
            {
               send_flag[current_level][processor][level][cnt++] = i;
            }
         }

         // Check the send flag generated above against the send flags on previous outer levels
         for (i = current_level+1; i < transition_level; i++)
         {
            for (j = 0; j < hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[i]; j++)
            {
               if (hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[i][j] == send_rank)
               {
                  // Compare the send_flag from the previous level to this send_flag. Note that send_flags are sorted by global index, so can do something similar to a merge here
                  HYPRE_Int old_cnt = 0;
                  HYPRE_Int new_cnt = 0;
                  HYPRE_Int actual_num_send_nodes = 0;
                  while (old_cnt < num_send_nodes[i][j][level] && new_cnt < num_send_nodes[current_level][processor][level])
                  {
                     HYPRE_Int old_global_index;
                     if (send_flag[i][j][level][old_cnt] >= 0)
                        old_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[ send_flag[i][j][level][old_cnt] ];
                     else
                        old_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[ -(send_flag[i][j][level][old_cnt]+1) ];
                     HYPRE_Int new_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[ send_flag[current_level][processor][level][new_cnt] ];
                     if (old_global_index > new_global_index)
                     {
                        send_flag[current_level][processor][level][ actual_num_send_nodes++ ] = send_flag[current_level][processor][level][ new_cnt++ ];
                     }
                     else if (old_global_index < new_global_index)
                     {
                        old_cnt++;
                     }
                     else if (old_global_index == new_global_index)
                     {
                        old_cnt++;
                        new_cnt++;
                     }
                  }
                  while (new_cnt < num_send_nodes[current_level][processor][level])
                  {
                     send_flag[current_level][processor][level][ actual_num_send_nodes++ ] = send_flag[current_level][processor][level][ new_cnt++ ];
                  }
                  num_send_nodes[current_level][processor][level] = actual_num_send_nodes;
                  send_flag[current_level][processor][level] = hypre_TReAlloc(send_flag[current_level][processor][level], HYPRE_Int, actual_num_send_nodes, HYPRE_MEMORY_HOST);
               }
            }
         }

         // Count up the buffer sizes
         (*send_flag_buffer_size) += num_send_nodes[current_level][processor][level];
         if (level != transition_level-1) (*buffer_size) += 2*num_send_nodes[current_level][processor][level];
         else (*buffer_size) += num_send_nodes[current_level][processor][level];
         for (i = 0; i < num_send_nodes[current_level][processor][level]; i++)
         {
            (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(compGrid[level])[ send_flag[current_level][processor][level][i] ]) + 1;
            if (level != num_levels-1) (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridPRows(compGrid[level])[ send_flag[current_level][processor][level][i] ]) + 1;
         }
      }
      else break;
   }

   // Allocate the buffer
   HYPRE_Complex *send_buffer = hypre_CTAlloc(HYPRE_Complex, (*buffer_size), HYPRE_MEMORY_HOST);

   // Pack the buffer
   cnt = 0;
   send_buffer[cnt++] = (HYPRE_Complex) num_psi_levels;
   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      // store the number of nodes on this level
      send_buffer[cnt++] = (HYPRE_Complex) num_send_nodes[current_level][processor][level];

      // copy all global indices
      for (i = 0; i < num_send_nodes[current_level][processor][level]; i++)
      {
         send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompGridGlobalIndices(compGrid[level])[ send_flag[current_level][processor][level][i] ];
      }

      // if not on last level, copy coarse gobal indices
      if (level != transition_level-1)
      {
         for (i = 0; i < num_send_nodes[current_level][processor][level]; i++)
         {
            send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ send_flag[current_level][processor][level][i] ];
         }
      }

      // now loop over matrix rows
      for (i = 0; i < num_send_nodes[current_level][processor][level]; i++)
      {
         // store the row length for matrix A
         row_length = hypre_ParCompMatrixRowSize( hypre_ParCompGridARows( compGrid[level] )[ send_flag[current_level][processor][level][i] ] );
         send_buffer[cnt++] = (HYPRE_Complex) row_length;
         
         // copy matrix entries for matrix A
         for (j = 0; j < row_length; j++)
         {
            send_buffer[cnt++] = hypre_ParCompMatrixRowData( hypre_ParCompGridARows( compGrid[level] )[ send_flag[current_level][processor][level][i] ] )[j];
         }
         // copy global indices for matrix A
         for (j = 0; j < row_length; j++)
         {
            send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( compGrid[level] )[ send_flag[current_level][processor][level][i] ] )[j];
         }

         if (level != num_levels-1)
            {
            // store the row length for matrix P
            row_length = hypre_ParCompMatrixRowSize( hypre_ParCompGridPRows( compGrid[level] )[ send_flag[current_level][processor][level][i] ] );
            send_buffer[cnt++] = (HYPRE_Complex) row_length;

            // copy matrix entries for matrix P
            for (j = 0; j < row_length; j++)
            {
               send_buffer[cnt++] = hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( compGrid[level] )[ send_flag[current_level][processor][level][i] ] )[j];
            }
            // copy global indices for matrix P
            for (j = 0; j < row_length; j++)
            {
               send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( compGrid[level] )[ send_flag[current_level][processor][level][i] ] )[j];
            }
         }
      }
   }

   // Clean up memory
   for (level = 0; level < transition_level; level++)
   {
      if (add_flag[level]) hypre_TFree(add_flag[level], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);

   // Return the send buffer
   return send_buffer;
}

HYPRE_Int
RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int *add_flag_coarse, 
   HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding)
{
   HYPRE_Int         i,index,coarse_grid_index;
   hypre_ParCompMatrixRow *A_row = hypre_ParCompGridARows(compGrid)[node];
   HYPRE_Int error_code = 0;

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
            if (m-1 > 0) error_code = RecursivelyBuildPsiComposite(index, m-1, compGrid, add_flag, add_flag_coarse, need_coarse_info, nodes_to_add, padding);
         }
         // If m = 1, we won't do another recursive call, so make sure to flag the coarse grid here if applicable
         if (need_coarse_info && m == 1)
         {
            coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid)[index];
            if ( coarse_grid_index != -1 ) 
            {
               // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
               add_flag_coarse[ coarse_grid_index ] = padding+1;
               *nodes_to_add = 1;
            }
         }
      }
      else
      {
         error_code = 1; 
         HYPRE_Int myid;
         hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
         if (myid == 1) hypre_printf("Error! Ran into a -1 index when building Psi_c\n");
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

   return error_code;
}

HYPRE_Int
UnpackRecvBuffer( HYPRE_Complex *recv_buffer, hypre_ParCompGrid **compGrid, 
      HYPRE_Int *num_sends, HYPRE_Int *num_recvs,
      HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
      HYPRE_Int ****recv_map, HYPRE_Int ***recv_map_send, 
      HYPRE_Int ***num_recv_nodes, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int transition_level,
      HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index, HYPRE_Int *num_added_nodes, HYPRE_Int **num_incoming_nodes, HYPRE_Int buffer_number, HYPRE_Int recv_rank )
{

   HYPRE_Int            level, i, j, k;
   HYPRE_Int            num_psi_levels, row_size, offset, level_start, global_index, add_node_cnt;
   // HYPRE_Int            *add_flag;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // initialize the counter
   HYPRE_Int            cnt = 0;

   // get the number of levels received
   num_psi_levels = (HYPRE_Int) recv_buffer[cnt++];

   // Init the recv_map_send_buffer_size !!! I think this can just be set a priori instead of counting it up in this function... !!!
   *recv_map_send_buffer_size = num_levels - current_level;

   // loop over coarser psi levels
   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      // get the number of nodes on this level
      num_incoming_nodes[buffer_number][level] = (HYPRE_Int) recv_buffer[cnt++];
      level_start = cnt;
      *recv_map_send_buffer_size += num_incoming_nodes[buffer_number][level];

      // Incoming nodes and existing (non-owned) nodes in the comp grid are both sorted by global index, so here we merge these lists together (getting rid of redundant nodes along the way)
      add_node_cnt = 0;
      HYPRE_Int num_owned_nodes = hypre_ParCompGridNumOwnedNodes(compGrid[level]);
      HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
      HYPRE_Int num_nonowned_nodes = num_nodes - num_owned_nodes;
      HYPRE_Int dest = num_owned_nodes;
      HYPRE_Int compGrid_cnt = 0;
      HYPRE_Int incoming_cnt = 0;
      HYPRE_Int *compGrid_dest = hypre_CTAlloc(HYPRE_Int, num_nonowned_nodes, HYPRE_MEMORY_HOST);
      HYPRE_Int *incoming_dest = hypre_CTAlloc(HYPRE_Int, num_incoming_nodes[buffer_number][level], HYPRE_MEMORY_HOST);

      while (incoming_cnt < num_incoming_nodes[buffer_number][level] && compGrid_cnt < num_nonowned_nodes)
      {
         HYPRE_Int incoming_global_index = (HYPRE_Int) recv_buffer[cnt];
         HYPRE_Int compGrid_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[ compGrid_cnt + num_owned_nodes ];
         if (incoming_global_index >= proc_first_index[level] && incoming_global_index <= proc_last_index[level])
         {
            incoming_dest[incoming_cnt++] = -1;
            cnt++;
         }
         else if (incoming_global_index == compGrid_global_index)
         {
            incoming_dest[incoming_cnt++] = -1;
            cnt++;
         }
         else if (incoming_global_index < compGrid_global_index)
         {
            incoming_dest[incoming_cnt++] = dest++;
            cnt++;
            add_node_cnt++;
         }
         else
         {
            compGrid_dest[compGrid_cnt++] = dest++;
         }
      }
      while (incoming_cnt < num_incoming_nodes[buffer_number][level])
      {
         HYPRE_Int incoming_global_index = (HYPRE_Int) recv_buffer[cnt];
         if (incoming_global_index >= proc_first_index[level] && incoming_global_index <= proc_last_index[level])
         {
            incoming_dest[incoming_cnt++] = -1;
            cnt++;
         }
         else
         {
            incoming_dest[incoming_cnt++] = dest++;
            add_node_cnt++;
            cnt++;
         }
      }
      while (compGrid_cnt < num_nonowned_nodes)
      {
         compGrid_dest[compGrid_cnt++] = dest++;
      }
      num_added_nodes[level] += add_node_cnt;

      // Set recv_map_send to incoming_dest
      recv_map_send[buffer_number][level] = incoming_dest;

      // if necessary, reallocate more space for compGrid
      offset = hypre_ParCompGridNumNodes(compGrid[level]);
      if (add_node_cnt + offset > hypre_ParCompGridMemSize(compGrid[level])) 
         hypre_ParCompGridResize(compGrid[level], add_node_cnt + offset, level != num_levels-1); // !!! Is there a better way to manage memory? !!!

      // Starting at the end of the list (to avoid overwriting info we want to access later), copy existing comp grid info to its new positions
      for (i = num_nonowned_nodes - 1; i >= 0; i--)
         hypre_ParCompGridGlobalIndices(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridGlobalIndices(compGrid[level])[i + num_owned_nodes];
      for (i = num_nonowned_nodes - 1; i >= 0; i--)
      {
         hypre_ParCompGridARows(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridARows(compGrid[level])[i + num_owned_nodes];
      }
      if (level != transition_level-1)
      {
         for (i = num_nonowned_nodes - 1; i >= 0; i--)
            hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[i + num_owned_nodes];
         for (i = num_nonowned_nodes - 1; i >= 0; i--)
            hypre_ParCompGridCoarseLocalIndices(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i + num_owned_nodes];
      }
      if (level != num_levels-1)
      {
         for (i = num_nonowned_nodes - 1; i >= 0; i--)
         {
            hypre_ParCompGridPRows(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridPRows(compGrid[level])[i + num_owned_nodes];
         }
      }

      // Fix up the send_flag and recv_map from previous levels
      for (i = current_level; i < num_levels; i++)
      {
         for (j = 0; j < num_sends[i]; j++)
         {
            for (k = 0; k < num_send_nodes[i][j][level]; k++)
            {
               if (send_flag[i][j][level][k] >= hypre_ParCompGridNumOwnedNodes(compGrid[level]))
                  send_flag[i][j][level][k] = compGrid_dest[ send_flag[i][j][level][k] - hypre_ParCompGridNumOwnedNodes(compGrid[level]) ];
               else if (-(send_flag[i][j][level][k]+1) >= hypre_ParCompGridNumOwnedNodes(compGrid[level]))
                  send_flag[i][j][level][k] = -(compGrid_dest[ -(send_flag[i][j][level][k]+1) - hypre_ParCompGridNumOwnedNodes(compGrid[level]) ] + 1);
            }
         }
      }
      for (i = current_level+1; i < num_levels; i++)
      {
         for (j = 0; j < num_recvs[i]; j++)
         {
            for (k = 0; k < num_recv_nodes[i][j][level]; k++)
            {
               if (recv_map[i][j][level][k] >= hypre_ParCompGridNumOwnedNodes(compGrid[level]))
                  recv_map[i][j][level][k] = compGrid_dest[ recv_map[i][j][level][k] - hypre_ParCompGridNumOwnedNodes(compGrid[level]) ];
            }
         }
      }
      for (i = 0; i < buffer_number; i++)
      {
         if (recv_map_send[i][level])
         {
            for (k = 0; k < num_incoming_nodes[i][level]; k++)
            {
               if (recv_map_send[i][level][k] >= hypre_ParCompGridNumOwnedNodes(compGrid[level]))
                  recv_map_send[i][level][k] = compGrid_dest[ recv_map_send[i][level][k] - hypre_ParCompGridNumOwnedNodes(compGrid[level]) ];
            }
         }
      }

      // Now copy in the new nodes to their appropriate positions
      cnt = level_start;
      for (i = 0; i < num_incoming_nodes[buffer_number][level]; i++) 
      {   
         if (incoming_dest[i] >= 0)
         {
            hypre_ParCompGridGlobalIndices(compGrid[level])[ incoming_dest[i] ] = (HYPRE_Int) recv_buffer[cnt];
            num_recv_nodes[current_level][buffer_number][level]++;
         }
         cnt++;
      }
      if (level != transition_level-1)
      {
         for (i = 0; i < num_incoming_nodes[buffer_number][level]; i++) 
         {   
            if (incoming_dest[i] >= 0) hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ incoming_dest[i] ] = (HYPRE_Int) recv_buffer[cnt];
            cnt++;
         }
      }
      for (i = 0; i < num_incoming_nodes[buffer_number][level]; i++)
      {
         if (incoming_dest[i] >= 0)
         {
            // get the row length of matrix A
            row_size = (HYPRE_Int) recv_buffer[cnt++];
            // Create row and allocate space
            hypre_ParCompGridARows(compGrid[level])[ incoming_dest[i] ] = hypre_ParCompMatrixRowCreate();
            hypre_ParCompMatrixRowSize( hypre_ParCompGridARows( compGrid[level] )[ incoming_dest[i] ] ) = row_size;
            hypre_ParCompMatrixRowData( hypre_ParCompGridARows( compGrid[level] )[ incoming_dest[i] ] ) = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
            hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( compGrid[level] )[ incoming_dest[i] ] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
            hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridARows( compGrid[level] )[ incoming_dest[i] ] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);

            // copy matrix entries of matrix A
            for (j = 0; j < row_size; j++)
            {
               hypre_ParCompMatrixRowData( hypre_ParCompGridARows( compGrid[level] )[ incoming_dest[i] ] )[j] = recv_buffer[cnt++];
            }
            // copy global indices of matrix A
            for (j = 0; j < row_size; j++)
            {
               hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( compGrid[level] )[ incoming_dest[i] ] )[j] = (HYPRE_Int) recv_buffer[cnt++];
            }
            // If not on coarsest level, do the same for P
            if (level != num_levels-1)
            {
               // get the row length of matrix P
               row_size = (HYPRE_Int) recv_buffer[cnt++];
               // Create row and allocate space
               hypre_ParCompGridPRows(compGrid[level])[ incoming_dest[i] ] = hypre_ParCompMatrixRowCreate();
               hypre_ParCompMatrixRowSize( hypre_ParCompGridPRows( compGrid[level] )[ incoming_dest[i] ] ) = row_size;
               hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( compGrid[level] )[ incoming_dest[i] ] ) = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
               hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( compGrid[level] )[ incoming_dest[i] ] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
               hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridPRows( compGrid[level] )[ incoming_dest[i] ] ) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
               // copy matrix entries of matrix P
               for (j = 0; j < row_size; j++)
               {
                  hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( compGrid[level] )[ incoming_dest[i] ] )[j] = recv_buffer[cnt++];
               }
               // copy global indices of matrix P
               for (j = 0; j < row_size; j++)
               {
                  hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( compGrid[level] )[ incoming_dest[i] ] )[j] = (HYPRE_Int) recv_buffer[cnt++];
               }
            }
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

      // clean up memory
      hypre_TFree(compGrid_dest, HYPRE_MEMORY_HOST);
   }

   return 0;
}

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_send_buffer, HYPRE_Int *num_incoming_nodes, HYPRE_Int current_level, HYPRE_Int num_levels)
{
   HYPRE_Int      level, i, cnt;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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
UnpackSendFlagBuffer(HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels)
{
   HYPRE_Int      level, i, cnt, num_nodes;
   cnt = 0;
   *send_buffer_size = 0;
   for (level = current_level; level < num_levels; level++)
   {
      num_nodes = send_flag_buffer[cnt++];

      for (i = 0; i < num_nodes; i++)
      {
         if (send_flag_buffer[cnt++] == -1) 
         {
            // Mark indices of nodes in send_flag that will not be sent in the future by mapping them to negative indices
            send_flag[level][i] = -(send_flag[level][i] + 1);
         }
         else (*send_buffer_size)++;
      }
   }

   return 0;
}


HYPRE_Int
FinalizeSendFlag(HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes, HYPRE_Int *num_sends, HYPRE_Int num_levels)
{
   HYPRE_Int      level, proc, inner_lvl, i, cnt;
   HYPRE_Int      num_nodes;

   cnt = 0;
   for (level = 0; level < num_levels; level++)
   {
      for (proc = 0; proc < num_sends[level]; proc++)
      {
         for (inner_lvl = level; inner_lvl < num_levels; inner_lvl++)
         {
            num_nodes = num_send_nodes[level][proc][inner_lvl];
            num_send_nodes[level][proc][inner_lvl] = 0;

            for (i = 0; i < num_nodes; i++)
            {
               if (send_flag[level][proc][inner_lvl][i] >= 0) 
               {
                  // discard indices of nodes in send_flag that will not be sent in the future and count the send buffer size
                  send_flag[level][proc][inner_lvl][ num_send_nodes[level][proc][inner_lvl]++ ] = send_flag[level][proc][inner_lvl][i];
               }
            }
            send_flag[level][proc][inner_lvl] = hypre_TReAlloc(send_flag[level][proc][inner_lvl], HYPRE_Int, num_send_nodes[level][proc][inner_lvl], HYPRE_MEMORY_HOST);
         }
      }
   }

   return 0;
}

HYPRE_Int
TestCompGrids1(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int transition_level, HYPRE_Int padding, HYPRE_Int num_ghost_layers, HYPRE_Int current_level, HYPRE_Int check_ghost_info)
{
   // TEST 1: See whether the parallel composite grid algorithm algorithm has constructed a composite grid with 
   // the same shape (and ghost node info) as we expect from serial, top-down composite grid generation
   HYPRE_Int            level,i;
   HYPRE_Int            need_coarse_info, nodes_to_add = 1;
   HYPRE_Int            **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            test_failed = 0;
   HYPRE_Int            error_code;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   // Allocate add_flag on each level and mark the owned dofs on the finest grid
   for (level = 0; level < transition_level; level++) add_flag[level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level]), HYPRE_MEMORY_HOST);
   for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid[current_level]); i++) add_flag[current_level][i] = padding + 1;

   // Serially generate comp grid from top down
   // Note that if nodes that should be present in the comp grid are not found, we will be alerted by the error message in RecursivelyBuildPsiComposite()
   for (level = current_level; level < transition_level; level++)
   {
      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         nodes_to_add = 0;

         // see whether we need coarse info on this level
         if (level != transition_level-1) need_coarse_info = 1;
         else need_coarse_info = 0;

         // Expand by the padding on this level and add coarse grid counterparts if applicable
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] == padding + 1)
            {
               // Recursively add the region of padding (flagging coarse nodes on the next level if applicable)
               if (need_coarse_info) error_code = RecursivelyBuildPsiComposite(i, padding, compGrid[level], add_flag[level], add_flag[level+1], need_coarse_info, &nodes_to_add, padding);
               else error_code = RecursivelyBuildPsiComposite(i, padding, compGrid[level], add_flag[level], NULL, need_coarse_info, &nodes_to_add, padding);
               if (error_code) test_failed = 1;
            }
         }

         // Expand by the number of ghost layers 
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 1) add_flag[level][i] = num_ghost_layers + 2;
            else if (add_flag[level][i] == 1) add_flag[level][i] = num_ghost_layers + 1;
         }
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            // Recursively add the region of ghost nodes (do not add any coarse nodes underneath)
            if (add_flag[level][i] == num_ghost_layers + 1) error_code = RecursivelyBuildPsiComposite(i, num_ghost_layers, compGrid[level], add_flag[level], NULL, 0, NULL, 0);
            if (error_code) test_failed = 1;
         }
      }
      else break;

      // Check whether add_flag has any zeros (zeros indicate that we have extra nodes in the comp grid that don't belong)
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) if (add_flag[level][i] == 0) 
      {
         test_failed = 1;
         if (myid == 1) printf("Error: extra nodes present in comp grid\n");
      }

      // Check to make sure we have the correct identification of ghost nodes
      if (level != transition_level-1 && check_ghost_info)
      {
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) 
         {
            if (add_flag[level][i] < num_ghost_layers + 1 && hypre_ParCompGridRealDofMarker(compGrid[level])[i] != 0) 
            {
               test_failed = 1;
               if (myid == 1) printf("Error: dof that should have been marked as ghost was marked as real\n");
            }
            if (add_flag[level][i] > num_ghost_layers && hypre_ParCompGridRealDofMarker(compGrid[level])[i] != 1) 
            {
               test_failed = 1;
               if (myid == 1) printf("Error: dof that should have been marked as real was marked as ghost\n");
            }
         }
      }
   }

   return test_failed;
}

HYPRE_Int
TestCompGrids2(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels)
{
   // TEST 2: See if the composite grid is set up such that restriction can occur correctly
   // The CoarseResidualMarker shows where we have all the required info to restrict a correct residual
   // Here we mark the locations where a restricted residual (rather than a residual just recalculated on the coarse grid) is REQUIRED (i.e. where is the coarse grid residual affected by fine grid relaxation)
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   HYPRE_Int            level,i,j;
   HYPRE_Int test_failed = 0;
   for (level = 0; level < num_levels-1; level++)
   { 
      HYPRE_Int *needs_restrict = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]), HYPRE_MEMORY_HOST);
      // For dof in the comp grid
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
      {
         // Look at the row of A for dof
         HYPRE_Int find_restrict_range = 0;
         for (j = hypre_ParCompGridARowPtr(compGrid[level])[i]; j < hypre_ParCompGridARowPtr(compGrid[level])[i+1]; j++)
         {
            // If dof is connected through A to a real node
            if (hypre_ParCompGridAColInd(compGrid[level])[j] >= 0)
            {
               if (hypre_ParCompGridRealDofMarker(compGrid[level])[ hypre_ParCompGridAColInd(compGrid[level])[j] ])
               {
                  // Then the residual here will change, so find where that residual will propogate on the coarse grid
                  find_restrict_range = 1;
                  break;
               }
            }
         }
         // If dof was connected to a real node
         if (find_restrict_range)
         {
            // Look at the row of P for dof
            for (j = hypre_ParCompGridPRowPtr(compGrid[level])[i]; j < hypre_ParCompGridPRowPtr(compGrid[level])[i+1]; j++)
            {
               // Mark everything in the restriction range of dof
               needs_restrict[ hypre_ParCompGridPColInd(compGrid[level])[j] ] = 1;
            }
         }
      }
      // Now check against the coarse residual marker
      // That is, coarse residual marker shows where we CAN restrict a correct residual and needs_restrict shows where we NEED to restrict a correct residual
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level+1]); i++)
      {
         if (needs_restrict[i])
         {
            if (hypre_ParCompGridCoarseResidualMarker(compGrid[level+1])[i] != 2)
            {
               test_failed = 1;
               // printf("Error: Need residual to be restricted at a location where it is not possible: proc %d, level %d, global index %d\n", myid, level+1, hypre_ParCompGridGlobalIndices(compGrid[level+1])[i]);
            }
         }
      }
      hypre_TFree(needs_restrict, HYPRE_MEMORY_HOST);
   }

   return test_failed;
}

HYPRE_Int
TestCompGrids3(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, hypre_ParCSRMatrix **A, hypre_ParCSRMatrix **P, hypre_ParVector **F)
{
   // TEST 3: See whether the dofs in the composite grid have the correct info.
   // Each processor in turn will broadcast out the info associate with its composite grids on each level.
   // The processors owning the original info will check to make sure their info matches the comp grid info that was broadcasted out.
   // This occurs for the matrix info (row pointer, column indices, and data for A and P) and the initial right-hand side 
   
   // Get MPI info
   HYPRE_Int myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   HYPRE_Int test_failed = 0;

   // For each processor and each level broadcast the residual data and global indices out and check agains the owning procs
   HYPRE_Int proc;
   for (proc = 0; proc < num_procs; proc++)
   {
      HYPRE_Int level;
      for (level = 0; level < num_levels; level++)
      {
         // Broadcast the number of nodes and num non zeros for A and P
         HYPRE_Int num_nodes = 0;
         HYPRE_Int num_owned_nodes = 0;
         HYPRE_Int num_coarse_nodes = 0;
         HYPRE_Int num_coarse_owned_nodes = 0;
         HYPRE_Int nnz_A = 0;
         HYPRE_Int nnz_P = 0;
         HYPRE_Int sizes_buf[6];
         HYPRE_Int i;
         if (myid == proc) 
         {
            num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
            num_owned_nodes = hypre_ParCompGridNumOwnedNodes(compGrid[level]);
            nnz_A = hypre_ParCompGridARowPtr(compGrid[level])[num_nodes];
            if (level != num_levels-1)
            {
               num_coarse_nodes = hypre_ParCompGridNumNodes(compGrid[level+1]);
               num_coarse_owned_nodes = hypre_ParCompGridNumOwnedNodes(compGrid[level+1]);
               nnz_P = hypre_ParCompGridPRowPtr(compGrid[level])[num_nodes];
            }
            else
            {
               num_coarse_nodes = 0;
               num_coarse_owned_nodes = 0;
               nnz_P = 0;
            }
            sizes_buf[0] = num_nodes;
            sizes_buf[1] = num_owned_nodes;
            sizes_buf[2] = num_coarse_nodes;
            sizes_buf[3] = num_coarse_owned_nodes;
            sizes_buf[4] = nnz_A;
            sizes_buf[5] = nnz_P;
         }
         hypre_MPI_Bcast(sizes_buf, 6, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);
         num_nodes = sizes_buf[0];
         num_owned_nodes = sizes_buf[1];
         num_coarse_nodes = sizes_buf[2];
         num_coarse_owned_nodes = sizes_buf[3];
         nnz_A = sizes_buf[4];
         nnz_P = sizes_buf[5];

         // Broadcast the global indices
         HYPRE_Int *global_indices;
         if (myid == proc) global_indices = hypre_ParCompGridGlobalIndices(compGrid[level]);
         else global_indices = hypre_CTAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(global_indices, num_nodes, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);


         // Broadcast the A row pointer
         HYPRE_Int *A_rowPtr;
         if (myid == proc) A_rowPtr = hypre_ParCompGridARowPtr(compGrid[level]);
         else A_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nodes+1, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(A_rowPtr, num_nodes+1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the A column indices
         HYPRE_Int *A_colInd;
         if (myid == proc) A_colInd = hypre_ParCompGridAColInd(compGrid[level]);
         else A_colInd = hypre_CTAlloc(HYPRE_Int, nnz_A, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(A_colInd, nnz_A, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the A data
         HYPRE_Complex *A_data;
         if (myid == proc) A_data = hypre_ParCompGridAData(compGrid[level]);
         else A_data = hypre_CTAlloc(HYPRE_Complex, nnz_A, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(A_data, nnz_A, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);

         HYPRE_Int *coarse_global_indices;
         HYPRE_Int *P_rowPtr;
         HYPRE_Int *P_colInd;
         HYPRE_Complex *P_data;
         if (level != num_levels-1)
         {
            // Broadcast the coarse global indices
            if (myid == proc) coarse_global_indices = hypre_ParCompGridGlobalIndices(compGrid[level+1]);
            else coarse_global_indices = hypre_CTAlloc(HYPRE_Int, num_coarse_nodes, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(coarse_global_indices, num_coarse_nodes, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P row ptr
            if (myid == proc) P_rowPtr = hypre_ParCompGridPRowPtr(compGrid[level]);
            else P_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nodes+1, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(P_rowPtr, num_nodes+1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P column indices
            if (myid == proc) P_colInd = hypre_ParCompGridPColInd(compGrid[level]);
            else P_colInd = hypre_CTAlloc(HYPRE_Int, nnz_P, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(P_colInd, nnz_P, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P data
            if (myid == proc) P_data = hypre_ParCompGridPData(compGrid[level]);
            else P_data = hypre_CTAlloc(HYPRE_Complex, nnz_P, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(P_data, nnz_P, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);
         }

         // Now, each processors checks their owned info against the composite grid info
         HYPRE_Int proc_first_index = hypre_ParCompGridGlobalIndices(compGrid[level])[0];
         HYPRE_Int proc_last_index;
         if (hypre_ParCompGridNumOwnedNodes(compGrid[level])) proc_last_index = hypre_ParCompGridGlobalIndices(compGrid[level])[ hypre_ParCompGridNumOwnedNodes(compGrid[level]) - 1 ];
         else proc_last_index = proc_first_index - 1;
         for (i = 0; i < num_nodes; i++)
         {
            if (global_indices[i] <= proc_last_index && global_indices[i] >= proc_first_index)
            {
               HYPRE_Int row_size;
               HYPRE_Int *row_col_ind;
               HYPRE_Complex *row_values;
               hypre_ParCSRMatrixGetRow( A[level], global_indices[i], &row_size, &row_col_ind, &row_values );
               if (row_size != A_rowPtr[i+1] - A_rowPtr[i])
               {
                  // printf("Error: proc %d has incorrect row size at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                  test_failed = 1;
               }
               HYPRE_Int j;
               for (j = A_rowPtr[i]; j < A_rowPtr[i+1]; j++)
               {
                  if (A_colInd[j] < 0)
                  {
                     // If the column index is -1, then the appropriate global index (in row_col_ind) should not be in the global_indices
                     // Can do binary searches on the global_indices over the sorted owned nodes and the sorted non-owned nodes
                     HYPRE_Int left = 0;
                     HYPRE_Int right = num_owned_nodes - 1;
                     HYPRE_Int index;
                     while (left <= right)
                     {
                        index = (left + right) / 2;
                        if (global_indices[index] < row_col_ind[j - A_rowPtr[i]]) left = index + 1;
                        else if (global_indices[index] > row_col_ind[j - A_rowPtr[i]]) right = index - 1;
                        else
                        {
                           test_failed = 1;
                           // printf("Error: proc %d has -1 col ind in A where it should not at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                        }
                     }
                     left = num_owned_nodes;
                     right = num_nodes - 1;
                     while (left <= right)
                     {
                        index = (left + right) / 2;
                        if (global_indices[index] < row_col_ind[j - A_rowPtr[i]]) left = index + 1;
                        else if (global_indices[index] > row_col_ind[j - A_rowPtr[i]]) right = index - 1;
                        else
                        {
                           test_failed = 1;
                           // printf("Error: proc %d has -1 col ind in A where it should not at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                        }
                     }
                  }
                  else if (global_indices[ A_colInd[j] ] != row_col_ind[j - A_rowPtr[i]])
                  {
                     // printf("Error: proc %d has incorrect A col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
                  if (A_data[j] != row_values[j - A_rowPtr[i]])
                  {
                     // printf("Error: proc %d has incorrect A data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
               }
               hypre_ParCSRMatrixRestoreRow( A[level], i, &row_size, &row_col_ind, &row_values );
               if (level != num_levels-1)
               {
                  hypre_ParCSRMatrixGetRow( P[level], global_indices[i], &row_size, &row_col_ind, &row_values );
                  if (row_size != P_rowPtr[i+1] - P_rowPtr[i])
                  {
                     // printf("Error: proc %d has incorrect row size at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
                  for (j = P_rowPtr[i]; j < P_rowPtr[i+1]; j++)
                  {
                     if (P_colInd[j] < 0)
                     {
                        // If the column index is -1, then the appropriate global index (in row_col_ind) should not be in the global_indices
                        // Can do binary searches on the global_indices over the sorted owned nodes and the sorted non-owned nodes
                        HYPRE_Int left = 0;
                        HYPRE_Int right = num_coarse_owned_nodes - 1;
                        HYPRE_Int index;
                        while (left <= right)
                        {
                           index = (left + right) / 2;
                           if (coarse_global_indices[index] < row_col_ind[j - P_rowPtr[i]]) left = index + 1;
                           else if (coarse_global_indices[index] > row_col_ind[j - P_rowPtr[i]]) right = index - 1;
                           else
                           {
                              test_failed = 1;
                              // printf("Error: proc %d has -1 col ind in P where it should not at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                           }
                        }
                        left = num_coarse_owned_nodes;
                        right = num_coarse_nodes - 1;
                        while (left <= right)
                        {
                           index = (left + right) / 2;
                           if (coarse_global_indices[index] < row_col_ind[j - P_rowPtr[i]]) left = index + 1;
                           else if (coarse_global_indices[index] > row_col_ind[j - P_rowPtr[i]]) right = index - 1;
                           else
                           {
                              test_failed = 1;
                              // printf("Error: proc %d has -1 col ind in P where it should not at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                           }
                        }
                     }
                     else if (coarse_global_indices[ P_colInd[j] ] != row_col_ind[j - P_rowPtr[i]])
                     {
                        // printf("Error: proc %d has incorrect P col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                        test_failed = 1;
                     }
                     if (P_data[j] != row_values[j - P_rowPtr[i]])
                     {
                        // printf("Error: proc %d has incorrect P data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                        test_failed = 1;
                     }
                  }
                  hypre_ParCSRMatrixRestoreRow( P[level], i, &row_size, &row_col_ind, &row_values );
               }
            }
         }

         // Clean up memory
         if (myid != proc) 
         {
            hypre_TFree(global_indices, HYPRE_MEMORY_HOST);
            hypre_TFree(A_rowPtr, HYPRE_MEMORY_HOST);
            hypre_TFree(A_colInd, HYPRE_MEMORY_HOST);
            hypre_TFree(A_data, HYPRE_MEMORY_HOST);
            if (level != num_levels-1)
            {
               hypre_TFree(coarse_global_indices, HYPRE_MEMORY_HOST);
               hypre_TFree(P_rowPtr, HYPRE_MEMORY_HOST);
               hypre_TFree(P_colInd, HYPRE_MEMORY_HOST);
               hypre_TFree(P_data, HYPRE_MEMORY_HOST);
            }
         }
      }
   }

   return test_failed;
}
