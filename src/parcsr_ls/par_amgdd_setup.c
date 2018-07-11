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
#define DEBUGGING_MESSAGES 0 // if true, prints a bunch of messages to the screen to let you know where in the algorithm you are

HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGrid *compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int padding );

HYPRE_Int
FindNeighborProcessors( hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, HYPRE_Int **add_flag,
   HYPRE_Int *num_starting_nodes, HYPRE_Int **starting_nodes,
   HYPRE_Int *search_proc_marker,
   HYPRE_Int *num_request_nodes, HYPRE_Int **request_nodes,
   HYPRE_Int *num_send_procs, HYPRE_Int *send_procs, 
   HYPRE_Int num_neighboring_procs, HYPRE_Int level );

HYPRE_Int
RecursivelyFindNeighborNodes(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, 
   HYPRE_Int *request_nodes, HYPRE_Int *num_request_nodes );

HYPRE_Complex*
PackSendBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, HYPRE_Int *send_flag_buffer_size, 
   HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int padding );

HYPRE_Int
RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int *add_flag_coarse, 
   HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding);

HYPRE_Int
UnpackRecvBuffer( HYPRE_Complex *recv_buffer, hypre_ParCompGrid **compGrid, 
      HYPRE_Int *num_sends, HYPRE_Int *num_recvs,
      HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
      HYPRE_Int ****recv_map, HYPRE_Int ***recv_map_send, 
      HYPRE_Int ***num_recv_nodes, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, 
      HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index, HYPRE_Int *num_added_nodes, HYPRE_Int **num_incoming_nodes, HYPRE_Int buffer_number, HYPRE_Int recv_rank );

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_send_buffer, HYPRE_Int *num_incoming_nodes, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
UnpackSendFlagBuffer(HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *send_buffer_size, HYPRE_Int *num_send_nodes, HYPRE_Int current_level, HYPRE_Int num_levels);

/*****************************************************************************
 *
 * Routine for setting up the composite grids in AMG-DD
 
 *****************************************************************************/

/*****************************************************************************
 * hypre_BoomerAMGDDCompGridSetup
 *****************************************************************************/

HYPRE_Int
hypre_BoomerAMGDDCompGridSetup( void *amg_vdata, HYPRE_Int padding, HYPRE_Int *timers, HYPRE_Int use_barriers )
{
   char filename[256];

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
      if (level != num_levels-2) hypre_ParCompGridInitialize( compGrid[level+1], F_array[level+1], CF_marker_array[level+1], proc_first_index[level+2], A_array[level+1], P_array[level+1] );
      else hypre_ParCompGridInitialize( compGrid[level+1], F_array[level+1], CF_marker_array[level+1], 0, A_array[level+1], NULL );
   }




   // !!! Debugging: !!!
   sprintf(filename, "outputs/A_rank%d_level%d.txt", myid, 3);
   hypre_ParCompGridMatlabAMatrixDump( compGrid[3], filename);







   // Now that the comp grids are initialized, store RHS back in F_array[0]
   hypre_ParVectorCopy(Vtemp,F_array[0]);

   // On each level, setup a long distance commPkg that has communication info for distance (eta + numGhostLayers)
   if (timers) hypre_BeginTiming(timers[0]);
   for (level = 0; level < num_levels; level++)
   {
      SetupNearestProcessorNeighbors(A_array[level], compGrid[level], compGridCommPkg, level, padding);
   }
   if (timers) hypre_EndTiming(timers[0]);
   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   
   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   #if DEBUGGING_MESSAGES
   if (myid == 0) printf("  Looping over levels\n");
   #endif

   for (level = num_levels-1; level > -1; level--)
   {

      // !!! Debugging: !!!
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      HYPRE_Int here_local = 1;
      HYPRE_Int here_global = 0;
      hypre_MPI_Allreduce(&here_local, &here_global, 1, HYPRE_MPI_INT, MPI_MAX, hypre_MPI_COMM_WORLD);
      if (myid == 0) printf("Level %d: all ranks here %d\n", level, here_global);
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);


      comm = hypre_ParCSRMatrixComm(A_array[level]);
      num_sends = hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level];
      num_recvs = hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level];
      #if DEBUGGING_MESSAGES
      printf("    Rank %d: Level %d:\n", myid, level);
      #endif
      if ( proc_last_index[level] >= proc_first_index[level] && num_sends ) // If there are any owned nodes on this level
      {



         // !!! Debugging: !!!
         HYPRE_Int color = 1;
         MPI_Comm subComm;
         MPI_Comm_split(hypre_MPI_COMM_WORLD, color, myid, &subComm);
         hypre_MPI_Barrier(subComm);
         here_local = 1;
         here_global = 0;
         hypre_MPI_Allreduce(&here_local, &here_global, 1, HYPRE_MPI_INT, MPI_MAX, subComm);
         HYPRE_Int subid;
         hypre_MPI_Comm_rank(subComm, &subid );
         if (subid == 0) printf("Level %d: all participating ranks began level %d\n", level, here_global);
         hypre_MPI_Barrier(subComm);





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



         if (level == 3)
         {
            FILE *file;
            char filename[256];
            sprintf(filename,"outputs/rank%d.txt",myid);
            file = fopen(filename,"w");
            for (i = 0; i < num_sends; i++) fprintf(file,"%d ", hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i]);
            fprintf(file, "\n");
            for (i = 0; i < num_recvs; i++) fprintf(file,"%d ", hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i]);
            fclose(file);
         }




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
                                             &(send_flag_buffer_size[i]), send_flag[level][i], num_send_nodes[level][i], i, level, num_levels, padding );
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
            // send_buffer_size[level][i] = GetBufferSize( psiComposite_send[i], level, num_levels, num_psi_levels_send[i] );
            hypre_MPI_Isend(&(send_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 0, comm, &requests[request_counter++]);
         }
         


         if (level == 3)
         {
            FILE *file;
            char filename[256];
            sprintf(filename,"outputs/send_buffer_sizes_rank%d.txt", myid);
            file = fopen(filename,"w");
            for (i = 0; i < num_sends; i++) fprintf(file,"%d ", send_buffer_size[level][i]);
            fprintf(file, "\n");
            for (i = 0; i < num_sends; i++) fprintf(file,"%d ", hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i]);
            fclose(file);
         }




         // wait for all buffer sizes to be received
         if (level == 3) printf("Rank %d, level %d, waiting on buffer sizes\n", myid, level);
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );
         if (level == 3) printf("Rank %d, level %d, done waiting on buffer sizes\n", myid, level);

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
         }

         // wait for buffers to be received
         if (level == 3) printf("Rank %d, level %d, waiting on buffers\n", myid, level);
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );
         if (level == 3) printf("Rank %d, level %d, done waiting on buffers\n", myid, level);

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
               num_recv_nodes, &(recv_map_send_buffer_size[i]), level, num_levels, proc_first_index, proc_last_index, num_added_nodes, num_incoming_nodes, i, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i]);
         }

         if (timers) hypre_EndTiming(timers[4]);

         // Setup local indices for the composite grid
         #if DEBUGGING_MESSAGES
         printf("      Rank %d: Setup local indices\n",myid);
         #endif

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         
         if (timers) hypre_BeginTiming(timers[5]);
         
         hypre_ParCompGridSetupLocalIndices(compGrid, num_added_nodes, num_levels, proc_first_index, proc_last_index);
         if (level == 0) hypre_ParCompGridSetupLocalIndicesP(compGrid, num_levels);
         
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
         if (level == 3) printf("Rank %d, level %d, waiting on send_flag buffers\n", myid, level);
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );
         if (level == 3) printf("Rank %d, level %d, done waiting on send_flag buffers\n", myid, level);

         #if DEBUGGING_MESSAGES
         hypre_printf("      Rank %d: done waiting on send map buffers\n", myid);
         #endif

         // unpack and setup the send flag arrays
         for (i = 0; i < num_sends; i++)
         {
            UnpackSendFlagBuffer(send_flag_buffer[i], send_flag[level][i], &(send_buffer_size[level][i]), num_send_nodes[level][i], level, num_levels);
         }

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



         // !!! Debugging: !!!
         hypre_MPI_Barrier(subComm);
         here_local = 1;
         here_global = 0;
         hypre_MPI_Allreduce(&here_local, &here_global, 1, HYPRE_MPI_INT, MPI_MAX, subComm);
         hypre_MPI_Comm_rank(subComm, &subid );
         if (subid == 0) printf("Level %d: all participating ranks finished level %d\n", level, here_global);
         hypre_MPI_Barrier(subComm);


      }
      else 
      {
         // printf("Rank %d, level %d, nothing to do on this level\n", myid, level);
         if (use_barriers)
         {
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         }


         // !!! Debugging: !!!
         HYPRE_Int color = 0;
         MPI_Comm subComm;
         MPI_Comm_split(hypre_MPI_COMM_WORLD, color, myid, &subComm);
         hypre_MPI_Barrier(subComm);
         here_local = 1;
         here_global = 0;
         hypre_MPI_Allreduce(&here_local, &here_global, 1, HYPRE_MPI_INT, MPI_MAX, subComm);
         HYPRE_Int subid;
         hypre_MPI_Comm_rank(subComm, &subid );
         if (subid == 0) printf("Level %d: all non participating ranks finished level %d\n", level, here_global);
         hypre_MPI_Barrier(subComm);

      }
      #if DEBUGGING_MESSAGES
      hypre_printf("      Rank %d: done with level %d\n", myid, level);
      #endif 
   }


   #if DEBUG_COMP_GRID
   for (level = 0; level < num_levels; level++)
   {
      // hypre_sprintf(filename, "/p/lscratchd/wbm/CompGrids/setupCompGridRank%dLevel%d.txt", myid, level);
      hypre_sprintf(filename, "outputs/CompGrids/setupCompGridRank%dLevel%d.txt", myid, level);
      // hypre_ParCompGridDebugPrint( compGrid[level], filename );
      // hypre_sprintf(filename, "outputs/CompGrids/plotCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridMatlabPlot( compGrid[level], filename );
      hypre_sprintf(filename, "outputs/CompGrids/setupACompRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridMatlabAMatrixDump( compGrid[level], filename );

      if (level != num_levels-1)
      {
         hypre_sprintf(filename, "outputs/CompGrids/setupPCompRank%dLevel%d.txt", myid, level);
         hypre_ParCompGridMatlabPMatrixDump( compGrid[level], filename );
      }
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

   // assign compGrid and compGridCommPkg info to the amg structure
   hypre_ParAMGDataCompGrid(amg_data) = compGrid;
   hypre_ParAMGDataCompGridCommPkg(amg_data) = compGridCommPkg;

   #if DEBUGGING_MESSAGES
   hypre_printf("Finished comp grid setup on rank %d\n", myid);
   #endif

   // Cleanup memory
   for (level = 0; level < num_levels; level++)
   {
      if (num_recv_nodes[level])
      {
         num_recvs = hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level];
         for (i = 0; i < num_recvs; i++)
         {
            hypre_TFree(num_recv_nodes[level][i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(num_recv_nodes[level], HYPRE_MEMORY_HOST);
      }
   }
   hypre_TFree(num_recv_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(num_added_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(global_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_first_index, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_last_index, HYPRE_MEMORY_HOST);
   
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

   // Get the default (distance 1) number of send procs
   HYPRE_Int      num_sends = hypre_ParCSRCommPkgNumSends(commPkg);

   #if DEBUG_COMP_GRID
   // Check to make sure original matrix has symmetric send/recv relationship
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
      // printf("Rank %d: nothing to do on level %d\n", myid, level);
   }
   else
   {
      // Initialize add_flag (this is how we will track nodes to send to each proc until routine finishes)
      HYPRE_Int      *send_procs = hypre_CTAlloc( HYPRE_Int, (padding+4)*num_sends, HYPRE_MEMORY_HOST); // Note: in the worst case, we extend the number of procs to send to by padding+4 times
      HYPRE_Int      **add_flag = hypre_CTAlloc( HYPRE_Int*, (padding+4)*num_sends, HYPRE_MEMORY_HOST);
      HYPRE_Int      *search_proc_marker = hypre_CTAlloc(HYPRE_Int, (padding+4)*num_sends, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         send_procs[i] = hypre_ParCSRCommPkgSendProc(commPkg,i);
         add_flag[i] = hypre_CTAlloc( HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
         for (j = start; j < finish; j++) add_flag[i][ hypre_ParCSRCommPkgSendMapElmt(commPkg,j) ] = padding + 4; // must be set to padding + numGhostLayers (note that the starting nodes are already distance 1 from their neighbors on the adjacent processor)
      }


      // !!! Debugging: !!!
      if (myid == 14 && level == 3) for (i = 0; i < num_sends; i++) if (send_procs[i] == 0) printf("At line 737, 14 sends to 0\n");

      // Setup initial num_starting_nodes and starting_nodes (these are the starting nodes when searching for long distance neighbors) !!! I don't think I actually have a good upper bound on sizes here... how to properly allocate/reallocate these? !!!
      HYPRE_Int *num_starting_nodes = hypre_CTAlloc( HYPRE_Int, (padding+4)*num_sends, HYPRE_MEMORY_HOST );
      HYPRE_Int **starting_nodes = hypre_CTAlloc( HYPRE_Int*, (padding+4)*num_sends, HYPRE_MEMORY_HOST );
      HYPRE_Int max_num_starting_nodes = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
         search_proc_marker[i] = 1;
         num_starting_nodes[i] = finish - start;
         if (num_starting_nodes[i] > max_num_starting_nodes) max_num_starting_nodes = num_starting_nodes[i];
      }
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         starting_nodes[i] = hypre_CTAlloc( HYPRE_Int, max_num_starting_nodes, HYPRE_MEMORY_HOST );
         for (j = 0; j < num_starting_nodes[i]; j++)
         {
            starting_nodes[i][j] = hypre_ParCSRCommPkgSendMapElmt(commPkg, j + start );
         }
      }

      // Find my own send nodes and communicate with neighbors to find off-processor long-range connections
      HYPRE_Int *num_request_nodes = hypre_CTAlloc(HYPRE_Int, (padding+4)*num_sends, HYPRE_MEMORY_HOST);
      HYPRE_Int **request_nodes = hypre_CTAlloc(HYPRE_Int*, (padding+4)*num_sends, HYPRE_MEMORY_HOST);

      FILE *file;
      char filename[256];
      for (i = 0; i < 3 + padding; i++)
      {
         if (level == 3)
         {
            for (j = 0; j < num_sends; j++)
            {
               sprintf(filename,"outputs/add_flag_proc%d_rank%d_i%d.txt", send_procs[j], myid, i);
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
               for (j = 0; j < num_sends; j++)
               {
                  sprintf(filename,"outputs/request_nodes_proc%d_rank%d_i%d.txt", send_procs[j], myid, 3+padding);
                  file = fopen(filename,"w");
                  HYPRE_Int k;
                  for (k = 0; k < num_request_nodes[j]; k++)
                  {
                     fprintf(file, "%d ", request_nodes[j][2*k]);
                  }
                  fprintf(file, "\n");
                  for (k = 0; k < num_request_nodes[j]; k++)
                  {
                     fprintf(file, "%d ", request_nodes[j][2*k+1]);
                  }
                  fclose(file);
               }
            }
         }

         FindNeighborProcessors(compGrid, A, add_flag, 
            num_starting_nodes, starting_nodes, 
            search_proc_marker,
            num_request_nodes, request_nodes,
            &num_sends, send_procs, 
            hypre_ParCSRCommPkgNumSends(commPkg), level); // Note that num_sends may change here

         // !!! Debugging: !!!
         if (myid == 14 && level == 3) for (j = 0; j < num_sends; j++) if (send_procs[j] == 0) printf("At line 797 after iteration %d, 14 sends to 0\n",i);
      }

      if (level == 3)
      {
         for (j = 0; j < num_sends; j++)
         {
            sprintf(filename,"outputs/add_flag_proc%d_rank%d_i%d.txt", send_procs[j], myid, 3+padding);
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
         }
         for (j = 0; j < num_sends; j++)
         {
            sprintf(filename,"outputs/request_nodes_proc%d_rank%d_i%d.txt", send_procs[j], myid, 3+padding);
            file = fopen(filename,"w");
            HYPRE_Int k;
            for (k = 0; k < num_request_nodes[j]; k++)
            {
               fprintf(file, "%d ", request_nodes[j][2*k]);
            }
            fprintf(file, "\n");
            for (k = 0; k < num_request_nodes[j]; k++)
            {
               fprintf(file, "%d ", request_nodes[j][2*k+1]);
            }
            fclose(file);
         }
      }

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

   // printf("Rank %d finishing SetupNearestProcessorNeighbors level %d\n", myid, level);

   return 0;
}

HYPRE_Int
FindNeighborProcessors( hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, HYPRE_Int **add_flag,
   HYPRE_Int *num_starting_nodes, HYPRE_Int **starting_nodes,
   HYPRE_Int *search_proc_marker,
   HYPRE_Int *num_request_nodes, HYPRE_Int **request_nodes,
   HYPRE_Int *num_send_procs, HYPRE_Int *send_procs, 
   HYPRE_Int num_neighboring_procs, HYPRE_Int level )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int      i,j,k,cnt;

   // Update add_flag by recursively adding neighbors
   for (i = 0; i < (*num_send_procs); i++)
   {
      if (search_proc_marker[i])
      {
         num_request_nodes[i] = 0;
         if (!request_nodes[i]) request_nodes[i] = hypre_CTAlloc( HYPRE_Int, 2*hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)), HYPRE_MEMORY_HOST );

         for (j = 0; j < num_starting_nodes[i]; j++)
         {
            if (add_flag[i][ starting_nodes[i][j] ] - 1 > 0)
               RecursivelyFindNeighborNodes( starting_nodes[i][j], add_flag[i][ starting_nodes[i][j] ] - 1, compGrid, add_flag[i], request_nodes[i], &(num_request_nodes[i]) );
         }

         num_starting_nodes[i] = 0;
      }
   }

   // Exchange message sizes
   HYPRE_Int send_size = 1;
   for (i = 0; i < (*num_send_procs); i++)
   {
      if (search_proc_marker[i])
      {
         if (num_request_nodes[i])
         {
            send_size += 2*num_request_nodes[i] + 2;
         }
      }
   }
   HYPRE_Int *recv_sizes = hypre_CTAlloc(HYPRE_Int, num_neighboring_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Request *requests = hypre_CTAlloc(hypre_MPI_Request, 4*num_neighboring_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Status *statuses = hypre_CTAlloc(hypre_MPI_Status, 4*num_neighboring_procs, HYPRE_MEMORY_HOST);
   HYPRE_Int request_cnt = 0;
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Irecv(&(recv_sizes[i]), 1, HYPRE_MPI_INT, send_procs[i], 4, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      hypre_MPI_Isend(&send_size, 1, HYPRE_MPI_INT, send_procs[i], 4, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }

   // Wait on the recv sizes
   hypre_MPI_Waitall(2*num_neighboring_procs, requests, statuses);

   // Allocate recv buffers
   HYPRE_Int **recv_buffers = hypre_CTAlloc(HYPRE_Int*, num_neighboring_procs, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_neighboring_procs; i++) recv_buffers[i] = hypre_CTAlloc(HYPRE_Int, recv_sizes[i], HYPRE_MEMORY_HOST);

   // Post the recvs
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Irecv(recv_buffers[i], recv_sizes[i], HYPRE_MPI_INT, send_procs[i], 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   
   // Setup the send buffer and post the sends
   HYPRE_Int *send_buffer = hypre_CTAlloc(HYPRE_Int, send_size, HYPRE_MEMORY_HOST);
   cnt = 1;
   HYPRE_Int num_request_procs = 0;
   for (i = 0; i < (*num_send_procs); i++)
   {
      if (search_proc_marker[i])
      {
         if (num_request_nodes[i])
         {
            num_request_procs++;
            send_buffer[cnt++] = send_procs[i];
            send_buffer[cnt++] = num_request_nodes[i];
            for (j = 0; j < num_request_nodes[i]; j++)
            {
               send_buffer[cnt++] = request_nodes[i][2*j];
               send_buffer[cnt++] = request_nodes[i][2*j+1];
            }
         }
      }
   }
   send_buffer[0] = num_request_procs;
   // hypre_printf("Rank %d: posting sends\n", myid);
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Isend(send_buffer, send_size, HYPRE_MPI_INT, send_procs[i], 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }

   // Wait 
   hypre_MPI_Waitall(2*num_neighboring_procs, &(requests[2*num_neighboring_procs]), &(statuses[2*num_neighboring_procs]));

   // Reset search_proc_marker
   for (i = 0; i < (*num_send_procs); i++) search_proc_marker[i] = 0;

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
            for (k = 0; k < (*num_send_procs); k++) if (send_procs[k] == incoming_proc) local_proc_index = k;
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
                  send_procs[local_proc_index] = incoming_proc;
                  add_flag[local_proc_index] = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRMatrixNumRows(A), HYPRE_MEMORY_HOST);
                  starting_nodes[local_proc_index] = hypre_CTAlloc(HYPRE_Int, num_incoming_nodes, HYPRE_MEMORY_HOST);
                  search_proc_marker[local_proc_index] = 1;
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
                     if (incoming_flag > add_flag[local_proc_index][local_index])
                     {
                        add_flag[local_proc_index][local_index] = incoming_flag;
                        starting_nodes[local_proc_index][ num_starting_nodes[local_proc_index]++ ] = local_index;
                        search_proc_marker[local_proc_index] = 1;
                     }
                  }
               }
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
         // Check whether we have already requested this node (!!! linear search, but should be over small set !!!)
         HYPRE_Int global_index = hypre_ParCompMatrixRowGlobalIndices(A_row)[i];
         HYPRE_Int add_requeset = 1;
         for (j = 0; j < (*num_request_nodes); j++)
         {
            if (request_nodes[2*j] == global_index)
            {
               add_requeset = 0;
               if (m > request_nodes[2*j+1]) request_nodes[2*j+1] = m;
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

HYPRE_Complex*
PackSendBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, 
   HYPRE_Int *send_flag_buffer_size, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes,
   HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int padding )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   
   HYPRE_Int            level,i,j,cnt,row_length,send_elmt,coarse_grid_index;
   HYPRE_Int            need_coarse_info, nodes_to_add = 0;
   HYPRE_Int            **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            num_psi_levels = 1;
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
   send_flag[current_level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level], HYPRE_MEMORY_HOST );
   ghost_marker[current_level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level], HYPRE_MEMORY_HOST );
   (*buffer_size) += 2;
   if (need_coarse_info) (*buffer_size) += 4*num_send_nodes[current_level];
   else (*buffer_size) += 2*num_send_nodes[current_level];
   for (i = start; i < finish; i++)
   {
      send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];
      send_flag[current_level][i - start] = send_elmt;
      ghost_marker[current_level][i - start] = hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[current_level][i];
      (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(compGrid[current_level])[send_elmt]) + 1;
      if (need_coarse_info) (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridPRows(compGrid[current_level])[send_elmt]) + 1;
   }
   
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
            if (add_flag[level][i] == padding + 1)
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
            if (add_flag[level][i] == 5) RecursivelyBuildPsiComposite(i, 4, compGrid[level], add_flag[level], NULL, 0, NULL, 0);
         }

         // Count up the buffer size
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 0)
            {
               num_send_nodes[level]++;
               (*send_flag_buffer_size)++;
               if (need_coarse_info) (*buffer_size) += 4;
               else (*buffer_size) += 2;
               (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(compGrid[level])[i]) + 1;
               if (need_coarse_info) (*buffer_size) += 2*hypre_ParCompMatrixRowSize(hypre_ParCompGridPRows(compGrid[level])[i]) + 1;
            }
         }

         // Save the indices (in global index ordering) so I don't have to keep looping over all nodes in compGrid when I pack the buffer
         send_flag[level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[level], HYPRE_MEMORY_HOST );
         ghost_marker[level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[level], HYPRE_MEMORY_HOST );
         cnt =  0;
         HYPRE_Int insert_owned_position;
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

         // Generate the send_flag in global index ordering
         // !!!!! TODO: Want to also get rid of potential redundancy on coarser levels here !!!!!!
         for (i = hypre_ParCompGridNumOwnedNodes(compGrid[level]); i < insert_owned_position; i++)
         {
            if (add_flag[level][i] > 0) 
            {
               send_flag[level][cnt] = i;
               if (add_flag[level][i] < 5) ghost_marker[level][cnt] = 1;
               cnt++;
            }
         }
         for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 0) 
            {
               send_flag[level][cnt] = i;
               if (add_flag[level][i] < 5) ghost_marker[level][cnt] = 1;
               cnt++;
            }
         }
         for (i = insert_owned_position; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 0) 
            {
               send_flag[level][cnt] = i;
               if (add_flag[level][i] < 5) ghost_marker[level][cnt] = 1;
               cnt++;
            }
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
      send_buffer[cnt++] = (HYPRE_Complex) num_send_nodes[level];

      // copy all global indices
      for (i = 0; i < num_send_nodes[level]; i++)
      {
         send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompGridGlobalIndices(compGrid[level])[ send_flag[level][i] ];
      }
      // copy all residual values
      for (i = 0; i < num_send_nodes[level]; i++)
      {
         send_buffer[cnt++] = hypre_ParCompGridF(compGrid[level])[ send_flag[level][i] ];
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
            send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ send_flag[level][i] ];
         }
      }

      // now loop over matrix rows
      for (i = 0; i < num_send_nodes[level]; i++)
      {
         // store the row length for matrix A
         row_length = hypre_ParCompMatrixRowSize( hypre_ParCompGridARows( compGrid[level] )[ send_flag[level][i] ] );
         send_buffer[cnt++] = (HYPRE_Complex) row_length;
         
         // copy matrix entries for matrix A
         for (j = 0; j < row_length; j++)
         {
            send_buffer[cnt++] = hypre_ParCompMatrixRowData( hypre_ParCompGridARows( compGrid[level] )[ send_flag[level][i] ] )[j];
         }
         // copy global indices for matrix A
         for (j = 0; j < row_length; j++)
         {
            send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( compGrid[level] )[ send_flag[level][i] ] )[j];
         }

         if (level != num_levels-1)
            {
            // store the row length for matrix P
            row_length = hypre_ParCompMatrixRowSize( hypre_ParCompGridPRows( compGrid[level] )[ send_flag[level][i] ] );
            send_buffer[cnt++] = (HYPRE_Complex) row_length;

            // copy matrix entries for matrix P
            for (j = 0; j < row_length; j++)
            {
               send_buffer[cnt++] = hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( compGrid[level] )[ send_flag[level][i] ] )[j];
            }
            // copy global indices for matrix P
            for (j = 0; j < row_length; j++)
            {
               send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( compGrid[level] )[ send_flag[level][i] ] )[j];
            }
         }
      }
   }

   // Clean up memory
   for (level = 0; level < num_levels; level++)
   {
      if (add_flag[level]) hypre_TFree(add_flag[level], HYPRE_MEMORY_HOST);
      if (ghost_marker[level]) hypre_TFree(ghost_marker[level], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);
   hypre_TFree(ghost_marker, HYPRE_MEMORY_HOST);

   // Return the send buffer
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
      if (index >= 0)
      {
         // And if we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[index] < m)
         {
            add_flag[index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) RecursivelyBuildPsiComposite(index, m-1, compGrid, add_flag, add_flag_coarse, need_coarse_info, nodes_to_add, padding);
         }
      }
      else hypre_printf("Error! Ran into a -1 index when building Psi_c\n");
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
UnpackRecvBuffer( HYPRE_Complex *recv_buffer, hypre_ParCompGrid **compGrid, 
      HYPRE_Int *num_sends, HYPRE_Int *num_recvs,
      HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
      HYPRE_Int ****recv_map, HYPRE_Int ***recv_map_send, 
      HYPRE_Int ***num_recv_nodes, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, 
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
         HYPRE_Int incoming_global_index = recv_buffer[cnt];
         HYPRE_Int compGrid_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[ compGrid_cnt + num_owned_nodes ];
         if (incoming_global_index >= proc_first_index[level] && incoming_global_index <= proc_last_index[level])
         {
            incoming_dest[incoming_cnt++] = -1;
            cnt++;
         }
         else if (incoming_global_index == compGrid_global_index)
         {
            incoming_dest[incoming_cnt++] = -1;
            // Check whether incoming redundant node is a real node (if so, ensure that the node is marked real in the comp grid)
            if (level != num_levels-1) if ( !( (HYPRE_Int) recv_buffer[cnt + 2*num_incoming_nodes[buffer_number][level]] ) ) hypre_ParCompGridGhostMarker(compGrid[level])[compGrid_cnt + num_owned_nodes] = 0;
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
         HYPRE_Int incoming_global_index = recv_buffer[cnt];
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
         hypre_ParCompGridF(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridF(compGrid[level])[i + num_owned_nodes];
      for (i = num_nonowned_nodes - 1; i >= 0; i--)
      {
         hypre_ParCompGridARows(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridARows(compGrid[level])[i + num_owned_nodes];
      }
      if (level != num_levels-1)
      {
         for (i = num_nonowned_nodes - 1; i >= 0; i--)
            hypre_ParCompGridGhostMarker(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridGhostMarker(compGrid[level])[i + num_owned_nodes];
         for (i = num_nonowned_nodes - 1; i >= 0; i--)
            hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[i + num_owned_nodes];
         for (i = num_nonowned_nodes - 1; i >= 0; i--)
            hypre_ParCompGridCoarseLocalIndices(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i + num_owned_nodes];
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
            hypre_ParCompGridGlobalIndices(compGrid[level])[ incoming_dest[i] ] = recv_buffer[cnt];
            num_recv_nodes[current_level][buffer_number][level]++;
         }
         cnt++;
      }
      for (i = 0; i < num_incoming_nodes[buffer_number][level]; i++) 
      {   
         if (incoming_dest[i] >= 0) hypre_ParCompGridF(compGrid[level])[ incoming_dest[i] ] = recv_buffer[cnt];
         cnt++;
      }
      if (level != num_levels-1)
      {
         for (i = 0; i < num_incoming_nodes[buffer_number][level]; i++) 
         {   
            if (incoming_dest[i] >= 0) hypre_ParCompGridGhostMarker(compGrid[level])[ incoming_dest[i] ] = recv_buffer[cnt];
            cnt++;
         }
         for (i = 0; i < num_incoming_nodes[buffer_number][level]; i++) 
         {   
            if (incoming_dest[i] >= 0) hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ incoming_dest[i] ] = recv_buffer[cnt];
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

      for (i = 0; i < num_nodes; i++)
      {
         if (send_flag_buffer[cnt++] != -1) 
         {
            // discard indices of nodes in send_flag that will not be sent in the future and count the send buffer size
            send_flag[level][ num_send_nodes[level]++ ] = send_flag[level][i];
            (*send_buffer_size)++;
         }
      }
      send_flag[level] = hypre_TReAlloc(send_flag[level], HYPRE_Int, num_send_nodes[level], HYPRE_MEMORY_HOST);
   }

   return 0;
}

