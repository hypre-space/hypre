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

#define DEBUG_COMP_GRID 0 // if true, runs some tests, prints out what is stored in the comp grids for each processor to a file
#define DEBUG_PROC_NEIGHBORS 0 // if true, dumps info on the add flag structures that determine nearest processor neighbors 
#define DEBUGGING_MESSAGES 0 // if true, prints a bunch of messages to the screen to let you know where in the algorithm you are

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int *recv_map_send_buffer, HYPRE_Int **recv_redundant_marker, HYPRE_Int *num_recv_nodes, HYPRE_Int *recv_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, hypre_ParCompGrid **compGrid);

HYPRE_Int
UnpackSendFlagBuffer(hypre_ParCompGrid **compGrid, HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, HYPRE_Int *send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
CommunicateRemainingMatrixInfo(hypre_ParAMGData* amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *communication_cost, HYPRE_Int symmetric);

HYPRE_Int
TestCompGrids1(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int current_level, HYPRE_Int check_ghost_info);

HYPRE_Int
TestCompGrids2(hypre_ParAMGData *amg_data);

HYPRE_Int 
CheckCompGridCommPkg(hypre_ParCompGridCommPkg *compGridCommPkg);

HYPRE_Int
FixUpRecvMaps(hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int ****recv_redundant_marker, HYPRE_Int start_level, HYPRE_Int num_levels);

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
                        HYPRE_Int *communication_cost,
                        HYPRE_Int verify_amgdd )
{

   // communication_cost measures # messages and comm volume from the perspective of THIS processor and takes the form:
   // communication_cost = [ [level 0], [level 1], ..., [level L] ]
   // [level] = [ 0: preprocesing # messages, 1: preprocessing comm volume, 
   //             2: setup # messages, 3: setup comm volume, 
   //             4: residual # messages, 5: residual comm volume ]

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   char filename[256];

   MPI_Comm 	      comm;
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;

   // If the underlying AMG data structure has not yet been set up, call BoomerAMGSetup()
   if (!hypre_ParAMGDataAArray(amg_data))
      hypre_BoomerAMGSetup(amg_vdata, A, b, x);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Began comp grid setup on rank %d\n", myid);
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif
 
   // Declare some frequently used variables
   HYPRE_Int      level,i,j;
   HYPRE_Int      **send_buffer_size;
   HYPRE_Int      **recv_buffer_size;
   HYPRE_Int      ***num_send_nodes;
   HYPRE_Int      ***num_recv_nodes;
   HYPRE_Int      ****send_flag;
   HYPRE_Int      ****recv_map;
   HYPRE_Int      ****recv_redundant_marker;
   HYPRE_Int      **send_buffer;
   HYPRE_Int      **recv_buffer;
   HYPRE_Int      **send_flag_buffer;
   HYPRE_Int      **recv_map_send_buffer;
   HYPRE_Int      *send_flag_buffer_size;
   HYPRE_Int      *recv_map_send_buffer_size;
   hypre_MPI_Request       *requests;
   hypre_MPI_Status        *status;
   HYPRE_Int               request_counter = 0;

   // get info from amg about how to setup amgdd
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);
   if (amgdd_start_level >= num_levels) amgdd_start_level = num_levels-1;
   HYPRE_Int pad = hypre_ParAMGDataAMGDDPadding(amg_data);
   HYPRE_Int variable_padding = hypre_ParAMGDataAMGDDVariablePadding(amg_data);
   HYPRE_Int num_ghost_layers = hypre_ParAMGDataAMGDDNumGhostLayers(amg_data);
   HYPRE_Int symmetric_tmp = hypre_ParAMGDataSym(amg_data);
   HYPRE_Int symmetric = 0;

   // !!! Debug
   HYPRE_Int *num_resizes = hypre_CTAlloc(HYPRE_Int, 3*num_levels, HYPRE_MEMORY_HOST);
   // !!! Debug
   HYPRE_Int total_bin_search_count = 0;
   HYPRE_Int total_redundant_sends = 0;

   // Allocate pointer for the composite grids
   hypre_ParCompGrid **compGrid = hypre_CTAlloc(hypre_ParCompGrid*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParAMGDataCompGrid(amg_data) = compGrid;

   // In the 1 processor case, just need to initialize the comp grids
   if (num_procs == 1)
   {
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         compGrid[level] = hypre_ParCompGridCreate();
         hypre_ParCompGridInitialize( amg_data, 0, level, symmetric );
      }
      hypre_ParCompGridFinalize(amg_data, compGrid, NULL, amgdd_start_level, num_levels, hypre_ParAMGDataAMGDDUseRD(amg_data), verify_amgdd);
      hypre_ParCompGridSetupRelax(amg_data);
      return 0;
   }

   // Get the padding on each level
   HYPRE_Int *padding = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   /* if (variable_padding > num_levels - amgdd_start_level) variable_padding = num_levels - amgdd_start_level; */
   if (variable_padding)
   {
      // padding[0] = 1;
      // for (level = amgdd_start_level; level < amgdd_start_level + variable_padding; level++) padding[level] = pad;
      // for (level = amgdd_start_level + variable_padding; level < num_levels; level++) padding[level] = 1;
      for (level = amgdd_start_level; level < num_levels; level++) padding[level] = pad;
      padding[num_levels-1] = variable_padding;
   }
   else
   {
      for (level = amgdd_start_level; level < num_levels; level++) padding[level] = pad;
   }
 
   // Initialize composite grid structures
   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[0]);
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      compGrid[level] = hypre_ParCompGridCreate();
      hypre_ParCompGridInitialize( amg_data, padding[level], level, symmetric );
   }
   if (timers) hypre_EndTiming(timers[0]);

   // Debugging: dump out the AMG hierarchy and the initialized composite grids
   #if DEBUG_COMP_GRID == 2
   for (level = 0; level < num_levels; level++)
   {
      sprintf(filename, "outputs/AMG_hierarchy/A_level%d", level);
      hypre_ParCSRMatrixPrint(hypre_ParAMGDataAArray(amg_data)[level], filename);
      if (level != num_levels-1)
      {
         sprintf(filename, "outputs/AMG_hierarchy/P_level%d", level);
         hypre_ParCSRMatrixPrint(hypre_ParAMGDataPArray(amg_data)[level], filename);
         if (hypre_ParAMGDataRestriction(amg_data))
         {
            sprintf(filename, "outputs/AMG_hierarchy/R_level%d", level);
            hypre_ParCSRMatrixPrint(hypre_ParAMGDataRArray(amg_data)[level], filename);
         }
      }
      hypre_sprintf(filename, "outputs/CompGrids/initCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrint( compGrid[level], filename );
   }
   #endif

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("  Done with comp grid initialization\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // Create the compGridCommPkg and grab a few frequently used variables
   hypre_ParCompGridCommPkg *compGridCommPkg = hypre_ParCompGridCommPkgCreate(num_levels);
   hypre_ParAMGDataCompGridCommPkg(amg_data) = compGridCommPkg;

   send_buffer_size = hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg);
   recv_buffer_size = hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg);
   send_flag = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg);
   num_send_nodes = hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg);
   recv_map = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg);
   num_recv_nodes = hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg);
   HYPRE_Int *nodes_added_on_level = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   recv_redundant_marker = hypre_CTAlloc(HYPRE_Int***, num_levels, HYPRE_MEMORY_HOST);

   // On each level, setup the compGridCommPkg so that it has communication info for distance (eta + numGhostLayers)
   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[1]);
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      SetupNearestProcessorNeighbors(hypre_ParAMGDataAArray(amg_data)[level], compGridCommPkg, level, padding, num_ghost_layers, communication_cost);
   }

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("  Done with SetupNearestProcessorNeighbors()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   if (timers) hypre_EndTiming(timers[1]);

   /////////////////////////////////////////////////////////////////

   // Loop over levels from coarsest to finest to build up the composite grids

   /////////////////////////////////////////////////////////////////

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("  Looping over levels\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   for (level = num_levels - 1; level >= amgdd_start_level; level--)
   {
      comm = hypre_ParCSRMatrixComm(hypre_ParAMGDataAArray(amg_data)[level]);
      HYPRE_Int num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level];
      HYPRE_Int num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level];

      if ( num_send_procs || num_recv_procs ) // If there are any owned nodes on this level
      {
         // Do some allocations
         requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         send_buffer = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
         send_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_send_procs, HYPRE_MEMORY_HOST);
         recv_buffer = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         recv_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_recv_procs, HYPRE_MEMORY_HOST);

         recv_map[level] = hypre_CTAlloc(HYPRE_Int**, num_recv_procs, HYPRE_MEMORY_HOST);
         recv_redundant_marker[level] = hypre_CTAlloc(HYPRE_Int**, num_recv_procs, HYPRE_MEMORY_HOST);
         num_recv_nodes[level] = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);

         send_flag_buffer = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
         send_flag_buffer_size = hypre_CTAlloc(HYPRE_Int, num_send_procs, HYPRE_MEMORY_HOST);
         recv_map_send_buffer = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         recv_map_send_buffer_size = hypre_CTAlloc(HYPRE_Int, num_recv_procs, HYPRE_MEMORY_HOST);

         //////////// Pack send buffers ////////////

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (timers) hypre_BeginTiming(timers[2]);
         for (i = 0; i < num_send_procs; i++)
         {
#if defined(HYPRE_USING_GPU)
            send_buffer[i] = PackSendBufferGPU(amg_data, compGrid, compGridCommPkg, &(send_buffer_size[level][i]), 
                                             &(send_flag_buffer_size[i]), send_flag, num_send_nodes, i, level, num_levels, padding, 
                                             num_ghost_layers, symmetric );
#else
            send_buffer[i] = PackSendBuffer(amg_data, compGrid, compGridCommPkg, &(send_buffer_size[level][i]), 
                                             &(send_flag_buffer_size[i]), send_flag, num_send_nodes, i, level, num_levels, padding, 
                                             num_ghost_layers, symmetric );
#endif
         }
         if (timers) hypre_EndTiming(timers[2]);

         //////////// Communicate buffer sizes ////////////

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (timers) hypre_BeginTiming(timers[3]);
  
         // post the receives for the buffer size
         for (i = 0; i < num_recv_procs; i++)
         {
            // !!! Check recv_buffer_size (make sure not overallocated)
            hypre_MPI_Irecv( &(recv_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]) );
         }

         // send the buffer sizes
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_MPI_Isend(&(send_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]));
            if (communication_cost)
            {
               communication_cost[level*10 + 2]++;
               communication_cost[level*10 + 3] += sizeof(HYPRE_Int);
            }
         }

         // wait for all buffer sizes to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests, status );
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         //////////// Communicate buffers ////////////

         // allocate space for the receive buffers and post the receives
         for (i = 0; i < num_recv_procs; i++)
         {
            recv_buffer[i] = hypre_CTAlloc(HYPRE_Int, recv_buffer_size[level][i], HYPRE_MEMORY_HOST );
            hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 1, comm, &(requests[request_counter++]));
         }

         // send the buffers
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_MPI_Isend(send_buffer[i], send_buffer_size[level][i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 1, comm, &(requests[request_counter++]));
            if (communication_cost)
            {
               communication_cost[level*10 + 2]++;
               communication_cost[level*10 + 3] += send_buffer_size[level][i]*sizeof(HYPRE_Int);
            }
         }

         // wait for buffers to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests, status );
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         if (timers) hypre_EndTiming(timers[3]);

         //////////// Unpack the received buffers ////////////

         if (timers) hypre_BeginTiming(timers[4]);
         HYPRE_Int **A_tmp_info = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_recv_procs; i++)
         {
            recv_map[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            recv_redundant_marker[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_recv_nodes[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);

            UnpackRecvBuffer(recv_buffer[i], compGrid, hypre_ParCSRMatrixCommPkg( hypre_ParAMGDataAArray(amg_data)[level] ),
               A_tmp_info,
               compGridCommPkg,
               send_flag, num_send_nodes, 
               recv_map, recv_redundant_marker, num_recv_nodes, 
               &(recv_map_send_buffer_size[i]), level, num_levels, nodes_added_on_level, i, num_resizes, 
               symmetric);
            
            recv_map_send_buffer[i] = hypre_CTAlloc(HYPRE_Int, recv_map_send_buffer_size[i], HYPRE_MEMORY_HOST);
            PackRecvMapSendBuffer(recv_map_send_buffer[i], recv_redundant_marker[level][i], num_recv_nodes[level][i], &(recv_buffer_size[level][i]), level, num_levels, compGrid);
         }
         if (timers) hypre_EndTiming(timers[4]);

         //////////// Setup local indices for the composite grid ////////////

         if (timers) hypre_BeginTiming(timers[5]);
         /* total_bin_search_count += hypre_ParCompGridSetupLocalIndices(compGrid, nodes_added_on_level, recv_map, num_recv_procs, A_tmp_info, level, num_levels, symmetric); */
         hypre_ParCompGridSetupLocalIndicesGPU(compGrid, nodes_added_on_level, recv_map, num_recv_procs, A_tmp_info, level, num_levels, symmetric);
         for (j = level; j < num_levels; j++) nodes_added_on_level[j] = 0;

         if (timers) hypre_EndTiming(timers[5]);

         //////////// Communicate redundancy info ////////////

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (timers) hypre_BeginTiming(timers[6]);  

         // post receives for send maps
         for (i = 0; i < num_send_procs; i++)
         {
            // !!! Check send_flag_buffer_size (make sure not overallocated)
            send_flag_buffer[i] = hypre_CTAlloc(HYPRE_Int, send_flag_buffer_size[i], HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv( send_flag_buffer[i], send_flag_buffer_size[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 2, comm, &(requests[request_counter++]));
         }

         // send the recv_map_send_buffer's
         for (i = 0; i < num_recv_procs; i++)
         {
            // !!! Check recv_map_send_buffer_size (make sure not overallocated)
            hypre_MPI_Isend( recv_map_send_buffer[i], recv_map_send_buffer_size[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 2, comm, &(requests[request_counter++]));
            if (communication_cost)
            {
               communication_cost[level*10 + 2]++;
               communication_cost[level*10 + 3] += recv_map_send_buffer_size[i]*sizeof(HYPRE_Int);
            }
         }

         // wait for maps to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests, status );
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);

         // unpack and setup the send flag arrays
         for (i = 0; i < num_send_procs; i++)
         {
            total_redundant_sends += UnpackSendFlagBuffer(compGrid, send_flag_buffer[i], send_flag[level][i], num_send_nodes[level][i], &(send_buffer_size[level][i]), level, num_levels);
         }

         if (timers) hypre_EndTiming(timers[6]);

         // clean up memory for this level
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_TFree(send_buffer[i], HYPRE_MEMORY_SHARED);
            hypre_TFree(send_flag_buffer[i], HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < num_recv_procs; i++)
         {
            hypre_TFree(recv_buffer[i], HYPRE_MEMORY_HOST);
            hypre_TFree(recv_map_send_buffer[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(send_buffer, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_buffer, HYPRE_MEMORY_HOST);
         hypre_TFree(send_flag_buffer, HYPRE_MEMORY_HOST);
         hypre_TFree(send_flag_buffer_size, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_map_send_buffer, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_map_send_buffer_size, HYPRE_MEMORY_HOST);
      }
      else 
      {
         if (use_barriers)
         {
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         }
      }
      #if DEBUGGING_MESSAGES
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      if (myid == 0) hypre_printf("All ranks: done with level %d\n", level);
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      #endif 

      #if DEBUG_COMP_GRID == 2
      HYPRE_Int l;
      for (l = 0; l < num_levels; l++)
      {
         hypre_sprintf(filename, "outputs/CompGrids/currentLevel%dCompGridRank%dLevel%d.txt", level, myid, l);
         hypre_ParCompGridDebugPrint( compGrid[l], filename );
      }
      #endif

      #if DEBUG_COMP_GRID
      HYPRE_Int error_code;
      error_code = TestCompGrids1(compGrid, num_levels, padding, num_ghost_layers, level, 1);
      if (error_code)
         hypre_printf("TestCompGrids1 failed! Rank %d, level %d\n", myid, level);
      else
         hypre_printf("TestCompGrids1 passed! Rank %d, level %d\n", myid, level);
      #endif
   }

   /////////////////////////////////////////////////////////////////

   // Done with loop over levels. Now just finalize things.

   /////////////////////////////////////////////////////////////////

   FixUpRecvMaps(compGrid, compGridCommPkg, recv_redundant_marker, amgdd_start_level, num_levels);
   
   #if DEBUG_COMP_GRID
   // Test whether comp grids have correct shape
   HYPRE_Int test_failed = 0;
   HYPRE_Int error_code;
   error_code = TestCompGrids1(compGrid, num_levels, padding, num_ghost_layers, 0, 1);
   if (error_code)
   {
      hypre_printf("TestCompGrids1 failed!\n");
      test_failed = 1;
   }
   else hypre_printf("TestCompGrids1 success\n");
   CheckCompGridCommPkg(compGridCommPkg);
   #endif

   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[7]);

   // Communicate data for A and all info for P
   CommunicateRemainingMatrixInfo(amg_data, compGrid, compGridCommPkg, communication_cost, symmetric);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with CommunicateRemainingMatrixInfo()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif 

   #if DEBUG_COMP_GRID
   error_code = TestCompGrids2(amg_data); // NOTE: test should come before setting up local indices for P (uses global col ind for P)
   if (error_code)
   {
      hypre_printf("TestCompGrids2New failed!\n");
      test_failed = 1;
   }
   else hypre_printf("TestCompGrids2New success\n");
   #endif

   if (timers) hypre_EndTiming(timers[7]);

   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[5]);

   // Setup the local indices for P
   hypre_ParCompGridSetupLocalIndicesP(amg_data, compGrid, amgdd_start_level, num_levels);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with hypre_ParCompGridSetupLocalIndicesP()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif 

   if (timers) hypre_EndTiming(timers[5]);

   #if DEBUG_COMP_GRID == 2
   for (level = 0; level < num_levels; level++)
   {
      hypre_sprintf(filename, "outputs/CompGrids/preFinalizeCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrint( compGrid[level], filename );
   }
   #endif

   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[8]);


   // Finalize the comp grid structures
   hypre_ParCompGridFinalize(amg_data, compGrid, compGridCommPkg, amgdd_start_level, num_levels, hypre_ParAMGDataAMGDDUseRD(amg_data), verify_amgdd);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with hypre_ParCompGridFinalize()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // Setup extra info for specific relaxation methods
   hypre_ParCompGridSetupRelax(amg_data);

   if (timers) hypre_EndTiming(timers[8]);

   // Count up the cost for subsequent residual communications
   if (communication_cost)
   {
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         communication_cost[level*10 + 4] += hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level];
         for (i = 0; i < hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level]; i++)
         {
            HYPRE_Int inner_level;
            for (inner_level = level; inner_level < num_levels; inner_level++)
            {
               communication_cost[level*10 + 5] += hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[level][i][inner_level]*sizeof(HYPRE_Complex);
            }
         }
      }
   }

   // Cleanup memory
   hypre_TFree(num_resizes, HYPRE_MEMORY_HOST);
   hypre_TFree(nodes_added_on_level, HYPRE_MEMORY_HOST);
   for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
   {
      for (j = 0; j < hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[i]; j++)
      {
         HYPRE_Int k;
         for (k = 0; k < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); k++)
         {
            if ( recv_redundant_marker[i][j][k] ) hypre_TFree( recv_redundant_marker[i][j][k], HYPRE_MEMORY_SHARED );
         }
         hypre_TFree( recv_redundant_marker[i][j], HYPRE_MEMORY_HOST );
      }
      hypre_TFree( recv_redundant_marker[i], HYPRE_MEMORY_HOST );
   }
   hypre_TFree( recv_redundant_marker, HYPRE_MEMORY_HOST );


   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Finished comp grid setup on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   #if DEBUG_COMP_GRID == 2
   for (level = 0; level < num_levels; level++)
   {
      hypre_sprintf(filename, "outputs/CompGrids/setupCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrint( compGrid[level], filename );
   }
   #endif

   // !!! Debug
   if (timers)
   {
      HYPRE_Int total_nonowned = 0;
      for (level = 0; level < num_levels; level++)
      {
         total_nonowned += hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
      }
      HYPRE_Int local_size_info[2];
      local_size_info[0] = total_nonowned;
      local_size_info[1] = total_redundant_sends;
      HYPRE_Int global_size_info[2];
      MPI_Reduce(local_size_info, global_size_info, 2, HYPRE_MPI_INT, MPI_SUM, 0, hypre_MPI_COMM_WORLD);
      if (myid == 0) printf("Total Setup Redundancy = %f\n", ((double) global_size_info[1] + global_size_info[0])/((double) global_size_info[0]));
   }


   #if DEBUG_COMP_GRID
   return test_failed;
   #else
   return 0;
   #endif
}

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int *recv_map_send_buffer, 
   HYPRE_Int **recv_redundant_marker, 
   HYPRE_Int *num_recv_nodes, 
   HYPRE_Int *recv_buffer_size,
   HYPRE_Int current_level, 
   HYPRE_Int num_levels,
   hypre_ParCompGrid **compGrid)
{
   HYPRE_Int      level, i, cnt, num_nodes;
   cnt = 0;
   *recv_buffer_size = 0;
   for (level = current_level+1; level < num_levels; level++)
   {
      // if there were nodes in psiComposite on this level
      if (recv_redundant_marker[level])
      {
         // store the number of nodes on this level
         num_nodes = num_recv_nodes[level];
         recv_map_send_buffer[cnt++] = num_nodes;

         for (i = 0; i < num_nodes; i++)
         {
            // store the map values for each node
            recv_map_send_buffer[cnt++] = recv_redundant_marker[level][i];
         }
      }
      // otherwise record that there were zero nodes on this level
      else recv_map_send_buffer[cnt++] = 0;
   }

   return 0;
}

HYPRE_Int
UnpackSendFlagBuffer(hypre_ParCompGrid **compGrid,
   HYPRE_Int *send_flag_buffer, 
   HYPRE_Int **send_flag, 
   HYPRE_Int *num_send_nodes,
   HYPRE_Int *send_buffer_size,
   HYPRE_Int current_level, 
   HYPRE_Int num_levels)
{
   // !!! Debug
   HYPRE_Int num_redundant_sends = 0;

   HYPRE_Int      level, i, cnt, num_nodes;
   cnt = 0;
   *send_buffer_size = 0;
   for (level = current_level+1; level < num_levels; level++)
   {
      num_nodes = send_flag_buffer[cnt++];
      num_send_nodes[level] = 0;

      for (i = 0; i < num_nodes; i++)
      {
         if (send_flag_buffer[cnt++] == 0) 
         {
            send_flag[level][ num_send_nodes[level]++ ] = send_flag[level][i];
            (*send_buffer_size)++;
         }
      }
      
      send_flag[level] = hypre_TReAlloc(send_flag[level], HYPRE_Int, num_send_nodes[level], HYPRE_MEMORY_SHARED);
      
      // !!! Debug
      num_redundant_sends += num_nodes - num_send_nodes[level];
   }

   return num_redundant_sends;
}

HYPRE_Int
CommunicateRemainingMatrixInfo(hypre_ParAMGData* amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *communication_cost, HYPRE_Int symmetric)
{
   HYPRE_Int outer_level,proc,level,i,j;
   HYPRE_Int num_levels = hypre_ParCompGridCommPkgNumLevels(compGridCommPkg);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);

   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   HYPRE_Int myid,num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   HYPRE_Int *P_row_cnt = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int *R_row_cnt = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int *A_row_cnt = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);

   for (outer_level = num_levels-1; outer_level >= amgdd_start_level; outer_level--)
   {

      // Initialize nonowned matrices for P (and R)
      if (outer_level != num_levels-1)
      {
         hypre_CSRMatrix *P_diag_original = hypre_ParCSRMatrixDiag(hypre_ParAMGDataPArray(amg_data)[outer_level]);
         hypre_CSRMatrix *P_offd_original = hypre_ParCSRMatrixOffd(hypre_ParAMGDataPArray(amg_data)[outer_level]);
         HYPRE_Int ave_nnz_per_row = 1;
         if (hypre_ParAMGDataPMaxElmts(amg_data)) // !!! Double check (when is this zero, negative, etc?)
            ave_nnz_per_row = hypre_ParAMGDataPMaxElmts(amg_data);
         else if (hypre_CSRMatrixNumRows(P_diag_original)) 
            ave_nnz_per_row = (HYPRE_Int) (hypre_CSRMatrixNumNonzeros(P_diag_original) / hypre_CSRMatrixNumRows(P_diag_original));
         HYPRE_Int max_nonowned_diag_nnz = hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level]) * ave_nnz_per_row;
         HYPRE_Int max_nonowned_offd_nnz = hypre_CSRMatrixNumNonzeros(P_offd_original);
         hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridP(compGrid[outer_level])) = hypre_CSRMatrixCreate(hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level]), hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level+1]), max_nonowned_diag_nnz);
         hypre_CSRMatrixInitialize(hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridP(compGrid[outer_level])));
         hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridP(compGrid[outer_level])) = hypre_CSRMatrixCreate(hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level]), hypre_ParCompGridNumOwnedNodes(compGrid[outer_level+1]), max_nonowned_offd_nnz);
         hypre_CSRMatrixInitialize(hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridP(compGrid[outer_level])));
      }
      if (hypre_ParAMGDataRestriction(amg_data) && outer_level != 0)
      {
         hypre_CSRMatrix *R_diag_original = hypre_ParCSRMatrixDiag(hypre_ParAMGDataPArray(amg_data)[outer_level-1]);
         hypre_CSRMatrix *R_offd_original = hypre_ParCSRMatrixOffd(hypre_ParAMGDataPArray(amg_data)[outer_level-1]);
         HYPRE_Int ave_nnz_per_row = 1;
         if (hypre_CSRMatrixNumRows(R_diag_original)) 
            ave_nnz_per_row = (HYPRE_Int) (hypre_CSRMatrixNumNonzeros(R_diag_original) / hypre_CSRMatrixNumRows(R_diag_original));
         HYPRE_Int max_nonowned_diag_nnz = hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level]) * ave_nnz_per_row;
         HYPRE_Int max_nonowned_offd_nnz = hypre_CSRMatrixNumNonzeros(R_offd_original);
         hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridR(compGrid[outer_level-1])) = hypre_CSRMatrixCreate(hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level]), hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level-1]), max_nonowned_diag_nnz);
         hypre_CSRMatrixInitialize(hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridR(compGrid[outer_level-1])));
         hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridR(compGrid[outer_level-1])) = hypre_CSRMatrixCreate(hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level]), hypre_ParCompGridNumOwnedNodes(compGrid[outer_level-1]), max_nonowned_offd_nnz);
         hypre_CSRMatrixInitialize(hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridR(compGrid[outer_level-1])));
      }

      // Get send/recv info from the comp grid comm pkg
      HYPRE_Int num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level];
      HYPRE_Int num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[outer_level];
      HYPRE_Int *send_procs = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[outer_level];
      HYPRE_Int *recv_procs = hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[outer_level];

      if (num_send_procs || num_recv_procs)
      {
         ////////////////////////////////////
         // Get the buffer sizes
         ////////////////////////////////////

         HYPRE_Int *send_sizes = hypre_CTAlloc(HYPRE_Int, 2*num_send_procs, HYPRE_MEMORY_HOST);
         for (proc = 0; proc < num_send_procs; proc++)
         {
            for (level = outer_level; level < num_levels; level++)
            {
               HYPRE_Int idx;
               HYPRE_Int A_row_size = 0;
               HYPRE_Int P_row_size = 0;
               HYPRE_Int R_row_size = 0;
               for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
               {
                  idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                  if (idx < 0) idx = -(idx + 1);

                  // Owned diag and offd
                  if (idx < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
                  {
                     diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[level]));
                     offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[level]));
                     A_row_size = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                     if (level != num_levels-1)
                     {
                        diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridP(compGrid[level]));
                        offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridP(compGrid[level]));
                        P_row_size = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                   + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                     }
                     if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
                     {
                        diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridR(compGrid[level-1]));
                        offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridR(compGrid[level-1]));
                        R_row_size = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                   + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                     }
                  }
                  // Nonowned diag and offd
                  else
                  {
                     idx -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
                     // Count diag and offd
                     diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level]));
                     offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level]));
                     A_row_size = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                     if (level != num_levels-1)
                     {
                        diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridP(compGrid[level]));
                        offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridP(compGrid[level]));
                        P_row_size = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                   + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                     }
                     if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
                     {
                        diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridR(compGrid[level-1]));
                        offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridR(compGrid[level-1]));
                        R_row_size = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                   + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                     }
                  }

                  send_sizes[2*proc] += A_row_size + P_row_size + R_row_size;
                  send_sizes[2*proc+1] += A_row_size + P_row_size + R_row_size;
               }
               if (level != num_levels-1) send_sizes[2*proc] += hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level];
               if (hypre_ParAMGDataRestriction(amg_data) && level != 0) send_sizes[2*proc] += hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level];
            }
         }


         HYPRE_Int **int_recv_buffers = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         HYPRE_Complex **complex_recv_buffers = hypre_CTAlloc(HYPRE_Complex*, num_recv_procs, HYPRE_MEMORY_HOST);

         // Communicate buffer sizes 
         hypre_MPI_Request *size_requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST);
         HYPRE_Int request_cnt = 0;
         hypre_MPI_Status *size_statuses = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST);
         HYPRE_Int *recv_sizes = hypre_CTAlloc(HYPRE_Int, 2*num_recv_procs, HYPRE_MEMORY_HOST);

         for (proc = 0; proc < num_recv_procs; proc++)
         {
            hypre_MPI_Irecv(&(recv_sizes[2*proc]), 2, HYPRE_MPI_INT, recv_procs[proc], 1, hypre_MPI_COMM_WORLD, &(size_requests[request_cnt++]));
         }
         for (proc = 0; proc < num_send_procs; proc++)
         {
            hypre_MPI_Isend(&(send_sizes[2*proc]), 2, HYPRE_MPI_INT, send_procs[proc], 1, hypre_MPI_COMM_WORLD, &(size_requests[request_cnt++]));
            
            if (communication_cost)
            {
               communication_cost[outer_level*10 + 2]++;
               communication_cost[outer_level*10 + 3] += 2*sizeof(HYPRE_Int);
            }
         }

         ////////////////////////////////////
         // Pack buffers
         ////////////////////////////////////

         // int_send_buffer = [ [level] , [level] , ... , [level] ]
         // level = [ [A col ind], [P_rows], ( [R_rows] ) ]
         // P_row = [ row_size, [col_ind] ]
         // complex_send_buffer = [ [level] , [level] , ... , [level] ]
         // level = [ [A_data] , [P_data], ( [R_data] ) ]

         hypre_MPI_Request *buf_requests = hypre_CTAlloc(hypre_MPI_Request, 2*(num_send_procs + num_recv_procs), HYPRE_MEMORY_HOST);
         request_cnt = 0;
         hypre_MPI_Status *buf_statuses = hypre_CTAlloc(hypre_MPI_Status, 2*(num_send_procs + num_recv_procs), HYPRE_MEMORY_HOST);
         HYPRE_Int **int_send_buffers = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
         HYPRE_Complex **complex_send_buffers = hypre_CTAlloc(HYPRE_Complex*, num_send_procs, HYPRE_MEMORY_HOST);
         for (proc = 0; proc < num_send_procs; proc++)
         {
            int_send_buffers[proc] = hypre_CTAlloc(HYPRE_Int, send_sizes[2*proc], HYPRE_MEMORY_HOST);
            complex_send_buffers[proc] = hypre_CTAlloc(HYPRE_Complex, send_sizes[2*proc+1], HYPRE_MEMORY_HOST);

            HYPRE_Int int_cnt = 0;
            HYPRE_Int complex_cnt = 0;
            for (level = outer_level; level < num_levels; level++)
            {
               // Pack A
               for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
               {
                  HYPRE_Int idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                  if (idx < 0) idx = -(idx + 1);

                  // Owned diag and offd
                  if (idx < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
                  {
                     diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[level]));
                     offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[level]));
                     for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx+1]; j++)
                     {
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(diag)[j] + hypre_ParCompGridFirstGlobalIndex(compGrid[level]);
                        complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                     }
                     for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx+1]; j++)
                     {
                        int_send_buffers[proc][int_cnt++] = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ hypre_CSRMatrixJ(offd)[j] ];
                        complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                     }
                  }
                  // Nonowned diag and offd
                  else
                  {
                     idx -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);

                     diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level]));
                     offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level]));
                     for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx+1]; j++)
                     {
                        if (hypre_CSRMatrixJ(diag)[j] < 0) int_send_buffers[proc][int_cnt++] = -(hypre_CSRMatrixJ(diag)[j]+1);
                        else int_send_buffers[proc][int_cnt++] = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ hypre_CSRMatrixJ(diag)[j] ];
                        complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                     }
                     for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx+1]; j++)
                     {
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(offd)[j] + hypre_ParCompGridFirstGlobalIndex(compGrid[level]);
                        complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                     }
                  }
               }
               // Pack P
               if (level != num_levels-1)
               {
                  for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) idx = -(idx + 1);

                     // Owned diag and offd
                     if (idx < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
                     {
                        diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridP(compGrid[level]));
                        offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridP(compGrid[level]));
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                                          + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                        for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx+1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(diag)[j] + hypre_ParCompGridFirstGlobalIndex(compGrid[level+1]);
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                        }
                        for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx+1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(offd)[j];
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                        }
                     }
                     // Nonowned diag and offd
                     else
                     {
                        idx -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
                        diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridP(compGrid[level]));
                        offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridP(compGrid[level]));
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                                          + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                        for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx+1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(diag)[j];
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                        }
                        for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx+1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(offd)[j] + hypre_ParCompGridFirstGlobalIndex(compGrid[level+1]);
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                        }
                     }
                  }
               }
               // Pack R
               if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
               {
                  for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) idx = -(idx + 1);

                     // Owned diag and offd
                     if (idx < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
                     {
                        diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridR(compGrid[level-1]));
                        offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridR(compGrid[level-1]));
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                                          + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                        for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx+1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(diag)[j] + hypre_ParCompGridFirstGlobalIndex(compGrid[level-1]);
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                        }
                        for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx+1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(offd)[j];
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                        }
                     }
                     // Nonowned diag and offd
                     else
                     {
                        idx -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
                        diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridR(compGrid[level-1]));
                        offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridR(compGrid[level-1]));
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                                          + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                        for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx+1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(diag)[j];
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                        }
                        for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx+1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(offd)[j] + hypre_ParCompGridFirstGlobalIndex(compGrid[level-1]);
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                        }
                     }
                  }
               }
            }
         }

         ////////////////////////////////////
         // Communicate
         ////////////////////////////////////

         for (proc = 0; proc < num_send_procs; proc++)
         {
            hypre_MPI_Isend(int_send_buffers[proc], send_sizes[2*proc], HYPRE_MPI_INT, send_procs[proc], 2, hypre_MPI_COMM_WORLD, &(buf_requests[request_cnt++]));
            hypre_MPI_Isend(complex_send_buffers[proc], send_sizes[2*proc+1], HYPRE_MPI_COMPLEX, send_procs[proc], 3, hypre_MPI_COMM_WORLD, &(buf_requests[request_cnt++]));
            if (communication_cost)
            {
               communication_cost[outer_level*10 + 2] += 2;
               communication_cost[outer_level*10 + 3] += send_sizes[2*proc]*sizeof(HYPRE_Int) + send_sizes[2*proc+1]*sizeof(HYPRE_Complex);
            }
         }

         // Wait on buffer sizes
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, size_requests, size_statuses );

         // Allocate and post recvs
         for (proc = 0; proc < num_recv_procs; proc++)
         {
            int_recv_buffers[proc] = hypre_CTAlloc(HYPRE_Int, recv_sizes[2*proc], HYPRE_MEMORY_HOST);
            complex_recv_buffers[proc] = hypre_CTAlloc(HYPRE_Complex, recv_sizes[2*proc+1], HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv(int_recv_buffers[proc], recv_sizes[2*proc], HYPRE_MPI_INT, recv_procs[proc], 2, hypre_MPI_COMM_WORLD, &(buf_requests[request_cnt++]));
            hypre_MPI_Irecv(complex_recv_buffers[proc], recv_sizes[2*proc+1], HYPRE_MPI_COMPLEX, recv_procs[proc], 3, hypre_MPI_COMM_WORLD, &(buf_requests[request_cnt++]));
         }

         // Wait on buffers
         hypre_MPI_Waitall( 2*(num_send_procs + num_recv_procs), buf_requests, buf_statuses );

         for (proc = 0; proc < num_send_procs; proc++) hypre_TFree(int_send_buffers[proc], HYPRE_MEMORY_HOST);
         for (proc = 0; proc < num_send_procs; proc++) hypre_TFree(complex_send_buffers[proc], HYPRE_MEMORY_HOST);
         hypre_TFree(int_send_buffers, HYPRE_MEMORY_HOST);
         hypre_TFree(complex_send_buffers, HYPRE_MEMORY_HOST);
         hypre_TFree(size_requests, HYPRE_MEMORY_HOST);
         hypre_TFree(size_statuses, HYPRE_MEMORY_HOST);
         hypre_TFree(buf_requests, HYPRE_MEMORY_HOST);
         hypre_TFree(buf_statuses, HYPRE_MEMORY_HOST);

         // P_tmp_info[buffer_number] = [ size, [row], size, [row], ... ]
         HYPRE_Int **P_tmp_info_int;
         HYPRE_Complex **P_tmp_info_complex;
         HYPRE_Int P_tmp_info_size = 0;
         HYPRE_Int P_tmp_info_cnt = 0;
         if (outer_level != num_levels-1)
         {
            for (proc = 0; proc < num_recv_procs; proc++) 
               P_tmp_info_size += hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][outer_level];
            P_tmp_info_size -= hypre_CSRMatrixNumCols(hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[outer_level])));
            P_tmp_info_int = hypre_CTAlloc(HYPRE_Int*, P_tmp_info_size, HYPRE_MEMORY_HOST);
            P_tmp_info_complex = hypre_CTAlloc(HYPRE_Complex*, P_tmp_info_size, HYPRE_MEMORY_HOST);
         }
         // R_tmp_info[buffer_number] = [ size, [row], size, [row], ... ]
         HYPRE_Int **R_tmp_info_int;
         HYPRE_Complex **R_tmp_info_complex;
         HYPRE_Int R_tmp_info_size = 0;
         HYPRE_Int R_tmp_info_cnt = 0;
         if (hypre_ParAMGDataRestriction(amg_data) && outer_level != 0)
         {
            for (proc = 0; proc < num_recv_procs; proc++) 
               R_tmp_info_size += hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][outer_level];
            R_tmp_info_size -= hypre_CSRMatrixNumCols(hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[outer_level])));
            R_tmp_info_int = hypre_CTAlloc(HYPRE_Int*, R_tmp_info_size, HYPRE_MEMORY_HOST);
            R_tmp_info_complex = hypre_CTAlloc(HYPRE_Complex*, R_tmp_info_size, HYPRE_MEMORY_HOST);
         }

         ////////////////////////////////////
         // Unpack recvs
         ////////////////////////////////////

         for (proc = 0; proc < num_recv_procs; proc++)
         {
            HYPRE_Int int_cnt = 0;
            HYPRE_Int complex_cnt = 0;

            for (level = outer_level; level < num_levels; level++)
            {               
               for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
               {
                  HYPRE_Int idx = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];

                  if (idx < 0) idx = -(idx + 1);

                  // !!! Optimization: I send (and setup) A info twice for ghosts overwritten as real
                  // !!! Double check ordering of incoming data vs. ordering of existing col ind
                  // Unpack A data
                  diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level]));
                  offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level]));
                  HYPRE_Int diag_rowptr = hypre_CSRMatrixI(diag)[idx];
                  HYPRE_Int offd_rowptr = hypre_CSRMatrixI(offd)[idx];

                  while (diag_rowptr < hypre_CSRMatrixI(diag)[idx+1] || offd_rowptr < hypre_CSRMatrixI(offd)[idx+1]) // !!! Double check
                  {
                     HYPRE_Int incoming_index = int_recv_buffers[proc][int_cnt++];

                     // See whether global index is owned
                     if (incoming_index >= hypre_ParCompGridFirstGlobalIndex(compGrid[level]) && incoming_index <= hypre_ParCompGridLastGlobalIndex(compGrid[level]))
                     {
                        // Don't overwrite data if already accounted for (ordering can change and screw things up)
                        if (level == outer_level || idx == A_row_cnt[level])
                        {
                           hypre_CSRMatrixData(offd)[offd_rowptr++] = complex_recv_buffers[proc][complex_cnt++];
                        }
                        else 
                        {
                           complex_cnt++;
                           offd_rowptr++;
                        }
                     }
                     else
                     {
                        // Don't overwrite data if already accounted for (ordering can change and screw things up)
                        if (level == outer_level || idx == A_row_cnt[level])
                        {
                           hypre_CSRMatrixData(diag)[diag_rowptr++] = complex_recv_buffers[proc][complex_cnt++];
                        }
                        else
                        {
                           complex_cnt++;
                           diag_rowptr++;
                        }
                     }
                  }
                  if (level != outer_level && idx == A_row_cnt[level]) A_row_cnt[level]++;
               }
               
               if (level == outer_level) A_row_cnt[level] += hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level];

               // Unpack P data and col indices
               if (level != num_levels-1)
               {
                  diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridP(compGrid[level]));
                  offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridP(compGrid[level]));

                  for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) idx = -(idx + 1);

                     // Setup orig commPkg recv dofs
                     if (idx == P_row_cnt[level])
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];

                        HYPRE_Int diag_rowptr = hypre_CSRMatrixI(diag)[idx];
                        HYPRE_Int offd_rowptr = hypre_CSRMatrixI(offd)[idx];

                        for (j = 0; j < row_size; j++)
                        {
                           HYPRE_Int incoming_index = int_recv_buffers[proc][int_cnt++];

                           // See whether global index is owned
                           if (incoming_index >= hypre_ParCompGridFirstGlobalIndex(compGrid[level+1]) && incoming_index <= hypre_ParCompGridLastGlobalIndex(compGrid[level+1]))
                           {
                              if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(offd))
                                 hypre_CSRMatrixResize(offd, hypre_CSRMatrixNumRows(offd), hypre_CSRMatrixNumCols(offd), ceil(1.5*hypre_CSRMatrixNumNonzeros(offd) + 1));
                              hypre_CSRMatrixJ(offd)[offd_rowptr] = incoming_index - hypre_ParCompGridFirstGlobalIndex(compGrid[level+1]);
                              hypre_CSRMatrixData(offd)[offd_rowptr] = complex_recv_buffers[proc][complex_cnt++];
                              offd_rowptr++;
                           }
                           else
                           {
                              if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(diag))
                                 hypre_CSRMatrixResize(diag, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(diag) + 1));
                              hypre_CSRMatrixJ(diag)[diag_rowptr] = incoming_index;
                              hypre_CSRMatrixData(diag)[diag_rowptr] = complex_recv_buffers[proc][complex_cnt++];
                              diag_rowptr++;
                           }
                        }
                        hypre_CSRMatrixI(diag)[idx+1] = diag_rowptr;
                        hypre_CSRMatrixI(offd)[idx+1] = offd_rowptr;

                        P_row_cnt[level]++;
                     }
                     // Store info for later setup on current outer level
                     else if (level == outer_level)
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];
                        P_tmp_info_int[P_tmp_info_cnt] = hypre_CTAlloc(HYPRE_Int, row_size+1, HYPRE_MEMORY_HOST);
                        P_tmp_info_complex[P_tmp_info_cnt] = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
                        P_tmp_info_int[P_tmp_info_cnt][0] = row_size;
                        for (j = 0; j < row_size; j++)
                        {
                           P_tmp_info_int[P_tmp_info_cnt][j+1] = int_recv_buffers[proc][int_cnt++];
                           P_tmp_info_complex[P_tmp_info_cnt][j] = complex_recv_buffers[proc][complex_cnt++];
                        }
                        P_tmp_info_cnt++;
                     }
                     // Otherwise, simply advance counters appropriately
                     else
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];
                        for (j = 0; j < row_size; j++)
                        {
                           int_cnt++;
                           complex_cnt++;
                        }
                     }
                  }
               }
               // Unpack R data and col indices
               if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
               {
                  diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridR(compGrid[level-1]));
                  offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridR(compGrid[level-1]));

                  for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) idx = -(idx + 1);

                     // Setup orig commPkg recv dofs
                     if (idx == R_row_cnt[level-1])
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];

                        HYPRE_Int diag_rowptr = hypre_CSRMatrixI(diag)[idx];
                        HYPRE_Int offd_rowptr = hypre_CSRMatrixI(offd)[idx];

                        for (j = 0; j < row_size; j++)
                        {
                           HYPRE_Int incoming_index = int_recv_buffers[proc][int_cnt++];

                           // See whether global index is owned
                           if (incoming_index >= hypre_ParCompGridFirstGlobalIndex(compGrid[level-1]) && incoming_index <= hypre_ParCompGridLastGlobalIndex(compGrid[level-1]))
                           {
                              if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(offd))
                                 hypre_CSRMatrixResize(offd, hypre_CSRMatrixNumRows(offd), hypre_CSRMatrixNumCols(offd), ceil(1.5*hypre_CSRMatrixNumNonzeros(offd) + 1));
                              hypre_CSRMatrixJ(offd)[offd_rowptr] = incoming_index - hypre_ParCompGridFirstGlobalIndex(compGrid[level-1]);
                              hypre_CSRMatrixData(offd)[offd_rowptr] = complex_recv_buffers[proc][complex_cnt++];
                              offd_rowptr++;
                           }
                           else
                           {
                              if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(diag))
                                 hypre_CSRMatrixResize(diag, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(diag) + 1));
                              hypre_CSRMatrixJ(diag)[diag_rowptr] = incoming_index;
                              hypre_CSRMatrixData(diag)[diag_rowptr] = complex_recv_buffers[proc][complex_cnt++];
                              diag_rowptr++;
                           }
                        }
                        hypre_CSRMatrixI(diag)[idx+1] = diag_rowptr;
                        hypre_CSRMatrixI(offd)[idx+1] = offd_rowptr;

                        R_row_cnt[level-1]++;
                     }
                     // Store info for later setup on current outer level
                     else if (level == outer_level)
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];
                        R_tmp_info_int[R_tmp_info_cnt] = hypre_CTAlloc(HYPRE_Int, row_size+1, HYPRE_MEMORY_HOST);
                        R_tmp_info_complex[R_tmp_info_cnt] = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
                        R_tmp_info_int[R_tmp_info_cnt][0] = row_size;
                        for (j = 0; j < row_size; j++)
                        {
                           R_tmp_info_int[R_tmp_info_cnt][j+1] = int_recv_buffers[proc][int_cnt++];
                           R_tmp_info_complex[R_tmp_info_cnt][j] = complex_recv_buffers[proc][complex_cnt++];
                        }
                        R_tmp_info_cnt++;
                     }
                     // Otherwise, simply advance counters appropriately
                     else
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];
                        for (j = 0; j < row_size; j++)
                        {
                           int_cnt++;
                           complex_cnt++;
                        }
                     }
                  }   
               }
            }
         }

         // Setup temporary info for P on current level
         if (outer_level != num_levels-1)
         {
            diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridP(compGrid[outer_level]));
            offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridP(compGrid[outer_level]));

            HYPRE_Int diag_rowptr = hypre_CSRMatrixI(diag)[ P_row_cnt[outer_level] ];
            HYPRE_Int offd_rowptr = hypre_CSRMatrixI(offd)[ P_row_cnt[outer_level] ];

            for (i = 0; i < P_tmp_info_size; i++)
            {
               if (P_tmp_info_int[i])
               {
                  HYPRE_Int row_size = P_tmp_info_int[i][0];
                  for (j = 0; j < row_size; j++)
                  {
                     HYPRE_Int incoming_index = P_tmp_info_int[i][j+1];

                     // See whether global index is owned
                     if (incoming_index >= hypre_ParCompGridFirstGlobalIndex(compGrid[outer_level+1]) && incoming_index <= hypre_ParCompGridLastGlobalIndex(compGrid[outer_level+1]))
                     {
                        if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(offd))
                           hypre_CSRMatrixResize(offd, hypre_CSRMatrixNumRows(offd), hypre_CSRMatrixNumCols(offd), ceil(1.5*hypre_CSRMatrixNumNonzeros(offd) + 1));
                        hypre_CSRMatrixJ(offd)[offd_rowptr] = incoming_index - hypre_ParCompGridFirstGlobalIndex(compGrid[outer_level+1]);
                        hypre_CSRMatrixData(offd)[offd_rowptr] = P_tmp_info_complex[i][j];
                        offd_rowptr++;
                     }
                     else
                     {
                        if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(diag))
                           hypre_CSRMatrixResize(diag, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(diag) + 1));
                        hypre_CSRMatrixJ(diag)[diag_rowptr] = incoming_index;
                        hypre_CSRMatrixData(diag)[diag_rowptr] = P_tmp_info_complex[i][j];
                        diag_rowptr++;
                     }

                  }
                  hypre_CSRMatrixI(diag)[P_row_cnt[outer_level]+1] = diag_rowptr;
                  hypre_CSRMatrixI(offd)[P_row_cnt[outer_level]+1] = offd_rowptr;
                  P_row_cnt[outer_level]++;

                  hypre_TFree(P_tmp_info_int[i], HYPRE_MEMORY_HOST);
                  hypre_TFree(P_tmp_info_complex[i], HYPRE_MEMORY_HOST);
               }
            }

            hypre_TFree(P_tmp_info_int, HYPRE_MEMORY_HOST);
            hypre_TFree(P_tmp_info_complex, HYPRE_MEMORY_HOST);
         }
         // Setup temporary info for R on current level
         if (hypre_ParAMGDataRestriction(amg_data) && outer_level != 0)
         {
            diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridR(compGrid[outer_level-1]));
            offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridR(compGrid[outer_level-1]));

            HYPRE_Int diag_rowptr = hypre_CSRMatrixI(diag)[ R_row_cnt[outer_level-1] ];
            HYPRE_Int offd_rowptr = hypre_CSRMatrixI(offd)[ R_row_cnt[outer_level-1] ];

            for (i = 0; i < R_tmp_info_size; i++)
            {
               if (R_tmp_info_int[i])
               {
                  HYPRE_Int row_size = R_tmp_info_int[i][0];
                  for (j = 0; j < row_size; j++)
                  {
                     HYPRE_Int incoming_index = R_tmp_info_int[i][j+1];

                     // See whether global index is owned
                     if (incoming_index >= hypre_ParCompGridFirstGlobalIndex(compGrid[outer_level-1]) && incoming_index <= hypre_ParCompGridLastGlobalIndex(compGrid[outer_level-1]))
                     {
                        if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(offd))
                           hypre_CSRMatrixResize(offd, hypre_CSRMatrixNumRows(offd), hypre_CSRMatrixNumCols(offd), ceil(1.5*hypre_CSRMatrixNumNonzeros(offd) + 1));
                        hypre_CSRMatrixJ(offd)[offd_rowptr] = incoming_index - hypre_ParCompGridFirstGlobalIndex(compGrid[outer_level-1]);
                        hypre_CSRMatrixData(offd)[offd_rowptr] = R_tmp_info_complex[i][j];
                        offd_rowptr++;
                     }
                     else
                     {
                        if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(diag))
                           hypre_CSRMatrixResize(diag, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(diag) + 1));
                        hypre_CSRMatrixJ(diag)[diag_rowptr] = incoming_index;
                        hypre_CSRMatrixData(diag)[diag_rowptr] = R_tmp_info_complex[i][j];
                        diag_rowptr++;
                     }

                  }
                  hypre_CSRMatrixI(diag)[R_row_cnt[outer_level-1]+1] = diag_rowptr;
                  hypre_CSRMatrixI(offd)[R_row_cnt[outer_level-1]+1] = offd_rowptr;
                  R_row_cnt[outer_level-1]++;

                  hypre_TFree(R_tmp_info_int[i], HYPRE_MEMORY_HOST);
                  hypre_TFree(R_tmp_info_complex[i], HYPRE_MEMORY_HOST);
               }
            }

            hypre_TFree(R_tmp_info_int, HYPRE_MEMORY_HOST);
            hypre_TFree(R_tmp_info_complex, HYPRE_MEMORY_HOST);
         }

         // Clean up memory
         for (proc = 0; proc < num_recv_procs; proc++) hypre_TFree(int_recv_buffers[proc], HYPRE_MEMORY_HOST);
         for (proc = 0; proc < num_recv_procs; proc++) hypre_TFree(complex_recv_buffers[proc], HYPRE_MEMORY_HOST);
         hypre_TFree(int_recv_buffers, HYPRE_MEMORY_HOST);
         hypre_TFree(complex_recv_buffers, HYPRE_MEMORY_HOST);
         hypre_TFree(send_sizes, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_sizes, HYPRE_MEMORY_HOST);
      }

      #if DEBUGGING_MESSAGES
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      if (myid == 0) hypre_printf("   All ranks: done with CommunicateRemainingMatrixInfo() level %d\n", outer_level);
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      #endif 
   }

   return 0;
}

HYPRE_Int
TestCompGrids1(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int current_level, HYPRE_Int check_ghost_info)
{
   // TEST 1: See whether the parallel composite grid algorithm algorithm has constructed a composite grid with 
   // the same shape (and ghost node info) as we expect from serial, top-down composite grid generation
   HYPRE_Int            level,i;
   HYPRE_Int            need_coarse_info, nodes_to_add = 1;
   HYPRE_Int            **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            test_failed = 0;
   HYPRE_Int            error_code = 0;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   // Allocate add_flag on each level and mark the owned dofs on the finest grid
   for (level = current_level; level < num_levels; level++) 
      add_flag[level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]), HYPRE_MEMORY_HOST);
   for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid[current_level]); i++) 
      add_flag[current_level][i] = padding[current_level] + 1;

   // Serially generate comp grid from top down
   // Note that if nodes that should be present in the comp grid are not found, we will be alerted by the error message in RecursivelyBuildPsiComposite()
   for (level = current_level; level < num_levels; level++)
   {
      HYPRE_Int total_num_nodes = hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
      
      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         nodes_to_add = 0;

         // see whether we need coarse info on this level
         if (level != num_levels-1) need_coarse_info = 1;
         else need_coarse_info = 0;

         // Expand by the padding on this level and add coarse grid counterparts if applicable
         for (i = 0; i < total_num_nodes; i++)
         {
            if (add_flag[level][i] == padding[level] + 1)
            {
               if (level != num_levels-1) error_code = RecursivelyBuildPsiComposite(i, padding[level], compGrid, add_flag, 1, &nodes_to_add, padding[level+1], level, 0);
               else error_code = RecursivelyBuildPsiComposite(i, padding[level], compGrid, add_flag, 0, NULL, 0, level, 0);
               if (error_code)
               {
                  hypre_printf("Error: expand padding\n");
                  test_failed = 1;
               }
            }
         }

         // Expand by the number of ghost layers 
         for (i = 0; i < total_num_nodes; i++)
         {
            if (add_flag[level][i] > 1) add_flag[level][i] = num_ghost_layers + 2;
            else if (add_flag[level][i] == 1) add_flag[level][i] = num_ghost_layers + 1;
         }
         for (i = 0; i < total_num_nodes; i++)
         {
            // Recursively add the region of ghost nodes (do not add any coarse nodes underneath)
            if (add_flag[level][i] == num_ghost_layers + 1) error_code = RecursivelyBuildPsiComposite(i, padding[level], compGrid, add_flag, 0, NULL, 0, level, 0);
            if (error_code)
            {
               hypre_printf("Error: recursively add the region of ghost nodes\n");
               test_failed = 1;
            }
         }
      }
      else break;

      // Check whether add_flag has any zeros (zeros indicate that we have extra nodes in the comp grid that don't belong) 
      for (i = 0; i < total_num_nodes; i++)
      {
         if (add_flag[level][i] == 0) 
         {
            test_failed = 1;
            if (i < hypre_ParCompGridNumOwnedNodes(compGrid[level])) 
               hypre_printf("Error: extra OWNED (i.e. test is broken) nodes present in comp grid, rank %d, level %d, i = %d, global index = %d\n", 
                  myid, level, i, i + hypre_ParCompGridFirstGlobalIndex(compGrid[level]));
            else
               hypre_printf("Error: extra nonowned nodes present in comp grid, rank %d, level %d, i = %d, global index = %d\n", 
                  myid, level, i - hypre_ParCompGridNumOwnedNodes(compGrid[level]), hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])]);
         }
      }

      // Check to make sure we have the correct identification of ghost nodes
      if (level != num_levels-1 && check_ghost_info)
      {
         for (i = hypre_ParCompGridNumOwnedNodes(compGrid[level]); i < total_num_nodes; i++) 
         {
            if (add_flag[level][i] < num_ghost_layers + 1 && hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])] != 0) 
            {
               test_failed = 1;
               hypre_printf("Error: dof that should have been marked as ghost was marked as real, rank %d, level %d, GID %d\n", 
                  myid, level, hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])]);
            }
            if (add_flag[level][i] > num_ghost_layers && hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])] == 0) 
            {
               test_failed = 1;
               hypre_printf("Error: dof that should have been marked as real was marked as ghost, rank %d, level %d\n", myid, level);
            }
         }
      }
   }

   // Clean up memory
   for (level = 0; level < num_levels; level++) if (add_flag[level]) hypre_TFree(add_flag[level], HYPRE_MEMORY_HOST);
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);

   return test_failed;
}

HYPRE_Int
TestCompGrids2(hypre_ParAMGData *amg_data)
{
   // TEST 2: See whether the dofs in the composite grid have the correct info.
   // Each processor in turn will broadcast out the info associate with its composite grids on each level.
   // The processors owning the original info will check to make sure their info matches the comp grid info that was broadcasted out.
   // This occurs for the matrix info (row pointer, column indices, and data for A and P) and the initial right-hand side 
   
   // Get MPI info
   HYPRE_Int myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   hypre_ParCompGrid **compGrid = hypre_ParAMGDataCompGrid(amg_data);
   hypre_ParCompGridCommPkg *compGridCommPkg = hypre_ParAMGDataCompGridCommPkg(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   hypre_ParCSRMatrix **A = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P = hypre_ParAMGDataPArray(amg_data);
   hypre_ParCSRMatrix **R = hypre_ParAMGDataRArray(amg_data);

   HYPRE_Int i,j,k;
   HYPRE_Int test_failed = 0;

   // For each processor and each level broadcast the global indices, coarse indices, and matrix info out and check against the owning procs
   HYPRE_Int proc;
   for (proc = 0; proc < num_procs; proc++)
   {
      HYPRE_Int level;
      for (level = 0; level < num_levels; level++)
      {
         /////////////////////////////////
         // Broadcast all info
         /////////////////////////////////

         hypre_CSRMatrix *A_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level]));
         hypre_CSRMatrix *A_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level]));
         hypre_CSRMatrix *P_diag = NULL;
         hypre_CSRMatrix *P_offd = NULL;
         hypre_CSRMatrix *R_diag = NULL;
         hypre_CSRMatrix *R_offd = NULL;


         // Broadcast the number of nodes and num non zeros for A, P, and R
         HYPRE_Int num_nonowned = 0;
         HYPRE_Int proc_first_index = 0;
         HYPRE_Int proc_fine_first_index = 0;
         HYPRE_Int proc_coarse_first_index = 0;
         HYPRE_Int nnz_A_diag = 0;
         HYPRE_Int nnz_A_offd = 0;
         HYPRE_Int nnz_P_diag = 0;
         HYPRE_Int nnz_P_offd = 0;
         HYPRE_Int nnz_R_diag = 0;
         HYPRE_Int nnz_R_offd = 0;
         HYPRE_Int sizes_buf[10];
         if (myid == proc) 
         {
            num_nonowned = hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
            proc_first_index = hypre_ParCompGridFirstGlobalIndex(compGrid[level]);
            nnz_A_diag = hypre_CSRMatrixI(A_diag)[num_nonowned];
            nnz_A_offd = hypre_CSRMatrixI(A_offd)[num_nonowned];
            if (level != num_levels-1)
            {
               proc_coarse_first_index = hypre_ParCompGridFirstGlobalIndex(compGrid[level+1]);
               P_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridP(compGrid[level]));
               P_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridP(compGrid[level]));
               nnz_P_diag = hypre_CSRMatrixI(P_diag)[num_nonowned];
               nnz_P_offd = hypre_CSRMatrixI(P_offd)[num_nonowned];
            }
            if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
            {
               proc_fine_first_index = hypre_ParCompGridFirstGlobalIndex(compGrid[level-1]);
               R_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridR(compGrid[level-1]));
               R_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridR(compGrid[level-1]));
               nnz_R_diag = hypre_CSRMatrixI(R_diag)[num_nonowned];
               nnz_R_offd = hypre_CSRMatrixI(R_offd)[num_nonowned];
            }
            sizes_buf[0] = num_nonowned;
            sizes_buf[1] = proc_first_index;
            sizes_buf[2] = proc_fine_first_index;
            sizes_buf[3] = proc_coarse_first_index;
            sizes_buf[4] = nnz_A_diag;
            sizes_buf[5] = nnz_A_offd;
            sizes_buf[6] = nnz_P_diag;
            sizes_buf[7] = nnz_P_offd;
            sizes_buf[8] = nnz_R_diag;
            sizes_buf[9] = nnz_R_offd;
         }
         hypre_MPI_Bcast(sizes_buf, 10, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);
         num_nonowned = sizes_buf[0];
         proc_first_index = sizes_buf[1];
         proc_fine_first_index = sizes_buf[2];
         proc_coarse_first_index = sizes_buf[3];
         nnz_A_diag = sizes_buf[4];
         nnz_A_offd = sizes_buf[5];
         nnz_P_diag = sizes_buf[6];
         nnz_P_offd = sizes_buf[7];
         nnz_R_diag = sizes_buf[8];
         nnz_R_offd = sizes_buf[9];

         // Broadcast the global indices
         HYPRE_Int *global_indices;
         if (myid == proc) global_indices = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]);
         else global_indices = hypre_CTAlloc(HYPRE_Int, num_nonowned, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(global_indices, num_nonowned, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the A diag row pointer
         HYPRE_Int *A_diag_rowPtr;
         if (myid == proc) A_diag_rowPtr = hypre_CSRMatrixI(A_diag);
         else A_diag_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(A_diag_rowPtr, num_nonowned+1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the A diag column indices
         HYPRE_Int *A_diag_colInd;
         if (myid == proc) A_diag_colInd = hypre_CSRMatrixJ(A_diag);
         else A_diag_colInd = hypre_CTAlloc(HYPRE_Int, nnz_A_diag, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(A_diag_colInd, nnz_A_diag, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the A diag data
         HYPRE_Complex *A_diag_data;
         if (myid == proc) A_diag_data = hypre_CSRMatrixData(A_diag);
         else A_diag_data = hypre_CTAlloc(HYPRE_Complex, nnz_A_diag, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(A_diag_data, nnz_A_diag, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the A offd row pointer
         HYPRE_Int *A_offd_rowPtr;
         if (myid == proc) A_offd_rowPtr = hypre_CSRMatrixI(A_offd);
         else A_offd_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(A_offd_rowPtr, num_nonowned+1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the A offd column indices
         HYPRE_Int *A_offd_colInd;
         if (myid == proc) A_offd_colInd = hypre_CSRMatrixJ(A_offd);
         else A_offd_colInd = hypre_CTAlloc(HYPRE_Int, nnz_A_offd, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(A_offd_colInd, nnz_A_offd, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the A offd data
         HYPRE_Complex *A_offd_data;
         if (myid == proc) A_offd_data = hypre_CSRMatrixData(A_offd);
         else A_offd_data = hypre_CTAlloc(HYPRE_Complex, nnz_A_offd, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(A_offd_data, nnz_A_offd, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);

         HYPRE_Int *coarse_indices;
         HYPRE_Int *P_diag_rowPtr;
         HYPRE_Int *P_diag_colInd;
         HYPRE_Complex *P_diag_data;
         HYPRE_Int *P_offd_rowPtr;
         HYPRE_Int *P_offd_colInd;
         HYPRE_Complex *P_offd_data;
         if (level != num_levels-1)
         {
            // Broadcast the coarse global indices
            coarse_indices = hypre_CTAlloc(HYPRE_Int, num_nonowned, HYPRE_MEMORY_HOST);
            if (myid == proc)
            {
               for (i = 0; i < num_nonowned; i++)
               {
                  HYPRE_Int coarse_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level])[i];
                  if (coarse_index >= 0)
                     coarse_indices[i] = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level+1])[ coarse_index ];
                  else
                     coarse_indices[i] = -1;
               }
            }
            hypre_MPI_Bcast(coarse_indices, num_nonowned, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P diag row pointer
            if (myid == proc) P_diag_rowPtr = hypre_CSRMatrixI(P_diag);
            else P_diag_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(P_diag_rowPtr, num_nonowned+1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P diag column indices
            if (myid == proc) P_diag_colInd = hypre_CSRMatrixJ(P_diag);
            else P_diag_colInd = hypre_CTAlloc(HYPRE_Int, nnz_P_diag, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(P_diag_colInd, nnz_P_diag, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P diag data
            if (myid == proc) P_diag_data = hypre_CSRMatrixData(P_diag);
            else P_diag_data = hypre_CTAlloc(HYPRE_Complex, nnz_P_diag, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(P_diag_data, nnz_P_diag, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P offd row pointer
            if (myid == proc) P_offd_rowPtr = hypre_CSRMatrixI(P_offd);
            else P_offd_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(P_offd_rowPtr, num_nonowned+1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P offd column indices
            if (myid == proc) P_offd_colInd = hypre_CSRMatrixJ(P_offd);
            else P_offd_colInd = hypre_CTAlloc(HYPRE_Int, nnz_P_offd, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(P_offd_colInd, nnz_P_offd, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P offd data
            if (myid == proc) P_offd_data = hypre_CSRMatrixData(P_offd);
            else P_offd_data = hypre_CTAlloc(HYPRE_Complex, nnz_P_offd, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(P_offd_data, nnz_P_offd, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);
         }

         HYPRE_Int *R_diag_rowPtr;
         HYPRE_Int *R_diag_colInd;
         HYPRE_Complex *R_diag_data;
         HYPRE_Int *R_offd_rowPtr;
         HYPRE_Int *R_offd_colInd;
         HYPRE_Complex *R_offd_data;
         if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
         {
            // Broadcast the R diag row pointer
            if (myid == proc) R_diag_rowPtr = hypre_CSRMatrixI(R_diag);
            else R_diag_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(R_diag_rowPtr, num_nonowned+1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P diag column indices
            if (myid == proc) R_diag_colInd = hypre_CSRMatrixJ(R_diag);
            else R_diag_colInd = hypre_CTAlloc(HYPRE_Int, nnz_R_diag, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(R_diag_colInd, nnz_R_diag, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P diag data
            if (myid == proc) R_diag_data = hypre_CSRMatrixData(R_diag);
            else R_diag_data = hypre_CTAlloc(HYPRE_Complex, nnz_R_diag, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(R_diag_data, nnz_R_diag, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P offd row pointer
            if (myid == proc) R_offd_rowPtr = hypre_CSRMatrixI(R_offd);
            else R_offd_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(R_offd_rowPtr, num_nonowned+1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P offd column indices
            if (myid == proc) R_offd_colInd = hypre_CSRMatrixJ(R_offd);
            else R_offd_colInd = hypre_CTAlloc(HYPRE_Int, nnz_R_offd, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(R_offd_colInd, nnz_R_offd, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the P offd data
            if (myid == proc) R_offd_data = hypre_CSRMatrixData(R_offd);
            else R_offd_data = hypre_CTAlloc(HYPRE_Complex, nnz_R_offd, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(R_offd_data, nnz_R_offd, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);
         }

         /////////////////////////////////
         // Check all info
         /////////////////////////////////

         // Now, each processors checks their owned info against the composite grid info
         HYPRE_Int my_first_index = hypre_ParCSRMatrixFirstRowIndex(A[level]);
         HYPRE_Int my_last_index = hypre_ParCSRMatrixLastRowIndex(A[level]);
         for (i = 0; i < num_nonowned; i++)
         {
            if (global_indices[i] <= my_last_index && global_indices[i] >= my_first_index)
            {
               /////////////////////////////////
               // Check A info
               /////////////////////////////////

               HYPRE_Int row_size;
               HYPRE_Int *row_col_ind;
               HYPRE_Complex *row_values;
               hypre_ParCSRMatrixGetRow( A[level], global_indices[i], &row_size, &row_col_ind, &row_values );

               // Check row size
               if (row_size != A_diag_rowPtr[i+1] - A_diag_rowPtr[i] + A_offd_rowPtr[i+1] - A_offd_rowPtr[i])
               {
                  hypre_printf("Error: proc %d has incorrect A row size at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                  test_failed = 1;
               }
               HYPRE_Int num_local_found = 0;

               // Check diag entries
               for (j = A_diag_rowPtr[i]; j < A_diag_rowPtr[i+1]; j++)
               {
                  // Get the global column index
                  HYPRE_Int col_index = A_diag_colInd[j];
                  HYPRE_Int global_col_index;
                  if (col_index < 0) global_col_index = -(col_index+1);
                  else global_col_index = global_indices[ col_index ];

                  // Check for that index in the local row
                  HYPRE_Int found = 0;
                  for (k = 0; k < row_size; k++)
                  {
                     if (global_col_index == row_col_ind[k])
                     {
                        found = 1;
                        num_local_found++;
                        if (A_diag_data[j] != row_values[k])
                        {
                           hypre_printf("Error: proc %d has incorrect A diag data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                           test_failed = 1;
                        }
                        break;
                     }
                  }
                  if (!found)
                  {
                     hypre_printf("Error: proc %d has incorrect A diag col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
               }
               
               // Check offd entries
               for (j = A_offd_rowPtr[i]; j < A_offd_rowPtr[i+1]; j++)
               {
                  // Get the global column index
                  HYPRE_Int col_index = A_offd_colInd[j];
                  HYPRE_Int global_col_index = col_index + proc_first_index;

                  // Check for that index in the local row
                  HYPRE_Int found = 0;
                  for (k = 0; k < row_size; k++)
                  {
                     if (global_col_index == row_col_ind[k])
                     {
                        found = 1;
                        num_local_found++;
                        if (A_offd_data[j] != row_values[k])
                        {
                           hypre_printf("Error: proc %d has incorrect A offd data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                           test_failed = 1;
                        }
                        break;
                     }
                  }
                  if (!found)
                  {
                     hypre_printf("Error: proc %d has incorrect A offd col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
               }

               // Make sure all local entries accounted for
               if (num_local_found != row_size)
               {
                  hypre_printf("Error: proc %d does not have all A row entries at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                  test_failed = 1;
               }
               
               hypre_ParCSRMatrixRestoreRow( A[level], global_indices[i], &row_size, &row_col_ind, &row_values );

               /////////////////////////////////
               // Check P info
               /////////////////////////////////

               if (level != num_levels-1)
               {
                  hypre_ParCSRMatrixGetRow( P[level], global_indices[i], &row_size, &row_col_ind, &row_values );

                  // Check row size
                  if (row_size != P_diag_rowPtr[i+1] - P_diag_rowPtr[i] + P_offd_rowPtr[i+1] - P_offd_rowPtr[i])
                  {
                     hypre_printf("Error: proc %d has incorrect P row size at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
                  HYPRE_Int num_local_found = 0;

                  // Check diag entries
                  for (j = P_diag_rowPtr[i]; j < P_diag_rowPtr[i+1]; j++)
                  {
                     // Get the global column index
                     HYPRE_Int global_col_index = P_diag_colInd[j];

                     // Check for that index in the local row
                     HYPRE_Int found = 0;
                     for (k = 0; k < row_size; k++)
                     {
                        if (global_col_index == row_col_ind[k])
                        {
                           found = 1;
                           num_local_found++;
                           if (P_diag_data[j] != row_values[k])
                           {
                              hypre_printf("Error: proc %d has incorrect P diag data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                              test_failed = 1;
                           }
                           break;
                        }
                     }
                     if (!found)
                     {
                        hypre_printf("Error: proc %d has incorrect P diag col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                        test_failed = 1;
                     }
                  }
                  
                  // Check offd entries
                  for (j = P_offd_rowPtr[i]; j < P_offd_rowPtr[i+1]; j++)
                  {
                     // Get the global column index
                     HYPRE_Int col_index = P_offd_colInd[j];
                     HYPRE_Int global_col_index = col_index + proc_coarse_first_index;

                     // Check for that index in the local row
                     HYPRE_Int found = 0;
                     for (k = 0; k < row_size; k++)
                     {
                        if (global_col_index == row_col_ind[k])
                        {
                           found = 1;
                           num_local_found++;
                           if (P_offd_data[j] != row_values[k])
                           {
                              hypre_printf("Error: proc %d has incorrect P offd data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                              test_failed = 1;
                           }
                           break;
                        }
                     }
                     if (!found)
                     {
                        hypre_printf("Error: proc %d has incorrect P offd col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                        test_failed = 1;
                     }
                  }

                  // Make sure all local entries accounted for
                  if (num_local_found != row_size)
                  {
                     hypre_printf("Error: proc %d does not have all P row entries at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
                  
                  hypre_ParCSRMatrixRestoreRow( P[level], global_indices[i], &row_size, &row_col_ind, &row_values );
               }
               
               /////////////////////////////////
               // Check R info
               /////////////////////////////////
               
               if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
               {
                  hypre_ParCSRMatrixGetRow( R[level-1], global_indices[i], &row_size, &row_col_ind, &row_values );

                  // Check row size
                  if (row_size != R_diag_rowPtr[i+1] - R_diag_rowPtr[i] + R_offd_rowPtr[i+1] - R_offd_rowPtr[i])
                  {
                     hypre_printf("Error: proc %d has incorrect R row size at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
                  HYPRE_Int num_local_found = 0;

                  // Check diag entries
                  for (j = R_diag_rowPtr[i]; j < R_diag_rowPtr[i+1]; j++)
                  {
                     // Get the global column index
                     HYPRE_Int global_col_index = R_diag_colInd[j];

                     // Check for that index in the local row
                     HYPRE_Int found = 0;
                     for (k = 0; k < row_size; k++)
                     {
                        if (global_col_index == row_col_ind[k])
                        {
                           found = 1;
                           num_local_found++;
                           if (R_diag_data[j] != row_values[k])
                           {
                              hypre_printf("Error: proc %d has incorrect R diag data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                              test_failed = 1;
                           }
                           break;
                        }
                     }
                     if (!found)
                     {
                        hypre_printf("Error: proc %d has incorrect R diag col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                        test_failed = 1;
                     }
                  }
                  
                  // Check offd entries
                  for (j = R_offd_rowPtr[i]; j < R_offd_rowPtr[i+1]; j++)
                  {
                     // Get the global column index
                     HYPRE_Int col_index = R_offd_colInd[j];
                     HYPRE_Int global_col_index = col_index + proc_fine_first_index;

                     // Check for that index in the local row
                     HYPRE_Int found = 0;
                     for (k = 0; k < row_size; k++)
                     {
                        if (global_col_index == row_col_ind[k])
                        {
                           found = 1;
                           num_local_found++;
                           if (R_offd_data[j] != row_values[k])
                           {
                              hypre_printf("Error: proc %d has incorrect R offd data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                              test_failed = 1;
                           }
                           break;
                        }
                     }
                     if (!found)
                     {
                        hypre_printf("Error: proc %d has incorrect R offd col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                        test_failed = 1;
                     }
                  }

                  // Make sure all local entries accounted for
                  if (num_local_found != row_size)
                  {
                     hypre_printf("Error: proc %d does not have all R row entries at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
                  
                  hypre_ParCSRMatrixRestoreRow( R[level-1], global_indices[i], &row_size, &row_col_ind, &row_values );
               }
            }
         }

         // Clean up memory
         if (myid != proc) 
         {
            hypre_TFree(global_indices, HYPRE_MEMORY_HOST);
            hypre_TFree(A_diag_rowPtr, HYPRE_MEMORY_HOST);
            hypre_TFree(A_diag_colInd, HYPRE_MEMORY_HOST);
            hypre_TFree(A_diag_data, HYPRE_MEMORY_HOST);
            hypre_TFree(A_offd_rowPtr, HYPRE_MEMORY_HOST);
            hypre_TFree(A_offd_colInd, HYPRE_MEMORY_HOST);
            hypre_TFree(A_offd_data, HYPRE_MEMORY_HOST);
            if (level != num_levels-1)
            {
               hypre_TFree(coarse_indices, HYPRE_MEMORY_HOST);
               hypre_TFree(P_diag_rowPtr, HYPRE_MEMORY_HOST);
               hypre_TFree(P_diag_colInd, HYPRE_MEMORY_HOST);
               hypre_TFree(P_diag_data, HYPRE_MEMORY_HOST);
               hypre_TFree(P_offd_rowPtr, HYPRE_MEMORY_HOST);
               hypre_TFree(P_offd_colInd, HYPRE_MEMORY_HOST);
               hypre_TFree(P_offd_data, HYPRE_MEMORY_HOST);
            }
            if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
            {
               hypre_TFree(R_diag_rowPtr, HYPRE_MEMORY_HOST);
               hypre_TFree(R_diag_colInd, HYPRE_MEMORY_HOST);
               hypre_TFree(R_diag_data, HYPRE_MEMORY_HOST);
               hypre_TFree(R_offd_rowPtr, HYPRE_MEMORY_HOST);
               hypre_TFree(R_offd_colInd, HYPRE_MEMORY_HOST);
               hypre_TFree(R_offd_data, HYPRE_MEMORY_HOST);
            }
         }
      }
   }

   return test_failed;
}

HYPRE_Int 
CheckCompGridCommPkg(hypre_ParCompGridCommPkg *compGridCommPkg)
{
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   HYPRE_Int i, j, level;
   HYPRE_Int num_levels = hypre_ParCompGridCommPkgNumLevels(compGridCommPkg);

   for (level = 0; level < num_levels; level++)
   {
      HYPRE_Int num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level];
      HYPRE_Int num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level];

      HYPRE_Int **send_buffers = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
      HYPRE_Int **recv_buffers = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);

      hypre_MPI_Request *requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
      hypre_MPI_Status *status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
      HYPRE_Int request_cnt = 0;

      // Pack send buffers   
      for (i = 0; i < num_send_procs; i++)
      {
         send_buffers[i] = hypre_CTAlloc(HYPRE_Int, num_levels - level, HYPRE_MEMORY_HOST);
         for (j = level; j < num_levels; j++) send_buffers[i][j - level] = hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[level][i][j];
      }

      // Allocate and recv buffers
      for (i = 0; i < num_recv_procs; i++)
      {
         recv_buffers[i] = hypre_CTAlloc(HYPRE_Int, num_levels - level, HYPRE_MEMORY_HOST);
         hypre_MPI_Irecv(recv_buffers[i], num_levels - level, HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 0, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      }

      // Send buffers
      for (i = 0; i < num_send_procs; i++)
      {
         hypre_MPI_Isend(send_buffers[i], num_levels - level, HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 0, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      }

      // Wait
      hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests, status );

      // Check correctness
      for (i = 0; i < num_recv_procs; i++)
      {
         for (j = level; j < num_levels; j++)
         {
            if (hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[level][i][j] != recv_buffers[i][j - level]) 
               hypre_printf("Error: level %d, inner level %d, rank %d sending to rank %d, nodes sent = %d, nodes recieved = %d\n",
                  level, j, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], myid, recv_buffers[i][j - level], hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[level][i][j]);
         }
      }

      // Clean up memory
      for (i = 0; i < num_send_procs; i++)
         hypre_TFree(send_buffers[i], HYPRE_MEMORY_HOST);
      for (i = 0; i < num_recv_procs; i++)
         hypre_TFree(recv_buffers[i], HYPRE_MEMORY_HOST);
      hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
      hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
      hypre_TFree(status, HYPRE_MEMORY_HOST);
   }
   return 0;
}

HYPRE_Int
FixUpRecvMaps(hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int ****recv_redundant_marker, HYPRE_Int start_level, HYPRE_Int num_levels)
{

   HYPRE_Int level, i;

   // Initial fix up of recv map: 
   // Get rid of redundant recvs and index from beginning of nonowned (instead of owned)
   if (compGridCommPkg)
   {
      for (level = start_level; level < num_levels; level++)
      {
         HYPRE_Int proc;
         for (proc = 0; proc < hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level]; proc++)
         {
            HYPRE_Int inner_level;
            for (inner_level = level; inner_level < num_levels; inner_level++)
            {
               // if there were nodes in psiComposite on this level
               if (hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[level][proc][inner_level])
               {
                  // store the number of nodes on this level
                  HYPRE_Int num_nodes = hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[level][proc][inner_level];
                  hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[level][proc][inner_level] = 0;

                  for (i = 0; i < num_nodes; i++)
                  {
                     HYPRE_Int redundant;
                     if (inner_level == level) redundant = 0;
                     else redundant = recv_redundant_marker[level][proc][inner_level][i];
                     if (!redundant)
                     {
                        HYPRE_Int map_val = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[level][proc][inner_level][i];
                        if (map_val < 0)
                           map_val += hypre_ParCompGridNumOwnedNodes(compGrid[inner_level]);
                        else
                           map_val -= hypre_ParCompGridNumOwnedNodes(compGrid[inner_level]);
                        hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[level][proc][inner_level][ hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[level][proc][inner_level]++ ] = map_val;
                     }
                  }
                  hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[level][proc][inner_level] = hypre_TReAlloc(hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[level][proc][inner_level], HYPRE_Int, hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[level][proc][inner_level], HYPRE_MEMORY_SHARED);
               }
            }
         }
      }
   }

   return 0;
}



