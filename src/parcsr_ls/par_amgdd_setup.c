/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#define HYPRE_TIMING

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"
#include "par_amgdd_helpers.cxx"

#define DEBUG_COMP_GRID 1 // if true, runs some tests, prints out what is stored in the comp grids for each processor to a file
#define DEBUG_PROC_NEIGHBORS 0 // if true, dumps info on the add flag structures that determine nearest processor neighbors 
#define DEBUGGING_MESSAGES 0 // if true, prints a bunch of messages to the screen to let you know where in the algorithm you are

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

   // !!! Timings
   HYPRE_Real total_timings[10];

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
   hypre_AMGDDCompGrid **compGrid = hypre_CTAlloc(hypre_AMGDDCompGrid*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParAMGDataAMGDDCompGrid(amg_data) = compGrid;

   // In the 1 processor case, just need to initialize the comp grids
   if (num_procs == 1)
   {
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         compGrid[level] = hypre_AMGDDCompGridCreate();
         hypre_AMGDDCompGridInitialize( amg_data, 0, level, symmetric );
      }
      hypre_AMGDDCompGridFinalize(amg_data, compGrid, NULL, amgdd_start_level, num_levels, hypre_ParAMGDataAMGDDUseRD(amg_data), verify_amgdd);
      hypre_AMGDDCompGridSetupRelax(amg_data);
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
      compGrid[level] = hypre_AMGDDCompGridCreate();
      hypre_AMGDDCompGridInitialize( amg_data, padding[level], level, symmetric );
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
      hypre_AMGDDCompGridDebugPrint( compGrid[level], filename );
   }
   #endif

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("  Done with comp grid initialization\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // Create the compGridCommPkg and grab a few frequently used variables
   hypre_AMGDDCommPkg *compGridCommPkg = hypre_AMGDDCommPkgCreate(num_levels);
   hypre_ParAMGDataAMGDDCommPkg(amg_data) = compGridCommPkg;

   send_buffer_size = hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg);
   recv_buffer_size = hypre_AMGDDCommPkgRecvBufferSize(compGridCommPkg);
   send_flag = hypre_AMGDDCommPkgSendFlag(compGridCommPkg);
   num_send_nodes = hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg);
   recv_map = hypre_AMGDDCommPkgRecvMap(compGridCommPkg);
   num_recv_nodes = hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg);
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
      HYPRE_Int num_send_procs = hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level];
      HYPRE_Int num_recv_procs = hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level];

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
            send_buffer[i] = PackSendBuffer(amg_data, compGrid, compGridCommPkg, &(send_buffer_size[level][i]), 
                                             &(send_flag_buffer_size[i]), send_flag, num_send_nodes, i, level, num_levels, padding, 
                                             num_ghost_layers, symmetric, total_timings );
         }
         if (timers) hypre_EndTiming(timers[2]);

         //////////// Communicate buffer sizes ////////////

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (timers) hypre_BeginTiming(timers[3]);
  
         // post the receives for the buffer size
         for (i = 0; i < num_recv_procs; i++)
         {
            // !!! Check recv_buffer_size (make sure not overallocated)
            hypre_MPI_Irecv( &(recv_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]) );
         }

         // send the buffer sizes
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_MPI_Isend(&(send_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]));
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
            hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_INT, hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 1, comm, &(requests[request_counter++]));
         }

         // send the buffers
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_MPI_Isend(send_buffer[i], send_buffer_size[level][i], HYPRE_MPI_INT, hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 1, comm, &(requests[request_counter++]));
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
         total_bin_search_count += hypre_AMGDDCompGridSetupLocalIndices(compGrid, nodes_added_on_level, recv_map, num_recv_procs, A_tmp_info, level, num_levels, symmetric);
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
            hypre_MPI_Irecv( send_flag_buffer[i], send_flag_buffer_size[i], HYPRE_MPI_INT, hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 2, comm, &(requests[request_counter++]));
         }

         // send the recv_map_send_buffer's
         for (i = 0; i < num_recv_procs; i++)
         {
            // !!! Check recv_map_send_buffer_size (make sure not overallocated)
            hypre_MPI_Isend( recv_map_send_buffer[i], recv_map_send_buffer_size[i], HYPRE_MPI_INT, hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 2, comm, &(requests[request_counter++]));
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
            hypre_TFree(send_buffer[i], HYPRE_MEMORY_HOST);
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
         hypre_AMGDDCompGridDebugPrint( compGrid[l], filename );
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

      // !!! Debug
      /* if (level == 6) */
      /* { */
      /*    MPI_Finalize(); */
      /*    exit(0); */
      /* } */
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
   hypre_AMGDDCompGridSetupLocalIndicesP(amg_data, compGrid, amgdd_start_level, num_levels);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with hypre_AMGDDCompGridSetupLocalIndicesP()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif 

   if (timers) hypre_EndTiming(timers[5]);

   #if DEBUG_COMP_GRID == 2
   for (level = 0; level < num_levels; level++)
   {
      hypre_sprintf(filename, "outputs/CompGrids/preFinalizeCompGridRank%dLevel%d.txt", myid, level);
      hypre_AMGDDCompGridDebugPrint( compGrid[level], filename );
   }
   #endif

   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[8]);


   // Finalize the comp grid structures
   hypre_AMGDDCompGridFinalize(amg_data, compGrid, compGridCommPkg, amgdd_start_level, num_levels, hypre_ParAMGDataAMGDDUseRD(amg_data), verify_amgdd);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with hypre_AMGDDCompGridFinalize()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // Setup extra info for specific relaxation methods
   hypre_AMGDDCompGridSetupRelax(amg_data);

   if (timers) hypre_EndTiming(timers[8]);

   // Count up the cost for subsequent residual communications
   if (communication_cost)
   {
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         communication_cost[level*10 + 4] += hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level];
         for (i = 0; i < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; i++)
         {
            HYPRE_Int inner_level;
            for (inner_level = level; inner_level < num_levels; inner_level++)
            {
               communication_cost[level*10 + 5] += hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level][i][inner_level]*sizeof(HYPRE_Complex);
            }
         }
      }
   }

   // Cleanup memory
   hypre_TFree(num_resizes, HYPRE_MEMORY_HOST);
   hypre_TFree(nodes_added_on_level, HYPRE_MEMORY_HOST);
   for (i = 0; i < hypre_AMGDDCommPkgNumLevels(compGridCommPkg); i++)
   {
      for (j = 0; j < hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[i]; j++)
      {
         HYPRE_Int k;
         for (k = 0; k < hypre_AMGDDCommPkgNumLevels(compGridCommPkg); k++)
         {
            if ( recv_redundant_marker[i][j][k] ) hypre_TFree( recv_redundant_marker[i][j][k], HYPRE_MEMORY_HOST );
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
      hypre_AMGDDCompGridDebugPrint( compGrid[level], filename );
   }
   #endif

   // !!! Debug
   /* if (timers) */
   /* { */
   /*    HYPRE_Int total_nonowned = 0; */
   /*    for (level = 0; level < num_levels; level++) */
   /*    { */
   /*       total_nonowned += hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]); */
   /*    } */
   /*    HYPRE_Int local_size_info[2]; */
   /*    local_size_info[0] = total_nonowned; */
   /*    local_size_info[1] = total_redundant_sends; */
   /*    HYPRE_Int global_size_info[2]; */
   /*    MPI_Reduce(local_size_info, global_size_info, 2, HYPRE_MPI_INT, MPI_SUM, 0, hypre_MPI_COMM_WORLD); */
   /*    if (myid == 0) printf("Total Setup Redundancy = %f\n", ((double) global_size_info[1] + global_size_info[0])/((double) global_size_info[0])); */
   /* } */

   // !!! Timings
   /* HYPRE_Real global_total_timings[10]; */
   /* MPI_Reduce(total_timings, global_total_timings, 6, HYPRE_MPI_REAL, MPI_SUM, 0, hypre_MPI_COMM_WORLD); */
   /* if (myid == 0) */
   /* { */
   /*    printf("Pack Send Buffer Times: \n"); */
   /*    printf("   Total: %e\n", global_total_timings[0]/num_procs); */
   /*    printf("   Expand: %e\n", global_total_timings[1]/num_procs); */
   /*    printf("   Add to send flag: %e\n", global_total_timings[2]/num_procs); */
   /*    printf("   Remove Redundancy: %e\n", global_total_timings[3]/num_procs); */
   /*    printf("   Mark Coarse: %e\n", global_total_timings[4]/num_procs); */
   /*    printf("   Adjust add flag: %e\n", global_total_timings[5]/num_procs); */
   /*    printf("\n"); */
   /* } */


   #if DEBUG_COMP_GRID
   return test_failed;
   #else
   return 0;
   #endif
}


