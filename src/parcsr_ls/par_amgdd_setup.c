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

HYPRE_Int*
PackSendBufferNew( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, HYPRE_Int *send_flag_buffer_size, 
   HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes, HYPRE_Int proc, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int *padding, 
   HYPRE_Int num_ghost_layers, HYPRE_Int symmetric );

HYPRE_Int
RecursivelyBuildPsiCompositeNew(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid **compGrids, HYPRE_Int **add_flags,
                           HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding, HYPRE_Int level, HYPRE_Int use_sort);

HYPRE_Int
PackRecvMapSendBufferNew(HYPRE_Int *recv_map_send_buffer, HYPRE_Int **recv_map, HYPRE_Int *num_recv_nodes, HYPRE_Int *recv_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, hypre_ParCompGrid **compGrid);

HYPRE_Int
UnpackSendFlagBufferNew(HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, HYPRE_Int *send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
CommunicateRemainingMatrixInfoNew(hypre_ParAMGData* amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *communication_cost, HYPRE_Int symmetric);

HYPRE_Int
TestCompGrids1New(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int current_level, HYPRE_Int check_ghost_info);

HYPRE_Int
TestCompGrids2New(hypre_ParAMGData *amg_data);

HYPRE_Int 
CheckCompGridCommPkg(hypre_ParCompGridCommPkg *compGridCommPkg);

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
   HYPRE_Int      **send_buffer_size_new;
   HYPRE_Int      **recv_buffer_size_new;
   HYPRE_Int      ***num_send_nodes_new;
   HYPRE_Int      ***num_recv_nodes_new;
   HYPRE_Int      ****send_flag_new;
   HYPRE_Int      ****recv_map_new;
   HYPRE_Int      **send_buffer_new;
   HYPRE_Int      **recv_buffer_new;
   HYPRE_Int      **send_flag_buffer_new;
   HYPRE_Int      **recv_map_send_buffer_new;
   HYPRE_Int      *send_flag_buffer_size_new;
   HYPRE_Int      *recv_map_send_buffer_size_new;
   hypre_MPI_Request       *requests_new;
   hypre_MPI_Status        *status_new;
   HYPRE_Int               request_counter_new = 0;

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

   // Allocate pointer for the composite grids
   hypre_ParCompGrid **compGrid = hypre_CTAlloc(hypre_ParCompGrid*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParAMGDataCompGrid(amg_data) = compGrid;

   // In the 1 processor case, just need to initialize the comp grids
   if (num_procs == 1)
   {
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         compGrid[level] = hypre_ParCompGridCreate();
         hypre_ParCompGridInitializeNew( amg_data, 0, level, symmetric );
      }
      hypre_ParCompGridFinalizeNew(amg_data, compGrid, NULL, amgdd_start_level, num_levels, hypre_ParAMGDataAMGDDUseRD(amg_data), verify_amgdd);
      hypre_ParCompGridSetupRelax(amg_data);
      return 0;
   }

   // Get the padding on each level
   HYPRE_Int *padding = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   if (variable_padding > num_levels - amgdd_start_level) variable_padding = num_levels - amgdd_start_level;
   if (variable_padding)
   {
      // padding[0] = 1;
      for (level = amgdd_start_level; level < amgdd_start_level + variable_padding; level++) padding[level] = pad;
      for (level = amgdd_start_level + variable_padding; level < num_levels; level++) padding[level] = 1;
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
      hypre_ParCompGridInitializeNew( amg_data, padding[level], level, symmetric );
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
      hypre_ParCompGridDebugPrintNew( compGrid[level], filename, 0 );
   }
   #endif

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("  Done with comp grid initialization\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif



   // Create the compGridCommPkg and grab a few frequently used variables
   hypre_ParCompGridCommPkg *compGridCommPkgNew = hypre_ParCompGridCommPkgCreate(num_levels);
   hypre_ParAMGDataCompGridCommPkg(amg_data) = compGridCommPkgNew;

   send_buffer_size_new = hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkgNew);
   recv_buffer_size_new = hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkgNew);
   send_flag_new = hypre_ParCompGridCommPkgSendFlag(compGridCommPkgNew);
   num_send_nodes_new = hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkgNew);
   recv_map_new = hypre_ParCompGridCommPkgRecvMap(compGridCommPkgNew);
   num_recv_nodes_new = hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkgNew);
   HYPRE_Int *nodes_added_on_level_new = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);




   // On each level, setup the compGridCommPkg so that it has communication info for distance (eta + numGhostLayers)
   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[1]);
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      SetupNearestProcessorNeighborsNew(hypre_ParAMGDataAArray(amg_data)[level], compGridCommPkgNew, level, padding, num_ghost_layers, communication_cost);
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
      HYPRE_Int num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkgNew)[level];
      HYPRE_Int num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkgNew)[level];

      if ( num_send_procs || num_recv_procs ) // If there are any owned nodes on this level
      {
         // Do some allocations
         requests_new = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status_new = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter_new = 0;

         send_buffer_new = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
         send_buffer_size_new[level] = hypre_CTAlloc(HYPRE_Int, num_send_procs, HYPRE_MEMORY_HOST);
         recv_buffer_new = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         recv_buffer_size_new[level] = hypre_CTAlloc(HYPRE_Int, num_recv_procs, HYPRE_MEMORY_HOST);

         send_flag_new[level] = hypre_CTAlloc(HYPRE_Int**, num_send_procs, HYPRE_MEMORY_HOST);
         num_send_nodes_new[level] = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
         recv_map_new[level] = hypre_CTAlloc(HYPRE_Int**, num_recv_procs, HYPRE_MEMORY_HOST);
         num_recv_nodes_new[level] = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);

         send_flag_buffer_new = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
         send_flag_buffer_size_new = hypre_CTAlloc(HYPRE_Int, num_send_procs, HYPRE_MEMORY_HOST);
         recv_map_send_buffer_new = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         recv_map_send_buffer_size_new = hypre_CTAlloc(HYPRE_Int, num_recv_procs, HYPRE_MEMORY_HOST);

         //////////// Pack send buffers ////////////

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (timers) hypre_BeginTiming(timers[2]);
         for (i = 0; i < num_send_procs; i++)
         {
            send_flag_new[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_send_nodes_new[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
            send_buffer_new[i] = PackSendBufferNew( compGrid, compGridCommPkgNew, &(send_buffer_size_new[level][i]), 
                                             &(send_flag_buffer_size_new[i]), send_flag_new, num_send_nodes_new, i, level, num_levels, padding, 
                                             num_ghost_layers, symmetric );
         }
         if (timers) hypre_EndTiming(timers[2]);

         //////////// Communicate buffer sizes ////////////

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (timers) hypre_BeginTiming(timers[3]);

         // post the receives for the buffer size
         for (i = 0; i < num_recv_procs; i++)
         {
            // !!! Check recv_buffer_size (make sure not overallocated)
            hypre_MPI_Irecv( &(recv_buffer_size_new[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkgNew)[level][i], 0, comm, &(requests_new[request_counter_new++]) );
         }

         // send the buffer sizes
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_MPI_Isend(&(send_buffer_size_new[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkgNew)[level][i], 0, comm, &(requests_new[request_counter_new++]));
            if (communication_cost)
            {
               communication_cost[level*10 + 2]++;
               communication_cost[level*10 + 3] += sizeof(HYPRE_Int);
            }
         }

         // wait for all buffer sizes to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests_new, status_new );
         hypre_TFree(requests_new, HYPRE_MEMORY_HOST);
         hypre_TFree(status_new, HYPRE_MEMORY_HOST);
         requests_new = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status_new = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter_new = 0;

         //////////// Communicate buffers ////////////

         // allocate space for the receive buffers and post the receives
         for (i = 0; i < num_recv_procs; i++)
         {
            recv_buffer_new[i] = hypre_CTAlloc(HYPRE_Int, recv_buffer_size_new[level][i], HYPRE_MEMORY_HOST );
            hypre_MPI_Irecv( recv_buffer_new[i], recv_buffer_size_new[level][i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkgNew)[level][i], 1, comm, &(requests_new[request_counter_new++]));
         }

         // send the buffers
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_MPI_Isend(send_buffer_new[i], send_buffer_size_new[level][i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkgNew)[level][i], 1, comm, &(requests_new[request_counter_new++]));
            if (communication_cost)
            {
               communication_cost[level*10 + 2]++;
               communication_cost[level*10 + 3] += send_buffer_size_new[level][i]*sizeof(HYPRE_Int);
            }
         }

         // wait for buffers to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests_new, status_new );
         hypre_TFree(requests_new, HYPRE_MEMORY_HOST);
         hypre_TFree(status_new, HYPRE_MEMORY_HOST);
         requests_new = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status_new = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter_new = 0;

         if (timers) hypre_EndTiming(timers[3]);

         //////////// Unpack the received buffers ////////////

         if (timers) hypre_BeginTiming(timers[4]);
         HYPRE_Int **A_tmp_info = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_recv_procs; i++)
         {
            recv_map_new[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_recv_nodes_new[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);

            UnpackRecvBufferNew(recv_buffer_new[i], compGrid, hypre_ParCSRMatrixCommPkg( hypre_ParAMGDataAArray(amg_data)[level] ),
               A_tmp_info,
               compGridCommPkgNew,
               send_flag_new, num_send_nodes_new, 
               recv_map_new, num_recv_nodes_new, 
               &(recv_map_send_buffer_size_new[i]), level, num_levels, nodes_added_on_level_new, i, num_resizes, symmetric);
            
            recv_map_send_buffer_new[i] = hypre_CTAlloc(HYPRE_Int, recv_map_send_buffer_size_new[i], HYPRE_MEMORY_HOST);
            PackRecvMapSendBufferNew(recv_map_send_buffer_new[i], recv_map_new[level][i], num_recv_nodes_new[level][i], &(recv_buffer_size_new[level][i]), level, num_levels, compGrid);
         }
         if (timers) hypre_EndTiming(timers[4]);

         //////////// Setup local indices for the composite grid ////////////

         if (timers) hypre_BeginTiming(timers[5]);
         total_bin_search_count += hypre_ParCompGridSetupLocalIndicesNew(compGrid, nodes_added_on_level_new, recv_map_new, num_recv_procs, A_tmp_info, level, num_levels, symmetric);
         for (j = level; j < num_levels; j++) nodes_added_on_level_new[j] = 0;

         if (timers) hypre_EndTiming(timers[5]);

         //////////// Communicate redundancy info ////////////

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (timers) hypre_BeginTiming(timers[6]);  

         // post receives for send maps
         for (i = 0; i < num_send_procs; i++)
         {
            // !!! Check send_flag_buffer_size (make sure not overallocated)
            send_flag_buffer_new[i] = hypre_CTAlloc(HYPRE_Int, send_flag_buffer_size_new[i], HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv( send_flag_buffer_new[i], send_flag_buffer_size_new[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkgNew)[level][i], 2, comm, &(requests_new[request_counter_new++]));
         }

         // send the recv_map_send_buffer's
         for (i = 0; i < num_recv_procs; i++)
         {
            // !!! Check recv_map_send_buffer_size (make sure not overallocated)
            hypre_MPI_Isend( recv_map_send_buffer_new[i], recv_map_send_buffer_size_new[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkgNew)[level][i], 2, comm, &(requests_new[request_counter_new++]));
            if (communication_cost)
            {
               communication_cost[level*10 + 2]++;
               communication_cost[level*10 + 3] += recv_map_send_buffer_size_new[i]*sizeof(HYPRE_Int);
            }
         }

         // wait for maps to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests_new, status_new );
         hypre_TFree(requests_new, HYPRE_MEMORY_HOST);
         hypre_TFree(status_new, HYPRE_MEMORY_HOST);

         // unpack and setup the send flag arrays
         for (i = 0; i < num_send_procs; i++)
         {
            UnpackSendFlagBufferNew(send_flag_buffer_new[i], send_flag_new[level][i], num_send_nodes_new[level][i], &(send_buffer_size_new[level][i]), level, num_levels);
         }

         if (timers) hypre_EndTiming(timers[6]);

         // clean up memory for this level
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_TFree(send_buffer_new[i], HYPRE_MEMORY_HOST);
            hypre_TFree(send_flag_buffer_new[i], HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < num_recv_procs; i++)
         {
            hypre_TFree(recv_buffer_new[i], HYPRE_MEMORY_HOST);
            hypre_TFree(recv_map_send_buffer_new[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(send_buffer_new, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_buffer_new, HYPRE_MEMORY_HOST);
         hypre_TFree(send_flag_buffer_new, HYPRE_MEMORY_HOST);
         hypre_TFree(send_flag_buffer_size_new, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_map_send_buffer_new, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_map_send_buffer_size_new, HYPRE_MEMORY_HOST);
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

      #if DEBUG_COMP_GRID
      HYPRE_Int error_code;
      error_code = TestCompGrids1New(compGrid, num_levels, padding, num_ghost_layers, level, 1);
      if (error_code)
         hypre_printf("TestCompGrids1 failed! Rank %d, level %d\n", myid, level);
      else
         hypre_printf("TestCompGrids1 passed! Rank %d, level %d\n", myid, level);
      #endif
   }

   /////////////////////////////////////////////////////////////////

   // Done with loop over levels. Now just finalize things.

   /////////////////////////////////////////////////////////////////

   #if DEBUG_COMP_GRID
   // Test whether comp grids have correct shape
   HYPRE_Int test_failed = 0;
   HYPRE_Int error_code;
   error_code = TestCompGrids1New(compGrid, num_levels, padding, num_ghost_layers, 0, 1);
   if (error_code)
   {
      hypre_printf("TestCompGrids1 failed!\n");
      test_failed = 1;
   }
   else hypre_printf("TestCompGrids1 success\n");
   CheckCompGridCommPkg(compGridCommPkgNew);
   #endif

   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[7]);

   // Communicate data for A and all info for P
   CommunicateRemainingMatrixInfoNew(amg_data, compGrid, compGridCommPkgNew, communication_cost, symmetric);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with CommunicateRemainingMatrixInfo()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif 

   #if DEBUG_COMP_GRID
   error_code = TestCompGrids2New(amg_data); // NOTE: test should come before setting up local indices for P (uses global col ind for P)
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
   hypre_ParCompGridSetupLocalIndicesPNew(compGrid, amgdd_start_level, num_levels);

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
      hypre_ParCompGridDebugPrintNew( compGrid[level], filename, 0 );
   }
   #endif

   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[8]);

   // Finalize the comp grid structures
   hypre_ParCompGridFinalizeNew(amg_data, compGrid, compGridCommPkgNew, amgdd_start_level, num_levels, hypre_ParAMGDataAMGDDUseRD(amg_data), verify_amgdd);

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
         communication_cost[level*10 + 4] += hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkgNew)[level];
         communication_cost[level*10 + 5] += hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkgNew)[level][ hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkgNew)[level] ]*sizeof(HYPRE_Complex);
      }
   }

   // Cleanup memory
   hypre_TFree(num_resizes, HYPRE_MEMORY_HOST);
   hypre_TFree(nodes_added_on_level_new, HYPRE_MEMORY_HOST);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Finished comp grid setup on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   #if DEBUG_COMP_GRID == 2
   for (level = 0; level < num_levels; level++)
   {
      HYPRE_Int coarse_num_nodes = 0;
      if (level != num_levels-1) coarse_num_nodes = hypre_ParCompGridNumNodes(compGrid[level+1]);
      hypre_sprintf(filename, "outputs/CompGrids/setupCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrintNew( compGrid[level], filename, coarse_num_nodes );
   }
   #endif

   #if DEBUG_COMP_GRID
   return test_failed;
   #else
   return 0;
   #endif
}

HYPRE_Int*
PackSendBufferNew( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, 
   HYPRE_Int *send_flag_buffer_size, HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
   HYPRE_Int proc, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int symmetric )
{
   // send_buffer = [ num_psi_levels , [level] , [level] , ... ]
   // level = [ num send nodes, [global indices] , [coarse global indices] , [A row sizes] , [A col ind: either global indices or local col indices within buffer] ]

   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int            level,i,j,k,cnt,row_length,send_elmt,coarse_grid_index,add_flag_index;
   HYPRE_Int            nodes_to_add = 0;
   HYPRE_Int            **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            num_psi_levels = 1;
   HYPRE_Int            coarse_proc;

   // Get where to look in commPkgSendMapElmts
   HYPRE_Int            start = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][proc];
   HYPRE_Int            finish = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][proc+1];

   // Get the sort maps
   HYPRE_Int            *sort_map;
   HYPRE_Int            *sort_map_coarse;

   // initialize send map buffer size
   (*send_flag_buffer_size) = num_levels - current_level - 1;

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Mark the nodes to send (including Psi_c grid plus ghost nodes)
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Count up the buffer size for the starting nodes
   num_send_nodes[current_level][proc][current_level] = finish - start;
   send_flag[current_level][proc][current_level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level][proc][current_level], HYPRE_MEMORY_HOST );
   add_flag[current_level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid[current_level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[current_level]), HYPRE_MEMORY_HOST);

   (*buffer_size) += 2;
   if (current_level != num_levels-1) (*buffer_size) += 3*num_send_nodes[current_level][proc][current_level];
   else (*buffer_size) += 2*num_send_nodes[current_level][proc][current_level];

   for (i = start; i < finish; i++)
   {
      send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];
      if (hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[current_level][i])
         send_flag[current_level][proc][current_level][i - start] = -(send_elmt + 1);
      else
         send_flag[current_level][proc][current_level][i - start] = send_elmt;
      add_flag[current_level][send_elmt] = i - start + 1;

      hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid[current_level]));
      hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid[current_level]));
      (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
      (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
   }

   // Add the nodes listed by the coarse grid counterparts if applicable
   // Note that the compGridCommPkg is set up to list all nodes within the padding plus ghost layers
   if (current_level != num_levels-1)
   {
      add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid[current_level+1]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[current_level+1]), HYPRE_MEMORY_HOST );
      for (i = start; i < finish; i++)
      {
         // flag nodes that are repeated on the next coarse grid
         if (!hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[current_level][i])
         {
            send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];
            coarse_grid_index = hypre_ParCompGridOwnedCoarseIndices(compGrid[current_level])[send_elmt];
            if ( coarse_grid_index != -1 ) 
            {
               add_flag[current_level+1][ coarse_grid_index ] = padding[current_level+1]+1;
               nodes_to_add = 1;
            }
         }
      }
   }

   // Now build out the psi_c composite grid (along with required ghost nodes) on coarser levels
   for (level = current_level + 1; level < num_levels; level++)
   {
      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         sort_map = hypre_ParCompGridNonOwnedSort(compGrid[level]);
         if (level != num_levels-1) sort_map_coarse = hypre_ParCompGridNonOwnedSort(compGrid[level+1]);
         HYPRE_Int *inv_sort_map = hypre_ParCompGridNonOwnedInvSort(compGrid[level]);
         
         num_psi_levels++;
         (*buffer_size)++;
         nodes_to_add = 0;

         // if we need coarse info, allocate space for the add flag on the next level
         if (level != num_levels-1) add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid[level+1]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level+1]), HYPRE_MEMORY_HOST );

         // Expand by the padding on this level and add coarse grid counterparts if applicable
         HYPRE_Int total_num_nodes = hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
         for (i = 0; i < total_num_nodes; i++)
         {
            if (i < hypre_ParCompGridNumOwnedNodes(compGrid[level])) add_flag_index = i;
            else add_flag_index = sort_map[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);

            if (add_flag[level][add_flag_index] == padding[level] + 1)
            {
               // Recursively add the region of padding (flagging coarse nodes on the next level if applicable)
               if (level != num_levels-1) RecursivelyBuildPsiCompositeNew(i, padding[level], compGrid, add_flag, 1, &nodes_to_add, padding[level+1], level, 1);
               else RecursivelyBuildPsiCompositeNew(i, padding[level], compGrid, add_flag, 0, NULL, 0, level, 1);
            }
         }

         // Expand by the number of ghost layers 
         for (i = 0; i < total_num_nodes; i++)
         {
            if (i < hypre_ParCompGridNumOwnedNodes(compGrid[level])) add_flag_index = i;
            else add_flag_index = sort_map[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);

            if (add_flag[level][add_flag_index] > 1) add_flag[level][add_flag_index] = num_ghost_layers + 2;
            else if (add_flag[level][add_flag_index] == 1) add_flag[level][add_flag_index] = num_ghost_layers + 1;
         }

         for (i = 0; i < total_num_nodes; i++)
         {
            if (i < hypre_ParCompGridNumOwnedNodes(compGrid[level])) add_flag_index = i;
            else add_flag_index = sort_map[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);

            // Recursively add the region of ghost nodes (do not add any coarse nodes underneath)
            if (add_flag[level][add_flag_index] == num_ghost_layers + 1) RecursivelyBuildPsiCompositeNew(i, num_ghost_layers, compGrid, add_flag, 0, NULL, 0, level, 1);
         }

         // Count up the total number of send nodes 
         for (i = 0; i < total_num_nodes; i++)
         {
            if (add_flag[level][i] > 0)
            {
               num_send_nodes[current_level][proc][level]++;
            }
         }

         // Save the indices (in global index ordering) 
         send_flag[current_level][proc][level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_HOST );
         cnt =  0;
         i = 0;
         // First the nonowned indices coming before the owned block
         if (hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
         {
            while (hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[inv_sort_map[i]] < hypre_ParCompGridFirstGlobalIndex(compGrid[level]))
            {
               add_flag_index = i + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               if (add_flag[level][add_flag_index] > num_ghost_layers)
               {
                  send_flag[current_level][proc][level][cnt] = inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
                  cnt++;
               }
               else if (add_flag[level][add_flag_index] > 0)
               {
                  send_flag[current_level][proc][level][cnt] = -(inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]) + 1);
                  cnt++;
               }
               i++;
               if (i == hypre_ParCompGridNumNonOwnedNodes(compGrid[level])) break;
            }
         }
         // Then the owned block
         for (add_flag_index = 0; add_flag_index < hypre_ParCompGridNumOwnedNodes(compGrid[level]); add_flag_index++)
         {
            if (add_flag[level][add_flag_index] > num_ghost_layers)
            {
               send_flag[current_level][proc][level][cnt] = add_flag_index;
               cnt++;
            }
            else if (add_flag[level][add_flag_index] > 0)
            {
               send_flag[current_level][proc][level][cnt] = -(add_flag_index+1);
               cnt++;
            }
         }
         // Finally the nonowned indices coming after the owned block
         while (i < hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
         {
            add_flag_index = i + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
            if (add_flag[level][add_flag_index] > num_ghost_layers)
            {
               send_flag[current_level][proc][level][cnt] = inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               cnt++;
            }
            else if (add_flag[level][add_flag_index] > 0)
            {
               send_flag[current_level][proc][level][cnt] = -(inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]) + 1);
               cnt++;
            }
            i++;
         }

         // Eliminate redundant send info by comparing with previous send_flags
         
         // !!! Debug
         // if (myid == 1) printf("current_level %d, proc %d, level %d: num send nodes before = %d\n",
         //    current_level, proc, level, num_send_nodes[current_level][proc][level]);


         HYPRE_Int current_send_proc = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[current_level][proc];
         HYPRE_Int prev_proc, prev_level;
         for (prev_level = current_level+1; prev_level <= level; prev_level++)
         {
            for (prev_proc = 0; prev_proc < hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[prev_level]; prev_proc++)
            {
               if (hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[prev_level][prev_proc] == current_send_proc)
               {
                  // send_flag's are in global index ordering on each level, so can merge 
                  HYPRE_Int prev_cnt = 0;
                  HYPRE_Int current_cnt = 0;
                  HYPRE_Int new_cnt = 0;
                  while (current_cnt < num_send_nodes[current_level][proc][level] && prev_cnt < num_send_nodes[prev_level][prev_proc][level])
                  {
                     if (send_flag[current_level][proc][level][current_cnt] > send_flag[prev_level][prev_proc][level][prev_cnt])
                     {
                        prev_cnt++;
                     }
                     else if (send_flag[current_level][proc][level][current_cnt] < send_flag[prev_level][prev_proc][level][prev_cnt])
                     {
                        send_flag[current_level][proc][level][new_cnt] = send_flag[current_level][proc][level][current_cnt];
                        new_cnt++;
                        current_cnt++;
                     }
                     else
                     {
                        current_cnt++;
                     }
                  }
                  while (current_cnt < num_send_nodes[current_level][proc][level])
                  {
                     send_flag[current_level][proc][level][new_cnt] = send_flag[current_level][proc][level][current_cnt];
                     new_cnt++;
                     current_cnt++;
                  }
                  num_send_nodes[current_level][proc][level] = new_cnt;
               }
            }
         }

         // !!! Debug
         // if (myid == 1) printf("current_level %d, proc %d, level %d: num send nodes after = %d\n",
         //    current_level, proc, level, num_send_nodes[current_level][proc][level]);

         // Count up the buffer sizes and adjust the add_flag
         memset(add_flag[level], 0, sizeof(HYPRE_Int)*(hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level])) );
         (*send_flag_buffer_size) += num_send_nodes[current_level][proc][level];
         if (level != num_levels-1) (*buffer_size) += 3*num_send_nodes[current_level][proc][level];
         else (*buffer_size) += 2*num_send_nodes[current_level][proc][level];
         
         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            add_flag[level][send_elmt] = i + 1;
            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
               (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
               (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
            }
            else
            {
               send_elmt -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
               (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
               (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
            }
         }
      }
      else break;
   }

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Pack the buffer
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   HYPRE_Int *send_buffer = hypre_CTAlloc(HYPRE_Int, (*buffer_size), HYPRE_MEMORY_HOST);
   cnt = 0;
   send_buffer[cnt++] = num_psi_levels;
   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      // store the number of nodes on this level
      send_buffer[cnt++] = num_send_nodes[current_level][proc][level];

      // copy all global indices
      for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
      {
         send_elmt = send_flag[current_level][proc][level][i];
         if (send_elmt < 0)
         {
            send_elmt = -(send_elmt + 1);

            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               send_buffer[cnt++] = -(send_elmt + hypre_ParCompGridFirstGlobalIndex(compGrid[level]) + 1);
            }
            else
            {
               send_buffer[cnt++] = -(hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid[level]) ] + 1);
            }
         }
         else 
         {
            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               send_buffer[cnt++] = send_elmt + hypre_ParCompGridFirstGlobalIndex(compGrid[level]);
            }
            else
            {
               send_buffer[cnt++] = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid[level]) ];
            }
         }
      }

      // if not on last level, copy coarse gobal indices
      if (level != num_levels-1)
      {
         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);

            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               if (hypre_ParCompGridOwnedCoarseIndices(compGrid[level])[ send_elmt ] >= 0)
                  send_buffer[cnt++] = hypre_ParCompGridOwnedCoarseIndices(compGrid[level])[ send_elmt ] + hypre_ParCompGridFirstGlobalIndex(compGrid[level+1]);
               else
                  send_buffer[cnt++] = hypre_ParCompGridOwnedCoarseIndices(compGrid[level])[ send_elmt ];
            }
            else 
            {
               HYPRE_Int nonowned_index = send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               HYPRE_Int nonowned_coarse_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level])[ nonowned_index ];
               
               if (nonowned_coarse_index >= 0)
                  send_buffer[cnt++] = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level+1])[ nonowned_coarse_index ];
               else if (nonowned_coarse_index == -1)
                  send_buffer[cnt++] = nonowned_coarse_index;
               else
                  send_buffer[cnt++] = -(nonowned_coarse_index+2);
            }
         }
      }

      // store the row length for matrix A
      for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
      {
         send_elmt = send_flag[current_level][proc][level][i];
         if (send_elmt < 0 && symmetric) send_buffer[cnt++] = 0;
         else
         {
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
               row_length = hypre_CSRMatrixI(diag)[ send_elmt + 1 ] - hypre_CSRMatrixI(diag)[ send_elmt ]
                          + hypre_CSRMatrixI(offd)[ send_elmt + 1 ] - hypre_CSRMatrixI(offd)[ send_elmt ];
            }
            else
            {
               HYPRE_Int nonowned_index = send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
               row_length = hypre_CSRMatrixI(diag)[ nonowned_index + 1 ] - hypre_CSRMatrixI(diag)[ nonowned_index ]
                          + hypre_CSRMatrixI(offd)[ nonowned_index + 1 ] - hypre_CSRMatrixI(offd)[ nonowned_index ];
            }
            send_buffer[cnt++] = row_length;
         }
      }

      // copy indices for matrix A (local connectivity within buffer where available, global index otherwise)
      sort_map = hypre_ParCompGridNonOwnedSort(compGrid[level]);
      for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
      {
         send_elmt = send_flag[current_level][proc][level][i];
         if (send_elmt < 0 && symmetric)
         {}
         else
         {
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);

            // Owned point
            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
               // Get diag connections
               for (j = hypre_CSRMatrixI(diag)[send_elmt]; j < hypre_CSRMatrixI(diag)[send_elmt+1]; j++)
               {
                  add_flag_index = hypre_CSRMatrixJ(diag)[j];
                  if (add_flag[level][add_flag_index] > 0)
                  {
                     send_buffer[cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
                  }
                  else
                  {
                     send_buffer[cnt++] = -(add_flag_index + hypre_ParCompGridFirstGlobalIndex(compGrid[level]) + 1); // -(GID + 1)
                  }
               }
               // Get offd connections
               for (j = hypre_CSRMatrixI(offd)[send_elmt]; j < hypre_CSRMatrixI(offd)[send_elmt+1]; j++)
               {
                  add_flag_index = hypre_CSRMatrixJ(offd)[j] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
                  if (add_flag[level][add_flag_index] > 0)
                  {
                     send_buffer[cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
                  }
                  else
                  {
                     send_buffer[cnt++] = -(hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ hypre_CSRMatrixJ(offd)[j] ] + 1); // -(GID + 1)
                  }
               }
            }
            // NonOwned point
            else
            {
               HYPRE_Int nonowned_index = send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
               // Get diag connections
               for (j = hypre_CSRMatrixI(diag)[nonowned_index]; j < hypre_CSRMatrixI(diag)[nonowned_index+1]; j++)
               {
                  if (hypre_CSRMatrixJ(diag)[j] >= 0)
                  {
                     add_flag_index = hypre_CSRMatrixJ(diag)[j] + hypre_ParCompGridNumOwnedNodes(compGrid[level]); 
                     if (add_flag[level][add_flag_index] > 0)
                     {
                        send_buffer[cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
                     }
                     else
                     {
                        send_buffer[cnt++] = -(hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ hypre_CSRMatrixJ(diag)[j] ] + 1); // -(GID + 1)
                     }
                  }
                  else
                  {
                     send_buffer[cnt++] = hypre_CSRMatrixJ(diag)[j]; // -(GID + 1)
                  }
               }
               // Get offd connections
               for (j = hypre_CSRMatrixI(offd)[nonowned_index]; j < hypre_CSRMatrixI(offd)[nonowned_index+1]; j++)
               {
                  add_flag_index = hypre_CSRMatrixJ(offd)[j];
                  if (add_flag[level][add_flag_index] > 0)
                  {
                     send_buffer[cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
                  }
                  else
                  {
                     send_buffer[cnt++] = -(add_flag_index + hypre_ParCompGridFirstGlobalIndex(compGrid[level]) + 1); // -(GID + 1)
                  }
               }
            }
         }
      }
   }

   // Clean up memory
   for (level = 0; level < num_levels; level++)
   {
      if (add_flag[level]) hypre_TFree(add_flag[level], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);

   // Return the send buffer
   return send_buffer;
}

HYPRE_Int
RecursivelyBuildPsiCompositeNew(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid **compGrids, HYPRE_Int **add_flags,
                           HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding, HYPRE_Int level, HYPRE_Int use_sort)
{
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int i,index,sort_index,coarse_grid_index;
   HYPRE_Int error_code = 0;

   hypre_ParCompGrid *compGrid = compGrids[level];
   HYPRE_Int *add_flag = add_flags[level];
   HYPRE_Int *sort_map = hypre_ParCompGridNonOwnedSort(compGrid);
   HYPRE_Int *add_flag_coarse = NULL;
   HYPRE_Int *sort_map_coarse = NULL;
   if (need_coarse_info)
   {
      add_flag_coarse = add_flags[level+1];
      sort_map_coarse = hypre_ParCompGridNonOwnedSort(compGrids[level+1]);
   }

   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;
   HYPRE_Int owned;
   if (node < hypre_ParCompGridNumOwnedNodes(compGrid))
   {
      owned = 1;
      diag = hypre_ParCompGridMatrixOwnedDiag( hypre_ParCompGridANew(compGrid) );
      offd = hypre_ParCompGridMatrixOwnedOffd( hypre_ParCompGridANew(compGrid) );
   }
   else
   {
      owned = 0;
      node = node - hypre_ParCompGridNumOwnedNodes(compGrid);
      diag = hypre_ParCompGridMatrixNonOwnedDiag( hypre_ParCompGridANew(compGrid) );
      offd = hypre_ParCompGridMatrixNonOwnedOffd( hypre_ParCompGridANew(compGrid) );      
   }

   // Look at neighbors in diag
   for (i = hypre_CSRMatrixI(diag)[node]; i < hypre_CSRMatrixI(diag)[node+1]; i++)
   {
      // Get the index of the neighbor
      index = hypre_CSRMatrixJ(diag)[i];

      if (index >= 0)
      {
         if (owned) sort_index = index;
         else
         {
            if (use_sort) sort_index = sort_map[index] + hypre_ParCompGridNumOwnedNodes(compGrid);
            else sort_index = index + hypre_ParCompGridNumOwnedNodes(compGrid);
         }

         // If we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[sort_index] < m)
         {
            add_flag[sort_index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) error_code = RecursivelyBuildPsiCompositeNew(index, m-1, compGrids, add_flags, need_coarse_info, nodes_to_add, padding, level, use_sort);
         }
         // If m = 1, we won't do another recursive call, so make sure to flag the coarse grid here if applicable
         if (need_coarse_info && m == 1)
         {
            if (owned) coarse_grid_index = hypre_ParCompGridOwnedCoarseIndices(compGrid)[index];
            else coarse_grid_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid)[index];

            if ( coarse_grid_index != -1 ) 
            {
               // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
               if (owned) sort_index = coarse_grid_index;
               else
               {
                  if (use_sort) sort_index = sort_map_coarse[coarse_grid_index] + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
                  else sort_index = coarse_grid_index + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
               }
               add_flag_coarse[ sort_index ] = padding+1;
               *nodes_to_add = 1;
            }
         }
      }
      else
      {
         error_code = 1;
         if (owned == 1) hypre_printf("Rank %d: Error! Negative col index encountered in owned matrix\n");
         else hypre_printf("Rank %d: Error! Ran into a -1 index in diag when building Psi_c\n", myid);
      }
   }

   // Look at neighbors in offd
   for (i = hypre_CSRMatrixI(offd)[node]; i < hypre_CSRMatrixI(offd)[node+1]; i++)
   {
      // Get the index of the neighbor
      index = hypre_CSRMatrixJ(offd)[i];

      if (index >= 0)
      {
         if (!owned) sort_index = index;
         else
         {
            if (use_sort) sort_index = sort_map[index] + hypre_ParCompGridNumOwnedNodes(compGrid);
            else sort_index = index + hypre_ParCompGridNumOwnedNodes(compGrid);
         }

         // If we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[sort_index] < m)
         {
            add_flag[sort_index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) error_code = RecursivelyBuildPsiCompositeNew(index, m-1, compGrids, add_flags, need_coarse_info, nodes_to_add, padding, level, use_sort);
         }
         // If m = 1, we won't do another recursive call, so make sure to flag the coarse grid here if applicable
         if (need_coarse_info && m == 1)
         {
            if (!owned) coarse_grid_index = hypre_ParCompGridOwnedCoarseIndices(compGrid)[index];
            else coarse_grid_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid)[index];

            if ( coarse_grid_index != -1 ) 
            {
               if (coarse_grid_index >= 0)
               {
                  // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
                  if (!owned) sort_index = coarse_grid_index;
                  else
                  {
                     if (use_sort) sort_index = sort_map_coarse[coarse_grid_index] + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
                     else sort_index = coarse_grid_index + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
                  }
                  add_flag_coarse[ sort_index ] = padding+1;
                  *nodes_to_add = 1;
               }
               else
               {
                  error_code = 1;
                  hypre_printf("Rank %d: Error! Ran into a coarse index that was not set up when building Psi_c\n", myid);
               }
            }
         }
      }
      else
      {
         error_code = 1; 
         if (owned == 1) hypre_printf("Rank %d: Error! Negative col index encountered in owned matrix\n");
         else hypre_printf("Rank %d: Error! Ran into a -1 index in nonowned_offd when building Psi_c\n", myid);
      }
   }

   // Flag this node on the next coarsest level if applicable
   if (need_coarse_info)
   {
      if (owned) coarse_grid_index = hypre_ParCompGridOwnedCoarseIndices(compGrid)[node];
      else coarse_grid_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid)[node];
      if ( coarse_grid_index != -1 ) 
      {
         // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
         if (owned) sort_index = coarse_grid_index;
         else
         {
            if (use_sort) sort_index = sort_map_coarse[coarse_grid_index] + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
            else sort_index = coarse_grid_index + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
         }
         add_flag_coarse[ sort_index ] = padding+1;
         *nodes_to_add = 1;
      }
   }

   return error_code;
}

HYPRE_Int
PackRecvMapSendBufferNew(HYPRE_Int *recv_map_send_buffer, 
   HYPRE_Int **recv_map, 
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
      if (recv_map[level])
      {
         // store the number of nodes on this level
         num_nodes = num_recv_nodes[level];
         recv_map_send_buffer[cnt++] = num_nodes;
         num_recv_nodes[level] = 0;

         for (i = 0; i < num_nodes; i++)
         {
            // store the map values for each node
            recv_map_send_buffer[cnt++] = recv_map[level][i];

            if (recv_map[level][i] >= 0)
            {
               if (hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[ recv_map[level][i] ] > 0)
               {
                  recv_map[level][ num_recv_nodes[level]++ ] = recv_map[level][i];
               }
               else
               {
                  recv_map[level][ num_recv_nodes[level]++ ] = -(recv_map[level][i] + 1);
               }
               (*recv_buffer_size)++;
            }
         }
         recv_map[level] = hypre_TReAlloc(recv_map[level], HYPRE_Int, num_recv_nodes[level], HYPRE_MEMORY_HOST);
      }
      // otherwise record that there were zero nodes on this level
      else recv_map_send_buffer[cnt++] = 0;
   }

   return 0;
}

HYPRE_Int
UnpackSendFlagBufferNew(HYPRE_Int *send_flag_buffer, 
   HYPRE_Int **send_flag, 
   HYPRE_Int *num_send_nodes,
   HYPRE_Int *send_buffer_size,
   HYPRE_Int current_level, 
   HYPRE_Int num_levels)
{
   HYPRE_Int      level, i, cnt, num_nodes;
   cnt = 0;
   *send_buffer_size = 0;
   for (level = current_level+1; level < num_levels; level++)
   {
      num_nodes = send_flag_buffer[cnt++];
      num_send_nodes[level] = 0;

      for (i = 0; i < num_nodes; i++)
      {
         if (send_flag_buffer[cnt++] >= 0) 
         {
            send_flag[level][ num_send_nodes[level]++ ] = send_flag[level][i];
            (*send_buffer_size)++;
         }
      }
      
      send_flag[level] = hypre_TReAlloc(send_flag[level], HYPRE_Int, num_send_nodes[level], HYPRE_MEMORY_HOST);
   }

   return 0;
}

HYPRE_Int
CommunicateRemainingMatrixInfoNew(hypre_ParAMGData* amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *communication_cost, HYPRE_Int symmetric)
{
   HYPRE_Int outer_level,proc,level,i,j;
   HYPRE_Int num_levels = hypre_ParCompGridCommPkgNumLevels(compGridCommPkg);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);

   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   HYPRE_Int myid,num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   HYPRE_Int ***temp_RColInd = NULL;
   HYPRE_Complex ***temp_RData = NULL;
   if (hypre_ParAMGDataRestriction(amg_data))
   {
      temp_RColInd = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);
      temp_RData = hypre_CTAlloc(HYPRE_Complex**, num_levels, HYPRE_MEMORY_HOST);
      for (outer_level = amgdd_start_level; outer_level < num_levels; outer_level++)
      {
         temp_RColInd[outer_level] = hypre_CTAlloc(HYPRE_Int*, hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level]), HYPRE_MEMORY_HOST);
         temp_RData[outer_level] = hypre_CTAlloc(HYPRE_Complex*, hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level]), HYPRE_MEMORY_HOST);
      }
   }

   HYPRE_Int *P_row_cnt = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
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
         hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[outer_level])) = hypre_CSRMatrixCreate(hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level]), hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level+1]), max_nonowned_diag_nnz);
         hypre_CSRMatrixInitialize(hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[outer_level])));
         hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid[outer_level])) = hypre_CSRMatrixCreate(hypre_ParCompGridNumNonOwnedNodes(compGrid[outer_level]), hypre_ParCompGridNumOwnedNodes(compGrid[outer_level+1]), max_nonowned_offd_nnz);
         hypre_CSRMatrixInitialize(hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid[outer_level])));
         if (hypre_ParAMGDataRestriction(amg_data))
         {
            // !!! TODO R
         }
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
                     diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
                     offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
                     A_row_size = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                     if (level != num_levels-1)
                     {
                        diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridPNew(compGrid[level]));
                        offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridPNew(compGrid[level]));
                        P_row_size = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                   + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                     }
                     // !!! TODO R
                  }
                  // Nonowned diag and offd
                  else
                  {
                     idx -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
                     // Count diag and offd
                     diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
                     offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
                     A_row_size = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                     if (level != num_levels-1)
                     {
                        diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[level]));
                        offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid[level]));
                        P_row_size = hypre_CSRMatrixI(diag)[idx+1] - hypre_CSRMatrixI(diag)[idx]
                                   + hypre_CSRMatrixI(offd)[idx+1] - hypre_CSRMatrixI(offd)[idx];
                     }
                     // !!! TODO R
                  }

                  send_sizes[2*proc] += A_row_size + P_row_size + R_row_size;
                  send_sizes[2*proc+1] += A_row_size + P_row_size + R_row_size;
               }
               if (level != num_levels-1) send_sizes[2*proc] += hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level];
               if (hypre_ParAMGDataRestriction(amg_data)) send_sizes[2*proc] += hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level];
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
                     diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
                     offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
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

                     diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
                     offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
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
                        diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridPNew(compGrid[level]));
                        offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridPNew(compGrid[level]));
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
                        diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[level]));
                        offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid[level]));
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
               if (level != 0 && hypre_ParAMGDataRestriction(amg_data))
               {
                  // !!! TODO R
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
            P_tmp_info_size -= hypre_CSRMatrixNumCols(hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid[outer_level])));
            P_tmp_info_int = hypre_CTAlloc(HYPRE_Int*, P_tmp_info_size, HYPRE_MEMORY_HOST);
            P_tmp_info_complex = hypre_CTAlloc(HYPRE_Complex*, P_tmp_info_size, HYPRE_MEMORY_HOST);
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
                  diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
                  offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
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
                  diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[level]));
                  offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid[level]));

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
                  // !!! TODO R

               }
            }
         }

         // Setup temporary info for P on current level
         if (outer_level != num_levels-1)
         {
            diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[outer_level]));
            offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid[outer_level]));

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
TestCompGrids1New(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int current_level, HYPRE_Int check_ghost_info)
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
               if (level != num_levels-1) error_code = RecursivelyBuildPsiCompositeNew(i, padding[level], compGrid, add_flag, 1, &nodes_to_add, padding[level+1], level, 0);
               else error_code = RecursivelyBuildPsiCompositeNew(i, padding[level], compGrid, add_flag, 0, NULL, 0, level, 0);
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
            if (add_flag[level][i] == num_ghost_layers + 1) error_code = RecursivelyBuildPsiCompositeNew(i, padding[level], compGrid, add_flag, 0, NULL, 0, level, 0);
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
TestCompGrids2New(hypre_ParAMGData *amg_data)
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

         hypre_CSRMatrix *A_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
         hypre_CSRMatrix *A_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
         hypre_CSRMatrix *P_diag = NULL;
         hypre_CSRMatrix *P_offd = NULL;

         // Broadcast the number of nodes and num non zeros for A, P, and R
         HYPRE_Int num_nonowned = 0;
         HYPRE_Int proc_first_index = 0;
         HYPRE_Int proc_coarse_first_index = 0;
         HYPRE_Int nnz_A_diag = 0;
         HYPRE_Int nnz_A_offd = 0;
         HYPRE_Int nnz_P_diag = 0;
         HYPRE_Int nnz_P_offd = 0;
         HYPRE_Int nnz_R_diag = 0;
         HYPRE_Int nnz_R_offd = 0;
         HYPRE_Int sizes_buf[9];
         if (myid == proc) 
         {
            num_nonowned = hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
            proc_first_index = hypre_ParCompGridFirstGlobalIndex(compGrid[level]);
            nnz_A_diag = hypre_CSRMatrixI(A_diag)[num_nonowned];
            nnz_A_offd = hypre_CSRMatrixI(A_offd)[num_nonowned];
            if (level != num_levels-1)
            {
               proc_coarse_first_index = hypre_ParCompGridFirstGlobalIndex(compGrid[level+1]);
               P_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[level]));
               P_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid[level]));
               nnz_P_diag = hypre_CSRMatrixI(P_diag)[num_nonowned];
               nnz_P_offd = hypre_CSRMatrixI(P_offd)[num_nonowned];
            }
            // !!! TODO R
            // if (level != 0 && hypre_ParCompGridRRowPtr(compGrid[level]))
            // {
            //    nnz_R = hypre_ParCompGridRRowPtr(compGrid[level])[num_nodes];
            // }
            sizes_buf[0] = num_nonowned;
            sizes_buf[1] = proc_first_index;
            sizes_buf[2] = proc_coarse_first_index;
            sizes_buf[3] = nnz_A_diag;
            sizes_buf[4] = nnz_A_offd;
            sizes_buf[5] = nnz_P_diag;
            sizes_buf[6] = nnz_P_offd;
            sizes_buf[7] = nnz_R_diag;
            sizes_buf[8] = nnz_R_offd;
         }
         hypre_MPI_Bcast(sizes_buf, 9, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);
         num_nonowned = sizes_buf[0];
         proc_first_index = sizes_buf[1];
         proc_coarse_first_index = sizes_buf[2];
         nnz_A_diag = sizes_buf[3];
         nnz_A_offd = sizes_buf[4];
         nnz_P_diag = sizes_buf[5];
         nnz_P_offd = sizes_buf[6];
         nnz_R_diag = sizes_buf[7];
         nnz_R_offd = sizes_buf[8];

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

         // !!! TODO R

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
               // !!! TODO R
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
            // !!! TODO R
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
