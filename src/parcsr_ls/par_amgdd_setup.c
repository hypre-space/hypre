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

HYPRE_Int*
PackSendBufferNew( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, HYPRE_Int *send_flag_buffer_size, 
   HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes, HYPRE_Int proc, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int *padding, 
   HYPRE_Int num_ghost_layers, HYPRE_Int symmetric );

HYPRE_Int*
PackSendBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, HYPRE_Int *send_flag_buffer_size, 
   HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes, HYPRE_Int proc, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int *padding, 
   HYPRE_Int num_ghost_layers, HYPRE_Int symmetric );

HYPRE_Int
RecursivelyBuildPsiCompositeNew(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid **compGrids, HYPRE_Int **add_flags,
                           HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding, HYPRE_Int level, HYPRE_Int use_sort);

HYPRE_Int
RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int *add_flag_coarse, 
   HYPRE_Int *sort_map, HYPRE_Int *sort_map_coarse, HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding, HYPRE_Int level);

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int *recv_map_send_buffer, HYPRE_Int **recv_map, HYPRE_Int *num_recv_nodes, HYPRE_Int *recv_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, hypre_ParCompGrid **compGrid);

HYPRE_Int
UnpackSendFlagBuffer(HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, HYPRE_Int *send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
CommunicateRemainingMatrixInfo(hypre_ParAMGData* amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *communication_cost, HYPRE_Int symmetric);

HYPRE_Int
TestCompGrids1New(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int current_level, HYPRE_Int check_ghost_info);

HYPRE_Int
TestCompGrids1(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int current_level, HYPRE_Int check_ghost_info);

HYPRE_Int
TestCompGrids2(hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int num_levels, hypre_ParCSRMatrix **A, hypre_ParCSRMatrix **P, hypre_ParCSRMatrix **R);

HYPRE_Int
CheckCompGridLocalIndices(hypre_ParAMGData *amg_data);

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
   HYPRE_Int      level,i,j,l;
   HYPRE_Int      num_send_procs, num_recv_procs;
   HYPRE_Int      **send_buffer_size;
   HYPRE_Int      **recv_buffer_size;
   HYPRE_Int      ***num_send_nodes;
   HYPRE_Int      ***num_recv_nodes;
   HYPRE_Int      ****send_flag;
   HYPRE_Int      ****recv_map;
   HYPRE_Int      **send_buffer;
   HYPRE_Int      **recv_buffer;
   HYPRE_Int      **send_flag_buffer;
   HYPRE_Int      **recv_map_send_buffer;
   HYPRE_Int      *send_flag_buffer_size;
   HYPRE_Int      *recv_map_send_buffer_size;
   hypre_MPI_Request       *requests;
   hypre_MPI_Status        *status;
   HYPRE_Int               request_counter = 0;



   // !!! New
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
         hypre_ParCompGridInitialize( amg_data, 0, level, symmetric );
         hypre_ParCompGridInitializeNew( amg_data, 0, level, symmetric );
      }
      hypre_ParCompGridFinalize(compGrid, NULL, amgdd_start_level, num_levels, hypre_ParAMGDataAMGDDUseRD(amg_data), verify_amgdd);
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
      hypre_ParCompGridInitialize( amg_data, padding[level], level, symmetric );
      hypre_ParCompGridInitializeNew( amg_data, padding[level], level, symmetric );
   }
   if (timers) hypre_EndTiming(timers[0]);

   // !!! Debug: check that initialization produces expected coarse indices
   // if (myid == 1)
   // {
   //    for (level = 0; level < hypre_ParAMGDataNumLevels(amg_data) - 1; level++)
   //    {
   //       printf("\nLevel %d hypre_ParCompGridOwnedCoarseIndices vs. hypre_ParCompGridCoarseLocalIndices:\n", level);
   //       for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid[level]); i++)
   //       {
   //          printf("%d, %d\n", hypre_ParCompGridOwnedCoarseIndices(compGrid[level])[i], hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i]);
   //       }
   //    }
   // }



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
      hypre_ParCompGridDebugPrint( compGrid[level], filename, 0 );
   }
   #endif


   // !!! Debug
   for (level = 0; level < num_levels; level++)
   {
      hypre_sprintf(filename, "outputs/CompGrids/initCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrintNew( compGrid[level], filename );
   }


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



   hypre_ParCompGridCommPkg *compGridCommPkgNew = hypre_ParCompGridCommPkgCreate(num_levels);
   // hypre_ParAMGDataCompGridCommPkg(amg_data) = compGridCommPkgNew; // !!! NEW: not yet assigning this

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
      SetupNearestProcessorNeighbors(hypre_ParAMGDataAArray(amg_data)[level], compGrid[level], compGridCommPkg, level, padding, num_ghost_layers, communication_cost);
      SetupNearestProcessorNeighborsNew(hypre_ParAMGDataAArray(amg_data)[level], compGrid[level], compGridCommPkgNew, level, padding, num_ghost_layers, communication_cost);
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
      num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level];
      num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level];

      if ( num_send_procs || num_recv_procs ) // If there are any owned nodes on this level
      {
         // !!! Debug
         // printf("Rank %d active on level %d\n", myid, level);

         // Do some allocations
         requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         send_buffer = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
         send_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_send_procs, HYPRE_MEMORY_HOST);
         recv_buffer = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         recv_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_recv_procs, HYPRE_MEMORY_HOST);

         send_flag[level] = hypre_CTAlloc(HYPRE_Int**, num_send_procs, HYPRE_MEMORY_HOST);
         num_send_nodes[level] = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
         recv_map[level] = hypre_CTAlloc(HYPRE_Int**, num_recv_procs, HYPRE_MEMORY_HOST);
         num_recv_nodes[level] = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);

         send_flag_buffer = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
         send_flag_buffer_size = hypre_CTAlloc(HYPRE_Int, num_send_procs, HYPRE_MEMORY_HOST);
         recv_map_send_buffer = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         recv_map_send_buffer_size = hypre_CTAlloc(HYPRE_Int, num_recv_procs, HYPRE_MEMORY_HOST);



         // !!! NEW
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
            send_flag[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_send_nodes[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
            send_buffer[i] = PackSendBuffer( compGrid, compGridCommPkg, &(send_buffer_size[level][i]), 
                                             &(send_flag_buffer_size[i]), send_flag, num_send_nodes, i, level, num_levels, padding, 
                                             num_ghost_layers, symmetric );
         }

         for (i = 0; i < num_send_procs; i++)
         {
            // !!! NEW
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
            hypre_MPI_Irecv( &(recv_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]) );

         }
         for (i = 0; i < num_recv_procs; i++)
         {
            hypre_MPI_Irecv( &(recv_buffer_size_new[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkgNew)[level][i], 0, comm, &(requests_new[request_counter_new++]) );
         }

         // send the buffer sizes
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_MPI_Isend(&(send_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]));
         }

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
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests, status );
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter = 0;


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
            recv_buffer[i] = hypre_CTAlloc(HYPRE_Int, recv_buffer_size[level][i], HYPRE_MEMORY_HOST );
            hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 1, comm, &(requests[request_counter++]));
         }
         for (i = 0; i < num_recv_procs; i++)
         {
            recv_buffer_new[i] = hypre_CTAlloc(HYPRE_Int, recv_buffer_size_new[level][i], HYPRE_MEMORY_HOST );
            hypre_MPI_Irecv( recv_buffer_new[i], recv_buffer_size_new[level][i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkgNew)[level][i], 1, comm, &(requests_new[request_counter_new++]));
         }

         // send the buffers
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_MPI_Isend(send_buffer[i], send_buffer_size[level][i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 1, comm, &(requests[request_counter++]));
         }
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_MPI_Isend(send_buffer_new[i], send_buffer_size_new[level][i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkgNew)[level][i], 1, comm, &(requests_new[request_counter_new++]));
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


         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests_new, status_new );
         hypre_TFree(requests_new, HYPRE_MEMORY_HOST);
         hypre_TFree(status_new, HYPRE_MEMORY_HOST);
         requests_new = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status_new = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter_new = 0;

         if (timers) hypre_EndTiming(timers[3]);

         //////////// Unpack the received buffers ////////////

         if (timers) hypre_BeginTiming(timers[4]);
         // !!! New
         HYPRE_Int **A_tmp_info = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_recv_procs; i++)
         {

            recv_map[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_recv_nodes[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
            
            UnpackRecvBuffer(recv_buffer[i], compGrid, compGridCommPkg, 
               send_flag, num_send_nodes, 
               recv_map, num_recv_nodes, 
               &(recv_map_send_buffer_size[i]), level, num_levels, nodes_added_on_level, i, num_resizes, symmetric);

            recv_map_send_buffer[i] = hypre_CTAlloc(HYPRE_Int, recv_map_send_buffer_size[i], HYPRE_MEMORY_HOST);
            PackRecvMapSendBuffer(recv_map_send_buffer[i], recv_map[level][i], num_recv_nodes[level][i], &(recv_buffer_size[level][i]), level, num_levels, compGrid);
         }
         for (i = 0; i < num_recv_procs; i++)
         {
            // !!! New
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
         total_bin_search_count += hypre_ParCompGridSetupLocalIndices(compGrid, nodes_added_on_level, amgdd_start_level, num_levels, symmetric_tmp);
         for (j = level; j < num_levels; j++) nodes_added_on_level[j] = 0;

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
            send_flag_buffer[i] = hypre_CTAlloc(HYPRE_Int, send_flag_buffer_size[i], HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv( send_flag_buffer[i], send_flag_buffer_size[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 2, comm, &(requests[request_counter++]));
         }
         for (i = 0; i < num_send_procs; i++)
         {
            send_flag_buffer_new[i] = hypre_CTAlloc(HYPRE_Int, send_flag_buffer_size_new[i], HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv( send_flag_buffer_new[i], send_flag_buffer_size_new[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkgNew)[level][i], 2, comm, &(requests_new[request_counter_new++]));
         }

         // send the recv_map_send_buffer's
         for (i = 0; i < num_recv_procs; i++)
         {
            // !!! Check recv_map_send_buffer_size (make sure not overallocated)
            hypre_MPI_Isend( recv_map_send_buffer[i], recv_map_send_buffer_size[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 2, comm, &(requests[request_counter++]));
         }
         for (i = 0; i < num_recv_procs; i++)
         {
            hypre_MPI_Isend( recv_map_send_buffer_new[i], recv_map_send_buffer_size_new[i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkgNew)[level][i], 2, comm, &(requests_new[request_counter_new++]));
            
            // !!! Debug
            // if (myid == 1 && i == 0)
            // {
            //    printf("recv_map_send_buffer_new = ");
            //    for (j = 0; j < recv_map_send_buffer_size_new[i]; j++)
            //    {
            //       printf("%d ", recv_map_send_buffer_new[i][j]);
            //    }
            //    printf("\n");
            // }

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

         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests_new, status_new );
         hypre_TFree(requests_new, HYPRE_MEMORY_HOST);
         hypre_TFree(status_new, HYPRE_MEMORY_HOST);

         // unpack and setup the send flag arrays
         for (i = 0; i < num_send_procs; i++)
         {
            UnpackSendFlagBuffer(send_flag_buffer[i], send_flag[level][i], num_send_nodes[level][i], &(send_buffer_size[level][i]), level, num_levels);
         }
         for (i = 0; i < num_send_procs; i++)
         {
            UnpackSendFlagBufferNew(send_flag_buffer_new[i], send_flag_new[level][i], num_send_nodes_new[level][i], &(send_buffer_size_new[level][i]), level, num_levels);
            
            // !!! Debug
            // if (myid == 1)
            // {
            //    printf("Send flag after unpack:\n");
            //    HYPRE_Int l;
            //    for (l = level; l < num_levels; l++)
            //    {
            //       printf("level %d\n", l);
            //       for (j = 0; j < num_send_nodes_new[level][0][l]; j++)
            //       {
            //          printf("%d, %d\n", send_flag_new[level][0][l][j], send_flag[level][0][l][j]);
            //       }
            //    }
            // }

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
      // #if DEBUGGING_MESSAGES
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      if (myid == 0) hypre_printf("All ranks: done with level %d\n", level);
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      // #endif 



      // #if DEBUG_COMP_GRID == 2
      for (i = level; i < num_levels; i++)
      {
         hypre_sprintf(filename, "outputs/CompGrids/level%dCompGridRank%dLevel%d", level, myid, i);
         hypre_ParCompGridDebugPrintNew( compGrid[i], filename );
      }
      // #endif




      #if DEBUG_COMP_GRID
      CheckCompGridLocalIndices(amg_data);
      HYPRE_Int error_code;
      error_code = TestCompGrids1(compGrid, num_levels, padding, num_ghost_layers, level, 1);
      error_code = TestCompGrids1New(compGrid, num_levels, padding, num_ghost_layers, level, 1);
      if (error_code)
         hypre_printf("TestCompGrids1 failed! Rank %d, level %d\n", myid, level);
      else
         hypre_printf("TestCompGrids1 passed! Rank %d, level %d\n", myid, level);
      #endif

      // !!! Debug
      // if (level == 2)
      // {
      //    MPI_Finalize();
      //    exit(0);
      // }

      // !!! Debug
      // if (myid == 1)
      // {
      //    printf("\nLevel %d, col_map_offd = \n", level);
      //    for (i = 0; i < hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(hypre_ParAMGDataAArray(amg_data)[level])); i++)
      //       printf("%d\n", hypre_ParCSRMatrixColMapOffd(hypre_ParAMGDataAArray(amg_data)[level])[i]);
      //    printf("\nLevel %d, nonowned comp grid global indices = \n", level);
      //    for (i = hypre_ParCompGridNumOwnedNodes(compGrid[level]); i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
      //       if (hypre_ParCompGridRealDofMarker(compGrid[level])[i])
      //          printf("%d\n", hypre_ParCompGridGlobalIndices(compGrid[level])[i]);
      // }
   }

   // !!! Debug
   MPI_Finalize();
   exit(0);

   /////////////////////////////////////////////////////////////////

   // Done with loop over levels. Now just finalize things.

   /////////////////////////////////////////////////////////////////

   // !!! Debug
   HYPRE_Int total_nnz = 0;
   for (level = 0; level < num_levels; level++) total_nnz += hypre_ParCompGridARowPtr(compGrid[level])[hypre_ParCompGridNumNodes(compGrid[level])];
   // printf("rank %d, total total_bin_search_count = %d, percentage = %f\n", myid, total_bin_search_count, ((double)total_bin_search_count)/((double)total_nnz));

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

   if (timers) hypre_EndTiming(timers[7]);

   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[5]);

   // Setup the local indices for P
   hypre_ParCompGridSetupLocalIndicesP(compGrid, amgdd_start_level, num_levels);

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
      hypre_ParCompGridDebugPrint( compGrid[level], filename, 0 );
   }
   #endif

   #if DEBUG_COMP_GRID
   error_code = TestCompGrids2(compGrid, compGridCommPkg, num_levels, hypre_ParAMGDataAArray(amg_data), hypre_ParAMGDataPArray(amg_data), hypre_ParAMGDataRArray(amg_data));
   if (error_code)
   {
      hypre_printf("TestCompGrids2 failed!\n");
      test_failed = 1;
   }
   else hypre_printf("TestCompGrids2 success\n");
   #endif

   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[8]);

   // Finalize the comp grid structures
   hypre_ParCompGridFinalize(compGrid, compGridCommPkg, amgdd_start_level, num_levels, hypre_ParAMGDataAMGDDUseRD(amg_data), verify_amgdd);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with hypre_ParCompGridFinalize()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // Finalize the send flag and the recv
   hypre_ParCompGridCommPkgFinalize(amg_data, compGridCommPkg, compGrid);
   
   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with hypre_ParCompGridCommPkgFinalize()\n");
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
         communication_cost[level*10 + 5] += hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][ hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level] ]*sizeof(HYPRE_Complex);
      }
   }

   // Cleanup memory
   hypre_TFree(num_resizes, HYPRE_MEMORY_HOST);
   hypre_TFree(nodes_added_on_level, HYPRE_MEMORY_HOST);

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
      hypre_ParCompGridDebugPrint( compGrid[level], filename, coarse_num_nodes );
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


   // !!! Debug
   // if (myid == 0 && current_level == 0)
   //    printf("buffer_size level 0 = %d\n", (*buffer_size));

   // !!! Debug
   // if (myid == 3 && current_level == 0 && hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][proc] == 0)
   //    printf("Rank 3 sending to 0 on level 0, nodes_to_add %d\n", nodes_to_add);

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
            while (hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[inv_sort_map[i]] < hypre_ParCompGridFirstGlobalIndex(compGrid[level])
               && i < hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
            {
               add_flag_index = i + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               if (add_flag[level][add_flag_index] > num_ghost_layers)
               {
                  add_flag[level][add_flag_index] = cnt + 1;
                  send_flag[current_level][proc][level][cnt] = inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
                  cnt++;
               }
               else if (add_flag[level][add_flag_index] > 0)
               {
                  add_flag[level][add_flag_index] = cnt + 1;
                  send_flag[current_level][proc][level][cnt] = -(inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]) + 1);
                  cnt++;
               }
               i++;
            }
         }
         // Then the owned block
         for (add_flag_index = 0; add_flag_index < hypre_ParCompGridNumOwnedNodes(compGrid[level]); add_flag_index++)
         {
            if (add_flag[level][add_flag_index] > num_ghost_layers)
            {
               add_flag[level][add_flag_index] = cnt + 1;
               send_flag[current_level][proc][level][cnt] = add_flag_index;
               cnt++;
            }
            else if (add_flag[level][add_flag_index] > 0)
            {
               add_flag[level][add_flag_index] = cnt + 1;
               send_flag[current_level][proc][level][cnt] = -(add_flag_index+1);
               cnt++;
            }
         }
         // Finally the nonowned indices coming after the owned block
         if (hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
         {
            while (i < hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
            {
               add_flag_index = i + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               if (add_flag[level][add_flag_index] > num_ghost_layers)
               {
                  add_flag[level][add_flag_index] = cnt + 1;
                  send_flag[current_level][proc][level][cnt] = inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
                  cnt++;
               }
               else if (add_flag[level][add_flag_index] > 0)
               {
                  add_flag[level][add_flag_index] = cnt + 1;
                  send_flag[current_level][proc][level][cnt] = -(inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]) + 1);
                  cnt++;
               }
               i++;
            }
         }

         // !!! TODO
         // Eliminate redundant send info by comparing with previous send_flags
         // HYPRE_Int current_send_proc = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[current_level][proc];
         // HYPRE_Int prev_proc;
         // for (prev_level = current_level+1; prev_level <= level; prev_level++)
         // {
         //    for (prev_proc = 0; prev_proc < hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[prev_level]; prev_proc++)
         //    if (hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[prev_level][prev_proc] == current_send_proc)
         //    {
         //       // send_flag is in global index ordering, 
         //       while ()
         //       {

         //       }
         //    }
         // }

         // Count up the buffer sizes
         (*send_flag_buffer_size) += num_send_nodes[current_level][proc][level];
         if (level != num_levels-1) (*buffer_size) += 3*num_send_nodes[current_level][proc][level];
         else (*buffer_size) += 2*num_send_nodes[current_level][proc][level];
         
         // !!! Debug
         // if (myid == 0 && current_level == 0)
         //    printf("buffer_size level %d pre rows = %d\n", level, (*buffer_size));

         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            
            // !!! Debug
            // if (myid == 0 && current_level == 0 && level == 2)
            //    printf("   send_elmt = %d, buffer_size = %d\n", send_elmt, (*buffer_size));

            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
               (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
               (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
               // !!! Debug
               if (myid == 0 && hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt] < 0)
                  printf("owned hypre_CSRMatrixI(diag)[%d] = %d, hypre_CSRMatrixI(diag)[%d] = %d\n", 
                     send_elmt+1, hypre_CSRMatrixI(diag)[send_elmt+1], send_elmt, hypre_CSRMatrixI(diag)[send_elmt]);
               if (myid == 0 && hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt] < 0)
                  printf("owned hypre_CSRMatrixI(offd)[%d] = %d, hypre_CSRMatrixI(offd)[%d] = %d\n", 
                     send_elmt+1, hypre_CSRMatrixI(offd)[send_elmt+1], send_elmt, hypre_CSRMatrixI(offd)[send_elmt]);
            }
            else
            {
               send_elmt -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid[level]));
               (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
               (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
               // !!! Debug
               if (myid == 0 && hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt] < 0)
                  printf("Rank %d, level %d, nonowned hypre_CSRMatrixI(diag)[%d] = %d, hypre_CSRMatrixI(diag)[%d] = %d, num nonowned = %d\n", 
                     myid, level, send_elmt+1, hypre_CSRMatrixI(diag)[send_elmt+1], send_elmt, hypre_CSRMatrixI(diag)[send_elmt], hypre_ParCompGridNumNonOwnedNodes(compGrid[level]));
               if (myid == 0 && hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt] < 0)
                  printf("Rank %d, level %d, nonowned hypre_CSRMatrixI(offd)[%d] = %d, hypre_CSRMatrixI(offd)[%d] = %d\n", 
                     myid, level, send_elmt+1, hypre_CSRMatrixI(offd)[send_elmt+1], send_elmt, hypre_CSRMatrixI(offd)[send_elmt]);
            }
         }
      }
      else break;

      // !!! Debug
      // if (myid == 0 && current_level == 0)
      //    printf("buffer_size level %d = %d\n", level, (*buffer_size));
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
               // !!! Debug
               if (level == current_level) printf("Error: Sending a nonowned point on current_level... should not happen\n");
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

      // !!! Debug
      // if (myid == 0 && current_level == 0)
      //    printf("cnt level %d pre rows = %d\n", level, cnt);

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

            // !!! Debug
            // if (myid == 0 && current_level == 0 && level == 2)
            //    printf("   send_elmt = %d, cnt = %d\n", send_elmt, cnt);

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
                  add_flag_index = sort_map[ hypre_CSRMatrixJ(offd)[j] ] + hypre_ParCompGridNumOwnedNodes(compGrid[level]); // !!! Double check... this is the tricky case. Use sort_map???
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
                     add_flag_index = sort_map[ hypre_CSRMatrixJ(diag)[j] ] + hypre_ParCompGridNumOwnedNodes(compGrid[level]); // !!! Double check... this is the tricky case. Use sort_map???
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

            // !!! Debug
            // if (myid == 0 && cnt > (*buffer_size)) printf("Rank %d, current_level %d, cnt = %d, buffer_size = %d\n", myid, current_level, cnt, (*buffer_size));
         }
      }
      // !!! Debug
      // if (myid == 0 && current_level == 0)
      //    printf("cnt level %d = %d\n", level, cnt);
   }

   // !!! Debug
   // if (cnt != (*buffer_size)) printf("Rank %d, current_level %d, cnt and buffer_size don't match in PackSendBufferNew()\n", myid, current_level);

   // Clean up memory
   for (level = 0; level < num_levels; level++)
   {
      if (add_flag[level]) hypre_TFree(add_flag[level], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);

   // Return the send buffer
   return send_buffer;
}

HYPRE_Int*
PackSendBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, 
   HYPRE_Int *send_flag_buffer_size, HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
   HYPRE_Int proc, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int symmetric )
{
   // send_buffer = [ num_psi_levels , [level] , [level] , ... ]
   // level = [ num send nodes, [global indices] , [coarse global indices] , [A row sizes] , [A col ind: either global indices or local col indices within buffer] ]

   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int            level,i,j,k,cnt,row_length,send_elmt,coarse_grid_index;
   HYPRE_Int            nodes_to_add = 0;
   HYPRE_Int            **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            **redundant_add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            **inv_send_flag = hypre_CTAlloc( HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST );
   HYPRE_Int            num_psi_levels = 1;
   HYPRE_Int            coarse_proc;

   // Get where to look in commPkgSendMapElmts
   HYPRE_Int            start = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][proc];
   HYPRE_Int            finish = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][proc+1];

   // Get the sort maps
   HYPRE_Int            *sort_map;
   HYPRE_Int            *sort_map_coarse;
   if (current_level != num_levels-1) sort_map_coarse = hypre_ParCompGridSortMap(compGrid[current_level+1]);

   // initialize send map buffer size
   (*send_flag_buffer_size) = num_levels - current_level;

   // Mark the nodes to send (including Psi_c grid plus ghost nodes)

   // Count up the buffer size for the starting nodes
   num_send_nodes[current_level][proc][current_level] = finish - start;
   send_flag[current_level][proc][current_level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level][proc][current_level], HYPRE_MEMORY_HOST );
   inv_send_flag[current_level] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[current_level]), HYPRE_MEMORY_HOST );

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
      inv_send_flag[current_level][send_elmt] = i - start + 1;

      (*buffer_size) += hypre_ParCompGridARowPtr(compGrid[current_level])[send_elmt+1] - hypre_ParCompGridARowPtr(compGrid[current_level])[send_elmt];
   }
   (*send_flag_buffer_size) += finish - start;

   // Add the nodes listed by the coarse grid counterparts if applicable
   // Note that the compGridCommPkg is set up to list all nodes within the padding plus ghost layers
   if (current_level != num_levels-1)
   {
      add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[current_level+1]), HYPRE_MEMORY_HOST );
      for (i = start; i < finish; i++)
      {
         // flag nodes that are repeated on the next coarse grid
         if (!hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[current_level][i])
         {
            send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];
            coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid[current_level])[send_elmt];
            if ( coarse_grid_index != -1 ) 
            {
               add_flag[current_level+1][ sort_map_coarse[coarse_grid_index] ] = padding[current_level+1]+1;
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
         sort_map = hypre_ParCompGridSortMap(compGrid[level]);
         if (level != num_levels-1) sort_map_coarse = hypre_ParCompGridSortMap(compGrid[level+1]);
         HYPRE_Int *inv_sort_map = hypre_ParCompGridInvSortMap(compGrid[level]);
         
         num_psi_levels++;
         (*buffer_size)++;
         nodes_to_add = 0;

         // if we need coarse info, allocate space for the add flag on the next level
         if (level != num_levels-1) add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]), HYPRE_MEMORY_HOST );

         // Expand by the padding on this level and add coarse grid counterparts if applicable
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][sort_map[i]] == padding[level] + 1)
            {
               // Recursively add the region of padding (flagging coarse nodes on the next level if applicable)
               if (level != num_levels-1) RecursivelyBuildPsiComposite(i, padding[level], compGrid[level], add_flag[level], add_flag[level+1], sort_map, sort_map_coarse, 1, &nodes_to_add, padding[level+1], level);
               else RecursivelyBuildPsiComposite(i, padding[level], compGrid[level], add_flag[level], NULL, sort_map, NULL, 0, &nodes_to_add, 0, level);
            }
         }

         // Expand by the number of ghost layers 
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][sort_map[i]] > 1) add_flag[level][sort_map[i]] = num_ghost_layers + 2;
            else if (add_flag[level][sort_map[i]] == 1) add_flag[level][sort_map[i]] = num_ghost_layers + 1;
         }
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            // Recursively add the region of ghost nodes (do not add any coarse nodes underneath)
            if (add_flag[level][sort_map[i]] == num_ghost_layers + 1) RecursivelyBuildPsiComposite(i, num_ghost_layers, compGrid[level], add_flag[level], NULL, sort_map, NULL, 0, NULL, 0, level);
         }

         // If we previously sent dofs to this proc on this level, mark the redundant_add_flag
         coarse_proc = -1;
         for (j = 0; j < hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level]; j++)
         {
            if (hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][j] == hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[current_level][proc]) coarse_proc = j;
         }
         if (coarse_proc != -1)
         {
            if (!redundant_add_flag[level]) redundant_add_flag[level] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level]), HYPRE_MEMORY_HOST );
            if (nodes_to_add) redundant_add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]), HYPRE_MEMORY_HOST );

            start = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][coarse_proc];
            finish = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][coarse_proc+1];
            if (nodes_to_add)
            {
               for (i = start; i < finish; i++)
               {
                  if (!hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level][i])
                  {
                     send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level][i];
                     coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid[level])[send_elmt];
                     if ( coarse_grid_index != -1 ) 
                     {
                        redundant_add_flag[level+1][ sort_map_coarse[coarse_grid_index] ] = padding[level+1]+1;
                     }
                  }
               }
            }

            // Expand by the padding on this level and add coarse grid counterparts if applicable
            HYPRE_Int dummy;
            for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
            {
               if (redundant_add_flag[level][sort_map[i]] == padding[level] + 1)
               {
                  // Recursively add the region of padding (flagging coarse nodes on the next level if applicable)
                  if (nodes_to_add) RecursivelyBuildPsiComposite(i, padding[level], compGrid[level], redundant_add_flag[level], redundant_add_flag[level+1], sort_map, sort_map_coarse, 1, &dummy, padding[level+1], level);
                  else RecursivelyBuildPsiComposite(i, padding[level], compGrid[level], redundant_add_flag[level], NULL, sort_map, NULL, 0, &dummy, 0, level);
               }
            }

            // Expand by the number of ghost layers 
            for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
            {
               if (redundant_add_flag[level][sort_map[i]] > 1) redundant_add_flag[level][sort_map[i]] = num_ghost_layers + 2;
               else if (redundant_add_flag[level][sort_map[i]] == 1) redundant_add_flag[level][sort_map[i]] = num_ghost_layers + 1;
            }
            for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
            {
               // Recursively add the region of ghost nodes (do not add any coarse nodes underneath)
               if (redundant_add_flag[level][sort_map[i]] == num_ghost_layers + 1) RecursivelyBuildPsiComposite(i, num_ghost_layers, compGrid[level], redundant_add_flag[level], NULL, sort_map, NULL, 0, NULL, 0, level);
            }

            // Make sure starting elements on this level are included
            for (i = start; i < finish; i++)
            {
               send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level][i];
               redundant_add_flag[level][sort_map[send_elmt]] = 1;
            }
         }

         // Eliminate redundant elements in add_flag
         if (redundant_add_flag[level])
         {
            for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
            {
               // Note: redundant add flag should not cancel out dofs flagged for send as real when same dof was previously sent as ghost
               if ( redundant_add_flag[level][sort_map[i]] && !(add_flag[level][sort_map[i]] > num_ghost_layers && redundant_add_flag[level][sort_map[i]] <= num_ghost_layers) )
               {
                  add_flag[level][sort_map[i]] = 0;
               }
            }
         }
         
         // Count up the total number of send nodes 
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] > 0)
            {
               num_send_nodes[current_level][proc][level]++;
            }
         }
         
         // Save the indices (in global index ordering) so I don't have to keep looping over all nodes in compGrid when I pack the buffer
         send_flag[current_level][proc][level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_HOST );
         inv_send_flag[level] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level]), HYPRE_MEMORY_HOST );
         cnt =  0;

         for (j = 0; j < hypre_ParCompGridNumNodes(compGrid[level]); j++)
         {
            if (add_flag[level][j] > num_ghost_layers)
            {
               inv_send_flag[level][ inv_sort_map[j] ] = cnt + 1;
               send_flag[current_level][proc][level][cnt++] = inv_sort_map[j];
            }
            else if (add_flag[level][j] > 0)
            {
               inv_send_flag[level][ inv_sort_map[j] ] = cnt + 1;
               send_flag[current_level][proc][level][cnt++] = -(inv_sort_map[j]+1);
            }
         }


         // Count up the buffer sizes
         (*send_flag_buffer_size) += num_send_nodes[current_level][proc][level];
         if (level != num_levels-1) (*buffer_size) += 3*num_send_nodes[current_level][proc][level];
         else (*buffer_size) += 2*num_send_nodes[current_level][proc][level];
         
         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            (*buffer_size) += hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt + 1 ] - hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt ];   
         }
      }
      else break;
   }

   // Allocate the buffer
   HYPRE_Int *send_buffer = hypre_CTAlloc(HYPRE_Int, (*buffer_size), HYPRE_MEMORY_HOST);

   // Pack the buffer
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
            send_buffer[cnt++] = -(hypre_ParCompGridGlobalIndices(compGrid[level])[ send_elmt ] + 1);
         }
         else 
         {
            send_buffer[cnt++] = hypre_ParCompGridGlobalIndices(compGrid[level])[ send_elmt ];
         }
      }

      // if not on last level, copy coarse gobal indices
      if (level != num_levels-1)
      {
         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            send_buffer[cnt++] = hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ send_elmt ];
         }
      }

      // store the row length for matrix A
      if (symmetric)
      {
         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_buffer[cnt++] = 0;
            else
            {
               row_length = hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt + 1 ]
                          - hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt ];
               send_buffer[cnt++] = row_length;
            }
         }
      }
      else
      {
         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            row_length = hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt + 1 ]
                       - hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt ];
            send_buffer[cnt++] = row_length;
         }
      }

      // copy global indices for matrix A
      if (symmetric)
      {
         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt >= 0)
            {
               row_length = hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt + 1 ]
                          - hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt ];
               HYPRE_Int offset = hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt ];

               for (j = 0; j < row_length; j++)
               {
                  HYPRE_Int local_index = hypre_ParCompGridAColInd(compGrid[level])[ offset + j ];
                  if (local_index >= 0)
                  {
                     if (inv_send_flag[level][local_index] > 0)
                     {
                        send_buffer[cnt++] = inv_send_flag[level][local_index] - 1;
                     }
                     else
                        send_buffer[cnt++] = -(hypre_ParCompGridAGlobalColInd(compGrid[level])[ offset + j ] + 1);
                  }
                  else
                     send_buffer[cnt++] = -(hypre_ParCompGridAGlobalColInd(compGrid[level])[ offset + j ] + 1);
               }
            }
         }
      }
      else
      {
         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            row_length = hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt + 1 ]
                       - hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt ];
            HYPRE_Int offset = hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt ];
            for (j = 0; j < row_length; j++)
            {
               HYPRE_Int local_index = hypre_ParCompGridAColInd(compGrid[level])[ offset + j ];
               if (local_index >= 0)
               {
                  if (inv_send_flag[level][local_index] > 0)
                  {
                     send_buffer[cnt++] = inv_send_flag[level][local_index] - 1;
                  }
                  else
                  {
                     send_buffer[cnt++] = -(hypre_ParCompGridAGlobalColInd(compGrid[level])[ offset + j ] + 1);
                  }
               }
               else
               {
                  send_buffer[cnt++] = -(hypre_ParCompGridAGlobalColInd(compGrid[level])[ offset + j ] + 1);
               }
            }
         }
      }
   }

   // Clean up memory
   for (level = 0; level < num_levels; level++)
   {
      if (add_flag[level]) hypre_TFree(add_flag[level], HYPRE_MEMORY_HOST);
      if (redundant_add_flag[level]) hypre_TFree(redundant_add_flag[level], HYPRE_MEMORY_HOST);
      if (inv_send_flag[level]) hypre_TFree(inv_send_flag[level], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);
   hypre_TFree(redundant_add_flag, HYPRE_MEMORY_HOST);
   hypre_TFree(inv_send_flag, HYPRE_MEMORY_HOST);

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
               // !!! Debug
               if (sort_index >= hypre_ParCompGridNumOwnedNodes(compGrids[level+1]) + hypre_ParCompGridNumNonOwnedNodes(compGrids[level+1]))
                  printf("Rank %d, level %d, sort_index = %d, num_owned = %d, num_nonowned = %d, coarse_grid_index = %d\n", myid, level, sort_index, hypre_ParCompGridNumOwnedNodes(compGrids[level+1]), hypre_ParCompGridNumNonOwnedNodes(compGrids[level+1]), coarse_grid_index);
               add_flag_coarse[ sort_index ] = padding+1;
               *nodes_to_add = 1;
            }
         }
      }
      else
      {
         error_code = 1;
         if (owned == 1) hypre_printf("Rank %d: Error! Negative col index encountered in owned matrix\n"); 
         // !!! Debug
         // else hypre_printf("Rank %d: Error! Ran into a -1 index in diag when building Psi_c\n", myid);
         else if (myid == 0) hypre_printf("Rank %d: Error! Level %d Ran into a -1 index in nonowned_diag, index = %d, node = %d (gid %d)\n", 
            myid, level, index, node, hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[node]);
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
RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int *add_flag_coarse, 
   HYPRE_Int *sort_map, HYPRE_Int *sort_map_coarse, HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding, HYPRE_Int level)
{
   HYPRE_Int         i,index,sort_index,coarse_grid_index;
   HYPRE_Int error_code = 0;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Look at neighbors
   for (i = hypre_ParCompGridARowPtr(compGrid)[node]; i < hypre_ParCompGridARowPtr(compGrid)[node+1]; i++)
   {
      // Get the index of the neighbor
      index = hypre_ParCompGridAColInd(compGrid)[i];

      // If the neighbor info is available on this proc
      if (index >= 0)
      {
         if (sort_map) sort_index = sort_map[index];
         else sort_index = index;

         // And if we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[sort_index] < m)
         {
            add_flag[sort_index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) error_code = RecursivelyBuildPsiComposite(index, m-1, compGrid, add_flag, add_flag_coarse, sort_map, sort_map_coarse, need_coarse_info, nodes_to_add, padding, level);
         }
         // If m = 1, we won't do another recursive call, so make sure to flag the coarse grid here if applicable
         if (need_coarse_info && m == 1)
         {
            coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid)[index];
            if ( coarse_grid_index != -1 ) 
            {
               // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
               if (sort_map_coarse) sort_index = sort_map_coarse[coarse_grid_index];
               else sort_index = coarse_grid_index;
               add_flag_coarse[ sort_index ] = padding+1;
               *nodes_to_add = 1;   
            }
         }
      }
      else
      {
         error_code = 1; 
         hypre_printf("Rank %d: Error! Ran into a -1 index when building Psi_c\n", myid);
         // if (myid == 3 && node == 1627) printf("Rank %d, level %d: Error! Ran into a -1 index when building Psi_c,\nnode = %d with global id %d, index = %d with global id = %d, m = %d\n",
         //    myid, level, node, hypre_ParCompGridGlobalIndices(compGrid)[node], index, hypre_ParCompGridAGlobalColInd(compGrid)[i], m);
      }
   }

   // Flag this node on the next coarsest level if applicable
   if (need_coarse_info)
   {
      coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid)[node];
      if ( coarse_grid_index != -1 ) 
      {
         // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
         if (sort_map_coarse) sort_index = sort_map_coarse[coarse_grid_index];
         else sort_index = coarse_grid_index;
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
PackRecvMapSendBuffer(HYPRE_Int *recv_map_send_buffer, 
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
   for (level = current_level; level < num_levels; level++)
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
               if (hypre_ParCompGridRealDofMarker(compGrid[level])[ recv_map[level][i] ] > 0)
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
UnpackSendFlagBuffer(HYPRE_Int *send_flag_buffer, 
   HYPRE_Int **send_flag, 
   HYPRE_Int *num_send_nodes,
   HYPRE_Int *send_buffer_size,
   HYPRE_Int current_level, 
   HYPRE_Int num_levels)
{
   HYPRE_Int      level, i, cnt, num_nodes;
   cnt = 0;
   *send_buffer_size = 0;
   for (level = current_level; level < num_levels; level++)
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
CommunicateRemainingMatrixInfo(hypre_ParAMGData* amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *communication_cost, HYPRE_Int symmetric)
{
   HYPRE_Int outer_level,proc,level,i,j;
   HYPRE_Int num_levels = hypre_ParCompGridCommPkgNumLevels(compGridCommPkg);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);

   HYPRE_Int myid,num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   HYPRE_Int ***temp_PColInd = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Complex ***temp_PData = hypre_CTAlloc(HYPRE_Complex**, num_levels, HYPRE_MEMORY_HOST);
   for (outer_level = amgdd_start_level; outer_level < num_levels; outer_level++)
   {
      HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[outer_level])[hypre_ParCompGridNumOwnedBlocks(compGrid[outer_level])];
      temp_PColInd[outer_level] = hypre_CTAlloc(HYPRE_Int*, hypre_ParCompGridNumNodes(compGrid[outer_level])
         - num_owned_nodes, HYPRE_MEMORY_HOST);
      temp_PData[outer_level] = hypre_CTAlloc(HYPRE_Complex*, hypre_ParCompGridNumNodes(compGrid[outer_level])
         - num_owned_nodes, HYPRE_MEMORY_HOST);
   }

   HYPRE_Int ***temp_RColInd = NULL;
   HYPRE_Complex ***temp_RData = NULL;
   if (hypre_ParAMGDataRestriction(amg_data))
   {
      temp_RColInd = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);
      temp_RData = hypre_CTAlloc(HYPRE_Complex**, num_levels, HYPRE_MEMORY_HOST);
      for (outer_level = amgdd_start_level; outer_level < num_levels; outer_level++)
      {
         HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[outer_level])[hypre_ParCompGridNumOwnedBlocks(compGrid[outer_level])];
         temp_RColInd[outer_level] = hypre_CTAlloc(HYPRE_Int*, hypre_ParCompGridNumNodes(compGrid[outer_level])
            - num_owned_nodes, HYPRE_MEMORY_HOST);
         temp_RData[outer_level] = hypre_CTAlloc(HYPRE_Complex*, hypre_ParCompGridNumNodes(compGrid[outer_level])
            - num_owned_nodes, HYPRE_MEMORY_HOST);
      }
   }

   // If no owned nodes, need to initialize start of PRowPtr
   for (level = amgdd_start_level; level < num_levels-1; level++)
   {
      if (!hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])])
         hypre_ParCompGridPRowPtr(compGrid[level])[0] = 0;
   }

   for (outer_level = num_levels-1; outer_level >= amgdd_start_level; outer_level--)
   {
      // Get send/recv info from the comp grid comm pkg
      HYPRE_Int num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level];
      HYPRE_Int num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[outer_level];
      HYPRE_Int *send_procs = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[outer_level];
      HYPRE_Int *recv_procs = hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[outer_level];

      if (num_send_procs || num_recv_procs)
      {
         // Get the buffer sizes
         HYPRE_Int *send_sizes = hypre_CTAlloc(HYPRE_Int, 2*num_send_procs, HYPRE_MEMORY_HOST);
         for (proc = 0; proc < num_send_procs; proc++)
         {
            for (level = outer_level; level < num_levels; level++)
            {      
               HYPRE_Int A_row_size = 0;
               HYPRE_Int P_row_size = 0;
               HYPRE_Int R_row_size = 0;
               HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
               for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
               {
                  HYPRE_Int idx, A_row_size;
                  idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                  if (idx < 0) idx = -(idx + 1);
                  A_row_size = hypre_ParCompGridARowPtr(compGrid[level])[idx+1] - hypre_ParCompGridARowPtr(compGrid[level])[idx];
                  
                  if (hypre_ParCompGridPRowPtr(compGrid[level]))
                  {
                     if (idx < num_owned_nodes) P_row_size = hypre_ParCompGridPRowPtr(compGrid[level])[idx+1] - hypre_ParCompGridPRowPtr(compGrid[level])[idx];
                     else P_row_size = hypre_ParCompGridPRowPtr(compGrid[level])[idx+1];
                  }
                  if (hypre_ParCompGridRRowPtr(compGrid[level]))
                  {
                     if (idx < num_owned_nodes) R_row_size = hypre_ParCompGridRRowPtr(compGrid[level])[idx+1] - hypre_ParCompGridRRowPtr(compGrid[level])[idx];
                     else R_row_size = hypre_ParCompGridRRowPtr(compGrid[level])[idx+1];
                  }

                  send_sizes[2*proc] += P_row_size + R_row_size;
                  send_sizes[2*proc+1] += A_row_size + P_row_size + R_row_size;
               }
               if (hypre_ParCompGridPRowPtr(compGrid[level])) send_sizes[2*proc] += hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level];
               if (hypre_ParCompGridRRowPtr(compGrid[level])) send_sizes[2*proc] += hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level];
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

         // Allocate, pack, and send buffers

         // int_send_buffer = [ [level] , [level] , ... , [level] ]
         // level = [ [P_rows], ( [R_rows] ) ]
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
            // Allocate
            int_send_buffers[proc] = hypre_CTAlloc(HYPRE_Int, send_sizes[2*proc], HYPRE_MEMORY_HOST);
            complex_send_buffers[proc] = hypre_CTAlloc(HYPRE_Complex, send_sizes[2*proc+1], HYPRE_MEMORY_HOST);

            // Pack
            HYPRE_Int int_cnt = 0;
            HYPRE_Int complex_cnt = 0;
            for (level = outer_level; level < num_levels; level++)
            {
               for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
               {
                  HYPRE_Int idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                  if (idx < 0) idx = -(idx + 1);
                  for (j = hypre_ParCompGridARowPtr(compGrid[level])[idx]; j < hypre_ParCompGridARowPtr(compGrid[level])[idx+1]; j++)
                  {
                     complex_send_buffers[proc][complex_cnt++] = hypre_ParCompGridAData(compGrid[level])[j];
                  }
               }
               if (level != num_levels-1)
               {      
                  HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
                  for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) idx = -(idx + 1);
                     if (idx < num_owned_nodes)
                     {
                        int_send_buffers[proc][int_cnt++] = hypre_ParCompGridPRowPtr(compGrid[level])[idx+1] - hypre_ParCompGridPRowPtr(compGrid[level])[idx];
                        for (j = hypre_ParCompGridPRowPtr(compGrid[level])[idx]; j < hypre_ParCompGridPRowPtr(compGrid[level])[idx+1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_ParCompGridPColInd(compGrid[level])[j];
                           complex_send_buffers[proc][complex_cnt++] = hypre_ParCompGridPData(compGrid[level])[j];
                        }
                     }
                     else
                     {
                        int_send_buffers[proc][int_cnt++] = hypre_ParCompGridPRowPtr(compGrid[level])[idx+1];
                        for (j = 0; j < hypre_ParCompGridPRowPtr(compGrid[level])[idx+1]; j++)
                        {
                           HYPRE_Int temp_idx = idx - num_owned_nodes;
                           int_send_buffers[proc][int_cnt++] = temp_PColInd[level][temp_idx][j];
                           complex_send_buffers[proc][complex_cnt++] = temp_PData[level][temp_idx][j];
                        }
                     }
                  }
               }
               if (level != 0 && hypre_ParAMGDataRestriction(amg_data))
               {
                  HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
                  for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) idx = -(idx + 1);
                     if (idx < num_owned_nodes)
                     {
                        int_send_buffers[proc][int_cnt++] = hypre_ParCompGridRRowPtr(compGrid[level])[idx+1] - hypre_ParCompGridRRowPtr(compGrid[level])[idx];
                        for (j = hypre_ParCompGridRRowPtr(compGrid[level])[idx]; j < hypre_ParCompGridRRowPtr(compGrid[level])[idx+1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_ParCompGridRColInd(compGrid[level])[j];
                           complex_send_buffers[proc][complex_cnt++] = hypre_ParCompGridRData(compGrid[level])[j];
                        }
                     }
                     else
                     {
                        int_send_buffers[proc][int_cnt++] = hypre_ParCompGridRRowPtr(compGrid[level])[idx+1];
                        for (j = 0; j < hypre_ParCompGridRRowPtr(compGrid[level])[idx+1]; j++)
                        {
                           HYPRE_Int temp_idx = idx - num_owned_nodes;
                           int_send_buffers[proc][int_cnt++] = temp_RColInd[level][temp_idx][j];
                           complex_send_buffers[proc][complex_cnt++] = temp_RData[level][temp_idx][j];
                        }
                     }
                  }
               }
            }
         }

         // Send
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
         
         // Unpack recvs
         for (proc = 0; proc < num_recv_procs; proc++)
         {
            HYPRE_Int int_cnt = 0;
            HYPRE_Int complex_cnt = 0;
            for (level = outer_level; level < num_levels; level++)
            {
               HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
               
               for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
               {
                  HYPRE_Int idx = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];
                  if (idx < 0) idx = -(idx + 1);

                  for (j = hypre_ParCompGridARowPtr(compGrid[level])[idx]; j < hypre_ParCompGridARowPtr(compGrid[level])[idx+1]; j++)
                  {
                     hypre_ParCompGridAData(compGrid[level])[j] = complex_recv_buffers[proc][complex_cnt++];
                  }
               }
               if (level != num_levels-1)
               {
                  for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) idx = -(idx + 1);
                     HYPRE_Int temp_idx = idx - num_owned_nodes;

                     HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];
                     hypre_ParCompGridPRowPtr(compGrid[level])[idx+1] = row_size;
                     if (!temp_PColInd[level][temp_idx])
                     {
                        temp_PColInd[level][temp_idx] = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
                        temp_PData[level][temp_idx] = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);

                        for (j = 0; j < row_size; j++)
                        {
                           temp_PColInd[level][temp_idx][j] = int_recv_buffers[proc][int_cnt++];
                           temp_PData[level][temp_idx][j] = complex_recv_buffers[proc][complex_cnt++];
                        }
                     }
                     else // !!! Question: is this else really necessary? Shouldn't there be no redundancy here?
                     {
                        int_cnt += row_size;
                        complex_cnt += row_size;
                     }
                  }
               }
               if (level != 0 && hypre_ParAMGDataRestriction(amg_data))
               {
                  for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) idx = -(idx + 1);
                     HYPRE_Int temp_idx = idx - num_owned_nodes;

                     HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];
                     hypre_ParCompGridRRowPtr(compGrid[level])[idx+1] = row_size;
                     if (!temp_RColInd[level][temp_idx])
                     {
                        temp_RColInd[level][temp_idx] = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
                        temp_RData[level][temp_idx] = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);

                        for (j = 0; j < row_size; j++)
                        {
                           temp_RColInd[level][temp_idx][j] = int_recv_buffers[proc][int_cnt++];
                           temp_RData[level][temp_idx][j] = complex_recv_buffers[proc][complex_cnt++];
                        }
                     }
                     else // !!! Question: is this else really necessary? Shouldn't there be no redundancy here?
                     {
                        int_cnt += row_size;
                        complex_cnt += row_size;
                     }
                  }
               }
            }
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

   // Fix up P and R
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      if (level != num_levels-1)
      {
         // Setup the row pointer (we stored the row sizes rather than pointer values as we unpacked)
         HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
         for (i = num_owned_nodes; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            hypre_ParCompGridPRowPtr(compGrid[level])[i+1] = hypre_ParCompGridPRowPtr(compGrid[level])[i] + hypre_ParCompGridPRowPtr(compGrid[level])[i+1];
         }

         // Make sure enough space is allocated for P
         if (hypre_ParCompGridPRowPtr(compGrid[level])[hypre_ParCompGridNumNodes(compGrid[level])] > hypre_ParCompGridPMemSize(compGrid[level]))
         {
            HYPRE_Int new_size = hypre_ParCompGridPRowPtr(compGrid[level])[hypre_ParCompGridNumNodes(compGrid[level])];
            hypre_ParCompGridResize(compGrid[level], new_size, level != num_levels-1, 2, symmetric);
         }

         // Copy col ind and data into the CSR structure
         for (i = num_owned_nodes; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            for (j = hypre_ParCompGridPRowPtr(compGrid[level])[i]; j < hypre_ParCompGridPRowPtr(compGrid[level])[i+1]; j++)
            {
               hypre_ParCompGridPColInd(compGrid[level])[j] = temp_PColInd[level][i - num_owned_nodes][j - hypre_ParCompGridPRowPtr(compGrid[level])[i]];
               hypre_ParCompGridPData(compGrid[level])[j] = temp_PData[level][i - num_owned_nodes][j - hypre_ParCompGridPRowPtr(compGrid[level])[i]];
            }
         }
      }

      if (level != 0 && hypre_ParAMGDataRestriction(amg_data))
      {
         // Setup the row pointer (we stored the row sizes rather than pointer values as we unpacked)
         HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
         for (i = num_owned_nodes; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            hypre_ParCompGridRRowPtr(compGrid[level])[i+1] = hypre_ParCompGridRRowPtr(compGrid[level])[i] + hypre_ParCompGridRRowPtr(compGrid[level])[i+1];
         }

         // Make sure enough space is allocated for R
         if (hypre_ParCompGridRRowPtr(compGrid[level])[hypre_ParCompGridNumNodes(compGrid[level])] > hypre_ParCompGridRMemSize(compGrid[level]))
         {
            HYPRE_Int new_size = hypre_ParCompGridRRowPtr(compGrid[level])[hypre_ParCompGridNumNodes(compGrid[level])];
            hypre_ParCompGridResize(compGrid[level], new_size, level != num_levels-1, 3, symmetric);
         }

         // Copy col ind and data into the CSR structure
         for (i = num_owned_nodes; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            for (j = hypre_ParCompGridRRowPtr(compGrid[level])[i]; j < hypre_ParCompGridRRowPtr(compGrid[level])[i+1]; j++)
            {
               hypre_ParCompGridRColInd(compGrid[level])[j] = temp_RColInd[level][i - num_owned_nodes][j - hypre_ParCompGridRRowPtr(compGrid[level])[i]];
               hypre_ParCompGridRData(compGrid[level])[j] = temp_RData[level][i - num_owned_nodes][j - hypre_ParCompGridRRowPtr(compGrid[level])[i]];
            }
         }
      }
   }
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]) - num_owned_nodes; i++)
      {
         hypre_TFree(temp_PColInd[level][i], HYPRE_MEMORY_HOST);
         hypre_TFree(temp_PData[level][i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(temp_PColInd[level], HYPRE_MEMORY_HOST);
      hypre_TFree(temp_PData[level], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(temp_PColInd, HYPRE_MEMORY_HOST);
   hypre_TFree(temp_PData, HYPRE_MEMORY_HOST);
   if (hypre_ParAMGDataRestriction(amg_data))
   {
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]) - num_owned_nodes; i++)
         {
            hypre_TFree(temp_RColInd[level][i], HYPRE_MEMORY_HOST);
            hypre_TFree(temp_RData[level][i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(temp_RColInd[level], HYPRE_MEMORY_HOST);
         hypre_TFree(temp_RData[level], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(temp_RColInd, HYPRE_MEMORY_HOST);
      hypre_TFree(temp_RData, HYPRE_MEMORY_HOST);
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

   // !!! Debug
   // printf("Rank %d in TestCompGrids1New\n", myid);

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
               // !!! Debug: commented out
               // hypre_printf("Error: recursively add the region of ghost nodes\n");
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
            // !!! Debug: restrict to rank 0
            // if (myid == 0)
            {
               if (i < hypre_ParCompGridNumOwnedNodes(compGrid[level])) 
                  hypre_printf("Error: extra OWNED (i.e. test is broken) nodes present in comp grid, rank %d, level %d, i = %d, global index = %d\n", 
                     myid, level, i, i + hypre_ParCompGridFirstGlobalIndex(compGrid[level]));
               else
                  hypre_printf("Error: extra nonowned nodes present in comp grid, rank %d, level %d, i = %d, global index = %d\n", 
                     myid, level, i, hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])]);
            }
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
   for (level = 0; level < num_levels; level++) add_flag[level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level]), HYPRE_MEMORY_HOST);
   HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[current_level])[hypre_ParCompGridNumOwnedBlocks(compGrid[current_level])];
   for (i = 0; i < num_owned_nodes; i++) add_flag[current_level][i] = padding[current_level] + 1;

   // Serially generate comp grid from top down
   // Note that if nodes that should be present in the comp grid are not found, we will be alerted by the error message in RecursivelyBuildPsiComposite()
   for (level = current_level; level < num_levels; level++)
   {
      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         nodes_to_add = 0;

         // see whether we need coarse info on this level
         if (level != num_levels-1) need_coarse_info = 1;
         else need_coarse_info = 0;

         // Expand by the padding on this level and add coarse grid counterparts if applicable
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][i] == padding[level] + 1)
            {
               if (need_coarse_info) error_code = RecursivelyBuildPsiComposite(i, padding[level], compGrid[level], add_flag[level], add_flag[level+1], NULL, NULL, need_coarse_info, &nodes_to_add, padding[level+1], level);
               else error_code = RecursivelyBuildPsiComposite(i, padding[level], compGrid[level], add_flag[level], NULL, NULL, NULL, need_coarse_info, &nodes_to_add, 0, level);
               if (error_code)
               {
                  hypre_printf("Error: expand padding\n");
                  test_failed = 1;
               }
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
            if (add_flag[level][i] == num_ghost_layers + 1) error_code = RecursivelyBuildPsiComposite(i, num_ghost_layers, compGrid[level], add_flag[level], NULL, NULL, NULL, 0, NULL, 0, level);
            if (error_code)
            {
               hypre_printf("Error: recursively add the region of ghost nodes\n");
               test_failed = 1;
            }
         }
      }
      else break;

      // Check whether add_flag has any zeros (zeros indicate that we have extra nodes in the comp grid that don't belong) 
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
      {
         if (add_flag[level][i] == 0) 
         {
            test_failed = 1;
            if (myid == 0) hypre_printf("Error: extra nodes present in comp grid, rank %d, level %d, i = %d, global index = %d\n", myid, level, i, hypre_ParCompGridGlobalIndices(compGrid[level])[i]);
         }
      }

      // Check to make sure we have the correct identification of ghost nodes
      if (level != num_levels-1 && check_ghost_info)
      {
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) 
         {
            if (add_flag[level][i] < num_ghost_layers + 1 && hypre_ParCompGridRealDofMarker(compGrid[level])[i] != 0) 
            {
               test_failed = 1;
               hypre_printf("Error: dof that should have been marked as ghost was marked as real, rank %d, level %d, GID %d\n", myid, level, hypre_ParCompGridGlobalIndices(compGrid[level])[i]);
            }
            if (add_flag[level][i] > num_ghost_layers && hypre_ParCompGridRealDofMarker(compGrid[level])[i] == 0) 
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
TestCompGrids2(hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int num_levels, hypre_ParCSRMatrix **A, hypre_ParCSRMatrix **P, hypre_ParCSRMatrix **R)
{
   // TEST 2: See whether the dofs in the composite grid have the correct info.
   // Each processor in turn will broadcast out the info associate with its composite grids on each level.
   // The processors owning the original info will check to make sure their info matches the comp grid info that was broadcasted out.
   // This occurs for the matrix info (row pointer, column indices, and data for A and P) and the initial right-hand side 
   
   // Get MPI info
   HYPRE_Int myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   HYPRE_Int i,j;
   HYPRE_Int test_failed = 0;

   // For each processor and each level broadcast the global indices and matrix info out and check agains the owning procs
   HYPRE_Int proc;
   for (proc = 0; proc < num_procs; proc++)
   {
      HYPRE_Int level;
      for (level = 0; level < num_levels; level++)
      {
         // Broadcast the number of nodes and num non zeros for A, P, and R
         HYPRE_Int num_nodes = 0;
         HYPRE_Int num_coarse_nodes = 0;
         HYPRE_Int num_fine_nodes = 0;
         HYPRE_Int nnz_A = 0;
         HYPRE_Int nnz_P = 0;
         HYPRE_Int nnz_R = 0;
         HYPRE_Int sizes_buf[6];
         if (myid == proc) 
         {
            num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
            nnz_A = hypre_ParCompGridARowPtr(compGrid[level])[num_nodes];
            if (level != num_levels-1)
            {
               num_coarse_nodes = hypre_ParCompGridNumNodes(compGrid[level+1]);
               nnz_P = hypre_ParCompGridPRowPtr(compGrid[level])[num_nodes];
            }
            else
            {
               num_coarse_nodes = 0;
               nnz_P = 0;
            }
            if (level != 0 && hypre_ParCompGridRRowPtr(compGrid[level]))
            {
               num_fine_nodes = hypre_ParCompGridNumNodes(compGrid[level-1]);
               nnz_R = hypre_ParCompGridRRowPtr(compGrid[level])[num_nodes];
            }
            else
            {
               num_fine_nodes = 0;
               nnz_R = 0;
            }
            sizes_buf[0] = num_nodes;
            sizes_buf[1] = num_coarse_nodes;
            sizes_buf[2] = num_fine_nodes;
            sizes_buf[3] = nnz_A;
            sizes_buf[4] = nnz_P;
            sizes_buf[5] = nnz_R;
         }
         hypre_MPI_Bcast(sizes_buf, 6, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);
         num_nodes = sizes_buf[0];
         num_coarse_nodes = sizes_buf[1];
         num_fine_nodes = sizes_buf[2];
         nnz_A = sizes_buf[3];
         nnz_P = sizes_buf[4];
         nnz_R = sizes_buf[5];

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

         HYPRE_Int *fine_global_indices;
         HYPRE_Int *R_rowPtr;
         HYPRE_Int *R_colInd;
         HYPRE_Complex *R_data;
         if (level != 0 && hypre_ParCompGridRRowPtr(compGrid[level]))
         {
            // Broadcast the coarse global indices
            if (myid == proc) fine_global_indices = hypre_ParCompGridGlobalIndices(compGrid[level-1]);
            else fine_global_indices = hypre_CTAlloc(HYPRE_Int, num_fine_nodes, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(fine_global_indices, num_fine_nodes, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the R row ptr
            if (myid == proc) R_rowPtr = hypre_ParCompGridRRowPtr(compGrid[level]);
            else R_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nodes+1, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(R_rowPtr, num_nodes+1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the R column indices
            if (myid == proc) R_colInd = hypre_ParCompGridRColInd(compGrid[level]);
            else R_colInd = hypre_CTAlloc(HYPRE_Int, nnz_R, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(R_colInd, nnz_R, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

            // Broadcast the R data
            if (myid == proc) R_data = hypre_ParCompGridRData(compGrid[level]);
            else R_data = hypre_CTAlloc(HYPRE_Complex, nnz_R, HYPRE_MEMORY_HOST);
            hypre_MPI_Bcast(R_data, nnz_R, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);
         }

         // Now, each processors checks their owned info against the composite grid info
         HYPRE_Int proc_first_index = hypre_ParCSRMatrixFirstRowIndex(A[level]);
         HYPRE_Int proc_last_index = hypre_ParCSRMatrixLastRowIndex(A[level]);
         for (i = 0; i < num_nodes; i++)
         {
            if (global_indices[i] <= proc_last_index && global_indices[i] >= proc_first_index)
            {
               HYPRE_Int row_size;
               HYPRE_Int *row_col_ind;
               HYPRE_Complex *row_values;
               if (A_rowPtr[i+1] - A_rowPtr[i] > 0)
               {
                  hypre_ParCSRMatrixGetRow( A[level], global_indices[i], &row_size, &row_col_ind, &row_values );
                  if (row_size != A_rowPtr[i+1] - A_rowPtr[i])
                  {
                     hypre_printf("Error: proc %d has incorrect A row size at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
                  for (j = A_rowPtr[i]; j < A_rowPtr[i+1]; j++)
                  {
                     if (A_colInd[j] >= 0)
                     {
                        if (global_indices[ A_colInd[j] ] != row_col_ind[j - A_rowPtr[i]])
                        {
                           hypre_printf("Error: proc %d has incorrect A col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                           test_failed = 1;
                        }
                        if (A_data[j] != row_values[j - A_rowPtr[i]])
                        {
                           hypre_printf("Error: proc %d has incorrect A data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                           test_failed = 1;
                        }
                     }
                  }
                  hypre_ParCSRMatrixRestoreRow( A[level], global_indices[i], &row_size, &row_col_ind, &row_values );
               }
               if (level != num_levels-1)
               {
                  hypre_ParCSRMatrixGetRow( P[level], global_indices[i], &row_size, &row_col_ind, &row_values );
                  if (row_size != P_rowPtr[i+1] - P_rowPtr[i])
                  {
                     hypre_printf("Error: proc %d has incorrect P row size at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
                  for (j = P_rowPtr[i]; j < P_rowPtr[i+1]; j++)
                  {
                     if (P_colInd[j] >= 0)
                     {
                        if (coarse_global_indices[ P_colInd[j] ] != row_col_ind[j - P_rowPtr[i]])
                        {
                           hypre_printf("Error: proc %d has incorrect P col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                           test_failed = 1;
                        }
                        if (P_data[j] != row_values[j - P_rowPtr[i]])
                        {
                           hypre_printf("Error: proc %d has incorrect P data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                           test_failed = 1;
                        }
                     }
                  }
                  hypre_ParCSRMatrixRestoreRow( P[level], global_indices[i], &row_size, &row_col_ind, &row_values );
               }
               if (level != 0 && hypre_ParCompGridRRowPtr(compGrid[level]))
               {
                  hypre_ParCSRMatrixGetRow( R[level-1], global_indices[i], &row_size, &row_col_ind, &row_values );
                  if (row_size != R_rowPtr[i+1] - R_rowPtr[i])
                  {
                     hypre_printf("Error: proc %d has incorrect R row size at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                     test_failed = 1;
                  }
                  for (j = R_rowPtr[i]; j < R_rowPtr[i+1]; j++)
                  {
                     if (R_colInd[j] >= 0)
                     {
                        if (fine_global_indices[ R_colInd[j] ] != row_col_ind[j - R_rowPtr[i]])
                        {
                           hypre_printf("Error: proc %d has incorrect R col index at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                           test_failed = 1;
                        }
                        if (R_data[j] != row_values[j - R_rowPtr[i]])
                        {
                           hypre_printf("Error: proc %d has incorrect R data at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                           test_failed = 1;
                        }
                     }
                  }
                  hypre_ParCSRMatrixRestoreRow( R[level-1], global_indices[i], &row_size, &row_col_ind, &row_values );
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
            if (level != 0 && hypre_ParCompGridRRowPtr(compGrid[level]))
            {
               hypre_TFree(fine_global_indices, HYPRE_MEMORY_HOST);
               hypre_TFree(R_rowPtr, HYPRE_MEMORY_HOST);
               hypre_TFree(R_colInd, HYPRE_MEMORY_HOST);
               hypre_TFree(R_data, HYPRE_MEMORY_HOST);
            }
         }
      }
   }

   return test_failed;
}

HYPRE_Int
CheckCompGridLocalIndices(hypre_ParAMGData *amg_data)
{
   hypre_ParCompGrid **compGrid = hypre_ParAMGDataCompGrid(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int level;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   for (level = 0; level < num_levels; level++)
   {
      HYPRE_Int i;
      HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
      HYPRE_Int A_nnz = hypre_ParCompGridARowPtr(compGrid[level])[num_nodes];

      for (i = 0; i < A_nnz; i++)
      {
         HYPRE_Int local_index = hypre_ParCompGridAColInd(compGrid[level])[i];
         if (local_index >= 0)
         {
            if (hypre_ParCompGridGlobalIndices(compGrid[level])[local_index] != hypre_ParCompGridAGlobalColInd(compGrid[level])[i])
               hypre_printf("Error: A global/local indices don't agree\n");
         }
         // !!! Implement test to make sure that if -1 index is encountered, the global index is, in fact, not found in the comp grid
         // Would also be good to do something similar for P after we setup P
      }

      if (level != num_levels-1)
      for (i = 0; i < num_nodes; i++)
      {
         HYPRE_Int local_index = hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i];
         if (local_index >= 0)
         {
            if (hypre_ParCompGridGlobalIndices(compGrid[level+1])[local_index] != hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[i])
               hypre_printf("Error: coarse local/global indices don't agree, rank %d, level %d\ni = %d, local_index = %d, global index at local = %d, coarse global index = %d\n", 
                  myid, level, i, local_index, hypre_ParCompGridGlobalIndices(compGrid[level+1])[local_index], hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[i]);
         }
      }
   }

   return 0;
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
