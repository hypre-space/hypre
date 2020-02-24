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
#define DEBUGGING_MESSAGES 1 // if true, prints a bunch of messages to the screen to let you know where in the algorithm you are
#define ENABLE_AGGLOMERATION 0 // if true, enable coarse level processor agglomeration, which requires linking with parmetis

#if ENABLE_AGGLOMERATION
#include "parmetis.h"
#endif

HYPRE_Int
FindTransitionLevel(hypre_ParAMGData *amg_data);

HYPRE_Int*
AllgatherCoarseLevels(hypre_ParAMGData *amg_data, MPI_Comm comm, HYPRE_Int fine_level, HYPRE_Int coarse_level, HYPRE_Int *communication_cost, HYPRE_Int agglomerating_levels, HYPRE_Int *padding);

HYPRE_Int 
PackCoarseLevels(hypre_ParAMGData *amg_data, HYPRE_Int fine_level, HYPRE_Int coarse_level, HYPRE_Int **int_buffer, HYPRE_Complex **complex_buffer, HYPRE_Int *buffer_sizes, HYPRE_Int agglomerating_levels);

HYPRE_Int 
UnpackCoarseLevels(hypre_ParAMGData *amg_data, MPI_Comm comm, HYPRE_Int *recv_int_buffer, HYPRE_Complex *recv_complex_buffer, HYPRE_Int fine_level, HYPRE_Int coarse_level, HYPRE_Int agglomerating_levels, HYPRE_Int *padding);

HYPRE_Int
AgglomerateProcessors(hypre_ParAMGData *amg_data, hypre_ParCompGridCommPkg *initCopyCompGridCommPkg, HYPRE_Int *padding, HYPRE_Int level, HYPRE_Int partition_size, HYPRE_Int *communication_cost);


#if ENABLE_AGGLOMERATION
HYPRE_Int
GetPartition(HYPRE_Int partition_size, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level,MPI_Comm local_comm, MPI_Comm global_comm);
#endif

HYPRE_Int
GetNeighborPartitionInfo(hypre_ParAMGData *amg_data, hypre_ParCompGridCommPkg *initCopyCompGridCommPkg, MPI_Comm local_comm, HYPRE_Int *proc_starts, HYPRE_Int partition, HYPRE_Int current_level, HYPRE_Int transition_level, HYPRE_Int *communication_cost);

HYPRE_Int
AllgatherCommunicationInfo(hypre_ParAMGData *amg_data, HYPRE_Int level, MPI_Comm comm,
   HYPRE_Int num_comm_partitions,
   HYPRE_Int *comm_partitions,
   HYPRE_Int **comm_partition_ranks,
   HYPRE_Int *comm_partition_num_send_elmts,
   HYPRE_Int **comm_partition_send_elmts,
   HYPRE_Int **comm_partition_ghost_marker,
   HYPRE_Int *proc_offsets,
   HYPRE_Int max_num_comm_procs,
   HYPRE_Int *communication_cost);

HYPRE_Int*
PackSendBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, HYPRE_Int *send_flag_buffer_size, 
   HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes, HYPRE_Int partition, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int *padding, 
   HYPRE_Int num_ghost_layers, HYPRE_Int symmetric );

HYPRE_Int
RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int *add_flag_coarse, 
   HYPRE_Int *sort_map, HYPRE_Int *sort_map_coarse, HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding, HYPRE_Int level);

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int *recv_map_send_buffer, HYPRE_Int **recv_map, HYPRE_Int *num_recv_nodes, HYPRE_Int *recv_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, hypre_ParCompGrid **compGrid);

HYPRE_Int
UnpackSendFlagBuffer(HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, HYPRE_Int *send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
CommunicateRemainingMatrixInfo(hypre_ParAMGData* amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *communication_cost);

HYPRE_Int
FinalizeCompGridCommPkg(hypre_ParAMGData* amg_data, hypre_ParCompGridCommPkg *compGridCommPkg, hypre_ParCompGrid **compGrid);

HYPRE_Int
TestCompGrids1(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int transition_level, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int current_level, HYPRE_Int check_ghost_info);

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

   /* Data Structure variables */
 
   // level counters, indices, and parameters
   HYPRE_Int                  amgdd_start_level;
   HYPRE_Int                  num_levels;
   HYPRE_Int                  *padding;
   HYPRE_Int                  pad;
   HYPRE_Int                  variable_padding;
   HYPRE_Int                  num_ghost_layers;
   HYPRE_Int                  symmetric;
   HYPRE_Int                  use_transition_level;
   HYPRE_Int                  transition_level;
   HYPRE_Int                  level,i,j,l;
   HYPRE_Int                  *nodes_added_on_level;

   // info from amg setup
   hypre_ParCSRMatrix         **A_array;
   HYPRE_Int                  *proc_first_index, *proc_last_index;

   // composite grids
   hypre_ParCompGrid          **compGrid;

   // info needed for later composite grid communication
   hypre_ParCompGridCommPkg   *compGridCommPkg;
   HYPRE_Int                  num_send_procs, num_recv_procs, num_send_partitions;
   HYPRE_Int                  **send_buffer_size;
   HYPRE_Int                  **recv_buffer_size;
   HYPRE_Int                  ***num_send_nodes;
   HYPRE_Int                  ***num_recv_nodes;
   HYPRE_Int                  ****send_flag;
   HYPRE_Int                  ****recv_map;

   // temporary arrays used for communication during comp grid setup
   HYPRE_Int                  **send_buffer;
   HYPRE_Int                  **recv_buffer;
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
   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);
   if (amgdd_start_level >= num_levels) amgdd_start_level = num_levels-1;
   pad = hypre_ParAMGDataAMGDDPadding(amg_data);
   variable_padding = hypre_ParAMGDataAMGDDVariablePadding(amg_data);
   num_ghost_layers = hypre_ParAMGDataAMGDDNumGhostLayers(amg_data);
   HYPRE_Int symmetric_tmp = hypre_ParAMGDataSym(amg_data);
   symmetric = 0;
   use_transition_level = hypre_ParAMGDataAMGDDUseTransitionLevel(amg_data);

   // Allocate pointer for the composite grids
   compGrid = hypre_CTAlloc(hypre_ParCompGrid*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParAMGDataCompGrid(amg_data) = compGrid;

   // In the 1 processor case, just need to initialize the comp grids
   if (num_procs == 1)
   {
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         compGrid[level] = hypre_ParCompGridCreate();
         hypre_ParCompGridInitialize( amg_data, 0, level, symmetric );
      }
      hypre_ParCompGridFinalize(compGrid, NULL, amgdd_start_level, num_levels, hypre_ParAMGDataAMGDDUseRD(amg_data), verify_amgdd);
      hypre_ParCompGridSetupRelax(amg_data);
      return 0;
   }

   // Figure out padding on each level
   padding = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
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

   // get first and last global indices on each level for this proc
   proc_first_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   proc_last_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   nodes_added_on_level = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   for (level = 0; level < num_levels; level++)
   {
      proc_first_index[level] = hypre_ParVectorFirstIndex(hypre_ParAMGDataFArray(amg_data)[level]);
      proc_last_index[level] = hypre_ParVectorLastIndex(hypre_ParAMGDataFArray(amg_data)[level]);
   }

   // Allocate space for some variables that store info on each level
   compGridCommPkg = hypre_ParCompGridCommPkgCreate();
   hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendPartitions(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendPartitionRanks(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg) = hypre_CTAlloc(MPI_Comm, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgAggGlobalComms(compGridCommPkg) = hypre_CTAlloc(MPI_Comm, num_levels, HYPRE_MEMORY_HOST);
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
   HYPRE_Int *num_resizes = hypre_CTAlloc(HYPRE_Int, 3*num_levels, HYPRE_MEMORY_HOST);

   // assign compGrid and compGridCommPkg info to the amg structure
   hypre_ParAMGDataCompGridCommPkg(amg_data) = compGridCommPkg;


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
   }
   #endif


   // If needed, find the transition level and initialize comp grids above and below the transition as appropriate
   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[0]);
   if (use_transition_level)
   {
      transition_level = FindTransitionLevel(amg_data);

      hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg) = transition_level;
      // if (myid == 0) hypre_printf("transition_level = %d\n", transition_level);
      // Do all gather so that all processors own the coarsest levels of the AMG hierarchy
      AllgatherCoarseLevels(amg_data, hypre_MPI_COMM_WORLD, transition_level, num_levels, communication_cost, 0, padding);

      // Initialize composite grids above the transition level
      for (level = amgdd_start_level; level < transition_level; level++)
      {
         compGrid[level] = hypre_ParCompGridCreate();
         hypre_ParCompGridInitialize( amg_data, padding[level], level, symmetric );
      }
   }
   // Otherwise just initialize comp grid on all levels
   else
   {
      transition_level = num_levels;
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         compGrid[level] = hypre_ParCompGridCreate();
         hypre_ParCompGridInitialize( amg_data, padding[level], level, symmetric );
      }   
   }
   if (timers) hypre_EndTiming(timers[0]);


   #if DEBUG_COMP_GRID == 2
   for (level = 0; level < num_levels; level++)
   {
      hypre_sprintf(filename, "outputs/CompGrids/initCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrint( compGrid[level], filename, 0 );
   }
   #endif

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("  Done with transition level setup and comp grid initialization\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // On each level, setup a long distance commPkg that has communication info for distance (eta + numGhostLayers)
   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[1]);
   for (level = amgdd_start_level; level < transition_level; level++)
   {
      SetupNearestProcessorNeighbors(A_array[level], compGrid[level], compGridCommPkg, level, padding, num_ghost_layers, communication_cost);  
   }

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("  Done with SetupNearestProcessorNeighbors()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   HYPRE_Int agglomeration_max_num_levels = hypre_ParAMGDataAMGDDAgglomerationMaxNumLevels(amg_data);
   if (agglomeration_max_num_levels)
   {
      // Get the max stencil size on all levels
      HYPRE_Int *local_stencil = hypre_CTAlloc(HYPRE_Int, transition_level, HYPRE_MEMORY_HOST);
      for (level = 0; level < transition_level; level++)
      {
         for (i = 0; i < hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level]; i++)
         {
            for (j = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][i]; j < hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][i+1]; j++)
            {
               if (!hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level][j])
               {
                  local_stencil[level]++;
                  break;
               }
            }
         }
      }
      HYPRE_Int *global_stencil = hypre_CTAlloc(HYPRE_Int, transition_level, HYPRE_MEMORY_HOST);
      hypre_MPI_Allreduce(local_stencil, global_stencil, transition_level, HYPRE_MPI_INT, MPI_MAX, hypre_MPI_COMM_WORLD);
      if (communication_cost)
      {
         communication_cost[0] += log(num_procs)/log(2);
         communication_cost[1] += sizeof(HYPRE_Int)*(num_procs-1);
      }

      HYPRE_Int agglomeration_threshold = hypre_ParAMGDataAMGDDAgglomerationThreshold(amg_data);
      HYPRE_Int agglomeration_partition_size = hypre_ParAMGDataAMGDDAgglomerationPartitionSize(amg_data);
      HYPRE_Int agglomeration_num_levels = 0;
      hypre_ParCompGridCommPkg *initCopyCompGridCommPkg = hypre_ParCompGridCommPkgCopy(compGridCommPkg);
      for (level = 1; level < transition_level; level++)
      {
         // if (myid == 0) hypre_printf("Max stencil on level %d is %d\n", level, global_stencil[level]);
         if (agglomeration_num_levels == agglomeration_max_num_levels) break;
         if (global_stencil[level] > agglomeration_threshold*global_stencil[0])
         {
            #if DEBUGGING_MESSAGES
            if (myid == 0) hypre_printf("Agglomerating processors on level %d\n", level); 
            #endif
            AgglomerateProcessors(amg_data, initCopyCompGridCommPkg, padding, level, agglomeration_partition_size, communication_cost);
            agglomeration_num_levels++;

            // Update global stencil info !!! Make sure this is how you want to do it (as opposed to accounting for num sends in the case of unequal partition sizes)
            for (l = 0; l < transition_level; l++)
            {
               local_stencil[l] = 0;
               for (i = 0; i < hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[l]; i++)
               {
                  for (j = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[l][i]; j < hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[l][i+1]; j++)
                  {
                     if (!hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[l][j])
                     {
                        local_stencil[l]++;
                        break;
                     }
                  }
               }
            }
            hypre_MPI_Allreduce(local_stencil, global_stencil, transition_level, HYPRE_MPI_INT, MPI_MAX, hypre_MPI_COMM_WORLD);
            if (communication_cost)
            {
               communication_cost[0] += log(num_procs)/log(2);
               communication_cost[1] += sizeof(HYPRE_Int)*(num_procs-1);
            }
         }
      }
      hypre_ParCompGridCommPkgDestroy(initCopyCompGridCommPkg);

      hypre_TFree(local_stencil, HYPRE_MEMORY_HOST);
      hypre_TFree(global_stencil, HYPRE_MEMORY_HOST);
   }

   if (timers) hypre_EndTiming(timers[1]);

   // !!! Debug
   HYPRE_Int total_bin_search_count = 0;


   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("  Looping over levels\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   for (level = transition_level - 1; level >= amgdd_start_level; level--)
   {
      comm = hypre_ParCSRMatrixComm(A_array[level]);
      num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level];
      num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level];
      num_send_partitions = hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level];

      if ( num_send_procs || num_recv_procs ) // If there are any owned nodes on this level
      {
         // allocate space for the buffers, buffer sizes, requests and status, psiComposite_send, psiComposite_recv, send and recv maps
         requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         send_buffer = hypre_CTAlloc(HYPRE_Int*, num_send_partitions, HYPRE_MEMORY_HOST);
         send_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_send_partitions, HYPRE_MEMORY_HOST);
         recv_buffer = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         recv_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_recv_procs, HYPRE_MEMORY_HOST);

         send_flag[level] = hypre_CTAlloc(HYPRE_Int**, num_send_partitions, HYPRE_MEMORY_HOST);
         num_send_nodes[level] = hypre_CTAlloc(HYPRE_Int*, num_send_partitions, HYPRE_MEMORY_HOST);
         recv_map[level] = hypre_CTAlloc(HYPRE_Int**, num_recv_procs, HYPRE_MEMORY_HOST);
         num_recv_nodes[level] = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);

         send_flag_buffer = hypre_CTAlloc(HYPRE_Int*, num_send_partitions, HYPRE_MEMORY_HOST);
         send_flag_buffer_size = hypre_CTAlloc(HYPRE_Int, num_send_partitions, HYPRE_MEMORY_HOST);
         recv_map_send_buffer = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         recv_map_send_buffer_size = hypre_CTAlloc(HYPRE_Int, num_recv_procs, HYPRE_MEMORY_HOST);

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (timers) hypre_BeginTiming(timers[2]);

         // pack send buffers
         for (i = 0; i < num_send_partitions; i++)
         {
            send_flag[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_send_nodes[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
            send_buffer[i] = PackSendBuffer( compGrid, compGridCommPkg, &(send_buffer_size[level][i]), 
                                             &(send_flag_buffer_size[i]), send_flag, num_send_nodes, i, level, num_levels, padding, 
                                             num_ghost_layers, symmetric );
         }

         if (timers) hypre_EndTiming(timers[2]);

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
            HYPRE_Int buffer_index = hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level][i];
            hypre_MPI_Isend(&(send_buffer_size[level][buffer_index]), 1, HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]));
            if (communication_cost)
            {
               communication_cost[level*10 + 2]++;
               communication_cost[level*10 + 3] += sizeof(HYPRE_Int);
            }
         }

         // wait for all buffer sizes to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests, status );

         // free and reallocate space for the requests and status
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         // Communicate buffers 
         // allocate space for the receive buffers and post the receives
         for (i = 0; i < num_recv_procs; i++)
         {
            recv_buffer[i] = hypre_CTAlloc(HYPRE_Int, recv_buffer_size[level][i], HYPRE_MEMORY_HOST );
            hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_INT, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 1, comm, &(requests[request_counter++]));
         }

         // send the buffers
         for (i = 0; i < num_send_procs; i++)
         {
            HYPRE_Int buffer_index = hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level][i];
            hypre_MPI_Isend(send_buffer[buffer_index], send_buffer_size[level][buffer_index], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 1, comm, &(requests[request_counter++]));
            if (communication_cost)
            {
               communication_cost[level*10 + 2]++;
               communication_cost[level*10 + 3] += send_buffer_size[level][i]*sizeof(HYPRE_Int);
            }
         }

         // wait for buffers to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests, status );

         // free and reallocate space for the requests and status
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         if (timers) hypre_EndTiming(timers[3]);


         // unpack the buffers
         if (timers) hypre_BeginTiming(timers[4]);
         
         for (i = 0; i < num_recv_procs; i++)
         {

            recv_map[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_recv_nodes[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
            UnpackRecvBuffer(recv_buffer[i], compGrid, compGridCommPkg, 
               send_flag, num_send_nodes, 
               recv_map, num_recv_nodes, 
               &(recv_map_send_buffer_size[i]), level, num_levels, transition_level, nodes_added_on_level, i, num_resizes, symmetric);
            
            recv_map_send_buffer[i] = hypre_CTAlloc(HYPRE_Int, recv_map_send_buffer_size[i], HYPRE_MEMORY_HOST);
            PackRecvMapSendBuffer(recv_map_send_buffer[i], recv_map[level][i], num_recv_nodes[level][i], &(recv_buffer_size[level][i]), level, num_levels, compGrid);
         }
         
         if (timers) hypre_EndTiming(timers[4]);

         // Setup local indices for the composite grid
         if (timers) hypre_BeginTiming(timers[5]);

         total_bin_search_count += hypre_ParCompGridSetupLocalIndices(compGrid, nodes_added_on_level, amgdd_start_level, transition_level, symmetric_tmp);
         for (j = level; j < num_levels; j++) nodes_added_on_level[j] = 0;

         if (timers) hypre_EndTiming(timers[5]);

         // Communicate redundancy info 
         // post receives for send maps
         for (i = 0; i < num_send_procs; i++)
         {
            // !!! Check send_flag_buffer_size (make sure not overallocated)
            HYPRE_Int buffer_index = hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level][i];
            send_flag_buffer[buffer_index] = hypre_CTAlloc(HYPRE_Int, send_flag_buffer_size[buffer_index], HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv( send_flag_buffer[buffer_index], send_flag_buffer_size[buffer_index], HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][buffer_index], 2, comm, &(requests[request_counter++]));
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

         if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (timers) hypre_BeginTiming(timers[6]);  

         // wait for maps to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests, status );

         // unpack and setup the send flag arrays
         for (i = 0; i < num_send_partitions; i++)
         {
            UnpackSendFlagBuffer(send_flag_buffer[i], send_flag[level][i], num_send_nodes[level][i], &(send_buffer_size[level][i]), level, num_levels);
         }

         if (timers) hypre_EndTiming(timers[6]);

         // clean up memory for this level
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_send_partitions; i++)
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

      #if DEBUG_COMP_GRID
      CheckCompGridLocalIndices(amg_data);
      HYPRE_Int error_code;
      error_code = TestCompGrids1(compGrid, num_levels, transition_level, padding, num_ghost_layers, level, 1);
      if (error_code)
      {
         hypre_printf("TestCompGrids1 failed! Rank %d, level %d\n", myid, level);
      }
      #endif
   }


   // !!! Debug
   HYPRE_Int total_nnz = 0;
   for (level = 0; level < num_levels; level++) total_nnz += hypre_ParCompGridARowPtr(compGrid[level])[hypre_ParCompGridNumNodes(compGrid[level])];
   printf("rank %d, total total_bin_search_count = %d, percentage = %f\n", myid, total_bin_search_count, ((double)total_bin_search_count)/((double)total_nnz));

   #if DEBUG_COMP_GRID
   // Test whether comp grids have correct shape
   HYPRE_Int test_failed = 0;
   HYPRE_Int error_code;
   error_code = TestCompGrids1(compGrid, num_levels, transition_level, padding, num_ghost_layers, 0, 1);
   if (error_code)
   {
      hypre_printf("TestCompGrids1 failed!\n");
      test_failed = 1;
   }
   else hypre_printf("TestCompGrids1 success\n");
   #endif

   // store communication info in compGridCommPkg
   hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg) = send_buffer_size;
   hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg) = recv_buffer_size;
   hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg) = num_send_nodes;
   hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg) = num_recv_nodes;
   hypre_ParCompGridCommPkgSendFlag(compGridCommPkg) = send_flag;
   hypre_ParCompGridCommPkgRecvMap(compGridCommPkg) = recv_map;

   #if DEBUG_COMP_GRID
   CheckCompGridCommPkg(compGridCommPkg);
   #endif

   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[7]);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with FinalizeSendFlag()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif 

   // Communicate data for A and all info for P
   CommunicateRemainingMatrixInfo(amg_data, compGrid, compGridCommPkg, communication_cost);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with CommunicateRemainingMatrixInfo()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif 

   if (timers) hypre_EndTiming(timers[7]);

   if (use_barriers) hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (timers) hypre_BeginTiming(timers[5]);

   // Setup the local indices for P
   hypre_ParCompGridSetupLocalIndicesP(compGrid, amgdd_start_level, transition_level);

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
   error_code = TestCompGrids2(compGrid, compGridCommPkg, transition_level, hypre_ParAMGDataAArray(amg_data), hypre_ParAMGDataPArray(amg_data), hypre_ParAMGDataRArray(amg_data));
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
   hypre_ParCompGridFinalize(compGrid, compGridCommPkg, amgdd_start_level, transition_level, hypre_ParAMGDataAMGDDUseRD(amg_data), verify_amgdd);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with hypre_ParCompGridFinalize()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // Finalize the send flag and the recv
   FinalizeCompGridCommPkg(amg_data, compGridCommPkg, compGrid);
   
   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("All ranks: done with FinalizeCompGridCommPkg()\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // Setup extra info for specific relaxation methods
   hypre_ParCompGridSetupRelax(amg_data);

   if (timers) hypre_EndTiming(timers[8]);

   // Count up the cost for subsequent residual communications
   if (communication_cost)
   {
      for (level = amgdd_start_level; level < transition_level; level++)
      {
         communication_cost[level*10 + 4] += hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level];
         communication_cost[level*10 + 5] += hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][ hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level] ]*sizeof(HYPRE_Complex);

         if (hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[level])
         {
            HYPRE_Int local_myid, local_num_procs;
            hypre_MPI_Comm_rank(hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[level], &local_myid);
            hypre_MPI_Comm_size(hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[level], &local_num_procs);
            communication_cost[level*10 + 4] += log(local_num_procs)/log(2);
            for (i = level; i < transition_level; i++)
            {
               if (i > level && hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[i]) break;
               communication_cost[i*10 + 5] += sizeof(HYPRE_Complex)*(local_num_procs-1)*(hypre_ParCompGridOwnedBlockStarts(compGrid[i])[local_myid+1] - hypre_ParCompGridOwnedBlockStarts(compGrid[i])[local_myid]);
            }
         }
      }
   }

   // Cleanup memory
   hypre_TFree(num_resizes, HYPRE_MEMORY_HOST);
   hypre_TFree(nodes_added_on_level, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_first_index, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_last_index, HYPRE_MEMORY_HOST);

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

HYPRE_Int
FindTransitionLevel(hypre_ParAMGData *amg_data)
{
   HYPRE_Int i;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int use_transition_level = hypre_ParAMGDataAMGDDUseTransitionLevel(amg_data);
   HYPRE_Int local_transition = num_levels-1;
   HYPRE_Int global_transition;

   // Transition level set as a prescribed level
   if (use_transition_level > 0) global_transition = use_transition_level;
   
   // Transition level is the finest level such that the global grid size stored is less than the amount of data sent on the fine grid
   if (use_transition_level < 0)
   {
      HYPRE_Int fine_grid_num_sends = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(hypre_ParAMGDataAArray(amg_data)[0]));
      for (i = hypre_ParAMGDataAMGDDStartLevel(amg_data); i < num_levels; i++)
      {
         if ( hypre_ParCSRMatrixGlobalNumRows(hypre_ParAMGDataAArray(amg_data)[i]) < fine_grid_num_sends )
         {
            local_transition = i;
            break;
         }
      }
      hypre_MPI_Allreduce(&local_transition, &global_transition, 1, HYPRE_MPI_INT, MPI_MAX, hypre_MPI_COMM_WORLD);
   }

   return global_transition;
}

HYPRE_Int*
AllgatherCoarseLevels(hypre_ParAMGData *amg_data, MPI_Comm comm, HYPRE_Int fine_level, HYPRE_Int coarse_level, HYPRE_Int *communication_cost, HYPRE_Int agglomerating_levels, HYPRE_Int *padding)
{
   // Get MPI comm size
   HYPRE_Int   num_procs;
   hypre_MPI_Comm_size(comm, &num_procs);

   // Pack up the buffer containing comp grid info that will be gathered onto all procs and get its size
   HYPRE_Int *buffer_sizes = hypre_CTAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
   HYPRE_Int *send_int_buffer;
   HYPRE_Complex *send_complex_buffer;
   PackCoarseLevels(amg_data, fine_level, coarse_level, &send_int_buffer, &send_complex_buffer, buffer_sizes, agglomerating_levels);

   // Do an allreduce to find the maximum buffer size
   HYPRE_Int *global_buffer_sizes = hypre_CTAlloc(HYPRE_Int, 2*num_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Allgather(buffer_sizes, 2, HYPRE_MPI_INT, global_buffer_sizes, 2, HYPRE_MPI_INT, comm);
   if (communication_cost)
   {
      communication_cost[fine_level*10 + 2] += log(num_procs)/log(2);
      communication_cost[fine_level*10 + 3] += 2*sizeof(HYPRE_Int)*(num_procs-1);
   }

   // Setup sizes and displacements for following Allgatherv calls
   HYPRE_Int *int_sizes = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
   HYPRE_Int *int_disps = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
   HYPRE_Int *complex_sizes = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
   HYPRE_Int *complex_disps = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);

   HYPRE_Int i;
   for (i = 0; i < num_procs; i++)
   {
      int_sizes[i] = global_buffer_sizes[2*i];
      complex_sizes[i] = global_buffer_sizes[2*i+1];
      if (i > 0)
      {
         int_disps[i] = int_disps[i-1] + int_sizes[i-1];
         complex_disps[i] = complex_disps[i-1] + complex_sizes[i-1];
      }
   }

   // Allocate a buffer to recieve comp grid info from all procs and do the allgather
   HYPRE_Int *recv_int_buffer = hypre_CTAlloc(HYPRE_Int, int_disps[num_procs-1] + int_sizes[num_procs-1], HYPRE_MEMORY_HOST);
   HYPRE_Complex *recv_complex_buffer = hypre_CTAlloc(HYPRE_Complex, complex_disps[num_procs-1] + complex_sizes[num_procs-1], HYPRE_MEMORY_HOST);

   hypre_MPI_Allgatherv(send_int_buffer, buffer_sizes[0], HYPRE_MPI_INT, recv_int_buffer, int_sizes, int_disps, HYPRE_MPI_INT, comm);
   hypre_MPI_Allgatherv(send_complex_buffer, buffer_sizes[1], HYPRE_MPI_COMPLEX, recv_complex_buffer, complex_sizes, complex_disps, HYPRE_MPI_COMPLEX, comm);
   if (communication_cost)
   {
      communication_cost[fine_level*10 + 2] += 2*log(num_procs)/log(2);
      communication_cost[fine_level*10 + 3] += (num_procs-1)*sizeof(HYPRE_Int)*buffer_sizes[0] + (num_procs-1)*sizeof(HYPRE_Complex)*buffer_sizes[1];
      communication_cost[fine_level*10 + 4] += log(num_procs)/log(2);
      communication_cost[fine_level*10 + 5] += (num_procs-1)*sizeof(HYPRE_Complex)*hypre_ParCSRMatrixNumRows(hypre_ParAMGDataAArray(amg_data)[fine_level]);
   }

   // Unpack the recv buffer and generate the comp grid structures for the coarse levels
   UnpackCoarseLevels(amg_data, comm, recv_int_buffer, recv_complex_buffer, fine_level, coarse_level, agglomerating_levels, padding);

   // Get processor starts info
   HYPRE_Int *proc_starts;
   if (agglomerating_levels)
   {
      HYPRE_Int num_comm_levels = coarse_level - fine_level;
      HYPRE_Int level;
      proc_starts = hypre_CTAlloc(HYPRE_Int, 2*num_procs*num_comm_levels, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_procs; i++)
      {
         HYPRE_Int offset = int_disps[i];
         for (level = fine_level; level < coarse_level; level++)
         {
            HYPRE_Int first_index = recv_int_buffer[offset++];
            HYPRE_Int last_index = recv_int_buffer[offset++];
            HYPRE_Int num_nodes = last_index - first_index + 1;
            proc_starts[2*(i*num_comm_levels + level - fine_level)] = first_index;
            proc_starts[2*(i*num_comm_levels + level - fine_level) + 1] = last_index;


            HYPRE_Int A_nnz = recv_int_buffer[offset++];
            HYPRE_Int P_nnz;
            if (level != hypre_ParAMGDataNumLevels(amg_data)-1) P_nnz = recv_int_buffer[offset++];
            if (agglomerating_levels && level != coarse_level-1) offset += num_nodes;
            offset += num_nodes + A_nnz;
            if (level != hypre_ParAMGDataNumLevels(amg_data)-1) offset += num_nodes + P_nnz;
         }
      }
   }

   // Clean up memory
   hypre_TFree(int_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(int_disps, HYPRE_MEMORY_HOST);
   hypre_TFree(complex_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(complex_disps, HYPRE_MEMORY_HOST);
   hypre_TFree(global_buffer_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(buffer_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(send_int_buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(send_complex_buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_int_buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_complex_buffer, HYPRE_MEMORY_HOST);

   if (agglomerating_levels)
   {
      return proc_starts;
   }
   else 
   {
      return NULL;
   }
}

HYPRE_Int
PackCoarseLevels(hypre_ParAMGData *amg_data, HYPRE_Int fine_level, HYPRE_Int coarse_level, HYPRE_Int **int_buffer, HYPRE_Complex **complex_buffer, HYPRE_Int *buffer_sizes, HYPRE_Int agglomerating_levels)
{
   // The buffers will have the following forms:
   // int_buffer = [ [level], (fine_level)           level = [ global index start,
   //                [level],                                  global index finish,
   //                [level],                                  A nnz,
   //                ...    ,                                  P nnz,
   //                [level] ] (coarse_level - 1)              [coarse global indices OR cf marker]
   //                                                          [A row sizes]
   //                                                          [A col indices]
   //                                                          [P row sizes]
   //                                                          [P col indices] ]
   //
   // complex_buffer = [ [level], (fine_level)       level = [ [A data],
   //                    [level],                              [P data] ]
   //                    ...    ,
   //                    [level] ] (coarse_level - 1)
   
   // Count up how large the buffers will be
   HYPRE_Int level,i,j;
   buffer_sizes[0] = 0;
   buffer_sizes[1] = 0;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   for (level = fine_level; level < coarse_level; level++)
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
      buffer_sizes[0] += num_nodes + A_nnz;
      if (level != coarse_level-1) buffer_sizes[0] += num_nodes;
      if (level != num_levels-1) buffer_sizes[0] += 1 + num_nodes + P_nnz;
      buffer_sizes[1] += A_nnz + P_nnz;
   }

   // Allocate and pack the buffer
   (*int_buffer) = hypre_CTAlloc(HYPRE_Int, buffer_sizes[0], HYPRE_MEMORY_HOST);
   (*complex_buffer) = hypre_CTAlloc(HYPRE_Complex, buffer_sizes[1], HYPRE_MEMORY_HOST);
   HYPRE_Int int_cnt = 0;
   HYPRE_Int complex_cnt = 0;
   for (level = fine_level; level < coarse_level; level++)
   {
      // Get num nodes and num non zeros for matrices on this level
      hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(amg_data)[level];
      hypre_ParCSRMatrix *P;
      if (level != num_levels-1) P = hypre_ParAMGDataPArray(amg_data)[level];
      HYPRE_Int num_nodes = hypre_ParCSRMatrixNumRows(A);
      HYPRE_Int A_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(A) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(A) );
      HYPRE_Int P_nnz = 0;
      if (level != num_levels-1) P_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(P) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(P) ); 
      HYPRE_Int first_index = hypre_ParVectorFirstIndex(hypre_ParAMGDataUArray(amg_data)[level]);
      HYPRE_Int last_index = hypre_ParVectorLastIndex(hypre_ParAMGDataUArray(amg_data)[level]);
      // Save the header data for this level
      (*int_buffer)[int_cnt++] = first_index;
      (*int_buffer)[int_cnt++] = last_index;
      (*int_buffer)[int_cnt++] = A_nnz;
      if (level != num_levels-1) (*int_buffer)[int_cnt++] = P_nnz;
      if (level != coarse_level-1)
      {
         // If desired, pack the coarse index info
         if (agglomerating_levels && !hypre_ParAMGDataCompGrid(amg_data)[level]) hypre_printf("Error: in order to send coarse indices, need to setup comp grid first.\n");
         else if (agglomerating_levels)
         {
            hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];
            for (i = 0; i < num_nodes; i++) (*int_buffer)[int_cnt++] = hypre_ParCompGridCoarseGlobalIndices(compGrid)[i];
         }
         else
         {
            for (i = 0; i < num_nodes; i++) (*int_buffer)[int_cnt++] = hypre_ParAMGDataCFMarkerArray(amg_data)[level][i];
         }
      }
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
UnpackCoarseLevels(hypre_ParAMGData *amg_data, MPI_Comm comm, HYPRE_Int *recv_int_buffer, HYPRE_Complex *recv_complex_buffer, HYPRE_Int fine_level, HYPRE_Int coarse_level, HYPRE_Int agglomerating_levels, HYPRE_Int *padding)
{
   // Get MPI comm size
   HYPRE_Int   num_procs;
   hypre_MPI_Comm_size(comm, &num_procs);

   // Loop over levels and generate new global comp grids 
   HYPRE_Int level,proc,i;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

   // Get the global number of nodes and number of nonzeros for the matrices (read in from buffer)
   HYPRE_Int *transition_res_recv_sizes;
   HYPRE_Int *transition_res_recv_disps;
   if (!agglomerating_levels)
   {
      transition_res_recv_sizes = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
      transition_res_recv_disps = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
   }
   HYPRE_Int *global_num_nodes = hypre_CTAlloc(HYPRE_Int, coarse_level - fine_level, HYPRE_MEMORY_HOST);
   HYPRE_Int *global_A_nnz = hypre_CTAlloc(HYPRE_Int, coarse_level - fine_level, HYPRE_MEMORY_HOST);
   HYPRE_Int *global_P_nnz = hypre_CTAlloc(HYPRE_Int, coarse_level - fine_level, HYPRE_MEMORY_HOST);
   HYPRE_Int int_cnt = 0;
   for (proc = 0; proc < num_procs; proc++)
   {
      for (level = fine_level; level < coarse_level; level++)
      {
         // Read header info for this proc
         HYPRE_Int first_index = recv_int_buffer[int_cnt++];
         HYPRE_Int last_index = recv_int_buffer[int_cnt++];
         HYPRE_Int num_nodes = last_index - first_index + 1;
         if (level == fine_level && !agglomerating_levels)
         {
            transition_res_recv_sizes[proc] = num_nodes;
            if (proc > 0) transition_res_recv_disps[proc] = transition_res_recv_disps[proc-1] + transition_res_recv_sizes[proc-1];
         }
         HYPRE_Int A_nnz = recv_int_buffer[int_cnt++];
         HYPRE_Int P_nnz = 0;
         if (level != num_levels-1) P_nnz = recv_int_buffer[int_cnt++];

         // Add the global totals
         global_num_nodes[level - fine_level] += num_nodes;
         global_A_nnz[level - fine_level] += A_nnz;
         if (level != num_levels-1) global_P_nnz[level - fine_level] += P_nnz;

         // Increment counter appropriately
         if (level != coarse_level-1) int_cnt += num_nodes;
         int_cnt += num_nodes + A_nnz;
         if (level != num_levels-1) int_cnt += num_nodes + P_nnz;
      }
   }

   // Create and allocate the comp grids
   for (level = fine_level; level < coarse_level; level++)
   {
      hypre_ParCompGrid *compGrid = hypre_ParCompGridCreate();
      if (hypre_ParAMGDataCompGrid(amg_data)[level]) hypre_ParCompGridDestroy(hypre_ParAMGDataCompGrid(amg_data)[level]);

      hypre_ParAMGDataCompGrid(amg_data)[level] = compGrid;
      if (agglomerating_levels)
      {
         HYPRE_Int mem_size = global_num_nodes[level - fine_level] + 2 * (padding[level] + hypre_ParAMGDataAMGDDNumGhostLayers(amg_data)) * hypre_CSRMatrixNumCols( hypre_ParCSRMatrixOffd( hypre_ParAMGDataAArray(amg_data)[level] ) );
         if (level != coarse_level-1) hypre_ParCompGridSetSize(compGrid, global_num_nodes[level - fine_level], mem_size, global_A_nnz[level - fine_level], global_P_nnz[level - fine_level], 2);
         else hypre_ParCompGridSetSize(compGrid, global_num_nodes[level - fine_level], mem_size, global_A_nnz[level - fine_level], global_P_nnz[level - fine_level], 1);
         hypre_ParCompGridNumOwnedBlocks(compGrid) = num_procs;
         if (hypre_ParCompGridOwnedBlockStarts(compGrid)) hypre_TFree(hypre_ParCompGridOwnedBlockStarts(compGrid), HYPRE_MEMORY_HOST);
         hypre_ParCompGridOwnedBlockStarts(compGrid) = hypre_CTAlloc(HYPRE_Int, num_procs+1, HYPRE_MEMORY_HOST);
      }
      else
      {
         hypre_ParCompGridSetSize(compGrid, global_num_nodes[level - fine_level], global_num_nodes[level - fine_level], global_A_nnz[level - fine_level], global_P_nnz[level - fine_level], 0);
         hypre_ParCompGridCFMarkerArray(compGrid) = hypre_CTAlloc(HYPRE_Int, global_num_nodes[level - fine_level], HYPRE_MEMORY_SHARED);
         hypre_ParCompGridNumOwnedBlocks(compGrid) = 1;
         if (hypre_ParCompGridOwnedBlockStarts(compGrid)) hypre_TFree(hypre_ParCompGridOwnedBlockStarts(compGrid), HYPRE_MEMORY_HOST);
         hypre_ParCompGridOwnedBlockStarts(compGrid) = hypre_CTAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
         hypre_ParCompGridOwnedBlockStarts(compGrid)[0] = 0;
         hypre_ParCompGridOwnedBlockStarts(compGrid)[1] = hypre_ParCSRMatrixNumRows(hypre_ParAMGDataAArray(amg_data)[level]);   
      }
      hypre_ParCompGridNumRealNodes(compGrid) = global_num_nodes[level - fine_level];
   }

   // Now get matrix info from all processors from recv_buffer
   HYPRE_Int *globalRowCnt_start = hypre_CTAlloc(HYPRE_Int, coarse_level - fine_level, HYPRE_MEMORY_HOST);
   HYPRE_Int *globalANnzCnt_start = hypre_CTAlloc(HYPRE_Int, coarse_level - fine_level, HYPRE_MEMORY_HOST);
   HYPRE_Int *globalPNnzCnt_start = hypre_CTAlloc(HYPRE_Int, coarse_level - fine_level, HYPRE_MEMORY_HOST);
   int_cnt = 0;
   HYPRE_Int complex_cnt = 0;
   for (proc = 0; proc < num_procs; proc++)
   {
      for (level = fine_level; level < coarse_level; level++)
      {
         // Get the comp grid
         hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];

         // Set the counters appropriately
         HYPRE_Int globalRowCnt = globalRowCnt_start[level - fine_level];
         HYPRE_Int globalANnzCnt = globalANnzCnt_start[level - fine_level];
         HYPRE_Int globalPNnzCnt = globalPNnzCnt_start[level - fine_level];

         // If this is the first processor read in, initialize the row pointers
         if (proc == 0 && global_num_nodes[level - fine_level])
         {
            hypre_ParCompGridARowPtr(compGrid)[0] = 0;
            if (level != num_levels-1)
            {
               hypre_ParCompGridPRowPtr(compGrid)[0] = 0;
            }
            globalRowCnt_start[level - fine_level] = 1;
            globalRowCnt = 1;
         }

         // Read header info for this proc
         HYPRE_Int first_index = recv_int_buffer[int_cnt++];
         HYPRE_Int last_index = recv_int_buffer[int_cnt++];
         HYPRE_Int num_nodes = last_index - first_index + 1;
         HYPRE_Int A_nnz = recv_int_buffer[int_cnt++];
         HYPRE_Int P_nnz = 0;
         if (level != num_levels-1) P_nnz = recv_int_buffer[int_cnt++];

         // If setting up full composite grid information
         if (agglomerating_levels)
         {
            hypre_ParCompGridOwnedBlockStarts(compGrid)[proc+1] = hypre_ParCompGridOwnedBlockStarts(compGrid)[proc] + num_nodes;
            
            // Setup the global indices
            for (i = 0; i < num_nodes; i++) hypre_ParCompGridGlobalIndices(compGrid)[globalRowCnt++ - 1] = i + first_index;
            globalRowCnt = globalRowCnt_start[level - fine_level];
         }

            // If necessary, read in coarse grid indices
         if (level != coarse_level-1)
         {
            if (agglomerating_levels)
            {
               for (i = 0; i < num_nodes; i++) hypre_ParCompGridCoarseGlobalIndices(compGrid)[globalRowCnt++ - 1] = recv_int_buffer[int_cnt++];
              globalRowCnt = globalRowCnt_start[level - fine_level];
            }
            else
            {
               for (i = 0; i < num_nodes; i++) if (recv_int_buffer[int_cnt++] == 1) hypre_ParCompGridCFMarkerArray(compGrid)[globalRowCnt++ - 1] = 1;
               globalRowCnt = globalRowCnt_start[level - fine_level];
            }
         }

         // Read in A row sizes and update row ptr
         for (i = 0; i < num_nodes; i++)
         {
            hypre_ParCompGridARowPtr(compGrid)[globalRowCnt] = hypre_ParCompGridARowPtr(compGrid)[globalRowCnt-1] + recv_int_buffer[int_cnt++];
            globalRowCnt++;
         }
         // Read in A col indices
         if (agglomerating_levels)
         {
            for (i = 0; i < A_nnz; i++)
            {
               hypre_ParCompGridAGlobalColInd(compGrid)[globalANnzCnt++] = recv_int_buffer[int_cnt++];
            }
         }
         else
         {
            for (i = 0; i < A_nnz; i++)
            {
               hypre_ParCompGridAColInd(compGrid)[globalANnzCnt++] = recv_int_buffer[int_cnt++];
            }
         }
         globalANnzCnt = globalANnzCnt_start[level - fine_level];
         // Read in A values
         for (i = 0; i < A_nnz; i++)
         {
            hypre_ParCompGridAData(compGrid)[globalANnzCnt++] = recv_complex_buffer[complex_cnt++];
         }

         if (level != num_levels-1)
         {
            // Read in P row sizes and update row ptr
            globalRowCnt = globalRowCnt_start[level - fine_level];

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
            globalPNnzCnt = globalPNnzCnt_start[level - fine_level];
            // Read in A values
            for (i = 0; i < P_nnz; i++)
            {
               hypre_ParCompGridPData(compGrid)[globalPNnzCnt++] = recv_complex_buffer[complex_cnt++];
            }
         }

         // Update counters 
         globalRowCnt_start[level - fine_level] = globalRowCnt;
         globalANnzCnt_start[level - fine_level] = globalANnzCnt;
         globalPNnzCnt_start[level - fine_level] = globalPNnzCnt;
      }
   }

   if (!agglomerating_levels)
   {
      hypre_ParCompGridCommPkgTransitionResRecvDisps(hypre_ParAMGDataCompGridCommPkg(amg_data)) = transition_res_recv_disps;
      hypre_ParCompGridCommPkgTransitionResRecvSizes(hypre_ParAMGDataCompGridCommPkg(amg_data)) = transition_res_recv_sizes;
   }

   hypre_TFree(global_num_nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(global_A_nnz, HYPRE_MEMORY_HOST);
   hypre_TFree(global_P_nnz, HYPRE_MEMORY_HOST);
   hypre_TFree(globalRowCnt_start, HYPRE_MEMORY_HOST);
   hypre_TFree(globalANnzCnt_start, HYPRE_MEMORY_HOST);
   hypre_TFree(globalPNnzCnt_start, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Int
AgglomerateProcessors(hypre_ParAMGData *amg_data, hypre_ParCompGridCommPkg *initCopyCompGridCommPkg, HYPRE_Int *padding, HYPRE_Int current_level, HYPRE_Int partition_size, HYPRE_Int *communication_cost)
{
   #if ENABLE_AGGLOMERATION
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   hypre_ParCompGridCommPkg *compGridCommPkg = hypre_ParAMGDataCompGridCommPkg(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg);
   if (transition_level < 0) transition_level = hypre_ParAMGDataNumLevels(amg_data);
   MPI_Comm previous_local_comm, previous_global_comm;
   HYPRE_Int level;
   for (level = current_level; level >=0; level--)
   {
      if (hypre_ParCompGridCommPkgAggGlobalComms(compGridCommPkg)[level])
      {
         previous_local_comm = hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[level];
         previous_global_comm = hypre_ParCompGridCommPkgAggGlobalComms(compGridCommPkg)[level];
         break;
      }
      if (level == 0)
      {
         previous_local_comm = MPI_COMM_SELF;
         previous_global_comm = hypre_MPI_COMM_WORLD;
      }
   }

   // Get the partitioning of the communication graph
   HYPRE_Int partition;
   partition = GetPartition(partition_size, compGridCommPkg, current_level, previous_local_comm, previous_global_comm);

   // Split the old communicator
   MPI_Comm local_comm;
   hypre_MPI_Comm_split(hypre_MPI_COMM_WORLD, partition, myid, &local_comm);
   hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[current_level] = local_comm;
   HYPRE_Int local_myid;
   hypre_MPI_Comm_rank(local_comm, &local_myid);
   MPI_Comm global_sub_comm;
   HYPRE_Int color;
   if (local_myid == 0) color = 0;
   else color = MPI_UNDEFINED;
   hypre_MPI_Comm_split(hypre_MPI_COMM_WORLD, color, partition, &global_sub_comm);
   hypre_ParCompGridCommPkgAggGlobalComms(compGridCommPkg)[current_level] = global_sub_comm;

   // Allgather grid info inside the local communicator
   HYPRE_Int *proc_starts = AllgatherCoarseLevels(amg_data, local_comm, current_level, transition_level, communication_cost, 1, padding);

   // Setup local indices 
   HYPRE_Int num_agg_levels = transition_level - current_level;
   HYPRE_Int i, proc, num_procs;
   hypre_MPI_Comm_size(local_comm, &num_procs);

   for (level = current_level; level < transition_level; level++)
   {
      hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];
      HYPRE_Int A_nnz = hypre_ParCompGridARowPtr(compGrid)[hypre_ParCompGridNumNodes(compGrid)];
      for (i = 0; i < A_nnz; i++)
      {
         HYPRE_Int global_index = hypre_ParCompGridAGlobalColInd(compGrid)[i];
         hypre_ParCompGridAColInd(compGrid)[i] = -global_index-1;
         HYPRE_Int offset = 0;
         for (proc = 0; proc < num_procs; proc++)
         {
            if (global_index >= proc_starts[2*(proc*num_agg_levels + level - current_level)] && global_index <= proc_starts[2*(proc*num_agg_levels + level - current_level) + 1])
            {
               hypre_ParCompGridAColInd(compGrid)[i] = offset + global_index - proc_starts[2*(proc*num_agg_levels + level - current_level)]; // !!! Check
            }
            offset += proc_starts[2*(proc*num_agg_levels + level - current_level) + 1] - proc_starts[2*(proc*num_agg_levels + level - current_level)] + 1;
         }
      }
   }
   for (level = current_level-1; level < transition_level-1; level++)
   {
      hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];
      HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid);
      for (i = 0; i < num_nodes; i++)
      {
         HYPRE_Int coarse_global_index = hypre_ParCompGridCoarseGlobalIndices(compGrid)[i];
         if (coarse_global_index < 0) hypre_ParCompGridCoarseLocalIndices(compGrid)[i] = -1;
         else
         {
            HYPRE_Int offset = 0;
            for (proc = 0; proc < num_procs; proc++)
            {
               if (coarse_global_index >= proc_starts[2*(proc*num_agg_levels + level + 1 - current_level)] && coarse_global_index <= proc_starts[2*(proc*num_agg_levels + level + 1 - current_level) + 1])
               {
                  hypre_ParCompGridCoarseLocalIndices(compGrid)[i] = offset + coarse_global_index - proc_starts[2*(proc*num_agg_levels + level + 1 - current_level)]; // !!! Check
               }
               offset += proc_starts[2*(proc*num_agg_levels + level + 1 - current_level) + 1] - proc_starts[2*(proc*num_agg_levels + level + 1 - current_level)] + 1;
            }
         }
      }
   }

   // Do neighbor communication to determine partition info for neighbors
   GetNeighborPartitionInfo(amg_data, initCopyCompGridCommPkg, local_comm, proc_starts, partition, current_level, transition_level, communication_cost);
   hypre_TFree(proc_starts, HYPRE_MEMORY_HOST);

   return 0;
   #else
   printf("Need to enable processor agglomeration.\n");
   return 0;
   #endif
}

#if ENABLE_AGGLOMERATION
HYPRE_Int 
GetPartition(HYPRE_Int partition_size, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, MPI_Comm local_comm, MPI_Comm global_comm)
{
   HYPRE_Int new_partition;

   if (global_comm != MPI_COMM_NULL)
   {

      HYPRE_Int num_old_partitions;
      hypre_MPI_Comm_size(global_comm, &num_old_partitions);
      HYPRE_Int i;

      // NOTE: size of the sub communicator should be the number of existing partitions and the ranks should be the partition IDs
      // We partition the communication graph among partitions
      // The vertices are the partitions (i.e. one vertex per rank in the communicator) with weight (???) !!! do I want the size of the partition? For now, no weight. 
      // The edges are communication connections between the partitions with weights (???) !!! do I want communication volume? For now, no weight. 
      idx_t *vtxdist = (idx_t*) calloc(num_old_partitions+1, sizeof(idx_t));
      for (i = 0; i < num_old_partitions; i++) vtxdist[i+1] = i+1;
      idx_t *xadj = (idx_t*) calloc(2, sizeof(idx_t));
      xadj[1] = hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level];
      idx_t *adjncy = (idx_t*) calloc(hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level], sizeof(idx_t));
      for (i = 0; i < hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level]; i++) adjncy[i] = hypre_ParCompGridCommPkgSendPartitions(compGridCommPkg)[level][i];
      idx_t *vwgt = NULL;
      idx_t *adjwgt = NULL;
      idx_t wgtflag = 0;
      idx_t numflag = 0;
      idx_t ncon = 1;
      idx_t nparts = num_old_partitions/partition_size;
      real_t *tpwgts = (real_t*) calloc(ncon*nparts, sizeof(real_t));
      for (i = 0; i < ncon*nparts; i++) tpwgts[i] = 1.0/nparts;
      real_t *ubvec = (real_t*) calloc(ncon, sizeof(real_t));
      for (i = 0; i < ncon; i++) ubvec[i] = 1.05;
      idx_t *options = (idx_t*) calloc(3, sizeof(idx_t));
      idx_t edgecut = 0;
      idx_t part = -1;

      ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, &part, &global_comm);
   
      new_partition = (HYPRE_Int) part;
      free(vtxdist);
      free(xadj);
      free(adjncy);
      free(tpwgts);
      free(ubvec);
      free(options);
   }   

   // Send new partition info to everyone in old partition
   hypre_MPI_Bcast(&new_partition, 1, HYPRE_MPI_INT, 0, local_comm);


   return new_partition;
}
#endif

HYPRE_Int
GetNeighborPartitionInfo(hypre_ParAMGData *amg_data, 
   hypre_ParCompGridCommPkg *initCopyCompGridCommPkg, 
   MPI_Comm local_comm, 
   HYPRE_Int *proc_starts,
   HYPRE_Int partition, 
   HYPRE_Int current_level, 
   HYPRE_Int transition_level,
   HYPRE_Int *communication_cost)
{
   HYPRE_Int myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   HYPRE_Int level,i,j,k;

   // Get previous communicator ranks of others in your new local communicator
   HYPRE_Int local_myid, local_num_procs;
   hypre_MPI_Comm_rank(local_comm, &local_myid);
   hypre_MPI_Comm_size(local_comm, &local_num_procs);
   int local_num_procs_plain_int;
   MPI_Comm_size(local_comm, &local_num_procs_plain_int);
   int *local_ranks = hypre_CTAlloc(int, local_num_procs, HYPRE_MEMORY_HOST);
   for (i = 0; i < local_num_procs; i++) local_ranks[i] = i;
   MPI_Group previous_group, local_group;
   MPI_Comm_group(hypre_MPI_COMM_WORLD, &previous_group);
   MPI_Comm_group(local_comm, &local_group);
   int *previous_ranks_plain_int = hypre_CTAlloc(int, local_num_procs, HYPRE_MEMORY_HOST);
   MPI_Group_translate_ranks(local_group, local_num_procs_plain_int, local_ranks, previous_group, previous_ranks_plain_int);
   hypre_TFree(local_ranks, HYPRE_MEMORY_HOST);
   HYPRE_Int *previous_ranks = hypre_CTAlloc(HYPRE_Int, local_num_procs, HYPRE_MEMORY_HOST);
   for (i = 0; i < local_num_procs; i++) previous_ranks[i] = (HYPRE_Int) previous_ranks_plain_int[i];
   hypre_TFree(previous_ranks_plain_int, HYPRE_MEMORY_HOST);

   for (level = current_level; level < transition_level; level++)
   {

      hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];

      // !!! Are these safe/efficient allocations???
      HYPRE_Int max_num_comm_procs;
      hypre_MPI_Allreduce(&(hypre_ParCompGridCommPkgNumSendProcs(initCopyCompGridCommPkg)[level]), &max_num_comm_procs, 1, HYPRE_MPI_INT, MPI_SUM, local_comm);
      // max_num_comm_procs *= 2; // !!! Check this: I have seen a case where setting max num comm procs using only the above allreduce is not sufficient since the number of send procs may be larger due to varying partition sizes (may need to send to )
      if (communication_cost)
      {
         communication_cost[level*10] += log(local_num_procs)/log(2);
         communication_cost[level*10 + 1] += sizeof(HYPRE_Int)*(local_num_procs-1);
      }

      HYPRE_Int num_comm_partitions = 0;
      HYPRE_Int *comm_partitions = hypre_CTAlloc(HYPRE_Int, max_num_comm_procs, HYPRE_MEMORY_HOST); 
      HYPRE_Int **comm_partition_ranks = hypre_CTAlloc(HYPRE_Int*, max_num_comm_procs, HYPRE_MEMORY_HOST);
      HYPRE_Int *comm_partition_num_send_elmts = hypre_CTAlloc(HYPRE_Int, max_num_comm_procs, HYPRE_MEMORY_HOST);
      HYPRE_Int **comm_partition_send_elmts = hypre_CTAlloc(HYPRE_Int*, max_num_comm_procs, HYPRE_MEMORY_HOST);
      HYPRE_Int **comm_partition_ghost_marker = hypre_CTAlloc(HYPRE_Int*, max_num_comm_procs, HYPRE_MEMORY_HOST);

      if (hypre_ParCompGridCommPkgNumSendProcs(initCopyCompGridCommPkg)[level])
      {

         // Get list of processors to communicate with that are NOT in your same partition
         HYPRE_Int num_comm_procs = 0;
         HYPRE_Int *comm_procs = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridCommPkgNumSendProcs(initCopyCompGridCommPkg)[level], HYPRE_MEMORY_HOST);
         for (i = 0; i < hypre_ParCompGridCommPkgNumSendProcs(initCopyCompGridCommPkg)[level]; i++)
         {
            HYPRE_Int do_comm = 1;
            for (j = 0; j < local_num_procs; j++)
            {
               if (hypre_ParCompGridCommPkgSendProcs(initCopyCompGridCommPkg)[level][i] == previous_ranks[j]) do_comm = 0;
            }
            if (do_comm) comm_procs[num_comm_procs++] = hypre_ParCompGridCommPkgSendProcs(initCopyCompGridCommPkg)[level][i];
         }

         // Declare and allocate storage for communications
         HYPRE_Int cnt = 0;
         hypre_MPI_Request *requests = hypre_CTAlloc(hypre_MPI_Request, 2*num_comm_procs, HYPRE_MEMORY_HOST);
         hypre_MPI_Status *statuses = hypre_CTAlloc(hypre_MPI_Status, 2*num_comm_procs, HYPRE_MEMORY_HOST);
         HYPRE_Int *send_buffer_sizes = hypre_CTAlloc(HYPRE_Int, num_comm_procs, HYPRE_MEMORY_HOST);
         HYPRE_Int *recv_buffer_sizes = hypre_CTAlloc(HYPRE_Int, num_comm_procs, HYPRE_MEMORY_HOST);
         HYPRE_Int **send_buffers = hypre_CTAlloc(HYPRE_Int*, num_comm_procs, HYPRE_MEMORY_HOST);
         HYPRE_Int **recv_buffers = hypre_CTAlloc(HYPRE_Int*, num_comm_procs, HYPRE_MEMORY_HOST);

         // Communicate buffer sizes
         for (i = 0; i < num_comm_procs; i++) 
         {
            hypre_MPI_Irecv(&(recv_buffer_sizes[i]), 1, HYPRE_MPI_INT, comm_procs[i], 8, hypre_MPI_COMM_WORLD, &(requests[cnt++]));
         }

         for (i = 0; i < num_comm_procs; i++)
         {
            send_buffer_sizes[i] = 2 + local_num_procs;
            hypre_MPI_Isend(&(send_buffer_sizes[i]), 1, HYPRE_MPI_INT, comm_procs[i], 8, hypre_MPI_COMM_WORLD, &(requests[cnt++]));
            if (communication_cost)
            {
               communication_cost[level*10]++;
               communication_cost[level*10 + 1] += sizeof(HYPRE_Int);
            }
            send_buffers[i] = hypre_CTAlloc(HYPRE_Int, send_buffer_sizes[i], HYPRE_MEMORY_HOST);
         }
         
         hypre_MPI_Waitall(cnt, requests, statuses);
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(statuses, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, 2*num_comm_procs, HYPRE_MEMORY_HOST);
         statuses = hypre_CTAlloc(hypre_MPI_Status, 2*num_comm_procs, HYPRE_MEMORY_HOST);

         // Communicate partition info: send_buffer = [partition ID, number of ranks in this partition, [ranks] ]
         cnt = 0;
         for (i = 0; i < num_comm_procs; i++)
         {
            recv_buffers[i] = hypre_CTAlloc(HYPRE_Int, recv_buffer_sizes[i], HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv(recv_buffers[i], recv_buffer_sizes[i], HYPRE_MPI_INT, comm_procs[i], 9, hypre_MPI_COMM_WORLD, &(requests[cnt++]));
         }
         hypre_TFree(recv_buffer_sizes, HYPRE_MEMORY_HOST);

         for (i = 0; i < num_comm_procs; i++)
         {
            send_buffers[i][0] = partition;
            send_buffers[i][1] = local_num_procs;
            for (k = 0; k < local_num_procs; k++) send_buffers[i][2 + k] = previous_ranks[k];
            hypre_MPI_Isend(send_buffers[i], send_buffer_sizes[i], HYPRE_MPI_INT, comm_procs[i], 9, hypre_MPI_COMM_WORLD, &(requests[cnt++]));
            if (communication_cost)
            {
               communication_cost[level*10]++;
               communication_cost[level*10 + 1] += sizeof(HYPRE_Int)*send_buffer_sizes[i];
            }
         }
         
         hypre_MPI_Waitall(cnt, requests, statuses);
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(statuses, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_comm_procs; i++) if (send_buffers[i]) hypre_TFree(send_buffers[i], HYPRE_MEMORY_HOST);
         hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
         hypre_TFree(send_buffer_sizes, HYPRE_MEMORY_HOST);

         // Compress and organize communication info into partition-wise representation
         for (i = 0; i < num_comm_procs; i++)
         {
            HYPRE_Int incoming_partition = recv_buffers[i][0];
            HYPRE_Int incoming_partition_size = recv_buffers[i][1];

            // Check whether we've already accounted for this partition
            HYPRE_Int new_incoming_partition = 1;
            HYPRE_Int partition_index = num_comm_partitions;
            for (j = 0; j < num_comm_partitions; j++)
            {
               if (incoming_partition == comm_partitions[j])
               {
                  new_incoming_partition = 0;
                  partition_index = j;
                  break;
               }
            }

            // Get the index we need from the original compGridCommPkg structure
            HYPRE_Int send_proc_index;
            for (j = 0; j < hypre_ParCompGridCommPkgNumSendProcs(initCopyCompGridCommPkg)[level]; j++)
            {
               if (hypre_ParCompGridCommPkgSendProcs(initCopyCompGridCommPkg)[level][j] == comm_procs[i])
               {
                  send_proc_index = j;
                  break;
               }
            }

            if (new_incoming_partition)
            {
               comm_partitions[partition_index] = incoming_partition;
               comm_partition_ranks[partition_index] = hypre_CTAlloc(HYPRE_Int, incoming_partition_size + 1, HYPRE_MEMORY_HOST);
               comm_partition_ranks[partition_index][0] = incoming_partition_size;
               for (j = 0; j < incoming_partition_size; j++) comm_partition_ranks[partition_index][j+1] = recv_buffers[i][2 + j];
               
               // !!! Check these allocations... I think they are safe... but maybe too large
               comm_partition_send_elmts[partition_index] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_HOST); 
               comm_partition_ghost_marker[partition_index] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_HOST);
               for (j = hypre_ParCompGridCommPkgSendMapStarts(initCopyCompGridCommPkg)[level][send_proc_index]; j < hypre_ParCompGridCommPkgSendMapStarts(initCopyCompGridCommPkg)[level][send_proc_index+1]; j++)
               {
                  comm_partition_send_elmts[partition_index][ comm_partition_num_send_elmts[partition_index] ] = hypre_ParCompGridCommPkgSendMapElmts(initCopyCompGridCommPkg)[level][j];
                  comm_partition_ghost_marker[partition_index][ comm_partition_num_send_elmts[partition_index] ] = hypre_ParCompGridCommPkgGhostMarker(initCopyCompGridCommPkg)[level][j];
                  comm_partition_num_send_elmts[partition_index]++;
               }

               num_comm_partitions++;
            }
            else
            {
               // Merge send elmt and ghost marker info
               HYPRE_Int existing_cnt = 0;
               HYPRE_Int new_cnt = hypre_ParCompGridCommPkgSendMapStarts(initCopyCompGridCommPkg)[level][send_proc_index];
               HYPRE_Int merged_cnt = 0;
               HYPRE_Int *merged_send_elmts = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_HOST);
               HYPRE_Int *merged_ghost_marker = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_HOST);
               while (existing_cnt < comm_partition_num_send_elmts[partition_index] && new_cnt < hypre_ParCompGridCommPkgSendMapStarts(initCopyCompGridCommPkg)[level][send_proc_index+1])
               {
                  HYPRE_Int existing_elmt = comm_partition_send_elmts[partition_index][existing_cnt];
                  HYPRE_Int new_elmt = hypre_ParCompGridCommPkgSendMapElmts(initCopyCompGridCommPkg)[level][new_cnt];
                  HYPRE_Int existing_ghost_marker = comm_partition_ghost_marker[partition_index][existing_cnt];
                  HYPRE_Int new_ghost_marker = hypre_ParCompGridCommPkgGhostMarker(initCopyCompGridCommPkg)[level][new_cnt];
                  if (existing_elmt < new_elmt)
                  {
                     merged_send_elmts[merged_cnt] = existing_elmt;
                     merged_ghost_marker[merged_cnt] = existing_ghost_marker;
                     existing_cnt++;
                  }
                  else if (new_elmt < existing_elmt)
                  {
                     merged_send_elmts[merged_cnt] = new_elmt;
                     merged_ghost_marker[merged_cnt] = new_ghost_marker;
                     new_cnt++;
                  }
                  else if (new_elmt == existing_elmt)
                  {
                     merged_send_elmts[merged_cnt] = new_elmt;
                     merged_ghost_marker[merged_cnt] = existing_ghost_marker & new_ghost_marker; // !!! Check this
                     new_cnt++;
                     existing_cnt++;
                  }
                  merged_cnt++;
               }
               while(existing_cnt < comm_partition_num_send_elmts[partition_index])
               {
                  HYPRE_Int existing_elmt = comm_partition_send_elmts[partition_index][existing_cnt];
                  HYPRE_Int existing_ghost_marker = comm_partition_ghost_marker[partition_index][existing_cnt];
                  merged_send_elmts[merged_cnt] = existing_elmt;
                  merged_ghost_marker[merged_cnt] = existing_ghost_marker;
                  existing_cnt++; 
                  merged_cnt++;                 
               }
               while(new_cnt < hypre_ParCompGridCommPkgSendMapStarts(initCopyCompGridCommPkg)[level][send_proc_index+1])
               {
                  HYPRE_Int new_elmt = hypre_ParCompGridCommPkgSendMapElmts(initCopyCompGridCommPkg)[level][new_cnt];
                  HYPRE_Int new_ghost_marker = hypre_ParCompGridCommPkgGhostMarker(initCopyCompGridCommPkg)[level][new_cnt];
                  merged_send_elmts[merged_cnt] = new_elmt;
                  merged_ghost_marker[merged_cnt] = new_ghost_marker;
                  new_cnt++;
                  merged_cnt++;
               }
               comm_partition_num_send_elmts[partition_index] = merged_cnt;

               hypre_TFree(comm_partition_send_elmts[partition_index], HYPRE_MEMORY_HOST);
               hypre_TFree(comm_partition_ghost_marker[partition_index], HYPRE_MEMORY_HOST);
               comm_partition_send_elmts[partition_index] = merged_send_elmts;
               comm_partition_ghost_marker[partition_index] = merged_ghost_marker;
            }
         }

         // Clean up memory
         for (i = 0; i < num_comm_procs; i++) hypre_TFree(recv_buffers[i], HYPRE_MEMORY_HOST);
         hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
         hypre_TFree(comm_procs, HYPRE_MEMORY_HOST);
      }


      // Allgather communication info in the local partition
      HYPRE_Int *proc_offsets = hypre_CTAlloc(HYPRE_Int, local_num_procs, HYPRE_MEMORY_HOST);
      for (i = 1; i < local_num_procs; i++)
      {
         proc_offsets[i] = proc_offsets[i-1] + proc_starts[2*((i-1)*(transition_level - current_level) + level - current_level) + 1] - proc_starts[2*((i-1)*(transition_level - current_level) + level - current_level)] + 1;
      }

      AllgatherCommunicationInfo(amg_data, 
         level, 
         local_comm,
         num_comm_partitions, 
         comm_partitions, 
         comm_partition_ranks, 
         comm_partition_num_send_elmts, 
         comm_partition_send_elmts, 
         comm_partition_ghost_marker,
         proc_offsets,
         max_num_comm_procs,
         communication_cost);

      // Clean up memory
      for (i = 0; i < max_num_comm_procs; i++)
      {
         if (comm_partition_ranks[i]) hypre_TFree(comm_partition_ranks[i], HYPRE_MEMORY_HOST);
         if (comm_partition_send_elmts[i]) hypre_TFree(comm_partition_send_elmts[i], HYPRE_MEMORY_HOST);
         if (comm_partition_ghost_marker[i]) hypre_TFree(comm_partition_ghost_marker[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(proc_offsets, HYPRE_MEMORY_HOST);
      hypre_TFree(comm_partition_ranks, HYPRE_MEMORY_HOST);
      hypre_TFree(comm_partition_send_elmts, HYPRE_MEMORY_HOST);
      hypre_TFree(comm_partition_ghost_marker, HYPRE_MEMORY_HOST);
      hypre_TFree(comm_partition_num_send_elmts, HYPRE_MEMORY_HOST);

   }
   hypre_TFree(previous_ranks, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Int
AllgatherCommunicationInfo(hypre_ParAMGData *amg_data, HYPRE_Int level, MPI_Comm comm,
   HYPRE_Int num_comm_partitions,
   HYPRE_Int *comm_partitions,
   HYPRE_Int **comm_partition_ranks,
   HYPRE_Int *comm_partition_num_send_elmts,
   HYPRE_Int **comm_partition_send_elmts,
   HYPRE_Int **comm_partition_ghost_marker,
   HYPRE_Int *proc_offsets,
   HYPRE_Int max_num_comm_procs,
   HYPRE_Int *communication_cost)
{
   HYPRE_Int num_procs, myid;
   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &num_procs);

   HYPRE_Int i,j,proc;

   // Count the buffer size
   HYPRE_Int buffer_size = 1;
   for (i = 0; i < num_comm_partitions; i++)
   {
      buffer_size += 3 + comm_partition_ranks[i][0] + 2*comm_partition_num_send_elmts[i];
   }

   // Allgather the buffer size
   HYPRE_Int *recvcounts = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Allgather(&buffer_size, 1, HYPRE_MPI_INT, recvcounts, 1, HYPRE_MPI_INT, comm);
   if (communication_cost)
   {
      communication_cost[level*10] += log(num_procs)/log(2);
      communication_cost[level*10 + 1] += sizeof(HYPRE_Int)*(num_procs-1);
   }

   // Setup the displs
   HYPRE_Int *displs = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
   displs[0] = 0;
   for (i = 0; i < num_procs-1; i++) displs[i+1] = displs[i] + recvcounts[i];

   // Allocate and pack the buffer
   // send_buffer = [num partitions, [partition info], [partition info], ...]
   // partition info = [partition ID, num ranks, [ranks], num send elmts, [send elmts and ghost marker, interleaved]]
   HYPRE_Int *send_buffer = hypre_CTAlloc(HYPRE_Int, buffer_size, HYPRE_MEMORY_HOST);
   HYPRE_Int cnt = 0;
   send_buffer[cnt++] = num_comm_partitions;
   for (i = 0; i < num_comm_partitions; i++)
   {
      send_buffer[cnt++] = comm_partitions[i];
      send_buffer[cnt++] = comm_partition_ranks[i][0];
      for (j = 0; j < comm_partition_ranks[i][0]; j++) send_buffer[cnt++] = comm_partition_ranks[i][j+1];
      send_buffer[cnt++] = comm_partition_num_send_elmts[i];
      for (j = 0; j < comm_partition_num_send_elmts[i]; j++)
      {
         send_buffer[cnt++] = comm_partition_send_elmts[i][j];
         send_buffer[cnt++] = comm_partition_ghost_marker[i][j];
      }
   }

   // Allgatherv the buffers
   HYPRE_Int *recv_buffer = hypre_CTAlloc(HYPRE_Int, displs[num_procs-1] + recvcounts[num_procs-1], HYPRE_MEMORY_HOST);
   hypre_MPI_Allgatherv(send_buffer, buffer_size, HYPRE_MPI_INT, recv_buffer, recvcounts, displs, HYPRE_MPI_INT, comm);
   if (communication_cost)
   {
      communication_cost[level*10] += log(num_procs)/log(2);
      communication_cost[level*10 + 1] += sizeof(HYPRE_Int)*(buffer_size)*(num_procs-1);
   }
   hypre_TFree(send_buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(recvcounts, HYPRE_MEMORY_HOST);
   hypre_TFree(displs, HYPRE_MEMORY_HOST);

   // Compress and organize all received communication info for later use
   hypre_ParCompGridCommPkg *compGridCommPkg = hypre_ParAMGDataCompGridCommPkg(amg_data);
   hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];
   cnt = 0;
   // Zero out comm_partition_num_send_elmts since we will use this variable to recount things
   for (i = 0; i < num_comm_partitions; i++) comm_partition_num_send_elmts[i] = 0;
   // Loop over info received from each processor
   for (proc = 0; proc < num_procs; proc++)
   {
      // Find where this processor's original owned dofs start in the agglomerated grid
      HYPRE_Int proc_offset = proc_offsets[proc];

      // Loop over info from different partitions
      HYPRE_Int num_incoming_partitions = recv_buffer[cnt++];
      for (i = 0; i < num_incoming_partitions; i++)
      {
         // Get the incoming partition id
         HYPRE_Int incoming_partition_id = recv_buffer[cnt++];
         HYPRE_Int new_partition = 1;
         HYPRE_Int partition_index = num_comm_partitions;
         for (j = 0; j < num_comm_partitions; j++)
         {
            if (incoming_partition_id == comm_partitions[j])
            {
               new_partition = 0;
               partition_index = j;
               break;
            }
         }

         HYPRE_Int incoming_num_ranks = recv_buffer[cnt++];
         if (new_partition)
         {
            // For a new incoming partition, must increment the num_comm_partitions and setup the rank info for this partition
            num_comm_partitions++;
            comm_partitions[partition_index] = incoming_partition_id;
            comm_partition_ranks[partition_index] = hypre_CTAlloc(HYPRE_Int, incoming_num_ranks+1, HYPRE_MEMORY_HOST);
            comm_partition_send_elmts[partition_index] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_HOST); // !!! Correct/efficient allocation?
            comm_partition_ghost_marker[partition_index] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_HOST);
            comm_partition_ranks[partition_index][0] = incoming_num_ranks;
            for (j = 0; j < incoming_num_ranks; j++) comm_partition_ranks[partition_index][j+1] = recv_buffer[cnt++];
         }
         else cnt += incoming_num_ranks;
         // Increment the num_send_elmts and copy info on send elmts and ghost marker
         HYPRE_Int incoming_num_send_elmts = recv_buffer[cnt++];
         for (j = 0; j < incoming_num_send_elmts; j++)
         {
            comm_partition_send_elmts[partition_index][ comm_partition_num_send_elmts[partition_index] ] = recv_buffer[cnt++] + proc_offset; 
            // if (global_myid == 0) fprintf(file,"recv_buffer = %d, comm_partition_send_elmts[%d] = %d\n", recv_buffer[cnt-1], comm_partition_num_send_elmts[partition_index], comm_partition_send_elmts[partition_index][ comm_partition_num_send_elmts[partition_index] ]);
            comm_partition_ghost_marker[partition_index][ comm_partition_num_send_elmts[partition_index] ] = recv_buffer[cnt++];
            comm_partition_num_send_elmts[partition_index]++;
         }
      }
   }
   hypre_TFree(recv_buffer, HYPRE_MEMORY_HOST);

   // Use accumulated and organized comm_partition info in order to setup compGridCommPkg correctly
   HYPRE_Int *send_procs = hypre_CTAlloc(HYPRE_Int, max_num_comm_procs, HYPRE_MEMORY_HOST);
   HYPRE_Int *recv_procs = hypre_CTAlloc(HYPRE_Int, num_comm_partitions, HYPRE_MEMORY_HOST);
   HYPRE_Int *send_proc_partitions = hypre_CTAlloc(HYPRE_Int, max_num_comm_procs, HYPRE_MEMORY_HOST);
   HYPRE_Int *send_map_starts = hypre_CTAlloc(HYPRE_Int, num_comm_partitions+1, HYPRE_MEMORY_HOST);
   HYPRE_Int **sorted_partition_ranks = hypre_CTAlloc(HYPRE_Int*, num_comm_partitions, HYPRE_MEMORY_HOST);
   HYPRE_Int total_send_elmts = 0;
   for (i = 0; i < num_comm_partitions; i++) total_send_elmts += comm_partition_num_send_elmts[i];
   HYPRE_Int *send_map_elmts = hypre_CTAlloc(HYPRE_Int, total_send_elmts, HYPRE_MEMORY_HOST);
   HYPRE_Int *ghost_marker = hypre_CTAlloc(HYPRE_Int, total_send_elmts, HYPRE_MEMORY_HOST);
   total_send_elmts = 0;   
   HYPRE_Int num_send_procs = 0;

   // Want to sort the partitions so that everyone in the same local partition has same list in same order.
   HYPRE_Int *sorted_indices = hypre_CTAlloc(HYPRE_Int, num_comm_partitions, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_comm_partitions; i++) sorted_indices[i] = i;
   hypre_qsort2i(comm_partitions, sorted_indices, 0, num_comm_partitions-1);

   for (i = 0; i < num_comm_partitions; i++)
   {
      // Figure out ranks of communication partners in this partition. 
      // NOTE: num_procs, myid are both for the local allgather comm. num_ranks is in other partition's local comm
      HYPRE_Int num_ranks = comm_partition_ranks[ sorted_indices[i] ][0];
      sorted_partition_ranks[i] = hypre_CTAlloc(HYPRE_Int, num_ranks+1, HYPRE_MEMORY_HOST);
      sorted_partition_ranks[i][0] = num_ranks;
      for (j = 0; j < num_ranks; j++) sorted_partition_ranks[i][j+1] = comm_partition_ranks[ sorted_indices[i] ][j+1];
      recv_procs[i] = comm_partition_ranks[ sorted_indices[i] ][1 + (myid % num_ranks)];
      for (j = 0; j < ((num_ranks-1)/num_procs)+1; j++)
      {
         if (j*num_procs + myid < num_ranks)
         {
            if (num_send_procs >= max_num_comm_procs)
            {
               // hypre_printf("num_send_procs = %d, but max_num_comm_procs = %d\n", num_send_procs, max_num_comm_procs);
               max_num_comm_procs *= 2;
               send_procs = hypre_TReAlloc(send_procs, HYPRE_Int, max_num_comm_procs, HYPRE_MEMORY_HOST);
               send_proc_partitions = hypre_TReAlloc(send_proc_partitions, HYPRE_Int, max_num_comm_procs, HYPRE_MEMORY_HOST);
            }
            send_procs[num_send_procs] = comm_partition_ranks[ sorted_indices[i] ][1 + j*num_procs + myid];
            send_proc_partitions[num_send_procs] = i;
            num_send_procs++;
         }
      }
      for (j = 0; j < comm_partition_num_send_elmts[ sorted_indices[i] ]; j++)
      {
         send_map_elmts[total_send_elmts + j] = comm_partition_send_elmts[ sorted_indices[i] ][j];
         ghost_marker[total_send_elmts + j] = comm_partition_ghost_marker[ sorted_indices[i] ][j];
      }
      send_map_starts[i] = total_send_elmts;
      total_send_elmts += comm_partition_num_send_elmts[ sorted_indices[i] ];
   }
   send_map_starts[num_comm_partitions] = total_send_elmts;
   hypre_TFree(sorted_indices, HYPRE_MEMORY_HOST);

   // Store the info in the compGridCommPkg
   hypre_TFree(hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCompGridCommPkgSendPartitions(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCompGridCommPkgSendPartitionRanks(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level], HYPRE_MEMORY_HOST);

   hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level] = num_send_procs;
   hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level] = num_comm_partitions;
   hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level] = num_comm_partitions;
   hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level] = send_procs;
   hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level] = recv_procs;
   hypre_ParCompGridCommPkgSendPartitions(compGridCommPkg)[level] = comm_partitions;
   hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level] = send_proc_partitions;
   hypre_ParCompGridCommPkgSendPartitionRanks(compGridCommPkg)[level] = sorted_partition_ranks;
   hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level] = send_map_starts;
   hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level] = send_map_elmts;
   hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level] = ghost_marker;

   return 0;
}

HYPRE_Int*
PackSendBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, 
   HYPRE_Int *send_flag_buffer_size, HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
   HYPRE_Int partition, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int symmetric )
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

   HYPRE_Int coarse_partition_id = hypre_ParCompGridCommPkgSendPartitions(compGridCommPkg)[current_level][partition];
   HYPRE_Int coarse_partition = partition;

   // Get where to look in commPkgSendMapElmts
   HYPRE_Int            start = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][partition];
   HYPRE_Int            finish = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][partition+1];

   // Get the transition level
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg);
   if (transition_level < 0) transition_level = num_levels;

   // Get the sort maps
   HYPRE_Int            *sort_map;
   HYPRE_Int            *sort_map_coarse;
   if (current_level != transition_level-1) sort_map_coarse = hypre_ParCompGridSortMap(compGrid[current_level+1]);

   // initialize send map buffer size
   (*send_flag_buffer_size) = num_levels - current_level;

   // Mark the nodes to send (including Psi_c grid plus ghost nodes)

   // Count up the buffer size for the starting nodes
   num_send_nodes[current_level][partition][current_level] = finish - start;
   send_flag[current_level][partition][current_level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level][partition][current_level], HYPRE_MEMORY_HOST );
   inv_send_flag[current_level] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[current_level]), HYPRE_MEMORY_HOST );

   (*buffer_size) += 2;
   if (current_level != transition_level-1) (*buffer_size) += 3*num_send_nodes[current_level][partition][current_level];
   else (*buffer_size) += 2*num_send_nodes[current_level][partition][current_level];

   for (i = start; i < finish; i++)
   {
      send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];
      if (hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[current_level][i])
         send_flag[current_level][partition][current_level][i - start] = -(send_elmt + 1);
      else
         send_flag[current_level][partition][current_level][i - start] = send_elmt;
      inv_send_flag[current_level][send_elmt] = i - start + 1;

      (*buffer_size) += hypre_ParCompGridARowPtr(compGrid[current_level])[send_elmt+1] - hypre_ParCompGridARowPtr(compGrid[current_level])[send_elmt];
   }
   (*send_flag_buffer_size) += finish - start;

   // Add the nodes listed by the coarse grid counterparts if applicable
   // Note that the compGridCommPkg is set up to list all nodes within the padding plus ghost layers
   if (current_level != transition_level-1)
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
   for (level = current_level + 1; level < transition_level; level++)
   {
      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         sort_map = hypre_ParCompGridSortMap(compGrid[level]);
         if (level != transition_level-1) sort_map_coarse = hypre_ParCompGridSortMap(compGrid[level+1]);
         HYPRE_Int *inv_sort_map = hypre_ParCompGridInvSortMap(compGrid[level]);
         
         num_psi_levels++;
         (*buffer_size)++;
         nodes_to_add = 0;

         // if we need coarse info, allocate space for the add flag on the next level
         if (level != transition_level-1) add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]), HYPRE_MEMORY_HOST );

         // Expand by the padding on this level and add coarse grid counterparts if applicable
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (add_flag[level][sort_map[i]] == padding[level] + 1)
            {
               // Recursively add the region of padding (flagging coarse nodes on the next level if applicable)
               if (level != transition_level-1) RecursivelyBuildPsiComposite(i, padding[level], compGrid[level], add_flag[level], add_flag[level+1], sort_map, sort_map_coarse, 1, &nodes_to_add, padding[level+1], level);
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

         // If we previously sent dofs to this partition on this level, mark the redundant_add_flag
         if (hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[level])
         {
            coarse_partition_id = -1;
            coarse_partition = -1;
            HYPRE_Int example_proc = hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[current_level][partition];
            for (j = 0; j < hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level]; j++)
            {
               for (k = 0; k < hypre_ParCompGridCommPkgSendPartitionRanks(compGridCommPkg)[level][j][0]; k++)
               {
                  if (example_proc == hypre_ParCompGridCommPkgSendPartitionRanks(compGridCommPkg)[level][j][k+1]) 
                  {
                     coarse_partition_id = hypre_ParCompGridCommPkgSendPartitions(compGridCommPkg)[level][j];
                     coarse_partition = j;
                  }
               }
            }
         }
         else
         {
            coarse_partition = -1;
            for (j = 0; j < hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level]; j++)
            {
               if (hypre_ParCompGridCommPkgSendPartitions(compGridCommPkg)[level][j] == coarse_partition_id) coarse_partition = j;
            }
         }
         if (coarse_partition != -1)
         {
            if (!redundant_add_flag[level]) redundant_add_flag[level] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level]), HYPRE_MEMORY_HOST );
            if (nodes_to_add) redundant_add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]), HYPRE_MEMORY_HOST );

            start = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][coarse_partition];
            finish = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][coarse_partition+1];
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
               num_send_nodes[current_level][partition][level]++;
            }
         }
         
         // Save the indices (in global index ordering) so I don't have to keep looping over all nodes in compGrid when I pack the buffer
         send_flag[current_level][partition][level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level][partition][level], HYPRE_MEMORY_HOST );
         inv_send_flag[level] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level]), HYPRE_MEMORY_HOST );
         cnt =  0;

         for (j = 0; j < hypre_ParCompGridNumNodes(compGrid[level]); j++)
         {
            if (add_flag[level][j] > num_ghost_layers)
            {
               inv_send_flag[level][ inv_sort_map[j] ] = cnt + 1;
               send_flag[current_level][partition][level][cnt++] = inv_sort_map[j];
            }
            else if (add_flag[level][j] > 0)
            {
               inv_send_flag[level][ inv_sort_map[j] ] = cnt + 1;
               send_flag[current_level][partition][level][cnt++] = -(inv_sort_map[j]+1);
            }
         }


         // Count up the buffer sizes
         (*send_flag_buffer_size) += num_send_nodes[current_level][partition][level];
         if (level != transition_level-1) (*buffer_size) += 3*num_send_nodes[current_level][partition][level];
         else (*buffer_size) += 2*num_send_nodes[current_level][partition][level];
         
         for (i = 0; i < num_send_nodes[current_level][partition][level]; i++)
         {
            send_elmt = send_flag[current_level][partition][level][i];
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
      send_buffer[cnt++] = num_send_nodes[current_level][partition][level];

      // copy all global indices
      for (i = 0; i < num_send_nodes[current_level][partition][level]; i++)
      {
         send_elmt = send_flag[current_level][partition][level][i];
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
      if (level != transition_level-1)
      {
         for (i = 0; i < num_send_nodes[current_level][partition][level]; i++)
         {
            send_elmt = send_flag[current_level][partition][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            send_buffer[cnt++] = hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ send_elmt ];
         }
      }

      // store the row length for matrix A
      if (symmetric)
      {
         for (i = 0; i < num_send_nodes[current_level][partition][level]; i++)
         {
            send_elmt = send_flag[current_level][partition][level][i];
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
         for (i = 0; i < num_send_nodes[current_level][partition][level]; i++)
         {
            send_elmt = send_flag[current_level][partition][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            row_length = hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt + 1 ]
                       - hypre_ParCompGridARowPtr(compGrid[level])[ send_elmt ];
            send_buffer[cnt++] = row_length;
         }
      }

      // copy global indices for matrix A
      if (symmetric)
      {
         for (i = 0; i < num_send_nodes[current_level][partition][level]; i++)
         {
            send_elmt = send_flag[current_level][partition][level][i];
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
         for (i = 0; i < num_send_nodes[current_level][partition][level]; i++)
         {
            send_elmt = send_flag[current_level][partition][level][i];
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
   for (level = 0; level < transition_level; level++)
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
         // hypre_printf("Rank %d: Error! Ran into a -1 index when building Psi_c\n", myid);
         if (myid == 1) printf("Rank %d, level %d: Error! Ran into a -1 index when building Psi_c,\nnode = %d with global id %d, index = %d with global id = %d, m = %d\n",
            myid, level, node, hypre_ParCompGridGlobalIndices(compGrid)[node], index, hypre_ParCompGridAGlobalColInd(compGrid)[i], m);
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

            if (recv_map[level][i] != -1)
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
         if (send_flag_buffer[cnt++] != -1) 
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
CommunicateRemainingMatrixInfo(hypre_ParAMGData* amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *communication_cost)
{
   HYPRE_Int outer_level,proc,part,level,i,j;
   HYPRE_Int num_levels = hypre_ParCompGridCommPkgNumLevels(compGridCommPkg);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg);
   if (transition_level < 0) transition_level = num_levels;

   HYPRE_Int myid,num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   HYPRE_Int ***temp_PColInd = hypre_CTAlloc(HYPRE_Int**, transition_level, HYPRE_MEMORY_HOST);
   HYPRE_Complex ***temp_PData = hypre_CTAlloc(HYPRE_Complex**, transition_level, HYPRE_MEMORY_HOST);
   for (outer_level = amgdd_start_level; outer_level < transition_level; outer_level++)
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
      temp_RColInd = hypre_CTAlloc(HYPRE_Int**, transition_level, HYPRE_MEMORY_HOST);
      temp_RData = hypre_CTAlloc(HYPRE_Complex**, transition_level, HYPRE_MEMORY_HOST);
      for (outer_level = amgdd_start_level; outer_level < transition_level; outer_level++)
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

   for (outer_level = transition_level-1; outer_level >= amgdd_start_level; outer_level--)
   {
      // Get send/recv info from the comp grid comm pkg
      HYPRE_Int num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level];
      HYPRE_Int num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[outer_level];
      HYPRE_Int num_send_partitions = hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[outer_level];
      HYPRE_Int *send_procs = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[outer_level];
      HYPRE_Int *recv_procs = hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[outer_level];

      if (num_send_procs || num_recv_procs)
      {
         // Get the buffer sizes
         HYPRE_Int *send_sizes = hypre_CTAlloc(HYPRE_Int, 2*num_send_partitions, HYPRE_MEMORY_HOST);
         for (part = 0; part < num_send_partitions; part++)
         {
            for (level = outer_level; level < num_levels; level++)
            {      
               HYPRE_Int A_row_size = 0;
               HYPRE_Int P_row_size = 0;
               HYPRE_Int R_row_size = 0;
               HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
               for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][part][level]; i++)
               {
                  HYPRE_Int idx, A_row_size;
                  idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][part][level][i];
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

                  send_sizes[2*part] += P_row_size + R_row_size;
                  send_sizes[2*part+1] += A_row_size + P_row_size + R_row_size;
               }
               if (hypre_ParCompGridPRowPtr(compGrid[level])) send_sizes[2*part] += hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][part][level];
               if (hypre_ParCompGridRRowPtr(compGrid[level])) send_sizes[2*part] += hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][part][level];
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
            HYPRE_Int buffer_index = hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[outer_level][proc];
            hypre_MPI_Isend(&(send_sizes[2*buffer_index]), 2, HYPRE_MPI_INT, send_procs[proc], 1, hypre_MPI_COMM_WORLD, &(size_requests[request_cnt++]));
            
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
         HYPRE_Int **int_send_buffers = hypre_CTAlloc(HYPRE_Int*, num_send_partitions, HYPRE_MEMORY_HOST);
         HYPRE_Complex **complex_send_buffers = hypre_CTAlloc(HYPRE_Complex*, num_send_partitions, HYPRE_MEMORY_HOST);
         for (part = 0; part < num_send_partitions; part++)
         {
            // Allocate
            int_send_buffers[part] = hypre_CTAlloc(HYPRE_Int, send_sizes[2*part], HYPRE_MEMORY_HOST);
            complex_send_buffers[part] = hypre_CTAlloc(HYPRE_Complex, send_sizes[2*part+1], HYPRE_MEMORY_HOST);

            // Pack
            HYPRE_Int int_cnt = 0;
            HYPRE_Int complex_cnt = 0;
            for (level = outer_level; level < num_levels; level++)
            {
               for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][part][level]; i++)
               {
                  HYPRE_Int idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][part][level][i];
                  if (idx < 0) idx = -(idx + 1);
                  for (j = hypre_ParCompGridARowPtr(compGrid[level])[idx]; j < hypre_ParCompGridARowPtr(compGrid[level])[idx+1]; j++)
                  {
                     complex_send_buffers[part][complex_cnt++] = hypre_ParCompGridAData(compGrid[level])[j];
                  }
               }
               if (level != num_levels-1)
               {      
                  HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
                  for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][part][level]; i++)
                  {
                     HYPRE_Int idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][part][level][i];
                     if (idx < 0) idx = -(idx + 1);
                     if (idx < num_owned_nodes)
                     {
                        int_send_buffers[part][int_cnt++] = hypre_ParCompGridPRowPtr(compGrid[level])[idx+1] - hypre_ParCompGridPRowPtr(compGrid[level])[idx];
                        for (j = hypre_ParCompGridPRowPtr(compGrid[level])[idx]; j < hypre_ParCompGridPRowPtr(compGrid[level])[idx+1]; j++)
                        {
                           int_send_buffers[part][int_cnt++] = hypre_ParCompGridPColInd(compGrid[level])[j];
                           complex_send_buffers[part][complex_cnt++] = hypre_ParCompGridPData(compGrid[level])[j];
                        }
                     }
                     else
                     {
                        int_send_buffers[part][int_cnt++] = hypre_ParCompGridPRowPtr(compGrid[level])[idx+1];
                        for (j = 0; j < hypre_ParCompGridPRowPtr(compGrid[level])[idx+1]; j++)
                        {
                           HYPRE_Int temp_idx = idx - num_owned_nodes;
                           int_send_buffers[part][int_cnt++] = temp_PColInd[level][temp_idx][j];
                           complex_send_buffers[part][complex_cnt++] = temp_PData[level][temp_idx][j];
                        }
                     }
                  }
               }
               if (level != 0 && hypre_ParAMGDataRestriction(amg_data))
               {
                  HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
                  for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][part][level]; i++)
                  {
                     HYPRE_Int idx = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][part][level][i];
                     if (idx < 0) idx = -(idx + 1);
                     if (idx < num_owned_nodes)
                     {
                        int_send_buffers[part][int_cnt++] = hypre_ParCompGridRRowPtr(compGrid[level])[idx+1] - hypre_ParCompGridRRowPtr(compGrid[level])[idx];
                        for (j = hypre_ParCompGridRRowPtr(compGrid[level])[idx]; j < hypre_ParCompGridRRowPtr(compGrid[level])[idx+1]; j++)
                        {
                           int_send_buffers[part][int_cnt++] = hypre_ParCompGridRColInd(compGrid[level])[j];
                           complex_send_buffers[part][complex_cnt++] = hypre_ParCompGridRData(compGrid[level])[j];
                        }
                     }
                     else
                     {
                        int_send_buffers[part][int_cnt++] = hypre_ParCompGridRRowPtr(compGrid[level])[idx+1];
                        for (j = 0; j < hypre_ParCompGridRRowPtr(compGrid[level])[idx+1]; j++)
                        {
                           HYPRE_Int temp_idx = idx - num_owned_nodes;
                           int_send_buffers[part][int_cnt++] = temp_RColInd[level][temp_idx][j];
                           complex_send_buffers[part][complex_cnt++] = temp_RData[level][temp_idx][j];
                        }
                     }
                  }
               }
            }
         }

         // Send
         for (proc = 0; proc < num_send_procs; proc++)
         {
            HYPRE_Int buffer_index = hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[outer_level][proc];
            hypre_MPI_Isend(int_send_buffers[buffer_index], send_sizes[2*buffer_index], HYPRE_MPI_INT, send_procs[proc], 2, hypre_MPI_COMM_WORLD, &(buf_requests[request_cnt++]));
            hypre_MPI_Isend(complex_send_buffers[buffer_index], send_sizes[2*buffer_index+1], HYPRE_MPI_COMPLEX, send_procs[proc], 3, hypre_MPI_COMM_WORLD, &(buf_requests[request_cnt++]));
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

         for (part = 0; part < num_send_partitions; part++) hypre_TFree(int_send_buffers[part], HYPRE_MEMORY_HOST);
         for (part = 0; part < num_send_partitions; part++) hypre_TFree(complex_send_buffers[part], HYPRE_MEMORY_HOST);
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
   for (level = amgdd_start_level; level < transition_level; level++)
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
            hypre_ParCompGridResize(compGrid[level], new_size, level != num_levels-1, 2);
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
            hypre_ParCompGridResize(compGrid[level], new_size, level != num_levels-1, 3);
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
   for (level = amgdd_start_level; level < transition_level; level++)
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
      for (level = amgdd_start_level; level < transition_level; level++)
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
FinalizeCompGridCommPkg(hypre_ParAMGData* amg_data, hypre_ParCompGridCommPkg *compGridCommPkg, hypre_ParCompGrid **compGrid)
{
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   HYPRE_Int outer_level, part, proc, level, i;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg);
   if (transition_level < 0) transition_level = num_levels;

   HYPRE_Int total_num_nodes = 0;
   HYPRE_Int *offsets = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      offsets[level] = total_num_nodes;
      total_num_nodes += hypre_ParCompGridNumNodes(compGrid[level]);
   }

   if (!hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)) hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   if (!hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)) hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   if (!hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)) hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   if (!hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)) hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);

   for (outer_level = amgdd_start_level; outer_level < transition_level; outer_level++)
   {
      // Finalize send info
      HYPRE_Int num_send_partitions = hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[outer_level];

      if (hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level]) hypre_TFree(hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level], HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level] = hypre_CTAlloc(HYPRE_Int, num_send_partitions+1, HYPRE_MEMORY_SHARED);

      hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][0] = 0;
      
      for (part = 0; part < num_send_partitions; part++)
      {
         hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][part+1] = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][part];

         for (level = outer_level; level < num_levels; level++)
         {
            for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][part][level]; i++)
            {
               if (hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][part][level][i] >= 0)
               {
                  hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][part+1]++;
               }
            }
         }
      }

      if (hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[outer_level]) hypre_TFree(hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[outer_level], HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[outer_level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][num_send_partitions], HYPRE_MEMORY_SHARED);

      HYPRE_Int num_send_nodes = 0;
      HYPRE_Int new_num_send_partitions = 0;
      for (part = 0; part < num_send_partitions; part++)
      {
         for (level = outer_level; level < num_levels; level++)
         {
            for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][part][level]; i++)
            {
               if (hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][part][level][i] >= 0)
               {
                  hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[outer_level][num_send_nodes++] = offsets[level] + hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][part][level][i];
               }
            }
         }

         if (hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][part+1] > hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][part])
         {
            new_num_send_partitions++;
         }
      }

      hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[outer_level] = new_num_send_partitions;
      HYPRE_Int num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level];
      hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level] = 0;
      HYPRE_Int new_cnt = 0;
      for (proc = 0; proc < num_send_procs; proc++)
      {
         part = hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[outer_level][proc];
         if (hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][part+1] > hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][part])
         {
            hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[outer_level][new_cnt] = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[outer_level][proc];
            hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[outer_level][new_cnt] = part;
            hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level]++;
            new_cnt++;
         }
      }
      new_cnt = 0;
      for (part = 0; part < num_send_partitions; part++)
      {
         if (hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][part+1] > hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][part])
         {
            hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][new_cnt+1] = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][part+1];
            for (proc = 0; proc < num_send_procs; proc++) if (hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[outer_level][proc] == part) hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[outer_level][proc] = new_cnt;
            new_cnt++;
         }
      }

      // Finalize recv info
      HYPRE_Int num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[outer_level];

      if (hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level]) hypre_TFree(hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level], HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level] = hypre_CTAlloc(HYPRE_Int, num_recv_procs+1, HYPRE_MEMORY_SHARED);

      hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][0] = 0;
      
      for (proc = 0; proc < num_recv_procs; proc++)
      {
         hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc+1] = hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc];

         for (level = outer_level; level < num_levels; level++)
         {
            for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
            {
               if (hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i] >= 0)
               {
                  hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc+1]++;
               }
            }
         }
      }
      
      if (hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)[outer_level]) hypre_TFree(hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)[outer_level], HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)[outer_level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][num_recv_procs], HYPRE_MEMORY_SHARED);

      HYPRE_Int num_recv_nodes = 0;
      HYPRE_Int new_num_recv_procs = 0;
      for (proc = 0; proc < num_recv_procs; proc++)
      {
         for (level = outer_level; level < num_levels; level++)
         {
            for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
            {
               if (hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i] >= 0)
               {
                  hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)[outer_level][num_recv_nodes++] = offsets[level] + hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];
               }
            }
         }

         if (hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc+1] > hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc])
         {
            new_num_recv_procs++;
         }
      }

      hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[outer_level] = new_num_recv_procs;
      new_cnt = 0;
      for (proc = 0; proc < num_recv_procs; proc++)
      {
         if (hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc+1] > hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc])
         {
            hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[outer_level][new_cnt] = hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[outer_level][proc];
            hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][new_cnt+1] = hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc+1];
            new_cnt++;
         }
      }
   }

   return 0;
}

HYPRE_Int
TestCompGrids1(hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int transition_level, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int current_level, HYPRE_Int check_ghost_info)
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
   for (level = 0; level < transition_level; level++) add_flag[level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level]), HYPRE_MEMORY_HOST);
   HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[current_level])[hypre_ParCompGridNumOwnedBlocks(compGrid[current_level])];
   for (i = 0; i < num_owned_nodes; i++) add_flag[current_level][i] = padding[current_level] + 1;

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
      // !!! NOTE: disabling this check for now since I think processor agglomeration may put extra nodes in (and that is OK)
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
      {
         if (add_flag[level][i] == 0) 
         {
            test_failed = 1;
            if (myid == 0) hypre_printf("Error: extra nodes present in comp grid, rank %d, level %d, i = %d, global index = %d\n", myid, level, i, hypre_ParCompGridGlobalIndices(compGrid[level])[i]);
         }
      }

      // Check to make sure we have the correct identification of ghost nodes
      if (level != transition_level-1 && check_ghost_info)
      {
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) 
         {
            // !!! NOTE: disabling this check since I've redefined how real dofs are determined after agglomeration (can now be many more real dofs than expected via simple top down construction)
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
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   for (level = 0; level < transition_level; level++)
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

      if (level != transition_level-1)
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
      HYPRE_Int num_send_partitions = hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level];

      HYPRE_Int **send_buffers = hypre_CTAlloc(HYPRE_Int*, num_send_partitions, HYPRE_MEMORY_HOST);
      HYPRE_Int **recv_buffers = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);

      hypre_MPI_Request *requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
      hypre_MPI_Status *status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
      HYPRE_Int request_cnt = 0;

      // Pack send buffers   
      for (i = 0; i < num_send_partitions; i++)
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
         HYPRE_Int buffer_index = hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level][i];
         hypre_MPI_Isend(send_buffers[buffer_index], num_levels - level, HYPRE_MPI_INT, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 0, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
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
      for (i = 0; i < num_send_partitions; i++)
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
