/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"

/*****************************************************************************
 *
 * Routine for setting up the composite grids in AMG-DD

 *****************************************************************************/

/*****************************************************************************
 * hypre_BoomerAMGDDSetup
 *****************************************************************************/

HYPRE_Int
hypre_BoomerAMGDDSetup( void *amgdd_vdata,
                        hypre_ParCSRMatrix *A,
                        hypre_ParVector *b,
                        hypre_ParVector *x)
{

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   char filename[256];

   MPI_Comm 	      comm;
   hypre_ParAMGDDData *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_ParAMGData   *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);

   // If the underlying AMG data structure has not yet been set up, call BoomerAMGSetup()
   if (!hypre_ParAMGDataAArray(amg_data))
      hypre_BoomerAMGSetup( (void*) amg_data, A, b, x);

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
   HYPRE_Int amgdd_start_level = hypre_ParAMGDDDataStartLevel(amgdd_data);
   if (amgdd_start_level >= num_levels) amgdd_start_level = num_levels-1;
   HYPRE_Int pad = hypre_ParAMGDDDataPadding(amgdd_data);
   HYPRE_Int num_ghost_layers = hypre_ParAMGDDDataNumGhostLayers(amgdd_data);

   // Allocate pointer for the composite grids
   hypre_AMGDDCompGrid **compGrid = hypre_CTAlloc(hypre_AMGDDCompGrid*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParAMGDDDataCompGrid(amgdd_data) = compGrid;

   // In the 1 processor case, just need to initialize the comp grids
   if (num_procs == 1)
   {
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         compGrid[level] = hypre_AMGDDCompGridCreate();
         hypre_AMGDDCompGridInitialize( amgdd_data, 0, level);
      }
      hypre_AMGDDCompGridFinalize(amgdd_data);
      hypre_AMGDDCompGridSetupRelax(amgdd_data);
      return 0;
   }

   // Get the padding on each level
   HYPRE_Int *padding = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   for (level = amgdd_start_level; level < num_levels; level++) padding[level] = pad;

   // Initialize composite grid structures
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      compGrid[level] = hypre_AMGDDCompGridCreate();
      hypre_AMGDDCompGridInitialize( amgdd_data, padding[level], level );
   }

   // Create the compGridCommPkg and grab a few frequently used variables
   hypre_AMGDDCommPkg *compGridCommPkg = hypre_AMGDDCommPkgCreate(num_levels);
   hypre_ParAMGDDDataCommPkg(amgdd_data) = compGridCommPkg;

   send_buffer_size = hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg);
   recv_buffer_size = hypre_AMGDDCommPkgRecvBufferSize(compGridCommPkg);
   send_flag = hypre_AMGDDCommPkgSendFlag(compGridCommPkg);
   num_send_nodes = hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg);
   recv_map = hypre_AMGDDCommPkgRecvMap(compGridCommPkg);
   num_recv_nodes = hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg);
   HYPRE_Int *nodes_added_on_level = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   recv_redundant_marker = hypre_CTAlloc(HYPRE_Int***, num_levels, HYPRE_MEMORY_HOST);

   // On each level, setup the compGridCommPkg so that it has communication info for distance (eta + numGhostLayers)
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      hypre_BoomerAMGDD_SetupNearestProcessorNeighbors(hypre_ParAMGDataAArray(amg_data)[level], compGridCommPkg, level, padding, num_ghost_layers);
   }

   /////////////////////////////////////////////////////////////////

   // Loop over levels from coarsest to finest to build up the composite grids

   /////////////////////////////////////////////////////////////////

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

         for (i = 0; i < num_send_procs; i++)
         {
            send_buffer[i] = hypre_BoomerAMGDD_PackSendBuffer(amgdd_data, i, level, padding, &(send_flag_buffer_size[i]));
         }

         //////////// Communicate buffer sizes ////////////

         // post the receives for the buffer size
         for (i = 0; i < num_recv_procs; i++)
         {
            hypre_MPI_Irecv( &(recv_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]) );
         }

         // send the buffer sizes
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_MPI_Isend(&(send_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]));
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
         }

         // wait for buffers to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests, status );
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         request_counter = 0;

         //////////// Unpack the received buffers ////////////

         HYPRE_Int **A_tmp_info = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_recv_procs; i++)
         {
            recv_map[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            recv_redundant_marker[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
            num_recv_nodes[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);

            hypre_BoomerAMGDD_UnpackRecvBuffer(recv_buffer[i], amgdd_data, A_tmp_info, recv_redundant_marker, &(recv_map_send_buffer_size[i]), nodes_added_on_level, level, i);

            recv_map_send_buffer[i] = hypre_CTAlloc(HYPRE_Int, recv_map_send_buffer_size[i], HYPRE_MEMORY_HOST);
            hypre_BoomerAMGDD_PackRecvMapSendBuffer(recv_map_send_buffer[i], recv_redundant_marker[level][i], num_recv_nodes[level][i], &(recv_buffer_size[level][i]), level, num_levels);
         }

         //////////// Setup local indices for the composite grid ////////////

         hypre_AMGDDCompGridSetupLocalIndices(compGrid, nodes_added_on_level, recv_map, num_recv_procs, A_tmp_info, level, num_levels);
         for (j = level; j < num_levels; j++) nodes_added_on_level[j] = 0;

         //////////// Communicate redundancy info ////////////

         // post receives for send maps
         for (i = 0; i < num_send_procs; i++)
         {
            send_flag_buffer[i] = hypre_CTAlloc(HYPRE_Int, send_flag_buffer_size[i], HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv( send_flag_buffer[i], send_flag_buffer_size[i], HYPRE_MPI_INT, hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 2, comm, &(requests[request_counter++]));
         }

         // send the recv_map_send_buffer's
         for (i = 0; i < num_recv_procs; i++)
         {
            hypre_MPI_Isend( recv_map_send_buffer[i], recv_map_send_buffer_size[i], HYPRE_MPI_INT, hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 2, comm, &(requests[request_counter++]));
         }

         // wait for maps to be received
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, requests, status );
         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);

         // unpack and setup the send flag arrays
         for (i = 0; i < num_send_procs; i++)
         {
            hypre_BoomerAMGDD_UnpackSendFlagBuffer(compGrid, send_flag_buffer[i], send_flag[level][i], num_send_nodes[level][i], &(send_buffer_size[level][i]), level, num_levels);
         }

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
   }

   /////////////////////////////////////////////////////////////////

   // Done with loop over levels. Now just finalize things.

   /////////////////////////////////////////////////////////////////

   hypre_BoomerAMGDD_FixUpRecvMaps(compGrid, compGridCommPkg, recv_redundant_marker, amgdd_start_level, num_levels);

   // Communicate data for A and all info for P
   hypre_BoomerAMGDD_CommunicateRemainingMatrixInfo(amgdd_data);

   // Setup the local indices for P
   hypre_AMGDDCompGridSetupLocalIndicesP(amgdd_data);

   // Finalize the comp grid structures
   hypre_AMGDDCompGridFinalize(amgdd_data);

   // Setup extra info for specific relaxation methods
   hypre_AMGDDCompGridSetupRelax(amgdd_data);

   // Cleanup memory
   hypre_TFree(padding, HYPRE_MEMORY_HOST);
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

   return 0;
}


