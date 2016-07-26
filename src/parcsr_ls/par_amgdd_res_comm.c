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
#include "par_amg.h"
#include "par_csr_block_matrix.h"	

#define DEBUG_COMP_GRID 1 // if true, prints out what is stored in the comp grids for each processor to a file
#define USE_BARRIERS 0 // if true, puts MPI barriers between major tasks in setup phase (for timing purposes)

HYPRE_Int 
GeneratePsiComposite( hypre_ParCompGrid **psiComposite, hypre_ParCompGrid **compGrid, hypre_ParCSRCommPkg *commPkg, HYPRE_Int *send_flag_buffer_size, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
GetBufferSize( hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels );

HYPRE_Complex*
PackSendBuffer( hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels, HYPRE_Int buffer_size );

HYPRE_Int
UnpackRecvBuffer( HYPRE_Complex *recv_buffer, hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
AddToCompGrid(hypre_ParCompGrid **compGrid, hypre_ParCompGrid **psiComposite, HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_size, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels, HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index, HYPRE_Int *num_added_nodes );

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_send_buffer, hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels);

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
   HYPRE_Int numGhostLayers = 3;
   #if DEBUG_COMP_GRID
   char filename[256];
   #endif

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // hypre_printf("Began comp grid setup on rank %d\n", myid);

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
   hypre_ParCSRCommPkg        *commPkg;
   HYPRE_Int                  **CF_marker_array;
   HYPRE_Int                  *proc_first_index, *proc_last_index, *global_nodes;
   hypre_IJAssumedPart        **apart;

   // composite grids and temporary psiComposite grids used for communication
   hypre_ParCompGrid          **compGrid;
   hypre_ParCompGrid          ***psiComposite_send;
   hypre_ParCompGrid          ***psiComposite_recv;

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
   proc_first_index = hypre_CTAlloc(HYPRE_Int, num_levels);
   proc_last_index = hypre_CTAlloc(HYPRE_Int, num_levels);
   global_nodes = hypre_CTAlloc(HYPRE_Int, num_levels);
   num_added_nodes = hypre_CTAlloc(HYPRE_Int, num_levels);
   apart = hypre_CTAlloc(hypre_IJAssumedPart*, num_levels);
   for (level = 0; level < num_levels; level++)
   {
      proc_first_index[level] = hypre_ParVectorFirstIndex(F_array[level]);
      proc_last_index[level] = hypre_ParVectorLastIndex(F_array[level]);
      global_nodes[level] = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
      apart[level] = hypre_ParCSRMatrixAssumedPartition(A_array[level]);
   }

   // Allocate space for some variables that store info on each level
   compGrid = hypre_CTAlloc(hypre_ParCompGrid*, num_levels);
   compGridCommPkg = hypre_ParCompGridCommPkgCreate();
   hypre_ParCompGridCommPkgNumLevels(compGridCommPkg) = num_levels;
   hypre_ParCompGridCommPkgNumSends(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels);
   hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels);
   hypre_ParCompGridCommPkgSendProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels);
   hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels);
   hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels);
   hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels);
   send_buffer_size = hypre_CTAlloc(HYPRE_Int*, num_levels);
   recv_buffer_size = hypre_CTAlloc(HYPRE_Int*, num_levels);
   send_flag = hypre_CTAlloc(HYPRE_Int***, num_levels);
   num_send_nodes = hypre_CTAlloc(HYPRE_Int**, num_levels);
   recv_map = hypre_CTAlloc(HYPRE_Int***, num_levels);
   numGhostFromProc =  hypre_CTAlloc(HYPRE_Int*, num_procs);
   ghostInfoOffset =  hypre_CTAlloc(HYPRE_Int*, num_procs);
   ghostGlobalIndex = hypre_CTAlloc(HYPRE_Int**, num_procs);
   numNewGhostNodes = hypre_CTAlloc(HYPRE_Int, num_levels);
   ghostUnpackIndex = hypre_CTAlloc(HYPRE_Int**, num_procs);
   for (i = 0; i < num_procs; i++)
   {
      numGhostFromProc[i] = hypre_CTAlloc(HYPRE_Int, num_levels);
      ghostInfoOffset[i] = hypre_CTAlloc(HYPRE_Int, num_levels);
      ghostGlobalIndex[i] = hypre_CTAlloc(HYPRE_Int*, num_levels);
      ghostUnpackIndex[i] = hypre_CTAlloc(HYPRE_Int*, num_levels);
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

   #if DEBUG_COMP_GRID
   for (k = 0; k < num_levels; k++)
   {
      hypre_sprintf(filename, "../../../scratch/CompGrids/compGridAfterInitRank%dLevel%d.txt", myid, k);
      hypre_ParCompGridDebugPrint( compGrid[k], filename );
   }
   #endif

   // Now that the comp grids are initialized, store RHS back in F_array[0]
   hypre_ParVectorCopy(Vtemp,F_array[0]);

   // For the given padding, eta, need to compute A^eta to determine processor neighbors of degree eta
   hypre_ParCSRMatrix **A_eta_array = hypre_CTAlloc(hypre_ParCSRMatrix*, num_levels);
   if (padding > 1)
   {
      for (level = 0; level < num_levels; level++)
      {
         A_eta_array[level] = hypre_ParMatmul(A_array[level], A_array[level]);
         for (i = 0; i < padding - 2; i++)
         {
            hypre_ParCSRMatrix *old_matrix = A_eta_array[level];
            A_eta_array[level] = hypre_ParMatmul(A_array[level], old_matrix);
            hypre_ParCSRMatrixDestroy(old_matrix); // Cleanup old matrices to prevent memory leak
         }
         // Create the commpkg for A^eta on this level
         hypre_MatvecCommPkgCreate ( A_eta_array[level] );
      }
   }
   else A_eta_array = A_array;

   // On the coarsest level, need to make sure eta is large enough such that all procs send all nodes to other procs with nodes
   HYPRE_Int need_to_expand_stencil = 1;
   while (need_to_expand_stencil)
   {
      need_to_expand_stencil = 0;
      commPkg =  hypre_ParCSRMatrixCommPkg( A_eta_array[num_levels - 1] );
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends( commPkg );
      for (i = 0; i < num_sends; i++)
      {
         HYPRE_Int start = hypre_ParCSRCommPkgSendMapStart(commPkg, i);
         HYPRE_Int finish = hypre_ParCSRCommPkgSendMapStart(commPkg, i+1);
         if ( (finish - start) != hypre_ParCSRMatrixNumRows( A_eta_array[num_levels - 1] ) ) need_to_expand_stencil = 1;
      }
      if (need_to_expand_stencil)
      {
         printf("Expanding stencil on coarsest level\n");
         hypre_ParCSRMatrix *old_matrix = A_eta_array[num_levels - 1];
         A_eta_array[num_levels - 1] = hypre_ParMatmul(A_array[num_levels - 1], old_matrix);
         if (old_matrix != A_array[num_levels - 1]) hypre_ParCSRMatrixDestroy(old_matrix);
         hypre_MatvecCommPkgCreate ( A_eta_array[num_levels - 1] );              
      }
   }

   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   for (level = num_levels-1; level > -1; level--)
   {
      if ( proc_last_index[level] >= proc_first_index[level] ) // If there are any owned nodes on this level
      {
         // Get the commPkg of matrix A^eta on this level
         commPkg = hypre_ParCSRMatrixCommPkg(A_eta_array[level]);
         comm = hypre_ParCSRCommPkgComm(commPkg);
         num_sends = hypre_ParCSRCommPkgNumSends(commPkg);
         num_recvs = hypre_ParCSRCommPkgNumRecvs(commPkg);

         // Copy over some info for comp grid comm pkg
         hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level] = num_sends;
         hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level] = num_recvs;
         hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, num_sends);
         hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, num_recvs);
         hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, num_sends + 1);
         hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStarts(commPkg)[num_sends] );
         memcpy( hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level], hypre_ParCSRCommPkgSendProcs(commPkg), num_sends * sizeof(HYPRE_Int) );
         memcpy( hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level], hypre_ParCSRCommPkgRecvProcs(commPkg), num_sends * sizeof(HYPRE_Int) );
         memcpy( hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level], hypre_ParCSRCommPkgSendMapStarts(commPkg), (num_sends + 1)*sizeof(HYPRE_Int) );
         memcpy( hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level], hypre_ParCSRCommPkgSendMapElmts(commPkg), hypre_ParCSRCommPkgSendMapStarts(commPkg)[num_sends]*sizeof(HYPRE_Int) );

         // allocate space for the buffers, buffer sizes, requests and status, psiComposite_send, psiComposite_recv, send and recv maps
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs );
         request_counter = 0;
         send_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_sends);
         recv_buffer_size[level] = hypre_CTAlloc(HYPRE_Int, num_recvs);
         send_buffer = hypre_CTAlloc(HYPRE_Complex*, num_sends);
         recv_buffer = hypre_CTAlloc(HYPRE_Complex*, num_recvs);
         psiComposite_send = hypre_CTAlloc(hypre_ParCompGrid**, num_sends);
         psiComposite_recv = hypre_CTAlloc(hypre_ParCompGrid**, num_recvs);
         num_psi_levels_send = hypre_CTAlloc(HYPRE_Int, num_sends);
         num_psi_levels_recv = hypre_CTAlloc(HYPRE_Int, num_recvs);

         send_flag[level] = hypre_CTAlloc(HYPRE_Int**, num_sends);
         num_send_nodes[level] = hypre_CTAlloc(HYPRE_Int*, num_sends);
         recv_map[level] = hypre_CTAlloc(HYPRE_Int**, num_recvs);
         recv_map_send = hypre_CTAlloc(HYPRE_Int**, num_recvs);
         recv_map_size = hypre_CTAlloc(HYPRE_Int*, num_recvs);
         send_flag_buffer = hypre_CTAlloc(HYPRE_Int*, num_sends);
         send_flag_buffer_size = hypre_CTAlloc(HYPRE_Int, num_sends);
         recv_map_send_buffer = hypre_CTAlloc(HYPRE_Int*, num_recvs);
         recv_map_send_buffer_size = hypre_CTAlloc(HYPRE_Int, num_recvs);

         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

         if (timers) hypre_BeginTiming(timers[1]);

         // loop over send procs
         for (i = 0; i < num_sends; i++)
         {
            // allocate space for psiComposite_send
            psiComposite_send[i] = hypre_CTAlloc(hypre_ParCompGrid*, num_levels);

            // generate psiComposite
            num_psi_levels_send[i] = GeneratePsiComposite( psiComposite_send[i], compGrid, commPkg, &(send_flag_buffer_size[i]), i, level, num_levels );
         }

         if (timers) hypre_EndTiming(timers[1]);

         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

         if (timers) hypre_BeginTiming(timers[2]);

         // post the receives for the buffer size
         for (i = 0; i < num_recvs; i++)
         {
            hypre_MPI_Irecv( &(recv_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCSRCommPkgRecvProc(commPkg, i), 0, comm, &requests[request_counter++] );
         }

         // send the buffer sizes
         for (i = 0; i < num_sends; i++)
         {
            send_buffer_size[level][i] = GetBufferSize( psiComposite_send[i], level, num_levels, num_psi_levels_send[i] );
            hypre_MPI_Isend(&(send_buffer_size[level][i]), 1, HYPRE_MPI_INT, hypre_ParCSRCommPkgSendProc(commPkg, i), 0, comm, &requests[request_counter++]);
         }
         
         // wait for all buffer sizes to be received
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );

         if (timers) hypre_EndTiming(timers[2]);


         // free and reallocate space for the requests and status
         hypre_TFree(requests);
         hypre_TFree(status);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs );
         request_counter = 0;

         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

         if (timers) hypre_BeginTiming(timers[3]);

         // allocate space for the receive buffers and post the receives
         for (i = 0; i < num_recvs; i++)
         {
            recv_buffer[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size[level][i] );
            hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_COMPLEX, hypre_ParCSRCommPkgRecvProc(commPkg, i), 1, comm, &requests[request_counter++]);
         }

         // pack and send the buffers
         for (i = 0; i < num_sends; i++)
         {
            send_buffer[i] = PackSendBuffer( psiComposite_send[i], level, num_levels, num_psi_levels_send[i], send_buffer_size[level][i] );
            hypre_MPI_Isend(send_buffer[i], send_buffer_size[level][i], HYPRE_MPI_COMPLEX, hypre_ParCSRCommPkgSendProc(commPkg, i), 1, comm, &requests[request_counter++]);
         }

         // wait for buffers to be received
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );

         if (timers) hypre_EndTiming(timers[3]);

         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

         if (timers) hypre_BeginTiming(timers[4]);

         // loop over received buffers
         for (i = 0; i < num_recvs; i++)
         {
            // unpack the buffers
            psiComposite_recv[i] = hypre_CTAlloc(hypre_ParCompGrid*, num_levels);
            num_psi_levels_recv[i] = UnpackRecvBuffer( recv_buffer[i], psiComposite_recv[i], level, num_levels );

            // allocate space for the recv map info
            recv_map_send[i] = hypre_CTAlloc(HYPRE_Int*, num_levels);
            recv_map_size[i] = hypre_CTAlloc(HYPRE_Int, num_levels);
            for (j = level; j < num_levels; j++)
            {
               if ( psiComposite_recv[i][j] ) recv_map_send[i][j] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(psiComposite_recv[i][j]) );               
            }

            // and add information to this composite grid
            AddToCompGrid(compGrid, psiComposite_recv[i], recv_map_send[i], recv_map_size[i], &(recv_map_send_buffer_size[i]), level, num_levels, num_psi_levels_recv[i], proc_first_index, proc_last_index, num_added_nodes );
         }

         // Setup local indices for the composite grid
         hypre_ParCompGridSetupLocalIndices(compGrid, num_added_nodes, num_levels, proc_first_index, proc_last_index);

         // Zero out num_added_nodes
         for (i = level; i < num_levels; i++) num_added_nodes[i] = 0;

         if (timers) hypre_EndTiming(timers[4]);

         #if DEBUG_COMP_GRID
         for (k = 0; k < num_levels; k++)
         {
            hypre_sprintf(filename, "../../../scratch/CompGrids/compGridAfterLevel%dRank%dLevel%d.txt", level, myid, k);
            hypre_ParCompGridDebugPrint( compGrid[k], filename );
         }
         #endif

         // If on the finest level, figure out ghost node info
         if (level == 0)
         {
            #if USE_BARRIERS
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            #endif

            if (timers) hypre_BeginTiming(timers[8]);

            for (k = 0; k < numGhostLayers; k++)
            {
               // Figure out what ghost nodes are needed (1 layer at a time)
               FindGhostNodes(compGrid, num_levels, proc_first_index, proc_last_index, numGhostFromProc, ghostInfoOffset, ghostGlobalIndex, numNewGhostNodes, ghostUnpackIndex, global_nodes, apart);
               // Communicate the ghost nodes and setup local indices
               CommunicateGhostNodes(compGrid, numGhostFromProc, ghostInfoOffset, ghostGlobalIndex, numNewGhostNodes, ghostUnpackIndex, num_levels, global_nodes);
               hypre_ParCompGridSetupLocalIndices(compGrid, numNewGhostNodes, num_levels, proc_first_index, proc_last_index);
               // Increment ghostInfoOffset and reset numGhostFromProc for the next iteration 
               for (j = 0; j < num_levels; j++)
               {
                  numNewGhostNodes[j] = 0;
                  for (i = 0; i < num_procs; i++)
                  {
                     ghostInfoOffset[i][j] += numGhostFromProc[i][j];
                     numGhostFromProc[i][j] = 0;
                  }
               }
               #if DEBUG_COMP_GRID
               for (j = 0; j < num_levels; j++)
               {
                  hypre_sprintf(filename, "../../../scratch/CompGrids/compGridAfterGhost%dRank%dLevel%d.txt", k+1, myid, j);
                  hypre_ParCompGridDebugPrint( compGrid[j], filename );
               }
               #endif
            }

            if (timers) hypre_EndTiming(timers[8]);
         }


         // free and reallocate space for the requests and status
         hypre_TFree(requests);
         hypre_TFree(status);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs );
         request_counter = 0;

         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

         if (timers) hypre_BeginTiming(timers[5]);         

         // post receives for send maps - NOTE: we want to receive this info from procs we sent to
         for (i = 0; i < num_sends; i++)
         {
            send_flag_buffer[i] = hypre_CTAlloc(HYPRE_Int, send_flag_buffer_size[i]);
            hypre_MPI_Irecv( send_flag_buffer[i], send_flag_buffer_size[i], HYPRE_MPI_INT, hypre_ParCSRCommPkgSendProc(commPkg, i), 2, comm, &requests[request_counter++]);
         }

         // send recv_map_send to procs received from to become their send maps - NOTE: we want to send this info from procs we received from
         for (i = 0; i < num_recvs; i++)
         {
            // pack up the recv_map_send's and send them
            recv_map_send_buffer[i] = hypre_CTAlloc(HYPRE_Int, recv_map_send_buffer_size[i]);
            PackRecvMapSendBuffer(recv_map_send[i], recv_map_send_buffer[i], psiComposite_recv[i], level, num_levels);
            hypre_MPI_Isend( recv_map_send_buffer[i], recv_map_send_buffer_size[i], HYPRE_MPI_INT, hypre_ParCSRCommPkgRecvProc(commPkg, i), 2, comm, &requests[request_counter++]);
         }

         // wait for maps to be received
         hypre_MPI_Waitall( num_sends + num_recvs, requests, status );

         // unpack and setup the send flag arrays
         for (i = 0; i < num_sends; i++)
         {
            send_flag[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels);
            num_send_nodes[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels);
            UnpackSendFlagBuffer(send_flag_buffer[i], send_flag[level][i], &(send_buffer_size[level][i]), num_send_nodes[level][i], level, num_levels);
         }

         if (timers) hypre_EndTiming(timers[5]);

         // finalize the recv maps and get final recv buffer size
         for (i = 0; i < num_recvs; i++)
         {
            // buffers will store number of nodes on each level
            recv_buffer_size[level][i] = num_levels - level;

            // allocate space for each level of the receive map for this proc
            recv_map[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels);

            // for each level
            for (j = level; j < num_levels; j++)
            {
               // if there is info for this proc on this level
               if (recv_map_send[i][j])
               {
                  // allocate the appropriate amount of space for the map
                  recv_map[level][i][j] = hypre_CTAlloc(HYPRE_Int, recv_map_size[i][j]);
                  cnt = 0;

                  for (k = 0; k < hypre_ParCompGridNumNodes(psiComposite_recv[i][j]); k++)
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
         hypre_TFree(requests);
         hypre_TFree(status);
         for (i = 0; i < num_sends; i++)
         {
            hypre_TFree(send_buffer[i]);
            hypre_TFree(send_flag_buffer[i]);
            for (j = 0; j < num_levels; j++)
            {
               if (psiComposite_send[i][j]) hypre_ParCompGridDestroy(psiComposite_send[i][j]);
            }
            hypre_TFree(psiComposite_send[i]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            hypre_TFree(recv_buffer[i]);
            hypre_TFree(recv_map_send_buffer[i]);
            hypre_TFree(recv_map_size[i]);
            for (j = 0; j < num_levels; j++)
            {
               if (psiComposite_recv[i][j]) hypre_ParCompGridDestroy(psiComposite_recv[i][j]);
               if (recv_map_send[i][j]) hypre_TFree(recv_map_send[i][j]);
            }
            hypre_TFree(psiComposite_recv[i]);
            hypre_TFree(recv_map_send[i]);
         }
         hypre_TFree(send_buffer);
         hypre_TFree(psiComposite_send);
         hypre_TFree(recv_buffer);
         hypre_TFree(psiComposite_recv);
         hypre_TFree(recv_map_send);
         hypre_TFree(send_flag_buffer);
         hypre_TFree(send_flag_buffer_size);
         hypre_TFree(recv_map_send_buffer);
         hypre_TFree(recv_map_send_buffer_size);
         hypre_TFree(recv_map_size);
         hypre_TFree(num_psi_levels_send);
         hypre_TFree(num_psi_levels_recv);
      }
      else 
      {
         #if USE_BARRIERS
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif
      }
   }


   #if DEBUG_COMP_GRID
   for (level = 0; level < num_levels; level++)
   {
      hypre_sprintf(filename, "../../../scratch/CompGrids/setupCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrint( compGrid[level], filename );
      if (myid == 0)
      {
         FILE             *file;
         hypre_sprintf(filename,"../../../scratch/CompGrids/global_num_nodes.txt");
         file = fopen(filename,"w");
         hypre_fprintf(file, "%d\n", hypre_ParCSRMatrixGlobalNumRows(A_array[0]) );
         fclose(file);
         // Print info on how to read files
         hypre_sprintf(filename,"../../../scratch/CompGrids/info.txt");
         file = fopen(filename,"w");
         hypre_fprintf(file, "num_nodes\nmem_size\nnum_owned_nodes\nnum_real_nodes\nsolution values\nresidual values\nglobal indices\ncoarse global indices\ncoarse local indices\nrows of matrix A: size, data, global indices, local indices\nrows of matrix P: size, data, global indices, local indices\nghost P rows: size, data, global indices, local indices\n");
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

   // hypre_printf("Finished comp grid setup on rank %d\n", myid);

   // Cleanup memory
   for (i = 1; i < num_procs; i++) hypre_TFree(numGhostFromProc[i]);
   hypre_TFree(numGhostFromProc);
   hypre_TFree(proc_first_index);
   hypre_TFree(proc_last_index);
   if (padding > 1)
   {
      for (level = 0; level < num_levels; level++)
      {
         hypre_ParCSRMatrixDestroy( A_eta_array[level] );
      }
      hypre_TFree( A_eta_array );
   }
   
   return 0;
}  


HYPRE_Int 
hypre_BoomerAMGDDResidualCommunication( void *amg_vdata )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // hypre_printf("Began residual communication on rank %d\n", myid);

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
   proc_first_index = hypre_CTAlloc(HYPRE_Int, num_levels);
   proc_last_index = hypre_CTAlloc(HYPRE_Int, num_levels);
   global_nodes = hypre_CTAlloc(HYPRE_Int, num_levels);
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
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs );
         request_counter = 0;
         send_buffer = hypre_CTAlloc(HYPRE_Complex*, num_sends);
         recv_buffer = hypre_CTAlloc(HYPRE_Complex*, num_recvs);


         // allocate space for the receive buffers and post the receives
         for (i = 0; i < num_recvs; i++)
         {
            recv_buffer[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size[level][i] );
            hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_COMPLEX, recv_procs[level][i], 0, comm, &requests[request_counter++]);
         }

         // pack and send the buffers
         for (i = 0; i < num_sends; i++)
         {
            send_buffer[i] = hypre_CTAlloc(HYPRE_Complex, send_buffer_size[level][i]);
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
         hypre_TFree(requests);
         hypre_TFree(status);
         for (i = 0; i < num_sends; i++)
         {
            hypre_TFree(send_buffer[i]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            hypre_TFree(recv_buffer[i]);
         }
         hypre_TFree(send_buffer);
         hypre_TFree(recv_buffer);
      }
   }

   // Communicate ghost residuals
   HYPRE_Int **numGhostFromProc = hypre_ParCompGridCommPkgNumGhostFromProc(compGridCommPkg);
   HYPRE_Int ***ghostGlobalIndex = hypre_ParCompGridCommPkgGhostGlobalIndex(compGridCommPkg);
   HYPRE_Int ***ghostUnpackIndex = hypre_ParCompGridCommPkgGhostUnpackIndex(compGridCommPkg);
   CommunicateGhostNodesResidualOnly(compGrid, numGhostFromProc, ghostGlobalIndex, ghostUnpackIndex, num_levels, global_nodes);

   #if DEBUG_COMP_GRID
   char filename[256];
   for (level = 0; level < num_levels; level++)
   {
      hypre_sprintf(filename, "../../../scratch/CompGrids/communicateCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrint( compGrid[level], filename );
   }
   #endif

   // store communication info in compGridCommPkg
   hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg) = send_buffer_size;
   hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg) = recv_buffer_size;
   hypre_ParCompGridCommPkgSendFlag(compGridCommPkg) = send_flag;
   hypre_ParCompGridCommPkgRecvMap(compGridCommPkg) = recv_map;

   // assign compGrid and compGridCommPkg info to the amg structure
   hypre_ParAMGDataCompGrid(amg_data) = compGrid;
   hypre_ParAMGDataCompGridCommPkg(amg_data) = compGridCommPkg;

   // hypre_printf("Finished residual communication on rank %d\n", myid);

   // Cleanup memory
   hypre_TFree(proc_first_index);
   hypre_TFree(proc_last_index);
   
   return 0;
}

HYPRE_Int
GeneratePsiComposite( hypre_ParCompGrid **psiComposite, hypre_ParCompGrid **compGrid, hypre_ParCSRCommPkg *commPkg, HYPRE_Int *send_flag_buffer_size, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels )
{
   HYPRE_Int                  level,i,j,cnt = 0;
   HYPRE_Int                  send_elmt;
   HYPRE_Int                  row_size;
   HYPRE_Int                  nodes_to_add = 0, coarse_grid_index, need_coarse_info;
   HYPRE_Int                  **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels );
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
   psiComposite[current_level] = hypre_ParCompGridCreate();
   hypre_ParCompGridSetSize(psiComposite[current_level], finish - start, need_coarse_info);
   if (need_coarse_info) add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[current_level+1]) );

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
         psiComposite[level] = hypre_ParCompGridCreate();
         hypre_ParCompGridSetSize(psiComposite[level], nodes_to_add, need_coarse_info);

         // if we need coarse info, allocate space for the add flag on the next level
         if (need_coarse_info)
         {
            add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]) );
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
      hypre_TFree(add_flag[level]);
   }
   hypre_TFree(add_flag);

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
   HYPRE_Complex     *send_buffer = hypre_CTAlloc(HYPRE_Complex, buffer_size);

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
      psiComposite[level] = hypre_ParCompGridCreate();

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
         hypre_ParCompMatrixRowData( hypre_ParCompGridARows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Complex, row_size);
         hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Int, row_size);
         hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridARows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Int, row_size);

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
            hypre_ParCompMatrixRowData( hypre_ParCompGridPRows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Complex, row_size);
            hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridPRows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Int, row_size);
            hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridPRows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Int, row_size);

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

      // increment num_nodes and num_real_nodes
      hypre_ParCompGridNumNodes(compGrid[current_level]) = ++num_nodes;
      hypre_ParCompGridNumRealNodes(compGrid[current_level]) = hypre_ParCompGridNumNodes(compGrid[current_level]);

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
         hypre_ParCompGridNumRealNodes(compGrid[level]) = hypre_ParCompGridNumNodes(psiComposite[level]);

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
            // we will search over the global indices NOT owned by this proc (nodes owned by this proc will already be accounted for)
            // this corresponds to local indices >= num owned nodes
            add_flag = 0;
            if ( hypre_ParCompGridGlobalIndices(psiComposite[level])[i] < proc_first_index[level] || hypre_ParCompGridGlobalIndices(psiComposite[level])[i] > proc_last_index[level] )
            {
               add_flag = 1;
               // search over nodes added to this comp grid (i.e. those with local index greater than num_owned_nodes)
               for (j = hypre_ParCompGridNumOwnedNodes(compGrid[level]); j < hypre_ParCompGridNumNodes(compGrid[level]); j++)
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
               hypre_ParCompGridNumRealNodes(compGrid[level]) = hypre_ParCompGridNumNodes(compGrid[level]);

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
PackRecvMapSendBuffer(HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_send_buffer, hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels)
{
   HYPRE_Int      level, i, cnt;
   HYPRE_Int      num_nodes;

   cnt = 0;
   for (level = current_level; level < num_levels; level++)
   {
      // if there were nodes in psiComposite on this level
      if (recv_map_send[level])
      {
         // get num nodes on this level
         num_nodes = hypre_ParCompGridNumNodes(psiComposite[level]);

         // store the number of nodes on this level
         recv_map_send_buffer[cnt++] = num_nodes;

         for (i = 0; i < num_nodes; i++)
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

      if (num_nodes) send_flag[level] = hypre_CTAlloc(HYPRE_Int, num_nodes);

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
   HYPRE_Int                  **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels );
   hypre_ParCompMatrixRow     *row;

   // Get where to look in commPkgSendMapElmts
   HYPRE_Int            start = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][processor];
   HYPRE_Int            finish = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][processor+1];

   // see whether we need coarse info and if so, set up add_flag
   if (current_level != num_levels-1) need_coarse_info = 1;
   else need_coarse_info = 0;
   if (need_coarse_info) add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[current_level+1]) );

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
            add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]) );
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
      hypre_TFree(add_flag[level]);
   }
   hypre_TFree(add_flag);

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

HYPRE_Int
CommunicateGhostNodes(hypre_ParCompGrid **compGrid, HYPRE_Int **numGhostFromProc, HYPRE_Int **ghostInfoOffset, HYPRE_Int ***ghostGlobalIndex, HYPRE_Int *numNewGhostNodes, HYPRE_Int ***ghostUnpackIndex, HYPRE_Int num_levels, HYPRE_Int *global_nodes )
{
   HYPRE_Int         num_procs;
   HYPRE_Int         totalNewGhostNodes = 0;
   HYPRE_Int         level,i,j,add_flag;
   HYPRE_Int         num_contacts, contact_obj_size, response_obj_size, max_response_size;
   HYPRE_Int         *contact_proc_list, *contact_send_buf, *contact_send_buf_starts, *response_buf_starts;
   HYPRE_Complex     *response_buf;
   hypre_DataExchangeResponse response_obj;

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   contact_obj_size = sizeof(HYPRE_Int);
   response_obj_size = sizeof(HYPRE_Complex);

   contact_proc_list = hypre_CTAlloc(HYPRE_Int, num_procs);

   num_contacts = 0;
   for (i = 0; i < num_procs; i++)
   {
      add_flag = 0;
      for (j = 0; j < num_levels; j++)
      {
         if (numGhostFromProc[i][j]) add_flag = 1;
      }
      if (add_flag)
      {
         contact_proc_list[num_contacts++] = i;
      }
   }

   max_response_size = 0; // !!! NOTE: THIS MUST BE THE SAME ON ALL PROCS! THIS HAS BITTEN YOU BEFORE, SO DON'T CHANGE IT UNLESS YOU MAKE SURE THE THING YOU USE IS THE SAME ON ALL PROCS
                                          // THAT IS, max_response_size on proc 0 is the same value as max_response_size on proc 1... HAS TO BE THIS WAY! OR IT WILL BOMB!
                                          // !!! Estimated of max response size... how big should this really be? !!! 
                                          // Right now assuming 2D uniform square mesh... 
   for (level = 0; level < num_levels; level++) 
   {
      totalNewGhostNodes += numNewGhostNodes[level]; // count the total number of ghost nodes to request (needed to allocate contact_send_buf)
      max_response_size += 31 * 2 * ( sqrt( global_nodes[level] ) );
                        // (estimated max amount of info per node) * (number of new ghost nodes)
                        // !!! Assumes a 9pt stencil for A and a 5pt stencil for P when calculating amount of info per node!!!
                        // !!! Assumes 2D domain: estimating number of new ghost nodes by taking length of 1 side of the domain and multiplying by 2
   }

   // printf("max_response_size = %d, 31*totalNewGhostNodes = %d\n", max_response_size, 31*totalNewGhostNodes);

   contact_proc_list = hypre_TReAlloc(contact_proc_list, HYPRE_Int, num_contacts);
   contact_send_buf_starts = hypre_CTAlloc(HYPRE_Int, num_contacts+1);
   contact_send_buf = hypre_CTAlloc( HYPRE_Int, totalNewGhostNodes + (num_levels * num_contacts) + num_contacts ); // entries for each requested row, num rows on each level on each proc, and num levels for each proc

   PackGhostNodeContact( num_levels, num_contacts, contact_proc_list, numGhostFromProc, ghostInfoOffset, ghostGlobalIndex, contact_send_buf, contact_send_buf_starts);
   response_obj.fill_response = FillGhostNodeResponse;
   response_obj.data1 = compGrid;
   response_obj.data2 = NULL;

   hypre_DataExchangeList( num_contacts, contact_proc_list, contact_send_buf, contact_send_buf_starts, contact_obj_size, 
      response_obj_size, &response_obj, max_response_size, 1, hypre_MPI_COMM_WORLD, (void**) (&response_buf), &response_buf_starts);

   UnpackGhostNodeResponse(compGrid, num_levels, num_contacts, contact_proc_list, response_buf, numNewGhostNodes, ghostInfoOffset, ghostUnpackIndex);

   // Clean up memory
   hypre_TFree(contact_proc_list);
   hypre_TFree(contact_send_buf);
   hypre_TFree(contact_send_buf_starts);
   hypre_TFree(response_buf);
   hypre_TFree(response_buf_starts);

   return 0;
}

HYPRE_Int
FillGhostNodeResponse(void* recvBuf, int contact_size, int contact_proc, void* ro, MPI_Comm comm, void** responseBuf, int* response_message_size)
{
   HYPRE_Int num_levels, numRequestedNodes, row_size;
   HYPRE_Int level, i, j, recv_cnt, send_cnt;
   hypre_ParCompMatrixRow *row;
   hypre_DataExchangeResponse *response_obj = ro;
   hypre_ParCompGrid **compGrid = response_obj->data1;
   HYPRE_Int *recv_buf = (HYPRE_Int*) recvBuf;
   HYPRE_Complex *response_buf = (HYPRE_Complex*) *responseBuf; // !!! do I know whether this is allocated large enough or do I have to check size in this function and realloc where necessary? !!!

   recv_cnt = 0;
   send_cnt = 0;
   num_levels = recv_buf[recv_cnt++];
   // For each level...
   for (level = 0; level < num_levels; level++)
   {
      // Pack number of nodes requested on this level
      numRequestedNodes = recv_buf[recv_cnt++];
      response_buf[send_cnt++] = (HYPRE_Complex) numRequestedNodes;
      // Then for each node...
      for (i = 0; i < numRequestedNodes; i++)
      {
         // Get the local index of ghost node to send (incoming index is a global index)
         HYPRE_Int localIndex = recv_buf[recv_cnt++] - hypre_ParCompGridGlobalIndices(compGrid[level])[0];

         // Pack the residual value and global index
         response_buf[send_cnt++] = hypre_ParCompGridF(compGrid[level])[ localIndex ];
         response_buf[send_cnt++] = hypre_ParCompGridGlobalIndices(compGrid[level])[ localIndex ];
         response_buf[send_cnt++] = hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ localIndex ];

         // Pack the info for the row of A
         row = hypre_ParCompGridARows(compGrid[level])[ localIndex ];
         row_size = hypre_ParCompMatrixRowSize(row);
         response_buf[send_cnt++] = (HYPRE_Complex) row_size;
         for (j = 0; j < row_size; j++)
         {
            response_buf[send_cnt++] = hypre_ParCompMatrixRowData(row)[j];
         }
         for (j = 0; j < row_size; j++)
         {
            response_buf[send_cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices(row)[j];
         }

         // Pack the info for the row of P
         row = hypre_ParCompGridPRows(compGrid[level])[ localIndex ];
         row_size = hypre_ParCompMatrixRowSize(row);
         response_buf[send_cnt++] = (HYPRE_Complex) row_size;
         for (j = 0; j < row_size; j++)
         {
            response_buf[send_cnt++] = hypre_ParCompMatrixRowData(row)[j];
         }
         for (j = 0; j < row_size; j++)
         {
            response_buf[send_cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices(row)[j];
         }
      }
   }

   *response_message_size = send_cnt;

   return 0;
}

HYPRE_Int
CommunicateGhostNodesResidualOnly(hypre_ParCompGrid **compGrid, HYPRE_Int **numGhostFromProc, HYPRE_Int ***ghostGlobalIndex, HYPRE_Int ***ghostUnpackIndex, HYPRE_Int num_levels, HYPRE_Int *global_nodes )
{
   HYPRE_Int         num_procs;
   HYPRE_Int         totalNewGhostNodes = 0;
   HYPRE_Int         level,i,j,add_flag;
   HYPRE_Int         num_contacts, contact_obj_size, response_obj_size, max_response_size;
   HYPRE_Int         *contact_proc_list, *contact_send_buf, *contact_send_buf_starts, *response_buf_starts;
   HYPRE_Complex     *response_buf;
   hypre_DataExchangeResponse response_obj;

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   contact_obj_size = sizeof(HYPRE_Int);
   response_obj_size = sizeof(HYPRE_Complex);

   contact_proc_list = hypre_CTAlloc(HYPRE_Int, num_procs);

   num_contacts = 0;
   for (i = 0; i < num_procs; i++)
   {
      add_flag = 0;
      for (j = 0; j < num_levels; j++)
      {
         if (numGhostFromProc[i][j]) add_flag = 1;
      }
      if (add_flag)
      {
         contact_proc_list[num_contacts++] = i;
      }
   }

   max_response_size = 0; // !!! NOTE: THIS MUST BE THE SAME ON ALL PROCS! THIS HAS BITTEN YOU BEFORE, SO DON'T CHANGE IT UNLESS YOU MAKE SURE THE THING YOU USE IS THE SAME ON ALL PROCS
                                          // THAT IS, max_response_size on proc 0 is the same value as max_response_size on proc 1... HAS TO BE THIS WAY! OR IT WILL BOMB!
                                          // !!! Estimated of max response size... how big should this really be? !!! 
                                          // Right now assuming 2D uniform square mesh... 
   for (level = 0; level < num_levels; level++) 
   {
      for (i = 0; i < num_procs; i++) totalNewGhostNodes += numGhostFromProc[i][level]; // count the total number of ghost nodes to request (needed to allocate contact_send_buf)
      max_response_size += num_levels + 1 + 12*( sqrt( global_nodes[level] / num_procs ) ); // !!! Estimated of max response size... how big should this really be? !!! 
                           // (estimated max amount of info per node) * (estimate num ghost nodes)
   }


   contact_proc_list = hypre_TReAlloc(contact_proc_list, HYPRE_Int, num_contacts);
   contact_send_buf_starts = hypre_CTAlloc(HYPRE_Int, num_contacts+1);
   contact_send_buf = hypre_CTAlloc( HYPRE_Int, totalNewGhostNodes + (num_levels * num_contacts) + num_contacts ); // entries for each requested row, num rows on each level on each proc, and num levels for each proc

   PackGhostNodeContact( num_levels, num_contacts, contact_proc_list, numGhostFromProc, NULL, ghostGlobalIndex, contact_send_buf, contact_send_buf_starts);
   response_obj.fill_response = FillGhostNodeResponseResidualOnly;
   response_obj.data1 = compGrid;
   response_obj.data2 = NULL;

   hypre_DataExchangeList( num_contacts, contact_proc_list, contact_send_buf, contact_send_buf_starts, contact_obj_size, 
      response_obj_size, &response_obj, max_response_size, 1, hypre_MPI_COMM_WORLD, (void**) (&response_buf), &response_buf_starts);

   UnpackGhostNodeResponseResidualOnly(compGrid, num_levels, num_contacts, contact_proc_list, response_buf, ghostUnpackIndex);

   // Clean up memory
   hypre_TFree(contact_proc_list);
   hypre_TFree(contact_send_buf);
   hypre_TFree(contact_send_buf_starts);
   hypre_TFree(response_buf);
   hypre_TFree(response_buf_starts);

   return 0;
}

HYPRE_Int
FillGhostNodeResponseResidualOnly(void* recvBuf, int contact_size, int contact_proc, void* ro, MPI_Comm comm, void** responseBuf, int* response_message_size)
{
   HYPRE_Int num_levels, numRequestedNodes;
   HYPRE_Int level, i, recv_cnt, send_cnt;
   hypre_DataExchangeResponse *response_obj = ro;
   hypre_ParCompGrid **compGrid = response_obj->data1;
   HYPRE_Int *recv_buf = (HYPRE_Int*) recvBuf;
   HYPRE_Complex *response_buf = (HYPRE_Complex*) *responseBuf; // !!! do I know whether this is allocated large enough or do I have to check size in this function and realloc where necessary? !!!

   recv_cnt = 0;
   send_cnt = 0;
   num_levels = recv_buf[recv_cnt++];
   // For each level...
   for (level = 0; level < num_levels; level++)
   {
      // Pack number of nodes requested on this level
      numRequestedNodes = recv_buf[recv_cnt++];
      response_buf[send_cnt++] = (HYPRE_Complex) numRequestedNodes;
      // Then for each node...
      for (i = 0; i < numRequestedNodes; i++)
      {
         // Get the local index of ghost node to send (incoming index is a global index)
         HYPRE_Int localIndex = recv_buf[recv_cnt++] - hypre_ParCompGridGlobalIndices(compGrid[level])[0];

         // Pack the residual value and global index
         response_buf[send_cnt++] = hypre_ParCompGridF(compGrid[level])[ localIndex ];
      }
   }

   *response_message_size = send_cnt;

   return 0;
}

HYPRE_Int
PackGhostNodeContact( HYPRE_Int num_levels, HYPRE_Int num_contacts, HYPRE_Int *contact_proc_list, HYPRE_Int **numGhostFromProc, HYPRE_Int **ghostInfoOffset, 
         HYPRE_Int ***ghostGlobalIndex, HYPRE_Int *contact_send_buf, HYPRE_Int *contact_send_buf_starts )
{
   HYPRE_Int i,proc,level,cnt, offset;
   HYPRE_Int numRequestedNodes;

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   cnt = 0;
   for (proc = 0; proc < num_contacts; proc++)
   {
      contact_send_buf_starts[proc] = cnt;
      contact_send_buf[cnt++] = num_levels;
      for (level = 0; level < num_levels; level++)
      {
         numRequestedNodes = numGhostFromProc[contact_proc_list[proc]][level];
         contact_send_buf[cnt++] = numRequestedNodes;
         if (ghostInfoOffset) offset = ghostInfoOffset[contact_proc_list[proc]][level];
         else offset = 0;
         for (i = offset; i < offset + numRequestedNodes; i++)
         {
            contact_send_buf[cnt++] = ghostGlobalIndex[contact_proc_list[proc]][level][i];
            // if (level == 2 && myid == 0) printf("On level 2, rank 0 requests node %d from proc 1\n", ghostGlobalIndex[contact_proc_list[proc]][level][i]);
         }
      }
   }
   contact_send_buf_starts[num_contacts] = cnt;

   return 0;
}

HYPRE_Int
UnpackGhostNodeResponse( hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int num_contacts, 
      HYPRE_Int *contact_proc_list, HYPRE_Complex *response_buf, HYPRE_Int *numNewGhostNodes, HYPRE_Int **ghostInfoOffset, HYPRE_Int ***ghostUnpackIndex )
{
   HYPRE_Int level, proc, i, j, cnt, offset;
   HYPRE_Int numRequestedNodes, row_size;
   hypre_ParCompMatrixRow *row;

   // Reallocate the comp grids on each level to make space for the new ghost nodes
   for (level = 0; level < num_levels; level++)
   {
      // If adding ghost nodes on this level, rellocate space
      if (numNewGhostNodes[level]) hypre_ParCompGridResize( compGrid[level], hypre_ParCompGridNumNodes(compGrid[level]) + numNewGhostNodes[level] );
   }

   // Unpack response buf info into comp grid
   // Note buffers from all processors are all squished sequentially into response_buf
   cnt = 0;
   for (proc = 0; proc < num_contacts; proc++)
   {
      // For each level...
      for (level = 0; level < num_levels; level++)
      {
         // Get the number of nodes incoming on this level
         numRequestedNodes = (HYPRE_Int) response_buf[cnt++];

         // For each node...
         offset = ghostInfoOffset[ contact_proc_list[proc] ][level];
         for (i = offset; i < offset + numRequestedNodes; i++)
         {
            // Find appropriate location to upack to
            HYPRE_Int unpackIndex = ghostUnpackIndex[contact_proc_list[proc]][level][i];

            // Unpack residual value and global index
            hypre_ParCompGridF(compGrid[level])[ unpackIndex ] = response_buf[cnt++];
            hypre_ParCompGridGlobalIndices(compGrid[level])[ unpackIndex ] = response_buf[cnt++];
            hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ unpackIndex ] = response_buf[cnt++];

            // Make sure soln value u is zero
            hypre_ParCompGridU(compGrid[level])[ unpackIndex ] = 0.0;

            // Create row of A, allocate space, and unpack info from buffer
            hypre_ParCompGridARows(compGrid[level])[ unpackIndex ] = hypre_ParCompMatrixRowCreate();
            row = hypre_ParCompGridARows(compGrid[level])[ unpackIndex ];
            row_size = (HYPRE_Int) response_buf[cnt++];
            hypre_ParCompMatrixRowSize(row) = row_size;
            hypre_ParCompMatrixRowData(row) = hypre_CTAlloc(HYPRE_Complex, row_size);
            hypre_ParCompMatrixRowGlobalIndices(row) = hypre_CTAlloc(HYPRE_Int, row_size);
            hypre_ParCompMatrixRowLocalIndices(row) = hypre_CTAlloc(HYPRE_Int, row_size);
            for (j = 0; j < row_size; j++)
            {
               hypre_ParCompMatrixRowData(row)[j] = response_buf[cnt++];
            }
            for (j = 0; j < row_size; j++)
            {
               hypre_ParCompMatrixRowGlobalIndices(row)[j] = (HYPRE_Int) response_buf[cnt++];
            }

            // Create row of P, allocate space, and unpack info from buffer
            hypre_ParCompGridPRows(compGrid[level])[ unpackIndex ] = hypre_ParCompMatrixRowCreate();
            row = hypre_ParCompGridPRows(compGrid[level])[ unpackIndex ];
            row_size = (HYPRE_Int) response_buf[cnt++];
            hypre_ParCompMatrixRowSize(row) = row_size;
            hypre_ParCompMatrixRowData(row) = hypre_CTAlloc(HYPRE_Complex, row_size);
            hypre_ParCompMatrixRowGlobalIndices(row) = hypre_CTAlloc(HYPRE_Int, row_size);
            hypre_ParCompMatrixRowLocalIndices(row) = hypre_CTAlloc(HYPRE_Int, row_size);
            for (j = 0; j < row_size; j++)
            {
               hypre_ParCompMatrixRowData(row)[j] = response_buf[cnt++];
            }
            for (j = 0; j < row_size; j++)
            {
               hypre_ParCompMatrixRowGlobalIndices(row)[j] = (HYPRE_Int) response_buf[cnt++];
            }
         }
      }
   }

   return 0;
}

HYPRE_Int
UnpackGhostNodeResponseResidualOnly( hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int num_contacts, 
      HYPRE_Int *contact_proc_list, HYPRE_Complex *response_buf, HYPRE_Int ***ghostUnpackIndex )
{
   HYPRE_Int level, proc, i, cnt;
   HYPRE_Int numRequestedNodes;

   // Unpack response buf info into comp grid
   // Note buffers from all processors are all squished sequentially into response_buf
   cnt = 0;
   for (proc = 0; proc < num_contacts; proc++)
   {
      // For each level...
      for (level = 0; level < num_levels; level++)
      {
         // Get the number of nodes incoming on this level
         numRequestedNodes = (HYPRE_Int) response_buf[cnt++];

         // For each node...
         for (i = 0; i < numRequestedNodes; i++)
         {
            // Find appropriate location to upack to
            HYPRE_Int unpackIndex = ghostUnpackIndex[contact_proc_list[proc]][level][i];

            // Unpack residual value
            hypre_ParCompGridF(compGrid[level])[ unpackIndex ] = response_buf[cnt++];
         }
      }
   }

   return 0;
}

HYPRE_Int
FindGhostNodes( hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index, HYPRE_Int **numGhostFromProc,
      HYPRE_Int **ghostInfoOffset, HYPRE_Int ***ghostGlobalIndex, HYPRE_Int *numNewGhostNodes, HYPRE_Int ***ghostUnpackIndex, HYPRE_Int *global_nodes, hypre_IJAssumedPart **apart )
{
   HYPRE_Int                  level, i, j, k, searchStart, searchEnd;
   HYPRE_Int                  row_size, ghostProcID, ghostAddFlag = 1, searchIndex;
   HYPRE_Int                  ghostRowStart, ghostRowEnd;
   hypre_ParCompMatrixRow     *row;
   HYPRE_Int                  ghostInfoLength;

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   for (level = 0; level < num_levels; level++)
   {
      // If we have already added a layer of ghost nodes, search out from them in the stencil of A
      if ( hypre_ParCompGridNumRealNodes(compGrid[level]) != hypre_ParCompGridNumNodes(compGrid[level]) )
      {
         searchStart = hypre_ParCompGridNumRealNodes(compGrid[level]);
         searchEnd = hypre_ParCompGridNumNodes(compGrid[level]);
      }
      else // Otherwise search out from the real nodes
      {
         searchStart = 0;
         searchEnd = hypre_ParCompGridNumNodes(compGrid[level]);
      }
      for (i = searchStart; i < searchEnd; i++)
      {
         row = hypre_ParCompGridARows(compGrid[level])[i];
         row_size = hypre_ParCompMatrixRowSize(row);

         for (j = 0; j < row_size; j++)
         {
            if (hypre_ParCompMatrixRowLocalIndices(row)[j] == -1)
            {
               // If so, figure out what processor the node lives on (!!! We use the assumed partition to find processor ID !!!)
               hypre_GetAssumedPartitionProcFromRow( hypre_MPI_COMM_WORLD, hypre_ParCompMatrixRowGlobalIndices(row)[j], 0, global_nodes[level], &ghostProcID );

               // Use assumed partition to get local index on the processor which owns the ghost node
               hypre_GetAssumedPartitionRowRange( hypre_MPI_COMM_WORLD, ghostProcID, 0, global_nodes[level], &ghostRowStart, &ghostRowEnd );
               searchIndex = hypre_ParCompMatrixRowGlobalIndices(row)[j];

               // ghostGlobalIndex[ghostProcID][level] and ghostUnpackIndex[ghostProcID][level] are initialized as NULL, so if they are still null when we need them, allocate
               if (!ghostGlobalIndex[ghostProcID][level]) 
               {
                  // likely that the number of ghost nodes will be at most 3 layers times the length of one side of the home domain (!!! 2D !!!)
                  ghostInfoLength = 3 * ( (int) sqrt( (double) (proc_last_index[level] - proc_first_index[level]) + 1 ) + 6 );
                  ghostGlobalIndex[ghostProcID][level] = hypre_CTAlloc(HYPRE_Int, ghostInfoLength);
                  ghostUnpackIndex[ghostProcID][level] = hypre_CTAlloc(HYPRE_Int, ghostInfoLength);
                  // Mark ends of arrays
                  ghostGlobalIndex[ghostProcID][level][ghostInfoLength - 1] = -1;
                  ghostUnpackIndex[ghostProcID][level][ghostInfoLength - 1] = -1;
               }

               // Check whether ghost node has already been accounted for (!!! involves linear search !!!)
               ghostAddFlag = 1;
               HYPRE_Int ghostInfoIndex = ghostInfoOffset[ghostProcID][level] + numGhostFromProc[ghostProcID][level];
               for (k = ghostInfoIndex - 1; k > -1; k--) // Note: doing the search backward (hopefully shorter)
               {
                  if (ghostGlobalIndex[ghostProcID][level][k] == searchIndex)
                  {
                     ghostAddFlag = 0;
                     break;
                  }
               }
               if (!ghostAddFlag)
               {
                  // If ghost node accounted for, just need to set local index to look in the right place
                  hypre_ParCompMatrixRowLocalIndices(row)[j] = ghostUnpackIndex[ghostProcID][level][ k ]; // !!! GETTING THERE I THINK... CHECK THIS !!!
               }
               else // otherwise, need to account for the request
               {
                  // Check to see whether we need to increase the size of ghostGlobalIndex[ghostProcID][level] and ghostUnpackIndex[ghostProcID][level]
                  if (ghostGlobalIndex[ghostProcID][level][ ghostInfoIndex ] == -1)
                  {
                     // Reallocate arrays
                     ghostGlobalIndex[ghostProcID][level] = hypre_TReAlloc(ghostGlobalIndex[ghostProcID][level], HYPRE_Int, 2*ghostInfoIndex + 1);
                     ghostUnpackIndex[ghostProcID][level] = hypre_TReAlloc(ghostUnpackIndex[ghostProcID][level], HYPRE_Int, 2*ghostInfoIndex + 1);
                     // Mark end of arrays
                     ghostGlobalIndex[ghostProcID][level][ 2*ghostInfoIndex ] = -1;
                     ghostUnpackIndex[ghostProcID][level][ 2*ghostInfoIndex ] = -1;
                  }
                  // Store the ghost index
                  ghostGlobalIndex[ghostProcID][level][ ghostInfoIndex ] = searchIndex;
                  // if (myid == 0 ) printf("level = %d, searchIndex = %d, global index = %d, ghostRowStart = %d, global_nodes = %d\n", level, searchIndex, hypre_ParCompMatrixRowGlobalIndices(row)[j], ghostRowStart, global_nodes[level]);
                  // Store info for unpacking ghost rows later, set local ghost index and increment numGhost counters
                  ghostUnpackIndex[ghostProcID][level][ ghostInfoIndex++ ] = hypre_ParCompGridNumNodes(compGrid[level]) + numNewGhostNodes[level]; // !!! DOUBLE CHECK !!!
                  hypre_ParCompMatrixRowLocalIndices(row)[j] = hypre_ParCompGridNumRealNodes(compGrid[level]) + numNewGhostNodes[level]; // !!! DOUBLE CHECK !!!
                  numGhostFromProc[ghostProcID][level]++;
                  numNewGhostNodes[level]++;
               }
            }
         }
      }
   }

   // Up to now, we have saved all the info acording to the assumed partition, so need to do some communication to figure out where the ghost nodes actually live and fix up our arrays
   LocateGhostNodes(numGhostFromProc, ghostGlobalIndex, ghostUnpackIndex, ghostInfoOffset, apart, num_levels, global_nodes);
   for (level = 0; level < num_levels; level++)
   {
      HYPRE_Int sum = 0;
      for (i = 0; i < num_procs; i++) sum +=numGhostFromProc[i][level];
      // printf("Rank %d, level %d: Sum of numGhostFromProc = %d, numNewGhostNodes = %d\n", myid, level, sum, numNewGhostNodes[level]);
   }
   

   // Coarsest level should always own all info (i.e. should not ask for ghost nodes)
   // If this is not the case, raise an error
   if (numNewGhostNodes[num_levels-1] != 0) printf("Error: Processor does not have real dofs for entire coarsest grid!\n");

   return 0;
}

HYPRE_Int 
LocateGhostNodes(HYPRE_Int **numGhostFromProc, HYPRE_Int ***ghostGlobalIndex, HYPRE_Int ***ghostUnpackIndex, HYPRE_Int **ghostInfoOffset, hypre_IJAssumedPart **apart, HYPRE_Int num_levels, HYPRE_Int *global_nodes)

{

   HYPRE_Int        num_procs, myid;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int        level, proc, k, j, i, cnt, proc_cnt;
   HYPRE_Int        range_start, range_end; 

   HYPRE_Int        num_contacts, *contact_procs=NULL, *contact_vec_starts=NULL;
   HYPRE_Int        *contact_buf=NULL;
   HYPRE_Int        *response_buf = NULL, *response_buf_starts=NULL;
    
   HYPRE_Int        num_ranges;
   HYPRE_Int        actualProcID;
   HYPRE_Int        *proc_ids;
   HYPRE_Int        *upper_bounds;

   HYPRE_Int        max_response_size;
   
   hypre_DataExchangeResponse        response_obj1;



   // Debugging:
   // if (myid == 2)
   // {
   //    FILE *file;
   //    char filename[255];
   //    hypre_sprintf(filename, "/Users/mitchell82/Desktop/before_ghostGlobalIndexRank%dLevel%d.txt", myid, 1);
   //    file = fopen(filename, "w");
   //    for (proc = 0; proc < num_procs; proc++)
   //    {
   //       hypre_fprintf(file, "Proc %d:\n", proc);
   //       for (i = 0; i < numGhostFromProc[proc][1]; i++)
   //       {
   //          hypre_fprintf(file, "  %d\n", ghostGlobalIndex[proc][1][i]);
   //       }
   //    }
   //    fclose(file);
   // }
   // if (myid == 2)
   // {
   //    FILE *file;
   //    char filename[255];
   //    hypre_sprintf(filename, "/Users/mitchell82/Desktop/before_ghostUnpackIndexRank%dLevel%d.txt", myid, 1);
   //    file = fopen(filename, "w");
   //    for (proc = 0; proc < num_procs; proc++)
   //    {
   //       hypre_fprintf(file, "Proc %d:\n", proc);
   //       for (i = 0; i < numGhostFromProc[proc][1]; i++)
   //       {
   //          hypre_fprintf(file, "  %d\n", ghostUnpackIndex[proc][1][i]);
   //       }
   //    }
   //    fclose(file);
   // }





   HYPRE_Int                  **old_numGhostFromProc;
   HYPRE_Int                  ***old_ghostGlobalIndex;
   HYPRE_Int                  ***old_ghostUnpackIndex;

   old_numGhostFromProc =  hypre_CTAlloc(HYPRE_Int*, num_procs);
   old_ghostGlobalIndex = hypre_CTAlloc(HYPRE_Int**, num_procs);
   old_ghostUnpackIndex = hypre_CTAlloc(HYPRE_Int**, num_procs);
   for (i = 0; i < num_procs; i++)
   {
      old_numGhostFromProc[i] = hypre_CTAlloc(HYPRE_Int, num_levels);
      old_ghostGlobalIndex[i] = hypre_CTAlloc(HYPRE_Int*, num_levels);
      old_ghostUnpackIndex[i] = hypre_CTAlloc(HYPRE_Int*, num_levels);
      for (j = 0; j < num_levels; j++)
      {
         // Copy over relevant old ghost node info where it exists
         if (numGhostFromProc[i][j])
         {
            old_numGhostFromProc[i][j] = numGhostFromProc[i][j];
            numGhostFromProc[i][j] = 0;
            old_ghostGlobalIndex[i][j] = hypre_CTAlloc(HYPRE_Int, old_numGhostFromProc[i][j]);
            old_ghostUnpackIndex[i][j] = hypre_CTAlloc(HYPRE_Int, old_numGhostFromProc[i][j]);
            for (k = 0; k < old_numGhostFromProc[i][j]; k++)
            {
               old_ghostGlobalIndex[i][j][k] = ghostGlobalIndex[i][j][ k + ghostInfoOffset[i][j] ];
               old_ghostUnpackIndex[i][j][k] = ghostUnpackIndex[i][j][ k + ghostInfoOffset[i][j] ];
            }
         }
         // Otherwise initialize to NULL
         else
         {
            old_ghostGlobalIndex[i][j] = NULL;
            old_ghostUnpackIndex[i][j] = NULL;
         }
      }
   }


   // Debugging:
   // for (level = 1; level < 2; level++)
   // {
   //     hypre_printf("Partition level %d\n", level);
   //     hypre_printf("   myid = %i, my assumed local range: [%i, %i]\n", myid, 
   //                       apart[level]->row_start, apart[level]->row_end);

   //    for (i=0; i<apart[level]->length; i++)
   //    {
   //      hypre_printf("   myid = %d, proc %d owns assumed partition range = [%d, %d]\n", 
   //              myid, apart[level]->proc_list[i], apart[level]->row_start_list[i], 
   //         apart[level]->row_end_list[i]);
   //    }

   //    hypre_printf("   myid = %d, length of apart[level] = %d\n", myid, apart[level]->length);
   // }




   // Setup contact info
   num_contacts = 0;
   HYPRE_Int size_est = 10*num_levels; // estimate that we need to contact at most 10 procs per level
   contact_procs = hypre_CTAlloc(HYPRE_Int, size_est);
   for (proc = 0; proc < num_procs; proc++)
   {
      for (level = 0; level < num_levels; level++)
      {
         if (old_numGhostFromProc[proc][level])
         {
            if (num_contacts == size_est)
            {
               size_est += 10;
               contact_procs = hypre_TReAlloc(contact_procs, HYPRE_Int, size_est);
            }
            contact_procs[num_contacts++] = proc;
            break;
         }
      }
   }
   contact_procs = hypre_TReAlloc(contact_procs, HYPRE_Int, num_contacts);
   contact_vec_starts = hypre_CTAlloc(HYPRE_Int, num_contacts+1);
   contact_buf = hypre_CTAlloc(HYPRE_Int, 2*num_contacts*num_levels);

   cnt = 0;
   for (proc_cnt = 0; proc_cnt < num_contacts; proc_cnt++)
   {
      proc = contact_procs[proc_cnt];
      contact_vec_starts[proc_cnt] = cnt;
      for (level = 0; level < num_levels; level++)
      {
         if (old_numGhostFromProc[proc][level])
         {
            // Find min and max over the nodes requested from this proc
            range_start = global_nodes[level];
            range_end = 0;
            for (i = 0; i < old_numGhostFromProc[proc][level]; i++)
            {
               if (old_ghostGlobalIndex[proc][level][i] < range_start) range_start = old_ghostGlobalIndex[proc][level][i];
               if (old_ghostGlobalIndex[proc][level][i] > range_end) range_end = old_ghostGlobalIndex[proc][level][i];               
            }
            contact_buf[cnt++] = range_start;
            contact_buf[cnt++] = range_end;
         }
         else
         {
            contact_buf[cnt++] = -1;
         }
      }
   }
   contact_vec_starts[num_contacts] = cnt;
   


   // Debugging:
   // if (myid == 2)
   // {
   //    hypre_printf("Rank %d: num_contacts = %d\n", myid, num_contacts);
   //    FILE *file;
   //    char filename[255];
   //    hypre_sprintf(filename, "/Users/mitchell82/Desktop/contact_bufRank%d.txt", myid);
   //    file = fopen(filename, "w");
   //    for (proc = 0; proc < num_contacts; proc++)
   //    {
   //       hypre_fprintf(file, "Proc %d:\n", contact_procs[proc]);
   //       for (i = contact_vec_starts[proc]; i < contact_vec_starts[proc+1]; i++)
   //       {
   //          hypre_fprintf(file, "  %d\n", contact_buf[i]);
   //       }
   //    }
   //    fclose(file);
   // }




   /*create response object*/
   response_obj1.fill_response = FillResponseForLocateGhostNodes;
   response_obj1.data1 =  apart; /* this is necessary so we can fill responses*/ 
   response_obj1.data2 = &num_levels;
   
   max_response_size = 9*num_levels;  /* 9 means we can fit 4 ranges on each level: each range is 2 items plus one item to say how many ranges*/
   
   hypre_DataExchangeList(num_contacts, contact_procs, 
                    contact_buf, contact_vec_starts, sizeof(HYPRE_Int), 
                     sizeof(HYPRE_Int), &response_obj1, max_response_size, 1, 
                     hypre_MPI_COMM_WORLD, (void**) &response_buf, &response_buf_starts);



   // Debugging:
   // if (myid == 2)
   // {
   //    FILE *file;
   //    char filename[255];
   //    hypre_sprintf(filename, "/Users/mitchell82/Desktop/response_bufRank%d.txt", myid);
   //    file = fopen(filename, "w");
   //    for (proc = 0; proc < num_contacts; proc++)
   //    {
   //       hypre_fprintf(file, "Proc %d:\n", contact_procs[proc]);
   //       for (i = response_buf_starts[proc]; i < response_buf_starts[proc+1]; i++)
   //       {
   //          hypre_fprintf(file, "  %d\n", response_buf[i]);
   //       }
   //    }
   //    fclose(file);
   // }



   // Unpack the response buffer and fill in new_numGhostFromProc, new_ghostGlobalIndex, new_ghostUnpackIndex
   cnt = 0; // cnt is now indexing the reponse buffer
   for (proc_cnt = 0; proc_cnt < num_contacts; proc_cnt++)
   {
      proc = contact_procs[proc_cnt];
      for (level = 0; level < num_levels; level++)
      {
         if (old_numGhostFromProc[proc][level])
         {
            // Get num ranges and the range starts and ends
            num_ranges = response_buf[cnt++];
            proc_ids = hypre_CTAlloc(HYPRE_Int, num_ranges);
            upper_bounds = hypre_CTAlloc(HYPRE_Int, num_ranges);
            for (i = 0; i < num_ranges; i++)
            {
               proc_ids[i] = response_buf[cnt++];
               upper_bounds[i] = response_buf[cnt++];
            }

            // if (myid == 0)
            // {
            //    printf("Rank 0 unpacking proc %d, level %d:\n", proc, level );
            //    printf("proc_ids = \n");
            //    for (i = 0; i < num_ranges; i++)
            //    {
            //       printf("   %d\n", proc_ids[i]);
            //    }
            //    printf("upper_bounds = \n");
            //    for (i = 0; i < num_ranges; i++)
            //    {
            //       printf("   %d\n", upper_bounds[i]);
            //    }
            // }

            // Loop over the nodes we asked for from this proc and copy their info to the arrays for the appropriate processors
            for (i = 0; i < old_numGhostFromProc[proc][level]; i++)
            {
               // Find which range this node belongs to
               j = 0;
               actualProcID = proc_ids[j];
               // if (myid == 0) printf("old_ghostGlobalIndex[%d][%d][%d] = %d, upper_bounds[j] = %d\n", proc, level, i, old_ghostGlobalIndex[proc][level][i], upper_bounds[j]);
               while (old_ghostGlobalIndex[proc][level][i] > upper_bounds[j] && j < num_ranges) 
               {
                  // if (myid == 0) printf("j = %d, upper_bounds[j] = %d\n", j, upper_bounds[j]);
                  actualProcID = proc_ids[++j];
               }
               // if (myid == 0) printf("actualProcID = %d\n", actualProcID);

               //If new info arrays are still null when we need them, allocate
               if (!ghostGlobalIndex[actualProcID][level]) 
               {
                  // likely that the number of ghost nodes will be at most 3 layers times the length of one side of the home domain (!!! 2D !!!)
                  HYPRE_Int ghostInfoLength = 3 * ( (int) sqrt( (double) (apart[level]->row_end - apart[level]->row_start) + 1 ) + 6 );
                  ghostGlobalIndex[actualProcID][level] = hypre_CTAlloc(HYPRE_Int, ghostInfoLength);
                  ghostUnpackIndex[actualProcID][level] = hypre_CTAlloc(HYPRE_Int, ghostInfoLength);
                  // Mark ends of arrays
                  ghostGlobalIndex[actualProcID][level][ghostInfoLength - 1] = -1;
                  ghostUnpackIndex[actualProcID][level][ghostInfoLength - 1] = -1;
               }

               HYPRE_Int ghostInfoIndex = ghostInfoOffset[actualProcID][level] + numGhostFromProc[actualProcID][level];
               // Check to see whether we need to increase the size of ghostGlobalIndex[actualProcID][level] and ghostUnpackIndex[actualProcID][level]
               if (ghostGlobalIndex[actualProcID][level][ ghostInfoIndex ] == -1)
               {
                  // Reallocate arrays
                  ghostGlobalIndex[actualProcID][level] = hypre_TReAlloc(ghostGlobalIndex[actualProcID][level], HYPRE_Int, 2*ghostInfoIndex + 1);
                  ghostUnpackIndex[actualProcID][level] = hypre_TReAlloc(ghostUnpackIndex[actualProcID][level], HYPRE_Int, 2*ghostInfoIndex + 1);
                  // Mark end of arrays
                  ghostGlobalIndex[actualProcID][level][ 2*ghostInfoIndex ] = -1;
                  ghostUnpackIndex[actualProcID][level][ 2*ghostInfoIndex ] = -1;
               }

               ghostGlobalIndex[actualProcID][level][ ghostInfoIndex ] = old_ghostGlobalIndex[proc][level][i];
               ghostUnpackIndex[actualProcID][level][ ghostInfoIndex ] = old_ghostUnpackIndex[proc][level][i];
               numGhostFromProc[actualProcID][level]++;

            }

            hypre_TFree(proc_ids);
            hypre_TFree(upper_bounds);
         }
      }
   }


   // Free up old ghost info arrays and communication buffers and info
   for (proc = 0; proc < num_procs; proc++)
   {
      for (level = 0; level < num_levels; level++)
      {
         if (old_ghostGlobalIndex[proc][level]) hypre_TFree(old_ghostGlobalIndex[proc][level]);
         if (old_ghostUnpackIndex[proc][level]) hypre_TFree(old_ghostUnpackIndex[proc][level]);
      }
      hypre_TFree(old_ghostGlobalIndex[proc]);
      hypre_TFree(old_ghostUnpackIndex[proc]);
      hypre_TFree(old_numGhostFromProc[proc]);
   }
   hypre_TFree(old_ghostGlobalIndex);
   hypre_TFree(old_ghostUnpackIndex);
   hypre_TFree(old_numGhostFromProc);

   hypre_TFree(contact_procs);
   hypre_TFree(contact_vec_starts);
   hypre_TFree(contact_buf);
   hypre_TFree(response_buf);
   hypre_TFree(response_buf_starts);




   // Debugging:
   // if (myid == 2)
   // {
   //    FILE *file;
   //    char filename[255];
   //    hypre_sprintf(filename, "/Users/mitchell82/Desktop/after_ghostGlobalIndexRank%dLevel%d.txt", myid, 1);
   //    file = fopen(filename, "w");
   //    for (proc = 0; proc < num_procs; proc++)
   //    {
   //       hypre_fprintf(file, "Proc %d:\n", proc);
   //       for (i = 0; i < numGhostFromProc[proc][1]; i++)
   //       {
   //          hypre_fprintf(file, "  %d\n", ghostGlobalIndex[proc][1][i]);
   //       }
   //    }
   //    fclose(file);
   // }
   // if (myid == 2)
   // {
   //    FILE *file;
   //    char filename[255];
   //    hypre_sprintf(filename, "/Users/mitchell82/Desktop/after_ghostUnpackIndexRank%dLevel%d.txt", myid, 1);
   //    file = fopen(filename, "w");
   //    for (proc = 0; proc < num_procs; proc++)
   //    {
   //       hypre_fprintf(file, "Proc %d:\n", proc);
   //       for (i = 0; i < numGhostFromProc[proc][1]; i++)
   //       {
   //          hypre_fprintf(file, "  %d\n", ghostUnpackIndex[proc][1][i]);
   //       }
   //    }
   //    fclose(file);
   // }




   return hypre_error_flag;

}

/*--------------------------------------------------------------------
 * FillResponseForLocateGhostNodes
 * Fill response function for determining the recv. processors
 * data exchange. This is copied from code used in setting up matvec comm pkg.
 *--------------------------------------------------------------------*/

HYPRE_Int
FillResponseForLocateGhostNodes(void *p_recv_contact_buf,
                                      HYPRE_Int contact_size, HYPRE_Int contact_proc, void *ro, 
                                      MPI_Comm comm, void **p_send_response_buf, 
                                      HYPRE_Int *response_message_size)
{
   HYPRE_Int    myid, tmp_id, row_end;
   HYPRE_Int    j, level;
   HYPRE_Int    row_val, index, size, contact_index;
   
   HYPRE_Int   *send_response_buf = (HYPRE_Int *) *p_send_response_buf;
   HYPRE_Int   *recv_contact_buf = (HYPRE_Int * ) p_recv_contact_buf;


   hypre_DataExchangeResponse  *response_obj = (hypre_DataExchangeResponse*)ro; 
   hypre_IJAssumedPart               **apart = (hypre_IJAssumedPart**)response_obj->data1;
   HYPRE_Int               num_levels = *( (HYPRE_Int*)response_obj->data2 );
   
   HYPRE_Int overhead = response_obj->send_response_overhead;

   /*-------------------------------------------------------------------
    * we are getting a range of off_d entries - need to see if we own them
    * or how many ranges to send back  - send back
    * with format [proc_id end_row  proc_id #end_row  proc_id #end_row etc...].
    *----------------------------------------------------------------------*/ 

   hypre_MPI_Comm_rank(comm, &myid );


   /* populate send_response_buf */
      
   index = 0; /*count entries in send_response_buf*/
   contact_index = 0;
   
   // Loop over levels
   for (level = 0; level < num_levels; level++)
   {
      row_val = recv_contact_buf[contact_index++]; /*beginning of range*/
      if (row_val != -1)
      {
         // Set aside first index to hold num ranges on this level
         HYPRE_Int num_ranges = 0;
         HYPRE_Int num_ranges_index = index++;

         hypre_IJAssumedPart *part = apart[level];
         j = 0; /*marks which partition of the assumed partition we are in */
         row_end = part->row_end_list[part->sort_index[j]];
         tmp_id = part->proc_list[part->sort_index[j]];

         /*check storage in send_buf for adding the ranges !!! NOT SURE IF THIS MAKE SENSE AFTER MY CHANGES... THIS IS A HOLD OVER FROM ORIGINAL FUNCTION !!!*/
         size = 2*(part->length);
               
         if ( response_obj->send_response_storage  < size  )
         {

            response_obj->send_response_storage =  hypre_max(size, 20); 
            send_response_buf = hypre_TReAlloc( send_response_buf, HYPRE_Int, 
                                               response_obj->send_response_storage + overhead );
            *p_send_response_buf = send_response_buf;    /* needed when using ReAlloc */
         }


         while (row_val > row_end) /*which partition to start in */
         {
            j++;
            row_end = part->row_end_list[part->sort_index[j]];   
            tmp_id = part->proc_list[part->sort_index[j]];
         }

         /*add this range*/
         send_response_buf[index++] = tmp_id;
         send_response_buf[index++] = row_end; 
         num_ranges++;

         j++; /*increase j to look in next partition */
            
          
         /*any more?  - now compare with end of range value*/
         row_val = recv_contact_buf[contact_index++]; /*end of range*/
         while ( j < part->length && row_val > row_end  )
         {
            row_end = part->row_end_list[part->sort_index[j]];  
            tmp_id = part->proc_list[part->sort_index[j]];

            send_response_buf[index++] = tmp_id;
            send_response_buf[index++] = row_end;  
            num_ranges++;

            j++;
            
         }
         // Set the num ranges entry for this level
         send_response_buf[num_ranges_index] = num_ranges;
      }
   }


   *response_message_size = index;
   *p_send_response_buf = send_response_buf;

   return hypre_error_flag;

}