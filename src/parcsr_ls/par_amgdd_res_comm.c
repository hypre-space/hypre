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

#define DEBUG_COMP_GRID 0

HYPRE_Int 
GeneratePsiComposite( hypre_ParCompGrid **psiComposite, hypre_ParCompGrid **compGrid, hypre_ParCSRCommPkg *commPkg, HYPRE_Int *send_flag_buffer_size, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
GetBufferSize( hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels );

HYPRE_Complex*
PackSendBuffer( hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels, HYPRE_Int buffer_size );

HYPRE_Int
UnpackRecvBuffer( HYPRE_Complex *recv_buffer, hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
AddToCompGrid(hypre_ParCompGrid **compGrid, hypre_ParCompGrid **psiComposite, HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_size, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels, HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index );

HYPRE_Int
PackRecvMapSendBuffer(HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_send_buffer, hypre_ParCompGrid **psiComposite, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
UnpackSendFlagBuffer(HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *send_buffer_size, HYPRE_Int *num_send_nodes, HYPRE_Int current_level, HYPRE_Int num_levels);

HYPRE_Int
PackResidualBuffer( HYPRE_Complex *send_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, hypre_ParCompGrid **compGrid, hypre_ParCSRCommPkg *commPkg, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
UnpackResidualBuffer( HYPRE_Complex *recv_buffer, HYPRE_Int **recv_map, hypre_ParCompGrid **compGrid, HYPRE_Int current_level, HYPRE_Int num_levels);

/*****************************************************************************
 *
 * Routine for communicating the composite grid residuals in AMG-DD
 *
 *****************************************************************************/

/*****************************************************************************
 * hypre_AMGDD_res_comm
 *****************************************************************************/

HYPRE_Int
hypre_BoomerAMGDDCompGridSetup( void               *amg_vdata )
{

   HYPRE_Int   myid;
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

   // info from amg setup
   hypre_ParCSRMatrix         **A_array;
   hypre_ParVector            **F_array;
   hypre_ParVector            **U_array;
   hypre_ParCSRMatrix         **P_array;
   hypre_ParCSRMatrix         **R_array;
   hypre_ParVector            *Vtemp;
   hypre_ParCSRCommPkg        *commPkg;
   HYPRE_Int                  **CF_marker_array;
   HYPRE_Int                  *proc_first_index, *proc_last_index;

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

   // timing variables
   HYPRE_Int                  time_index0, time_index1, time_index2, time_index3, time_index4, time_index5;

   // initialize timings and begin global timing
   time_index0 = hypre_InitializeTiming("Entire setup phase time");
   time_index1 = hypre_InitializeTiming("Generate Psi Composite");
   time_index2 = hypre_InitializeTiming("Communicate the buffer sizes");
   time_index3 = hypre_InitializeTiming("Communicate the buffers");
   time_index4 = hypre_InitializeTiming("Unpack and add nodes to composite grid");
   time_index5 = hypre_InitializeTiming("Communicate array to flag repeated nodes");
   hypre_BeginTiming(time_index0);

   // get info from amg
   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   R_array = hypre_ParAMGDataRArray(amg_data);
   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);
   Vtemp = hypre_ParAMGDataVtemp(amg_data);
   CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);
   num_levels = hypre_ParAMGDataNumLevels(amg_data);

   // get first and last global indices on each level for this proc
   proc_first_index = hypre_CTAlloc(HYPRE_Int, num_levels);
   proc_last_index = hypre_CTAlloc(HYPRE_Int, num_levels);
   for (level = 0; level < num_levels; level++)
   {
      proc_first_index[level] = hypre_ParVectorFirstIndex(F_array[level]);
      proc_last_index[level] = hypre_ParVectorLastIndex(F_array[level]);
   }

   // Allocate space for some variables that store info on each level
   compGrid = hypre_CTAlloc(hypre_ParCompGrid*, num_levels);
   compGridCommPkg = hypre_ParCompGridCommPkgCreate();
   hypre_ParCompGridCommPkgNumLevels(compGridCommPkg) = num_levels;
   hypre_ParCompGridCommPkgNumSends(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels);
   hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels);
   send_buffer_size = hypre_CTAlloc(HYPRE_Int*, num_levels);
   recv_buffer_size = hypre_CTAlloc(HYPRE_Int*, num_levels);
   send_flag = hypre_CTAlloc(HYPRE_Int***, num_levels);
   num_send_nodes = hypre_CTAlloc(HYPRE_Int**, num_levels);
   recv_map = hypre_CTAlloc(HYPRE_Int***, num_levels);

   /* Form residual and restrict down to all levels and initialize composite grids 
      Note that from here on, residuals will be stored in F_array and the fine grid RHS will be stored in Vtemp */
   hypre_ParVectorCopy(F_array[0],Vtemp);
   alpha = -1.0;
   beta = 1.0;
   hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0],
                        beta, F_array[0]);

   compGrid[0] = hypre_ParCompGridCreate();
   hypre_ParCompGridInitialize( compGrid[0], F_array[0], CF_marker_array[0], proc_first_index[1], A_array[0] );

   for (level = 0; level < num_levels-1; level++)
   {
      alpha = 1.0;
      beta = 0.0;
      hypre_ParCSRMatrixMatvecT(alpha,P_array[level],F_array[level],
                            beta,F_array[level+1]);

      compGrid[level+1] = hypre_ParCompGridCreate();
      if (level != num_levels-2) hypre_ParCompGridInitialize( compGrid[level+1], F_array[level+1], CF_marker_array[level+1], proc_first_index[level+2], A_array[level+1] );
      else hypre_ParCompGridInitialize( compGrid[level+1], F_array[level+1], CF_marker_array[level+1], 0, A_array[level+1] );
   }

   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   for (level = num_levels-1; level > -1; level--)
   {      
      if ( proc_last_index[level] >= proc_first_index[level] ) // If there are any owned nodes on this level
      {
         // Get the commPkg of matrix A on this level
         commPkg = hypre_ParCSRMatrixCommPkg(A_array[level]);
         comm = hypre_ParCSRCommPkgComm(commPkg);
         num_sends = hypre_ParCSRCommPkgNumSends(commPkg);
         num_recvs = hypre_ParCSRCommPkgNumRecvs(commPkg);

         // Set info for comp grid comm pkg
         hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level] = num_sends;
         hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level] = num_recvs;

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

         
         hypre_BeginTiming(time_index1);

         // loop over send procs
         for (i = 0; i < num_sends; i++)
         {
            // allocate space for psiComposite_send
            psiComposite_send[i] = hypre_CTAlloc(hypre_ParCompGrid*, num_levels);

            // generate psiComposite
            num_psi_levels_send[i] = GeneratePsiComposite( psiComposite_send[i], compGrid, commPkg, &(send_flag_buffer_size[i]), i, level, num_levels );
         }

         hypre_EndTiming(time_index1);


         hypre_BeginTiming(time_index2);

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

         hypre_EndTiming(time_index2);


         // free and reallocate space for the requests and status
         hypre_TFree(requests);
         hypre_TFree(status);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs );
         request_counter = 0;

         
         hypre_BeginTiming(time_index3);

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

         hypre_EndTiming(time_index3);


         
         hypre_BeginTiming(time_index4);

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
            AddToCompGrid(compGrid, psiComposite_recv[i], recv_map_send[i], recv_map_size[i], &(recv_map_send_buffer_size[i]), level, num_levels, num_psi_levels_recv[i], proc_first_index, proc_last_index);
         }

         hypre_EndTiming(time_index4);


         // free and reallocate space for the requests and status
         hypre_TFree(requests);
         hypre_TFree(status);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs );
         status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs );
         request_counter = 0;


         hypre_BeginTiming(time_index5);         

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

         hypre_EndTiming(time_index5);

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
      }
   }


   #if DEBUG_COMP_GRID
   char filename[256];
   for (level = 0; level < num_levels; level++)
   {
      hypre_sprintf(filename, "Outputs/setupCompGridRank%dLevel%d.txt", myid, level);
      hypre_ParCompGridDebugPrint( compGrid[level], filename );
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

   // hypre_printf("Finished comp grid setup on rank %d\n", myid);

   // finish global timing and print timing info
   hypre_EndTiming(time_index0);

   hypre_PrintTiming("Setup composite grids", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index0);
   hypre_FinalizeTiming(time_index1);
   hypre_FinalizeTiming(time_index2);
   hypre_FinalizeTiming(time_index3);
   hypre_FinalizeTiming(time_index4);
   hypre_FinalizeTiming(time_index5);
   hypre_ClearTiming();

   return hypre_error_flag;
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
   hypre_ParCSRMatrix         **R_array;
   hypre_ParVector            *Vtemp;
   hypre_ParCSRCommPkg        *commPkg;
   HYPRE_Int                  **CF_marker_array;
   HYPRE_Int                  *proc_first_index, *proc_last_index;
   hypre_ParCompGrid          **compGrid;

   // info from comp grid comm pkg
   hypre_ParCompGridCommPkg   *compGridCommPkg;
   HYPRE_Int                  num_sends, num_recvs;
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

   // timing variables
   HYPRE_Int                  time_index0, time_index1;

   // begin timing
   time_index0 = hypre_InitializeTiming("Communicate Residual");
   time_index1 = hypre_InitializeTiming("Pack and send residual (generate psi composite)");
   hypre_BeginTiming(time_index0);

   // get info from amg
   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   R_array = hypre_ParAMGDataRArray(amg_data);
   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);
   Vtemp = hypre_ParAMGDataVtemp(amg_data);
   CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);
   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   compGrid = hypre_ParAMGDataCompGrid(amg_data);
   compGridCommPkg = hypre_ParAMGDataCompGridCommPkg(amg_data);

   // get info from comp grid comm pkg
   send_buffer_size = hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg);
   recv_buffer_size = hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg);
   num_send_nodes = hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg);
   send_flag = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg);
   recv_map = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg);

   // get first and last global indices on each level for this proc
   proc_first_index = hypre_CTAlloc(HYPRE_Int, num_levels);
   proc_last_index = hypre_CTAlloc(HYPRE_Int, num_levels);
   for (level = 0; level < num_levels; level++)
   {
      proc_first_index[level] = hypre_ParVectorFirstIndex(F_array[level]);
      proc_last_index[level] = hypre_ParVectorLastIndex(F_array[level]);
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

   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   for (level = num_levels-1; level > -1; level--)
   {      
      if ( proc_last_index[level] >= proc_first_index[level] ) // If there are any owned nodes on this level
      {
         // Get the commPkg of matrix A on this level
         commPkg = hypre_ParCSRMatrixCommPkg(A_array[level]);
         comm = hypre_ParCSRCommPkgComm(commPkg);
         num_sends = hypre_ParCSRCommPkgNumSends(commPkg);
         num_recvs = hypre_ParCSRCommPkgNumRecvs(commPkg);

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
            hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_COMPLEX, hypre_ParCSRCommPkgRecvProc(commPkg, i), 0, comm, &requests[request_counter++]);
         }

         
         hypre_BeginTiming(time_index1);

         // pack and send the buffers
         for (i = 0; i < num_sends; i++)
         {
            send_buffer[i] = hypre_CTAlloc(HYPRE_Complex, send_buffer_size[level][i]);
            PackResidualBuffer(send_buffer[i], send_flag[level][i], num_send_nodes[level][i], compGrid, commPkg, i, level, num_levels);
            hypre_MPI_Isend(send_buffer[i], send_buffer_size[level][i], HYPRE_MPI_COMPLEX, hypre_ParCSRCommPkgSendProc(commPkg, i), 0, comm, &requests[request_counter++]);
         }

         hypre_EndTiming(time_index1);

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


   #if DEBUG_COMP_GRID
   char filename[256];
   for (level = 0; level < num_levels; level++)
   {
      hypre_sprintf(filename, "Outputs/communicateCompGridRank%dLevel%d.txt", myid, level);
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

   // finish timing and print timing info
   hypre_EndTiming(time_index0);

   hypre_PrintTiming("Communicate Residual", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index1);
   hypre_FinalizeTiming(time_index0);
   hypre_ClearTiming();
   
   return hypre_error_flag;
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
         // store the row length
         row_length = hypre_ParCompMatrixRowSize( hypre_ParCompGridARows( psiComposite[level] )[i] );
         send_buffer[cnt++] = (HYPRE_Complex) row_length;

         // copy matrix entries
         for (j = 0; j < row_length; j++)
         {
            send_buffer[cnt++] = hypre_ParCompMatrixRowData( hypre_ParCompGridARows( psiComposite[level] )[i] )[j];
         }
         // copy global indices
         for (j = 0; j < row_length; j++)
         {
            send_buffer[cnt++] = (HYPRE_Complex) hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( psiComposite[level] )[i] )[j];
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
         // get the row length
         row_size = (HYPRE_Int) recv_buffer[cnt++];
         // Create row and allocate space
         hypre_ParCompGridARows(psiComposite[level])[i] = hypre_ParCompMatrixRowCreate();
         hypre_ParCompMatrixRowSize( hypre_ParCompGridARows( psiComposite[level] )[i] ) = row_size;
         hypre_ParCompMatrixRowData( hypre_ParCompGridARows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Complex, row_size);
         hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Int, row_size);
         hypre_ParCompMatrixRowLocalIndices( hypre_ParCompGridARows( psiComposite[level] )[i] ) = hypre_CTAlloc(HYPRE_Int, row_size);

         // copy matrix entries
         for (j = 0; j < row_size; j++)
         {
            hypre_ParCompMatrixRowData( hypre_ParCompGridARows( psiComposite[level] )[i] )[j] = recv_buffer[cnt++];
         }
         // copy global indices
         for (j = 0; j < row_size; j++)
         {
            hypre_ParCompMatrixRowGlobalIndices( hypre_ParCompGridARows( psiComposite[level] )[i] )[j] = (HYPRE_Int) recv_buffer[cnt++];
         }
      }
   }

   return num_psi_levels;
}

HYPRE_Int
AddToCompGrid( hypre_ParCompGrid **compGrid, hypre_ParCompGrid **psiComposite, HYPRE_Int **recv_map_send, HYPRE_Int *recv_map_size, HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int num_psi_levels, HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index )
{
   HYPRE_Int         level,i,j;
   HYPRE_Int         need_coarse_info;
   HYPRE_Int         num_nodes;
   HYPRE_Int         add_flag;
   HYPRE_Int         *num_added_nodes = hypre_CTAlloc(HYPRE_Int, num_levels);

   num_nodes = hypre_ParCompGridNumNodes(compGrid[current_level]);
   *recv_map_send_buffer_size = num_levels - current_level;

   // copy all info on this level (there will not be redundant info)
   for (i = 0; i < hypre_ParCompGridNumNodes(psiComposite[current_level]); i++) 
   {
      // check whether we need to allocate more space in order to add to compGrid on this level
      hypre_ParCompGridResize(compGrid[current_level]);

      // copy data into compGrid
      hypre_ParCompGridCopyNode( psiComposite[current_level], compGrid[current_level], i, num_nodes );

      // generate the receive map for this proc on this level
      recv_map_send[current_level][i] = num_nodes;
      recv_map_size[current_level]++;
      (*recv_map_send_buffer_size)++;

      // increment num_nodes
      num_nodes++;
      hypre_ParCompGridNumNodes(compGrid[current_level]) = num_nodes;

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
               hypre_ParCompGridResize(compGrid[level]);

               // copy data into compGrid
               hypre_ParCompGridCopyNode( psiComposite[level], compGrid[level], i, num_nodes );

               // generate the receive map for this proc on this level
               recv_map_send[level][i] = num_nodes;
               recv_map_size[level]++;
               (*recv_map_send_buffer_size)++;

               // increment num_nodes
               num_nodes++;
               hypre_ParCompGridNumNodes(compGrid[level]) = num_nodes;

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

   // all global info copied above, so now setup all local indices
   hypre_ParCompGridSetupLocalIndices(compGrid, num_added_nodes, num_levels, proc_first_index, proc_last_index);

   return hypre_error_flag;
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

   return hypre_error_flag;
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

   return hypre_error_flag;
}

HYPRE_Int
PackResidualBuffer( HYPRE_Complex *send_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, hypre_ParCompGrid **compGrid, hypre_ParCSRCommPkg *commPkg, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels )
{
   HYPRE_Int                  level,i,j,cnt = 0, flag_cnt;
   HYPRE_Int                  send_elmt;
   HYPRE_Int                  row_size;
   HYPRE_Int                  nodes_to_add = 0, coarse_grid_index, need_coarse_info;
   HYPRE_Int                  **add_flag = hypre_CTAlloc( HYPRE_Int*, num_levels );
   hypre_ParCompMatrixRow     *row;

   // Get where to look in commPkgSendMapElmts
   HYPRE_Int            start = hypre_ParCSRCommPkgSendMapStart(commPkg, processor);
   HYPRE_Int            finish = hypre_ParCSRCommPkgSendMapStart(commPkg, processor+1);

   // see whether we need coarse info and if so, set up add_flag
   if (current_level != num_levels-1) need_coarse_info = 1;
   else need_coarse_info = 0;
   if (need_coarse_info) add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[current_level+1]) );

   // pack the number of nodes sent on this level
   send_buffer[cnt++] = finish - start;

   // copy correct data into psiComposite from compGrid using sendMapElmts from commPkg
   // !!! CHECK ORDERING OF COMP GRIDS VS ORDERING OF SENDMAPELMTS... SHOULD BE OK I THINK... BUT CHECK !!!
   for (i = start; i < finish; i++)
   {
      // see whether we need coarse info
      if (current_level != num_levels-1) need_coarse_info = 1;
      else need_coarse_info = 0;

      // get index of element to send
      send_elmt = hypre_ParCSRCommPkgSendMapElmt(commPkg, i);

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

   return hypre_error_flag;
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

   return hypre_error_flag;
}
