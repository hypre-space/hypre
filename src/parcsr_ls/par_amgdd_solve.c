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

#define TEST_RES_COMM 1
#define DEBUGGING_MESSAGES 0

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"	

HYPRE_Int
AddSolution( void *amg_vdata );

HYPRE_Real
GetCompositeResidual(hypre_ParCompGrid *compGrid);

HYPRE_Int
ZeroInitialGuess( void *amg_vdata );

HYPRE_Int
PackResidualBuffer( HYPRE_Int proc, HYPRE_Complex *send_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
UnpackResidualBuffer( HYPRE_Int proc, HYPRE_Complex *recv_buffer, HYPRE_Int **recv_map, HYPRE_Int *num_recv_nodes, hypre_ParCompGrid **compGrid, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
TestResComm(hypre_ParAMGData *amg_data);

HYPRE_Int 
hypre_BoomerAMGDDSolve( void *amg_vdata,
                                 hypre_ParCSRMatrix *A,
                                 hypre_ParVector *f,
                                 hypre_ParVector *u )
{

   HYPRE_Int test_failed = 0;
   HYPRE_Int error_code;
   HYPRE_Int cycle_count = 0;
   HYPRE_Real alpha, beta;
   HYPRE_Real resid_nrm, resid_nrm_init, rhs_norm, relative_resid;

   // Get info from amg_data
   hypre_ParAMGData   *amg_data = amg_vdata;
   HYPRE_Real tol = hypre_ParAMGDataTol(amg_data);
   HYPRE_Int min_iter = hypre_ParAMGDataMinIter(amg_data);
   HYPRE_Int max_iter = hypre_ParAMGDataMaxIter(amg_data);
   HYPRE_Int converge_type = hypre_ParAMGDataConvergeType(amg_data);

   // Set the fine grid operator, left-hand side, and right-hand side
   hypre_ParAMGDataAArray(amg_data)[0] = A;
   hypre_ParAMGDataUArray(amg_data)[0] = u;
   hypre_ParAMGDataFArray(amg_data)[0] = f;

   // Store the original fine grid right-hand side in Vtemp and use f as the current fine grid residual
   hypre_ParVectorCopy(f, hypre_ParAMGDataVtemp(amg_data));
   alpha = -1.0;
   beta = 1.0;
   hypre_ParCSRMatrixMatvec(alpha, A, u, beta, f);

   // Setup convergence tolerance info
   if (tol > 0.)
   {
      resid_nrm = sqrt(hypre_ParVectorInnerProd(f,f));
      resid_nrm_init = resid_nrm;
      if (0 == converge_type)
      {
         rhs_norm = sqrt(hypre_ParVectorInnerProd(hypre_ParAMGDataVtemp(amg_data), hypre_ParAMGDataVtemp(amg_data)));
         if (rhs_norm)
         {
            relative_resid = resid_nrm_init / rhs_norm;
         }
         else
         {
            relative_resid = resid_nrm_init;
         }
      }
      else
      {
         /* converge_type != 0, test convergence with ||r|| / ||r0|| */
         relative_resid = 1.0;
      }
   }
   else
   {
      relative_resid = 1.;
   }

   // Main cycle loop
   while ( (relative_resid >= tol || cycle_count < min_iter) && cycle_count < max_iter )
   {
      // Do the AMGDD cycle
      error_code = hypre_BoomerAMGDD_Cycle(amg_vdata);
      if (error_code) test_failed = 1;

      // Calculate a new resiudal
      if (tol > 0.)
      {
         hypre_ParVectorCopy(hypre_ParAMGDataVtemp(amg_data), f);
         hypre_ParCSRMatrixMatvec(alpha, A, u, beta, f);
         resid_nrm = sqrt(hypre_ParVectorInnerProd(f,f));
         if (0 == converge_type)
         {
            if (rhs_norm)
            {
               relative_resid = resid_nrm / rhs_norm;
            }
            else
            {
               relative_resid = resid_nrm;
            }
         }
         else
         {
            relative_resid = resid_nrm / resid_nrm_init;
         }

         hypre_ParAMGDataRelativeResidualNorm(amg_data) = relative_resid;
      }
      ++cycle_count;

      hypre_ParAMGDataNumIterations(amg_data) = cycle_count;
   }

   // Copy RHS back into f
   hypre_ParVectorCopy(hypre_ParAMGDataVtemp(amg_data), f);

   return test_failed;
}

HYPRE_Int
hypre_BoomerAMGDD_Cycle( void *amg_vdata )
{
	HYPRE_Int   myid;
	hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

	HYPRE_Int i,j,k,level;
   HYPRE_Int cycle_count = 0;
	hypre_ParAMGData	*amg_data = amg_vdata;
	hypre_ParCompGrid 	**compGrid = hypre_ParAMGDataCompGrid(amg_data);
  	HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int min_fac_iter = hypre_ParAMGDataMinFACIter(amg_data);
   HYPRE_Int max_fac_iter = hypre_ParAMGDataMaxFACIter(amg_data);
   HYPRE_Real fac_tol = hypre_ParAMGDataFACTol(amg_data);

	// Form residual and do residual communication
   HYPRE_Int test_failed = 0;
	test_failed = hypre_BoomerAMGDDResidualCommunication( amg_vdata );

	// Set zero initial guess for all comp grids on all levels
	ZeroInitialGuess( amg_vdata );

   // Setup convergence tolerance info
   HYPRE_Real resid_nrm = 1.;
   if (fac_tol != 0.) resid_nrm = GetCompositeResidual(hypre_ParAMGDataCompGrid(amg_data)[0]);
   HYPRE_Real resid_nrm_init = resid_nrm;
   HYPRE_Real relative_resid = 1.;
   HYPRE_Real conv_fact = 0;
   
	// Do the cycles
   if (fac_tol == 0.0)
   {
      while ( cycle_count < max_fac_iter )
      {
         // Do FAC cycle
         hypre_BoomerAMGDD_FAC_Cycle( amg_vdata );

         ++cycle_count;
         hypre_ParAMGDataNumFACIterations(amg_data) = cycle_count;
      }
   }
   else if (fac_tol > 0)
   {
      while ( (relative_resid >= fac_tol || cycle_count < min_fac_iter) && cycle_count < max_fac_iter )
      {
         // Do FAC cycle
   		hypre_BoomerAMGDD_FAC_Cycle( amg_vdata );

         // Check convergence and up the cycle count
         resid_nrm = GetCompositeResidual(hypre_ParAMGDataCompGrid(amg_data)[0]);
         relative_resid = resid_nrm / resid_nrm_init;

         ++cycle_count;
         hypre_ParAMGDataNumFACIterations(amg_data) = cycle_count;
   	}
   }
   else if (fac_tol < 0)
   {
      fac_tol = -fac_tol;
      while ( (conv_fact <= fac_tol || conv_fact >= 1.0 || cycle_count < min_fac_iter) && cycle_count < max_fac_iter )
      {
         // Do FAC cycle
         hypre_BoomerAMGDD_FAC_Cycle( amg_vdata );

         // Check convergence and up the cycle count
         resid_nrm = GetCompositeResidual(hypre_ParAMGDataCompGrid(amg_data)[0]);
         conv_fact = resid_nrm / resid_nrm_init;
         resid_nrm_init = resid_nrm;
         ++cycle_count;
         hypre_ParAMGDataNumFACIterations(amg_data) = cycle_count;
      }
   }
   
	// Update fine grid solution
   AddSolution( amg_vdata );

	return test_failed;
}

HYPRE_Int
AddSolution( void *amg_vdata )
{
	hypre_ParAMGData	*amg_data = amg_vdata;
	HYPRE_Complex 		*u = hypre_VectorData( hypre_ParVectorLocalVector( hypre_ParAMGDataUArray(amg_data)[0] ) );
	hypre_ParCompGrid 	**compGrid = hypre_ParAMGDataCompGrid(amg_data);
	HYPRE_Complex 		*u_comp = hypre_ParCompGridU(compGrid[0]);
	HYPRE_Int 			num_owned_nodes = hypre_ParCompGridNumOwnedNodes(compGrid[0]);
	HYPRE_Int 			i;

	for (i = 0; i < num_owned_nodes; i++) u[i] += u_comp[i];

	return 0;
}

HYPRE_Real
GetCompositeResidual(hypre_ParCompGrid *compGrid)
{
   HYPRE_Int i,j;
   HYPRE_Real res_norm = 0.0;
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      if (hypre_ParCompGridRealDofMarker(compGrid)[i])
      {
         HYPRE_Real res = hypre_ParCompGridF(compGrid)[i];
         for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
         {
            res -= hypre_ParCompGridAData(compGrid)[j] * hypre_ParCompGridU(compGrid)[ hypre_ParCompGridAColInd(compGrid)[j] ];
         }
         res_norm += res*res;
      }
   }

   return sqrt(res_norm);
}

HYPRE_Int
ZeroInitialGuess( void *amg_vdata )
{
	HYPRE_Int   myid;
	hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

	hypre_ParAMGData	*amg_data = amg_vdata;
   HYPRE_Int i;

   HYPRE_Int level;
   for (level = 0; level < hypre_ParAMGDataNumLevels(amg_data); level++)
   {
      hypre_ParCompGrid    *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++) hypre_ParCompGridU(compGrid)[i] = 0.0;
   }
	
	return 0;
}

HYPRE_Int 
hypre_BoomerAMGDDResidualCommunication( void *amg_vdata )
{
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Began residual communication on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   MPI_Comm          comm;
   hypre_ParAMGData   *amg_data = amg_vdata;
   
   /* Data Structure variables */

   // level counters, indices, and parameters
   HYPRE_Int                  num_levels;
   HYPRE_Real                 alpha, beta;
   HYPRE_Int                  level,i,j;

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
   HYPRE_Int                  ***num_recv_nodes;
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
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg);
   if (transition_level < 0) transition_level = num_levels;
   send_procs = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg);
   recv_procs = hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg);
   send_buffer_size = hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg);
   recv_buffer_size = hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg);
   num_send_nodes = hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg);
   num_recv_nodes = hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg);
   send_flag = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg);
   recv_map = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg);

   // get first and last global indices on each level for this proc
   proc_first_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   proc_last_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   global_nodes = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   for (level = 0; level < num_levels; level++)
   {
      proc_first_index[level] = hypre_ParVectorFirstIndex(F_array[level]);
      proc_last_index[level] = hypre_ParVectorLastIndex(F_array[level]);
      global_nodes[level] = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
   }

   // Restrict residual down to all levels (or just to the transition level) and initialize composite grids
   for (level = 0; level < transition_level-1; level++)
   {
      alpha = 1.0;
      beta = 0.0;
      hypre_ParCSRMatrixMatvecT(alpha,P_array[level],F_array[level],
                            beta,F_array[level+1]);
   }
   if (transition_level != num_levels)
   {
      alpha = 1.0;
      beta = 0.0;
      hypre_ParCSRMatrixMatvecT(alpha,P_array[transition_level-1],F_array[transition_level-1],
                            beta,F_array[transition_level]);
   }

   // copy new restricted residual into comp grid structure
   for (level = 0; level < transition_level; level++)
   {
      // Access the residual data
      residual_local = hypre_ParVectorLocalVector(F_array[level]);
      residual_data = hypre_VectorData(residual_local);
      for (i = 0; i < hypre_VectorSize(residual_local); i++)
      {
         hypre_ParCompGridF(compGrid[level])[i] = residual_data[i];
      }
   }

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("About to do coarse levels allgather on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // Do Allgather of transition level 
   if (transition_level != num_levels)
   {
      residual_local = hypre_ParVectorLocalVector(F_array[transition_level]);
      residual_data = hypre_VectorData(residual_local);
      
      hypre_MPI_Allgatherv(residual_data, hypre_VectorSize(residual_local), HYPRE_MPI_COMPLEX, hypre_ParCompGridF(compGrid[transition_level]), hypre_ParCompGridCommPkgTransitionResRecvSizes(compGridCommPkg), hypre_ParCompGridCommPkgTransitionResRecvDisps(compGridCommPkg), HYPRE_MPI_COMPLEX, hypre_MPI_COMM_WORLD);
   }

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Entering looop over levels in residual communication on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   for (level = transition_level - 1; level >= 0; level--)
   {      
      // Get some communication info
      comm = hypre_ParCSRMatrixComm(A_array[level]);
      num_sends = hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[level];
      num_recvs = hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[level];

      if ( proc_last_index[level] >= proc_first_index[level] && num_sends ) // If there are any owned nodes on this level
      {

         // allocate space for the buffers, buffer sizes, requests and status, psiComposite_send, psiComposite_recv, send and recv maps
         recv_buffer = hypre_CTAlloc(HYPRE_Complex*, num_recvs, HYPRE_MEMORY_HOST);









         if (hypre_ParCompGridCommPkgUseAllgatherv(compGridCommPkg)[level])
         {


            // pack and send the buffers
            HYPRE_Complex *allgatherv_send_buffer = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridCommPkgResRecvcounts(compGridCommPkg)[level][myid], HYPRE_MEMORY_HOST);
            HYPRE_Int cnt = 0;
            allgatherv_send_buffer[cnt++] = num_sends;
            for (i = 0; i < num_sends; i++) allgatherv_send_buffer[cnt++] = send_procs[level][i];
            for (i = 0; i < num_sends; i++) allgatherv_send_buffer[cnt++] = send_buffer_size[level][i];




            // !!! Debug
            // if (myid == 0)
            // {
            //    HYPRE_Int sum_num_send_nodes = 0;
            //    HYPRE_Int sum_send_buffer_size = 0;
            //    for (i = 0; i < num_sends; i++)
            //    {
            //       sum_send_buffer_size += send_buffer_size[level][i];
            //       HYPRE_Int inner_level;
            //       for (inner_level = level; inner_level < num_levels; inner_level++) sum_num_send_nodes += num_send_nodes[level][i][inner_level];
            //    }
            //    printf("sum send_buffer_size = %d\n", sum_send_buffer_size);
            //    printf("sum num_send_nodes = %d\n", sum_num_send_nodes);
            //    printf("allgatherv_send_buffer size = %d\n", hypre_ParCompGridCommPkgResRecvcounts(compGridCommPkg)[level][myid]);
            // }


            HYPRE_Int offset = num_sends*2 + 1;
            for (i = 0; i < num_sends; i++)
            {
            // !!! Debug
               // if (myid == 0) printf("offset = %d\n", offset);
               PackResidualBuffer(send_procs[level][i], &(allgatherv_send_buffer[offset]), send_flag[level][i], num_send_nodes[level][i], compGrid, compGridCommPkg, i, level, num_levels);
               offset += send_buffer_size[level][i];
            }




            HYPRE_Complex *allgatherv_recv_buffer = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridCommPkgResRecvcounts(compGridCommPkg)[level][num_procs-1] + hypre_ParCompGridCommPkgResDispls(compGridCommPkg)[level][num_procs-1], HYPRE_MEMORY_HOST);
            hypre_MPI_Allgatherv(allgatherv_send_buffer, hypre_ParCompGridCommPkgResRecvcounts(compGridCommPkg)[level][myid], HYPRE_MPI_COMPLEX, 
               allgatherv_recv_buffer, hypre_ParCompGridCommPkgResRecvcounts(compGridCommPkg)[level], 
               hypre_ParCompGridCommPkgResDispls(compGridCommPkg)[level], HYPRE_MPI_COMPLEX, hypre_MPI_COMM_WORLD);



            // Copy appropriate buffers
            for (i = 0; i < num_recvs; i++)
            {

               // Find appropriate location to copy from 
               HYPRE_Int cnt = hypre_ParCompGridCommPkgResDispls(compGridCommPkg)[level][recv_procs[level][i]];
               HYPRE_Int num_psi_buffers = allgatherv_recv_buffer[cnt++];
               HYPRE_Int buffer_index = 0;
               for (j = 0; j < num_psi_buffers; j++) if (allgatherv_recv_buffer[cnt++] == myid) buffer_index = j;
               HYPRE_Int read_location = cnt + num_psi_buffers;
               for (j = 0; j < buffer_index; j++) read_location += allgatherv_recv_buffer[cnt++];

               // Allocate and fill the recv buffer
               recv_buffer[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size[level][i], HYPRE_MEMORY_HOST );
               cnt = read_location;
               for (j = 0; j < recv_buffer_size[level][i]; j++) recv_buffer[i][j] = allgatherv_recv_buffer[cnt++];




            }















         }
         else
         {
            request_counter = 0;
            requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs, HYPRE_MEMORY_HOST );
            status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs, HYPRE_MEMORY_HOST );
            send_buffer = hypre_CTAlloc(HYPRE_Complex*, num_sends, HYPRE_MEMORY_HOST);

            // allocate space for the receive buffers and post the receives
            for (i = 0; i < num_recvs; i++)
            {
               recv_buffer[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size[level][i], HYPRE_MEMORY_HOST );
               hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_COMPLEX, recv_procs[level][i], 3, comm, &requests[request_counter++]);
            }

            // pack and send the buffers
            for (i = 0; i < num_sends; i++)
            {
               send_buffer[i] = hypre_CTAlloc(HYPRE_Complex, send_buffer_size[level][i], HYPRE_MEMORY_HOST);
               PackResidualBuffer(send_procs[level][i], send_buffer[i], send_flag[level][i], num_send_nodes[level][i], compGrid, compGridCommPkg, i, level, num_levels);
               hypre_MPI_Isend(send_buffer[i], send_buffer_size[level][i], HYPRE_MPI_COMPLEX, send_procs[level][i], 3, comm, &requests[request_counter++]);
            }

            #if DEBUGGING_MESSAGES
            hypre_printf("    Posted sends and recvs on level %d, rank %d\n", level, myid);
            #endif

            // wait for buffers to be received
            hypre_MPI_Waitall( num_sends + num_recvs, requests, status );

            #if DEBUGGING_MESSAGES
            hypre_printf("    Finished waiting on level %d, rank %d\n", level, myid);
            #endif


            hypre_TFree(requests, HYPRE_MEMORY_HOST);
            hypre_TFree(status, HYPRE_MEMORY_HOST);
            for (i = 0; i < num_sends; i++)
            {
               hypre_TFree(send_buffer[i], HYPRE_MEMORY_HOST);
            }
            hypre_TFree(send_buffer, HYPRE_MEMORY_HOST);
         }

         // loop over received buffers
         for (i = 0; i < num_recvs; i++)
         {
            // unpack the buffers
            UnpackResidualBuffer(recv_procs[level][i], recv_buffer[i], recv_map[level][i], num_recv_nodes[level][i], compGrid, level, num_levels);
         }

         // clean up memory for this level
         for (i = 0; i < num_recvs; i++)
         {
            hypre_TFree(recv_buffer[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(recv_buffer, HYPRE_MEMORY_HOST);
      }

      #if DEBUGGING_MESSAGES
      hypre_printf("    Finished residual communication on level %d, rank %d\n", level, myid);
      #endif

   }

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Finished residual communication on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   #if TEST_RES_COMM
   HYPRE_Int test_failed = TestResComm(amg_data);
   #endif

   // Cleanup memory
   hypre_TFree(proc_first_index, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_last_index, HYPRE_MEMORY_HOST);
   hypre_TFree(global_nodes, HYPRE_MEMORY_HOST);
   
   #if TEST_RES_COMM
   return test_failed;
   #else
   return 0;
   #endif
}

HYPRE_Int
PackResidualBuffer( HYPRE_Int proc, HYPRE_Complex *send_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int processor, HYPRE_Int current_level, HYPRE_Int num_levels )
{
   HYPRE_Int                  level,i,cnt = 0;

   // pack the send buffer
   for (level = current_level; level < num_levels; level++)
   {
      for (i = 0; i < num_send_nodes[level]; i++) send_buffer[cnt++] = hypre_ParCompGridF(compGrid[level])[ send_flag[level][i] ];
   }

   return 0;

}

HYPRE_Int
UnpackResidualBuffer( HYPRE_Int proc, HYPRE_Complex *recv_buffer, HYPRE_Int **recv_map, HYPRE_Int *num_recv_nodes, hypre_ParCompGrid **compGrid, HYPRE_Int current_level, HYPRE_Int num_levels)
{
   HYPRE_Int                  level,i,cnt = 0, map_cnt, num_nodes;

   // loop over levels
   for (level = current_level; level < num_levels; level++)
   {
      for (i = 0; i < num_recv_nodes[level]; i++) hypre_ParCompGridF(compGrid[level])[ recv_map[level][i] ] = recv_buffer[cnt++];
   }

   return 0;
}

HYPRE_Int
TestResComm(hypre_ParAMGData *amg_data)
{
   // Get MPI info
   HYPRE_Int myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   // Get info from the amg data structure
   hypre_ParCompGrid **compGrid = hypre_ParAMGDataCompGrid(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;

   HYPRE_Int test_failed = 0;

   // For each processor and each level broadcast the residual data and global indices out and check agains the owning procs
   HYPRE_Int proc;
   HYPRE_Int i;
   for (proc = 0; proc < num_procs; proc++)
   {
      HYPRE_Int level;
      for (level = 0; level < transition_level; level++)
      {
         // Broadcast the number of nodes
         HYPRE_Int num_nodes = 0;
         if (myid == proc) num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
         hypre_MPI_Bcast(&num_nodes, 1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the composite residual
         HYPRE_Complex *comp_res;
         if (myid == proc) comp_res = hypre_ParCompGridF(compGrid[level]);
         else comp_res = hypre_CTAlloc(HYPRE_Complex, num_nodes, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(comp_res, num_nodes, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the global indices
         HYPRE_Int *global_indices;
         if (myid == proc) global_indices = hypre_ParCompGridGlobalIndices(compGrid[level]);
         else global_indices = hypre_CTAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(global_indices, num_nodes, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Now, each processors checks their owned residual value against the composite residual
         HYPRE_Int proc_first_index = hypre_ParVectorFirstIndex(hypre_ParAMGDataUArray(amg_data)[level]);
         HYPRE_Int proc_last_index = hypre_ParVectorLastIndex(hypre_ParAMGDataUArray(amg_data)[level]);
         for (i = 0; i < num_nodes; i++)
         {
            if (global_indices[i] <= proc_last_index && global_indices[i] >= proc_first_index)
            {
               if (comp_res[i] != hypre_VectorData(hypre_ParVectorLocalVector(hypre_ParAMGDataFArray(amg_data)[level]))[global_indices[i] - proc_first_index] )
               {
                  // printf("Error: on proc %d has incorrect residual at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                  test_failed = 1;
               }
            }
         }

         // Clean up memory
         if (myid != proc) 
         {
            hypre_TFree(comp_res, HYPRE_MEMORY_HOST);
            hypre_TFree(global_indices, HYPRE_MEMORY_HOST);
         }
      }
      if (transition_level != num_levels)
      {
         HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[transition_level]);

         // Broadcast the composite residual
         HYPRE_Complex *comp_res;
         if (myid == proc) comp_res = hypre_ParCompGridF(compGrid[transition_level]);
         else comp_res = hypre_CTAlloc(HYPRE_Complex, num_nodes, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(comp_res, num_nodes, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);

         // Now, each processors checks their owned residual value against the composite residual
         HYPRE_Int proc_first_index = hypre_ParVectorFirstIndex(hypre_ParAMGDataUArray(amg_data)[transition_level]);
         HYPRE_Int proc_last_index = hypre_ParVectorLastIndex(hypre_ParAMGDataUArray(amg_data)[transition_level]);
         for (i = 0; i < num_nodes; i++)
         {
            if (i <= proc_last_index && i >= proc_first_index)
            {
               if (comp_res[i] != hypre_VectorData(hypre_ParVectorLocalVector(hypre_ParAMGDataFArray(amg_data)[transition_level]))[i - proc_first_index] )
               {
                  // printf("Error: on proc %d has incorrect residual at global index %d on transition_level %d, checked by rank %d\n", proc, i, transition_level, myid);
                  test_failed = 1;
               }
            }
         }

         // Clean up memory
         if (myid != proc) 
         {
            hypre_TFree(comp_res, HYPRE_MEMORY_HOST);
         }         
      }
   }

   return test_failed;
}
