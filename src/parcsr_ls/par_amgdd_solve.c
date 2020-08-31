/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"


HYPRE_Int
hypre_BoomerAMGDDSolve( void *amgdd_vdata,
                                 hypre_ParCSRMatrix *A,
                                 hypre_ParVector *f,
                                 hypre_ParVector *u )
{

   HYPRE_Int error_code;
   HYPRE_Int cycle_count = 0;
   HYPRE_Real resid_nrm, resid_nrm_init, rhs_norm, relative_resid;

   // Get info from amg_data
   hypre_ParAMGDDData   *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_ParAMGData     *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);
   HYPRE_Real tol = hypre_ParAMGDataTol(amg_data);
   HYPRE_Int min_iter = hypre_ParAMGDataMinIter(amg_data);
   HYPRE_Int max_iter = hypre_ParAMGDataMaxIter(amg_data);
   HYPRE_Int converge_type = hypre_ParAMGDataConvergeType(amg_data);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDDDataStartLevel(amgdd_data);

   // Setup extra temporary variable to hold the solution if necessary
   if (!hypre_ParAMGDataZtemp(amg_data))
   {
      hypre_ParAMGDataZtemp(amg_data) = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(hypre_ParAMGDataAArray(amg_data)[amgdd_start_level]),
                                    hypre_ParCSRMatrixGlobalNumRows(hypre_ParAMGDataAArray(amg_data)[amgdd_start_level]),
                                    hypre_ParCSRMatrixRowStarts(hypre_ParAMGDataAArray(amg_data)[amgdd_start_level]));
      hypre_ParVectorInitialize(hypre_ParAMGDataZtemp(amg_data));
      hypre_ParVectorSetPartitioningOwner(hypre_ParAMGDataZtemp(amg_data),0);
   }

   // Set the fine grid operator, left-hand side, and right-hand side
   hypre_ParAMGDataAArray(amg_data)[0] = A;
   hypre_AMGDDCompGrid *compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data)[0];
   if (A != hypre_ParAMGDataAArray(amg_data)[0])
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"WARNING: calling hypre_BoomerAMGDDSolve with different matrix than what was used for initial setup. "
            "Non-owned parts of fine-grid matrix and fine-grid communication patterns may be incorrect.\n");
      hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid)) = hypre_ParCSRMatrixDiag(A);
      hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid)) = hypre_ParCSRMatrixOffd(A);
   }
   hypre_ParAMGDataUArray(amg_data)[0] = u;
   hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridU(compGrid)) = hypre_ParVectorLocalVector(u);
   hypre_ParAMGDataFArray(amg_data)[0] = f;
   hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridF(compGrid)) = hypre_ParVectorLocalVector(f);

   // Setup convergence tolerance info
   if (tol > 0.)
   {
      hypre_ParVectorCopy(f, hypre_ParAMGDataVtemp(amg_data));
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, hypre_ParAMGDataVtemp(amg_data));
      resid_nrm = sqrt(hypre_ParVectorInnerProd(hypre_ParAMGDataVtemp(amg_data),hypre_ParAMGDataVtemp(amg_data)));
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
      // Do normal AMG V-cycle downsweep to where we start AMG-DD
      if (amgdd_start_level > 0)
      {
         hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = amgdd_start_level - 1;
         hypre_ParAMGDataPartialCycleControl(amg_data) = 0;
         hypre_BoomerAMGCycle( (void*) amg_data, hypre_ParAMGDataFArray(amg_data), hypre_ParAMGDataUArray(amg_data));
      }
      else
      {
         // Store the original fine grid right-hand side in Vtemp and use f as the current fine grid residual
         hypre_ParVectorCopy(hypre_ParAMGDataFArray(amg_data)[amgdd_start_level], hypre_ParAMGDataVtemp(amg_data));
         hypre_ParCSRMatrixMatvec(-1.0, hypre_ParAMGDataAArray(amg_data)[amgdd_start_level], hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], 1.0, hypre_ParAMGDataFArray(amg_data)[amgdd_start_level]);
      }

      error_code = hypre_BoomerAMGDD_Cycle(amgdd_data);

      // Do normal AMG V-cycle upsweep back up to the fine grid
      if (amgdd_start_level > 0)
      {
         // Interpolate
         hypre_ParCSRMatrixMatvec(1.0, hypre_ParAMGDataPArray(amg_data)[amgdd_start_level-1], hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], 1.0, hypre_ParAMGDataUArray(amg_data)[amgdd_start_level-1]);
         // V-cycle back to finest grid
         hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = amgdd_start_level - 1;
         hypre_ParAMGDataPartialCycleControl(amg_data) = 1;

         hypre_BoomerAMGCycle( (void*) amg_data, hypre_ParAMGDataFArray(amg_data), hypre_ParAMGDataUArray(amg_data));

         hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = - 1;
         hypre_ParAMGDataPartialCycleControl(amg_data) = -1;
      }
      else
      {
         // Copy RHS back into f
         hypre_ParVectorCopy(hypre_ParAMGDataVtemp(amg_data), hypre_ParAMGDataFArray(amg_data)[amgdd_start_level]);
      }

      // Calculate a new resiudal
      if (tol > 0.)
      {
         hypre_ParVectorCopy(f, hypre_ParAMGDataVtemp(amg_data));
         hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, hypre_ParAMGDataVtemp(amg_data));
         resid_nrm = sqrt(hypre_ParVectorInnerProd(hypre_ParAMGDataVtemp(amg_data),hypre_ParAMGDataVtemp(amg_data)));
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

   return error_code;
}

HYPRE_Int
hypre_BoomerAMGDD_Cycle( hypre_ParAMGDDData *amgdd_data )
{
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

	HYPRE_Int i,level;
   hypre_ParAMGData     *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);
  	HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDDDataStartLevel(amgdd_data);
   HYPRE_Int fac_num_cycles = hypre_ParAMGDDDataFACNumCycles(amgdd_data);

   // do residual communication
   hypre_BoomerAMGDD_ResidualCommunication( amgdd_data );

   // Save the original solution (updated at the end of the AMG-DD cycle)
   hypre_ParVectorCopy(hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], hypre_ParAMGDataZtemp(amg_data));

   // Zero solution on all levels
   for (level = amgdd_start_level; level < hypre_ParAMGDataNumLevels(amg_data); level++)
   {
      hypre_AMGDDCompGridVectorSetConstantValues( hypre_AMGDDCompGridU(hypre_ParAMGDDDataCompGrid(amgdd_data)[level]), 0.0);
      if (hypre_AMGDDCompGridQ(hypre_ParAMGDDDataCompGrid(amgdd_data)[level])) hypre_AMGDDCompGridVectorSetConstantValues( hypre_AMGDDCompGridQ(hypre_ParAMGDDDataCompGrid(amgdd_data)[level]), 0.0);
   }

   for (level = amgdd_start_level; level < num_levels; level++)
   {
      hypre_AMGDDCompGridVectorSetConstantValues( hypre_AMGDDCompGridT( hypre_ParAMGDDDataCompGrid(amgdd_data)[level] ), 0.0 );
      hypre_AMGDDCompGridVectorSetConstantValues( hypre_AMGDDCompGridS( hypre_ParAMGDDDataCompGrid(amgdd_data)[level] ), 0.0 );
   }

	// Do the cycles
   HYPRE_Int first_iteration = 1;
   for (i = 0; i < fac_num_cycles; i++)
   {
      // Do FAC cycle
      hypre_BoomerAMGDD_FAC( (void*) amgdd_data, first_iteration );
      first_iteration = 0;
   }

	// Update fine grid solution
   hypre_ParVectorAxpy( 1.0, hypre_ParAMGDataZtemp(amg_data), hypre_ParAMGDataUArray(amg_data)[amgdd_start_level]);

	return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_ResidualCommunication( hypre_ParAMGDDData *amgdd_data )
{
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   MPI_Comm          comm;
   hypre_ParAMGData     *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);

   /* Data Structure variables */

   // level counters, indices, and parameters
   HYPRE_Int                  num_levels, amgdd_start_level;
   HYPRE_Int                  level,i;

   // info from amg
   hypre_ParCSRMatrix         **A_array;
   hypre_ParVector            **F_array;
   hypre_ParCSRMatrix         **R_array;
   hypre_AMGDDCompGrid          **compGrid;

   // info from comp grid comm pkg
   hypre_AMGDDCommPkg   *compGridCommPkg;

   // temporary arrays used for communication during comp grid setup
   HYPRE_Complex              **send_buffers;
   HYPRE_Complex              **recv_buffers;

   // mpi stuff
   hypre_MPI_Request          *requests;
   hypre_MPI_Status           *status;
   HYPRE_Int                  request_counter = 0;

   // get info from amg
   A_array = hypre_ParAMGDataAArray(amg_data);
   R_array = hypre_ParAMGDataRArray(amg_data);
   F_array = hypre_ParAMGDataFArray(amg_data);
   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   amgdd_start_level = hypre_ParAMGDDDataStartLevel(amgdd_data);
   compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data);
   compGridCommPkg = hypre_ParAMGDDDataCommPkg(amgdd_data);

   // Restrict residual down to all levels
   for (level = amgdd_start_level; level < num_levels-1; level++)
   {
      if ( hypre_ParAMGDataRestriction(amg_data) ) hypre_ParCSRMatrixMatvec(1.0, R_array[level], F_array[level], 0.0, F_array[level+1]);
      else hypre_ParCSRMatrixMatvecT(1.0, R_array[level], F_array[level], 0.0, F_array[level+1]);
   }

   if (num_procs > 1)
   {
      /* Outer loop over levels:
      Start from coarsest level and work up to finest */
      for (level = num_levels - 1; level >= amgdd_start_level; level--)
      {
         // Get some communication info
         comm = hypre_ParCSRMatrixComm(A_array[level]);
         HYPRE_Int num_sends = hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level];
         HYPRE_Int num_recvs = hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level];

         if ( num_sends || num_recvs ) // If there are any owned nodes on this level
         {
            // allocate space for the buffers, buffer sizes, requests and status, psiComposite_send, psiComposite_recv, send and recv maps
            recv_buffers = hypre_CTAlloc(HYPRE_Complex*, num_recvs, HYPRE_MEMORY_HOST);
            send_buffers = hypre_CTAlloc(HYPRE_Complex*, num_sends, HYPRE_MEMORY_HOST);
            request_counter = 0;
            requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs, HYPRE_MEMORY_HOST );
            status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs, HYPRE_MEMORY_HOST );

            // allocate space for the receive buffers and post the receives
            for (i = 0; i < num_recvs; i++)
            {
               HYPRE_Int recv_buffer_size = hypre_AMGDDCommPkgRecvBufferSize(compGridCommPkg)[level][i];
               recv_buffers[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size, HYPRE_MEMORY_HOST);
               hypre_MPI_Irecv(recv_buffers[i], recv_buffer_size, HYPRE_MPI_COMPLEX, hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 3, comm, &requests[request_counter++]);
            }

            for (i = 0; i < num_sends; i++)
            {
               HYPRE_Int send_buffer_size = hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg)[level][i];
               send_buffers[i] = hypre_BoomerAMGDD_PackResidualBuffer(compGrid, compGridCommPkg, level, i);
               hypre_MPI_Isend(send_buffers[i], send_buffer_size, HYPRE_MPI_COMPLEX, hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 3, comm, &requests[request_counter++]);
            }

            // wait for buffers to be received
            hypre_MPI_Waitall( request_counter, requests, status );

            hypre_TFree(requests, HYPRE_MEMORY_HOST);
            hypre_TFree(status, HYPRE_MEMORY_HOST);
            for (i = 0; i < num_sends; i++) hypre_TFree(send_buffers[i], HYPRE_MEMORY_HOST);
            hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);

            // Unpack recv buffers
            for (i = 0; i < num_recvs; i++)
            {
               hypre_BoomerAMGDD_UnpackResidualBuffer(recv_buffers[i], compGrid, compGridCommPkg, level, i);
            }

            // clean up memory for this level
            for (i = 0; i < num_recvs; i++) hypre_TFree(recv_buffers[i], HYPRE_MEMORY_HOST);
            hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
         }
      }
   }

   return 0;
}

HYPRE_Complex*
hypre_BoomerAMGDD_PackResidualBuffer( hypre_AMGDDCompGrid **compGrid, hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc )
{
   HYPRE_Int level,i;

   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Complex *buffer = hypre_CTAlloc(HYPRE_Complex, hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg)[current_level][proc], HYPRE_MEMORY_HOST);

   HYPRE_Int cnt = 0;
   for (level = current_level; level < hypre_AMGDDCommPkgNumLevels(compGridCommPkg); level++)
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[current_level][proc][level]; i++)
      {
         HYPRE_Int send_elmt = hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[current_level][proc][level][i];
         if (send_elmt < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
            buffer[cnt++] = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridF(compGrid[level])))[send_elmt];
         else
         {
            send_elmt -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
            buffer[cnt++] = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridF(compGrid[level])))[send_elmt];
         }
      }
   }

   return buffer;
}

HYPRE_Int
hypre_BoomerAMGDD_UnpackResidualBuffer( HYPRE_Complex *buffer, hypre_AMGDDCompGrid **compGrid, hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc )
{
   HYPRE_Int level,i;

   HYPRE_Int cnt = 0;
   for (level = current_level; level < hypre_AMGDDCommPkgNumLevels(compGridCommPkg); level++)
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[current_level][proc][level]; i++)
      {
         HYPRE_Int recv_elmt = hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[current_level][proc][level][i];
         hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridF(compGrid[level])))[recv_elmt] = buffer[cnt++];
      }
   }

   return 0;
}


