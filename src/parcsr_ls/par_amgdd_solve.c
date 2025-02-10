/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

HYPRE_Int
hypre_BoomerAMGDDSolve( void               *amgdd_vdata,
                        hypre_ParCSRMatrix *A,
                        hypre_ParVector    *f,
                        hypre_ParVector    *u )
{
   hypre_ParAMGDDData   *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_ParAMGData     *amg_data   = hypre_ParAMGDDDataAMG(amgdd_data);

   hypre_AMGDDCompGrid **compGrids;
   hypre_ParCSRMatrix  **A_array;
   hypre_ParCSRMatrix  **P_array;
   hypre_ParVector     **F_array;
   hypre_ParVector     **U_array;
   hypre_ParVector      *res = NULL;
   hypre_ParVector      *Vtemp;
   hypre_ParVector      *Ztemp;

   HYPRE_Int             myid;
   HYPRE_Int             min_iter;
   HYPRE_Int             max_iter;
   HYPRE_Int             converge_type;
   HYPRE_Int             i, level;
   HYPRE_Int             num_levels;
   HYPRE_Int             amgdd_start_level;
   HYPRE_Int             fac_num_cycles;
   HYPRE_Int             cycle_count;
   HYPRE_Int             amg_print_level;
   HYPRE_Int             amg_logging;
   HYPRE_Real            tol;
   HYPRE_Real            resid_nrm = 0.0;
   HYPRE_Real            resid_nrm_init = 1.0;
   HYPRE_Real            rhs_norm = 1.0;
   HYPRE_Real            old_resid;
   HYPRE_Real            relative_resid;
   HYPRE_Real            conv_factor;
   HYPRE_Real            alpha = -1.0;
   HYPRE_Real            beta = 1.0;
   HYPRE_Real            ieee_check = 0.0;

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /* Set some data */
   amgdd_start_level = hypre_ParAMGDDDataStartLevel(amgdd_data);
   fac_num_cycles    = hypre_ParAMGDDDataFACNumCycles(amgdd_data);
   compGrids         = hypre_ParAMGDDDataCompGrid(amgdd_data);
   amg_print_level   = hypre_ParAMGDataPrintLevel(amg_data);
   amg_logging       = hypre_ParAMGDataLogging(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   converge_type     = hypre_ParAMGDataConvergeType(amg_data);
   min_iter          = hypre_ParAMGDataMinIter(amg_data);
   max_iter          = hypre_ParAMGDataMaxIter(amg_data);
   A_array           = hypre_ParAMGDataAArray(amg_data);
   P_array           = hypre_ParAMGDataPArray(amg_data);
   F_array           = hypre_ParAMGDataFArray(amg_data);
   U_array           = hypre_ParAMGDataUArray(amg_data);
   Vtemp             = hypre_ParAMGDataVtemp(amg_data);
   Ztemp             = hypre_ParAMGDDDataZtemp(amg_data);
   tol               = hypre_ParAMGDataTol(amg_data);
   cycle_count       = 0;
   if (amg_logging > 1)
   {
      res = hypre_ParAMGDataResidual(amg_data);
   }

   // Setup extra temporary variable to hold the solution if necessary
   if (!Ztemp)
   {
      Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[amgdd_start_level]),
                                    hypre_ParCSRMatrixGlobalNumRows(A_array[amgdd_start_level]),
                                    hypre_ParCSRMatrixRowStarts(A_array[amgdd_start_level]));
      hypre_ParVectorInitialize(Ztemp);
      hypre_ParAMGDDDataZtemp(amg_data) = Ztemp;
   }

   /*-----------------------------------------------------------------------
    * Write the solver parameters
    *-----------------------------------------------------------------------*/
   if (myid == 0 && amg_print_level > 1)
   {
      hypre_BoomerAMGWriteSolverParams(amg_data);
   }

   /*-----------------------------------------------------------------------
    * Set the fine grid operator, left-hand side, and right-hand side
    *-----------------------------------------------------------------------*/
   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;
   if (A != A_array[0])
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "WARNING: calling hypre_BoomerAMGDDSolve with different matrix than what was used for initial setup. "
                        "Non-owned parts of fine-grid matrix and fine-grid communication patterns may be incorrect.\n");
      hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrids[0])) = hypre_ParCSRMatrixDiag(A);
      hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrids[0])) = hypre_ParCSRMatrixOffd(A);
   }

   if (compGrids[0])
   {
      hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridU(compGrids[0])) = hypre_ParVectorLocalVector(u);
      hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridF(compGrids[0])) = hypre_ParVectorLocalVector(f);
   }

   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print
    *-----------------------------------------------------------------------*/
   if (amg_print_level > 1 || amg_logging > 1 || tol > 0.)
   {
      if (amg_logging > 1)
      {
         hypre_ParVectorCopy(F_array[0], res);
         if (tol > 0.)
         {
            hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, res);
         }
         resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(res, res));
      }
      else
      {
         hypre_ParVectorCopy(F_array[0], Vtemp);
         if (tol > 0.)
         {
            hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
         }
         resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
      }

      /* Since it does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (resid_nrm != 0.)
      {
         ieee_check = resid_nrm / resid_nrm; /* INF -> NaN conversion */
      }

      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (amg_print_level > 0)
         {
            hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            hypre_printf("ERROR -- hypre_BoomerAMGDDSolve: INFs and/or NaNs detected in input.\n");
            hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         hypre_error(HYPRE_ERROR_GENERIC);

         return hypre_error_flag;
      }

      /* r0 */
      resid_nrm_init = resid_nrm;

      if (0 == converge_type)
      {
         rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(f, f));
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

   if (myid == 0 && amg_print_level > 1)
   {
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n",
                   resid_nrm_init, relative_resid);
   }

   /*-----------------------------------------------------------------------
    *    Main cycle loop
    *-----------------------------------------------------------------------*/
   while ( (relative_resid >= tol || cycle_count < min_iter) && cycle_count < max_iter )
   {
      // Do normal AMG V-cycle down-sweep to where we start AMG-DD
      if (amgdd_start_level > 0)
      {
         hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = amgdd_start_level - 1;
         hypre_ParAMGDataPartialCycleControl(amg_data) = 0;
         hypre_BoomerAMGCycle( (void*) amg_data, F_array, U_array);
      }
      else
      {
         // Store the original fine grid right-hand side in Vtemp and use f as the current fine grid residual
         hypre_ParVectorCopy(F_array[amgdd_start_level], Vtemp);
         hypre_ParCSRMatrixMatvec(alpha, A_array[amgdd_start_level],
                                  U_array[amgdd_start_level], beta,
                                  F_array[amgdd_start_level]);
      }

      // AMG-DD cycle
      hypre_BoomerAMGDD_ResidualCommunication(amgdd_data);

      // Save the original solution (updated at the end of the AMG-DD cycle)
      hypre_ParVectorCopy(U_array[amgdd_start_level], Ztemp);

      // Zero solution on all levels
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         hypre_AMGDDCompGridVectorSetConstantValues(hypre_AMGDDCompGridU(compGrids[level]), 0.0);

         if (hypre_AMGDDCompGridQ(compGrids[level]))
         {
            hypre_AMGDDCompGridVectorSetConstantValues(hypre_AMGDDCompGridQ(compGrids[level]), 0.0);
         }
      }

      for (level = amgdd_start_level; level < num_levels; level++)
      {
         hypre_AMGDDCompGridVectorSetConstantValues(hypre_AMGDDCompGridT(compGrids[level]), 0.0 );
         hypre_AMGDDCompGridVectorSetConstantValues(hypre_AMGDDCompGridS(compGrids[level]), 0.0 );
      }

      // Do FAC cycles
      if (fac_num_cycles > 0)
      {
         hypre_BoomerAMGDD_FAC((void*) amgdd_data, 1);
      }
      for (i = 1; i < fac_num_cycles; i++)
      {
         hypre_BoomerAMGDD_FAC((void*) amgdd_data, 0);
      }

      // Update fine grid solution
      hypre_ParVectorAxpy(1.0, Ztemp, U_array[amgdd_start_level]);

      // Do normal AMG V-cycle up-sweep back up to the fine grid
      if (amgdd_start_level > 0)
      {
         // Interpolate
         hypre_ParCSRMatrixMatvec(1.0, P_array[amgdd_start_level - 1],
                                  U_array[amgdd_start_level], 1.0,
                                  U_array[amgdd_start_level - 1]);
         // V-cycle back to finest grid
         hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = amgdd_start_level - 1;
         hypre_ParAMGDataPartialCycleControl(amg_data) = 1;

         hypre_BoomerAMGCycle((void*) amg_data, F_array, U_array);

         hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = - 1;
         hypre_ParAMGDataPartialCycleControl(amg_data) = -1;
      }
      else
      {
         // Copy RHS back into f
         hypre_ParVectorCopy(Vtemp, F_array[amgdd_start_level]);
      }

      /*---------------------------------------------------------------
       * Compute fine-grid residual and residual norm
       *----------------------------------------------------------------*/
      if (amg_print_level > 1 || amg_logging > 1 || tol > 0.)
      {
         old_resid = resid_nrm;

         if (amg_logging > 1)
         {
            hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[0], U_array[0], beta,
                                               F_array[0], res);
            resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(res, res));
         }
         else
         {
            hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[0], U_array[0], beta,
                                               F_array[0], Vtemp);
            resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
         }

         if (old_resid)
         {
            conv_factor = resid_nrm / old_resid;
         }
         else
         {
            conv_factor = resid_nrm;
         }

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

      if (myid == 0 && amg_print_level > 1)
      {
         hypre_printf("    Cycle %2d   %e    %f     %e \n", cycle_count,
                      resid_nrm, conv_factor, relative_resid);
      }

      // Update cycle counter
      ++cycle_count;
      hypre_ParAMGDataNumIterations(amg_data) = cycle_count;
   }

   if (cycle_count == max_iter && tol > 0.)
   {
      if (myid == 0)
      {
         hypre_printf("\n\n==============================================");
         hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
         hypre_printf("      within the allowed %d V-cycles\n", max_iter);
         hypre_printf("==============================================");
      }

      hypre_error(HYPRE_ERROR_CONV);
   }

   if (myid == 0 && amg_print_level > 1)
   {
      hypre_printf("\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * TODO: Don't reallocate requests/sends at each level. Implement
 *       a hypre_AMGDDCommPkgHandle data structure (see hypre_ParCSRCommHandle)
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGDD_ResidualCommunication( hypre_ParAMGDDData *amgdd_data )
{
   hypre_ParAMGData      *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);

   // info from amg
   hypre_ParCSRMatrix   **A_array;
   hypre_ParCSRMatrix   **R_array;
   hypre_ParVector      **F_array;
   hypre_AMGDDCommPkg    *compGridCommPkg;
   hypre_AMGDDCompGrid  **compGrid;

   // temporary arrays used for communication during comp grid setup
   HYPRE_Complex        **send_buffers;
   HYPRE_Complex        **recv_buffers;

   // MPI stuff
   MPI_Comm               comm;
   hypre_MPI_Request     *requests;
   hypre_MPI_Status      *status;
   HYPRE_Int              request_counter = 0;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_sends, num_recvs;
   HYPRE_Int              send_buffer_size, recv_buffer_size;

   HYPRE_Int              num_levels, amgdd_start_level;
   HYPRE_Int              level, i;

   // Get info from amg
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   amgdd_start_level = hypre_ParAMGDDDataStartLevel(amgdd_data);
   compGrid          = hypre_ParAMGDDDataCompGrid(amgdd_data);
   compGridCommPkg   = hypre_ParAMGDDDataCommPkg(amgdd_data);
   A_array           = hypre_ParAMGDataAArray(amg_data);
   R_array           = hypre_ParAMGDataRArray(amg_data);
   F_array           = hypre_ParAMGDataFArray(amg_data);

   // Restrict residual down to all levels
   for (level = amgdd_start_level; level < num_levels - 1; level++)
   {
      if (hypre_ParAMGDataRestriction(amg_data))
      {
         hypre_ParCSRMatrixMatvec(1.0, R_array[level], F_array[level], 0.0, F_array[level + 1]);
      }
      else
      {
         hypre_ParCSRMatrixMatvecT(1.0, R_array[level], F_array[level], 0.0, F_array[level + 1]);
      }
   }

   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   for (level = num_levels - 1; level >= amgdd_start_level; level--)
   {
      // Get some communication info
      comm = hypre_ParCSRMatrixComm(A_array[level]);
      hypre_MPI_Comm_size(comm, &num_procs);

      if (num_procs > 1)
      {
         num_sends = hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level];
         num_recvs = hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level];

         if ( num_sends || num_recvs ) // If there are any owned nodes on this level
         {
            // allocate space for the buffers, buffer sizes, requests and status, psiComposite_send, psiComposite_recv, send and recv maps
            recv_buffers = hypre_CTAlloc(HYPRE_Complex *, num_recvs, HYPRE_MEMORY_HOST);
            send_buffers = hypre_CTAlloc(HYPRE_Complex *, num_sends, HYPRE_MEMORY_HOST);
            request_counter = 0;
            requests = hypre_CTAlloc(hypre_MPI_Request, num_sends + num_recvs, HYPRE_MEMORY_HOST);
            status = hypre_CTAlloc(hypre_MPI_Status, num_sends + num_recvs, HYPRE_MEMORY_HOST);

            // allocate space for the receive buffers and post the receives
            for (i = 0; i < num_recvs; i++)
            {
               recv_buffer_size = hypre_AMGDDCommPkgRecvBufferSize(compGridCommPkg)[level][i];
               recv_buffers[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size, HYPRE_MEMORY_HOST);
               hypre_MPI_Irecv(recv_buffers[i], recv_buffer_size, HYPRE_MPI_COMPLEX,
                               hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 3, comm, &requests[request_counter++]);
            }

            for (i = 0; i < num_sends; i++)
            {
               send_buffer_size = hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg)[level][i];
               send_buffers[i] = hypre_BoomerAMGDD_PackResidualBuffer(compGrid, compGridCommPkg, level, i);
               hypre_MPI_Isend(send_buffers[i], send_buffer_size, HYPRE_MPI_COMPLEX,
                               hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 3, comm, &requests[request_counter++]);
            }

            // wait for buffers to be received
            hypre_MPI_Waitall( request_counter, requests, status );

            hypre_TFree(requests, HYPRE_MEMORY_HOST);
            hypre_TFree(status, HYPRE_MEMORY_HOST);
            for (i = 0; i < num_sends; i++)
            {
               hypre_TFree(send_buffers[i], HYPRE_MEMORY_HOST);
            }
            hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);

            // Unpack recv buffers
            for (i = 0; i < num_recvs; i++)
            {
               hypre_BoomerAMGDD_UnpackResidualBuffer(recv_buffers[i], compGrid, compGridCommPkg, level, i);
            }

            // clean up memory for this level
            for (i = 0; i < num_recvs; i++)
            {
               hypre_TFree(recv_buffers[i], HYPRE_MEMORY_HOST);
            }
            hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
         }
      }
   }

   return hypre_error_flag;
}

HYPRE_Complex*
hypre_BoomerAMGDD_PackResidualBuffer( hypre_AMGDDCompGrid **compGrid,
                                      hypre_AMGDDCommPkg   *compGridCommPkg,
                                      HYPRE_Int             current_level,
                                      HYPRE_Int             proc )
{
   HYPRE_Complex  *buffer;
   HYPRE_Int       level, i;
   HYPRE_Int       send_elmt;
   HYPRE_Int       cnt = 0;

   buffer = hypre_CTAlloc(HYPRE_Complex,
                          hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg)[current_level][proc], HYPRE_MEMORY_HOST);
   for (level = current_level; level < hypre_AMGDDCommPkgNumLevels(compGridCommPkg); level++)
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[current_level][proc][level]; i++)
      {
         send_elmt = hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[current_level][proc][level][i];
         if (send_elmt < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
         {
            buffer[cnt++] = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridF(
                                                                               compGrid[level])))[send_elmt];
         }
         else
         {
            send_elmt -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
            buffer[cnt++] = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridF(
                                                                                  compGrid[level])))[send_elmt];
         }
      }
   }

   return buffer;
}

HYPRE_Int
hypre_BoomerAMGDD_UnpackResidualBuffer( HYPRE_Complex        *buffer,
                                        hypre_AMGDDCompGrid **compGrid,
                                        hypre_AMGDDCommPkg   *compGridCommPkg,
                                        HYPRE_Int             current_level,
                                        HYPRE_Int             proc )
{
   HYPRE_Int  recv_elmt;
   HYPRE_Int  level, i;
   HYPRE_Int  cnt = 0;

   for (level = current_level; level < hypre_AMGDDCommPkgNumLevels(compGridCommPkg); level++)
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[current_level][proc][level]; i++)
      {
         recv_elmt = hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[current_level][proc][level][i];
         hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridF(
                                                               compGrid[level])))[recv_elmt] = buffer[cnt++];
      }
   }

   return hypre_error_flag;
}
