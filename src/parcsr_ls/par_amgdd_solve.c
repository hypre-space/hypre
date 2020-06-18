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

#define TEST_RES_COMM 0
#define DEBUGGING_MESSAGES 0

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"	

HYPRE_Int
AddSolution( void *amg_vdata );

HYPRE_Int
ZeroInitialGuess( void *amg_vdata );

HYPRE_Int
ZeroCompositeRHS( void *amg_vdata );

HYPRE_Complex*
PackResidualBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc );

HYPRE_Complex*
PackSolutionBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc );

HYPRE_Int
UnpackResidualBuffer( HYPRE_Complex *buffer, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc );

HYPRE_Int
UnpackSolutionBuffer( HYPRE_Complex *buffer, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc );

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
   HYPRE_Int i;
   HYPRE_Real resid_nrm, resid_nrm_init, rhs_norm, relative_resid;

   // Get info from amg_data
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Real tol = hypre_ParAMGDataTol(amg_data);
   HYPRE_Int min_iter = hypre_ParAMGDataMinIter(amg_data);
   HYPRE_Int max_iter = hypre_ParAMGDataMaxIter(amg_data);
   HYPRE_Int converge_type = hypre_ParAMGDataConvergeType(amg_data);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);
   
   // Setup extra temporary variable to hold the solution if necessary
   if (!hypre_ParAMGDataZtemp(amg_data))
   {
      hypre_ParAMGDataZtemp(amg_data) = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(hypre_ParAMGDataAArray(amg_data)[amgdd_start_level]),
                                    hypre_ParCSRMatrixGlobalNumRows(hypre_ParAMGDataAArray(amg_data)[amgdd_start_level]),
                                    hypre_ParCSRMatrixRowStarts(hypre_ParAMGDataAArray(amg_data)[amgdd_start_level]));
      hypre_ParVectorInitialize(hypre_ParAMGDataZtemp(amg_data));
      hypre_ParVectorSetPartitioningOwner(hypre_ParAMGDataZtemp(amg_data),0);
   }

   // !!! New: trying a solve around just the processor boundary. This is temporary implementation (something much more efficient could be done). Just looking at convergence.
   HYPRE_Int *boundary_marker;
   if (hypre_ParAMGDataAMGDDNumGlobalRelax(amg_data) < 0) 
   {
      boundary_marker = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRMatrixNumRows(hypre_ParAMGDataAArray(amg_data)[amgdd_start_level]), HYPRE_MEMORY_HOST);
      HYPRE_Int i;
      hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(hypre_ParAMGDataAArray(amg_data)[amgdd_start_level]);
      for (i = 0; i < hypre_ParCSRCommPkgSendMapStart(comm_pkg, hypre_ParCSRCommPkgNumSends(comm_pkg)); i++) boundary_marker[ hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i) ] = 1;
   }


   // Set the fine grid operator, left-hand side, and right-hand side
   hypre_ParAMGDataAArray(amg_data)[0] = A;
   hypre_ParAMGDataUArray(amg_data)[0] = u;
   hypre_ParAMGDataFArray(amg_data)[0] = f;

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
         hypre_BoomerAMGPartialCycle(amg_vdata, hypre_ParAMGDataFArray(amg_data), hypre_ParAMGDataUArray(amg_data), amgdd_start_level-1, 0);

      // Do the AMGDD cycle
      if (hypre_ParAMGDataAMGDDUseRD(amg_data) > 1) // Do DD cycles followed by RD cycles
      {
         hypre_ParVectorCopy(hypre_ParAMGDataFArray(amg_data)[amgdd_start_level], hypre_ParAMGDataVtemp(amg_data));
         if (amgdd_start_level == 0) hypre_ParCSRMatrixMatvec(-1.0, hypre_ParAMGDataAArray(amg_data)[amgdd_start_level], hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], 1.0, hypre_ParAMGDataFArray(amg_data)[amgdd_start_level]);

         HYPRE_Int num_cycles = hypre_ParAMGDataAMGDDUseRD(amg_data) - 1; // number of each cycle type is 
         hypre_ParAMGDataAMGDDUseRD(amg_data) = 0;
         for (i = 0; i < num_cycles; i++)
         {
            error_code = hypre_BoomerAMGDD_Cycle(amg_vdata);
            if (error_code) test_failed = 1;
            hypre_ParVectorCopy(hypre_ParAMGDataVtemp(amg_data), hypre_ParAMGDataFArray(amg_data)[amgdd_start_level]);
            hypre_ParCSRMatrixMatvec(-1.0, hypre_ParAMGDataAArray(amg_data)[amgdd_start_level], hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], 1.0, hypre_ParAMGDataFArray(amg_data)[amgdd_start_level]);
         }
         hypre_ParAMGDataAMGDDUseRD(amg_data) = 1;
         for (i = 0; i < num_cycles; i++)
         {
            error_code = hypre_BoomerAMGDD_Cycle(amg_vdata);
            if (error_code) test_failed = 1;
            if (i != num_cycles-1)
            {
               hypre_ParVectorCopy(hypre_ParAMGDataVtemp(amg_data), hypre_ParAMGDataFArray(amg_data)[amgdd_start_level]);
               hypre_ParCSRMatrixMatvec(-1.0, hypre_ParAMGDataAArray(amg_data)[amgdd_start_level], hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], 1.0, hypre_ParAMGDataFArray(amg_data)[amgdd_start_level]);
            }
         }
         hypre_ParAMGDataAMGDDUseRD(amg_data) = 2;

         if (amgdd_start_level == 0) hypre_ParVectorCopy(hypre_ParAMGDataVtemp(amg_data), hypre_ParAMGDataFArray(amg_data)[amgdd_start_level]);
      }
      else
      {
         // If on the finest level, need to convert to a residual/correction equation
         if (amgdd_start_level == 0)
         {
            // Store the original fine grid right-hand side in Vtemp and use f as the current fine grid residual
            hypre_ParVectorCopy(hypre_ParAMGDataFArray(amg_data)[amgdd_start_level], hypre_ParAMGDataVtemp(amg_data));
            hypre_ParCSRMatrixMatvec(-1.0, hypre_ParAMGDataAArray(amg_data)[amgdd_start_level], hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], 1.0, hypre_ParAMGDataFArray(amg_data)[amgdd_start_level]);
         }

         error_code = hypre_BoomerAMGDD_Cycle(amg_vdata);
         if (error_code) test_failed = 1;

         if (amgdd_start_level == 0)
         {
            // Copy RHS back into f
            hypre_ParVectorCopy(hypre_ParAMGDataVtemp(amg_data), hypre_ParAMGDataFArray(amg_data)[amgdd_start_level]);
         }
      }

      /* HYPRE_Int relax_type, i; */
      /* for (i = 0; i < hypre_ParAMGDataAMGDDNumGlobalRelax(amg_data); i++) */
      /*    for (relax_type = 13; relax_type < 15; relax_type++) */
      /*       hypre_BoomerAMGRelax( hypre_ParAMGDataAArray(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataFArray(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataCFMarkerArray(amg_data)[amgdd_start_level], */
      /*                            relax_type, */
      /*                            0, */
      /*                            hypre_ParAMGDataRelaxWeight(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataOmega(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataL1Norms(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataVtemp(amg_data), */
      /*                            hypre_ParAMGDataZtemp(amg_data) ); */
      /* if (hypre_ParAMGDataAMGDDNumGlobalRelax(amg_data) < 0) */
      /*    for (i = 0; i < -hypre_ParAMGDataAMGDDNumGlobalRelax(amg_data); i++) */
      /*       for (relax_type = 13; relax_type < 15; relax_type++) */
      /*          hypre_BoomerAMGRelax( hypre_ParAMGDataAArray(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataFArray(amg_data)[amgdd_start_level], */
      /*                            boundary_marker, */
      /*                            relax_type, */
      /*                            1, */
      /*                            hypre_ParAMGDataRelaxWeight(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataOmega(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataL1Norms(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], */
      /*                            hypre_ParAMGDataVtemp(amg_data), */
      /*                            hypre_ParAMGDataZtemp(amg_data) ); */

      // Do normal AMG V-cycle upsweep back up to the fine grid
      if (amgdd_start_level > 0) 
      {
         // Interpolate
         hypre_ParCSRMatrixMatvec(1.0, hypre_ParAMGDataPArray(amg_data)[amgdd_start_level-1], hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], 1.0, hypre_ParAMGDataUArray(amg_data)[amgdd_start_level-1]);
         // V-cycle back to finest grid
         hypre_BoomerAMGPartialCycle(amg_vdata, hypre_ParAMGDataFArray(amg_data), hypre_ParAMGDataUArray(amg_data), amgdd_start_level-1, 1);
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

   // !!! New: clean up
   if (hypre_ParAMGDataAMGDDNumGlobalRelax(amg_data) < 0) hypre_TFree(boundary_marker, HYPRE_MEMORY_HOST);

   return test_failed;
}

HYPRE_Int
hypre_BoomerAMGDD_Cycle( void *amg_vdata )
{
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

	HYPRE_Int i,level;
   HYPRE_Int cycle_count = 0;
	hypre_ParAMGData	*amg_data = (hypre_ParAMGData*) amg_vdata;
  	HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);
   HYPRE_Int min_fac_iter = hypre_ParAMGDataMinFACIter(amg_data);
   HYPRE_Int max_fac_iter = hypre_ParAMGDataMaxFACIter(amg_data);
   HYPRE_Real fac_tol = hypre_ParAMGDataFACTol(amg_data);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Began AMG-DD cycle on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   HYPRE_Int test_failed = 0;
   if (hypre_ParAMGDataAMGDDUseRD(amg_data))
   {
      // If doing range decomposition, simply zero out RHS outside owned dofs
      ZeroCompositeRHS( amg_vdata );
   }
   else
   {
      // Otherwise, do residual communication
      test_failed = hypre_BoomerAMGDDResidualCommunication( amg_vdata );
   }

	// Set zero initial guess for all comp grids on all levels
	ZeroInitialGuess( amg_vdata );

   // Setup convergence tolerance info
   HYPRE_Real resid_nrm = 1.;
   HYPRE_Real resid_nrm_init = resid_nrm;
   HYPRE_Real relative_resid = 1.;
   HYPRE_Real conv_fact = 0;
   
   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("About to do FAC cycles on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   for (level = amgdd_start_level; level < num_levels; level++)
   {
      hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridT( hypre_ParAMGDataCompGrid(amg_data)[level] ), 0.0 );
      hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridS( hypre_ParAMGDataCompGrid(amg_data)[level] ), 0.0 );
   }

	// Do the cycles
   HYPRE_Int first_iteration = 1;
   if (fac_tol == 0.0)
   {
      while ( cycle_count < max_fac_iter )
      {
         // Do FAC cycle
         hypre_BoomerAMGDD_FAC_Cycle( amg_vdata, first_iteration );
         first_iteration = 0;

         ++cycle_count;
         hypre_ParAMGDataNumFACIterations(amg_data) = cycle_count;
      }
   }
   else if (fac_tol > 0)
   {
      // !!! To do:
      if (myid == 0) printf("fac_tol != 0 not yet implemented\n");
   }
   else if (fac_tol < 0)
   {
      // !!! To do:
      if (myid == 0) printf("fac_tol != 0 not yet implemented\n");
   }
   


	// Update fine grid solution
   if (hypre_ParAMGDataAMGDDUseRD(amg_data))
   {
      // If using RD, need to communicate and add composite solutions
      if (num_procs > 1) test_failed = hypre_BoomerAMGRDSolutionCommunication( amg_vdata );      
   }
   else
   {
      // Otherwise simply do a local add over the owned dofs
      AddSolution( amg_vdata );
   }

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Finished AMG-DD cycle on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

	return test_failed;
}

HYPRE_Int
AddSolution( void *amg_vdata )
{
	hypre_ParAMGData	*amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);

   // Original solution was stored in Z temp, so just add this back in
   hypre_ParVectorAxpy( 1.0, hypre_ParAMGDataZtemp(amg_data), hypre_ParAMGDataUArray(amg_data)[amgdd_start_level]);
   
	return 0;
}

HYPRE_Int
ZeroInitialGuess( void *amg_vdata )
{
   HYPRE_Int level;
	hypre_ParAMGData	*amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);

   // Save the original solution (updated at the end of the AMG-DD cycle)
   hypre_ParVectorCopy(hypre_ParAMGDataUArray(amg_data)[amgdd_start_level], hypre_ParAMGDataZtemp(amg_data));

   // Zero solution on all levels
   for (level = amgdd_start_level; level < hypre_ParAMGDataNumLevels(amg_data); level++)
   {
      hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridU(hypre_ParAMGDataCompGrid(amg_data)[level]), 0.0);
      if (hypre_ParCompGridQ(hypre_ParAMGDataCompGrid(amg_data)[level])) hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridQ(hypre_ParAMGDataCompGrid(amg_data)[level]), 0.0);
   }
	
	return 0;
}

HYPRE_Int
ZeroCompositeRHS( void *amg_vdata )
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) amg_vdata;

   HYPRE_Int level;
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);
   hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[amgdd_start_level];
   
   // Set RHS = 0, then copy in residual (from AMG F) into the owned portion
   hypre_SeqVectorSetConstantValues(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridF(compGrid)), 0.0);

   // Restrict rhs to all levels to get appropriate initial values
   for (level = amgdd_start_level; level < hypre_ParAMGDataNumLevels(amg_data)-1; level++)
   {
      hypre_ParCompGrid *compGrid_f = hypre_ParAMGDataCompGrid(amg_data)[level];
      hypre_ParCompGrid *compGrid_c = hypre_ParAMGDataCompGrid(amg_data)[level+1];
      hypre_ParCompGridMatvec(1.0, hypre_ParCompGridR(compGrid_f), hypre_ParCompGridF(compGrid_f), 0.0, hypre_ParCompGridF(compGrid_c));
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
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   
   /* Data Structure variables */

   // level counters, indices, and parameters
   HYPRE_Int                  num_levels, amgdd_start_level;
   HYPRE_Real                 alpha, beta;
   HYPRE_Int                  level,i;

   // info from amg
   hypre_ParCSRMatrix         **A_array;
   hypre_ParVector            **F_array;
   hypre_ParCSRMatrix         **R_array;
   hypre_ParCompGrid          **compGrid;

   // info from comp grid comm pkg
   hypre_ParCompGridCommPkg   *compGridCommPkg;

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
   amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);
   compGrid = hypre_ParAMGDataCompGrid(amg_data);
   compGridCommPkg = hypre_ParAMGDataCompGridCommPkg(amg_data);

   // Restrict residual down to all levels 
   for (level = amgdd_start_level; level < num_levels-1; level++)
   {
      if ( hypre_ParAMGDataRestriction(amg_data) ) hypre_ParCSRMatrixMatvec(1.0, R_array[level], F_array[level], 0.0, F_array[level+1]);
      else hypre_ParCSRMatrixMatvecT(1.0, R_array[level], F_array[level], 0.0, F_array[level+1]);
   }

   if (num_procs > 1)
   {
      #if DEBUGGING_MESSAGES
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      if (myid == 0) hypre_printf("Entering loop over levels in residual communication on all ranks\n");
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      #endif

      /* Outer loop over levels:
      Start from coarsest level and work up to finest */
      for (level = num_levels - 1; level >= amgdd_start_level; level--)
      {
         // Get some communication info
         comm = hypre_ParCSRMatrixComm(A_array[level]);
         HYPRE_Int num_sends = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level];
         HYPRE_Int num_recvs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level];

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
               HYPRE_Int recv_buffer_size = hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg)[level][i];
               if (!recv_buffer_size) printf("Posted recv for empty buffer\n");
               recv_buffers[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size, HYPRE_MEMORY_HOST);
               hypre_MPI_Irecv(recv_buffers[i], recv_buffer_size, HYPRE_MPI_COMPLEX, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 3, comm, &requests[request_counter++]);
            }

            for (i = 0; i < num_sends; i++)
            {
               HYPRE_Int send_buffer_size = hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg)[level][i];
               if (!send_buffer_size) printf("Posted send for empty buffer\n");
               send_buffers[i] = PackResidualBuffer(compGrid, compGridCommPkg, level, i);
               hypre_MPI_Isend(send_buffers[i], send_buffer_size, HYPRE_MPI_COMPLEX, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 3, comm, &requests[request_counter++]);
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
               UnpackResidualBuffer(recv_buffers[i], compGrid, compGridCommPkg, level, i);
            }

            // clean up memory for this level
            for (i = 0; i < num_recvs; i++) hypre_TFree(recv_buffers[i], HYPRE_MEMORY_HOST);
            hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
         }

         #if DEBUGGING_MESSAGES
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (myid == 0) hypre_printf("   Finished residual communication on level %d on all ranks\n", level);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif

      }

      #if DEBUGGING_MESSAGES
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      if (myid == 0) hypre_printf("Finished residual communication on all ranks\n");
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      #endif

   }
   
   #if TEST_RES_COMM
   return TestResComm(amg_data);
   #else
   return 0;
   #endif
}

HYPRE_Int
hypre_BoomerAMGRDSolutionCommunication( void *amg_vdata )
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
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   
   /* Data Structure variables */

   // level counters, indices, and parameters
   HYPRE_Int                  num_levels, amgdd_start_level;
   HYPRE_Real                 alpha, beta;
   HYPRE_Int                  level,i;

   // info from amg
   hypre_ParCSRMatrix         **A_array;
   hypre_ParVector            **F_array;
   hypre_ParVector            **U_array;
   hypre_ParCSRMatrix         **P_array;
   hypre_ParCompGrid          **compGrid;

   // info from comp grid comm pkg
   hypre_ParCompGridCommPkg   *compGridCommPkg;
   HYPRE_Int                  num_send_procs, num_recv_procs;

   // temporary arrays used for communication during comp grid setup
   HYPRE_Complex              **send_buffers;
   HYPRE_Complex              **recv_buffers;

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
   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);
   compGrid = hypre_ParAMGDataCompGrid(amg_data);
   compGridCommPkg = hypre_ParAMGDataCompGridCommPkg(amg_data);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("About to do coarse levels allgather on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Entering loop over levels in residual communication on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   /* Outer loop over levels:
   Start from finest level and work down to coarsest */
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      // Get some communication info
      comm = hypre_ParCSRMatrixComm(A_array[level]);
      num_send_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level];
      num_recv_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level];

      if ( num_send_procs || num_recv_procs ) // If there are any owned nodes on this level
      {
         // Get some communication info
         comm = hypre_ParCSRMatrixComm(A_array[level]);
         HYPRE_Int num_sends = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level];
         HYPRE_Int num_recvs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level];

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
               HYPRE_Int recv_buffer_size = hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg)[level][i];
               if (!recv_buffer_size) printf("Posted recv for empty buffer\n");
               recv_buffers[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size, HYPRE_MEMORY_HOST);
               hypre_MPI_Irecv(recv_buffers[i], recv_buffer_size, HYPRE_MPI_COMPLEX, hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i], 3, comm, &requests[request_counter++]);
            }

            for (i = 0; i < num_sends; i++)
            {
               HYPRE_Int send_buffer_size = hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg)[level][i];
               if (!send_buffer_size) printf("Posted send for empty buffer\n");
               send_buffers[i] = PackSolutionBuffer(compGrid, compGridCommPkg, level, i);
               hypre_MPI_Isend(send_buffers[i], send_buffer_size, HYPRE_MPI_COMPLEX, hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][i], 3, comm, &requests[request_counter++]);
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
               UnpackSolutionBuffer(recv_buffers[i], compGrid, compGridCommPkg, level, i);
            }

            // clean up memory for this level
            for (i = 0; i < num_recvs; i++) hypre_TFree(recv_buffers[i], HYPRE_MEMORY_HOST);
            hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
         }

         #if DEBUGGING_MESSAGES
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         if (myid == 0) hypre_printf("   Finished solution communication on level %d on all ranks\n", level);
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         #endif
      }
   }

   // Copy owned parts of updates into ParVectors. Using F from AMG, since this is free.
   for (level  = amgdd_start_level; level < num_levels; level++)
   {
      hypre_SeqVectorCopy(hypre_ParCompGridVectorOwned(hypre_ParCompGridQ(compGrid[level])), hypre_ParVectorLocalVector(F_array[level]));
   }

   // Interpolate updates to fine grid
   for (level = num_levels-2; level >= amgdd_start_level; level--)
   {
      hypre_ParCSRMatrixMatvec(1.0, P_array[level], F_array[level+1], 1.0, F_array[level]);
   }

   // Add update to solution
   hypre_ParVectorCopy(hypre_ParAMGDataZtemp(amg_data), U_array[amgdd_start_level]);
   hypre_ParVectorAxpy( 1.0, F_array[amgdd_start_level], U_array[amgdd_start_level]);
   
   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Finished residual communication on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   #if TEST_RES_COMM
   return TestResComm(amg_data);
   #else
   return 0;
   #endif
}

HYPRE_Complex*
PackResidualBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc )
{
   HYPRE_Int level,i;

   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Complex *buffer = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg)[current_level][proc], HYPRE_MEMORY_HOST);

   HYPRE_Int cnt = 0;
   for (level = current_level; level < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); level++)
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[current_level][proc][level]; i++)
      {
         HYPRE_Int send_elmt = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[current_level][proc][level][i];
         if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            buffer[cnt++] = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridF(compGrid[level])))[send_elmt];
         else
         {
            send_elmt -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
            buffer[cnt++] = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridF(compGrid[level])))[send_elmt];
         }
      }
   }

   return buffer;
}

HYPRE_Complex*
PackSolutionBuffer( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc )
{
   HYPRE_Int level,i;

   HYPRE_Complex *buffer = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg)[current_level][proc], HYPRE_MEMORY_HOST);

   HYPRE_Int cnt = 0;
   for (level = current_level; level < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); level++)
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[current_level][proc][level]; i++)
      {
         HYPRE_Int send_elmt = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[current_level][proc][level][i];
         if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            buffer[cnt++] = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridU(compGrid[level])))[send_elmt];
         else
         {
            send_elmt -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
            buffer[cnt++] = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridU(compGrid[level])))[send_elmt];
         }
      }
   }

   return buffer;
}


HYPRE_Int
UnpackResidualBuffer( HYPRE_Complex *buffer, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc )
{
   HYPRE_Int level,i;

   HYPRE_Int cnt = 0;
   for (level = current_level; level < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); level++)
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[current_level][proc][level]; i++)
      {
         HYPRE_Int recv_elmt = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[current_level][proc][level][i];
         hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridF(compGrid[level])))[recv_elmt] = buffer[cnt++];
      }
   }

   return 0;
}

HYPRE_Int
UnpackSolutionBuffer( HYPRE_Complex *buffer, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc )
{
   HYPRE_Int level,i;

   HYPRE_Int cnt = 0;
   for (level = current_level; level < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); level++)
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[current_level][proc][level]; i++)
      {
         HYPRE_Int recv_elmt = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[current_level][proc][level][i];
         hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridU(compGrid[level])))[recv_elmt] += buffer[cnt++];
      }
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

   HYPRE_Int test_failed = 0;

   // For each processor and each level broadcast the residual data and global indices out and check agains the owning procs
   HYPRE_Int proc;
   HYPRE_Int i;
   for (proc = 0; proc < num_procs; proc++)
   {
      HYPRE_Int level;
      for (level = 0; level < num_levels; level++)
      {
         // Broadcast the number of nodes
         HYPRE_Int num_real_nodes = hypre_ParCompGridNumNonOwnedRealNodes(compGrid[level]);
         hypre_MPI_Bcast(&num_real_nodes, 1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the composite residual
         HYPRE_Complex *comp_res;
         if (myid == proc) comp_res = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridF(compGrid[level])));
         else comp_res = hypre_CTAlloc(HYPRE_Complex, num_real_nodes, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(comp_res, num_real_nodes, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the global indices
         HYPRE_Int *global_indices;
         if (myid == proc) global_indices = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]);
         else global_indices = hypre_CTAlloc(HYPRE_Int, num_real_nodes, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(global_indices, num_real_nodes, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Now, each processors checks their owned residual value against the composite residual
         HYPRE_Int proc_first_index = hypre_ParVectorFirstIndex(hypre_ParAMGDataUArray(amg_data)[level]);
         HYPRE_Int proc_last_index = hypre_ParVectorLastIndex(hypre_ParAMGDataUArray(amg_data)[level]);
         for (i = 0; i < num_real_nodes; i++)
         {
            if (global_indices[i] <= proc_last_index && global_indices[i] >= proc_first_index)
            {
               if (comp_res[i] != hypre_VectorData(hypre_ParVectorLocalVector(hypre_ParAMGDataFArray(amg_data)[level]))[global_indices[i] - proc_first_index] )
               {
                  printf("Error: on proc %d has incorrect residual at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
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
   }

   return test_failed;
}
