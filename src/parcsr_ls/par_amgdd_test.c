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

#define MEASURE_TEST_COMP_RES 0
#define DUMP_INTERMEDIATE_TEST_SOLNS 0
#define DEBUGGING_MESSAGES 0

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"
#ifdef HYPRE_USING_CALIPER
#include <caliper/cali.h>
#endif


HYPRE_Int
SetRelaxMarker(hypre_AMGDDCompGrid *compGrid, hypre_ParVector *relax_marker, HYPRE_Int proc);

HYPRE_Real
GetTestCompositeResidual(hypre_ParCSRMatrix *A, hypre_ParVector *U_comp, hypre_ParVector *res, hypre_Vector *relax_marker, HYPRE_Int proc);

HYPRE_Int
TestBoomerAMGSolve( void               *amg_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *f,
                   hypre_ParVector    *u,
                   hypre_ParVector    **relax_marker,
                   HYPRE_Int          proc,
                   hypre_ParVector    **Q_array         );

HYPRE_Int
TestBoomerAMGCycle( void              *amg_vdata,
                   hypre_ParVector  **F_array,
                   hypre_ParVector  **U_array,
                   hypre_ParVector  **relax_marker,
                   HYPRE_Int        proc,
                   hypre_ParVector  **Q_array   );



HYPRE_Int
hypre_BoomerAMGDDTestSolve( void               *amg_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *f,
                   hypre_ParVector    *u                )
{
  // We will simulate an AMG-DD cycle by performing standard, parallel AMG V-cycles with suppressed relaxation to generate the solutions for each subproblem

  // Get MPI info
  HYPRE_Int myid, num_procs;
  hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
  hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

  HYPRE_Int proc, level;

  #if DEBUGGING_MESSAGES
  hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
  if (myid == 0) hypre_printf("Began AMG-DD test solve on all ranks\n");
  hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
  #endif

  // Get AMG info
  hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;
  hypre_ParVector  *U_comp = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, hypre_ParVectorGlobalSize(u), hypre_ParVectorPartitioning(u));
  hypre_ParVectorSetPartitioningOwner(U_comp, 0);
  hypre_ParVectorInitialize(U_comp);
  HYPRE_Int num_comp_cycles = hypre_ParAMGDataAMGDDFACNumCycles(amg_data);
  HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

  // Generate the residual
  hypre_ParVector *res = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, hypre_ParVectorGlobalSize(f), hypre_ParVectorPartitioning(f));
  hypre_ParVectorSetPartitioningOwner(res, 0);
  hypre_ParVectorInitialize(res);
  hypre_ParVectorCopy(f, res);
  hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, res );
  hypre_ParVector *res_copy = NULL;
  if (hypre_ParAMGDataAMGDDUseRD(amg_data))
  {
    res_copy = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, hypre_ParVectorGlobalSize(f), hypre_ParVectorPartitioning(f));
    hypre_ParVectorSetPartitioningOwner(res_copy, 0);
    hypre_ParVectorInitialize(res_copy);
    hypre_ParVectorCopy(res, res_copy);
  }



  // !!! TRY: testing whether my representation of Q is correct for RD
  hypre_ParVector *U_copy = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, hypre_ParVectorGlobalSize(u), hypre_ParVectorPartitioning(u));
  hypre_ParVectorSetPartitioningOwner(U_copy, 0);
  hypre_ParVectorInitialize(U_copy);
  hypre_ParVectorCopy(u, U_copy);
  hypre_ParVector **Q_array = NULL;
  if (hypre_ParAMGDataAMGDDUseRD(amg_data))
  {
    Q_array = hypre_CTAlloc(hypre_ParVector*, num_levels, HYPRE_MEMORY_HOST);
    for (level = 0; level < num_levels; level++)
    {
      Q_array[level] = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, hypre_ParVectorGlobalSize(hypre_ParAMGDataUArray(amg_data)[level]), hypre_ParVectorPartitioning(hypre_ParAMGDataUArray(amg_data)[level]));
      hypre_ParVectorSetPartitioningOwner(Q_array[level], 0);
      hypre_ParVectorInitialize(Q_array[level]);
    }
  }




  // Loop over processors
  for (proc = 0; proc < num_procs; proc++)
  {
    #if DEBUGGING_MESSAGES
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    if (myid == 0) hypre_printf("About to set relax marker on all ranks for proc %d\n", proc);
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    #endif


    // Setup vectors on each level that dictate where relaxation should be suppressed
    hypre_ParVector  **relax_marker = hypre_CTAlloc(hypre_ParVector*, num_levels, HYPRE_MEMORY_HOST);
    for (level = 0; level < num_levels; level++)
    {
      // Create and initialize the relax_marker vector on this level
      relax_marker[level] = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, hypre_ParVectorGlobalSize(hypre_ParAMGDataUArray(amg_data)[level]), hypre_ParVectorPartitioning(hypre_ParAMGDataUArray(amg_data)[level]));
      hypre_ParVectorSetPartitioningOwner(relax_marker[level],0);
      hypre_ParVectorInitialize(relax_marker[level]);
      // Now set the values according to the relevant comp grid
      if (level < num_levels) SetRelaxMarker(hypre_ParAMGDataAMGDDCompGrid(amg_data)[level], relax_marker[level], proc);
      else hypre_ParVectorSetConstantValues(relax_marker[level], 1.0);
    }

    #if DEBUGGING_MESSAGES
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    if (myid == 0) hypre_printf("Done setting relax marker on all ranks for proc %d\n", proc);
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    #endif

    // Set the initial guess for the AMG solve to 0
    hypre_ParVectorSetConstantValues(U_comp,0);

    // If doing RD, setup RHS
    if (hypre_ParAMGDataAMGDDUseRD(amg_data))
    {
      hypre_ParVectorSetConstantValues(res, 0.0);
      if (myid == proc) hypre_SeqVectorCopy(hypre_ParVectorLocalVector(res_copy), hypre_ParVectorLocalVector(res));
    }

    // Perform AMG solve with suppressed relaxation
    #if MEASURE_TEST_COMP_RES
    HYPRE_Real *res_norm = hypre_CTAlloc(HYPRE_Real, num_comp_cycles+1, HYPRE_MEMORY_HOST);
    res_norm[0] = GetTestCompositeResidual(A, U_comp, res, hypre_ParVectorLocalVector(relax_marker[0]), proc);
    #endif
    HYPRE_Int i;
    for (i = 0; i < num_comp_cycles; i++)
    {
      TestBoomerAMGSolve(amg_data, A, res, U_comp, relax_marker, proc, Q_array);


      #if MEASURE_TEST_COMP_RES
      res_norm[i+1] = GetTestCompositeResidual(A, U_comp, res, hypre_ParVectorLocalVector(relax_marker[0]), proc);
      #endif
    }

    #if DEBUGGING_MESSAGES
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    if (myid == 0) hypre_printf("Done with TestBoomerAMGSolve on all ranks for proc %d\n", proc);
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    #endif

    #if MEASURE_TEST_COMP_RES
    if (myid == 0)
    {
      FILE *file;
      char filename[256];
      sprintf(filename, "outputs/test_comp_res_proc%d.txt", proc);
      file = fopen(filename, "w");
      for (i = 0; i < num_comp_cycles+1; i++) fprintf(file, "%e ", res_norm[i]);
      fprintf(file, "\n");
    }
    #endif

    // Update global solution
    if (hypre_ParAMGDataAMGDDUseRD(amg_data))
    {
      hypre_ParVectorAxpy(1.0, U_comp, u);
    }
    else
    {
      // Update the values in the global solution for this proc
      if (myid == proc)
      {
        // add local part of U_comp to local part of u 
        hypre_SeqVectorAxpy( 1.0, hypre_ParVectorLocalVector(U_comp), hypre_ParVectorLocalVector(u));
      }
    }
    
    // Clean up memory
    for (level = 0; level < num_levels; level++) hypre_ParVectorDestroy(relax_marker[level]);
    hypre_TFree(relax_marker, HYPRE_MEMORY_HOST);
  }


  // !!! TRY: see if Q interpolated from all levels up to the finest is same as the total update to U
  if (Q_array)
  {
    for (level = num_levels-2; level >= 0; level--)
    {
      hypre_ParCSRMatrixMatvec(1.0, hypre_ParAMGDataPArray(amg_data)[level], Q_array[level+1], 1.0, Q_array[level]);
    }
    hypre_ParVectorAxpy(-1.0, u, U_copy);
    hypre_ParVectorScale(-1.0, U_copy);

    #if DUMP_INTERMEDIATE_TEST_SOLNS
    char filename[256];
    sprintf(filename, "outputs/test/q");
    hypre_ParVectorPrint(Q_array[0], filename);
    sprintf(filename, "outputs/test/update");
    hypre_ParVectorPrint(U_copy, filename);
    #endif
  }







  // Reset fine grid solution and right-hand side vectors for amg_data structure
  hypre_ParAMGDataUArray(amg_data)[0] = u;
  hypre_ParAMGDataFArray(amg_data)[0] = f;

  // Clean up memory
  hypre_ParVectorDestroy(U_comp);
  hypre_ParVectorDestroy(res);

  // !!! TRY: clean up Q_array
  if (Q_array)
  {
    for (level = 0; level < num_levels; level++)
    {
      hypre_ParVectorDestroy(Q_array[level]);
    }
    hypre_TFree(Q_array, HYPRE_MEMORY_HOST);
  }
  hypre_ParVectorDestroy(U_copy);



  return 0;
}

HYPRE_Int
SetRelaxMarker(hypre_AMGDDCompGrid *compGrid, hypre_ParVector *relax_marker, HYPRE_Int proc)
{
  HYPRE_Int i;
  HYPRE_Int myid;
  hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

  // Check whether the global indices are still around
  if (!hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid) && hypre_AMGDDCompGridNumNonOwnedNodes(compGrid))
  {
    if (myid == 0) printf("Error: need to setup AMG-DD with debugging flag set.\n");
    hypre_MPI_Finalize();
    exit(0);
  }

  // Broadcast the number of nodes in the composite grid on this level for the root proc
  HYPRE_Int num_nonowned_real = hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid);
  hypre_MPI_Bcast(&num_nonowned_real, 1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

  // Broadcast the global indices of the dofs in the composite grid
  HYPRE_Int *global_indices;
  if (myid == proc) global_indices = hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid);
  else global_indices = hypre_CTAlloc(HYPRE_Int, num_nonowned_real, HYPRE_MEMORY_HOST);
  hypre_MPI_Bcast(global_indices, num_nonowned_real, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

  // Loop over the global indices and mark where to do relaxation
  HYPRE_Int proc_first_index = hypre_ParVectorFirstIndex(relax_marker);
  HYPRE_Int proc_last_index = hypre_ParVectorLastIndex(relax_marker);
  for (i = 0; i < num_nonowned_real; i++)
  {
    if (global_indices[i] >= proc_first_index && global_indices[i] <= proc_last_index)
    {
      hypre_VectorData(hypre_ParVectorLocalVector(relax_marker))[global_indices[i] - proc_first_index] = 1;
    }
  }

  // Set relax marker on active proc
  if (myid == proc)
  {
    for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
      hypre_VectorData(hypre_ParVectorLocalVector(relax_marker))[i] = 1;
  }

  if (myid != proc) hypre_TFree(global_indices, HYPRE_MEMORY_HOST);

  return 0;
}

HYPRE_Real
GetTestCompositeResidual(hypre_ParCSRMatrix *A, hypre_ParVector *U_comp, hypre_ParVector *res, hypre_Vector *relax_marker, HYPRE_Int proc)
{

  HYPRE_Int myid;
  hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

  hypre_ParVector *intermediate_res = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, hypre_ParVectorGlobalSize(res), hypre_ParVectorPartitioning(res));
  hypre_ParVectorInitialize(intermediate_res);

  // Do the residual calculation in parallel
  hypre_ParVectorCopy(res, intermediate_res);
  hypre_ParCSRMatrixMatvec(-1.0, A, U_comp, 1.0, intermediate_res );
  hypre_Vector *local_res = hypre_ParVectorLocalVector(intermediate_res);

  // Locally find the residual norm counting only real nodes, then reduce over processors to get overall res norm
  HYPRE_Real local_res_norm = 0;
  HYPRE_Int i;
  for (i = 0; i < hypre_VectorSize(local_res); i++)
  {
    if (hypre_VectorData(relax_marker)[i]) local_res_norm += hypre_VectorData(local_res)[i]*hypre_VectorData(local_res)[i];
  }
  HYPRE_Real res_norm = 0;
  MPI_Reduce(&local_res_norm, &res_norm, 1, HYPRE_MPI_REAL, MPI_SUM, 0, hypre_MPI_COMM_WORLD);

  hypre_ParVectorSetPartitioningOwner(intermediate_res, 0);
  hypre_ParVectorDestroy(intermediate_res);

  return sqrt(res_norm);
}

























/*--------------------------------------------------------------------
 * TestBoomerAMGSolve
 *--------------------------------------------------------------------*/

HYPRE_Int
TestBoomerAMGSolve( void               *amg_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *f,
                   hypre_ParVector    *u,
                   hypre_ParVector    **relax_marker,
                   HYPRE_Int          proc,
                   hypre_ParVector    **Q_array         )
{

   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);   






    #if DEBUGGING_MESSAGES
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    if (myid == 0) hypre_printf("Inside TestBoomerAMGSolve on all ranks, comm = %d\n", comm);
    if (myid == 0) hypre_printf("comm world = %d\n", hypre_MPI_COMM_WORLD);
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    #endif






   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */

   HYPRE_Int      amg_print_level;
   HYPRE_Int      amg_logging;
   HYPRE_Int      cycle_count;
   HYPRE_Int      num_levels;
   /* HYPRE_Int      num_unknowns; */
   HYPRE_Int    converge_type;
   HYPRE_Real   tol;

   HYPRE_Int block_mode;
   

   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   hypre_ParCSRBlockMatrix **A_block_array;


   /*  Local variables  */

   HYPRE_Int      j;
   HYPRE_Int      Solve_err_flag;
   HYPRE_Int      min_iter;
   HYPRE_Int      max_iter;
   HYPRE_Int      num_procs, my_id;
   HYPRE_Int      additive;
   HYPRE_Int      mult_additive;
   HYPRE_Int      simple;

   HYPRE_Real   alpha = 1.0;
   HYPRE_Real   beta = -1.0;
   HYPRE_Real   cycle_op_count;
   HYPRE_Real   total_coeffs;
   HYPRE_Real   total_variables;
   HYPRE_Real  *num_coeffs;
   HYPRE_Real  *num_variables;
   HYPRE_Real   cycle_cmplxty = 0.0;
   HYPRE_Real   operat_cmplxty;
   HYPRE_Real   grid_cmplxty;
   HYPRE_Real   conv_factor = 0.0;
   HYPRE_Real   resid_nrm = 1.0;
   HYPRE_Real   resid_nrm_init = 0.0;
   HYPRE_Real   relative_resid;
   HYPRE_Real   rhs_norm = 0.0;
   HYPRE_Real   old_resid;
   HYPRE_Real   ieee_check = 0.;

   hypre_ParVector  *Vtemp;
   hypre_ParVector  *Residual;

   HYPRE_ANNOTATION_BEGIN("BoomerAMG.solve");
   hypre_MPI_Comm_size(comm, &num_procs);   
   hypre_MPI_Comm_rank(comm,&my_id);






    #if DEBUGGING_MESSAGES
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    if (myid == 0) hypre_printf("Past the first comm size call\n");
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    #endif










   amg_print_level    = hypre_ParAMGDataPrintLevel(amg_data);
   amg_logging      = hypre_ParAMGDataLogging(amg_data);
   if ( amg_logging > 1 )
      Residual = hypre_ParAMGDataResidual(amg_data);
   /* num_unknowns  = hypre_ParAMGDataNumUnknowns(amg_data); */
   num_levels       = hypre_ParAMGDataNumLevels(amg_data);
   A_array          = hypre_ParAMGDataAArray(amg_data);
   F_array          = hypre_ParAMGDataFArray(amg_data);
   U_array          = hypre_ParAMGDataUArray(amg_data);

   converge_type    = hypre_ParAMGDataConvergeType(amg_data);
   tol              = hypre_ParAMGDataTol(amg_data);
   min_iter         = hypre_ParAMGDataMinIter(amg_data);
   max_iter         = hypre_ParAMGDataMaxIter(amg_data);
   additive         = hypre_ParAMGDataAdditive(amg_data);
   simple           = hypre_ParAMGDataSimple(amg_data);
   mult_additive    = hypre_ParAMGDataMultAdditive(amg_data);

   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;

   block_mode = hypre_ParAMGDataBlockMode(amg_data);

   A_block_array          = hypre_ParAMGDataABlockArray(amg_data);


/*   Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
   hypre_ParVectorInitialize(Vtemp);
   hypre_ParVectorSetPartitioningOwner(Vtemp,0);
   hypre_ParAMGDataVtemp(amg_data) = Vtemp;
*/
   Vtemp = hypre_ParAMGDataVtemp(amg_data);

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/


   if (my_id == 0 && amg_print_level > 1)
      hypre_BoomerAMGWriteSolverParams(amg_data); 

   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag and assorted bookkeeping variables
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;

   total_coeffs = 0;
   total_variables = 0;
   cycle_count = 0;
   operat_cmplxty = 0;
   grid_cmplxty = 0;

   /*-----------------------------------------------------------------------
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && amg_print_level > 1 && tol > 0.)
     hypre_printf("\n\nAMG SOLUTION INFO:\n");


   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print 
    *-----------------------------------------------------------------------*/

   if (amg_print_level > 1 || amg_logging > 1 || tol > 0.)
   {
     if ( amg_logging > 1 ) {
        hypre_ParVectorCopy(F_array[0], Residual );
        if (tol > 0)
      hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Residual );
        resid_nrm = sqrt(hypre_ParVectorInnerProd( Residual, Residual ));
     }
     else {
        hypre_ParVectorCopy(F_array[0], Vtemp);
        if (tol > 0)
           hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
        resid_nrm = sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
     }

     /* Since it is does not diminish performance, attempt to return an error flag
        and notify users when they supply bad input. */
     if (resid_nrm != 0.) ieee_check = resid_nrm/resid_nrm; /* INF -> NaN conversion */
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
          hypre_printf("ERROR -- hypre_BoomerAMGSolve: INFs and/or NaNs detected in input.\n");
          hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
          hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
        }
        hypre_error(HYPRE_ERROR_GENERIC);
        HYPRE_ANNOTATION_END("BoomerAMG.solve");
        return hypre_error_flag;
     }

     /* r0 */
     resid_nrm_init = resid_nrm;

     if (0 == converge_type)
     {
        rhs_norm = sqrt(hypre_ParVectorInnerProd(f, f));
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

   if (my_id == 0 && amg_print_level > 1)
   {     
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n", resid_nrm_init,
                   relative_resid);
   }

   /*-----------------------------------------------------------------------
    *    Main V-cycle loop
    *-----------------------------------------------------------------------*/
   
   while ( (relative_resid >= tol || cycle_count < min_iter) && cycle_count < max_iter )
   {
      hypre_ParAMGDataCycleOpCount(amg_data) = 0;
      /* Op count only needed for one cycle */
      if ((additive < 0 || additive >= num_levels) 
      && (mult_additive < 0 || mult_additive >= num_levels)
      && (simple < 0 || simple >= num_levels) )
      {
        // printf("Rank %d about to call TestBoomerAMGCycle(), cycle_count = %d, max_iter = %d\n", my_id, cycle_count, max_iter);
         TestBoomerAMGCycle(amg_data, F_array, U_array, relax_marker, proc, Q_array); 
      }
      else
         hypre_BoomerAMGAdditiveCycle(amg_data); 
      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      if (amg_print_level > 1 || amg_logging > 1 || tol > 0.)
      {
        old_resid = resid_nrm;

        if ( amg_logging > 1 ) {
           hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[0], U_array[0], beta, F_array[0], Residual );
           resid_nrm = sqrt(hypre_ParVectorInnerProd( Residual, Residual ));
        }
        else {
           hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[0], U_array[0], beta, F_array[0], Vtemp);
           resid_nrm = sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
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

      ++cycle_count;

      hypre_ParAMGDataNumIterations(amg_data) = cycle_count;
#ifdef CUMNUMIT
      ++hypre_ParAMGDataCumNumIterations(amg_data);
#endif

      if (my_id == 0 && amg_print_level > 1)
      { 
         hypre_printf("    Cycle %2d   %e    %f     %e \n", cycle_count,
                      resid_nrm, conv_factor, relative_resid);
      }
   }

   if (cycle_count == max_iter && tol > 0.)
   {
      Solve_err_flag = 1;
      hypre_error(HYPRE_ERROR_CONV);
   }

   /*-----------------------------------------------------------------------
    *    Compute closing statistics
    *-----------------------------------------------------------------------*/

   if (cycle_count > 0 && resid_nrm_init) 
     conv_factor = pow((resid_nrm/resid_nrm_init),(1.0/(HYPRE_Real) cycle_count));
   else
     conv_factor = 1.;

   if (amg_print_level > 1) 
   {
      num_coeffs       = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
      num_variables    = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
      num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A);
      num_variables[0] = hypre_ParCSRMatrixGlobalNumRows(A);

      if (block_mode)
      {
         for (j = 1; j < num_levels; j++)
         {
            num_coeffs[j]    = (HYPRE_Real) hypre_ParCSRBlockMatrixNumNonzeros(A_block_array[j]);
            num_variables[j] = (HYPRE_Real) hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[j]);
         }
         num_coeffs[0]    = hypre_ParCSRBlockMatrixDNumNonzeros(A_block_array[0]);
         num_variables[0] = hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[0]);

      }
      else
      {
         for (j = 1; j < num_levels; j++)
         {
            num_coeffs[j]    = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A_array[j]);
            num_variables[j] = (HYPRE_Real) hypre_ParCSRMatrixGlobalNumRows(A_array[j]);
         }
      }
   

      for (j=0;j<hypre_ParAMGDataNumLevels(amg_data);j++)
      {
         total_coeffs += num_coeffs[j];
         total_variables += num_variables[j];
      }

      cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

      if (num_variables[0])
         grid_cmplxty = total_variables / num_variables[0];
      if (num_coeffs[0])
      {
         operat_cmplxty = total_coeffs / num_coeffs[0];
         cycle_cmplxty = cycle_op_count / num_coeffs[0];
      }

      if (my_id == 0)
      {
         if (Solve_err_flag == 1)
         {
            hypre_printf("\n\n==============================================");
            hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
            hypre_printf("      within the allowed %d V-cycles\n",max_iter);
            hypre_printf("==============================================");
         }
         hypre_printf("\n\n Average Convergence Factor = %f",conv_factor);
         hypre_printf("\n\n     Complexity:    grid = %f\n",grid_cmplxty);
         hypre_printf("                operator = %f\n",operat_cmplxty);
         hypre_printf("                   cycle = %f\n\n\n\n",cycle_cmplxty);
      }

      hypre_TFree(num_coeffs, HYPRE_MEMORY_HOST);
      hypre_TFree(num_variables, HYPRE_MEMORY_HOST);
   }
   HYPRE_ANNOTATION_END("BoomerAMG.solve");
   




    #if DEBUGGING_MESSAGES
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    if (myid == 0) hypre_printf("Finished with TestBoomerAMGSolve on all ranks\n");
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    #endif







   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * TestBoomerAMGCycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
TestBoomerAMGCycle( void              *amg_vdata,
                   hypre_ParVector  **F_array,
                   hypre_ParVector  **U_array,
                   hypre_ParVector  **relax_marker,
                   HYPRE_Int        proc,
                   hypre_ParVector  ** Q_array   )
{

   // !!! TODO

  return(0);
}


