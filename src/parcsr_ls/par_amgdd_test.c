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
SetRelaxMarker(hypre_ParCompGrid *compGrid, hypre_ParVector *relax_marker, HYPRE_Int proc);

HYPRE_Real
GetTestCompositeResidual(hypre_ParCSRMatrix *A, hypre_ParVector *U_comp, hypre_ParVector *res, hypre_Vector *relax_marker, HYPRE_Int proc);

HYPRE_Int
TestBoomerAMGSolve( void               *amg_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *f,
                   hypre_ParVector    *u,
                   hypre_ParVector    **relax_marker,
                   HYPRE_Int proc         );

HYPRE_Int
TestBoomerAMGCycle( void              *amg_vdata,
                   hypre_ParVector  **F_array,
                   hypre_ParVector  **U_array,
                   hypre_ParVector  **relax_marker,
                   HYPRE_Int proc   );



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
  HYPRE_Int num_comp_cycles = hypre_ParAMGDataMaxFACIter(amg_data);
  HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
  HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
  if (transition_level < 0) transition_level = num_levels;

  // Generate the residual and store in f
  hypre_ParVector *res = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, hypre_ParVectorGlobalSize(f), hypre_ParVectorPartitioning(f));
  hypre_ParVectorSetPartitioningOwner(res, 0);
  hypre_ParVectorInitialize(res);
  hypre_ParVectorCopy(f, res);
  hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, res );


  // Loop over processors
  HYPRE_Int proc, level;
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
      if (level < transition_level) SetRelaxMarker(hypre_ParAMGDataCompGrid(amg_data)[level], relax_marker[level], proc);
      else hypre_ParVectorSetConstantValues(relax_marker[level], 1.0);
    }

    #if DEBUGGING_MESSAGES
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    if (myid == 0) hypre_printf("Done setting relax marker on all ranks for proc %d\n", proc);
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    #endif

    // Set the initial guess for the AMG solve to 0
    hypre_ParVectorSetConstantValues(U_comp,0);

    // Perform AMG solve with suppressed relaxation
    #if MEASURE_TEST_COMP_RES
    HYPRE_Real *res_norm = hypre_CTAlloc(HYPRE_Real, num_comp_cycles+1, HYPRE_MEMORY_HOST);
    res_norm[0] = GetTestCompositeResidual(A, U_comp, res, hypre_ParVectorLocalVector(relax_marker[0]), proc);
    #endif
    HYPRE_Int i;
    for (i = 0; i < num_comp_cycles; i++)
    {
      hypre_BoomerAMGSetMaxIter(amg_data, 1);
      TestBoomerAMGSolve(amg_data, A, res, U_comp, relax_marker, proc);


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

    // Update the values in the global solution for this proc
    if (myid == proc)
    {
      // add local part of U_comp to local part of u 
      hypre_SeqVectorAxpy( 1.0, hypre_ParVectorLocalVector(U_comp), hypre_ParVectorLocalVector(u));
    }
    
    // Clean up memory
    for (level = 0; level < num_levels; level++) hypre_ParVectorDestroy(relax_marker[level]);
    hypre_TFree(relax_marker, HYPRE_MEMORY_HOST);
  }

  // Reset fine grid solution and right-hand side vectors for amg_data structure
  hypre_ParAMGDataUArray(amg_data)[0] = u;
  hypre_ParAMGDataFArray(amg_data)[0] = f;

  // Clean up memory
  hypre_ParVectorDestroy(U_comp);
  hypre_ParVectorDestroy(res);

  return 0;
}

HYPRE_Int
SetRelaxMarker(hypre_ParCompGrid *compGrid, hypre_ParVector *relax_marker, HYPRE_Int proc)
{
  HYPRE_Int i;
  HYPRE_Int myid;
  hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

  // Check whether the global indices are still around
  if (!hypre_ParCompGridGlobalIndices(compGrid))
  {
    if (myid == 0) printf("Error: need to setup AMG-DD with debugging flag set.\n");
    hypre_MPI_Finalize();
    exit(0);
  }

  // Broadcast the number of nodes in the composite gird on this level for the root proc
  HYPRE_Int num_nodes = 0;
  HYPRE_Int is_real;
  if (myid == proc)
  {
    for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
    {
      // if (hypre_ParCompGridRealDofMarker(compGrid)) is_real = hypre_ParCompGridRealDofMarker(compGrid)[i];
      // else is_real = 1;
      // if (is_real) num_nodes++;
      if (hypre_ParCompGridARowPtr(compGrid)[i+1] - hypre_ParCompGridARowPtr(compGrid)[i] > 0) num_nodes++;
    }
  }
  hypre_MPI_Bcast(&num_nodes, 1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

  // Broadcast the global indices of the dofs in the composite grid
  HYPRE_Int *global_indices = hypre_CTAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
  if (myid == proc)
  {
    HYPRE_Int cnt = 0;
    for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
    {
      // if (hypre_ParCompGridRealDofMarker(compGrid)) is_real = hypre_ParCompGridRealDofMarker(compGrid)[i];
      // else is_real = 1;
      // if (is_real) global_indices[cnt++] = hypre_ParCompGridGlobalIndices(compGrid)[i];
      if (hypre_ParCompGridARowPtr(compGrid)[i+1] - hypre_ParCompGridARowPtr(compGrid)[i] > 0) global_indices[cnt++] = hypre_ParCompGridGlobalIndices(compGrid)[i];
    }
  }
  hypre_MPI_Bcast(global_indices, num_nodes, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

  // Loop over the global indices and mark where to do relaxation
  HYPRE_Int proc_first_index = hypre_ParVectorFirstIndex(relax_marker);
  HYPRE_Int proc_last_index = hypre_ParVectorLastIndex(relax_marker);
  for (i = 0; i < num_nodes; i++)
  {
    if (global_indices[i] >= proc_first_index && global_indices[i] <= proc_last_index)
    {
      hypre_VectorData(hypre_ParVectorLocalVector(relax_marker))[global_indices[i] - proc_first_index] = 1;
    }
  }

  hypre_TFree(global_indices, HYPRE_MEMORY_HOST);

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
                   HYPRE_Int proc         )
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
         TestBoomerAMGCycle(amg_data, F_array, U_array, relax_marker, proc); 
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
                   HYPRE_Int proc   )
{




    #if DEBUGGING_MESSAGES
  HYPRE_Int myid;
  hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    if (myid == 0) hypre_printf("Inside TestBoomerAMGCycle\n");
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    #endif





   char filename[256];

   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   HYPRE_Solver *smoother;
   /* Data Structure variables */

   hypre_ParCSRMatrix    **A_array;
   hypre_ParCSRMatrix    **P_array;
   hypre_ParCSRMatrix    **R_array;
   hypre_ParVector    *Utemp;
   hypre_ParVector    *Vtemp;
   hypre_ParVector    *Rtemp;
   hypre_ParVector    *Ptemp;
   hypre_ParVector    *Ztemp;
   hypre_ParVector    *Aux_U;
   hypre_ParVector    *Aux_F;

   hypre_ParCSRBlockMatrix    **A_block_array;
   hypre_ParCSRBlockMatrix    **P_block_array;
   hypre_ParCSRBlockMatrix    **R_block_array;

   HYPRE_Real   *Ztemp_data;
   HYPRE_Real   *Ptemp_data;
   HYPRE_Int     **CF_marker_array;
   /* HYPRE_Int     **unknown_map_array;
   HYPRE_Int     **point_map_array;
   HYPRE_Int     **v_at_point_array; */

   HYPRE_Real    cycle_op_count;
   HYPRE_Int       cycle_type;
   HYPRE_Int       num_levels;
   HYPRE_Int       max_levels;

   HYPRE_Real   *num_coeffs;
   HYPRE_Int      *num_grid_sweeps;
   HYPRE_Int      *grid_relax_type;
   HYPRE_Int     **grid_relax_points;

   HYPRE_Int     block_mode;

   HYPRE_Int      cheby_order;

 /* Local variables  */
   HYPRE_Int      *lev_counter;
   HYPRE_Int       Solve_err_flag;
   HYPRE_Int       k;
   HYPRE_Int       i, j, jj;
   HYPRE_Int       level;
   HYPRE_Int       cycle_param;
   HYPRE_Int       coarse_grid;
   HYPRE_Int       fine_grid;
   HYPRE_Int       Not_Finished;
   HYPRE_Int       num_sweep;
   HYPRE_Int       cg_num_sweep = 1;
   HYPRE_Int       relax_type;
   HYPRE_Int       relax_points;
   HYPRE_Int       relax_order;
   HYPRE_Int       relax_local;
   HYPRE_Int       old_version = 0;
   HYPRE_Real   *relax_weight;
   HYPRE_Real   *omega;
   HYPRE_Real    alfa, beta, gammaold;
   HYPRE_Real    gamma = 1.0;
   HYPRE_Int       local_size;
/*   HYPRE_Int      *smooth_option; */
   HYPRE_Int       smooth_type;
   HYPRE_Int       smooth_num_levels;
   HYPRE_Int       num_threads, my_id;

   /* RL */
   HYPRE_Int       restrict_type;
   
   HYPRE_Real    alpha;
   HYPRE_Real  **l1_norms = NULL;
   HYPRE_Real   *l1_norms_level;
   HYPRE_Real   **ds = hypre_ParAMGDataChebyDS(amg_data);
   HYPRE_Real   **coefs = hypre_ParAMGDataChebyCoefs(amg_data);

   HYPRE_Int seq_cg = 0;

   MPI_Comm comm;

#if 0
   HYPRE_Real   *D_mat;
   HYPRE_Real   *S_vec;
#endif

#ifdef HYPRE_USING_CALIPER
   cali_id_t iter_attr =
     cali_create_attribute("hypre.par_cycle.level", CALI_TYPE_INT, CALI_ATTR_DEFAULT);
#endif
   
   /* Acquire data and allocate storage */

   num_threads = hypre_NumThreads();

   A_array           = hypre_ParAMGDataAArray(amg_data);
   P_array           = hypre_ParAMGDataPArray(amg_data);
   R_array           = hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = hypre_ParAMGDataCFMarkerArray(amg_data);
   Vtemp             = hypre_ParAMGDataVtemp(amg_data);
   Rtemp             = hypre_ParAMGDataRtemp(amg_data);
   Ptemp             = hypre_ParAMGDataPtemp(amg_data);
   Ztemp             = hypre_ParAMGDataZtemp(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   max_levels        = hypre_ParAMGDataMaxLevels(amg_data);
   cycle_type        = hypre_ParAMGDataCycleType(amg_data);

   A_block_array     = hypre_ParAMGDataABlockArray(amg_data);
   P_block_array     = hypre_ParAMGDataPBlockArray(amg_data);
   R_block_array     = hypre_ParAMGDataRBlockArray(amg_data);
   block_mode        = hypre_ParAMGDataBlockMode(amg_data);

   num_grid_sweeps     = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order         = hypre_ParAMGDataRelaxOrder(amg_data);
   relax_weight        = hypre_ParAMGDataRelaxWeight(amg_data);
   omega               = hypre_ParAMGDataOmega(amg_data);
   smooth_type         = hypre_ParAMGDataSmoothType(amg_data);
   smooth_num_levels   = hypre_ParAMGDataSmoothNumLevels(amg_data);
   l1_norms            = hypre_ParAMGDataL1Norms(amg_data);
   /* smooth_option       = hypre_ParAMGDataSmoothOption(amg_data); */
   /* RL */
   restrict_type = hypre_ParAMGDataRestriction(amg_data); 
   
   /*max_eig_est = hypre_ParAMGDataMaxEigEst(amg_data);
   min_eig_est = hypre_ParAMGDataMinEigEst(amg_data);
   cheby_fraction = hypre_ParAMGDataChebyFraction(amg_data);*/
   cheby_order = hypre_ParAMGDataChebyOrder(amg_data);

   cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

   lev_counter = hypre_CTAlloc(HYPRE_Int,  num_levels, HYPRE_MEMORY_HOST);

   if (hypre_ParAMGDataParticipate(amg_data)) seq_cg = 1;

   /* Initialize */

   Solve_err_flag = 0;

   if (grid_relax_points) old_version = 1;

   num_coeffs = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
   num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
   comm = hypre_ParCSRMatrixComm(A_array[0]);
   hypre_MPI_Comm_rank(comm,&my_id);

   if (block_mode)
   {
      for (j = 1; j < num_levels; j++)
         num_coeffs[j] = hypre_ParCSRBlockMatrixNumNonzeros(A_block_array[j]);

   }
   else
   {
       for (j = 1; j < num_levels; j++)
         num_coeffs[j] = hypre_ParCSRMatrixDNumNonzeros(A_array[j]);
   }

   /*---------------------------------------------------------------------
    *    Initialize cycling control counter
    *
    *     Cycling is controlled using a level counter: lev_counter[k]
    *
    *     Each time relaxation is performed on level k, the
    *     counter is decremented by 1. If the counter is then
    *     negative, we go to the next finer level. If non-
    *     negative, we go to the next coarser level. The
    *     following actions control cycling:
    *
    *     a. lev_counter[0] is initialized to 1.
    *     b. lev_counter[k] is initialized to cycle_type for k>0.
    *
    *     c. During cycling, when going down to level k, lev_counter[k]
    *        is set to the max of (lev_counter[k],cycle_type)
    *---------------------------------------------------------------------*/

   Not_Finished = 1;

   lev_counter[0] = 1;
   for (k = 1; k < num_levels; ++k)
   {
      lev_counter[k] = cycle_type;
   }

   level = 0;
   cycle_param = 1;

   smoother = hypre_ParAMGDataSmoother(amg_data);

   if (smooth_num_levels > 0)
   {
      if (smooth_type == 7 || smooth_type == 8
          || smooth_type == 17 || smooth_type == 18
          || smooth_type == 9 || smooth_type == 19)
      {
         HYPRE_Int actual_local_size = hypre_ParVectorActualLocalSize(Vtemp);
         Utemp = hypre_ParVectorCreate(comm,hypre_ParVectorGlobalSize(Vtemp),
                        hypre_ParVectorPartitioning(Vtemp));
         hypre_ParVectorOwnsPartitioning(Utemp) = 0;
         local_size
            = hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp));
         if (local_size < actual_local_size)
         {
            hypre_VectorData(hypre_ParVectorLocalVector(Utemp)) =
    hypre_CTAlloc(HYPRE_Complex,  actual_local_size, HYPRE_MEMORY_HOST);
            hypre_ParVectorActualLocalSize(Utemp) = actual_local_size;
         }
         else
       hypre_ParVectorInitialize(Utemp);
      }
   }

   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/

  #ifdef HYPRE_USING_CALIPER
  cali_set_int(iter_attr, level);
  #endif

  while (Not_Finished)
  {





    // This is before smoothing occurs, so save a copy of U
    hypre_ParVector *U_copy = hypre_ParVectorCreate(comm, hypre_ParVectorGlobalSize(U_array[level]),hypre_ParVectorPartitioning(U_array[level]));
    hypre_ParVectorInitialize(U_copy);
    hypre_ParVectorSetPartitioningOwner(U_copy, 0);
    hypre_ParVectorCopy( U_array[level], U_copy );





    if (num_levels > 1)
    {
      local_size = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[level]));
      hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp)) = local_size;
      if (smooth_num_levels <= level)
      {
        cg_num_sweep = 1;
        num_sweep = num_grid_sweeps[cycle_param];
        Aux_U = U_array[level];
        Aux_F = F_array[level];
      }
      else if (smooth_type > 9)
      {
        hypre_VectorSize(hypre_ParVectorLocalVector(Ztemp)) = local_size;
        hypre_VectorSize(hypre_ParVectorLocalVector(Rtemp)) = local_size;
        hypre_VectorSize(hypre_ParVectorLocalVector(Ptemp)) = local_size;
        Ztemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Ztemp));
        Ptemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Ptemp));
        hypre_ParVectorSetConstantValues(Ztemp,0);
        alpha = -1.0;
        beta = 1.0;
        //printf("par_cycle.c 1 %d\n",level);
        hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[level],
                    U_array[level], beta, F_array[level], Rtemp);
        //printf("par_cycle.c 1 Done\n");
        cg_num_sweep = hypre_ParAMGDataSmoothNumSweeps(amg_data);
        num_sweep = num_grid_sweeps[cycle_param];
        Aux_U = Ztemp;
        Aux_F = Rtemp;
      }
      else
      {
        cg_num_sweep = 1;
        num_sweep = hypre_ParAMGDataSmoothNumSweeps(amg_data);
        Aux_U = U_array[level];
        Aux_F = F_array[level];
      }
      relax_type = grid_relax_type[cycle_param];
    }
    else /* AB: 4/08: removed the max_levels > 1 check - should do this when max-levels = 1 also */
    {
      /* If no coarsening occurred, apply a simple smoother once */
      Aux_U = U_array[level];
      Aux_F = F_array[level];
      num_sweep = 1;
      /* TK: Use the user relax type (instead of 0) to allow for setting a
      convergent smoother (e.g. in the solution of singular problems). */
      relax_type = hypre_ParAMGDataUserRelaxType(amg_data);
      if (relax_type == -1) relax_type = 6;
    }

    if (l1_norms != NULL)
    l1_norms_level = l1_norms[level];
    else
    l1_norms_level = NULL;

    if (cycle_param == 3 && seq_cg)
    {
      hypre_seqAMGCycle(amg_data, level, F_array, U_array);
    }
    #ifdef HAVE_DSUPERLU
    else if (cycle_param == 3 && hypre_ParAMGDataDSLUSolver(amg_data) != NULL)
    {
      hypre_SLUDistSolve(hypre_ParAMGDataDSLUSolver(amg_data), Aux_F, Aux_U);
    }
    #endif
    else
    {

      /*------------------------------------------------------------------
      * Do the relaxation num_sweep times
      *-----------------------------------------------------------------*/
      for (jj = 0; jj < cg_num_sweep; jj++)
      {
      
        if (smooth_num_levels > level && smooth_type > 9)
          hypre_ParVectorSetConstantValues(Aux_U,0);

        for (j = 0; j < num_sweep; j++)
        {
        

          if (num_levels == 1 && max_levels > 1)
          {
            relax_points = 0;
            relax_local = 0;
          }
          else
          {
            if (old_version)
            relax_points = grid_relax_points[cycle_param][j];
            relax_local = relax_order;
          }

          /*-----------------------------------------------
          * VERY sloppy approximation to cycle complexity
          *-----------------------------------------------*/
          if (old_version && level < num_levels -1)
          {
            switch (relax_points)
            {
              case 1:
                cycle_op_count += num_coeffs[level+1];
              break;

              case -1:
                cycle_op_count += (num_coeffs[level]-num_coeffs[level+1]);
              break;
            }
          }
          else
          {
            cycle_op_count += num_coeffs[level];
          }
          /*-----------------------------------------------
          Choose Smoother
          -----------------------------------------------*/

          if (smooth_num_levels > level &&
          (smooth_type == 7 || smooth_type == 8 ||
          smooth_type == 9 || smooth_type == 19 ||
          smooth_type == 17 || smooth_type == 18))
          {
            hypre_VectorSize(hypre_ParVectorLocalVector(Utemp)) = local_size;
            alpha = -1.0;
            beta = 1.0;
            //printf("par_cycle.c 2 %d\n",level);
            hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[level],
                        U_array[level], beta, Aux_F, Vtemp);
            if (smooth_type == 8 || smooth_type == 18)
            HYPRE_ParCSRParaSailsSolve(smoother[level],
                         (HYPRE_ParCSRMatrix) A_array[level],
                         (HYPRE_ParVector) Vtemp,
                         (HYPRE_ParVector) Utemp);
            else if (smooth_type == 7 || smooth_type == 17)
            HYPRE_ParCSRPilutSolve(smoother[level],
                         (HYPRE_ParCSRMatrix) A_array[level],
                         (HYPRE_ParVector) Vtemp,
                         (HYPRE_ParVector) Utemp);
            else if (smooth_type == 9 || smooth_type == 19)
            HYPRE_EuclidSolve(smoother[level],
                         (HYPRE_ParCSRMatrix) A_array[level],
                         (HYPRE_ParVector) Vtemp,
                         (HYPRE_ParVector) Utemp);
            hypre_ParVectorAxpy(relax_weight[level],Utemp,Aux_U);
          }
          else if (smooth_num_levels > level &&
          (smooth_type == 6 || smooth_type == 16))
          {
            HYPRE_SchwarzSolve(smoother[level],
                       (HYPRE_ParCSRMatrix) A_array[level],
                       (HYPRE_ParVector) Aux_F,
                        (HYPRE_ParVector) Aux_U);
          }
          /*else if (relax_type == 99)*/
          else if (relax_type == 9 || relax_type == 99)
          { /* Gaussian elimination */
            hypre_GaussElimSolve(amg_data, level, relax_type);
          }
          else if (relax_type == 18)
          {   /* L1 - Jacobi*/
            if (relax_order == 1 && cycle_param < 3)
            {
              /* need to do CF - so can't use the AMS one */
              HYPRE_Int i;
              HYPRE_Int loc_relax_points[2];
              if (cycle_param < 2)
              {
                loc_relax_points[0] = 1;
                loc_relax_points[1] = -1;
              }
              else
              {
                loc_relax_points[0] = -1;
                loc_relax_points[1] = 1;
              }
              for (i=0; i < 2; i++)
                hypre_ParCSRRelax_L1_Jacobi(A_array[level],
                                           Aux_F,
                                           CF_marker_array[level],
                                           loc_relax_points[i],
                                           relax_weight[level],
                                           l1_norms[level],
                                           Aux_U,
                                         Vtemp);
            }
            else /* not CF - so use through AMS */
            {
              #if defined(HYPRE_USE_GPU)|| defined(HYPRE_USING_OPENMP_OFFLOAD) || defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
              //printf("par_cycle.c 3 %d\n",level);
                hypre_ParCSRRelax(A_array[level], 
                             Aux_F,
                             1,
                             1,
                             l1_norms_level,
                             relax_weight[level],
                             omega[level],0,0,0,0,
                             Aux_U,
                             Vtemp, 
              Ztemp);
              //printf("par_cycle.c 3 done %d\n",level);
              #else
              if (num_threads == 1)
                hypre_ParCSRRelax(A_array[level],
                             Aux_F,
                             1,
                             1,
                             l1_norms_level,
                             relax_weight[level],
                             omega[level],0,0,0,0,
                             Aux_U,
                             Vtemp,
                             Ztemp);

              else
                hypre_ParCSRRelaxThreads(A_array[level],
                                    Aux_F,
                                    1,
                                    1,
                                    l1_norms_level,
                                    relax_weight[level],
                                    omega[level],
                                    Aux_U,
                                    Vtemp,
                                    Ztemp);
              #endif
            }
          }
          else if (relax_type == 15)
          {  /* CG */
            if (j ==0) /* do num sweep iterations of CG */
              hypre_ParCSRRelax_CG( smoother[level],
                              A_array[level],
                              Aux_F,
                              Aux_U,
                              num_sweep);
          }
          else if (relax_type == 16)
          { /* scaled Chebyshev */
            HYPRE_Int scale = hypre_ParAMGDataChebyScale(amg_data);
            HYPRE_Int variant = hypre_ParAMGDataChebyVariant(amg_data);
            hypre_ParCSRRelax_Cheby_Solve(A_array[level], Aux_F,
                             ds[level], coefs[level],
                             cheby_order, scale,
                             variant, Aux_U, Vtemp, Ztemp );
          }
          else if (relax_type ==17)
          {
            hypre_BoomerAMGRelax_FCFJacobi(A_array[level],
                                    Aux_F,
                                    CF_marker_array[level],
                                    relax_weight[level],
                                    Aux_U,
                                    Vtemp);
          }
          else if (old_version)
          {
            /* 
            printf("cycle %d: relax_type %d, relax_points %d\n", 
                 cycle_param, relax_type, relax_points);
            */
            Solve_err_flag = hypre_BoomerAMGRelax(A_array[level],
                                           Aux_F,
                                           CF_marker_array[level],
                                           relax_type, 
                                           relax_points,
                                           relax_weight[level],
                                           omega[level],
                                           l1_norms_level,
                                           Aux_U,
                                           Vtemp,
                                           Ztemp);
          }
          else
          {
            /* smoother than can have CF ordering */
            if (block_mode)
            {
              Solve_err_flag = hypre_BoomerAMGBlockRelaxIF(A_block_array[level],
                                                          Aux_F,
                                                          CF_marker_array[level],
                                                          relax_type,
                                                          relax_local,
                                                          cycle_param,
                                                          relax_weight[level],
                                                          omega[level],
                                                          Aux_U,
                                                          Vtemp);
            }
            else
            {
              Solve_err_flag = hypre_BoomerAMGRelaxIF(A_array[level],
                                                  Aux_F,
                                                  CF_marker_array[level],
                                                  relax_type,
                                                  relax_local,
                                                  cycle_param,
                                                  relax_weight[level],
                                                  omega[level],
                                                  l1_norms_level,
                                                  Aux_U,
                                                  Vtemp,
                                                  Ztemp);
            }
          }

          if (Solve_err_flag != 0)
          return(Solve_err_flag);
        }
        if  (smooth_num_levels > level && smooth_type > 9)
        {
          gammaold = gamma;
          gamma = hypre_ParVectorInnerProd(Rtemp,Ztemp);
          if (jj == 0)
            hypre_ParVectorCopy(Ztemp,Ptemp);
          else
          {
            beta = gamma/gammaold;
            for (i=0; i < local_size; i++)
            Ptemp_data[i] = Ztemp_data[i] + beta*Ptemp_data[i];
          }

          hypre_ParCSRMatrixMatvec(1.0,A_array[level],Ptemp,0.0,Vtemp);
          alfa = gamma /hypre_ParVectorInnerProd(Ptemp,Vtemp);
          hypre_ParVectorAxpy(alfa,Ptemp,U_array[level]);
          hypre_ParVectorAxpy(-alfa,Vtemp,Rtemp);
        }
      }
    }








    // This is where relaxation is done but we haven't yet messed with restriction/interpolation, so reset U back to U_copy at appropriate locations
    // Want U_array[level] to hold the correct values (i.e. relaxed everywhere except at relax_marker)

    hypre_Vector  *local_U = hypre_ParVectorLocalVector(U_array[level]);
    hypre_Vector  *local_U_before_relax = hypre_ParVectorLocalVector(U_copy);
    hypre_Vector  *local_relax_marker = hypre_ParVectorLocalVector(relax_marker[level]); 
    for (i = 0; i < hypre_VectorSize(local_U); i++)
    {
      if (!(hypre_VectorData(local_relax_marker)[i])) 
      {
        hypre_VectorData(local_U)[i] = hypre_VectorData(local_U_before_relax)[i];
      }
    }
    hypre_ParVectorDestroy(U_copy);











    /*------------------------------------------------------------------
    * Decrement the control counter and determine which grid to visit next
    *-----------------------------------------------------------------*/

    --lev_counter[level];

    if (lev_counter[level] >= 0 && level != num_levels-1)
    {

      /*---------------------------------------------------------------
      * Visit coarser level next.
      * Compute residual using hypre_ParCSRMatrixMatvec.
      * Perform restriction using hypre_ParCSRMatrixMatvecT.
      * Reset counters and cycling parameters for coarse level
      *--------------------------------------------------------------*/



      #if DUMP_INTERMEDIATE_TEST_SOLNS
      sprintf(filename, "outputs/u%d_level%d_relax1", proc, level);
      hypre_ParVectorPrint(U_array[level], filename);
      if (level == 0)
      {
        sprintf(filename, "outputs/f%d_level%d", proc, level);
        hypre_ParVectorPrint(F_array[level], filename);
      }
      #endif





      fine_grid = level;
      coarse_grid = level + 1;

      hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0);

      alpha = -1.0;
      beta = 1.0;

      if (block_mode)
      {
        hypre_ParVectorCopy(F_array[fine_grid],Vtemp);
        hypre_ParCSRBlockMatrixMatvec(alpha, A_block_array[fine_grid], U_array[fine_grid],
                            beta, Vtemp);
      }
      else
      {
        // JSP: avoid unnecessary copy using out-of-place version of SpMV
        //printf("par_cycle.c 4 %d\n",level);
        hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[fine_grid], U_array[fine_grid],
                                   beta, F_array[fine_grid], Vtemp);
        //printf("par_cycle.c 4 done %d\n",level);
        //SyncVectorToHost(hypre_ParVectorLocalVector(Vtemp));
      }




      #if DUMP_INTERMEDIATE_TEST_SOLNS
      sprintf(filename, "outputs/r%d_level%d", proc, level);
      hypre_ParVectorPrint(Vtemp, filename);
      #endif




      alpha = 1.0;
      beta = 0.0;

      if (block_mode)
      {
        hypre_ParCSRBlockMatrixMatvecT(alpha,R_block_array[fine_grid],Vtemp,
                        beta,F_array[coarse_grid]);
      }
      else
      {
        if (restrict_type)
        {
          /* RL: no transpose for R */
          hypre_ParCSRMatrixMatvec(alpha, R_array[fine_grid], Vtemp,
                              beta, F_array[coarse_grid]);
        }
        else
        {
          //SyncVectorToHost(hypre_ParVectorLocalVector(Vtemp));
          //SyncVectorToHost(hypre_ParVectorLocalVector(F_array[coarse_grid]));
          //printf("par_cycle.c 5 %d\n",level,PrintPointerAttributes( hypre_ParCSRMatrixDiag(R_array[fine_grid])->data));
          hypre_ParCSRMatrixMatvecT(alpha,R_array[fine_grid],Vtemp,
                               beta,F_array[coarse_grid]);
          //printf("par_cycle.c 5 Done %d\n",level);
          //UpdateDRC(hypre_ParVectorLocalVector(F_array[coarse_grid]));
          //SyncVectorToHost(hypre_ParVectorLocalVector(F_array[coarse_grid]));
        }
      }



      #if DUMP_INTERMEDIATE_TEST_SOLNS
      sprintf(filename, "outputs/f%d_level%d", proc, level+1);
      hypre_ParVectorPrint(F_array[coarse_grid], filename);
      #endif




      ++level;
      lev_counter[level] = hypre_max(lev_counter[level],cycle_type);
      cycle_param = 1;
      if (level == num_levels-1) cycle_param = 3;

      #ifdef HYPRE_USING_CALIPER
      cali_set_int(iter_attr, level);  /* set the level for caliper here */
      #endif
    }

    else if (level != 0)
    {
      /*---------------------------------------------------------------
      * Visit finer level next.
      * Interpolate and add correction using hypre_ParCSRMatrixMatvec.
      * Reset counters and cycling parameters for finer level.
      *--------------------------------------------------------------*/

      fine_grid = level - 1;
      coarse_grid = level;
      alpha = 1.0;
      beta = 1.0;
      if (block_mode)
      {
        hypre_ParCSRBlockMatrixMatvec(alpha, P_block_array[fine_grid],
                       U_array[coarse_grid],
                       beta, U_array[fine_grid]);
      }
      else
      {
        //printf("par_cycle.c 6 %d\n",level);
        hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid],
                         U_array[coarse_grid],
                         beta, U_array[fine_grid]);
        //printf("par_cycle.c 6 done %d\n",level);
      }


      #if DUMP_INTERMEDIATE_TEST_SOLNS
      sprintf(filename, "outputs/u%d_level%d_relax2", proc, level);
      hypre_ParVectorPrint(U_array[level], filename);
      sprintf(filename, "outputs/u%d_level%d_project", proc, level-1);
      hypre_ParVectorPrint(U_array[level-1], filename);
      #endif



      --level;
      cycle_param = 2;

      #ifdef HYPRE_USING_CALIPER
      cali_set_int(iter_attr, level);  /* set the level for caliper here */
      #endif
    }
    else
    {
      Not_Finished = 0;
    }
  }




  #if DUMP_INTERMEDIATE_TEST_SOLNS
  sprintf(filename, "outputs/u%d_level%d_relax2", proc, 0);
  hypre_ParVectorPrint(U_array[0], filename);
  #endif



  #ifdef HYPRE_USING_CALIPER
  cali_end(iter_attr);  /* unset "iter" */
  #endif

  hypre_ParAMGDataCycleOpCount(amg_data) = cycle_op_count;

  hypre_TFree(lev_counter, HYPRE_MEMORY_HOST);
  hypre_TFree(num_coeffs, HYPRE_MEMORY_HOST);
  if (smooth_num_levels > 0)
  {
    if (smooth_type == 7 || smooth_type == 8 || smooth_type == 9 ||
    smooth_type == 17 || smooth_type == 18 || smooth_type == 19 )
      hypre_ParVectorDestroy(Utemp);
  }



    #if DEBUGGING_MESSAGES
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    if (myid == 0) hypre_printf("Finished with TestBoomerAMGCycle\n");
    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
    #endif

  return(Solve_err_flag);
}


