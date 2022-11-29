/*BHEADER**********************************************************************
 * Copyright (c) 2017,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Ulrike Yang (yang11@llnl.gov) et al. CODE-LLNL-738-322.
 * This file is part of AMG.  See files README and COPYRIGHT for details.
 *
 * AMG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This software is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTIBILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
 * GNU General Public License for more details.
 *
 ***********************************************************************EHEADER*/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface. AJC 7/99.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int BuildIJLaplacian27pt (HYPRE_Int argc , char *argv [], HYPRE_BigInt *size , HYPRE_IJMatrix *A_ptr );
HYPRE_BigInt hypre_map27( HYPRE_BigInt  ix, HYPRE_BigInt  iy, HYPRE_BigInt  iz,
      HYPRE_Int  px, HYPRE_Int  py, HYPRE_Int  pz,
      HYPRE_BigInt  Cx, HYPRE_BigInt  Cy, HYPRE_BigInt  Cz, 
      HYPRE_Int nx, HYPRE_Int nxy);

#ifdef __cplusplus
}
#endif
#define SECOND_TIME 0
 
hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int           arg_index;
   HYPRE_Int           print_usage;
   HYPRE_Int           solver_id;
   HYPRE_Int           problem_id;
   HYPRE_Int           ioutdat;
   HYPRE_Int           poutdat;
   HYPRE_Int           debug_flag;
   HYPRE_Int           i; 
   HYPRE_Int           num_iterations;
   HYPRE_Int           max_iter = 1000;
   HYPRE_Int           mg_max_iter = 100;
   HYPRE_Real          final_res_norm;
   void               *object;

   HYPRE_IJMatrix      ij_A; 
   HYPRE_IJVector      ij_b;
   HYPRE_IJVector      ij_x;

   HYPRE_ParCSRMatrix  parcsr_A;
   HYPRE_ParVector     b;
   HYPRE_ParVector     x;

   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond=NULL, pcg_precond_gotten;

   HYPRE_Int           myid = 0;
   HYPRE_Int           num_procs = 1;
   HYPRE_Int	       agg_num_levels = 1;

   HYPRE_Int	       time_index;
   MPI_Comm            comm = hypre_MPI_COMM_WORLD;
   HYPRE_BigInt first_local_row, last_local_row;
   HYPRE_BigInt first_local_col, last_local_col;
   HYPRE_Int local_num_rows;

   /* parameters for BoomerAMG */
   HYPRE_Int    P_max_elmts = 8;
   HYPRE_Int    coarsen_type = 8;
   HYPRE_Int    num_sweeps = 2;  
   HYPRE_Int    relax_type = 18;   
   HYPRE_Int    rap2=1;
   //HYPRE_Int    mod_rap2=0;
   HYPRE_Int    keepTranspose = 0;
   HYPRE_Real   tol = 1.e-8, pc_tol = 0.;
   HYPRE_Real   atol = 0.0;

   HYPRE_Real   wall_time;
   HYPRE_BigInt system_size;
   HYPRE_Real   cum_nnz_AP = 1;
   HYPRE_Real   FOM1 = 0, FOM2 = 0;

   /* parameters for GMRES */
   HYPRE_Int	    k_dim = 20;
   /* interpolation */
   HYPRE_Int      interp_type  = 17; /* default value */

   HYPRE_Int      print_system = 0;
   HYPRE_Int      print_stats = 0;

   HYPRE_Int rel_change = 0;

   HYPRE_Real *values;

#if defined(HYPRE_USING_MEMORY_TRACKER)
   HYPRE_Int print_mem_tracker = 0;
   char mem_tracker_name[HYPRE_MAX_FILE_NAME_LEN] = {0};
#endif

   /* default execution policy and memory space */
#if defined(HYPRE_TEST_USING_HOST)
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_HOST;
   //HYPRE_ExecutionPolicy exec2_policy = HYPRE_EXEC_HOST;
#else
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_DEVICE;
   //HYPRE_ExecutionPolicy exec2_policy = HYPRE_EXEC_DEVICE;
#endif
   for (arg_index = 1; arg_index < argc; arg_index ++)
   {
      if ( strcmp(argv[arg_index], "-memory_host") == 0 )
      {
         memory_location = HYPRE_MEMORY_HOST;
      }
      else if ( strcmp(argv[arg_index], "-memory_device") == 0 )
      {
         memory_location = HYPRE_MEMORY_DEVICE;
      }
      else if ( strcmp(argv[arg_index], "-exec_host") == 0 )
      {
         default_exec_policy = HYPRE_EXEC_HOST;
      }
      else if ( strcmp(argv[arg_index], "-exec_device") == 0 )
      {
         default_exec_policy = HYPRE_EXEC_DEVICE;
      }
      /*else if ( strcmp(argv[arg_index], "-exec2_host") == 0 )
      {
         exec2_policy = HYPRE_EXEC_HOST;
      }
      else if ( strcmp(argv[arg_index], "-exec2_device") == 0 )
      {
         exec2_policy = HYPRE_EXEC_DEVICE;
      }*/
   }

   /*if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      keepTranspose = 1;
      coarsen_type  = 8;
      mod_rap2      = 1;
   }*/

#ifdef HYPRE_USING_DEVICE_POOL
   /* device pool allocator */
   hypre_uint mempool_bin_growth   = 8,
              mempool_min_bin      = 3,
              mempool_max_bin      = 9;
   size_t mempool_max_cached_bytes = 2000LL * 1024 * 1024;
#endif

   /*-----------------------------------------------------------
    * Initialize MPI
    *-----------------------------------------------------------*/

   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   debug_flag = 0;

   solver_id = 1;
   problem_id = 1;

   ioutdat = 0;
   poutdat = 0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      else
      {
         arg_index++;
      }
   }

   /* defaults for GMRES */

   k_dim = 100;

   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-problem") == 0 )
      {
         arg_index++;
         problem_id = atoi(argv[arg_index++]);
         if (problem_id == 2) 
	 {
	    solver_id = 3;
	 }
         
      }
      else if ( strcmp(argv[arg_index], "-printstats") == 0 )
      {
         arg_index++;
         ioutdat  = 1;
         poutdat  = 1;
         print_stats  = 1;
      }
      else if ( strcmp(argv[arg_index], "-printallstats") == 0 )
      {
         arg_index++;
         ioutdat  = 3;
         poutdat  = 1;
         print_stats  = 1;
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-keepT") == 0 )
      {
         arg_index++;
         keepTranspose = 1;
         rap2 = 0;
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
 
   if ( (print_usage) && (myid == 0) )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [<options>]\n", argv[0]);
      hypre_printf("\n");
      hypre_printf("  -problem <ID>: problem ID\n");
      hypre_printf("       1 = solves 1 large problem with AMG-PCG (default) \n");
      hypre_printf("       2 = simulates a time-dependent loop with AMG-GMRES\n");
      hypre_printf("\n");
      hypre_printf("  -n <nx> <ny> <nz>: problem size per MPI process (default: nx=ny=nz=10)\n");
      hypre_printf("\n");
      hypre_printf("  -P <px> <py> <pz>: processor topology (default: px=py=pz=1)\n");
      hypre_printf("\n");
      hypre_printf("  -print       : prints the system\n");
      hypre_printf("  -printstats  : prints preconditioning and convergence stats\n");
      hypre_printf("  -printallstats  : prints preconditioning and convergence stats\n");
      hypre_printf("                    including residual norms for each iteration\n");
      hypre_printf("\n"); 
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("Running with these driver parameters:\n");
      hypre_printf("  solver ID    = %d\n\n", solver_id);
   }

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before HYPRE_Init() and should not be changed after
    *-----------------------------------------------------------------*/
   hypre_bind_device(myid, num_procs, comm);

   time_index = hypre_InitializeTiming("Hypre init");
   hypre_BeginTiming(time_index);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   HYPRE_Init();

   hypre_EndTiming(time_index);
   hypre_GetTiming("Hypre init times", &wall_time, comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

#ifdef HYPRE_USING_DEVICE_POOL
   /* To be effective, hypre_SetCubMemPoolSize must immediately follow HYPRE_Init */
   HYPRE_SetGPUMemoryPoolSize( mempool_bin_growth, mempool_min_bin,
                               mempool_max_bin, mempool_max_cached_bytes );
#endif

#if defined(HYPRE_USING_MEMORY_TRACKER)
   hypre_MemoryTrackerSetPrint(print_mem_tracker);
   if (mem_tracker_name[0]) { hypre_MemoryTrackerSetFileName(mem_tracker_name); }
#endif

   /* default memory location */
   HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   HYPRE_SetExecutionPolicy(default_exec_policy);

#if defined(HYPRE_USING_GPU)
   HYPRE_Int ierr;
   ierr = HYPRE_SetSpMVUseVendor(spmv_use_vendor); hypre_assert(ierr == 0);
   /* use vendor implementation for SpGEMM */
   ierr = HYPRE_SetSpGemmUseVendor(spgemm_use_vendor); hypre_assert(ierr == 0);
   ierr = hypre_SetSpGemmAlgorithm(spgemm_alg); hypre_assert(ierr == 0);
   ierr = hypre_SetSpGemmBinned(spgemm_binned); hypre_assert(ierr == 0);
   ierr = hypre_SetSpGemmRownnzEstimateMethod(spgemm_rowest_mtd); hypre_assert(ierr == 0);
   if (spgemm_rowest_nsamples > 0) { ierr = hypre_SetSpGemmRownnzEstimateNSamples(spgemm_rowest_nsamples); hypre_assert(ierr == 0); }
   if (spgemm_rowest_mult > 0.0) { ierr = hypre_SetSpGemmRownnzEstimateMultFactor(spgemm_rowest_mult); hypre_assert(ierr == 0); }
   /* use cuRand for PMIS */
   HYPRE_SetUseGpuRand(use_curand);
#endif

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("Spatial Operator");
   hypre_BeginTiming(time_index);

   BuildIJLaplacian27pt(argc, argv, &system_size, &ij_A);
   HYPRE_IJMatrixGetObject(ij_A, &object);
   parcsr_A = (HYPRE_ParCSRMatrix) object;


   hypre_EndTiming(time_index);
   hypre_GetTiming("Generate Matrix", &wall_time, comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("RHS and Initial Guess");
   hypre_BeginTiming(time_index);

   HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                              &first_local_row, &last_local_row ,
                                              &first_local_col, &last_local_col );

   local_num_rows = (HYPRE_Int)(last_local_row - first_local_row + 1);

   if (myid == 0)
   {
      hypre_printf("  RHS vector has unit components\n");
      hypre_printf("  Initial guess is 0\n");
   }

   /* RHS */
   HYPRE_IJVectorCreate(comm, first_local_row, last_local_row, &ij_b);
   HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_b);

   /* Initial guess */
   HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
   HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_x);

   values = hypre_CTAlloc(HYPRE_Complex, local_num_rows, memory_location);

   HYPRE_IJVectorSetValues(ij_x, local_num_rows, NULL, values);

   for (i = 0; i < local_num_rows; i++)
      values[i] = 1.0;

   HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
   hypre_TFree(values, memory_location);

   HYPRE_IJVectorGetObject( ij_b, &object );
   b = (HYPRE_ParVector) object;

   HYPRE_IJVectorGetObject( ij_x, &object );
   x = (HYPRE_ParVector) object;

   hypre_EndTiming(time_index);
   hypre_GetTiming("IJ Vector Setup", &wall_time, comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_IJMatrixPrint(ij_A, "IJ.out.A");
      HYPRE_IJVectorPrint(ij_b, "IJ.out.b");
      HYPRE_IJVectorPrint(ij_x, "IJ.out.x0");

   }

   /*-----------------------------------------------------------
    * Problem 1: Solve one large problem with AMG-PCG
    *-----------------------------------------------------------*/


   if (problem_id == 1 )
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_MPI_Barrier(comm);
      hypre_BeginTiming(time_index);
      HYPRE_ParCSRPCGCreate(comm, &pcg_solver);
      HYPRE_PCGSetMaxIter(pcg_solver, max_iter);
      HYPRE_PCGSetTol(pcg_solver, tol);
      HYPRE_PCGSetTwoNorm(pcg_solver, 1);
      HYPRE_PCGSetRelChange(pcg_solver, rel_change);
      HYPRE_PCGSetPrintLevel(pcg_solver, ioutdat);
      HYPRE_PCGSetAbsoluteTol(pcg_solver, atol);

      /* use BoomerAMG as preconditioner */
      if (myid == 0 && print_stats) hypre_printf("Solver: AMG-PCG\n");
      HYPRE_BoomerAMGCreate(&pcg_precond); 
      HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
      HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
      HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
      HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
      HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
      HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
      if (relax_type > -1) HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
      HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
      HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
      HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
      HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
      HYPRE_BoomerAMGSetCumNnzAP(pcg_precond, cum_nnz_AP);
      HYPRE_PCGSetMaxIter(pcg_solver, mg_max_iter);
      HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                             pcg_precond);
 
      HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
         hypre_printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
         return(-1);
      }
      else 
         if (myid == 0 && print_stats)
            hypre_printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");

      HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, 
                     (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_MPI_Barrier(comm);
      hypre_EndTiming(time_index);
      hypre_GetTiming("Problem 1: AMG Setup Time", &wall_time, comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);

      HYPRE_BoomerAMGGetCumNnzAP(pcg_precond, &cum_nnz_AP);

      FOM1 = cum_nnz_AP/ wall_time;

      if (myid == 0)
            printf ("\nFOM_Setup: nnz_AP / Setup Phase Time: %e\n\n", FOM1);
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_MPI_Barrier(comm);
      hypre_BeginTiming(time_index);
 
      HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, 
                     (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_MPI_Barrier(comm);
      hypre_EndTiming(time_index);
      hypre_GetTiming("Problem 1: AMG-PCG Solve Time", &wall_time, comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);
 
      HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

   
      HYPRE_BoomerAMGDestroy(pcg_precond);

      HYPRE_ParCSRPCGDestroy(pcg_solver);

      FOM2 = cum_nnz_AP*(HYPRE_Real)num_iterations/ wall_time;

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
         printf ("\nFOM_Solve: nnz_AP * Iterations / Solve Phase Time: %e\n\n", FOM2);
         FOM1 += 3.0*FOM2;
         FOM1 /= 4.0;
         printf ("\n\nFigure of Merit (FOM_1): %e\n\n", FOM1);
      }
 
   }

   /*-----------------------------------------------------------
    * Problem 2: simulate time-dependent problem AMG-GMRES
    *-----------------------------------------------------------*/

   if (problem_id == 2)
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_MPI_Barrier(comm);
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRGMRESCreate(comm, &pcg_solver);
      HYPRE_GMRESSetKDim(pcg_solver, k_dim);
      HYPRE_GMRESSetMaxIter(pcg_solver, max_iter);
      HYPRE_GMRESSetTol(pcg_solver, tol);
      HYPRE_GMRESSetAbsoluteTol(pcg_solver, atol);
      HYPRE_GMRESSetLogging(pcg_solver, 1);
      HYPRE_GMRESSetPrintLevel(pcg_solver, ioutdat);
      HYPRE_GMRESSetRelChange(pcg_solver, rel_change);
 
      /* use BoomerAMG as preconditioner */
      if (myid == 0 && print_stats) hypre_printf("Solver: AMG-GMRES\n");

      HYPRE_BoomerAMGCreate(&pcg_precond); 
      HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
      HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
      HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
      HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
      HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
      HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
      if (relax_type > -1) HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
      HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
      HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
      HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
      HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
      HYPRE_BoomerAMGSetCumNnzAP(pcg_precond, cum_nnz_AP);
      HYPRE_GMRESSetMaxIter(pcg_solver, mg_max_iter);
      HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                               pcg_precond);

      HYPRE_GMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         hypre_printf("HYPRE_GMRESGetPrecond got bad precond\n");
         return(-1);
      }
      else
         if (myid == 0 && print_stats)
            hypre_printf("HYPRE_GMRESGetPrecond got good precond\n");
      HYPRE_GMRESSetup (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_MPI_Barrier(comm);
      hypre_EndTiming(time_index);
      hypre_GetTiming("Problem 2: AMG Setup Time", &wall_time, comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);

      HYPRE_BoomerAMGGetCumNnzAP(pcg_precond, &cum_nnz_AP); 

      FOM2 = cum_nnz_AP/ wall_time;

      if (myid == 0)
            printf ("\nFOM_Setup: nnz_AP / Setup Phase Time: %e\n\n", FOM2);

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_MPI_Barrier(comm);
      hypre_BeginTiming(time_index);
 
      HYPRE_GMRESSolve (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_MPI_Barrier(comm);
      hypre_EndTiming(time_index);
      hypre_GetTiming("Problem 2: AMG-GMRES Solve Time", &wall_time, comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);
 
      HYPRE_GMRESGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
      
      HYPRE_BoomerAMGDestroy(pcg_precond);

      HYPRE_ParCSRGMRESDestroy(pcg_solver);
      FOM2 = cum_nnz_AP*(HYPRE_Real)num_iterations/ wall_time;

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
         printf ("\nFOM_Solve: nnz_AP * Iterations / Solve Phase Time: %e\n\n", FOM2);
         FOM1 += 3.0*FOM2;
         FOM1 /= 4.0;
         printf ("\n\nFigure of Merit (FOM_1): %e\n\n", FOM1);
      }
   }
 

   /*-----------------------------------------------------------
    * Print the solution
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_IJVectorPrint(ij_x, "IJ.out.x");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_IJMatrixDestroy(ij_A);

   HYPRE_IJVectorDestroy(ij_b);

   HYPRE_IJVectorDestroy(ij_x);

   /* Finalize hypre */
   HYPRE_Finalize();


   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}

/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildIJLaplacian27pt( HYPRE_Int         argc,
                       char            *argv[],
                       HYPRE_BigInt      *system_size_ptr,
                       HYPRE_IJMatrix  *ij_A_ptr     )
{
   MPI_Comm        comm = hypre_MPI_COMM_WORLD;
   HYPRE_Int    nx, ny, nz;
   HYPRE_Int       P, Q, R;

   HYPRE_IJMatrix  ij_A;

   HYPRE_Int       num_procs, myid;
   HYPRE_Int       px, py, pz;
   HYPRE_Real     *value;
   HYPRE_Int      *diag_i;
   HYPRE_Int      *offd_i;
   HYPRE_BigInt   *row_nums;
   HYPRE_BigInt   *col_nums;
   HYPRE_Int      *num_cols;
   HYPRE_Complex  *data;

   HYPRE_BigInt row_index;
   HYPRE_Int i;
   HYPRE_Int local_size;
   HYPRE_BigInt global_size;

   HYPRE_Int nxy;
   HYPRE_BigInt nx_global, ny_global, nz_global;
   HYPRE_Int all_threads;
   HYPRE_Int *nnz;
   HYPRE_BigInt Cx, Cy, Cz;
   HYPRE_Int arg_index;
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );
   all_threads = hypre_NumThreads();
   nnz = hypre_CTAlloc(HYPRE_Int, all_threads, memory_location);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = num_procs;;
   Q  = 1;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   nx_global = (HYPRE_BigInt)(P*nx); 
   ny_global = (HYPRE_BigInt)(Q*ny); 
   nz_global = (HYPRE_BigInt)(R*nz); 
   global_size = nx_global*ny_global*nz_global;
   if (myid == 0)

   {
      hypre_printf("  Laplacian_27pt:\n");
      hypre_printf("    (Nx, Ny, Nz) = (%b, %b, %b)\n", nx_global, ny_global, nz_global);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute px,py,pz from P,Q,R and myid */
   px = myid % P;
   py = (( myid - px)/P) % Q;
   pz = ( myid - px - P*py)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   value = hypre_CTAlloc(HYPRE_Real, 2, memory_location);

   value[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
      value[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
      value[0] = 2.0;
   value[1] = -1.;

   local_size = nx*ny*nz;

   row_nums = hypre_CTAlloc(HYPRE_BigInt, local_size, memory_location);
   num_cols = hypre_CTAlloc(HYPRE_Int, local_size, memory_location);
   row_index = (HYPRE_BigInt)(myid*local_size);

   HYPRE_IJMatrixCreate( comm, row_index, (HYPRE_BigInt)(row_index+local_size-1),
                               row_index, (HYPRE_BigInt)(row_index+local_size-1),
                               &ij_A );

   HYPRE_IJMatrixSetObjectType( ij_A, HYPRE_PARCSR );

   nxy = nx*ny;

   diag_i = hypre_CTAlloc(HYPRE_Int, local_size, memory_location);
   offd_i = hypre_CTAlloc(HYPRE_Int, local_size, memory_location);

   Cx = (HYPRE_BigInt)(nx*(ny*nz-1));
   Cy = (HYPRE_BigInt)(nxy*(P*nz-1));
   Cz = (HYPRE_BigInt)(local_size*(P*Q-1));
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel
#endif
   {
    HYPRE_BigInt ix, iy, iz;
    HYPRE_Int cnt, o_cnt;
    HYPRE_BigInt ix_start, ix_end;
    HYPRE_BigInt iy_start, iy_end;
    HYPRE_BigInt iz_start, iz_end;
    HYPRE_Int num_threads, my_thread;
    HYPRE_Int all_nnz=0;
    HYPRE_Int size, rest;
    HYPRE_Int new_row_index;
    num_threads = hypre_NumActiveThreads();
    my_thread = hypre_GetThreadNum();
    size = nz/num_threads;
    rest = nz - size*num_threads;
    ix_start = nx*px;
    ix_end = ix_start+nx;
    iy_start = ny*py;
    iy_end = iy_start+ny;
    if (my_thread < rest)
    {
       iz_start = (HYPRE_BigInt)(nz*pz + my_thread*size+my_thread);
       iz_end = (HYPRE_BigInt)(nz*pz + (my_thread+1)*size+my_thread+1);
       cnt = (my_thread*size+my_thread)*nxy-1;
    }
    else
    {
       iz_start = (HYPRE_BigInt)(nz*pz + my_thread*size+rest);
       iz_end = (HYPRE_BigInt)(nz*pz + (my_thread+1)*size+rest);
       cnt = (my_thread*size+rest)*nxy-1;
    }
    o_cnt = cnt;

    for (iz = iz_start;  iz < iz_end; iz++)
    {
      for (iy = iy_start;  iy < iy_end; iy++)
      {
         for (ix = ix_start; ix < ix_end; ix++)
         {
            cnt++;
            o_cnt++;
            diag_i[cnt]++;
            if (iz > (HYPRE_BigInt)(nz*pz)) 
            {
               diag_i[cnt]++;
               if (iy > (HYPRE_BigInt)(ny*py)) 
               {
                  diag_i[cnt]++;
      	          if (ix > (HYPRE_BigInt)(nx*px))
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx_global-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
               if (ix > (HYPRE_BigInt)(nx*px))
                  diag_i[cnt]++;
               else
               {
                  if (ix) 
                  {
                     offd_i[o_cnt]++;
                  }
               }
               if (ix+1 < (HYPRE_BigInt)(nx*(px+1))) 
                  diag_i[cnt]++;
               else
               {
                  if (ix+1 < nx_global) 
                  {
                     offd_i[o_cnt]++; 
                  }
               }
               if (iy+1 < (HYPRE_BigInt)(ny*(py+1)))
               {
                  diag_i[cnt]++;
      	          if (ix > (HYPRE_BigInt)(nx*px))
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy+1 < ny_global) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx*px)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx_global-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
            }
            else
            {
               if (iz)
	       {
		  offd_i[o_cnt]++;
                  if (iy > (HYPRE_BigInt)(ny*py))
                  {
                     offd_i[o_cnt]++;
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	        offd_i[o_cnt]++;
      	             }
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	   	        offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > (HYPRE_BigInt)(nx*px))
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx_global-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
                  if (ix > (HYPRE_BigInt)(nx*px))
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (ix+1 < (HYPRE_BigInt)(nx*(px+1)))
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix+1 < nx_global) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (iy+1 < (HYPRE_BigInt)(ny*(py+1)))
                  {
                     offd_i[o_cnt]++;
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	           offd_i[o_cnt]++;
      	             }
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	   	           offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny_global) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > (HYPRE_BigInt)(nx*px))
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx_global-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
               }
            }
            if (iy > (HYPRE_BigInt)(ny*py)) 
            {
               diag_i[cnt]++;
   	       if (ix > (HYPRE_BigInt)(nx*px))
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix) 
   		     offd_i[o_cnt]++;
   	       }
   	       if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix+1 < nx_global) 
   		     offd_i[o_cnt]++;
   	       }
            }
            else
            {
               if (iy) 
               {
                  offd_i[o_cnt]++;
   	          if (ix > (HYPRE_BigInt)(nx*px))
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix < nx_global-1)
   	          {
   	             offd_i[o_cnt]++;
   	          }
               }
            }
            if (ix > (HYPRE_BigInt)(nx*px))
               diag_i[cnt]++;
            else
            {
               if (ix) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (ix+1 < (HYPRE_BigInt)(nx*(px+1)))
               diag_i[cnt]++;
            else
            {
               if (ix+1 < nx_global) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (iy+1 < (HYPRE_BigInt)(ny*(py+1)))
            {
               diag_i[cnt]++;
   	       if (ix > (HYPRE_BigInt)(nx*px))
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix) 
   		     offd_i[o_cnt]++;
   	       }
   	       if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix+1 < nx_global) 
   		     offd_i[o_cnt]++;
   	       }
            }
            else
            {
               if (iy+1 < ny_global) 
               {
                  offd_i[o_cnt]++;
   	          if (ix > (HYPRE_BigInt)(nx*px))
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix < nx_global-1)
   	          {
   	             offd_i[o_cnt]++;
   	          }
               }
            }
            if (iz+1 < (HYPRE_BigInt)(nz*(pz+1)))
            {
               diag_i[cnt]++;
               if (iy > (HYPRE_BigInt)(ny*py))
               {
                  diag_i[cnt]++;
      	          if (ix > (HYPRE_BigInt)(nx*px))
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx_global-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
               if (ix > (HYPRE_BigInt)(nx*px))
                  diag_i[cnt]++;
               else
               {
                  if (ix) 
                  {
                     offd_i[o_cnt]++; 
                  }
               }
               if (ix+1 < (HYPRE_BigInt)(nx*(px+1)))
                  diag_i[cnt]++;
               else
               {
                  if (ix+1 < nx_global) 
                  {
                     offd_i[o_cnt]++; 
                  }
               }
               if (iy+1 < (HYPRE_BigInt)(ny*(py+1)))
               {
                  diag_i[cnt]++;
      	          if (ix > (HYPRE_BigInt)(nx*px))
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy+1 < ny_global) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx_global-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
            }
            else
            {
               if (iz+1 < nz_global)
	       {
		  offd_i[o_cnt]++;
                  if (iy > (HYPRE_BigInt)(ny*py))
                  {
                     offd_i[o_cnt]++;
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	        offd_i[o_cnt]++;
      	             }
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	   	        offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > (HYPRE_BigInt)(nx*px))
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx_global-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
                  if (ix > (HYPRE_BigInt)(nx*px))
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (ix+1 < (HYPRE_BigInt)(nx*(px+1)))
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix+1 < nx_global) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (iy+1 < (HYPRE_BigInt)(ny*(py+1)))
                  {
                     offd_i[o_cnt]++;
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	           offd_i[o_cnt]++;
      	             }
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	   	           offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny_global) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > (HYPRE_BigInt)(nx*px))
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx_global-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
               }
            }
            nnz[my_thread] += diag_i[cnt]+offd_i[o_cnt];
            row_nums[cnt] = row_index+cnt;
            num_cols[cnt] = diag_i[cnt]+offd_i[o_cnt];
         }
      }
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   if (my_thread == 0)
   {
      for (i=1; i< num_threads; i++)
         nnz[i]+= nnz[i-1];

      all_nnz = nnz[num_threads-1];
      col_nums = hypre_CTAlloc(HYPRE_BigInt, all_nnz, memory_location);
      data = hypre_CTAlloc(HYPRE_Complex, all_nnz, memory_location);

      HYPRE_IJMatrixSetDiagOffdSizes( ij_A, diag_i, offd_i);
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   if (my_thread) 
   {
      cnt = nnz[my_thread-1];
      new_row_index = row_index+(HYPRE_BigInt)((iz_start-nz*pz)*nxy);
   }
   else
   {
      cnt = 0;
      new_row_index = row_index;
   }
   for (iz = iz_start;  iz < iz_end; iz++)
   {
      for (iy = iy_start;  iy < iy_end; iy++)
      {
         for (ix = ix_start; ix < ix_end; ix++)
         {
            col_nums[cnt] = new_row_index;
            data[cnt++] = value[0];
            if (iz > (HYPRE_BigInt)(nz*pz))
            {
               if (iy > (HYPRE_BigInt)(ny*py))
               {
      	          if (ix > (HYPRE_BigInt)(nx*px))
      	          {
      	             col_nums[cnt] = new_row_index-nxy-nx-1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	          }
      	          col_nums[cnt] = new_row_index-nxy-nx;
      	          data[cnt++] = value[1];
      	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	          {
      	             col_nums[cnt] = new_row_index-nxy-nx+1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	          }
               }
               else
               {
                  if (iy) 
                  {
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px-1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      		     col_nums[cnt] = hypre_map27(ix,iy-1,iz-1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix < nx_global-1)
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px+1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
                  }
               }
               if (ix > (HYPRE_BigInt)(nx*px))
      	       {   
      	          col_nums[cnt] = new_row_index-nxy-1;
      	          data[cnt++] = value[1];
      	       }   
               else
               {
                  if (ix) 
                  {
      		     col_nums[cnt] = hypre_map27(ix-1,iy,iz-1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
               }
      	       col_nums[cnt] = new_row_index-nxy;
      	       data[cnt++] = value[1];
               if (ix+1 < (HYPRE_BigInt)(nx*(px+1)))
      	       {   
      	          col_nums[cnt] = new_row_index-nxy+1;
      	          data[cnt++] = value[1];
      	       }   
               else
               {
                  if (ix+1 < nx_global) 
                  {
      		     col_nums[cnt] = hypre_map27(ix+1,iy,iz-1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
               }
               if (iy+1 < (HYPRE_BigInt)(ny*(py+1)))
               {
      	          if (ix > (HYPRE_BigInt)(nx*px))
      	          {
      	             col_nums[cnt] = new_row_index-nxy+nx-1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
                     {
      		        col_nums[cnt] = hypre_map27(ix-1,iy+1,iz-1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
      	          }
      	          col_nums[cnt] = new_row_index-nxy+nx;
      	          data[cnt++] = value[1];
      	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	          {
      	             col_nums[cnt] = new_row_index-nxy+nx+1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
                     {
      		        col_nums[cnt] = hypre_map27(ix+1,iy+1,iz-1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
      	          }
               }
               else
               {
                  if (iy+1 < ny_global) 
                  {
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy+1,iz-1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy+1,iz-1,px-1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      		     col_nums[cnt] = hypre_map27(ix,iy+1,iz-1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy+1,iz-1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix < nx_global-1)
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy+1,iz-1,px+1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
                  }
               }
            }
            else
            {
               if (iz)
	       {
                  if (iy > (HYPRE_BigInt)(ny*py))
                  {
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px-1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
      		     col_nums[cnt] = hypre_map27(ix,iy-1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	                {
      		           col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px+1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
      	                if (ix > (HYPRE_BigInt)(nx*px))
      	                {
      		           col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px,py-1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy-1,iz-1,px-1,py-1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      		        col_nums[cnt] = hypre_map27(ix,iy-1,iz-1,px,py-1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	                if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	                {
      		           col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px,py-1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix < nx_global-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy-1,iz-1,px+1,py-1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
                     }
                  }
                  if (ix > (HYPRE_BigInt)(nx*px))
                  {
      		     col_nums[cnt] =hypre_map27(ix-1,iy,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix) 
                     {
      		        col_nums[cnt] =hypre_map27(ix-1,iy,iz-1,px-1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
                  }
      		  col_nums[cnt] =hypre_map27(ix,iy,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
                  if (ix+1 < (HYPRE_BigInt)(nx*(px+1)))
                  {
      		     col_nums[cnt] =hypre_map27(ix+1,iy,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix+1 < nx_global) 
                     {
      		        col_nums[cnt] =hypre_map27(ix+1,iy,iz-1,px+1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
                  }
                  if (iy+1 < (HYPRE_BigInt)(ny*(py+1)))
                  {
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy+1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz-1,px-1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
      		     col_nums[cnt] =hypre_map27(ix,iy+1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy+1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz-1,px+1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny_global) 
                     {
      	                if (ix > (HYPRE_BigInt)(nx*px))
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz-1,px,py+1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz-1,px-1,py+1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      		        col_nums[cnt] =hypre_map27(ix,iy+1,iz-1,px,py+1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	                if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz-1,px,py+1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix < nx_global-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz-1,px+1,py+1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
                     }
                  }
               }
            }
            if (iy > (HYPRE_BigInt)(ny*py))
            {
   	       if (ix > (HYPRE_BigInt)(nx*px))
   	       {
   	          col_nums[cnt] = new_row_index-nx-1;
   	          data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix) 
   	          {
      		     col_nums[cnt] =hypre_map27(ix-1,iy-1,iz,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	       }
   	       col_nums[cnt] = new_row_index-nx;
   	       data[cnt++] = value[1];
   	       if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
   	       {
   	          col_nums[cnt] = new_row_index-nx+1;
   	          data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix+1 < nx_global) 
   	          {
      		     col_nums[cnt] =hypre_map27(ix+1,iy-1,iz,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	       }
            }
            else
            {
               if (iy) 
               {
   	          if (ix > (HYPRE_BigInt)(nx*px))
   	          {
      		     col_nums[cnt] =hypre_map27(ix-1,iy-1,iz,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	          else if (ix)
   	          {
      		     col_nums[cnt] =hypre_map27(ix-1,iy-1,iz,px-1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
      		  col_nums[cnt] =hypre_map27(ix,iy-1,iz,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
   	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
   	          {
      		     col_nums[cnt] =hypre_map27(ix+1,iy-1,iz,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	          else if (ix < nx_global-1)
   	          {
      		     col_nums[cnt] =hypre_map27(ix+1,iy-1,iz,px+1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
               }
            }
            if (ix > (HYPRE_BigInt)(nx*px))
            {
               col_nums[cnt] = new_row_index-1;
               data[cnt++] = value[1];
            }
            else
            {
               if (ix) 
               {
      		  col_nums[cnt] =hypre_map27(ix-1,iy,iz,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
               }
            }
            if (ix+1 < (HYPRE_BigInt)(nx*(px+1)))
            {
               col_nums[cnt] = new_row_index+1;
               data[cnt++] = value[1];
            }
            else
            {
               if (ix+1 < nx_global) 
               {
      		  col_nums[cnt] =hypre_map27(ix+1,iy,iz,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
               }
            }
            if (iy+1 < (HYPRE_BigInt)(ny*(py+1)))
            {
   	       if (ix > (HYPRE_BigInt)(nx*px))
   	       {
                  col_nums[cnt] = new_row_index+nx-1;
                  data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix) 
                  {
      		     col_nums[cnt] =hypre_map27(ix-1,iy+1,iz,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
   	       }
               col_nums[cnt] = new_row_index+nx;
               data[cnt++] = value[1];
   	       if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
   	       {
                  col_nums[cnt] = new_row_index+nx+1;
                  data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix+1 < nx_global) 
                  {
      		     col_nums[cnt] =hypre_map27(ix+1,iy+1,iz,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
   	       }
            }
            else
            {
               if (iy+1 < ny_global) 
               {
   	          if (ix > (HYPRE_BigInt)(nx*px))
   	          {
      		     col_nums[cnt] =hypre_map27(ix-1,iy+1,iz,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	          else if (ix)
   	          {
      		     col_nums[cnt] =hypre_map27(ix-1,iy+1,iz,px-1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
      		  col_nums[cnt] =hypre_map27(ix,iy+1,iz,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
   	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
   	          {
      		     col_nums[cnt] =hypre_map27(ix+1,iy+1,iz,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	          else if (ix < nx_global-1)
   	          {
      		     col_nums[cnt] =hypre_map27(ix+1,iy+1,iz,px+1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
               }
            }
            if (iz+1 < (HYPRE_BigInt)(nz*(pz+1)))
            {
               if (iy > (HYPRE_BigInt)(ny*py))
               {
      	          if (ix > (HYPRE_BigInt)(nx*px))
      	          {
      	             col_nums[cnt] = new_row_index+nxy-nx-1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
   	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
   	             }
      	          }
      	          col_nums[cnt] = new_row_index+nxy-nx;
      	          data[cnt++] = value[1];
      	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	          {
      	             col_nums[cnt] = new_row_index+nxy-nx+1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
   	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
   	             }
      	          }
               }
               else
               {
                  if (iy) 
                  {
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px-1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      		     col_nums[cnt] =hypre_map27(ix,iy-1,iz+1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix < nx_global-1)
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px+1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
                  }
               }
               if (ix > (HYPRE_BigInt)(nx*px))
               {
                  col_nums[cnt] = new_row_index+nxy-1;
                  data[cnt++] = value[1];
               }
               else
               {
                  if (ix) 
                  {
      		     col_nums[cnt] =hypre_map27(ix-1,iy,iz+1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
               }
               col_nums[cnt] = new_row_index+nxy;
               data[cnt++] = value[1];
               if (ix+1 < (HYPRE_BigInt)(nx*(px+1)))
               {
                  col_nums[cnt] = new_row_index+nxy+1;
                  data[cnt++] = value[1];
               }
               else
               {
                  if (ix+1 < nx_global) 
                  {
      		     col_nums[cnt] =hypre_map27(ix+1,iy,iz+1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
               }
               if (iy+1 < (HYPRE_BigInt)(ny*(py+1)))
               {
      	          if (ix > (HYPRE_BigInt)(nx*px))
      	          {
                     col_nums[cnt] = new_row_index+nxy+nx-1;
                     data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
                     {
      		        col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
      	          }
                  col_nums[cnt] = new_row_index+nxy+nx;
                  data[cnt++] = value[1];
      	          if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	          {
                     col_nums[cnt] = new_row_index+nxy+nx+1;
                     data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
                     {
      		        col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
      	          }
               }
               else
               {
                  if (iy+1 < ny_global) 
                  {
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px-1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      		     col_nums[cnt] =hypre_map27(ix,iy+1,iz+1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix < nx_global-1)
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px+1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
                  }
               }
            }
            else
            {
               if (iz+1 < nz_global)
	       {
                  if (iy > (HYPRE_BigInt)(ny*py))
                  {
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px-1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
      		     col_nums[cnt] =hypre_map27(ix,iy-1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px+1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
      	                if (ix > (HYPRE_BigInt)(nx*px))
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px,py-1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px-1,py-1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      		        col_nums[cnt] =hypre_map27(ix,iy-1,iz+1,px,py-1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	                if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px,py-1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix < nx_global-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px+1,py-1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
                     }
                  }
                  if (ix > (HYPRE_BigInt)(nx*px))
                  {
      		     col_nums[cnt] =hypre_map27(ix-1,iy,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix) 
                     {
      		        col_nums[cnt] =hypre_map27(ix-1,iy,iz+1,px-1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
                  }
      		  col_nums[cnt] =hypre_map27(ix,iy,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
                  if (ix+1 < (HYPRE_BigInt)(nx*(px+1)))
                  {
      		     col_nums[cnt] =hypre_map27(ix+1,iy,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix+1 < nx_global) 
                     {
      		        col_nums[cnt] =hypre_map27(ix+1,iy,iz+1,px+1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
                  }
                  if (iy+1 < (HYPRE_BigInt)(ny*(py+1)))
                  {
      	             if (ix > (HYPRE_BigInt)(nx*px))
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px-1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
      		     col_nums[cnt] =hypre_map27(ix,iy+1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px+1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny_global) 
                     {
      	                if (ix > (HYPRE_BigInt)(nx*px))
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px,py+1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px-1,py+1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      		        col_nums[cnt] =hypre_map27(ix,iy+1,iz+1,px,py+1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	                if (ix < (HYPRE_BigInt)(nx*(px+1)-1))
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px,py+1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix < nx_global-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px+1,py+1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
                     }
                  }
               }
            }
            new_row_index++;
         }
      }
    }
   } /*end parallel loop */

   HYPRE_IJMatrixInitialize(ij_A);

   HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   HYPRE_IJMatrixSetValues(ij_A, local_size, num_cols, row_nums,
			col_nums, data);

   HYPRE_IJMatrixAssemble(ij_A);

   hypre_TFree(diag_i, memory_location);
   hypre_TFree(offd_i, memory_location);
   hypre_TFree(num_cols, memory_location);
   hypre_TFree(col_nums, memory_location);
   hypre_TFree(row_nums, memory_location);
   hypre_TFree(data, memory_location);
   hypre_TFree(value, memory_location);
   hypre_TFree(nnz, memory_location);

   *system_size_ptr = global_size;
   *ij_A_ptr = ij_A;

   return (0);
}

HYPRE_BigInt
hypre_map27( HYPRE_BigInt  ix,
      HYPRE_BigInt  iy,
      HYPRE_BigInt  iz,
      HYPRE_Int  px,
      HYPRE_Int  py,
      HYPRE_Int  pz,
      HYPRE_BigInt  Cx,
      HYPRE_BigInt  Cy,
      HYPRE_BigInt  Cz,
      HYPRE_Int nx,
      HYPRE_Int nxy)
{
   HYPRE_BigInt global_index = pz*Cz + py*Cy +px*Cx + iz*nxy + iy*nx + ix;

   return global_index;
}

