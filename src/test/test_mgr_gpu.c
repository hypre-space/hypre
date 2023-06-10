/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE_krylov.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif
#define SECOND_TIME 0




hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int                 arg_index;
   HYPRE_Int                 print_usage;
   HYPRE_Int                 build_matrix_type;
   HYPRE_Int                 build_matrix_arg_index;
   HYPRE_Int                 build_rhs_type;
   HYPRE_Int                 build_rhs_arg_index;
   HYPRE_Int                 use_block_cf = 0;
   HYPRE_Int                 use_point_marker_array = 0;
   HYPRE_Int                 use_reserved_coarse_grid;
   HYPRE_Int                 build_block_cf_arg_index;
   HYPRE_Int                 build_marker_array_arg_index;
   HYPRE_Int                 solver_id;
   HYPRE_Int                 print_system = 0;
   HYPRE_Int                 ierr = 0;
   HYPRE_Int                 i;
   HYPRE_Int                 num_iterations;
   HYPRE_Real                final_res_norm;
   void                      *object;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int spgemm_use_vendor = 1;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_HOST;
#endif
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;

   /* CUB Allocator */
   hypre_uint mempool_bin_growth   = 8,
              mempool_min_bin      = 3,
              mempool_max_bin      = 9;
   size_t mempool_max_cached_bytes = 2000LL * 1024 * 1024;

   HYPRE_IJMatrix      ij_A = NULL;
   //HYPRE_IJMatrix      ij_M = NULL;
   HYPRE_IJVector      ij_b = NULL;
   HYPRE_IJVector      ij_x = NULL;

   HYPRE_ParCSRMatrix  parcsr_A = NULL;
   //HYPRE_ParCSRMatrix  parcsr_M = NULL;
   HYPRE_ParVector     b = NULL;
   HYPRE_ParVector     x = NULL;

   HYPRE_Solver        amg_solver = NULL;
   HYPRE_Solver        pcg_solver = NULL;
   HYPRE_Solver        pcg_precond = NULL, pcg_precond_gotten = NULL;

   HYPRE_Int           num_procs, myid;

   HYPRE_Int           time_index;
   MPI_Comm            comm = hypre_MPI_COMM_WORLD;

   HYPRE_BigInt first_local_row, last_local_row, local_num_rows;
   HYPRE_BigInt first_local_col, last_local_col, local_num_cols;
#ifdef HAVE_DSUPERLU
   HYPRE_Int    dslu_threshold = -1;
#endif

   /* parameters for GMRES */
   HYPRE_Int     k_dim = 100;
   HYPRE_Real    tol = 1e-6;
   HYPRE_Real    atol = 1e-14;
   HYPRE_Real    pc_tol = 0.0;
   HYPRE_Int     max_iter = 1;

   /* mgr options */
   HYPRE_Int mgr_bsize = 3;
   HYPRE_Int mgr_nlevels = 2;
   HYPRE_Int mgr_num_reserved_nodes = 0;
   HYPRE_Int mgr_non_c_to_f = 1;
   HYPRE_Int P_max_elmts = 0;

   HYPRE_BigInt  *mgr_idx_array = NULL;
   HYPRE_Int     *mgr_point_marker_array = NULL;
   HYPRE_Int     *mgr_num_cindexes = NULL;
   HYPRE_Int     **mgr_cindexes = NULL;
   HYPRE_BigInt  *mgr_reserved_coarse_indexes = NULL;

   HYPRE_Int mgr_relax_type = 18;
   HYPRE_Int mgr_num_relax_sweeps = 1;

   HYPRE_Int mgr_gsmooth_type = 0;
   HYPRE_Int mgr_num_gsmooth_sweeps = 0;

   HYPRE_Int mgr_restrict_type = 0;
   HYPRE_Int mgr_num_restrict_sweeps = 0;
   //HYPRE_Int *mgr_level_restrict_type = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
   //mgr_level_restrict_type[0] = 0;
   //mgr_level_restrict_type[1] = 0;

   HYPRE_Int mgr_interp_type = 2;
   HYPRE_Int mgr_num_interp_sweeps = 0;
   //HYPRE_Int *mgr_level_interp_type = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
   //mgr_level_interp_type[0] = 2;
   //mgr_level_interp_type[1] = 2;

   HYPRE_Int *mgr_coarse_grid_method = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
   mgr_coarse_grid_method[0] = 0;
   mgr_coarse_grid_method[1] = 0;

   mgr_cindexes = hypre_CTAlloc(HYPRE_Int*, mgr_nlevels, HYPRE_MEMORY_HOST);
   HYPRE_Int *lv1 = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
   HYPRE_Int *lv2 = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
   lv1[0] = 0;
   lv1[1] = 1;
   lv2[0] = 0;
   mgr_cindexes[0] = lv1;
   mgr_cindexes[1] = lv2;

   mgr_num_cindexes = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
   mgr_num_cindexes[0] = 2;
   mgr_num_cindexes[1] = 1;

   HYPRE_Int mgr_frelax_method = 0;
   //HYPRE_Int *mgr_level_frelax_method = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
   //mgr_level_frelax_method[0] = 2;
   //mgr_level_frelax_method[1] = 0;

   //HYPRE_Int *mgr_frelax_num_functions = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
   //mgr_frelax_num_functions[0] = 3;

   char* indexList = NULL;
   /* end mgr options */

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   build_matrix_type = -1;
   build_matrix_arg_index = argc;
   build_rhs_type = 2;
   build_rhs_arg_index = argc;

   solver_id = 72;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = -1;
         build_matrix_arg_index = arg_index;
      }
      /*
      if ( strcmp(argv[arg_index], "-precondfromfile") == 0 )
      {
        arg_index++;
        build_precond_type      = -1;
        build_precond_arg_index = arg_index;
      }
      */
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 0;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-indexList") == 0)
      {
         arg_index++;
         use_reserved_coarse_grid = 1;
         indexList = (argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-blockCF") == 0)
      {
         arg_index++;
         use_block_cf = 1;
         build_block_cf_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-markerArray") == 0)
      {
         arg_index++;
         use_point_marker_array = 1;
         build_marker_array_arg_index = arg_index;
      }
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      else if ( strcmp(argv[arg_index], "-exec_host") == 0 )
      {
         arg_index++;
         default_exec_policy = HYPRE_EXEC_HOST;
      }
      else if ( strcmp(argv[arg_index], "-exec_device") == 0 )
      {
         arg_index++;
         default_exec_policy = HYPRE_EXEC_DEVICE;
      }
      else if ( strcmp(argv[arg_index], "-mm_vendor") == 0 )
      {
         arg_index++;
         spgemm_use_vendor = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-memory_host") == 0 )
      {
         arg_index++;
         memory_location = HYPRE_MEMORY_HOST;
      }
#endif
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
   * Print driver parameters
   *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("Running with these driver parameters:\n");
      hypre_printf("  solver ID    = %d\n\n", solver_id);
   }

   time_index = hypre_InitializeTiming("Hypre init");
   hypre_BeginTiming(time_index);

   /*-----------------------------------------------------------
   * Initialize : must be the first HYPRE function to call
   *-----------------------------------------------------------*/
   HYPRE_Initialize();

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Hypre init times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /* To be effective, hypre_SetCubMemPoolSize must immediately follow HYPRE_Init */
   hypre_SetCubMemPoolSize( mempool_bin_growth, mempool_min_bin,
                            mempool_max_bin, mempool_max_cached_bytes );

   HYPRE_SetMemoryLocation(memory_location);

   HYPRE_SetExecutionPolicy(default_exec_policy);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   ierr = HYPRE_SetSpGemmUseVendor(spgemm_use_vendor); hypre_assert(ierr == 0);
#endif


   /*-----------------------------------------------------------
   * Setup the matrix
   *-----------------------------------------------------------*/
   if (myid == 0)
   {
      hypre_printf("Reading the system matrix\n");
   }
   time_index = hypre_InitializeTiming("Reading Input Matrix");
   hypre_BeginTiming(time_index);
   ierr = HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                              HYPRE_PARCSR, &ij_A );
   if (ierr)
   {
      hypre_printf("ERROR: Problem reading in the system matrix!\n");
      exit(1);
   }
   else
   {
      hypre_printf("Done reading the system matrix\n");
   }
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Reading Input Matrix", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   if (build_matrix_type < 0)
   {
      ierr = HYPRE_IJMatrixGetLocalRange( ij_A,
                                          &first_local_row, &last_local_row,
                                          &first_local_col, &last_local_col );

      local_num_rows = last_local_row - first_local_row + 1;
      local_num_cols = last_local_col - first_local_col + 1;
      ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_A = (HYPRE_ParCSRMatrix) object;
   }

   if ( build_rhs_type == 0 )
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
         hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      ierr = HYPRE_IJVectorRead( argv[build_rhs_arg_index], hypre_MPI_COMM_WORLD,
                                 HYPRE_PARCSR, &ij_b );
      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in the right-hand-side!\n");
         exit(1);
      }
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorAssemble(ij_x);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 2 )
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector has unit components\n");
         hypre_printf("  Initial guess is 0\n");
      }

      HYPRE_Real *values_h = hypre_CTAlloc(HYPRE_Real, local_num_rows, HYPRE_MEMORY_HOST);
      HYPRE_Real *values_d = hypre_CTAlloc(HYPRE_Real, local_num_rows, memory_location);
      for (i = 0; i < local_num_rows; i++)
      {
         values_h[i] = 1.0;
      }
      hypre_TMemcpy(values_d, values_h, HYPRE_Real, local_num_rows, memory_location, HYPRE_MEMORY_HOST);

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_v2(ij_b, memory_location);
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values_d);
      HYPRE_IJVectorAssemble(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      hypre_Memset(values_d, 0, local_num_rows * sizeof(HYPRE_Real), HYPRE_MEMORY_DEVICE);
      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_v2(ij_x, memory_location);
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      HYPRE_IJVectorAssemble(ij_x);
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
      hypre_TFree(values_d, memory_location);
   }

   if (indexList != NULL)
   {
      mgr_reserved_coarse_indexes = hypre_CTAlloc(HYPRE_BigInt, mgr_num_reserved_nodes,
                                                  HYPRE_MEMORY_HOST);
      FILE* ifp;
      ifp = fopen(indexList, "r");
      if (ifp == NULL)
      {
         fprintf(stderr, "Can't open input file for index list!\n");
         exit(1);
      }
      fscanf(ifp, "%d", &mgr_num_reserved_nodes);
      fprintf(stderr, "There are %d additional indices\n", mgr_num_reserved_nodes);
      for (i = 0; i < mgr_num_reserved_nodes; i++)
      {
         fscanf(ifp, "%d", &mgr_reserved_coarse_indexes[i]);
      }
   }
   else
   {
      mgr_num_reserved_nodes = 0;
      mgr_reserved_coarse_indexes = NULL;
   }

   if (use_block_cf)
   {
      mgr_idx_array = hypre_CTAlloc(HYPRE_BigInt, mgr_bsize, HYPRE_MEMORY_HOST);
      FILE *ifp;
      char fname[80];
      hypre_sprintf(fname, "%s.%05i", argv[build_block_cf_arg_index], myid);
      hypre_printf("Reading block CF indices from %s \n", fname);
      ifp = fopen(fname, "r");
      if (ifp == NULL)
      {
         fprintf(stderr, "Can't open input file for block CF indices!\n");
         exit(1);
      }
      for (i = 0; i < mgr_bsize; i++)
      {
         fscanf(ifp, "%d", &mgr_idx_array[i]);
      }
   }

   mgr_point_marker_array = hypre_CTAlloc(HYPRE_Int, local_num_rows, HYPRE_MEMORY_HOST);
   if (use_point_marker_array)
   {
      FILE *ifp;
      char fname[80];
      hypre_sprintf(fname, "%s.%05i", argv[build_marker_array_arg_index], myid);
      hypre_printf("Reading marker array from %s \n", fname);
      ifp = fopen(fname, "r");
      if (ifp == NULL)
      {
         fprintf(stderr, "Can't open input file for block CF indices!\n");
         exit(1);
      }
      for (i = 0; i < local_num_rows; i++)
      {
         fscanf(ifp, "%d", &mgr_point_marker_array[i]);
      }
   }

   /*-----------------------------------------------------------
   * Migrate the system to the wanted memory space
   *-----------------------------------------------------------*/
   hypre_ParCSRMatrixMigrate(parcsr_A, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(b, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(x, hypre_HandleMemoryLocation(hypre_handle()));

   if (solver_id == 72)
   {
      // Initialize main solver
      HYPRE_ParCSRFlexGMRESCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_FlexGMRESSetKDim(pcg_solver, k_dim);
      HYPRE_FlexGMRESSetMaxIter(pcg_solver, max_iter);
      //HYPRE_FlexGMRESSetMaxIter(pcg_solver, 0);
      HYPRE_FlexGMRESSetTol(pcg_solver, tol);
      HYPRE_FlexGMRESSetAbsoluteTol(pcg_solver, atol);
      HYPRE_FlexGMRESSetLogging(pcg_solver, 1);
      HYPRE_FlexGMRESSetPrintLevel(pcg_solver, 2);

      /* use MGR preconditioning */
      if (myid == 0) { hypre_printf("Solver:  MGR-FlexGMRES\n"); }

      HYPRE_MGRCreate(&pcg_precond);

      /* set MGR data by block */
      if (use_point_marker_array)
      {
         HYPRE_MGRSetCpointsByPointMarkerArray( pcg_precond, mgr_bsize, mgr_nlevels, mgr_num_cindexes,
                                                mgr_cindexes, mgr_point_marker_array);
      }
      else
      {
         HYPRE_MGRSetCpointsByBlock( pcg_precond, mgr_bsize, mgr_nlevels, mgr_num_cindexes, mgr_cindexes);
      }
      /* set reserved coarse nodes */
      if (mgr_num_reserved_nodes) { HYPRE_MGRSetReservedCoarseNodes(pcg_precond, mgr_num_reserved_nodes, mgr_reserved_coarse_indexes); }

      /* set intermediate coarse grid strategy */
      HYPRE_MGRSetNonCpointsToFpoints(pcg_precond, mgr_non_c_to_f);
      /* set F relaxation strategy */
      HYPRE_MGRSetFRelaxMethod(pcg_precond, mgr_frelax_method);
      //HYPRE_MGRSetLevelFRelaxNumFunctions(pcg_precond, mgr_frelax_num_functions);
      /* set relax type for single level F-relaxation and post-relaxation */
      HYPRE_MGRSetRelaxType(pcg_precond, mgr_relax_type);
      HYPRE_MGRSetNumRelaxSweeps(pcg_precond, mgr_num_relax_sweeps);
      /* set interpolation type */
      HYPRE_MGRSetInterpType(pcg_precond, mgr_interp_type);
      HYPRE_MGRSetNumInterpSweeps(pcg_precond, mgr_num_interp_sweeps);
      /* set restriction type */
      HYPRE_MGRSetRestrictType(pcg_precond, mgr_restrict_type);
      HYPRE_MGRSetNumRestrictSweeps(pcg_precond, mgr_num_restrict_sweeps);
      /* set coarse grid method */
      HYPRE_MGRSetCoarseGridMethod(pcg_precond, mgr_coarse_grid_method);
      /* set print level */
      HYPRE_MGRSetPrintLevel(pcg_precond, 1);
      /* set max iterations */
      HYPRE_MGRSetMaxIter(pcg_precond, 1);
      HYPRE_MGRSetTol(pcg_precond, pc_tol);
      HYPRE_MGRSetTruncateCoarseGridThreshold(pcg_precond, 1e-20);

      HYPRE_MGRSetGlobalsmoothType(pcg_precond, mgr_gsmooth_type);
      HYPRE_MGRSetMaxGlobalsmoothIters( pcg_precond, mgr_num_gsmooth_sweeps );
      //hypre_MGRPrintCoarseSystem( pcg_precond, 1 );

      HYPRE_BoomerAMGCreate(&amg_solver);
      HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);
      //HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);
      HYPRE_BoomerAMGSetCoarsenType(amg_solver, 8);
      //HYPRE_BoomerAMGSetInterpType(amg_solver, 6);
      HYPRE_BoomerAMGSetNumFunctions(amg_solver, 1);
      HYPRE_BoomerAMGSetRelaxType(amg_solver, 18);
      HYPRE_BoomerAMGSetNumSweeps(amg_solver, 2);
      HYPRE_BoomerAMGSetMaxIter(amg_solver, 1);
      HYPRE_BoomerAMGSetTol(amg_solver, 0.0);
      HYPRE_BoomerAMGSetMaxRowSum(amg_solver, 1.0);

      /* set the MGR coarse solver. Comment out to use default CG solver in MGR */
      HYPRE_MGRSetCoarseSolver( pcg_precond, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, amg_solver);

      /* setup MGR-PCG solver */
      HYPRE_FlexGMRESSetPrecond(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MGRSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_MGRSetup,
                                pcg_precond);

      HYPRE_FlexGMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         hypre_printf("HYPRE_FlexGMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else
      {
         if (myid == 0)
         {
            hypre_printf("HYPRE_FlexGMRESGetPrecond got good precond\n");
         }
      }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      hypre_SyncCudaDevice(hypre_handle());
#endif

      // Setup main solver
      time_index = hypre_InitializeTiming("FlexGMRES Setup");
      hypre_BeginTiming(time_index);
      HYPRE_FlexGMRESSetup
      (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      //HYPRE_MGRSetup(pcg_precond, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      hypre_SyncCudaDevice(hypre_handle());
#endif

      time_index = hypre_InitializeTiming("FlexGMRES Solve");
      hypre_BeginTiming(time_index);

      //hypre_ParVectorSetConstantValues(x, 0.0);
      HYPRE_FlexGMRESSolve
      (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      //HYPRE_MGRSolve(pcg_precond, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      hypre_SyncCudaDevice(hypre_handle());
#endif

      if (print_system)
      {
         HYPRE_IJVectorPrint(ij_x, "x.out");
      }
      HYPRE_FlexGMRESGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      time_index = hypre_InitializeTiming("FlexGMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_FlexGMRESSetup(pcg_solver, (HYPRE_Matrix)parcsr_A,
                           (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("FlexGMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_FlexGMRESSolve(pcg_solver, (HYPRE_Matrix)parcsr_A,
                           (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
#endif

      // free memory for flex FlexGMRES
      if (pcg_solver) { HYPRE_ParCSRFlexGMRESDestroy(pcg_solver); }
      if (amg_solver) { HYPRE_BoomerAMGDestroy(amg_solver); }
      if (pcg_precond) { HYPRE_MGRDestroy(pcg_precond); }

      // Print out solver summary
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("FlexGMRES Iterations = %d\n", num_iterations);
         hypre_printf("Final FlexGMRES Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }
   }
   else if (solver_id == 98)
   {
      // Test building MGR interpolation on device
      comm = hypre_ParCSRMatrixComm(parcsr_A);
      hypre_ParCSRMatrix *P = NULL;
      hypre_IntArray *dof_func_buff = NULL;
      HYPRE_BigInt *coarse_pnts_global = NULL;
      HYPRE_Int nloc =  hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(parcsr_A));
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(parcsr_A) );

      hypre_IntArray *CF_marker = hypre_IntArrayCreate(nloc);
      hypre_IntArrayInitialize_v2(CF_marker, HYPRE_MEMORY_HOST);

      for (i = 0; i < nloc; i++)
      {
         if (i % 3 == 0)
         {
            hypre_IntArrayData(CF_marker)[i] = 1;
         }
         else
         {
            hypre_IntArrayData(CF_marker)[i] = -1;
         }
      }

      hypre_BoomerAMGCoarseParms(comm, nloc, 1, NULL, CF_marker, &dof_func_buff, &coarse_pnts_global);
      if (exec == HYPRE_EXEC_HOST)
      {
         hypre_MGRBuildP(parcsr_A, hypre_IntArrayData(CF_marker), coarse_pnts_global, 2, 0, &P);
         hypre_ParCSRMatrixPrintIJ(P, 0, 0, "P_host");
      }
#if defined(HYPRE_USING_CUDA)
      else
      {
         hypre_MGRBuildPDevice(parcsr_A, hypre_IntArrayData(CF_marker), coarse_pnts_global, 2, &P);
         hypre_ParCSRMatrixPrintIJ(P, 0, 0, "P_device");
      }
#endif
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/
   // free the matrix, the rhs and the initial guess
   HYPRE_IJMatrixDestroy(ij_A);
   HYPRE_IJVectorDestroy(ij_b);
   HYPRE_IJVectorDestroy(ij_x);

   hypre_TFree(mgr_num_cindexes, HYPRE_MEMORY_HOST);
   //hypre_TFree(mgr_level_frelax_method, HYPRE_MEMORY_HOST);
   //hypre_TFree(mgr_frelax_num_functions, HYPRE_MEMORY_HOST);
   hypre_TFree(mgr_idx_array, HYPRE_MEMORY_HOST);
   hypre_TFree(mgr_point_marker_array, HYPRE_MEMORY_HOST);
   hypre_TFree(mgr_coarse_grid_method, HYPRE_MEMORY_HOST);
   hypre_TFree(lv1, HYPRE_MEMORY_HOST);
   hypre_TFree(lv2, HYPRE_MEMORY_HOST);
   hypre_TFree(mgr_cindexes, HYPRE_MEMORY_HOST);
   //hypre_TFree(mgr_level_interp_type, HYPRE_MEMORY_HOST);
   //hypre_TFree(mgr_level_restrict_type, HYPRE_MEMORY_HOST);
   if (mgr_num_reserved_nodes > 0) { hypre_TFree(mgr_reserved_coarse_indexes, HYPRE_MEMORY_HOST); }

   HYPRE_Finalize();
   hypre_MPI_Finalize();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_ResetCudaDevice(hypre_handle());
#endif

   return (0);
}
