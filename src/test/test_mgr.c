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
#include "_hypre_utilities.hpp"
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


HYPRE_Int BuildParFromFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                            HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int ReadParVectorFromFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                 HYPRE_ParVector *b_ptr );


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
   HYPRE_Int                 use_point_marker_array = 0;
   HYPRE_Int                 build_marker_array_arg_index;
   HYPRE_Int                 solver_id;
   HYPRE_Int                 example_id;
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
   HYPRE_IJVector      ij_b = NULL;
   HYPRE_IJVector      ij_x = NULL;

   HYPRE_ParCSRMatrix  parcsr_A = NULL;
   HYPRE_ParVector     b = NULL;
   HYPRE_ParVector     x = NULL;

   HYPRE_Solver        amg_solver = NULL;
   HYPRE_Solver        krylov_solver = NULL;
   HYPRE_Solver        krylov_precond = NULL, krylov_precond_gotten = NULL;

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
   HYPRE_Real    tol = 1e-8;
   HYPRE_Real    atol = 1e-14;
   HYPRE_Real    pc_tol = 0.0;
   HYPRE_Int     max_iter = 100;

   /* parameters for MGR */
   HYPRE_Int mgr_bsize = 4;  // block size of the system
   HYPRE_Int mgr_nlevels = 3;  // number of reduction levels (3-level method means 2-level reduction)
   HYPRE_Int mgr_non_c_to_f = 1;  // option to use user-provided reduction strategy

   HYPRE_Int     *mgr_point_marker_array = NULL;
   HYPRE_Int     *mgr_num_cindexes = NULL;
   HYPRE_Int     **mgr_cindexes = NULL;
   HYPRE_Int     *lv1 = NULL, *lv2 = NULL, *lv3 = NULL;

   /* F-relaxation option
    * 0  - Jacobi (only CPU)
    * 18 - L1 Jacobi (GPU-enabled)
    */
   HYPRE_Int mgr_relax_type = 18;
   HYPRE_Int mgr_num_relax_sweeps = 1;  // number of F-relax iterations, 0 for traditional CPR

   /* Global smoother option
    *  0  - (block) Jacobi (only CPU)
    *  16 - ILU(0) (GPU-enabled)
    */
   HYPRE_Int mgr_gsmooth_type = 16;
   HYPRE_Int mgr_num_gsmooth_sweeps = 0;  // number of global smoothing steps

   /* Interpolation/Restriction option
    * 0 - Injection
    * 2 - Jacobi diagonal scaling
    */
   HYPRE_Int mgr_restrict_type = 0;
   HYPRE_Int mgr_interp_type = 2;


   HYPRE_Int  *lvl_cg_method = NULL;
   HYPRE_Int  *lvl_interp_type = NULL;
   HYPRE_Int  *lvl_gsmooth_type = NULL;
   HYPRE_Int  *lvl_gsmooth_iters = NULL;

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
   example_id = 0;

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
      else if ( strcmp(argv[arg_index], "-fromparcsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsparcsrfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 1;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 0;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-markerArray") == 0)
      {
         arg_index++;
         use_point_marker_array = 1;
         build_marker_array_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-gmres") == 0 )
      {
         arg_index++;
         solver_id = 72;
      }
      else if ( strcmp(argv[arg_index], "-fgmres") == 0 )
      {
         arg_index++;
         solver_id = 73;
      }
      else if ( strcmp(argv[arg_index], "-bicgstab") == 0 )
      {
         arg_index++;
         solver_id = 74;
      }
      /*
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
        arg_index++;
        solver_id = atoi(argv[arg_index++]);
      }
      */
      else if ( strcmp(argv[arg_index], "-example") == 0 )
      {
         arg_index++;
         example_id = atoi(argv[arg_index++]);
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

   /* To be effective, hypre_SetCubMemPoolSize must immediately follow HYPRE_Initialize */
   hypre_SetCubMemPoolSize( mempool_bin_growth, mempool_min_bin,
                            mempool_max_bin, mempool_max_cached_bytes );

   hypre_HandleMemoryLocation(hypre_handle())    = memory_location;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_HandleDefaultExecPolicy(hypre_handle()) = default_exec_policy;
   hypre_HandleSpgemmUseVendor(hypre_handle()) = spgemm_use_vendor;
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
   if (build_matrix_type == -1)
   {
      ierr = HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                                 HYPRE_PARCSR, &ij_A );
   }
   else if ( build_matrix_type == 0 )
   {
      ierr = BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   if (ierr)
   {
      hypre_printf("ERROR: Problem reading in the system matrix!\n");
      exit(1);
   }
   else
   {
      if (myid == 0) { hypre_printf("Done reading the system matrix\n"); }
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
   else
   {
      /*-----------------------------------------------------------
       * Copy the parcsr matrix into the IJMatrix through interface calls
       *-----------------------------------------------------------*/
      ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                              &first_local_row, &last_local_row,
                                              &first_local_col, &last_local_col );

      local_num_rows = (HYPRE_Int)(last_local_row - first_local_row + 1);
      local_num_cols = (HYPRE_Int)(last_local_col - first_local_col + 1);
   }
   // build rhs from file in IJ format
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
   else if (build_rhs_type == 1) // build rhs from parcsr file
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
         hypre_printf("  Initial guess is 0\n");
      }

      ij_b = NULL;
      ReadParVectorFromFile(argc, argv, build_rhs_arg_index, &b);

      /* initial guess */
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

   if (use_point_marker_array)
   {
      mgr_point_marker_array = hypre_CTAlloc(HYPRE_Int, local_num_rows, HYPRE_MEMORY_HOST);
      FILE *ifp;
      char fname[80];
      hypre_sprintf(fname, "%s.%d", argv[build_marker_array_arg_index], myid);
      ifp = fopen(fname, "r");
      if (ifp == NULL)
      {
         fprintf(stderr, "Can't open input file for marker array!\n");
         exit(1);
      }
      else
      {
         if (myid == 0) { hypre_printf("  Cpoint marker array read from file %s\n", argv[build_rhs_arg_index]); }
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

   if (example_id == 0)
   {
      /*
       * Example of a 2x2 block system
       * E.g. incompressible two-phase flow with TPFA
       *
       * A = [ A_pp  A_ps
       *       A_sp  A_ss ]
       * where p, s are cell-centered pressure and saturation
       *
       * Two-level MGR reduction strategy that mimics a variant of CPR
       *   - Global relaxation with ILU
       *   - 1st level: eliminate the saturation
       *   - 2nd level: solve the coarse grid (Schur-complement pressure system) with BoomerAMG
       */

      if (myid == 0) { hypre_printf("MGR example: 2x2 Block System\n"); }
      /* mgr options */
      mgr_bsize = 2;  // block size of the system
      mgr_nlevels = 1;  // number of reduction levels (2-level method means 1-level reduction)
      mgr_non_c_to_f = 1;  // option to use user-provided reduction strategy

      /* F-relaxation option
       * 0  - Jacobi (only CPU)
       * 18 - L1 Jacobi (GPU-enabled)
       */
      mgr_relax_type = 18;
      mgr_num_relax_sweeps = 0;  // number of F-relax iterations, 0 for traditional CPR

      /* Global smoother option
       * 0  - (block) Jacobi (only CPU)
       * 16 - ILU(0) (GPU-enabled)
       */
      mgr_gsmooth_type = 16;
      mgr_num_gsmooth_sweeps = 1;  // number of global smoothing steps

      /* Interpolation/Restriction option
       * 0 - Injection
       * 2 - Jacobi diagonal scaling
       */
      mgr_restrict_type = 0;
      mgr_interp_type = 2;

      /* array for number of C-points at each level */
      mgr_num_cindexes = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
      mgr_num_cindexes[0] = 1;  // 1 C-point for 1st-level reduction, i.e. pressure

      /* array for indices of C-points at each level */
      mgr_cindexes = hypre_CTAlloc(HYPRE_Int*, mgr_nlevels, HYPRE_MEMORY_HOST);
      lv1 = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
      lv1[0] = 0;  // pressure is a C-point
      mgr_cindexes[0] = lv1;
      /* end mgr options */
   }
   else if (example_id == 1)
   {
      /*
       * Example for 3x3 block system
       * E.g. 2-phase, 2-component flow with TPFA
       *
       * A = [ A_{pp}       A_{p\rho_1}       A_{p\rho_2}
       *       A_{\rho_1p}  A_{\rho_1\rho_1}  A_{\rho_1\rho_2}
       *       A_{\rho_2p}  A_{\rho_2\rho_1}  A_{\rho_2\rho_2} ]
       * where p, \rho_1, \rho_2 are cell-centered pressure and densities
       * of component 1 and 2, respectively
       *
       * Three-level MGR reduction strategy that mimics a variant of CPR
       *   - Global relaxation with ILU
       *   - 1st level: eliminate second component density \rho_2
       *   - 2nd level: eliminate first component density \rho_1
       *   - 3rd level: solve the coarse grid (Schur-complement pressure system) with BoomerAMG
      */

      if (myid == 0) { hypre_printf("MGR example: 3x3 Block System\n"); }
      /* mgr options */
      mgr_bsize = 3;  // block size of the system
      mgr_nlevels = 2;  // number of reduction levels
      mgr_non_c_to_f = 1;  // option to use user-provided reduction strategy

      /* F-relaxation option
       * 0  - Jacobi (only CPU)
       * 18 - L1 Jacobi (GPU-enabled)
       */
      mgr_relax_type = 18;
      mgr_num_relax_sweeps = 0;  // number of F-relax iterations, 0 for traditional CPR

      /* Global smoother option
       *  0  - (block) Jacobi (only CPU)
       *  16 - ILU(0) (GPU-enabled)
       */
      mgr_gsmooth_type = 16;
      mgr_num_gsmooth_sweeps = 1;  // number of global smoothing steps

      /* Interpolation/Restriction option
       * 0 - Injection
       * 2 - Jacobi diagonal scaling
       */
      mgr_restrict_type = 0;
      mgr_interp_type = 2;

      /* array for number of C-points at each level */
      mgr_num_cindexes = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
      mgr_num_cindexes[0] = 2;  // 2 C-points for 1st-level reduction, i.e. p, \rho_1
      mgr_num_cindexes[1] = 1;  // 1 C-point for 2nd-level reduction, i.e. p

      /* array for indices of C-points at each level */
      mgr_cindexes = hypre_CTAlloc(HYPRE_Int*, mgr_nlevels, HYPRE_MEMORY_HOST);
      lv1 = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
      lv2 = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
      lv1[0] = 0;  // pressure is a C-point at level 1
      lv1[1] = 1;  // \rho_1 is also a C-point at level 1
      lv2[0] = 0;  // only pressure is a C-point at level 2
      mgr_cindexes[0] = lv1;
      mgr_cindexes[1] = lv2;
      /* end mgr options */
   }
   else if (example_id == 2)
   {
      /*
       * Example for 4x4 block system
       * E.g. Compositional flow with wells
       *
       * A = [ A_{pp}       A_{p\rho_1}       A_{p\rho_2}
       *       A_{\rho_1p}  A_{\rho_1\rho_1}  A_{\rho_1\rho_2}
       *       A_{\rho_2p}  A_{\rho_2\rho_1}  A_{\rho_2\rho_2} ]
       * where p, \rho_1, \rho_2 are cell-centered pressure and densities
       * of component 1 and 2, respectively
       *
       * Three-level MGR reduction strategy that mimics a variant of CPR
       *   - Global relaxation with ILU
       *   - 1st level: eliminate second component density \rho_2
       *   - 2nd level: eliminate first component density \rho_1
       *   - 3rd level: solve the coarse grid (Schur-complement pressure system) with BoomerAMG
      */

      if (myid == 0) { hypre_printf("MGR example: 4x4 Block System\n"); }
      /* mgr options */
      mgr_bsize = 7;  // block size of the system
      mgr_nlevels = 3;  // number of reduction levels
      mgr_non_c_to_f = 1;  // option to use user-provided reduction strategy

      // Set global smoothing type/iterations at each level
      // Use block-GS for the condensed system

      lvl_cg_method = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
      lvl_interp_type = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
      lvl_gsmooth_type = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
      lvl_gsmooth_iters = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);

      lvl_cg_method[0] = 0; // Standard Galerkin
      lvl_cg_method[1] = 0; // Standard Galerkin
      lvl_cg_method[2] = 3; // Quasi-IMPES reduction

      lvl_interp_type[0] = 12; // Exact well block elimination with block-Jacobi
      lvl_interp_type[1] = 2; // Diagonal scaling (Jacobi)
      lvl_interp_type[2] = 0; // Injection

      lvl_gsmooth_type[1] = 1;
      lvl_gsmooth_iters[1] = 1;

      /* F-relaxation option
       * 0  - Jacobi (only CPU)
       * 18 - L1 Jacobi (GPU-enabled)
       */
      mgr_relax_type = 0;
      mgr_num_relax_sweeps = 1;  // number of F-relax iterations, 0 for traditional CPR

      /* Global smoother option
       *  0  - (block) Jacobi (only CPU)
       *  16 - ILU(0) (GPU-enabled)
       */
      mgr_gsmooth_type = 16;
      mgr_num_gsmooth_sweeps = 0;  // number of global smoothing steps

      /* Interpolation/Restriction option
       * 0 - Injection
       * 2 - Jacobi diagonal scaling
       */
      mgr_restrict_type = 0;
      mgr_interp_type = 12;

      /* array for number of C-points at each level */
      mgr_num_cindexes = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
      mgr_num_cindexes[0] = 3;  // 3 C-points for 1st-level reduction, i.e. p, \rho_1, \rho_2
      mgr_num_cindexes[1] = 2;  // 2 C-point for 2nd-level reduction, i.e. p, \rho_1
      mgr_num_cindexes[2] = 1;  // 1 C-point for 3rd-level reduction, i.e. p

      /* array for indices of C-points at each level */
      mgr_cindexes = hypre_CTAlloc(HYPRE_Int*, mgr_nlevels, HYPRE_MEMORY_HOST);
      lv1 = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
      lv2 = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
      lv3 = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
      lv1[0] = 0;  // pressure is a C-point at level 1
      lv1[1] = 1;  // \rho_1 is also a C-point at level 1
      lv1[2] = 2;  // \rho_2 is also a C-point at level 1
      lv2[0] = 0;  // pressure is a C-point at level 2
      lv2[1] = 1;  // \rho_1 pressure is a C-point at level 2
      lv3[0] = 0;  // only pressure is a C-point at level 3
      mgr_cindexes[0] = lv1;
      mgr_cindexes[1] = lv2;
      mgr_cindexes[2] = lv3;
      /* end mgr options */
   }

   /* use MGR preconditioning */
   HYPRE_MGRCreate(&krylov_precond);

   /* set MGR data for reduction hierarchy */
   if (use_point_marker_array)
   {
      /* set MGR data by point_marker_array
       * This is the recommended way to set MGR data when
       * unknowns have arbitrary order.
       * mgr_point_marker_array is a local array that
       * has the same length as the number of unknowns on this rank
       * and stores the unique point type associated with each physical variable.
       * For example,
       *     solution vector      mgr_point_marker_array
       *           p_1                     0
       *           p_2                     0
       *            .                      .
       *            .                      .
       *            .                      .
       *           p_{N-1}                 0
       *           p_N                     0
       *           \rho_1_1                1
       *           \rho_2_1                2
       *           \rho_1_2                1
       *           \rho_2_2                2
       *            .                      .
       *            .                      .
       *            .                      .
       *           \rho_1_N                1
       *           \rho_2_N                2
       */
      HYPRE_MGRSetCpointsByPointMarkerArray( krylov_precond, mgr_bsize, mgr_nlevels, mgr_num_cindexes,
                                             mgr_cindexes, mgr_point_marker_array);
   }
   else
   {
      /* set MGR data by block
       * This is the way to set MGR data if the unknowns
       * for different physical fields are interleaved/collocated.
       * For example, the solution vector x = [p_1, s_1, p_2, s_2, ..., p_N, s_N]
       */
      HYPRE_MGRSetCpointsByBlock( krylov_precond, mgr_bsize, mgr_nlevels, mgr_num_cindexes, mgr_cindexes);
   }

   /* set intermediate coarse grid strategy */
   HYPRE_MGRSetNonCpointsToFpoints(krylov_precond, mgr_non_c_to_f);
   /* set relax type for single level F-relaxation and post-relaxation */
   HYPRE_MGRSetRelaxType(krylov_precond, mgr_relax_type);
   HYPRE_MGRSetNumRelaxSweeps(krylov_precond, mgr_num_relax_sweeps);
   /* set interpolation type */
   //   HYPRE_MGRSetInterpType(krylov_precond, mgr_interp_type);
   HYPRE_MGRSetLevelInterpType(krylov_precond, lvl_interp_type);
   /* set restriction type */
   HYPRE_MGRSetRestrictType(krylov_precond, mgr_restrict_type);
   /* set print level */
   HYPRE_MGRSetPrintLevel(krylov_precond, 1);
   /* set max iterations */
   HYPRE_MGRSetMaxIter(krylov_precond, 1);
   HYPRE_MGRSetTol(krylov_precond, pc_tol);
   HYPRE_MGRSetTruncateCoarseGridThreshold(krylov_precond, 1e-20);

   /* set global smoother options */
   HYPRE_MGRSetLevelSmoothType( krylov_precond, lvl_gsmooth_type );
   HYPRE_MGRSetLevelSmoothIters( krylov_precond, lvl_gsmooth_iters );
   //   HYPRE_MGRSetGlobalsmoothType(krylov_precond, mgr_gsmooth_type);
   //   HYPRE_MGRSetMaxGlobalsmoothIters( krylov_precond, mgr_num_gsmooth_sweeps );

   HYPRE_MGRSetCoarseGridMethod( krylov_precond, lvl_cg_method );
   HYPRE_MGRSetTruncateCoarseGridThreshold( krylov_precond,
                                            1e-20 ); // Low tolerance to remove only zeros

   /* Create MGR coarse solver */
   HYPRE_BoomerAMGCreate(&amg_solver);
   HYPRE_BoomerAMGSetMaxIter(amg_solver, 1);
   HYPRE_BoomerAMGSetTol(amg_solver, 0.0);
   HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);
   HYPRE_BoomerAMGSetNumFunctions(amg_solver, 1);
   HYPRE_BoomerAMGSetMaxRowSum(amg_solver, 0.9);
#if defined(HYPPRE_USING_CUDA)
   HYPRE_BoomerAMGSetCoarsenType(amg_solver, 8);
   HYPRE_BoomerAMGSetRelaxType(amg_solver, 18);
   HYPRE_BoomerAMGSetNumSweeps(amg_solver, 2);
#else
   HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);
#endif

   /* set the MGR coarse solver. Comment out to use default CG solver (with BoomerAMG) in MGR */
   HYPRE_MGRSetCoarseSolver( krylov_precond, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, amg_solver);


   /* Create Krylov solver */
   if (solver_id == 72)
   {
      if (myid == 0) { hypre_printf("Solver:  MGR-GMRES\n"); }

      // Initialize main solver
      HYPRE_ParCSRGMRESCreate(hypre_MPI_COMM_WORLD, &krylov_solver);
      HYPRE_GMRESSetKDim(krylov_solver, k_dim);
      HYPRE_GMRESSetMaxIter(krylov_solver, max_iter);
      HYPRE_GMRESSetTol(krylov_solver, tol);
      HYPRE_GMRESSetAbsoluteTol(krylov_solver, atol);
      HYPRE_GMRESSetLogging(krylov_solver, 1);
      HYPRE_GMRESSetPrintLevel(krylov_solver, 2);

      /* setup MGR-PCG solver */
      HYPRE_GMRESSetPrecond(krylov_solver,
                            (HYPRE_PtrToSolverFcn) HYPRE_MGRSolve,
                            (HYPRE_PtrToSolverFcn) HYPRE_MGRSetup,
                            krylov_precond);

      HYPRE_GMRESGetPrecond(krylov_solver, &krylov_precond_gotten);
      if (krylov_precond_gotten != krylov_precond)
      {
         hypre_printf("HYPRE_GMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else
      {
         if (myid == 0)
         {
            hypre_printf("HYPRE_GMRESGetPrecond got good precond\n");
         }
      }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      cudaDeviceSynchronize();
#endif

      // Setup main solver
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSetup
      (krylov_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      cudaDeviceSynchronize();
#endif

      // Do the solve
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve
      (krylov_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      cudaDeviceSynchronize();
#endif

      // Get number of iterations and residual
      HYPRE_GMRESGetNumIterations(krylov_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(krylov_solver, &final_res_norm);

      // free memory for flex GMRES
      if (krylov_solver) { HYPRE_ParCSRGMRESDestroy(krylov_solver); }
      if (amg_solver) { HYPRE_BoomerAMGDestroy(amg_solver); }
      if (krylov_precond) { HYPRE_MGRDestroy(krylov_precond); }

      // Print out solver summary
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("GMRES Iterations = %d\n", num_iterations);
         hypre_printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }
   }
   else if (solver_id == 73)
   {
      if (myid == 0) { hypre_printf("Solver:  MGR-GMRES\n"); }

      // Initialize main solver
      HYPRE_ParCSRFlexGMRESCreate(hypre_MPI_COMM_WORLD, &krylov_solver);
      HYPRE_FlexGMRESSetKDim(krylov_solver, k_dim);
      HYPRE_FlexGMRESSetMaxIter(krylov_solver, max_iter);
      HYPRE_FlexGMRESSetTol(krylov_solver, tol);
      HYPRE_FlexGMRESSetAbsoluteTol(krylov_solver, atol);
      HYPRE_FlexGMRESSetLogging(krylov_solver, 1);
      HYPRE_FlexGMRESSetPrintLevel(krylov_solver, 2);

      /* setup MGR-PCG solver */
      HYPRE_FlexGMRESSetPrecond(krylov_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MGRSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_MGRSetup,
                                krylov_precond);

      HYPRE_FlexGMRESGetPrecond(krylov_solver, &krylov_precond_gotten);
      if (krylov_precond_gotten != krylov_precond)
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
      cudaDeviceSynchronize();
#endif

      // Setup main solver
      time_index = hypre_InitializeTiming("FlexGMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_FlexGMRESSetup
      (krylov_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      cudaDeviceSynchronize();
#endif

      // Do the solve
      time_index = hypre_InitializeTiming("FlexGMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_FlexGMRESSolve
      (krylov_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      cudaDeviceSynchronize();
#endif

      // Get number of iterations and residual
      HYPRE_FlexGMRESGetNumIterations(krylov_solver, &num_iterations);
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm(krylov_solver, &final_res_norm);

      // free memory for flex FlexGMRES
      if (krylov_solver) { HYPRE_ParCSRFlexGMRESDestroy(krylov_solver); }
      if (amg_solver) { HYPRE_BoomerAMGDestroy(amg_solver); }
      if (krylov_precond) { HYPRE_MGRDestroy(krylov_precond); }

      // Print out solver summary
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("FlexGMRES Iterations = %d\n", num_iterations);
         hypre_printf("Final FlexGMRES Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }
   }
   else if (solver_id == 74)
   {
      if (myid == 0) { hypre_printf("Solver:  MGR-BiCGSTAB\n"); }

      // Initialize main solver
      HYPRE_ParCSRBiCGSTABCreate(hypre_MPI_COMM_WORLD, &krylov_solver);
      HYPRE_BiCGSTABSetMaxIter(krylov_solver, max_iter);
      HYPRE_BiCGSTABSetTol(krylov_solver, tol);
      HYPRE_BiCGSTABSetAbsoluteTol(krylov_solver, atol);
      HYPRE_BiCGSTABSetLogging(krylov_solver, 1);
      HYPRE_BiCGSTABSetPrintLevel(krylov_solver, 2);

      /* setup MGR-PCG solver */
      HYPRE_BiCGSTABSetPrecond(krylov_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_MGRSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_MGRSetup,
                               krylov_precond);

      HYPRE_BiCGSTABGetPrecond(krylov_solver, &krylov_precond_gotten);
      if (krylov_precond_gotten != krylov_precond)
      {
         hypre_printf("HYPRE_BiCGSTABGetPrecond got bad precond\n");
         return (-1);
      }
      else
      {
         if (myid == 0)
         {
            hypre_printf("HYPRE_BiCGSTABGetPrecond got good precond\n");
         }
      }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      cudaDeviceSynchronize();
#endif

      // Setup main solver
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSetup
      (krylov_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      cudaDeviceSynchronize();
#endif

      // Do the solve
      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve
      (krylov_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      cudaDeviceSynchronize();
#endif

      // Get number of iterations and residual
      HYPRE_BiCGSTABGetNumIterations(krylov_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm(krylov_solver, &final_res_norm);

      // free memory for flex BiCGSTAB
      if (krylov_solver) { HYPRE_ParCSRBiCGSTABDestroy(krylov_solver); }
      if (amg_solver) { HYPRE_BoomerAMGDestroy(amg_solver); }
      if (krylov_precond) { HYPRE_MGRDestroy(krylov_precond); }

      // Print out solver summary
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("BiCGSTAB Iterations = %d\n", num_iterations);
         hypre_printf("Final BiCGSTAB Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/
   // free the matrix, the rhs and the initial guess
   HYPRE_IJMatrixDestroy(ij_A);
   HYPRE_IJVectorDestroy(ij_b);
   HYPRE_IJVectorDestroy(ij_x);

   hypre_TFree(mgr_num_cindexes, HYPRE_MEMORY_HOST);
   hypre_TFree(mgr_point_marker_array, HYPRE_MEMORY_HOST);
   for (i = 0; i < mgr_nlevels; i++)
   {
      hypre_TFree(mgr_cindexes[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(mgr_cindexes, HYPRE_MEMORY_HOST);

   HYPRE_Finalize();
   hypre_MPI_Finalize();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   cudaDeviceReset();
#endif

   return (0);
}

/*----------------------------------------------------------------------
 * Build rhs from file. Expects two files on each processor.
 * filename.n contains the data and
 * and filename.INFO.n contains global row
 * numbers
 *----------------------------------------------------------------------*/

HYPRE_Int
ReadParVectorFromFile( HYPRE_Int            argc,
                       char                *argv[],
                       HYPRE_Int            arg_index,
                       HYPRE_ParVector      *b_ptr     )
{
   char               *filename;

   HYPRE_ParVector b;

   HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      hypre_printf("  Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  From ParFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   HYPRE_ParVectorRead(hypre_MPI_COMM_WORLD, filename, &b);

   *b_ptr = b;

   return (0);
}


/*----------------------------------------------------------------------
 * Build matrix from file. Expects three files on each processor.
 * filename.D.n contains the diagonal part, filename.O.n contains
 * the offdiagonal part and filename.INFO.n contains global row
 * and column numbers, number of columns of offdiagonal matrix
 * and the mapping of offdiagonal column numbers to global column numbers.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParFromFile( HYPRE_Int                  argc,
                  char                *argv[],
                  HYPRE_Int                  arg_index,
                  HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix A;

   HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   HYPRE_ParCSRMatrixRead(hypre_MPI_COMM_WORLD, filename, &A);

   *A_ptr = A;

   return (0);
}
