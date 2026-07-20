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
#include "HYPRE_krylov.h"

#if defined (HYPRE_USING_CUDA)
#include <cuda_profiler_api.h>
#endif
#if defined(HYPRE_USING_CUSPARSE)
#define DISABLE_CUSPARSE_DEPRECATED
#include <cusparse.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int BuildParFromFile (MPI_Comm comm, HYPRE_Int argc, char *argv [],
                            HYPRE_Int arg_index, HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int ReadParVectorFromFile (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                 HYPRE_ParVector *b_ptr );
HYPRE_Int BuildParLaplacian (MPI_Comm comm, HYPRE_Int argc, char *argv [],
                             HYPRE_Int arg_index, HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParDifConv (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                           HYPRE_ParCSRMatrix *A_ptr);
HYPRE_Int BuildParFromOneFile (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                               HYPRE_Int num_functions, HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildFuncTagsFromFiles (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                  HYPRE_ParCSRMatrix A, HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildFuncTagsFromOneFile (MPI_Comm comm, HYPRE_Int argc, char *argv [],
                                    HYPRE_Int arg_index,
                                    HYPRE_ParCSRMatrix A, HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildFuncTagsInterleaved (HYPRE_Int local_size, HYPRE_Int num_functions,
                                    HYPRE_MemoryLocation memory_location,
                                    HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildFuncTagsContiguous (HYPRE_Int local_size, HYPRE_Int num_functions,
                                   HYPRE_MemoryLocation memory_location,
                                   HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildRhsParFromOneFile (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                  HYPRE_ParCSRMatrix A, HYPRE_ParVector *b_ptr );
HYPRE_Int BuildSolParFromOneFile (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                  HYPRE_ParCSRMatrix A, HYPRE_ParVector *x_ptr );
HYPRE_Int BuildBigArrayFromOneFile (MPI_Comm comm, HYPRE_Int argc, char *argv [],
                                    const char *array_name,
                                    HYPRE_Int arg_index, HYPRE_BigInt *partitioning, HYPRE_Int *size, HYPRE_BigInt **array_ptr);
HYPRE_Int BuildParLaplacian9pt (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian27pt (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                 HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian125pt (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                  HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParRotate7pt (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                             HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParVarDifConv (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                              HYPRE_ParCSRMatrix *A_ptr, HYPRE_ParVector *rhs_ptr );
HYPRE_ParCSRMatrix GenerateSysLaplacian (MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                         HYPRE_BigInt nz,
                                         HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                         HYPRE_Int num_fun, HYPRE_Real *mtrx, HYPRE_Real *value);
HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef (MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                              HYPRE_BigInt nz,
                                              HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                              HYPRE_Int num_fun, HYPRE_Real *mtrx, HYPRE_Real *value);
HYPRE_Int SetSysVcoefValues(HYPRE_Int num_fun, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                            HYPRE_Real vcx, HYPRE_Real vcy, HYPRE_Real vcz, HYPRE_Int mtx_entry, HYPRE_Real *values);
HYPRE_Int BuildParCoordinates (MPI_Comm comm, HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                               HYPRE_Int *coorddim_ptr, float **coord_ptr );
HYPRE_Int
SetBoomerAMGOptions(HYPRE_Solver amg_pc, HYPRE_Precision precond_precision);                               
#ifdef __cplusplus
}
#endif

hypre_int
main( hypre_int argc,
      char *argv[] )
{
   MPI_Comm            comm = hypre_MPI_COMM_WORLD;

   HYPRE_Int           arg_index;
   HYPRE_Int           print_usage;
   HYPRE_Int           log_level = 0;
   HYPRE_Int           build_matrix_type;
   HYPRE_Int           build_matrix_arg_index;
   HYPRE_Int           build_rhs_type;
   HYPRE_Int           build_rhs_arg_index;
   HYPRE_Int           build_src_type;
   HYPRE_Int           build_src_arg_index;
   HYPRE_Int           build_x0_type;
   HYPRE_Int           build_x0_arg_index;
   HYPRE_Int           build_funcs_type;
   HYPRE_Int           build_funcs_arg_index;
   HYPRE_Int           num_components = 1;
   HYPRE_Int           solver_id;
   HYPRE_Int           recompute_res = 1;   /* What should be the default here? */
   HYPRE_Int           ioutdat;
   HYPRE_Int           poutdat;
   HYPRE_Int           print_matrix_info = 0;
   HYPRE_Int           debug_flag;
   HYPRE_Int           ierr = 0;
   HYPRE_Int           i, c;
   HYPRE_Int           max_levels = 25;
   HYPRE_Int           num_iterations;
   HYPRE_Int           max_iter = 1000;
   HYPRE_Int           nodal = 0;
   HYPRE_Int           nodal_diag = 0;
   HYPRE_Int           keep_same_sign = 0;
   HYPRE_Real          norm;
   HYPRE_Real          b_dot_b;
   void               *object;

   HYPRE_IJMatrix      ij_A = NULL;
   HYPRE_IJVector      ij_b = NULL;
   HYPRE_IJVector      ij_x = NULL;
   HYPRE_IJVector      *ij_rbm  = NULL;

   HYPRE_ParCSRMatrix  parcsr_A = NULL;
   HYPRE_ParVector     b = NULL;
   HYPRE_ParVector     x = NULL;
   HYPRE_ParVector     *interp_vecs = NULL;
   HYPRE_ParVector     x0_save = NULL;

   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond = NULL;
   HYPRE_Solver        pcg_precond_gotten;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int          *dof_func = NULL;
   HYPRE_Int           free_dof_func = 1;
   HYPRE_Int           filter_functions = 0;
   HYPRE_Int           num_functions = 1;
   HYPRE_Int           num_paths = 1;
   HYPRE_Int           agg_num_levels = 0;
   HYPRE_Int           ns_coarse = 1;

   HYPRE_Int           time_index;
   HYPRE_Int           local_num_rows, local_num_cols;
   HYPRE_BigInt        first_local_row, last_local_row;
   HYPRE_BigInt        first_local_col, last_local_col;
   HYPRE_Int           variant, overlap, domain_type;
   HYPRE_Real          schwarz_rlx_weight;

   HYPRE_Int           use_nonsymm_schwarz = 0;
   HYPRE_Int           build_rbm = 0;
   HYPRE_Int           build_rbm_index = 0;
   HYPRE_Int           num_interp_vecs = 0;
   HYPRE_Int           interp_vec_variant = 0;
   HYPRE_Int           Q_max = 0;
   HYPRE_Real          Q_trunc = 0;

   /* Specific tests */
   HYPRE_Int           lazy_device_init = 0;
   HYPRE_Int           device_id = -1;
   HYPRE_Int           test_error = 0;
   
   /* max dt */
   const HYPRE_Real    dt_inf = 1.0e30;

   HYPRE_Real          dt = dt_inf;

   /* solve -Ax = b, for testing SND matrices */
   HYPRE_Int           negA = 0;

   /* parameters for BoomerAMG */
   HYPRE_Int      coarsen_cut_factor = 0;
   HYPRE_Real     strong_threshold;
   HYPRE_Real     trunc_factor;
   HYPRE_Real     jacobi_trunc_threshold;
   HYPRE_Real     S_commpkg_switch = 1.0;
   HYPRE_Int      P_max_elmts = 4;
   HYPRE_Int      cycle_type;
   HYPRE_Int      fcycle;
   HYPRE_Int      coarsen_type = 10;
   HYPRE_Int      measure_type = 0;
   HYPRE_Int      num_sweeps = 1;
   HYPRE_Int      IS_type;
   HYPRE_Int      relax_type = -1;
   HYPRE_Int      add_relax_type = 18;
   HYPRE_Int      relax_coarse = -1;
   HYPRE_Int      relax_up = -1;
   HYPRE_Int      relax_down = -1;
   HYPRE_Int      relax_order = 0;
   HYPRE_Int      level_w = -1;
   HYPRE_Int      level_ow = -1;
   HYPRE_Int      smooth_type = 6;
   HYPRE_Int      smooth_num_levels = 0;
   HYPRE_Int      smooth_num_sweeps = 1;
   HYPRE_Int      coarse_threshold = 9;
   HYPRE_Int      min_coarse_size = 0;
   /* redundant coarse grid solve */
   HYPRE_Int      seq_threshold = 0;
   HYPRE_Int      redundant = 0;
   /* additive versions */
   HYPRE_Int    additive = -1;
   HYPRE_Int    mult_add = -1;
   HYPRE_Int    simple = -1;
   HYPRE_Int    add_last_lvl = -1;
   HYPRE_Int    add_P_max_elmts = 0;
   HYPRE_Real   add_trunc_factor = 0;
   HYPRE_Int    rap2     = 0;
   HYPRE_Int    mod_rap2 = 0;
   HYPRE_Int    keepTranspose = 0;
#ifdef HYPRE_USING_DSUPERLU
   HYPRE_Int    dslu_threshold = -1;
#endif
   HYPRE_Real   relax_wt;
   HYPRE_Real   add_relax_wt = 1.0;
   HYPRE_Real   relax_wt_level = 0.0;
   HYPRE_Real   outer_wt;
   HYPRE_Real   outer_wt_level = 0;
   HYPRE_Real   tol = 1.e-8, pc_tol = 0.;
   HYPRE_Real   atol = 0.0;
   HYPRE_Real   max_row_sum = 1.;
   HYPRE_Int    precon_cycles = 1;

   HYPRE_Int  cheby_order = 2;
   HYPRE_Int  cheby_eig_est = 10;
   HYPRE_Int  cheby_variant = 0;
   HYPRE_Int  cheby_scale = 1;
   HYPRE_Real cheby_fraction = .3;

#if defined(HYPRE_USING_GPU) && !defined(HYPRE_TEST_USING_HOST)
#if defined(HYPRE_USING_CUSPARSE) && CUSPARSE_VERSION >= 11000
   /* CUSPARSE_SPMV_ALG_DEFAULT doesn't provide deterministic results */
   HYPRE_Int  spmv_use_vendor = 0;
#else
   HYPRE_Int  spmv_use_vendor = 1;
#endif
   HYPRE_Int  use_curand = 1;
#if defined(HYPRE_USING_CUDA)
   HYPRE_Int  spgemm_use_vendor = 0;
#else
   HYPRE_Int  spgemm_use_vendor = 1;
#endif
   HYPRE_Int  spgemm_alg = 1;
   HYPRE_Int  spgemm_binned = 0;
   HYPRE_Int  spgemm_rowest_mtd = 3;
   HYPRE_Int  spgemm_rowest_nsamples = -1; /* default */
   HYPRE_Real spgemm_rowest_mult = -1.0; /* default */
   HYPRE_Int  gpu_aware_mpi = 0;
#endif

   /* for CGC BM Aug 25, 2006 */
   HYPRE_Int      cgcits = 1;
   /* for coordinate plotting BM Oct 24, 2006 */
   HYPRE_Int      plot_grids = 0;
   HYPRE_Int      coord_dim  = 3;
   float         *coordinates = NULL;
   char           plot_file_name[256];

   /* parameters for GMRES */
   HYPRE_Int    k_dim;
   /* interpolation */
   HYPRE_Int    interp_type  = 6; /* default value */
   HYPRE_Int    post_interp_type  = 0; /* default value */
   /* aggressive coarsening */
   HYPRE_Int    agg_interp_type  = 4; /* default value */
   HYPRE_Int    agg_P_max_elmts  = 0; /* default value */
   HYPRE_Int    agg_P12_max_elmts  = 0; /* default value */
   HYPRE_Real   agg_trunc_factor  = 0; /* default value */
   HYPRE_Real   agg_P12_trunc_factor  = 0; /* default value */

   HYPRE_Int    print_system = 0;
   HYPRE_Int    print_system_binary = 0;
   HYPRE_Int    print_system_csr = 0;
   HYPRE_Int    rel_change = 0;
   HYPRE_Int    second_time = 0;
   HYPRE_Int    benchmark = 0;

   HYPRE_Real     *nongalerk_tol = NULL;
   HYPRE_Int       nongalerk_num_tol = 0;

   /* precision options */
   HYPRE_Int precision_id; /* 0=flt, 1=dbl, 2=ldbl */
   HYPRE_Precision solver_precision = HYPRE_REAL_DOUBLE;
   HYPRE_Precision precond_precision = HYPRE_REAL_SINGLE;  
   HYPRE_ParCSRMatrix A_pc = NULL;
   HYPRE_ParCSRMatrix A_slvr = NULL;
   HYPRE_ParVector b_slvr = NULL;
   HYPRE_ParVector b_pc = NULL;
   HYPRE_ParVector x_slvr = NULL;
   HYPRE_ParVector x_pc = NULL;

   long double     final_res_norm_ldbl;
   double          final_res_norm_dbl;
   float           final_res_norm_flt;

   void        *final_res_norm = &final_res_norm_dbl;
   /* Size of solver and preconditioner data types */
   size_t      slvr_size_t = sizeof(hypre_double);
   size_t      pc_size_t = sizeof(hypre_float);

#if defined(HYPRE_USING_MEMORY_TRACKER)
   HYPRE_Int print_mem_tracker = 0;
   char mem_tracker_name[HYPRE_MAX_FILE_NAME_LEN] = {0};
#endif

   /* default execution policy and memory space */
#if defined(HYPRE_TEST_USING_HOST)
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_HOST;
   HYPRE_ExecutionPolicy exec2_policy = HYPRE_EXEC_HOST;
#else
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_DEVICE;
   HYPRE_ExecutionPolicy exec2_policy = HYPRE_EXEC_DEVICE;
#endif

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

   /* Should we test library initialization? */
   for (arg_index = 1; arg_index < argc; arg_index ++)
   {
      if ( strcmp(argv[arg_index], "-ll") == 0 )
      {
         arg_index++;
         log_level = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-lazy_device_init") == 0)
      {
         lazy_device_init = atoi(argv[++arg_index]);
      }
      else if (strcmp(argv[arg_index], "-device_id") == 0)
      {
         device_id = atoi(argv[++arg_index]);
      }
      else if ( strcmp(argv[arg_index], "-memory_host") == 0 )
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
      else if ( strcmp(argv[arg_index], "-exec2_host") == 0 )
      {
         exec2_policy = HYPRE_EXEC_HOST;
      }
      else if ( strcmp(argv[arg_index], "-exec2_device") == 0 )
      {
         exec2_policy = HYPRE_EXEC_DEVICE;
      }
   }

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before HYPRE_Initialize() and should not be changed after
    *-----------------------------------------------------------------*/
   if (default_exec_policy == HYPRE_EXEC_DEVICE ||
       exec2_policy == HYPRE_EXEC_DEVICE )
   {
      hypre_bind_device_id(device_id, myid, num_procs, comm);
   }

   time_index = hypre_InitializeTiming("Hypre init");
   hypre_BeginTiming(time_index);

   HYPRE_Initialize();

   /* We set the execution policy early so that hypre_EndTiming
      knows whether to call hypre_DeviceSync or not. */
   HYPRE_SetExecutionPolicy(default_exec_policy);

   if (!lazy_device_init &&
       (default_exec_policy == HYPRE_EXEC_DEVICE || exec2_policy == HYPRE_EXEC_DEVICE))
   {
      HYPRE_DeviceInitialize();
      if (log_level > 0)
      {
         HYPRE_PrintDeviceInfo();
         hypre_MPI_Barrier(comm);
      }
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Hypre init times", comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      keepTranspose = 1;
      coarsen_type  = 8;
      mod_rap2      = 1;
   }

#if defined (HYPRE_USING_UMPIRE)
   size_t umpire_dev_pool_size    = 4294967296; // 4 GiB
   size_t umpire_uvm_pool_size    = 4294967296; // 4 GiB
   size_t umpire_pinned_pool_size = 4294967296; // 4 GiB
   size_t umpire_host_pool_size   = 4294967296; // 4 GiB
#endif

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
   build_matrix_type = 2;
   build_matrix_arg_index = argc;
   build_rhs_type = 2;
   build_rhs_arg_index = argc;
   build_src_type = -1;
   build_src_arg_index = argc;
   build_x0_type = -1;
   build_x0_arg_index = argc;
   build_funcs_type = 0;
   build_funcs_arg_index = argc;
   IS_type = 1;
   debug_flag = 0;
   solver_id = 0;
   ioutdat = 2;
   poutdat = 1;
   hypre_sprintf (plot_file_name, "AMGgrids.CF.dat");

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-frombinfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = -2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromfile") == 0 )
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
      else if ( strcmp(argv[arg_index], "-fromonecsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 3;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 4;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-125pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 5;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 6;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-vardifconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 7;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rotate") == 0 )
      {
         arg_index++;
         build_matrix_type      = 8;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-test_error") == 0 )
      {
         arg_index++;
         test_error = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-funcsfromonefile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 1;
         build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-funcsfromfile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 2;
         build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-funcsinterleaved") == 0 )
      {
         arg_index++;
         build_funcs_type = 3;
      }
      else if ( strcmp(argv[arg_index], "-funcscontiguous") == 0 )
      {
         arg_index++;
         build_funcs_type = 4;
      }
      else if ( strcmp(argv[arg_index], "-mat-info") == 0 )
      {
         arg_index++;
         print_matrix_info = 1;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rbm") == 0 )
      {
         arg_index++;
         build_rbm      = 1;
         num_interp_vecs = atoi(argv[arg_index++]);
         build_rbm_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-nc") == 0 )
      {
         arg_index++;
         num_components = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 0;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfrombinfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = -2;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromonefile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 1;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsparcsrfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 7;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 2;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsrand") == 0 )
      {
         arg_index++;
         build_rhs_type      = 3;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-xisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 4;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhszero") == 0 )
      {
         arg_index++;
         build_rhs_type      = 5;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcfromfile") == 0 )
      {
         arg_index++;
         build_src_type      = 0;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcfromonefile") == 0 )
      {
         arg_index++;
         build_src_type      = 1;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcisone") == 0 )
      {
         arg_index++;
         build_src_type      = 2;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcrand") == 0 )
      {
         arg_index++;
         build_src_type      = 3;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srczero") == 0 )
      {
         arg_index++;
         build_src_type      = 4;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-x0fromonefile") == 0 )
      {
         arg_index++;
         build_x0_type       = -3;
         build_x0_arg_index  = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-x0frombinfile") == 0 )
      {
         arg_index++;
         build_x0_type       = -2;
         build_x0_arg_index  = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-x0fromfile") == 0 )
      {
         arg_index++;
         build_x0_type       = 0;
         build_x0_arg_index  = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-x0parcsrfile") == 0 )
      {
         arg_index++;
         build_x0_type      = 7;
         build_x0_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-x0rand") == 0 )
      {
         arg_index++;
         build_x0_type       = 1;
         build_x0_arg_index  = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-CFfromfile") == 0 )
      {
         arg_index++;
         coarsen_type      = 999;
      }
      else if ( strcmp(argv[arg_index], "-cljp") == 0 )
      {
         arg_index++;
         coarsen_type      = 0;
      }
      else if ( strcmp(argv[arg_index], "-cljp1") == 0 )
      {
         arg_index++;
         coarsen_type      = 7;
      }
      else if ( strcmp(argv[arg_index], "-cgc") == 0 )
      {
         arg_index++;
         coarsen_type      = 21;
         cgcits            = 200;
      }
      else if ( strcmp(argv[arg_index], "-cgce") == 0 )
      {
         arg_index++;
         coarsen_type      = 22;
         cgcits            = 200;
      }
      else if ( strcmp(argv[arg_index], "-pmis") == 0 )
      {
         arg_index++;
         coarsen_type      = 8;
      }
      else if ( strcmp(argv[arg_index], "-pmis1") == 0 )
      {
         arg_index++;
         coarsen_type      = 9;
      }
      else if ( strcmp(argv[arg_index], "-cr1") == 0 )
      {
         arg_index++;
         coarsen_type      = 98;
      }
      else if ( strcmp(argv[arg_index], "-cr") == 0 )
      {
         arg_index++;
         coarsen_type      = 99;
      }
      else if ( strcmp(argv[arg_index], "-hmis") == 0 )
      {
         arg_index++;
         coarsen_type      = 10;
      }
      else if ( strcmp(argv[arg_index], "-ruge") == 0 )
      {
         arg_index++;
         coarsen_type      = 1;
      }
      else if ( strcmp(argv[arg_index], "-ruge1p") == 0 )
      {
         arg_index++;
         coarsen_type      = 11;
      }
      else if ( strcmp(argv[arg_index], "-ruge2b") == 0 )
      {
         arg_index++;
         coarsen_type      = 2;
      }
      else if ( strcmp(argv[arg_index], "-ruge3") == 0 )
      {
         arg_index++;
         coarsen_type      = 3;
      }
      else if ( strcmp(argv[arg_index], "-ruge3c") == 0 )
      {
         arg_index++;
         coarsen_type      = 4;
      }
      else if ( strcmp(argv[arg_index], "-rugerlx") == 0 )
      {
         arg_index++;
         coarsen_type      = 5;
      }
      else if ( strcmp(argv[arg_index], "-falgout") == 0 )
      {
         arg_index++;
         coarsen_type      = 6;
      }
      else if ( strcmp(argv[arg_index], "-gm") == 0 )
      {
         arg_index++;
         measure_type      = 1;
      }
      else if ( strcmp(argv[arg_index], "-is") == 0 )
      {
         arg_index++;
         IS_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index++;
         relax_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx_coarse") == 0 )
      {
         arg_index++;
         relax_coarse = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx_down") == 0 )
      {
         arg_index++;
         relax_down = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx_up") == 0 )
      {
         arg_index++;
         relax_up = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smtype") == 0 )
      {
         arg_index++;
         smooth_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smlv") == 0 )
      {
         arg_index++;
         smooth_num_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxl") == 0 )
      {
         arg_index++;
         max_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         debug_flag = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nf") == 0 )
      {
         arg_index++;
         num_functions = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ff") == 0 )
      {
         arg_index++;
         filter_functions = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_nl") == 0 )
      {
         arg_index++;
         agg_num_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-npaths") == 0 )
      {
         arg_index++;
         num_paths = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns") == 0 )
      {
         arg_index++;
         num_sweeps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns_coarse") == 0 )
      {
         arg_index++;
         ns_coarse = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sns") == 0 )
      {
         arg_index++;
         smooth_num_sweeps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-max_iter") == 0 )
      {
         arg_index++;
         max_iter = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dt") == 0 )
      {
         arg_index++;
         dt = (HYPRE_Real)atof(argv[arg_index++]);
         build_rhs_type = -1;
         if ( build_src_type == -1 ) { build_src_type = 2; }
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_TEST_USING_HOST)
      else if ( strcmp(argv[arg_index], "-gpu_mpi") == 0 )
      {
         arg_index++;
         gpu_aware_mpi = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mm_vendor") == 0 )
      {
         arg_index++;
         spgemm_use_vendor = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mv_vendor") == 0 )
      {
         arg_index++;
         spmv_use_vendor = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_alg") == 0 )
      {
         arg_index++;
         spgemm_alg  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_binned") == 0 )
      {
         arg_index++;
         spgemm_binned  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_rowest") == 0 )
      {
         arg_index++;
         spgemm_rowest_mtd  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_rowestmult") == 0 )
      {
         arg_index++;
         spgemm_rowest_mult  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_rowestnsamples") == 0 )
      {
         arg_index++;
         spgemm_rowest_nsamples  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-use_curand") == 0 )
      {
         arg_index++;
         use_curand = atoi(argv[arg_index++]);
      }
#endif
#if defined (HYPRE_USING_UMPIRE)
      else if ( strcmp(argv[arg_index], "-umpire_dev_pool_size") == 0 )
      {
         arg_index++;
         umpire_dev_pool_size = (size_t) 1073741824 * atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-umpire_uvm_pool_size") == 0 )
      {
         arg_index++;
         umpire_uvm_pool_size = (size_t) 1073741824 * atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-umpire_pinned_pool_size") == 0 )
      {
         arg_index++;
         umpire_pinned_pool_size = (size_t) 1073741824 * atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-umpire_host_pool_size") == 0 )
      {
         arg_index++;
         umpire_host_pool_size = (size_t) 1073741824 * atoi(argv[arg_index++]);
      }
#endif
      else if ( strcmp(argv[arg_index], "-negA") == 0 )
      {
         arg_index++;
         negA = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-second_time") == 0 )
      {
         arg_index++;
         second_time = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-benchmark") == 0 )
      {
         arg_index++;
         benchmark = atoi(argv[arg_index++]);
      }
#if defined(HYPRE_USING_MEMORY_TRACKER)
      else if ( strcmp(argv[arg_index], "-print_mem_tracker") == 0 )
      {
         arg_index++;
         print_mem_tracker = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mem_tracker_filename") == 0 )
      {
         arg_index++;
         snprintf(mem_tracker_name, HYPRE_MAX_FILE_NAME_LEN, "%s", argv[arg_index++]);
      }
#endif
      else if ( strcmp(argv[arg_index], "-solver_precision") == 0 )
      {
         arg_index++;
         precision_id = atoi(argv[arg_index++]);

         switch (precision_id)
         {
            case 0:
               solver_precision = HYPRE_REAL_SINGLE;
               slvr_size_t = sizeof(hypre_float);
               final_res_norm = &final_res_norm_flt;
               break;
            case 1:
               solver_precision = HYPRE_REAL_DOUBLE;
               slvr_size_t = sizeof(hypre_double);
               final_res_norm = &final_res_norm_dbl;
               break;
            case 2:
               solver_precision = HYPRE_REAL_LONGDOUBLE;
               slvr_size_t = sizeof(hypre_long_double);
               final_res_norm = &final_res_norm_ldbl;
               break;
            default:
               { HYPRE_Int value = 0; hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown solver precision"); return value; }            
         }
      }
      else if ( strcmp(argv[arg_index], "-pc_precision") == 0 )
      {
         arg_index++;
         precision_id = atoi(argv[arg_index++]);

         switch (precision_id)
         {
            case 0:
               precond_precision = HYPRE_REAL_SINGLE;
               pc_size_t = sizeof(hypre_float);
               break;
            case 1:
               precond_precision = HYPRE_REAL_DOUBLE;
               pc_size_t = sizeof(hypre_double);
               break;
            case 2:
               precond_precision = HYPRE_REAL_LONGDOUBLE;
               pc_size_t = sizeof(hypre_long_double);
               break;
            default:
               { HYPRE_Int value = 0; hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown pc precision"); return value; }            
         }
      }
      else
      {
         arg_index++;
      }
   }

   /* Unused variables */
   HYPRE_UNUSED_VAR(slvr_size_t);
   HYPRE_UNUSED_VAR(pc_size_t);

   /* begin CGC BM Aug 25, 2006 */
   if (coarsen_type == 21 || coarsen_type == 22)
   {
      arg_index = 0;
      while ( (arg_index < argc) && (!print_usage) )
      {
         if ( strcmp(argv[arg_index], "-cgcits") == 0 )
         {
            arg_index++;
            cgcits = atoi(argv[arg_index++]);
         }
         else
         {
            arg_index++;
         }
      }
   }

   /* defaults for BoomerAMG */
   if (solver_id == 1 || solver_id == 11 || solver_id == 21 || solver_id == 31)
   {
      strong_threshold = 0.25;
      trunc_factor = 0.;
      jacobi_trunc_threshold = 0.01;
      cycle_type = 1;
      fcycle = 0;
      relax_wt = 1.;
      outer_wt = 1.;
   }

   /* defaults for Schwarz */
   variant = 0;  /* multiplicative */
   overlap = 1;  /* 1 layer overlap */
   domain_type = 2; /* through agglomeration */
   schwarz_rlx_weight = 1.;

   /* defaults for GMRES */
   k_dim = 5;

   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         relax_wt = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-wl") == 0 )
      {
         arg_index++;
         relax_wt_level = (HYPRE_Real)atof(argv[arg_index++]);
         level_w = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ow") == 0 )
      {
         arg_index++;
         outer_wt = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-owl") == 0 )
      {
         arg_index++;
         outer_wt_level = (HYPRE_Real)atof(argv[arg_index++]);
         level_ow = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sw") == 0 )
      {
         arg_index++;
         schwarz_rlx_weight = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-coarse_th") == 0 )
      {
         arg_index++;
         coarse_threshold  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-min_cs") == 0 )
      {
         arg_index++;
         min_coarse_size  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seq_th") == 0 )
      {
         arg_index++;
         seq_threshold  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-red") == 0 )
      {
         arg_index++;
         redundant  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cutf") == 0 )
      {
         arg_index++;
         coarsen_cut_factor = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         strong_threshold  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-CF") == 0 )
      {
         arg_index++;
         relax_order = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-atol") == 0 )
      {
         arg_index++;
         atol  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index++;
         max_row_sum  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tr") == 0 )
      {
         arg_index++;
         trunc_factor  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Pmx") == 0 )
      {
         arg_index++;
         P_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-interpvecvar") == 0 )
      {
         arg_index++;
         interp_vec_variant  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Qtr") == 0 )
      {
         arg_index++;
         Q_trunc  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Qmx") == 0 )
      {
         arg_index++;
         Q_max = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-jtr") == 0 )
      {
         arg_index++;
         jacobi_trunc_threshold  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Ssw") == 0 )
      {
         arg_index++;
         S_commpkg_switch = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-recompute") == 0 )
      {
         arg_index++;
         recompute_res = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-iout") == 0 )
      {
         arg_index++;
         ioutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pout") == 0 )
      {
         arg_index++;
         poutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-var") == 0 )
      {
         arg_index++;
         variant  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-use_ns") == 0 )
      {
         arg_index++;
         use_nonsymm_schwarz = 1;
      }
      else if ( strcmp(argv[arg_index], "-ov") == 0 )
      {
         arg_index++;
         overlap  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dom") == 0 )
      {
         arg_index++;
         domain_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-blk_sm") == 0 )
      {
         arg_index++;
         smooth_num_levels = atoi(argv[arg_index++]);
         overlap = 0;
         smooth_type = 6;
         domain_type = 1;
      }
      else if ( strcmp(argv[arg_index], "-mu") == 0 )
      {
         arg_index++;
         cycle_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fmg") == 0 )
      {
         arg_index++;
         fcycle  = 1;
      }
      else if ( strcmp(argv[arg_index], "-interptype") == 0 )
      {
         arg_index++;
         interp_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_interp") == 0 )
      {
         arg_index++;
         agg_interp_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_Pmx") == 0 )
      {
         arg_index++;
         agg_P_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_P12_mx") == 0 )
      {
         arg_index++;
         agg_P12_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_tr") == 0 )
      {
         arg_index++;
         agg_trunc_factor  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_P12_tr") == 0 )
      {
         arg_index++;
         agg_P12_trunc_factor  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-postinterptype") == 0 )
      {
         arg_index++;
         post_interp_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nodal") == 0 )
      {
         arg_index++;
         nodal  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rel_change") == 0 )
      {
         arg_index++;
         rel_change = 1;
      }
      else if ( strcmp(argv[arg_index], "-nodal_diag") == 0 )
      {
         arg_index++;
         nodal_diag  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-keepSS") == 0 )
      {
         arg_index++;
         keep_same_sign  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_order") == 0 )
      {
         arg_index++;
         cheby_order = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_eig_est") == 0 )
      {
         arg_index++;
         cheby_eig_est = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_variant") == 0 )
      {
         arg_index++;
         cheby_variant = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_scale") == 0 )
      {
         arg_index++;
         cheby_scale = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_fraction") == 0 )
      {
         arg_index++;
         cheby_fraction = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-additive") == 0 )
      {
         arg_index++;
         additive  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mult_add") == 0 )
      {
         arg_index++;
         mult_add  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-simple") == 0 )
      {
         arg_index++;
         simple  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_end") == 0 )
      {
         arg_index++;
         add_last_lvl  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_Pmx") == 0 )
      {
         arg_index++;
         add_P_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_tr") == 0 )
      {
         arg_index++;
         add_trunc_factor  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_rlx") == 0 )
      {
         arg_index++;
         add_relax_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_w") == 0 )
      {
         arg_index++;
         add_relax_wt = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rap") == 0 )
      {
         arg_index++;
         rap2  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mod_rap2") == 0 )
      {
         arg_index++;
         mod_rap2  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-keepT") == 0 )
      {
         arg_index++;
         keepTranspose  = atoi(argv[arg_index++]);
      }
#ifdef HYPRE_USING_DSUPERLU
      else if ( strcmp(argv[arg_index], "-dslu_th") == 0 )
      {
         arg_index++;
         dslu_threshold  = atoi(argv[arg_index++]);
      }
#endif
      else if ( strcmp(argv[arg_index], "-nongalerk_tol") == 0 )
      {
         arg_index++;
         nongalerk_num_tol = atoi(argv[arg_index++]);
         nongalerk_tol = hypre_CTAlloc(HYPRE_Real,  nongalerk_num_tol, HYPRE_MEMORY_HOST);
         for (i = 0; i < nongalerk_num_tol; i++)
         {
            nongalerk_tol[i] = (HYPRE_Real)atof(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-printbin") == 0 )
      {
         arg_index++;
         print_system_binary = 1;
      }
      else if ( strcmp(argv[arg_index], "-printcsr") == 0 )
      {
         arg_index++;
         print_system_csr = 1;
      }
      /* BM Oct 23, 2006 */
      else if ( strcmp(argv[arg_index], "-plot_grids") == 0 )
      {
         arg_index++;
         plot_grids = 1;
      }
      else if ( strcmp(argv[arg_index], "-plot_file_name") == 0 )
      {
         arg_index++;
         hypre_sprintf (plot_file_name, "%s", argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-precon_cycles") == 0 )
      {
         arg_index++;
         precon_cycles = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }
   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/

   if ( print_usage )
   {
      if ( myid == 0 )
      {
         hypre_printf("\n");
         hypre_printf("Usage: %s [<options>]\n", argv[0]);
         hypre_printf("\n");
         hypre_printf("  -ll <val>                  : hypre's log level. \n");
         hypre_printf("      0 = (default) No messaging.\n");
         hypre_printf("      1 = Display memory usage statistics for each MPI rank.\n");
         hypre_printf("      2 = Display aggregate memory usage statistics over MPI ranks.\n");
         hypre_printf("\n");
         hypre_printf("  -fromfile <filename>       : ");
         hypre_printf("matrix read from multiple files (IJ format)\n");
         hypre_printf("  -frombinfile <filename>    : ");
         hypre_printf("matrix read from multiple binary files (IJ format)\n");
         hypre_printf("  -fromparcsrfile <filename> : ");
         hypre_printf("matrix read from multiple files (ParCSR format)\n");
         hypre_printf("  -fromonecsrfile <filename> : ");
         hypre_printf("matrix read from a single file (CSR format)\n");
         hypre_printf("\n");
         hypre_printf("  -laplacian             : build 5pt 2D laplacian problem (default) \n");
         hypre_printf("  -sysL <num functions>  : build SYSTEMS laplacian 7pt operator\n");
         hypre_printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
         hypre_printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
         hypre_printf("  -125pt [<opts>]        : build 125pt (27pt squared) 3D laplacian\n");
         hypre_printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
         hypre_printf("  -vardifconv [<opts>]   : build variable conv.-diffusion problem\n");
         hypre_printf("  -rotate [<opts>]       : build 7pt rotated laplacian problem\n");
         hypre_printf("    -n <nx> <ny> <nz>    : total problem size \n");
         hypre_printf("    -P <Px> <Py> <Pz>    : processor topology\n");
         hypre_printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
         hypre_printf("    -a <ax> <ay> <az>    : convection coefficients\n");
         hypre_printf("    -atype <type>        : FD scheme for convection \n");
         hypre_printf("           0=Forward (default)       1=Backward\n");
         hypre_printf("           2=Centered                3=Upwind\n");
         hypre_printf("\n");
         hypre_printf("  -rbm <val> <filename>  : rigid body mode vectors\n");
         hypre_printf("  -nc <val>              : number of components of a vector (multivector)\n");
         hypre_printf("  -rhsfromfile           : ");
         hypre_printf("rhs read from multiple files (IJ format)\n");
         hypre_printf("  -rhsfrombinfile        : ");
         hypre_printf("rhs read from multiple binary files (IJ format)\n");
         hypre_printf("  -rhsfromonefile        : ");
         hypre_printf("rhs read from a single file (CSR format)\n");
         hypre_printf("  -rhsparcsrfile        :  ");
         hypre_printf("rhs read from multiple files (ParCSR format)\n");
         hypre_printf("  -Ffromonefile          : ");
         hypre_printf("list of F points from a single file\n");
         hypre_printf("  -SFfromonefile          : ");
         hypre_printf("list of isolated F points from a single file\n");
         hypre_printf("  -rhsrand               : rhs is random vector\n");
         hypre_printf("  -rhsisone              : rhs is vector with unit coefficients (default)\n");
         hypre_printf("  -xisone                : solution of all ones\n");
         hypre_printf("  -rhszero               : rhs is zero vector\n");
         hypre_printf("\n");
         hypre_printf("  -dt <val>              : specify finite backward Euler time step\n");
         hypre_printf("                         :    -rhsfromfile, -rhsfromonefile, -rhsrand,\n");
         hypre_printf("                         :    -rhsrand, or -xisone will be ignored\n");
         hypre_printf("  -srcfromfile           : ");
         hypre_printf("backward Euler source read from multiple files (IJ format)\n");
         hypre_printf("  -srcfromonefile        : ");
         hypre_printf("backward Euler source read from a single file (IJ format)\n");
         hypre_printf("  -srcrand               : ");
         hypre_printf("backward Euler source is random vector with coefficients in range 0 - 1\n");
         hypre_printf("  -srcisone              : ");
         hypre_printf("backward Euler source is vector with unit coefficients (default)\n");
         hypre_printf("  -srczero               : ");
         hypre_printf("backward Euler source is zero-vector\n");
         hypre_printf("  -x0fromfile            : ");
         hypre_printf("initial guess x0 read from multiple files (IJ format)\n");
         hypre_printf("  -solver <ID>           : solver ID\n");
         hypre_printf("       0=AMG               1=AMG-PCG\n");
         hypre_printf("       2=DS-PCG            3=AMG-GMRES\n");
         hypre_printf("       9=AMG-BiCGSTAB\n");
         hypre_printf("       10=DS-BiCGSTAB\n");
         hypre_printf("       20=Hybrid solver/ DiagScale, AMG\n");
         hypre_printf("       21=Cheby-PCG       22=Cheby-GMRES\n");
         hypre_printf("       60=DS-FlexGMRES    61=AMG-FlexGMRES\n");
         hypre_printf("\n");
         hypre_printf("  -cljp                 : CLJP coarsening\n");
         hypre_printf("  -cljp1                : CLJP coarsening, fixed random\n");
         hypre_printf("  -cgc                  : CGC coarsening\n");
         hypre_printf("  -cgce                 : CGC-E coarsening\n");
         hypre_printf("  -pmis                 : PMIS coarsening\n");
         hypre_printf("  -pmis1                : PMIS coarsening, fixed random\n");
         hypre_printf("  -hmis                 : HMIS coarsening (default)\n");
         hypre_printf("  -ruge                 : Ruge-Stueben coarsening (local)\n");
         hypre_printf("  -ruge1p               : Ruge-Stueben coarsening 1st pass only(local)\n");
         hypre_printf("  -ruge3                : third pass on boundary\n");
         hypre_printf("  -ruge3c               : third pass on boundary, keep c-points\n");
         hypre_printf("  -falgout              : local Ruge_Stueben followed by CLJP\n");
         hypre_printf("  -gm                   : use global measures\n");
         hypre_printf("\n");
         hypre_printf("  -interptype  <val>    : set interpolation type\n");
         hypre_printf("       0=Classical modified interpolation  \n");
         hypre_printf("       0=Classical modified interpolation for hyperbolic PDEs \n");
         hypre_printf("       3=direct interpolation with separation of weights  \n");
         hypre_printf("       15=direct interpolation\n");
         hypre_printf("       4=multipass interpolation  \n");
         hypre_printf("       5=multipass interpolation with separation of weights  \n");
         hypre_printf("       6=extended classical modified interpolation (default) \n");
         hypre_printf("       7=extended (only if no common C neighbor) interpolation  \n");
         hypre_printf("       8=standard interpolation  \n");
         hypre_printf("       9=standard interpolation with separation of weights  \n");
         hypre_printf("      12=FF interpolation  \n");
         hypre_printf("      13=FF1 interpolation  \n");

         hypre_printf("      16=use modified unknown interpolation for a system (w/unknown or hybrid approach) \n");
         hypre_printf("      17=use non-systems interp = 6 for a system (w/unknown or hybrid approach) \n");
         hypre_printf("      18=use non-systems interp = 8 for a system (w/unknown or hybrid approach) \n");
         hypre_printf("      19=use non-systems interp = 0 for a system (w/unknown or hybrid approach) \n");

         hypre_printf("      10=classical block interpolation for nodal systems AMG\n");
         hypre_printf("      11=classical block interpolation with diagonal blocks for nodal systems AMG\n");
         hypre_printf("      20=same as 10, but don't add weak connect. to diag \n");
         hypre_printf("      21=same as 11, but don't add weak connect. to diag \n");
         hypre_printf("      22=classical block interpolation w/Ruge's variant for nodal systems AMG \n");
         hypre_printf("      23=same as 22, but use row sums for diag scaling matrices,for nodal systems AMG \n");
         hypre_printf("      24=direct block interpolation for nodal systems AMG\n");
         hypre_printf("     100=One point interpolation [a Boolean matrix]\n");
         hypre_printf("\n");

         hypre_printf("  -rlx  <val>            : relaxation type\n");
         hypre_printf("       0=Weighted Jacobi  \n");
         hypre_printf("       1=Gauss-Seidel (very slow!)  \n");
         hypre_printf("       3=Hybrid Gauss-Seidel  \n");
         hypre_printf("       4=Hybrid backward Gauss-Seidel  \n");
         hypre_printf("       6=Hybrid symmetric Gauss-Seidel  \n");
         hypre_printf("       8= symmetric L1-Gauss-Seidel  \n");
         hypre_printf("       13= forward L1-Gauss-Seidel  \n");
         hypre_printf("       14= backward L1-Gauss-Seidel  \n");
         hypre_printf("       15=CG  \n");
         hypre_printf("       16=Chebyshev  \n");
         hypre_printf("       17=FCF-Jacobi  \n");
         hypre_printf("       18=L1-Jacobi (may be used with -CF) \n");
         hypre_printf("       9=Gauss elimination (use for coarsest grid only)  \n");
         hypre_printf("       99=Gauss elimination with pivoting (use for coarsest grid only)  \n");
         hypre_printf("       20= Nodal Weighted Jacobi (for systems only) \n");
         hypre_printf("       23= Nodal Hybrid Jacobi/Gauss-Seidel (for systems only) \n");
         hypre_printf("       26= Nodal Hybrid Symmetric Gauss-Seidel  (for systems only)\n");
         hypre_printf("       29= Nodal Gauss elimination (use for coarsest grid only)  \n");
         hypre_printf("  -rlx_coarse  <val>       : set relaxation type for coarsest grid\n");
         hypre_printf("  -rlx_down    <val>       : set relaxation type for down cycle\n");
         hypre_printf("  -rlx_up      <val>       : set relaxation type for up cycle\n");
         hypre_printf("  -cheby_order  <val> : set order (1-4) for Chebyshev poly. smoother (default is 2)\n");
         hypre_printf("  -cheby_fraction <val> : fraction of the spectrum for Chebyshev poly. smoother (default is .3)\n");
         hypre_printf("  -nodal  <val>            : nodal system type\n");
         hypre_printf("       0 = Unknown approach \n");
         hypre_printf("       1 = Frobenius norm  \n");
         hypre_printf("       2 = Sum of Abs.value of elements  \n");
         hypre_printf("       3 = Largest magnitude element (includes its sign)  \n");
         hypre_printf("       4 = Inf. norm  \n");
         hypre_printf("       5 = One norm  (note: use with block version only) \n");
         hypre_printf("       6 = Sum of all elements in block  \n");
         hypre_printf("  -nodal_diag <val>        :how to treat diag elements\n");
         hypre_printf("       0 = no special treatment \n");
         hypre_printf("       1 = make diag = neg.sum of the off_diag  \n");
         hypre_printf("       2 = make diag = neg. of diag \n");
         hypre_printf("  -ns <val>              : Use <val> sweeps on each level\n");
         hypre_printf("                           (default C/F down, F/C up, F/C fine\n");
         hypre_printf("  -ns_coarse  <val>       : set no. of sweeps for coarsest grid\n");
         hypre_printf("\n");
         hypre_printf("  -mu   <val>            : set AMG cycles (1=V, 2=W, etc.)\n");
         hypre_printf("  -cutf <val>            : set coarsening cut factor for dense rows\n");
         hypre_printf("  -th   <val>            : set AMG threshold Theta = val \n");
         hypre_printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
         hypre_printf("  -Pmx  <val>            : set maximal no. of elmts per row for AMG interpolation (default: 4)\n");
         hypre_printf("  -jtr  <val>            : set truncation threshold for Jacobi interpolation = val \n");
         hypre_printf("  -Ssw  <val>            : set S-commpkg-switch = val \n");
         hypre_printf("  -mxrs <val>            : set AMG maximum row sum threshold for dependency weakening \n");
         hypre_printf("  -ff <0/1>              : set filtering based on functions for systems AMG\n");
         hypre_printf("  -nf <val>              : set number of functions for systems AMG\n");

         hypre_printf("  -postinterptype <val>  : invokes <val> no. of Jacobi interpolation steps after main interpolation\n");
         hypre_printf("\n");
         hypre_printf("  -cgcitr <val>          : set maximal number of coarsening iterations for CGC\n");
         hypre_printf("                         : 1  PCG  (default)\n");
         hypre_printf("                         : 2  GMRES\n");
         hypre_printf("                         : 3  BiCGSTAB\n");

         hypre_printf("  -w   <val>             : set Jacobi relax weight = val\n");
         hypre_printf("  -k   <val>             : dimension Krylov space for GMRES\n");
         hypre_printf("  -aug   <val>           : number of augmentation vectors for LGMRES (-k indicates total approx space size)\n");

         hypre_printf("  -mxl  <val>            : maximum number of levels (AMG)\n");
         hypre_printf("  -tol  <val>            : set solver convergence tolerance = val\n");
         hypre_printf("  -atol  <val>           : set solver absolute convergence tolerance = val\n");
         hypre_printf("  -max_iter  <val>       : set max iterations\n");
         hypre_printf("  -agg_nl  <val>         : set number of aggressive coarsening levels (default:0)\n");
         hypre_printf("  -np  <val>             : set number of paths of length 2 for aggr. coarsening\n");
         hypre_printf("\n");
         hypre_printf("  -iout <val>            : set output flag\n");
         hypre_printf("       0 = no output\n");
         hypre_printf("       1 = matrix and basic solver stats\n");
         hypre_printf("       2 = abs. and rel. residual norms\n");
         hypre_printf("       3 = abs. residual norms for multi-tag vectors (GMRES/FlexGMRES)\n");
         hypre_printf("       4 = tagged rel. residual norms for multi-tag vectors (GMRES/FlexGMRES)\n");
         hypre_printf("       5 = rel. residual norms for multi-tag vectors (GMRES/FlexGMRES)\n");
         hypre_printf("       6 = abs. and rel. error norms (GMRES only)\n");
         hypre_printf("       7 = abs. error norms for multi-tag vectors (GMRES only)\n");
         hypre_printf("       8 = tagged rel. error norms for multi-tag vectors (GMRES only)\n");
         hypre_printf("       9 = rel. error norms for multi-tag vectors (GMRES only)\n");
         hypre_printf("\n");
         hypre_printf("  -dbg <val>             : set debug flag\n");
         hypre_printf("       0=no debugging\n       1=internal timing\n       2=interpolation truncation\n       3=more detailed timing in coarsening routine\n");
         hypre_printf("\n");
         hypre_printf("  -print                 : print out the system\n");
         hypre_printf("\n");
         hypre_printf("  -plot_grids            : print out information for plotting the grids\n");
         hypre_printf("  -plot_file_name <val>  : file name for plotting output\n");
         hypre_printf("\n");
         hypre_printf("  -smtype <val>      :smooth type\n");
         hypre_printf("  -smlv <val>        :smooth num levels\n");
         hypre_printf("  -ov <val>          :over lap:\n");
         hypre_printf("  -dom <val>         :domain type\n");
         hypre_printf("  -use_ns            : use non-symm schwarz smoother\n");
         hypre_printf("  -var <val>         : schwarz smoother variant (0-3) \n");
         hypre_printf("  -blk_sm <val>      : same as '-smtype 6 -ov 0 -dom 1 -smlv <val>'\n");
         hypre_printf("  -nongalerk_tol <val> <list>    : specify the NonGalerkin drop tolerance\n");
         hypre_printf("                                   and list contains the values, where last value\n");
         hypre_printf("                                   in list is repeated if val < num_levels in AMG\n");

#if defined(HYPRE_USING_GPU)
         /* GPU options */
         hypre_printf("GPU options:\n");
         hypre_printf("  -lazy_device_init <0/1>    : delay device initialization until first use (default 0)\n");
         hypre_printf("  -device_id <val>           : bind this MPI rank to a specific GPU device (default auto)\n");
         hypre_printf("  -memory_host               : use host memory for IJ matrix/vector data\n");
         hypre_printf("  -memory_device             : use device memory for IJ matrix/vector data\n");
         hypre_printf("  -exec_host                 : use host execution policy\n");
         hypre_printf("  -exec_device               : use device execution policy\n");
         hypre_printf("  -exec2_host                : use host execution policy for the second setup/solve\n");
         hypre_printf("  -exec2_device              : use device execution policy for the second setup/solve\n");
         hypre_printf("  -gpu_mpi <0/1>             : use GPU-aware MPI with device buffers (default 0)\n");
         hypre_printf("  -mv_vendor <0/1>           : use vendor SpMV implementation\n");
         hypre_printf("  -mm_vendor <0/1>           : use vendor SpGEMM implementation\n");
         hypre_printf("  -spgemm_alg <val>          : set SpGEMM algorithm (1-3)\n");
         hypre_printf("  -spgemm_binned <0/1>       : use binned SpGEMM kernels\n");
         hypre_printf("  -spgemm_rowest <val>       : set SpGEMM row-nnz estimate method (1-3)\n");
         hypre_printf("  -spgemm_rowestmult <val>   : set SpGEMM row-nnz estimate multiplier\n");
         hypre_printf("  -spgemm_rowestnsamples <val> : set SpGEMM row-nnz estimate sample count\n");
         hypre_printf("  -use_curand <0/1>          : use GPU random number generation\n");
         /* end GPU options */
#endif
#if defined (HYPRE_USING_UMPIRE)
         /* UMPIRE options */
         hypre_printf("  -umpire_dev_pool_size <val>      : device memory pool size (GiB)\n");
         hypre_printf("  -umpire_uvm_pool_size <val>      : device unified virtual memory pool size (GiB)\n");
         hypre_printf("  -umpire_pinned_pool_size <val>   : pinned memory pool size (GiB)\n");
         hypre_printf("  -umpire_host_pool_size <val>     : host memory pool size (GiB)\n");
         /* end UMPIRE options */
#endif

      }

      goto final;
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
#if defined(HYPRE_DEVELOP_STRING) && defined(HYPRE_DEVELOP_BRANCH)
      hypre_printf("\nUsing HYPRE_DEVELOP_STRING: %s (branch %s; the develop branch)\n\n",
                   HYPRE_DEVELOP_STRING, HYPRE_DEVELOP_BRANCH);

#elif defined(HYPRE_DEVELOP_STRING) && !defined(HYPRE_DEVELOP_BRANCH)
      hypre_printf("\nUsing HYPRE_DEVELOP_STRING: %s (branch %s; not the develop branch)\n\n",
                   HYPRE_DEVELOP_STRING, HYPRE_BRANCH_NAME);

#elif defined(HYPRE_RELEASE_VERSION)
      hypre_printf("\nUsing HYPRE_RELEASE_VERSION: %s\n\n",
                   HYPRE_RELEASE_VERSION);
#endif

      hypre_printf("Running with these driver parameters:\n");
      hypre_printf("  solver ID    = %d\n\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set various things
    *-----------------------------------------------------------*/

   HYPRE_SetPrintErrorMode(1);
   if (test_error == 1)
   {
      HYPRE_SetPrintErrorVerbosity(-1, 0);                   /* turn all errors off */
      HYPRE_SetPrintErrorVerbosity(HYPRE_ERROR_GENERIC, 1);  /* turn generic errors on */
   }

#if defined(HYPRE_USING_UMPIRE)
   /* Setup Umpire pools */
   HYPRE_SetUmpireDevicePoolName("HYPRE_DEVICE_POOL_TEST");
   HYPRE_SetUmpireUMPoolName("HYPRE_UM_POOL_TEST");
   HYPRE_SetUmpireHostPoolName("HYPRE_HOST_POOL_TEST");
   HYPRE_SetUmpirePinnedPoolName("HYPRE_PINNED_POOL_TEST");

   HYPRE_SetUmpireDevicePoolSize(umpire_dev_pool_size);
   HYPRE_SetUmpireUMPoolSize(umpire_uvm_pool_size);
   HYPRE_SetUmpireHostPoolSize(umpire_host_pool_size);
   HYPRE_SetUmpirePinnedPoolSize(umpire_pinned_pool_size);
#endif

#if defined(HYPRE_USING_MEMORY_TRACKER)
   hypre_MemoryTrackerSetPrint(print_mem_tracker);
   if (mem_tracker_name[0])
   {
      hypre_MemoryTrackerSetFileName(mem_tracker_name);
   }
#endif

   /* Set log level */
   HYPRE_SetLogLevel(log_level);

   /* default memory location */
   HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   HYPRE_SetExecutionPolicy(default_exec_policy);

#if defined(HYPRE_USING_GPU) && !defined(HYPRE_TEST_USING_HOST)
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
   HYPRE_SetGpuAwareMPI(gpu_aware_mpi);
#endif

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   if ( myid == 0 && dt != dt_inf)
   {
      hypre_printf("  Backward Euler time step with dt = %e\n", dt);
      hypre_printf("  Dirichlet 0 BCs are implicit in the spatial operator\n");
   }

   time_index = hypre_InitializeTiming("Spatial Operator");
   hypre_BeginTiming(time_index);
   if ( build_matrix_type == -2 )
   {
      ierr = HYPRE_IJMatrixReadBinary( argv[build_matrix_arg_index], comm,
                                       HYPRE_PARCSR, &ij_A );
      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in the system matrix!\n");
         hypre_MPI_Abort(comm, 1);
      }
   }
   else if ( build_matrix_type == -1 )
   {
      ierr = HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                                 HYPRE_PARCSR, &ij_A );
      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in the system matrix!\n");
         hypre_MPI_Abort(comm, 1);
      }
   }
   else if ( build_matrix_type == 0 )
   {
      BuildParFromFile(comm, argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParFromOneFile(comm, argc, argv, build_matrix_arg_index, num_functions,
                          &parcsr_A);
   }
   else if ( build_matrix_type == 2 )
   {
      BuildParLaplacian(comm, argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 3 )
   {
      BuildParLaplacian9pt(comm, argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 4 )
   {
      BuildParLaplacian27pt(comm, argc, argv, build_matrix_arg_index, &parcsr_A);

#if defined(HYPRE_USING_GPU)
      hypre_CSRMatrixSpMVAnalysisDevice(hypre_ParCSRMatrixDiag(parcsr_A));
#endif
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParLaplacian125pt(comm, argc, argv, build_matrix_arg_index, &parcsr_A);

#if defined(HYPRE_USING_GPU)
      hypre_CSRMatrixSpMVAnalysisDevice(hypre_ParCSRMatrixDiag(parcsr_A));
#endif
   }
   else if ( build_matrix_type == 6 )
   {
      BuildParDifConv(comm, argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 7 )
   {
      BuildParVarDifConv(comm, argc, argv, build_matrix_arg_index, &parcsr_A, &b);
      build_rhs_type      = 6;
      build_src_type      = 5;
   }
   else if ( build_matrix_type == 8 )
   {
      BuildParRotate7pt(comm, argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else
   {
      hypre_printf("You have asked for an unsupported problem with\n");
      hypre_printf("build_matrix_type = %d.\n", build_matrix_type);

      hypre_MPI_Abort(comm, 1);
   }

   /* BM Oct 23, 2006 */
   if (plot_grids)
   {
      if (build_matrix_type > 1 &&  build_matrix_type < 9)
         BuildParCoordinates (comm, argc, argv, build_matrix_arg_index,
                              &coord_dim, &coordinates);
      else
      {
         hypre_printf("Warning: coordinates are not yet printed for build_matrix_type = %d.\n",
                      build_matrix_type);
      }
   }

   if (build_matrix_type < 0)
   {
      ierr = HYPRE_IJMatrixGetLocalRange( ij_A,
                                          &first_local_row, &last_local_row,
                                          &first_local_col, &last_local_col );

      local_num_rows = (HYPRE_Int)(last_local_row - first_local_row + 1);
      local_num_cols = (HYPRE_Int)(last_local_col - first_local_col + 1);
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
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Generate Matrix", comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Set up the interp vector
    *-----------------------------------------------------------*/
   if (build_rbm)
   {
      char new_file_name[80];
      /* RHS */
      interp_vecs = hypre_CTAlloc(HYPRE_ParVector, num_interp_vecs, HYPRE_MEMORY_HOST);
      ij_rbm = hypre_CTAlloc(HYPRE_IJVector, num_interp_vecs, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_interp_vecs; i++)
      {
         hypre_sprintf(new_file_name, "%s.%d", argv[build_rbm_index], i);
         ierr = HYPRE_IJVectorRead( new_file_name, comm,
                                    HYPRE_PARCSR, &ij_rbm[i] );
         ierr = HYPRE_IJVectorGetObject( ij_rbm[i], &object );
         interp_vecs[i] = (HYPRE_ParVector) object;
      }
      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in rbm!\n");
         exit(1);
      }
   }

   /* Print matrix information */
   if (print_matrix_info)
   {
      HYPRE_BigInt global_num_rows, global_num_cols, global_num_nonzeros;

      if (parcsr_A)
      {
         HYPRE_BigInt ilower, iupper, jlower, jupper;

         HYPRE_ParCSRMatrixGetLocalRange(parcsr_A, &ilower, &iupper, &jlower, &jupper);
         HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &ij_A);
         HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
         hypre_IJMatrixObject(ij_A) = parcsr_A;
         hypre_IJMatrixAssembleFlag(ij_A) = 1;
      }

      HYPRE_IJMatrixGetGlobalInfo(ij_A,
                                  &global_num_rows,
                                  &global_num_cols,
                                  &global_num_nonzeros);

      if (parcsr_A)
      {
         hypre_IJMatrixObject(ij_A) = NULL;
         HYPRE_IJMatrixDestroy(ij_A);
      }

      if (myid == 0)
      {
         hypre_printf("  Matrix Information:\n");
         hypre_printf("    Global number of rows:     %b\n", global_num_rows);
         hypre_printf("    Global number of columns:  %b\n", global_num_cols);
         hypre_printf("    Global number of nonzeros: %b\n", global_num_nonzeros);
      }
   }

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/
   time_index = hypre_InitializeTiming("RHS and Initial Guess");
   hypre_BeginTiming(time_index);

   if (myid == 0)
   {
      hypre_printf("  Number of vector components: %d\n", num_components);
   }

   if (num_components > 1 && !(build_rhs_type > 1 && build_rhs_type < 6))
   {
      hypre_printf("num_components > 1 not implemented for this RHS choice!\n");
      hypre_MPI_Abort(comm, 1);
   }

   if (build_rhs_type == 0 || build_rhs_type == -2)
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
         if (build_x0_type == -1)
         {
            hypre_printf("  Initial guess is 0\n");
         }
      }

      /* RHS */
      if (!build_rhs_type)
      {
         ierr = HYPRE_IJVectorRead(argv[build_rhs_arg_index],
                                   comm, HYPRE_PARCSR, &ij_b);
      }
      else
      {
         ierr = HYPRE_IJVectorReadBinary(argv[build_rhs_arg_index],
                                         comm, HYPRE_PARCSR, &ij_b);
      }

      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in the right-hand-side!\n");
         exit(1);
      }
      HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorAssemble(ij_x);
      HYPRE_IJVectorGetObject(ij_x, &object);
      x = (HYPRE_ParVector) object;
   }
   else if (build_rhs_type == 1)
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
         if (build_x0_type == -1)
         {
            hypre_printf("  Initial guess is 0\n");
         }
      }

      ij_b = NULL;
      BuildRhsParFromOneFile(comm, argc, argv, build_rhs_arg_index, parcsr_A, &b);

      /* initial guess */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorAssemble(ij_x);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if (build_rhs_type == 2)
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector has unit coefficients\n");
         if (build_x0_type == -1)
         {
            hypre_printf("  Initial guess is 0\n");
         }
      }

      HYPRE_Complex *values_h = hypre_CTAlloc(HYPRE_Real, local_num_rows, HYPRE_MEMORY_HOST);
      HYPRE_Complex *values_d = hypre_CTAlloc(HYPRE_Real, local_num_rows, memory_location);
      for (i = 0; i < local_num_rows; i++)
      {
         values_h[i] = 1.0;
      }
      hypre_TMemcpy(values_d, values_h, HYPRE_Complex, local_num_rows,
                    memory_location, HYPRE_MEMORY_HOST);

      /* RHS */
      HYPRE_IJVectorCreate(comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorSetNumComponents(ij_b, num_components);
      HYPRE_IJVectorInitialize_v2(ij_b, memory_location);
      for (c = 0; c < num_components; c++)
      {
         HYPRE_IJVectorSetComponent(ij_b, c);
         HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values_d);
      }
      HYPRE_IJVectorAssemble(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      hypre_Memset(values_d, 0, (size_t)local_num_rows * sizeof(HYPRE_Complex), memory_location);
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorSetNumComponents(ij_x, num_components);
      HYPRE_IJVectorInitialize_v2(ij_x, memory_location);
      for (c = 0; c < num_components; c++)
      {
         HYPRE_IJVectorSetComponent(ij_x, c);
         HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      }
      HYPRE_IJVectorAssemble(ij_x);
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
      hypre_TFree(values_d, memory_location);
   }
   else if (build_rhs_type == 3)
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector has random coefficients and unit 2-norm\n");
         if (build_x0_type == -1)
         {
            hypre_printf("  Initial guess is 0\n");
         }
      }

      /* RHS */
      HYPRE_IJVectorCreate(comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorSetNumComponents(ij_b, num_components);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* For purposes of this test, HYPRE_ParVector functions are used, but
         these are not necessary.  For a clean use of the interface, the user
         "should" modify coefficients of ij_x by using functions
         HYPRE_IJVectorSetValues or HYPRE_IJVectorAddToValues */

      HYPRE_ParVectorSetRandomValues(b, 22775);
      HYPRE_ParVectorInnerProd(b, b, &norm);
      norm = 1. / hypre_sqrt(norm);
      ierr = HYPRE_ParVectorScale(norm, b);

      /* Initial guess */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetNumComponents(ij_x, num_components);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorAssemble(ij_x);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if (build_rhs_type == 4)
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector set for solution with unit coefficients\n");
         if (build_x0_type == -1)
         {
            hypre_printf("  Initial guess is 0\n");
         }
      }

      HYPRE_Real *values_h = hypre_CTAlloc(HYPRE_Real, local_num_cols, HYPRE_MEMORY_HOST);
      HYPRE_Real *values_d = hypre_CTAlloc(HYPRE_Real, local_num_cols, memory_location);
      for (i = 0; i < local_num_cols; i++)
      {
         values_h[i] = 1.;
      }
      hypre_TMemcpy(values_d, values_h, HYPRE_Real, local_num_cols,
                    memory_location, HYPRE_MEMORY_HOST);

      /* Temporary use of solution vector */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorSetNumComponents(ij_x, num_components);
      HYPRE_IJVectorInitialize(ij_x);
      for (c = 0; c < num_components; c++)
      {
         HYPRE_IJVectorSetComponent(ij_x, c);
         HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      }
      HYPRE_IJVectorAssemble(ij_x);
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
      hypre_TFree(values_d, memory_location);

      /* RHS */
      HYPRE_IJVectorCreate(comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorSetNumComponents(ij_b, num_components);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      HYPRE_ParCSRMatrixMatvec(1.0, parcsr_A, x, 0.0, b);

      /* Zero initial guess */
      hypre_IJVectorZeroValues(ij_x);
   }
   else if (build_rhs_type == 5)
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector is 0\n");
         hypre_printf("  Initial guess has unit coefficients\n");
      }

      HYPRE_Real *values_h = hypre_CTAlloc(HYPRE_Real, local_num_cols, HYPRE_MEMORY_HOST);
      HYPRE_Real *values_d = hypre_CTAlloc(HYPRE_Real, local_num_cols, memory_location);
      for (i = 0; i < local_num_cols; i++)
      {
         values_h[i] = 1.;
      }
      hypre_TMemcpy(values_d, values_h, HYPRE_Real, local_num_cols,
                    memory_location, HYPRE_MEMORY_HOST);

      /* RHS */
      HYPRE_IJVectorCreate(comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorSetNumComponents(ij_b, num_components);
      HYPRE_IJVectorInitialize(ij_b);
      HYPRE_IJVectorAssemble(ij_b);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetNumComponents(ij_x, num_components);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);
      for (c = 0; c < num_components; c++)
      {
         HYPRE_IJVectorSetComponent(ij_x, c);
         HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      }
      HYPRE_IJVectorAssemble(ij_x);
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
      hypre_TFree(values_d, memory_location);
   }
   else if (build_rhs_type == 6)
   {
      ij_b = NULL;
   }
   else if (build_rhs_type == 7)
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
         if (build_x0_type == -1)
         {
            hypre_printf("  Initial guess is 0\n");
         }
      }

      ij_b = NULL;
      ReadParVectorFromFile(comm, argc, argv, build_rhs_arg_index, &b);

      /* initial guess */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorAssemble(ij_x);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else
   {
      if (build_rhs_type != -1)
      {
         if (myid == 0)
         {
            hypre_printf("Error: Invalid build_rhs_type!\n");
         }
         hypre_MPI_Abort(comm, 1);
      }
   }

   if ( build_src_type == 0)
   {
      if (myid == 0)
      {
         hypre_printf("  Source vector read from file %s\n", argv[build_src_arg_index]);
         hypre_printf("  Initial unknown vector in evolution is 0\n");
      }

      ierr = HYPRE_IJVectorRead( argv[build_src_arg_index], comm,
                                 HYPRE_PARCSR, &ij_b );
      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in the right-hand-side!\n");
         exit(1);
      }
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial unknown vector */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorAssemble(ij_x);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if (build_src_type == 1)
   {
      BuildRhsParFromOneFile(comm, argc, argv, build_src_arg_index, parcsr_A, &b);
      ij_b = NULL;

      /* Initial unknown vector */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorAssemble(ij_x);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 2 )
   {
      if (myid == 0)
      {
         hypre_printf("  Source vector has unit coefficients\n");
         hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      HYPRE_Real *values_h = hypre_CTAlloc(HYPRE_Real, local_num_rows, HYPRE_MEMORY_HOST);
      HYPRE_Real *values_d = hypre_CTAlloc(HYPRE_Real, local_num_rows, memory_location);
      for (i = 0; i < local_num_rows; i++)
      {
         values_h[i] = 1.;
      }
      hypre_TMemcpy(values_d, values_h, HYPRE_Real, local_num_rows,
                    memory_location, HYPRE_MEMORY_HOST);

      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values_d);
      HYPRE_IJVectorAssemble(ij_b);
      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
      hypre_TFree(values_d, memory_location);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      HYPRE_IJVectorAssemble(ij_x);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 3 )
   {
      if (myid == 0)
      {
         hypre_printf("  Source vector has random coefficients in range 0 - 1\n");
         hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      HYPRE_Real *values_h = hypre_CTAlloc(HYPRE_Real, local_num_rows, HYPRE_MEMORY_HOST);
      HYPRE_Real *values_d = hypre_CTAlloc(HYPRE_Real, local_num_rows, memory_location);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values_h[i] = hypre_Rand();
      }
      hypre_TMemcpy(values_d, values_h, HYPRE_Real, local_num_rows, memory_location, HYPRE_MEMORY_HOST);

      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values_d);
      HYPRE_IJVectorAssemble(ij_b);
      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
      hypre_TFree(values_d, memory_location);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      HYPRE_IJVectorAssemble(ij_x);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 4 )
   {
      if (myid == 0)
      {
         hypre_printf("  Source vector is 0 \n");
         hypre_printf("  Initial unknown vector has random coefficients in range 0 - 1\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      HYPRE_Real *values_h = hypre_CTAlloc(HYPRE_Real, local_num_rows, HYPRE_MEMORY_HOST);
      HYPRE_Real *values_d = hypre_CTAlloc(HYPRE_Real, local_num_rows, memory_location);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values_h[i] = hypre_Rand() / dt;
      }
      hypre_TMemcpy(values_d, values_h, HYPRE_Real, local_num_rows, memory_location, HYPRE_MEMORY_HOST);

      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values_d);
      HYPRE_IJVectorAssemble(ij_b);
      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
      hypre_TFree(values_d, memory_location);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      values_h = hypre_CTAlloc(HYPRE_Real, local_num_cols, HYPRE_MEMORY_HOST);
      values_d = hypre_CTAlloc(HYPRE_Real, local_num_cols, memory_location);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
      {
         values_h[i] = hypre_Rand();
      }
      hypre_TMemcpy(values_d, values_h, HYPRE_Real, local_num_cols, memory_location, HYPRE_MEMORY_HOST);

      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      HYPRE_IJVectorAssemble(ij_x);
      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
      hypre_TFree(values_d, memory_location);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 5 )
   {
      if (myid == 0)
      {
         hypre_printf("  Initial guess is random \n");
      }

      /* Initial guess */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      HYPRE_Real *values_h = hypre_CTAlloc(HYPRE_Real, local_num_cols, HYPRE_MEMORY_HOST);
      HYPRE_Real *values_d = hypre_CTAlloc(HYPRE_Real, local_num_cols, memory_location);
      hypre_SeedRand(myid + 2747);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
      {
         values_h[i] = hypre_Rand();
      }
      hypre_TMemcpy(values_d, values_h, HYPRE_Real, local_num_cols, memory_location, HYPRE_MEMORY_HOST);

      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      HYPRE_IJVectorAssemble(ij_x);
      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
      hypre_TFree(values_d, memory_location);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }

   /* initial guess */
   if (build_x0_type == 0 || build_x0_type == -2)
   {
      /* from file */
      if (myid == 0)
      {
         hypre_printf("  Initial guess vector read from file %s\n", argv[build_x0_arg_index]);
      }

      /* x0 */
      if (ij_x)
      {
         HYPRE_IJVectorDestroy(ij_x);
      }

      if (!build_x0_type)
      {
         ierr = HYPRE_IJVectorRead(argv[build_x0_arg_index],
                                   comm, HYPRE_PARCSR, &ij_x);
      }
      else
      {
         ierr = HYPRE_IJVectorReadBinary(argv[build_x0_arg_index],
                                         comm, HYPRE_PARCSR, &ij_x);
      }

      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in x0!\n");
         exit(1);
      }
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if (build_x0_type == -3)
   {
      if (myid == 0)
      {
         hypre_printf("  Initial guess vector read from file %s\n", argv[build_x0_arg_index]);
      }

      if (ij_x)
      {
         HYPRE_IJVectorDestroy(ij_x);
      }
      ij_x = NULL;

      BuildSolParFromOneFile(comm, argc, argv, build_x0_arg_index, parcsr_A, &x);
   }
   else if (build_x0_type == 7)
   {
      /* from file */
      if (myid == 0)
      {
         hypre_printf("  Initial guess vector read from file %s\n", argv[build_x0_arg_index]);
      }

      ReadParVectorFromFile(comm, argc, argv, build_x0_arg_index, &x);
   }
   else if (build_x0_type == 1)
   {
      /* random */
      if (myid == 0)
      {
         hypre_printf("  Initial guess is random \n");
      }

      if (ij_x)
      {
         HYPRE_IJVectorDestroy(ij_x);
      }

      /* Initial guess */
      HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      HYPRE_Real *values_h = hypre_CTAlloc(HYPRE_Real, local_num_cols, HYPRE_MEMORY_HOST);
      HYPRE_Real *values_d = hypre_CTAlloc(HYPRE_Real, local_num_cols, memory_location);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
      {
         values_h[i] = hypre_Rand();
      }
      hypre_TMemcpy(values_d, values_h, HYPRE_Real, local_num_cols, memory_location, HYPRE_MEMORY_HOST);

      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      HYPRE_IJVectorAssemble(ij_x);
      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
      hypre_TFree(values_d, memory_location);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }

   /* Setup dof_func array if needed */
   if (num_functions > 1)
   {
      if (build_funcs_type == 1)
      {
         if (myid == 0)
         {
            hypre_printf("  Calling BuildFuncTagsFromOneFile\n");
         }
         BuildFuncTagsFromOneFile(comm, argc, argv, build_funcs_arg_index, parcsr_A, &dof_func);
      }
      else if (build_funcs_type == 2)
      {
         if (myid == 0)
         {
            hypre_printf("  Calling BuildFuncTagsFromFiles\n");
         }
         BuildFuncTagsFromFiles(argc, argv, build_funcs_arg_index, parcsr_A, &dof_func);
      }
      else if (build_funcs_type == 3)
      {
         if (myid == 0)
         {
            hypre_printf("  Calling BuildFuncTagsInterleaved with num_functions = %d\n", num_functions);
         }
         BuildFuncTagsInterleaved(local_num_rows, num_functions, memory_location, &dof_func);
      }
      else if (build_funcs_type == 4)
      {
         if (myid == 0)
         {
            hypre_printf("  Calling BuildFuncTagsContiguous with num_functions = %d\n", num_functions);
         }
         BuildFuncTagsContiguous(local_num_rows, num_functions, memory_location, &dof_func);
      }
      else
      {
         hypre_printf ("  Number of functions = %d \n", num_functions);
      }

      if (dof_func)
      {
         HYPRE_IJVectorSetTags(ij_x, 0, num_functions, dof_func);
         HYPRE_IJVectorSetTags(ij_b, 0, num_functions, dof_func);
      }
   }

   /*-----------------------------------------------------------
    * Finalize IJVector Setup timings
    *-----------------------------------------------------------*/

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Vector Setup", comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (negA)
   {
      hypre_ParCSRMatrixScale(parcsr_A, -1);
   }

   if (print_system)
   {
      if (ij_A)
      {
         HYPRE_IJMatrixPrint(ij_A, "IJ.out.A");
      }
      else if (parcsr_A)
      {
         hypre_ParCSRMatrixPrintIJ(parcsr_A, 0, 0, "IJ.out.A");
      }
      else
      {
         if (!myid)
         {
            hypre_printf(" Matrix A not found!\n");
         }
      }

      if (ij_b)
      {
         HYPRE_IJVectorPrint(ij_b, "IJ.out.b");
      }
      else if (b)
      {
         HYPRE_ParVectorPrint(b, "ParVec.out.b");
      }
      HYPRE_IJVectorPrint(ij_x, "IJ.out.x0");
   }

   if (print_system_binary)
   {
      if (ij_A)
      {
         HYPRE_IJMatrixPrintBinary(ij_A, "IJ.out.A");
      }
      else
      {
         hypre_ParCSRMatrixPrintBinaryIJ(parcsr_A, 0, 0, "IJ.out.A");
      }

      if (ij_b)
      {
         HYPRE_IJVectorPrintBinary(ij_b, "IJ.out.b");
      }
      else if (b)
      {
         HYPRE_ParVectorPrintBinaryIJ(b, "IJ.out.b");
      }

      if (ij_x)
      {
         HYPRE_IJVectorPrintBinary(ij_x, "IJ.out.x0");
      }
      else if (x)
      {
         HYPRE_ParVectorPrintBinaryIJ(x, "IJ.out.x0");
      }
   }

   if (print_system_csr)
   {
      if (parcsr_A)
      {
         hypre_ParCSRMatrixPrint(parcsr_A, "csr.out.A");
      }
      else
      {
         if (!myid)
         {
            hypre_printf(" Matrix A in parcsr format not found!\n");
         }
      }

      if (b)
      {
         HYPRE_ParVectorPrint(b, "csr.out.b");
      }

      if (x)
      {
         HYPRE_ParVectorPrint(x, "csr.out.x0");
      }
   }

   /*-----------------------------------------------------------
    * Migrate the system to the wanted memory space
    *-----------------------------------------------------------*/

   hypre_ParCSRMatrixMigrate(parcsr_A, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(b, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(x, hypre_HandleMemoryLocation(hypre_handle()));

   /* clone data for solver and preconditioner */
   A_slvr = HYPRE_ParCSRMatrixClone_mp(parcsr_A, solver_precision);
   A_pc = HYPRE_ParCSRMatrixClone_mp(parcsr_A, precond_precision);
   x_slvr = HYPRE_ParVectorClone_mp(x, solver_precision);
   x_pc = HYPRE_ParVectorClone_mp(x, precond_precision);
   b_slvr = HYPRE_ParVectorClone_mp(b, solver_precision);
   b_pc = HYPRE_ParVectorClone_mp(b, precond_precision);

   if (benchmark)
   {
      poutdat = 0;
      second_time = 1;
   }

   /* save the initial guess for the 2nd time */
   if (second_time)
   {
      x0_save = HYPRE_ParVectorClone_mp(x, solver_precision);
   }

   /* Compute RHS squared norm */
   if (ij_b)
   {
      HYPRE_IJVectorInnerProd(ij_b, ij_b, &b_dot_b);
   }
   else if (b)
   {
      HYPRE_ParVectorInnerProd(b, b, &b_dot_b);
   }
   else
   {
      if (!myid)
      {
         hypre_printf(" Error: Vector b not set!\n");
      }
      hypre_MPI_Abort(comm, 1);
   }
   
   /* Create and set options for AMG preconditioner */
   if(solver_id == 1 || solver_id == 11 || solver_id == 21 || solver_id == 31)
   {
      HYPRE_BoomerAMGCreate_pre(precond_precision, &pcg_precond);
      /* BM Aug 25, 2006 */
      HYPRE_BoomerAMGSetCGCIts_pre(precond_precision, pcg_precond, cgcits);
      HYPRE_BoomerAMGSetInterpType_pre(precond_precision, pcg_precond, interp_type);
      HYPRE_BoomerAMGSetPostInterpType_pre(precond_precision, pcg_precond, post_interp_type);
      HYPRE_BoomerAMGSetTol_pre(precond_precision, pcg_precond, pc_tol);
      HYPRE_BoomerAMGSetCoarsenType_pre(precond_precision, pcg_precond, coarsen_type);
      HYPRE_BoomerAMGSetCoarsenCutFactor_pre(precond_precision, pcg_precond, coarsen_cut_factor);
      HYPRE_BoomerAMGSetMeasureType_pre(precond_precision, pcg_precond, measure_type);
      HYPRE_BoomerAMGSetStrongThreshold_pre(precond_precision, pcg_precond, strong_threshold);
      HYPRE_BoomerAMGSetSeqThreshold_pre(precond_precision, pcg_precond, seq_threshold);
      HYPRE_BoomerAMGSetRedundant_pre(precond_precision, pcg_precond, redundant);
      HYPRE_BoomerAMGSetMaxCoarseSize_pre(precond_precision, pcg_precond, coarse_threshold);
      HYPRE_BoomerAMGSetMinCoarseSize_pre(precond_precision, pcg_precond, min_coarse_size);
      HYPRE_BoomerAMGSetTruncFactor_pre(precond_precision, pcg_precond, trunc_factor);
      HYPRE_BoomerAMGSetPMaxElmts_pre(precond_precision, pcg_precond, P_max_elmts);
      HYPRE_BoomerAMGSetJacobiTruncThreshold_pre(precond_precision, pcg_precond, jacobi_trunc_threshold);
      HYPRE_BoomerAMGSetSCommPkgSwitch_pre(precond_precision, pcg_precond, S_commpkg_switch);
      HYPRE_BoomerAMGSetPrintLevel_pre(precond_precision, pcg_precond, poutdat);
      HYPRE_BoomerAMGSetPrintFileName_pre(precond_precision, pcg_precond, "driver.out.log");
      HYPRE_BoomerAMGSetMaxIter_pre(precond_precision, pcg_precond, precon_cycles);
      HYPRE_BoomerAMGSetCycleType_pre(precond_precision, pcg_precond, cycle_type);
      HYPRE_BoomerAMGSetFCycle_pre(precond_precision, pcg_precond, fcycle);
      HYPRE_BoomerAMGSetNumSweeps_pre(precond_precision, pcg_precond, num_sweeps);
      HYPRE_BoomerAMGSetISType_pre(precond_precision, pcg_precond, IS_type);
      if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_pre(precond_precision, pcg_precond, relax_type); }
      if (relax_down > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType_pre(precond_precision, pcg_precond, relax_down, 1);
      }
      if (relax_up > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType_pre(precond_precision, pcg_precond, relax_up, 2);
      }
      if (relax_coarse > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType_pre(precond_precision, pcg_precond, relax_coarse, 3);
      }
      HYPRE_BoomerAMGSetAddRelaxType_pre(precond_precision, pcg_precond, add_relax_type);
      HYPRE_BoomerAMGSetAddRelaxWt_pre(precond_precision, pcg_precond, add_relax_wt);
      HYPRE_BoomerAMGSetChebyOrder_pre(precond_precision, pcg_precond, cheby_order);
      HYPRE_BoomerAMGSetChebyFraction_pre(precond_precision, pcg_precond, cheby_fraction);
      HYPRE_BoomerAMGSetChebyEigEst_pre(precond_precision, pcg_precond, cheby_eig_est);
      HYPRE_BoomerAMGSetChebyVariant_pre(precond_precision, pcg_precond, cheby_variant);
      HYPRE_BoomerAMGSetChebyScale_pre(precond_precision, pcg_precond, cheby_scale);
      HYPRE_BoomerAMGSetRelaxOrder_pre(precond_precision, pcg_precond, relax_order);
      HYPRE_BoomerAMGSetRelaxWt_pre(precond_precision, pcg_precond, relax_wt);
      HYPRE_BoomerAMGSetOuterWt_pre(precond_precision, pcg_precond, outer_wt);
      if (level_w > -1)
      {
         HYPRE_BoomerAMGSetLevelRelaxWt_pre(precond_precision, pcg_precond, relax_wt_level, level_w);
      }
      if (level_ow > -1)
      {
         HYPRE_BoomerAMGSetLevelOuterWt_pre(precond_precision, pcg_precond, outer_wt_level, level_ow);
      }
      HYPRE_BoomerAMGSetSmoothType_pre(precond_precision, pcg_precond, smooth_type);
      HYPRE_BoomerAMGSetSmoothNumLevels_pre(precond_precision, pcg_precond, smooth_num_levels);
      HYPRE_BoomerAMGSetSmoothNumSweeps_pre(precond_precision, pcg_precond, smooth_num_sweeps);
      HYPRE_BoomerAMGSetMaxLevels_pre(precond_precision, pcg_precond, max_levels);
      HYPRE_BoomerAMGSetMaxRowSum_pre(precond_precision, pcg_precond, max_row_sum);
      HYPRE_BoomerAMGSetDebugFlag_pre(precond_precision, pcg_precond, debug_flag);
      HYPRE_BoomerAMGSetFilterFunctions_pre(precond_precision, pcg_precond, filter_functions);
      HYPRE_BoomerAMGSetNumFunctions_pre(precond_precision, pcg_precond, num_functions);
      HYPRE_BoomerAMGSetAggNumLevels_pre(precond_precision, pcg_precond, agg_num_levels);
      HYPRE_BoomerAMGSetAggInterpType_pre(precond_precision, pcg_precond, agg_interp_type);
      HYPRE_BoomerAMGSetAggTruncFactor_pre(precond_precision, pcg_precond, agg_trunc_factor);
      HYPRE_BoomerAMGSetAggP12TruncFactor_pre(precond_precision, pcg_precond, agg_P12_trunc_factor);
      HYPRE_BoomerAMGSetAggPMaxElmts_pre(precond_precision, pcg_precond, agg_P_max_elmts);
      HYPRE_BoomerAMGSetAggP12MaxElmts_pre(precond_precision, pcg_precond, agg_P12_max_elmts);
      HYPRE_BoomerAMGSetNumPaths_pre(precond_precision, pcg_precond, num_paths);
      HYPRE_BoomerAMGSetNodal_pre(precond_precision, pcg_precond, nodal);
      HYPRE_BoomerAMGSetNodalDiag_pre(precond_precision, pcg_precond, nodal_diag);
      HYPRE_BoomerAMGSetKeepSameSign_pre(precond_precision, pcg_precond, keep_same_sign);
      HYPRE_BoomerAMGSetVariant_pre(precond_precision, pcg_precond, variant);
      HYPRE_BoomerAMGSetOverlap_pre(precond_precision, pcg_precond, overlap);
      HYPRE_BoomerAMGSetDomainType_pre(precond_precision, pcg_precond, domain_type);
      HYPRE_BoomerAMGSetSchwarzUseNonSymm_pre(precond_precision, pcg_precond, use_nonsymm_schwarz);
      HYPRE_BoomerAMGSetSchwarzRlxWeight_pre(precond_precision, pcg_precond, schwarz_rlx_weight);
      HYPRE_BoomerAMGSetCycleNumSweeps_pre(precond_precision, pcg_precond, ns_coarse, 3);
      HYPRE_BoomerAMGSetAdditive_pre(precond_precision, pcg_precond, additive);
      HYPRE_BoomerAMGSetMultAdditive_pre(precond_precision, pcg_precond, mult_add);
      HYPRE_BoomerAMGSetSimple_pre(precond_precision, pcg_precond, simple);
      HYPRE_BoomerAMGSetAddLastLvl_pre(precond_precision, pcg_precond, add_last_lvl);
      HYPRE_BoomerAMGSetMultAddPMaxElmts_pre(precond_precision, pcg_precond, add_P_max_elmts);
      HYPRE_BoomerAMGSetMultAddTruncFactor_pre(precond_precision, pcg_precond, add_trunc_factor);
      HYPRE_BoomerAMGSetRAP2_pre(precond_precision, pcg_precond, rap2);
      HYPRE_BoomerAMGSetModuleRAP2_pre(precond_precision, pcg_precond, mod_rap2);
      HYPRE_BoomerAMGSetKeepTranspose_pre(precond_precision, pcg_precond, keepTranspose);
#ifdef HYPRE_USING_DSUPERLU
      HYPRE_BoomerAMGSetDSLUThreshold_pre(precond_precision, pcg_precond, dslu_threshold);
#endif
      if (nongalerk_tol)
      {
         HYPRE_BoomerAMGSetNonGalerkinTol_pre(precond_precision, pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
         for (i = 0; i < nongalerk_num_tol - 1; i++)
         {
            HYPRE_BoomerAMGSetLevelNonGalerkinTol_pre(precond_precision, pcg_precond, nongalerk_tol[i], i);
         }
      }
      if (build_rbm)
      {
         HYPRE_BoomerAMGSetInterpVectors_pre(precond_precision, pcg_precond, num_interp_vecs, interp_vecs);
         HYPRE_BoomerAMGSetInterpVecVariant_pre(precond_precision, pcg_precond, interp_vec_variant);
         HYPRE_BoomerAMGSetInterpVecQMax_pre(precond_precision, pcg_precond, Q_max);
         HYPRE_BoomerAMGSetInterpVecAbsQTrunc_pre(precond_precision, pcg_precond, Q_trunc);
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   if (solver_id == 0  || solver_id == 1)
   {
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Run-1");
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRPCGCreate_pre(solver_precision, comm, &pcg_solver);
      HYPRE_PCGSetMaxIter_pre(solver_precision, pcg_solver, max_iter);
      HYPRE_PCGSetTol_pre(solver_precision, pcg_solver, tol);
      HYPRE_PCGSetTwoNorm_pre(solver_precision, pcg_solver, 1);
      HYPRE_PCGSetRelChange_pre(solver_precision, pcg_solver, rel_change);
      HYPRE_PCGSetPrintLevel_pre(solver_precision, pcg_solver, ioutdat);
      HYPRE_PCGSetAbsoluteTol_pre(solver_precision, pcg_solver, atol);
      HYPRE_PCGSetRecomputeResidual_pre(solver_precision, pcg_solver, recompute_res);

      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { 
            if(solver_precision == precond_precision)
            {
               if(solver_precision == HYPRE_REAL_SINGLE)
               {
                  hypre_printf("Solver: SINGLE PRECISION AMG-PCG\n"); 
               }
               else if(solver_precision == HYPRE_REAL_DOUBLE)
               {
                  hypre_printf("Solver: DOUBLE PRECISION AMG-PCG\n"); 
               }
               else
               {
                  hypre_printf("Solver: LONG DOUBLE PRECISION AMG-PCG\n"); 
               }             
            }
            else
            {
               hypre_printf("Solver: MIXED PRECISION AMG-PCG\n"); 
            }
         }
         /* Set the preconditioning matrix */
         HYPRE_PCGSetPrecondMatrix_pre( solver_precision, (HYPRE_Solver)pcg_solver, (HYPRE_Matrix)A_pc);
         HYPRE_PCGSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_mp,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_mp,
                             pcg_precond);                         
      }
      else if (solver_id == 0)
      {
         /* use diagonal scaling as preconditioner */
         pcg_precond = NULL;
         if (myid == 0) {
            if(solver_precision == HYPRE_REAL_SINGLE)
            {
               hypre_printf("Solver: SINGLE PRECISION DS-PCG\n"); 
               HYPRE_PCGSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_flt,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_flt,
                             pcg_precond);
            }
            else if(solver_precision == HYPRE_REAL_DOUBLE)
            {
               hypre_printf("Solver: DOUBLE PRECISION DS-PCG\n"); 
               HYPRE_PCGSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_dbl,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_dbl,
                             pcg_precond);
            }
            else
            {
               hypre_printf("Solver: LONG DOUBLE PRECISION DS-PCG\n"); 
               HYPRE_PCGSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_long_dbl,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_long_dbl,
                             pcg_precond);
            }
         }
      }
      HYPRE_PCGGetPrecond_pre(solver_precision, pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
         hypre_printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         hypre_printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");
      }

      hypre_GpuProfilingPushRange("PCG-Setup-1");
      HYPRE_PCGSetup_pre(solver_precision, pcg_solver, (HYPRE_Matrix) A_slvr,
                     (HYPRE_Vector) b_slvr, (HYPRE_Vector) x_slvr);
      hypre_GpuProfilingPopRange();
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);
      hypre_GpuProfilingPushRange("PCG-Solve-1");
      HYPRE_PCGSolve_pre(solver_precision, pcg_solver, (HYPRE_Matrix)A_slvr,
                     (HYPRE_Vector)b_slvr, (HYPRE_Vector)x_slvr);
      hypre_GpuProfilingPopRange();
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      HYPRE_ANNOTATE_REGION_END("%s", "Run-1");

      if (second_time)
      {
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Run-2");
         HYPRE_SetExecutionPolicy(exec2_policy);

         /* run a second time [for timings, to check for memory leaks] */
         HYPRE_ParVectorSetRandomValues_pre(solver_precision, x_slvr, 775);
#if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND)
         hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         HYPRE_ParVectorCopy_mp(x0_save, x_slvr);

#if defined(HYPRE_USING_CUDA)
         cudaProfilerStart();
#endif

         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);

         hypre_GpuProfilingPushRange("PCG-Setup-2");

         HYPRE_PCGSetup_pre(solver_precision, pcg_solver, (HYPRE_Matrix) A_slvr,
                        (HYPRE_Vector) b_slvr, (HYPRE_Vector) x_slvr);

         hypre_GpuProfilingPopRange();

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("PCG Solve");
         hypre_BeginTiming(time_index);

         hypre_GpuProfilingPushRange("PCG-Solve-2");

         HYPRE_PCGSolve_pre(solver_precision, pcg_solver, (HYPRE_Matrix)A_slvr,
                        (HYPRE_Vector)b_slvr, (HYPRE_Vector)x_slvr);

         hypre_GpuProfilingPopRange();

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_ANNOTATE_REGION_END("%s", "Run-2");
#if defined(HYPRE_USING_CUDA)
         cudaProfilerStop();
#endif
      }

      HYPRE_PCGGetNumIterations_pre(solver_precision, pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm_pre(solver_precision, pcg_solver, final_res_norm);
      HYPRE_ParCSRPCGDestroy_pre(solver_precision, pcg_solver);

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("PCG Iterations = %d\n", num_iterations);
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if (solver_id == 10 || solver_id == 11)
   {
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Run-1");
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRGMRESCreate_pre(solver_precision, comm, &pcg_solver);
      HYPRE_GMRESSetKDim_pre(solver_precision, pcg_solver, k_dim);
      HYPRE_GMRESSetMaxIter_pre(solver_precision, pcg_solver, max_iter);
      HYPRE_GMRESSetTol_pre(solver_precision, pcg_solver, tol);
      HYPRE_GMRESSetAbsoluteTol_pre(solver_precision, pcg_solver, atol);
      HYPRE_GMRESSetLogging_pre(solver_precision, pcg_solver, 1);
      HYPRE_GMRESSetPrintLevel_pre(solver_precision, pcg_solver, ioutdat);
      HYPRE_GMRESSetRelChange_pre(solver_precision, pcg_solver, rel_change);

      if (solver_id == 11)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { 
            if(solver_precision == precond_precision)
            {
               if(solver_precision == HYPRE_REAL_SINGLE)
               {
                  hypre_printf("Solver: SINGLE PRECISION AMG-GMRES\n"); 
               }
               else if(solver_precision == HYPRE_REAL_DOUBLE)
               {
                  hypre_printf("Solver: DOUBLE PRECISION AMG-GMRES\n"); 
               }
               else
               {
                  hypre_printf("Solver: LONG DOUBLE PRECISION AMG-GMRES\n"); 
               }             
            }
            else
            {
               hypre_printf("Solver: MIXED PRECISION AMG-GMRES\n"); 
            }
         }
         /* Set the preconditioning matrix */
         HYPRE_GMRESSetPrecondMatrix_pre(solver_precision, (HYPRE_Solver)pcg_solver, (HYPRE_Matrix)A_pc);
         HYPRE_GMRESSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_mp,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_mp,
                             pcg_precond);                         
      }
      else if (solver_id == 10)
      {
         /* use diagonal scaling as preconditioner */
         pcg_precond = NULL;
         if (myid == 0) {
            if(solver_precision == HYPRE_REAL_SINGLE)
            {
               hypre_printf("Solver: SINGLE PRECISION DS-GMRES\n"); 
               HYPRE_GMRESSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_flt,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_flt,
                             pcg_precond);
            }
            else if(solver_precision == HYPRE_REAL_DOUBLE)
            {
               hypre_printf("Solver: DOUBLE PRECISION DS-GMRES\n"); 
               HYPRE_GMRESSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_dbl,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_dbl,
                             pcg_precond);
            }
            else
            {
               hypre_printf("Solver: LONG DOUBLE PRECISION DS-GMRES\n"); 
               HYPRE_GMRESSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_long_dbl,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_long_dbl,
                             pcg_precond);
            }
         }
      }
      HYPRE_GMRESGetPrecond_pre(solver_precision, pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
         hypre_printf("HYPRE_GMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         hypre_printf("HYPRE_GMRESGetPrecond got good precond\n");
      }

      HYPRE_GMRESSetup_pre(solver_precision, pcg_solver, (HYPRE_Matrix)A_slvr, (HYPRE_Vector)b_slvr, (HYPRE_Vector)x_slvr);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve_pre(solver_precision, pcg_solver, (HYPRE_Matrix)A_slvr, (HYPRE_Vector)b_slvr, (HYPRE_Vector)x_slvr);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      HYPRE_ANNOTATE_REGION_END("%s", "Run-1");

      if (second_time)
      {
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Run-2");
         /* run a second time [for timings, to check for memory leaks] */
         HYPRE_ParVectorSetRandomValues_pre(solver_precision, x_slvr, 775);
#if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND)
         hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
        HYPRE_ParVectorCopy_mp(x0_save, x_slvr);

         HYPRE_GMRESSetup_pre(solver_precision, pcg_solver,
                          (HYPRE_Matrix) A_slvr,
                          (HYPRE_Vector) b_slvr,
                          (HYPRE_Vector) x_slvr);
         HYPRE_GMRESSolve_pre(solver_precision, pcg_solver,
                          (HYPRE_Matrix) A_slvr,
                          (HYPRE_Vector) b_slvr,
                          (HYPRE_Vector) x_slvr);
         HYPRE_ANNOTATE_REGION_END("%s", "Run-2");
      }
   
      HYPRE_GMRESGetNumIterations_pre(solver_precision, pcg_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm_pre(solver_precision, pcg_solver, final_res_norm);
      HYPRE_ParCSRGMRESDestroy_pre(solver_precision, pcg_solver);

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("GMRES Iterations = %d\n", num_iterations);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using FlexGMRES
    *-----------------------------------------------------------*/

   if (solver_id == 20 || solver_id == 21 )
   {
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Run-1");
      time_index = hypre_InitializeTiming("FlexGMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRFlexGMRESCreate_pre(solver_precision, comm, &pcg_solver);
      HYPRE_FlexGMRESSetKDim_pre(solver_precision, pcg_solver, k_dim);
      HYPRE_FlexGMRESSetMaxIter_pre(solver_precision, pcg_solver, max_iter);
      HYPRE_FlexGMRESSetTol_pre(solver_precision, pcg_solver, tol);
      HYPRE_FlexGMRESSetAbsoluteTol_pre(solver_precision, pcg_solver, atol);
      HYPRE_FlexGMRESSetLogging_pre(solver_precision, pcg_solver, 1);
      HYPRE_FlexGMRESSetPrintLevel_pre(solver_precision, pcg_solver, ioutdat);

      if (solver_id == 21)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { 
            if(solver_precision == precond_precision)
            {
               if(solver_precision == HYPRE_REAL_SINGLE)
               {
                  hypre_printf("Solver: SINGLE PRECISION AMG-FlexGMRES\n"); 
               }
               else if(solver_precision == HYPRE_REAL_DOUBLE)
               {
                  hypre_printf("Solver: DOUBLE PRECISION AMG-FlexGMRES\n"); 
               }
               else
               {
                  hypre_printf("Solver: LONG DOUBLE PRECISION AMG-FlexGMRES\n"); 
               }             
            }
            else
            {
               hypre_printf("Solver: MIXED PRECISION AMG-FlexGMRES\n"); 
            }
         }
         /* Set the preconditioning matrix */
         HYPRE_FlexGMRESSetPrecondMatrix_pre(solver_precision, (HYPRE_Solver)pcg_solver, (HYPRE_Matrix)A_pc);
         HYPRE_FlexGMRESSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_mp,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_mp,
                             pcg_precond);
      }
      else if (solver_id == 20)
      {
         /* use diagonal scaling as preconditioner */
         pcg_precond = NULL;
         if (myid == 0) {
            if(solver_precision == HYPRE_REAL_SINGLE)
            {
               hypre_printf("Solver: SINGLE PRECISION DS-FlexGMRES\n"); 
               HYPRE_FlexGMRESSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_flt,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_flt,
                             pcg_precond);
            }
            else if(solver_precision == HYPRE_REAL_DOUBLE)
            {
               hypre_printf("Solver: DOUBLE PRECISION DS-FlexGMRES\n"); 
               HYPRE_FlexGMRESSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_dbl,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_dbl,
                             pcg_precond);
            }
            else
            {
               hypre_printf("Solver: LONG DOUBLE PRECISION DS-FlexGMRES\n"); 
               HYPRE_FlexGMRESSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_long_dbl,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_long_dbl,
                             pcg_precond);
            }
         }
      }

      HYPRE_FlexGMRESGetPrecond_pre(solver_precision, pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
         hypre_printf("HYPRE_FlexGMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         hypre_printf("HYPRE_FlexGMRESGetPrecond got good precond\n");
      }

      HYPRE_FlexGMRESSetup_pre
      (solver_precision, pcg_solver, (HYPRE_Matrix)A_slvr, (HYPRE_Vector)b_slvr, (HYPRE_Vector)x_slvr);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("FlexGMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_FlexGMRESSolve_pre
      (solver_precision, pcg_solver, (HYPRE_Matrix)A_slvr, (HYPRE_Vector)b_slvr, (HYPRE_Vector)x_slvr);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      HYPRE_ANNOTATE_REGION_END("%s", "Run-1");

      if (second_time)
      {
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Run-2");
         /* run a second time [for timings, to check for memory leaks] */
         HYPRE_ParVectorSetRandomValues_pre(solver_precision, x_slvr, 775);
#if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND)
         hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
        HYPRE_ParVectorCopy_mp(x0_save, x_slvr);

         HYPRE_FlexGMRESSetup_pre(solver_precision, pcg_solver,
                          (HYPRE_Matrix) A_slvr,
                          (HYPRE_Vector) b_slvr,
                          (HYPRE_Vector) x_slvr);
         HYPRE_FlexGMRESSolve_pre(solver_precision, pcg_solver,
                          (HYPRE_Matrix) A_slvr,
                          (HYPRE_Vector) b_slvr,
                          (HYPRE_Vector) x_slvr);
         HYPRE_ANNOTATE_REGION_END("%s", "Run-2");
      }

      HYPRE_FlexGMRESGetNumIterations_pre(solver_precision, pcg_solver, &num_iterations);
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm_pre(solver_precision, pcg_solver, final_res_norm);
      HYPRE_ParCSRFlexGMRESDestroy_pre(solver_precision, pcg_solver);

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("FlexGMRES Iterations = %d\n", num_iterations);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGSTAB
    *-----------------------------------------------------------*/

   if (solver_id == 30 || solver_id == 31 )
   {
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Run-1");
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRBiCGSTABCreate_pre(solver_precision, comm, &pcg_solver);
      HYPRE_BiCGSTABSetMaxIter_pre(solver_precision, pcg_solver, max_iter);
      HYPRE_BiCGSTABSetTol_pre(solver_precision, pcg_solver, tol);
      HYPRE_BiCGSTABSetAbsoluteTol_pre(solver_precision, pcg_solver, atol);
      HYPRE_BiCGSTABSetLogging_pre(solver_precision, pcg_solver, ioutdat);
      HYPRE_BiCGSTABSetPrintLevel_pre(solver_precision, pcg_solver, ioutdat);

      if (solver_id == 31)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { 
            if(solver_precision == precond_precision)
            {
               if(solver_precision == HYPRE_REAL_SINGLE)
               {
                  hypre_printf("Solver: SINGLE PRECISION AMG-BiCGSTAB\n"); 
               }
               else if(solver_precision == HYPRE_REAL_DOUBLE)
               {
                  hypre_printf("Solver: DOUBLE PRECISION AMG-BiCGSTAB\n"); 
               }
               else
               {
                  hypre_printf("Solver: LONG DOUBLE PRECISION AMG-BiCGSTAB\n"); 
               }             
            }
            else
            {
               hypre_printf("Solver: MIXED PRECISION AMG-BiCGSTAB\n"); 
            }
         }
         /* Set the preconditioning matrix */
         HYPRE_BiCGSTABSetPrecondMatrix_pre(solver_precision, (HYPRE_Solver)pcg_solver, (HYPRE_Matrix)A_pc);
         HYPRE_BiCGSTABSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_mp,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_mp,
                             pcg_precond);
      }
      else if (solver_id == 30)
      {
         /* use diagonal scaling as preconditioner */
         pcg_precond = NULL;
         if (myid == 0) {
            if(solver_precision == HYPRE_REAL_SINGLE)
            {
               hypre_printf("Solver: SINGLE PRECISION DS-BiCGSTAB\n"); 
               HYPRE_BiCGSTABSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_flt,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_flt,
                             pcg_precond);
            }
            else if(solver_precision == HYPRE_REAL_DOUBLE)
            {
               hypre_printf("Solver: DOUBLE PRECISION DS-BiCGSTAB\n"); 
               HYPRE_BiCGSTABSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_dbl,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_dbl,
                             pcg_precond);
            }
            else
            {
               hypre_printf("Solver: LONG DOUBLE PRECISION DS-BiCGSTAB\n"); 
               HYPRE_BiCGSTABSetPrecond_pre(solver_precision, pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale_long_dbl,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup_long_dbl,
                             pcg_precond);
            }
         }
      }

      HYPRE_BiCGSTABGetPrecond_pre(solver_precision, pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
         hypre_printf("HYPRE_BiCGSTABGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         hypre_printf("HYPRE_BiCGSTABGetPrecond got good precond\n");
      }
      
      HYPRE_BiCGSTABSetup_pre
      (solver_precision, pcg_solver, (HYPRE_Matrix)A_slvr, (HYPRE_Vector)b_slvr, (HYPRE_Vector)x_slvr);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve_pre
      (solver_precision, pcg_solver, (HYPRE_Matrix)A_slvr, (HYPRE_Vector)b_slvr, (HYPRE_Vector)x_slvr);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      HYPRE_ANNOTATE_REGION_END("%s", "Run-1");

      if (second_time)
      {
         /* run a second time [for timings, to check for memory leaks] */
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Run-2");
         HYPRE_ParVectorSetRandomValues_pre(solver_precision, x_slvr, 775);
#if defined(HYPRE_USING_CURAND) || defined(HYPRE_USING_ROCRAND)
         hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         HYPRE_ParVectorCopy_mp(x0_save, x_slvr);

         HYPRE_BiCGSTABSetup_pre(solver_precision, pcg_solver,
                          (HYPRE_Matrix) A_slvr,
                          (HYPRE_Vector) b_slvr,
                          (HYPRE_Vector) x_slvr);
         HYPRE_BiCGSTABSolve_pre(solver_precision, pcg_solver,
                          (HYPRE_Matrix) A_slvr,
                          (HYPRE_Vector) b_slvr,
                          (HYPRE_Vector) x_slvr);
         HYPRE_ANNOTATE_REGION_END("%s", "Run-2");
      }
      HYPRE_BiCGSTABGetNumIterations_pre(solver_precision, pcg_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm_pre(solver_precision, pcg_solver, final_res_norm);
      HYPRE_ParCSRBiCGSTABDestroy_pre(solver_precision, pcg_solver);

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("BiCGSTAB Iterations = %d\n", num_iterations);
      }
   }
  /*-----------------------------------------------------------
   * Some post-processing and cleanup 
   *-----------------------------------------------------------*/
   if (myid == 0)
   {
      switch (solver_precision)
      {
         case HYPRE_REAL_SINGLE:
            hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm_flt);
            break;
         case HYPRE_REAL_DOUBLE:
            hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm_dbl);
            break;
         case HYPRE_REAL_LONGDOUBLE:
            hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm_ldbl);
            break;
      }
      hypre_printf("\n");
   }   

   /* update solution vector x */
   HYPRE_ParVectorCopy_mp( x_slvr, x);
   /* cleanup data preconditioning */
   if (solver_id == 1 || solver_id == 11 || solver_id == 21 || solver_id == 31)
   {
      HYPRE_BoomerAMGDestroy_pre(precond_precision, pcg_precond);
      HYPRE_ParCSRMatrixDestroy_pre(precond_precision, A_pc);
      HYPRE_ParVectorDestroy_pre(precond_precision, x_pc);
      HYPRE_ParVectorDestroy_pre(precond_precision, b_pc);
   }   
   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_IJVectorPrint(ij_x, "IJ.out.x");
   }

   if (print_system_binary)
   {
      if (ij_x)
      {
         HYPRE_IJVectorPrintBinary(ij_x, "IJ.out.x");
      }
      else if (x)
      {
         HYPRE_ParVectorPrintBinaryIJ(x, "IJ.out.x");
      }
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

final:

   HYPRE_ParCSRMatrixDestroy_pre(solver_precision, A_slvr);
   HYPRE_ParVectorDestroy_pre(solver_precision, x_slvr);
   HYPRE_ParVectorDestroy_pre(solver_precision, b_slvr);
   
   HYPRE_ParVectorDestroy(x0_save);

   if (build_matrix_type == -1 || build_matrix_type == -2)
   {
      if (ij_A)
      {
         HYPRE_IJMatrixDestroy(ij_A);
      }
   }
   else
   {
      HYPRE_ParCSRMatrixDestroy(parcsr_A);
   }

   /* for build_rhs_type = 1, 6 or 7, we did not create ij_b  - just b*/
   if (build_rhs_type == 1 || build_rhs_type == 6 || build_rhs_type == 7)
   {
      HYPRE_ParVectorDestroy(b);
   }
   else
   {
      if (ij_b) { HYPRE_IJVectorDestroy(ij_b); }
   }

   if (build_x0_type == -3)
   {
      HYPRE_ParVectorDestroy(x);
   }
   else
   {
      if (ij_x) { HYPRE_IJVectorDestroy(ij_x); }
   }

   if (build_rbm)
   {
      if (ij_rbm)
      {
         for (i = 0; i < num_interp_vecs; i++)
         {
            if (ij_rbm[i]) { HYPRE_IJVectorDestroy(ij_rbm[i]); }
         }
      }
      hypre_TFree(ij_rbm, HYPRE_MEMORY_HOST);
      hypre_TFree(interp_vecs, HYPRE_MEMORY_HOST);
   }
   if (nongalerk_tol)
   {
      hypre_TFree(nongalerk_tol, HYPRE_MEMORY_HOST);
   }

   /* AMG takes ownership of dof_func, so we free it explicitly only when not using AMG */
   if (free_dof_func)
   {
      hypre_TFree(dof_func, memory_location);
   }

   if (test_error == 1)
   {
      /* Test GetErrorMessages() */
      char      *buffer, *msg;
      HYPRE_Int  bufsz;
      HYPRE_GetErrorMessages(&buffer, &bufsz);
      hypre_MPI_Barrier(comm);
      for (msg = buffer; msg < (buffer + bufsz); msg += strlen(msg) + 1)
      {
         hypre_fprintf(stderr, "%d: %s", myid, msg);
      }
      hypre_TFree(buffer, HYPRE_MEMORY_HOST);
   }
   else
   {
      if (myid == 0)
      {
         HYPRE_PrintErrorMessages(comm);
      }
   }

   /* Free the memory buffer allocated for storing error messages when using mode 1 for printing errors
      Note: This call is redundant since the cleanup is already handled in HYPRE_Finalize. */
   HYPRE_ClearErrorMessages();

   /* Finalize Hypre */
   HYPRE_Finalize();

   /* Finalize MPI */
   hypre_MPI_Finalize();

#if defined(HYPRE_USING_MEMORY_TRACKER)
   if (memory_location == HYPRE_MEMORY_HOST)
   {
      if (hypre_total_bytes[hypre_MEMORY_DEVICE] || hypre_total_bytes[hypre_MEMORY_UNIFIED])
      {
         hypre_printf("Error: nonzero GPU memory allocated with the HOST mode\n");
         hypre_assert(0);
      }
   }
#endif

   /* when using cuda-memcheck --leak-check full, uncomment this */
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_TEST_USING_HOST)
   hypre_ResetDevice();
#endif

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
BuildParFromFile( MPI_Comm             comm,
                  HYPRE_Int            argc,
                  char                *argv[],
                  HYPRE_Int            arg_index,
                  HYPRE_ParCSRMatrix  *A_ptr )
{
   char               *filename;

   HYPRE_ParCSRMatrix A;

   HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid);

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

   HYPRE_ParCSRMatrixRead(comm, filename, &A);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build rhs from file. Expects two files on each processor.
 * filename.n contains the data and
 * and filename.INFO.n contains global row
 * numbers
 *----------------------------------------------------------------------*/

HYPRE_Int
ReadParVectorFromFile( MPI_Comm             comm,
                       HYPRE_Int            argc,
                       char                *argv[],
                       HYPRE_Int            arg_index,
                       HYPRE_ParVector      *b_ptr )
{
   char           *filename;
   HYPRE_ParVector b;
   HYPRE_Int       myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid);

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

   HYPRE_ParVectorRead(comm, filename, &b);

   *b_ptr = b;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian( MPI_Comm             comm,
                   HYPRE_Int            argc,
                   char                *argv[],
                   HYPRE_Int            arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_BigInt        nx, ny, nz;
   HYPRE_Int           P, Q, R;
   HYPRE_Real          cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q, r;
   HYPRE_Int           num_fun = 1;
   HYPRE_Real         *values;
   HYPRE_Real         *mtrx;

   HYPRE_Real          ep = .1;

   HYPRE_Int           system_vcoef = 0;
   HYPRE_Int           sys_opt = 0;
   HYPRE_Int           vcoef_opt = 0;


   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

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
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = (HYPRE_Real)atof(argv[arg_index++]);
         cy = (HYPRE_Real)atof(argv[arg_index++]);
         cz = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sysL") == 0 )
      {
         arg_index++;
         num_fun = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sysL_opt") == 0 )
      {
         arg_index++;
         sys_opt = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sys_vcoef") == 0 )
      {
         /* have to use -sysL for this to */
         arg_index++;
         system_vcoef = 1;
      }
      else if ( strcmp(argv[arg_index], "-sys_vcoef_opt") == 0 )
      {
         arg_index++;
         vcoef_opt = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ep") == 0 )
      {
         arg_index++;
         ep = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian:   num_fun = %d\n", num_fun);
      hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  4, HYPRE_MEMORY_HOST);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0 * cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0 * cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0 * cz;
   }

   if (num_fun == 1)
      A = (HYPRE_ParCSRMatrix) GenerateLaplacian(comm,
                                                 nx, ny, nz, P, Q, R, p, q, r, values);
   else
   {
      mtrx = hypre_CTAlloc(HYPRE_Real,  num_fun * num_fun, HYPRE_MEMORY_HOST);

      if (num_fun == 2)
      {
         if (sys_opt == 1) /* identity  */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 2)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 20.0;
         }
         else if (sys_opt == 3) /* similar to barry's talk - ex1 */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 2.0;
            mtrx[2] = 2.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 4) /* can use with vcoef to get barry's ex*/
         {
            mtrx[0] = 1.0;
            mtrx[1] = 1.0;
            mtrx[2] = 1.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 5) /* barry's talk - ex1 */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 1.1;
            mtrx[2] = 1.1;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 6) /*  */
         {
            mtrx[0] = 1.1;
            mtrx[1] = 1.0;
            mtrx[2] = 1.0;
            mtrx[3] = 1.1;
         }

         else /* == 0 */
         {
            mtrx[0] = 2;
            mtrx[1] = 1;
            mtrx[2] = 1;
            mtrx[3] = 2;
         }
      }
      else if (num_fun == 3)
      {
         if (sys_opt == 1)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 0.0;
            mtrx[4] = 1.0;
            mtrx[5] = 0.0;
            mtrx[6] = 0.0;
            mtrx[7] = 0.0;
            mtrx[8] = 1.0;
         }
         else if (sys_opt == 2)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 0.0;
            mtrx[4] = 20.0;
            mtrx[5] = 0.0;
            mtrx[6] = 0.0;
            mtrx[7] = 0.0;
            mtrx[8] = .01;
         }
         else if (sys_opt == 3)
         {
            mtrx[0] = 1.01;
            mtrx[1] = 1;
            mtrx[2] = 0.0;
            mtrx[3] = 1;
            mtrx[4] = 2;
            mtrx[5] = 1;
            mtrx[6] = 0.0;
            mtrx[7] = 1;
            mtrx[8] = 1.01;
         }
         else if (sys_opt == 4) /* barry ex4 */
         {
            mtrx[0] = 3;
            mtrx[1] = 1;
            mtrx[2] = 0.0;
            mtrx[3] = 1;
            mtrx[4] = 4;
            mtrx[5] = 2;
            mtrx[6] = 0.0;
            mtrx[7] = 2;
            mtrx[8] = .25;
         }
         else /* == 0 */
         {
            mtrx[0] = 2.0;
            mtrx[1] = 1.0;
            mtrx[2] = 0.0;
            mtrx[3] = 1.0;
            mtrx[4] = 2.0;
            mtrx[5] = 1.0;
            mtrx[6] = 0.0;
            mtrx[7] = 1.0;
            mtrx[8] = 2.0;
         }

      }
      else if (num_fun == 4)
      {
         mtrx[0] = 1.01;
         mtrx[1] = 1;
         mtrx[2] = 0.0;
         mtrx[3] = 0.0;
         mtrx[4] = 1;
         mtrx[5] = 2;
         mtrx[6] = 1;
         mtrx[7] = 0.0;
         mtrx[8] = 0.0;
         mtrx[9] = 1;
         mtrx[10] = 1.01;
         mtrx[11] = 0.0;
         mtrx[12] = 2;
         mtrx[13] = 1;
         mtrx[14] = 0.0;
         mtrx[15] = 1;
      }

      if (!system_vcoef)
      {
         A = (HYPRE_ParCSRMatrix) GenerateSysLaplacian(comm,
                                                       nx, ny, nz, P, Q,
                                                       R, p, q, r, num_fun, mtrx, values);
      }
      else
      {
         HYPRE_Real *mtrx_values;

         mtrx_values = hypre_CTAlloc(HYPRE_Real,  num_fun * num_fun * 4, HYPRE_MEMORY_HOST);

         if (num_fun == 2)
         {
            if (vcoef_opt == 1)
            {
               /* Barry's talk * - must also have sys_opt = 4, all fail */
               mtrx[0] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, .10, 1.0, 0, mtrx_values);

               mtrx[1]  = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, .1, 1.0, 1.0, 1, mtrx_values);

               mtrx[2] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, .01, 1.0, 1.0, 2, mtrx_values);

               mtrx[3] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, .02, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 2)
            {
               /* Barry's talk * - ex2 - if have sys-opt = 4*/
               mtrx[0] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, .010, 1.0, 0, mtrx_values);

               mtrx[1]  = 200.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 1, mtrx_values);

               mtrx[2] = 200.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 2, mtrx_values);

               mtrx[3] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, .02, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 3) /* use with default sys_opt  - ulrike ex 3*/
            {

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 4) /* use with default sys_opt  - ulrike ex 4*/
            {
               HYPRE_Real ep2 = ep;

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, ep * 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, ep2 * 1.0, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 5) /* use with default sys_opt  - */
            {
               HYPRE_Real  alp, beta;
               alp = .001;
               beta = 10;

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, alp * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, beta * 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, alp * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, beta * 1.0, 1.0, 3, mtrx_values);
            }
            else  /* = 0 */
            {
               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 2.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, 1.0, 0.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 3.0, 1.0, 3, mtrx_values);
            }
         }
         else if (num_fun == 3)
         {
            mtrx[0] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, .01, 1, 0, mtrx_values);

            mtrx[1] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 1, mtrx_values);

            mtrx[2] = 0.0;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 2, mtrx_values);

            mtrx[3] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 3, mtrx_values);

            mtrx[4] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 2, .02, 1, 4, mtrx_values);

            mtrx[5] = 2;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 5, mtrx_values);

            mtrx[6] = 0.0;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 6, mtrx_values);

            mtrx[7] = 2;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 7, mtrx_values);

            mtrx[8] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1.5, .04, 1, 8, mtrx_values);
         }

         A = (HYPRE_ParCSRMatrix) GenerateSysLaplacianVCoef(comm,
                                                            nx, ny, nz, P, Q,
                                                            R, p, q, r, num_fun, mtrx, mtrx_values);

         hypre_TFree(mtrx_values, HYPRE_MEMORY_HOST);
      }

      hypre_TFree(mtrx, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * returns the sign of a real number
 *  1 : positive
 *  0 : zero
 * -1 : negative
 *----------------------------------------------------------------------*/
static inline HYPRE_Int sign_double(HYPRE_Real a)
{
   return ( (0.0 < a) - (0.0 > a) );
}

/*----------------------------------------------------------------------
 * Build standard 7-point convection-diffusion operator
 * Parameters given in command line.
 * Operator:
 *
 *  -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f
 *
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParDifConv( MPI_Comm             comm,
                 HYPRE_Int            argc,
                 char                *argv[],
                 HYPRE_Int            arg_index,
                 HYPRE_ParCSRMatrix  *A_ptr)
{
   HYPRE_BigInt        nx, ny, nz;
   HYPRE_Int           P, Q, R;
   HYPRE_Real          cx, cy, cz;
   HYPRE_Real          ax, ay, az, atype;
   HYPRE_Real          hinx, hiny, hinz;
   HYPRE_Int           sign_prod;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   ax = 1.;
   ay = 1.;
   az = 1.;

   atype = 0;

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
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = (HYPRE_Real)atof(argv[arg_index++]);
         cy = (HYPRE_Real)atof(argv[arg_index++]);
         cz = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = (HYPRE_Real)atof(argv[arg_index++]);
         ay = (HYPRE_Real)atof(argv[arg_index++]);
         az = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-atype") == 0 )
      {
         arg_index++;
         atype = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Convection-Diffusion: \n");
      hypre_printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");
      hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   hinx = 1. / (HYPRE_Real)(nx + 1);
   hiny = 1. / (HYPRE_Real)(ny + 1);
   hinz = 1. / (HYPRE_Real)(nz + 1);

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/
   /* values[7]:
    *    [0]: center
    *    [1]: X-
    *    [2]: Y-
    *    [3]: Z-
    *    [4]: X+
    *    [5]: Y+
    *    [6]: Z+
    */
   values = hypre_CTAlloc(HYPRE_Real,  7, HYPRE_MEMORY_HOST);

   values[0] = 0.;

   if (0 == atype) /* forward scheme for conv */
   {
      values[1] = -cx / (hinx * hinx);
      values[2] = -cy / (hiny * hiny);
      values[3] = -cz / (hinz * hinz);
      values[4] = -cx / (hinx * hinx) + ax / hinx;
      values[5] = -cy / (hiny * hiny) + ay / hiny;
      values[6] = -cz / (hinz * hinz) + az / hinz;

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx) - 1.*ax / hinx;
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny) - 1.*ay / hiny;
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz) - 1.*az / hinz;
      }
   }
   else if (1 == atype) /* backward scheme for conv */
   {
      values[1] = -cx / (hinx * hinx) - ax / hinx;
      values[2] = -cy / (hiny * hiny) - ay / hiny;
      values[3] = -cz / (hinz * hinz) - az / hinz;
      values[4] = -cx / (hinx * hinx);
      values[5] = -cy / (hiny * hiny);
      values[6] = -cz / (hinz * hinz);

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx) + 1.*ax / hinx;
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny) + 1.*ay / hiny;
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz) + 1.*az / hinz;
      }
   }
   else if (3 == atype) /* upwind scheme */
   {
      sign_prod = sign_double(cx) * sign_double(ax);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[1] = -cx / (hinx * hinx) - ax / hinx;
         values[4] = -cx / (hinx * hinx);
         if (nx > 1)
         {
            values[0] += 2.0 * cx / (hinx * hinx) + 1.*ax / hinx;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[1] = -cx / (hinx * hinx);
         values[4] = -cx / (hinx * hinx) + ax / hinx;
         if (nx > 1)
         {
            values[0] += 2.0 * cx / (hinx * hinx) - 1.*ax / hinx;
         }
      }

      sign_prod = sign_double(cy) * sign_double(ay);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[2] = -cy / (hiny * hiny) - ay / hiny;
         values[5] = -cy / (hiny * hiny);
         if (ny > 1)
         {
            values[0] += 2.0 * cy / (hiny * hiny) + 1.*ay / hiny;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[2] = -cy / (hiny * hiny);
         values[5] = -cy / (hiny * hiny) + ay / hiny;
         if (ny > 1)
         {
            values[0] += 2.0 * cy / (hiny * hiny) - 1.*ay / hiny;
         }
      }

      sign_prod = sign_double(cz) * sign_double(az);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[3] = -cz / (hinz * hinz) - az / hinz;
         values[6] = -cz / (hinz * hinz);
         if (nz > 1)
         {
            values[0] += 2.0 * cz / (hinz * hinz) + 1.*az / hinz;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[3] = -cz / (hinz * hinz);
         values[6] = -cz / (hinz * hinz) + az / hinz;
         if (nz > 1)
         {
            values[0] += 2.0 * cz / (hinz * hinz) - 1.*az / hinz;
         }
      }
   }
   else /* centered difference scheme */
   {
      values[1] = -cx / (hinx * hinx) - ax / (2.*hinx);
      values[2] = -cy / (hiny * hiny) - ay / (2.*hiny);
      values[3] = -cz / (hinz * hinz) - az / (2.*hinz);
      values[4] = -cx / (hinx * hinx) + ax / (2.*hinx);
      values[5] = -cy / (hiny * hiny) + ay / (2.*hiny);
      values[6] = -cz / (hinz * hinz) + az / (2.*hinz);

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx);
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny);
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz);
      }
   }

   A = (HYPRE_ParCSRMatrix) GenerateDifConv(comm,
                                            nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from one file on Proc. 0. Expects matrix to be in
 * CSR format. Distributes matrix across processors giving each about
 * the same number of rows.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParFromOneFile( MPI_Comm             comm,
                     HYPRE_Int            argc,
                     char                *argv[],
                     HYPRE_Int            arg_index,
                     HYPRE_Int            num_functions,
                     HYPRE_ParCSRMatrix  *A_ptr )
{
   char               *filename;

   HYPRE_CSRMatrix     A_CSR = NULL;
   HYPRE_BigInt       *row_part = NULL;
   HYPRE_BigInt       *col_part = NULL;

   HYPRE_Int           myid, numprocs;
   HYPRE_Int           i, rest, size, num_nodes, num_dofs;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &numprocs);

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

      /*-----------------------------------------------------------
       * Generate the matrix
       *-----------------------------------------------------------*/

      A_CSR = HYPRE_CSRMatrixRead(filename);
   }

   if (myid == 0 && num_functions > 1)
   {
      HYPRE_CSRMatrixGetNumRows(A_CSR, &num_dofs);
      num_nodes = num_dofs / num_functions;
      if (num_dofs == num_functions * num_nodes)
      {
         row_part = hypre_CTAlloc(HYPRE_BigInt,  numprocs + 1, HYPRE_MEMORY_HOST);

         row_part[0] = 0;
         size = num_nodes / numprocs;
         rest = num_nodes - size * numprocs;
         for (i = 0; i < rest; i++)
         {
            row_part[i + 1] = row_part[i] + (size + 1) * num_functions;
         }
         for (i = rest; i < numprocs; i++)
         {
            row_part[i + 1] = row_part[i] + size * num_functions;
         }

         col_part = row_part;
      }
   }

   HYPRE_CSRMatrixToParCSRMatrix(comm, A_CSR, row_part, col_part, A_ptr);

   if (myid == 0)
   {
      HYPRE_CSRMatrixDestroy(A_CSR);
   }

   return (0);
}

/*----------------------------------------------------------------------
 * Build x0 from one file on Proc. 0. Distributes vector across processors
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildSolParFromOneFile( MPI_Comm             comm,
                        HYPRE_Int            argc,
                        char                *argv[],
                        HYPRE_Int            arg_index,
                        HYPRE_ParCSRMatrix   parcsr_A,
                        HYPRE_ParVector     *x_ptr     )
{
   char           *filename;
   HYPRE_Int       myid;
   HYPRE_BigInt   *partitioning;
   HYPRE_ParVector x;
   HYPRE_Vector    x_CSR = NULL;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid );
   partitioning = hypre_ParCSRMatrixColStarts(parcsr_A);

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
    * Read the initial guess from file and create parallel vector
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      x_CSR = HYPRE_VectorRead(filename);
   }
   HYPRE_VectorToParVector(comm, x_CSR, partitioning, &x);

   *x_ptr = x;

   HYPRE_VectorDestroy(x_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildFuncTagsFromFiles( HYPRE_Int            argc,
                        char                *argv[],
                        HYPRE_Int            arg_index,
                        HYPRE_ParCSRMatrix   parcsr_A,
                        HYPRE_Int          **dof_func_ptr )
{
   HYPRE_UNUSED_VAR(argc);
   HYPRE_UNUSED_VAR(argv);
   HYPRE_UNUSED_VAR(arg_index);
   HYPRE_UNUSED_VAR(parcsr_A);
   HYPRE_UNUSED_VAR(dof_func_ptr);

   /*----------------------------------------------------------------------
    * Build Function array from files on different processors
    *----------------------------------------------------------------------*/

   hypre_printf("Feature is not implemented yet!\n");
   return (0);
}

/*----------------------------------------------------------------------
 * Build Function array from file on master process
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildFuncTagsFromOneFile( MPI_Comm             comm,
                          HYPRE_Int            argc,
                          char                *argv[],
                          HYPRE_Int            arg_index,
                          HYPRE_ParCSRMatrix   parcsr_A,
                          HYPRE_Int          **dof_func_ptr )
{
   char                 *filename;

   HYPRE_Int             myid, num_procs;
   HYPRE_BigInt          first_row_index;
   HYPRE_BigInt          last_row_index;
   HYPRE_BigInt         *partitioning;
   HYPRE_Int            *dof_func = NULL;
   HYPRE_Int            *dof_func_local;
   HYPRE_Int             i, j;
   HYPRE_BigInt          local_size;
   HYPRE_Int             global_size;
   hypre_MPI_Request    *requests;
   hypre_MPI_Status     *status, status0;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &num_procs);

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
      FILE *fp;
      hypre_printf("  Funcs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * read in the data
       *-----------------------------------------------------------*/
      fp = fopen(filename, "r");

      hypre_fscanf(fp, "%d", &global_size);
      dof_func = hypre_CTAlloc(HYPRE_Int,  global_size, HYPRE_MEMORY_HOST);

      for (j = 0; j < global_size; j++)
      {
         hypre_fscanf(fp, "%d", &dof_func[j]);
      }

      fclose(fp);
   }

   HYPRE_ParCSRMatrixGetGlobalRowPartitioning(parcsr_A, 0, &partitioning);
   first_row_index = hypre_ParCSRMatrixFirstRowIndex(parcsr_A);
   last_row_index  = hypre_ParCSRMatrixLastRowIndex(parcsr_A);
   local_size      = last_row_index - first_row_index + 1;
   dof_func_local  = hypre_CTAlloc(HYPRE_Int, local_size, HYPRE_MEMORY_HOST);
   if (myid == 0)
   {
      requests = hypre_CTAlloc(hypre_MPI_Request, num_procs - 1, HYPRE_MEMORY_HOST);
      status = hypre_CTAlloc(hypre_MPI_Status, num_procs - 1, HYPRE_MEMORY_HOST);
      for (i = 1; i < num_procs; i++)
      {
         hypre_MPI_Isend(&dof_func[partitioning[i]],
                         (HYPRE_Int)(partitioning[i + 1] - partitioning[i]),
                         HYPRE_MPI_INT, i, 0, comm, &requests[i - 1]);
      }
      for (i = 0; i < (HYPRE_Int)local_size; i++)
      {
         dof_func_local[i] = dof_func[i];
      }
      hypre_MPI_Waitall(num_procs - 1, requests, status);
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
      hypre_TFree(status, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_MPI_Recv(dof_func_local, (HYPRE_Int)local_size, HYPRE_MPI_INT, 0, 0, comm, &status0);
   }

   *dof_func_ptr = dof_func_local;

   if (myid == 0) { hypre_TFree(dof_func, HYPRE_MEMORY_HOST); }

   if (partitioning) { hypre_TFree(partitioning, HYPRE_MEMORY_HOST); }

   return (0);
}

/*----------------------------------------------------------------------
 * Build interleaved function array (0,1,2,0,1,2,...) for local data
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildFuncTagsInterleaved( HYPRE_Int              local_size,
                          HYPRE_Int              num_functions,
                          HYPRE_MemoryLocation   memory_location,
                          HYPRE_Int            **dof_func_ptr )
{
   HYPRE_Int *dof_func_h, *dof_func;
   HYPRE_Int  i;

   /* Allocate array */
   dof_func_h = hypre_CTAlloc(HYPRE_Int, local_size, HYPRE_MEMORY_HOST);

   /* Fill array with interleaved function numbers (0,1,2,...,0,1,2,...) */
   for (i = 0; i < local_size; i++)
   {
      dof_func_h[i] = i % num_functions;
   }

   /* Copy to device */
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      dof_func = hypre_CTAlloc(HYPRE_Int, local_size, memory_location);
      hypre_TMemcpy(dof_func, dof_func_h, HYPRE_Int, local_size,
                    memory_location, HYPRE_MEMORY_HOST);

      /* Free host memory */
      hypre_TFree(dof_func_h, HYPRE_MEMORY_HOST);
   }
   else
   {
      dof_func = dof_func_h;
   }

   *dof_func_ptr = dof_func;

   return (0);
}

/*----------------------------------------------------------------------
 * Build contiguous function array (0,0,0,...,1,1,1,...) for local data
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildFuncTagsContiguous( HYPRE_Int              local_size,
                         HYPRE_Int              num_functions,
                         HYPRE_MemoryLocation   memory_location,
                         HYPRE_Int            **dof_func_ptr )
{
   HYPRE_Int *dof_func_h, *dof_func;
   HYPRE_Int  i;
   HYPRE_Int  block_size, remainder;
   HYPRE_Int  current_func, count;

   /* Allocate array on host initially */
   dof_func_h = hypre_CTAlloc(HYPRE_Int, local_size, HYPRE_MEMORY_HOST);

   /* Calculate block size for each function */
   block_size = local_size / num_functions;
   remainder  = local_size % num_functions;

   /* Fill array with contiguous function numbers (0,0,0,...,1,1,1,...) */
   current_func = 0;
   count = 0;

   for (i = 0; i < local_size; i++)
   {
      dof_func_h[i] = current_func;
      count++;

      /* Move to next function when we've filled the current block */
      if (current_func < remainder)
      {
         /* First 'remainder' functions get one extra element */
         if (count >= block_size + 1)
         {
            current_func++;
            count = 0;
         }
      }
      else
      {
         if (count >= block_size)
         {
            current_func++;
            count = 0;
         }
      }
   }

   /* Copy to device if needed */
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      dof_func = hypre_CTAlloc(HYPRE_Int, local_size, memory_location);
      hypre_TMemcpy(dof_func, dof_func_h, HYPRE_Int, local_size,
                    memory_location, HYPRE_MEMORY_HOST);

      /* Free host memory */
      hypre_TFree(dof_func_h, HYPRE_MEMORY_HOST);
   }
   else
   {
      dof_func = dof_func_h;
   }

   *dof_func_ptr = dof_func;

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildRhsParFromOneFile( MPI_Comm             comm,
                        HYPRE_Int            argc,
                        char                *argv[],
                        HYPRE_Int            arg_index,
                        HYPRE_ParCSRMatrix   parcsr_A,
                        HYPRE_ParVector     *b_ptr     )
{
   char           *filename;
   HYPRE_Int       myid;
   HYPRE_BigInt   *partitioning;
   HYPRE_ParVector b;
   HYPRE_Vector    b_CSR = NULL;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid);
   partitioning = hypre_ParCSRMatrixRowStarts(parcsr_A);

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
    * Read the vector from file and create parallel vector
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      b_CSR = HYPRE_VectorRead(filename);
   }
   HYPRE_VectorToParVector(comm, b_CSR, partitioning, &b);

   *b_ptr = b;

   HYPRE_VectorDestroy(b_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildBigArrayFromOneFile( MPI_Comm             comm,
                          HYPRE_Int            argc,
                          char                *argv[],
                          const char          *array_name,
                          HYPRE_Int            arg_index,
                          HYPRE_BigInt        *partitioning,
                          HYPRE_Int           *size,
                          HYPRE_BigInt       **array_ptr )
{
   char           *filename = NULL;
   FILE           *fp;
   HYPRE_Int       myid;
   HYPRE_Int       num_procs;
   HYPRE_Int       global_size;
   HYPRE_BigInt   *global_array = NULL;
   HYPRE_BigInt   *array        = NULL;
   HYPRE_BigInt   *send_buffer  = NULL;
   HYPRE_Int      *send_counts  = NULL;
   HYPRE_Int      *displs       = NULL;
   HYPRE_Int      *array_procs  = NULL;
   HYPRE_Int       j, jj, proc;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &num_procs);

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      if (myid == 0)
      {
         hypre_printf("Error: No filename specified \n");
      }
      hypre_MPI_Abort(comm, 1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
   if (myid == 0)
   {
      hypre_printf("  %s array FromFile: %s\n", array_name, filename);

      /*-----------------------------------------------------------
       * Read data
       *-----------------------------------------------------------*/
      fp = fopen(filename, "r");

      hypre_fscanf(fp, "%d", &global_size);
      global_array = hypre_CTAlloc(HYPRE_BigInt, global_size, HYPRE_MEMORY_HOST);
      for (j = 0; j < global_size; j++)
      {
         hypre_fscanf(fp, "%d", &global_array[j]);
      }

      fclose(fp);
   }

   /*-----------------------------------------------------------
    * Distribute data
    *-----------------------------------------------------------*/
   if (myid == 0)
   {
      send_counts = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
      displs      = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
      array_procs = hypre_CTAlloc(HYPRE_Int, global_size, HYPRE_MEMORY_HOST);
      send_buffer = hypre_CTAlloc(HYPRE_BigInt, global_size, HYPRE_MEMORY_HOST);
      for (j = 0; j < global_size; j++)
      {
         for (proc = 0; proc < (num_procs + 1); proc++)
         {
            if (global_array[j] < partitioning[proc])
            {
               proc--; break;
            }
         }

         if (proc < num_procs)
         {
            send_counts[proc]++;
            array_procs[j] = proc;
         }
         else
         {
            array_procs[j] = -1; // Not found
         }
      }

      for (proc = 0; proc < (num_procs - 1); proc++)
      {
         displs[proc + 1] = displs[proc] + send_counts[proc];
      }
   }
   hypre_MPI_Scatter(send_counts, 1, HYPRE_MPI_INT, size, 1, HYPRE_MPI_INT, 0, comm);

   if (myid == 0)
   {
      for (proc = 0; proc < num_procs; proc++)
      {
         send_counts[proc] = 0;
      }

      for (j = 0; j < global_size; j++)
      {
         proc = array_procs[j];
         if (proc > -1)
         {
            jj = displs[proc] + send_counts[proc];
            send_buffer[jj] = global_array[j];
            send_counts[proc]++;
         }
      }
   }

   array = hypre_CTAlloc(HYPRE_BigInt, *size, HYPRE_MEMORY_HOST);
   hypre_MPI_Scatterv(send_buffer, send_counts, displs, HYPRE_MPI_BIG_INT,
                      array, *size, HYPRE_MPI_BIG_INT, 0, comm);
   *array_ptr = array;

   /* Free memory */
   if (myid == 0)
   {
      hypre_TFree(send_counts, HYPRE_MEMORY_HOST);
      hypre_TFree(send_buffer, HYPRE_MEMORY_HOST);
      hypre_TFree(displs, HYPRE_MEMORY_HOST);
      hypre_TFree(array_procs, HYPRE_MEMORY_HOST);
      hypre_TFree(global_array, HYPRE_MEMORY_HOST);
   }

   return 0;
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian9pt( MPI_Comm             comm,
                      HYPRE_Int            argc,
                      char                *argv[],
                      HYPRE_Int            arg_index,
                      HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_BigInt        nx, ny;
   HYPRE_Int           P, Q;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;

   P  = 1;
   Q  = num_procs;

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
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian 9pt:\n");
      hypre_printf("    (nx, ny) = (%b, %b)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p) / P;

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  2, HYPRE_MEMORY_HOST);

   values[1] = -1.;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0;
   }
   if (ny > 1)
   {
      values[0] += 2.0;
   }
   if (nx > 1 && ny > 1)
   {
      values[0] += 4.0;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(comm,
                                                 nx, ny, P, Q, p, q, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian27pt( MPI_Comm             comm,
                       HYPRE_Int            argc,
                       char                *argv[],
                       HYPRE_Int            arg_index,
                       HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_BigInt        nx, ny, nz;
   HYPRE_Int           P, Q, R;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
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

   if ((P * Q * R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian_27pt:\n");
      hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  2, HYPRE_MEMORY_HOST);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
   {
      values[0] = 8.0;
   }
   if (nx * ny == 1 || nx * nz == 1 || ny * nz == 1)
   {
      values[0] = 2.0;
   }
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(comm,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build 125-point laplacian in 3D (27-pt squared)
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian125pt( MPI_Comm             comm,
                        HYPRE_Int            argc,
                        char                *argv[],
                        HYPRE_Int            arg_index,
                        HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_BigInt              nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix        A, B;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real               *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
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

   if ((P * Q * R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian_125pt:\n");
      hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  2, HYPRE_MEMORY_HOST);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
   {
      values[0] = 8.0;
   }
   if (nx * ny == 1 || nx * nz == 1 || ny * nz == 1)
   {
      values[0] = 2.0;
   }
   values[1] = -1.;

   B = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(comm,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);
   A = (HYPRE_ParCSRMatrix) hypre_ParCSRMatMat(B, B);

   HYPRE_ParCSRMatrixDestroy(B);
   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build 7-point in 2D
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParRotate7pt( MPI_Comm             comm,
                   HYPRE_Int            argc,
                   char                *argv[],
                   HYPRE_Int            arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_BigInt        nx, ny;
   HYPRE_Int           P, Q;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q;
   HYPRE_Real          eps = 0.0, alpha = 1.0;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;

   P  = 1;
   Q  = num_procs;

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
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-alpha") == 0 )
      {
         arg_index++;
         alpha  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Rotate 7pt:\n");
      hypre_printf("    alpha = %f, eps = %f\n", alpha, eps);
      hypre_printf("    (nx, ny) = (%b, %b)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p) / P;

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   A = (HYPRE_ParCSRMatrix) GenerateRotate7pt(comm,
                                              nx, ny, P, Q, p, q, alpha, eps);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point difference operator using centered differences
 *
 *  eps*(a(x,y,z) ux)x + (b(x,y,z) uy)y + (c(x,y,z) uz)z
 *  d(x,y,z) ux + e(x,y,z) uy + f(x,y,z) uz + g(x,y,z) u
 *
 *  functions a,b,c,d,e,f,g need to be defined inside par_vardifconv.c
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParVarDifConv( MPI_Comm             comm,
                    HYPRE_Int            argc,
                    char                *argv[],
                    HYPRE_Int            arg_index,
                    HYPRE_ParCSRMatrix  *A_ptr,
                    HYPRE_ParVector     *rhs_ptr )
{
   HYPRE_BigInt        nx, ny, nz;
   HYPRE_Int           P, Q, R;

   HYPRE_ParCSRMatrix  A;
   HYPRE_ParVector     rhs;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q, r;
   HYPRE_Int           type;
   HYPRE_Real          eps;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;
   P  = 1;
   Q  = num_procs;
   R  = 1;
   eps = 1.0;

   /* type: 0   : default FD;
    *       1-3 : FD and examples 1-3 in Ruge-Stuben paper */
   type = 0;

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
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vardifconvRS") == 0 )
      {
         arg_index++;
         type = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  ell PDE: eps = %f\n", eps);
      hypre_printf("    Dx(aDxu) + Dy(bDyu) + Dz(cDzu) + d Dxu + e Dyu + f Dzu  + g u= f\n");
      hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
   }
   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   if (0 == type)
   {
      A = (HYPRE_ParCSRMatrix) GenerateVarDifConv(comm,
                                                  nx, ny, nz, P, Q, R, p, q, r, eps, &rhs);
   }
   else
   {
      A = (HYPRE_ParCSRMatrix) GenerateRSVarDifConv(comm,
                                                    nx, ny, nz, P, Q, R, p, q, r, eps, &rhs,
                                                    type);
   }

   *A_ptr = A;
   *rhs_ptr = rhs;

   return (0);
}

/**************************************************************************/

HYPRE_Int SetSysVcoefValues(HYPRE_Int num_fun, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                            HYPRE_Real vcx,
                            HYPRE_Real vcy, HYPRE_Real vcz, HYPRE_Int mtx_entry, HYPRE_Real *values)
{


   HYPRE_Int sz = num_fun * num_fun;

   values[1 * sz + mtx_entry] = -vcx;
   values[2 * sz + mtx_entry] = -vcy;
   values[3 * sz + mtx_entry] = -vcz;
   values[0 * sz + mtx_entry] = 0.0;

   if (nx > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcx;
   }
   if (ny > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcy;
   }
   if (nz > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcz;
   }

   return 0;

}

/*----------------------------------------------------------------------
 * Build coordinates for 1D/2D/3D
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParCoordinates( MPI_Comm             comm,
                     HYPRE_Int            argc,
                     char                *argv[],
                     HYPRE_Int            arg_index,
                     HYPRE_Int           *coorddim_ptr,
                     float              **coord_ptr )
{
   HYPRE_BigInt              nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;

   HYPRE_Int                 coorddim;
   float                    *coordinates;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
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

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the coordinates
    *-----------------------------------------------------------*/

   coorddim = 3;
   if (nx < 2) { coorddim--; }
   if (ny < 2) { coorddim--; }
   if (nz < 2) { coorddim--; }

   if (coorddim > 0)
   {
      coordinates = hypre_GenerateCoordinates(comm, nx, ny, nz, P, Q, R, p, q, r, coorddim);
   }
   else
   {
      coordinates = NULL;
   }

   *coorddim_ptr = coorddim;
   *coord_ptr = coordinates;
   return (0);
}
