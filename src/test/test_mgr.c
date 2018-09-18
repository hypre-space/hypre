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

#ifdef HAVE_DSUPERLU
#include "superlu_ddefs.h"
#endif

/* begin lobpcg */

#define NO_SOLVER -9198

#include <assert.h>
#include <time.h>

#include "fortran_matrix.h"
#include "HYPRE_lobpcg.h"

#include "interpreter.h"
#include "multivector.h"
#include "HYPRE_MatvecFunctions.h"

/* max dt */
#define DT_INF 1.0e30
HYPRE_Int
BuildParIsoLaplacian( HYPRE_Int argc, char** argv, HYPRE_ParCSRMatrix *A_ptr );

/* end lobpcg */

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
  HYPRE_Int                 add = 0;
  HYPRE_Int                 build_matrix_type;
  HYPRE_Int                 build_matrix_arg_index;
  HYPRE_Int                 build_rhs_type;
  HYPRE_Int                 build_rhs_arg_index;
  HYPRE_Int                 build_x0_type;
  HYPRE_Int                 build_x0_arg_index;
  HYPRE_Int                 use_block_cf = 0;
  HYPRE_Int                 use_reserved_coarse_grid;
  HYPRE_Int                 build_block_cf_arg_index;
  HYPRE_Int                 solver_id;
  HYPRE_Int                 poutdat;
  HYPRE_Int                 debug_flag;
  HYPRE_Int                 ierr = 0;
  HYPRE_Int                 i,j; 
  HYPRE_Int                 max_levels = 25;
  HYPRE_Int                 num_iterations;
  HYPRE_Int                 pcg_num_its, dscg_num_its;
  HYPRE_Int                 max_iter = 400;
  HYPRE_Int                 mg_max_iter = 100;
  HYPRE_Int                 nodal = 0;
  HYPRE_Int                 nodal_diag = 0;
  HYPRE_Real          cf_tol = 0.9;
  HYPRE_Real          norm;
  HYPRE_Real          final_res_norm;
  void               *object;

  HYPRE_IJMatrix      ij_A = NULL; 
  HYPRE_IJVector      ij_b = NULL;
  HYPRE_IJVector      ij_x = NULL;

  HYPRE_ParCSRMatrix  parcsr_A = NULL;
  HYPRE_ParVector     b = NULL;
  HYPRE_ParVector     x;

  HYPRE_Solver        aux_precond, aux_solver;
  HYPRE_Solver        amg_solver;
  HYPRE_Solver        pcg_solver;
  HYPRE_Solver        pcg_precond=NULL, pcg_precond_gotten;

  HYPRE_Int                 num_procs, myid;
  HYPRE_Int                 local_row;
  HYPRE_Int                *row_sizes;
  HYPRE_Int                *diag_sizes;
  HYPRE_Int                *offdiag_sizes;
  HYPRE_Int                *rows;
  HYPRE_Int                 size;
  HYPRE_Int                *ncols;
  HYPRE_Int                *col_inds;
  HYPRE_Int                *dof_func;
  HYPRE_Int          num_paths = 1;
  HYPRE_Int          agg_num_levels = 0;
  HYPRE_Int          ns_coarse = 1, ns_down = -1, ns_up = -1;

  HYPRE_Int          time_index;
  MPI_Comm            comm = hypre_MPI_COMM_WORLD;
  HYPRE_Int M, N;
  HYPRE_Int first_local_row, last_local_row, local_num_rows;
  HYPRE_Int first_local_col, last_local_col, local_num_cols;
  HYPRE_Int variant, overlap, domain_type;
  HYPRE_Real schwarz_rlx_weight;
  HYPRE_Real *values, val;

  HYPRE_Int use_nonsymm_schwarz = 0;
  HYPRE_Int test_ij = 0;
  HYPRE_Int build_rbm = 0;
  HYPRE_Int build_rbm_index = 0;

  const HYPRE_Real dt_inf = DT_INF;
  HYPRE_Real dt = dt_inf;

  /* parameters for BoomerAMG */
  HYPRE_Real   A_drop_tol = 0.0;
  HYPRE_Real   strong_threshold;
  HYPRE_Real   trunc_factor;
  HYPRE_Real   jacobi_trunc_threshold;
  HYPRE_Real   S_commpkg_switch = 1.0;
  HYPRE_Real   CR_rate = 0.7;
  HYPRE_Real   CR_strong_th = 0.0;
  HYPRE_Int      CR_use_CG = 0;
  HYPRE_Int      P_max_elmts = 4;
  HYPRE_Int      cycle_type;
  HYPRE_Int      coarsen_type = 10;
  HYPRE_Int      measure_type = 0;
  HYPRE_Int      num_sweeps = 1;  
  HYPRE_Int      num_CR_relax_steps = 2;   
  HYPRE_Int      relax_type = -1;   
  HYPRE_Int      add_relax_type = 18;   
  HYPRE_Int      relax_coarse = -1;   
  HYPRE_Int      relax_up = -1;   
  HYPRE_Int      relax_down = -1;   
  HYPRE_Int      relax_order = 0;   
  HYPRE_Int      level_w = -1;
  HYPRE_Int      level_ow = -1;
/* HYPRE_Int      smooth_lev; */
/* HYPRE_Int      smooth_rlx = 8; */
  HYPRE_Int     smooth_type = 6;
  HYPRE_Int     smooth_num_levels = 0;
  HYPRE_Int      smooth_num_sweeps = 1;
  HYPRE_Int      coarse_threshold = 9;
  HYPRE_Int      min_coarse_size = 0;
/* redundant coarse grid solve */
  HYPRE_Int      seq_threshold = 0;
  HYPRE_Int      redundant = 0;
/* additive versions */
  HYPRE_Int additive = -1;
  HYPRE_Int mult_add = -1;
  HYPRE_Int simple = -1;
  HYPRE_Int add_last_lvl = -1;
  HYPRE_Int add_P_max_elmts = 0;
  HYPRE_Real add_trunc_factor = 0;

  HYPRE_Int    rap2=0;
  HYPRE_Int    keepTranspose = 0;
#ifdef HAVE_DSUPERLU
  HYPRE_Int    dslu_threshold = -1;
#endif
  HYPRE_Real   relax_wt; 
  HYPRE_Real   add_relax_wt = 1.0; 
  HYPRE_Real   relax_wt_level; 
  HYPRE_Real   outer_wt;
  HYPRE_Real   outer_wt_level;
  HYPRE_Real   tol = 1.e-6, pc_tol = 0.;
  HYPRE_Real   atol = 0.0;
  HYPRE_Real   max_row_sum = 1.;
  HYPRE_Int    converge_type = 0;

  HYPRE_Int cheby_order = 2;
  HYPRE_Int cheby_eig_est = 10;
  HYPRE_Int cheby_variant = 0;
  HYPRE_Int cheby_scale = 1;
  HYPRE_Real cheby_fraction = .3;

  /* for CGC BM Aug 25, 2006 */
  HYPRE_Int      cgcits = 1;
  /* for coordinate plotting BM Oct 24, 2006 */
  HYPRE_Int      plot_grids = 0;
  HYPRE_Int      coord_dim  = 3;
  float    *coordinates = NULL;
  char    plot_file_name[256];
  
  /* parameters for GMRES */
  HYPRE_Int     k_dim = 100;

  /* interpolation */
  HYPRE_Int      interp_type  = 6; /* default value */
  HYPRE_Int      post_interp_type  = 0; /* default value */

  /* mgr options */
  HYPRE_Int mgr_bsize = 2;
  HYPRE_Int mgr_nlevels = 1;
  HYPRE_Int mgr_num_reserved_nodes = 0;
  HYPRE_Int mgr_non_c_to_f = 1;
  HYPRE_Int *mgr_frelax_method = NULL;
  HYPRE_Int *mgr_frelax_num_functions= NULL;
  HYPRE_Int *mgr_idx_array = NULL;
  HYPRE_Int *mgr_num_cindexes = NULL; 
  HYPRE_Int **mgr_cindexes = NULL;
  HYPRE_Int *mgr_reserved_coarse_indexes = NULL;
  HYPRE_Int mgr_relax_type = 0;
  HYPRE_Int mgr_num_relax_sweeps = 1;
  HYPRE_Int mgr_num_interp_sweeps = 0;
  HYPRE_Int mgr_gsmooth_type = 0;
  HYPRE_Int mgr_num_gsmooth_sweeps = 0;
  HYPRE_Int *mgr_restrict_type = NULL;
  HYPRE_Int mgr_num_restrict_sweeps = 0;   
  HYPRE_Int *mgr_interp_type = NULL;
  mgr_interp_type = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
  mgr_interp_type[0] = 2;
  //mgr_interp_type[1] = 2;
  mgr_restrict_type = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
  mgr_restrict_type[0] = 0;
  //mgr_restrict_type[1] = 0;

  /* Set mgr options */
  /*
  mgr_bsize = 2;
  mgr_nlevels = 1;
  //mgr_num_reserved_nodes = 0;
  mgr_non_c_to_f = 1;
  mgr_frelax_method = 1;
  mgr_relax_type = 0;
  mgr_num_relax_sweeps = 2;
  mgr_interp_type = 2;
  mgr_num_interp_sweeps = 2;
  mgr_gsmooth_type = 0;
  mgr_num_gsmooth_sweeps = 0;
  mgr_restrict_type = 2;
  mgr_num_restrict_sweeps = 2;
  */

  HYPRE_Int *block_cindices[mgr_nlevels];
  HYPRE_Int lv1[mgr_bsize];
  HYPRE_Int lv2[mgr_bsize];
  lv1[0] = 1;
  //lv1[1] = 2;
  //lv2[0] = 2;
  block_cindices[0] = &lv1;
  //block_cindices[1] = &lv2;
  HYPRE_Int block_num_coarse_indices[mgr_nlevels];
  block_num_coarse_indices[0] = 1;
  //block_num_coarse_indices[1] = 1;

  mgr_cindexes = &block_cindices;
  mgr_num_cindexes = &block_num_coarse_indices;
  mgr_idx_array = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
  //mgr_idx_array[0] = 0;
  //mgr_idx_array[1] = 55539;
  //mgr_idx_array[2] = 71923;
  //mgr_idx_array[1] = 52800;

  mgr_frelax_method = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
  mgr_frelax_method[0] = 1;
  //mgr_frelax_method[1] = 0;

  mgr_frelax_num_functions = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
  mgr_frelax_num_functions[0] = 1;

  char* indexList = NULL;
  /* end mgr options */

  HYPRE_Int air = 0;

  /*-----------------------------------------------------------
   * Initialize some stuff
   *-----------------------------------------------------------*/

  /* Initialize MPI */
  hypre_MPI_Init(&argc, &argv);

  hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
  hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
  
  /* GPU Init stuff inside */
  hypre_init();

  /*
    hypre_InitMemoryDebug(myid);
  */
  /*-----------------------------------------------------------
   * Set defaults
   *-----------------------------------------------------------*/
 
  build_matrix_type = -1;
  build_matrix_arg_index = argc;
  build_rhs_type = 2;
  build_rhs_arg_index = argc;
  build_x0_type = -1;
  build_x0_arg_index = argc;
  debug_flag = 0;

  solver_id = 72;

  poutdat = 0;

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
    else if ( strcmp(argv[arg_index], "-solver") == 0 )
    {
      arg_index++;
      solver_id = atoi(argv[arg_index++]);
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
    else
    {
      arg_index++;
    }
  }

  ierr = HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                    HYPRE_PARCSR, &ij_A );
  if (ierr)
  {
    hypre_printf("ERROR: Problem reading in the system matrix!\n");
    exit(1);
  }

  if (build_matrix_type < 0)
  {
    ierr = HYPRE_IJMatrixGetLocalRange( ij_A,
                            &first_local_row, &last_local_row ,
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

    values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
    for (i = 0; i < local_num_cols; i++)
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
    hypre_TFree(values, HYPRE_MEMORY_HOST);

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

    /* RHS */
    HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
    HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_b);

    values = hypre_CTAlloc(HYPRE_Real,  local_num_rows, HYPRE_MEMORY_HOST);
    for (i = 0; i < local_num_rows; i++) 
      values[i] = 1.0; 
    HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
    hypre_TFree(values, HYPRE_MEMORY_HOST);

    ierr = HYPRE_IJVectorGetObject( ij_b, &object );
    b = (HYPRE_ParVector) object;

    /* Initial guess */
    HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
    HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_x);

    values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
    for (i = 0; i < local_num_cols; i++) 
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
    hypre_TFree(values, HYPRE_MEMORY_HOST);

    ierr = HYPRE_IJVectorGetObject( ij_x, &object );
    x = (HYPRE_ParVector) object;
  }


  if (indexList != NULL) {
    FILE* ifp;
    ifp = fopen(indexList,"r");
    if (ifp == NULL) {
      fprintf(stderr, "Can't open input file for index list!\n");
      exit(1);
    }
    fscanf(ifp, "%d", &mgr_num_reserved_nodes);
    fprintf(stderr, "There are %d additional indices\n", mgr_num_reserved_nodes);
    mgr_reserved_coarse_indexes = hypre_CTAlloc(HYPRE_Int, mgr_num_reserved_nodes, HYPRE_MEMORY_HOST);
    HYPRE_Int idx = 0;
    for (i = 0; i < mgr_num_reserved_nodes; i++) {
      //additional_coarse_indices[i] = 3*(i+1) + 3*n_gas_active;
      //additional_coarse_indices[i] = 3*(i+1);
      fscanf(ifp, "%d", &idx);
      //fprintf(stderr, "Reading index %d\n", idx);
      mgr_reserved_coarse_indexes[i] = idx;
    }
  } else {
    mgr_num_reserved_nodes = 0;
    mgr_reserved_coarse_indexes = NULL;
  }

  if (use_block_cf)
  {
    FILE *ifp;
    char fname[80];
    //hypre_sprintf(fname,'%s.%5d',argv[build_block_cf_arg_index],myid);
    hypre_sprintf(fname, "%s.%05i", argv[build_block_cf_arg_index],myid);
    hypre_printf("Reading block CF indices from %s \n", fname);
    ifp = fopen(fname,"r");
    if (ifp == NULL) {
      fprintf(stderr, "Can't open input file for block CF indices!\n");
      exit(1);
    }
    for (i = 0; i < mgr_bsize; i++)
    {
      fscanf(ifp, "%d", &mgr_idx_array[i]);
    }
  }


  if( solver_id == 72 )
  {
    time_index = hypre_InitializeTiming("FlexGMRES Setup");
    hypre_BeginTiming(time_index);
 
    HYPRE_ParCSRFlexGMRESCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
    HYPRE_FlexGMRESSetKDim(pcg_solver, k_dim);
    HYPRE_FlexGMRESSetMaxIter(pcg_solver, max_iter);
    HYPRE_FlexGMRESSetTol(pcg_solver, tol);
    HYPRE_FlexGMRESSetAbsoluteTol(pcg_solver, atol);
    HYPRE_FlexGMRESSetLogging(pcg_solver, 1);
    HYPRE_FlexGMRESSetPrintLevel(pcg_solver, 2);

    ierr = HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &aux_precond);
    HYPRE_EuclidSetLevel(aux_precond, 5);
    //HYPRE_EuclidSetSparseA(aux_precond, 1.0e-5);
    //HYPRE_EuclidSetBJ(aux_precond, 1);
    HYPRE_ParCSRFlexGMRESCreate(hypre_MPI_COMM_WORLD, &aux_solver);
    HYPRE_FlexGMRESSetMaxIter(aux_solver, 10);
    HYPRE_FlexGMRESSetTol(aux_solver, 1e-12);
    HYPRE_FlexGMRESSetPrecond(aux_solver,
                (HYPRE_PtrToSolverFcn)HYPRE_EuclidSolve,
                (HYPRE_PtrToSolverFcn)HYPRE_EuclidSetup,
                 aux_precond);

    /*
    HYPRE_BoomerAMGCreate(&aux_precond);
    HYPRE_BoomerAMGSetRelaxOrder(aux_precond, 1);
    HYPRE_BoomerAMGSetMaxIter(aux_precond, 1);
    HYPRE_ParCSRFlexGMRESCreate(hypre_MPI_COMM_WORLD, &aux_solver);
    HYPRE_FlexGMRESSetMaxIter(aux_solver, 10);
    HYPRE_FlexGMRESSetPrecond(aux_solver,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                 aux_precond);
    */

    /* use MGR preconditioning */
    if (myid == 0) hypre_printf("Solver:  MGR-FlexGMRES\n");
  
    HYPRE_MGRCreate(&pcg_precond);

    /*
    mgr_num_cindexes = hypre_CTAlloc(HYPRE_Int,  mgr_nlevels, HYPRE_MEMORY_HOST);
    for(i=0; i<mgr_nlevels; i++)
    { // assume 1 coarse index per level //
      mgr_num_cindexes[i] = 1;
    }
    mgr_cindexes = hypre_CTAlloc(HYPRE_Int*,  mgr_nlevels, HYPRE_MEMORY_HOST);
    for(i=0; i<mgr_nlevels; i++)
    {
      mgr_cindexes[i] = hypre_CTAlloc(HYPRE_Int,  mgr_num_cindexes[i], HYPRE_MEMORY_HOST);
    }
    for(i=0; i<mgr_nlevels; i++)
    { // assume coarse point is at index 0 //
      mgr_cindexes[i][0] = 0;
    }
    mgr_reserved_coarse_indexes = hypre_CTAlloc(HYPRE_Int,  mgr_num_reserved_nodes, HYPRE_MEMORY_HOST);
    for(i=0; i<mgr_num_reserved_nodes; i++)
    { // generate artificial reserved nodes //
      mgr_reserved_coarse_indexes[i] = last_local_row-i;//2*i+1;
    }
    */
  
    /* set MGR data by block */
    if (use_block_cf) {
       HYPRE_MGRSetCpointsByGlobalBlock( pcg_precond, mgr_bsize, mgr_nlevels, mgr_idx_array, mgr_num_cindexes, mgr_cindexes);
    } else {
       HYPRE_MGRSetCpointsByLocalBlock( pcg_precond, mgr_bsize, mgr_nlevels, mgr_num_cindexes,mgr_cindexes);
    }
    /* set reserved coarse nodes */
    if(mgr_num_reserved_nodes)HYPRE_MGRSetReservedCoarseNodes(pcg_precond, mgr_num_reserved_nodes, mgr_reserved_coarse_indexes);
  
    /* set intermediate coarse grid strategy */
    HYPRE_MGRSetNonCpointsToFpoints(pcg_precond, mgr_non_c_to_f);
    /* set F relaxation strategy */
    HYPRE_MGRSetFRelaxMethod(pcg_precond, mgr_frelax_method);
    /* set F relaxation number of functions*/
    HYPRE_MGRSetFRelaxNumFunctions(pcg_precond, mgr_frelax_num_functions);
    /* set relax type for single level F-relaxation and post-relaxation */
    HYPRE_MGRSetRelaxType(pcg_precond, mgr_relax_type);
    HYPRE_MGRSetNumRelaxSweeps(pcg_precond, mgr_num_relax_sweeps);
    /* set restrict type */
    HYPRE_MGRSetRestrictType(pcg_precond, mgr_restrict_type);
    HYPRE_MGRSetNumRestrictSweeps(pcg_precond, mgr_num_restrict_sweeps);
    /* set interpolation type */
    HYPRE_MGRSetInterpType(pcg_precond, mgr_interp_type);
    HYPRE_MGRSetNumInterpSweeps(pcg_precond, mgr_num_interp_sweeps);
    /* set P_max_elmts for coarse grid */
    HYPRE_MGRSetPMaxElmts(pcg_precond, P_max_elmts);
    /* set print level */
    HYPRE_MGRSetPrintLevel(pcg_precond, poutdat);
    /* set max iterations */
    HYPRE_MGRSetMaxIter(pcg_precond, 1);
    HYPRE_MGRSetTol(pcg_precond, pc_tol);

    HYPRE_MGRSetGlobalsmoothType(pcg_precond, mgr_gsmooth_type);
    HYPRE_MGRSetMaxGlobalsmoothIters( pcg_precond, mgr_num_gsmooth_sweeps );   
    hypre_MGRPrintCoarseSystem( pcg_precond, 0 );
  
    /* create AMG coarse grid solver */
  
    HYPRE_BoomerAMGCreate(&amg_solver); 
    /* BM Aug 25, 2006 */
    //HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
    //HYPRE_BoomerAMGSetInterpType(amg_solver, 0);
    //HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
    //HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
    //HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0);
    /* note: log is written to standard output, not to file */
    HYPRE_BoomerAMGSetPrintLevel(amg_solver, 0);
    /*HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
    HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
    HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);
    if (relax_down > -1)
      HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
    if (relax_up > -1)
      HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
    if (relax_coarse > -1)
      HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
    HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
    HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
    HYPRE_BoomerAMGSetTol(amg_solver, 0.0);*/
    HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);
    HYPRE_BoomerAMGSetMaxIter(amg_solver, 1);
    /*
    HYPRE_BoomerAMGSetSmoothType(amg_solver, 9);
    HYPRE_BoomerAMGSetEuLevel(amg_solver, 5);
    HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, 4);
    HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, 1);
    */

    /* set the MGR coarse solver. Comment out to use default CG solver in MGR */
    HYPRE_MGRSetCoarseSolver( pcg_precond, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, amg_solver);
    //HYPRE_MGRSetCoarseSolver( pcg_precond, (HYPRE_PtrToParSolverFcn)HYPRE_FlexGMRESSolve, (HYPRE_PtrToParSolverFcn)HYPRE_FlexGMRESSetup, aux_solver); 
  
    /* setup MGR-PCG solver */
    HYPRE_FlexGMRESSetMaxIter(pcg_solver, mg_max_iter);
    HYPRE_FlexGMRESSetPrecond(pcg_solver,
        (HYPRE_PtrToSolverFcn) HYPRE_MGRSolve,
        (HYPRE_PtrToSolverFcn) HYPRE_MGRSetup,
                      pcg_precond);


    HYPRE_FlexGMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
    if (pcg_precond_gotten != pcg_precond)
    {
      hypre_printf("HYPRE_FlexGMRESGetPrecond got bad precond\n");
      return(-1);
    }
    else
      if (myid == 0)
        hypre_printf("HYPRE_FlexGMRESGetPrecond got good precond\n");


    HYPRE_FlexGMRESSetup
      (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

    hypre_EndTiming(time_index);
    hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_index);
    hypre_ClearTiming();

    time_index = hypre_InitializeTiming("FlexGMRES Solve");
    hypre_BeginTiming(time_index);

    HYPRE_FlexGMRESSolve
      (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

    hypre_EndTiming(time_index);
    hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_index);
    hypre_ClearTiming();

    HYPRE_FlexGMRESGetNumIterations(pcg_solver, &num_iterations);
    HYPRE_FlexGMRESGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);

    // free memory for flex gmres
    HYPRE_ParCSRFlexGMRESDestroy(pcg_solver);
  
    /* free memory for MGR */
    /*
    if(mgr_num_cindexes)
      hypre_TFree(mgr_num_cindexes, HYPRE_MEMORY_HOST);
    mgr_num_cindexes = NULL; 
  
    if(mgr_reserved_coarse_indexes)
      hypre_TFree(mgr_reserved_coarse_indexes, HYPRE_MEMORY_HOST);
    mgr_reserved_coarse_indexes = NULL; 
  
    if(mgr_cindexes)
    {
      for( i=0; i<mgr_nlevels; i++)
      {
      if(mgr_cindexes[i])
        hypre_TFree(mgr_cindexes[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(mgr_cindexes, HYPRE_MEMORY_HOST);
      mgr_cindexes = NULL;
    }
    */
    HYPRE_BoomerAMGDestroy(amg_solver);
    HYPRE_MGRDestroy(pcg_precond);

    // Print out solver summary
    if (myid == 0)
    {
      hypre_printf("\n");
      hypre_printf("FlexGMRES Iterations = %d\n", num_iterations);
      hypre_printf("Final FlexGMRES Relative Residual Norm = %e\n", final_res_norm);
      hypre_printf("\n");
    }
  }
  else if (solver_id == 70)
  {
    if (myid == 0) hypre_printf("Solver:  MGR\n");
    time_index = hypre_InitializeTiming("MGR Setup");
    hypre_BeginTiming(time_index);
    
    HYPRE_Solver mgr_solver;
    HYPRE_MGRCreate(&mgr_solver);

    /* set MGR data by block */
    if (use_block_cf) {
       HYPRE_MGRSetCpointsByGlobalBlock( pcg_precond, mgr_bsize, mgr_nlevels, mgr_idx_array, mgr_num_cindexes, mgr_cindexes);
    } else {
       HYPRE_MGRSetCpointsByLocalBlock( pcg_precond, mgr_bsize, mgr_nlevels, mgr_num_cindexes,mgr_cindexes);
    } 

    /* set reserved coarse nodes */
    if(mgr_num_reserved_nodes)HYPRE_MGRSetReservedCoarseNodes(mgr_solver, mgr_num_reserved_nodes, mgr_reserved_coarse_indexes);
    
    /* set intermediate coarse grid strategy */
    HYPRE_MGRSetNonCpointsToFpoints(mgr_solver, mgr_non_c_to_f);
    /* set F relaxation strategy */
    HYPRE_MGRSetFRelaxMethod(mgr_solver, mgr_frelax_method);
    /* set relax type for single level F-relaxation and post-relaxation */
    HYPRE_MGRSetRelaxType(mgr_solver, mgr_relax_type);
    HYPRE_MGRSetNumRelaxSweeps(mgr_solver, mgr_num_relax_sweeps);
    /* set interpolation type */
    HYPRE_MGRSetRestrictType(mgr_solver, mgr_restrict_type);
    HYPRE_MGRSetNumRestrictSweeps(mgr_solver, mgr_num_restrict_sweeps);
    HYPRE_MGRSetInterpType(mgr_solver, mgr_interp_type);
    HYPRE_MGRSetNumInterpSweeps(mgr_solver, mgr_num_interp_sweeps);
    /* set print level */
    HYPRE_MGRSetPrintLevel(mgr_solver, poutdat);
    /* set max iterations */
    HYPRE_MGRSetMaxIter(mgr_solver, max_iter);
    HYPRE_MGRSetTol(mgr_solver, tol);

    HYPRE_MGRSetGlobalsmoothType(mgr_solver, mgr_gsmooth_type);
    HYPRE_MGRSetMaxGlobalsmoothIters( mgr_solver, mgr_num_gsmooth_sweeps );
    
    /* create AMG coarse grid solver */
    /*    
    HYPRE_BoomerAMGCreate(&amg_solver); 
    HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
    HYPRE_BoomerAMGSetInterpType(amg_solver, 0);
    HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
    HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
    HYPRE_BoomerAMGSetTol(amg_solver, tol);
    HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0);
    HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
    HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
    HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);
    if (relax_down > -1)
      HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
    if (relax_up > -1)
      HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
    if (relax_coarse > -1)
      HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
    HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);
    HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
    HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
    HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
    if(mgr_nlevels < 1 || mgr_bsize < 2)
    {
      HYPRE_BoomerAMGSetMaxIter(amg_solver, max_iter);
      HYPRE_BoomerAMGSetPrintLevel(amg_solver, 3);
    }
    else
    {
      HYPRE_BoomerAMGSetMaxIter(amg_solver, 1);
      HYPRE_BoomerAMGSetTol(amg_solver, 0.0);
      HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);
    }
    */

    /* create AMG coarse grid solver */
  
    HYPRE_BoomerAMGCreate(&amg_solver); 
    /* BM Aug 25, 2006 */
    //HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
    //HYPRE_BoomerAMGSetInterpType(amg_solver, 0);
    //HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
    //HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
    //HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0);
    /* note: log is written to standard output, not to file */
    HYPRE_BoomerAMGSetPrintLevel(amg_solver, 2);
    /*HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
    HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
    HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);
    if (relax_down > -1)
      HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
    if (relax_up > -1)
      HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
    if (relax_coarse > -1)
      HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
    HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
    HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
    HYPRE_BoomerAMGSetTol(amg_solver, 0.0);*/
    HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);
    HYPRE_BoomerAMGSetMaxIter(amg_solver, 1);
    HYPRE_BoomerAMGSetSmoothType(amg_solver, 9);
    HYPRE_BoomerAMGSetEuLevel(amg_solver, 1);
    HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, 1);
    HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, 1);

    /* set the MGR coarse solver. Comment out to use default CG solver in MGR */
    HYPRE_MGRSetCoarseSolver( mgr_solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, amg_solver);
    
    /* setup MGR solver */
    HYPRE_MGRSetup(mgr_solver, parcsr_A, b, x);

    hypre_EndTiming(time_index);
    hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_index);
    hypre_ClearTiming();
 
    time_index = hypre_InitializeTiming("MGR Solve");
    hypre_BeginTiming(time_index);
    
    /* MGR solve */
    HYPRE_MGRSolve(mgr_solver, parcsr_A, b, x);
    
    hypre_EndTiming(time_index);
    hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_index);
    hypre_ClearTiming();

    HYPRE_MGRGetNumIterations(mgr_solver, &num_iterations);
    HYPRE_MGRGetFinalRelativeResidualNorm(mgr_solver, &final_res_norm);

    if (myid == 0)
    {
      hypre_printf("\n");
      hypre_printf("MGR Iterations = %d\n", num_iterations);
      hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
      hypre_printf("\n");
    }

#if SECOND_TIME
    /* run a second time to check for memory leaks */
    HYPRE_ParVectorSetRandomValues(x, 775);
    HYPRE_MGRSetup(mgr_solver, parcsr_A, b, x);
    HYPRE_MGRSolve(mgr_solver, parcsr_A, b, x);
#endif
    
    /* free memory */
    if(mgr_num_cindexes)
      hypre_TFree(mgr_num_cindexes, HYPRE_MEMORY_HOST);
    mgr_num_cindexes = NULL; 
    
    if(mgr_reserved_coarse_indexes)
      hypre_TFree(mgr_reserved_coarse_indexes, HYPRE_MEMORY_HOST);
    mgr_reserved_coarse_indexes = NULL; 
    
    if(mgr_cindexes)
    {
      for( i=0; i<mgr_nlevels; i++)
      {
        if(mgr_cindexes[i])
          hypre_TFree(mgr_cindexes[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(mgr_cindexes, HYPRE_MEMORY_HOST);
      mgr_cindexes = NULL;
    }

    HYPRE_BoomerAMGDestroy(amg_solver);      
    HYPRE_MGRDestroy(mgr_solver);
  }

  /*-----------------------------------------------------------
   * Finalize things
   *-----------------------------------------------------------*/
  // free the matrix, the rhs and the initial guess
  HYPRE_IJMatrixDestroy(ij_A);
  HYPRE_IJVectorDestroy(ij_b);
  HYPRE_IJVectorDestroy(ij_x);

  /*
  hypre_FinalizeMemoryDebug();
  */

  /* GPU finalize stuff inside */
  hypre_finalize();
  
  hypre_MPI_Finalize();

  return (0);
}
