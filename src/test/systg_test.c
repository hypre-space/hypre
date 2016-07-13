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
//#include <stdlib.h>
//#include <stdio.h>
#include <math.h>
 
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"

HYPRE_Int BuildParFromFile (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParRhsFromFile (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParVector *b_ptr );

HYPRE_Int BuildParLaplacian (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParSysLaplacian (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParDifConv (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParFromOneFile (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_Int num_functions , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildFuncsFromFiles (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix A , HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildFuncsFromOneFile (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix A , HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildRhsParFromOneFile (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_Int *partitioning , HYPRE_ParVector *b_ptr );
HYPRE_Int BuildParLaplacian9pt (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian27pt (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParRotate7pt (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParVarDifConv (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr , HYPRE_ParVector *rhs_ptr );
HYPRE_ParCSRMatrix GenerateSysLaplacian (MPI_Comm comm, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz, 
                                         HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                         HYPRE_Int num_fun, HYPRE_Real *mtrx, HYPRE_Real *value);
HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef (MPI_Comm comm, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz, 
                                              HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                              HYPRE_Int num_fun, HYPRE_Real *mtrx, HYPRE_Real *value);
HYPRE_Int SetSysVcoefValues(HYPRE_Int num_fun, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz, HYPRE_Real vcx, HYPRE_Real vcy, HYPRE_Real vcz, HYPRE_Int mtx_entry, HYPRE_Real *values);

HYPRE_Int BuildParCoordinates (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_Int *coorddim_ptr , float **coord_ptr );
                                                                                
extern HYPRE_Int hypre_FlexGMRESModifyPCAMGExample(void *precond_data, HYPRE_Int iterations, 
                                                   HYPRE_Real rel_residual_norm);

extern HYPRE_Int hypre_FlexGMRESModifyPCDefault(void *precond_data, HYPRE_Int iteration, 
                                                HYPRE_Real rel_residual_norm);


#define SECOND_TIME 0
 
hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int                 arg_index;
   HYPRE_Int                 print_usage;
   HYPRE_Int                 sparsity_known = 0;
   HYPRE_Int                 add = 0;
   HYPRE_Int                 off_proc = 0;
   HYPRE_Int                 chunk = 0;
   HYPRE_Int                 omp_flag = 0;
   HYPRE_Int                 build_matrix_type;
   HYPRE_Int                 build_matrix_arg_index;
   HYPRE_Int                 build_rhs_type;
   HYPRE_Int                 build_rhs_arg_index;
   HYPRE_Int                 build_src_type;
   HYPRE_Int                 build_src_arg_index;
   HYPRE_Int                 build_funcs_type;
   HYPRE_Int                 build_funcs_arg_index;
   HYPRE_Int                 solver_id;
   HYPRE_Int                 solver_type = 1;
   HYPRE_Int                 ioutdat;
   HYPRE_Int                 poutdat;
   HYPRE_Int                 debug_flag;
   HYPRE_Int                 ierr = 0;
   HYPRE_Int                 i,j; 
   HYPRE_Int                 max_levels = 25;
   HYPRE_Int                 num_iterations;
   HYPRE_Int                 pcg_num_its, dscg_num_its;
   HYPRE_Int                 max_iter = 1;
   HYPRE_Int                 mg_max_iter = 1000;
   HYPRE_Int                 nodal = 0;
   HYPRE_Int                 nodal_diag = 0;
   HYPRE_Real          cf_tol = 0.9;
   HYPRE_Real          norm;
   HYPRE_Real          final_res_norm;
   void               *object;

   HYPRE_IJMatrix      ij_A; 
   HYPRE_IJVector      ij_b;
   HYPRE_IJVector      ij_x;

   HYPRE_ParCSRMatrix  parcsr_A;
   HYPRE_ParCSRMatrix  parcsr_B;
   HYPRE_ParCSRMatrix  parcsr_C;
   
   HYPRE_ParVector     b;
   HYPRE_ParVector     x;
   HYPRE_ParVector     bb;
   
   HYPRE_Solver        amg_solver;
   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond=NULL, pcg_precond_gotten;
   HYPRE_Solver        aux_solver=NULL,aux_precond=NULL;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 num_threads;
   HYPRE_Int                 local_row;
   HYPRE_Int                *row_sizes;
   HYPRE_Int                *diag_sizes;
   HYPRE_Int                *offdiag_sizes;
   HYPRE_Int                *rows;
   HYPRE_Int                 size;
   HYPRE_Int                *ncols;
   HYPRE_Int                *col_inds;
   HYPRE_Int                *dof_func;
   HYPRE_Int		       num_functions = 1;
   HYPRE_Int		       num_paths = 1;
   HYPRE_Int		       agg_num_levels = 0;
   HYPRE_Int		       ns_coarse = 1;

   HYPRE_Int		       time_index;
   MPI_Comm            comm = hypre_MPI_COMM_WORLD;
   HYPRE_Int M, N;
   HYPRE_Int first_local_row, last_local_row, local_num_rows;
   HYPRE_Int first_local_col, last_local_col, local_num_cols;
   HYPRE_Int variant, overlap, domain_type;
   HYPRE_Real schwarz_rlx_weight;
   HYPRE_Real *values, val;

   HYPRE_Int use_nonsymm_schwarz = 0;
   HYPRE_Int test_ij = 0;

   const HYPRE_Real dt_inf = 1.e40;
   HYPRE_Real dt = dt_inf;

   /* parameters for BoomerAMG */
   HYPRE_Real   strong_threshold;
   HYPRE_Real   trunc_factor;
   HYPRE_Real   jacobi_trunc_threshold;
   HYPRE_Real   S_commpkg_switch = 1.0;
   HYPRE_Real   CR_rate = 0.7;
   HYPRE_Real   CR_strong_th = 0.0;
   HYPRE_Int      CR_use_CG = 0;
   HYPRE_Int      P_max_elmts = 0;
   HYPRE_Int      cycle_type;
   HYPRE_Int      coarsen_type = 6;
   HYPRE_Int      measure_type = 0;
   HYPRE_Int      num_sweeps = 1;  
   HYPRE_Int      IS_type;   
   HYPRE_Int      num_CR_relax_steps = 2;   
   HYPRE_Int      relax_type;   
   HYPRE_Int      relax_coarse = -1;   
   HYPRE_Int      relax_up = -1;   
   HYPRE_Int      relax_down = -1;   
   HYPRE_Int      relax_order = 1;   
   HYPRE_Int      level_w = -1;
   HYPRE_Int      level_ow = -1;
/* HYPRE_Int	    smooth_lev; */
/* HYPRE_Int	    smooth_rlx = 8; */
   HYPRE_Int	    smooth_type = 6;
   HYPRE_Int	    smooth_num_levels = 0;
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
   HYPRE_Int add_P_max_elmts = 0;
   HYPRE_Real add_trunc_factor = 0;

   HYPRE_Real   relax_wt; 
   HYPRE_Real   relax_wt_level; 
   HYPRE_Real   outer_wt;
   HYPRE_Real   outer_wt_level;
   HYPRE_Real   tol = 1.e-5, pc_tol = 0.;
   HYPRE_Real   atol = 0.0;
   HYPRE_Real   max_row_sum = 1.;

   HYPRE_Int cheby_order = 2;
   HYPRE_Real cheby_fraction = .3;

   /* for CGC BM Aug 25, 2006 */
   HYPRE_Int      cgcits = 1;
   /* for coordinate plotting BM Oct 24, 2006 */
   HYPRE_Int      plot_grids = 0;
   HYPRE_Int      coord_dim  = 3;
   float    *coordinates = NULL;
   char    plot_file_name[256];
   
   /* parameters for ParaSAILS */
   HYPRE_Real   sai_threshold = 0.1;
   HYPRE_Real   sai_filter = 0.1;

   /* parameters for PILUT */
   HYPRE_Real   drop_tol = -1;
   HYPRE_Int      nonzeros_to_keep = -1;

   /* parameters for Euclid or ILU smoother in AMG */
   HYPRE_Real   eu_ilut = 0.0;
   HYPRE_Real   eu_sparse_A = 0.0;
   HYPRE_Int	    eu_bj = 0;
   HYPRE_Int	    eu_level = -1;
   HYPRE_Int	    eu_stats = 0;
   HYPRE_Int	    eu_mem = 0;
   HYPRE_Int	    eu_row_scale = 0; /* Euclid only */

   /* parameters for GMRES */
   HYPRE_Int	    k_dim;
   /* parameters for LGMRES */
   HYPRE_Int	    aug_dim;
   /* parameters for GSMG */
   HYPRE_Int      gsmg_samples = 5;
   /* interpolation */
   HYPRE_Int      interp_type  = 0; /* default value */
   HYPRE_Int      post_interp_type  = 0; /* default value */
   /* aggressive coarsening */
   HYPRE_Int      agg_interp_type  = 4; /* default value */
   HYPRE_Int      agg_P_max_elmts  = 0; /* default value */
   HYPRE_Int      agg_P12_max_elmts  = 0; /* default value */
   HYPRE_Real   agg_trunc_factor  = 0; /* default value */
   HYPRE_Real   agg_P12_trunc_factor  = 0; /* default value */

   HYPRE_Int      print_system = 0;

   HYPRE_Int rel_change = 0;

   HYPRE_Real     *nongalerk_tol = NULL;
   HYPRE_Int       nongalerk_num_tol = 0;

   HYPRE_Int *row_nums = NULL;
   HYPRE_Int *num_cols = NULL;
   HYPRE_Int *col_nums = NULL;
   HYPRE_Int i_indx, j_indx, num_rows;
   HYPRE_Real *data = NULL;

   HYPRE_Int blk_size  = 3;
   HYPRE_Int num_wells = 0;
   
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   num_threads = hypre_NumThreads();
/*
  hypre_InitMemoryDebug(myid);
*/
   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   build_matrix_type = 2;
   build_matrix_arg_index = argc;
   build_rhs_type = 2;
   build_rhs_arg_index = argc;
   build_src_type = -1;
   build_src_arg_index = argc;
   build_funcs_type = 0;
   build_funcs_arg_index = argc;
   relax_type = 3;
   IS_type = 1;
   debug_flag = 0;

   solver_id = 0;

   ioutdat = 3;
   poutdat = 1;

   hypre_sprintf (plot_file_name,"AMGgrids.CF.dat");

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
      else if ( strcmp(argv[arg_index], "-fromonecsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-test_ij") == 0 )
      {
         arg_index++;
         test_ij = 1;
      }
      else if ( strcmp(argv[arg_index], "-exact_size") == 0 )
      {
         arg_index++;
         sparsity_known = 1;
      }
      else if ( strcmp(argv[arg_index], "-storage_low") == 0 )
      {
         arg_index++;
         sparsity_known = 2;
      }
      else if ( strcmp(argv[arg_index], "-add") == 0 )
      {
         arg_index++;
         add = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-chunk") == 0 )
      {
         arg_index++;
         chunk = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-off_proc") == 0 )
      {
         arg_index++;
         off_proc = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-omp") == 0 )
      {
         arg_index++;
         omp_flag = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
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
      else if ( strcmp(argv[arg_index], "-crcg") == 0 )
      {
         arg_index++;
         CR_use_CG = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-ncr") == 0 )
      {
         arg_index++;
         num_CR_relax_steps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-crth") == 0 )
      {
         arg_index++;
         CR_rate = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-crst") == 0 )
      {
         arg_index++;
         CR_strong_th = atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-mg_max_iter") == 0 )
      {
         arg_index++;
         mg_max_iter = atoi(argv[arg_index++]);
      }

      else if ( strcmp(argv[arg_index], "-dt") == 0 )
      {
         arg_index++;
         dt = atof(argv[arg_index++]);
         build_rhs_type = -1;
         if ( build_src_type == -1 ) build_src_type = 2;
      }
	  else if ( strcmp(argv[arg_index], "-blksize") == 0 )
	  {
		  arg_index++;
		  blk_size = atoi(argv[arg_index++]);
	  }
	  else if ( strcmp(argv[arg_index], "-numWells") == 0)
	  {
		  arg_index++;
		  num_wells = atoi(argv[arg_index++]);
	  }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      else
      {
         arg_index++;
      }
   }

   /* begin CGC BM Aug 25, 2006 */
   if (coarsen_type == 21 || coarsen_type == 22) {
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

   if (solver_id == 8 || solver_id == 18)
   {
      max_levels = 1;
   }

   /* defaults for BoomerAMG */
   if (solver_id == 0 || solver_id == 1 || solver_id == 3 || solver_id == 5
       || solver_id == 9 || solver_id == 13 || solver_id == 14
       || solver_id == 15 || solver_id == 20 || solver_id == 30 || solver_id == 51 || solver_id == 61 || solver_id == 62)
   {
      strong_threshold = 0.25;
      trunc_factor = 0.;
      jacobi_trunc_threshold = 0.01;
      cycle_type = 1;
      relax_wt = 1.;
      outer_wt = 1.;

      /* for CGNR preconditioned with Boomeramg, only relaxation scheme 0 is
         implemented, i.e. Jacobi relaxation, and needs to be used without CF
         ordering */
      if (solver_id == 5) 
      {     
         relax_type = 0;
         relax_order = 0;
      }
	  if (solver_id == 0 || solver_id == 30 || solver_id == 62)
	  {
		  relax_order = 0;
	  }
	  
   }

   /* defaults for Schwarz */

   variant = 0;  /* multiplicative */
   overlap = 1;  /* 1 layer overlap */
   domain_type = 2; /* through agglomeration */
   schwarz_rlx_weight = 1.;

   /* defaults for GMRES */

   k_dim = 5;

   /* defaults for LGMRES - should use a larger k_dim, though*/
   aug_dim = 2;

   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-aug") == 0 )
      {
         arg_index++;
         aug_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         relax_wt = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-wl") == 0 )
      {
         arg_index++;
         relax_wt_level = atof(argv[arg_index++]);
         level_w = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ow") == 0 )
      {
         arg_index++;
         outer_wt = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-owl") == 0 )
      {
         arg_index++;
         outer_wt_level = atof(argv[arg_index++]);
         level_ow = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sw") == 0 )
      {
         arg_index++;
         schwarz_rlx_weight = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-coarse_th") == 0 )
      {
         arg_index++;
         coarse_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-min_cs") == 0 )
      {
         arg_index++;
         min_coarse_size  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seq_th") == 0 )
      {
         arg_index++;
         seq_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-red") == 0 )
      {
         arg_index++;
         redundant  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         strong_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-CF") == 0 )
      {
         arg_index++;
         relax_order = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-atol") == 0 )
      {
         arg_index++;
         atol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index++;
         max_row_sum  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_th") == 0 )
      {
         arg_index++;
         sai_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_filt") == 0 )
      {
         arg_index++;
         sai_filter  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-drop_tol") == 0 )
      {
         arg_index++;
         drop_tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nonzeros_to_keep") == 0 )
      {
         arg_index++;
         nonzeros_to_keep  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ilut") == 0 )
      {
         arg_index++;
         eu_ilut  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sparseA") == 0 )
      {
         arg_index++;
         eu_sparse_A  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rowScale") == 0 )
      {
         arg_index++;
         eu_row_scale  = 1;
      }
      else if ( strcmp(argv[arg_index], "-level") == 0 )
      {
         arg_index++;
         eu_level  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-bj") == 0 )
      {
         arg_index++;
         eu_bj  = 1;
      }
      else if ( strcmp(argv[arg_index], "-eu_stats") == 0 )
      {
         arg_index++;
         eu_stats  = 1;
      }
      else if ( strcmp(argv[arg_index], "-eu_mem") == 0 )
      {
         arg_index++;
         eu_mem  = 1;
      }
      else if ( strcmp(argv[arg_index], "-tr") == 0 )
      {
         arg_index++;
         trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Pmx") == 0 )
      {
         arg_index++;
         P_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-jtr") == 0 )
      {
         arg_index++;
         jacobi_trunc_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Ssw") == 0 )
      {
         arg_index++;
         S_commpkg_switch = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver_type") == 0 )
      {
         arg_index++;
         solver_type  = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-numsamp") == 0 )
      {
         arg_index++;
         gsmg_samples  = atoi(argv[arg_index++]);
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
         agg_trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_P12_tr") == 0 )
      {
         arg_index++;
         agg_P12_trunc_factor  = atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-cheby_order") == 0 )
      {
         arg_index++;
         cheby_order = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_fraction") == 0 )
      {
         arg_index++;
         cheby_fraction = atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-add_Pmx") == 0 )
      {
         arg_index++;
         add_P_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_tr") == 0 )
      {
         arg_index++;
         add_trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nongalerk_tol") == 0 )
      {
         arg_index++;
         nongalerk_num_tol = atoi(argv[arg_index++]);
         nongalerk_tol = hypre_CTAlloc(HYPRE_Real, nongalerk_num_tol);
         for (i = 0; i < nongalerk_num_tol; i++)
            nongalerk_tol[i] = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
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
	 hypre_sprintf (plot_file_name,"%s",argv[arg_index++]);
      }
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

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   if ( build_matrix_type == -1 )
   {
      ierr = HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                                 HYPRE_PARCSR, &ij_A );
      if (ierr)
      {
		  hypre_printf("ERROR %d: Problem reading in the system matrix!\n",ierr);
         exit(1);
      }
   }

   if (build_matrix_type < 0)
   {
      ierr = HYPRE_IJMatrixGetLocalRange( ij_A,
                                          &first_local_row, &last_local_row ,
                                          &first_local_col, &last_local_col );

      local_num_rows = last_local_row - first_local_row + 1;
      local_num_cols = last_local_col - first_local_col + 1;
      ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
      //parcsr_A = (HYPRE_ParCSRMatrix) object;
	  parcsr_B = (HYPRE_ParCSRMatrix) object;
	  
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Generate Matrix", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   HYPRE_Int *row_A_partition;
   HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_B , &row_A_partition );
   b = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(parcsr_B),
								  hypre_ParCSRMatrixGlobalNumRows(parcsr_B),
								  hypre_ParCSRMatrixRowStarts(parcsr_B));
   hypre_ParVectorInitialize(b);
   hypre_ParVectorSetPartitioningOwner(b,0);
   
   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("RHS and Initial Guess");
   hypre_BeginTiming(time_index);
   
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
      bb = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

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

      values = hypre_CTAlloc(HYPRE_Real, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
         values[i] = 1.0;
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      bb = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 3 )
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector has random components and unit 2-norm\n");
         hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b); 
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* For purposes of this test, HYPRE_ParVector functions are used, but
         these are not necessary.  For a clean use of the interface, the user
         "should" modify components of ij_x by using functions
         HYPRE_IJVectorSetValues or HYPRE_IJVectorAddToValues */

      HYPRE_ParVectorSetRandomValues(b, 22775);
      HYPRE_ParVectorInnerProd(b,b,&norm);
      norm = 1./sqrt(norm);
      ierr = HYPRE_ParVectorScale(norm, b);      

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 4 )
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector set for solution with unit components\n");
         hypre_printf("  Initial guess is 0\n");
      }

      /* Temporary use of solution vector */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 1.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b); 
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      HYPRE_ParCSRMatrixMatvec(1.,parcsr_A,x,0.,b);

      /* Initial guess */
      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);
   }
   else if ( build_rhs_type == 5 )
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector is 0\n");
         hypre_printf("  Initial guess has unit components\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      values = hypre_CTAlloc(HYPRE_Real, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 1.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }

   if ( build_src_type == 0 )
   {
      if (myid == 0)
      {
         hypre_printf("  Source vector read from file %s\n", argv[build_src_arg_index]);
         hypre_printf("  Initial unknown vector in evolution is 0\n");
      }

      ierr = HYPRE_IJVectorRead( argv[build_src_arg_index], hypre_MPI_COMM_WORLD, 
                                 HYPRE_PARCSR, &ij_b );
      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in the right-hand-side!\n");
         exit(1);
      }
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial unknown vector */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 2 )
   {
      if (myid == 0)
      {
         hypre_printf("  Source vector has unit components\n");
         hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      values = hypre_CTAlloc(HYPRE_Real, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
         values[i] = 1.;
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 3 )
   {
      if (myid == 0)
      {
         hypre_printf("  Source vector has random components in range 0 - 1\n");
         hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      values = hypre_CTAlloc(HYPRE_Real, local_num_rows);

      hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
         values[i] = hypre_Rand();

      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 4 )
   {
      if (myid == 0)
      {
         hypre_printf("  Source vector is 0 \n");
         hypre_printf("  Initial unknown vector has random components in range 0 - 1\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      values = hypre_CTAlloc(HYPRE_Real, local_num_rows);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
         values[i] = hypre_Rand()/dt;
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
         values[i] = hypre_Rand();
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Vector Setup", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

      printf("build_rhs_type == %d\n",num_functions);
   if (num_functions > 1)
   {
      dof_func = NULL;
      if (build_funcs_type == 1)
      {
	 BuildFuncsFromOneFile(argc, argv, build_funcs_arg_index, parcsr_A, &dof_func);
      }
      else if (build_funcs_type == 2)
      {
	 BuildFuncsFromFiles(argc, argv, build_funcs_arg_index, parcsr_A, &dof_func);
      }
      else
      {
         if (myid == 0)
	    hypre_printf (" Number of functions = %d \n", num_functions);
      }
   }
 
   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_IJMatrixPrint(ij_A, "IJ.out.A");
      HYPRE_IJVectorPrint(ij_b, "IJ.out.b");
      HYPRE_IJVectorPrint(ij_x, "IJ.out.x0");

      /* HYPRE_ParCSRMatrixPrint( parcsr_A, "new_mat.A" );*/
   }

   /*-----------------------------------------------------------
    * Solve the system using the hybrid solver
    *-----------------------------------------------------------*/

   if (solver_id == 20)
   {
      if (myid == 0) hypre_printf("Solver:  AMG\n");
      time_index = hypre_InitializeTiming("AMG_hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRHybridCreate(&amg_solver);
      HYPRE_ParCSRHybridSetTol(amg_solver, tol);
      HYPRE_ParCSRHybridSetAbsoluteTol(amg_solver, atol);
      HYPRE_ParCSRHybridSetConvergenceTol(amg_solver, cf_tol);
      HYPRE_ParCSRHybridSetSolverType(amg_solver, solver_type);
      HYPRE_ParCSRHybridSetLogging(amg_solver, ioutdat);
      HYPRE_ParCSRHybridSetPrintLevel(amg_solver, poutdat);
      HYPRE_ParCSRHybridSetDSCGMaxIter(amg_solver, max_iter);
      HYPRE_ParCSRHybridSetPCGMaxIter(amg_solver, mg_max_iter);
      HYPRE_ParCSRHybridSetCoarsenType(amg_solver, coarsen_type);
      HYPRE_ParCSRHybridSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_ParCSRHybridSetTruncFactor(amg_solver, trunc_factor);
      HYPRE_ParCSRHybridSetPMaxElmts(amg_solver, P_max_elmts);
      HYPRE_ParCSRHybridSetMaxLevels(amg_solver, max_levels);
      HYPRE_ParCSRHybridSetMaxRowSum(amg_solver, max_row_sum);
      HYPRE_ParCSRHybridSetNumSweeps(amg_solver, num_sweeps);
      HYPRE_ParCSRHybridSetRelaxType(amg_solver, relax_type);
      HYPRE_ParCSRHybridSetAggNumLevels(amg_solver, agg_num_levels);
      HYPRE_ParCSRHybridSetNumPaths(amg_solver, num_paths);
      HYPRE_ParCSRHybridSetNumFunctions(amg_solver, num_functions);
      HYPRE_ParCSRHybridSetNodal(amg_solver, nodal);
      if (relax_down > -1)
         HYPRE_ParCSRHybridSetCycleRelaxType(amg_solver, relax_down, 1);
      if (relax_up > -1)
         HYPRE_ParCSRHybridSetCycleRelaxType(amg_solver, relax_up, 2);
      if (relax_coarse > -1)
         HYPRE_ParCSRHybridSetCycleRelaxType(amg_solver, relax_coarse, 3);
      HYPRE_ParCSRHybridSetRelaxOrder(amg_solver, relax_order);
      HYPRE_ParCSRHybridSetMaxCoarseSize(amg_solver, coarse_threshold);
      HYPRE_ParCSRHybridSetMinCoarseSize(amg_solver, min_coarse_size);
      HYPRE_ParCSRHybridSetSeqThreshold(amg_solver, seq_threshold);
      HYPRE_ParCSRHybridSetRelaxWt(amg_solver, relax_wt);
      HYPRE_ParCSRHybridSetOuterWt(amg_solver, outer_wt);
      if (level_w > -1)
         HYPRE_ParCSRHybridSetLevelRelaxWt(amg_solver, relax_wt_level, level_w);
      if (level_ow > -1)
         HYPRE_ParCSRHybridSetLevelOuterWt(amg_solver, outer_wt_level, level_ow);
  
      HYPRE_ParCSRHybridSetup(amg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("ParCSR Hybrid Solve");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRHybridSolve(amg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_ParCSRHybridGetNumIterations(amg_solver, &num_iterations);
      HYPRE_ParCSRHybridGetPCGNumIterations(amg_solver, &pcg_num_its);
      HYPRE_ParCSRHybridGetDSCGNumIterations(amg_solver, &dscg_num_its);
      HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(amg_solver, 
                                                     &final_res_norm);

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("PCG_Iterations = %d\n", pcg_num_its);
         hypre_printf("DSCG_Iterations = %d\n", dscg_num_its);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      HYPRE_ParCSRHybridDestroy(amg_solver);
   }
   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/
   if (solver_id == -1)
   {
	   HYPRE_Solver systg_solver;
	   
	   HYPRE_SysTGCreate(&systg_solver);

	   HYPRE_Int coarse_index = 0;
	   
	   HYPRE_SysTGSetBlockData( systg_solver, blk_size, 1, &coarse_index);
	   
	   HYPRE_SysTGSetNumWells(systg_solver,num_wells);
	   HYPRE_SysTGSetup(systg_solver, parcsr_A, b, x);

	   hypre_ParVector     	*Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(parcsr_A),
														   hypre_ParCSRMatrixGlobalNumRows(parcsr_A),
														   hypre_ParCSRMatrixRowStarts(parcsr_A));
	   hypre_ParVector     	*Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(parcsr_A),
														   hypre_ParCSRMatrixGlobalNumRows(parcsr_A),
														   hypre_ParCSRMatrixRowStarts(parcsr_A));
	   hypre_ParVectorInitialize(Vtemp);
	   hypre_ParVectorSetPartitioningOwner(Vtemp,0);
	   
	   printf("Starting the relaxation\n");
	   
	   hypre_blockRelax(parcsr_A, b,x, blk_size,num_wells,Vtemp,Ztemp);
	   hypre_blockRelax(parcsr_A, b,x, blk_size,num_wells,Vtemp,Ztemp);
   }
   
   if (solver_id == 0)
   {
      if (myid == 0) hypre_printf("Solver:  SysTG\n");
      time_index = hypre_InitializeTiming("SysTG Setup");
      hypre_BeginTiming(time_index);
      
	  HYPRE_Solver systg_solver;
	  HYPRE_SysTGCreate(&systg_solver);
      
      
      HYPRE_BoomerAMGCreate(&amg_solver); 
      /* BM Aug 25, 2006 */	  
      HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
      HYPRE_BoomerAMGSetInterpType(amg_solver, interp_type);
      HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
      HYPRE_BoomerAMGSetNumSamples(amg_solver, gsmg_samples);
      HYPRE_BoomerAMGSetCoarsenType(amg_solver, coarsen_type);
      HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_BoomerAMGSetSeqThreshold(amg_solver, seq_threshold);
      HYPRE_BoomerAMGSetRedundant(amg_solver, redundant);
      HYPRE_BoomerAMGSetMaxCoarseSize(amg_solver, coarse_threshold);
      HYPRE_BoomerAMGSetMinCoarseSize(amg_solver, min_coarse_size);
      HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
      HYPRE_BoomerAMGSetPMaxElmts(amg_solver, P_max_elmts);
      HYPRE_BoomerAMGSetJacobiTruncThreshold(amg_solver, jacobi_trunc_threshold);
      HYPRE_BoomerAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
/* note: log is written to standard output, not to file */
      HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log"); 
      HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
      HYPRE_BoomerAMGSetISType(amg_solver, IS_type);
      HYPRE_BoomerAMGSetNumCRRelaxSteps(amg_solver, num_CR_relax_steps);
      HYPRE_BoomerAMGSetCRRate(amg_solver, CR_rate);
      HYPRE_BoomerAMGSetCRStrongTh(amg_solver, CR_strong_th);
      HYPRE_BoomerAMGSetCRUseCG(amg_solver, CR_use_CG);
      HYPRE_BoomerAMGSetRelaxType(amg_solver, relax_type);
      if (relax_down > -1)
         HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
      if (relax_up > -1)
         HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
      if (relax_coarse > -1)
         HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
      HYPRE_BoomerAMGSetChebyOrder(amg_solver, cheby_order);
      HYPRE_BoomerAMGSetChebyFraction(amg_solver, cheby_fraction);
      HYPRE_BoomerAMGSetRelaxWt(amg_solver, relax_wt);
      HYPRE_BoomerAMGSetOuterWt(amg_solver, outer_wt);
      HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      if (level_w > -1)
         HYPRE_BoomerAMGSetLevelRelaxWt(amg_solver, relax_wt_level, level_w);
      if (level_ow > -1)
         HYPRE_BoomerAMGSetLevelOuterWt(amg_solver, outer_wt_level, level_ow);
      HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
      HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
      HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, smooth_num_levels);
      HYPRE_BoomerAMGSetMaxRowSum(amg_solver, max_row_sum);
      HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);
      HYPRE_BoomerAMGSetVariant(amg_solver, variant);
      HYPRE_BoomerAMGSetOverlap(amg_solver, overlap);
      HYPRE_BoomerAMGSetDomainType(amg_solver, domain_type);
      HYPRE_BoomerAMGSetSchwarzUseNonSymm(amg_solver, use_nonsymm_schwarz);
      HYPRE_BoomerAMGSetSchwarzRlxWeight(amg_solver, schwarz_rlx_weight);
      if (eu_level < 0) eu_level = 0;
      HYPRE_BoomerAMGSetEuLevel(amg_solver, eu_level);
      HYPRE_BoomerAMGSetEuBJ(amg_solver, eu_bj);
      HYPRE_BoomerAMGSetEuSparseA(amg_solver, eu_sparse_A);
      HYPRE_BoomerAMGSetNumFunctions(amg_solver, num_functions);
      HYPRE_BoomerAMGSetAggNumLevels(amg_solver, agg_num_levels);
      HYPRE_BoomerAMGSetAggInterpType(amg_solver, agg_interp_type);
      HYPRE_BoomerAMGSetAggTruncFactor(amg_solver, agg_trunc_factor);
      HYPRE_BoomerAMGSetAggP12TruncFactor(amg_solver, agg_P12_trunc_factor);
      HYPRE_BoomerAMGSetAggPMaxElmts(amg_solver, agg_P_max_elmts);
      HYPRE_BoomerAMGSetAggP12MaxElmts(amg_solver, agg_P12_max_elmts);
      HYPRE_BoomerAMGSetNumPaths(amg_solver, num_paths);
      HYPRE_BoomerAMGSetNodal(amg_solver, nodal);
      HYPRE_BoomerAMGSetNodalDiag(amg_solver, nodal_diag);
      HYPRE_BoomerAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);
      if (num_functions > 1)
	 HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);
      HYPRE_BoomerAMGSetAdditive(amg_solver, additive);
      HYPRE_BoomerAMGSetMultAdditive(amg_solver, mult_add);
      HYPRE_BoomerAMGSetSimple(amg_solver, simple);
      HYPRE_BoomerAMGSetAddPMaxElmts(amg_solver, add_P_max_elmts);
      HYPRE_BoomerAMGSetAddTruncFactor(amg_solver, add_trunc_factor);
      /* BM Oct 23, 2006 */
      if (plot_grids) {
         HYPRE_BoomerAMGSetPlotGrids (amg_solver, 1);
         HYPRE_BoomerAMGSetPlotFileName (amg_solver, plot_file_name);
         HYPRE_BoomerAMGSetCoordDim (amg_solver, coord_dim);
         HYPRE_BoomerAMGSetCoordinates (amg_solver, coordinates);
      }
	  HYPRE_BoomerAMGSetRelaxOrder(amg_solver, relax_order);
	  HYPRE_BoomerAMGSetMaxIter(amg_solver, 1);
	  HYPRE_BoomerAMGSetTol(amg_solver, tol);
	  HYPRE_BoomerAMGSetNonGalerkTol(amg_solver, nongalerk_num_tol, nongalerk_tol);
	  HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);
	  
	  HYPRE_Int coarse_index = 0;
	  HYPRE_SysTGSetBlockData( systg_solver, blk_size, 1, &coarse_index);
	  
	  HYPRE_SysTGSetNumWells(systg_solver,num_wells);
	  
	  HYPRE_SysTGSetCoarseSolver( systg_solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, amg_solver);
	  HYPRE_SysTGSetMaxCoarseLevels(systg_solver, 1);
	  HYPRE_SysTGSetRelaxType(systg_solver, 0);
	  HYPRE_SysTGSetNumRelaxSweeps(systg_solver, 1);
	  HYPRE_SysTGSetNumInterpSweeps(systg_solver, 1);
	  HYPRE_SysTGSetPrintLevel(systg_solver, 2);
	  HYPRE_SysTGSetMaxGlobalsmoothIters(systg_solver, 0);
	  HYPRE_SysTGSetGlobalsmoothType(systg_solver, 0);

	  parcsr_A = parcsr_B;
	  b = bb;
	  

	  HYPRE_SysTGSetup(systg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("SysTG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_SysTGSolve(systg_solver, parcsr_A, b, x);
      HYPRE_SysTGGetNumIterations(systg_solver, &num_iterations);
      HYPRE_SysTGGetResidualNorm(systg_solver, &final_res_norm);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();


      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("SysTG Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

#if SECOND_TIME
      /* run a second time to check for memory leaks */
//      HYPRE_ParVectorSetRandomValues(x, 775);
//      HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);
//      HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
#endif

	  HYPRE_SysTGDestroy(systg_solver);

      HYPRE_BoomerAMGDestroy(amg_solver);


   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES 
    *-----------------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4 || solver_id == 7 ||
       solver_id == 15 || solver_id == 18 || solver_id == 44 ||
       solver_id == 30)
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRGMRESCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_GMRESSetKDim(pcg_solver, k_dim);
      HYPRE_GMRESSetMaxIter(pcg_solver, max_iter);
      HYPRE_GMRESSetTol(pcg_solver, tol);
      HYPRE_GMRESSetAbsoluteTol(pcg_solver, atol);
      HYPRE_GMRESSetLogging(pcg_solver, 1);
      HYPRE_GMRESSetPrintLevel(pcg_solver, ioutdat);
      HYPRE_GMRESSetRelChange(pcg_solver, rel_change);

      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) hypre_printf("Solver: AMG-GMRES\n");

         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
         if (relax_down > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         if (relax_up > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         if (relax_coarse > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level,level_w);
         if (level_ow > -1)
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond,outer_wt_level,level_ow);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         HYPRE_BoomerAMGSetSchwarzUseNonSymm(pcg_precond, use_nonsymm_schwarz);
         HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) eu_level = 0;
         HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetNonGalerkTol(pcg_precond, nongalerk_num_tol, nongalerk_tol);
         HYPRE_GMRESSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                               pcg_precond);
      }
      else if(solver_id == 30)
      {
         /* use SysTG as preconditioner */
         if (myid == 0) hypre_printf("Solver: SysTG-GMRES\n");

         HYPRE_SysTGCreate(&pcg_precond);
     
///*         
         HYPRE_BoomerAMGCreate(&aux_solver); 
         HYPRE_BoomerAMGSetCGCIts(aux_solver, cgcits);
         HYPRE_BoomerAMGSetInterpType(aux_solver, interp_type);
         HYPRE_BoomerAMGSetPostInterpType(aux_solver, post_interp_type);
         HYPRE_BoomerAMGSetNumSamples(aux_solver, gsmg_samples);
         HYPRE_BoomerAMGSetCoarsenType(aux_solver, coarsen_type);
         HYPRE_BoomerAMGSetMeasureType(aux_solver, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(aux_solver, strong_threshold);
         HYPRE_BoomerAMGSetSeqThreshold(aux_solver, seq_threshold);
         HYPRE_BoomerAMGSetRedundant(aux_solver, redundant);
         HYPRE_BoomerAMGSetMaxCoarseSize(aux_solver, coarse_threshold);
         HYPRE_BoomerAMGSetMinCoarseSize(aux_solver, min_coarse_size);
         HYPRE_BoomerAMGSetTruncFactor(aux_solver, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(aux_solver, P_max_elmts);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(aux_solver, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetSCommPkgSwitch(aux_solver, S_commpkg_switch);
         HYPRE_BoomerAMGSetPrintFileName(aux_solver, "driver.out.log");
         HYPRE_BoomerAMGSetCycleType(aux_solver, cycle_type);
         HYPRE_BoomerAMGSetNumSweeps(aux_solver, num_sweeps);
         HYPRE_BoomerAMGSetISType(aux_solver, IS_type);
         HYPRE_BoomerAMGSetNumCRRelaxSteps(aux_solver, num_CR_relax_steps);
         HYPRE_BoomerAMGSetCRRate(aux_solver, CR_rate);
         HYPRE_BoomerAMGSetCRStrongTh(aux_solver, CR_strong_th);
         HYPRE_BoomerAMGSetCRUseCG(aux_solver, CR_use_CG);
         HYPRE_BoomerAMGSetRelaxType(aux_solver, relax_type);
         if (relax_down > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(aux_solver, relax_down, 1);
         if (relax_up > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(aux_solver, relax_up, 2);
         if (relax_coarse > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(aux_solver, relax_coarse, 3);
         HYPRE_BoomerAMGSetChebyOrder(aux_solver, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(aux_solver, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxWt(aux_solver, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(aux_solver, outer_wt);
         if (level_w > -1)
            HYPRE_BoomerAMGSetLevelRelaxWt(aux_solver, relax_wt_level,level_w);
         if (level_ow > -1)
            HYPRE_BoomerAMGSetLevelOuterWt(aux_solver,outer_wt_level,level_ow);
         HYPRE_BoomerAMGSetSmoothType(aux_solver, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(aux_solver, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(aux_solver, smooth_num_sweeps);
         HYPRE_BoomerAMGSetMaxLevels(aux_solver, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(aux_solver, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(aux_solver, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels(aux_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType(aux_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor(aux_solver, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor(aux_solver, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts(aux_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts(aux_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths(aux_solver, num_paths);
         HYPRE_BoomerAMGSetNodal(aux_solver, nodal);
         HYPRE_BoomerAMGSetNodalDiag(aux_solver, nodal_diag);
         HYPRE_BoomerAMGSetVariant(aux_solver, variant);
         HYPRE_BoomerAMGSetOverlap(aux_solver, overlap);
         HYPRE_BoomerAMGSetDomainType(aux_solver, domain_type);
         HYPRE_BoomerAMGSetSchwarzUseNonSymm(aux_solver, use_nonsymm_schwarz);
         HYPRE_BoomerAMGSetSchwarzRlxWeight(aux_solver, schwarz_rlx_weight);
         if (eu_level < 0) eu_level = 0;
         HYPRE_BoomerAMGSetEuLevel(aux_solver, eu_level);
         HYPRE_BoomerAMGSetEuBJ(aux_solver, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(aux_solver, eu_sparse_A);
         HYPRE_BoomerAMGSetCycleNumSweeps(aux_solver, ns_coarse, 3);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(aux_solver, dof_func);
         HYPRE_BoomerAMGSetAdditive(aux_solver, additive);
         HYPRE_BoomerAMGSetMultAdditive(aux_solver, mult_add);
         HYPRE_BoomerAMGSetSimple(aux_solver, simple);
         HYPRE_BoomerAMGSetAddPMaxElmts(aux_solver, add_P_max_elmts);
         HYPRE_BoomerAMGSetAddTruncFactor(aux_solver, add_trunc_factor);
         HYPRE_BoomerAMGSetNonGalerkTol(aux_solver, nongalerk_num_tol, nongalerk_tol);

		 HYPRE_BoomerAMGSetRelaxOrder(aux_solver, relax_order);
		 HYPRE_BoomerAMGSetTol(aux_solver, pc_tol);
		 HYPRE_BoomerAMGSetMaxIter(aux_solver, 1);
		 HYPRE_BoomerAMGSetPrintLevel(aux_solver, 0);
//*/         
	 HYPRE_Int coarse_index = 0;
	 HYPRE_SysTGSetBlockData( pcg_precond, blk_size, 1, &coarse_index);
	 HYPRE_SysTGSetNumWells(pcg_precond,num_wells);
	 HYPRE_SysTGSetCoarseSolver( pcg_precond, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup, aux_solver);
	 HYPRE_SysTGSetMaxCoarseLevels(pcg_precond, 1);
	 HYPRE_SysTGSetRelaxType(pcg_precond, 0);
	 HYPRE_SysTGSetNumRelaxSweeps(pcg_precond, 2);
	 HYPRE_SysTGSetNumInterpSweeps(pcg_precond, 2);    
	 HYPRE_SysTGSetMaxIters(pcg_precond, 1);
	 HYPRE_SysTGSetMaxGlobalsmoothIters(pcg_precond, 1);
	  HYPRE_SysTGSetGlobalsmoothType(pcg_precond, 0);
         HYPRE_GMRESSetMaxIter(pcg_solver, mg_max_iter);         
         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_SysTGSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_SysTGSetup,
                               pcg_precond);

		 parcsr_A = parcsr_B;

		 b = bb;
		 
      }      
      else if (solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) hypre_printf("Solver: DS-GMRES\n");
         pcg_precond = NULL;

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                               pcg_precond);
      }
      else if (solver_id == 7)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) hypre_printf("Solver: PILUT-GMRES\n");

         ierr = HYPRE_ParCSRPilutCreate( hypre_MPI_COMM_WORLD, &pcg_precond ); 
         if (ierr) {
            hypre_printf("Error in ParPilutCreate\n");
         }

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                               pcg_precond);

         if (drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               drop_tol );

         if (nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               nonzeros_to_keep );
      }
   
      else if (solver_id == 44)
      {
         /* use Euclid preconditioning */
         if (myid == 0) hypre_printf("Solver: Euclid-GMRES\n");

         HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &pcg_precond);

         if (eu_level > -1) HYPRE_EuclidSetLevel(pcg_precond, eu_level);
         if (eu_ilut) HYPRE_EuclidSetILUT(pcg_precond, eu_ilut);
         if (eu_sparse_A) HYPRE_EuclidSetSparseA(pcg_precond, eu_sparse_A);
         if (eu_row_scale) HYPRE_EuclidSetRowScale(pcg_precond, eu_row_scale);
         if (eu_bj) HYPRE_EuclidSetBJ(pcg_precond, eu_bj);
         HYPRE_EuclidSetStats(pcg_precond, eu_stats);
         HYPRE_EuclidSetMem(pcg_precond, eu_mem);
         /*HYPRE_EuclidSetParams(pcg_precond, argc, argv);*/

         HYPRE_GMRESSetPrecond (pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                pcg_precond);
      }
 
      HYPRE_GMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         hypre_printf("HYPRE_GMRESGetPrecond got bad precond\n");
         return(-1);
      }
      else
         if (myid == 0)
            hypre_printf("HYPRE_GMRESGetPrecond got good precond\n");
      HYPRE_GMRESSetup
         (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_GMRESSolve
         (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_GMRESGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_GMRESSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, 
                       (HYPRE_Vector)x);
      HYPRE_GMRESSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, 
                       (HYPRE_Vector)x);
#endif

      HYPRE_ParCSRGMRESDestroy(pcg_solver);
 
      if (solver_id == 3 || solver_id == 15)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      if(solver_id == 30)
      {
         HYPRE_BoomerAMGDestroy(aux_solver);
         HYPRE_SysTGDestroy(pcg_precond);
      }

      if (solver_id == 7)
      {
         HYPRE_ParCSRPilutDestroy(pcg_precond);
      }
      else if (solver_id == 18)
      {
	 HYPRE_ParaSailsDestroy(pcg_precond);
      }
      else if (solver_id == 44)
      {
         HYPRE_EuclidDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("GMRES Iterations = %d\n", num_iterations);
         hypre_printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using FlexGMRES 
    *-----------------------------------------------------------*/

   if (solver_id == 60 || solver_id == 61 || solver_id == 62 || solver_id == 63)
   {
      time_index = hypre_InitializeTiming("FlexGMRES Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRFlexGMRESCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_FlexGMRESSetKDim(pcg_solver, k_dim);
      HYPRE_FlexGMRESSetMaxIter(pcg_solver, max_iter);
      HYPRE_FlexGMRESSetTol(pcg_solver, tol);
      HYPRE_FlexGMRESSetAbsoluteTol(pcg_solver, atol);
      HYPRE_FlexGMRESSetLogging(pcg_solver, 1);
      HYPRE_FlexGMRESSetPrintLevel(pcg_solver, ioutdat);
 
      if (solver_id == 61)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) hypre_printf("Solver: AMG-FlexGMRES\n");

         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
         if (relax_down > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         if (relax_up > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         if (relax_coarse > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level,level_w);
         if (level_ow > -1)
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond,outer_wt_level,level_ow);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         HYPRE_BoomerAMGSetSchwarzUseNonSymm(pcg_precond, use_nonsymm_schwarz);
         HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) eu_level = 0;
         HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetNonGalerkTol(pcg_precond, nongalerk_num_tol, nongalerk_tol);
         HYPRE_FlexGMRESSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_FlexGMRESSetPrecond(pcg_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 60)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) hypre_printf("Solver: DS-FlexGMRES\n");
         pcg_precond = NULL;

         HYPRE_FlexGMRESSetPrecond(pcg_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                   (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
      }
	  else if (solver_id == 62)
	  {
		  if (myid == 0) hypre_printf("Solver: SysTG-flexGMRES\n");

		  HYPRE_SysTGCreate(&pcg_precond);
		  
		  /* HYPRE_ParCSRGMRESCreate(hypre_MPI_COMM_WORLD, &aux_solver); */
		  /* HYPRE_GMRESSetKDim(aux_solver, k_dim); */
		  /* HYPRE_GMRESSetMaxIter(aux_solver, 10); */
		  /* HYPRE_GMRESSetTol(aux_solver, 1e-1); */
		  /* HYPRE_GMRESSetAbsoluteTol(aux_solver, 1e-1); */
		  /* HYPRE_GMRESSetLogging(aux_solver, 1); */
		  /* HYPRE_GMRESSetPrintLevel(aux_solver, 2); */
		  /* HYPRE_GMRESSetRelChange(aux_solver, rel_change); */
		  //aux_precond = NULL;
		  //HYPRE_GMRESSetPrecond (aux_solver,
		  //						 (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
		  //						 (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
		  //						 aux_precond);
		  HYPRE_ParCSRFlexGMRESCreate(hypre_MPI_COMM_WORLD, &aux_solver);
		  HYPRE_FlexGMRESSetKDim(aux_solver, 10);
		  HYPRE_FlexGMRESSetMaxIter(aux_solver, 10);
		  HYPRE_FlexGMRESSetTol(aux_solver, 1e-1);
		  HYPRE_FlexGMRESSetAbsoluteTol(aux_solver, 1e-1);
		  HYPRE_FlexGMRESSetLogging(aux_solver, 0);
		  HYPRE_FlexGMRESSetPrintLevel(aux_solver, 3); 
 

		  //HYPRE_Int *cpt_coarse_index;
		  //cpt_coarse_index = hypre_CTAlloc(HYPRE_Int, 79*4);

		  //for (i = 0;i < 79*4;i++)
		  //	  cpt_coarse_index[i] = hypre_ParCSRMatrixGlobalNumRows(parcsr_B)-79*4+i;

		  //HYPRE_Int *cpt_coarse_index;
		  //cpt_coarse_index = hypre_CTAlloc(HYPRE_Int, 98);
		  //for (i = 0;i < 98;i++)
		  //	  cpt_coarse_index[i] = (hypre_ParCSRMatrixGlobalNumRows(parcsr_B)-98)/3+i;
		  
		  HYPRE_BoomerAMGCreate(&aux_precond);
		  HYPRE_BoomerAMGSetMaxIter(aux_precond, 1);
		  HYPRE_BoomerAMGSetRelaxOrder(aux_precond,0);
		  HYPRE_BoomerAMGSetSmoothNumSweeps(aux_precond, 1);
		  //HYPRE_BoomerAMGSetStrongThreshold(aux_precond, 1e-8);
		  //hypre_BoomerAMGSetCoarseningCpoint(aux_precond,10,4*79,cpt_coarse_index);
		  //hypre_BoomerAMGSetCoarseningCpoint(aux_precond,10,98,cpt_coarse_index);

		  
		  HYPRE_BoomerAMGSetTol(aux_precond, 0);

		  HYPRE_FlexGMRESSetPrecond(aux_solver,
		  							(HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
		  							(HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
		  							aux_precond);

		  /* ierr = HYPRE_EuclidCreate(comm, &aux_precond); */
		  /* HYPRE_EuclidSetLevel(aux_precond, 5); */
		  /* HYPRE_EuclidSetSparseA(aux_precond, 1.0e-5); */
		  /* HYPRE_EuclidSetBJ(pcg_precond, 1); */
		  /* HYPRE_FlexGMRESSetPrecond(aux_solver, */
		  /* 							(HYPRE_PtrToSolverFcn)HYPRE_EuclidSolve, */
		  /* 							(HYPRE_PtrToSolverFcn)HYPRE_EuclidSetup, */
		  /* 							aux_precond); */
		  
		  //HYPRE_BoomerAMGCreate(&aux_solver);
		  //HYPRE_BoomerAMGSetMaxIter(aux_solver, 1);
		  //HYPRE_BoomerAMGSetRelaxOrder(aux_solver, relax_order);
		  //HYPRE_BoomerAMGSetTol(aux_solver, 1e-1);
		  //HYPRE_BoomerAMGSetRelaxOrder(aux_solver, 0);
		  //HYPRE_BoomerAMGSetPrintLevel(aux_solver, 2);
		  //HYPRE_BoomerAMGSetStrongThreshold(aux_solver, 1e-10);
		  //HYPRE_BoomerAMGSetSmoothNumSweeps(aux_solver, 2);
		  
		  HYPRE_Int coarse_index = 0;//blk_size-1;
		  HYPRE_SysTGSetBlockData( pcg_precond, blk_size, 1, &coarse_index);
		  HYPRE_SysTGSetNumWells(pcg_precond,num_wells);
		  //HYPRE_SysTGSetCoarseSolver( pcg_precond, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup, aux_solver);
		  //HYPRE_SysTGSetCoarseSolver( pcg_precond, (HYPRE_PtrToParSolverFcn)HYPRE_GMRESSolve, (HYPRE_PtrToParSolverFcn)HYPRE_GMRESSetup, aux_solver); 
		  HYPRE_SysTGSetCoarseSolver( pcg_precond, (HYPRE_PtrToParSolverFcn)HYPRE_FlexGMRESSolve, (HYPRE_PtrToParSolverFcn)HYPRE_FlexGMRESSetup, aux_solver);
		  HYPRE_SysTGSetMaxCoarseLevels(pcg_precond, 1);
		  HYPRE_SysTGSetRelaxType(pcg_precond, 0);
		  HYPRE_SysTGSetNumRelaxSweeps(pcg_precond, 1);
		  HYPRE_SysTGSetNumInterpSweeps(pcg_precond, 1);
		  HYPRE_SysTGSetMaxIters(pcg_precond, 1);
		    
		  HYPRE_SysTGSetMaxGlobalsmoothIters(pcg_precond, 0);
		  HYPRE_SysTGSetGlobalsmoothType(pcg_precond, 0);
	 
		  HYPRE_FlexGMRESSetMaxIter(pcg_solver, mg_max_iter);
  
		  //hypre_block_jacobi_scaling(parcsr_B,&parcsr_C,pcg_precond,0);
		  parcsr_A = parcsr_B;
		  //parcsr_A = hypre_ParMatmul(parcsr_C,parcsr_B);
		  //parcsr_A = hypre_ParMatmul(parcsr_B,parcsr_C);

		  
		  //HYPRE_ParCSRMatrixMatvec(1.0,parcsr_C,bb,0.0,b);
		  b = bb; 
		   
		  
		  HYPRE_FlexGMRESSetPrecond(pcg_solver,
									(HYPRE_PtrToSolverFcn) HYPRE_SysTGSolve,
									(HYPRE_PtrToSolverFcn) HYPRE_SysTGSetup,
									pcg_precond);
	  }
	  else if (solver_id == 63)
	  {
		  ierr = HYPRE_EuclidCreate(comm, &pcg_precond);
//              	    HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &m_euclid);
		  HYPRE_EuclidSetLevel(pcg_precond, 5);
		  //HYPRE_EuclidSetILUT(pcg_precond, 1.0e-2);
		  //HYPRE_EuclidSetSparseA(pcg_precond, 1.0e-10);
//              	    HYPRE_EuclidSetRowScale(m_euclid,1);
		  HYPRE_EuclidSetBJ(pcg_precond, 1);
//              	    HYPRE_EuclidSetStats(m_euclid, 0);
//              	    HYPRE_EuclidSetMem(m_euclid, 0);

		  //hypre_block_jacobi_scaling(parcsr_B,&parcsr_C,pcg_precond,0);
		  //parcsr_A = hypre_ParMatmul(parcsr_C,parcsr_B);
		  //parcsr_A = hypre_ParMatmul(parcsr_B,parcsr_C);

		  parcsr_A = parcsr_B;
		  {
			  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(parcsr_A);
			  HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
			  HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
			  HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);
			  HYPRE_Int              n = hypre_CSRMatrixNumRows(A_diag);
			  HYPRE_Int i,j,k;
			  for (i = 0;i < n; i++)
			  {
				  for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
				  {
					  k = A_diag_j[j];
					  if (k == i && fabs(A_diag_data[j]) < 1e-5)
					  {
						  A_diag_data[j] = 1e-3;
						  //printf("(%d,%d) = %e\n",i,k,A_diag_data[j]);
					  }
					  if (k == i && k == 723)
						  printf("(%d,%d) = %e\n",i,k,A_diag_data[j]);
				  }
			  }
			  hypre_CSRMatrixData(A_diag) = A_diag_data;
			  hypre_CSRMatrixI(A_diag) = A_diag_i;
			  hypre_CSRMatrixJ(A_diag) = A_diag_j;
		  }
		  
		  

		  //HYPRE_ParCSRMatrixMatvec(1.0,parcsr_C,bb,0.0,b);
		  b = bb;

		  
		  
		  HYPRE_FlexGMRESSetPrecond(pcg_solver,
									(HYPRE_PtrToSolverFcn)HYPRE_EuclidSolve,
									(HYPRE_PtrToSolverFcn)HYPRE_EuclidSetup,
									pcg_precond);
	  }
	  

      HYPRE_FlexGMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         hypre_printf("HYPRE_FlexGMRESGetPrecond got bad precond\n");
         return(-1);
      }
      else
         if (myid == 0)
            hypre_printf("HYPRE_FlexGMRESGetPrecond got good precond\n");


      /* this is optional - could be a user defined one instead (see ex5.c)*/
      HYPRE_FlexGMRESSetModifyPC( pcg_solver, 
                                  (HYPRE_PtrToModifyPCFcn) hypre_FlexGMRESModifyPCDefault);

	  printf("before setup\n");
	  
      HYPRE_FlexGMRESSetup
         (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

	  printf("end setup\n");
	  
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

      HYPRE_ParCSRFlexGMRESDestroy(pcg_solver);
 
      if (solver_id == 61)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("FlexGMRES Iterations = %d\n", num_iterations);
         hypre_printf("Final FlexGMRES Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGSTAB 
    *-----------------------------------------------------------*/

   if (solver_id == 9 || solver_id == 10 || solver_id == 11 || solver_id == 45)
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRBiCGSTABCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_BiCGSTABSetMaxIter(pcg_solver, max_iter);
      HYPRE_BiCGSTABSetTol(pcg_solver, tol);
      HYPRE_BiCGSTABSetAbsoluteTol(pcg_solver, atol);
      HYPRE_BiCGSTABSetLogging(pcg_solver, ioutdat);
      HYPRE_BiCGSTABSetPrintLevel(pcg_solver, ioutdat);
 
      if (solver_id == 9)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) hypre_printf("Solver: AMG-BiCGSTAB\n");
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
         if (relax_down > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         if (relax_up > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         if (relax_coarse > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level,level_w);
         if (level_ow > -1)
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond,outer_wt_level,level_ow);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         HYPRE_BoomerAMGSetSchwarzUseNonSymm(pcg_precond, use_nonsymm_schwarz);
       
         HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) eu_level = 0;
         HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetNonGalerkTol(pcg_precond, nongalerk_num_tol, nongalerk_tol);
         HYPRE_BiCGSTABSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                  pcg_precond);
      }
      else if (solver_id == 10)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) hypre_printf("Solver: DS-BiCGSTAB\n");
         pcg_precond = NULL;

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                  pcg_precond);
      }
      else if (solver_id == 11)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) hypre_printf("Solver: PILUT-BiCGSTAB\n");

         ierr = HYPRE_ParCSRPilutCreate( hypre_MPI_COMM_WORLD, &pcg_precond ); 
         if (ierr) {
            hypre_printf("Error in ParPilutCreate\n");
         }

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                                  pcg_precond);

         if (drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               drop_tol );

         if (nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               nonzeros_to_keep );
      }
      else if (solver_id == 45)
      {
         /* use Euclid preconditioning */
         if (myid == 0) hypre_printf("Solver: Euclid-BICGSTAB\n");

         HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         if (eu_level > -1) HYPRE_EuclidSetLevel(pcg_precond, eu_level);
         if (eu_ilut) HYPRE_EuclidSetILUT(pcg_precond, eu_ilut);
         if (eu_sparse_A) HYPRE_EuclidSetSparseA(pcg_precond, eu_sparse_A);
         if (eu_row_scale) HYPRE_EuclidSetRowScale(pcg_precond, eu_row_scale);
         if (eu_bj) HYPRE_EuclidSetBJ(pcg_precond, eu_bj);
         HYPRE_EuclidSetStats(pcg_precond, eu_stats);
         HYPRE_EuclidSetMem(pcg_precond, eu_mem);

         /*HYPRE_EuclidSetParams(pcg_precond, argc, argv);*/

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                  pcg_precond);
      }
 
      HYPRE_BiCGSTABSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, 
                          (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_BiCGSTABSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, 
                          (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_BiCGSTABGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_BiCGSTABSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, 
                          (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_BiCGSTABSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, 
                          (HYPRE_Vector)b, (HYPRE_Vector)x);
#endif

      HYPRE_ParCSRBiCGSTABDestroy(pcg_solver);
 
      if (solver_id == 9)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (solver_id == 11)
      {
         HYPRE_ParCSRPilutDestroy(pcg_precond);
      }
      else if (solver_id == 45)
      {
         HYPRE_EuclidDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("BiCGSTAB Iterations = %d\n", num_iterations);
         hypre_printf("Final BiCGSTAB Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using CGNR 
    *-----------------------------------------------------------*/

   if (solver_id == 5 || solver_id == 6)
   {
      time_index = hypre_InitializeTiming("CGNR Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRCGNRCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_CGNRSetMaxIter(pcg_solver, max_iter);
      HYPRE_CGNRSetTol(pcg_solver, tol);
      HYPRE_CGNRSetLogging(pcg_solver, ioutdat);
 
      if (solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) hypre_printf("Solver: AMG-CGNR\n");
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
         if (relax_down > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         if (relax_up > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         if (relax_coarse > -1)
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level,level_w);
         if (level_ow > -1)
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond,outer_wt_level,level_ow);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetNonGalerkTol(pcg_precond, nongalerk_num_tol, nongalerk_tol);
         HYPRE_CGNRSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_CGNRSetPrecond(pcg_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolveT,
                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                              pcg_precond);
      }
      else if (solver_id == 6)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) hypre_printf("Solver: DS-CGNR\n");
         pcg_precond = NULL;

         HYPRE_CGNRSetPrecond(pcg_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                              pcg_precond);
      }
 
      HYPRE_CGNRGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         hypre_printf("HYPRE_ParCSRCGNRGetPrecond got bad precond\n");
         return(-1);
      }
      else
         if (myid == 0)
            hypre_printf("HYPRE_ParCSRCGNRGetPrecond got good precond\n");
      HYPRE_CGNRSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, 
                      (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("CGNR Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_CGNRSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, 
                      (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_CGNRGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_CGNRGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_CGNRSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, 
                      (HYPRE_Vector)x);
      HYPRE_CGNRSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, 
                      (HYPRE_Vector)x);
#endif

      HYPRE_ParCSRCGNRDestroy(pcg_solver);
 
      if (solver_id == 5)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   /* RDF: Why is this here? */
   if (!(build_rhs_type ==1 || build_rhs_type ==7))
      HYPRE_IJVectorGetObjectType(ij_b, &j);

   if (print_system)
   {
      HYPRE_IJVectorPrint(ij_x, "IJ.out.x");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   if (test_ij || build_matrix_type == -1) HYPRE_IJMatrixDestroy(ij_A);
   else HYPRE_ParCSRMatrixDestroy(parcsr_A);

   /* for build_rhs_type = 1 or 7, we did not create ij_b  - just b*/
   if (build_rhs_type ==1 || build_rhs_type ==7)
      HYPRE_ParVectorDestroy(b);
   else
      HYPRE_IJVectorDestroy(ij_b);

   HYPRE_IJVectorDestroy(ij_x);

/*
  hypre_FinalizeMemoryDebug();
*/

   hypre_MPI_Finalize();

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
 
   HYPRE_ParCSRMatrixRead(hypre_MPI_COMM_WORLD, filename,&A);

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
BuildParRhsFromFile( HYPRE_Int                  argc,
                     char                *argv[],
                     HYPRE_Int                  arg_index,
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
      hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  RhsFromParFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   HYPRE_ParVectorRead(hypre_MPI_COMM_WORLD, filename,&b);

   *b_ptr = b;

   return (0);
}




/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian( HYPRE_Int                  argc,
                   char                *argv[],
                   HYPRE_Int                  arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   HYPRE_Real          cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Int                 num_fun = 1;
   HYPRE_Real         *values;
   HYPRE_Real         *mtrx;

   HYPRE_Real          ep = .1;
   
   HYPRE_Int                 system_vcoef = 0;
   HYPRE_Int                 sys_opt = 0;
   HYPRE_Int                 vcoef_opt = 0;
   
   
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
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
         ep = atof(argv[arg_index++]); 
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
 
   if (myid == 0)
   {
      hypre_printf("  Laplacian:   num_fun = %d\n", num_fun);
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(HYPRE_Real, 4);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   if (num_fun == 1)
      A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD, 
                                                 nx, ny, nz, P, Q, R, p, q, r, values);
   else
   {
      mtrx = hypre_CTAlloc(HYPRE_Real, num_fun*num_fun);

      if (num_fun == 2)
      {
         if (sys_opt ==1) /* identity  */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0; 
            mtrx[2] = 0.0;
            mtrx[3] = 1.0; 
         }
         else if (sys_opt ==2)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0; 
            mtrx[2] = 0.0;
            mtrx[3] = 20.0; 
         }
         else if (sys_opt ==3) /* similar to barry's talk - ex1 */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 2.0; 
            mtrx[2] = 2.0;
            mtrx[3] = 1.0; 
         }
         else if (sys_opt ==4) /* can use with vcoef to get barry's ex*/
         {
            mtrx[0] = 1.0;
            mtrx[1] = 1.0; 
            mtrx[2] = 1.0;
            mtrx[3] = 1.0; 
         }
         else if (sys_opt ==5) /* barry's talk - ex1 */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 1.1; 
            mtrx[2] = 1.1;
            mtrx[3] = 1.0; 
         }
         else if (sys_opt ==6) /*  */
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
         if (sys_opt ==1)
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
         else if (sys_opt ==2)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 0.0;
            mtrx[4] = 20.0;
            mtrx[5] = 0.0;
            mtrx[6] = 0.0;
            mtrx[7] = 0.0;
            mtrx[8] =.01;
         }
         else if (sys_opt ==3)
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
         else if (sys_opt ==4) /* barry ex4 */
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
         A = (HYPRE_ParCSRMatrix) GenerateSysLaplacian(hypre_MPI_COMM_WORLD, 
                                                       nx, ny, nz, P, Q, 
                                                       R, p, q, r, num_fun, mtrx, values);
      }
      else
      {
       
    
         HYPRE_Real *mtrx_values;

         mtrx_values = hypre_CTAlloc(HYPRE_Real, num_fun*num_fun*4);

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
               SetSysVcoefValues(num_fun, nx, ny, nz, ep*1.0, 1.0, 1.0, 0, mtrx_values);
               
               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 1, mtrx_values);
               
               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep*1.0, 1.0, 1.0, 2, mtrx_values);
               
               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 4) /* use with default sys_opt  - ulrike ex 4*/
            {
               HYPRE_Real ep2 = ep;
               
               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep*1.0, 1.0, 1.0, 0, mtrx_values);
               
               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, ep*1.0, 1.0, 1, mtrx_values);
               
               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep*1.0, 1.0, 1.0, 2, mtrx_values);
               
               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, ep2*1.0, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 5) /* use with default sys_opt  - */
            {
               HYPRE_Real  alp, beta;
               alp = .001;
               beta = 10;
               
               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, alp*1.0, 1.0, 1.0, 0, mtrx_values);
               
               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, beta*1.0, 1.0, 1, mtrx_values);
               
               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, alp*1.0, 1.0, 1.0, 2, mtrx_values);
               
               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, beta*1.0, 1.0, 3, mtrx_values);
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

         A = (HYPRE_ParCSRMatrix) GenerateSysLaplacianVCoef(hypre_MPI_COMM_WORLD, 
                                                            nx, ny, nz, P, Q, 
                                                            R, p, q, r, num_fun, mtrx, mtrx_values);





         hypre_TFree(mtrx_values);
      }

      hypre_TFree(mtrx);
   }

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
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
BuildParDifConv( HYPRE_Int                  argc,
                 char                *argv[],
                 HYPRE_Int                  arg_index,
                 HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   HYPRE_Real          cx, cy, cz;
   HYPRE_Real          ax, ay, az;
   HYPRE_Real          hinx,hiny,hinz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = atof(argv[arg_index++]);
         ay = atof(argv[arg_index++]);
         az = atof(argv[arg_index++]);
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
 
   if (myid == 0)
   {
      hypre_printf("  Convection-Diffusion: \n");
      hypre_printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");  
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   hinx = 1./(nx+1);
   hiny = 1./(ny+1);
   hinz = 1./(nz+1);

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(HYPRE_Real, 7);

   values[1] = -cx/(hinx*hinx);
   values[2] = -cy/(hiny*hiny);
   values[3] = -cz/(hinz*hinz);
   values[4] = -cx/(hinx*hinx) + ax/hinx;
   values[5] = -cy/(hiny*hiny) + ay/hiny;
   values[6] = -cz/(hinz*hinz) + az/hinz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx/(hinx*hinx) - 1.*ax/hinx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy/(hiny*hiny) - 1.*ay/hiny;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz/(hinz*hinz) - 1.*az/hinz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateDifConv(hypre_MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

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
BuildParFromOneFile( HYPRE_Int                  argc,
                     char                *argv[],
                     HYPRE_Int                  arg_index,
                     HYPRE_Int                  num_functions,
                     HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix  A;
   HYPRE_CSRMatrix  A_CSR = NULL;

   HYPRE_Int                 myid, numprocs;
   HYPRE_Int                 i, rest, size, num_nodes, num_dofs;
   HYPRE_Int		      *row_part;
   HYPRE_Int		      *col_part;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &numprocs );

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

   row_part = NULL;
   col_part = NULL;
   if (myid == 0 && num_functions > 1)
   {
      HYPRE_CSRMatrixGetNumRows(A_CSR, &num_dofs);
      num_nodes = num_dofs/num_functions;
      if (num_dofs != num_functions*num_nodes)
      {
	 row_part = NULL;
	 col_part = NULL;
      }
      else
      {
         row_part = hypre_CTAlloc(HYPRE_Int, numprocs+1);
	 row_part[0] = 0;
	 size = num_nodes/numprocs;
	 rest = num_nodes-size*numprocs;
	 for (i=0; i < numprocs; i++)
	 {
	    row_part[i+1] = row_part[i]+size*num_functions;
	    if (i < rest) row_part[i+1] += num_functions;
         }
         col_part = row_part;
      }
   }

   HYPRE_CSRMatrixToParCSRMatrix(hypre_MPI_COMM_WORLD, A_CSR, row_part, col_part, &A);

   *A_ptr = A;

   if (myid == 0) HYPRE_CSRMatrixDestroy(A_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildFuncsFromFiles(    HYPRE_Int                  argc,
                        char                *argv[],
                        HYPRE_Int                  arg_index,
                        HYPRE_ParCSRMatrix   parcsr_A,
                        HYPRE_Int                **dof_func_ptr     )
{
/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

   hypre_printf (" Feature is not implemented yet!\n");	
   return(0);

}


HYPRE_Int
BuildFuncsFromOneFile(  HYPRE_Int                  argc,
                        char                *argv[],
                        HYPRE_Int                  arg_index,
                        HYPRE_ParCSRMatrix   parcsr_A,
                        HYPRE_Int                **dof_func_ptr     )
{
   char           *filename;

   HYPRE_Int             myid, num_procs;
   HYPRE_Int            *partitioning;
   HYPRE_Int            *dof_func;
   HYPRE_Int            *dof_func_local;
   HYPRE_Int             i, j;
   HYPRE_Int             local_size, global_size;
   hypre_MPI_Request	  *requests;
   hypre_MPI_Status	  *status, status0;
   MPI_Comm	   comm;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   comm = hypre_MPI_COMM_WORLD;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

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
      dof_func = hypre_CTAlloc(HYPRE_Int, global_size);

      for (j = 0; j < global_size; j++)
      {
         hypre_fscanf(fp, "%d", &dof_func[j]);
      }

      fclose(fp);
 
   }
   HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A, &partitioning);
   local_size = partitioning[myid+1]-partitioning[myid];
   dof_func_local = hypre_CTAlloc(HYPRE_Int,local_size);

   if (myid == 0)
   {
      requests = hypre_CTAlloc(hypre_MPI_Request,num_procs-1);
      status = hypre_CTAlloc(hypre_MPI_Status,num_procs-1);
      j = 0;
      for (i=1; i < num_procs; i++)
         hypre_MPI_Isend(&dof_func[partitioning[i]],
                         partitioning[i+1]-partitioning[i],
                         HYPRE_MPI_INT, i, 0, comm, &requests[j++]);
      for (i=0; i < local_size; i++)
         dof_func_local[i] = dof_func[i];
      hypre_MPI_Waitall(num_procs-1,requests, status);
      hypre_TFree(requests);
      hypre_TFree(status);
   }
   else
   {
      hypre_MPI_Recv(dof_func_local,local_size,HYPRE_MPI_INT,0,0,comm,&status0);
   }

   *dof_func_ptr = dof_func_local;

   if (myid == 0) hypre_TFree(dof_func);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors 
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildRhsParFromOneFile( HYPRE_Int                  argc,
                        char                *argv[],
                        HYPRE_Int                  arg_index,
                        HYPRE_Int                 *partitioning,
                        HYPRE_ParVector     *b_ptr     )
{
   char           *filename;

   HYPRE_ParVector b;
   HYPRE_Vector    b_CSR=NULL;

   HYPRE_Int             myid;

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
      hypre_printf("  Rhs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix 
       *-----------------------------------------------------------*/
 
      b_CSR = HYPRE_VectorRead(filename);
   }
   HYPRE_VectorToParVector(hypre_MPI_COMM_WORLD, b_CSR, partitioning,&b); 

   *b_ptr = b;

   HYPRE_VectorDestroy(b_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian9pt( HYPRE_Int                  argc,
                      char                *argv[],
                      HYPRE_Int                  arg_index,
                      HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny;
   HYPRE_Int                 P, Q;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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

   if ((P*Q) != num_procs)
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
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p)/P;

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(HYPRE_Real, 2);

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

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(hypre_MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}
/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian27pt( HYPRE_Int                  argc,
                       char                *argv[],
                       HYPRE_Int                  arg_index,
                       HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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

   if ((P*Q*R) != num_procs)
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
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(HYPRE_Real, 2);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
      values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
      values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build 7-point in 2D 
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParRotate7pt( HYPRE_Int                  argc,
                   char                *argv[],
                   HYPRE_Int                  arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny;
   HYPRE_Int                 P, Q;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q;
   HYPRE_Real          eps, alpha;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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
         alpha  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q) != num_procs)
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
      hypre_printf("    alpha = %f, eps = %f\n", alpha,eps);
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p)/P;

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/

   A = (HYPRE_ParCSRMatrix) GenerateRotate7pt(hypre_MPI_COMM_WORLD,
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
 *
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParVarDifConv( HYPRE_Int                  argc,
                    char                *argv[],
                    HYPRE_Int                  arg_index,
                    HYPRE_ParCSRMatrix  *A_ptr    ,
                    HYPRE_ParVector  *rhs_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;
   HYPRE_ParVector  rhs;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real          eps;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = atof(argv[arg_index++]);
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

   if (myid == 0)
   {
      hypre_printf("  ell PDE: eps = %f\n", eps);
      hypre_printf("    Dx(aDxu) + Dy(bDyu) + Dz(cDzu) + d Dxu + e Dyu + f Dzu  + g u= f\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
   }
   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   A = (HYPRE_ParCSRMatrix) GenerateVarDifConv(hypre_MPI_COMM_WORLD,
                                               nx, ny, nz, P, Q, R, p, q, r, eps, &rhs);

   *A_ptr = A;
   *rhs_ptr = rhs;

   return (0);
}

/**************************************************************************/


HYPRE_Int SetSysVcoefValues(HYPRE_Int num_fun, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz, HYPRE_Real vcx, 
                            HYPRE_Real vcy, HYPRE_Real vcz, HYPRE_Int mtx_entry, HYPRE_Real *values)
{


   HYPRE_Int sz = num_fun*num_fun;

   values[1*sz + mtx_entry] = -vcx;
   values[2*sz + mtx_entry] = -vcy;
   values[3*sz + mtx_entry] = -vcz;
   values[0*sz + mtx_entry] = 0.0;

   if (nx > 1)
   {
      values[0*sz + mtx_entry] += 2.0*vcx;
   }
   if (ny > 1)
   {
      values[0*sz + mtx_entry] += 2.0*vcy;
   }
   if (nz > 1)
   {
      values[0*sz + mtx_entry] += 2.0*vcz;
   }

   return 0;
   
}
                                                                                
/*----------------------------------------------------------------------
 * Build coordinates for 1D/2D/3D
 *----------------------------------------------------------------------*/
                                                                                
HYPRE_Int
BuildParCoordinates( HYPRE_Int                  argc,
                     char                *argv[],
                     HYPRE_Int                  arg_index,
                     HYPRE_Int                 *coorddim_ptr,
                     float               **coord_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
                                                                                
   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
                                                                                
   HYPRE_Int                 coorddim;
   float               *coordinates;
                                                                                
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
                                                                                
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
                                                                                
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
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );
                                                                                
   /*-----------------------------------------------------------
    * Generate the coordinates
    *-----------------------------------------------------------*/
                                                                                
   coorddim = 3;
   if (nx<2) coorddim--;
   if (ny<2) coorddim--;
   if (nz<2) coorddim--;
                                                                                
   if (coorddim>0)
      coordinates = GenerateCoordinates (hypre_MPI_COMM_WORLD,
                                         nx, ny, nz, P, Q, R, p, q, r, coorddim);
   else
      coordinates=NULL;
                                                                                
   *coorddim_ptr = coorddim;
   *coord_ptr = coordinates;
   return (0);
}

