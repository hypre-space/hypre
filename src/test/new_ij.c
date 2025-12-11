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
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int BuildParFromFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                            HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParRhsFromFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                               HYPRE_ParVector *b_ptr );

HYPRE_Int BuildParLaplacian (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                             HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParSysLaplacian (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParDifConv (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                           HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParFromOneFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                               HYPRE_Int num_functions, HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildFuncsFromFiles (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                               HYPRE_ParCSRMatrix A, HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildFuncsFromOneFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                 HYPRE_ParCSRMatrix A, HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildRhsParFromOneFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                  HYPRE_Int *partitioning, HYPRE_ParVector *b_ptr );
HYPRE_Int BuildParLaplacian9pt (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian27pt (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                 HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParRotate7pt (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                             HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParVarDifConv (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                              HYPRE_ParCSRMatrix *A_ptr, HYPRE_ParVector *rhs_ptr );
HYPRE_ParCSRMatrix GenerateSysLaplacian (MPI_Comm comm, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                                         HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                         HYPRE_Int num_fun, HYPRE_Real *mtrx, HYPRE_Real *value);
HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef (MPI_Comm comm, HYPRE_Int nx, HYPRE_Int ny,
                                              HYPRE_Int nz,
                                              HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                              HYPRE_Int num_fun, HYPRE_Real *mtrx, HYPRE_Real *value);
HYPRE_Int SetSysVcoefValues(HYPRE_Int num_fun, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                            HYPRE_Real vcx, HYPRE_Real vcy, HYPRE_Real vcz, HYPRE_Int mtx_entry, HYPRE_Real *values);

HYPRE_Int BuildParCoordinates (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                               HYPRE_Int *coorddim_ptr, float **coord_ptr );

extern HYPRE_Int hypre_FlexGMRESModifyPCAMGExample(void *precond_data, HYPRE_Int iterations,
                                                   HYPRE_Real rel_residual_norm);

extern HYPRE_Int hypre_FlexGMRESModifyPCDefault(void *precond_data, HYPRE_Int iteration,
                                                HYPRE_Real rel_residual_norm);

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
   HYPRE_Int                 i, j;
   HYPRE_Int                 max_levels = 25;
   HYPRE_Int                 num_iterations;
   HYPRE_Int                 pcg_num_its, dscg_num_its;
   HYPRE_Int                 max_iter = 1000;
   HYPRE_Int                 mg_max_iter = 100;
   HYPRE_Int                 nodal = 0;
   HYPRE_Int                 nodal_diag = 0;
   HYPRE_Real          cf_tol = 0.9;
   HYPRE_Real          norm;
   HYPRE_Real          final_res_norm;
   void               *object;

   HYPRE_IJMatrix      ij_A;
   HYPRE_IJVector      ij_b;
   HYPRE_IJVector      ij_x;
   HYPRE_IJVector      *ij_rbm;

   HYPRE_ParCSRMatrix  parcsr_A;
   HYPRE_ParVector     b;
   HYPRE_ParVector     x;
   HYPRE_ParVector     *interp_vecs = NULL;

   HYPRE_Solver        amg_solver;
   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond = NULL, pcg_precond_gotten;

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
   HYPRE_Int             num_functions = 1;
   HYPRE_Int             num_paths = 1;
   HYPRE_Int             agg_num_levels = 0;
   HYPRE_Int             ns_coarse = 1;

   HYPRE_Int             time_index;
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
   HYPRE_Int num_interp_vecs = 0;
   HYPRE_Int interp_vec_variant = 0;
   HYPRE_Int Q_max = 0;
   HYPRE_Real Q_trunc = 0;

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
   /* HYPRE_Int       smooth_lev; */
   /* HYPRE_Int       smooth_rlx = 8; */
   HYPRE_Int       smooth_type = 6;
   HYPRE_Int       smooth_num_levels = 0;
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

   HYPRE_Int    rap2 = 0;
   HYPRE_Int    keepTranspose = 0;
   HYPRE_Real   relax_wt;
   HYPRE_Real   relax_wt_level;
   HYPRE_Real   outer_wt;
   HYPRE_Real   outer_wt_level;
   HYPRE_Real   tol = 1.e-8, pc_tol = 0.;
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
   HYPRE_Int       eu_bj = 0;
   HYPRE_Int       eu_level = -1;
   HYPRE_Int       eu_stats = 0;
   HYPRE_Int       eu_mem = 0;
   HYPRE_Int       eu_row_scale = 0; /* Euclid only */

   /* parameters for GMRES */
   HYPRE_Int       k_dim;
   /* parameters for LGMRES */
   HYPRE_Int       aug_dim;
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

   hypre_sprintf (plot_file_name, "AMGgrids.CF.dat");

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
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 5;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-vardifconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 6;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rotate") == 0 )
      {
         arg_index++;
         build_matrix_type      = 7;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-test_ij") == 0 )
      {
         arg_index++;
         test_ij = 1;
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
      else if ( strcmp(argv[arg_index], "-rbm") == 0 )
      {
         arg_index++;
         build_rbm      = 1;
         num_interp_vecs = atoi(argv[arg_index++]);
         build_rbm_index = arg_index;
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
         CR_rate = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-crst") == 0 )
      {
         arg_index++;
         CR_strong_th = (HYPRE_Real)atof(argv[arg_index++]);
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
         dt = (HYPRE_Real)atof(argv[arg_index++]);
         build_rhs_type = -1;
         if ( build_src_type == -1 ) { build_src_type = 2; }
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

   if (solver_id == 8 || solver_id == 18)
   {
      max_levels = 1;
   }

   /* defaults for BoomerAMG */
   if (solver_id == 0 || solver_id == 1 || solver_id == 3 || solver_id == 5
       || solver_id == 9 || solver_id == 13 || solver_id == 14
       || solver_id == 15 || solver_id == 20 || solver_id == 51 || solver_id == 61)
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
         coarse_threshold  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-min_cs") == 0 )
      {
         arg_index++;
         min_coarse_size  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seq_th") == 0 )
      {
         arg_index++;
         seq_threshold  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-red") == 0 )
      {
         arg_index++;
         redundant  = (HYPRE_Real)atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol  = (HYPRE_Real)atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-sai_th") == 0 )
      {
         arg_index++;
         sai_threshold  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_filt") == 0 )
      {
         arg_index++;
         sai_filter  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-drop_tol") == 0 )
      {
         arg_index++;
         drop_tol  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nonzeros_to_keep") == 0 )
      {
         arg_index++;
         nonzeros_to_keep  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ilut") == 0 )
      {
         arg_index++;
         eu_ilut  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sparseA") == 0 )
      {
         arg_index++;
         eu_sparse_A  = (HYPRE_Real)atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-cheby_order") == 0 )
      {
         arg_index++;
         cheby_order = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-rap") == 0 )
      {
         arg_index++;
         rap2  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-keepT") == 0 )
      {
         arg_index++;
         keepTranspose  = atoi(argv[arg_index++]);
      }
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
      hypre_printf("  -fromfile <filename>       : ");
      hypre_printf("matrix read from multiple files (IJ format)\n");
      hypre_printf("  -fromparcsrfile <filename> : ");
      hypre_printf("matrix read from multiple files (ParCSR format)\n");
      hypre_printf("  -fromonecsrfile <filename> : ");
      hypre_printf("matrix read from a single file (CSR format)\n");
      hypre_printf("\n");
      hypre_printf("  -laplacian [<options>] : build 5pt 2D laplacian problem (default) \n");
      hypre_printf("  -sysL <num functions>  : build SYSTEMS laplacian 7pt operator\n");
      hypre_printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
      hypre_printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
      hypre_printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
      hypre_printf("    -n <nx> <ny> <nz>    : total problem size \n");
      hypre_printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      hypre_printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      hypre_printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      hypre_printf("\n");
      hypre_printf("  -exact_size            : inserts immediately into ParCSR structure\n");
      hypre_printf("  -storage_low           : allocates not enough storage for aux struct\n");
      hypre_printf("  -concrete_parcsr       : use parcsr matrix type as concrete type\n");
      hypre_printf("\n");
      hypre_printf("  -rhsfromfile           : ");
      hypre_printf("rhs read from multiple files (IJ format)\n");
      hypre_printf("  -rhsfromonefile        : ");
      hypre_printf("rhs read from a single file (CSR format)\n");
      hypre_printf("  -rhsparcsrfile        :  ");
      hypre_printf("rhs read from multiple files (ParCSR format)\n");
      hypre_printf("  -rhsrand               : rhs is random vector\n");
      hypre_printf("  -rhsisone              : rhs is vector with unit components (default)\n");
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
      hypre_printf("backward Euler source is random vector with components in range 0 - 1\n");
      hypre_printf("  -srcisone              : ");
      hypre_printf("backward Euler source is vector with unit components (default)\n");
      hypre_printf("  -srczero               : ");
      hypre_printf("backward Euler source is zero-vector\n");
      hypre_printf("\n");
      hypre_printf("  -solver <ID>           : solver ID\n");
      hypre_printf("       0=AMG               1=AMG-PCG        \n");
      hypre_printf("       2=DS-PCG            3=AMG-GMRES      \n");
      hypre_printf("       4=DS-GMRES          5=AMG-CGNR       \n");
      hypre_printf("       6=DS-CGNR           7=PILUT-GMRES    \n");
      hypre_printf("       8=ParaSails-PCG     9=AMG-BiCGSTAB   \n");
      hypre_printf("       10=DS-BiCGSTAB     11=PILUT-BiCGSTAB \n");
      hypre_printf("       12=Schwarz-PCG     13=GSMG           \n");
      hypre_printf("       14=GSMG-PCG        15=GSMG-GMRES\n");
      hypre_printf("       18=ParaSails-GMRES\n");
      hypre_printf("       20=Hybrid solver/ DiagScale, AMG \n");
      hypre_printf("       43=Euclid-PCG      44=Euclid-GMRES   \n");
      hypre_printf("       45=Euclid-BICGSTAB\n");
      hypre_printf("       50=DS-LGMRES         51=AMG-LGMRES     \n");
      hypre_printf("       60=DS-FlexGMRES         61=AMG-FlexGMRES     \n");
      hypre_printf("\n");
      hypre_printf("  -cljp                 : CLJP coarsening \n");
      hypre_printf("  -cljp1                : CLJP coarsening, fixed random \n");
      hypre_printf("  -cgc                  : CGC coarsening \n");
      hypre_printf("  -cgce                 : CGC-E coarsening \n");
      hypre_printf("  -pmis                 : PMIS coarsening \n");
      hypre_printf("  -pmis1                : PMIS coarsening, fixed random \n");
      hypre_printf("  -hmis                 : HMIS coarsening \n");
      hypre_printf("  -ruge                 : Ruge-Stueben coarsening (local)\n");
      hypre_printf("  -ruge1p               : Ruge-Stueben coarsening 1st pass only(local)\n");
      hypre_printf("  -ruge3                : third pass on boundary\n");
      hypre_printf("  -ruge3c               : third pass on boundary, keep c-points\n");
      hypre_printf("  -falgout              : local Ruge_Stueben followed by CLJP\n");
      hypre_printf("  -gm                   : use global measures\n");
      hypre_printf("\n");
      hypre_printf("  -interptype  <val>    : set interpolation type\n");
      hypre_printf("       0=Classical modified interpolation (default)  \n");
      hypre_printf("       1=least squares interpolation (for GSMG only)  \n");
      hypre_printf("       0=Classical modified interpolation for hyperbolic PDEs \n");
      hypre_printf("       3=direct interpolation with separation of weights  \n");
      hypre_printf("       4=multipass interpolation  \n");
      hypre_printf("       5=multipass interpolation with separation of weights  \n");
      hypre_printf("       6=extended classical modified interpolation  \n");
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
      hypre_printf("  -ns_down    <val>       : set no. of sweeps for down cycle\n");
      hypre_printf("  -ns_up      <val>       : set no. of sweeps for up cycle\n");
      hypre_printf("\n");
      hypre_printf("  -mu   <val>            : set AMG cycles (1=V, 2=W, etc.)\n");
      hypre_printf("  -th   <val>            : set AMG threshold Theta = val \n");
      hypre_printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
      hypre_printf("  -Pmx  <val>            : set maximal no. of elmts per row for AMG interpolation \n");
      hypre_printf("  -jtr  <val>            : set truncation threshold for Jacobi interpolation = val \n");
      hypre_printf("  -Ssw  <val>            : set S-commpkg-switch = val \n");
      hypre_printf("  -mxrs <val>            : set AMG maximum row sum threshold for dependency weakening \n");
      hypre_printf("  -nf <val>              : set number of functions for systems AMG\n");
      hypre_printf("  -numsamp <val>         : set number of sample vectors for GSMG\n");

      hypre_printf("  -postinterptype <val>  : invokes <val> no. of Jacobi interpolation steps after main interpolation\n");
      hypre_printf("\n");
      hypre_printf("  -cgcitr <val>          : set maximal number of coarsening iterations for CGC\n");
      hypre_printf("  -solver_type <val>     : sets solver within Hybrid solver\n");
      hypre_printf("                         : 1  PCG  (default)\n");
      hypre_printf("                         : 2  GMRES\n");
      hypre_printf("                         : 3  BiCGSTAB\n");

      hypre_printf("  -w   <val>             : set Jacobi relax weight = val\n");
      hypre_printf("  -k   <val>             : dimension Krylov space for GMRES\n");
      hypre_printf("  -aug   <val>           : number of augmentation vectors for LGMRES (-k indicates total approx space size)\n");

      hypre_printf("  -mxl  <val>            : maximum number of levels (AMG, ParaSAILS)\n");
      hypre_printf("  -tol  <val>            : set solver convergence tolerance = val\n");
      hypre_printf("  -atol  <val>           : set solver absolute convergence tolerance = val\n");
      hypre_printf("  -max_iter  <val>       : set max iterations\n");
      hypre_printf("  -mg_max_iter  <val>    : set max iterations for mg solvers\n");
      hypre_printf("  -agg_nl  <val>         : set number of aggressive coarsening levels (default:0)\n");
      hypre_printf("  -np  <val>             : set number of paths of length 2 for aggr. coarsening\n");
      hypre_printf("\n");
      hypre_printf("  -sai_th   <val>        : set ParaSAILS threshold = val \n");
      hypre_printf("  -sai_filt <val>        : set ParaSAILS filter = val \n");
      hypre_printf("\n");
      hypre_printf("  -level   <val>         : set k in ILU(k) for Euclid \n");
      hypre_printf("  -bj <val>              : enable block Jacobi ILU for Euclid \n");
      hypre_printf("  -ilut <val>            : set drop tolerance for ILUT in Euclid\n");
      hypre_printf("                           Note ILUT is sequential only!\n");
      hypre_printf("  -sparseA <val>         : set drop tolerance in ILU(k) for Euclid \n");
      hypre_printf("  -rowScale <val>        : enable row scaling in Euclid \n");
      hypre_printf("\n");
      hypre_printf("  -drop_tol  <val>       : set threshold for dropping in PILUT\n");
      hypre_printf("  -nonzeros_to_keep <val>: number of nonzeros in each row to keep\n");
      hypre_printf("\n");
      hypre_printf("  -iout <val>            : set output flag\n");
      hypre_printf("       0=no output    1=matrix stats\n");
      hypre_printf("       2=cycle stats  3=matrix & cycle stats\n");
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
   if ( build_matrix_type == -1 )
   {
      ierr = HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                                 HYPRE_PARCSR, &ij_A );
      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in the system matrix!\n");
         exit(1);
      }
   }
   else if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParFromOneFile(argc, argv, build_matrix_arg_index, num_functions,
                          &parcsr_A);
   }
   else if ( build_matrix_type == 2 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 3 )
   {
      BuildParLaplacian9pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 4 )
   {
      BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParDifConv(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 6 )
   {
      BuildParVarDifConv(argc, argv, build_matrix_arg_index, &parcsr_A, &b);
      /*HYPRE_ParCSRMatrixPrint(parcsr_A,"mat100");*/
   }
   else if ( build_matrix_type == 7 )
   {
      BuildParRotate7pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }

   else
   {
      hypre_printf("You have asked for an unsupported problem with\n");
      hypre_printf("build_matrix_type = %d.\n", build_matrix_type);
      return (-1);
   }
   /* BM Oct 23, 2006 */
   if (plot_grids)
   {
      if (build_matrix_type > 1 &&  build_matrix_type < 8)
         BuildParCoordinates (argc, argv, build_matrix_arg_index,
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

      local_num_rows = last_local_row - first_local_row + 1;
      local_num_cols = last_local_col - first_local_col + 1;
   }
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Generate Matrix", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /* Check the ij interface - not necessary if one just wants to test solvers */
   if (test_ij && build_matrix_type > -1)
   {
      HYPRE_Int mx_size = 5;
      time_index = hypre_InitializeTiming("Generate IJ matrix");
      hypre_BeginTiming(time_index);

      ierr += HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );

      ierr += HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                    first_local_col, last_local_col,
                                    &ij_A );

      ierr += HYPRE_IJMatrixSetObjectType( ij_A, HYPRE_PARCSR );
      num_rows = local_num_rows;
      if (off_proc)
      {
         if (myid != num_procs - 1) { num_rows++; }
         if (myid) { num_rows++; }
      }
      /* The following shows how to build an IJMatrix if one has only an
         estimate for the row sizes */
      row_nums = hypre_CTAlloc(HYPRE_Int,  num_rows, HYPRE_MEMORY_HOST);
      num_cols = hypre_CTAlloc(HYPRE_Int,  num_rows, HYPRE_MEMORY_HOST);
      if (sparsity_known == 1)
      {
         diag_sizes = hypre_CTAlloc(HYPRE_Int,  local_num_rows, HYPRE_MEMORY_HOST);
         offdiag_sizes = hypre_CTAlloc(HYPRE_Int,  local_num_rows, HYPRE_MEMORY_HOST);
      }
      else
      {
         size = 5;
         if (sparsity_known == 0)
         {
            if (build_matrix_type == 2) { size = 7; }
            if (build_matrix_type == 3) { size = 9; }
            if (build_matrix_type == 4) { size = 27; }
         }
         row_sizes = hypre_CTAlloc(HYPRE_Int,  num_rows, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_rows; i++)
         {
            row_sizes[i] = size;
         }
      }
      local_row = 0;
      if (build_matrix_type == 2) { mx_size = 7; }
      if (build_matrix_type == 3) { mx_size = 9; }
      if (build_matrix_type == 4) { mx_size = 27; }
      col_nums = hypre_CTAlloc(HYPRE_Int,  mx_size * num_rows, HYPRE_MEMORY_HOST);
      data = hypre_CTAlloc(HYPRE_Real,  mx_size * num_rows, HYPRE_MEMORY_HOST);
      i_indx = 0;
      j_indx = 0;
      if (off_proc && myid)
      {
         num_cols[i_indx] = 2;
         row_nums[i_indx++] = first_local_row - 1;
         col_nums[j_indx] = first_local_row - 1;
         data[j_indx++] = 6.;
         col_nums[j_indx] = first_local_row - 2;
         data[j_indx++] = -1;
      }
      for (i = 0; i < local_num_rows; i++)
      {
         row_nums[i_indx] = first_local_row + i;
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, first_local_row + i, &size,
                                           &col_inds, &values);
         num_cols[i_indx++] = size;
         for (j = 0; j < size; j++)
         {
            col_nums[j_indx] = col_inds[j];
            data[j_indx++] = values[j];
            if (sparsity_known == 1)
            {
               if (col_inds[j] < first_local_row || col_inds[j] > last_local_row)
               {
                  offdiag_sizes[local_row]++;
               }
               else
               {
                  diag_sizes[local_row]++;
               }
            }
         }
         local_row++;
         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, first_local_row + i, &size,
                                               &col_inds, &values );
      }
      if (off_proc && myid != num_procs - 1)
      {
         num_cols[i_indx] = 2;
         row_nums[i_indx++] = last_local_row + 1;
         col_nums[j_indx] = last_local_row + 2;
         data[j_indx++] = -1.;
         col_nums[j_indx] = last_local_row + 1;
         data[j_indx++] = 6;
      }

      /*ierr += HYPRE_IJMatrixSetRowSizes ( ij_A, (const HYPRE_Int *) num_cols );*/
      if (sparsity_known == 1)
         ierr += HYPRE_IJMatrixSetDiagOffdSizes( ij_A, (const HYPRE_Int *) diag_sizes,
                                                 (const HYPRE_Int *) offdiag_sizes );
      else
      {
         ierr = HYPRE_IJMatrixSetRowSizes ( ij_A, (const HYPRE_Int *) row_sizes );
      }

      ierr += HYPRE_IJMatrixInitialize( ij_A );

      if (omp_flag) { HYPRE_IJMatrixSetOMPFlag(ij_A, 1); }

      if (chunk)
      {
         if (add)
            ierr += HYPRE_IJMatrixAddToValues(ij_A, num_rows, num_cols, row_nums,
                                              (const HYPRE_Int *) col_nums,
                                              (const HYPRE_Real *) data);
         else
            ierr += HYPRE_IJMatrixSetValues(ij_A, num_rows, num_cols, row_nums,
                                            (const HYPRE_Int *) col_nums,
                                            (const HYPRE_Real *) data);
      }
      else
      {
         j_indx = 0;
         for (i = 0; i < num_rows; i++)
         {
            if (add)
               ierr += HYPRE_IJMatrixAddToValues( ij_A, 1, &num_cols[i], &row_nums[i],
                                                  (const HYPRE_Int *) &col_nums[j_indx],
                                                  (const HYPRE_Real *) &data[j_indx] );
            else
               ierr += HYPRE_IJMatrixSetValues( ij_A, 1, &num_cols[i], &row_nums[i],
                                                (const HYPRE_Int *) &col_nums[j_indx],
                                                (const HYPRE_Real *) &data[j_indx] );
            j_indx += num_cols[i];
         }
      }
      hypre_TFree(col_nums, HYPRE_MEMORY_HOST);
      hypre_TFree(data, HYPRE_MEMORY_HOST);
      hypre_TFree(row_nums, HYPRE_MEMORY_HOST);
      hypre_TFree(num_cols, HYPRE_MEMORY_HOST);
      if (sparsity_known == 1)
      {
         hypre_TFree(diag_sizes, HYPRE_MEMORY_HOST);
         hypre_TFree(offdiag_sizes, HYPRE_MEMORY_HOST);
      }
      else
      {
         hypre_TFree(row_sizes, HYPRE_MEMORY_HOST);
      }

      ierr += HYPRE_IJMatrixAssemble( ij_A );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("IJ Matrix Setup", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      if (ierr)
      {
         hypre_printf("Error in driver building IJMatrix from parcsr matrix. \n");
         return (-1);
      }

      /* This is to emphasize that one can IJMatrixAddToValues after an
         IJMatrixRead or an IJMatrixAssemble.  After an IJMatrixRead,
         assembly is unnecessary if the sparsity pattern of the matrix is
         not changed somehow.  If one has not used IJMatrixRead, one has
         the opportunity to IJMatrixAddTo before a IJMatrixAssemble. */

      ncols    = hypre_CTAlloc(HYPRE_Int,  last_local_row - first_local_row + 1, HYPRE_MEMORY_HOST);
      rows     = hypre_CTAlloc(HYPRE_Int,  last_local_row - first_local_row + 1, HYPRE_MEMORY_HOST);
      col_inds = hypre_CTAlloc(HYPRE_Int,  last_local_row - first_local_row + 1, HYPRE_MEMORY_HOST);
      values   = hypre_CTAlloc(HYPRE_Real,  last_local_row - first_local_row + 1, HYPRE_MEMORY_HOST);

      if (dt < dt_inf)
      {
         val = 1. / dt;
      }
      else
      {
         val = 0.;   /* Use zero to avoid unintentional loss of significance */
      }

      for (i = first_local_row; i <= last_local_row; i++)
      {
         j = i - first_local_row;
         rows[j] = i;
         ncols[j] = 1;
         col_inds[j] = i;
         values[j] = val;
      }

      ierr += HYPRE_IJMatrixAddToValues( ij_A,
                                         local_num_rows,
                                         ncols, rows,
                                         (const HYPRE_Int *) col_inds,
                                         (const HYPRE_Real *) values );

      hypre_TFree(values, HYPRE_MEMORY_HOST);
      hypre_TFree(col_inds, HYPRE_MEMORY_HOST);
      hypre_TFree(rows, HYPRE_MEMORY_HOST);
      hypre_TFree(ncols, HYPRE_MEMORY_HOST);

      /* If sparsity pattern is not changed since last IJMatrixAssemble call,
         this should be a no-op */

      ierr += HYPRE_IJMatrixAssemble( ij_A );

      /*-----------------------------------------------------------
       * Fetch the resulting underlying matrix out
       *-----------------------------------------------------------*/
      if (build_matrix_type > -1)
      {
         ierr += HYPRE_ParCSRMatrixDestroy(parcsr_A);
      }

      ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_A = (HYPRE_ParCSRMatrix) object;

   }

   /*-----------------------------------------------------------
    * Set up the interp vector
    *-----------------------------------------------------------*/

   if ( build_rbm)
   {
      char new_file_name[80];
      /* RHS */
      interp_vecs = hypre_CTAlloc(HYPRE_ParVector, num_interp_vecs, HYPRE_MEMORY_HOST);
      ij_rbm = hypre_CTAlloc(HYPRE_IJVector, num_interp_vecs, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_interp_vecs; i++)
      {
         hypre_sprintf(new_file_name, "%s.%d", argv[build_rbm_index], i);
         ierr = HYPRE_IJVectorRead( new_file_name, hypre_MPI_COMM_WORLD,
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
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 1 )
   {

#if 0
      hypre_printf("build_rhs_type == 1 not currently implemented\n");
      return (-1);
#else
      /* RHS - this has not been tested for multiple processors*/
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, NULL, &b);

      hypre_printf("  Initial guess is 0\n");

      ij_b = NULL;

      /* initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

#endif
   }
   else if ( build_rhs_type == 7 )
   {

      /* rhs */
      BuildParRhsFromFile(argc, argv, build_rhs_arg_index, &b);

      hypre_printf("  Initial guess is 0\n");

      ij_b = NULL;

      /* initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
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
      {
         values[i] = 1.0;
      }
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
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

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
      HYPRE_ParVectorInnerProd(b, b, &norm);
      norm = 1. / hypre_sqrt(norm);
      ierr = HYPRE_ParVectorScale(norm, b);

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

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

      values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 1.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      HYPRE_ParCSRMatrixMatvec(1., parcsr_A, x, 0., b);

      /* Initial guess */
      values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);
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

      values = hypre_CTAlloc(HYPRE_Real,  local_num_rows, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = 0.;
      }
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
      {
         values[i] = 1.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }

   if ( build_src_type == 0 )
   {
#if 0
      /* RHS */
      BuildRhsParFromFile(argc, argv, build_src_arg_index, &b);
#endif

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

      values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 1 )
   {
      hypre_printf("build_src_type == 1 not currently implemented\n");
      return (-1);

#if 0
      BuildRhsParFromOneFile(argc, argv, build_src_arg_index, part_b, &b);
#endif
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

      values = hypre_CTAlloc(HYPRE_Real,  local_num_rows, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = 1.;
      }
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

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
      values = hypre_CTAlloc(HYPRE_Real,  local_num_rows, HYPRE_MEMORY_HOST);

      hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = hypre_Rand();
      }

      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

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

      values = hypre_CTAlloc(HYPRE_Real,  local_num_rows, HYPRE_MEMORY_HOST);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = hypre_Rand() / dt;
      }
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = hypre_Rand();
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Vector Setup", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

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
         {
            hypre_printf (" Number of functions = %d \n", num_functions);
         }
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
      if (myid == 0) { hypre_printf("Solver:  AMG\n"); }
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
      {
         HYPRE_ParCSRHybridSetCycleRelaxType(amg_solver, relax_down, 1);
      }
      if (relax_up > -1)
      {
         HYPRE_ParCSRHybridSetCycleRelaxType(amg_solver, relax_up, 2);
      }
      if (relax_coarse > -1)
      {
         HYPRE_ParCSRHybridSetCycleRelaxType(amg_solver, relax_coarse, 3);
      }
      HYPRE_ParCSRHybridSetRelaxOrder(amg_solver, relax_order);
      HYPRE_ParCSRHybridSetMaxCoarseSize(amg_solver, coarse_threshold);
      HYPRE_ParCSRHybridSetMinCoarseSize(amg_solver, min_coarse_size);
      HYPRE_ParCSRHybridSetSeqThreshold(amg_solver, seq_threshold);
      HYPRE_ParCSRHybridSetRelaxWt(amg_solver, relax_wt);
      HYPRE_ParCSRHybridSetOuterWt(amg_solver, outer_wt);
      if (level_w > -1)
      {
         HYPRE_ParCSRHybridSetLevelRelaxWt(amg_solver, relax_wt_level, level_w);
      }
      if (level_ow > -1)
      {
         HYPRE_ParCSRHybridSetLevelOuterWt(amg_solver, outer_wt_level, level_ow);
      }

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

   if (solver_id == 0)
   {
      if (myid == 0) { hypre_printf("Solver:  AMG\n"); }
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGCreate(&amg_solver);
      /* BM Aug 25, 2006 */
      HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
      HYPRE_BoomerAMGSetInterpType(amg_solver, interp_type);
      HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
      HYPRE_BoomerAMGSetNumSamples(amg_solver, gsmg_samples);
      HYPRE_BoomerAMGSetCoarsenType(amg_solver, coarsen_type);
      HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      HYPRE_BoomerAMGSetTol(amg_solver, tol);
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
      HYPRE_BoomerAMGSetPrintLevel(amg_solver, 3);
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
      {
         HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
      }
      if (relax_up > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
      }
      if (relax_coarse > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
      }
      HYPRE_BoomerAMGSetChebyOrder(amg_solver, cheby_order);
      HYPRE_BoomerAMGSetChebyFraction(amg_solver, cheby_fraction);
      HYPRE_BoomerAMGSetRelaxOrder(amg_solver, relax_order);
      HYPRE_BoomerAMGSetRelaxWt(amg_solver, relax_wt);
      HYPRE_BoomerAMGSetOuterWt(amg_solver, outer_wt);
      HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      if (level_w > -1)
      {
         HYPRE_BoomerAMGSetLevelRelaxWt(amg_solver, relax_wt_level, level_w);
      }
      if (level_ow > -1)
      {
         HYPRE_BoomerAMGSetLevelOuterWt(amg_solver, outer_wt_level, level_ow);
      }
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
      if (eu_level < 0) { eu_level = 0; }
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
      {
         HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);
      }
      HYPRE_BoomerAMGSetAdditive(amg_solver, additive);
      HYPRE_BoomerAMGSetMultAdditive(amg_solver, mult_add);
      HYPRE_BoomerAMGSetSimple(amg_solver, simple);
      HYPRE_BoomerAMGSetMultAddPMaxElmts(amg_solver, add_P_max_elmts);
      HYPRE_BoomerAMGSetMultAddTruncFactor(amg_solver, add_trunc_factor);

      HYPRE_BoomerAMGSetMaxIter(amg_solver, mg_max_iter);
      HYPRE_BoomerAMGSetRAP2(amg_solver, rap2);
      HYPRE_BoomerAMGSetKeepTranspose(amg_solver, keepTranspose);
      /*HYPRE_BoomerAMGSetNonGalerkTol(amg_solver, nongalerk_num_tol, nongalerk_tol);*/
      if (nongalerk_tol)
      {
         HYPRE_BoomerAMGSetNonGalerkinTol(amg_solver, nongalerk_tol[nongalerk_num_tol - 1]);
         for (i = 0; i < nongalerk_num_tol - 1; i++)
         {
            HYPRE_BoomerAMGSetLevelNonGalerkinTol(amg_solver, nongalerk_tol[i], i);
         }
      }
      if (build_rbm)
      {
         HYPRE_BoomerAMGSetInterpVectors(amg_solver, num_interp_vecs, interp_vecs);
         HYPRE_BoomerAMGSetInterpVecVariant(amg_solver, interp_vec_variant);
         HYPRE_BoomerAMGSetInterpVecQMax(amg_solver, Q_max);
         HYPRE_BoomerAMGSetInterpVecAbsQTrunc(amg_solver, Q_trunc);
      }

      /* BM Oct 23, 2006 */
      if (plot_grids)
      {
         HYPRE_BoomerAMGSetPlotGrids (amg_solver, 1);
         HYPRE_BoomerAMGSetPlotFileName (amg_solver, plot_file_name);
         HYPRE_BoomerAMGSetCoordDim (amg_solver, coord_dim);
         HYPRE_BoomerAMGSetCoordinates (amg_solver, coordinates);
      }

      HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BoomerAMGGetNumIterations(amg_solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(amg_solver, &final_res_norm);

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("BoomerAMG Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);
      HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
#endif

      HYPRE_BoomerAMGDestroy(amg_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using GSMG
    *-----------------------------------------------------------*/

   if (solver_id == 13)
   {
      /* reset some smoother parameters */

      relax_order = 0;

      if (myid == 0) { hypre_printf("Solver:  GSMG\n"); }
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGCreate(&amg_solver);
      HYPRE_BoomerAMGSetGSMG(amg_solver, 4); /* specify GSMG */
      /* BM Aug 25, 2006 */
      HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
      HYPRE_BoomerAMGSetInterpType(amg_solver, interp_type);
      HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
      HYPRE_BoomerAMGSetNumSamples(amg_solver, gsmg_samples);
      HYPRE_BoomerAMGSetCoarsenType(amg_solver, coarsen_type);
      HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      HYPRE_BoomerAMGSetTol(amg_solver, tol);
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
      HYPRE_BoomerAMGSetPrintLevel(amg_solver, 3);
      HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log");
      HYPRE_BoomerAMGSetMaxIter(amg_solver, mg_max_iter);
      HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
      HYPRE_BoomerAMGSetRelaxType(amg_solver, relax_type);
      if (relax_down > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
      }
      if (relax_up > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
      }
      if (relax_coarse > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
      }
      HYPRE_BoomerAMGSetChebyOrder(amg_solver, cheby_order);
      HYPRE_BoomerAMGSetChebyFraction(amg_solver, cheby_fraction);
      HYPRE_BoomerAMGSetRelaxOrder(amg_solver, relax_order);
      HYPRE_BoomerAMGSetRelaxWt(amg_solver, relax_wt);
      HYPRE_BoomerAMGSetOuterWt(amg_solver, outer_wt);
      if (level_w > -1)
      {
         HYPRE_BoomerAMGSetLevelRelaxWt(amg_solver, relax_wt_level, level_w);
      }
      if (level_ow > -1)
      {
         HYPRE_BoomerAMGSetLevelOuterWt(amg_solver, outer_wt_level, level_ow);
      }
      HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
      HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
      HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, smooth_num_levels);
      HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      HYPRE_BoomerAMGSetMaxRowSum(amg_solver, max_row_sum);
      HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);
      HYPRE_BoomerAMGSetVariant(amg_solver, variant);
      HYPRE_BoomerAMGSetOverlap(amg_solver, overlap);
      HYPRE_BoomerAMGSetDomainType(amg_solver, domain_type);
      HYPRE_BoomerAMGSetSchwarzUseNonSymm(amg_solver, use_nonsymm_schwarz);
      HYPRE_BoomerAMGSetSchwarzRlxWeight(amg_solver, schwarz_rlx_weight);
      if (eu_level < 0) { eu_level = 0; }
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
      if (num_functions > 1)
      {
         HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);
      }
      HYPRE_BoomerAMGSetAdditive(amg_solver, additive);
      HYPRE_BoomerAMGSetMultAdditive(amg_solver, mult_add);
      HYPRE_BoomerAMGSetSimple(amg_solver, simple);
      HYPRE_BoomerAMGSetMultAddPMaxElmts(amg_solver, add_P_max_elmts);
      HYPRE_BoomerAMGSetMultAddTruncFactor(amg_solver, add_trunc_factor);
      HYPRE_BoomerAMGSetRAP2(amg_solver, rap2);
      HYPRE_BoomerAMGSetKeepTranspose(amg_solver, keepTranspose);
      if (nongalerk_tol)
      {
         HYPRE_BoomerAMGSetNonGalerkinTol(amg_solver, nongalerk_tol[nongalerk_num_tol - 1]);
         for (i = 0; i < nongalerk_num_tol - 1; i++)
         {
            HYPRE_BoomerAMGSetLevelNonGalerkinTol(amg_solver, nongalerk_tol[i], i);
         }
      }

      HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);
      HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
#endif

      HYPRE_BoomerAMGDestroy(amg_solver);
   }

   if (solver_id == 999)
   {
      HYPRE_IJMatrix ij_M;
      HYPRE_ParCSRMatrix  parcsr_mat;

      /* use ParaSails preconditioner */
      if (myid == 0) { hypre_printf("Test ParaSails Build IJMatrix\n"); }

      HYPRE_IJMatrixPrint(ij_A, "parasails.in");

      HYPRE_ParaSailsCreate(hypre_MPI_COMM_WORLD, &pcg_precond);
      HYPRE_ParaSailsSetParams(pcg_precond, 0., 0);
      HYPRE_ParaSailsSetFilter(pcg_precond, 0.);
      HYPRE_ParaSailsSetLogging(pcg_precond, ioutdat);

      HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_mat = (HYPRE_ParCSRMatrix) object;

      HYPRE_ParaSailsSetup(pcg_precond, parcsr_mat, NULL, NULL);
      HYPRE_ParaSailsBuildIJMatrix(pcg_precond, &ij_M);
      HYPRE_IJMatrixPrint(ij_M, "parasails.out");

      if (myid == 0) { hypre_printf("Printed to parasails.out.\n"); }
      exit(0);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2 || solver_id == 8 ||
       solver_id == 12 || solver_id == 14 || solver_id == 43)
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter(pcg_solver, max_iter);
      HYPRE_PCGSetTol(pcg_solver, tol);
      HYPRE_PCGSetTwoNorm(pcg_solver, 1);
      HYPRE_PCGSetRelChange(pcg_solver, rel_change);
      HYPRE_PCGSetPrintLevel(pcg_solver, ioutdat);
      HYPRE_PCGSetAbsoluteTol(pcg_solver, atol);

      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { hypre_printf("Solver: AMG-PCG\n"); }
         HYPRE_BoomerAMGCreate(&pcg_precond);
         /* BM Aug 25, 2006 */
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
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
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
         if (eu_level < 0) { eu_level = 0; }
         HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
         if (nongalerk_tol)
         {
            HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         if (build_rbm)
         {
            HYPRE_BoomerAMGSetInterpVectors(pcg_precond, num_interp_vecs, interp_vecs);
            HYPRE_BoomerAMGSetInterpVecVariant(pcg_precond, interp_vec_variant);
            HYPRE_BoomerAMGSetInterpVecQMax(pcg_precond, Q_max);
            HYPRE_BoomerAMGSetInterpVecAbsQTrunc(pcg_precond, Q_trunc);
         }
         HYPRE_PCGSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                             pcg_precond);
      }
      else if (solver_id == 2)
      {

         /* use diagonal scaling as preconditioner */
         if (myid == 0) { hypre_printf("Solver: DS-PCG\n"); }
         pcg_precond = NULL;

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                             pcg_precond);
      }
      else if (solver_id == 8)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) { hypre_printf("Solver: ParaSails-PCG\n"); }

         HYPRE_ParaSailsCreate(hypre_MPI_COMM_WORLD, &pcg_precond);
         HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
         HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
         HYPRE_ParaSailsSetLogging(pcg_precond, poutdat);

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup,
                             pcg_precond);
      }
      else if (solver_id == 12)
      {
         /* use Schwarz preconditioner */
         if (myid == 0) { hypre_printf("Solver: Schwarz-PCG\n"); }

         HYPRE_SchwarzCreate(&pcg_precond);
         HYPRE_SchwarzSetVariant(pcg_precond, variant);
         HYPRE_SchwarzSetOverlap(pcg_precond, overlap);
         HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
         HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);
         HYPRE_SchwarzSetNonSymm(pcg_precond, use_nonsymm_schwarz);
         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSetup,
                             pcg_precond);
      }
      else if (solver_id == 14)
      {
         /* use GSMG as preconditioner */

         /* reset some smoother parameters */

         /* fine grid */
         relax_order = 0;

         if (myid == 0) { hypre_printf("Solver: GSMG-PCG\n"); }
         HYPRE_BoomerAMGCreate(&pcg_precond);
         HYPRE_BoomerAMGSetGSMG(pcg_precond, 4);
         /* BM Aug 25, 2006 */
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
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) { eu_level = 0; }
         HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
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
         HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
         if (nongalerk_tol)
         {
            HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         HYPRE_PCGSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                             pcg_precond);
      }
      else if (solver_id == 43)
      {
         /* use Euclid preconditioning */
         if (myid == 0) { hypre_printf("Solver: Euclid-PCG\n"); }

         HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally
            parse the command line.
         */
         if (eu_level > -1) { HYPRE_EuclidSetLevel(pcg_precond, eu_level); }
         if (eu_ilut) { HYPRE_EuclidSetILUT(pcg_precond, eu_ilut); }
         if (eu_sparse_A) { HYPRE_EuclidSetSparseA(pcg_precond, eu_sparse_A); }
         if (eu_row_scale) { HYPRE_EuclidSetRowScale(pcg_precond, eu_row_scale); }
         if (eu_bj) { HYPRE_EuclidSetBJ(pcg_precond, eu_bj); }
         HYPRE_EuclidSetStats(pcg_precond, eu_stats);
         HYPRE_EuclidSetMem(pcg_precond, eu_mem);

         /*HYPRE_EuclidSetParams(pcg_precond, argc, argv);*/

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                             pcg_precond);
      }

      HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
         hypre_printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         hypre_printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");
      }

      HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A,
                     (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)parcsr_A,
                     (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A,
                     (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)parcsr_A,
                     (HYPRE_Vector)b, (HYPRE_Vector)x);
#endif

      HYPRE_ParCSRPCGDestroy(pcg_solver);

      if (solver_id == 1)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      else if (solver_id == 8)
      {
         HYPRE_ParaSailsDestroy(pcg_precond);
      }
      else if (solver_id == 12)
      {
         HYPRE_SchwarzDestroy(pcg_precond);
      }
      else if (solver_id == 14)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      else if (solver_id == 43)
      {
         HYPRE_EuclidDestroy(pcg_precond);
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
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4 || solver_id == 7 ||
       solver_id == 15 || solver_id == 18 || solver_id == 44)
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
         if (myid == 0) { hypre_printf("Solver: AMG-GMRES\n"); }

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
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
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
         if (eu_level < 0) { eu_level = 0; }
         HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
         if (nongalerk_tol)
         {
            HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         if (build_rbm)
         {
            HYPRE_BoomerAMGSetInterpVectors(pcg_precond, 1, interp_vecs);
            HYPRE_BoomerAMGSetInterpVecVariant(pcg_precond, interp_vec_variant);
            HYPRE_BoomerAMGSetInterpVecQMax(pcg_precond, Q_max);
            HYPRE_BoomerAMGSetInterpVecAbsQTrunc(pcg_precond, Q_trunc);
         }
         HYPRE_GMRESSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                               pcg_precond);
      }
      else if (solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { hypre_printf("Solver: DS-GMRES\n"); }
         pcg_precond = NULL;

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                               pcg_precond);
      }
      else if (solver_id == 7)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) { hypre_printf("Solver: PILUT-GMRES\n"); }

         ierr = HYPRE_ParCSRPilutCreate( hypre_MPI_COMM_WORLD, &pcg_precond );
         if (ierr)
         {
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
      else if (solver_id == 15)
      {
         /* use GSMG as preconditioner */

         /* reset some smoother parameters */

         relax_order = 0;

         if (myid == 0) { hypre_printf("Solver: GSMG-GMRES\n"); }
         HYPRE_BoomerAMGCreate(&pcg_precond);
         HYPRE_BoomerAMGSetGSMG(pcg_precond, 4);
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
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         HYPRE_BoomerAMGSetSchwarzUseNonSymm(pcg_precond, use_nonsymm_schwarz);
         HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) { eu_level = 0; }
         HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
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
         HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
         if (nongalerk_tol)
         {
            HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         HYPRE_GMRESSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                               pcg_precond);
      }
      else if (solver_id == 18)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) { hypre_printf("Solver: ParaSails-GMRES\n"); }

         HYPRE_ParaSailsCreate(hypre_MPI_COMM_WORLD, &pcg_precond);
         HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
         HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
         HYPRE_ParaSailsSetLogging(pcg_precond, poutdat);
         HYPRE_ParaSailsSetSym(pcg_precond, 0);

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup,
                               pcg_precond);
      }
      else if (solver_id == 44)
      {
         /* use Euclid preconditioning */
         if (myid == 0) { hypre_printf("Solver: Euclid-GMRES\n"); }

         HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &pcg_precond);

         if (eu_level > -1) { HYPRE_EuclidSetLevel(pcg_precond, eu_level); }
         if (eu_ilut) { HYPRE_EuclidSetILUT(pcg_precond, eu_ilut); }
         if (eu_sparse_A) { HYPRE_EuclidSetSparseA(pcg_precond, eu_sparse_A); }
         if (eu_row_scale) { HYPRE_EuclidSetRowScale(pcg_precond, eu_row_scale); }
         if (eu_bj) { HYPRE_EuclidSetBJ(pcg_precond, eu_bj); }
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
         return (-1);
      }
      else if (myid == 0)
      {
         hypre_printf("HYPRE_GMRESGetPrecond got good precond\n");
      }
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
      HYPRE_GMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
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
    * Solve the system using LGMRES
    *-----------------------------------------------------------*/

   if (solver_id == 50 || solver_id == 51 )
   {
      time_index = hypre_InitializeTiming("LGMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRLGMRESCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_LGMRESSetKDim(pcg_solver, k_dim);
      HYPRE_LGMRESSetAugDim(pcg_solver, aug_dim);
      HYPRE_LGMRESSetMaxIter(pcg_solver, max_iter);
      HYPRE_LGMRESSetTol(pcg_solver, tol);
      HYPRE_LGMRESSetAbsoluteTol(pcg_solver, atol);
      HYPRE_LGMRESSetLogging(pcg_solver, 1);
      HYPRE_LGMRESSetPrintLevel(pcg_solver, ioutdat);

      if (solver_id == 51)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { hypre_printf("Solver: AMG-LGMRES\n"); }

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
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
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
         if (eu_level < 0) { eu_level = 0; }
         HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
         if (nongalerk_tol)
         {
            HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         HYPRE_LGMRESSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_LGMRESSetPrecond(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                pcg_precond);
      }
      else if (solver_id == 50)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { hypre_printf("Solver: DS-LGMRES\n"); }
         pcg_precond = NULL;

         HYPRE_LGMRESSetPrecond(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                pcg_precond);
      }

      HYPRE_LGMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         hypre_printf("HYPRE_LGMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         hypre_printf("HYPRE_LGMRESGetPrecond got good precond\n");
      }
      HYPRE_LGMRESSetup
      (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("LGMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_LGMRESSolve
      (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_LGMRESGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_LGMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

      HYPRE_ParCSRLGMRESDestroy(pcg_solver);

      if (solver_id == 51)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("LGMRES Iterations = %d\n", num_iterations);
         hypre_printf("Final LGMRES Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using FlexGMRES
    *-----------------------------------------------------------*/

   if (solver_id == 60 || solver_id == 61 )
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
         if (myid == 0) { hypre_printf("Solver: AMG-FlexGMRES\n"); }

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
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
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
         if (eu_level < 0) { eu_level = 0; }
         HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
         if (nongalerk_tol)
         {
            HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         HYPRE_FlexGMRESSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_FlexGMRESSetPrecond(pcg_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 60)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { hypre_printf("Solver: DS-FlexGMRES\n"); }
         pcg_precond = NULL;

         HYPRE_FlexGMRESSetPrecond(pcg_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                   (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
      }

      HYPRE_FlexGMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         hypre_printf("HYPRE_FlexGMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         hypre_printf("HYPRE_FlexGMRESGetPrecond got good precond\n");
      }


      /* this is optional - could be a user defined one instead (see ex5.c)*/
      HYPRE_FlexGMRESSetModifyPC( pcg_solver,
                                  (HYPRE_PtrToModifyPCFcn) hypre_FlexGMRESModifyPCDefault);


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
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

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
         if (myid == 0) { hypre_printf("Solver: AMG-BiCGSTAB\n"); }
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
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
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
         if (eu_level < 0) { eu_level = 0; }
         HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
         if (nongalerk_tol)
         {
            HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         HYPRE_BiCGSTABSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                  pcg_precond);
      }
      else if (solver_id == 10)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { hypre_printf("Solver: DS-BiCGSTAB\n"); }
         pcg_precond = NULL;

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                  pcg_precond);
      }
      else if (solver_id == 11)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) { hypre_printf("Solver: PILUT-BiCGSTAB\n"); }

         ierr = HYPRE_ParCSRPilutCreate( hypre_MPI_COMM_WORLD, &pcg_precond );
         if (ierr)
         {
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
         if (myid == 0) { hypre_printf("Solver: Euclid-BICGSTAB\n"); }

         HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally
            parse the command line.
         */
         if (eu_level > -1) { HYPRE_EuclidSetLevel(pcg_precond, eu_level); }
         if (eu_ilut) { HYPRE_EuclidSetILUT(pcg_precond, eu_ilut); }
         if (eu_sparse_A) { HYPRE_EuclidSetSparseA(pcg_precond, eu_sparse_A); }
         if (eu_row_scale) { HYPRE_EuclidSetRowScale(pcg_precond, eu_row_scale); }
         if (eu_bj) { HYPRE_EuclidSetBJ(pcg_precond, eu_bj); }
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
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
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
         if (myid == 0) { hypre_printf("Solver: AMG-CGNR\n"); }
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
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
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
         {
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
         if (nongalerk_tol)
         {
            HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
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
         if (myid == 0) { hypre_printf("Solver: DS-CGNR\n"); }
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
         return (-1);
      }
      else if (myid == 0)
      {
         hypre_printf("HYPRE_ParCSRCGNRGetPrecond got good precond\n");
      }
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
      HYPRE_CGNRGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

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
   if (!(build_rhs_type == 1 || build_rhs_type == 7))
   {
      HYPRE_IJVectorGetObjectType(ij_b, &j);
   }

   if (print_system)
   {
      HYPRE_IJVectorPrint(ij_x, "IJ.out.x");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   if (test_ij || build_matrix_type == -1) { HYPRE_IJMatrixDestroy(ij_A); }
   else { HYPRE_ParCSRMatrixDestroy(parcsr_A); }

   /* for build_rhs_type = 1 or 7, we did not create ij_b  - just b*/
   if (build_rhs_type == 1 || build_rhs_type == 7)
   {
      HYPRE_ParVectorDestroy(b);
   }
   else
   {
      HYPRE_IJVectorDestroy(ij_b);
   }

   HYPRE_IJVectorDestroy(ij_x);

   if (build_rbm)
   {
      for (i = 0; i < num_interp_vecs; i++)
      {
         HYPRE_IJVectorDestroy(ij_rbm[i]);
      }
      hypre_TFree(ij_rbm, HYPRE_MEMORY_HOST);
      hypre_TFree(interp_vecs, HYPRE_MEMORY_HOST);
   }
   if (nongalerk_tol) { hypre_TFree(nongalerk_tol, HYPRE_MEMORY_HOST); }

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

   HYPRE_ParCSRMatrixRead(hypre_MPI_COMM_WORLD, filename, &A);

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

   HYPRE_ParVectorRead(hypre_MPI_COMM_WORLD, filename, &b);

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
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
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
      A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD,
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
         A = (HYPRE_ParCSRMatrix) GenerateSysLaplacian(hypre_MPI_COMM_WORLD,
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

         A = (HYPRE_ParCSRMatrix) GenerateSysLaplacianVCoef(hypre_MPI_COMM_WORLD,
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
   HYPRE_Real          hinx, hiny, hinz;

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
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   hinx = 1. / (nx + 1);
   hiny = 1. / (ny + 1);
   hinz = 1. / (nz + 1);

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  7, HYPRE_MEMORY_HOST);

   values[1] = -cx / (hinx * hinx);
   values[2] = -cy / (hiny * hiny);
   values[3] = -cz / (hinz * hinz);
   values[4] = -cx / (hinx * hinx) + ax / hinx;
   values[5] = -cy / (hiny * hiny) + ay / hiny;
   values[6] = -cz / (hinz * hinz) + az / hinz;

   values[0] = 0.;
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

   A = (HYPRE_ParCSRMatrix) GenerateDifConv(hypre_MPI_COMM_WORLD,
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
   HYPRE_Int            *row_part;
   HYPRE_Int            *col_part;

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
      num_nodes = num_dofs / num_functions;
      if (num_dofs != num_functions * num_nodes)
      {
         row_part = NULL;
         col_part = NULL;
      }
      else
      {
         row_part = hypre_CTAlloc(HYPRE_Int,  numprocs + 1, HYPRE_MEMORY_HOST);
         row_part[0] = 0;
         size = num_nodes / numprocs;
         rest = num_nodes - size * numprocs;
         for (i = 0; i < numprocs; i++)
         {
            row_part[i + 1] = row_part[i] + size * num_functions;
            if (i < rest) { row_part[i + 1] += num_functions; }
         }
         col_part = row_part;
      }
   }

   HYPRE_CSRMatrixToParCSRMatrix(hypre_MPI_COMM_WORLD, A_CSR, row_part, col_part, &A);

   *A_ptr = A;

   if (myid == 0) { HYPRE_CSRMatrixDestroy(A_CSR); }

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
   return (0);

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
   hypre_MPI_Request   *requests;
   hypre_MPI_Status    *status, status0;
   MPI_Comm    comm;

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
      dof_func = hypre_CTAlloc(HYPRE_Int,  global_size, HYPRE_MEMORY_HOST);

      for (j = 0; j < global_size; j++)
      {
         hypre_fscanf(fp, "%d", &dof_func[j]);
      }

      fclose(fp);

   }
   HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A, &partitioning);
   local_size = partitioning[myid + 1] - partitioning[myid];
   dof_func_local = hypre_CTAlloc(HYPRE_Int, local_size, HYPRE_MEMORY_HOST);

   if (myid == 0)
   {
      requests = hypre_CTAlloc(hypre_MPI_Request, num_procs - 1, HYPRE_MEMORY_HOST);
      status = hypre_CTAlloc(hypre_MPI_Status, num_procs - 1, HYPRE_MEMORY_HOST);
      j = 0;
      for (i = 1; i < num_procs; i++)
         hypre_MPI_Isend(&dof_func[partitioning[i]],
                         partitioning[i + 1] - partitioning[i],
                         HYPRE_MPI_INT, i, 0, comm, &requests[j++]);
      for (i = 0; i < local_size; i++)
      {
         dof_func_local[i] = dof_func[i];
      }
      hypre_MPI_Waitall(num_procs - 1, requests, status);
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
      hypre_TFree(status, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_MPI_Recv(dof_func_local, local_size, HYPRE_MPI_INT, 0, 0, comm, &status0);
   }

   *dof_func_ptr = dof_func_local;

   if (myid == 0) { hypre_TFree(dof_func, HYPRE_MEMORY_HOST); }

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
   HYPRE_Vector    b_CSR = NULL;

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
   HYPRE_VectorToParVector(hypre_MPI_COMM_WORLD, b_CSR, partitioning, &b);

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
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
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

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(hypre_MPI_COMM_WORLD,
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
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
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

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

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
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
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
                    HYPRE_ParCSRMatrix  *A_ptr,
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
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
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

   A = (HYPRE_ParCSRMatrix) GenerateVarDifConv(hypre_MPI_COMM_WORLD,
                                               nx, ny, nz, P, Q, R, p, q, r, eps, &rhs);

   *A_ptr = A;
   *rhs_ptr = rhs;

   return (0);
}

/**************************************************************************/


HYPRE_Int SetSysVcoefValues(HYPRE_Int num_fun, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
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
      coordinates = hypre_GenerateCoordinates(hypre_MPI_COMM_WORLD,
                                              nx, ny, nz, P, Q, R, p, q, r, coorddim);
   }
   else
   {
      coordinates = NULL;
   }

   *coorddim_ptr = coorddim;
   *coord_ptr = coordinates;
   return (0);
}
