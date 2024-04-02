/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*!
   This file contains a mocked-up example, based on ex5.c in the examples directory. 
   The goal is to give an idea of how a user may utilize hypre to assemble matrix data 
   and access solvers in a way that would facilitate a mixed-1recision solution of the 
   linear system. This particular driver demonstrates how the mixed-precision build may 
   be used to develop mixed-precision solvers, such as the defect-correction-based solver
   implemented here. Feel free to ask questions, make comments or suggestions 
   regarding any of the information below.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "_hypre_utilities.h"
#include "hypre_utilities_mup.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_mv_mp.h"
#include "hypre_parcsr_mv_mup.h"

#include "HYPRE_IJ_mv.h"
#include "hypre_IJ_mv_mup.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_ls_mp.h"
#include "hypre_parcsr_ls_mup.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"
#include "hypre_krylov_mup.h"
//#include "hypre_utilities_mp.h"

#include <time.h>

#define my_min(a,b)  (((a)<(b)) ? (a) : (b))

HYPRE_Int BuildParLaplacian_mp( HYPRE_Int argc, char *argv[], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParSysLaplacian_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParDifConv_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParLaplacian9pt_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParLaplacian27pt_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParLaplacian125pt_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParRotate7pt_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParVarDifConv_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr,
                   HYPRE_ParVector *rhs_flt_ptr, HYPRE_ParVector *rhs_dbl_ptr );

HYPRE_Int HYPRE_DefectCorrectionSolver(HYPRE_ParCSRMatrix A, 
		HYPRE_ParCSRMatrix B, 
		HYPRE_ParVector x, 
		HYPRE_ParVector b,
		HYPRE_Solver solver,
		HYPRE_PtrToSolverFcn approx_solve,
		HYPRE_Int maxits);

int main (int argc, char *argv[])
{
   int arg_index;
   int myid, num_procs;
   int ilower, iupper;
   int jlower, jupper;
   int solver_id = 0;
   double one = 1.0;
   double zero = 0.;
   int num_iterations;
   double dfinal_res_norm;
   float  final_res_norm;
   int	  time_index;   
   float  wall_time;   
   double max_row_sum = 1.0;
   int build_matrix_type = 2;
   int build_rhs_type = 2;
   int build_matrix_arg_index;
   int build_rhs_arg_index;
   int mg_max_iter = 50;
   int max_iter = 1000;
   int coarsen_type = 10;
   int interp_type = 6;
   int P_max_elmts = 4;
   double trunc_factor = 0.0;
   double strong_threshold = 0.25;
   int relax_type = 8;
   int relax_up = 14;
   int relax_down = 13;
   int relax_coarse = 9;
   int num_sweeps = 1;
   int ns_down = -1;
   int ns_up = -1;
   int ns_coarse = -1;
   int max_levels = 25;
   int debug_flag = 0;
   int agg_num_levels = 0;
   int num_paths = 1;
   int agg_interp_type = 4;
   int agg_P_max_elmts = 0;
   double agg_trunc_factor = 0;
   int agg_P12_max_elmts = 0;
   double agg_P12_trunc_factor = 0;
   int smooth_type = 6;
   int smooth_num_levels = 0;
   int smooth_num_sweeps = 1;
   double tol = 1.e-8;
   int ioutdat = 0;
   int poutdat = 0;
   int flex = 0;
   int num_functions = 1;
   int nodal = 0;
   int nodal_diag = 0;
   int keep_same_sign = 0;
   int cycle_type = 1;
   int relax_order = 0;
   double relax_wt = 1.0;
   double outer_wt = 1.0;
   int k_dim = 10;
   int two_norm = 0;
   int all = 0;
   int precision = 0;

   /*! Matrix and preconditioner declarations. Here, we declare IJMatrices and parcsr matrices
       for the solver (A, parcsr_A) and the preconditioner (B, A_dbl). I have included two suggestions 
       below on how we would utilize both of these matrices. 
   */

   HYPRE_ParCSRMatrix A_flt;
   HYPRE_IJVector ij_b_flt;
   HYPRE_ParVector b_flt;
   HYPRE_IJVector ij_x_flt;
   HYPRE_ParVector x_flt;

   HYPRE_ParCSRMatrix A_dbl;
   HYPRE_IJVector ij_b_dbl;
   HYPRE_ParVector b_dbl;
   HYPRE_IJVector ij_x_dbl;
   HYPRE_ParVector x_dbl;   

   HYPRE_Precision *precision_array;

   /*! Solver and preconditioner and declarations and solver_precision variable. Internally, HYPRE_SolverPrecision 
       is an enum struct containing HYPRE_REAL_float, HYPRE_REAL_SINGLE and HYPRE_REAL_LONG.
   */
   HYPRE_Solver solver, precond;
   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /*! We set up the linear system following ex5. */
   /* Some default problem parameters */
   solver_id = 0;
   build_matrix_type = 2;
   build_matrix_arg_index = argc;
   build_rhs_type = 2;
   build_rhs_arg_index = argc;

   /*--------------------------
    * Parse command line
    *--------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);         
      }
      else if ( strcmp(argv[arg_index], "-laplace") == 0 )
      {
         arg_index++;
         build_matrix_type = 2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         build_matrix_type = 3;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         build_matrix_type = 4;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         build_matrix_type = 6;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-vardifconv") == 0 )
      {
         arg_index++;
         build_matrix_type = 7;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rotate") == 0 )
      {
         arg_index++;
         build_matrix_type      = 8;
         build_matrix_arg_index = arg_index;
      }
     else if ( strcmp(argv[arg_index], "-125pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 5;
         build_matrix_arg_index = arg_index;
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
      else if ( strcmp(argv[arg_index], "-rhszero") == 0 )
      {
         arg_index++;
         build_rhs_type      = 4;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-cljp") == 0 )
      {
         arg_index++;
         coarsen_type      = 0;
      }
      else if ( strcmp(argv[arg_index], "-pmis") == 0 )
      {
         arg_index++;
         coarsen_type      = 8;
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
      else if ( strcmp(argv[arg_index], "-falgout") == 0 )
      {
         arg_index++;
         coarsen_type      = 6;
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
      else if ( strcmp(argv[arg_index], "-ns_down") == 0 )
      {
         arg_index++;
         ns_down = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns_up") == 0 )
      {
         arg_index++;
         ns_up = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-nodal") == 0 )
      {
         arg_index++;
         nodal  = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-mu") == 0 )
      {
         arg_index++;
         cycle_type  = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-flex") == 0 )
      {
         arg_index++;
         flex  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index++;
         max_row_sum  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-two_norm") == 0 )
      {
         arg_index++;
         two_norm  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         k_dim  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-all") == 0 )
      {
         arg_index++;
         all = 1;
      }
      else if ( strcmp(argv[arg_index], "-double") == 0 )
      {
         arg_index++;
         precision = 0;
         all = 0;
      }
      else if ( strcmp(argv[arg_index], "-single") == 0 )
      {
         arg_index++;
         precision = 1;
         all = 0;
      }
      else if ( strcmp(argv[arg_index], "-mixed") == 0 )
      {
         arg_index++;
         precision = 2;
         all = 0;
      }
      else
      {
         arg_index++;
      }
   }

   if (build_matrix_type == 2)
   {
      BuildParLaplacian_mp(argc, argv, build_matrix_arg_index, &A_flt, &A_dbl);
   }
   else if (build_matrix_type == 3)
   {
      BuildParLaplacian9pt_mp(argc, argv, build_matrix_arg_index, &A_flt, &A_dbl);
   }
   else if (build_matrix_type == 4)
   {
      BuildParLaplacian27pt_mp(argc, argv, build_matrix_arg_index, &A_flt, &A_dbl);
   }
   else if (build_matrix_type == 5)
   {
      BuildParLaplacian125pt_mp(argc, argv, build_matrix_arg_index, &A_flt, &A_dbl);
   }
   else if (build_matrix_type == 6)
   {
      BuildParDifConv_mp(argc, argv, build_matrix_arg_index, &A_flt, &A_dbl);
   }
   else if (build_matrix_type == 7)
   {
      BuildParVarDifConv_mp(argc, argv, build_matrix_arg_index, &A_flt, &A_dbl, &b_flt, &b_dbl);
      build_rhs_type = 6;
   }
   else if (build_matrix_type == 8)
   {
      BuildParRotate7pt_mp(argc, argv, build_matrix_arg_index, &A_flt, &A_dbl);
   }

   HYPRE_ParCSRMatrixGetLocalRange_flt( A_flt,
                                       &ilower, &iupper, &jlower, &jupper);

   if (build_rhs_type == 2)
   {
      double one = 1.0;
      if (myid == 0)
      {
         hypre_printf_dbl("  RHS vector has unit coefficients\n");
         hypre_printf_dbl("  Initial guess is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &ij_b_flt);
      HYPRE_IJVectorSetObjectType_flt(ij_b_flt, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(ij_b_flt);
      HYPRE_IJVectorAssemble_flt(ij_b_flt);
      HYPRE_IJVectorGetObject_flt( ij_b_flt, (void **) &b_flt );
      HYPRE_ParVectorSetConstantValues_flt(b_flt, (float)one);
      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &ij_b_dbl);
      HYPRE_IJVectorSetObjectType_dbl(ij_b_dbl, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(ij_b_dbl);
      HYPRE_IJVectorAssemble_dbl(ij_b_dbl);
      HYPRE_IJVectorGetObject_dbl( ij_b_dbl, (void **) &b_dbl );
      HYPRE_ParVectorSetConstantValues_dbl(b_dbl, one);
      /* X0 */
      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &ij_x_flt);
      HYPRE_IJVectorSetObjectType_flt(ij_x_flt, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(ij_x_flt);
      HYPRE_IJVectorAssemble_flt(ij_x_flt);
      HYPRE_IJVectorGetObject_flt( ij_x_flt, (void **) &x_flt );
      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &ij_x_dbl);
      HYPRE_IJVectorSetObjectType_dbl(ij_x_dbl, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(ij_x_dbl);
      HYPRE_IJVectorAssemble_dbl(ij_x_dbl);
      HYPRE_IJVectorGetObject_dbl( ij_x_dbl, (void **) &x_dbl );
   }
   else if (build_rhs_type == 3)
   {
      double one = 1.0;
      if (myid == 0)
      {
         hypre_printf_dbl("  RHS vector has random coefficients\n");
         hypre_printf_dbl("  Initial guess is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &ij_b_flt);
      HYPRE_IJVectorSetObjectType_flt(ij_b_flt, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(ij_b_flt);
      HYPRE_IJVectorAssemble_flt(ij_b_flt);
      HYPRE_IJVectorGetObject_flt( ij_b_flt, (void **) &b_flt );
      HYPRE_ParVectorSetRandomValues_flt(b_flt, 22775);
      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &ij_b_dbl);
      HYPRE_IJVectorSetObjectType_dbl(ij_b_dbl, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(ij_b_dbl);
      HYPRE_IJVectorAssemble_dbl(ij_b_dbl);
      HYPRE_IJVectorGetObject_dbl( ij_b_dbl, (void **) &b_dbl );
      HYPRE_ParVectorSetRandomValues_dbl(b_dbl, 22775);
      /* X0 */
      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &ij_x_flt);
      HYPRE_IJVectorSetObjectType_flt(ij_x_flt, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(ij_x_flt);
      HYPRE_IJVectorAssemble_flt(ij_x_flt);
      HYPRE_IJVectorGetObject_flt( ij_x_flt, (void **) &x_flt );
      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &ij_x_dbl);
      HYPRE_IJVectorSetObjectType_dbl(ij_x_dbl, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(ij_x_dbl);
      HYPRE_IJVectorAssemble_dbl(ij_x_dbl);
      HYPRE_IJVectorGetObject_dbl( ij_x_dbl, (void **) &x_dbl );
   }
   else if (build_rhs_type == 4)
   {
      if (myid == 0)
      {
         hypre_printf_dbl("  RHS vector has unit coefficients\n");
         hypre_printf_dbl("  Initial guess is random\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &ij_b_flt);
      HYPRE_IJVectorSetObjectType_flt(ij_b_flt, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(ij_b_flt);
      HYPRE_IJVectorAssemble_flt(ij_b_flt);
      HYPRE_IJVectorGetObject_flt( ij_b_flt, (void **) &b_flt );
      HYPRE_ParVectorSetConstantValues_flt(b_flt, (float)zero);
      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &ij_b_dbl);
      HYPRE_IJVectorSetObjectType_dbl(ij_b_dbl, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(ij_b_dbl);
      HYPRE_IJVectorAssemble_dbl(ij_b_dbl);
      HYPRE_IJVectorGetObject_dbl( ij_b_dbl, (void **) &b_dbl );
      HYPRE_ParVectorSetConstantValues_dbl(b_dbl, zero);
      /* X0 */
      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &ij_x_flt);
      HYPRE_IJVectorSetObjectType_flt(ij_x_flt, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(ij_x_flt);
      HYPRE_IJVectorAssemble_flt(ij_x_flt);
      HYPRE_IJVectorGetObject_flt( ij_x_flt, (void **) &x_flt );
      HYPRE_ParVectorSetRandomValues_flt(x_flt, 22775);
      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &ij_x_dbl);
      HYPRE_IJVectorSetObjectType_dbl(ij_x_dbl, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(ij_x_dbl);
      HYPRE_IJVectorAssemble_dbl(ij_x_dbl);
      HYPRE_IJVectorGetObject_dbl( ij_x_dbl, (void **) &x_dbl );
      HYPRE_ParVectorSetRandomValues_dbl(x_dbl, 22775);
   }

   if (solver_id == 11 || solver_id == 13 || solver_id == 15)
   {
      HYPRE_Int i;
      precision_array = (HYPRE_Precision *) hypre_CAlloc_dbl((size_t)(max_levels), (size_t)sizeof(HYPRE_Precision), HYPRE_MEMORY_HOST);
      precision_array[0] = HYPRE_REAL_DOUBLE;
      for (i=1; i < max_levels; i++)
      {
         precision_array[i] = HYPRE_REAL_SINGLE;
      }
      precision = 2;
   }
   /*! Done with linear system setup. Now proceed to solve the system. */
   // PCG solve
   if (solver_id < 2 || solver_id == 11)
   {
// Double precision
    if (precision == 0 || all) 
    {
      /* reset solution vector */
      if (build_rhs_type < 4 || build_rhs_type == 6) HYPRE_ParVectorSetConstantValues_dbl(x_dbl, zero);
      else  HYPRE_ParVectorSetRandomValues_dbl(x_dbl, 22775);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming_dbl("DBL Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      // Create PCG solver
      HYPRE_ParCSRPCGCreate_dbl(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter_dbl(pcg_solver, max_iter);
      HYPRE_PCGSetTol_dbl(pcg_solver, tol);
      HYPRE_PCGSetTwoNorm_dbl(pcg_solver, two_norm);
      HYPRE_PCGSetPrintLevel_dbl(pcg_solver, ioutdat);
      HYPRE_PCGSetFlex_dbl(pcg_solver, flex);
      HYPRE_PCGSetRecomputeResidual_dbl(pcg_solver, 1);      
      
      
      /* Now set up the AMG preconditioner and specify any parameters */
     if (solver_id == 1)
     {
        if (myid == 0) hypre_printf_dbl("\n\n***** Solver: DOUBLE PRECISION AMG-PCG *****\n");

         HYPRE_PCGSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate_dbl(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_dbl(amg_solver, poutdat); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_dbl(amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_dbl(amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_dbl(amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_dbl(amg_solver, 0.0); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_dbl(amg_solver, 1); /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold_dbl(amg_solver, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_dbl(amg_solver, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_dbl(amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_dbl(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_dbl(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_dbl(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_dbl(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_dbl(amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_dbl(amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_dbl(amg_solver, relax_wt);
         HYPRE_BoomerAMGSetOuterWt_dbl(amg_solver, outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_dbl(amg_solver, max_levels);
         HYPRE_BoomerAMGSetSmoothType_dbl(amg_solver, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumSweeps_dbl(amg_solver, smooth_num_sweeps);
         HYPRE_BoomerAMGSetSmoothNumLevels_dbl(amg_solver, smooth_num_levels);
         HYPRE_BoomerAMGSetMaxRowSum_dbl(amg_solver, max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_dbl(amg_solver, debug_flag);
         HYPRE_BoomerAMGSetNumFunctions_dbl(amg_solver, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels_dbl(amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_dbl(amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_dbl(amg_solver, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_dbl(amg_solver, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_dbl(amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_dbl(amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_dbl(amg_solver, num_paths);
         HYPRE_BoomerAMGSetNodal_dbl(amg_solver, nodal);
         HYPRE_BoomerAMGSetNodalDiag_dbl(amg_solver, nodal_diag);
         HYPRE_BoomerAMGSetKeepSameSign_dbl(amg_solver, keep_same_sign);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_dbl(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_dbl(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_dbl(amg_solver, ns_up,     2);
         }
         
         // Set the preconditioner for PCG
         HYPRE_PCGSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_dbl);
        
         HYPRE_PCGSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_dbl,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_dbl,
                                amg_solver);

         HYPRE_PCGGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRPCGGetPrecond got good precond\n");
         }
      }
      else if (solver_id ==0)
      {
        if (myid == 0) hypre_printf_dbl("\n\n***** Solver: DOUBLE PRECISION DS-PCG *****\n");
      }
      // Setup PCG solver
      HYPRE_PCGSetup_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Double precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();
      fflush(NULL);

      time_index = hypre_InitializeTiming_dbl("DBL Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      //  PCG solve
      HYPRE_PCGSolve_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Double precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();

      HYPRE_PCGGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
        hypre_printf_dbl("final relative residual norm = %e \n", dfinal_res_norm);
      	hypre_printf_dbl("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRPCGDestroy_dbl(pcg_solver);
      if(solver_id == 1) HYPRE_BoomerAMGDestroy_dbl(amg_solver);
    }
// Single precision
    if (precision == 1 || all)
    {
      /* reset solution vector */
      if (build_rhs_type < 4)  HYPRE_ParVectorSetConstantValues_flt(x_flt, zero);
      else  HYPRE_ParVectorSetRandomValues_flt(x_flt, 22775);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming_dbl("FLT Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      // Create PCG solver
      HYPRE_ParCSRPCGCreate_flt(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter_flt(pcg_solver, max_iter);
      HYPRE_PCGSetTol_flt(pcg_solver, (float)tol);
      HYPRE_PCGSetTwoNorm_flt(pcg_solver, two_norm);
      HYPRE_PCGSetPrintLevel_flt(pcg_solver, ioutdat);
      HYPRE_PCGSetFlex_flt(pcg_solver, flex);
      HYPRE_PCGSetRecomputeResidual_flt(pcg_solver, 1);      
      
      
      /* Now set up the AMG preconditioner and specify any parameters */
      if(solver_id == 1)
      {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: SINGLE PRECISION AMG-PCG *****\n");
         HYPRE_PCGSetMaxIter_flt(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate_flt(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_flt(amg_solver, poutdat); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_flt(amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_flt(amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_flt(amg_solver, (float)zero); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_flt(amg_solver, 1); /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold_flt(amg_solver, (float)strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_flt(amg_solver, (float)trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_flt(amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_flt(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_flt(amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_flt(amg_solver, (float)relax_wt);
         HYPRE_BoomerAMGSetOuterWt_flt(amg_solver, (float)outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_flt(amg_solver, max_levels);
         HYPRE_BoomerAMGSetSmoothType_flt(amg_solver, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumSweeps_flt(amg_solver, smooth_num_sweeps);
         HYPRE_BoomerAMGSetSmoothNumLevels_flt(amg_solver, smooth_num_levels);
         HYPRE_BoomerAMGSetMaxRowSum_flt(amg_solver, (float)max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_flt(amg_solver, debug_flag);
         HYPRE_BoomerAMGSetNumFunctions_flt(amg_solver, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels_flt(amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_flt(amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_flt(amg_solver, (float)agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_flt(amg_solver, (float)agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_flt(amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_flt(amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_flt(amg_solver, num_paths);
         HYPRE_BoomerAMGSetNodal_flt(amg_solver, nodal);
         HYPRE_BoomerAMGSetNodalDiag_flt(amg_solver, nodal_diag);
         HYPRE_BoomerAMGSetKeepSameSign_flt(amg_solver, keep_same_sign);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_up,     2);
         }
      
         // Set the preconditioner for PCG
         HYPRE_PCGSetPrecondMatrix_flt(pcg_solver, (HYPRE_Matrix)A_flt);
        
         HYPRE_PCGSetPrecond_flt(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_flt,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_flt,
                                amg_solver);

         HYPRE_PCGGetPrecond_flt(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRPCGGetPrecond got good precond\n");
         }
      }
      else if (solver_id == 0)
      {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: SINGLE PRECISION DS-PCG *****\n");
      }
      // Setup PCG solver
      HYPRE_PCGSetup_flt(pcg_solver, (HYPRE_Matrix)A_flt, (HYPRE_Vector)b_flt, (HYPRE_Vector)x_flt);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Single precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();
      fflush(NULL);

      time_index = hypre_InitializeTiming_dbl("FLT Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      //  PCG solve
      HYPRE_PCGSolve_flt(pcg_solver, (HYPRE_Matrix)A_flt,  (HYPRE_Vector)b_flt, (HYPRE_Vector)x_flt);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Single precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();

      HYPRE_PCGGetNumIterations_flt(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm_flt(pcg_solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf_dbl("final relative residual norm = %e \n", final_res_norm);
         hypre_printf_dbl("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRPCGDestroy_flt(pcg_solver);
      if(solver_id == 1) HYPRE_BoomerAMGDestroy_flt(amg_solver);
    }
// mixed-precision
    if (precision == 2 || all)
    {
      /* reset solution vector */
      if (build_rhs_type < 4)  HYPRE_ParVectorSetConstantValues_dbl(x_dbl, zero);
      else  HYPRE_ParVectorSetRandomValues_dbl(x_dbl, 22775);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming_dbl("DBL Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      // Create PCG solver
      HYPRE_ParCSRPCGCreate_dbl(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter_dbl(pcg_solver, max_iter);
      HYPRE_PCGSetTol_dbl(pcg_solver, tol);
      HYPRE_PCGSetTwoNorm_dbl(pcg_solver, two_norm);
      HYPRE_PCGSetPrintLevel_dbl(pcg_solver, ioutdat);
      HYPRE_PCGSetFlex_dbl(pcg_solver, flex);
      HYPRE_PCGSetRecomputeResidual_dbl(pcg_solver, 1);      
      
      
      /* Now set up the AMG preconditioner and specify any parameters */
     if (solver_id == 1)
     {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: MIXED PRECISION AMG-PCG *****\n");
         HYPRE_PCGSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate_flt(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_flt(amg_solver, poutdat); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_flt(amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_flt(amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_flt(amg_solver, (float)zero); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_flt(amg_solver, 1); /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold_flt(amg_solver, (float)strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_flt(amg_solver, (float)trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_flt(amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_flt(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_flt(amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_flt(amg_solver, (float)relax_wt);
         HYPRE_BoomerAMGSetOuterWt_flt(amg_solver, (float)outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_flt(amg_solver, max_levels);
         HYPRE_BoomerAMGSetSmoothType_flt(amg_solver, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumSweeps_flt(amg_solver, smooth_num_sweeps);
         HYPRE_BoomerAMGSetSmoothNumLevels_flt(amg_solver, smooth_num_levels);
         HYPRE_BoomerAMGSetMaxRowSum_flt(amg_solver, (float)max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_flt(amg_solver, debug_flag);
         HYPRE_BoomerAMGSetNumFunctions_flt(amg_solver, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels_flt(amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_flt(amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_flt(amg_solver, (float)agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_flt(amg_solver, (float)agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_flt(amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_flt(amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_flt(amg_solver, num_paths);
         HYPRE_BoomerAMGSetNodal_flt(amg_solver, nodal);
         HYPRE_BoomerAMGSetNodalDiag_flt(amg_solver, nodal_diag);
         HYPRE_BoomerAMGSetKeepSameSign_flt(amg_solver, keep_same_sign);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_up,     2);
         }
      
         // Set the preconditioner for PCG (single precision matrix)
         HYPRE_PCGSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_flt);
         // Set the preconditioner for PCG.
         // This actually sets a pointer to a single precision AMG solver.
         // The setup and solve functions just allow us to accept double precision
         // rhs and sol vectors from the PCG solver to do the preconditioner solve.        
         HYPRE_PCGSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_mp,
                                amg_solver);

         HYPRE_PCGGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRPCGGetPrecond got good precond\n");
         }
      }
      /* Now set up the MPAMG preconditioner and specify any parameters */
      else if (solver_id == 11)
      {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: MIXED PRECISION AMG-PCG *****\n");
         HYPRE_PCGSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_MPAMGCreate_mp(&amg_solver);
         HYPRE_MPAMGSetPrintLevel_mp(amg_solver, poutdat); 
         HYPRE_MPAMGSetCoarsenType_mp(amg_solver, coarsen_type);
         HYPRE_MPAMGSetInterpType_mp(amg_solver, interp_type);
         HYPRE_MPAMGSetNumSweeps_mp(amg_solver, num_sweeps);
         HYPRE_MPAMGSetTol_mp(amg_solver, zero); 
         HYPRE_MPAMGSetMaxIter_mp(amg_solver, 1); 
         HYPRE_MPAMGSetStrongThreshold_mp(amg_solver, strong_threshold);
         HYPRE_MPAMGSetTruncFactor_mp(amg_solver, trunc_factor);
         HYPRE_MPAMGSetPMaxElmts_mp(amg_solver, P_max_elmts);
         HYPRE_MPAMGSetNumSweeps_mp(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_MPAMGSetRelaxType_mp(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_MPAMGSetCycleRelaxType_mp(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_MPAMGSetCycleRelaxType_mp(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_MPAMGSetCycleRelaxType_mp(amg_solver, relax_coarse, 3);
         }
         HYPRE_MPAMGSetRelaxOrder_mp(amg_solver, relax_order);
         HYPRE_MPAMGSetRelaxWt_mp(amg_solver, relax_wt);
         HYPRE_MPAMGSetOuterWt_mp(amg_solver, outer_wt);
         HYPRE_MPAMGSetMaxLevels_mp(amg_solver, max_levels);
         HYPRE_MPAMGSetMaxRowSum_mp(amg_solver, max_row_sum);
         HYPRE_MPAMGSetDebugFlag_mp(amg_solver, debug_flag);
         HYPRE_MPAMGSetNumFunctions_mp(amg_solver, num_functions);
         HYPRE_MPAMGSetAggNumLevels_mp(amg_solver, agg_num_levels);
         HYPRE_MPAMGSetAggInterpType_mp(amg_solver, agg_interp_type);
         HYPRE_MPAMGSetAggTruncFactor_mp(amg_solver, agg_trunc_factor);
         HYPRE_MPAMGSetAggP12TruncFactor_mp(amg_solver, agg_P12_trunc_factor);
         HYPRE_MPAMGSetAggPMaxElmts_mp(amg_solver, agg_P_max_elmts);
         HYPRE_MPAMGSetAggP12MaxElmts_mp(amg_solver, agg_P12_max_elmts);
         HYPRE_MPAMGSetNumPaths_mp(amg_solver, num_paths);
         HYPRE_MPAMGSetNodal_mp(amg_solver, nodal);
         HYPRE_MPAMGSetNodalDiag_mp(amg_solver, nodal_diag);
         HYPRE_MPAMGSetKeepSameSign_mp(amg_solver, keep_same_sign);
         HYPRE_MPAMGSetPrecisionArray_mp(amg_solver, precision_array);
         if (ns_coarse > -1)
         {
            HYPRE_MPAMGSetCycleNumSweeps_mp(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_MPAMGSetCycleNumSweeps_mp(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_MPAMGSetCycleNumSweeps_mp(amg_solver, ns_up,     2);
         }
      
         // Set the preconditioner for PCG (single precision matrix)
         HYPRE_PCGSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_dbl);
         // Set the preconditioner for PCG.
         // This actually sets a pointer to a single precision AMG solver.
         // The setup and solve functions just allow us to accept double precision
         // rhs and sol vectors from the PCG solver to do the preconditioner solve.        
         HYPRE_PCGSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSetup_mp,
                                amg_solver);

         HYPRE_PCGGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRPCGGetPrecond got good precond\n");
         }
      }
      else if (solver_id == 0)
      {
        if (myid == 0) hypre_printf_dbl("\n\n***** Solver: MIXED PRECISION DS-PCG *****\n");
      }
      // Setup PCG solver (double precision)
      HYPRE_PCGSetup_dbl(pcg_solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Mixed precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();
      fflush(NULL);

      time_index = hypre_InitializeTiming_dbl("DBL Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      //  PCG solve (double precision)
      HYPRE_PCGSolve_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Mixed precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();

      HYPRE_PCGGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
        hypre_printf_dbl("final relative residual norm = %e \n", dfinal_res_norm);
        hypre_printf_dbl("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRPCGDestroy_dbl(pcg_solver);
      if(solver_id == 1) HYPRE_BoomerAMGDestroy_flt(amg_solver);
      if(solver_id == 11) HYPRE_MPAMGDestroy_mp(amg_solver);
    } //end PCG   
   }   
   else if (solver_id < 4 || solver_id == 13)  //GMRES
   {
// double-precision
    if (precision == 0 || all)
    {
      /* reset solution vector */
      if (build_rhs_type < 4 || build_rhs_type == 6) HYPRE_ParVectorSetConstantValues_dbl(x_dbl, zero);
      else  HYPRE_ParVectorSetRandomValues_dbl(x_dbl, 22775);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming_dbl("DBL Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      // Create GMRES solver
      HYPRE_ParCSRGMRESCreate_dbl(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_GMRESSetKDim_dbl(pcg_solver, k_dim);
      HYPRE_GMRESSetMaxIter_dbl(pcg_solver, max_iter);
      HYPRE_GMRESSetTol_dbl(pcg_solver, tol);
      HYPRE_GMRESSetPrintLevel_dbl(pcg_solver, ioutdat);
//      HYPRE_PCGSetRecomputeResidual_dbl(pcg_solver, recompute_res);      
      
      
      /* Now set up the AMG preconditioner and specify any parameters */
     if (solver_id == 3)
     {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: DOUBLE PRECISION AMG-GMRES *****\n");

         HYPRE_GMRESSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate_dbl(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_dbl(amg_solver, poutdat); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_dbl(amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_dbl(amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_dbl(amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_dbl(amg_solver, 0.0); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_dbl(amg_solver, 1); /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold_dbl(amg_solver, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_dbl(amg_solver, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_dbl(amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_dbl(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_dbl(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_dbl(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_dbl(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_dbl(amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_dbl(amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_dbl(amg_solver, relax_wt);
         HYPRE_BoomerAMGSetOuterWt_dbl(amg_solver, outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_dbl(amg_solver, max_levels);
         HYPRE_BoomerAMGSetSmoothType_dbl(amg_solver, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumSweeps_dbl(amg_solver, smooth_num_sweeps);
         HYPRE_BoomerAMGSetSmoothNumLevels_dbl(amg_solver, smooth_num_levels);
         HYPRE_BoomerAMGSetMaxRowSum_dbl(amg_solver, max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_dbl(amg_solver, debug_flag);
         HYPRE_BoomerAMGSetNumFunctions_dbl(amg_solver, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels_dbl(amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_dbl(amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_dbl(amg_solver, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_dbl(amg_solver, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_dbl(amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_dbl(amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_dbl(amg_solver, num_paths);
         HYPRE_BoomerAMGSetNodal_dbl(amg_solver, nodal);
         HYPRE_BoomerAMGSetNodalDiag_dbl(amg_solver, nodal_diag);
         HYPRE_BoomerAMGSetKeepSameSign_dbl(amg_solver, keep_same_sign);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_dbl(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_dbl(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_dbl(amg_solver, ns_up,     2);
         }
         
         // Set the preconditioner for GMRES
         HYPRE_GMRESSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_dbl);
        
         HYPRE_GMRESSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_dbl,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_dbl,
                                amg_solver);

         HYPRE_GMRESGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRGMRESGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRGMRESGetPrecond got good precond\n");
         }
      }
      else if (solver_id ==2)
      {
        if (myid == 0) hypre_printf_dbl("\n\n***** Solver: DOUBLE PRECISION DS-GMRES *****\n");
      }
      // Setup GMRES solver
      HYPRE_GMRESSetup_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Double precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();
      fflush(NULL);

      time_index = hypre_InitializeTiming_dbl("DBL Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      //  GMRES solve
      HYPRE_GMRESSolve_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Double precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();

      HYPRE_GMRESGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
        hypre_printf_dbl("final relative residual norm = %e \n", dfinal_res_norm);
      	hypre_printf_dbl("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRGMRESDestroy_dbl(pcg_solver);
      if(solver_id == 3) HYPRE_BoomerAMGDestroy_dbl(amg_solver);
    }
// Single precision
    if (precision == 1 || all)
    {
      /* reset solution vector */
      if (build_rhs_type < 4)  HYPRE_ParVectorSetConstantValues_flt(x_flt, zero);
      else  HYPRE_ParVectorSetRandomValues_flt(x_flt, 22775);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming_dbl("FLT Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      // Create GMRES solver
      HYPRE_ParCSRGMRESCreate_flt(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_GMRESSetMaxIter_flt(pcg_solver, max_iter);
      HYPRE_GMRESSetTol_flt(pcg_solver, (float)tol);
      HYPRE_GMRESSetKDim_flt(pcg_solver, k_dim);
      HYPRE_GMRESSetPrintLevel_flt(pcg_solver, ioutdat);
      
      /* Now set up the AMG preconditioner and specify any parameters */
      if(solver_id == 3)
      {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: SINGLE PRECISION AMG-GMRES *****\n");
         HYPRE_GMRESSetMaxIter_flt(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate_flt(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_flt(amg_solver, poutdat); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_flt(amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_flt(amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_flt(amg_solver, (float)zero); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_flt(amg_solver, 1); /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold_flt(amg_solver, (float)strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_flt(amg_solver, (float)trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_flt(amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_flt(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_flt(amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_flt(amg_solver, (float)relax_wt);
         HYPRE_BoomerAMGSetOuterWt_flt(amg_solver, (float)outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_flt(amg_solver, max_levels);
         HYPRE_BoomerAMGSetSmoothType_flt(amg_solver, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumSweeps_flt(amg_solver, smooth_num_sweeps);
         HYPRE_BoomerAMGSetSmoothNumLevels_flt(amg_solver, smooth_num_levels);
         HYPRE_BoomerAMGSetMaxRowSum_flt(amg_solver, (float)max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_flt(amg_solver, debug_flag);
         HYPRE_BoomerAMGSetNumFunctions_flt(amg_solver, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels_flt(amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_flt(amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_flt(amg_solver, (float)agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_flt(amg_solver, (float)agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_flt(amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_flt(amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_flt(amg_solver, num_paths);
         HYPRE_BoomerAMGSetNodal_flt(amg_solver, nodal);
         HYPRE_BoomerAMGSetNodalDiag_flt(amg_solver, nodal_diag);
         HYPRE_BoomerAMGSetKeepSameSign_flt(amg_solver, keep_same_sign);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_up,     2);
         }
      
         // Set the preconditioner for GMRES
         HYPRE_GMRESSetPrecondMatrix_flt(pcg_solver, (HYPRE_Matrix)A_flt);
        
         HYPRE_GMRESSetPrecond_flt(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_flt,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_flt,
                                amg_solver);

         HYPRE_GMRESGetPrecond_flt(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRGMRESGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRGMRESGetPrecond got good precond\n");
         }
      }
      else if (solver_id == 2)
      {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: SINGLE PRECISION DS-GMRES *****\n");
      }
      // Setup GMRES solver
      HYPRE_GMRESSetup_flt(pcg_solver, (HYPRE_Matrix)A_flt, (HYPRE_Vector)b_flt, (HYPRE_Vector)x_flt);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Single precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();
      fflush(NULL);

      time_index = hypre_InitializeTiming_dbl("FLT Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      //  GMRES solve
      HYPRE_GMRESSolve_flt(pcg_solver, (HYPRE_Matrix)A_flt,  (HYPRE_Vector)b_flt, (HYPRE_Vector)x_flt);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Single precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();

      HYPRE_GMRESGetNumIterations_flt(pcg_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm_flt(pcg_solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf_dbl("final relative residual norm = %e \n", final_res_norm);
         hypre_printf_dbl("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRGMRESDestroy_flt(pcg_solver);
      if(solver_id == 3) HYPRE_BoomerAMGDestroy_flt(amg_solver);
    }
// mixed-precision
    {
      /* reset solution vector */
      if (build_rhs_type < 4)  HYPRE_ParVectorSetConstantValues_dbl(x_dbl, zero);
      else  HYPRE_ParVectorSetRandomValues_dbl(x_dbl, 22775);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming_dbl("DBL Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      // Create GMRES solver
      HYPRE_ParCSRGMRESCreate_dbl(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_GMRESSetMaxIter_dbl(pcg_solver, max_iter);
      HYPRE_GMRESSetTol_dbl(pcg_solver, tol);
      HYPRE_GMRESSetKDim_dbl(pcg_solver, k_dim);
      HYPRE_GMRESSetPrintLevel_dbl(pcg_solver, ioutdat);
      
      
      /* Now set up the AMG preconditioner and specify any parameters */
     if(solver_id == 3)
     {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: MIXED PRECISION AMG-GMRES *****\n");
         HYPRE_GMRESSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate_flt(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_flt(amg_solver, poutdat); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_flt(amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_flt(amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_flt(amg_solver, (float)zero); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_flt(amg_solver, 1); /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold_flt(amg_solver, (float)strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_flt(amg_solver, (float)trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_flt(amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_flt(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_flt(amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_flt(amg_solver, (float)relax_wt);
         HYPRE_BoomerAMGSetOuterWt_flt(amg_solver, (float)outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_flt(amg_solver, max_levels);
         HYPRE_BoomerAMGSetSmoothType_flt(amg_solver, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumSweeps_flt(amg_solver, smooth_num_sweeps);
         HYPRE_BoomerAMGSetSmoothNumLevels_flt(amg_solver, smooth_num_levels);
         HYPRE_BoomerAMGSetMaxRowSum_flt(amg_solver, (float)max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_flt(amg_solver, debug_flag);
         HYPRE_BoomerAMGSetNumFunctions_flt(amg_solver, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels_flt(amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_flt(amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_flt(amg_solver, (float)agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_flt(amg_solver, (float)agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_flt(amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_flt(amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_flt(amg_solver, num_paths);
         HYPRE_BoomerAMGSetNodal_flt(amg_solver, nodal);
         HYPRE_BoomerAMGSetNodalDiag_flt(amg_solver, nodal_diag);
         HYPRE_BoomerAMGSetKeepSameSign_flt(amg_solver, keep_same_sign);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_up,     2);
         }
      
         // Set the preconditioner for GMRES (single precision matrix)
         HYPRE_GMRESSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_flt);
         // Set the preconditioner for GMRES.
         // This actually sets a pointer to a single precision AMG solver.
         // The setup and solve functions just allow us to accept double precision
         // rhs and sol vectors from the GMRES solver to do the preconditioner solve.        
         HYPRE_GMRESSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_mp,
                                amg_solver);

         HYPRE_GMRESGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRGMRESGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRGMRESGetPrecond got good precond\n");
         }
      }
      /* Now set up the MPAMG preconditioner and specify any parameters */
      else if (solver_id == 13)
      {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: MIXED PRECISION AMG-PCG *****\n");
         HYPRE_PCGSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_MPAMGCreate_mp(&amg_solver);
         /*HYPRE_MPAMGSetPrintLevel_mp(amg_solver, poutdat); 
         HYPRE_MPAMGSetCoarsenType_mp(amg_solver, coarsen_type);
         HYPRE_MPAMGSetInterpType_mp(amg_solver, interp_type);
         HYPRE_MPAMGSetNumSweeps_mp(amg_solver, num_sweeps);
         HYPRE_MPAMGSetTol_mp(amg_solver, zero); 
         HYPRE_MPAMGSetMaxIter_mp(amg_solver, 1); 
         HYPRE_MPAMGSetStrongThreshold_mp(amg_solver, strong_threshold);
         HYPRE_MPAMGSetTruncFactor_mp(amg_solver, trunc_factor);
         HYPRE_MPAMGSetPMaxElmts_mp(amg_solver, P_max_elmts);
         HYPRE_MPAMGSetNumSweeps_mp(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_MPAMGSetRelaxType_mp(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_MPAMGSetCycleRelaxType_mp(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_MPAMGSetCycleRelaxType_mp(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_MPAMGSetCycleRelaxType_mp(amg_solver, relax_coarse, 3);
         }
         HYPRE_MPAMGSetRelaxOrder_mp(amg_solver, relax_order);
         HYPRE_MPAMGSetRelaxWt_mp(amg_solver, relax_wt);
         HYPRE_MPAMGSetOuterWt_mp(amg_solver, outer_wt);
         HYPRE_MPAMGSetMaxLevels_mp(amg_solver, max_levels);
         HYPRE_MPAMGSetMaxRowSum_mp(amg_solver, max_row_sum);
         HYPRE_MPAMGSetDebugFlag_mp(amg_solver, debug_flag);
         HYPRE_MPAMGSetNumFunctions_mp(amg_solver, num_functions);
         HYPRE_MPAMGSetAggNumLevels_mp(amg_solver, agg_num_levels);
         HYPRE_MPAMGSetAggInterpType_mp(amg_solver, agg_interp_type);
         HYPRE_MPAMGSetAggTruncFactor_mp(amg_solver, agg_trunc_factor);
         HYPRE_MPAMGSetAggP12TruncFactor_mp(amg_solver, agg_P12_trunc_factor);
         HYPRE_MPAMGSetAggPMaxElmts_mp(amg_solver, agg_P_max_elmts);
         HYPRE_MPAMGSetAggP12MaxElmts_mp(amg_solver, agg_P12_max_elmts);
         HYPRE_MPAMGSetNumPaths_mp(amg_solver, num_paths);
         HYPRE_MPAMGSetNodal_mp(amg_solver, nodal);
         HYPRE_MPAMGSetNodalDiag_mp(amg_solver, nodal_diag);
         HYPRE_MPAMGSetKeepSameSign_mp(amg_solver, keep_same_sign);
         HYPRE_MPAMGSetPrecisionArray_mp(amg_solver, precision_array);
         if (ns_coarse > -1)
         {
            HYPRE_MPAMGSetCycleNumSweeps_mp(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_MPAMGSetCycleNumSweeps_mp(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_MPAMGSetCycleNumSweeps_mp(amg_solver, ns_up,     2);
         }*/
         // Set the preconditioner for GMRES (single precision matrix)
         HYPRE_GMRESSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_dbl);
         // Set the preconditioner for GMRES.
         // This actually sets a pointer to a single precision AMG solver.
         // The setup and solve functions just allow us to accept double precision
         // rhs and sol vectors from the GMRES solver to do the preconditioner solve.        
         HYPRE_GMRESSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSetup_mp,
                                amg_solver);

         HYPRE_GMRESGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRGMRESGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRGMRESGetPrecond got good precond\n");
         }
      }
      
      else if (solver_id == 2)
      {
        if (myid == 0) hypre_printf_dbl("\n\n***** Solver: MIXED PRECISION DS-GMRES *****\n");
      }
      // Setup GMRES solver (double precision)
      HYPRE_GMRESSetup_dbl(pcg_solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Mixed precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();
      fflush(NULL);

      time_index = hypre_InitializeTiming_dbl("DBL Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      //  GMRES solve (double precision)
      HYPRE_GMRESSolve_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Mixed precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();

      HYPRE_GMRESGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
        hypre_printf_dbl("final relative residual norm = %e \n", dfinal_res_norm);
        hypre_printf_dbl("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy gmres solver
      HYPRE_ParCSRGMRESDestroy_dbl(pcg_solver);
      if(solver_id == 3) HYPRE_BoomerAMGDestroy_flt(amg_solver);
      if(solver_id == 13) HYPRE_MPAMGDestroy_mp(amg_solver);
    } // end GMRES   
   }   
   // BiCGSTAB solve
   else if (solver_id < 6 || solver_id == 15)
   {
// Double precision
    {
      /* reset solution vector */
      if (build_rhs_type < 4 || build_rhs_type == 6) HYPRE_ParVectorSetConstantValues_dbl(x_dbl, zero);
      else  HYPRE_ParVectorSetRandomValues_dbl(x_dbl, 22775);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming_dbl("DBL Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      // Create BiCGSTAB solver
      HYPRE_ParCSRBiCGSTABCreate_dbl(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_BiCGSTABSetMaxIter_dbl(pcg_solver, max_iter);
      HYPRE_BiCGSTABSetTol_dbl(pcg_solver, tol);
      HYPRE_BiCGSTABSetPrintLevel_dbl(pcg_solver, ioutdat);
      
      
      /* Now set up the AMG preconditioner and specify any parameters */
     if (solver_id == 5)
     {
        if (myid == 0) hypre_printf_dbl("\n\n***** Solver: DOUBLE PRECISION AMG-BiCGSTAB *****\n");

         HYPRE_BiCGSTABSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate_dbl(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_dbl(amg_solver, poutdat); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_dbl(amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_dbl(amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_dbl(amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_dbl(amg_solver, 0.0); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_dbl(amg_solver, 1); /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold_dbl(amg_solver, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_dbl(amg_solver, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_dbl(amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_dbl(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_dbl(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_dbl(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_dbl(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_dbl(amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_dbl(amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_dbl(amg_solver, relax_wt);
         HYPRE_BoomerAMGSetOuterWt_dbl(amg_solver, outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_dbl(amg_solver, max_levels);
         HYPRE_BoomerAMGSetSmoothType_dbl(amg_solver, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumSweeps_dbl(amg_solver, smooth_num_sweeps);
         HYPRE_BoomerAMGSetSmoothNumLevels_dbl(amg_solver, smooth_num_levels);
         HYPRE_BoomerAMGSetMaxRowSum_dbl(amg_solver, max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_dbl(amg_solver, debug_flag);
         HYPRE_BoomerAMGSetNumFunctions_dbl(amg_solver, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels_dbl(amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_dbl(amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_dbl(amg_solver, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_dbl(amg_solver, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_dbl(amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_dbl(amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_dbl(amg_solver, num_paths);
         HYPRE_BoomerAMGSetNodal_dbl(amg_solver, nodal);
         HYPRE_BoomerAMGSetNodalDiag_dbl(amg_solver, nodal_diag);
         HYPRE_BoomerAMGSetKeepSameSign_dbl(amg_solver, keep_same_sign);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_dbl(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_dbl(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_dbl(amg_solver, ns_up,     2);
         }
         
         // Set the preconditioner for BiCGSTAB
         HYPRE_BiCGSTABSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_dbl);
        
         HYPRE_BiCGSTABSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_dbl,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_dbl,
                                amg_solver);

         HYPRE_BiCGSTABGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRBiCGSTABGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRBiCGSTABGetPrecond got good precond\n");
         }
      }
      else if (solver_id == 4)
      {
        if (myid == 0) hypre_printf_dbl("\n\n***** Solver: DOUBLE PRECISION DS-BiCGSTAB *****\n");
      }
      // Setup BiCGSTAB solver
      HYPRE_BiCGSTABSetup_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Double precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();
      fflush(NULL);

      time_index = hypre_InitializeTiming_dbl("DBL Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      //  BiCGSTAB solve
      HYPRE_BiCGSTABSolve_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Double precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();

      HYPRE_BiCGSTABGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
        hypre_printf_dbl("final relative residual norm = %e \n", dfinal_res_norm);
      	hypre_printf_dbl("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRBiCGSTABDestroy_dbl(pcg_solver);
      if (solver_id == 5) HYPRE_BoomerAMGDestroy_dbl(amg_solver);
    }
// Single precision
    if (precision == 1 || all)
    {
      /* reset solution vector */
      if (build_rhs_type < 4)  HYPRE_ParVectorSetConstantValues_flt(x_flt, zero);
      else  HYPRE_ParVectorSetRandomValues_flt(x_flt, 22775);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming_dbl("FLT Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      // Create BiCGSTAB solver
      HYPRE_ParCSRBiCGSTABCreate_flt(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_BiCGSTABSetMaxIter_flt(pcg_solver, max_iter);
      HYPRE_BiCGSTABSetTol_flt(pcg_solver, (float)tol);
      HYPRE_BiCGSTABSetPrintLevel_flt(pcg_solver, ioutdat);
//      HYPRE_BiCGSTABSetRecomputeResidual_flt(pcg_solver, recompute_res);      
      
      
      /* Now set up the AMG preconditioner and specify any parameters */
      if (solver_id == 5)
      {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: SINGLE PRECISION AMG-BiCGSTAB *****\n");
         HYPRE_BiCGSTABSetMaxIter_flt(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate_flt(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_flt(amg_solver, poutdat); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_flt(amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_flt(amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_flt(amg_solver, (float)zero); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_flt(amg_solver, 1); /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold_flt(amg_solver, (float)strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_flt(amg_solver, (float)trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_flt(amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_flt(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_flt(amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_flt(amg_solver, (float)relax_wt);
         HYPRE_BoomerAMGSetOuterWt_flt(amg_solver, (float)outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_flt(amg_solver, max_levels);
         HYPRE_BoomerAMGSetSmoothType_flt(amg_solver, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumSweeps_flt(amg_solver, smooth_num_sweeps);
         HYPRE_BoomerAMGSetSmoothNumLevels_flt(amg_solver, smooth_num_levels);
         HYPRE_BoomerAMGSetMaxRowSum_flt(amg_solver, (float)max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_flt(amg_solver, debug_flag);
         HYPRE_BoomerAMGSetNumFunctions_flt(amg_solver, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels_flt(amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_flt(amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_flt(amg_solver, (float)agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_flt(amg_solver, (float)agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_flt(amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_flt(amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_flt(amg_solver, num_paths);
         HYPRE_BoomerAMGSetNodal_flt(amg_solver, nodal);
         HYPRE_BoomerAMGSetNodalDiag_flt(amg_solver, nodal_diag);
         HYPRE_BoomerAMGSetKeepSameSign_flt(amg_solver, keep_same_sign);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_up,     2);
         }
      
         // Set the preconditioner for BiCGSTAB
         HYPRE_BiCGSTABSetPrecondMatrix_flt(pcg_solver, (HYPRE_Matrix)A_flt);
        
         HYPRE_BiCGSTABSetPrecond_flt(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_flt,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_flt,
                                amg_solver);

         HYPRE_BiCGSTABGetPrecond_flt(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRBiCGSTABGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRBiCGSTABGetPrecond got good precond\n");
         }
      }
      else if (solver_id == 4)
      {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: SINGLE PRECISION DS-BiCGSTAB *****\n");
      }
      // Setup BiCGSTAB solver
      HYPRE_BiCGSTABSetup_flt(pcg_solver, (HYPRE_Matrix)A_flt, (HYPRE_Vector)b_flt, (HYPRE_Vector)x_flt);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Single precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();
      fflush(NULL);

      time_index = hypre_InitializeTiming_dbl("FLT Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      //  BiCGSTAB solve
      HYPRE_BiCGSTABSolve_flt(pcg_solver, (HYPRE_Matrix)A_flt,  (HYPRE_Vector)b_flt, (HYPRE_Vector)x_flt);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Single precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();

      HYPRE_BiCGSTABGetNumIterations_flt(pcg_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm_flt(pcg_solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf_dbl("final relative residual norm = %e \n", final_res_norm);
         hypre_printf_dbl("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRBiCGSTABDestroy_flt(pcg_solver);
      if(solver_id == 5) HYPRE_BoomerAMGDestroy_flt(amg_solver);
    }
// mixed-precision
    if (precision == 2 || all)
    {
      /* reset solution vector */
      if (build_rhs_type < 4)  HYPRE_ParVectorSetConstantValues_dbl(x_dbl, zero);
      else  HYPRE_ParVectorSetRandomValues_dbl(x_dbl, 22775);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming_dbl("DBL Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      // Create BiCGSTAB solver
      HYPRE_ParCSRBiCGSTABCreate_dbl(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_BiCGSTABSetMaxIter_dbl(pcg_solver, max_iter);
      HYPRE_BiCGSTABSetTol_dbl(pcg_solver, tol);
      HYPRE_BiCGSTABSetPrintLevel_dbl(pcg_solver, ioutdat);
//      HYPRE_BiCGSTABSetRecomputeResidual_dbl(pcg_solver, recompute_res);      
      
      
      /* Now set up the AMG preconditioner and specify any parameters */
     if (solver_id == 5)
     {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: MIXED PRECISION AMG-BiCGSTAB *****\n");
         HYPRE_BiCGSTABSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate_flt(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_flt(amg_solver, poutdat); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_flt(amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_flt(amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_flt(amg_solver, (float)zero); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_flt(amg_solver, 1); /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold_flt(amg_solver, (float)strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_flt(amg_solver, (float)trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_flt(amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_flt(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_flt(amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_flt(amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_flt(amg_solver, (float)relax_wt);
         HYPRE_BoomerAMGSetOuterWt_flt(amg_solver, (float)outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_flt(amg_solver, max_levels);
         HYPRE_BoomerAMGSetSmoothType_flt(amg_solver, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumSweeps_flt(amg_solver, smooth_num_sweeps);
         HYPRE_BoomerAMGSetSmoothNumLevels_flt(amg_solver, smooth_num_levels);
         HYPRE_BoomerAMGSetMaxRowSum_flt(amg_solver, (float)max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_flt(amg_solver, debug_flag);
         HYPRE_BoomerAMGSetNumFunctions_flt(amg_solver, num_functions);
         HYPRE_BoomerAMGSetAggNumLevels_flt(amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_flt(amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_flt(amg_solver, (float)agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_flt(amg_solver, (float)agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_flt(amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_flt(amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_flt(amg_solver, num_paths);
         HYPRE_BoomerAMGSetNodal_flt(amg_solver, nodal);
         HYPRE_BoomerAMGSetNodalDiag_flt(amg_solver, nodal_diag);
         HYPRE_BoomerAMGSetKeepSameSign_flt(amg_solver, keep_same_sign);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_flt(amg_solver, ns_up,     2);
         }
      
         // Set the preconditioner for BiCGSTAB (single precision matrix)
         HYPRE_BiCGSTABSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_flt);
         // Set the preconditioner for BiCGSTAB.
         // This actually sets a pointer to a single precision AMG solver.
         // The setup and solve functions just allow us to accept double precision
         // rhs and sol vectors from the BiCGSTAB solver to do the preconditioner solve.        
         HYPRE_BiCGSTABSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_mp,
                                amg_solver);

         HYPRE_BiCGSTABGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRBiCGSTABGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRBiCGSTABGetPrecond got good precond\n");
         }
      }
      /* Now set up the MPAMG preconditioner and specify any parameters */
      else if (solver_id == 15)
      {
         if (myid == 0) hypre_printf_dbl("\n\n***** Solver: MIXED PRECISION AMG-PCG *****\n");
         HYPRE_PCGSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_MPAMGCreate_mp(&amg_solver);
         /*HYPRE_MPAMGSetPrintLevel_mp(amg_solver, poutdat); 
         HYPRE_MPAMGSetCoarsenType_mp(amg_solver, coarsen_type);
         HYPRE_MPAMGSetInterpType_mp(amg_solver, interp_type);
         HYPRE_MPAMGSetNumSweeps_mp(amg_solver, num_sweeps);
         HYPRE_MPAMGSetTol_mp(amg_solver, zero); 
         HYPRE_MPAMGSetMaxIter_mp(amg_solver, 1); 
         HYPRE_MPAMGSetStrongThreshold_mp(amg_solver, strong_threshold);
         HYPRE_MPAMGSetTruncFactor_mp(amg_solver, trunc_factor);
         HYPRE_MPAMGSetPMaxElmts_mp(amg_solver, P_max_elmts);
         HYPRE_MPAMGSetNumSweeps_mp(amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_MPAMGSetRelaxType_mp(amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_MPAMGSetCycleRelaxType_mp(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_MPAMGSetCycleRelaxType_mp(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_MPAMGSetCycleRelaxType_mp(amg_solver, relax_coarse, 3);
         }
         HYPRE_MPAMGSetRelaxOrder_mp(amg_solver, relax_order);
         HYPRE_MPAMGSetRelaxWt_mp(amg_solver, relax_wt);
         HYPRE_MPAMGSetOuterWt_mp(amg_solver, outer_wt);
         HYPRE_MPAMGSetMaxLevels_mp(amg_solver, max_levels);
         HYPRE_MPAMGSetMaxRowSum_mp(amg_solver, max_row_sum);
         HYPRE_MPAMGSetDebugFlag_mp(amg_solver, debug_flag);
         HYPRE_MPAMGSetNumFunctions_mp(amg_solver, num_functions);
         HYPRE_MPAMGSetAggNumLevels_mp(amg_solver, agg_num_levels);
         HYPRE_MPAMGSetAggInterpType_mp(amg_solver, agg_interp_type);
         HYPRE_MPAMGSetAggTruncFactor_mp(amg_solver, agg_trunc_factor);
         HYPRE_MPAMGSetAggP12TruncFactor_mp(amg_solver, agg_P12_trunc_factor);
         HYPRE_MPAMGSetAggPMaxElmts_mp(amg_solver, agg_P_max_elmts);
         HYPRE_MPAMGSetAggP12MaxElmts_mp(amg_solver, agg_P12_max_elmts);
         HYPRE_MPAMGSetNumPaths_mp(amg_solver, num_paths);
         HYPRE_MPAMGSetNodal_mp(amg_solver, nodal);
         HYPRE_MPAMGSetNodalDiag_mp(amg_solver, nodal_diag);
         HYPRE_MPAMGSetKeepSameSign_mp(amg_solver, keep_same_sign);
         HYPRE_MPAMGSetPrecisionArray_mp(amg_solver, precision_array);
         if (ns_coarse > -1)
         {
            HYPRE_MPAMGSetCycleNumSweeps_mp(amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_MPAMGSetCycleNumSweeps_mp(amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_MPAMGSetCycleNumSweeps_mp(amg_solver, ns_up,     2);
         }*/
         // Set the preconditioner for BiCGSTAB (single precision matrix)
         HYPRE_BiCGSTABSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_dbl);
         // Set the preconditioner for BiCGSTAB.
         // This actually sets a pointer to a single precision AMG solver.
         // The setup and solve functions just allow us to accept double precision
         // rhs and sol vectors from the BiCGSTAB solver to do the preconditioner solve.        
         HYPRE_BiCGSTABSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSetup_mp,
                                amg_solver);

         HYPRE_BiCGSTABGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf_dbl("HYPRE_ParCSRBiCGSTABGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf_dbl("HYPRE_ParCSRBiCGSTABGetPrecond got good precond\n");
         }
      }
      
      else if (solver_id == 4)
      {
        if (myid == 0) hypre_printf_dbl("\n\n***** Solver: MIXED PRECISION DS-BiCGSTAB *****\n");
      }
      // Setup BiCGSTAB solver (double precision)
      HYPRE_BiCGSTABSetup_dbl(pcg_solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Mixed precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();
      fflush(NULL);

      time_index = hypre_InitializeTiming_dbl("DBL Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming_dbl(time_index);
      //  BiCGSTAB solve (double precision)
      HYPRE_BiCGSTABSolve_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming_dbl(time_index);
      hypre_PrintTiming_dbl("Mixed precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming_dbl(time_index);
      hypre_ClearTiming_dbl();

      HYPRE_BiCGSTABGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
        hypre_printf_dbl("final relative residual norm = %e \n", dfinal_res_norm);
        hypre_printf_dbl("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRBiCGSTABDestroy_dbl(pcg_solver);
      if (solver_id == 5) HYPRE_BoomerAMGDestroy_flt(amg_solver);
      if (solver_id == 15) HYPRE_MPAMGDestroy_mp(amg_solver);
    } //end BiCGSTAB   
   }   
    
   /* Clean up */
   HYPRE_IJVectorDestroy_flt(ij_b_flt);
   HYPRE_IJVectorDestroy_flt(ij_x_flt);

   HYPRE_IJVectorDestroy_dbl(ij_b_dbl);
   HYPRE_IJVectorDestroy_dbl(ij_x_dbl);

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}

HYPRE_Int HYPRE_DefectCorrectionSolver(HYPRE_ParCSRMatrix A, 
		HYPRE_ParCSRMatrix B, 
		HYPRE_ParVector x, 
		HYPRE_ParVector b,
		HYPRE_Solver solver,
		HYPRE_PtrToSolverFcn approx_solve,
		HYPRE_Int maxits)
{

   

   return 0;

}
/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian_mp( HYPRE_Int            argc,
                      char                *argv[],
                      HYPRE_Int            arg_index,
                      HYPRE_ParCSRMatrix  *A_flt_ptr,
                      HYPRE_ParCSRMatrix  *A_dbl_ptr     )
{
   HYPRE_BigInt    nx, ny, nz;
   HYPRE_Int       P, Q, R;
   double          cx, cy, cz;

   HYPRE_ParCSRMatrix  A_flt;
   HYPRE_ParCSRMatrix  A_dbl;

   HYPRE_Int       num_procs, myid;
   HYPRE_Int       p, q, r;
   HYPRE_Int       num_fun = 1;
   double         *values_dbl;
   float          *values_flt;
   double         *mtrx_dbl;
   float          *mtrx_flt;

   HYPRE_Int       sys_opt = 0;
   HYPRE_Int       i;


   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

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
      hypre_printf_dbl("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf_dbl("  Laplacian:   num_fun = %d\n", num_fun);
      hypre_printf_dbl("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf_dbl("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf_dbl("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
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

   values_dbl = (double*) calloc(4, sizeof(double));
   values_flt = (float*) calloc(4, sizeof(float));

   values_dbl[1] = -cx;
   values_dbl[2] = -cy;
   values_dbl[3] = -cz;
   values_flt[1] = -(float)cx;
   values_flt[2] = -(float)cy;
   values_flt[3] = -(float)cz;

   values_dbl[0] = 0.;
   values_flt[0] = 0.;
   if (nx > 1)
   {
      values_dbl[0] += 2.0 * cx;
      values_flt[0] += 2.0 * (float)cx;
   }
   if (ny > 1)
   {
      values_dbl[0] += 2.0 * cy;
      values_flt[0] += 2.0 * (float)cy;
   }
   if (nz > 1)
   {
      values_dbl[0] += 2.0 * cz;
      values_flt[0] += 2.0 * (float)cz;
   }

   if (num_fun == 1)
   {
      A_dbl = (HYPRE_ParCSRMatrix) GenerateLaplacian_dbl(MPI_COMM_WORLD,
                                                 nx, ny, nz, P, Q, R, p, q, r, values_dbl);
      A_flt = (HYPRE_ParCSRMatrix) GenerateLaplacian_flt(MPI_COMM_WORLD,
                                                 nx, ny, nz, P, Q, R, p, q, r, values_flt);
   }
   else
   {
      mtrx_dbl = (double*) calloc(num_fun * num_fun, sizeof(double));
      mtrx_flt = (float*) calloc(num_fun * num_fun, sizeof(float));

      if (num_fun == 2)
      {
         if (sys_opt == 1) /* identity  */
         {
            mtrx_dbl[0] = 1.0;
            mtrx_dbl[1] = 0.0;
            mtrx_dbl[2] = 0.0;
            mtrx_dbl[3] = 1.0;
         }
         else if (sys_opt == 2)
         {
            mtrx_dbl[0] = 1.0;
            mtrx_dbl[1] = 0.0;
            mtrx_dbl[2] = 0.0;
            mtrx_dbl[3] = 20.0;
         }
         else if (sys_opt == 3) /* similar to barry's talk - ex1 */
         {
            mtrx_dbl[0] = 1.0;
            mtrx_dbl[1] = 2.0;
            mtrx_dbl[2] = 2.0;
            mtrx_dbl[3] = 1.0;
         }
         else if (sys_opt == 4) /* can use with vcoef to get barry's ex*/
         {
            mtrx_dbl[0] = 1.0;
            mtrx_dbl[1] = 1.0;
            mtrx_dbl[2] = 1.0;
            mtrx_dbl[3] = 1.0;
         }
         else if (sys_opt == 5) /* barry's talk - ex1 */
         {
            mtrx_dbl[0] = 1.0;
            mtrx_dbl[1] = 1.1;
            mtrx_dbl[2] = 1.1;
            mtrx_dbl[3] = 1.0;
         }
         else if (sys_opt == 6) /*  */
         {
            mtrx_dbl[0] = 1.1;
            mtrx_dbl[1] = 1.0;
            mtrx_dbl[2] = 1.0;
            mtrx_dbl[3] = 1.1;
         }

         else /* == 0 */
         {
            mtrx_dbl[0] = 2;
            mtrx_dbl[1] = 1;
            mtrx_dbl[2] = 1;
            mtrx_dbl[3] = 2;
         }
      }
      else if (num_fun == 3)
      {
         if (sys_opt == 1)
         {
            mtrx_dbl[0] = 1.0;
            mtrx_dbl[1] = 0.0;
            mtrx_dbl[2] = 0.0;
            mtrx_dbl[3] = 0.0;
            mtrx_dbl[4] = 1.0;
            mtrx_dbl[5] = 0.0;
            mtrx_dbl[6] = 0.0;
            mtrx_dbl[7] = 0.0;
            mtrx_dbl[8] = 1.0;
         }
         else if (sys_opt == 2)
         {
            mtrx_dbl[0] = 1.0;
            mtrx_dbl[1] = 0.0;
            mtrx_dbl[2] = 0.0;
            mtrx_dbl[3] = 0.0;
            mtrx_dbl[4] = 20.0;
            mtrx_dbl[5] = 0.0;
            mtrx_dbl[6] = 0.0;
            mtrx_dbl[7] = 0.0;
            mtrx_dbl[8] = .01;
         }
         else if (sys_opt == 3)
         {
            mtrx_dbl[0] = 1.01;
            mtrx_dbl[1] = 1;
            mtrx_dbl[2] = 0.0;
            mtrx_dbl[3] = 1;
            mtrx_dbl[4] = 2;
            mtrx_dbl[5] = 1;
            mtrx_dbl[6] = 0.0;
            mtrx_dbl[7] = 1;
            mtrx_dbl[8] = 1.01;
         }
         else if (sys_opt == 4) /* barry ex4 */
         {
            mtrx_dbl[0] = 3;
            mtrx_dbl[1] = 1;
            mtrx_dbl[2] = 0.0;
            mtrx_dbl[3] = 1;
            mtrx_dbl[4] = 4;
            mtrx_dbl[5] = 2;
            mtrx_dbl[6] = 0.0;
            mtrx_dbl[7] = 2;
            mtrx_dbl[8] = .25;
         }
         else /* == 0 */
         {
            mtrx_dbl[0] = 2.0;
            mtrx_dbl[1] = 1.0;
            mtrx_dbl[2] = 0.0;
            mtrx_dbl[3] = 1.0;
            mtrx_dbl[4] = 2.0;
            mtrx_dbl[5] = 1.0;
            mtrx_dbl[6] = 0.0;
            mtrx_dbl[7] = 1.0;
            mtrx_dbl[8] = 2.0;
         }

      }
      else if (num_fun == 4)
      {
         mtrx_dbl[0] = 1.01;
         mtrx_dbl[1] = 1;
         mtrx_dbl[2] = 0.0;
         mtrx_dbl[3] = 0.0;
         mtrx_dbl[4] = 1;
         mtrx_dbl[5] = 2;
         mtrx_dbl[6] = 1;
         mtrx_dbl[7] = 0.0;
         mtrx_dbl[8] = 0.0;
         mtrx_dbl[9] = 1;
         mtrx_dbl[10] = 1.01;
         mtrx_dbl[11] = 0.0;
         mtrx_dbl[12] = 2;
         mtrx_dbl[13] = 1;
         mtrx_dbl[14] = 0.0;
         mtrx_dbl[15] = 1;
      }

      for (i=0; i<num_fun*num_fun; i++)
      {
         mtrx_flt[i] = (float)mtrx_dbl[i];
      }

      A_dbl = (HYPRE_ParCSRMatrix) GenerateSysLaplacian_dbl(MPI_COMM_WORLD,
                                                       nx, ny, nz, P, Q,
                                                       R, p, q, r, num_fun, mtrx_dbl, values_dbl);
      A_flt = (HYPRE_ParCSRMatrix) GenerateSysLaplacian_flt(MPI_COMM_WORLD,
                                                       nx, ny, nz, P, Q,
                                                       R, p, q, r, num_fun, mtrx_flt, values_flt);

      free(mtrx_dbl);
      free(mtrx_flt);
   }

   free(values_dbl);
   free(values_flt);

   *A_flt_ptr = A_flt;
   *A_dbl_ptr = A_dbl;

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
BuildParDifConv_mp( HYPRE_Int            argc,
                    char                *argv[],
                    HYPRE_Int            arg_index,
                    HYPRE_ParCSRMatrix  *A_flt_ptr,
                    HYPRE_ParCSRMatrix  *A_dbl_ptr)
{
   HYPRE_BigInt    nx, ny, nz;
   HYPRE_Int       P, Q, R;
   double          cx, cy, cz;
   double          ax, ay, az, atype;
   double          hinx, hiny, hinz;
   HYPRE_Int       sign_prod;

   HYPRE_ParCSRMatrix  A_flt;
   HYPRE_ParCSRMatrix  A_dbl;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q, r, i;
   double             *values_dbl;
   float              *values_flt;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

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
      hypre_printf_dbl("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf_dbl("  Convection-Diffusion: \n");
      hypre_printf_dbl("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");
      hypre_printf_dbl("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf_dbl("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf_dbl("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf_dbl("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   hinx = 1. / (double)(nx + 1);
   hiny = 1. / (double)(ny + 1);
   hinz = 1. / (double)(nz + 1);

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
   values_dbl = (double*) calloc(7, sizeof(double));
   values_flt = (float*) calloc(7, sizeof(float));

   values_dbl[0] = 0.;

   if (0 == atype) /* forward scheme for conv */
   {
      values_dbl[1] = -cx / (hinx * hinx);
      values_dbl[2] = -cy / (hiny * hiny);
      values_dbl[3] = -cz / (hinz * hinz);
      values_dbl[4] = -cx / (hinx * hinx) + ax / hinx;
      values_dbl[5] = -cy / (hiny * hiny) + ay / hiny;
      values_dbl[6] = -cz / (hinz * hinz) + az / hinz;

      if (nx > 1)
      {
         values_dbl[0] += 2.0 * cx / (hinx * hinx) - 1.*ax / hinx;
      }
      if (ny > 1)
      {
         values_dbl[0] += 2.0 * cy / (hiny * hiny) - 1.*ay / hiny;
      }
      if (nz > 1)
      {
         values_dbl[0] += 2.0 * cz / (hinz * hinz) - 1.*az / hinz;
      }
   }
   else if (1 == atype) /* backward scheme for conv */
   {
      values_dbl[1] = -cx / (hinx * hinx) - ax / hinx;
      values_dbl[2] = -cy / (hiny * hiny) - ay / hiny;
      values_dbl[3] = -cz / (hinz * hinz) - az / hinz;
      values_dbl[4] = -cx / (hinx * hinx);
      values_dbl[5] = -cy / (hiny * hiny);
      values_dbl[6] = -cz / (hinz * hinz);

      if (nx > 1)
      {
         values_dbl[0] += 2.0 * cx / (hinx * hinx) + 1.*ax / hinx;
      }
      if (ny > 1)
      {
         values_dbl[0] += 2.0 * cy / (hiny * hiny) + 1.*ay / hiny;
      }
      if (nz > 1)
      {
         values_dbl[0] += 2.0 * cz / (hinz * hinz) + 1.*az / hinz;
      }
   }
   else if (3 == atype) /* upwind scheme */
   {
      sign_prod = sign_double(cx) * sign_double(ax);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values_dbl[1] = -cx / (hinx * hinx) - ax / hinx;
         values_dbl[4] = -cx / (hinx * hinx);
         if (nx > 1)
         {
            values_dbl[0] += 2.0 * cx / (hinx * hinx) + 1.*ax / hinx;
         }
      }
      else /* diff sign use forward scheme */
      {
         values_dbl[1] = -cx / (hinx * hinx);
         values_dbl[4] = -cx / (hinx * hinx) + ax / hinx;
         if (nx > 1)
         {
            values_dbl[0] += 2.0 * cx / (hinx * hinx) - 1.*ax / hinx;
         }
      }

      sign_prod = sign_double(cy) * sign_double(ay);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values_dbl[2] = -cy / (hiny * hiny) - ay / hiny;
         values_dbl[5] = -cy / (hiny * hiny);
         if (ny > 1)
         {
            values_dbl[0] += 2.0 * cy / (hiny * hiny) + 1.*ay / hiny;
         }
      }
      else /* diff sign use forward scheme */
      {
         values_dbl[2] = -cy / (hiny * hiny);
         values_dbl[5] = -cy / (hiny * hiny) + ay / hiny;
         if (ny > 1)
         {
            values_dbl[0] += 2.0 * cy / (hiny * hiny) - 1.*ay / hiny;
         }
      }

      sign_prod = sign_double(cz) * sign_double(az);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values_dbl[3] = -cz / (hinz * hinz) - az / hinz;
         values_dbl[6] = -cz / (hinz * hinz);
         if (nz > 1)
         {
            values_dbl[0] += 2.0 * cz / (hinz * hinz) + 1.*az / hinz;
         }
      }
      else /* diff sign use forward scheme */
      {
         values_dbl[3] = -cz / (hinz * hinz);
         values_dbl[6] = -cz / (hinz * hinz) + az / hinz;
         if (nz > 1)
         {
            values_dbl[0] += 2.0 * cz / (hinz * hinz) - 1.*az / hinz;
         }
      }
   }
   else /* centered difference scheme */
   {
      values_dbl[1] = -cx / (hinx * hinx) - ax / (2.*hinx);
      values_dbl[2] = -cy / (hiny * hiny) - ay / (2.*hiny);
      values_dbl[3] = -cz / (hinz * hinz) - az / (2.*hinz);
      values_dbl[4] = -cx / (hinx * hinx) + ax / (2.*hinx);
      values_dbl[5] = -cy / (hiny * hiny) + ay / (2.*hiny);
      values_dbl[6] = -cz / (hinz * hinz) + az / (2.*hinz);

      if (nx > 1)
      {
         values_dbl[0] += 2.0 * cx / (hinx * hinx);
      }
      if (ny > 1)
      {
         values_dbl[0] += 2.0 * cy / (hiny * hiny);
      }
      if (nz > 1)
      {
         values_dbl[0] += 2.0 * cz / (hinz * hinz);
      }
   }

   for (i=0; i<7; i++)
   {
      values_flt[i] = (float)values_dbl[i];
   }
	   
   A_dbl = (HYPRE_ParCSRMatrix) GenerateDifConv_dbl(MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values_dbl);
   A_flt = (HYPRE_ParCSRMatrix) GenerateDifConv_flt(MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values_flt);

   free(values_dbl);
   free(values_flt);

   *A_dbl_ptr = A_dbl;
   *A_flt_ptr = A_flt;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian9pt_mp( HYPRE_Int            argc,
                         char                *argv[],
                         HYPRE_Int            arg_index,
                         HYPRE_ParCSRMatrix  *A_flt_ptr,
                         HYPRE_ParCSRMatrix  *A_dbl_ptr     )
{
   HYPRE_BigInt         nx, ny;
   HYPRE_Int            P, Q;

   HYPRE_ParCSRMatrix   A_flt;
   HYPRE_ParCSRMatrix   A_dbl;

   HYPRE_Int            num_procs, myid;
   HYPRE_Int            p, q;
   double              *values_dbl;
   float               *values_flt;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

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
      hypre_printf_dbl("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf_dbl("  Laplacian 9pt:\n");
      hypre_printf_dbl("    (nx, ny) = (%b, %b)\n", nx, ny);
      hypre_printf_dbl("    (Px, Py) = (%d, %d)\n\n", P,  Q);
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

   values_dbl = (double*) calloc(2, sizeof(double));
   values_flt = (float*) calloc(2, sizeof(float));

   values_dbl[1] = -1.;
   values_flt[1] = -1.;

   values_dbl[0] = 0.;
   values_flt[0] = 0.;
   if (nx > 1)
   {
      values_dbl[0] += 2.0;
      values_flt[0] += 2.0;
   }
   if (ny > 1)
   {
      values_dbl[0] += 2.0;
      values_flt[0] += 2.0;
   }
   if (nx > 1 && ny > 1)
   {
      values_dbl[0] += 4.0;
      values_flt[0] += 4.0;
   }

   A_flt = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt_flt(MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values_flt);

   free(values_flt);

   A_dbl = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt_dbl(MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values_dbl);

   free(values_dbl);

   *A_dbl_ptr = A_dbl;

   *A_flt_ptr = A_flt;

   return (0);
}

/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian27pt_mp( HYPRE_Int            argc,
                          char                *argv[],
                          HYPRE_Int            arg_index,
                          HYPRE_ParCSRMatrix  *A_flt_ptr,
                          HYPRE_ParCSRMatrix  *A_dbl_ptr     )
{
   HYPRE_BigInt        nx, ny, nz;
   HYPRE_Int           P, Q, R;

   HYPRE_ParCSRMatrix  A_flt;
   HYPRE_ParCSRMatrix  A_dbl;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q, r;
   float              *values_flt;
   double             *values_dbl;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

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
      hypre_printf_dbl("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf_dbl("  Laplacian_27pt:\n");
      hypre_printf_dbl("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf_dbl("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
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

   values_flt = (float*) calloc(2, sizeof(float));
   values_dbl = (double*) calloc(2, sizeof(double));

   values_flt[0] = 26.0;
   values_dbl[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
   {
      values_flt[0] = 8.0;
      values_dbl[0] = 8.0;
   }
   if (nx * ny == 1 || nx * nz == 1 || ny * nz == 1)
   {
      values_flt[0] = 2.0;
      values_dbl[0] = 2.0;
   }
   values_flt[1] = -1.;
   values_dbl[1] = -1.;

   A_flt = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt_flt(MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values_flt);

   free(values_flt);

   A_dbl = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt_dbl(MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values_dbl);

   free(values_dbl);

   *A_dbl_ptr = A_dbl;
   *A_flt_ptr = A_flt;

   return (0);
}

/*----------------------------------------------------------------------
 * Build 125-point laplacian in 3D (27-pt squared)
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian125pt_mp( HYPRE_Int            argc,
                           char                *argv[],
                           HYPRE_Int            arg_index,
                           HYPRE_ParCSRMatrix  *A_flt_ptr,
                           HYPRE_ParCSRMatrix  *A_dbl_ptr     )
{
   HYPRE_BigInt              nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix        A_flt, B_flt;
   HYPRE_ParCSRMatrix        A_dbl, B_dbl;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   float                    *values_flt;
   double                   *values_dbl;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

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
      hypre_printf_dbl("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf_dbl("  Laplacian_125pt:\n");
      hypre_printf_dbl("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf_dbl("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
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

   values_flt = (float*) calloc(2, sizeof(float));
   values_dbl = (double*) calloc(2, sizeof(double));

   values_flt[0] = 26.0;
   values_dbl[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
   {
      values_flt[0] = 8.0;
      values_dbl[0] = 8.0;
   }
   if (nx * ny == 1 || nx * nz == 1 || ny * nz == 1)
   {
      values_flt[0] = 2.0;
      values_dbl[0] = 2.0;
   }
   values_flt[1] = -1.;
   values_dbl[1] = -1.;

   B_flt = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt_flt(MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values_flt);
   A_flt = (HYPRE_ParCSRMatrix) hypre_ParCSRMatMat_flt(B_flt, B_flt);

   HYPRE_ParCSRMatrixDestroy_flt(B_flt);
   free(values_flt);

   B_dbl = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt_dbl(MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values_dbl);
   A_dbl = (HYPRE_ParCSRMatrix) hypre_ParCSRMatMat_dbl(B_dbl, B_dbl);

   HYPRE_ParCSRMatrixDestroy_dbl(B_dbl);
   free(values_dbl);

   *A_dbl_ptr = A_dbl;
   *A_flt_ptr = A_flt;

   return (0);
}

/*----------------------------------------------------------------------
 * Build 7-point in 2D
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParRotate7pt_mp( HYPRE_Int            argc,
                      char                *argv[],
                      HYPRE_Int            arg_index,
                      HYPRE_ParCSRMatrix  *A_flt_ptr, 
                      HYPRE_ParCSRMatrix  *A_dbl_ptr     )
{
   HYPRE_BigInt              nx, ny;
   HYPRE_Int                 P, Q;

   HYPRE_ParCSRMatrix        A_flt;
   HYPRE_ParCSRMatrix        A_dbl;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q;
   double                    eps, alpha;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

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
      hypre_printf_dbl("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf_dbl("  Rotate 7pt:\n");
      hypre_printf_dbl("    alpha = %f, eps = %f\n", alpha, eps);
      hypre_printf_dbl("    (nx, ny) = (%b, %b)\n", nx, ny);
      hypre_printf_dbl("    (Px, Py) = (%d, %d)\n", P,  Q);
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

   A_flt = (HYPRE_ParCSRMatrix) GenerateRotate7pt_flt(MPI_COMM_WORLD,
                                              nx, ny, P, Q, p, q, (float)alpha, (float)eps);
   A_dbl = (HYPRE_ParCSRMatrix) GenerateRotate7pt_dbl(MPI_COMM_WORLD,
                                              nx, ny, P, Q, p, q, alpha, eps);

   *A_flt_ptr = A_flt;
   *A_dbl_ptr = A_dbl;

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
BuildParVarDifConv_mp( HYPRE_Int            argc,
                       char                *argv[],
                       HYPRE_Int            arg_index,
                       HYPRE_ParCSRMatrix  *A_flt_ptr,
                       HYPRE_ParCSRMatrix  *A_dbl_ptr,
                       HYPRE_ParVector     *rhs_flt_ptr,
                       HYPRE_ParVector     *rhs_dbl_ptr     )
{
   HYPRE_BigInt        nx, ny, nz;
   HYPRE_Int           P, Q, R;

   HYPRE_ParCSRMatrix  A_flt;
   HYPRE_ParCSRMatrix  A_dbl;
   HYPRE_ParVector     rhs_flt;
   HYPRE_ParVector     rhs_dbl;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q, r;
   HYPRE_Int           type;
   double              eps;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

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
      hypre_printf_dbl("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf_dbl("  ell PDE: eps = %f\n", eps);
      hypre_printf_dbl("    Dx(aDxu) + Dy(bDyu) + Dz(cDzu) + d Dxu + e Dyu + f Dzu  + g u= f\n");
      hypre_printf_dbl("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf_dbl("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
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
      A_dbl = (HYPRE_ParCSRMatrix) GenerateVarDifConv_dbl(MPI_COMM_WORLD,
                                                          nx, ny, nz, P, Q, R, p, q, r, 
							  eps, &rhs_dbl);
      A_flt = (HYPRE_ParCSRMatrix) GenerateVarDifConv_flt(MPI_COMM_WORLD,
                                                          nx, ny, nz, P, Q, R, p, q, r, 
							  (float)eps, &rhs_flt);
   }
   else
   {
      A_dbl = (HYPRE_ParCSRMatrix) GenerateRSVarDifConv_dbl(MPI_COMM_WORLD,
                                                            nx, ny, nz, P, Q, R, p, q, r, 
							    eps, &rhs_dbl, type);
      A_flt = (HYPRE_ParCSRMatrix) GenerateRSVarDifConv_flt(MPI_COMM_WORLD,
                                                            nx, ny, nz, P, Q, R, p, q, r, 
							    (float)eps, &rhs_flt, type);
   }

   *A_flt_ptr = A_flt;
   *rhs_flt_ptr = rhs_flt;
   *A_dbl_ptr = A_dbl;
   *rhs_dbl_ptr = rhs_dbl;

   return (0);
}


