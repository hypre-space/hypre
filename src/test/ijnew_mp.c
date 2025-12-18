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
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"

#include <time.h>

#define my_min(a,b)  (((a)<(b)) ? (a) : (b))

HYPRE_Int BuildParLaplacian_mp( HYPRE_Int argc, char *argv[], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParDifConv_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParLaplacian9pt_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParLaplacian27pt_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParRotate7pt_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr );
HYPRE_Int BuildParVarDifConv_mp (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                   HYPRE_ParCSRMatrix *A_flt_ptr, HYPRE_ParCSRMatrix *A_dbl_ptr,
                   HYPRE_ParVector *rhs_flt_ptr, HYPRE_ParVector *rhs_dbl_ptr );

hypre_int 
main (hypre_int argc, char *argv[])
{
   HYPRE_Int arg_index;
   HYPRE_Int print_usage;
   HYPRE_Int myid, num_procs;
   HYPRE_Int ilower, iupper;
   HYPRE_Int jlower, jupper;
   HYPRE_Int solver_id = 0;
   HYPRE_Real one = 1.0;
   HYPRE_Real zero = 0.;
   HYPRE_Int num_iterations;
   HYPRE_Real dfinal_res_norm;
   HYPRE_Real  final_res_norm;
   HYPRE_Int	  time_index;   
   HYPRE_Real  wall_time;   
   HYPRE_Real max_row_sum = 1.0;
   HYPRE_Int build_matrix_type = 2;
   HYPRE_Int build_rhs_type = 2;
   HYPRE_Int build_matrix_arg_index;
   HYPRE_Int build_rhs_arg_index;
   HYPRE_Int mg_max_iter = 100;
   HYPRE_Int max_iter = 1000;
   HYPRE_Int coarsen_type = 10;
   HYPRE_Int interp_type = 6;
   HYPRE_Int P_max_elmts = 4;
   HYPRE_Real trunc_factor = 0.0;
   HYPRE_Real strong_threshold = 0.25;
   HYPRE_Int relax_type = 8;
   HYPRE_Int relax_up = 14;
   HYPRE_Int relax_down = 13;
   HYPRE_Int relax_coarse = 9;
   HYPRE_Int num_sweeps = 1;
   HYPRE_Int ns_down = -1;
   HYPRE_Int ns_up = -1;
   HYPRE_Int ns_coarse = -1;
   HYPRE_Int max_levels = 25;
   HYPRE_Int debug_flag = 0;
   HYPRE_Int agg_num_levels = 0;
   HYPRE_Int num_paths = 1;
   HYPRE_Int agg_interp_type = 4;
   HYPRE_Int agg_P_max_elmts = 0;
   HYPRE_Real agg_trunc_factor = 0;
   HYPRE_Int agg_P12_max_elmts = 0;
   HYPRE_Real agg_P12_trunc_factor = 0;
   HYPRE_Int smooth_type = 6;
   HYPRE_Int smooth_num_levels = 0;
   HYPRE_Int smooth_num_sweeps = 1;
   HYPRE_Real tol = 1.e-8;
   HYPRE_Int ioutdat = 2;
   HYPRE_Int poutdat = 1;
   HYPRE_Int flex = 1;
   HYPRE_Int cycle_type = 1;
   HYPRE_Int relax_order = 0;
   HYPRE_Real relax_wt = 1.0;
   HYPRE_Real outer_wt = 1.0;
   HYPRE_Int k_dim = 10;
   HYPRE_Int two_norm = 0;
   HYPRE_Int all = 0;
   HYPRE_Int precision = 0;

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

   HYPRE_Precision *precision_array, precision0;
   HYPRE_Int *prec_elmts;
   HYPRE_Int i, num_prec_elmts = 0;

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
   print_usage = 0;
   arg_index = 1;
   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
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
      else if ( strcmp(argv[arg_index], "-interp") == 0 )
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
      else if ( strcmp(argv[arg_index], "-prec_array") == 0 )
      {
         arg_index++;
         num_prec_elmts  = atoi(argv[arg_index++]);
         prec_elmts = (HYPRE_Int *) hypre_CAlloc((size_t)(num_prec_elmts), 
			 (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
         for (i=0; i < num_prec_elmts; i++)
            prec_elmts[i]  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

      if ( print_usage )
   {
      if ( myid == 0 )
      {
         hypre_printf("\n");
         hypre_printf("Usage: %s [<options>]\n", argv[0]);
         hypre_printf("\n");
         hypre_printf("  -laplacian             : build 5pt 2D laplacian problem (default) \n");
         hypre_printf("  -sysL <num functions>  : build SYSTEMS laplacian 7pt operator\n");
         hypre_printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
         hypre_printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
         hypre_printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
         hypre_printf("  -vardifconv [<opts>]   : build variable conv.-diffusion problem\n");
         hypre_printf("  -rotate [<opts>]       : build 7pt rotated laplacian problem\n");
         hypre_printf("    -n <nx> <ny> <nz>    : total problem size \n");
         hypre_printf("    -P <Px> <Py> <Pz>    : processor topology\n");
         hypre_printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
         hypre_printf("    -a <ax> <ay> <az>    : convection coefficients\n");
         hypre_printf("\n");
         hypre_printf("  -rhsrand               : rhs is random vector\n");
         hypre_printf("  -rhsisone              : rhs is vector with unit coefficients (default)\n");
         hypre_printf("  -xisone                : solution of all ones\n");
         hypre_printf("  -rhszero               : rhs is zero vector\n");
         hypre_printf("\n");
         hypre_printf("\n");
         hypre_printf("  -solver <ID>           : solver ID\n");
         hypre_printf("       10=MPAMG            11=MPAMG-PCG        \n");
         hypre_printf("       13=MPAMG-GMRES      15=MPAMG-BiCGSTAB  \n");
         hypre_printf("\n");
         hypre_printf("  -prec_array  <n> <p0> <p1> ... <pn>: set level precisions \n");
	 hypre_printf("        0 = double precision \n");
	 hypre_printf("        1 = single precision \n");
	 hypre_printf("        2 = long double precision \n");
	 hypre_printf("        Note that <p0> should be 0 always \n");
	 hypre_printf("        precisions at all levels greater than n will be set to <pn> \n");
	 hypre_printf("        e.g., -prec_array 2 0 1 will set coarse level to double precision \n");
	 hypre_printf("        and all other levels to single precision \n");
         hypre_printf("\n");
         hypre_printf("\n");
         hypre_printf("  -cljp                 : CLJP coarsening \n");
         hypre_printf("  -pmis                 : PMIS coarsening \n");
         hypre_printf("  -hmis                 : HMIS coarsening (default)\n");
         hypre_printf("  -ruge                 : Ruge-Stueben coarsening (local)\n");
         hypre_printf("  -ruge1p               : Ruge-Stueben coarsening 1st pass only(local)\n");
         hypre_printf("  -ruge3                : third pass on boundary\n");
         hypre_printf("  -falgout              : local Ruge_Stueben followed by CLJP\n");
         hypre_printf("\n");
         hypre_printf("  -interptype  <val>    : set interpolation type\n");
         hypre_printf("       0=Classical modified interpolation  \n");
         hypre_printf("       4=multipass interpolation  \n");
         hypre_printf("       6=extended classical modified interpolation (default) \n");
         hypre_printf("       8=standard interpolation  \n");
         hypre_printf("\n");

         /* RL */
         hypre_printf("  -rlx  <val>            : relaxation type\n");
         hypre_printf("       0=Weighted Jacobi  \n");
         hypre_printf("       3=Hybrid Gauss-Seidel  \n");
         hypre_printf("       4=Hybrid backward Gauss-Seidel  \n");
         hypre_printf("       6=Hybrid symmetric Gauss-Seidel  \n");
         hypre_printf("       8= symmetric L1-Gauss-Seidel  \n");
         hypre_printf("       13= forward L1-Gauss-Seidel  \n");
         hypre_printf("       14= backward L1-Gauss-Seidel  \n");
         hypre_printf("       18=L1-Jacobi \n");
         hypre_printf("       9=Gauss elimination (use for coarsest grid only)  \n");
         hypre_printf("  -rlx_coarse  <val>       : set relaxation type for coarsest grid\n");
         hypre_printf("  -rlx_down    <val>       : set relaxation type for down cycle\n");
         hypre_printf("  -rlx_up      <val>       : set relaxation type for up cycle\n");
         hypre_printf("  -ns <val>              : Use <val> sweeps on each level\n");
         hypre_printf("                           (default C/F down, F/C up, F/C fine\n");
         hypre_printf("  -ns_coarse  <val>       : set no. of sweeps for coarsest grid\n");
         /* RL restore these */
         hypre_printf("  -ns_down    <val>       : set no. of sweeps for down cycle\n");
         hypre_printf("  -ns_up      <val>       : set no. of sweeps for up cycle\n");
         hypre_printf("\n");
         hypre_printf("  -th   <val>            : set AMG strength threshold  \n");
         hypre_printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
         hypre_printf("  -Pmx  <val>            : set maximal no. of elmts per row for AMG interpolation (default: 4)\n");
         hypre_printf("  -mxrs <val>            : set AMG maximum row sum threshold for dependency weakening \n");

         hypre_printf("  -w   <val>             : set Jacobi relax weight = val\n");
         hypre_printf("  -flex <val>            : use flexible CG  \n");
         hypre_printf("       1 = on (default in driver)\n");
         hypre_printf("       0 = off\n");
         hypre_printf("  -k   <val>             : dimension Krylov space for GMRES\n");
         hypre_printf("  -mxl  <val>            : maximum number of levels (MPAMG)\n");
         hypre_printf("  -tol  <val>            : set solver convergence tolerance = val\n");
         hypre_printf("  -max_iter  <val>       : set max iterations for Krylov solvers\n");
         hypre_printf("  -mg_max_iter  <val>    : set max iterations for mg solvers\n");
         hypre_printf("  -agg_nl  <val>         : set number of aggressive coarsening levels (default:0)\n");
         hypre_printf("  -np  <val>             : set number of paths of length 2 for aggr. coarsening\n");
         hypre_printf("\n");
         hypre_printf("  -iout <val>            : set output flag solver \n");
         hypre_printf("       0 = no output\n");
         hypre_printf("\n");
         hypre_printf("  -pout <val>            : set output flag preconditioner\n");
         hypre_printf("       0 = no output\n");
         hypre_printf("\n");
         hypre_printf("  -print                 : print out the system\n");
         hypre_printf("\n");
      }

      goto final;
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
   else if (build_matrix_type == 6)
   {
      BuildParDifConv_mp(argc, argv, build_matrix_arg_index, &A_flt, &A_dbl);
   }
   else if (build_matrix_type == 7)
   {
      BuildParVarDifConv_mp(argc, argv, build_matrix_arg_index, &A_flt, &A_dbl, &b_flt, &b_dbl);
      //build_rhs_type = 6;
   }
   else if (build_matrix_type == 8)
   {
      BuildParRotate7pt_mp(argc, argv, build_matrix_arg_index, &A_flt, &A_dbl);
   }

   HYPRE_ParCSRMatrixGetLocalRange_flt( A_flt,
                                       &ilower, &iupper, &jlower, &jupper);

   if (build_rhs_type == 2)
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector has unit coefficients\n");
         hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &ij_b_flt);
      HYPRE_IJVectorSetObjectType_flt(ij_b_flt, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(ij_b_flt);
      HYPRE_IJVectorAssemble_flt(ij_b_flt);
      HYPRE_IJVectorGetObject_flt( ij_b_flt, (void **) &b_flt );
      HYPRE_ParVectorSetConstantValues_flt(b_flt, (hypre_float) one);
      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &ij_b_dbl);
      HYPRE_IJVectorSetObjectType_dbl(ij_b_dbl, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(ij_b_dbl);
      HYPRE_IJVectorAssemble_dbl(ij_b_dbl);
      HYPRE_IJVectorGetObject_dbl( ij_b_dbl, (void **) &b_dbl );
      HYPRE_ParVectorSetConstantValues_dbl(b_dbl, (hypre_double) one);
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
      HYPRE_Real one = 1.0;
      if (myid == 0)
      {
         hypre_printf("  RHS vector has random coefficients\n");
         hypre_printf("  Initial guess is 0\n");
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
         hypre_printf("  RHS vector has unit coefficients\n");
         hypre_printf("  Initial guess is random\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &ij_b_flt);
      HYPRE_IJVectorSetObjectType_flt(ij_b_flt, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(ij_b_flt);
      HYPRE_IJVectorAssemble_flt(ij_b_flt);
      HYPRE_IJVectorGetObject_flt( ij_b_flt, (void **) &b_flt );
      HYPRE_ParVectorSetConstantValues_flt(b_flt, (hypre_float) zero);
      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &ij_b_dbl);
      HYPRE_IJVectorSetObjectType_dbl(ij_b_dbl, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(ij_b_dbl);
      HYPRE_IJVectorAssemble_dbl(ij_b_dbl);
      HYPRE_IJVectorGetObject_dbl( ij_b_dbl, (void **) &b_dbl );
      HYPRE_ParVectorSetConstantValues_dbl(b_dbl, (hypre_double) zero);
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

   if (solver_id == 10 || solver_id == 11 || solver_id == 13 || solver_id == 15)
   {
      precision_array = (HYPRE_Precision *) hypre_CAlloc((size_t)(max_levels), (size_t)sizeof(HYPRE_Precision), HYPRE_MEMORY_HOST);
      precision_array[0] = HYPRE_REAL_DOUBLE;
      if (num_prec_elmts)
      {
         for (i=0; i < num_prec_elmts; i++)
	 {
	    if (prec_elmts[i] == 0) precision_array[i] = HYPRE_REAL_DOUBLE;
	    else if (prec_elmts[i] == 1) precision_array[i] = HYPRE_REAL_SINGLE;
	    else if (prec_elmts[i] == 2) precision_array[i] = HYPRE_REAL_LONGDOUBLE;
	 }
      }
      for (i=num_prec_elmts; i < max_levels; i++)
      {
         if (i>0) precision_array[i] = precision_array[i-1];
      }
      precision0 = precision_array[0];
   }
   /*! Done with linear system setup. Now proceed to solve the system. */
   if (solver_id == 10)
   {
   // mixed precision
    {
      HYPRE_Solver amg_solver;

      if (myid == 0) { hypre_printf("Solver:  MPAMG mixed precision\n"); }
      time_index = hypre_InitializeTiming("MPAMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGCreate(&amg_solver);
      
      HYPRE_BoomerAMGSetMaxIter_pre(precision0, amg_solver, mg_max_iter);
      HYPRE_BoomerAMGSetInterpType_pre(precision0, amg_solver, interp_type);
      HYPRE_BoomerAMGSetCoarsenType_pre(precision0, amg_solver, coarsen_type);
      HYPRE_BoomerAMGSetTol_pre(precision0, amg_solver, tol);
      HYPRE_BoomerAMGSetStrongThreshold_pre(precision0, amg_solver, strong_threshold);
      HYPRE_BoomerAMGSetTruncFactor_pre(precision0, amg_solver, trunc_factor);
      HYPRE_BoomerAMGSetPMaxElmts_pre(precision0, amg_solver, P_max_elmts);
      /* note: log is written to standard output, not to file */
      HYPRE_BoomerAMGSetPrintLevel_pre(precision0, amg_solver, poutdat);
      //HYPRE_MPAMGSetCycleType_mp(amg_solver, cycle_type);
      HYPRE_BoomerAMGSetNumSweeps_pre(precision0, amg_solver, num_sweeps);
      if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_pre(precision0, amg_solver, relax_type); }
      if (relax_down > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_down, 1);
      }
      if (relax_up > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_up, 2);
      }
      if (relax_coarse > -1)
      {
         HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_coarse, 3);
      }
      HYPRE_BoomerAMGSetRelaxOrder_pre(precision0, amg_solver, relax_order);
      HYPRE_BoomerAMGSetRelaxWt_pre(precision0, amg_solver, relax_wt);
      HYPRE_BoomerAMGSetOuterWt_pre(precision0, amg_solver, outer_wt);
      HYPRE_BoomerAMGSetMaxLevels_pre(precision0, amg_solver, max_levels);
      HYPRE_BoomerAMGSetMaxRowSum_pre(precision0, amg_solver, max_row_sum);
      HYPRE_BoomerAMGSetDebugFlag_pre(precision0, amg_solver, debug_flag);
      HYPRE_BoomerAMGSetAggNumLevels_pre(precision0, amg_solver, agg_num_levels);
      HYPRE_BoomerAMGSetAggInterpType_pre(precision0, amg_solver, agg_interp_type);
      HYPRE_BoomerAMGSetAggTruncFactor_pre(precision0, amg_solver, agg_trunc_factor);
      HYPRE_BoomerAMGSetAggP12TruncFactor_pre(precision0, amg_solver, agg_P12_trunc_factor);
      HYPRE_BoomerAMGSetAggPMaxElmts_pre(precision0, amg_solver, agg_P_max_elmts);
      HYPRE_BoomerAMGSetAggP12MaxElmts_pre(precision0, amg_solver, agg_P12_max_elmts);
      HYPRE_BoomerAMGSetNumPaths_pre(precision0, amg_solver, num_paths);
      HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_coarse, 3);
      if (ns_down > -1)
      {
         HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_down,   1);
      }
      if (ns_up > -1)
      {
         HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_up,     2);
      }
      HYPRE_MPAMGSetPrecisionArray_mp(amg_solver, precision_array);
      
      if (precision0 == HYPRE_REAL_SINGLE)
      {
         HYPRE_MPAMGSetup_mp(amg_solver, (HYPRE_ParCSRMatrix) A_flt, (HYPRE_ParVector) b_flt, (HYPRE_ParVector) x_flt);
      }
      else
      {
         HYPRE_MPAMGSetup_mp(amg_solver, (HYPRE_ParCSRMatrix) A_dbl, (HYPRE_ParVector) b_dbl, (HYPRE_ParVector) x_dbl);
      }

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("MPAMG Solve");
      hypre_BeginTiming(time_index);

      if (precision0 == HYPRE_REAL_SINGLE)
      {
         HYPRE_MPAMGSolve_mp(amg_solver, (HYPRE_ParCSRMatrix) A_flt, (HYPRE_ParVector) b_flt, (HYPRE_ParVector) x_flt);
      }
      else
      {
         HYPRE_MPAMGSolve_mp(amg_solver, (HYPRE_ParCSRMatrix) A_dbl, (HYPRE_ParVector) b_dbl, (HYPRE_ParVector) x_dbl);
      }

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      
      HYPRE_BoomerAMGGetNumIterations_pre(precision0, amg_solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm_pre(precision0, amg_solver, &dfinal_res_norm);

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("MPAMG Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", dfinal_res_norm);
         hypre_printf("\n");
      }

      HYPRE_BoomerAMGDestroy(amg_solver);
    }
   }
   // PCG solve
   else if (solver_id == 11)
   {
      /* reset solution vector */
      if (build_rhs_type < 4)  
      {
	 HYPRE_ParVectorSetConstantValues_dbl(x_dbl, (hypre_double) zero);
	 HYPRE_ParVectorSetConstantValues_flt(x_flt, (hypre_float) zero);
      }
      else  
      {
	 HYPRE_ParVectorSetRandomValues_dbl(x_dbl, 22775);
	 HYPRE_ParVectorSetRandomValues_flt(x_flt, 22775);
      }

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming("DBL Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      // Create PCG solver
      HYPRE_ParCSRPCGCreate_dbl(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter_dbl(pcg_solver, max_iter);
      HYPRE_PCGSetTol_dbl(pcg_solver, tol);
      HYPRE_PCGSetTwoNorm_dbl(pcg_solver, two_norm);
      HYPRE_PCGSetPrintLevel_dbl(pcg_solver, ioutdat);
      HYPRE_PCGSetFlex_dbl(pcg_solver, flex);
      HYPRE_PCGSetRecomputeResidual_dbl(pcg_solver, 1);      
      
      
      /* Now set up the MPAMG preconditioner and specify any parameters */
      {
         if (myid == 0) hypre_printf("\n\n***** Solver: MIXED PRECISION AMG-PCG *****\n");
         HYPRE_PCGSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_pre(precision0, amg_solver, poutdat); 
         HYPRE_BoomerAMGSetCoarsenType_pre(precision0, amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_pre(precision0, amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_pre(precision0, amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_pre(precision0, amg_solver, zero); 
         HYPRE_BoomerAMGSetMaxIter_pre(precision0, amg_solver, 1); 
         HYPRE_BoomerAMGSetStrongThreshold_pre(precision0, amg_solver, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_pre(precision0, amg_solver, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_pre(precision0, amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_pre(precision0, amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_pre(precision0, amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_pre(precision0, amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_pre(precision0, amg_solver, relax_wt);
         HYPRE_BoomerAMGSetOuterWt_pre(precision0, amg_solver, outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_pre(precision0, amg_solver, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum_pre(precision0, amg_solver, max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_pre(precision0, amg_solver, debug_flag);
         HYPRE_BoomerAMGSetAggNumLevels_pre(precision0, amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_pre(precision0, amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_pre(precision0, amg_solver, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_pre(precision0, amg_solver, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_pre(precision0, amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_pre(precision0, amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_pre(precision0, amg_solver, num_paths);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_up,     2);
         }
         HYPRE_MPAMGSetPrecisionArray_mp(amg_solver, precision_array);
      
         // Set the preconditioner for PCG (double precision matrix)
         if (precision0 == HYPRE_REAL_SINGLE)
	 {
	    HYPRE_PCGSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_flt);
            HYPRE_PCGSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGPrecSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGPrecSetup_mp,
                                amg_solver);
	 }
	 else if (precision0 == HYPRE_REAL_DOUBLE)
	 {
            HYPRE_PCGSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_dbl);
            HYPRE_PCGSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSetup_mp,
                                amg_solver);
	 }
         // Set the preconditioner for PCG.
         // This actually sets a pointer to a single precision AMG solver.
         // The setup and solve functions just allow us to accept double precision
         // rhs and sol vectors from the PCG solver to do the preconditioner solve.        

         HYPRE_PCGGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");
         }
      }
      // Setup PCG solver (double precision)
      HYPRE_PCGSetup_dbl(pcg_solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Mixed precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);

      time_index = hypre_InitializeTiming("DBL Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      //  PCG solve (double precision)
      HYPRE_PCGSolve_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Mixed precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
        hypre_printf("final relative residual norm = %e \n", dfinal_res_norm);
        hypre_printf("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRPCGDestroy_dbl(pcg_solver);
      HYPRE_BoomerAMGDestroy(amg_solver);
   }   
   else if (solver_id == 13)  //GMRES
   {
      /* reset solution vector */
      if (build_rhs_type < 4)  
      {
	 HYPRE_ParVectorSetConstantValues_dbl(x_dbl, (hypre_double) zero);
	 HYPRE_ParVectorSetConstantValues_flt(x_flt, (hypre_float) zero);
      }
      else  
      {
	 HYPRE_ParVectorSetRandomValues_dbl(x_dbl, 22775);
	 HYPRE_ParVectorSetRandomValues_flt(x_flt, 22775);
      }

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming("DBL Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      // Create GMRES solver
      HYPRE_ParCSRGMRESCreate_dbl(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_GMRESSetMaxIter_dbl(pcg_solver, max_iter);
      HYPRE_GMRESSetTol_dbl(pcg_solver, tol);
      HYPRE_GMRESSetKDim_dbl(pcg_solver, k_dim);
      HYPRE_GMRESSetPrintLevel_dbl(pcg_solver, ioutdat);
      
      
      /* Now set up the MPAMG preconditioner and specify any parameters */
      {
         if (myid == 0) hypre_printf("\n\n***** Solver: MIXED PRECISION AMG-PCG *****\n");
         HYPRE_GMRESSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_pre(precision0, amg_solver, poutdat); 
         HYPRE_BoomerAMGSetCoarsenType_pre(precision0, amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_pre(precision0, amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_pre(precision0, amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetTol_pre(precision0, amg_solver, zero); 
         HYPRE_BoomerAMGSetMaxIter_pre(precision0, amg_solver, 1); 
         HYPRE_BoomerAMGSetStrongThreshold_pre(precision0, amg_solver, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_pre(precision0, amg_solver, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts_pre(precision0, amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_pre(precision0, amg_solver, num_sweeps);
         if (relax_type > -1) { HYPRE_BoomerAMGSetRelaxType_pre(precision0, amg_solver, relax_type); }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_pre(precision0, amg_solver, relax_order);
         HYPRE_BoomerAMGSetRelaxWt_pre(precision0, amg_solver, relax_wt);
         HYPRE_BoomerAMGSetOuterWt_pre(precision0, amg_solver, outer_wt);
         HYPRE_BoomerAMGSetMaxLevels_pre(precision0, amg_solver, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum_pre(precision0, amg_solver, max_row_sum);
         HYPRE_BoomerAMGSetDebugFlag_pre(precision0, amg_solver, debug_flag);
         HYPRE_BoomerAMGSetAggNumLevels_pre(precision0, amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_pre(precision0, amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetAggTruncFactor_pre(precision0, amg_solver, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_pre(precision0, amg_solver, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_pre(precision0, amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_pre(precision0, amg_solver, agg_P12_max_elmts);
         HYPRE_BoomerAMGSetNumPaths_pre(precision0, amg_solver, num_paths);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_up,     2);
         }
         HYPRE_MPAMGSetPrecisionArray_mp(amg_solver, precision_array);
	
         // Set the preconditioner for GMRES (single precision matrix)
         if (precision_array[0] == HYPRE_REAL_SINGLE)
	 {
	    HYPRE_GMRESSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_flt);
            HYPRE_GMRESSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGPrecSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGPrecSetup_mp,
                                amg_solver);
	 }
	 else if (precision_array[0] == HYPRE_REAL_DOUBLE)
	 {
            HYPRE_GMRESSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_dbl);
            HYPRE_GMRESSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSetup_mp,
                                amg_solver);
	 }

         HYPRE_GMRESGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf("HYPRE_ParCSRGMRESGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf("HYPRE_ParCSRGMRESGetPrecond got good precond\n");
         }
      }
      
      // Setup GMRES solver (double precision)
      HYPRE_GMRESSetup_dbl(pcg_solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Mixed precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);

      time_index = hypre_InitializeTiming("DBL Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      //  GMRES solve (double precision)
      HYPRE_GMRESSolve_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Mixed precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
        hypre_printf("final relative residual norm = %e \n", dfinal_res_norm);
        hypre_printf("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy gmres solver
      HYPRE_ParCSRGMRESDestroy_dbl(pcg_solver);
      HYPRE_BoomerAMGDestroy(amg_solver);
   }   
   // BiCGSTAB solve
   else if (solver_id == 15)
   {
      /* reset solution vector */
      if (build_rhs_type < 4)  
      {
	 HYPRE_ParVectorSetConstantValues_dbl(x_dbl, (hypre_double) zero);
	 HYPRE_ParVectorSetConstantValues_flt(x_flt, (hypre_float) zero);
      }
      else  
      {
	 HYPRE_ParVectorSetRandomValues_dbl(x_dbl, 22775);
	 HYPRE_ParVectorSetRandomValues_flt(x_flt, 22775);
      }

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver pcg_precond_gotten;      

      time_index = hypre_InitializeTiming("DBL Setup");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      // Create BiCGSTAB solver
      HYPRE_ParCSRBiCGSTABCreate_dbl(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_BiCGSTABSetMaxIter_dbl(pcg_solver, max_iter);
      HYPRE_BiCGSTABSetTol_dbl(pcg_solver, tol);
      HYPRE_BiCGSTABSetPrintLevel_dbl(pcg_solver, ioutdat);
      
      
      /* Now set up the MPAMG preconditioner and specify any parameters */
      {
         if (myid == 0) hypre_printf("\n\n***** Solver: MIXED PRECISION AMG-PCG *****\n");
         HYPRE_PCGSetMaxIter_dbl(pcg_solver, mg_max_iter);
         HYPRE_BoomerAMGCreate(&amg_solver);
         HYPRE_BoomerAMGSetPrintLevel_pre(precision0, amg_solver, poutdat); 
         HYPRE_BoomerAMGSetCoarsenType_pre(precision0, amg_solver, coarsen_type);
         HYPRE_BoomerAMGSetInterpType_pre(precision0, amg_solver, interp_type);
         HYPRE_BoomerAMGSetNumSweeps_pre(precision0, amg_solver, num_sweeps);
         HYPRE_BoomerAMGSetMaxIter_pre(precision0, amg_solver, 1); 
         HYPRE_BoomerAMGSetPMaxElmts_pre(precision0, amg_solver, P_max_elmts);
         HYPRE_BoomerAMGSetNumSweeps_pre(precision0, amg_solver, num_sweeps);
         if (relax_type > -1) 
	 { 
	    HYPRE_BoomerAMGSetRelaxType_pre(precision0, amg_solver, relax_type); 
	 }
         if (relax_down > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType_pre(precision0, amg_solver, relax_coarse, 3);
         }
         HYPRE_BoomerAMGSetRelaxOrder_pre(precision0, amg_solver, relax_order);
         HYPRE_BoomerAMGSetMaxLevels_pre(precision0, amg_solver, max_levels);
         HYPRE_BoomerAMGSetDebugFlag_pre(precision0, amg_solver, debug_flag);
         HYPRE_BoomerAMGSetAggNumLevels_pre(precision0, amg_solver, agg_num_levels);
         HYPRE_BoomerAMGSetAggInterpType_pre(precision0, amg_solver, agg_interp_type);
         HYPRE_BoomerAMGSetNumPaths_pre(precision0, amg_solver, num_paths);
         HYPRE_BoomerAMGSetTol_pre(precision0, amg_solver, zero); 
         HYPRE_BoomerAMGSetMaxRowSum_pre(precision0, amg_solver, max_row_sum);
         HYPRE_BoomerAMGSetStrongThreshold_pre(precision0, amg_solver, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor_pre(precision0, amg_solver, trunc_factor);
         HYPRE_BoomerAMGSetRelaxWt_pre(precision0, amg_solver, relax_wt);
         HYPRE_BoomerAMGSetOuterWt_pre(precision0, amg_solver, outer_wt);
         HYPRE_BoomerAMGSetAggTruncFactor_pre(precision0, amg_solver, agg_trunc_factor);
         HYPRE_BoomerAMGSetAggP12TruncFactor_pre(precision0, amg_solver, agg_P12_trunc_factor);
         HYPRE_BoomerAMGSetAggPMaxElmts_pre(precision0, amg_solver, agg_P_max_elmts);
         HYPRE_BoomerAMGSetAggP12MaxElmts_pre(precision0, amg_solver, agg_P12_max_elmts);
         if (ns_coarse > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_coarse, 3);
	 }
         if (ns_down > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_down,   1);
         }
         if (ns_up > -1)
         {
            HYPRE_BoomerAMGSetCycleNumSweeps_pre(precision0, amg_solver, ns_up,     2);
         }
         HYPRE_MPAMGSetPrecisionArray_mp(amg_solver, precision_array);
         // Set the preconditioner for BiCGSTAB (single precision matrix)
         if (precision0 == HYPRE_REAL_SINGLE)
	 {
	    HYPRE_BiCGSTABSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_flt);
            HYPRE_BiCGSTABSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGPrecSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGPrecSetup_mp,
                                amg_solver);
	 }
	 else if (precision0 == HYPRE_REAL_DOUBLE)
	 {
            HYPRE_BiCGSTABSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)A_dbl);
            HYPRE_BiCGSTABSetPrecond_dbl(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSolve_mp,
                                (HYPRE_PtrToSolverFcn) HYPRE_MPAMGSetup_mp,
                                amg_solver);
	 }
         // Set the preconditioner for BiCGSTAB.
         // This actually sets a pointer to a single precision AMG solver.
         // The setup and solve functions just allow us to accept double precision
         // rhs and sol vectors from the BiCGSTAB solver to do the preconditioner solve.        

         HYPRE_BiCGSTABGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf("HYPRE_ParCSRBiCGSTABGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf("HYPRE_ParCSRBiCGSTABGetPrecond got good precond\n");
         }
      }
      
      // Setup BiCGSTAB solver (double precision)
      HYPRE_BiCGSTABSetup_dbl(pcg_solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Mixed precision Setup Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);

      time_index = hypre_InitializeTiming("DBL Solve");
      MPI_Barrier(MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      //  BiCGSTAB solve (double precision)
      HYPRE_BiCGSTABSolve_dbl(pcg_solver, (HYPRE_Matrix)A_dbl,  (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

      MPI_Barrier(MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Mixed precision Solve Time", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BiCGSTABGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
        hypre_printf("final relative residual norm = %e \n", dfinal_res_norm);
        hypre_printf("Iteration count = %d \n", num_iterations);         
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRBiCGSTABDestroy_dbl(pcg_solver);
      HYPRE_BoomerAMGDestroy(amg_solver);
   }   
    
   /* Clean up */
   HYPRE_IJVectorDestroy_flt(ij_b_flt);
   HYPRE_IJVectorDestroy_flt(ij_x_flt);

   HYPRE_IJVectorDestroy_dbl(ij_b_dbl);
   HYPRE_IJVectorDestroy_dbl(ij_x_dbl);

   /* Finalize MPI*/
   MPI_Finalize();
final:

   return(0);
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
   double         *values_dbl;
   float          *values_flt;

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

   A_dbl = (HYPRE_ParCSRMatrix) GenerateLaplacian_dbl(MPI_COMM_WORLD,
                                                 nx, ny, nz, P, Q, R, p, q, r, values_dbl);
   A_flt = (HYPRE_ParCSRMatrix) GenerateLaplacian_flt(MPI_COMM_WORLD,
                                                 nx, ny, nz, P, Q, R, p, q, r, values_flt);
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


