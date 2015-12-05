/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.30 $
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

/* begin lobpcg */

#define NO_SOLVER -9198

#include <assert.h>
#include <time.h>

#include "fortran_matrix.h"
#include "HYPRE_lobpcg.h"

#include "interpreter.h"
#include "multivector.h"
#include "HYPRE_MatvecFunctions.h"

HYPRE_Int
BuildParIsoLaplacian( HYPRE_Int argc, char** argv, HYPRE_ParCSRMatrix *A_ptr );

/* end lobpcg */

HYPRE_Int BuildParFromFile (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParDifConv (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParFromOneFile2(HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_Int num_functions , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildFuncsFromFiles (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix A , HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildFuncsFromOneFile (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix A , HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildRhsParFromOneFile2(HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_Int *partitioning , HYPRE_ParVector *b_ptr );
HYPRE_Int BuildParLaplacian9pt (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian27pt (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );

#define SECOND_TIME 0
 
hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int                 arg_index;
   HYPRE_Int                 print_usage;
   HYPRE_Int                 sparsity_known = 0;
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
   HYPRE_Int                 i,j,k; 
   HYPRE_Int                 indx, rest, tms;
   HYPRE_Int                 max_levels = 25;
   HYPRE_Int                 num_iterations; 
   HYPRE_Int                 pcg_num_its; 
   HYPRE_Int                 dscg_num_its; 
   HYPRE_Int                 pcg_max_its; 
   HYPRE_Int                 dscg_max_its; 
   double              cf_tol = 0.9;
   double              norm;
   double              final_res_norm;
   void               *object;

   HYPRE_IJMatrix      ij_A; 
   HYPRE_IJVector      ij_b;
   HYPRE_IJVector      ij_x;

   HYPRE_ParCSRMatrix  parcsr_A;
   HYPRE_ParVector     b;
   HYPRE_ParVector     x;

   HYPRE_Solver        amg_solver;
   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond, pcg_precond_gotten;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 local_row;
   HYPRE_Int                *row_sizes;
   HYPRE_Int                *diag_sizes;
   HYPRE_Int                *offdiag_sizes;
   HYPRE_Int                 size;
   HYPRE_Int                *col_inds;

   HYPRE_Int		       time_index;
   MPI_Comm            comm = hypre_MPI_COMM_WORLD;
   HYPRE_Int M, N;
   HYPRE_Int first_local_row, last_local_row, local_num_rows;
   HYPRE_Int first_local_col, last_local_col, local_num_cols;
   HYPRE_Int local_num_vars;
   HYPRE_Int variant, overlap, domain_type;
   double schwarz_rlx_weight;
   double *values;

   const double dt_inf = 1.e40;
   double dt = dt_inf;

   /* parameters for BoomerAMG */
   double   strong_threshold;
   double   trunc_factor = 0.0;
   double   jacobi_trunc_threshold = 0.0;
   HYPRE_Int      P_max_elmts = 0;
   HYPRE_Int      cycle_type;
   HYPRE_Int      coarsen_type = 6;
   HYPRE_Int      hybrid = 1;
   HYPRE_Int      measure_type = 0;
   HYPRE_Int     *num_grid_sweeps;  
   HYPRE_Int     *grid_relax_type;   
   HYPRE_Int    **grid_relax_points;
/* HYPRE_Int	    smooth_lev; */
/* HYPRE_Int	    smooth_rlx = 8; */
   HYPRE_Int	    smooth_type = 6;
   HYPRE_Int	    smooth_num_levels = 0;
   HYPRE_Int      relax_default;
   HYPRE_Int      smooth_num_sweeps = 1;
   HYPRE_Int      num_sweep = 1;
   double  *relax_weight; 
   double  *omega;
   double   tol = 1.e-8, pc_tol = 0.;
   double   max_row_sum = 1.;
   HYPRE_Int      interp_type  = 0; /* default value */
   HYPRE_Int      post_interp_type  = 0; 
   HYPRE_Int     *dof_func;
   HYPRE_Int      num_functions = 1;
   HYPRE_Int      num_paths = 1;
   HYPRE_Int      agg_num_levels = 0;
   /* for CGC BM Aug 25, 2006 */
   HYPRE_Int      cgcits = 1;

   /* parameters for ParaSAILS */
   double   sai_threshold = 0.1;
   double   sai_filter = 0.1;

   /* parameters for PILUT */
   double   drop_tol = -1;
   HYPRE_Int      nonzeros_to_keep = -1;

   /* parameters for GMRES */
   HYPRE_Int	    k_dim;

   /* parameters for GSMG */
   HYPRE_Int      gsmg_samples = 5;

   HYPRE_Int      print_system = 0;

   /* begin lobpcg */

   HYPRE_Int lobpcgFlag = 0;
   HYPRE_Int lobpcgGen = 0;
   HYPRE_Int constrained = 0;
   HYPRE_Int vFromFileFlag = 0;
   HYPRE_Int lobpcgSeed = 0;
   HYPRE_Int blockSize = 1;
   HYPRE_Int verbosity = 1;
   HYPRE_Int iterations;
   HYPRE_Int maxIterations = 100;
   HYPRE_Int checkOrtho = 0;
   HYPRE_Int printLevel = 0; /* also c.f. poutdat */
   HYPRE_Int two_norm = 1;
   HYPRE_Int pcgIterations = 0;
   HYPRE_Int pcgMode = 1;
   double pcgTol = 1e-2;
   double nonOrthF;
   
   FILE* filePtr;

   mv_MultiVectorPtr eigenvectors = NULL;
   mv_MultiVectorPtr constraints = NULL;
   mv_MultiVectorPtr workspace = NULL;

   double* eigenvalues = NULL;

   double* residuals;
   utilities_FortranMatrix* residualNorms;
   utilities_FortranMatrix* residualNormsHistory;
   utilities_FortranMatrix* eigenvaluesHistory;
   utilities_FortranMatrix* printBuffer;
   utilities_FortranMatrix* gramXX;
   utilities_FortranMatrix* identity;

   HYPRE_Solver        lobpcg_solver;

   mv_InterfaceInterpreter* interpreter;
   HYPRE_MatvecFunctions matvec_fn;

   HYPRE_IJMatrix      ij_B; 
   HYPRE_ParCSRMatrix  parcsr_B;

   /* end lobpcg */

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
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
   relax_default = 3;
   debug_flag = 0;

   solver_id = 0;

   ioutdat = 3;
   poutdat = 1;

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
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;

	 /* begin lobpcg */
	 if ( strcmp(argv[arg_index], "none") == 0 ) {
	   solver_id = NO_SOLVER;
	   arg_index++;
	 }
	 else /* end lobpcg */
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
      else if ( strcmp(argv[arg_index], "-ruge") == 0 )
      {
         arg_index++;
         coarsen_type      = 1;
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
      else if ( strcmp(argv[arg_index], "-hmis") == 0 )
      {
         arg_index++;
         coarsen_type      = 10;
      }
      else if ( strcmp(argv[arg_index], "-ruge1p") == 0 )
      {
         arg_index++;
         coarsen_type      = 11;
      }
      else if ( strcmp(argv[arg_index], "-agg_nl") == 0 )
      {
         arg_index++;
         agg_num_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-np") == 0 )
      {
         arg_index++;
         num_paths = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nohybrid") == 0 )
      {
         arg_index++;
         hybrid      = -1;
      }    
      else if ( strcmp(argv[arg_index], "-gm") == 0 )
      {
         arg_index++;
         measure_type      = 1;
      }    
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index++;
         relax_default = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-ns") == 0 )
      {
         arg_index++;
         num_sweep = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sns") == 0 )
      {
         arg_index++;
         smooth_num_sweeps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dt") == 0 )
      {
         arg_index++;
         dt = atof(argv[arg_index++]);
         build_rhs_type = -1;
         if ( build_src_type == -1 ) build_src_type = 2;
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      /* begin lobpcg */
      else if ( strcmp(argv[arg_index], "-lobpcg") == 0 ) 
	{	      				 /* use lobpcg */
	  arg_index++;
	  lobpcgFlag = 1;
	}
      else if ( strcmp(argv[arg_index], "-gen") == 0 ) 
	{	      				 /* generalized evp */
	  arg_index++;
	  lobpcgGen = 1;
	}
      else if ( strcmp(argv[arg_index], "-con") == 0 ) 
	{	      				 /* constrained evp */
	  arg_index++;
	  constrained = 1;
	}
      else if ( strcmp(argv[arg_index], "-orthchk") == 0 )
	{			/* lobpcg: check orthonormality */
	  arg_index++;
	  checkOrtho = 1;
	}
      else if ( strcmp(argv[arg_index], "-vfromfile") == 0 )
	{		 /* lobpcg: get initial vectors from file */
	  arg_index++;
	  vFromFileFlag = 1;
	}
      else if ( strcmp(argv[arg_index], "-vrand") == 0 ) 
	{				/* lobpcg: block size */
	  arg_index++;
	  blockSize = atoi(argv[arg_index++]);
	}
      else if ( strcmp(argv[arg_index], "-seed") == 0 )
	{		           /* lobpcg: seed for srand */
	  arg_index++;
	  lobpcgSeed = atoi(argv[arg_index++]);
	}
      else if ( strcmp(argv[arg_index], "-itr") == 0 ) 
	{		     /* lobpcg: max # of iterations */
	  arg_index++;
	  maxIterations = atoi(argv[arg_index++]);
	}
      else if ( strcmp(argv[arg_index], "-verb") == 0 ) 
	{			  /* lobpcg: verbosity level */
	  arg_index++;
	  verbosity = atoi(argv[arg_index++]);
	}
      else if ( strcmp(argv[arg_index], "-vout") == 0 )
	{			      /* lobpcg: print level */
	  arg_index++;
	  printLevel = atoi(argv[arg_index++]);
	}
      else if ( strcmp(argv[arg_index], "-pcgitr") == 0 ) 
	{		       /* lobpcg: inner pcg iterations */
	  arg_index++;
	  pcgIterations = atoi(argv[arg_index++]);
	}
      else if ( strcmp(argv[arg_index], "-pcgtol") == 0 ) 
	{		       /* lobpcg: inner pcg iterations */
	  arg_index++;
	  pcgTol = atof(argv[arg_index++]);
	}
      else if ( strcmp(argv[arg_index], "-pcgmode") == 0 ) 
	{		 /* lobpcg: initial guess for inner pcg */
	  arg_index++;	      /* 0: zero, otherwise rhs */
	  pcgMode = atoi(argv[arg_index++]);
	}
      /* end lobpcg */
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

   /* begin lobpcg */

   if ( solver_id == 0 && lobpcgFlag )
     solver_id = 1;

   /* end lobpcg */

   if (solver_id == 8 || solver_id == 18)
   {
     max_levels = 1;
   }

   /* defaults for BoomerAMG */
   if (solver_id == 0 || solver_id == 1 || solver_id == 3 || solver_id == 5
	|| solver_id == 9 || solver_id == 13 || solver_id == 14
 	|| solver_id == 15 || solver_id == 20)
   {
   strong_threshold = 0.25;
   trunc_factor = 0.;
   cycle_type = 1;

   num_grid_sweeps   = hypre_CTAlloc(HYPRE_Int,4);
   grid_relax_type   = hypre_CTAlloc(HYPRE_Int,4);
   grid_relax_points = hypre_CTAlloc(HYPRE_Int *,4);
   relax_weight      = hypre_CTAlloc(double, max_levels);
   omega      = hypre_CTAlloc(double, max_levels);

   for (i=0; i < max_levels; i++)
   {
	relax_weight[i] = 1.;
	omega[i] = 1.;
   }

   /* for CGNR preconditioned with Boomeramg, only relaxation scheme 0 is
      implemented, i.e. Jacobi relaxation, and it needs to be used without
      CF ordering */
   if (solver_id == 5) 
   {
      /* fine grid */
      relax_default = 7;
      grid_relax_type[0] = relax_default; 
      num_grid_sweeps[0] = num_sweep;
      grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, num_sweep); 
      for (i=0; i<num_sweep; i++)
      {
         grid_relax_points[0][i] = 0;
      } 
      /* down cycle */
      grid_relax_type[1] = relax_default; 
       num_grid_sweeps[1] = num_sweep;
      grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, num_sweep); 
      for (i=0; i<num_sweep; i++)
      {
         grid_relax_points[1][i] = 0;
      } 
      /* up cycle */
      grid_relax_type[2] = relax_default; 
      num_grid_sweeps[2] = num_sweep;
      grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, num_sweep); 
      for (i=0; i<num_sweep; i++)
      {
         grid_relax_points[2][i] = 0;
      } 
   }
   else if (coarsen_type == 5)
   {
      /* fine grid */
      num_grid_sweeps[0] = 3;
      grid_relax_type[0] = relax_default; 
      grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, 3); 
      grid_relax_points[0][0] = -2;
      grid_relax_points[0][1] = -1;
      grid_relax_points[0][2] = 1;
   
      /* down cycle */
      num_grid_sweeps[1] = 4;
      grid_relax_type[1] = relax_default; 
      grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, 4); 
      grid_relax_points[1][0] = -1;
      grid_relax_points[1][1] = 1;
      grid_relax_points[1][2] = -2;
      grid_relax_points[1][3] = -2;
   
      /* up cycle */
      num_grid_sweeps[2] = 4;
      grid_relax_type[2] = relax_default; 
      grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, 4); 
      grid_relax_points[2][0] = -2;
      grid_relax_points[2][1] = -2;
      grid_relax_points[2][2] = 1;
      grid_relax_points[2][3] = -1;
   }
   else
   {   
      /* fine grid */
      grid_relax_type[0] = relax_default; 
      /*num_grid_sweeps[0] = num_sweep;
      grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, num_sweep); 
      for (i=0; i<num_sweep; i++)
      {
         grid_relax_points[0][i] = 0;
      } */
      num_grid_sweeps[0] = 2*num_sweep;
      grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, 2*num_sweep); 
      for (i=0; i<2*num_sweep; i+=2)
      {
         grid_relax_points[0][i] = 1;
         grid_relax_points[0][i+1] = -1;
      } 

      /* down cycle */
      grid_relax_type[1] = relax_default; 
      /* num_grid_sweeps[1] = num_sweep;
      grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, num_sweep); 
      for (i=0; i<num_sweep; i++)
      {
         grid_relax_points[1][i] = 0;
      } */
      num_grid_sweeps[1] = 2*num_sweep;
      grid_relax_type[1] = relax_default; 
      grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, 2*num_sweep); 
      for (i=0; i<2*num_sweep; i+=2)
      {
         grid_relax_points[1][i] = 1;
         grid_relax_points[1][i+1] = -1;
      }

      /* up cycle */
      grid_relax_type[2] = relax_default; 
      /* num_grid_sweeps[2] = num_sweep;
      grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, num_sweep); 
      for (i=0; i<num_sweep; i++)
      {
         grid_relax_points[2][i] = 0;
      } */
      num_grid_sweeps[2] = 2*num_sweep;
      grid_relax_type[2] = relax_default; 
      grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, 2*num_sweep); 
      for (i=0; i<2*num_sweep; i+=2)
      {
         grid_relax_points[2][i] = -1;
         grid_relax_points[2][i+1] = 1;
      }
   }

   /* coarsest grid */
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 9;
   grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int, 1);
   grid_relax_points[3][0] = 0;
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
        if (solver_id == 0 || solver_id == 1 || solver_id == 3 
		|| solver_id == 5 || solver_id == 13 || solver_id == 14
		|| solver_id == 15 || solver_id == 9 || solver_id == 20)
        {
         relax_weight[0] = atof(argv[arg_index++]);
         for (i=1; i < max_levels; i++)
	   relax_weight[i] = relax_weight[0];
        }
      }
      else if ( strcmp(argv[arg_index], "-om") == 0 )
      {
         arg_index++;
        if (solver_id == 0 || solver_id == 1 || solver_id == 3 
		|| solver_id == 5 || solver_id == 13 || solver_id == 14
		|| solver_id == 15 || solver_id == 9 || solver_id == 20)
        {
         omega[0] = atof(argv[arg_index++]);
         for (i=1; i < max_levels; i++)
	   omega[i] = omega[0];
        }
      }
      else if ( strcmp(argv[arg_index], "-sw") == 0 )
      {
         arg_index++;
         schwarz_rlx_weight = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         strong_threshold  = atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
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
      hypre_printf("\n");
      hypre_printf("  -cljp                 : CLJP coarsening \n");
      hypre_printf("  -cgc                  : CGC coarsening \n");
      hypre_printf("  -cgce                 : CGC-E coarsening \n");
      hypre_printf("  -ruge                 : Ruge coarsening (local)\n");
      hypre_printf("  -ruge3                : third pass on boundary\n");
      hypre_printf("  -ruge3c               : third pass on boundary, keep c-points\n");
      hypre_printf("  -ruge2b               : 2nd pass is global\n");
      hypre_printf("  -rugerlx              : relaxes special points\n");
      hypre_printf("  -falgout              : local ruge followed by LJP\n");
      hypre_printf("  -nohybrid             : no switch in coarsening\n");
      hypre_printf("  -gm                   : use global measures\n");
      hypre_printf("\n");
      hypre_printf("  -rlx  <val>            : relaxation type\n");
      hypre_printf("       0=Weighted Jacobi  \n");
      hypre_printf("       1=Gauss-Seidel (very slow!)  \n");
      hypre_printf("       3=Hybrid Jacobi/Gauss-Seidel  \n");
      hypre_printf("  -ns <val>              : Use <val> sweeps on each level\n");
      hypre_printf("                           (default C/F down, F/C up, F/C fine\n");
      hypre_printf("\n"); 
      hypre_printf("  -mu   <val>            : set AMG cycles (1=V, 2=W, etc.)\n"); 
      hypre_printf("  -th   <val>            : set AMG threshold Theta = val \n");
      hypre_printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
      hypre_printf("  -Pmx  <val>            : set maximal no. of elmts per row for AMG interpolation \n");
      hypre_printf("  -jtr  <val>            : set truncation threshold for Jacobi interpolation = val \n");

      hypre_printf("  -mxrs <val>            : set AMG maximum row sum threshold for dependency weakening \n");
      hypre_printf("  -nf <val>              : set number of functions for systems AMG\n");
      hypre_printf("  -numsamp <val>         : set number of sample vectors for GSMG\n");
      hypre_printf("  -interptype <val>      : set to 1 to get LS interpolation (for GSMG only)\n");
      hypre_printf("                         : set to 2 to get interpolation for hyperbolic equations\n");
      hypre_printf("                         : set to 3 to get direct interpolation (with weight separation)\n");
      hypre_printf("                         : set to 4 to get multipass interpolation\n");
      hypre_printf("                         : set to 5 to get multipass interpolation with weight separation\n");
      hypre_printf("                         : set to 6 to get extended interpolation\n");
      hypre_printf("                         : set to 7 to get FF interpolation\n");
      hypre_printf("                         : set to 8 to get standard interpolation\n");
      hypre_printf("                         : set to 9 to get standard interpolation with weight separation\n");
      hypre_printf("                         : set to 10 for nodal standard interpolation (for systems only) \n");
      hypre_printf("                         : set to 11 for diagonal nodal standard interpolation (for systems only) \n");
      hypre_printf("  -postinterptype <val>  : invokes <val> no. of Jacobi interpolation steps after main interpolation\n");
      hypre_printf("  -cgcitr <val>          : set maximal number of coarsening iterations for CGC\n");
      hypre_printf("  -solver_type <val>     : sets solver within Hybrid solver\n");
      hypre_printf("                         : 1  PCG  (default)\n");
      hypre_printf("                         : 2  GMRES\n");
      hypre_printf("                         : 3  BiCGSTAB\n");
     
      hypre_printf("  -w   <val>             : set Jacobi relax weight = val\n");
      hypre_printf("  -k   <val>             : dimension Krylov space for GMRES\n");
      hypre_printf("  -mxl  <val>            : maximum number of levels (AMG, ParaSAILS)\n");
      hypre_printf("  -tol  <val>            : set solver convergence tolerance = val\n");
      hypre_printf("\n");
      hypre_printf("  -sai_th   <val>        : set ParaSAILS threshold = val \n");
      hypre_printf("  -sai_filt <val>        : set ParaSAILS filter = val \n");
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

      /* begin lobpcg */

      hypre_printf("LOBPCG options:\n");
      hypre_printf("\n");
      hypre_printf("  -lobpcg                 : run LOBPCG instead of PCG\n");
      hypre_printf("\n");
      hypre_printf("  -gen                    : solve generalized EVP with B = Laplacian\n");
      hypre_printf("\n");
      hypre_printf("  -con                    : solve constrained EVP using 'vectors.*.*'\n");
      hypre_printf("                            as constraints (see -vout 1 below)\n");
      hypre_printf("\n");
      hypre_printf("  -solver none            : no HYPRE preconditioner is used\n");
      hypre_printf("\n");
      hypre_printf("  -itr <val>              : maximal number of LOBPCG iterations\n");
      hypre_printf("                            (default 100);\n");
      hypre_printf("\n");
      hypre_printf("  -vrand <val>            : compute <val> eigenpairs using random\n");
      hypre_printf("                            initial vectors (default 1)\n");
      hypre_printf("\n");
      hypre_printf("  -seed <val>             : use <val> as the seed for the random\n");
      hypre_printf("                            number generator(default seed is based\n");
      hypre_printf("                            on the time of the run)\n");
      hypre_printf("\n");
      hypre_printf("  -vfromfile              : read initial vectors from files\n"); 
      hypre_printf("                            vectors.i.j where i is vector number\n");
      hypre_printf("                            and j is processor number\n");
      hypre_printf("\n");
      hypre_printf("  -orthchk                : check eigenvectors for orthonormality\n");
      hypre_printf("\n");
      hypre_printf("  -verb <val>             : verbosity level\n");
      hypre_printf("  -verb 0                 : no print\n");
      hypre_printf("  -verb 1                 : print initial eigenvalues and residuals,\n");
      hypre_printf("                            the iteration number, the number of\n");
      hypre_printf("                            non-convergent eigenpairs and final\n");
      hypre_printf("                            eigenvalues and residuals (default)\n");
      hypre_printf("  -verb 2                 : print eigenvalues and residuals on each\n");
      hypre_printf("                            iteration\n");
      hypre_printf("\n");
      hypre_printf("  -pcgitr <val>           : maximal number of inner PCG iterations\n");
      hypre_printf("                            for preconditioning (default 1);\n");
      hypre_printf("                            if <val> = 0 then the preconditioner\n");
      hypre_printf("                            is applied directly\n");
      hypre_printf("\n");
      hypre_printf("  -pcgtol <val>           : residual tolerance for inner iterations\n");
      hypre_printf("                            (default 0.01)\n");
      hypre_printf("\n");
      hypre_printf("  -vout <val>             : file output level\n");
      hypre_printf("  -vout 0                 : no files created (default)\n");
      hypre_printf("  -vout 1                 : write eigenvalues to values.txt, residuals\n");
      hypre_printf("                            to residuals.txt and eigenvectors to \n");
      hypre_printf("                            vectors.i.j where i is vector number\n");
      hypre_printf("                            and j is processor number\n");
      hypre_printf("  -vout 2                 : in addition to the above, write the\n");
      hypre_printf("                            eigenvalues history (the matrix whose\n");
      hypre_printf("                            i-th column contains eigenvalues at\n");
      hypre_printf("                            (i+1)-th iteration) to val_hist.txt and\n");
      hypre_printf("                            residuals history to res_hist.txt\n");
      hypre_printf("\nNOTE: in this test driver LOBPCG only works with solvers 1, 2, 8, 12, 14 and 43\n");
      hypre_printf("\ndefault solver is 1\n");
      hypre_printf("\n");

      /* end lobpcg */

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

   if ( build_matrix_type == -1 )
   {
      HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                          HYPRE_PARCSR, &ij_A );
   }
   else if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParFromOneFile2(argc, argv, build_matrix_arg_index, num_functions,
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
   else
   {
      hypre_printf("You have asked for an unsupported problem with\n");
      hypre_printf("build_matrix_type = %d.\n", build_matrix_type);
      return(-1);
   }

   time_index = hypre_InitializeTiming("Spatial operator");
   hypre_BeginTiming(time_index);

   if (build_matrix_type < 0)
   {
     ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
     parcsr_A = (HYPRE_ParCSRMatrix) object;

     ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
               &first_local_row, &last_local_row ,
               &first_local_col, &last_local_col );

     local_num_rows = last_local_row - first_local_row + 1;
     local_num_cols = last_local_col - first_local_col + 1;
   }
   else
   {

     /*-----------------------------------------------------------
      * Copy the parcsr matrix into the IJMatrix through interface calls
      *-----------------------------------------------------------*/
 
     ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
               &first_local_row, &last_local_row ,
               &first_local_col, &last_local_col );

     local_num_rows = last_local_row - first_local_row + 1;
     local_num_cols = last_local_col - first_local_col + 1;
     ierr += HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );

     ierr += HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                   first_local_col, last_local_col,
                                   &ij_A );

     ierr += HYPRE_IJMatrixSetObjectType( ij_A, HYPRE_PARCSR );
  

/* the following shows how to build an IJMatrix if one has only an
   estimate for the row sizes */
     if (sparsity_known == 1)
     {   
/*  build IJMatrix using exact row_sizes for diag and offdiag */

       diag_sizes = hypre_CTAlloc(HYPRE_Int, local_num_rows);
       offdiag_sizes = hypre_CTAlloc(HYPRE_Int, local_num_rows);
       local_row = 0;
       for (i=first_local_row; i<= last_local_row; i++)
       {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size, 
                                           &col_inds, &values );
 
         for (j=0; j < size; j++)
         {
           if (col_inds[j] < first_local_row || col_inds[j] > last_local_row)
             offdiag_sizes[local_row]++;
           else
             diag_sizes[local_row]++;
         }
         local_row++;
         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size, 
                                               &col_inds, &values );
       }
       ierr += HYPRE_IJMatrixSetDiagOffdSizes( ij_A, 
                                        (const HYPRE_Int *) diag_sizes,
                                        (const HYPRE_Int *) offdiag_sizes );
       hypre_TFree(diag_sizes);
       hypre_TFree(offdiag_sizes);
     
       ierr = HYPRE_IJMatrixInitialize( ij_A );

       for (i=first_local_row; i<= last_local_row; i++)
       {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                           &col_inds, &values );

         ierr += HYPRE_IJMatrixSetValues( ij_A, 1, &size, &i,
                                          (const HYPRE_Int *) col_inds,
                                          (const double *) values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                               &col_inds, &values );
       }
     }
     else
     {
       row_sizes = hypre_CTAlloc(HYPRE_Int, local_num_rows);

       size = 5; /* this is in general too low, and supposed to test
                    the capability of the reallocation of the interface */ 

       if (sparsity_known == 0) /* tries a more accurate estimate of the
                                    storage */
       {
         if (build_matrix_type == 2) size = 7;
         if (build_matrix_type == 3) size = 9;
         if (build_matrix_type == 4) size = 27;
       }

       for (i=0; i < local_num_rows; i++)
         row_sizes[i] = size;

       ierr = HYPRE_IJMatrixSetRowSizes ( ij_A, (const HYPRE_Int *) row_sizes );

       hypre_TFree(row_sizes);

       ierr = HYPRE_IJMatrixInitialize( ij_A );

       /* Loop through all locally stored rows and insert them into ij_matrix */
       for (i=first_local_row; i<= last_local_row; i++)
       {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                           &col_inds, &values );

         ierr += HYPRE_IJMatrixSetValues( ij_A, 1, &size, &i,
                                          (const HYPRE_Int *) col_inds,
                                          (const double *) values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                               &col_inds, &values );
       }
     }

     ierr += HYPRE_IJMatrixAssemble( ij_A );

   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Matrix Setup", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   if (ierr)
   {
     hypre_printf("Error in driver building IJMatrix from parcsr matrix. \n");
     return(-1);
   }

   /* This is to emphasize that one can IJMatrixAddToValues after an
      IJMatrixRead or an IJMatrixAssemble.  After an IJMatrixRead,
      assembly is unnecessary if the sparsity pattern of the matrix is
      not changed somehow.  If one has not used IJMatrixRead, one has
      the opportunity to IJMatrixAddTo before a IJMatrixAssemble. */


   if (build_matrix_type > -1)
   {
       ierr = HYPRE_IJMatrixInitialize( ij_A );

       /* Loop through all locally stored rows and insert them into ij_matrix */
       for (i=first_local_row; i<= last_local_row; i++)
       {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                           &col_inds, &values );

         ierr += HYPRE_IJMatrixSetValues( ij_A, 1, &size, &i,
                                          (const HYPRE_Int *) col_inds,
                                          (const double *) values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                               &col_inds, &values );
       }

   /* If sparsity pattern is not changed since last IJMatrixAssemble call,
      this should be a no-op */

   ierr += HYPRE_IJMatrixAssemble( ij_A );

   /*-----------------------------------------------------------
    * Fetch the resulting underlying matrix out
    *-----------------------------------------------------------*/

     ierr += HYPRE_ParCSRMatrixDestroy(parcsr_A);

   ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
   parcsr_A = (HYPRE_ParCSRMatrix) object;
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
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 1 )
   {
      hypre_printf("build_rhs_type == 1 not currently implemented\n");
      return(-1);

#if 0
/* RHS */
      BuildRhsParFromOneFile2(argc, argv, build_rhs_arg_index, part_b, &b);
#endif
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

      values = hypre_CTAlloc(double, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
         values[i] = 1.0;
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
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

/* For purposes of this test, HYPRE_ParVector functions are used, but these are 
   not necessary.  For a clean use of the interface, the user "should"
   modify components of ij_x by using functions HYPRE_IJVectorSetValues or
   HYPRE_IJVectorAddToValues */

      HYPRE_ParVectorSetRandomValues(b, 22775);
      HYPRE_ParVectorInnerProd(b,b,&norm);
      norm = 1./sqrt(norm);
      ierr = HYPRE_ParVectorScale(norm, b);      

/* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
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

      values = hypre_CTAlloc(double, local_num_cols);
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
      values = hypre_CTAlloc(double, local_num_cols);
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

      values = hypre_CTAlloc(double, local_num_rows);
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

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 1.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

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

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial unknown vector */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 1 )
   {
      hypre_printf("build_src_type == 1 not currently implemented\n");
      return(-1);

#if 0
      BuildRhsParFromOneFile2(argc, argv, build_src_arg_index, part_b, &b);
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

      values = hypre_CTAlloc(double, local_num_rows);
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
      values = hypre_CTAlloc(double, local_num_cols);
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
      values = hypre_CTAlloc(double, local_num_rows);

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
      values = hypre_CTAlloc(double, local_num_cols);
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

      values = hypre_CTAlloc(double, local_num_rows);
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
      values = hypre_CTAlloc(double, local_num_cols);
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
   
   /* HYPRE_IJMatrixPrint(ij_A, "driver.out.A");
   HYPRE_IJVectorPrint(ij_x, "driver.out.x0"); */

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
         local_num_vars = local_num_rows;
         dof_func = hypre_CTAlloc(HYPRE_Int,local_num_vars);
         if (myid == 0)
	    hypre_printf (" Number of unknown functions = %d \n", num_functions);
         rest = first_local_row-((first_local_row/num_functions)*num_functions);
         indx = num_functions-rest;
         if (rest == 0) indx = 0;
         k = num_functions - 1;
         for (j = indx-1; j > -1; j--)
	    dof_func[j] = k--;
         tms = local_num_vars/num_functions;
         if (tms*num_functions+indx > local_num_vars) tms--;
         for (j=0; j < tms; j++)
         {
	    for (k=0; k < num_functions; k++)
	       dof_func[indx++] = k;
         }
         k = 0;
         while (indx < local_num_vars)
	    dof_func[indx++] = k++;
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
   }

   /*-----------------------------------------------------------
    * Solve the system using the hybrid solver
    *-----------------------------------------------------------*/

   if (solver_id == 20)
   {
      dscg_max_its = 1000;
      pcg_max_its = 200;
      if (myid == 0) hypre_printf("Solver:  AMG\n");
      time_index = hypre_InitializeTiming("AMG_hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRHybridCreate(&amg_solver); 
      HYPRE_ParCSRHybridSetTol(amg_solver, tol);
      HYPRE_ParCSRHybridSetConvergenceTol(amg_solver, cf_tol);
      HYPRE_ParCSRHybridSetSolverType(amg_solver, solver_type);
      HYPRE_ParCSRHybridSetLogging(amg_solver, ioutdat);
      HYPRE_ParCSRHybridSetPrintLevel(amg_solver, poutdat);
      HYPRE_ParCSRHybridSetDSCGMaxIter(amg_solver, dscg_max_its );
      HYPRE_ParCSRHybridSetPCGMaxIter(amg_solver, pcg_max_its ); 
      HYPRE_ParCSRHybridSetCoarsenType(amg_solver, (hybrid*coarsen_type));
      HYPRE_ParCSRHybridSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_ParCSRHybridSetTruncFactor(amg_solver, trunc_factor);
      HYPRE_ParCSRHybridSetNumGridSweeps(amg_solver, num_grid_sweeps);
      HYPRE_ParCSRHybridSetGridRelaxType(amg_solver, grid_relax_type);
      HYPRE_ParCSRHybridSetRelaxWeight(amg_solver, relax_weight);
      HYPRE_ParCSRHybridSetOmega(amg_solver, omega);
      HYPRE_ParCSRHybridSetGridRelaxPoints(amg_solver, grid_relax_points);
      HYPRE_ParCSRHybridSetMaxLevels(amg_solver, max_levels);
      HYPRE_ParCSRHybridSetMaxRowSum(amg_solver, max_row_sum);
  
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
      if (myid == 0) hypre_printf("Solver:  AMG\n");
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGCreate(&amg_solver); 
      /* BM Aug 25, 2006 */
      HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
      HYPRE_BoomerAMGSetInterpType(amg_solver, interp_type);
      HYPRE_BoomerAMGSetNumSamples(amg_solver, gsmg_samples);
      HYPRE_BoomerAMGSetCoarsenType(amg_solver, (hybrid*coarsen_type));
      HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      HYPRE_BoomerAMGSetTol(amg_solver, tol);
      HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
      HYPRE_BoomerAMGSetPMaxElmts(amg_solver, P_max_elmts);
      HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
      HYPRE_BoomerAMGSetJacobiTruncThreshold(amg_solver, jacobi_trunc_threshold);
/* note: log is written to standard output, not to file */
      HYPRE_BoomerAMGSetPrintLevel(amg_solver, 3);
      HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log"); 
      HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      HYPRE_BoomerAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      HYPRE_BoomerAMGSetGridRelaxType(amg_solver, grid_relax_type);
      HYPRE_BoomerAMGSetRelaxWeight(amg_solver, relax_weight);
      HYPRE_BoomerAMGSetOmega(amg_solver, omega);
      HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
      HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
      HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, smooth_num_levels);
      HYPRE_BoomerAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      HYPRE_BoomerAMGSetMaxRowSum(amg_solver, max_row_sum);
      HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);
      HYPRE_BoomerAMGSetVariant(amg_solver, variant);
      HYPRE_BoomerAMGSetOverlap(amg_solver, overlap);
      HYPRE_BoomerAMGSetDomainType(amg_solver, domain_type);
      HYPRE_BoomerAMGSetSchwarzRlxWeight(amg_solver, schwarz_rlx_weight);
      HYPRE_BoomerAMGSetNumFunctions(amg_solver, num_functions);
      HYPRE_BoomerAMGSetNumPaths(amg_solver, num_paths);
      HYPRE_BoomerAMGSetAggNumLevels(amg_solver, agg_num_levels);
      if (num_functions > 1)
	 HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);

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

   /*-----------------------------------------------------------
    * Solve the system using GSMG
    *-----------------------------------------------------------*/

   if (solver_id == 13)
   {
      /* reset some smoother parameters */
 
      /* fine grid */
      num_grid_sweeps[0] = num_sweep;
      grid_relax_type[0] = relax_default;
      hypre_TFree (grid_relax_points[0]);
      grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, num_sweep);
      for (i=0; i<num_sweep; i++)
         grid_relax_points[0][i] = 0;
 
      /* down cycle */
      num_grid_sweeps[1] = num_sweep;
      grid_relax_type[1] = relax_default;
      hypre_TFree (grid_relax_points[1]);
      grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, num_sweep);
      for (i=0; i<num_sweep; i++)
         grid_relax_points[1][i] = 0;
 
      /* up cycle */
      num_grid_sweeps[2] = num_sweep;
      grid_relax_type[2] = relax_default;
      hypre_TFree (grid_relax_points[2]);
      grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, num_sweep);
      for (i=0; i<num_sweep; i++)
         grid_relax_points[2][i] = 0;
 
      /* coarsest grid */
      num_grid_sweeps[3] = 1;
      grid_relax_type[3] = 9;
      hypre_TFree (grid_relax_points[3]);
      grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int, 1);
      grid_relax_points[3][0] = 0;
 
      if (myid == 0) hypre_printf("Solver:  GSMG\n");
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_BoomerAMGCreate(&amg_solver);
      /* BM Aug 25, 2006 */
      HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
      HYPRE_BoomerAMGSetGSMG(amg_solver, 4); /* specify GSMG */
      HYPRE_BoomerAMGSetInterpType(amg_solver, interp_type);
      HYPRE_BoomerAMGSetNumSamples(amg_solver, gsmg_samples);
      HYPRE_BoomerAMGSetCoarsenType(amg_solver, (hybrid*coarsen_type));
      HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      HYPRE_BoomerAMGSetTol(amg_solver, tol);
      HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
      HYPRE_BoomerAMGSetPMaxElmts(amg_solver, P_max_elmts);
      HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
      HYPRE_BoomerAMGSetJacobiTruncThreshold(amg_solver, jacobi_trunc_threshold);
/* note: log is written to standard output, not to file */
      HYPRE_BoomerAMGSetPrintLevel(amg_solver, 3);
      HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log");
      HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      HYPRE_BoomerAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      HYPRE_BoomerAMGSetGridRelaxType(amg_solver, grid_relax_type);
      HYPRE_BoomerAMGSetRelaxWeight(amg_solver, relax_weight);
      HYPRE_BoomerAMGSetOmega(amg_solver, omega);
      HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
      HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
      HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, smooth_num_levels);
      HYPRE_BoomerAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      HYPRE_BoomerAMGSetMaxRowSum(amg_solver, max_row_sum);
      HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);
      HYPRE_BoomerAMGSetVariant(amg_solver, variant);
      HYPRE_BoomerAMGSetOverlap(amg_solver, overlap);
      HYPRE_BoomerAMGSetDomainType(amg_solver, domain_type);
      HYPRE_BoomerAMGSetSchwarzRlxWeight(amg_solver, schwarz_rlx_weight);
      HYPRE_BoomerAMGSetNumFunctions(amg_solver, num_functions);
      HYPRE_BoomerAMGSetNumPaths(amg_solver, num_paths);
      HYPRE_BoomerAMGSetAggNumLevels(amg_solver, agg_num_levels);
      if (num_functions > 1)
         HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);
 
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
      if (myid == 0) hypre_printf("Test ParaSails Build IJMatrix\n");

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

      if (myid == 0) hypre_printf("Printed to parasails.out.\n");
      exit(0);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG 
    *-----------------------------------------------------------*/

   /* begin lobpcg */
   if ( !lobpcgFlag && ( solver_id == 1 || solver_id == 2 || solver_id == 8 || 
			 solver_id == 12 || solver_id == 14 || solver_id == 43 ) )
   /*end lobpcg */
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter(pcg_solver, 1000);
      HYPRE_PCGSetTol(pcg_solver, tol);
      HYPRE_PCGSetTwoNorm(pcg_solver, two_norm);
      HYPRE_PCGSetRelChange(pcg_solver, 0);
      HYPRE_PCGSetPrintLevel(pcg_solver, ioutdat);
 
      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) hypre_printf("Solver: AMG-PCG\n");
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         /* BM Aug 25, 2006 */
         HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 2)
      {
         
         /* use diagonal scaling as preconditioner */
         if (myid == 0) hypre_printf("Solver: DS-PCG\n");
         pcg_precond = NULL;

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                             pcg_precond);
      }
      else if (solver_id == 8)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) hypre_printf("Solver: ParaSails-PCG\n");

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
         if (myid == 0) hypre_printf("Solver: Schwarz-PCG\n");

	 HYPRE_SchwarzCreate(&pcg_precond);
	 HYPRE_SchwarzSetVariant(pcg_precond, variant);
	 HYPRE_SchwarzSetOverlap(pcg_precond, overlap);
	 HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
         HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);

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
         num_grid_sweeps[0] = num_sweep;
         grid_relax_type[0] = relax_default;
         hypre_TFree (grid_relax_points[0]);
         grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, num_sweep);
         for (i=0; i<num_sweep; i++)
            grid_relax_points[0][i] = 0;
    
         /* down cycle */
         num_grid_sweeps[1] = num_sweep;
         grid_relax_type[1] = relax_default;
         hypre_TFree (grid_relax_points[1]);
         grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, num_sweep);
         for (i=0; i<num_sweep; i++)
            grid_relax_points[1][i] = 0;
    
         /* up cycle */
         num_grid_sweeps[2] = num_sweep;
         grid_relax_type[2] = relax_default;
         hypre_TFree (grid_relax_points[2]);
         grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, num_sweep);
         for (i=0; i<num_sweep; i++)
            grid_relax_points[2][i] = 0;
    
         /* coarsest grid */
         num_grid_sweeps[3] = 1;
         grid_relax_type[3] = 9;
         hypre_TFree (grid_relax_points[3]);
         grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int, 1);
         grid_relax_points[3][0] = 0;
 
         if (myid == 0) hypre_printf("Solver: GSMG-PCG\n");
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         /* BM Aug 25, 2006 */
         HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         HYPRE_BoomerAMGSetGSMG(pcg_precond, 4); 
         HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 43)
      {
         /* use Euclid preconditioning */
         if (myid == 0) hypre_printf("Solver: Euclid-PCG\n");

         HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                             pcg_precond);
      }
 
      HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
        hypre_printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
        return(-1);
      }
      else 
        if (myid == 0)
          hypre_printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");

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

   /* begin lobpcg */

   /*-----------------------------------------------------------
    * Solve the eigenvalue problem using LOBPCG 
    *-----------------------------------------------------------*/

   if ( lobpcgFlag ) {

     interpreter = hypre_CTAlloc(mv_InterfaceInterpreter,1);

     HYPRE_ParCSRSetupInterpreter( interpreter );
     HYPRE_ParCSRSetupMatvec(&matvec_fn);

     if (myid != 0)
       verbosity = 0;

     if ( lobpcgGen ) {
       BuildParIsoLaplacian(argc, argv, &parcsr_B);

       ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_B,
					       &first_local_row, &last_local_row ,
					       &first_local_col, &last_local_col );

       local_num_rows = last_local_row - first_local_row + 1;
       local_num_cols = last_local_col - first_local_col + 1;
       ierr += HYPRE_ParCSRMatrixGetDims( parcsr_B, &M, &N );

       ierr += HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
				     first_local_col, last_local_col,
				     &ij_B );
      
       ierr += HYPRE_IJMatrixSetObjectType( ij_B, HYPRE_PARCSR );  
    
       if (sparsity_known == 1) {   
	 diag_sizes = hypre_CTAlloc(HYPRE_Int, local_num_rows);
	 offdiag_sizes = hypre_CTAlloc(HYPRE_Int, local_num_rows);
	 local_row = 0;
	 for (i=first_local_row; i<= last_local_row; i++) {
	   ierr += HYPRE_ParCSRMatrixGetRow( parcsr_B, i, &size, 
					     &col_inds, &values );
	   for (j=0; j < size; j++)
	     {
	       if (col_inds[j] < first_local_row || col_inds[j] > last_local_row)
		 offdiag_sizes[local_row]++;
	       else
		 diag_sizes[local_row]++;
	     }
	   local_row++;
	   ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_B, i, &size, 
						 &col_inds, &values );
	 }
	 ierr += HYPRE_IJMatrixSetDiagOffdSizes( ij_B, 
						 (const HYPRE_Int *) diag_sizes,
						 (const HYPRE_Int *) offdiag_sizes );
	 hypre_TFree(diag_sizes);
	 hypre_TFree(offdiag_sizes);
     
	 ierr = HYPRE_IJMatrixInitialize( ij_B );

	 for (i=first_local_row; i<= last_local_row; i++)
	   {
	     ierr += HYPRE_ParCSRMatrixGetRow( parcsr_B, i, &size,
					       &col_inds, &values );

	     ierr += HYPRE_IJMatrixSetValues( ij_B, 1, &size, &i,
					      (const HYPRE_Int *) col_inds,
					      (const double *) values );

	     ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_B, i, &size,
						   &col_inds, &values );
	   }
       }
       else
	 {
	   row_sizes = hypre_CTAlloc(HYPRE_Int, local_num_rows);
	
	   size = 5; /* this is in general too low, and supposed to test
			the capability of the reallocation of the interface */ 

	   if (sparsity_known == 0) /* tries a more accurate estimate of the
				       storage */
	     {
	       if (build_matrix_type == 2) size = 7;
	       if (build_matrix_type == 3) size = 9;
	       if (build_matrix_type == 4) size = 27;
	     }

	   for (i=0; i < local_num_rows; i++)
	     row_sizes[i] = size;

	   ierr = HYPRE_IJMatrixSetRowSizes ( ij_B, (const HYPRE_Int *) row_sizes );

	   hypre_TFree(row_sizes);

	   ierr = HYPRE_IJMatrixInitialize( ij_B );

	   /* Loop through all locally stored rows and insert them into ij_matrix */
	   for (i=first_local_row; i<= last_local_row; i++)
	     {
	       ierr += HYPRE_ParCSRMatrixGetRow( parcsr_B, i, &size,
						 &col_inds, &values );

	       ierr += HYPRE_IJMatrixSetValues( ij_B, 1, &size, &i,
						(const HYPRE_Int *) col_inds,
						(const double *) values );

	       ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_B, i, &size,
						     &col_inds, &values );
	     }
	 }

       ierr += HYPRE_IJMatrixAssemble( ij_B );

       ierr += HYPRE_ParCSRMatrixDestroy(parcsr_B);

       ierr += HYPRE_IJMatrixGetObject( ij_B, &object);
       parcsr_B = (HYPRE_ParCSRMatrix) object;

     } /* if ( lobpcgGen ) */

     if ( pcgIterations > 0 ) { /* do inner pcg iterations */

       time_index = hypre_InitializeTiming("PCG Setup");
       hypre_BeginTiming(time_index);

       HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
       HYPRE_PCGSetMaxIter(pcg_solver, pcgIterations);
       HYPRE_PCGSetTol(pcg_solver, pcgTol);
       HYPRE_PCGSetTwoNorm(pcg_solver, two_norm);
       HYPRE_PCGSetRelChange(pcg_solver, 0);
       HYPRE_PCGSetPrintLevel(pcg_solver, 0);
 
       HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond);

       if (solver_id == 1)
	 {
	   /* use BoomerAMG as preconditioner */
	   if (myid == 0) hypre_printf("Solver: AMG-PCG\n");
	   HYPRE_BoomerAMGCreate(&pcg_precond); 
           /* BM Aug 25, 2006 */
           HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
	   HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
	   HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
	   HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
	   HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
	   HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
	   HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
	   HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
           HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
           HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
           HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
	   HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
	   HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
	   HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
	   HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
	   HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
	   HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
	   HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
	   HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
	   HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
	   HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
	   HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
	   HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
	   HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
	   HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
	   HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
           HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
           HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
	   HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
	   HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
	   HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
	   HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
	   if (num_functions > 1)
	     HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
	   HYPRE_PCGSetPrecond(pcg_solver,
			       (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
			       (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
			       pcg_precond);
	 }
       else if (solver_id == 2)
	 {
	    
	   /* use diagonal scaling as preconditioner */
	   if (myid == 0) hypre_printf("Solver: DS-PCG\n");
	   pcg_precond = NULL;
	    
	   HYPRE_PCGSetPrecond(pcg_solver,
			       (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
			       (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
			       pcg_precond);
	 }
       else if (solver_id == 8)
	 {
	   /* use ParaSails preconditioner */
	   if (myid == 0) hypre_printf("Solver: ParaSails-PCG\n");
	    
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
	   if (myid == 0) hypre_printf("Solver: Schwarz-PCG\n");
	    
	   HYPRE_SchwarzCreate(&pcg_precond);
	   HYPRE_SchwarzSetVariant(pcg_precond, variant);
	   HYPRE_SchwarzSetOverlap(pcg_precond, overlap);
	   HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
	   HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);
	    
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
	   num_grid_sweeps[0] = num_sweep;
	   grid_relax_type[0] = relax_default;
	   hypre_TFree (grid_relax_points[0]);
	   grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, num_sweep);
	   for (i=0; i<num_sweep; i++)
	     grid_relax_points[0][i] = 0;
	    
	   /* down cycle */
	   num_grid_sweeps[1] = num_sweep;
	   grid_relax_type[1] = relax_default;
	   hypre_TFree (grid_relax_points[1]);
	   grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, num_sweep);
	   for (i=0; i<num_sweep; i++)
	     grid_relax_points[1][i] = 0;
	    
	   /* up cycle */
	   num_grid_sweeps[2] = num_sweep;
	   grid_relax_type[2] = relax_default;
	   hypre_TFree (grid_relax_points[2]);
	   grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, num_sweep);
	   for (i=0; i<num_sweep; i++)
	     grid_relax_points[2][i] = 0;
	    
	   /* coarsest grid */
	   num_grid_sweeps[3] = 1;
	   grid_relax_type[3] = 9;
	   hypre_TFree (grid_relax_points[3]);
	   grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int, 1);
	   grid_relax_points[3][0] = 0;
	    
	   if (myid == 0) hypre_printf("Solver: GSMG-PCG\n");
	   HYPRE_BoomerAMGCreate(&pcg_precond); 
           /* BM Aug 25, 2006 */
           HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
	   HYPRE_BoomerAMGSetGSMG(pcg_precond, 4); 
	   HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
	   HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
	   HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
	   HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
	   HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
	   HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
	   HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
           HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
           HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
           HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
	   HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
	   HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
	   HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
	   HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
	   HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
	   HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
	   HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
	   HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
	   HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
	   HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
	   HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
	   HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
	   HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
	   HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
	   HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
	   HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
	   HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
	   HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
	   HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
           HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
           HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
	   if (num_functions > 1)
	     HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
	   HYPRE_PCGSetPrecond(pcg_solver,
			       (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
			       (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
			       pcg_precond);
	 }
       else if (solver_id == 43)
	 {
	   /* use Euclid preconditioning */
	   if (myid == 0) hypre_printf("Solver: Euclid-PCG\n");
	    
	   HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &pcg_precond);
	    
	   /* note: There are three three methods of setting run-time 
	      parameters for Euclid: (see HYPRE_parcsr_ls.h); here
	      we'll use what I think is simplest: let Euclid internally 
	      parse the command line.
	   */   
	   HYPRE_EuclidSetParams(pcg_precond, argc, argv);
	    
	   HYPRE_PCGSetPrecond(pcg_solver,
			       (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
			       (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
			       pcg_precond);
	 }
       else if (solver_id != NO_SOLVER )
	 {
	   if ( verbosity )
	     hypre_printf("Solver ID not recognized - running inner PCG iterations without preconditioner\n\n");
	 }
	
       HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
       if (pcg_precond_gotten !=  pcg_precond)
	 {
	   hypre_printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
	   return(-1);
	 }
       else 
	 if (myid == 0)
	   hypre_printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");
	
       /*      HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, 
	       (HYPRE_Vector)b, (HYPRE_Vector)x); */
	
       hypre_EndTiming(time_index);
       hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
       hypre_FinalizeTiming(time_index);
       hypre_ClearTiming();
	
       HYPRE_LOBPCGCreate(interpreter, &matvec_fn, &lobpcg_solver);

       HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
       HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
       HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
       HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);
	
       HYPRE_LOBPCGSetPrecond(lobpcg_solver,
			      (HYPRE_PtrToSolverFcn) HYPRE_PCGSolve,
			      (HYPRE_PtrToSolverFcn) HYPRE_PCGSetup,
			      pcg_solver);

       HYPRE_LOBPCGSetupT(lobpcg_solver, (HYPRE_Matrix)parcsr_A, 
			  (HYPRE_Vector)x);

       HYPRE_LOBPCGSetup(lobpcg_solver, (HYPRE_Matrix)parcsr_A, 
			 (HYPRE_Vector)b, (HYPRE_Vector)x);

       if ( lobpcgGen )
	 HYPRE_LOBPCGSetupB(lobpcg_solver, (HYPRE_Matrix)parcsr_B, 
			    (HYPRE_Vector)x);

       if ( vFromFileFlag ) {
	 eigenvectors = mv_MultiVectorWrap( interpreter,
                         hypre_ParCSRMultiVectorRead(hypre_MPI_COMM_WORLD, 
					             interpreter, 
					             "vectors" ),1);
	 hypre_assert( eigenvectors != NULL );
	 blockSize = mv_MultiVectorWidth( eigenvectors );
       }
       else {
	 eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
								 blockSize, 
								 x );
	 if ( lobpcgSeed )
	   mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
	 else
	   mv_MultiVectorSetRandom( eigenvectors, (HYPRE_Int)time(0) );
       }

       if ( constrained ) {
	 constraints = mv_MultiVectorWrap( interpreter,
                         hypre_ParCSRMultiVectorRead(hypre_MPI_COMM_WORLD,
                                                     interpreter,
                                                     "vectors" ),1);
	 hypre_assert( constraints != NULL );
       }	

       eigenvalues = (double*) calloc( blockSize, sizeof(double) );
	
       time_index = hypre_InitializeTiming("LOBPCG Solve");
       hypre_BeginTiming(time_index);
	
       HYPRE_LOBPCGSolve(lobpcg_solver, constraints, eigenvectors, eigenvalues );
	
       hypre_EndTiming(time_index);
       hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
       hypre_FinalizeTiming(time_index);
       hypre_ClearTiming();


       if ( checkOrtho ) {
		
	 gramXX = utilities_FortranMatrixCreate();
	 identity = utilities_FortranMatrixCreate();

	 utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
	 utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

	 if ( lobpcgGen ) {
	   workspace = mv_MultiVectorCreateCopy( eigenvectors, 0 );
	   hypre_LOBPCGMultiOperatorB( lobpcg_solver, 
	                               mv_MultiVectorGetData(eigenvectors),
	                               mv_MultiVectorGetData(workspace) );
	   lobpcg_MultiVectorByMultiVector( eigenvectors, workspace, gramXX );
	 }
	 else
	   lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );

	 utilities_FortranMatrixSetToIdentity( identity );
	 utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
	 nonOrthF = utilities_FortranMatrixFNorm( gramXX );
	 if ( myid == 0 )
	   hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
		
	 utilities_FortranMatrixDestroy( gramXX );
	 utilities_FortranMatrixDestroy( identity );
	  
       }

       if ( printLevel ) {
		  
	 hypre_ParCSRMultiVectorPrint( mv_MultiVectorGetData(eigenvectors), "vectors" );

	 if ( myid == 0 ) {	  
	   if ( (filePtr = fopen("values.txt", "w")) ) {
	     hypre_fprintf(filePtr, "%d\n", blockSize);
	     for ( i = 0; i < blockSize; i++ )
	       hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
	     fclose(filePtr);
	   }
	    
	   if ( (filePtr = fopen("residuals.txt", "w")) ) {
	     residualNorms = HYPRE_LOBPCGResidualNorms( lobpcg_solver );
	     residuals = utilities_FortranMatrixValues( residualNorms );
	     hypre_fprintf(filePtr, "%d\n", blockSize);
	     for ( i = 0; i < blockSize; i++ )
	       hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
	     fclose(filePtr);
	   }
	    
	   if ( printLevel > 1 ) {
	      
	     printBuffer = utilities_FortranMatrixCreate();
	    
	     iterations = HYPRE_LOBPCGIterations( lobpcg_solver );

	     eigenvaluesHistory = HYPRE_LOBPCGEigenvaluesHistory( lobpcg_solver );
	     utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
						 1, blockSize, 1, iterations + 1, printBuffer );
	     utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

	     residualNormsHistory = HYPRE_LOBPCGResidualNormsHistory( lobpcg_solver );
	     utilities_FortranMatrixSelectBlock(residualNormsHistory, 
						1, blockSize, 1, iterations + 1, printBuffer );
	     utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );
	      
	     utilities_FortranMatrixDestroy( printBuffer );
	   }
	 }
       }

       HYPRE_LOBPCGDestroy(lobpcg_solver);
       mv_MultiVectorDestroy( eigenvectors );
       if ( constrained )
	 mv_MultiVectorDestroy( constraints );
       if ( lobpcgGen )
	 mv_MultiVectorDestroy( workspace );
       free( eigenvalues );
	
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
	
     }
     else { /* pcgIterations <= 0 --> use the preconditioner directly */
	
       time_index = hypre_InitializeTiming("LOBPCG Setup");
       hypre_BeginTiming(time_index);
       if (myid != 0) 
	 verbosity = 0;
 
       HYPRE_LOBPCGCreate(interpreter, &matvec_fn, &pcg_solver);
       HYPRE_LOBPCGSetMaxIter(pcg_solver, maxIterations);
       HYPRE_LOBPCGSetTol(pcg_solver, tol);
       HYPRE_LOBPCGSetPrintLevel(pcg_solver, verbosity);
	
       HYPRE_LOBPCGGetPrecond(pcg_solver, &pcg_precond);
	
       if (solver_id == 1)
	 {
	   /* use BoomerAMG as preconditioner */
	   if (myid == 0) 
	     hypre_printf("Solver: AMG-PCG\n");
	    
	   HYPRE_BoomerAMGCreate(&pcg_precond); 
           /* BM Aug 25, 2006 */
           HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
	   HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
	   HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
	   HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
	   HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
	   HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
	   HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
	   HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
           HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
           HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
           HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
	   HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
	   HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
	   HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
	   HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
	   HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
	   HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
	   HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
	   HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
	   HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
	   HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
	   HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
	   HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
	   HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
	   HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
	   HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
           HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
           HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
	   HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
	   HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
	   HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
	   HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
	   if (num_functions > 1)
	     HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
	    
	   HYPRE_LOBPCGSetPrecond(pcg_solver,
				  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
				  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
				  pcg_precond);
	 }
       else if (solver_id == 2)
	 {
	    
	   /* use diagonal scaling as preconditioner */
	   if (myid == 0) 
	     hypre_printf("Solver: DS-PCG\n");
         
	   pcg_precond = NULL;
	    
	   HYPRE_LOBPCGSetPrecond(pcg_solver,
				  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
				  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
				  pcg_precond);
	 }
       else if (solver_id == 8)
	 {
	   /* use ParaSails preconditioner */
	   if (myid == 0) 
	     hypre_printf("Solver: ParaSails-PCG\n");
	 
	   HYPRE_ParaSailsCreate(hypre_MPI_COMM_WORLD, &pcg_precond);
	   HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
	   HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
	   HYPRE_ParaSailsSetLogging(pcg_precond, poutdat);
	    
	   HYPRE_LOBPCGSetPrecond(pcg_solver,
				  (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
				  (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup,
				  pcg_precond);
	 }
       else if (solver_id == 12)
	 {
	   /* use Schwarz preconditioner */
	   if (myid == 0) 
	     hypre_printf("Solver: Schwarz-PCG\n");
	    
	   HYPRE_SchwarzCreate(&pcg_precond);		
	   HYPRE_SchwarzSetVariant(pcg_precond, variant);		
	   HYPRE_SchwarzSetOverlap(pcg_precond, overlap);		
	   HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
	   HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);
	    
	   HYPRE_LOBPCGSetPrecond(pcg_solver,
				  (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSolve,
				  (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSetup,
				  pcg_precond);
	 }
       else if (solver_id == 14)
	 {
	   /* use GSMG as preconditioner */
	    
	   /* reset some smoother parameters */
	    
	   /* fine grid */
	   num_grid_sweeps[0] = num_sweep;
	   grid_relax_type[0] = relax_default;
	   hypre_TFree (grid_relax_points[0]);
	   grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, num_sweep);
	   for (i=0; i<num_sweep; i++)
	     grid_relax_points[0][i] = 0;
	    
	   /* down cycle */
	   num_grid_sweeps[1] = num_sweep;
	   grid_relax_type[1] = relax_default;
	   hypre_TFree (grid_relax_points[1]);
	   grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, num_sweep);
	   for (i=0; i<num_sweep; i++)
	     grid_relax_points[1][i] = 0;
	    
	   /* up cycle */
	   num_grid_sweeps[2] = num_sweep;
	   grid_relax_type[2] = relax_default;
	   hypre_TFree (grid_relax_points[2]);
	   grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, num_sweep);
	   for (i=0; i<num_sweep; i++)
	     grid_relax_points[2][i] = 0;
	    
	   /* coarsest grid */
	   num_grid_sweeps[3] = 1;
	   grid_relax_type[3] = 9;
	   hypre_TFree (grid_relax_points[3]);
	   grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int, 1);
	   grid_relax_points[3][0] = 0;
	    
	   if (myid == 0) hypre_printf("Solver: GSMG-PCG\n");
	   HYPRE_BoomerAMGCreate(&pcg_precond); 
           /* BM Aug 25, 2006 */
           HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
	   HYPRE_BoomerAMGSetGSMG(pcg_precond, 4); 
	   HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
	   HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
	   HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
	   HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
	   HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
	   HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
	   HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
           HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
           HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
           HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
	   HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
	   HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
	   HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
	   HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
	   HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
	   HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
	   HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
	   HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
	   HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
	   HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
	   HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
	   HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
	   HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
	   HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
	   HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
	   HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
	   HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
	   HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
	   HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
           HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
           HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
	   if (num_functions > 1)
	     HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
	    
	   HYPRE_LOBPCGSetPrecond(pcg_solver,
				  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
				  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
				  pcg_precond);
	 }
       else if (solver_id == 43)
	 {
	   /* use Euclid preconditioning */
	   if (myid == 0) 
	     hypre_printf("Solver: Euclid-PCG\n");
	    
	   HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &pcg_precond);
	    
	   /* note: There are three three methods of setting run-time 
	      parameters for Euclid: (see HYPRE_parcsr_ls.h); here
	      we'll use what I think is simplest: let Euclid internally 
	      parse the command line.
	   */   
	   HYPRE_EuclidSetParams(pcg_precond, argc, argv);
	    
	   HYPRE_LOBPCGSetPrecond(pcg_solver,
				  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
				  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
				  pcg_precond);
	 }
       else if (solver_id != NO_SOLVER )
	 {
	   if ( verbosity )
	     hypre_printf("Solver ID not recognized - running LOBPCG without preconditioner\n\n");
	 }
	
       HYPRE_LOBPCGGetPrecond(pcg_solver, &pcg_precond_gotten);
       if (pcg_precond_gotten !=  pcg_precond && pcgIterations)
	 {
	   hypre_printf("HYPRE_ParCSRLOBPCGGetPrecond got bad precond\n");
	   return(-1);
	 }
       else 
	 if (myid == 0)
	   hypre_printf("HYPRE_ParCSRLOBPCGGetPrecond got good precond\n");
	
       HYPRE_LOBPCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, 
			 (HYPRE_Vector)b, (HYPRE_Vector)x);
 
       if ( lobpcgGen )
	 HYPRE_LOBPCGSetupB(pcg_solver, (HYPRE_Matrix)parcsr_B, 
			    (HYPRE_Vector)x);

       hypre_EndTiming(time_index);
       hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
       hypre_FinalizeTiming(time_index);
       hypre_ClearTiming();

       if ( vFromFileFlag ) {
         eigenvectors = mv_MultiVectorWrap( interpreter,
                         hypre_ParCSRMultiVectorRead(hypre_MPI_COMM_WORLD,
                                                     interpreter,
                                                     "vectors" ),1);
	 hypre_assert( eigenvectors != NULL );
	 blockSize = mv_MultiVectorWidth( eigenvectors );
       }
       else {
	 eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
								 blockSize, 
								 x );
	 if ( lobpcgSeed )
	   mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
	 else
	   mv_MultiVectorSetRandom( eigenvectors, (HYPRE_Int)time(0) );
       }
	
       if ( constrained ) {
         constraints = mv_MultiVectorWrap( interpreter,
                         hypre_ParCSRMultiVectorRead(hypre_MPI_COMM_WORLD,
                                                     interpreter,
                                                     "vectors" ),1);
	 hypre_assert( constraints != NULL );
       }	

       eigenvalues = (double*) calloc( blockSize, sizeof(double) );
	
       time_index = hypre_InitializeTiming("LOBPCG Solve");
       hypre_BeginTiming(time_index);
	
       HYPRE_LOBPCGSolve(pcg_solver, constraints, eigenvectors, eigenvalues);
	
       hypre_EndTiming(time_index);
       hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
       hypre_FinalizeTiming(time_index);
       hypre_ClearTiming(); 
	
       if ( checkOrtho ) {
	  
	 gramXX = utilities_FortranMatrixCreate();
	 identity = utilities_FortranMatrixCreate();

	 utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
	 utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

	 if ( lobpcgGen ) {
	   workspace = mv_MultiVectorCreateCopy( eigenvectors, 0 );
           hypre_LOBPCGMultiOperatorB( pcg_solver,
                                       mv_MultiVectorGetData(eigenvectors),
                                       mv_MultiVectorGetData(workspace) );
	   lobpcg_MultiVectorByMultiVector( eigenvectors, workspace, gramXX );
	 }
	 else
	   lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );

	 utilities_FortranMatrixSetToIdentity( identity );
	 utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
	 nonOrthF = utilities_FortranMatrixFNorm( gramXX );
	 if ( myid == 0 )
	   hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
		
	 utilities_FortranMatrixDestroy( gramXX );
	 utilities_FortranMatrixDestroy( identity );
	  
       }

       if ( printLevel ) {
	  
         hypre_ParCSRMultiVectorPrint( mv_MultiVectorGetData(eigenvectors), "vectors" );
	  
	 if ( myid == 0 ) {
	   if ( (filePtr = fopen("values.txt", "w")) ) {
	     hypre_fprintf(filePtr, "%d\n", blockSize);
	     for ( i = 0; i < blockSize; i++ )
	       hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
	     fclose(filePtr);
	   }
	    
	   if ( (filePtr = fopen("residuals.txt", "w")) ) {
	     residualNorms = HYPRE_LOBPCGResidualNorms( pcg_solver );
	     residuals = utilities_FortranMatrixValues( residualNorms );
	     hypre_fprintf(filePtr, "%d\n", blockSize);
	     for ( i = 0; i < blockSize; i++ )
	       hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
	     fclose(filePtr);
	   }

	   if ( printLevel > 1 ) {

	     printBuffer = utilities_FortranMatrixCreate();

	     iterations = HYPRE_LOBPCGIterations( pcg_solver );

	     eigenvaluesHistory = HYPRE_LOBPCGEigenvaluesHistory( pcg_solver );
	     utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
						 1, blockSize, 1, iterations + 1, printBuffer );
	     utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

	     residualNormsHistory = HYPRE_LOBPCGResidualNormsHistory( pcg_solver );
	     utilities_FortranMatrixSelectBlock(residualNormsHistory,
						1, blockSize, 1, iterations + 1, printBuffer );
	     utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );
	      
	     utilities_FortranMatrixDestroy( printBuffer );
	   }
	 }
       } 

#if SECOND_TIME
       /* run a second time to check for memory leaks */
       mv_MultiVectorSetRandom( eigenvectors, 775 );
       HYPRE_LOBPCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, 
			 (HYPRE_Vector)b, (HYPRE_Vector)x);
       HYPRE_LOBPCGSolve(pcg_solver, constraints, eigenvectors, eigenvalues );
#endif

       HYPRE_LOBPCGDestroy(pcg_solver);
 
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

       mv_MultiVectorDestroy( eigenvectors );
       if ( constrained )
	 mv_MultiVectorDestroy( constraints );
       if ( lobpcgGen )
	 mv_MultiVectorDestroy( workspace );
       free( eigenvalues );
	
     } /* if ( pcgIterations > 0 ) */

     hypre_TFree( interpreter );

     if ( lobpcgGen )
       HYPRE_IJMatrixDestroy(ij_B);

   } /* if ( lobpcgFlag ) */

   /* end lobpcg */

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
      HYPRE_GMRESSetMaxIter(pcg_solver, 1000);
      HYPRE_GMRESSetTol(pcg_solver, tol);
      HYPRE_GMRESSetLogging(pcg_solver, 1);
      HYPRE_GMRESSetPrintLevel(pcg_solver, ioutdat);
 
      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) hypre_printf("Solver: AMG-GMRES\n");

         HYPRE_BoomerAMGCreate(&pcg_precond); 
         /* BM Aug 25, 2006 */
         HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                               pcg_precond);
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
      else if (solver_id == 15)
      {
         /* use GSMG as preconditioner */

         /* reset some smoother parameters */
    
         /* fine grid */
         num_grid_sweeps[0] = num_sweep;
         grid_relax_type[0] = relax_default;
         hypre_TFree (grid_relax_points[0]);
         grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, num_sweep);
         for (i=0; i<num_sweep; i++)
            grid_relax_points[0][i] = 0;
    
         /* down cycle */
         num_grid_sweeps[1] = num_sweep;
         grid_relax_type[1] = relax_default;
         hypre_TFree (grid_relax_points[1]);
         grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, num_sweep);
         for (i=0; i<num_sweep; i++)
            grid_relax_points[1][i] = 0;
    
         /* up cycle */
         num_grid_sweeps[2] = num_sweep;
         grid_relax_type[2] = relax_default;
         hypre_TFree (grid_relax_points[2]);
         grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, num_sweep);
         for (i=0; i<num_sweep; i++)
            grid_relax_points[2][i] = 0;
    
         /* coarsest grid */
         num_grid_sweeps[3] = 1;
         grid_relax_type[3] = 9;
         hypre_TFree (grid_relax_points[3]);
         grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int, 1);
         grid_relax_points[3][0] = 0;
 
         if (myid == 0) hypre_printf("Solver: GSMG-GMRES\n");
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         /* BM Aug 25, 2006 */
         HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         HYPRE_BoomerAMGSetGSMG(pcg_precond, 4); 
         HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_GMRESSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 18)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) hypre_printf("Solver: ParaSails-GMRES\n");

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
         if (myid == 0) hypre_printf("Solver: Euclid-GMRES\n");

         HYPRE_EuclidCreate(hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

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
    * Solve the system using BiCGSTAB 
    *-----------------------------------------------------------*/

   if (solver_id == 9 || solver_id == 10 || solver_id == 11 || solver_id == 45)
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRBiCGSTABCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_BiCGSTABSetMaxIter(pcg_solver, 1000);
      HYPRE_BiCGSTABSetTol(pcg_solver, tol);
      HYPRE_BiCGSTABSetLogging(pcg_solver, ioutdat);
      HYPRE_BiCGSTABSetPrintLevel(pcg_solver, ioutdat);
 
      if (solver_id == 9)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) hypre_printf("Solver: AMG-BiCGSTAB\n");
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         /* BM Aug 25, 2006 */
         HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
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
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

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
      HYPRE_CGNRSetMaxIter(pcg_solver, 1000);
      HYPRE_CGNRSetTol(pcg_solver, tol);
      HYPRE_CGNRSetLogging(pcg_solver, ioutdat);
 
      if (solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) hypre_printf("Solver: AMG-CGNR\n");
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         /* BM Aug 25, 2006 */
         HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
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
      if (myid == 0 /* begin lobpcg */ && !lobpcgFlag /* end lobpcg */)
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

   HYPRE_IJVectorGetObjectType(ij_b, &j);
   /* HYPRE_IJVectorPrint(ij_b, "driver.out.b");
   HYPRE_IJVectorPrint(ij_x, "driver.out.x"); */

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_IJMatrixDestroy(ij_A);
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
   double              cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   double             *values;

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
      hypre_printf("  Laplacian:\n");
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
 
   values = hypre_CTAlloc(double, 4);

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

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD, 
		nx, ny, nz, P, Q, R, p, q, r, values);

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
   double              cx, cy, cz;
   double              ax, ay, az;
   double              hinx,hiny,hinz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   double             *values;

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
 
   values = hypre_CTAlloc(double, 7);

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
BuildParFromOneFile2(HYPRE_Int                  argc,
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
BuildRhsParFromOneFile2(HYPRE_Int                  argc,
                        char                *argv[],
                        HYPRE_Int                  arg_index,
                        HYPRE_Int                 *partitioning,
                        HYPRE_ParVector     *b_ptr     )
{
   char           *filename;

   HYPRE_ParVector b;
   HYPRE_Vector    b_CSR;

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
   double             *values;

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
 
   values = hypre_CTAlloc(double, 2);

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
   double             *values;

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
 
   values = hypre_CTAlloc(double, 2);

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

/* begin lobpcg */

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParIsoLaplacian( HYPRE_Int argc, char** argv, HYPRE_ParCSRMatrix *A_ptr )
{

   HYPRE_Int                 nx, ny, nz;
   double              cx, cy, cz;

   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   double             *values;

   HYPRE_Int arg_index;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   P  = 1;
   Q  = num_procs;
   R  = 1;

   nx = 10;
   ny = 10;
   nz = 10;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;


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
      hypre_printf("  Laplacian:\n");
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
 
   values = hypre_CTAlloc(double, 4);

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

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD, 
		nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

/* end lobpcg */
