/* >>> to do: think about ref counting - may not be set up foolproofly */
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
#include <assert.h>

#include "utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"

#include "Hypre.h"
#include "Hypre_ParCSRMatrix_Impl.h"
#include "Hypre_ParCSRVector_Impl.h"

int BuildParFromFile (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParDifConv (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParFromOneFile (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildFuncsFromFiles (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix A , int **dof_func_ptr );
int BuildFuncsFromOneFile (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix A , int **dof_func_ptr );
int BuildRhsParFromOneFile (int argc , char *argv [], int arg_index , int *partitioning , HYPRE_ParVector *b_ptr );
int BuildParLaplacian9pt (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian27pt (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );

#define SECOND_TIME 0
 
int
main( int   argc,
      char *argv[] )
{
   int                 arg_index;
   int                 print_usage;
   int                 sparsity_known = 0;
   int                 build_matrix_type;
   int                 build_matrix_arg_index;
   int                 build_rhs_type;
   int                 build_rhs_arg_index;
   int                 build_src_type;
   int                 build_src_arg_index;
   int                 build_funcs_type;
   int                 build_funcs_arg_index;
   int                 solver_id;
   int                 ioutdat;
   int                 log_level;
   int                 debug_flag;
   int                 ierr = 0;
   int                 i,j,k; 
   int                 indx, rest, tms;
   int                 max_levels = 25;
   int                 num_iterations; 
   double              norm, tmp;
   double              final_res_norm;
   void               *object;
   SIDL_BaseInterface  Hypre_object;

   HYPRE_IJMatrix      ij_A; 
   Hypre_IJBuildMatrix Hypre_ij_A ;
   HYPRE_IJVector      ij_b;
   HYPRE_IJVector      ij_x;
   Hypre_IJBuildVector Hypre_ij_b;
   Hypre_IJBuildVector Hypre_ij_x;
   Hypre_IJBuildVector Hypre_ij_y;
   Hypre_IJBuildVector Hypre_ij_y2;

   HYPRE_ParCSRMatrix  parcsr_A;
   Hypre_ParCSRMatrix  Hypre_parcsr_A;
   Hypre_Operator      Hypre_op_A;
   HYPRE_ParVector     b;
   HYPRE_ParVector     x;
   Hypre_ParCSRVector     Hypre_b;
   Hypre_ParCSRVector     Hypre_x;
   Hypre_ParCSRVector     Hypre_y;
   Hypre_ParCSRVector     Hypre_y2;
   Hypre_Vector        y,Hypre_Vector_x, Hypre_Vector_b;

   HYPRE_Solver        amg_solver;
   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond, pcg_precond_gotten;
   Hypre_ParAMG        Hypre_AMG;
   Hypre_PCG           Hypre_PCG;
   Hypre_ParDiagScale  Hypre_ParDiagScale;
   Hypre_Solver        Hypre_SolverPC;

   int                 num_procs, myid;
   int                 local_row;
   int                *row_sizes;
   struct SIDL_int__array* Hypre_row_sizes;
   int                *diag_sizes;
   struct SIDL_int__array* Hypre_diag_sizes;
   int                 lower[3], upper[3], stride[3]; /* Do not really need to be 3, but safe */
   int                *offdiag_sizes;
   struct SIDL_int__array* Hypre_offdiag_sizes;
   int                *rows;
   struct SIDL_int__array* Hypre_rows;
   int                 size;
   int                *ncols;
   struct SIDL_int__array* Hypre_ncols;
   int                *col_inds;
   struct SIDL_int__array* Hypre_col_inds;
   int                *dof_func;
   int		       num_functions = 1;
   struct Hypre_ParCSRVector__data * temp_vecdata;

   int		       time_index;
   MPI_Comm            comm = MPI_COMM_WORLD;
   int M, N;
   int first_local_row, last_local_row, local_num_rows;
   int first_local_col, last_local_col, local_num_cols;
   int local_num_vars;
   int variant, overlap, domain_type;
   double schwarz_rlx_weight;
   double *values, val;
   struct SIDL_double__array* Hypre_values;
   struct SIDL_int__array* Hypre_indices;
   struct SIDL_int__array* Hypre_num_grid_sweeps=NULL;
   struct SIDL_int__array* Hypre_grid_relax_type=NULL;
   struct SIDL_double__array* Hypre_relax_weight=NULL;
   struct SIDL_int__array* Hypre_smooth_option=NULL;
   struct SIDL_int__array* Hypre_grid_relax_points=NULL;
   struct SIDL_int__array* Hypre_dof_func=NULL;


   int dimsl[2], dimsu[2];

   const double dt_inf = 1.e40;
   double dt = dt_inf;

   /* parameters for BoomerAMG */
   double   strong_threshold;
   double   trunc_factor;
   int      cycle_type;
   int      coarsen_type = 6;
   int      hybrid = 1;
   int      measure_type = 0;
   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points;
   int	    smooth_lev;   
   int	    smooth_rlx = 8;
   int	   *smooth_option;
   int      relax_default;
   int      smooth_num_sweep = 1;
   int      num_sweep = 1;
   double  *relax_weight; 
   double   tol = 1.e-6, pc_tol = 0.;
   double   max_row_sum = 1.;

   /* parameters for ParaSAILS */
   double   sai_threshold = 0.1;
   double   sai_filter = 0.1;

   /* parameters for PILUT */
   double   drop_tol = -1;
   int      nonzeros_to_keep = -1;

   /* parameters for GMRES */
   int	    k_dim;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
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

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   print_usage = 0;
   arg_index = 1;
   smooth_option = hypre_CTAlloc(int, max_levels);
   for (i=0; i < max_levels; i++)
   {
        smooth_option[i] = -1;
   }

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromijfile") == 0 )
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
      else if ( strcmp(argv[arg_index], "-smooth") == 0 )
      {
         arg_index++;
         smooth_lev = atoi(argv[arg_index++]);
         smooth_rlx = atoi(argv[arg_index++]);
         for (i=0; i < smooth_lev; i++)
            smooth_option[i] = smooth_rlx;
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
         smooth_num_sweep = atoi(argv[arg_index++]);
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
      else
      {
         arg_index++;
      }
   }

   /* for CGNR preconditioned with Boomeramg, only relaxation scheme 2 is
      implemented, i.e. Jacobi relaxation with Matvec */
   if (solver_id == 5) relax_default = 2;

   if (solver_id == 8 || solver_id == 18)
   {
     max_levels = 1;
   }

   /* defaults for BoomerAMG */
   if (solver_id == 0 || solver_id == 1 || solver_id == 3 || solver_id == 5
	|| solver_id == 9)
   {
   strong_threshold = 0.25;
   trunc_factor = 0.;
   cycle_type = 1;

   num_grid_sweeps   = hypre_CTAlloc(int,4);
   grid_relax_type   = hypre_CTAlloc(int,4);
   grid_relax_points = hypre_CTAlloc(int *,4);
   relax_weight      = hypre_CTAlloc(double, max_levels);

   for (i=0; i < max_levels; i++)
	relax_weight[i] = 1.;

   if (coarsen_type == 5)
   {
      /* fine grid */
      num_grid_sweeps[0] = 3;
      grid_relax_type[0] = relax_default; 
      grid_relax_points[0] = hypre_CTAlloc(int, 3); 
      grid_relax_points[0][0] = -2;
      grid_relax_points[0][1] = -1;
      grid_relax_points[0][2] = 1;
   
      /* down cycle */
      num_grid_sweeps[1] = 4;
      grid_relax_type[1] = relax_default; 
      grid_relax_points[1] = hypre_CTAlloc(int, 4); 
      grid_relax_points[1][0] = -1;
      grid_relax_points[1][1] = 1;
      grid_relax_points[1][2] = -2;
      grid_relax_points[1][3] = -2;
   
      /* up cycle */
      num_grid_sweeps[2] = 4;
      grid_relax_type[2] = relax_default; 
      grid_relax_points[2] = hypre_CTAlloc(int, 4); 
      grid_relax_points[2][0] = -2;
      grid_relax_points[2][1] = -2;
      grid_relax_points[2][2] = 1;
      grid_relax_points[2][3] = -1;
   }
   else
   {   
      /* fine grid */
      num_grid_sweeps[0] = 2*num_sweep;
      grid_relax_type[0] = relax_default; 
      grid_relax_points[0] = hypre_CTAlloc(int, 2*num_sweep); 
      for (i=0; i<2*num_sweep; i+=2)
      {
         grid_relax_points[0][i] = 1;
         grid_relax_points[0][i+1] = -1;
      }

      /* down cycle */
      num_grid_sweeps[1] = 2*num_sweep;
      grid_relax_type[1] = relax_default; 
      grid_relax_points[1] = hypre_CTAlloc(int, 2*num_sweep); 
      for (i=0; i<2*num_sweep; i+=2)
      {
         grid_relax_points[1][i] = 1;
         grid_relax_points[1][i+1] = -1;
      }

      /* up cycle */
      num_grid_sweeps[2] = 2*num_sweep;
      grid_relax_type[2] = relax_default; 
      grid_relax_points[2] = hypre_CTAlloc(int, 2*num_sweep); 
      for (i=0; i<2*num_sweep; i+=2)
      {
         grid_relax_points[2][i] = -1;
         grid_relax_points[2][i+1] = 1;
      }
   }

   /* coarsest grid */
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 9;
   grid_relax_points[3] = hypre_CTAlloc(int, 1);
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
		|| solver_id == 5 )
        {
         relax_weight[0] = atof(argv[arg_index++]);
         for (i=1; i < max_levels; i++)
	   relax_weight[i] = relax_weight[0];
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
      else if ( strcmp(argv[arg_index], "-iout") == 0 )
      {
         arg_index++;
         ioutdat  = atoi(argv[arg_index++]);
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
      printf("\n");
      printf("Usage: %s [<options>]\n", argv[0]);
      printf("\n");
      printf("  -fromijfile <filename>     : ");
      printf("matrix read in IJ format from distributed files\n");
      printf("  -fromparcsrfile <filename> : ");
      printf("matrix read in ParCSR format from distributed files\n");
      printf("  -fromonecsrfile <filename> : ");
      printf("matrix read in CSR format from a file on one processor\n");
      printf("\n");
      printf("  -laplacian [<options>] : build 5pt 2D laplacian problem (default) \n");
      printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
      printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
      printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
      printf("    -n <nx> <ny> <nz>    : total problem size \n");
      printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      printf("\n");
      printf("  -exact_size            : inserts immediately into ParCSR structure\n");
      printf("  -storage_low           : allocates not enough storage for aux struct\n");
      printf("  -concrete_parcsr       : use parcsr matrix type as concrete type\n");
      printf("\n");
      printf("  -rhsfromfile           : rhs read in IJ form from distributed files\n");
      printf("  -rhsfromonefile        : rhs read from a file one one processor\n");
      printf("  -rhsrand               : rhs is random vector\n");
      printf("  -rhsisone              : rhs is vector with unit components (default)\n");
      printf("  -xisone                : solution of all ones\n");
      printf("  -rhszero               : rhs is zero vector\n");
      printf("\n");
      printf("  -dt <val>              : specify finite backward Euler time step\n");
      printf("                         :    -rhsfromfile, -rhsfromonefile, -rhsrand,\n");
      printf("                         :    -rhsrand, or -xisone will be ignored\n");
      printf("  -srcfromfile           : backward Euler source read in IJ form from distributed files\n");
      printf("  -srcfromonefile        : ");
      printf("backward Euler source read from a file on one processor\n");
      printf("  -srcrand               : ");
      printf("backward Euler source is random vector with components in range 0 - 1\n");
      printf("  -srcisone              : ");
      printf("backward Euler source is vector with unit components (default)\n");
      printf("  -srczero               : ");
      printf("backward Euler source is zero-vector\n");
      printf("\n");
      printf("  -solver <ID>           : solver ID\n");
      printf("       0=AMG               1=AMG-PCG        \n");
      printf("       2=DS-PCG            3=AMG-GMRES      \n");
      printf("       4=DS-GMRES          5=AMG-CGNR       \n");     
      printf("       6=DS-CGNR           7=PILUT-GMRES    \n");     
      printf("       8=ParaSails-PCG     9=AMG-BiCGSTAB   \n");
      printf("       10=DS-BiCGSTAB     11=PILUT-BiCGSTAB \n");
      printf("       12=Schwarz-PCG     18=ParaSails-GMRES\n");     
      printf("       43=Euclid-PCG      44=Euclid-GMRES   \n");
      printf("       45=Euclid-BICGSTAB\n");
      printf("\n");
      printf("   -cljp                 : CLJP coarsening \n");
      printf("   -ruge                 : Ruge coarsening (local)\n");
      printf("   -ruge3                : third pass on boundary\n");
      printf("   -ruge3c               : third pass on boundary, keep c-points\n");
      printf("   -ruge2b               : 2nd pass is global\n");
      printf("   -rugerlx              : relaxes special points\n");
      printf("   -falgout              : local ruge followed by LJP\n");
      printf("   -nohybrid             : no switch in coarsening\n");
      printf("   -gm                   : use global measures\n");
      printf("\n");
      printf("  -rlx  <val>            : relaxation type\n");
      printf("       0=Weighted Jacobi  \n");
      printf("       1=Gauss-Seidel (very slow!)  \n");
      printf("       3=Hybrid Jacobi/Gauss-Seidel  \n");
      printf("  -ns <val>              : Use <val> sweeps on each level\n");
      printf("                           (default C/F down, F/C up, F/C fine\n");
      printf("\n"); 
      printf("  -mu   <val>            : set AMG cycles (1=V, 2=W, etc.)\n"); 
      printf("  -th   <val>            : set AMG threshold Theta = val \n");
      printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
      printf("  -mxrs <val>            : set AMG maximum row sum threshold for dependency weakening \n");
      printf("  -nf <val>              : set number of functions for systems AMG\n");
     
      printf("  -w   <val>             : set Jacobi relax weight = val\n");
      printf("  -k   <val>             : dimension Krylov space for GMRES\n");
      printf("  -mxl  <val>            : maximum number of levels (AMG, ParaSAILS)\n");
      printf("  -tol  <val>            : set solver convergence tolerance = val\n");
      printf("\n");
      printf("  -sai_th   <val>        : set ParaSAILS threshold = val \n");
      printf("  -sai_filt <val>        : set ParaSAILS filter = val \n");
      printf("\n");  
      printf("  -drop_tol  <val>       : set threshold for dropping in PILUT\n");
      printf("  -nonzeros_to_keep <val>: number of nonzeros in each row to keep\n");
      printf("\n");  
      printf("  -iout <val>            : set output flag\n");
      printf("       0=no output    1=matrix stats\n"); 
      printf("       2=cycle stats  3=matrix & cycle stats\n"); 
      printf("\n");  
      printf("  -dbg <val>             : set debug flag\n");
      printf("       0=no debugging\n       1=internal timing\n       2=interpolation truncation\n       3=more detailed timing in coarsening routine\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("Running with these driver parameters:\n");
      printf("  solver ID    = %d\n\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   if ( myid == 0 && dt != dt_inf)
   {
      printf("  Backward Euler time step with dt = %e\n", dt);
      printf("  Dirichlet 0 BCs are implicit in the spatial operator\n");
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
      BuildParFromOneFile(argc, argv, build_matrix_arg_index, &parcsr_A);
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
      printf("You have asked for an unsupported problem with\n");
      printf("build_matrix_type = %d.\n", build_matrix_type);
      return(-1);
   }

   time_index = hypre_InitializeTiming("Spatial operator");
   hypre_BeginTiming(time_index);

   if (build_matrix_type < 2)
   {
     ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
     parcsr_A = (HYPRE_ParCSRMatrix) object;

     ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
               &first_local_row, &last_local_row ,
               &first_local_col, &last_local_col );

     local_num_rows = last_local_row - first_local_row + 1;
     local_num_cols = last_local_col - first_local_col + 1;

     ierr = HYPRE_IJMatrixInitialize( ij_A );

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

     /* In Babel, must set the type before calling Hypre's create
        (note that the default name for the "new-like" constructor automatically provided
        by Babel is also create, albeit with two underscores. This name
        confusion is merely coincidental). AJC 08/01. */
     Hypre_parcsr_A = Hypre_ParCSRMatrix__create();

     printf("finished __create\n");

     Hypre_ij_A = (Hypre_IJBuildMatrix) Hypre_ParCSRMatrix__cast2
        ( Hypre_parcsr_A, "Hypre.IJBuildMatrix");

     printf("finished cast\n");
     if ( Hypre_ij_A == NULL )
     {
        printf("Cast failed\n");
        return;
     }

     /* The following will cancel each other out, but it is good practice
        to perform them */
     Hypre_IJBuildMatrix_addReference( Hypre_ij_A );
     Hypre_ParCSRMatrix_deleteReference( Hypre_parcsr_A );

     ierr += Hypre_IJBuildMatrix_SetCommunicator( Hypre_ij_A, (void *)comm );
     ierr += Hypre_IJBuildMatrix_Create( Hypre_ij_A, 
                                   first_local_row, last_local_row,
                                   first_local_col, last_local_col );
     printf("finished Create\n");
				   
     printf("Sparsity known = %d\n", sparsity_known);

/* the following shows how to build an IJMatrix if one has only an
   estimate for the row sizes */
     if (sparsity_known == 1)
     {   
/*  build IJMatrix using exact row_sizes for diag and offdiag */

       diag_sizes = hypre_CTAlloc(int, local_num_rows);
       offdiag_sizes = hypre_CTAlloc(int, local_num_rows);
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
       /*
       ierr += HYPRE_IJMatrixSetDiagOffdSizes( ij_A, 
                                        (const int *) diag_sizes,
                                        (const int *) offdiag_sizes );
       */

       /* Set the indexing into the diag_sizes and offdiag_sizes SIDL arrays
          to match C conventions */
       lower[0] = 0;
       upper[0] = local_num_rows - 1;
   
       printf("Creating SIDL_int_arrays\n");

       /* Create the arrays that are used as parameters to SetDiagOffdSizes */
       Hypre_diag_sizes = SIDL_int__array_create( 1, lower, upper );

       Hypre_offdiag_sizes =  SIDL_int__array_create( 1, lower, upper );

       /* Fill the SIDL arrays from the C arrays. We do it this way to keep these
          test codes as similar as possible. It could also done as a single block
          using the "borrow" function */
       for ( i=0; i < local_num_rows; i++ )
       {
          SIDL_int__array_set1( Hypre_diag_sizes, i, diag_sizes[ i ] );
          SIDL_int__array_set1( Hypre_offdiag_sizes, i, offdiag_sizes[ i ] );
       }

       ierr += Hypre_IJBuildMatrix_SetDiagOffdSizes( Hypre_ij_A, Hypre_diag_sizes,
                                        Hypre_offdiag_sizes );
       printf("Finished SetDiagOffdSizes \n");

       hypre_TFree(diag_sizes);
       hypre_TFree(offdiag_sizes);

       SIDL_int__array_destroy( Hypre_diag_sizes );
       SIDL_int__array_destroy( Hypre_offdiag_sizes );

     
       ierr = HYPRE_IJMatrixInitialize( ij_A );

       for (i=first_local_row; i<= last_local_row; i++)
       {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                           &col_inds, &values );

         ierr += HYPRE_IJMatrixSetValues( ij_A, 1, &size, &i,
                                          (const int *) col_inds,
                                          (const double *) values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                               &col_inds, &values );
       }
     }
     else
     {

       /* Create the SIDL arrays */
       /* Set the indexing into the diag_sizes and offdiag_sizes SIDL arrays
          to match C conventions */
       lower[0] = 0;
       upper[0] = local_num_rows - 1;
       printf("about to create SIDL array\n");

       Hypre_row_sizes = SIDL_int__array_create( 1, lower, upper );
       printf("finished to create SIDL array\n");


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
       {

          SIDL_int__array_set1( Hypre_row_sizes, i, size );
       }

         printf("about to call SIDL set row sizes\n");
       ierr = Hypre_IJBuildMatrix_SetRowSizes( Hypre_ij_A, Hypre_row_sizes );
          printf("finishedij row sizes\n");

       SIDL_int__array_destroy( Hypre_row_sizes );

       ierr = Hypre_IJBuildMatrix_Initialize( Hypre_ij_A );

       /* reuse Hypre_row_sizes */
       lower[ 0 ] = 0;
       upper[ 0 ] = 0;

       Hypre_row_sizes = SIDL_int__array_create( 1, lower, upper );

       Hypre_ncols = SIDL_int__array_create( 1, lower, upper );

       stride[ 0 ] = 1;

       /* Loop through all locally stored rows and insert them into ij_matrix */
       for (i=first_local_row; i<= last_local_row; i++)
       {
          ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                            &col_inds, &values );
         SIDL_int__array_set1( Hypre_row_sizes, 0, size );
         SIDL_int__array_set1( Hypre_ncols, 0, i );

         upper[ 0 ] = size - 1;

         Hypre_col_inds = SIDL_int__array_borrow( col_inds, 1, 
                          lower, upper, stride );
         Hypre_values = SIDL_double__array_borrow( values, 1,
                          lower, upper, stride );

         ierr += Hypre_IJBuildMatrix_SetValues( Hypre_ij_A, 1, Hypre_row_sizes,
                                          Hypre_ncols,
                                          Hypre_col_inds,
                                          Hypre_values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                               &col_inds, &values );
       }
       SIDL_int__array_destroy( Hypre_row_sizes );
       SIDL_int__array_destroy( Hypre_ncols );
     }

     ierr += Hypre_IJBuildMatrix_Assemble( Hypre_ij_A );

     SIDL_int__array_destroy( Hypre_col_inds );
     SIDL_double__array_destroy( Hypre_values );

   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Matrix Setup", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   if (ierr)
   {
     printf("Error in driver building IJMatrix from parcsr matrix. \n");
     return(-1);
   }

   /* This is to emphasize that one can IJMatrixAddToValues after an
      IJMatrixRead or an IJMatrixAssemble.  After an IJMatrixRead,
      assembly is unnecessary if the sparsity pattern of the matrix is
      not changed somehow.  If one has not used IJMatrixRead, one has
      the opportunity to IJMatrixAddTo before a IJMatrixAssemble. */

   ncols    = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
   rows     = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
   col_inds = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
   values   = hypre_CTAlloc(double, last_local_row - first_local_row + 1);
   
   upper[ 0 ] = last_local_row - first_local_row;

   Hypre_ncols = SIDL_int__array_borrow( ncols, 1, 
                          lower, upper, stride );
   Hypre_rows = SIDL_int__array_borrow( rows, 1,
                          lower, upper, stride );
   Hypre_col_inds = SIDL_int__array_borrow( col_inds, 1, 
                          lower, upper, stride );
   Hypre_values = SIDL_double__array_borrow( values, 1,
                          lower, upper, stride );

   if (dt < dt_inf)
     val = 1./dt;
   else 
     val = 0.;  /* Use zero to avoid unintentional loss of significance */

   for (i = first_local_row; i <= last_local_row; i++)
   {
     j = i - first_local_row;
     rows[j] = i;
     ncols[j] = 1;
     col_inds[j] = i;
     values[j] = val;
   }
      
   ierr += Hypre_IJBuildMatrix_AddToValues( Hypre_ij_A,
                                      local_num_rows,
                                      Hypre_ncols, Hypre_rows,
                                      Hypre_col_inds,
                                      Hypre_values );

   hypre_TFree(values);
   hypre_TFree(col_inds);
   hypre_TFree(rows);
   hypre_TFree(ncols);

   SIDL_int__array_destroy( Hypre_ncols );
   SIDL_int__array_destroy( Hypre_rows );
   SIDL_int__array_destroy( Hypre_col_inds );
   SIDL_double__array_destroy( Hypre_values );

   /* If sparsity pattern is not changed since last IJMatrixAssemble call,
      this should be a no-op */

   ierr += Hypre_IJBuildMatrix_Assemble( Hypre_ij_A );

   /*-----------------------------------------------------------
    * Fetch the resulting underlying matrix out
    *-----------------------------------------------------------*/

   if (build_matrix_type > 1)
     ierr += HYPRE_ParCSRMatrixDestroy(parcsr_A);

   ierr += Hypre_IJBuildMatrix_GetObject( Hypre_ij_A, &Hypre_object );

   /* Done with the IJBuildMatrix, delete the reference */
   Hypre_IJBuildMatrix_deleteReference( Hypre_ij_A );

   /* The QueryInterface below checks to see if the returned object can
      return a Hypre.ParCSRMatrix. The "cast" is necessary because of the
      restrictions of the C language, and is merely to please the compiler.
      It is the QueryInterface that actually has semantic meaning. */
   Hypre_parcsr_A = SIDL_BaseInterface__cast2
      ( SIDL_BaseInterface_queryInterface( Hypre_object, "Hypre.ParCSRMatrix"),
        "Hypre.ParCSRMatrix" );

   if ( Hypre_parcsr_A == NULL )
   {
      printf("Cast/QI failed\n");
      return;
   }

   {
   /* Break encapsulation so that the rest of the driver stays the same */
   struct Hypre_ParCSRMatrix__data * temp_data;
   temp_data = Hypre_ParCSRMatrix__get_data( Hypre_parcsr_A );
   
   ij_A = temp_data ->ij_A ;

   ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
   parcsr_A = (HYPRE_ParCSRMatrix) object;
   
   }

   /* return; */

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("RHS and Initial Guess");
   hypre_BeginTiming(time_index);

   if ( build_rhs_type == 0 )
   {
      if (myid == 0)
      {
        printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
        printf("  Initial guess is 0\n");
      }

/* RHS */
      ierr = HYPRE_IJVectorRead( argv[build_rhs_arg_index], MPI_COMM_WORLD, 
                                 HYPRE_PARCSR, &ij_b );
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
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
      printf("build_rhs_type == 1 not currently implemented\n");
      return(-1);

#if 0
/* RHS */
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, part_b, &b);
#endif
   }
   else if ( build_rhs_type == 2 )
   {
      if (myid == 0)
      {
        printf("  RHS vector has unit components\n");
        printf("  Initial guess is 0\n");
      }

/* RHS */
      Hypre_b = Hypre_ParCSRVector__create();
      Hypre_ij_b = (Hypre_IJBuildVector) Hypre_ParCSRVector__cast2
         ( Hypre_b, "Hypre.IJBuildVector" );
      /* adjust reference counting system for new data type: */
      Hypre_IJBuildVector_addReference( Hypre_ij_b );
      Hypre_ParCSRVector_deleteReference( Hypre_b );
      ierr += Hypre_IJBuildVector_SetCommunicator( Hypre_ij_b, (void *)comm );
      ierr += Hypre_IJBuildVector_Create( Hypre_ij_b, (void *)comm,
                                          first_local_row,last_local_row );
      ierr += Hypre_IJBuildVector_Initialize( Hypre_ij_b );

      dimsl[0] = 0;  dimsu[0] = local_num_rows;
      Hypre_indices = SIDL_int__array_create( 1, dimsl, dimsu );
      Hypre_values = SIDL_double__array_create( 1, dimsl, dimsu );
      for ( i=0; i<local_num_rows; ++i ) {
         SIDL_int__array_set1( Hypre_indices, i, i );
         SIDL_double__array_set1( Hypre_values, i, 1 );
      }
      Hypre_IJBuildVector_SetValues( Hypre_ij_b, local_num_rows, Hypre_indices, Hypre_values );
      SIDL_int__array_destroy( Hypre_indices );
      SIDL_double__array_destroy( Hypre_values );

     ierr += Hypre_IJBuildVector_Assemble( Hypre_ij_b );

     ierr += Hypre_IJBuildVector_GetObject( Hypre_ij_b, &Hypre_object );
     /* Done with the IJBuildVector, delete the reference */
     Hypre_IJBuildVector_deleteReference( Hypre_ij_b );
     Hypre_b = SIDL_BaseInterface__cast2
      ( SIDL_BaseInterface_queryInterface( Hypre_object, "Hypre.ParCSRVector"),
        "Hypre.ParCSRVector" );
     if ( Hypre_b == NULL ) {
        printf("Cast/QI failed\n");
        return;
     }

      /* Break encapsulation so that the rest of the driver stays the same */

      temp_vecdata = Hypre_ParCSRVector__get_data( Hypre_b );
      ij_b = temp_vecdata ->ij_b ;
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;


/* Initial guess */
      Hypre_x = Hypre_ParCSRVector__create();
      Hypre_ij_x = (Hypre_IJBuildVector) Hypre_ParCSRVector__cast2
         ( Hypre_x, "Hypre.IJBuildVector" );
      /* adjust reference counting system for new data type: */
      Hypre_IJBuildVector_addReference( Hypre_ij_x );
      Hypre_ParCSRVector_deleteReference( Hypre_x );
      ierr += Hypre_IJBuildVector_SetCommunicator( Hypre_ij_x, (void *)comm );
      ierr += Hypre_IJBuildVector_Create( Hypre_ij_x, (void *)comm,
                                          first_local_col,last_local_col );
      ierr += Hypre_IJBuildVector_Initialize( Hypre_ij_x );

      dimsl[0] = 0;  dimsu[0] = local_num_cols;
      Hypre_indices = SIDL_int__array_create( 1, dimsl, dimsu );
      Hypre_values = SIDL_double__array_create( 1, dimsl, dimsu );
      for ( i=0; i<local_num_cols; ++i ) {
         SIDL_int__array_set1( Hypre_indices, i, i );
         SIDL_double__array_set1( Hypre_values, i, 0 );
      }
      Hypre_IJBuildVector_SetValues( Hypre_ij_x, local_num_cols, Hypre_indices, Hypre_values );
      SIDL_int__array_destroy( Hypre_indices );
      SIDL_double__array_destroy( Hypre_values );

     ierr += Hypre_IJBuildVector_Assemble( Hypre_ij_x );

     ierr += Hypre_IJBuildVector_GetObject( Hypre_ij_x, &Hypre_object );
     /* Done with the IJBuildVector, delete the reference */
     Hypre_IJBuildVector_deleteReference( Hypre_ij_x );
     Hypre_x = SIDL_BaseInterface__cast2
      ( SIDL_BaseInterface_queryInterface( Hypre_object, "Hypre.ParCSRVector"),
        "Hypre.ParCSRVector" );
     if ( Hypre_x == NULL ) {
        printf("Cast/QI failed\n");
        return;
     }


      /* Break encapsulation so that the rest of the driver stays the same */

      temp_vecdata = Hypre_ParCSRVector__get_data( Hypre_x );
      ij_x = temp_vecdata ->ij_b ;
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

   }
   else if ( build_rhs_type == 3 )
   {
      if (myid == 0)
      {
        printf("  RHS vector has random components and unit 2-norm\n");
        printf("  Initial guess is 0\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b); 
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
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
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
        printf("  RHS vector set for solution with unit components\n");
        printf("  Initial guess is 0\n");
      }

/* Temporary use of solution vector */
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
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
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b); 
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
        printf("  RHS vector is 0\n");
        printf("  Initial guess has unit components\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
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
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
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
        printf("  Source vector read from file %s\n", argv[build_src_arg_index]);
        printf("  Initial unknown vector in evolution is 0\n");
      }

      ierr = HYPRE_IJVectorRead( argv[build_src_arg_index], MPI_COMM_WORLD, 
                                 HYPRE_PARCSR, &ij_b );

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial unknown vector */
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
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
      printf("build_src_type == 1 not currently implemented\n");
      return(-1);

#if 0
      BuildRhsParFromOneFile(argc, argv, build_src_arg_index, part_b, &b);
#endif
   }
   else if ( build_src_type == 2 )
   {
      if (myid == 0)
      {
        printf("  Source vector has unit components\n");
        printf("  Initial unknown vector is 0\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
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
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
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
        printf("  Source vector has random components in range 0 - 1\n");
        printf("  Initial unknown vector is 0\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
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
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
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
        printf("  Source vector is 0 \n");
        printf("  Initial unknown vector has random components in range 0 - 1\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
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
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
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
   hypre_PrintTiming("IJ Vector Setup", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   HYPRE_IJMatrixPrint(ij_A, "driver.out.A");
   HYPRE_IJVectorPrint(ij_x, "driver.out.x0");
   Hypre_ParCSRMatrix_Print( Hypre_parcsr_A, "driver.out.HA");
   Hypre_ParCSRVector_Print( Hypre_b, "driver.out.Hb0");
   Hypre_ParCSRVector_Print( Hypre_x, "driver.out.Hx0");

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
         dof_func = hypre_CTAlloc(int,local_num_vars);
         if (myid == 0)
	    printf (" Number of unknown functions = %d \n", num_functions);
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
    * Matrix-Vector and Vector Operation Debugging code begun by adapting
    * from Rob Falgout's sstruct tests
    *-----------------------------------------------------------*/

#define DEBUG 1
#if DEBUG
   {
      FILE *file;
      char  filename[255];
                       
      /*  Apply, y=A*b: result is 1's on the interior of the grid */
      Hypre_y = Hypre_ParCSRVector__create();
      Hypre_ij_y = (Hypre_IJBuildVector) Hypre_ParCSRVector__cast2
         ( Hypre_y, "Hypre.IJBuildVector" );
      /* adjust reference counting system for new data type: */
      Hypre_IJBuildVector_addReference( Hypre_ij_y );
      Hypre_ParCSRVector_deleteReference( Hypre_y );
      ierr += Hypre_IJBuildVector_SetCommunicator( Hypre_ij_y, (void *)comm );
      ierr += Hypre_IJBuildVector_Create( Hypre_ij_y, (void *)comm,
                                          first_local_col,last_local_col );
      ierr += Hypre_IJBuildVector_Initialize( Hypre_ij_y );
      y = Hypre_ParCSRVector__cast2( Hypre_y, "Hypre.Vector" );

      Hypre_ParCSRMatrix_Apply( Hypre_parcsr_A, Hypre_ParCSRVector__cast2( Hypre_b, "Hypre.Vector" ),
                                &y );
      Hypre_ParCSRMatrix_Print( Hypre_parcsr_A, "test.A" );
      Hypre_ParCSRVector_Print( Hypre_y, "test.apply" );
      Hypre_IJBuildVector_deleteReference( Hypre_ij_y ); /* delete y */

      /* SetValues, x=1; result is all 1's */
      dimsl[0] = 0;   dimsu[0] = local_num_cols;
      Hypre_indices = SIDL_int__array_create( 1, dimsl, dimsu );
      Hypre_values = SIDL_double__array_create( 1, dimsl, dimsu );
      for ( i=0; i<local_num_cols; ++i ) {
         SIDL_int__array_set1( Hypre_indices, i, i );
         SIDL_double__array_set1( Hypre_values, i, 1.0 );
      }
      Hypre_IJBuildVector_SetValues( Hypre_ij_x, local_num_cols, Hypre_indices, Hypre_values );
      SIDL_int__array_destroy( Hypre_indices );
      SIDL_double__array_destroy( Hypre_values );
      Hypre_ParCSRVector_Print( Hypre_x, "test.setvalues" );

      /* Copy, b=x; result is all 1's */
      Hypre_Vector_x = (Hypre_Vector)Hypre_ParCSRVector__cast2( Hypre_x, "Hypre.Vector" );
      Hypre_ParCSRVector_Copy( Hypre_b, Hypre_Vector_x );
      Hypre_ParCSRVector_Print( Hypre_b, "test.copy" );

      /* Clone y=b; result is all 1's */
      Hypre_ParCSRVector_Clone( Hypre_b, &y );
      Hypre_y = Hypre_Vector__cast2( y, "Hypre.ParCSRVector" );
      Hypre_ParCSRVector_Print( Hypre_y, "test.clone" );
      Hypre_Vector_deleteReference( y );

      /* Read y2=y; result is all 1's */
      Hypre_y2 = Hypre_ParCSRVector__create();
      Hypre_ij_y2 = (Hypre_IJBuildVector) Hypre_ParCSRVector__cast2
         ( Hypre_y2, "Hypre.IJBuildVector" );
      Hypre_IJBuildVector_addReference( Hypre_ij_y2 );
      Hypre_ParCSRVector_deleteReference( Hypre_y2 );
      ierr += Hypre_IJBuildVector_SetCommunicator( Hypre_ij_y2, (void *)comm );
      ierr += Hypre_IJBuildVector_Create( Hypre_ij_y2, (void *)comm,
                                          first_local_col,last_local_col );
      ierr += Hypre_IJBuildVector_Initialize( Hypre_ij_y2 );
      Hypre_ParCSRVector_Read( Hypre_y2, "test.clone", (void *)comm );
      Hypre_ParCSRVector_Print( Hypre_y2, "test.read" );
      Hypre_IJBuildVector_deleteReference( Hypre_ij_y2 );

      /* GetRow, b[i], tested but not printed */
      dimsl[0] = 0;   dimsu[0] = local_num_cols;
      Hypre_indices = SIDL_int__array_create( 1, dimsl, dimsu );
      Hypre_values = SIDL_double__array_create( 1, dimsl, dimsu );
      Hypre_ParCSRVector_GetRow( Hypre_b, 6, &local_num_cols, &Hypre_indices, &Hypre_values );
      tmp = SIDL_double__array_get1( Hypre_values, 0 );
      assert( tmp == 1.0 );
      SIDL_int__array_destroy( Hypre_indices );
      SIDL_double__array_destroy( Hypre_values );

      /* Scale, x=2*x; result is all 2's */
      Hypre_ParCSRVector_Scale( Hypre_x, 2.0 );
      Hypre_ParCSRVector_Print( Hypre_x, "test.scale" );

      /* Dot, tmp = b.x; at this point all b[i]==1, all x[i]==2 */
      Hypre_ParCSRVector_Dot( Hypre_b, Hypre_Vector_x, &tmp );
      assert( tmp==2*local_num_cols );

      /* Axpy, b=b-0.5*x; result is all 0's */
      Hypre_ParCSRVector_Axpy( Hypre_b, -0.5, Hypre_Vector_x );
      Hypre_ParCSRVector_Print( Hypre_b, "test.axpy" );

      /* tested by other parts of this driver program: ParCSRVector_GetObject */

      /* Clear and AddToValues, b=1, which restores its initial value of 1 */
      dimsl[0] = 0;   dimsu[0] = local_num_cols;
      Hypre_indices = SIDL_int__array_create( 1, dimsl, dimsu );
      Hypre_values = SIDL_double__array_create( 1, dimsl, dimsu );
      for ( i=0; i<local_num_cols; ++i ) {
         SIDL_int__array_set1( Hypre_indices, i, i );
         SIDL_double__array_set1( Hypre_values, i, 1.0 );
      }
      Hypre_ParCSRVector_Clear( Hypre_b );
      Hypre_IJBuildVector_AddToValues( Hypre_ij_b, local_num_cols, Hypre_indices, Hypre_values );
      SIDL_int__array_destroy( Hypre_indices );
      SIDL_double__array_destroy( Hypre_values );
      Hypre_ParCSRVector_Print( Hypre_b, "test.addtovalues" );

      /* Clear,x=0, which restores its initial value of 0 */
      Hypre_ParCSRVector_Clear( Hypre_x );
      Hypre_ParCSRVector_Print( Hypre_x, "test.clear" );
   }
#endif


   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      if (myid == 0) printf("Solver:  AMG\n");
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

#define USE_BABEL_INTERFACE
#ifdef USE_BABEL_INTERFACE
      /* To call a Hypre solver:
         create, set comm, set operator, set other parameters,
         Setup (noop in this case), Apply */
      Hypre_AMG = Hypre_ParAMG__create();
      Hypre_Vector_b = (Hypre_Vector)Hypre_ParCSRVector__cast2( Hypre_b, "Hypre.Vector" );
      Hypre_Vector_x = (Hypre_Vector)Hypre_ParCSRVector__cast2( Hypre_x, "Hypre.Vector" );
      Hypre_op_A = (Hypre_Operator) Hypre_ParCSRMatrix__cast2( Hypre_parcsr_A, "Hypre.Operator" );
      ierr += Hypre_ParAMG_SetCommunicator( Hypre_AMG, (void *)comm );
      Hypre_ParAMG_SetOperator( Hypre_AMG, Hypre_op_A );

      printf("**** before calling Hypre_ParAMGSet*Parameter\n");
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "CoarsenType", (hybrid*coarsen_type));
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "MeasureType", measure_type);
      Hypre_ParAMG_SetDoubleParameter( Hypre_AMG, "Tol", tol);
      Hypre_ParAMG_SetDoubleParameter( Hypre_AMG, "StrongThreshold", strong_threshold);
      Hypre_ParAMG_SetDoubleParameter( Hypre_AMG, "TruncFactor", trunc_factor);
      /* note: log output not specified ... */
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "PrintLevel", ioutdat ); 
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "CycleType", cycle_type);
        dimsl[0] = 0;   dimsu[0] = 4;
        Hypre_num_grid_sweeps = SIDL_int__array_create( 1, dimsl, dimsu );
          for ( i=0; i<4; ++i )
             SIDL_int__array_set1( Hypre_num_grid_sweeps, i, num_grid_sweeps[i] );
      Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "NumGridSweeps", Hypre_num_grid_sweeps );
        dimsl[0] = 0;   dimsu[0] = 4;
        Hypre_grid_relax_type = SIDL_int__array_create( 1, dimsl, dimsu );
        for ( i=0; i<4; ++i )
           SIDL_int__array_set1( Hypre_grid_relax_type, i, grid_relax_type[i] );
      Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "GridRelaxType", Hypre_grid_relax_type );
        dimsl[0] = 0;   dimsu[0] = max_levels;
        Hypre_relax_weight = SIDL_double__array_create( 1, dimsl, dimsu );
        for ( i=0; i<max_levels; ++i )
           SIDL_double__array_set1( Hypre_relax_weight, i, relax_weight[i] );
      Hypre_ParAMG_SetDoubleArrayParameter( Hypre_AMG, "RelaxWeight", Hypre_relax_weight );
        dimsl[0] = 0;   dimsu[0] = max_levels;
        Hypre_smooth_option = SIDL_int__array_create( 1, dimsl, dimsu );
        for ( i=0; i<max_levels; ++i )
           SIDL_int__array_set1( Hypre_smooth_option, i, smooth_option[i] );
      Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "SmoothOption", Hypre_smooth_option );
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "SmoothNumSweep", smooth_num_sweep);
        dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
        Hypre_grid_relax_points = SIDL_int__array_create( 2, dimsl, dimsu );
        for ( i=0; i<4; ++i ) for ( j=0; j<4; ++j )
           SIDL_int__array_set2( Hypre_grid_relax_points, i, j, grid_relax_points[i][j] );
      Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "GridRelaxPoints", Hypre_grid_relax_points );
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "MaxLevels", max_levels);
      Hypre_ParAMG_SetDoubleParameter( Hypre_AMG, "MaxRowSum", max_row_sum);
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "DebugFlag", debug_flag);
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "Variant", variant);
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "Overlap", overlap);
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "DomainType", domain_type);
      Hypre_ParAMG_SetDoubleParameter( Hypre_AMG, "SchwarzRlxWeight", schwarz_rlx_weight);
      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "NumFunctions", num_functions);
      if (num_functions > 1) {
          dimsl[0] = 0;   dimsu[0] = num_functions;
           Hypre_dof_func = SIDL_int__array_create( 1, dimsl, dimsu );
           for ( i=0; i<num_functions; ++i )
              SIDL_int__array_set1( Hypre_dof_func, i, dof_func[i] );
	 Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "DofFunc", Hypre_dof_func );
      }
      log_level = 3;
      Hypre_ParAMG_SetLogging( Hypre_AMG, log_level );

      ierr += Hypre_ParAMG_Setup( Hypre_AMG, Hypre_Vector_b, Hypre_Vector_x );
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      ierr += Hypre_ParAMG_Apply( Hypre_AMG, Hypre_Vector_b, &Hypre_Vector_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      if ( log_level > 2 ) {
         /* print residual... */
           Hypre_y = Hypre_ParCSRVector__create();
           Hypre_ij_y = (Hypre_IJBuildVector) Hypre_ParCSRVector__cast2
              ( Hypre_y, "Hypre.IJBuildVector" );
           /* adjust reference counting system for new data type: */
           Hypre_IJBuildVector_addReference( Hypre_ij_y );
           Hypre_ParCSRVector_deleteReference( Hypre_y );
           ierr += Hypre_IJBuildVector_SetCommunicator( Hypre_ij_y, (void *)comm );
           ierr += Hypre_IJBuildVector_Create( Hypre_ij_y, (void *)comm,
                                               first_local_col,last_local_col );
           ierr += Hypre_IJBuildVector_Initialize( Hypre_ij_y );
           y = Hypre_ParCSRVector__cast2( Hypre_y, "Hypre.Vector" );
           ierr += Hypre_ParAMG_GetResidual( Hypre_AMG, &y );
           Hypre_ParCSRVector_Print( Hypre_y, "test.residual" );
           Hypre_IJBuildVector_deleteReference( Hypre_ij_y ); /* delete y */
      }

      /* Break encapsulation so that the rest of the driver stays the same */
      temp_vecdata = Hypre_ParCSRVector__get_data( Hypre_x );
      ij_x = temp_vecdata ->ij_b ;
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

#else
      HYPRE_BoomerAMGCreate(&amg_solver); 
      HYPRE_BoomerAMGSetCoarsenType(amg_solver, (hybrid*coarsen_type));
      HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      HYPRE_BoomerAMGSetTol(amg_solver, tol);
      HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
/* note: log is written to standard output, not to file */
      HYPRE_BoomerAMGSetPrintLevel(amg_solver, ioutdat);
      HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log"); 
      HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      HYPRE_BoomerAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      HYPRE_BoomerAMGSetGridRelaxType(amg_solver, grid_relax_type);
      HYPRE_BoomerAMGSetRelaxWeight(amg_solver, relax_weight);
      HYPRE_BoomerAMGSetSmoothOption(amg_solver, smooth_option);
      HYPRE_BoomerAMGSetSmoothNumSweep(amg_solver, smooth_num_sweep);
      HYPRE_BoomerAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      HYPRE_BoomerAMGSetMaxRowSum(amg_solver, max_row_sum);
      HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);
      HYPRE_BoomerAMGSetVariant(amg_solver, variant);
      HYPRE_BoomerAMGSetOverlap(amg_solver, overlap);
      HYPRE_BoomerAMGSetDomainType(amg_solver, domain_type);
      HYPRE_BoomerAMGSetSchwarzRlxWeight(amg_solver, schwarz_rlx_weight);
      HYPRE_BoomerAMGSetNumFunctions(amg_solver, num_functions);
      if (num_functions > 1)
	 HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);

      HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);
      HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
#endif

      HYPRE_BoomerAMGDestroy(amg_solver);
#endif /* USE_BABEL_INTERFACE*/
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG 
    *-----------------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2 || solver_id == 8 || 
	solver_id == 12 || solver_id == 43)
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);
 
#ifdef USE_BABEL_INTERFACE
      Hypre_PCG = Hypre_PCG__create();
      Hypre_Vector_b = (Hypre_Vector)Hypre_ParCSRVector__cast2( Hypre_b, "Hypre.Vector" );
      Hypre_Vector_x = (Hypre_Vector)Hypre_ParCSRVector__cast2( Hypre_x, "Hypre.Vector" );
      Hypre_op_A = (Hypre_Operator) Hypre_ParCSRMatrix__cast2( Hypre_parcsr_A, "Hypre.Operator" );
      ierr += Hypre_PCG_SetCommunicator( Hypre_PCG, (void *)comm );
      Hypre_PCG_SetOperator( Hypre_PCG, Hypre_op_A );
      Hypre_PCG_SetIntParameter( Hypre_PCG, "Max Iter", 500 );
      Hypre_PCG_SetDoubleParameter( Hypre_PCG, "Tol", tol );
      Hypre_PCG_SetIntParameter( Hypre_PCG, "Two Norm", 1 );
      Hypre_PCG_SetIntParameter( Hypre_PCG, "Rel Change", 0 );
      Hypre_PCG_SetPrintLevel( Hypre_PCG, 1 );
      ierr += Hypre_PCG_Setup( Hypre_PCG, Hypre_Vector_b, Hypre_Vector_x );

#else
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter(pcg_solver, 500);
      HYPRE_PCGSetTol(pcg_solver, tol);
      HYPRE_PCGSetTwoNorm(pcg_solver, 1);
      HYPRE_PCGSetRelChange(pcg_solver, 0);
      HYPRE_PCGSetPrintLevel(pcg_solver, 1);
#endif /* USE_BABEL_INTERFACE*/
 
      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
	 ioutdat = 1;
         if (myid == 0) printf("Solver: AMG-PCG\n");
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, ioutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetSmoothOption(pcg_precond, smooth_option);
         HYPRE_BoomerAMGSetSmoothNumSweep(pcg_precond, smooth_num_sweep);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
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

#ifdef USE_BABEL_INTERFACE
      /* To call a Hypre solver:
         create, set comm, set operator, set other parameters,
         Setup (noop in this case), Apply */
      Hypre_ParDiagScale = Hypre_ParDiagScale__create();
      ierr += Hypre_ParDiagScale_SetCommunicator( Hypre_ParDiagScale, (void *)comm );
      Hypre_ParDiagScale_SetOperator( Hypre_ParDiagScale, Hypre_op_A );
      ierr += Hypre_ParDiagScale_Setup( Hypre_ParDiagScale,
                                        Hypre_Vector_b, Hypre_Vector_x );
      Hypre_SolverPC = (Hypre_Solver) Hypre_ParDiagScale__cast2
         ( Hypre_ParDiagScale, "Hypre.Solver" );
      ierr += Hypre_PCG_SetPreconditioner( Hypre_PCG, Hypre_SolverPC );
#else
         if (myid == 0) printf("Solver: DS-PCG\n");
         pcg_precond = NULL;

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                             pcg_precond);
         hypre_TFree(smooth_option);
#endif /* USE_BABEL_INTERFACE */

      }
      else if (solver_id == 8)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-PCG\n");

	 HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &pcg_precond);
         HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
         HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
         HYPRE_ParaSailsSetLogging(pcg_precond, ioutdat);

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup,
                             pcg_precond);
         hypre_TFree(smooth_option);
      }
      else if (solver_id == 12)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: Schwarz-PCG\n");

	 HYPRE_SchwarzCreate(&pcg_precond);
	 HYPRE_SchwarzSetVariant(pcg_precond, variant);
	 HYPRE_SchwarzSetOverlap(pcg_precond, overlap);
	 HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
         HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSetup,
                             pcg_precond);
         hypre_TFree(smooth_option);
      }
      else if (solver_id == 43)
      {
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-PCG\n");

         HYPRE_EuclidCreate(MPI_COMM_WORLD, &pcg_precond);

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
         hypre_TFree(smooth_option);
      }
 
#ifdef USE_BABEL_INTERFACE
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      ierr += Hypre_PCG_Apply( Hypre_PCG, Hypre_Vector_b, &Hypre_Vector_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += Hypre_PCG_GetIntValue( Hypre_PCG, "Number of Iterations",
                                     &num_iterations );
      ierr += Hypre_PCG_GetDoubleValue( Hypre_PCG, "Final Relative Residual Norm",
                                &final_res_norm );

      if ( log_level > 2 ) {
         /* print residual... */
           Hypre_y = Hypre_ParCSRVector__create();
           Hypre_ij_y = (Hypre_IJBuildVector) Hypre_ParCSRVector__cast2
              ( Hypre_y, "Hypre.IJBuildVector" );
           /* adjust reference counting system for new data type: */
           Hypre_IJBuildVector_addReference( Hypre_ij_y );
           Hypre_ParCSRVector_deleteReference( Hypre_y );
           ierr += Hypre_IJBuildVector_SetCommunicator( Hypre_ij_y, (void *)comm );
           ierr += Hypre_IJBuildVector_Create( Hypre_ij_y, (void *)comm,
                                               first_local_col,last_local_col );
           ierr += Hypre_IJBuildVector_Initialize( Hypre_ij_y );
           y = Hypre_ParCSRVector__cast2( Hypre_y, "Hypre.Vector" );
           ierr += Hypre_PCG_GetResidual( Hypre_PCG, &y );
           Hypre_ParCSRVector_Print( Hypre_y, "test.residual" );
           Hypre_IJBuildVector_deleteReference( Hypre_ij_y ); /* delete y */
      }

      /* Break encapsulation so that the rest of the driver stays the same */
      temp_vecdata = Hypre_ParCSRVector__get_data( Hypre_x );
      ij_x = temp_vecdata ->ij_b ;
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

#else
      HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
        printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
        return(-1);
      }
      else 
        if (myid == 0)
          printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");
      HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
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
      else if (solver_id == 43)
      {
	/* HYPRE_EuclidPrintParams(pcg_precond); */
        HYPRE_EuclidDestroy(pcg_precond);
      }

#endif /* USE_BABEL_INTERFACE */

      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
 
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES 
    *-----------------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4 || solver_id == 7 
                      || solver_id == 18 || solver_id == 44)
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_GMRESSetKDim(pcg_solver, k_dim);
      HYPRE_GMRESSetMaxIter(pcg_solver, 1000);
      HYPRE_GMRESSetTol(pcg_solver, tol);
      HYPRE_GMRESSetLogging(pcg_solver, 1);
 
      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-GMRES\n");

         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, ioutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetSmoothOption(pcg_precond, smooth_option);
         HYPRE_BoomerAMGSetSmoothNumSweep(pcg_precond, smooth_num_sweep);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
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
         if (myid == 0) printf("Solver: DS-GMRES\n");
         pcg_precond = NULL;

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                               pcg_precond);
         hypre_TFree(smooth_option);
      }
      else if (solver_id == 7)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) printf("Solver: PILUT-GMRES\n");

         ierr = HYPRE_ParCSRPilutCreate( MPI_COMM_WORLD, &pcg_precond ); 
         if (ierr) {
	   printf("Error in ParPilutCreate\n");
         }

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                               pcg_precond);
         hypre_TFree(smooth_option);

         if (drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
               drop_tol );

         if (nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
               nonzeros_to_keep );
      }
      else if (solver_id == 18)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-GMRES\n");

	 HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &pcg_precond);
	 HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
         HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
         HYPRE_ParaSailsSetLogging(pcg_precond, ioutdat);
	 HYPRE_ParaSailsSetSym(pcg_precond, 0);

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup,
                               pcg_precond);
         hypre_TFree(smooth_option);
      }
      else if (solver_id == 44)
      {
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-GMRES\n");

         HYPRE_EuclidCreate(MPI_COMM_WORLD, &pcg_precond);

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
         hypre_TFree(smooth_option);
      }
 
      HYPRE_GMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
        printf("HYPRE_GMRESGetPrecond got bad precond\n");
        return(-1);
      }
      else
        if (myid == 0)
          printf("HYPRE_GMRESGetPrecond got good precond\n");
      HYPRE_GMRESSetup
         (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_GMRESSolve
         (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_GMRESGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_GMRESSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_GMRESSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
#endif

      HYPRE_ParCSRGMRESDestroy(pcg_solver);
 
      if (solver_id == 3)
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
        /* HYPRE_EuclidPrintParams(pcg_precond); */
        HYPRE_EuclidDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         printf("\n");
         printf("GMRES Iterations = %d\n", num_iterations);
         printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using BiCGSTAB 
    *-----------------------------------------------------------*/

   if (solver_id == 9 || solver_id == 10 || solver_id == 11 || solver_id == 45)
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_BiCGSTABSetMaxIter(pcg_solver, 1000);
      HYPRE_BiCGSTABSetTol(pcg_solver, tol);
      HYPRE_BiCGSTABSetLogging(pcg_solver, 1);
 
      if (solver_id == 9)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-BiCGSTAB\n");

         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, ioutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetSmoothOption(pcg_precond, smooth_option);
         HYPRE_BoomerAMGSetSmoothNumSweep(pcg_precond, smooth_num_sweep);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
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
         if (myid == 0) printf("Solver: DS-BiCGSTAB\n");
         pcg_precond = NULL;

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                  pcg_precond);
         hypre_TFree(smooth_option);
      }
      else if (solver_id == 11)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) printf("Solver: PILUT-BiCGSTAB\n");

         ierr = HYPRE_ParCSRPilutCreate( MPI_COMM_WORLD, &pcg_precond ); 
         if (ierr) {
	   printf("Error in ParPilutCreate\n");
         }

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                                  pcg_precond);
         hypre_TFree(smooth_option);

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
         if (myid == 0) printf("Solver: Euclid-BICGSTAB\n");

         HYPRE_EuclidCreate(MPI_COMM_WORLD, &pcg_precond);

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
         hypre_TFree(smooth_option);
      }
 
      HYPRE_BiCGSTABSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_BiCGSTABSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_BiCGSTABGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_BiCGSTABSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_BiCGSTABSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
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
        /* HYPRE_EuclidPrintParams(pcg_precond); */
        HYPRE_EuclidDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         printf("\n");
         printf("BiCGSTAB Iterations = %d\n", num_iterations);
         printf("Final BiCGSTAB Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using CGNR 
    *-----------------------------------------------------------*/

   if (solver_id == 5 || solver_id == 6)
   {
      time_index = hypre_InitializeTiming("CGNR Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRCGNRCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_CGNRSetMaxIter(pcg_solver, 1000);
      HYPRE_CGNRSetTol(pcg_solver, tol);
      HYPRE_CGNRSetLogging(pcg_solver, 1);
 
      if (solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-CGNR\n");

         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, ioutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetSmoothOption(pcg_precond, smooth_option);
         HYPRE_BoomerAMGSetSmoothNumSweep(pcg_precond, smooth_num_sweep);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
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
         if (myid == 0) printf("Solver: DS-CGNR\n");
         pcg_precond = NULL;

         HYPRE_CGNRSetPrecond(pcg_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                              pcg_precond);
         hypre_TFree(smooth_option);
      }
 
      HYPRE_CGNRGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
        printf("HYPRE_ParCSRCGNRGetPrecond got bad precond\n");
        return(-1);
      }
      else
        if (myid == 0)
          printf("HYPRE_ParCSRCGNRGetPrecond got good precond\n");
      HYPRE_CGNRSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("CGNR Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_CGNRSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_CGNRGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_CGNRGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_CGNRSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_CGNRSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
#endif

      HYPRE_ParCSRCGNRDestroy(pcg_solver);
 
      if (solver_id == 5)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   HYPRE_IJVectorGetObjectType(ij_b, &j);
   HYPRE_IJVectorPrint(ij_b, "driver.out.b");
   HYPRE_IJVectorPrint(ij_x, "driver.out.x");

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_IJMatrixDestroy(ij_A);
   HYPRE_IJVectorDestroy(ij_b);
   HYPRE_IJVectorDestroy(ij_x);


   if( Hypre_num_grid_sweeps )
      SIDL_int__array_destroy( Hypre_num_grid_sweeps );
   if( Hypre_grid_relax_type )
      SIDL_int__array_destroy( Hypre_grid_relax_type );
   if( Hypre_relax_weight )
      SIDL_double__array_destroy( Hypre_relax_weight );
   if( Hypre_smooth_option )
      SIDL_int__array_destroy( Hypre_smooth_option );
   if( Hypre_grid_relax_points )
      SIDL_int__array_destroy( Hypre_grid_relax_points );
   if( Hypre_dof_func )
      SIDL_int__array_destroy( Hypre_dof_func );


/*
   hypre_FinalizeMemoryDebug();
*/

   MPI_Finalize();

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

int
BuildParFromFile( int                  argc,
                  char                *argv[],
                  int                  arg_index,
                  HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix A;

   int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   HYPRE_ParCSRMatrixRead(MPI_COMM_WORLD, filename,&A);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParLaplacian( int                  argc,
                   char                *argv[],
                   int                  arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

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
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian:\n");
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
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

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(MPI_COMM_WORLD, 
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

int
BuildParDifConv( int                  argc,
                 char                *argv[],
                 int                  arg_index,
                 HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;
   double              ax, ay, az;
   double              hinx,hiny,hinz;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

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

   hinx = 1./(nx+1);
   hiny = 1./(ny+1);
   hinz = 1./(nz+1);

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
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Convection-Diffusion: \n");
      printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");  
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
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

   A = (HYPRE_ParCSRMatrix) GenerateDifConv(MPI_COMM_WORLD,
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

int
BuildParFromOneFile( int                  argc,
                     char                *argv[],
                     int                  arg_index,
                     HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix  A;
   HYPRE_CSRMatrix  A_CSR = NULL;

   int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix 
       *-----------------------------------------------------------*/
 
      A_CSR = HYPRE_CSRMatrixRead(filename);
   }
   HYPRE_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_CSR, NULL, NULL, &A);

   *A_ptr = A;

   if (myid == 0) HYPRE_CSRMatrixDestroy(A_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

int
BuildFuncsFromFiles(    int                  argc,
                        char                *argv[],
                        int                  arg_index,
                        HYPRE_ParCSRMatrix   parcsr_A,
                        int                **dof_func_ptr     )
{
/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

	printf (" Feature is not implemented yet!\n");	
	return(0);

}


int
BuildFuncsFromOneFile(  int                  argc,
                        char                *argv[],
                        int                  arg_index,
                        HYPRE_ParCSRMatrix   parcsr_A,
                        int                **dof_func_ptr     )
{
   char           *filename;

   int             myid, num_procs;
   int            *partitioning;
   int            *dof_func;
   int            *dof_func_local;
   int             i, j;
   int             local_size, global_size;
   MPI_Request	  *requests;
   MPI_Status	  *status, status0;
   MPI_Comm	   comm;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   comm = MPI_COMM_WORLD;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      FILE *fp;
      printf("  Funcs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * read in the data
       *-----------------------------------------------------------*/
      fp = fopen(filename, "r");

      fscanf(fp, "%d", &global_size);
      dof_func = hypre_CTAlloc(int, global_size);

      for (j = 0; j < global_size; j++)
      {
         fscanf(fp, "%d", &dof_func[j]);
      }

      fclose(fp);
 
   }
   HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A, &partitioning);
   local_size = partitioning[myid+1]-partitioning[myid];
   dof_func_local = hypre_CTAlloc(int,local_size);

   if (myid == 0)
   {
        requests = hypre_CTAlloc(MPI_Request,num_procs-1);
        status = hypre_CTAlloc(MPI_Status,num_procs-1);
        j = 0;
        for (i=1; i < num_procs; i++)
                MPI_Isend(&dof_func[partitioning[i]],
		partitioning[i+1]-partitioning[i],
                MPI_INT, i, 0, comm, &requests[j++]);
        for (i=0; i < local_size; i++)
                dof_func_local[i] = dof_func[i];
        MPI_Waitall(num_procs-1,requests, status);
        hypre_TFree(requests);
        hypre_TFree(status);
   }
   else
   {
        MPI_Recv(dof_func_local,local_size,MPI_INT,0,0,comm,&status0);
   }

   *dof_func_ptr = dof_func_local;

   if (myid == 0) hypre_TFree(dof_func);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors 
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

int
BuildRhsParFromOneFile( int                  argc,
                        char                *argv[],
                        int                  arg_index,
                        int                 *partitioning,
                        HYPRE_ParVector     *b_ptr     )
{
   char           *filename;

   HYPRE_ParVector b;
   HYPRE_Vector    b_CSR;

   int             myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Rhs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix 
       *-----------------------------------------------------------*/
 
      b_CSR = HYPRE_VectorRead(filename);
   }
   HYPRE_VectorToParVector(MPI_COMM_WORLD, b_CSR, partitioning,&b); 

   *b_ptr = b;

   HYPRE_VectorDestroy(b_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParLaplacian9pt( int                  argc,
                      char                *argv[],
                      int                  arg_index,
                      HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny;
   int                 P, Q;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q;
   double             *values;

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

   if ((P*Q) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian 9pt:\n");
      printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
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

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(MPI_COMM_WORLD,
                                  nx, ny, P, Q, p, q, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}
/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParLaplacian27pt( int                  argc,
                       char                *argv[],
                       int                  arg_index,
                       HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

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

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian_27pt:\n");
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
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

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(MPI_COMM_WORLD,
                               nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}
