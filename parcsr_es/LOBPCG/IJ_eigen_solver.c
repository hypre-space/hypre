/*BHEADER**********************************************************************
 * IJ_eigen_solver.c 
 *
 * $Revision$
 * Date: 10/7/2002
 * Authors: M. Argentati and A. Knyazev
 *
 * Test driver for lobpcg.
 * Do `driver -help' for usage info.
 * This driver is base on HYPRE IJ_linear_solvers 
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

/*------------------------------------------------------------------------*/
/* HYPRE includes                                                         */
/*------------------------------------------------------------------------*/
#include "HYPRE.h"
#include "IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "HYPRE_lobpcg.h"

/* lobpcg */
#define TRUE  1
#define FALSE 0
#define MATRIX_INPUT_MTX      2
#define assert2(ierr)   

static lobpcg_options opts;
static Matx *X;
HYPRE_ParVector *eigenvector; /* array of eigenvectors */

static HYPRE_Solver       pcg_solver;
static HYPRE_Solver       pcg_precond, pcg_precond_gotten;
static HYPRE_ParCSRMatrix  parcsr_A;
static HYPRE_ParCSRMatrix  parcsr_T;
static HYPRE_LobpcgData   lobpcgdata;

/* function prototypes for functions that are passed to lobpcg */
int Func_A1(HYPRE_ParVector x,HYPRE_ParVector y);
int Funct_Solve_A(HYPRE_ParVector x,HYPRE_ParVector y);
int Funct_Solve_T(HYPRE_ParVector x,HYPRE_ParVector y);

/* misc lobpcg function prototypes */
int Get_Lobpcg_Options(int argc,char **args,lobpcg_options *opts);
Matx *Mat_Alloc1();
int Mat_Size(Matx *A,int rc);
int Mat_Size_Mtx(char *file_name,int rc);
void HYPRE_Load_IJAMatrix2(HYPRE_ParCSRMatrix  *A_ptr,
     int matrix_input_type, char *matfile,int *partitioning);
int readmatrix(char Ain[],Matx *A,mst mat_storage_type,int *partitioning);
void PrintVector(double *data,int n,char fname[]);
int Mat_Free(Matx *A);
void PrintMatrix(double **data,int m,int n,char fname[]);

/* IJ_linear_solvers prototypes */
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
   int                 debug_flag;
   int                 ierr = 0;
   int                 i,j,k; 
   int                 indx, rest, tms;
   int                 max_levels = 25;
   int                 num_iterations; 
   double              final_res_norm;
   void               *object;

   HYPRE_IJMatrix      ij_A; 
   HYPRE_IJVector      ij_b;
   HYPRE_IJVector      ij_x;

   /*HYPRE_ParCSRMatrix  parcsr_A;*/
   HYPRE_ParVector     b=NULL;
   HYPRE_ParVector     x=NULL;

/*   HYPRE_Solver        amg_solver; */

/* lobpcg */
/* make these global variables
   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond, pcg_precond_gotten;
*/

   int                 num_procs, myid;
   int                 local_row;
   int                *row_sizes;
   int                *diag_sizes;
   int                *offdiag_sizes;
   int                *rows;
   int                 size;
   int                *ncols;
   int                *col_inds;
   int                *dof_func;
   int		       num_functions = 1;

   int		       time_index;
   MPI_Comm            comm = MPI_COMM_WORLD;
   int M, N;
   int first_local_row, last_local_row, local_num_rows;
   int first_local_col, last_local_col, local_num_cols;
   int local_num_vars;
   int variant, overlap, domain_type;
   double schwarz_rlx_weight;
   double *values, val;

   const double dt_inf = 1.e40;
   double dt = dt_inf;

   /* parameters for BoomerAMG */
   double   strong_threshold=0;
   double   trunc_factor=0;
   int      cycle_type=0;
   int      coarsen_type = 6;
   int      hybrid = 1;
   int      measure_type = 0;
   int     *num_grid_sweeps=NULL;  
   int     *grid_relax_type=NULL;   
   int    **grid_relax_points=NULL;
   int	    smooth_lev;
   int	    smooth_rlx = 8;
   int      relax_default;
   int      smooth_num_sweep = 1;
   int      num_sweep = 1;
   double  *relax_weight=NULL; 
   double   tol = 1.e-6, pc_tol = 0.;
   double   max_row_sum = 1.;

   /* parameters for ParaSAILS */
   double   sai_threshold = 0.1;
   double   sai_filter = 0.1;

   /* lobpcg */
   /* parameters for lobpcg */
   int     m = 0;
   int     bsize,iterations;
   int     M2, N2;
   int     *row_partitioning,*part2;
   int     lobpcg_flag=TRUE; /* do LobpcgSolve if TRUE */
   int     num_nonzeros=0;
   double  orth_frob_norm=0;

   /*HYPRE_LobpcgData lobpcgdata;*/
   double *eigval,**resvec,**eigvalhistory;
   int (*FuncT)(HYPRE_ParVector x,HYPRE_ParVector y)=NULL; /* function pointer */

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

   /* solver_id = 0; */
   /* lobpcg */
   solver_id = 12; /* set default to Schwarz-PCG */
   ioutdat = 3;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   /* lobpcg */
   /* get lobpcg options */
   arg_index = 1;
   while (arg_index < argc)
   {
      /* run IJ_linear_solvers without lobpcg */
      if ( strcmp(argv[arg_index], "-nolobpcg") == 0 )
      {
         arg_index++;
         lobpcg_flag=FALSE;
      }
      else
      {
         arg_index++;
      }
   }
   if (lobpcg_flag==TRUE)
   {
      opts.pcg_tol=LOBPCG_PCG_SOLVE_TOL;
      opts.pcg_max_itr=LOBPCG_PCG_SOLVE_MAXITR;
      Get_Lobpcg_Options(argc,argv,&opts);
   }
   else Get_Lobpcg_Options(argc,argv,&opts);

   print_usage = 0;
   arg_index = 1;

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
      else if ( strcmp(argv[arg_index], "-smtype") == 0 )
      {
         arg_index++;
         smooth_rlx = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smlv") == 0 )
      {
         arg_index++;
         smooth_lev = atoi(argv[arg_index++]);
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

   /* check allowable solvers for pcg and lobpcg */
   if (!(solver_id == 1 || solver_id == 2 || solver_id == 8 ||
           solver_id == 12 || solver_id == 43))
   {
         if (myid==0)
         {
            printf("\nSolver invalid (solver_id=%d)\n",solver_id);
            printf("Valid solvers for PCG/LOBPCG are: 1=AMG-PCG,2=DS-PCG,8=ParaSails-PCG\n");
            printf("12=Schwarz-PCG (default),43=Euclid-PCG\n");
         }
         exit(1);
   }   

   /* lobpcg */
   if (lobpcg_flag==TRUE)
   {
      /* allocate storage */
      X=Mat_Alloc1();

      /* check for matrix market file */
      if (opts.flag_A) build_matrix_type = 6;

      /* check for preconditioner */
      if(opts.flag_T)
      {
         if (solver_id>0) FuncT=Funct_Solve_T; /* use matrix T for solve */
         else FuncT=NULL;  /* no solve/precondioner */
      }
      else
      {
         if(solver_id>0) FuncT=Funct_Solve_A; /* use A itself for solve */
         else FuncT=NULL;  /* no solve/precondioner */
      }
   }


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



   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
   if ( (print_usage) && (myid == 0) )
   {
      printf("\n");
      printf("Usage: %s [<options>]\n", argv[0]);

      /* lobpcg */
      if (lobpcg_flag==TRUE)
      {
         printf("Specific lobpcg options:\n");
         printf("  -ain  <filename> : read symmetric input matrix in ascii Matrix Market format\n");
         printf("  -vin  <filename> : read initial eigenvectors (columns) in ascii Matrix Market format\n");
         printf("  -tin  <filename> : read symmetric positive definit preconditioner matrix in ascii Matrix Market format\n");
         printf("  -vrand <int>     : randomize initial eigenvectors\n");
         printf("  -veye  <int>     : change initial eigenvectors into an m x bsize identity\n");
         printf("  -chk   <int>     : check orthogonality of final eigenvectors \n");
         printf("  -itr  <int>      : override default max iteration count\n");
         printf("  -tol  <scalar>   : tolerance override\n");
         printf("  -v    <int>      : verbose - display a lot of output\n");
         printf("  -pcgitr <int>    : maximum iterations for pcg solve (default=1)\n");
         printf("                   : Note: set this to 0 for no precondioner\n");
         printf("  -pcgtol <scalar> : tol for pcg solve\n");
         printf("  -nolobpcg        : don't execute LobpcgSolve, only IJ_linear_solvers\n");
         printf("  -printA          : print out matrix A\n");
         printf("\n");
         printf("Internally generated files used by lobpcg:\n");
         printf("  -laplacian [<options>] : build 7pt 3D laplacian problem (default)\n");
         printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
         printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
         printf("\n");
         printf("Solvers supported for lobpcg:\n");
         printf("  -solver <ID>\n");
         printf("     1=AMG-PCG\n");
         printf("     2=DS-PCG\n");
         printf("     8=ParaSails-PCG\n");
         printf("     12=Schwarz-PCG (default)\n");  
         printf("     43=Euclid-PCG\n");
         printf("\n");
         printf("Run these lobpcg programs:\n");
         printf("  IJ_eigen_solver -ain poisson21.mtx -vrand 5  -solver 43 -pcgitr 10\n");
         printf("  IJ_eigen_solver -9pt -solver 43 -n 20 20\n");
         printf("Run lobpcg on multiple processors\n");
         printf("  mpirun -np 2 IJ_eigen_solver -27pt -solver 12 -vrand 5 -pcgitr 2\n");
         printf("\n");
         printf("\n");
      }
      
      printf(" WARNING: SOME OPTIONS MAY BE BROKEN!\n\n");
      printf("  -fromijfile <filename>     : ");
      printf("matrix read in IJ format from distributed files\n");
      printf("  -fromparcsrfile <filename> : ");
      printf("matrix read in ParCSR format from distributed files\n");
      printf("  -fromonecsrfile <filename> : ");
      printf("matrix read in CSR format from a file on one processor\n");
      printf("\n");
      printf("  -laplacian [<options>] : build 7pt 3D laplacian problem (default) \n");
      printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
      printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
      printf("    -n <nx> <ny> <nz>    : total problem size \n");
      printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("\n");
      printf("  -exact_size            : inserts immediately into ParCSR structure\n");
      printf("  -storage_low           : allocates not enough storage for aux struct\n");
      printf("  -concrete_parcsr       : use parcsr matrix type as concrete type\n");
      printf("\n");
      printf("  -solver <ID>           : solver ID\n");
      printf("       1=AMG-PCG        \n");
      printf("       2=DS-PCG         \n");     
      printf("       8=ParaSails-PCG  \n");
      printf("       12=Schwarz-PCG   \n");     
      printf("       43=Euclid-PCG    \n");
      printf("\n");
      printf("  -cljp                 : CLJP coarsening \n");
      printf("  -ruge                 : Ruge coarsening (local)\n");
      printf("  -ruge3                : third pass on boundary\n");
      printf("  -ruge3c               : third pass on boundary, keep c-points\n");
      printf("  -ruge2b               : 2nd pass is global\n");
      printf("  -rugerlx              : relaxes special points\n");
      printf("  -falgout              : local ruge followed by LJP\n");
      printf("  -nohybrid             : no switch in coarsening\n");
      printf("  -gm                   : use global measures\n");
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
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("\nNumber of Processors: %d\n",num_procs);
      printf("Running with these driver parameters:\n");
      printf("  Solver ID    = %d\n\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

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
   else if ( build_matrix_type == 6 )
   {
      /* read  matrix from matrix market file */
      M=Mat_Size_Mtx(opts.Ain,1); /* get number of rows */
      ierr=hypre_GeneratePartitioning(M,num_procs,&row_partitioning);
      HYPRE_Load_IJAMatrix2(&parcsr_A,MATRIX_INPUT_MTX,opts.Ain,row_partitioning);
      HYPRE_ParCSRMatrixGetDims(parcsr_A,&M,&N);
      if (myid == 0) printf("  Input Matrix A: %d x %d:\n\n",M,N);
   }
   else
   {
      printf("You have asked for an unsupported problem with\n");
      printf("build_matrix_type = %d.\n", build_matrix_type);
      return(-1);
   }

   time_index = hypre_InitializeTiming("Spatial operator");
   hypre_BeginTiming(time_index);

   /* lobpcg */
   if (lobpcg_flag==TRUE)
   {
      /* get partitioning */
      if (opts.flag_A ==FALSE)
         HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A,&row_partitioning);
      HYPRE_ParCSRMatrixGetDims(parcsr_A,&M,&N);
      m=M;

      /* read and assemble matrix T used for solve/preconditioning */
      if(opts.flag_T)
      {
         HYPRE_Load_IJAMatrix2(&parcsr_T,MATRIX_INPUT_MTX,opts.Tin,row_partitioning);
      }

      /* check number of rows of X and T */
      if(opts.flag_A)
      {
         if(opts.flag_T)
         {
            HYPRE_ParCSRMatrixGetDims(parcsr_T,&M2,&N2);
            if (m!=M2)
            {
               printf("Number or rows of matrix T does not equal number of rows of A\n");
               exit(1);
            }
         }
      }
   }
   
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

     ierr += HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                   first_local_col, last_local_col,
                                   &ij_A );

     ierr += HYPRE_IJMatrixSetObjectType( ij_A, HYPRE_PARCSR );
  

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
       ierr += HYPRE_IJMatrixSetDiagOffdSizes( ij_A, 
                                        (const int *) diag_sizes,
                                        (const int *) offdiag_sizes );
       hypre_TFree(diag_sizes);
       hypre_TFree(offdiag_sizes);
     
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
       row_sizes = hypre_CTAlloc(int, local_num_rows);

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

       ierr = HYPRE_IJMatrixSetRowSizes ( ij_A, (const int *) row_sizes );

       hypre_TFree(row_sizes);

       ierr = HYPRE_IJMatrixInitialize( ij_A );

       /* Loop through all locally stored rows and insert them into ij_matrix */
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

     ierr += HYPRE_IJMatrixAssemble( ij_A );

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
      
   ierr += HYPRE_IJMatrixAddToValues( ij_A,
                                      local_num_rows,
                                      ncols, rows,
                                      (const int *) col_inds,
                                      (const double *) values );

   hypre_TFree(values);
   hypre_TFree(col_inds);
   hypre_TFree(rows);
   hypre_TFree(ncols);

   /* If sparsity pattern is not changed since last IJMatrixAssemble call,
      this should be a no-op */

   ierr += HYPRE_IJMatrixAssemble( ij_A );

   /*-----------------------------------------------------------
    * Fetch the resulting underlying matrix out
    *-----------------------------------------------------------*/

   if (build_matrix_type > 1)
     ierr += HYPRE_ParCSRMatrixDestroy(parcsr_A);

   ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
   parcsr_A = (HYPRE_ParCSRMatrix) object;

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("RHS and Initial Guess");
   hypre_BeginTiming(time_index);
   
   /* lobpcg */
   if (lobpcg_flag==TRUE)
   {
      bsize=0;
      /* check to see if we need to read in matrix X */
      if(opts.flag_V)
      {
         readmatrix(opts.Vin,X,HYPRE_VECTORS,row_partitioning);
         bsize=Mat_Size(X,2);
      }

      /* check number of rows of X and A */
      if(opts.flag_A)
      {
         if(opts.flag_V)
         {
           if (m!=Mat_Size(X,1))
           {
              printf("Number or rows of matrix X does not equal number of rows of A\n");
              exit(1);
           }
         }
      }

      /* check for random or identity initial eigenvectors */
      if (opts.Vrand>0) bsize=opts.Vrand;
      else if (opts.Veye>0) bsize=opts.Veye;
      else if (opts.flag_V != TRUE && opts.Vrand ==0)
      {
          /* default is one random vector */
          bsize=1;
          opts.Vrand=1;
      }
      if (bsize>0)
      {
         /* allocate memory */
         if ((eigenvector=(HYPRE_ParVector *) malloc(bsize*sizeof(HYPRE_ParVector)))==NULL)
         {
           printf("Could not allocate memory.\n");
           assert(0);
         }

         /* create initial parallel vectors */
         for (i=0; i<bsize; i++)
         {
           part2=CopyPartition(row_partitioning);
           ierr=HYPRE_ParVectorCreate(MPI_COMM_WORLD,m,part2,&eigenvector[i]);assert2(ierr);
           ierr=HYPRE_ParVectorInitialize(eigenvector[i]);assert2(ierr);
         }
         
         /* copy input vectors to eigenvector */
         if(opts.flag_V)
         {
            for (i=0; i<bsize; i++)
            {
               ierr=HYPRE_ParVectorCopy(X->vsPar[i],eigenvector[i]);assert2(ierr);
            }

            /* free memory */
            Mat_Free(X);free(X);
         }
      }
   }
   
   /*If lobpcg_flag!=TRUE we do not need to run this part, */
   /*but I was not able to comment it out without breaking the driver */
   if ( build_rhs_type == 2 )
   {
      if (lobpcg_flag!=TRUE)
      {   
      	if (myid == 0)
      	{
       	 printf("  RHS vector has unit components\n");
       	 printf("  Initial guess is 0\n");
      	}
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

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else 
   {
      printf("only build_rhs_type == 2 is implemented here\n");
      return(-1);
#if 0
/* RHS */
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, part_b, &b);
#endif
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Vector Setup", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   /* lobpcg */
   if (lobpcg_flag==FALSE) 
   {
      HYPRE_IJMatrixPrint(ij_A, "driver.out.A");
      HYPRE_IJVectorPrint(ij_x, "driver.out.x0");
   }
   else if ((lobpcg_flag==TRUE) && (opts.printA==TRUE))
   {
      HYPRE_IJMatrixPrint(ij_A, "driver.out.A");
      /* capability to print out X containing input/output vectors
         will be added at later date */
   }

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
    * Solve the system using PCG/LOBPCG 
    *-----------------------------------------------------------*/
   if (solver_id == 1 || solver_id == 2 || solver_id == 8 || 
	solver_id == 12 || solver_id == 43)
   {
      time_index = hypre_InitializeTiming("PCG/LOBPCG Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);

      /* lobpcg */
      if (lobpcg_flag==TRUE)
      {
         HYPRE_PCGSetMaxIter(pcg_solver, opts.pcg_max_itr);
         HYPRE_PCGSetTol(pcg_solver, opts.pcg_tol);
         HYPRE_PCGSetTwoNorm(pcg_solver, 1);
         HYPRE_PCGSetRelChange(pcg_solver, 0);
         HYPRE_PCGSetLogging(pcg_solver, 1);

         /*------------------------------------------------------------------
         * Setup lobpcg
         *---------------------------------------------------------------*/
         HYPRE_LobpcgCreate(&lobpcgdata);
         if (opts.flag_tol) HYPRE_LobpcgSetTolerance(lobpcgdata,opts.tol);
         if (opts.flag_itr) HYPRE_LobpcgSetMaxIterations(lobpcgdata,opts.max_iter_count);
         if (opts.flag_v) HYPRE_LobpcgSetVerbose(lobpcgdata);
         if (opts.flag_orth_check==TRUE) HYPRE_LobpcgSetOrthCheck(lobpcgdata);

         if (opts.Vrand>0) HYPRE_LobpcgSetRandom(lobpcgdata);
         else if (opts.Veye>0) HYPRE_LobpcgSetEye(lobpcgdata);

         HYPRE_LobpcgSetBlocksize(lobpcgdata,bsize);
         HYPRE_LobpcgSetSolverFunction(lobpcgdata,FuncT);
      }
      else
      {
         if (opts.pcg_max_flag==FALSE) opts.pcg_max_itr=500; 
         HYPRE_PCGSetMaxIter(pcg_solver, opts.pcg_max_itr);
         HYPRE_PCGSetTol(pcg_solver, tol);
         HYPRE_PCGSetTwoNorm(pcg_solver, 1);
         HYPRE_PCGSetRelChange(pcg_solver, 0);
         HYPRE_PCGSetLogging(pcg_solver, 1);
      }
 
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
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_rlx);
         HYPRE_BoomerAMGSetSmoothNumLevels(pcg_solver, smooth_lev);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweep);
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
         if (myid == 0) printf("Solver: DS-PCG\n");
         pcg_precond = NULL;

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                             pcg_precond);
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
      }
 
      HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
        printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
        return(-1);
      }
      else 
        if (myid == 0)
          printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");

      /* lobpcg */
      if (lobpcg_flag==TRUE)
      {
         HYPRE_LobpcgSetup(lobpcgdata);
	 if(!opts.flag_T)
         {
	 	ierr=HYPRE_ParCSRPCGSetup(pcg_solver,parcsr_A,b,x);assert2(ierr); 
 	 }
 	 else
 	 {
		ierr=HYPRE_ParCSRPCGSetup(pcg_solver,parcsr_T,b,x);assert2(ierr);
 	 }
      }
      else
      {
         HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      }
      
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG/LOBPCG Solve");
      hypre_BeginTiming(time_index);
 
      /* lobpcg */
      if (lobpcg_flag==TRUE)
      {
         /* setup to get total number of nonzeros */
         hypre_ParCSRMatrixSetNumNonzeros((hypre_ParCSRMatrix *) parcsr_A);
         if (myid==0)
         {
            printf("=============================================\n");
            printf("Lobpcg Solver:\n");
            printf("=============================================\n");
            printf("  Matrix: %d x %d:\n",M,N);
            num_nonzeros=hypre_ParCSRMatrixNumNonzeros((hypre_ParCSRMatrix *) parcsr_A);
            printf("  Number of nonzeros: %d\n",num_nonzeros);
            printf("  pcg_max_itr=\t%d\n", opts.pcg_max_itr);
            printf("  pcg_tol=\t%12.6e\n",opts.pcg_tol);
         }
         HYPRE_LobpcgSolve(lobpcgdata,Func_A1,eigenvector,&eigval);

        /*------------------------------------------------------------------
         * get lobpcg output and print to matrix market files
         *---------------------------------------------------------------*/
         HYPRE_LobpcgGetIterations(lobpcgdata,&iterations);
         HYPRE_LobpcgGetResvec(lobpcgdata,&resvec);
         HYPRE_LobpcgGetEigvalHistory(lobpcgdata,&eigvalhistory);
         HYPRE_LobpcgGetBlocksize(lobpcgdata,&bsize);

         if (opts.flag_orth_check==TRUE)
         {
           HYPRE_LobpcgGetOrthCheckNorm(lobpcgdata,&orth_frob_norm);
           if (myid==0) printf("Eigenvector Orth. Check: FrobNorm(V'V-I)=%12.6e\n",
             orth_frob_norm);
         }

         /* print lobpcg results to matrix market files */
         if (myid==0)
         {
            printf("\n\n");
            PrintVector(eigval,bsize,"eigval.mtx");
            PrintMatrix(resvec,bsize,iterations,"resvec.mtx");
            PrintMatrix(eigvalhistory,bsize,iterations,"eigvalhistory.mtx");
         }
      }
      else
      {
         HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      }
 
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

      /* lobpcg */
      if (lobpcg_flag==TRUE)
      {
         HYPRE_LobpcgDestroy(lobpcgdata);
      }

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
        HYPRE_EuclidDestroy(pcg_precond);
      }
      
      if (lobpcg_flag!=TRUE)
      {
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
}
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   HYPRE_IJVectorGetObjectType(ij_b, &j);
   /* lobpcg */
   if (lobpcg_flag==FALSE)
   {
      HYPRE_IJVectorPrint(ij_b, "driver.out.b");
      HYPRE_IJVectorPrint(ij_x, "driver.out.x");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_IJMatrixDestroy(ij_A);
   HYPRE_IJVectorDestroy(ij_b);
   HYPRE_IJVectorDestroy(ij_x);

/*
   hypre_FinalizeMemoryDebug();
*/

   /* lobpcg */
   /* final cleanup */
   if (lobpcg_flag==TRUE)
   {
      /* destroy parallel eigenvectors */
      for (i=0; i<bsize; i++)
      {
         ierr=HYPRE_ParVectorDestroy(eigenvector[i]);assert2(ierr);
      }
   }

   MPI_Finalize();

   return (0);
}




/* lobpcg */
/*****************************************************************************/
int Func_A1(HYPRE_ParVector x,HYPRE_ParVector y)
{
   /* compute y=A*x */
   int ierr=0;

   ierr=HYPRE_ParCSRMatrixMatvec(1.0,parcsr_A,x,0.0,y);assert2(ierr);

   return 0;
}

/*****************************************************************************/
int Funct_Solve_A(HYPRE_ParVector b,HYPRE_ParVector x)
{
   /* Solve A*x=b  */
   /* use A itself */
   int ierr=0;

   ierr=HYPRE_ParCSRPCGSolve(pcg_solver,parcsr_A,b,x);assert2(ierr);

   return 0;
}

/*****************************************************************************/
int Funct_Solve_T(HYPRE_ParVector b,HYPRE_ParVector x)
{
   /* Solve T*x=b  */
   /* use matrix T */
   int ierr=0;

   ierr=HYPRE_ParCSRPCGSolve(pcg_solver,parcsr_T,b,x);assert2(ierr);

   return 0;
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
   int            *dof_func=NULL;
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
   HYPRE_Vector    b_CSR=NULL;

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
