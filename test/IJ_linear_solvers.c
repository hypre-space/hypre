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

#include "utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"

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
   int                 generate_matrix = 0;
   int                 build_matrix_type;
   int                 build_matrix_arg_index;
   int                 build_rhs_type;
   int                 build_rhs_arg_index;
   int                 build_funcs_type;
   int                 build_funcs_arg_index;
   int                 solver_id;
   int                 ioutdat;
   int                 debug_flag;
   int                 ierr,i,j,k; 
   int                 indx, rest, tms;
   int                 max_levels = 25;
   int                 num_iterations; 
   int                 num_sweep = 1;
   int                 smooth_num_sweep = 1;
   double              max_row_sum = 1.0;
   double              norm;
   double              final_res_norm;

   HYPRE_IJMatrix      ij_matrix; 
   HYPRE_IJVector      ij_b;
   HYPRE_IJVector      ij_x;
   HYPRE_IJVector      ij_v;

   /* concrete underlying type for ij_matrix defaults to parcsr. AJC. */
   /* int                 ij_matrix_storage_type=HYPRE_PARCSR; */
   int                 ij_vector_storage_type=HYPRE_PARCSR;

   HYPRE_ParCSRMatrix  parcsr_A;
   HYPRE_ParCSRMatrix  A;
   HYPRE_ParVector     b;
   HYPRE_ParVector     x;

   HYPRE_Solver        amg_solver;
   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond, pcg_precond_gotten;

   int                 num_procs, myid;
   int                 global_n;
   int                 local_row;
   const int          *partitioning;
   int                *part_b;
   int                *part_x;
   int                *glob_vec_indices;
   int                *value_indices;
   int                *row_sizes;
   int                *diag_sizes;
   int                *offdiag_sizes;
   int                *dof_func;
   int		       num_functions = 1;

   int		       time_index;
   MPI_Comm comm;
   int M, N;
   int first_local_row, last_local_row;
   int first_local_col, last_local_col;
   int size, *col_ind;
   int local_num_vars;
   int variant, overlap, domain_type;
   double schwarz_rlx_weight;
   double *values;

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
   double  *relax_weight; 
   double   tol = 1.0e-6, pc_tol = 0.;

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
 
   build_matrix_type      = 1;
   build_matrix_arg_index = argc;
   build_rhs_type = 0;
   build_rhs_arg_index = argc;
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
   smooth_option     = hypre_CTAlloc(int, max_levels);
   for (i=0; i < max_levels; i++)
   {
        smooth_option[i] = -1;
   }

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromonefile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
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
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
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
      else if ( strcmp(argv[arg_index], "-exact_size") == 0 )
      {
         arg_index++;
         generate_matrix = 1;
      }
      else if ( strcmp(argv[arg_index], "-storage_low") == 0 )
      {
         arg_index++;
         generate_matrix = 2;
      }
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         /* ij_matrix_storage_type      = HYPRE_PARCSR; */
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
         build_rhs_type      = 1;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromonefile") == 0 )
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
         build_rhs_type      = 5;
         build_rhs_arg_index = arg_index;
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
      else if ( strcmp(argv[arg_index], "-xisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 4;
         build_rhs_arg_index = arg_index;
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

   /* defaults for BoomerAMG */
   if (solver_id == 0 || solver_id == 1 || solver_id == 3 || solver_id == 5
	|| solver_id == 9)
   {
   strong_threshold = 0.25;
   trunc_factor = 0.0;
   cycle_type = 1;

   num_grid_sweeps   = hypre_CTAlloc(int,4);
   grid_relax_type   = hypre_CTAlloc(int,4);
   grid_relax_points = hypre_CTAlloc(int *,4);
   relax_weight      = hypre_CTAlloc(double, max_levels);

   for (i=0; i < max_levels; i++)
	relax_weight[i] = 1.0;

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
   schwarz_rlx_weight = 1.0;

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
         schwarz_rlx_weight  = atof(argv[arg_index++]);
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
      printf("  -fromfile <filename>   : problem defining matrix from distributed file\n");
      printf("  -fromonefile <filename>: problem defining matrix from standard CSR file\n");
      printf("\n");
      printf("  -laplacian [<options>] : build laplacian problem\n");
      printf("  -9pt [<opts>] : build 9pt 2D laplacian problem\n");
      printf("  -27pt [<opts>] : build 27pt 3D laplacian problem\n");
      printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
      printf("    -n <nx> <ny> <nz>    : total problem size \n");
      printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      printf("\n");
      printf("   -exact_size           : inserts immediately into ParCSR structure\n");
      printf("   -storage_low          : allocates not enough storage for aux struct\n");
      printf("   -concrete_parcsr      : use parcsr matrix type as concrete type\n");
      printf("\n");
      printf("   -rhsfromfile          : from distributed file (NOT YET)\n");
      printf("   -rhsfromonefile       : from vector file \n");
      printf("   -rhsrand              : rhs is random vector\n");
      printf("   -xisone               : solution of all ones\n");
      printf("\n");
      printf("  -solver <ID>           : solver ID\n");
      printf("       0=AMG         1=AMG-PCG       \n");
      printf("       2=DS-PCG      3=AMG-GMRES     \n");
      printf("       4=DS-GMRES    5=AMG-CGNR      \n");     
      printf("       6=DS-CGNR     7=PILUT-GMRES   \n");     
      printf("       8=ParaSails-PCG   \n");     
      printf("       9=AMG-BiCGSTAB    \n");
      printf("       10=DS-BiCGSTAB    \n");
      printf("       11=PILUT-BiCGSTAB \n");
      printf("       12=Schwarz-PCG \n");
      printf("       18=ParaSails-GMRES\n");     
      printf("       43=Euclid-PCG\n");
      printf("       44=Euclid-GMRES\n");
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
      printf("  -mxl  <val>            : maximum number of AMG levels\n");
      printf("\n"); 
      printf("  -mu   <val>            : set AMG cycles (1=V, 2=W, etc.)\n"); 
      printf("  -th   <val>            : set AMG threshold Theta = val \n");
      printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
      printf("  -tol  <val>            : set AMG convergence tolerance to val\n");
      printf("  -mxrs <val>            : set AMG maximum row sum threshold for dependency weakening \n");
      printf("  -nf <val>              : set number of functions for systems AMG\n");
     
      printf("  -w   <val>             : set Jacobi relax weight = val\n");
      printf("  -k   <val>             : dimension Krylov space for GMRES\n");
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
    * Check a few things
    *-----------------------------------------------------------*/

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("Running with these driver parameters:\n");
      printf("  solver ID    = %d\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("IJ Interface");
   hypre_BeginTiming(time_index);

   if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 2 )
   {
      BuildParFromOneFile(argc, argv, build_matrix_arg_index, &parcsr_A);
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
      printf("You have asked for an unsupported problem, problem = %d.\n", 
		build_matrix_type);
      return(-1);
   }

    
   /*-----------------------------------------------------------
    * Copy the parcsr matrix into the IJMatrix through interface calls
    *-----------------------------------------------------------*/

   ierr = HYPRE_ParCSRMatrixGetComm( parcsr_A, &comm );
   ierr += HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );

   ierr += HYPRE_IJMatrixCreate( comm, &ij_matrix, M, N );

   ierr += HYPRE_IJMatrixSetLocalStorageType(
                 ij_matrix, HYPRE_PARCSR );
   ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
             &first_local_row, &last_local_row ,
             &first_local_col, &last_local_col );

   ierr = HYPRE_IJMatrixSetLocalSize( ij_matrix,
                last_local_row-first_local_row+1,
                last_local_col-first_local_col+1 );

/* the following shows how to build an ij_matrix if one has only an
   estimate for the row sizes */
   if (generate_matrix == 1)
   {   
/*  build ij_matrix using exact row_sizes for diag and offdiag */

      diag_sizes = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
      offdiag_sizes = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
      local_row = 0;
      for (i=first_local_row; i<= last_local_row; i++)
      {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size, 
		&col_ind, &values );

         for (j=0; j < size; j++)
         {
	 if (col_ind[j] < first_local_row || col_ind[j] > last_local_row)
	       offdiag_sizes[local_row]++;
	 else
	       diag_sizes[local_row]++;
         }
         local_row++;
         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size, 
		&col_ind, &values );
      }
      ierr += HYPRE_IJMatrixSetDiagRowSizes ( ij_matrix, 
					(const int *) diag_sizes );
      ierr += HYPRE_IJMatrixSetOffDiagRowSizes ( ij_matrix, 
					(const int *) offdiag_sizes );
      hypre_TFree(diag_sizes);
      hypre_TFree(offdiag_sizes);
      
      ierr = HYPRE_IJMatrixInitialize( ij_matrix );
      
      for (i=first_local_row; i<= last_local_row; i++)
      {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size, 
		&col_ind, &values );

         ierr += HYPRE_IJMatrixInsertRow( ij_matrix, size, i, col_ind, values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size, 
		&col_ind, &values );

      }
      ierr += HYPRE_IJMatrixAssemble( ij_matrix );
   }
   else
   {
      row_sizes = hypre_CTAlloc(int, last_local_row - first_local_row + 1);

      size = 5; /* this is in general too low, and supposed to test
		   the capability of the reallocation of the interface */ 
      if (generate_matrix == 0) /* tries a more accurate estimate of the
				   storage */
      {
	 if (build_matrix_type == 1) size = 7;
	 if (build_matrix_type == 3) size = 9;
	 if (build_matrix_type == 4) size = 27;
      }

      for (i=0; i < last_local_row - first_local_row + 1; i++)
         row_sizes[i] = size;

      ierr = HYPRE_IJMatrixSetRowSizes ( ij_matrix, (const int *) row_sizes );

      hypre_TFree(row_sizes);

      ierr = HYPRE_IJMatrixInitialize( ij_matrix );

      /* Loop through all locally stored rows and insert them into ij_matrix */
      for (i=first_local_row; i<= last_local_row; i++)
      {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size, 
		&col_ind, &values );

         ierr += HYPRE_IJMatrixInsertRow( ij_matrix, size, i, col_ind, values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size, 
		&col_ind, &values );
      }
      ierr += HYPRE_IJMatrixAssemble( ij_matrix );
   }
   if (ierr)
   {
       printf("Error in driver building IJMatrix from parcsr matrix. \n");
       return(-1);
   }

   /*-----------------------------------------------------------
    * Fetch the resulting underlying matrix out
    *-----------------------------------------------------------*/

    A = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage( ij_matrix);

#if 0
    /* compare the two matrices that should be the same */
    HYPRE_ParCSRMatrixPrint(parcsr_A, "driver.out.parcsr_A");
    HYPRE_ParCSRMatrixPrint(A, "driver.out.A");
#endif

    HYPRE_ParCSRMatrixDestroy(parcsr_A);
   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   ierr = HYPRE_IJMatrixGetRowPartitioning( ij_matrix, &partitioning);

   part_b = hypre_CTAlloc(int, num_procs+1);
   part_x = hypre_CTAlloc(int, num_procs+1);
   for (i=0; i < num_procs+1; i++)
   {
      part_b[i] = partitioning[i];
      part_x[i] = partitioning[i];
   }
   /* HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning); */
   HYPRE_ParCSRMatrixGetDims(A, &global_n, &global_n);

   hypre_EndTiming(time_index);

  /*-------------------------------------------------------------------
   * Separately test IJVector functions
   *-------------------------------------------------------------------*/

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, &ij_v, global_n);
   HYPRE_IJVectorSetLocalStorageType(ij_v, ij_vector_storage_type );
   HYPRE_IJVectorSetPartitioning(ij_v, (const int *) part_x);
   HYPRE_IJVectorInitialize(ij_v);
   HYPRE_IJVectorZeroLocalComponents(ij_v); 

   values = hypre_CTAlloc(double, part_x[myid+1] - part_x[myid]);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = (double)i;

   HYPRE_IJVectorAddToLocalComponentsInBlock(ij_v, 
                                             part_x[myid], 
                                             part_x[myid+1]-1,
                                             NULL,
                                             values);

   HYPRE_IJVectorGetLocalComponentsInBlock(ij_v, 
                                           part_x[myid], 
                                           part_x[myid+1]-1,
                                           NULL,
                                           values);

   ierr = 0;
   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     if (values[i] != (double)i) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorZeroLocalComponents or\n");
     printf("HYPRE_IJVector(AddTo,Get)LocalComponentsInBlock calls bad\n");
     printf("IJVector Error 1 with ierr = %d\n", ierr);
     exit(1);
   }

  /*-------------------------------------------------------------------
   * Check HYPRE_IJVectorSet(Get)LocalComponentsInBlock calls
   *
   * All local components changed -- NULL value_indices
   *-------------------------------------------------------------------*/

   HYPRE_IJVectorZeroLocalComponents(ij_v); 

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 1.;

   HYPRE_IJVectorSetLocalComponentsInBlock(ij_v, 
                                           part_x[myid], 
                                           part_x[myid+1]-1,
                                           NULL,
                                           values);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = (double)i;

   HYPRE_IJVectorAddToLocalComponentsInBlock(ij_v, 
                                             part_x[myid], 
                                             part_x[myid+1]-1,
                                             NULL,
                                             values);

   HYPRE_IJVectorGetLocalComponentsInBlock(ij_v, 
                                           part_x[myid], 
                                           part_x[myid+1]-1,
                                           NULL,
                                           values);

   ierr = 0;
   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     if (values[i] != (double)i + 1.) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(AddTo,Get)LocalComponentsInBlock\n");
     printf("calls with NULL value_indices bad\n");
     printf("IJVector Error 2 with ierr = %d\n", ierr);
     exit(1);
   }

  /*-------------------------------------------------------------------
   * Only first component changed -- NULL value_indices
   *-------------------------------------------------------------------*/

   HYPRE_IJVectorZeroLocalComponents(ij_v); 

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 1.;

   HYPRE_IJVectorSetLocalComponentsInBlock(ij_v, 
                                           part_x[myid], 
                                           part_x[myid],
                                           NULL,
                                           values);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 99.;

   HYPRE_IJVectorAddToLocalComponentsInBlock(ij_v, 
                                             part_x[myid], 
                                             part_x[myid],
                                             NULL,
                                             values);

   HYPRE_IJVectorGetLocalComponentsInBlock(ij_v, 
                                           part_x[myid], 
                                           part_x[myid+1]-1,
                                           NULL,
                                           values);

   ierr = 0;
   if (values[0] != 100.) ++ierr;
   for (i = 1; i < part_x[myid+1] - part_x[myid]; i++)
     if (values[i] != 0.) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(AddTo,Get)LocalComponentsInBlock\n");
     printf("calls with NULL value_indices bad\n");
     printf("IJVector Error 3 with ierr = %d\n", ierr);
     exit(1);
   }

  /*-------------------------------------------------------------------
   * Only last component changed -- NULL value_indices 
   *-------------------------------------------------------------------*/

   HYPRE_IJVectorZeroLocalComponents(ij_v); 

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 1.;

   HYPRE_IJVectorSetLocalComponentsInBlock(ij_v, 
                                           part_x[myid+1]-1, 
                                           part_x[myid+1]-1,
                                           NULL,
                                           values);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 99.;

   HYPRE_IJVectorAddToLocalComponentsInBlock(ij_v, 
                                             part_x[myid+1]-1, 
                                             part_x[myid+1]-1,
                                             NULL,
                                             values);

   HYPRE_IJVectorGetLocalComponentsInBlock(ij_v, 
                                           part_x[myid], 
                                           part_x[myid+1]-1,
                                           NULL,
                                           values);

   ierr = 0;
   for (i = 0; i < part_x[myid+1] - part_x[myid] - 1; i++)
     if (values[i] != 0.) ++ierr;
   if (values[part_x[myid+1] - part_x[myid] - 1] != 100.) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(AddTo,Get)LocalComponentsInBlock\n");
     printf("calls with NULL value_indices bad\n");
     printf("IJVector Error 4 with ierr = %d\n",ierr);
     exit(1);
   }

  /*-------------------------------------------------------------------
   * Last component set from first value of values of values array
   *-------------------------------------------------------------------*/

   HYPRE_IJVectorZeroLocalComponents(ij_v); 

   value_indices = hypre_CTAlloc(int,1);

   value_indices[0] = 0;

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 0.;

   values[0] = 1.;

   HYPRE_IJVectorSetLocalComponentsInBlock(ij_v, 
                                           part_x[myid+1]-1, 
                                           part_x[myid+1]-1,
                                           value_indices,
                                           values);


   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 99.;

   HYPRE_IJVectorAddToLocalComponentsInBlock(ij_v, 
                                             part_x[myid+1]-1, 
                                             part_x[myid+1]-1,
                                             value_indices,
                                             values);

   HYPRE_IJVectorGetLocalComponentsInBlock(ij_v, 
                                           part_x[myid], 
                                           part_x[myid+1]-1,
                                           NULL,
                                           values);

   ierr = 0;
   for (i = 0; i < part_x[myid+1] - part_x[myid] - 1; i++)
     if (values[i] != 0.) ++ierr;
   if (values[part_x[myid+1] - part_x[myid] - 1] != 100.) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(AddTo,Get)LocalComponentsInBlock\n");
     printf("calls bad\n");
     printf("IJVector Error 5 with ierr = %d\n",ierr);
     exit(1);
   }

   hypre_TFree(value_indices);

  /*-------------------------------------------------------------------
   * Check HYPRE_IJVectorSet(Get)LocalComponents calls
   *
   * All local components changed -- NULL glob_vec_indices,
   *                                 NULL value_indices
   *-------------------------------------------------------------------*/

   HYPRE_IJVectorZeroLocalComponents(ij_v); 

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 1.;

   HYPRE_IJVectorSetLocalComponents(ij_v,
                                    part_x[myid+1] - part_x[myid],
                                    NULL, NULL, values);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = (double)i;

   HYPRE_IJVectorAddToLocalComponents(ij_v,
                                      (part_x[myid+1] - part_x[myid])/2,
                                      NULL, NULL, values);

   HYPRE_IJVectorGetLocalComponents(ij_v,
                                    part_x[myid+1] - part_x[myid],
                                    NULL, NULL, values);

   ierr = 0;
   for (i = 0; i < (part_x[myid+1]-part_x[myid])/2; i++)
     if (values[i] != (double)i + 1.) ++ierr;
   for (i = (part_x[myid+1]-part_x[myid])/2; i < part_x[myid+1]-part_x[myid]; i++)
     if (values[i] != 1.) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(AddTo,Get)LocalComponents\n");
     printf("calls with NULL glob_vec_indices, NULL value_indices bad\n");
     printf("IJVector Error 6 with ierr = %d\n", ierr);
     exit(1);
   }

  /*-------------------------------------------------------------------
   * All local components changed, assigned reverse-ordered values
   *   as specified by glob_vec_indices -- NULL value_indices
   *-------------------------------------------------------------------*/

   HYPRE_IJVectorZeroLocalComponents(ij_v); 

   glob_vec_indices = hypre_CTAlloc(int, part_x[myid+1] - part_x[myid]);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
   {
     values[i] = (double)i;
     glob_vec_indices[i] = part_x[myid+1] - 1 - i;
   }

   HYPRE_IJVectorSetLocalComponents(ij_v,
                                    part_x[myid+1] - part_x[myid],
                                    glob_vec_indices, NULL, values);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = (double)i*i;

   HYPRE_IJVectorAddToLocalComponents(ij_v,
                                    part_x[myid+1] - part_x[myid],
                                    glob_vec_indices, NULL, values);

   HYPRE_IJVectorGetLocalComponents(ij_v,
                                    part_x[myid+1] - part_x[myid],
                                    glob_vec_indices, NULL, values);

   hypre_TFree(glob_vec_indices);

   ierr = 0;
   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     if (values[i] != (double)(i*i + i)) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(Get)LocalComponents\n");
     printf("calls with NULL value_indices bad\n");
     printf("IJVector Error 7 with ierr = %d\n", ierr);
     exit(1);
   }

  /*-------------------------------------------------------------------
   * All local components changed, assigned reverse-ordered values
   *   as specified by glob_vec_indices -- NULL glob_vec_indices
   *-------------------------------------------------------------------*/

   HYPRE_IJVectorZeroLocalComponents(ij_v); 

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 0.;

   value_indices = hypre_CTAlloc(int, 2);

   value_indices[0] = part_x[myid+1] - part_x[myid] - 1;
   value_indices[1] = 0;

   values[0] = 1.;
   values[part_x[myid+1] - part_x[myid] - 1] = -1.;

   HYPRE_IJVectorSetLocalComponents(ij_v, 2,
                                    NULL, value_indices, values);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 99.;

   /* Inspect all of the vector data */
   HYPRE_IJVectorGetLocalComponents(ij_v,
                                    part_x[myid+1] - part_x[myid],
                                    NULL, NULL, values);

   ierr = 0;
   if (values[0] != -1.) ++ierr;
   if (values[1] != 1.) ++ierr;
   for (i = 2; i < part_x[myid+1] - part_x[myid]; i++)
     if (values[i] != 0.) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(Get)LocalComponents\n");
     printf("calls with NULL glob_vec_indices bad\n");
     printf("IJVector Error 8 with ierr = %d\n", ierr);
     exit(1);
   }

   values[0] = 99.;
   values[part_x[myid+1] - part_x[myid] - 1] = -99.;
   HYPRE_IJVectorAddToLocalComponents(ij_v, 2,
                                      NULL, value_indices, values);

   /* This should switch the extreme values wrt the initial values array */
   value_indices[0] = 0;
   value_indices[1] = part_x[myid+1] - part_x[myid] - 1;

   HYPRE_IJVectorGetLocalComponents(ij_v, 2,
                                    NULL, value_indices, values);

   ierr = 0;
   if (values[0] != -100.) ++ierr;
   if (values[1] != 1.) ++ierr;
   for (i = 2; i < part_x[myid+1] - part_x[myid] - 1; i++)
     if (values[i] != 0.) ++ierr;
   if (values[part_x[myid+1]-part_x[myid]-1] != 100.) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(AddTo,Get)LocalComponents\n");
     printf("calls with NULL glob_vec_indices bad\n");
     printf("IJVector Error 9 with ierr = %d\n", ierr);
     exit(1);
   }

   hypre_TFree(value_indices);

  /*-------------------------------------------------------------------
   * Extreme local components assigned from opposite ends of
   *   values array
   *-------------------------------------------------------------------*/

   HYPRE_IJVectorZeroLocalComponents(ij_v); 

   glob_vec_indices = hypre_CTAlloc(int, 2);
   value_indices    = hypre_CTAlloc(int, 2);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 0.;

   value_indices[0] = part_x[myid+1] - part_x[myid] - 1;
   value_indices[1] = 0;

   values[0] = 1.;
   values[part_x[myid+1] - part_x[myid] - 1] = -1.;

   glob_vec_indices[0] = part_x[myid]; 
   glob_vec_indices[1] = part_x[myid+1] - 1; 

   HYPRE_IJVectorSetLocalComponents(ij_v, 2,
                                    glob_vec_indices,
                                    value_indices,
                                    values);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 99.;

   /* Inspect all of the vector data */
   HYPRE_IJVectorGetLocalComponents(ij_v,
                                    part_x[myid+1] - part_x[myid],
                                    NULL, NULL,
                                    values);

   ierr = 0;
   if (values[0] != -1.) ++ierr;
   for (i = 1; i < part_x[myid+1] - part_x[myid] - 1; i++)
     if (values[i] != 0.) ++ierr;
   if (values[part_x[myid+1]-part_x[myid]-1] != 1.) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(Get)LocalComponents\n");
     printf("calls bad\n");
     printf("IJVector Error 10 with ierr = %d\n", ierr);
     printf("myid = %d IJVector components\n",myid);
     exit(1);
   }

   values[0] = 99.;
   values[part_x[myid+1] - part_x[myid] - 1] = -99.;
   HYPRE_IJVectorAddToLocalComponents(ij_v, 2,
                                      glob_vec_indices,
                                      value_indices,
                                      values);

   /* This should switch the extreme values wrt the initial values array */
   value_indices[0] = 0;
   value_indices[1] = part_x[myid+1] - part_x[myid] - 1;

   HYPRE_IJVectorGetLocalComponents(ij_v, 2,
                                    glob_vec_indices,
                                    value_indices,
                                    values);

   ierr = 0;
   if (values[0] != -100.) ++ierr;
   for (i = 1; i < part_x[myid+1] - part_x[myid] - 1; i++)
     if (values[i] != 0.) ++ierr;
   if (values[part_x[myid+1]-part_x[myid]-1] != 100.) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(AddTo,Get)LocalComponents\n");
     printf("calls bad\n");
     printf("IJVector Error 11 with ierr = %d\n", ierr);
     exit(1);
   }

   hypre_TFree(value_indices);
   hypre_TFree(glob_vec_indices);

   hypre_TFree(values);

  /*-------------------------------------------------------------------
   * End testing of IJVector functions
   *-------------------------------------------------------------------*/

   hypre_BeginTiming(time_index);

   if (build_rhs_type == 1)
   {
      /* BuildRHSParFromFile(argc, argv, build_rhs_arg_index, &b); */
      printf("Rhs from file not yet implemented.  Defaults to b=0\n");
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, &ij_b, global_n);
      HYPRE_IJVectorSetLocalStorageType(ij_b,ij_vector_storage_type );
      HYPRE_IJVectorSetPartitioning(ij_b, (const int *) part_b);
      HYPRE_IJVectorInitialize(ij_b);
      HYPRE_IJVectorZeroLocalComponents(ij_b); 

      HYPRE_IJVectorCreate(MPI_COMM_WORLD, &ij_x, global_n);
      HYPRE_IJVectorSetLocalStorageType(ij_x,ij_vector_storage_type );
      HYPRE_IJVectorSetPartitioning(ij_x, (const int *) part_x);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorZeroLocalComponents(ij_x); 

      values = hypre_CTAlloc(double, part_x[myid+1] - part_x[myid]);

      for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
         values[i] = 1.0;

      HYPRE_IJVectorSetLocalComponentsInBlock(ij_x, 
                                              part_x[myid], 
                                              part_x[myid+1]-1,
                                              NULL,
                                              values);
      hypre_TFree(values);

   /*-----------------------------------------------------------
    * Fetch the resulting underlying vectors out
    *-----------------------------------------------------------*/

      b = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage( ij_b );
      x = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage( ij_x );

   }
   else if ( build_rhs_type == 2 )
   {
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, part_b, &b);

      HYPRE_IJVectorCreate(MPI_COMM_WORLD, &ij_x, global_n);
      HYPRE_IJVectorSetLocalStorageType(ij_x,ij_vector_storage_type );
      HYPRE_IJVectorSetPartitioning(ij_x, (const int *) part_x);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorZeroLocalComponents(ij_x); 
      x = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage( ij_x );

   }
   else if ( build_rhs_type == 3 )
   {

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_b,&b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParVectorSetRandomValues(b, 22775);
      HYPRE_ParVectorInnerProd(b,b,&norm);
      norm = 1.0/sqrt(norm);
      ierr = HYPRE_ParVectorScale(norm, b);      

      HYPRE_IJVectorCreate(MPI_COMM_WORLD, &ij_x, global_n);
      HYPRE_IJVectorSetLocalStorageType(ij_x,ij_vector_storage_type );
      HYPRE_IJVectorSetPartitioning(ij_x, (const int *) part_x);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorZeroLocalComponents(ij_x); 
      x = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage( ij_x );
   }
   else if ( build_rhs_type == 4 )
   {

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_x, &x);
      HYPRE_ParVectorInitialize(x);
      HYPRE_ParVectorSetConstantValues(x, 1.0);

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_b, &b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParCSRMatrixMatvec(1.0,A,x,0.0,b);

      HYPRE_ParVectorSetConstantValues(x, 0.0);
   }
   else if ( build_rhs_type == 5 )
   {

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_x, &x);
      HYPRE_ParVectorInitialize(x);
      HYPRE_ParVectorSetConstantValues(x, 1.0);

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_b, &b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParVectorSetConstantValues(b, 0.0);
   }
   else /* if ( build_rhs_type == 0 ) */
   {
      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_b, &b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParVectorSetConstantValues(b, 1.0);

      HYPRE_IJVectorCreate(MPI_COMM_WORLD, &ij_x, global_n);
      HYPRE_IJVectorSetLocalStorageType(ij_x,ij_vector_storage_type );
      HYPRE_IJVectorSetPartitioning(ij_x, (const int *) part_x);
      HYPRE_IJVectorInitialize(ij_x);
      HYPRE_IJVectorZeroLocalComponents(ij_x); 
      x = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage( ij_x );
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Interface", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   if (num_functions > 1)
   {
      dof_func = NULL;
      if (build_funcs_type == 1)
      {
	 BuildFuncsFromOneFile(argc, argv, build_funcs_arg_index, A, &dof_func);
      }
      else if (build_funcs_type == 2)
      {
	 BuildFuncsFromFiles(argc, argv, build_funcs_arg_index, A, &dof_func);
      }
      else
      {
         local_num_vars = last_local_row -first_local_row+1;
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
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      if (myid == 0) printf("Solver:  AMG\n");
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGCreate(&amg_solver); 
      HYPRE_BoomerAMGSetCoarsenType(amg_solver, (hybrid*coarsen_type));
      HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      HYPRE_BoomerAMGSetTol(amg_solver, tol);
      HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
/* note: log is written to standard output, not to file */
      HYPRE_BoomerAMGSetLogging(amg_solver, ioutdat, "driver.out.log"); 
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

      HYPRE_BoomerAMGSetup(amg_solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGSolve(amg_solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_BoomerAMGSetup(amg_solver, A, b, x);
      HYPRE_BoomerAMGSolve(amg_solver, A, b, x);
#endif

      HYPRE_BoomerAMGDestroy(amg_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG 
    *-----------------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2 || solver_id == 8 || 
	solver_id == 12 || solver_id == 43)
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter(pcg_solver, 500);
      HYPRE_PCGSetTol(pcg_solver, tol);
      HYPRE_PCGSetTwoNorm(pcg_solver, 1);
      HYPRE_PCGSetRelChange(pcg_solver, 0);
      HYPRE_PCGSetLogging(pcg_solver, 1);
 
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
         HYPRE_BoomerAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
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
         if (myid == 0) printf("Solver: DS-PCG\n");
         pcg_precond = NULL;

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
	 hypre_TFree(smooth_option);
      }
      else if (solver_id == 8)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-PCG\n");

	 HYPRE_ParCSRParaSailsCreate(MPI_COMM_WORLD, &pcg_precond);
	 HYPRE_ParCSRParaSailsSetParams(pcg_precond, 0.1, 1);

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
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

         HYPRE_ParCSREuclidCreate(MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_ParCSREuclidSetParams(pcg_precond, argc, argv);

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSREuclidSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSREuclidSetup,
                                   pcg_precond);
	 hypre_TFree(smooth_option);
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
      HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
#endif

      HYPRE_ParCSRPCGDestroy(pcg_solver);
 
      if (solver_id == 1)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      else if (solver_id == 8)
      {
	 HYPRE_ParCSRParaSailsDestroy(pcg_precond);
      }
      else if (solver_id == 12)
      {
	 HYPRE_SchwarzDestroy(pcg_precond);
      }
      else if (solver_id == 43)
      {
        HYPRE_ParCSREuclidPrintParams(pcg_precond);
        HYPRE_ParCSREuclidDestroy(pcg_precond);
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
         HYPRE_BoomerAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
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
         if (myid == 0) printf("Solver: Pilut-GMRES\n");

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

	 HYPRE_ParCSRParaSailsCreate(MPI_COMM_WORLD, &pcg_precond);
	 HYPRE_ParCSRParaSailsSetParams(pcg_precond, 0.1, 1);
	 HYPRE_ParCSRParaSailsSetSym(pcg_precond, 0);

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
                               pcg_precond);
	 hypre_TFree(smooth_option);
      }
      else if (solver_id == 44)
      {
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-GMRES\n");

         HYPRE_ParCSREuclidCreate(MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_ParCSREuclidSetParams(pcg_precond, argc, argv);

         HYPRE_GMRESSetPrecond (pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_ParCSREuclidSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_ParCSREuclidSetup,
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
         (pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_GMRESSolve
         (pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_GMRESGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_GMRESSetup(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_GMRESSolve(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
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
	 HYPRE_ParCSRParaSailsDestroy(pcg_precond);
      }
      else if (solver_id == 44)
      {
        HYPRE_ParCSREuclidPrintParams(pcg_precond);
        HYPRE_ParCSREuclidDestroy(pcg_precond);
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
         HYPRE_BoomerAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
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
         if (myid == 0) printf("Solver: Pilut-BiCGSTAB\n");

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

         HYPRE_ParCSREuclidCreate(MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_ParCSREuclidSetParams(pcg_precond, argc, argv);

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSREuclidSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSREuclidSetup,
                                  pcg_precond);
	 hypre_TFree(smooth_option);
      }
 
      HYPRE_BiCGSTABSetup(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_BiCGSTABSolve(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_ParCSRBiCGSTABGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_BiCGSTABSetup(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_BiCGSTABSolve(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
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
        HYPRE_ParCSREuclidPrintParams(pcg_precond);
        HYPRE_ParCSREuclidDestroy(pcg_precond);
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
         HYPRE_BoomerAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
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
      HYPRE_CGNRSetup(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("CGNR Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_CGNRSolve(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_CGNRGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_CGNRGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_CGNRSetup(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_CGNRSolve(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
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

#if 0
   HYPRE_PrintCSRVector(x, "driver.out.x");
#endif


   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_IJVectorDestroy(ij_v);
   hypre_TFree(part_x);
   hypre_TFree(part_b);

   HYPRE_IJMatrixDestroy(ij_matrix);
   if (build_rhs_type == 1)
      HYPRE_IJVectorDestroy(ij_b);
   else
      HYPRE_ParVectorDestroy(b);
   if (build_rhs_type > -1 && build_rhs_type < 4)
      HYPRE_IJVectorDestroy(ij_x);
   else
      HYPRE_ParVectorDestroy(x);
/*
   hypre_FinalizeMemoryDebug();
*/
   /* Finalize MPI */
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

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

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
      printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
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

   values[0] = 0.0;
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

   hinx = 1.0/(nx+1);
   hiny = 1.0/(ny+1);
   hinz = 1.0/(nz+1);

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

   ax = 1.0;
   ay = 1.0;
   az = 1.0;

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
      printf("    (ax, ay, az) = (%f, %f, %f)\n", ax, ay, az);
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

   values[0] = 0.0;
   if (nx > 1)
   {
      values[0] += 2.0*cx/(hinx*hinx) - 1.0*ax/hinx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy/(hiny*hiny) - 1.0*ay/hiny;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz/(hinz*hinz) - 1.0*az/hinz;
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
                        HYPRE_ParCSRMatrix   A,
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
                        HYPRE_ParCSRMatrix   A,
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
   HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
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
                        int		    *partitioning,
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
      printf("    (Px, Py) = (%d, %d)\n", P,  Q);
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

   values[1] = -1.0;

   values[0] = 0.0;
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
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
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
   values[1] = -1.0;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(MPI_COMM_WORLD,
                               nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}
