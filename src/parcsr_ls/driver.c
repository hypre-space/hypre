
#include "headers.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interfoace (parcsr storage).
 * Do `driver -help' for usage info.
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   int                 arg_index;
   int                 print_usage;
   int                 build_matrix_type;
   int                 build_matrix_arg_index;
   int                 build_rhs_type;
   int                 build_rhs_arg_index;
   int                 solver_id;
   int                 ioutdat;
   int                 debug_flag;
   int                 ierr,i; 
   int                 max_levels = 25;
   int                 num_iterations; 
   double              norm;
   double              final_res_norm;


   HYPRE_ParCSRMatrix  A;
   HYPRE_ParVector     b;
   HYPRE_ParVector     x;

   HYPRE_Solver        amg_solver;
   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond;

   int                 num_procs, myid;
   int                 global_m, global_n;
   int                *partitioning;

   int		       time_index;

   /* parameters for BoomerAMG */
   double   strong_threshold;
   double   trunc_factor;
   int      cycle_type;
   int      coarsen_type = 0;
   int      hybrid = 1;
   int      measure_type = 0;
   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points;
   int      relax_default;
   double  *relax_weight; 
   double   tol = 1.0e-6;

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
   relax_default = 3;
   debug_flag = 0;

   solver_id = 0;

   ioutdat = 3;

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
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromonefile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
         build_matrix_arg_index = arg_index;
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
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         debug_flag = atoi(argv[arg_index++]);
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
   strong_threshold = 0.25;
   trunc_factor = 0.0;
   cycle_type = 1;

   num_grid_sweeps = hypre_CTAlloc(int,4);
   grid_relax_type = hypre_CTAlloc(int,4);
   grid_relax_points = hypre_CTAlloc(int *,4);
   relax_weight = hypre_CTAlloc(double,max_levels);

   for (i=0; i < max_levels; i++)
	relax_weight[i] = 0.0;
   if (coarsen_type == 5)
   {
      /* fine grid */
      num_grid_sweeps[0] = 3;
      grid_relax_type[0] = relax_default; 
      grid_relax_points[0] = hypre_CTAlloc(int, 4); 
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
      num_grid_sweeps[0] = 2;
      grid_relax_type[0] = relax_default; 
      grid_relax_points[0] = hypre_CTAlloc(int, 2); 
      grid_relax_points[0][0] = 1;
      grid_relax_points[0][1] = -1;
  
      /* down cycle */
      num_grid_sweeps[1] = 2;
      grid_relax_type[1] = relax_default; 
      grid_relax_points[1] = hypre_CTAlloc(int, 2); 
      grid_relax_points[1][0] = 1;
      grid_relax_points[1][1] = -1;
   
      /* up cycle */
      num_grid_sweeps[2] = 2;
      grid_relax_type[2] = relax_default; 
      grid_relax_points[2] = hypre_CTAlloc(int, 2); 
      grid_relax_points[2][0] = -1;
      grid_relax_points[2][1] = 1;
   }
   /* coarsest grid */
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 9;
   grid_relax_points[3] = hypre_CTAlloc(int, 1);
   grid_relax_points[3][0] = 0;

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
         relax_weight[0] = atof(argv[arg_index++]);
         for (i=1; i < max_levels; i++)
	   relax_weight[i] = relax_weight[0];
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
      printf("  -fromfile <filename>   : matrix from distributed file\n");
      printf("  -fromonefile <filename>: matrix from standard CSR file\n");
      printf("\n");
      printf("  -laplacian [<options>] : build laplacian matrix\n");
      printf("  -9pt [<opts>] : build 9pt 2D laplacian matrix\n");
      printf("  -27pt [<opts>] : build 27pt 3D laplacian matrix\n");
      printf("  -difconv [<opts>]      : build convection-diffusion matrix\n");
      printf("    -n <nx> <ny> <nz>    : problem size per processor\n");
      printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      printf("\n");
      printf("   -rhsfromfile          : from distributed file (NOT YET)\n");
      printf("   -rhsfromonefile       : from vector file \n");
      printf("   -rhsrand              : rhs is random vector, ||x||=1\n");
      printf("   -xisone               : rhs of all ones\n");
      printf("\n");
      printf("  -solver <ID>           : solver ID\n");
      printf("       1=AMG-PCG    2=DS-PCG   \n");
      printf("       3=AMG-GMRES  4=DS-GMRES  \n");     
      printf("       5=AMG-CGNR   6=DS-CGNR  \n");     
      printf("       7=PILUT-GMRES  \n");     
      printf("\n");
      printf("   -ruge                 : Ruge coarsening (local)\n");
      printf("   -ruge3                : third pass on boundary\n");
      printf("   -ruge3c               : third pass on boundary, keep c-points\n");
      printf("   -ruge2b               : 2nd pass is global\n");
      printf("   -rugerlx              : relaxes special points\n");
      printf("   -falgout              : local ruge followed by LJP\n");
      printf("   -nohybrid             : no switch in coarsening\n");
      printf("   -gm                   : use global measures\n");
      printf("\n");
      printf("  -rlx <val>             : relaxation type\n");
      printf("       0=Weighted Jacobi  \n");
      printf("       3=Hybrid Jacobi/Gauss-Seidel  \n");
      printf("\n");  
      printf("  -th <val>              : set AMG threshold Theta = val \n");
      printf("  -tr <val>              : set AMG interpolation truncation factor = val \n");
      printf("  -tol <val>             : set AMG convergence tolerance to val\n");
      printf("  -w  <val>              : set Jacobi relax weight = val\n");
      printf("  -k  <val>              : dimension Krylov space for GMRES\n");
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

   if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 2 )
   {
      BuildParFromOneFile(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 3 )
   {
      BuildParLaplacian9pt(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 4 )
   {
      BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParDifConv(argc, argv, build_matrix_arg_index, &A);
   }
   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

#if 0
   HYPRE_ParCSRMatrixPrint(A, "driver.out.A");
#endif

   if (build_rhs_type == 1)
   {
      /* BuildRHSParFromFile(argc, argv, build_rhs_arg_index, &b); */
      printf("Rhs from file not yet implemented.  Defaults to b=0\n");
      HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_m, partitioning,&b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParVectorSetConstantValues(b, 0.0);

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, partitioning,&x);
      HYPRE_ParVectorInitialize(x);
      HYPRE_ParVectorSetConstantValues(x, 1.0);
   }
   else if ( build_rhs_type == 2 )
   {
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, A, &b);

      HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, partitioning,&x);
      HYPRE_ParVectorInitialize(x);
      HYPRE_ParVectorSetConstantValues(x, 0.0);      
   }
   else if ( build_rhs_type == 3 )
   {

      HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_m, partitioning,&b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParVectorSetRandomValues(b, 22775);
      HYPRE_ParVectorInnerProd(b,b,&norm);
      norm = 1.0/sqrt(norm);
      ierr = HYPRE_ParVectorScale(norm, b);      

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, partitioning,&x);
      HYPRE_ParVectorInitialize(x);
      HYPRE_ParVectorSetConstantValues(x, 0.0);      
   }
   else if ( build_rhs_type == 4 )
   {

      HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, partitioning,&x);
      HYPRE_ParVectorInitialize(x);
      HYPRE_ParVectorSetConstantValues(x, 1.0);      

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_m, partitioning,&b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParCSRMatrixMatvec(1.0,A,x,0.0,b);

      HYPRE_ParVectorSetConstantValues(x, 0.0);      
   }
   else /* if ( build_rhs_type == 0 ) */
   {
      HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_m, partitioning,&b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParVectorSetConstantValues(b, 0.0);

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, partitioning,&x);
      HYPRE_ParVectorInitialize(x);
      HYPRE_ParVectorSetConstantValues(x, 1.0);
   }
   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGCreate(&amg_solver); 
      HYPRE_BoomerAMGSetCoarsenType(amg_solver, (hybrid*coarsen_type));
      HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      HYPRE_BoomerAMGSetTol(amg_solver, tol);
      HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
      HYPRE_BoomerAMGSetLogging(amg_solver, ioutdat, "driver.out.log");
      HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      HYPRE_BoomerAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      HYPRE_BoomerAMGSetGridRelaxType(amg_solver, grid_relax_type);
      HYPRE_BoomerAMGSetRelaxWeight(amg_solver, relax_weight);
      HYPRE_BoomerAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);

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

      HYPRE_BoomerAMGDestroy(amg_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG 
    *-----------------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2)
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRPCGSetMaxIter(pcg_solver, 500);
      HYPRE_ParCSRPCGSetTol(pcg_solver, tol);
      HYPRE_ParCSRPCGSetTwoNorm(pcg_solver, 1);
      HYPRE_ParCSRPCGSetRelChange(pcg_solver, 0);
      HYPRE_ParCSRPCGSetLogging(pcg_solver, 1);
 
      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_BoomerAMGSolve,
                                   HYPRE_BoomerAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 2)
      {
         /* use diagonal scaling as preconditioner */

         pcg_precond = NULL;

         HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
      }
 
      HYPRE_ParCSRPCGSetup(pcg_solver, A, b, x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRPCGSolve(pcg_solver, A, b, x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
      HYPRE_ParCSRPCGDestroy(pcg_solver);
 
      if (solver_id == 1)
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
    * Solve the system using GMRES 
    *-----------------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4 || solver_id == 7)
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRGMRESSetKDim(pcg_solver, k_dim);
      HYPRE_ParCSRGMRESSetMaxIter(pcg_solver, 100);
      HYPRE_ParCSRGMRESSetTol(pcg_solver, tol);
      HYPRE_ParCSRGMRESSetLogging(pcg_solver, 1);
 
      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */

         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                   HYPRE_BoomerAMGSolve,
                                   HYPRE_BoomerAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */

         pcg_precond = NULL;

         HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
      }
      else if (solver_id == 7)
      {
         /* use PILUT as preconditioner */
         ierr = HYPRE_ParCSRPilutCreate( MPI_COMM_WORLD, &pcg_precond ); 
         if (ierr) {
	   printf("Error in ParPilutCreate\n");
         }

         HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                   HYPRE_ParCSRPilutSolve,
                                   HYPRE_ParCSRPilutSetup,
                                   pcg_precond);

         if (drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
               drop_tol );

         if (nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
               nonzeros_to_keep );
      }
 
      HYPRE_ParCSRGMRESSetup(pcg_solver, A, b, x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRGMRESSolve(pcg_solver, A, b, x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_ParCSRGMRESGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
      HYPRE_ParCSRGMRESDestroy(pcg_solver);
 
      if (solver_id == 3)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (solver_id == 7)
      {
         HYPRE_ParCSRPilutDestroy(pcg_precond);
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
    * Solve the system using CGNR 
    *-----------------------------------------------------------*/

   if (solver_id == 5 || solver_id == 6)
   {
      time_index = hypre_InitializeTiming("CGNR Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRCGNRCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRCGNRSetMaxIter(pcg_solver, 1000);
      HYPRE_ParCSRCGNRSetTol(pcg_solver, tol);
      HYPRE_ParCSRCGNRSetLogging(pcg_solver, 1);
 
      if (solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_ParCSRCGNRSetPrecond(pcg_solver,
                                   HYPRE_BoomerAMGSolve,
                                   HYPRE_BoomerAMGSolveT,
                                   HYPRE_BoomerAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 6)
      {
         /* use diagonal scaling as preconditioner */

         pcg_precond = NULL;

         HYPRE_ParCSRCGNRSetPrecond(pcg_solver,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
      }
 
      HYPRE_ParCSRCGNRSetup(pcg_solver, A, b, x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("CGNR Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRCGNRSolve(pcg_solver, A, b, x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_ParCSRCGNRGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
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

   HYPRE_ParCSRMatrixDestroy(A);
   HYPRE_ParVectorDestroy(b);
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

   HYPRE_ParCSRMatrix  A;

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
 
   A = HYPRE_ParCSRMatrixRead(MPI_COMM_WORLD, filename);

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

   A = (HYPRE_ParCSRMatrix) 
	GenerateLaplacian(MPI_COMM_WORLD, nx, ny, nz, P, Q, R, p, q, r, values);

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
   HYPRE_CSRMatrix  A_CSR;

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
   A = HYPRE_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_CSR, NULL, NULL);

   *A_ptr = A;

   HYPRE_CSRMatrixDestroy(A_CSR);

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
                        HYPRE_ParCSRMatrix   A,
                        HYPRE_ParVector     *b_ptr     )
{
   char               *filename;

   HYPRE_ParVector  b;
   HYPRE_Vector     b_CSR;

   int                 myid;
   int 		      *partitioning;

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
   HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
   b = HYPRE_VectorToParVector(MPI_COMM_WORLD, b_CSR, partitioning);

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
