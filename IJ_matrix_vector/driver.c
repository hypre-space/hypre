
#include "headers.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
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

   HYPRE_IJMatrix     *ij_matrix;
   /* concrete underlying type for ij_matrix defaults to parcsr. AJC. */
   int                 ij_matrix_storage_type=HYPRE_PARCSR_MATRIX;

   hypre_ParCSRMatrix *A;
   hypre_ParVector    *b;
   hypre_ParVector    *x;

   HYPRE_Solver        amg_solver;
   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond;

   int                 num_procs, myid;

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
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         ij_matrix_storage_type      = HYPRE_PARCSR_MATRIX;
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
      printf("  -fromfile <filename>   : problem defining matrix from distributed file\n");
      printf("  -fromonefile <filename>: problem defining matrix from standard CSR file\n");
      printf("\n");
      printf("  -laplacian [<options>] : build laplacian problem\n");
      printf("  -9pt [<opts>] : build 9pt 2D laplacian problem\n");
      printf("  -27pt [<opts>] : build 27pt 3D laplacian problem\n");
      printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
      printf("    -n <nx> <ny> <nz>    : problem size per processor\n");
      printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      printf("\n");
      printf("   -concrete_parcsr      : use parcsr matrix type as concrete type\n");
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

   if ( build_matrix_type == 3 )
   {
      IJMatrixBuildParLaplacian9pt(argc, argv, build_matrix_arg_index, &A, &ij_matrix, ij_matrix_storage_type);
   }
   else
   {
      printf("You have asked for an unsupported problem, problem = \n.", build_matrix_type);
      return(-1);
   }

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

#if 0
   hypre_PrintParCSRMatrix(A, "driver.out.A");
#endif

   if (build_rhs_type == 1)
   {
      /* BuildRHSParFromFile(argc, argv, build_rhs_arg_index, &b); */
      printf("Rhs from file not yet implemented.  Defaults to b=0\n");
      b = hypre_CreateParVector(MPI_COMM_WORLD,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_SetParVectorPartitioningOwner(b, 0);
      hypre_InitializeParVector(b);
      hypre_SetParVectorConstantValues(b, 0.0);

      x = hypre_CreateParVector(MPI_COMM_WORLD,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_SetParVectorPartitioningOwner(x, 0);
      hypre_InitializeParVector(x);
      hypre_SetParVectorConstantValues(x, 1.0);
   }
   else if ( build_rhs_type == 2 )
   {
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, A, &b);

      x = hypre_CreateParVector(MPI_COMM_WORLD,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_SetParVectorPartitioningOwner(x, 0);
      hypre_InitializeParVector(x);
      hypre_SetParVectorConstantValues(x, 0.0);      
   }
   else if ( build_rhs_type == 3 )
   {

      b = hypre_CreateParVector(MPI_COMM_WORLD,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_SetParVectorPartitioningOwner(b, 0);
      hypre_InitializeParVector(b);
      hypre_SetParVectorRandomValues(b, 22775);
      norm = 1.0/sqrt(hypre_ParInnerProd(b,b));
      ierr = hypre_ScaleParVector(norm, b);      

      x = hypre_CreateParVector(MPI_COMM_WORLD,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_SetParVectorPartitioningOwner(x, 0);
      hypre_InitializeParVector(x);
      hypre_SetParVectorConstantValues(x, 0.0);      
   }
   else if ( build_rhs_type == 4 )
   {

      x = hypre_CreateParVector(MPI_COMM_WORLD,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_SetParVectorPartitioningOwner(x, 0);
      hypre_InitializeParVector(x);
      hypre_SetParVectorConstantValues(x, 1.0);      

      b = hypre_CreateParVector(MPI_COMM_WORLD,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_SetParVectorPartitioningOwner(b, 0);
      hypre_InitializeParVector(b);
      hypre_ParMatvec(1.0,A,x,0.0,b);

      hypre_SetParVectorConstantValues(x, 0.0);      
   }
   else /* if ( build_rhs_type == 0 ) */
   {
      b = hypre_CreateParVector(MPI_COMM_WORLD,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_SetParVectorPartitioningOwner(b, 0);
      hypre_InitializeParVector(b);
      hypre_SetParVectorConstantValues(b, 0.0);

      x = hypre_CreateParVector(MPI_COMM_WORLD,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_SetParVectorPartitioningOwner(x, 0);
      hypre_InitializeParVector(x);
      hypre_SetParVectorConstantValues(x, 1.0);
   }
   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      amg_solver = HYPRE_ParAMGInitialize(); 
      HYPRE_ParAMGSetCoarsenType(amg_solver, (hybrid*coarsen_type));
      HYPRE_ParAMGSetMeasureType(amg_solver, measure_type);
      HYPRE_ParAMGSetTol(amg_solver, tol);
      HYPRE_ParAMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_ParAMGSetTruncFactor(amg_solver, trunc_factor);
      HYPRE_ParAMGSetLogging(amg_solver, ioutdat, "driver.out.log");
      HYPRE_ParAMGSetCycleType(amg_solver, cycle_type);
      HYPRE_ParAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      HYPRE_ParAMGSetGridRelaxType(amg_solver, grid_relax_type);
      HYPRE_ParAMGSetRelaxWeight(amg_solver, relax_weight);
      HYPRE_ParAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      HYPRE_ParAMGSetMaxLevels(amg_solver, max_levels);
      HYPRE_ParAMGSetDebugFlag(amg_solver, debug_flag);

      HYPRE_ParAMGSetup(amg_solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_ParAMGSolve(amg_solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_ParAMGFinalize(amg_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG 
    *-----------------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2)
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRPCGInitialize(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRPCGSetMaxIter(pcg_solver, 500);
      HYPRE_ParCSRPCGSetTol(pcg_solver, tol);
      HYPRE_ParCSRPCGSetTwoNorm(pcg_solver, 1);
      HYPRE_ParCSRPCGSetRelChange(pcg_solver, 0);
      HYPRE_ParCSRPCGSetLogging(pcg_solver, 1);
 
      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
         pcg_precond = HYPRE_ParAMGInitialize(); 
         HYPRE_ParAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_ParAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_ParAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_ParAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_ParAMGSetMaxIter(pcg_precond, 1);
         HYPRE_ParAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_ParAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_ParAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_ParAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_ParAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_ParAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSetup,
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
      HYPRE_ParCSRPCGFinalize(pcg_solver);
 
      if (solver_id == 1)
      {
         HYPRE_ParAMGFinalize(pcg_precond);
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
 
      HYPRE_ParCSRGMRESInitialize(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRGMRESSetKDim(pcg_solver, k_dim);
      HYPRE_ParCSRGMRESSetMaxIter(pcg_solver, 100);
      HYPRE_ParCSRGMRESSetTol(pcg_solver, tol);
      HYPRE_ParCSRGMRESSetLogging(pcg_solver, 1);
 
      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */

         pcg_precond = HYPRE_ParAMGInitialize(); 
         HYPRE_ParAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_ParAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_ParAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_ParAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_ParAMGSetMaxIter(pcg_precond, 1);
         HYPRE_ParAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_ParAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_ParAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_ParAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_ParAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_ParAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                   HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSetup,
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
         ierr = HYPRE_ParCSRPilutInitialize( MPI_COMM_WORLD, &pcg_precond ); 
         if (ierr) {
	   printf("Error in ParPilutInitialize\n");
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
      HYPRE_ParCSRGMRESFinalize(pcg_solver);
 
      if (solver_id == 3)
      {
         HYPRE_ParAMGFinalize(pcg_precond);
      }

      if (solver_id == 7)
      {
         HYPRE_ParCSRPilutFinalize(pcg_precond);
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
 
      HYPRE_ParCSRCGNRInitialize(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRCGNRSetMaxIter(pcg_solver, 1000);
      HYPRE_ParCSRCGNRSetTol(pcg_solver, tol);
      HYPRE_ParCSRCGNRSetLogging(pcg_solver, 1);
 
      if (solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         pcg_precond = HYPRE_ParAMGInitialize(); 
         HYPRE_ParAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_ParAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_ParAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_ParAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_ParAMGSetMaxIter(pcg_precond, 1);
         HYPRE_ParAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_ParAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_ParAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_ParAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_ParAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_ParAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_ParCSRCGNRSetPrecond(pcg_solver,
                                   HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSolveT,
                                   HYPRE_ParAMGSetup,
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
      HYPRE_ParCSRCGNRFinalize(pcg_solver);
 
      if (solver_id == 5)
      {
         HYPRE_ParAMGFinalize(pcg_precond);
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
   hypre_PrintCSRVector(x, "driver.out.x");
#endif


   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_DestroyParCSRMatrix(A);
   hypre_DestroyParVector(b);
   hypre_DestroyParVector(x);
/*
   hypre_FinalizeMemoryDebug();
*/
   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}

