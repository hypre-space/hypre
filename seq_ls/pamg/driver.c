
#include "headers.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (csr storage).
 * Do `driver -help' for usage info.
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   int                 arg_index;
/*   int                 time_index; */
   int                 print_usage;
   int                 build_matrix_type;
   int                 build_matrix_arg_index;
   int                 build_rhs_type;
   int                 build_rhs_arg_index;
   int                 build_funcs_type;
   int                 build_funcs_arg_index;
   int                *dof_func;
   int                 interp_type;
   int                 solver_id;
   int                 relax_default;
   int                 coarsen_type;
   int                 max_levels;
   int                 num_functions;
   int                 num_relax_steps;
   double	       norm;
   double	       tol;
   int                 i, j, k;
   int                 k_dim;
   int		       ierr = 0;

#if 0
   hypre_ParCSRMatrix *A;
   hypre_ParVector    *b;
   hypre_ParVector    *x;
#endif
   hypre_CSRMatrix    *A;
   hypre_Vector       *b;
   hypre_Vector       *x;

   HYPRE_Solver        amg_solver;
   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond;

/*   int                 num_procs, myid;   */

#if 0
   int 		      *global_part;
#endif

      double   strong_threshold;
      int      cycle_type;
      int      ioutdat;
      int      num_iterations;  
      int      num_sweep;  
      int     *num_grid_sweeps;  
      int     *grid_relax_type;   
      int    **grid_relax_points; 
      double  *relax_weight;
      double   final_res_norm;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   dof_func = NULL;

   /* Initialize MPI */
   /* MPI_Init(&argc, &argv); */

#if 0 
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
#endif

/*   num_procs = 1; 
     myid = 0;         */
/*
   hypre_InitMemoryDebug(myid);
*/
   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   build_matrix_type      = 1;
   build_matrix_arg_index = argc;
   build_rhs_type      = 0;
   build_rhs_arg_index = argc;
   coarsen_type      = 0;
   relax_default = 0;
   interp_type = 0;
   num_functions = 1;
   num_relax_steps = 1;
   build_funcs_type = 0;
   build_funcs_arg_index = argc;
   tol = 1.e-6;
   k_dim = 5;

   solver_id = 0;
   max_levels = 25;

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
      else if ( strcmp(argv[arg_index], "-ruge") == 0 )
      {
         arg_index++;
         coarsen_type = 1;
      }
      else if ( strcmp(argv[arg_index], "-rugeL") == 0 )
      {
         arg_index++;
         coarsen_type = 2;
      }
      else if ( strcmp(argv[arg_index], "-cr") == 0 )
      {
         arg_index++;
         coarsen_type = 3;
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nrlx") == 0 )
      {
         arg_index++;
         num_relax_steps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index++;
         relax_default = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-maxlev") == 0 )
      {
         arg_index++;
         max_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rbm") == 0 )
      {
         arg_index++;
         interp_type      = 1;
      }
      else if ( strcmp(argv[arg_index], "-cri") == 0 )
      {
         arg_index++;
         interp_type      = 2;
      }
      else if ( strcmp(argv[arg_index], "-nf") == 0 )
      {
         arg_index++;
         num_functions = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-funcsfromfile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 1;
         build_funcs_arg_index = arg_index;
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

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
 
   if (print_usage)
   {
      printf("\n");
      printf("Usage: %s [<options>]\n", argv[0]);
      printf("\n");
      printf("  -fromfile <filename>   : build matrix from file\n");
      printf("\n");
      printf("  -laplacian (default)   : build laplacian 7pt operator\n");
      printf("  -9pt                   : build laplacian 9pt operator\n");
      printf("  -27pt                  : build laplacian 27pt operator\n");
      printf("  -difconv               : build 7pt diffusion-convection\n");
      printf("  -ruge                  : use classical coarsening \n");
      printf("  -rugeL                 : use classical coarsening \n");
      printf("                           (more efficient version) \n");
      printf("  -cr                    : use coarsening based on compatible relaxation\n");
      printf("  -nrlx <nx>             : no. relaxation step for cr coarsening\n");
      printf("  -rbm                   : use rigid body motion interpolation \n");
      printf("  -rlx <rlxtype>         : rlxtype = 0 Jacobi relaxation \n");
      printf("                                     1 Gauss-Seidel relaxation \n");
      printf("  -w <rlxweight>         : defines relaxation weight for Jacobi\n");
      printf("  -ns <val>              : Use <val> sweeps on each level\n");
      printf("                           (default C/F down, F/C up, F/C fine\n");
      printf("  -mu <val>              : sets cycle type, 1=V, 2=W, etc\n");
      printf("  -th <threshold>        : defines threshold for coarsenings\n");
      printf("  -maxlev <ml>           : defines max. no. of levels\n");
      printf("    -n <nx> <ny> <nz>    : problem size per processor\n");
      printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      printf("\n");
      printf("  -solver <ID>           : solver ID\n");
      printf("      0=AMG, 1=AMG-CG, 2=CG, 3=AMG-GMRES, 4=GMRES\n");
      printf("\n");

      exit(1);
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
  
      printf("Running with these driver parameters:\n");
      printf("  solver ID    = %d\n", solver_id);
 

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   if ( build_matrix_type == 0 )
   {
      BuildFromFile(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildLaplacian(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 3 )
   {
      BuildLaplacian9pt(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 4 )
   {
      BuildLaplacian27pt(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 5 )
   {
      BuildDifConv(argc, argv, build_matrix_arg_index, &A);
   }
   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

#if 0
   hypre_CSRMatrixPrint(A, "driver.out.A");
#endif

   if ( build_rhs_type == 1 )
   {
      BuildRhsFromFile(argc, argv, build_rhs_arg_index, A, &b);
 
      x = hypre_VectorCreate( hypre_CSRMatrixNumRows(A));
      hypre_VectorInitialize(x);
      hypre_VectorSetConstantValues(x, 0.0);      
   }
   else if ( build_rhs_type == 3 )
   {
      b = hypre_VectorCreate( hypre_CSRMatrixNumRows(A));
      hypre_VectorInitialize(b);
      hypre_VectorSetRandomValues(b, 22775);
      norm = 1.0/sqrt(hypre_VectorInnerProd(b,b));
      ierr = hypre_VectorScale(norm, b);      
 
      x = hypre_VectorCreate( hypre_CSRMatrixNumRows(A));
      hypre_VectorInitialize(x);
      hypre_VectorSetConstantValues(x, 0.0);      
   }
   else if ( build_rhs_type == 4 )
   {
      x = hypre_VectorCreate( hypre_CSRMatrixNumRows(A));
      hypre_VectorInitialize(x);
      hypre_VectorSetConstantValues(x, 1.0);      
 
      b = hypre_VectorCreate( hypre_CSRMatrixNumRows(A));
      hypre_VectorInitialize(b);
      hypre_CSRMatrixMatvec(1.0,A,x,0.0,b);
 
      hypre_VectorSetConstantValues(x, 0.0);      
   }
   else
   {
      b = hypre_VectorCreate(hypre_CSRMatrixNumRows(A));
      hypre_VectorInitialize(b);
      hypre_VectorSetConstantValues(b, 0.0);

      x = hypre_VectorCreate(hypre_CSRMatrixNumRows(A));
      hypre_VectorInitialize(x);
      hypre_VectorSetConstantValues(x, 1.0);
   }
   if ( build_funcs_type == 1 )
   {
      BuildFuncsFromFile(argc, argv, build_funcs_arg_index, &dof_func);    
   }
   else /* if (num_functions > 1) */ 
   {
      printf("\n Number of unknown functions = %d\n\n",num_functions);
      dof_func = hypre_CTAlloc(int,hypre_CSRMatrixNumRows(A));
 
      for (j = 0; j < hypre_CSRMatrixNumRows(A)/num_functions; j++)
      {
         for (k = 0; k < num_functions; k++) dof_func[j*num_functions+k] = k;
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

      strong_threshold = 0.25;
      cycle_type       = 1;
      ioutdat = 3;
      num_sweep = 1;

      num_grid_sweeps = hypre_CTAlloc(int,4);
      grid_relax_type = hypre_CTAlloc(int,4);
      grid_relax_points = hypre_CTAlloc(int *,4);
      relax_weight = hypre_CTAlloc(double,max_levels);
      for (i=0; i < max_levels; i++)
	 relax_weight[i] = 1.0;






      arg_index = 0;
      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-w") == 0 )
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
         else if ( strcmp(argv[arg_index], "-iout") == 0 )
         {
            arg_index++;
            ioutdat  = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-mu") == 0 )
         {
            arg_index++;
            cycle_type = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-ns") == 0 )
         {
            arg_index++;
            num_sweep = atoi(argv[arg_index++]);
         }
         else
         {
            arg_index++;
         }
      }

   if (solver_id == 0 || solver_id == 1 || solver_id == 3)
   {
/*----------------------------------------------------------------
 *
 * Option to have numerous relaxation sweeps per level.  In this case
 * we use C/F relaxation going down, F/C going up, for all sweeps
 *
 *----------------------------------------------------------------*/

     
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


      /* coarsest grid */
      num_grid_sweeps[3] = 1;
      grid_relax_type[3] = 9;
      grid_relax_points[3] = hypre_CTAlloc(int, 1);
      grid_relax_points[3][0] = 0;
   }

   if (solver_id == 0)
   {

      /* time_index = hypre_InitializeTiming("Setup");
      hypre_BeginTiming(time_index); */

      amg_solver = HYPRE_AMGInitialize();
      HYPRE_AMGSetCoarsenType(amg_solver, coarsen_type);
      HYPRE_AMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_AMGSetNumRelaxSteps(amg_solver, num_relax_steps);
      HYPRE_AMGSetLogging(amg_solver, ioutdat, "driver.out.log");
      HYPRE_AMGSetCycleType(amg_solver, cycle_type);
      HYPRE_AMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      HYPRE_AMGSetGridRelaxType(amg_solver, grid_relax_type);
      HYPRE_AMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      HYPRE_AMGSetRelaxWeight(amg_solver, relax_weight);
      HYPRE_AMGSetMaxLevels(amg_solver, max_levels);
      HYPRE_AMGSetInterpType(amg_solver, interp_type);
      HYPRE_AMGSetNumFunctions(amg_solver, num_functions);
      HYPRE_AMGSetDofFunc(amg_solver, dof_func);

      HYPRE_AMGSetup(amg_solver, (HYPRE_CSRMatrix) A, (HYPRE_Vector) b, 
			(HYPRE_Vector) x);
      /* hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming(); 
 
      time_index = hypre_InitializeTiming("Solve");
      hypre_BeginTiming(time_index); */

      HYPRE_AMGSolve(amg_solver, (HYPRE_CSRMatrix) A, (HYPRE_Vector) b, 
			(HYPRE_Vector) x);

      /* hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming(); */
 
      HYPRE_AMGFinalize(amg_solver);
   }

   /*-----------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2)
   {
      /* time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index); */

      HYPRE_CSRPCGCreate( &pcg_solver);
      HYPRE_CSRPCGSetMaxIter(pcg_solver, 500);
      HYPRE_CSRPCGSetTol(pcg_solver, tol);
      HYPRE_CSRPCGSetTwoNorm(pcg_solver, 1);
      HYPRE_CSRPCGSetRelChange(pcg_solver, 0);
      HYPRE_CSRPCGSetLogging(pcg_solver, 1);

      if (solver_id == 1)
      {
	 /* use AMG as preconditioner */
	 printf ("Solver: AMG-PCG\n");
         pcg_precond = HYPRE_AMGInitialize();
         HYPRE_AMGSetCoarsenType(pcg_precond, coarsen_type);
         HYPRE_AMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_AMGSetNumRelaxSteps(pcg_precond, num_relax_steps);
         HYPRE_AMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_AMGSetMaxIter(pcg_precond, 1);
         HYPRE_AMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_AMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_AMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_AMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_AMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_AMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_AMGSetInterpType(pcg_precond, interp_type);
         HYPRE_AMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_AMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_CSRPCGSetPrecond(pcg_solver, HYPRE_AMGSolve, HYPRE_AMGSetup, 
				pcg_precond);
      }
      else if (solver_id == 2)
      {
	 /* use diagonal scaling as preconditioner */
	 printf ("Solver: DS-PCG\n");
	 pcg_precond = NULL;
	 HYPRE_CSRPCGSetPrecond(pcg_solver, HYPRE_CSRDiagScale, 
				HYPRE_CSRDiagScaleSetup, pcg_precond);
      }

      HYPRE_CSRPCGSetup(pcg_solver, (HYPRE_CSRMatrix) A, (HYPRE_Vector) b, 
			(HYPRE_Vector) x);
      /* hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index); */

      HYPRE_CSRPCGSolve(pcg_solver, (HYPRE_CSRMatrix) A, (HYPRE_Vector) b, 
			(HYPRE_Vector) x);

      /* hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming(); */

      HYPRE_CSRPCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_CSRPCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

      HYPRE_CSRPCGDestroy(pcg_solver);
 
      if (solver_id == 1) HYPRE_AMGFinalize(pcg_precond);
   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
   }

   /*-----------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4)
   {
      /* time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index); */

      HYPRE_CSRGMRESCreate( &pcg_solver);
      HYPRE_CSRGMRESSetMaxIter(pcg_solver, 500);
      HYPRE_CSRGMRESSetTol(pcg_solver, tol);
      HYPRE_CSRGMRESSetKDim(pcg_solver, k_dim);
      HYPRE_CSRGMRESSetLogging(pcg_solver, 1);

      if (solver_id == 3)
      {
	 /* use AMG as preconditioner */
	 printf ("Solver: AMG-GMRES\n");
         pcg_precond = HYPRE_AMGInitialize();
         HYPRE_AMGSetCoarsenType(pcg_precond, coarsen_type);
         HYPRE_AMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_AMGSetNumRelaxSteps(pcg_precond, num_relax_steps);
         HYPRE_AMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_AMGSetMaxIter(pcg_precond, 1);
         HYPRE_AMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_AMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_AMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_AMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_AMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_AMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_AMGSetInterpType(pcg_precond, interp_type);
         HYPRE_AMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_AMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_CSRGMRESSetPrecond(pcg_solver, HYPRE_AMGSolve, HYPRE_AMGSetup, 
				pcg_precond);
      }
      else if (solver_id == 2)
      {
	 /* use diagonal scaling as preconditioner */
	 printf ("Solver: DS-GMRES\n");
	 pcg_precond = NULL;
	 HYPRE_CSRGMRESSetPrecond(pcg_solver, HYPRE_CSRDiagScale, 
				HYPRE_CSRDiagScaleSetup, pcg_precond);
      }

      HYPRE_CSRGMRESSetup(pcg_solver, (HYPRE_CSRMatrix) A, (HYPRE_Vector) b, 
			(HYPRE_Vector) x);
      /* hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index); */

      HYPRE_CSRGMRESSolve(pcg_solver, (HYPRE_CSRMatrix) A, (HYPRE_Vector) b, 
			(HYPRE_Vector) x);

      /* hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming(); */

      HYPRE_CSRGMRESGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_CSRGMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

      HYPRE_CSRGMRESDestroy(pcg_solver);
 
      if (solver_id == 1) HYPRE_AMGFinalize(pcg_precond);
   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
   }

#if 0
   hypre_PrintCSRVector(x, "driver.out.x");
#endif

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

#if 0
   hypre_DestroyParCSRMatrix(A);
   hypre_DestroyParVector(b);
   hypre_DestroyParVector(x);
#endif
   hypre_CSRMatrixDestroy(A);
   hypre_VectorDestroy(b);
   hypre_VectorDestroy(x);

#if 0
   hypre_TFree(global_part);
#endif
/*
   hypre_FinalizeMemoryDebug();
*/
   /* Finalize MPI */
   /* MPI_Finalize(); */
#if 0
#endif

   return (ierr);
}

/*----------------------------------------------------------------------
 * Build matrix from file.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildFromFile( int               argc,
               char             *argv[],
               int               arg_index,
               hypre_CSRMatrix **A_ptr     )
{
   char               *filename;

#if 0
   hypre_ParCSRMatrix *A;
   int 		      *global_part;
#endif
   hypre_CSRMatrix    *A;

   int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

#if 0 
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
#endif

   myid = 0;

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
      printf("  Operator FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   A = hypre_CSRMatrixRead(filename);

   *A_ptr = A;
#if 0
   *global_part_ptr = global_part;
#endif

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildLaplacian( int               argc,
                char             *argv[],
                int               arg_index,
                hypre_CSRMatrix **A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;

#if 0
   hypre_ParCSRMatrix *A;
   int 		      *global_part;
   int                 p, q, r;
#endif
   hypre_CSRMatrix    *A;

   int                 num_procs, myid;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

#if 0 
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
#endif

   num_procs = 1;
   myid = 0;

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
/*
   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }
*/
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

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */

#if 0   
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );
   A = hypre_GenerateLaplacian(MPI_COMM_WORLD,
                               nx, ny, nz, P, Q, R, p, q, r,
                               values, &global_part);
#endif
   A = hypre_GenerateLaplacian(nx, ny, nz, P, Q, R, values);

   hypre_TFree(values);

   *A_ptr = A;
#if 0
   *global_part_ptr = global_part;
#endif

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D 
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildLaplacian9pt( int               argc,
                   char             *argv[],
                   int               arg_index,
                   hypre_CSRMatrix **A_ptr     )
{
   int                 nx, ny;
   int                 P, Q;

#if 0
   hypre_ParCSRMatrix *A;
   int 		      *global_part;
   int                 p, q;
#endif
   hypre_CSRMatrix    *A;

   int                 num_procs, myid;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

#if 0 
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
#endif

   num_procs = 1;
   myid = 0;

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
/*
   if ((P*Q) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }
*/
   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian:\n");
      printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      printf("    (Px, Py) = (%d, %d)\n", P,  Q);
   }

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

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */

#if 0   
   p = myid % P;
   q = ( myid - p)/P;
   A = hypre_GenerateLaplacian9pt(MPI_COMM_WORLD,
                               nx, ny, P, Q, p, q,
                               values, &global_part);
#endif
   A = hypre_GenerateLaplacian9pt(nx, ny, P, Q, values);

   hypre_TFree(values);

   *A_ptr = A;
#if 0
   *global_part_ptr = global_part;
#endif

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 27-point laplacian in 3D 
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildLaplacian27pt(int               argc,
                   char             *argv[],
                   int               arg_index,
                   hypre_CSRMatrix **A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;

#if 0
   hypre_ParCSRMatrix *A;
   int 		      *global_part;
   int                 p, q, r;
#endif
   hypre_CSRMatrix    *A;

   int                 num_procs, myid;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q, R and myid */
#if 0 
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
   p = myid % P;
   q = ( myid - p)/P;

#endif

   num_procs = 1;
   myid = 0;

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
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  27pt_Laplacian:\n");
      printf("    (nx, ny) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py) = (%d, %d, %d)\n", P,  Q, R);
   }

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 2);

   values[1] = -1.0;

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
   {
      values[0] = 8.0;
   }
   if (ny*nx == 1 || ny*nz == 1 || nx*ny == 1)
   {
      values[0] = 2.0;
   }

   A = hypre_GenerateLaplacian27pt(nx, ny, nz, P, Q, R, values);

   hypre_TFree(values);

   *A_ptr = A;
#if 0
   *global_part_ptr = global_part;
#endif

   return (0);
}
/*----------------------------------------------------------------------
 * Build standard 7-point convection-diffusion in 3D anisotropy.
 * Parameters given in command line.
 *
 *  Operator:
 *
 *  -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f
 *
 *----------------------------------------------------------------------*/

int
BuildDifConv(   int               argc,
                char             *argv[],
                int               arg_index,
                hypre_CSRMatrix **A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;
   double              ax, ay, az;
   double              hinx,hiny,hinz;

#if 0
   hypre_ParCSRMatrix *A;
   int 		      *global_part;
   int                 p, q, r;
#endif
   hypre_CSRMatrix    *A;

   int                 num_procs, myid;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
#if 0 
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );
#endif

   num_procs = 1;
   myid = 0;

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
/*
   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }
*/
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

   A = hypre_GenerateDifConv(nx, ny, nz, P, Q, R, values);

   hypre_TFree(values);

   *A_ptr = A;
#if 0
   *global_part_ptr = global_part;
#endif

   return (0);
}

/********************************************************************
 *      
 * Build RHS
 *
 *********************************************************************/

int
BuildRhsFromFile( int                  argc,
                  char                *argv[],
                  int                  arg_index,
                  hypre_CSRMatrix  *A,
                  hypre_Vector    **b_ptr     )
{
   char               *filename;
 
   hypre_Vector *b;
 
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
 
   printf("  Rhs FromFile: %s\n", filename);
 
   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   b = hypre_VectorRead(filename);
 
   *b_ptr = b;
 
   return (0);
}


/********************************************************************
 *      
 * Build dof_func vector
 *
 *********************************************************************/

int
BuildFuncsFromFile( int                  argc,
                    char                *argv[],
                    int                  arg_index,
                    int                 **dof_func_ptr     )
{
   char               *filename;

   FILE    *fp;

   int     *dof_func;
   int      size;
   
   int      j;


 
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
 
   printf("\n  FuncsFromFile: %s\n", filename);
 
   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(filename, "r");

   fscanf(fp, "%d", &size);
   dof_func = hypre_CTAlloc(int, size);

   for (j = 0; j < size; j++)
   {
      fscanf(fp, "%d", &dof_func[j]);
   }

   fclose(fp);
  *dof_func_ptr = dof_func;
 
   return (0);
}

