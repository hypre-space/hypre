
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
   int                 time_index;
   int                 print_usage;
   int                 build_matrix_type;
   int                 build_matrix_arg_index;
   int                 build_rhs_type;
   int                 build_rhs_arg_index;
   int                 solver_id;
   int                 relax_default;
   int                 coarsen_type;
   int                 max_levels;
   double	       norm;
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

   int                 num_procs, myid;

#if 0
   int 		      *global_part;
#endif

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

#if 0 
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
#endif

   num_procs = 1;
   myid = 0;
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
 
   if ( (print_usage) && (myid == 0) )
   {
      printf("\n");
      printf("Usage: %s [<options>]\n", argv[0]);
      printf("\n");
      printf("  -fromfile <filename>   : build matrix from file\n");
      printf("\n");
      printf("  -laplacian [<options>] : build laplacian matrix\n");
      printf("    -n <nx> <ny> <nz>    : problem size per processor\n");
      printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("\n");
      printf("  -solver <ID>           : solver ID\n");
      printf("\n");

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
#if 0
      BuildParFromFile(argc, argv, build_matrix_arg_index, &A, &global_part);
#endif
      BuildFromFile(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 1 )
   {
#if 0
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &A, &global_part);
#endif
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
   hypre_PrintParCSRMatrix(A, "driver.out.A");
#endif
#if 0
   hypre_PrintCSRMatrix(A, "driver.out.A");
#endif

#if 0
   b = hypre_CreateParVector(MPI_COMM_WORLD, volume, global_part[myid],
	global_part[myid+1]-global_part[myid]);
   hypre_InitializeParVector(b);
   hypre_SetParVectorConstantValues(b,1.0);

   x = hypre_CreateParVector(MPI_COMM_WORLD, volume, global_part[myid],
	global_part[myid+1]-global_part[myid]);
   hypre_InitializeParVector(x);
   hypre_SetParVectorConstantValues(x,0.0);
#endif

   if ( build_rhs_type == 1 )
   {
      BuildRhsFromFile(argc, argv, build_rhs_arg_index, A, &b);
 
      x = hypre_CreateVector( hypre_CSRMatrixNumRows(A));
      hypre_InitializeVector(x);
      hypre_SetVectorConstantValues(x, 0.0);      
   }
   else if ( build_rhs_type == 3 )
   {
      b = hypre_CreateVector( hypre_CSRMatrixNumRows(A));
      hypre_InitializeVector(b);
      hypre_SetVectorRandomValues(b, 22775);
      norm = 1.0/sqrt(hypre_InnerProd(b,b));
      ierr = hypre_ScaleVector(norm, b);      
 
      x = hypre_CreateVector( hypre_CSRMatrixNumRows(A));
      hypre_InitializeVector(x);
      hypre_SetVectorConstantValues(x, 0.0);      
   }
   else if ( build_rhs_type == 4 )
   {
      x = hypre_CreateVector( hypre_CSRMatrixNumRows(A));
      hypre_InitializeVector(x);
      hypre_SetVectorConstantValues(x, 1.0);      
 
      b = hypre_CreateVector( hypre_CSRMatrixNumRows(A));
      hypre_InitializeVector(b);
      hypre_Matvec(1.0,A,x,0.0,b);
 
      hypre_SetVectorConstantValues(x, 0.0);      
   }
   else
   {
      b = hypre_CreateVector(hypre_CSRMatrixNumRows(A));
      hypre_InitializeVector(b);
      hypre_SetVectorConstantValues(b, 0.0);

      x = hypre_CreateVector(hypre_CSRMatrixNumRows(A));
      hypre_InitializeVector(x);
      hypre_SetVectorConstantValues(x, 1.0);
   }
   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      double   strong_threshold;
      int      cycle_type;
      int      ioutdat;
      int     *num_grid_sweeps;  
      int     *grid_relax_type;   
      int    **grid_relax_points; 
      double   relax_weight;

      strong_threshold = 0.25;
      cycle_type       = 1;
      relax_weight = 1.0;
      ioutdat = 3;

      num_grid_sweeps = hypre_CTAlloc(int,4);
      grid_relax_type = hypre_CTAlloc(int,4);
      grid_relax_points = hypre_CTAlloc(int *,4);

      /* fine grid */
      num_grid_sweeps[0] = 2;
      grid_relax_type[0] = relax_default; 
      grid_relax_points[0] = hypre_CTAlloc(int, 2); 
      grid_relax_points[0][0] = -1;
      grid_relax_points[0][1] = 1;

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

      /* coarsest grid */
      num_grid_sweeps[3] = 1;
      grid_relax_type[3] = 9;
      grid_relax_points[3] = hypre_CTAlloc(int, 1);
      grid_relax_points[3][0] = 0;

      arg_index = 0;
      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-w") == 0 )
         {
            arg_index++;
            relax_weight = atof(argv[arg_index++]);
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
         else
         {
            arg_index++;
         }
      }

      time_index = hypre_InitializeTiming("Setup");
      hypre_BeginTiming(time_index);

      amg_solver = HYPRE_AMGInitialize();
      HYPRE_AMGSetCoarsenType(amg_solver, coarsen_type);
      HYPRE_AMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_AMGSetLogging(amg_solver, ioutdat, "driver.out.log");
      HYPRE_AMGSetCycleType(amg_solver, cycle_type);
      HYPRE_AMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      HYPRE_AMGSetGridRelaxType(amg_solver, grid_relax_type);
      HYPRE_AMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      HYPRE_AMGSetRelaxWeight(amg_solver, relax_weight);
      HYPRE_AMGSetMaxLevels(amg_solver, max_levels);
      HYPRE_AMGSetup(amg_solver, A, b, x);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("Solve");
      hypre_BeginTiming(time_index);

      HYPRE_AMGSolve(amg_solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_AMGFinalize(amg_solver);
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

#if 0
   hypre_DestroyParCSRMatrix(A);
   hypre_DestroyParVector(b);
   hypre_DestroyParVector(x);
#endif
   hypre_DestroyCSRMatrix(A);
   hypre_DestroyVector(b);
   hypre_DestroyVector(x);

#if 0
   hypre_TFree(global_part);
#endif
/*
   hypre_FinalizeMemoryDebug();
*/
   /* Finalize MPI */
   MPI_Finalize();
#if 0
#endif

   return (0);
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
      printf("  FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   A = hypre_ReadCSRMatrix(filename);

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
#endif
   hypre_CSRMatrix    *A;

   int                 num_procs, myid, volume;
   int                 p, q, r;
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
    * Set up the grid structure
    *-----------------------------------------------------------*/

   volume  = nx*ny*nz;

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

#if 0   
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
#endif
   hypre_CSRMatrix    *A;

   int                 num_procs, myid, volume;
   int                 p, q;
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
    * Set up the grid structure
    *-----------------------------------------------------------*/

   volume  = nx*ny;

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

#if 0   
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
#endif
   hypre_CSRMatrix    *A;

   int                 num_procs, myid, volume;
   int                 p, q, r;
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
    * Set up the grid structure
    *-----------------------------------------------------------*/

   volume  = nx*ny*nz;

   /* compute p,q,r from P,Q, R and myid */
   p = myid % P;
   q = ( myid - p)/P;

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
#endif
   hypre_CSRMatrix    *A;

   int                 num_procs, myid, volume;
   int                 p, q, r;
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
    * Set up the grid structure
    *-----------------------------------------------------------*/

   volume  = nx*ny*nz;

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
 
   b = hypre_ReadVector(filename);
 
   *b_ptr = b;
 
   return (0);
}

