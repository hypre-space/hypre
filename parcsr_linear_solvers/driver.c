
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "hypre_utilities.h"
#include "hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (parcsr storage).
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
   int                 solver_id;

   hypre_ParCSRMatrix *A;
   hypre_ParVector    *b;
   hypre_ParVector    *x;

   HYPRE_Solver        amg_solver;

   int                 num_procs, myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   hypre_InitMemoryDebug(myid);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   build_matrix_type      = 1;
   build_matrix_arg_index = argc;

   solver_id = 0;

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
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
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
      BuildParFromFile(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &A);
   }

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

#if 0
   hypre_PrintParCSRMatrix(A, "driver.out.A");
#endif

   b = hypre_CreateParVector(MPI_COMM_WORLD,
                             hypre_ParCSRMatrixNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_SetParVectorPartitioningOwner(b, 0);
   hypre_InitializeParVector(b);
   hypre_SetParVectorConstantValues(b, 0.0);

   x = hypre_CreateParVector(MPI_COMM_WORLD,
                             hypre_ParCSRMatrixNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_SetParVectorPartitioningOwner(x, 0);
   hypre_InitializeParVector(x);
   hypre_SetParVectorConstantValues(x, 1.0);

   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      double   strong_threshold;
      int      cycle_type;
      int     *num_grid_sweeps;  
      int     *grid_relax_type;   
      int    **grid_relax_points; 

      strong_threshold = 0.25;
      cycle_type       = 1;

      num_grid_sweeps = hypre_CTAlloc(int,4);
      grid_relax_type = hypre_CTAlloc(int,4);
      grid_relax_points = hypre_CTAlloc(int *,4);

      /* fine grid */
      num_grid_sweeps[0] = 2;
      grid_relax_type[0] = 0; 
      grid_relax_points[0] = hypre_CTAlloc(int, 2); 
      grid_relax_points[0][0] = 1;
      grid_relax_points[0][1] = -1;

      /* down cycle */
      num_grid_sweeps[1] = 2;
      grid_relax_type[1] = 0; 
      grid_relax_points[1] = hypre_CTAlloc(int, 2); 
      grid_relax_points[1][0] = 1;
      grid_relax_points[1][1] = -1;

      /* up cycle */
      num_grid_sweeps[2] = 2;
      grid_relax_type[2] = 0; 
      grid_relax_points[2] = hypre_CTAlloc(int, 2); 
      grid_relax_points[2][0] = -1;
      grid_relax_points[2][1] = 1;

      /* coarsest grid */
      num_grid_sweeps[3] = 1;
      grid_relax_type[3] = 9;
      grid_relax_points[3] = hypre_CTAlloc(int, 1);
      grid_relax_points[3][0] = 0;

      amg_solver = HYPRE_ParAMGInitialize();
      HYPRE_ParAMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_ParAMGSetLogging(amg_solver, 3, "driver.out.log");
      HYPRE_ParAMGSetCycleType(amg_solver, cycle_type);
      HYPRE_ParAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      HYPRE_ParAMGSetGridRelaxType(amg_solver, grid_relax_type);
      HYPRE_ParAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      HYPRE_ParAMGSetMaxLevels(amg_solver, 25);
      HYPRE_ParAMGSetup(amg_solver, A, b, x);

      HYPRE_ParAMGSolve(amg_solver, A, b, x);

      HYPRE_ParAMGFinalize(amg_solver);
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

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from file.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParFromFile( int                  argc,
                  char                *argv[],
                  int                  arg_index,
                  hypre_ParCSRMatrix **A_ptr     )
{
   char               *filename;

   hypre_ParCSRMatrix *A;

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
 
   A = hypre_ReadParCSRMatrix(MPI_COMM_WORLD, filename);

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
                   hypre_ParCSRMatrix **A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;

   hypre_ParCSRMatrix *A;

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

   A = hypre_GenerateLaplacian(MPI_COMM_WORLD,
                               nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

