
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (parcsr storage)
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 * Standard 7-point laplacian in 3D with grid and anisotropy determined
 * as command line arguments.  Do `driver -help' for usage info.
 *----------------------------------------------------------------------*/

int
main( int   argc,
      char *argv[] )
{
   int                 arg_index;
   int                 print_usage;
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;
   int                 solver_id;

   hypre_ParCSRMatrix  *A;
   hypre_ParVector     *b;
   hypre_ParVector     *x;

   int                 num_procs, myid, volume;

   int                 p, q, r;

   double             *values;

   int 		      *global_part;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );


   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 2;
   ny = 4;
   nz = 1;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.0;
   cy = 1.0;
   cz = 0.0;

   solver_id = 0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   print_usage = 0;
   arg_index = 1;
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
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
         break;
      }
      else
      {
         print_usage = 1;
         break;
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
      printf("  -n <nx> <ny> <nz>    : problem size per processor\n");
      printf("  -P <Px> <Py> <Pz>    : processor topology\n");
      printf("  -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("  -solver <ID>         : solver ID\n");
      printf("\n");

      exit(1);
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
      printf("Running with these driver parameters:\n");
      printf("  (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("  (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("  (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      printf("  solver ID    = %d\n", solver_id);
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
   values[0] = 2.0*(cx+cy+cz);
   
   A = hypre_GenerateLaplacian(MPI_COMM_WORLD,nx,ny,nz,P,Q,R,p,q,r,
	values);

   global_part = hypre_ParCSRMatrixRowStarts(A);

   hypre_PrintParCSRMatrix(A, "Laplace");

   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   b = hypre_CreateParVector(MPI_COMM_WORLD, volume, global_part[myid],
	global_part[myid+1]-global_part[myid]);
   hypre_InitializeParVector(b);
   hypre_SetParVectorConstantValues(b,1.0);

   x = hypre_CreateParVector(MPI_COMM_WORLD, volume, global_part[myid],
	global_part[myid+1]-global_part[myid]);
   hypre_InitializeParVector(x);
   hypre_SetParVectorConstantValues(x,0.0);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_DestroyParCSRMatrix(A);
   hypre_DestroyParVector(b);
   hypre_DestroyParVector(x);

   hypre_TFree(global_part);
   hypre_TFree(values);
   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}


