 
#include "mpi.h"
#include "ZZZ.h"
#include "headers.h"  /* This is temporary! */
 
/* Include Petsc linear solver headers for output */
#include "sles.h"

/* debugging header
#include <cegdb.h> */

/* malloc debug stuff */
char amps_malloclog[256];
 
/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (PETSc under the hood)
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 * Example 1: (nonsymmetric storage)
 *
 *    Standard 5-point laplacian in 2D on a 10 x 7 grid,
 *    ignoring boundary conditions for simplicity.
 *----------------------------------------------------------------------*/

int   main(argc, argv)
int   argc;
char *argv[];
{
  /* control constants */
   int     num_runs = 1, curr_run;


   double  zero=0.0, one=1.0;
   int     Nx = 8;
   int     Ny = 6;
   int     nx, ny;
   int     ilower[2];
   int     iupper[2];
   int     ilower2[2];
   int     iupper2[2];

   int     offsets[5][2] = {{ 0,  0},
                            {-1,  0},
                            { 1,  0},
                            { 0, -1},
                            { 0,  1}};

   double  coeffs[5] = { 4, -1, -1, -1, -1};

   int     index[2];
   int     i;

   ZZZ_StructGrid          grid;    /* grid structure */
   ZZZ_StructStencil       stencil; /* stencil structure */
   ZZZ_StructMatrix matrix;  /* matrix structure */
   ZZZ_StructVector rhs;     /* vector structure */
   ZZZ_StructVector soln;    /* vector structure */
   ZZZ_StructSolver solver;  /* solver structure */

   Mat        *A_PETSc; /*strictly for output and checking */
   Vec        *rhs_PETSc, *soln_PETSc;
   int         num_procs, myid;
   int         P, Q, p, q;

   /* Initialize Petsc */ /* In regular code should not be done here */
   PetscInitialize(&argc, &argv, (char *)0, NULL);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   cegdb(&argc, &argv, myid);

   /* malloc debug stuff */
   malloc_logpath = amps_malloclog;
   sprintf(malloc_logpath, "malloc.log.%04d", myid);


   /* determine grid distribution */
   P = Q = (int) sqrt(num_procs);
   p = myid % P;
   q = (myid - p) / P;
   nx = Nx / P;
   ny = Ny / Q;
   ilower[0] = nx*p;
   ilower[1] = ny*q;
   iupper[0] = ilower[0] + nx - 1;
   iupper[1] = ilower[1] + ny - 1;

   /* Loop to test the code further */
   for (curr_run=0; curr_run< num_runs; curr_run++)
     {

   /* set up the grid structure */
   grid = ZZZ_NewStructGrid(2);
   ZZZ_SetStructGridExtents(grid, ilower, iupper);
   if ((p == 1) && (q == 0))
   {
      ilower2[0] = ilower[0] + nx;
      ilower2[1] = ilower[1];
      iupper2[0] = iupper[0] + nx;
      iupper2[1] = iupper[1];
      ZZZ_SetStructGridExtents(grid, ilower2, iupper2);
   }
   ZZZ_AssembleStructGrid(grid);

   /* set up the stencil structure */
   stencil = ZZZ_NewStructStencil(2, 5);
   for (i = 0; i < 5; i++)
      ZZZ_SetStructStencilElement(stencil, i, offsets[i]);

   /* set up the matrix structure */
   matrix = ZZZ_NewStructMatrix( MPI_COMM_WORLD, grid, stencil);

   ZZZ_SetStructMatrixStorageType( matrix, ZZZ_PETSC_MATRIX );

   /* Fill in the matrix elements */
   for (index[1] = ilower[1]; index[1] <= iupper[1]; index[1]++)
      for (index[0] = ilower[0]; index[0] <= iupper[0]; index[0]++)
         ZZZ_SetStructMatrixCoeffs(matrix, index, coeffs);
   if ((p == 1) && (q == 0))
   {
      for (index[1] = ilower2[1]; index[1] <= iupper2[1]; index[1]++)
	 for (index[0] = ilower2[0]; index[0] <= iupper2[0]; index[0]++)
	    ZZZ_SetStructMatrixCoeffs(matrix, index, coeffs);
   }
   ZZZ_AssembleStructMatrix(matrix);

   /* Output information about the matrix */
   /* Note: the user cannot do this!! */
   A_PETSc = (Mat *) zzz_StructMatrixData( (zzz_StructMatrix *) matrix );

   MatView( *A_PETSc, VIEWER_STDOUT_WORLD );
/*   MatView( *A_PETSc, VIEWER_FORMAT_ASCII_MATLAB );*/

   /* set up the vector structures for RHS and soln */
   rhs = ZZZ_NewStructVector( MPI_COMM_WORLD, grid, stencil );
   soln = ZZZ_NewStructVector( MPI_COMM_WORLD, grid, stencil );

   /* Fill in elements for RHS and soln */
   ZZZ_SetStructVector( rhs, &zero );
   ZZZ_SetStructVector( soln, &one );

   /* Set up the solver structure */
   solver = ZZZ_NewStructSolver( MPI_COMM_WORLD, grid, stencil );

   ZZZ_StructSolverSetup( solver, matrix, soln, rhs );

   /* Solve the linear system represented in the solver structure */
   ZZZ_StructSolverSolve( solver );

   /* Output pertinent information on solver */
   /* Note: the user cannot do this!! */
   soln_PETSc = (Vec *) zzz_StructVectorData( (zzz_StructVector *) soln );

   VecView( *soln_PETSc, VIEWER_STDOUT_WORLD );

   /* Free stuff up in opposite order of allocating */
   ZZZ_FreeStructSolver( solver );
   ZZZ_FreeStructVector( soln );
   ZZZ_FreeStructVector( rhs );
   ZZZ_FreeStructMatrix( matrix );
   ZZZ_FreeStructStencil( stencil );
   ZZZ_FreeStructGrid( grid );

     } /*End of loop over num_runs */

   /* malloc debug stuff */
   malloc_verify(0);
   malloc_heap_map();
   malloc_shutdown();
}

