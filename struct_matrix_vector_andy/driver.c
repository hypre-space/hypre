 
#include "mpi.h"
#include "HYPRE.h"
 
#ifdef HYPRE_DEBUG
/* debugging header */
#include <cegdb.h>
#endif

#ifdef HYPRE_DEBUG
/* malloc debug stuff */
#include <gmalloc.h>
char amps_malloclog[256];

#endif
 
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
   int     Nx = 4;
   int     Ny = 3;
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
   int     i, ierr=0;

   HYPRE_StructGrid          grid;    /* grid structure */
   HYPRE_StructStencil       stencil; /* stencil structure */
   HYPRE_StructMatrix matrix;  /* matrix structure */
   HYPRE_StructVector rhs;     /* vector structure */
   HYPRE_StructVector soln;    /* vector structure */
   HYPRE_StructSolver solver;  /* solver structure */

   int         num_procs, myid;
   int         P, Q, p, q;

   /* Initialize Petsc */ /* In regular code should not be done here */
   PetscInitialize(&argc, &argv, (char *)0, NULL);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

#ifdef HYPRE_DEBUG
   cegdb(&argc, &argv, myid);

   /* malloc debug stuff */
   malloc_logpath = amps_malloclog;
   sprintf(malloc_logpath, "malloc.log.%04d", myid);
#endif


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
   grid = HYPRE_NewStructGrid(2);
   HYPRE_SetStructGridExtents(grid, ilower, iupper);
      printf("Proc %d setting grid extents (%d, %d) to (%d, %d)\n",
              myid, ilower[0], ilower[1], iupper[0], iupper[1]);
   if ((p == 1) && (q == 0))
   {
      ilower2[0] = ilower[0] + nx;
      ilower2[1] = ilower[1];
      iupper2[0] = iupper[0] + nx;
      iupper2[1] = iupper[1];
      HYPRE_SetStructGridExtents(grid, ilower2, iupper2);
      printf("Proc %d setting grid extents (%d, %d) to (%d, %d)\n",
              myid, ilower2[0], ilower2[1], iupper2[0], iupper2[1]);
   }
   HYPRE_AssembleStructGrid(grid);

   /* set up the stencil structure */
   stencil = HYPRE_NewStructStencil(2, 5);
   for (i = 0; i < 5; i++)
      HYPRE_SetStructStencilElement(stencil, i, offsets[i]);

   /* set up the matrix structure */
   matrix = HYPRE_NewStructMatrix( MPI_COMM_WORLD, grid, stencil);

   HYPRE_SetStructMatrixStorageType( matrix, HYPRE_PETSC_MATRIX );

   /* Fill in the matrix elements */
   for (index[1] = ilower[1]; index[1] <= iupper[1]; index[1]++)
      for (index[0] = ilower[0]; index[0] <= iupper[0]; index[0]++)
         HYPRE_SetStructMatrixCoeffs(matrix, index, coeffs);
   if ((p == 1) && (q == 0))
   {
      for (index[1] = ilower2[1]; index[1] <= iupper2[1]; index[1]++)
	 for (index[0] = ilower2[0]; index[0] <= iupper2[0]; index[0]++)
	    HYPRE_SetStructMatrixCoeffs(matrix, index, coeffs);
   }
   ierr = HYPRE_AssembleStructMatrix(matrix);
   if( ierr ) {printf("Error returned by AssembleStructMatrix\n"); return;}

   /* Output information about the matrix */
   HYPRE_PrintStructMatrix(matrix);

   /* set up the vector structures for RHS and soln */
   rhs = HYPRE_NewStructVector( MPI_COMM_WORLD, grid, stencil );
   soln = HYPRE_NewStructVector( MPI_COMM_WORLD, grid, stencil );

   /* Fill in elements for RHS and soln */
   HYPRE_SetStructVector( rhs, &one );
   HYPRE_SetStructVector( soln, &zero );

   /* Set up the solver structure */
   solver = HYPRE_NewStructSolver( MPI_COMM_WORLD, grid, stencil );

   ierr = HYPRE_StructSolverSetup( solver, matrix, soln, rhs );
   if( ierr ) {printf("Error returned by StructSolverSetup\n"); return;}

   /* Solve the linear system represented in the solver structure */
   HYPRE_StructSolverSolve( solver );

   /* Output pertinent information on solver */
   HYPRE_PrintStructVector( soln );

   /* Free stuff up in opposite order of allocating */
   HYPRE_FreeStructSolver( solver );
   HYPRE_FreeStructVector( soln );
   HYPRE_FreeStructVector( rhs );
   HYPRE_FreeStructMatrix( matrix );
   HYPRE_FreeStructStencil( stencil );
   HYPRE_FreeStructGrid( grid );

     } /*End of loop over num_runs */

#ifdef HYPRE_DEBUG
   /* malloc debug stuff */
   malloc_verify(0);
   malloc_heap_map();
   malloc_shutdown();
#endif
}

