
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "mpi.h"
#include "../utilities/memory.h"
#include "../utilities/timing.h"
#include "../struct_matrix_vector/HYPRE_mv.h"
#include "HYPRE_ls.h"
 
#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 * Standard 7-point laplacian in 3D with grid and anisotropy determined
 * as command line arguments.
 *
 * Command line arguments: nx ny nz P Q R dx dy dz
 *   n[xyz] = size of local problem
 *   [PQR]  = process topology
 *   d[xyz] = diffusion coefficients
 *----------------------------------------------------------------------*/

int   main(argc, argv)
int   argc;
char *argv[];
{
   int                 A_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   HYPRE_StructMatrix  A;
   HYPRE_StructVector  b;
   HYPRE_StructVector  x;

   HYPRE_StructSolver  smg_solver;
   int                 num_iterations;
   int                 time_index;

   int                 num_procs, myid;

   int                 nx, ny, nz;
   int                 P, Q, R;
   double              dx, dy, dz;
   int                 p, q, r;
   int                 ilower[3];
   int                 iupper[3];
   int                 volume;
                     
   int                 dim = 3;
                     
   int                 offsets[4][3] = {{-1,  0,  0},
                                        { 0, -1,  0},
                                        { 0,  0, -1},
                                        { 0,  0,  0}};
                     
   HYPRE_StructGrid    grid;
   HYPRE_StructStencil stencil;

   int                 stencil_indices[4];
   double             *values;

   int                 i, s, d;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

#ifdef HYPRE_DEBUG
   cegdb(&argc, &argv, myid);
#endif

   hypre_InitMemoryDebug(myid);

   if (argc > 9)
   {
      nx = atoi(argv[1]);
      ny = atoi(argv[2]);
      nz = atoi(argv[3]);
      P  = atoi(argv[4]);
      Q  = atoi(argv[5]);
      R  = atoi(argv[6]);
      dx = atof(argv[7]);
      dy = atof(argv[8]);
      dz = atof(argv[9]);
   }
   else
   {
      printf("Usage: mpirun -np %d %s <nx,ny,nz,P,Q,R,dx,dy,dz> ,\n\n",
             num_procs, argv[0]);
      printf("     where nx X ny X nz is the problem size per processor;\n");
      printf("           P  X  Q X  R is the processor topology;\n");
      printf("           dx, dy, dz   are the diffusion coefficients.\n");

      exit(1);
   }

   volume = nx*ny*nz;

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/
 
   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /* compute ilower and iupper from p,q,r and nx,ny,nz */
   ilower[0] = nx*p;
   iupper[0] = nx*(p+1) - 1;
   ilower[1] = ny*q;
   iupper[1] = ny*(q+1) - 1;
   ilower[2] = nz*r;
   iupper[2] = nz*(r+1) - 1;

   grid = HYPRE_NewStructGrid(MPI_COMM_WORLD, dim);
   HYPRE_SetStructGridExtents(grid, ilower, iupper);
   HYPRE_AssembleStructGrid(grid);

   /*-----------------------------------------------------------
    * Set up the stencil structure
    *-----------------------------------------------------------*/
 
   stencil = HYPRE_NewStructStencil(dim, 4);
   for (s = 0; s < 4; s++)
   {
      HYPRE_SetStructStencilElement(stencil, s, offsets[s]);
   }

   /*-----------------------------------------------------------
    * Set up the matrix structure
    *-----------------------------------------------------------*/
 
   A = HYPRE_NewStructMatrix(MPI_COMM_WORLD, grid, stencil);
   HYPRE_SetStructMatrixSymmetric(A, 1);
   HYPRE_SetStructMatrixNumGhost(A, A_num_ghost);
   HYPRE_InitializeStructMatrix(A);

   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(double, 4*volume);

   /* Set the coefficients for the grid */
   for (i = 0; i < 4*volume; i += 4)
   {
      for (s = 0; s < 4; s++)
      {
         stencil_indices[s] = s;
         values[i  ] = -dx;
         values[i+1] = -dy;
         values[i+2] = -dz;
         values[i+3] = 2.0*(dx+dy+dz);
      }
   }
   HYPRE_SetStructMatrixBoxValues(A, ilower, iupper, 4,
                                  stencil_indices, values);

   /* Zero out stencils reaching to real boundary */
   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   for (d = 0; d < 3; d++)
   {
      if( ilower[d] == 0 )
      {
         i = iupper[d];
         iupper[d] = 0;
         stencil_indices[0] = d;
         HYPRE_SetStructMatrixBoxValues(A, ilower, iupper,
                                        1, stencil_indices, values);
         iupper[d] = i;
      }
   }

   HYPRE_AssembleStructMatrix(A);

   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(double, volume);

   b = HYPRE_NewStructVector(MPI_COMM_WORLD, grid, stencil);
   HYPRE_InitializeStructVector(b);
   for (i = 0; i < volume; i++)
   {
      values[i] = 1.0;
   }
   HYPRE_SetStructVectorBoxValues(b, ilower, iupper, values);
   HYPRE_AssembleStructVector(b);

   x = HYPRE_NewStructVector(MPI_COMM_WORLD, grid, stencil);
   HYPRE_InitializeStructVector(x);
   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   HYPRE_SetStructVectorBoxValues(x, ilower, iupper, values);
   HYPRE_AssembleStructVector(x);
 
   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("SMG Setup");
   hypre_BeginTiming(time_index);

   smg_solver = HYPRE_StructSMGInitialize(MPI_COMM_WORLD);
   HYPRE_SMGSetMemoryUse(smg_solver, 0);
   HYPRE_SMGSetMaxIter(smg_solver, 50);
   HYPRE_SMGSetTol(smg_solver, 1.0e-06);
   HYPRE_SMGSetNumPreRelax(smg_solver, 1);
   HYPRE_SMGSetNumPostRelax(smg_solver, 1);
   HYPRE_StructSMGSetup(smg_solver, A, b, x);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   time_index = hypre_InitializeTiming("SMG Solve");
   hypre_BeginTiming(time_index);

   HYPRE_StructSMGSolve(smg_solver, A, b, x);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

#if 0
   HYPRE_PrintStructVector("zout_x", x, 0);
#endif

   HYPRE_SMGGetNumIterations(smg_solver, &num_iterations);
   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("\n");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_StructSMGFinalize(smg_solver);
   HYPRE_FreeStructGrid(grid);
   HYPRE_FreeStructMatrix(A);
   HYPRE_FreeStructVector(b);
   HYPRE_FreeStructVector(x);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}


