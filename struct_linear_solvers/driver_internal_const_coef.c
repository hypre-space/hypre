#include "headers.h"
 
#ifdef ZZZ_DEBUG
#include <cegdb.h>
#endif

#ifdef ZZZ_DEBUG
char malloc_logpath_memory[256];
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
   MPI_Comm           *comm;

   int                 A_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
   int                 b_num_ghost[6] = { 0, 0, 0, 0, 0, 0};
   int                 x_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   zzz_StructMatrix   *A;
   zzz_StructVector   *b;
   zzz_StructVector   *x;

   void               *smg_data;
   int                 num_iterations;
   int                 time_index;

   int                 num_procs, myid;

   int                 nx, ny, nz;
   int                 P, Q, R;
   double              dx, dy, dz;
   int                 p, q, r;
   zzz_Index           ilower;
   zzz_Index           iupper;
   zzz_Box            *box;
                     
   zzz_Index           ilower_temp;
   zzz_Index           iupper_temp;
   zzz_Box            *box_temp;

   int                 dim = 3;
                     
   int                 offsets[4][3] = {{ 0,  0, -1},
                                        { 0, -1,  0},
                                        {-1,  0,  0},
                                        { 0,  0,  0}};
                     
   zzz_StructGrid     *grid;
   zzz_StructStencil  *stencil;
   zzz_Index          *stencil_shape;

   double             *values;
   int                *stencil_indices;

   int                 i, d;

   double              local_wall_time, wall_time;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   comm = zzz_TAlloc(MPI_Comm, 1);
   MPI_Comm_dup(MPI_COMM_WORLD, comm);
   MPI_Comm_size(*comm, &num_procs );
   MPI_Comm_rank(*comm, &myid );

#ifdef ZZZ_DEBUG
   cegdb(&argc, &argv, myid);
#endif

#ifdef ZZZ_DEBUG
   malloc_logpath = malloc_logpath_memory;
   sprintf(malloc_logpath, "malloc.log.%04d", myid);
#endif

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

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/
 
   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /* compute ilower and iupper from p,q,r and nx,ny,nz */
   zzz_IndexX(ilower) = nx*p;
   zzz_IndexX(iupper) = nx*(p+1) - 1;
   zzz_IndexY(ilower) = ny*q;
   zzz_IndexY(iupper) = ny*(q+1) - 1;
   zzz_IndexZ(ilower) = nz*r;
   zzz_IndexZ(iupper) = nz*(r+1) - 1;

   grid = zzz_NewStructGrid(comm, dim);
   zzz_SetStructGridExtents(grid, ilower, iupper);
   zzz_AssembleStructGrid(grid);

   /*-----------------------------------------------------------
    * Set up the stencil structure
    *-----------------------------------------------------------*/
 
   stencil_shape = zzz_CTAlloc(zzz_Index, 4);
   for (i = 0; i < 4; i++)
   {
      for (d = 0; d < dim; d++)
         zzz_IndexD(stencil_shape[i], d) = offsets[i][d];
   }
   stencil = zzz_NewStructStencil(dim, 4, stencil_shape);

   /*-----------------------------------------------------------
    * Set up the matrix structure
    *-----------------------------------------------------------*/
 
   A = zzz_NewStructMatrix(comm, grid, stencil);
   zzz_StructMatrixSymmetric(A) = 1;
   zzz_SetStructMatrixNumGhost(A, A_num_ghost);
   zzz_InitializeStructMatrix(A);

   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/

   stencil_indices = zzz_CTAlloc(int, 1);
   box = zzz_NewBox(ilower, iupper);
   values = zzz_CTAlloc(double, zzz_BoxVolume(box));

   /* Set each coefficient for the grid individually */

   stencil_indices[0] = 0;
   for (i=0; i < zzz_BoxVolume(box); i++)
      values[i] = -dz;
   zzz_SetStructMatrixBoxValues(A, box, 1, stencil_indices, values);

   stencil_indices[0] = 1;
   for (i=0; i < zzz_BoxVolume(box); i++)
      values[i] = -dy;
   zzz_SetStructMatrixBoxValues(A, box, 1, stencil_indices, values);

   stencil_indices[0] = 2;
   for (i=0; i < zzz_BoxVolume(box); i++)
      values[i] = -dx;
   zzz_SetStructMatrixBoxValues(A, box, 1, stencil_indices, values);

   stencil_indices[0] = 3;
   for (i=0; i < zzz_BoxVolume(box); i++)
      values[i] = 2.0*(dx+dy+dz);
   zzz_SetStructMatrixBoxValues(A, box, 1, stencil_indices, values);

   zzz_TFree(values);

   /* Zero out stencils reaching to real boundary */

   if( zzz_IndexZ(ilower) == 0 )
   {
       stencil_indices[0] = 0;
       zzz_CopyIndex(ilower, ilower_temp);
       zzz_CopyIndex(iupper, iupper_temp);
       zzz_IndexZ(iupper_temp) = 0;
       box_temp = zzz_NewBox(ilower_temp, iupper_temp);
       values = zzz_CTAlloc(double, zzz_BoxVolume(box_temp));
       zzz_SetStructMatrixBoxValues(A, box_temp, 1, stencil_indices, values);
       zzz_FreeBox(box_temp);
       zzz_TFree(values);
    }
   if( zzz_IndexY(ilower) == 0 )
   {
       stencil_indices[0] = 1;
       zzz_CopyIndex(ilower, ilower_temp);
       zzz_CopyIndex(iupper, iupper_temp);
       zzz_IndexY(iupper_temp) = 0;
       box_temp = zzz_NewBox(ilower_temp, iupper_temp);
       values = zzz_CTAlloc(double, zzz_BoxVolume(box_temp));
       zzz_SetStructMatrixBoxValues(A, box_temp, 1, stencil_indices, values);
       zzz_FreeBox(box_temp);
       zzz_TFree(values);
    }
   if( zzz_IndexX(ilower) == 0 )
   {
       stencil_indices[0] = 2;
       zzz_CopyIndex(ilower, ilower_temp);
       zzz_CopyIndex(iupper, iupper_temp);
       zzz_IndexX(iupper_temp) = 0;
       box_temp = zzz_NewBox(ilower_temp, iupper_temp);
       values = zzz_CTAlloc(double, zzz_BoxVolume(box_temp));
       zzz_SetStructMatrixBoxValues(A, box_temp, 1, stencil_indices, values);
       zzz_FreeBox(box_temp);
       zzz_TFree(values);
    }

    zzz_FreeBox(box);

   zzz_AssembleStructMatrix(A);

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   b = zzz_NewStructVector(comm, zzz_StructMatrixGrid(A));
   zzz_SetStructVectorNumGhost(b, b_num_ghost);
   zzz_InitializeStructVector(b);
   zzz_AssembleStructVector(b);
   zzz_SetStructVectorConstantValues(b, 1.0);

   x = zzz_NewStructVector(comm, zzz_StructMatrixGrid(A));
   zzz_SetStructVectorNumGhost(x, x_num_ghost);
   zzz_InitializeStructVector(x);
   zzz_AssembleStructVector(x);
   zzz_SetStructVectorConstantValues(x, 0.0);
 
   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   smg_data = zzz_SMGInitialize(comm);
   zzz_SMGSetMemoryUse(smg_data, 0);
   zzz_SMGSetMaxIter(smg_data, 50);
   zzz_SMGSetTol(smg_data, 1.0e-06);
   zzz_SMGSetNumPreRelax(smg_data, 1);
   zzz_SMGSetNumPostRelax(smg_data, 1);
   zzz_SMGSetLogging(smg_data, 0);
   zzz_SMGSetup(smg_data, A, b, x);

   time_index = zzz_InitializeTiming("Driver");
   zzz_BeginTiming(time_index);
   local_wall_time = - time_getWallclockSeconds();
   zzz_SMGSolve(smg_data, A, b, x);
   local_wall_time += time_getWallclockSeconds();
   zzz_EndTiming(time_index);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   zzz_PrintStructVector("zout_x", x, 0);

   zzz_SMGGetNumIterations(smg_data, &num_iterations);
   if (myid == 0)
   {
      printf("Iterations = %d\n", num_iterations);
   }

   MPI_Allreduce(&local_wall_time, &wall_time, 1,
                 MPI_DOUBLE, MPI_MAX, *comm);
   if (myid == 0)
   {
      printf("Time = %f seconds\n", wall_time);
   }

   zzz_PrintTiming(comm);
   
   zzz_SMGPrintLogging(smg_data, myid);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   zzz_FinalizeTiming(time_index);
   zzz_SMGFinalize(smg_data);
   zzz_FreeStructGrid(zzz_StructMatrixGrid(A));
   zzz_FreeStructMatrix(A);
   zzz_FreeStructVector(b);
   zzz_FreeStructVector(x);
   zzz_TFree(comm);

#ifdef ZZZ_DEBUG
   malloc_verify(0);
   malloc_shutdown();
#endif

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}


