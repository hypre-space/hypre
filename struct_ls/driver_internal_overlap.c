
#include "headers.h"
 
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
   int                   A_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
   int                   b_num_ghost[6] = { 0, 0, 0, 0, 0, 0};
   int                   x_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   hypre_StructGrid     *grid;
   hypre_StructMatrix   *A;
   hypre_StructVector   *b;
   hypre_StructVector   *x;

   hypre_StructGrid     *overlapped_grid;
   hypre_StructVector   *overlapped_x;
   hypre_CommPkg        *comm_pkg;

   void                 *smg_data;
   int                   num_iterations;
   int                   time_index;
                       
   int                   num_procs, myid;
                       
   int                   nx, ny, nz;
   int                   P, Q, R;
   double                dx, dy, dz;
   int                   p, q, r;
   hypre_Index           ilower;
   hypre_Index           iupper;
   hypre_Box            *box;
                     
   hypre_Index           ilower_temp;
   hypre_Index           iupper_temp;
   hypre_Box            *box_temp;

   hypre_Index           global_ilower;
   hypre_Index           global_iupper;

   int                   dim = 3;
                       
   int                   offsets[4][3] = {{ 0,  0, -1},
                                          { 0, -1,  0},
                                          {-1,  0,  0},
                                          { 0,  0,  0}};
                     
   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;

   double               *values;
   int                  *stencil_indices;
                       
   int                   i, d;

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

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/
 
   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /* compute ilower and iupper from p,q,r and nx,ny,nz */
   hypre_IndexX(ilower) = nx*p;
   hypre_IndexX(iupper) = nx*(p+1) - 1;
   hypre_IndexY(ilower) = ny*q;
   hypre_IndexY(iupper) = ny*(q+1) - 1;
   hypre_IndexZ(ilower) = nz*r;
   hypre_IndexZ(iupper) = nz*(r+1) - 1;

   /* compute global_ilower and global_iupper */
   hypre_IndexX(global_ilower) = 0;
   hypre_IndexX(global_iupper) = nx*P - 1;
   hypre_IndexY(global_ilower) = 0;
   hypre_IndexY(global_iupper) = ny*Q - 1;
   hypre_IndexZ(global_ilower) = 0;
   hypre_IndexZ(global_iupper) = nz*R - 1;

   grid = hypre_NewStructGrid(MPI_COMM_WORLD, dim);
   hypre_SetStructGridExtents(grid, ilower, iupper);
   hypre_AssembleStructGrid(grid);

   /*-----------------------------------------------------------
    * Set up an overlapped grid structure
    *-----------------------------------------------------------*/
 
   /* use overlapped extents to set up the problem */
   for (d = 0; d < 3; d++)
   {
      if (hypre_IndexD(ilower, d) != hypre_IndexD(global_ilower, d))
         hypre_IndexD(ilower, d)--;
      if (hypre_IndexD(iupper, d) != hypre_IndexD(global_iupper, d))
         hypre_IndexD(iupper, d)++;
   }

   /* create an overlapped grid */
   overlapped_grid = hypre_NewStructGrid(MPI_COMM_WORLD, dim);
   hypre_SetStructGridExtents(overlapped_grid, ilower, iupper);
   hypre_AssembleStructGrid(overlapped_grid);

   /*-----------------------------------------------------------
    * Set up the stencil structure
    *-----------------------------------------------------------*/
 
   stencil_shape = hypre_CTAlloc(hypre_Index, 4);
   for (i = 0; i < 4; i++)
   {
      for (d = 0; d < dim; d++)
         hypre_IndexD(stencil_shape[i], d) = offsets[i][d];
   }
   stencil = hypre_NewStructStencil(dim, 4, stencil_shape);

   /*-----------------------------------------------------------
    * Set up the matrix structure
    *-----------------------------------------------------------*/
 
   A = hypre_NewStructMatrix(MPI_COMM_WORLD, grid, stencil);
   hypre_StructMatrixSymmetric(A) = 1;
   hypre_SetStructMatrixNumGhost(A, A_num_ghost);
   hypre_InitializeStructMatrix(A);

   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/

   stencil_indices = hypre_CTAlloc(int, 1);
   box = hypre_NewBox(ilower, iupper);
   values = hypre_CTAlloc(double, hypre_BoxVolume(box));

   /* Set each coefficient for the grid individually */

   stencil_indices[0] = 0;
   for (i=0; i < hypre_BoxVolume(box); i++)
      values[i] = -dz;
   hypre_SetStructMatrixBoxValues(A, box, 1, stencil_indices, values);

   stencil_indices[0] = 1;
   for (i=0; i < hypre_BoxVolume(box); i++)
      values[i] = -dy;
   hypre_SetStructMatrixBoxValues(A, box, 1, stencil_indices, values);

   stencil_indices[0] = 2;
   for (i=0; i < hypre_BoxVolume(box); i++)
      values[i] = -dx;
   hypre_SetStructMatrixBoxValues(A, box, 1, stencil_indices, values);

   stencil_indices[0] = 3;
   for (i=0; i < hypre_BoxVolume(box); i++)
      values[i] = 2.0*(dx+dy+dz);
   hypre_SetStructMatrixBoxValues(A, box, 1, stencil_indices, values);

   hypre_TFree(values);

   /* Zero out stencils reaching to real boundary */

   if( hypre_IndexZ(ilower) == hypre_IndexZ(global_ilower) )
   {
      stencil_indices[0] = 0;
      hypre_CopyIndex(ilower, ilower_temp);
      hypre_CopyIndex(iupper, iupper_temp);
      hypre_IndexZ(iupper_temp) = 0;
      box_temp = hypre_NewBox(ilower_temp, iupper_temp);
      values = hypre_CTAlloc(double, hypre_BoxVolume(box_temp));
      hypre_SetStructMatrixBoxValues(A, box_temp, 1, stencil_indices, values);
      hypre_FreeBox(box_temp);
      hypre_TFree(values);
   }
   if( hypre_IndexY(ilower) == hypre_IndexY(global_ilower) )
   {
      stencil_indices[0] = 1;
      hypre_CopyIndex(ilower, ilower_temp);
      hypre_CopyIndex(iupper, iupper_temp);
      hypre_IndexY(iupper_temp) = 0;
      box_temp = hypre_NewBox(ilower_temp, iupper_temp);
      values = hypre_CTAlloc(double, hypre_BoxVolume(box_temp));
      hypre_SetStructMatrixBoxValues(A, box_temp, 1, stencil_indices, values);
      hypre_FreeBox(box_temp);
      hypre_TFree(values);
   }
   if( hypre_IndexX(ilower) == hypre_IndexX(global_ilower) )
   {
      stencil_indices[0] = 2;
      hypre_CopyIndex(ilower, ilower_temp);
      hypre_CopyIndex(iupper, iupper_temp);
      hypre_IndexX(iupper_temp) = 0;
      box_temp = hypre_NewBox(ilower_temp, iupper_temp);
      values = hypre_CTAlloc(double, hypre_BoxVolume(box_temp));
      hypre_SetStructMatrixBoxValues(A, box_temp, 1, stencil_indices, values);
      hypre_FreeBox(box_temp);
      hypre_TFree(values);
   }

   hypre_FreeBox(box);
   hypre_TFree(stencil_indices);

   hypre_AssembleStructMatrix(A);

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   b = hypre_NewStructVector(MPI_COMM_WORLD, grid);
   hypre_SetStructVectorNumGhost(b, b_num_ghost);
   hypre_InitializeStructVector(b);
   hypre_AssembleStructVector(b);
   hypre_SetStructVectorConstantValues(b, 1.0);

   x = hypre_NewStructVector(MPI_COMM_WORLD, grid);
   hypre_SetStructVectorNumGhost(x, x_num_ghost);
   hypre_InitializeStructVector(x);
   hypre_AssembleStructVector(x);
   hypre_SetStructVectorConstantValues(x, 0.0);
 
   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("SMG Setup");
   hypre_BeginTiming(time_index);

   smg_data = hypre_SMGInitialize(MPI_COMM_WORLD);
   hypre_SMGSetMemoryUse(smg_data, 0);
   hypre_SMGSetMaxIter(smg_data, 10);
   hypre_SMGSetTol(smg_data, 1.0e-06);
   hypre_SMGSetNumPreRelax(smg_data, 1);
   hypre_SMGSetNumPostRelax(smg_data, 1);
   hypre_SMGSetLogging(smg_data, 0);
   hypre_SMGSetup(smg_data, A, b, x);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   time_index = hypre_InitializeTiming("SMG Solve");
   hypre_BeginTiming(time_index);

   hypre_SMGSolve(smg_data, A, b, x);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   /*-----------------------------------------------------------
    * Put the solution on the overlapped grid
    *-----------------------------------------------------------*/

   overlapped_x = hypre_NewStructVector(MPI_COMM_WORLD, overlapped_grid);
   hypre_SetStructVectorNumGhost(overlapped_x, b_num_ghost);
   hypre_InitializeStructVector(overlapped_x);
   hypre_AssembleStructVector(overlapped_x);
   comm_pkg = hypre_GetMigrateStructVectorCommPkg(x, overlapped_x);
   hypre_MigrateStructVector(comm_pkg, x, overlapped_x);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   hypre_PrintStructVector("zout_x", x, 0);
   hypre_PrintStructVector("zout_overlapped_x", overlapped_x, 0);

   hypre_SMGGetNumIterations(smg_data, &num_iterations);
   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("\n");
   }

   hypre_SMGPrintLogging(smg_data, myid);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_SMGFinalize(smg_data);

   hypre_FreeStructGrid(grid);
   hypre_FreeStructMatrix(A);
   hypre_FreeStructVector(b);
   hypre_FreeStructVector(x);

   hypre_FreeStructGrid(overlapped_grid);
   hypre_FreeStructVector(overlapped_x);
   hypre_FreeCommPkg(comm_pkg);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}


