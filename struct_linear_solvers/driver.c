
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../utilities/hypre_utilities.h"
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
   int                 bx, by, bz;
   double              cx, cy, cz;
   int                 solver_id;

   int                 A_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   HYPRE_StructMatrix  A;
   HYPRE_StructVector  b;
   HYPRE_StructVector  x;

   HYPRE_StructSolver  smg_solver;
   HYPRE_StructSolver  pcg_solver;
   HYPRE_StructSolver  pcg_precond;
   int                 num_iterations;
   int                 time_index;

   int                 num_procs, myid;

   int                 p, q, r;
   int                 ilower[50][3];
   int                 iupper[50][3];
   int                 istart[3] = {-17, 0, 32};
   int                 nblocks, volume;

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
   int                 ix, iy, iz, ib;

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

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = num_procs;
   Q  = 1;
   R  = 1;

   bx = 1;
   by = 1;
   bz = 1;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

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
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         bx = atoi(argv[arg_index++]);
         by = atoi(argv[arg_index++]);
         bz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
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
      printf("  -b <bx> <by> <bz>    : blocking per processor\n");
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

   if (bx*by*bz > 50)
   {
      printf("Error: Maximum number of blocks allowed per processor is 50\n");
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
      printf("  (bx, by, bz) = (%d, %d, %d)\n", bx, by, bz);
      printf("  (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      printf("  solver ID    = %d\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   volume  = nx*ny*nz;
   nblocks = bx*by*bz;

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /* compute ilower and iupper from (p,q,r), (bx,by,bz), and (nx,ny,nz) */
   ib = 0;
   for (iz = 0; iz < bz; iz++)
      for (iy = 0; iy < by; iy++)
         for (ix = 0; ix < bx; ix++)
         {
            ilower[ib][0] = istart[0]+ nx*(bx*p+ix);
            iupper[ib][0] = istart[0]+ nx*(bx*p+ix+1) - 1;
            ilower[ib][1] = istart[1]+ ny*(by*q+iy);
            iupper[ib][1] = istart[1]+ ny*(by*q+iy+1) - 1;
            ilower[ib][2] = istart[2]+ nz*(bz*r+iz);
            iupper[ib][2] = istart[2]+ nz*(bz*r+iz+1) - 1;
            ib++;
         }

   grid = HYPRE_NewStructGrid(MPI_COMM_WORLD, dim);
   for (ib = 0; ib < nblocks; ib++)
   {
      HYPRE_SetStructGridExtents(grid, ilower[ib], iupper[ib]);
   }
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
         values[i  ] = -cx;
         values[i+1] = -cy;
         values[i+2] = -cz;
         values[i+3] = 2.0*(cx+cy+cz);
      }
   }
   for (ib = 0; ib < nblocks; ib++)
   {
      HYPRE_SetStructMatrixBoxValues(A, ilower[ib], iupper[ib], 4,
                                     stencil_indices, values);
   }

   /* Zero out stencils reaching to real boundary */
   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   for (d = 0; d < 3; d++)
   {
      for (ib = 0; ib < nblocks; ib++)
      {
         if( ilower[ib][d] == istart[d] )
         {
            i = iupper[ib][d];
            iupper[ib][d] = istart[d];
            stencil_indices[0] = d;
            HYPRE_SetStructMatrixBoxValues(A, ilower[ib], iupper[ib],
                                           1, stencil_indices, values);
            iupper[ib][d] = i;
         }
      }
   }

   HYPRE_AssembleStructMatrix(A);
#if 0
   HYPRE_PrintStructMatrix("driver.out.A", A, 0);
#endif

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
   for (ib = 0; ib < nblocks; ib++)
   {
      HYPRE_SetStructVectorBoxValues(b, ilower[ib], iupper[ib], values);
   }
   HYPRE_AssembleStructVector(b);
#if 0
   HYPRE_PrintStructVector("driver.out.b", b, 0);
#endif

   x = HYPRE_NewStructVector(MPI_COMM_WORLD, grid, stencil);
   HYPRE_InitializeStructVector(x);
   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   for (ib = 0; ib < nblocks; ib++)
   {
      HYPRE_SetStructVectorBoxValues(x, ilower[ib], iupper[ib], values);
   }
   HYPRE_AssembleStructVector(x);
#if 0
   HYPRE_PrintStructVector("driver.out.x0", x, 0);
#endif
 
   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Solve the system using SMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      time_index = hypre_InitializeTiming("SMG Setup");
      hypre_BeginTiming(time_index);

      smg_solver = HYPRE_StructSMGInitialize(MPI_COMM_WORLD);
      HYPRE_StructSMGSetMemoryUse(smg_solver, 0);
      HYPRE_StructSMGSetMaxIter(smg_solver, 50);
      HYPRE_StructSMGSetRelChange(smg_solver, 0);
      HYPRE_StructSMGSetTol(smg_solver, 1.0e-06);
      HYPRE_StructSMGSetNumPreRelax(smg_solver, 1);
      HYPRE_StructSMGSetNumPostRelax(smg_solver, 1);
      HYPRE_StructSMGSetLogging(smg_solver, 0);
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
   
      HYPRE_StructSMGGetNumIterations(smg_solver, &num_iterations);
      HYPRE_StructSMGFinalize(smg_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   if (solver_id > 0)
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      pcg_solver = HYPRE_StructPCGInitialize(MPI_COMM_WORLD);
      HYPRE_StructPCGSetMaxIter(pcg_solver, 50);
      HYPRE_StructPCGSetTol(pcg_solver, 1.0e-06);
      HYPRE_StructPCGSetTwoNorm(pcg_solver, 1);
      HYPRE_StructPCGSetRelChange(pcg_solver, 0);
      HYPRE_StructPCGSetLogging(pcg_solver, 0);

      if (solver_id == 1)
      {
         /* use symmetric SMG as preconditioner */
         pcg_precond = HYPRE_StructSMGInitialize(MPI_COMM_WORLD);
         HYPRE_StructSMGSetMemoryUse(pcg_precond, 0);
         HYPRE_StructSMGSetMaxIter(pcg_precond, 1);
         HYPRE_StructSMGSetTol(pcg_precond, 0.0);
         HYPRE_StructSMGSetNumPreRelax(pcg_precond, 1);
         HYPRE_StructSMGSetNumPostRelax(pcg_precond, 1);
         HYPRE_StructSMGSetLogging(pcg_precond, 0);
         HYPRE_StructPCGSetPrecond(pcg_solver,
                                   HYPRE_StructSMGSolve,
                                   HYPRE_StructSMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 2)
      {
         /* use diagonal scaling as preconditioner */
         pcg_precond = NULL;
         HYPRE_StructPCGSetPrecond(pcg_solver,
                                   HYPRE_StructDiagScale,
                                   HYPRE_StructDiagScaleSetup,
                                   pcg_precond);
      }

      HYPRE_StructPCGSetup(pcg_solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructPCGSolve(pcg_solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructPCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_StructPCGFinalize(pcg_solver);

      if (solver_id == 1)
      {
         HYPRE_StructSMGFinalize(pcg_precond);
      }
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

#if 0
   HYPRE_PrintStructVector("driver.out.x", x, 0);
#endif

   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("\n");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_FreeStructGrid(grid);
   HYPRE_FreeStructMatrix(A);
   HYPRE_FreeStructVector(b);
   HYPRE_FreeStructVector(x);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}


