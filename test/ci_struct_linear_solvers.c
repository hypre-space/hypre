#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "../CI_struct_matrix_vector/HYPRE_CI_struct_matrix_vector_types.h"  
#include "../CI_struct_matrix_vector/HYPRE_CI_struct_matrix_vector_protos.h"  
#include "../CI_struct_linear_solvers/HYPRE_CI_struct_linear_solvers_types.h"  
#include "../CI_struct_linear_solvers/HYPRE_CI_struct_linear_solvers_protos.h"  
 
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

   int                 A_num_ghost[6] = { 0, 0, 0, 0, 0, 0};

   int                 ierr;
                     
   HYPRE_StructInterfaceMatrix  A;
   HYPRE_StructInterfaceVector  b;
   HYPRE_StructInterfaceVector  x;

   HYPRE_StructInterfaceSolver  pilut_solver; 
   int                 num_iterations;
   int                 time_index;
   double              final_res_norm;

   int                 num_procs, myid;

   int                 p, q, r;
   int                 dim;
   int                 n_pre, n_post;
   int                 nblocks, volume;

   int               **iupper;
   int               **ilower;

   int                *istart;

   int               **offsets;

   HYPRE_StructGrid    grid;
   HYPRE_StructStencil stencil;

   int                *stencil_indices;
   double             *values;

   int                 i, s, d;
   int                 ix, iy, iz, ib;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

#ifdef HYPRE_USE_PTHREADS
   HYPRE_InitPthreads(MPI_COMM_WORLD);
#endif  

 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   /* Initialize Petsc */ /* In regular code should not be done here */
   PetscInitialize( NULL, NULL, NULL, NULL);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );


   hypre_InitMemoryDebug(myid);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   dim = 3;

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

   n_pre  = 1;
   n_post = 1;

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
      else if ( strcmp(argv[arg_index], "-v") == 0 )
      {
         arg_index++;
         n_pre = atoi(argv[arg_index++]);
         n_post = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-d") == 0 )
      {
         arg_index++;
         dim = atoi(argv[arg_index++]);
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
      printf("  -n <nx> <ny> <nz>    : problem size per block\n");
      printf("  -P <Px> <Py> <Pz>    : processor topology\n");
      printf("  -b <bx> <by> <bz>    : blocking per processor\n");
      printf("  -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("  -v <n_pre> <n_post>  : number of pre and post relaxations\n");
      printf("  -d <dim>             : problem dimension (2 or 3)\n");
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
      printf("  (nx, ny, nz)    = (%d, %d, %d)\n", nx, ny, nz);
      printf("  (Px, Py, Pz)    = (%d, %d, %d)\n", P,  Q,  R);
      printf("  (bx, by, bz)    = (%d, %d, %d)\n", bx, by, bz);
      printf("  (cx, cy, cz)    = (%f, %f, %f)\n", cx, cy, cz);
      printf("  (n_pre, n_post) = (%d, %d)\n", n_pre, n_post);
      printf("  dim             = %d\n", dim);
      printf("  solver ID       = %d\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   istart = hypre_CTAlloc(int, dim);

   switch (dim)
   {
      case 1:
         volume  = nx;
         nblocks = bx;
         istart[0] = -17;
         stencil_indices = hypre_CTAlloc(int, 2);
         offsets = hypre_CTAlloc(int*, 2);
         offsets[0] = hypre_CTAlloc(int, 1);
         offsets[0][0] = -1; 
         offsets[1] = hypre_CTAlloc(int, 1);
         offsets[1][0] = 0; 
         /* compute p from P and myid */
         p = myid % P;
         break;
      case 2:
         volume  = nx*ny;
         nblocks = bx*by;
         istart[0] = -17;
         istart[1] = 0;
         stencil_indices = hypre_CTAlloc(int, 3);
         offsets = hypre_CTAlloc(int*, 3);
         offsets[0] = hypre_CTAlloc(int, 2);
         offsets[0][0] = -1; 
         offsets[0][1] = 0; 
         offsets[1] = hypre_CTAlloc(int, 2);
         offsets[1][0] = 0; 
         offsets[1][1] = -1; 
         offsets[2] = hypre_CTAlloc(int, 2);
         offsets[2][0] = 0; 
         offsets[2][1] = 0; 
         /* compute p,q from P,Q and myid */
         p = myid % P;
         q = (( myid - p)/P) % Q;
         break;
      case 3:
         volume  = nx*ny*nz;
         nblocks = bx*by*bz;
         istart[0] = -17;
         istart[1] = 0;
         istart[2] = 32;
         stencil_indices = hypre_CTAlloc(int, 4);
         offsets = hypre_CTAlloc(int*, 4);
         offsets[0] = hypre_CTAlloc(int, 3);
         offsets[0][0] = -1; 
         offsets[0][1] = 0; 
         offsets[0][2] = 0; 
         offsets[1] = hypre_CTAlloc(int, 3);
         offsets[1][0] = 0; 
         offsets[1][1] = -1; 
         offsets[1][2] = 0; 
         offsets[2] = hypre_CTAlloc(int, 3);
         offsets[2][0] = 0; 
         offsets[2][1] = 0; 
         offsets[2][2] = -1; 
         offsets[3] = hypre_CTAlloc(int, 3);
         offsets[3][0] = 0; 
         offsets[3][1] = 0; 
         offsets[3][2] = 0; 
         /* compute p,q,r from P,Q,R and myid */
         p = myid % P;
         q = (( myid - p)/P) % Q;
         r = ( myid - p - P*q)/( P*Q );
         break;
   }

   ilower = hypre_CTAlloc(int*, nblocks);
   iupper = hypre_CTAlloc(int*, nblocks);
   for (i = 0; i < nblocks; i++)
   {
      ilower[i] = hypre_CTAlloc(int, dim);
      iupper[i] = hypre_CTAlloc(int, dim);
   }

   for (i = 0; i < dim; i++)
   {
      A_num_ghost[2*i] = 1;
      A_num_ghost[2*i + 1] = 1;
   }

   /* compute ilower and iupper from (p,q,r), (bx,by,bz), and (nx,ny,nz) */
   ib = 0;
   switch (dim)
   {
      case 1:
         for (ix = 0; ix < bx; ix++)
         {
            ilower[ib][0] = istart[0]+ nx*(bx*p+ix);
            iupper[ib][0] = istart[0]+ nx*(bx*p+ix+1) - 1;
            ib++;
         }
         break;
      case 2:
         for (iy = 0; iy < by; iy++)
            for (ix = 0; ix < bx; ix++)
            {
               ilower[ib][0] = istart[0]+ nx*(bx*p+ix);
               iupper[ib][0] = istart[0]+ nx*(bx*p+ix+1) - 1;
               ilower[ib][1] = istart[1]+ ny*(by*q+iy);
               iupper[ib][1] = istart[1]+ ny*(by*q+iy+1) - 1;
               ib++;
            }
         break;
      case 3:
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
         break;
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
 
   stencil = HYPRE_NewStructStencil(dim, dim + 1);
   for (s = 0; s < dim + 1; s++)
   {
      HYPRE_SetStructStencilElement(stencil, s, offsets[s]);
   }

   /*-----------------------------------------------------------
    * Set up the matrix structure
    *-----------------------------------------------------------*/
 
   A = HYPRE_NewStructInterfaceMatrix(MPI_COMM_WORLD, grid, stencil);
   HYPRE_SetStructInterfaceMatrixSymmetric(A, 1);  
   HYPRE_SetStructInterfaceMatrixNumGhost(A, A_num_ghost); 
   HYPRE_InitializeStructInterfaceMatrix(A); 

   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(double, (dim +1)*volume);

   /* Set the coefficients for the grid */
   for (i = 0; i < (dim + 1)*volume; i += (dim + 1))
   {
      for (s = 0; s < (dim + 1); s++)
      {
         stencil_indices[s] = s;
         switch (dim)
         {
            case 1:
               values[i  ] = -cx;
               values[i+1] = 2.0*(cx);
               break;
            case 2:
               values[i  ] = -cx;
               values[i+1] = -cy;
               values[i+2] = 2.0*(cx+cy);
               break;
            case 3:
               values[i  ] = -cx;
               values[i+1] = -cy;
               values[i+2] = -cz;
               values[i+3] = 2.0*(cx+cy+cz);
               break;
         }
      }
   }
   for (ib = 0; ib < nblocks; ib++)
   {
      HYPRE_SetStructInterfaceMatrixBoxValues(A, ilower[ib], iupper[ib], 
(dim+1),
                                     stencil_indices, values);   
   }

   /* Zero out stencils reaching to real boundary */
   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   for (d = 0; d < dim; d++)
   {
      for (ib = 0; ib < nblocks; ib++)
      {
         if( ilower[ib][d] == istart[d] )
         {
            i = iupper[ib][d];
            iupper[ib][d] = istart[d];
            stencil_indices[0] = d;
            HYPRE_SetStructInterfaceMatrixBoxValues(A, ilower[ib], iupper[ib],
                                           1, stencil_indices, values); 
            iupper[ib][d] = i;
         }
      }
   }

   HYPRE_AssembleStructInterfaceMatrix(A);
#if 0
   HYPRE_PrintStructInterfaceMatrix("driver.out.A", A, 0);
#endif

   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(double, volume);

   b = HYPRE_NewStructInterfaceVector(MPI_COMM_WORLD, grid, stencil);
   HYPRE_InitializeStructInterfaceVector(b);  
   for (i = 0; i < volume; i++)
   {
      values[i] = 1.0;
   }
   for (ib = 0; ib < nblocks; ib++)
   {
      HYPRE_SetStructInterfaceVectorBoxValues(b, ilower[ib], iupper[ib], 
values); /***** ? *****/
   }
   HYPRE_AssembleStructInterfaceVector(b);
#if 0
   HYPRE_PrintStructInterfaceVector("driver.out.b", b, 0);
#endif

   x = HYPRE_NewStructInterfaceVector(MPI_COMM_WORLD, grid, stencil);
   HYPRE_InitializeStructInterfaceVector(x);
   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   for (ib = 0; ib < nblocks; ib++)
   {
      HYPRE_SetStructInterfaceVectorBoxValues(x, ilower[ib], iupper[ib], 
values); /***** ? *****/
   }
   HYPRE_AssembleStructInterfaceVector(x);
#if 0
   HYPRE_PrintStructInterfaceVector("driver.out.x0", x, 0);
#endif
 
   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Solve the system using PETSc and pilut
    *-----------------------------------------------------------*/

   if (solver_id == 2)
   {
      time_index = hypre_InitializeTiming("Petsc/pilut Setup");
      hypre_BeginTiming(time_index);

      pilut_solver = HYPRE_NewStructInterfaceSolver(MPI_COMM_WORLD, grid, stencil);

      ierr = HYPRE_StructInterfaceSolverInitialize( pilut_solver );
      if (ierr) 
	{
	  printf("Solver Initialize failed, halting.\n");
          return(0);
        }

      /* Solver parameters */
      /* For pilut preconditioner
      HYPRE_StructInterfaceSolverSetDropToleranc( pilut_solver, 0.0001 ):
      HYPRE_StructInterfaceSolverSetFactorRowSize( pilut_solver, 20 );
      */

      ierr = HYPRE_StructInterfaceSolverSetup(pilut_solver, A, x, b);
      if (ierr) 
	{
	  printf("Solver Setup failed, halting.\n");
          return(0);
        }

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("Petsc/Pilut Solve");
      hypre_BeginTiming(time_index);

      ierr = HYPRE_StructInterfaceSolverSolve(pilut_solver );
      if (ierr < 0) 
	{
	  printf("Solver Solve-phase failed, halting.\n");
          return(0);
        }

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      HYPRE_FreeStructInterfaceSolver(pilut_solver); 
   } else
   {
      printf("You have asked for a solver that is not supported.\n");
   }


   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_FreeStructGrid(grid);
   HYPRE_FreeStructInterfaceMatrix(A);
   HYPRE_FreeStructInterfaceVector(b);
   HYPRE_FreeStructInterfaceVector(x);

   for (i = 0; i < nblocks; i++)
   {
      hypre_TFree(iupper[i]);
      hypre_TFree(ilower[i]);
   }
   hypre_TFree(ilower);
   hypre_TFree(iupper);
   hypre_TFree(stencil_indices);
   hypre_TFree(istart);

   for ( i = 0; i < (dim + 1); i++)
      hypre_TFree(offsets[i]);
   hypre_TFree(offsets);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   MPI_Finalize();

#ifdef HYPRE_USE_PTHREADS
   HYPRE_DestroyPthreads();
#endif  

   return (0);
}






