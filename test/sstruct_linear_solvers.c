#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE_sstruct_ls.h"
 
#define DEBUG 0

#if DEBUG
#include "sstruct_matrix_vector.h"
#endif

#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface (semi-structured storage)
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 * Standard 7-point laplacian in 3D with grid and anisotropy determined
 * as command line arguments.  Do `driver -help' for usage info.
 *----------------------------------------------------------------------*/

int
main( int   argc,
      char *argv[] )
{
   int                   arg_index;
   int                   print_usage;
   int                   nx, ny, nz;
   int                   P, Q, R;
   int                   bx, by, bz;
   int                   solver_id;
   HYPRE_SStructVariable vtypes[2] = {HYPRE_SSTRUCT_VARIABLE_CELL,
                                      HYPRE_SSTRUCT_VARIABLE_NODE};
                        
   HYPRE_SStructMatrix  A;
   HYPRE_SStructVector  b;
   HYPRE_SStructVector  x;

   HYPRE_SStructSolver  solver;
   HYPRE_SStructSolver  precond;
   int                  num_iterations;
   int                  time_index;
   double               final_res_norm;
                        
   int                  num_procs, myid;
                        
   int                  p, q, r;
   int                  dim;
   int                  nblocks;

   int                  Cvolume, Nvolume;

   int                **Cilower;
   int                **Ciupper;
   int                **Nilower;
   int                **Niupper;

   int                 *index0, index0_mem[3];
   int                 *index1, index1_mem[3];
   int                  Cistart[3];

   int                  offsets[7][3] = {{ 0, 0, 0},
                                         {-1, 0, 0},
                                         { 1, 0, 0},
                                         { 0,-1, 0},
                                         { 0, 1, 0},
                                         { 0, 0,-1},
                                         { 0, 0, 1}};
   int                  CNoffsets[8][3] = {{-1,-1,-1},
                                           { 0,-1,-1},
                                           {-1, 0,-1},
                                           { 0, 0,-1},
                                           {-1, 0, 0},
                                           { 0,-1, 0},
                                           {-1, 0, 0},
                                           { 0,-1, 0}};
   int                  NCoffsets[8][3] = {{ 0, 0, 0},
                                           { 1, 0, 0},
                                           { 0, 1, 0},
                                           { 1, 1, 0},
                                           { 0, 0, 1},
                                           { 1, 0, 1},
                                           { 0, 1, 1},
                                           { 1, 1, 1}};

   HYPRE_SStructGrid    grid;
   HYPRE_SStructGraph   graph;
   HYPRE_SStructStencil Cstencil,             Nstencil;
   int                  Cstencil_size,        Nstencil_size;
   int                  CCstencil_size,       NNstencil_size;
   int                  CNstencil_size,       NCstencil_size;
   int                  CCstencil_indexes[7], NNstencil_indexes[7];
   int                  CNstencil_indexes[8], NCstencil_indexes[8];
   double              *CCvalues,            *NNvalues;
   double              *CNvalues,            *NCvalues;
   int                  CAstencil_index;
   double              *CAvalues;
                       
   int                  i, j, k, s, d, part;
   int                  ib, jb, kb, block;
   int                  ix, iy, iz;
                       
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

   solver_id = 39;

   Cistart[0] = 1;
   Cistart[1] = 1;
   Cistart[2] = 1;

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
      printf("  -d <dim>             : problem dimension (2 or 3)\n");
      printf("  -solver <ID>         : solver ID (default = 0)\n");
      printf("                         38 - GMRES with diagonal scaling\n");
      printf("                         39 - GMRES\n");
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
      printf("  dim             = %d\n", dim);
      printf("  solver ID       = %d\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   MPI_Barrier(MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("SStruct Interface");
   hypre_BeginTiming(time_index);

   switch (dim)
   {
      case 1:
         Cvolume = nx;
         Nvolume = (nx+1);
         CCstencil_size = 3;
         CNstencil_size = 2;
         NNstencil_size = 3;
         NCstencil_size = 2;
         nblocks = bx;
         break;
      case 2:
         Cvolume = nx*ny;
         Nvolume = (nx+1)*(ny+1);
         CCstencil_size = 5;
         CNstencil_size = 4;
         NNstencil_size = 5;
         NCstencil_size = 4;
         nblocks = bx*by;
         break;
      case 3:
         Cvolume = nx*ny*nz;
         Nvolume = (nx+1)*(ny+1)*(nz+1);
         CCstencil_size = 7;
         CNstencil_size = 8;
         NNstencil_size = 7;
         NCstencil_size = 8;
         nblocks = bx*by*bz;
         break;
   }

   Cilower = hypre_CTAlloc(int*, nblocks);
   Ciupper = hypre_CTAlloc(int*, nblocks);
   Nilower = hypre_CTAlloc(int*, nblocks);
   Niupper = hypre_CTAlloc(int*, nblocks);
   for (i = 0; i < nblocks; i++)
   {
      Cilower[i] = hypre_CTAlloc(int, dim);
      Ciupper[i] = hypre_CTAlloc(int, dim);
      Nilower[i] = hypre_CTAlloc(int, dim);
      Niupper[i] = hypre_CTAlloc(int, dim);
   }

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /* compute ilower and iupper from (p,q,r), (bx,by,bz), and (nx,ny,nz) */
   block = 0;
   switch (dim)
   {
      case 1:
         for (ix = 0; ix < bx; ix++)
         {
            Cilower[block][0] = Cistart[0]+ nx*(bx*p+ix);
            Ciupper[block][0] = Cistart[0]+ nx*(bx*p+ix+1) - 1;
            Nilower[block][0] = Cilower[block][0] - 1;
            Niupper[block][0] = Ciupper[block][0];
            block++;
         }
         break;
      case 2:
         for (iy = 0; iy < by; iy++)
            for (ix = 0; ix < bx; ix++)
            {
               Cilower[block][0] = Cistart[0]+ nx*(bx*p+ix);
               Ciupper[block][0] = Cistart[0]+ nx*(bx*p+ix+1) - 1;
               Cilower[block][1] = Cistart[1]+ ny*(by*q+iy);
               Ciupper[block][1] = Cistart[1]+ ny*(by*q+iy+1) - 1;
               Nilower[block][0] = Cilower[block][0] - 1;
               Niupper[block][0] = Ciupper[block][0];
               Nilower[block][1] = Cilower[block][1] - 1;
               Niupper[block][1] = Ciupper[block][1];
               block++;
            }
         break;
      case 3:
         for (iz = 0; iz < bz; iz++)
            for (iy = 0; iy < by; iy++)
               for (ix = 0; ix < bx; ix++)
               {
                  Cilower[block][0] = Cistart[0]+ nx*(bx*p+ix);
                  Ciupper[block][0] = Cistart[0]+ nx*(bx*p+ix+1) - 1;
                  Cilower[block][1] = Cistart[1]+ ny*(by*q+iy);
                  Ciupper[block][1] = Cistart[1]+ ny*(by*q+iy+1) - 1;
                  Cilower[block][2] = Cistart[2]+ nz*(bz*r+iz);
                  Ciupper[block][2] = Cistart[2]+ nz*(bz*r+iz+1) - 1;
                  Nilower[block][0] = Cilower[block][0] - 1;
                  Niupper[block][0] = Ciupper[block][0];
                  Nilower[block][1] = Cilower[block][1] - 1;
                  Niupper[block][1] = Ciupper[block][1];
                  Nilower[block][2] = Cilower[block][2] - 1;
                  Niupper[block][2] = Ciupper[block][2];
                  block++;
               }
         break;
   } 

   HYPRE_SStructGridCreate(MPI_COMM_WORLD, dim, 2, &grid);
   for (part = 0; part < 2; part++)
   {
      for (block = 0; block < nblocks; block++)
      {
         HYPRE_SStructGridSetExtents(grid, part,
                                     Cilower[block], Ciupper[block]);
      }
      HYPRE_SStructGridSetVariables(grid, part, 2, vtypes);
   }
   HYPRE_SStructGridAssemble(grid);

   /*-----------------------------------------------------------
    * Set up the stencil structure for cell-centered variables
    *-----------------------------------------------------------*/
 
   Cstencil_size = CCstencil_size + CNstencil_size;
   HYPRE_SStructStencilCreate(dim, Cstencil_size, &Cstencil);
   s = 0;
   for (i = 0; i < CCstencil_size; i++)
   {
      HYPRE_SStructStencilSetEntry(Cstencil, s, offsets[i], 0);
      CCstencil_indexes[i] = s;
      s++;
   }
   for (i = 0; i < CNstencil_size; i++)
   {
      HYPRE_SStructStencilSetEntry(Cstencil, s, CNoffsets[i], 1);
      CNstencil_indexes[i] = s;
      s++;
   }
   CAstencil_index = s;

   /*-----------------------------------------------------------
    * Set up the stencil structure for node-centered variables
    *-----------------------------------------------------------*/
 
   Nstencil_size = NNstencil_size + NCstencil_size;
   HYPRE_SStructStencilCreate(dim, Nstencil_size, &Nstencil);
   s = 0;
   for (i = 0; i < NNstencil_size; i++)
   {
      HYPRE_SStructStencilSetEntry(Nstencil, s, offsets[i], 1);
      NNstencil_indexes[i] = s;
      s++;
   }
   for (i = 0; i < NCstencil_size; i++)
   {
      HYPRE_SStructStencilSetEntry(Nstencil, s, NCoffsets[i], 0);
      NCstencil_indexes[i] = s;
      s++;
   }

   /*-----------------------------------------------------------
    * Set up the graph structure
    *-----------------------------------------------------------*/
 
   HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);
   for (part = 0; part < 2; part++)
   {
      HYPRE_SStructGraphSetStencil(graph, part, 0, Cstencil);
      HYPRE_SStructGraphSetStencil(graph, part, 1, Nstencil);
   }

   /* Add entries */
   index0 = index0_mem;
   index1 = index1_mem;
   for (kb = 0; kb < bz; kb++)
   {
      for (jb = 0; jb < by; jb++)
      {
         block = jb*bx + kb*bx*by;

         switch (dim)
         {
            case 1:
            index0[0] = Ciupper[block + bx - 1][0];
            index1[0] = Cilower[block][0];
            
            /* glue part 0 to part 1 */
            HYPRE_SStructGraphAddEntries(graph, 0, index0, 0,
                                         1, 1, &index1, 0);
            /* glue part 1 to part 0 */
            HYPRE_SStructGraphAddEntries(graph, 1, index1, 0,
                                         1, 0, &index0, 0);
            break;

            case 2:
            for (iy = Cilower[block][1]; iy <= Ciupper[block][1]; iy++)
            {
               index0[0] = Ciupper[block + bx - 1][0];
               index1[0] = Cilower[block][0];
               index0[1] = iy;
               index1[1] = iy;
               
               /* glue part 0 to part 1 */
               HYPRE_SStructGraphAddEntries(graph, 0, index0, 0,
                                            1, 1, &index1, 0);
               /* glue part 1 to part 0 */
               HYPRE_SStructGraphAddEntries(graph, 1, index1, 0,
                                            1, 0, &index0, 0);
            }
            break;

            case 3:
            for (iz = Cilower[block][2]; iz <= Ciupper[block][2]; iz++)
            {
               for (iy = Cilower[block][1]; iy <= Ciupper[block][1]; iy++)
               {
                  index0[0] = Ciupper[block + bx - 1][0];
                  index1[0] = Cilower[block][0];
                  index0[1] = iy;
                  index1[1] = iy;
                  index0[2] = iz;
                  index1[2] = iz;

                  /* glue part 0 to part 1 */
                  HYPRE_SStructGraphAddEntries(graph, 0, index0, 0,
                                               1, 1, &index1, 0);
                  /* glue part 1 to part 0 */
                  HYPRE_SStructGraphAddEntries(graph, 1, index1, 0,
                                               1, 0, &index0, 0);
               }
            }
            break;
         }
      }
   }

   HYPRE_SStructGraphAssemble(graph);

   /*-----------------------------------------------------------
    * Set up the matrix structure
    *-----------------------------------------------------------*/
 
   HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);
   /* TODO HYPRE_SStructMatrixSetSymmetric(A, 1); */
   HYPRE_SStructMatrixInitialize(A);

   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/

   CCvalues = hypre_CTAlloc(double, CCstencil_size*Cvolume);
   CNvalues = hypre_CTAlloc(double, CNstencil_size*Cvolume);
   NNvalues = hypre_CTAlloc(double, NNstencil_size*Nvolume);
   NCvalues = hypre_CTAlloc(double, NCstencil_size*Nvolume);

   for (i = 0; i < Cvolume; i++)
   {
      k = i*CCstencil_size;
      CCvalues[k++] = Cstencil_size;
      for (j = 1; j < CCstencil_size; j++)
      {
         CCvalues[k++] = -1.0;
      }
      k = i*CNstencil_size;
      for (j = 0; j < CNstencil_size; j++)
      {
         CNvalues[k++] = -1.0;
      }
   }

   for (i = 0; i < Nvolume; i++)
   {
      k = i*NNstencil_size;
      NNvalues[k++] = Nstencil_size;
      for (j = 1; j < NNstencil_size; j++)
      {
         NNvalues[k++] = -1.0;
      }
      k = i*NCstencil_size;
      for (j = 0; j < NCstencil_size; j++)
      {
         NCvalues[k++] = -1.0;
      }
   }

   for (part = 0; part < 2; part++)
   {
      for (block = 0; block < nblocks; block++)
      {
#if 1
         HYPRE_SStructMatrixSetBoxValues(A, part,
                                         Cilower[block], Ciupper[block], 0,
                                         CCstencil_size, CCstencil_indexes,
                                         CCvalues);
         HYPRE_SStructMatrixSetBoxValues(A, part,
                                         Cilower[block], Ciupper[block], 0,
                                         CNstencil_size, CNstencil_indexes,
                                         CNvalues);
         HYPRE_SStructMatrixSetBoxValues(A, part,
                                         Nilower[block], Niupper[block], 1,
                                         NNstencil_size, NNstencil_indexes,
                                         NNvalues);
         HYPRE_SStructMatrixSetBoxValues(A, part,
                                         Nilower[block], Niupper[block], 1,
                                         NCstencil_size, NCstencil_indexes,
                                         NCvalues);
#else
#endif
      }
   }

#if 0 /* TODO - deal with boundaries */
   /* Zero out stencils reaching to real boundary */
   for (i = 0; i < CCstencil_size*Cvolume; i++)
   {
      values[i] = 0.0;
   }
   for (d = 0; d < dim; d++)
   {
      stencil_indices[0] = d;
      for (block = 0; block < nblocks; block++)
      {
         if( ilower[block][d] == istart[d] )
         {
            i = iupper[block][d];
            iupper[block][d] = istart[d];
            HYPRE_SStructMatrixSetBoxValues(A, ilower[block], iupper[block],
                                            1, stencil_indices, values);
            iupper[block][d] = i;
         }
      }
   }
#endif

   /* Glue the two parts together */
   CAvalues = CNvalues;
   for (kb = 0; kb < bz; kb++)
   {
      for (jb = 0; jb < by; jb++)
      {
         /* glue part 0 to part 1 */
         part = 0;
         block = (bx-1) + jb*bx + kb*bx*by;
         i = Cilower[block][0];
         Cilower[block][0] = Ciupper[block][0];
         HYPRE_SStructMatrixSetBoxValues(A, part,
                                         Cilower[block], Ciupper[block], 0,
                                         1, &CAstencil_index, CAvalues);
         Cilower[block][0] = i;

         /* glue part 1 to part 0 */
         part = 1;
         block = jb*bx + kb*bx*by;
         i = Ciupper[block][0];
         Ciupper[block][0] = Cilower[block][0];
         HYPRE_SStructMatrixSetBoxValues(A, part,
                                         Cilower[block], Ciupper[block], 0,
                                         1, &CAstencil_index, CAvalues);
         Ciupper[block][0] = i;
      }
   }

   HYPRE_SStructMatrixAssemble(A);
#if DEBUG
   HYPRE_SStructMatrixPrint("driver.out.A", A, 0);
#endif

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
   HYPRE_SStructVectorInitialize(b);
   for (i = 0; i < Cvolume; i++)
   {
      CCvalues[i] = 1.0;
   }
   for (i = 0; i < Nvolume; i++)
   {
      NNvalues[i] = 1.0;
   }
   for (part = 0; part < 2; part++)
   {
      for (block = 0; block < nblocks; block++)
      {
         HYPRE_SStructVectorSetBoxValues(b, part,
                                         Cilower[block], Ciupper[block], 0,
                                         CCvalues);
         HYPRE_SStructVectorSetBoxValues(b, part,
                                         Nilower[block], Niupper[block], 1,
                                         NNvalues);
      }
   }
   HYPRE_SStructVectorAssemble(b);
#if DEBUG
   HYPRE_SStructVectorPrint("driver.out.b", b, 0);
#endif

   HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);
   HYPRE_SStructVectorInitialize(x);
   for (i = 0; i < Cvolume; i++)
   {
      CCvalues[i] = 0.0;
   }
   for (i = 0; i < Nvolume; i++)
   {
      NNvalues[i] = 0.0;
   }
   for (part = 0; part < 2; part++)
   {
      for (block = 0; block < nblocks; block++)
      {
         HYPRE_SStructVectorSetBoxValues(x, part,
                                         Cilower[block], Ciupper[block], 0,
                                         CCvalues);
         HYPRE_SStructVectorSetBoxValues(x, part,
                                         Nilower[block], Niupper[block], 1,
                                         NNvalues);
      }
   }
   HYPRE_SStructVectorAssemble(x);
#if DEBUG
   HYPRE_SStructVectorPrint("driver.out.x0", x, 0);
#endif

#if DEBUG
   hypre_SStructMatvec(1.0, A, b, 0.0, x);
   HYPRE_SStructVectorPrint("driver.out.matvec", x, 0);

   hypre_SStructCopy(b, x);
   HYPRE_SStructVectorPrint("driver.out.copy", x, 0);

   hypre_SStructScale(2.0, x);
   HYPRE_SStructVectorPrint("driver.out.scale", x, 0);

   hypre_SStructAxpy(-2.0, b, x);
   HYPRE_SStructVectorPrint("driver.out.axpy", x, 0);
#endif
 
   hypre_TFree(CCvalues);
   hypre_TFree(CNvalues);
   hypre_TFree(NNvalues);
   hypre_TFree(NCvalues);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("SStruct Interface", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if ((solver_id > 29) && (solver_id < 40))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructGMRESCreate(MPI_COMM_WORLD, &solver);
      HYPRE_SStructGMRESSetKDim(solver, 5);
      HYPRE_SStructGMRESSetMaxIter(solver, 100);
      HYPRE_SStructGMRESSetTol(solver, 1.0e-06);
      HYPRE_SStructGMRESSetLogging(solver, 1);

      if (solver_id == 30)
      {
      }

      else if (solver_id == 31)
      {
      }

      else if (solver_id == 38)
      {
#if 0 /* TODO */
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         HYPRE_SStructGMRESSetPrecond(solver,
                                      HYPRE_SStructDiagScale,
                                      HYPRE_SStructDiagScaleSetup,
                                      precond);
#endif
      }

      HYPRE_SStructGMRESSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_SStructGMRESSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_SStructGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_SStructGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_SStructGMRESDestroy(solver);

      if (solver_id == 30)
      {
      }
      else if (solver_id == 31)
      {
      }
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

#if DEBUG
   HYPRE_SStructVectorPrint("driver.out.x", x, 0);
#endif

   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_SStructGridDestroy(grid);
   HYPRE_SStructStencilDestroy(Cstencil);
   HYPRE_SStructStencilDestroy(Nstencil);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);

   for (i = 0; i < nblocks; i++)
   {
      hypre_TFree(Ciupper[i]);
      hypre_TFree(Cilower[i]);
      hypre_TFree(Niupper[i]);
      hypre_TFree(Nilower[i]);
   }
   hypre_TFree(Cilower);
   hypre_TFree(Ciupper);
   hypre_TFree(Nilower);
   hypre_TFree(Niupper);

   hypre_TFree(CCvalues);
   hypre_TFree(CNvalues);
   hypre_TFree(NNvalues);
   hypre_TFree(NCvalues);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}
