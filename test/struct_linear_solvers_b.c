#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE_struct_ls.h"
 
#include "Hypre_Box_Stub.h"
#include "Hypre_StructStencil_Stub.h"
#include "Hypre_StructGrid_Stub.h"
#include "Hypre_StructMatrix_Stub.h"
#include "Hypre_StructMatrixBuilder_Stub.h"
#include "Hypre_StructVector_Stub.h"
#include "Hypre_StructVectorBuilder_Stub.h"
#include "Hypre_MPI_Com_Stub.h"
#include "Hypre_StructJacobi_Stub.h"
#include "Hypre_StructSMG_Stub.h"
#include "Hypre_PCG_Stub.h"

#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 * Modified to use the Babel interface.
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 * Standard 7-point laplacian in 3D with grid and anisotropy determined
 * as command line arguments.  Do `driver -help' for usage info.
 *----------------------------------------------------------------------*/

int
main( int   argc,
      char *argv[] )
{
   int                 ierr;
   int                 arg_index;
   int                 print_usage;
   int                 nx, ny, nz;
   int                 P, Q, R;
   int                 bx, by, bz;
   int                 px, py, pz;
   double              cx, cy, cz;
   int                 solver_id;

   /*double              dxyz[3];*/

   int                 A_num_ghost[6] = {0, 0, 0, 0, 0, 0};
   double              doubtemp;
                     
   Hypre_StructMatrixBuilder MatBldr;
   Hypre_StructVectorBuilder VecBldr;
   Hypre_LinearOperator lo_test;
   Hypre_StructMatrix  A_SM;
   Hypre_LinearOperator A_LO;
   Hypre_StructVector  b_SV;
   Hypre_Vector  b_V;
   Hypre_StructVector  x_SV;
   Hypre_Vector  x_V;

   Hypre_Solver  solver;
   Hypre_Solver  precond;
   Hypre_StructJacobi  solver_SJ;
   Hypre_StructSMG solver_SMG;
   Hypre_PCG  solver_PCG;

   Hypre_MPI_Com comm;
   Hypre_Box * box;
   Hypre_Box bbox;
   int symmetric;
   array1int arroffsets;
   array1int intvals;
   array1int intvals_lo;
   array1int intvals_hi;
   array1double doubvals;
   array1int num_ghost;
   array1int periodic_arr;
   int int1, int2, int3, int4, int5, int6, int7, int8, int9, int10;
   int int11, int12, int13, int14, int15;

/*    HYPRE_StructMatrix  A; */
/*    HYPRE_StructVector  b; */
/*    HYPRE_StructVector  x; */

/*    HYPRE_StructSolver  solver; */
/*    HYPRE_StructSolver  precond; */
   int                 num_iterations;
   int                 time_index;
   double              final_res_norm;

   int                 num_procs, myid;

   int                 p, q, r;
   int                 dim;
   int                 n_pre, n_post;
   int                 nblocks, volume;
   int                 skip;
   int                 jump;

   int               **iupper;
   int               **ilower;

   int                 istart[3];
   int                 periodic[3];

   int               **offsets;

   Hypre_StructGrid grid;
   Hypre_StructStencil stencil;
/*    HYPRE_StructGrid    grid; */
/*    HYPRE_StructStencil stencil; */

   int                *stencil_indices;
   double             *values;

   int                 i, isave, s, d;
   int                 ix, iy, iz, ib;

   int                 periodic_error = 0;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
   /* Make a MPI_Com object. */
   comm = Hypre_MPI_Com_Constructor( MPI_COMM_WORLD );


#ifdef HYPRE_DEBUG
   cegdb(&argc, &argv, myid);
#endif

   hypre_InitMemoryDebug(myid);

   arroffsets.lower = &int1;
   arroffsets.upper = &int2;
   intvals.lower = &int3;
   intvals.upper = &int4;
   intvals_lo.lower = &int5;
   intvals_lo.upper = &int6;
   intvals_hi.lower = &int7;
   intvals_hi.upper = &int8;
   num_ghost.lower = &int9;
   num_ghost.upper = &int10;
   periodic_arr.lower = &int11;
   periodic_arr.upper = &int12;
   doubvals.lower = &int13;
   doubvals.upper = &int14;

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   dim = 3;

   skip = 0;
   jump = 0;

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

   istart[0] = -3;
   istart[1] = -3;
   istart[2] = -3;

   px = 0;
   py = 0;
   pz = 0;

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
      else if ( strcmp(argv[arg_index], "-p") == 0 )
      {
         arg_index++;
         px = atoi(argv[arg_index++]);
         py = atoi(argv[arg_index++]);
         pz = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-skip") == 0 )
      {
         arg_index++;
         skip = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-jump") == 0 )
      {
         arg_index++;
         jump = atoi(argv[arg_index++]);
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
      printf("  -p <px> <py> <pz>    : periodicity in each dimension\n");
      printf("  -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("  -v <n_pre> <n_post>  : number of pre and post relaxations\n");
      printf("  -d <dim>             : problem dimension (2 or 3)\n");
      printf("  -skip <s>            : skip some relaxation in PFMG (0 or 1)\n");
      printf("  -jump <num>          : num levels to jump in SparseMSG\n");
      printf("  -solver <ID>         : solver ID (default = 0)\n");
      printf("                         0  - SMG\n");
      printf("                         1  - PFMG\n");
      printf("                         2  - SparseMSG\n");
      printf("                         10 - CG with SMG precond\n");
      printf("                         11 - CG with PFMG precond\n");
      printf("                         12 - CG with SparseMSG precond\n");
      printf("                         17 - CG with 2-step Jacobi\n");
      printf("                         18 - CG with diagonal scaling\n");
      printf("                         19 - CG\n");
      printf("                         20 - Hybrid with SMG precond\n");
      printf("                         21 - Hybrid with PFMG precond\n");
      printf("                         22 - Hybrid with SparseMSG precond\n");
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

   if ((px+py+pz) != 0 && solver_id != 0 )
   {
      printf("Error: Periodic implemented only for solver 0, SMG \n");
      periodic_error++;
   }
   if (periodic_error != 0)
   {
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
      printf("  (px, py, pz)    = (%d, %d, %d)\n", px, py, pz);
      printf("  (cx, cy, cz)    = (%f, %f, %f)\n", cx, cy, cz);
      printf("  (n_pre, n_post) = (%d, %d)\n", n_pre, n_post);
      printf("  dim             = %d\n", dim);
      printf("  skip            = %d\n", skip);
      printf("  jump            = %d\n", jump);
      printf("  solver ID       = %d\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up periodic flags and set istart = 0 for periodic dims
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("Struct Interface");
   hypre_BeginTiming(time_index);

   periodic[0] = px;
   periodic[1] = py;
   periodic[2] = pz;
   for (i = 0; i < dim; i++)
   {
      if (periodic[i] != 0)
         istart[i] = 0;
   }

   /*-----------------------------------------------------------
    * Set up dxyz for PFMG solver
    *-----------------------------------------------------------*/

#if 0
   dxyz[0] = 1.0e+123;
   dxyz[1] = 1.0e+123;
   dxyz[2] = 1.0e+123;
   if (cx > 0)
   {
      dxyz[0] = sqrt(1.0 / cx);
   }
   if (cy > 0)
   {
      dxyz[1] = sqrt(1.0 / cy);
   }
   if (cz > 0)
   {
      dxyz[2] = sqrt(1.0 / cz);
   }
#endif

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   switch (dim)
   {
      case 1:
         volume  = nx;
         nblocks = bx;
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

   box = hypre_CTAlloc( Hypre_Box, nblocks );
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
   intvals_lo.lower[0] = 0;
   intvals_lo.upper[0] = dim;
   intvals_hi.lower[0] = 0;
   intvals_hi.upper[0] = dim;
   switch (dim)
   {
      case 1:
         for (ix = 0; ix < bx; ix++)
         {
            ilower[ib][0] = istart[0]+ nx*(bx*p+ix);
            iupper[ib][0] = istart[0]+ nx*(bx*p+ix+1) - 1;
            intvals_lo.data = ilower[ib];
            intvals_hi.data = iupper[ib];
            box[ib] = Hypre_Box_Constructor( intvals_lo, intvals_hi, dim );
            Hypre_Box_Setup( box[ib] );
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
               intvals_lo.data = ilower[ib];
               intvals_hi.data = iupper[ib];
               box[ib] = Hypre_Box_Constructor( intvals_lo, intvals_hi, dim );
               Hypre_Box_Setup( box[ib] );
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
                  intvals_lo.data = ilower[ib];
                  intvals_hi.data = iupper[ib];
                  box[ib] = Hypre_Box_Constructor(
                     intvals_lo, intvals_hi, dim );
                  Hypre_Box_Setup( box[ib] );
                  ib++;
               }
         break;
   } 

   grid = Hypre_StructGrid_Constructor( comm, dim );
   for (ib = 0; ib < nblocks; ib++)
   {
      Hypre_StructGrid_SetGridExtents( grid, box[ib] );
   }

   periodic_arr.lower[0] = 0;
   periodic_arr.upper[0] = dim;
   periodic_arr.data = periodic;
   Hypre_StructGrid_SetParameterIntArray( grid, "periodic", periodic_arr );

   Hypre_StructGrid_Setup( grid );

/*    HYPRE_StructGridCreate(MPI_COMM_WORLD, dim, &grid); */
/*    for (ib = 0; ib < nblocks; ib++) */
/*    { */
/*       HYPRE_StructGridSetExtents(grid, ilower[ib], iupper[ib]); */
/*    } */
/*    HYPRE_StructGridSetPeriodic(grid, periodic); */
/*    HYPRE_StructGridAssemble(grid); */

   /*-----------------------------------------------------------
    * Set up the stencil structure
    *-----------------------------------------------------------*/
 
   stencil = Hypre_StructStencil_Constructor( dim, dim+1 );
   for (s = 0; s < dim + 1; s++)
   {
      arroffsets.data = offsets[s];
      Hypre_StructStencil_SetElement( stencil, s, arroffsets );
   };
   Hypre_StructStencil_Setup( stencil );

/*    HYPRE_StructStencilCreate(dim, dim + 1, &stencil); */
/*    for (s = 0; s < dim + 1; s++) */
/*    { */
/*       HYPRE_StructStencilSetElement(stencil, s, offsets[s]); */
/*    } */

   /*-----------------------------------------------------------
    * Set up the matrix structure
    *-----------------------------------------------------------*/
 
/*    HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A); */
/*    HYPRE_StructMatrixSetSymmetric(A, 1); */
/*    HYPRE_StructMatrixSetNumGhost(A, A_num_ghost); */
/*    HYPRE_StructMatrixInitialize(A); */
/*  jfp: it is important to set numGhost before Initialize.
    Note that HYPRE has an Assemble function too. */

   symmetric = 1;
   num_ghost.lower[0] = 0;
   num_ghost.upper[0] = 2*dim;
   num_ghost.data = A_num_ghost;
   MatBldr = Hypre_StructMatrixBuilder_Constructor
      ( grid, stencil, symmetric, num_ghost );
   Hypre_StructMatrixBuilder_Start( MatBldr, grid, stencil, symmetric, num_ghost );

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
   intvals.lower[0] = 0;
   intvals.upper[0] = dim+1;
   doubvals.lower[0] = 0;
   doubvals.upper[0] = (dim+1)*volume;
   intvals.data = stencil_indices;
   doubvals.data = values;
   for (ib = 0; ib < nblocks; ib++)
   {
      Hypre_StructMatrixBuilder_SetBoxValues( MatBldr, box[ib], intvals, doubvals );
/*       HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib], (dim+1), */
/*                                      stencil_indices, values); */
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
         if( ilower[ib][d] == istart[d] && periodic[d] == 0 )
         {  /* at boundary */

            /* Make boundary box by flattening out box[ib] for direction d. */
            isave = iupper[ib][d];
            iupper[ib][d] = istart[d];
            intvals_lo.lower[0] = 0;
            intvals_lo.upper[0] = dim;
            intvals_hi.lower[0] = 0;
            intvals_hi.upper[0] = dim;
            intvals_lo.data = ilower[ib];
            intvals_hi.data = iupper[ib];
            bbox = Hypre_Box_Constructor( intvals_lo, intvals_hi, dim );
            Hypre_Box_Setup( bbox );
   
            /* Put stencil point d (the one in direction d from the "middle"),
               into stencil_indices, so the corresponding matrix entry will
               get zeroed. */
            stencil_indices[0] = d;
            intvals.lower[0] = 0;
            intvals.upper[0] = 1;
            intvals.data = stencil_indices;
            doubvals.lower[0] = 0;
            doubvals.upper[0] = volume;
            doubvals.data = values;

            Hypre_StructMatrixBuilder_SetBoxValues
               ( MatBldr, bbox, intvals, doubvals );
/*             HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib], */
/*                                            1, stencil_indices, values); */
            iupper[ib][d] = isave;
            Hypre_Box_destructor( bbox );
         }
      }
   }

   ierr += Hypre_StructMatrixBuilder_Setup( MatBldr );
   ierr += Hypre_StructMatrixBuilder_GetConstructedObject( MatBldr, &A_LO );
   A_SM = (Hypre_StructMatrix) Hypre_LinearOperator_castTo
      ( A_LO, "Hypre.StructMatrix" );

#if 0
   Hypre_StructMatrix_print( A_SM );
/*   HYPRE_StructMatrixPrint("driver.out.A", A, 0); */
#endif

   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(double, volume);

   VecBldr = Hypre_StructVectorBuilder_Constructor( grid );
   Hypre_StructVectorBuilder_Start( VecBldr, grid );
/*    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, stencil, &b); */
/*    HYPRE_StructVectorInitialize(b); */

   /*-----------------------------------------------------------
    * For periodic b.c. in all directions, need rhs to satisfy 
    * compatibility condition. Achieved by setting a source and
    *  sink of equal strength.  All other problems have rhs = 1.
    *-----------------------------------------------------------*/

   if ((dim == 2 && px != 0 && py != 0) ||
       (dim == 3 && px != 0 && py != 0 && pz != 0))
   {
      for (i = 0; i < volume; i++)
      {
         values[i] = 0.0;
      }
      values[0]          =  1.0;
      values[volume - 1] = -1.0;
   }
   else
   {
      for (i = 0; i < volume; i++)
      {
         values[i] = 1.0;
      }
   }

   doubvals.data = values;
   for (ib = 0; ib < nblocks; ib++)
   {
      Hypre_StructVectorBuilder_SetBoxValues( VecBldr, box[ib], doubvals );
/*       HYPRE_StructVectorSetBoxValues(b, ilower[ib], iupper[ib], values); */
   }
/*   HYPRE_StructVectorAssemble(b); */
   Hypre_StructVectorBuilder_Setup( VecBldr );
   Hypre_StructVectorBuilder_GetConstructedObject( VecBldr, &b_V );
   b_SV = (Hypre_StructVector) Hypre_Vector_castTo
      ( b_V, "Hypre.StructVector" );

#if 0
   Hypre_StructVector_Print( b_SV );
/*   HYPRE_StructVectorPrint("driver.out.b", b, 0); */
#endif

   Hypre_StructVectorBuilder_Start( VecBldr, grid );
/*    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, stencil, &x); */
/*    HYPRE_StructVectorInitialize(x); */
   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   doubvals.data = values;
   for (ib = 0; ib < nblocks; ib++)
   {
      Hypre_StructVectorBuilder_SetBoxValues( VecBldr, box[ib], doubvals );
/*       HYPRE_StructVectorSetBoxValues(x, ilower[ib], iupper[ib], values); */
   }
/*   HYPRE_StructVectorAssemble(x); */
   Hypre_StructVectorBuilder_Setup( VecBldr );
   Hypre_StructVectorBuilder_GetConstructedObject( VecBldr, &x_V );
   x_SV = (Hypre_StructVector) Hypre_Vector_castTo
      ( x_V, "Hypre.StructVector" );

#if 0
   Hypre_StructVector_Print( x_SV );
/*   HYPRE_StructVectorPrint("driver.out.x0", x, 0); */
#endif
 
   hypre_TFree(values);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Struct Interface", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();


/* JfP: temporarily, call Jacobi iteration, using as a model the
   code which calls multigrid ... */

   if ( solver_id==201 )
   {
      time_index = hypre_InitializeTiming("Jacobi Setup");
      hypre_BeginTiming(time_index);

      solver_SJ = Hypre_StructJacobi_Constructor( comm );

      Hypre_StructJacobi_SetParameterDouble( solver_SJ, "tol", 1.0e-4 );
      Hypre_StructJacobi_SetParameterInt( solver_SJ, "max_iter", 500 );

      Hypre_StructJacobi_Setup( solver_SJ, A_LO, b_V, x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Jacobi Solve");
      hypre_BeginTiming(time_index);

      Hypre_StructJacobi_Apply( solver_SJ, b_V, &x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
/* "solver" and "lo_test" are not used, but the following lines demonstrate
   how we can simulate inheritance (of StructJacobi from Solver in this case;
   "implements" is a sort of inheritance).
   The second line only knows that it is using something with the Solver
   interface; it does not know about its relationship with the
   Hypre_StructJacobi object.
   */
      solver = (Hypre_Solver) Hypre_StructJacobi_castTo( solver_SJ,
                                                         "Hypre.Solver" ); 
      Hypre_Solver_GetSystemOperator( solver, &lo_test );

      Hypre_StructJacobi_destructor( solver_SJ );
   }


/* Conjugate Gradient */
   if ((solver_id > 9) && (solver_id < 20))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      solver_PCG = Hypre_PCG_Constructor( comm );

      Hypre_PCG_SetParameterInt( solver_PCG, "max_iter", 50 );
      Hypre_PCG_SetParameterDouble( solver_PCG, "tol", 1.0e-06);
      Hypre_PCG_SetParameterInt( solver_PCG, "2-norm", 1);
      Hypre_PCG_SetParameterInt( solver_PCG, "relative change test", 0);
      Hypre_PCG_SetParameterInt( solver_PCG, "log", 1);

      if (solver_id == 10)
      {
         /* use symmetric SMG as preconditioner */
         solver_SMG = Hypre_StructSMG_Constructor( comm );
         precond = (Hypre_Solver) Hypre_StructSMG_castTo
            ( solver_SMG, "Hypre.Solver" ); 

         Hypre_StructSMG_SetParameterInt( solver_SMG, "memory use", 0 );
         Hypre_StructSMG_SetParameterInt( solver_SMG, "max iter", 1 );
         Hypre_StructSMG_SetParameterDouble( solver_SMG, "tol", 0.0 );
         Hypre_StructSMG_SetParameterInt( solver_SMG, "zero guess", 1 );
         Hypre_StructSMG_SetParameterInt( solver_SMG, "rel change", 0 );
         Hypre_StructSMG_SetParameterInt( solver_SMG, "num prerelax", n_pre );
         Hypre_StructSMG_SetParameterInt( solver_SMG, "num postrelax", n_post );
         Hypre_StructSMG_SetParameterInt( solver_SMG, "logging", 0 );

         Hypre_StructSMG_Setup( solver_SMG, A_LO, b_V, x_V );
      }
      else if (solver_id == 17)
      {
         /* use two-step Jacobi as preconditioner */
         solver_SJ = Hypre_StructJacobi_Constructor( comm );
         precond = (Hypre_Solver) Hypre_StructJacobi_castTo
            ( solver_SJ, "Hypre.Solver" ); 
         Hypre_StructJacobi_SetParameterDouble( solver_SJ, "tol", 0.0 );
         Hypre_StructJacobi_SetParameterInt( solver_SJ, "max_iter", 2 );
         Hypre_StructJacobi_SetParameterInt( solver_SJ, "zero guess", 0 );
      
         Hypre_StructJacobi_Setup( solver_SJ, A_LO, b_V, x_V );
      }
      else {
         printf( "Preconditioner not supported! Solver_id=%i\n", solver_id );
      }
      
      Hypre_PCG_SetPreconditioner( solver_PCG, precond );

      Hypre_PCG_Setup( solver_PCG, A_LO, b_V, x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      Hypre_PCG_Apply( solver_PCG, b_V, &x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      Hypre_PCG_GetConvergenceInfo( solver_PCG, "number of iterations",
                                    &doubtemp );
      num_iterations = floor(doubtemp*1.001); /* round(doubtemp) */
      Hypre_PCG_GetConvergenceInfo( solver_PCG, "residual norm",
                                    &final_res_norm);

/* not available yet      Hypre_PCG_PrintLogging( solver_PCG ); */

      Hypre_PCG_destructor( solver_PCG );

      if (solver_id == 10)
      {
         Hypre_StructSMG_destructor( solver_SMG );
      }
/*
      else if (solver_id == 11)
      {
         HYPRE_StructPFMGDestroy(precond);
      }
      else if (solver_id == 12)
      {
         HYPRE_StructSparseMSGDestroy(precond);
      }
*/
      else if (solver_id == 17)
      {
         Hypre_StructJacobi_destructor( solver_SJ );
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using SMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      time_index = hypre_InitializeTiming("SMG Setup");
      hypre_BeginTiming(time_index);

      solver_SMG = Hypre_StructSMG_Constructor( comm );

      Hypre_StructSMG_SetParameterInt( solver_SMG, "memory use", 0 );
      Hypre_StructSMG_SetParameterInt( solver_SMG, "max iter", 50 );
      Hypre_StructSMG_SetParameterDouble( solver_SMG, "tol", 1.0e-6 );
      Hypre_StructSMG_SetParameterInt( solver_SMG, "rel change", 0 );
      Hypre_StructSMG_SetParameterInt( solver_SMG, "num prerelax", n_pre );
      Hypre_StructSMG_SetParameterInt( solver_SMG, "num postrelax", n_post );
      Hypre_StructSMG_SetParameterInt( solver_SMG, "logging", 1 );

      Hypre_StructSMG_Setup( solver_SMG, A_LO, b_V, x_V );
/*
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructSMGSetMemoryUse(solver, 0);
      HYPRE_StructSMGSetMaxIter(solver, 50);
      HYPRE_StructSMGSetTol(solver, 1.0e-06);
      HYPRE_StructSMGSetRelChange(solver, 0);
      HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
      HYPRE_StructSMGSetNumPostRelax(solver, n_post);
      HYPRE_StructSMGSetLogging(solver, 1);
      HYPRE_StructSMGSetup(solver, A, b, x);
*/

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SMG Solve");
      hypre_BeginTiming(time_index);

      Hypre_StructSMG_Apply( solver_SMG, b_V, &x_V );
/*      
      HYPRE_StructSMGSolve(solver, A, b, x);
*/
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      Hypre_StructSMG_GetConvergenceInfo(
         solver_SMG, "num iterations", &doubtemp );
      num_iterations = floor( 1.001*doubtemp );
      Hypre_StructSMG_GetConvergenceInfo(
         solver_SMG, "final relative residual norm", &final_res_norm );
/*
      HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
*/
      Hypre_StructSMG_destructor( solver_SMG );
/*    HYPRE_StructSMGDestroy(solver); */
   }

#if 0
/* most solvers not implemented yet (JfP jan2000) ... */

   /*-----------------------------------------------------------
    * Solve the system using PFMG
    *-----------------------------------------------------------*/

   else if (solver_id == 1)
   {
      time_index = hypre_InitializeTiming("PFMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructPFMGSetMaxIter(solver, 50);
      HYPRE_StructPFMGSetTol(solver, 1.0e-06);
      HYPRE_StructPFMGSetRelChange(solver, 0);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructPFMGSetRelaxType(solver, 1);
      HYPRE_StructPFMGSetNumPreRelax(solver, n_pre);
      HYPRE_StructPFMGSetNumPostRelax(solver, n_post);
      HYPRE_StructPFMGSetSkipRelax(solver, skip);
      /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
      HYPRE_StructPFMGSetLogging(solver, 1);
      HYPRE_StructPFMGSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PFMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructPFMGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      HYPRE_StructPFMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructPFMGDestroy(solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using SparseMSG
    *-----------------------------------------------------------*/

   else if (solver_id == 2)
   {
      time_index = hypre_InitializeTiming("SparseMSG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSparseMSGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructSparseMSGSetMaxIter(solver, 50);
      HYPRE_StructSparseMSGSetJump(solver, jump);
      HYPRE_StructSparseMSGSetTol(solver, 1.0e-06);
      HYPRE_StructSparseMSGSetRelChange(solver, 0);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructSparseMSGSetRelaxType(solver, 1);
      HYPRE_StructSparseMSGSetNumPreRelax(solver, n_pre);
      HYPRE_StructSparseMSGSetNumPostRelax(solver, n_post);
      HYPRE_StructSparseMSGSetLogging(solver, 1);
      HYPRE_StructSparseMSGSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SparseMSG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructSparseMSGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      HYPRE_StructSparseMSGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(solver,
                                                        &final_res_norm);
      HYPRE_StructSparseMSGDestroy(solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using CG
    *-----------------------------------------------------------*/

   if ((solver_id > 9) && (solver_id < 20))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructPCGSetMaxIter(solver, 50);
      HYPRE_StructPCGSetTol(solver, 1.0e-06);
      HYPRE_StructPCGSetTwoNorm(solver, 1);
      HYPRE_StructPCGSetRelChange(solver, 0);
      HYPRE_StructPCGSetLogging(solver, 1);

      if (solver_id == 10)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSMGSetMemoryUse(precond, 0);
         HYPRE_StructSMGSetMaxIter(precond, 1);
         HYPRE_StructSMGSetTol(precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(precond);
         HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(precond, n_post);
         HYPRE_StructSMGSetLogging(precond, 0);
         HYPRE_StructPCGSetPrecond(solver,
                                   HYPRE_StructSMGSolve,
                                   HYPRE_StructSMGSetup,
                                   precond);
      }

      else if (solver_id == 11)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructPFMGSetMaxIter(precond, 1);
         HYPRE_StructPFMGSetTol(precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_StructPFMGSetRelaxType(precond, 1);
         HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(precond, skip);
         /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
         HYPRE_StructPFMGSetLogging(precond, 0);
         HYPRE_StructPCGSetPrecond(solver,
                                   HYPRE_StructPFMGSolve,
                                   HYPRE_StructPFMGSetup,
                                   precond);
      }

      else if (solver_id == 12)
      {
         /* use symmetric SparseMSG as preconditioner */
         HYPRE_StructSparseMSGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSparseMSGSetMaxIter(precond, 1);
         HYPRE_StructSparseMSGSetJump(precond, jump);
         HYPRE_StructSparseMSGSetTol(precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_StructSparseMSGSetRelaxType(precond, 1);
         HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
         HYPRE_StructSparseMSGSetLogging(precond, 0);
         HYPRE_StructPCGSetPrecond(solver,
                                   HYPRE_StructSparseMSGSolve,
                                   HYPRE_StructSparseMSGSetup,
                                   precond);
      }

      else if (solver_id == 17)
      {
         /* use two-step Jacobi as preconditioner */
         HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructJacobiSetMaxIter(precond, 2);
         HYPRE_StructJacobiSetTol(precond, 0.0);
         HYPRE_StructJacobiSetZeroGuess(precond);
         HYPRE_StructPCGSetPrecond(solver,
                                   HYPRE_StructJacobiSolve,
                                   HYPRE_StructJacobiSetup,
                                   precond);
      }

      else if (solver_id == 18)
      {
         /* use diagonal scaling as preconditioner */
#ifdef HYPRE_USE_PTHREADS
         for (i = 0; i < hypre_NumThreads; i++)
         {
            precond[i] = NULL;
         }
#else
         precond = NULL;
#endif
         HYPRE_StructPCGSetPrecond(solver,
                                   HYPRE_StructDiagScale,
                                   HYPRE_StructDiagScaleSetup,
                                   precond);
      }

      HYPRE_StructPCGSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructPCGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructPCGGetNumIterations(solver, &num_iterations);
      HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructPCGDestroy(solver);

      if (solver_id == 10)
      {
         HYPRE_StructSMGDestroy(precond);
      }
      else if (solver_id == 11)
      {
         HYPRE_StructPFMGDestroy(precond);
      }
      else if (solver_id == 12)
      {
         HYPRE_StructSparseMSGDestroy(precond);
      }
      else if (solver_id == 17)
      {
         HYPRE_StructJacobiDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using Hybrid
    *-----------------------------------------------------------*/

   if ((solver_id > 19) && (solver_id < 30))
   {
      time_index = hypre_InitializeTiming("Hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructHybridSetDSCGMaxIter(solver, 100);
      HYPRE_StructHybridSetPCGMaxIter(solver, 50);
      HYPRE_StructHybridSetTol(solver, 1.0e-06);
      HYPRE_StructHybridSetConvergenceTol(solver, 0.90);
      HYPRE_StructHybridSetTwoNorm(solver, 1);
      HYPRE_StructHybridSetRelChange(solver, 0);
      HYPRE_StructHybridSetLogging(solver, 1);

      if (solver_id == 20)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSMGSetMemoryUse(precond, 0);
         HYPRE_StructSMGSetMaxIter(precond, 1);
         HYPRE_StructSMGSetTol(precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(precond);
         HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(precond, n_post);
         HYPRE_StructSMGSetLogging(precond, 0);
         HYPRE_StructHybridSetPrecond(solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      precond);
      }

      else if (solver_id == 21)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructPFMGSetMaxIter(precond, 1);
         HYPRE_StructPFMGSetTol(precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_StructPFMGSetRelaxType(precond, 1);
         HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(precond, skip);
         /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
         HYPRE_StructPFMGSetLogging(precond, 0);
         HYPRE_StructHybridSetPrecond(solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      precond);
      }

      else if (solver_id == 22)
      {
         /* use symmetric SparseMSG as preconditioner */
         HYPRE_StructSparseMSGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSparseMSGSetJump(precond, jump);
         HYPRE_StructSparseMSGSetMaxIter(precond, 1);
         HYPRE_StructSparseMSGSetTol(precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_StructSparseMSGSetRelaxType(precond, 1);
         HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
         HYPRE_StructSparseMSGSetLogging(precond, 0);
         HYPRE_StructHybridSetPrecond(solver,
                                      HYPRE_StructSparseMSGSolve,
                                      HYPRE_StructSparseMSGSetup,
                                      precond);
      }

      HYPRE_StructHybridSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructHybridGetNumIterations(solver, &num_iterations);
      HYPRE_StructHybridGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructHybridDestroy(solver);

      if (solver_id == 20)
      {
         HYPRE_StructSMGDestroy(precond);
      }
      else if (solver_id == 21)
      {
         HYPRE_StructPFMGDestroy(precond);
      }
      else if (solver_id == 22)
      {
         HYPRE_StructSparseMSGDestroy(precond);
      }
   }

#endif

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

#if 0
   Hypre_StructVector_Print( x );
/*   HYPRE_StructVectorPrint("driver.out.x", x, 0); */
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

   Hypre_MPI_Com_destructor(comm);
   Hypre_StructStencil_destructor(stencil);
   Hypre_StructGrid_destructor(grid);
   for (ib = 0; ib < nblocks; ib++)
      Hypre_Box_destructor(box[ib]);
   Hypre_StructMatrix_destructor(A_SM);
   Hypre_StructVector_destructor(b_SV);
   Hypre_StructVector_destructor(x_SV);
            
/*    HYPRE_StructGridDestroy(grid); */
/*    HYPRE_StructStencilDestroy(stencil); */
/*    HYPRE_StructMatrixDestroy(A); */
/*    HYPRE_StructVectorDestroy(b); */
/*    HYPRE_StructVectorDestroy(x); */

   for (i = 0; i < nblocks; i++)
   {
      hypre_TFree(iupper[i]);
      hypre_TFree(ilower[i]);
   }
   hypre_TFree(ilower);
   hypre_TFree(iupper);
   hypre_TFree(stencil_indices);
   hypre_TFree(box);

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
