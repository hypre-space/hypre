#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE_struct_ls.h"
#include "struct_mv.h"
 
#include "bHYPRE_StructBuildMatrix.h"
#include "bHYPRE_StructMatrix.h"
#include "bHYPRE_StructBuildVector.h"
#include "bHYPRE_StructVector.h"
#include "bHYPRE_Operator.h"
#include "bHYPRE_Solver.h"
#include "bHYPRE_StructSMG.h"
#include "bHYPRE_PCG.h"
#include "bHYPRE_StructGrid.h"
#include "bHYPRE_StructStencil.h"
#include "bHYPRE_StructGrid_Impl.h"

#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 * Modified to use the Babel interface.
 * ********* obsolete, needs to be reworked ********
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 * Standard 7-point laplacian in 3D with grid and anisotropy determined
 * as command line arguments.  Do `driver -help' for usage info.
 *----------------------------------------------------------------------*/

int SetStencilBndry
( bHYPRE_StructMatrix A_b, bHYPRE_StructGrid grid, int* periodic );

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
   /* obsolete double              doubtemp;*/
                     
/* obsolete...
   bHYPRE_StructBuildMatrix MatBldr;
   bHYPRE_StructBuildVector VecBldr;
*/
/* not currently used   bHYPRE_Operator lo_test;*/
   bHYPRE_StructMatrix  A_b;
/* not currently used   bHYPRE_Operator A_LO;*/
   bHYPRE_Operator A_O;
   bHYPRE_StructVector  b_SV;
   bHYPRE_Vector  b_V;
   bHYPRE_StructVector  x_SV;
   bHYPRE_Vector  x_V;

/* not currently used   bHYPRE_Solver  solver;*/
   bHYPRE_Solver  precond;
/*   bHYPRE_StructJacobi  solver_SJ;*/
   bHYPRE_StructSMG solver_SMG;
   bHYPRE_PCG  solver_PCG;

   int zero = 0;
   int one = 1;
   int size;
   struct sidl_int__array* sidl_upper;
   struct sidl_int__array* sidl_lower;
   struct sidl_int__array* sidl_periodic;
   struct sidl_int__array* sidl_offsets;
   struct sidl_int__array* sidl_num_ghost;
   struct sidl_int__array* sidl_stencil_indices;
   struct sidl_double__array* sidl_values;

   MPI_Comm comm = MPI_COMM_WORLD;
   int symmetric;

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

   bHYPRE_StructGrid grid;
   bHYPRE_StructStencil stencil;
/*    HYPRE_StructGrid    grid; */
/*    HYPRE_StructStencil stencil; */

   int                *stencil_indices;
   double             *values;

   int                 i, s;
/* not currently used   int                 isave, d;*/
   int                 ix, iy, iz, ib;

   int                 periodic_error = 0;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
 
   ierr = 0;

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

   sidl_upper= sidl_int__array_create1d( dim );
   sidl_lower= sidl_int__array_create1d( dim );
   grid = bHYPRE_StructGrid__create();
   ierr += bHYPRE_StructGrid_SetCommunicator( grid, (void *) comm );
   ierr += bHYPRE_StructGrid_SetDimension( grid, dim );
   for (ib = 0; ib < nblocks; ib++)
   {
      for ( i=0; i < dim; i++ )
      {
         sidl_int__array_set1( sidl_upper, i, iupper[ib][i] );
         sidl_int__array_set1( sidl_lower, i, ilower[ib][i] );
      }
      bHYPRE_StructGrid_SetExtents( grid, sidl_lower, sidl_upper );
   }
   sidl_int__array_deleteRef( sidl_upper );
   sidl_int__array_deleteRef( sidl_lower );

   sidl_periodic= sidl_int__array_create1d( dim );
   for ( i=0; i < dim; i++ )
   {
      sidl_int__array_set1( sidl_periodic, i, periodic[i] );
   }
   bHYPRE_StructGrid_SetPeriodic( grid, sidl_periodic );
   sidl_int__array_deleteRef( sidl_periodic );

   bHYPRE_StructGrid_Assemble( grid );

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
 
   stencil = bHYPRE_StructStencil__create();
   ierr += bHYPRE_StructStencil_SetDimension( stencil, dim );
   ierr += bHYPRE_StructStencil_SetSize( stencil, dim+1 );
   /* ...SetSize recognizes that ...SetDimension has been called, so it calls
      HYPRE_StructStencilCreate */
   sidl_offsets= sidl_int__array_create1d( dim );
   for (s = 0; s < dim + 1; s++)
   {
      for ( i=0; i < dim; i++ )
      {
         sidl_int__array_set1( sidl_offsets, i, offsets[s][i] );
      }
      bHYPRE_StructStencil_SetElement( stencil, s, sidl_offsets );
   };
   sidl_int__array_deleteRef( sidl_offsets );

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
   /* ... this test code, and probably the present Babel interface, only works
      with symmetric matrix storage */
   A_b = bHYPRE_StructMatrix__create();
   ierr += bHYPRE_StructMatrix_SetCommunicator( A_b, (void *) comm );
   ierr += bHYPRE_StructMatrix_SetGrid( A_b, grid );
   /* ... the above matrix Set functions _must_ be called before the following ones ... */
   ierr += bHYPRE_StructMatrix_SetStencil( A_b, stencil );
   ierr += bHYPRE_StructMatrix_SetSymmetric( A_b, symmetric );
   size = 2*dim;
   sidl_num_ghost= sidl_int__array_borrow( A_num_ghost, 1, &zero, &size, &one );
   ierr += bHYPRE_StructMatrix_SetNumGhost( A_b, sidl_num_ghost );
   sidl_int__array_deleteRef( sidl_num_ghost );
   ierr += bHYPRE_StructMatrix_Initialize( A_b );

/* obsolete...
   num_ghost.lower[0] = 0;
   num_ghost.upper[0] = 2*dim;
   num_ghost.data = A_num_ghost;
   MatBldr = bHYPRE_StructMatrixBuilder_Constructor
      ( grid, stencil, symmetric, num_ghost );
   bHYPRE_StructMatrixBuilder_Start( MatBldr, grid, stencil, symmetric, num_ghost );
*/

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
   sidl_lower= sidl_int__array_create1d( dim );
   sidl_upper= sidl_int__array_create1d( dim );
   size = dim+1;
   sidl_stencil_indices= sidl_int__array_borrow( stencil_indices, 1, &zero, &size, &one );
   size = (dim+1)*volume;
   sidl_values= sidl_double__array_borrow( values, 1, &zero, &size, &one );
   for (ib = 0; ib < nblocks; ib++)
   {
      for ( i=0; i < dim; i++ )
      {
         sidl_int__array_set1( sidl_lower, i, ilower[ib][i] );
         sidl_int__array_set1( sidl_upper, i, iupper[ib][i] );
      }
      ierr += bHYPRE_StructMatrix_SetBoxValues
         ( A_b, sidl_lower, sidl_upper, dim+1,
           sidl_stencil_indices,
           sidl_values );
   }
   sidl_int__array_deleteRef( sidl_lower );
   sidl_int__array_deleteRef( sidl_upper );
   sidl_int__array_deleteRef( sidl_stencil_indices );
   sidl_double__array_deleteRef( sidl_values );
/* obsolete...
   intvals.lower[0] = 0;
   intvals.upper[0] = dim+1;
   doubvals.lower[0] = 0;
   doubvals.upper[0] = (dim+1)*volume;
   intvals.data = stencil_indices;
   doubvals.data = values;
   for (ib = 0; ib < nblocks; ib++)
   {
      bHYPRE_StructMatrixBuilder_SetBoxValues( MatBldr, box[ib], intvals, doubvals );
   }
*/
/*       HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib], (dim+1), */
/*                                      stencil_indices, values); */

   /* Zero out stencils reaching to real boundary */
   /* this section moved to a new function, SetStencilBndry */
   ierr += SetStencilBndry( A_b, grid, periodic); 
#if 0
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
/* obsolete
            intvals_lo.lower[0] = 0;
            intvals_lo.upper[0] = dim;
            intvals_hi.lower[0] = 0;
            intvals_hi.upper[0] = dim;
            intvals_lo.data = ilower[ib];
            intvals_hi.data = iupper[ib];
            bbox = bHYPRE_Box_Constructor( intvals_lo, intvals_hi, dim );
            bHYPRE_Box_Setup( bbox );
*/
            /* Put stencil point d (the one in direction d from the "middle"),
               into stencil_indices, so the corresponding matrix entry will
               get zeroed. */
            stencil_indices[0] = d;

/* obsolete
            intvals.lower[0] = 0;
            intvals.upper[0] = 1;
            intvals.data = stencil_indices;
            doubvals.lower[0] = 0;
            doubvals.upper[0] = volume;
            doubvals.data = values;
            bHYPRE_StructMatrixBuilder_SetBoxValues
               ( MatBldr, bbox, intvals, doubvals );
*/
/*             HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib], */
/*                                            1, stencil_indices, values); */
            iupper[ib][d] = isave;
            bHYPRE_Box_destructor( bbox );
         }
      }
   }
#endif

   ierr += bHYPRE_StructMatrix_Assemble( A_b );
/* obsolete...
   ierr += bHYPRE_StructMatrixBuilder_Setup( MatBldr );
   ierr += bHYPRE_StructMatrixBuilder_GetConstructedObject( MatBldr, &A_LO );
   A_b = (bHYPRE_StructMatrix) bHYPRE_LinearOperator_castTo
      ( A_LO, "bHYPRE.StructMatrix" );
*/

#if 0
   bHYPRE_StructMatrix_print( A_b );
/*   HYPRE_StructMatrixPrint("driver.out.A", A, 0); */
#endif

   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(double, volume);

   b_SV = bHYPRE_StructVector__create();
   ierr += bHYPRE_StructVector_SetCommunicator( b_SV, (void *) comm );
   ierr += bHYPRE_StructVector_SetGrid( b_SV, grid );
   ierr += bHYPRE_StructVector_Initialize( b_SV );
/* obsolete...
   VecBldr = bHYPRE_StructVectorBuilder_Constructor( grid );
   bHYPRE_StructVectorBuilder_Start( VecBldr, grid );
*/
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

   sidl_lower= sidl_int__array_create1d( dim );
   sidl_upper= sidl_int__array_create1d( dim );
   sidl_values= sidl_double__array_borrow( values, 1, &zero, &volume, &one );
   for (ib = 0; ib < nblocks; ib++)
   {
      for ( i=0; i < dim; i++ )
      {
         sidl_int__array_set1( sidl_lower, i, ilower[ib][i] );
         sidl_int__array_set1( sidl_upper, i, iupper[ib][i] );
      }
      ierr += bHYPRE_StructVector_SetBoxValues( b_SV, sidl_lower, sidl_upper, sidl_values );
   }
   sidl_int__array_deleteRef( sidl_lower );
   sidl_int__array_deleteRef( sidl_upper );
   sidl_double__array_deleteRef( sidl_values );

/* obsolete...
   doubvals.data = values;
   for (ib = 0; ib < nblocks; ib++)
   {
      bHYPRE_StructVectorBuilder_SetBoxValues( VecBldr, box[ib], doubvals );
   }
*/
/*       HYPRE_StructVectorSetBoxValues(b, ilower[ib], iupper[ib], values); */
/*   HYPRE_StructVectorAssemble(b); */
   bHYPRE_StructVector_Assemble( b_SV );
/* obsolete...
   bHYPRE_StructVectorBuilder_GetConstructedObject( VecBldr, &b_V );
   b_SV = (bHYPRE_StructVector) bHYPRE_Vector_castTo
      ( b_V, "bHYPRE.StructVector" );
*/

#if 0
   bHYPRE_StructVector_Print( b_SV );
/*   HYPRE_StructVectorPrint("driver.out.b", b, 0); */
#endif

/* obsolete...
   bHYPRE_StructVectorBuilder_Start( VecBldr, grid );
*/
/*    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, stencil, &x); */
/*    HYPRE_StructVectorInitialize(x); */
   x_SV = bHYPRE_StructVector__create();
   ierr += bHYPRE_StructVector_SetCommunicator( x_SV, (void *) comm );
   ierr += bHYPRE_StructVector_SetGrid( x_SV, grid );
   ierr += bHYPRE_StructVector_Initialize( x_SV );

   sidl_lower= sidl_int__array_create1d( dim );
   sidl_upper= sidl_int__array_create1d( dim );
   sidl_values= sidl_double__array_borrow( values, 1, &zero, &volume, &one );
   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   for (ib = 0; ib < nblocks; ib++)
   {
      for ( i=0; i < dim; i++ )
      {
         sidl_int__array_set1( sidl_lower, i, ilower[ib][i] );
         sidl_int__array_set1( sidl_upper, i, iupper[ib][i] );
      }
      ierr += bHYPRE_StructVector_SetBoxValues( x_SV, sidl_lower, sidl_upper, sidl_values );
   }
   sidl_int__array_deleteRef( sidl_lower );
   sidl_int__array_deleteRef( sidl_upper );
   sidl_double__array_deleteRef( sidl_values );

   bHYPRE_StructVector_Assemble( b_SV );
/* obsolete...
   doubvals.data = values;
   for (ib = 0; ib < nblocks; ib++)
   {
      bHYPRE_StructVectorBuilder_SetBoxValues( VecBldr, box[ib], doubvals );
   }
*/
/*       HYPRE_StructVectorSetBoxValues(x, ilower[ib], iupper[ib], values); */
/*   HYPRE_StructVectorAssemble(x); */
/* obsolete...
   bHYPRE_StructVectorBuilder_Setup( VecBldr );
   bHYPRE_StructVectorBuilder_GetConstructedObject( VecBldr, &x_V );
   x_SV = (bHYPRE_StructVector) bHYPRE_Vector_castTo
      ( x_V, "bHYPRE.StructVector" );
*/

#if 0
   bHYPRE_StructVector_Print( x_SV );
/*   HYPRE_StructVectorPrint("driver.out.x0", x, 0); */
#endif
 
   hypre_TFree(values);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Struct Interface", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();


/* JfP: temporarily, call Jacobi iteration, using as a model the
   code which calls multigrid ... */
#if 0
   if ( solver_id==201 )
   {
      time_index = hypre_InitializeTiming("Jacobi Setup");
      hypre_BeginTiming(time_index);

      solver_SJ = bHYPRE_StructJacobi_Constructor( comm );

      bHYPRE_StructJacobi_SetDoubleParameter( solver_SJ, "tol", 1.0e-4 );
      bHYPRE_StructJacobi_SetParameterInt( solver_SJ, "max_iter", 500 );

      bHYPRE_StructJacobi_Setup( solver_SJ, A_LO, b_V, x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Jacobi Solve");
      hypre_BeginTiming(time_index);

      bHYPRE_StructJacobi_Apply( solver_SJ, b_V, &x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
/* "solver" and "lo_test" are not used, but the following lines demonstrate
   how we can simulate inheritance (of StructJacobi from Solver in this case;
   "implements" is a sort of inheritance).
   The second line only knows that it is using something with the Solver
   interface; it does not know about its relationship with the
   bHYPRE_StructJacobi object.
   */
      solver = (bHYPRE_Solver) bHYPRE_StructJacobi_castTo( solver_SJ,
                                                         "bHYPRE.Solver" ); 
      bHYPRE_Solver_GetSystemOperator( solver, &lo_test );

      bHYPRE_StructJacobi_destructor( solver_SJ );
   }
#endif

/* Conjugate Gradient */
   if ((solver_id > 9) && (solver_id < 20))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);


      solver_PCG = bHYPRE_PCG__create();
      ierr += bHYPRE_PCG_SetCommunicator( solver_PCG, (void *) comm );
      A_O = bHYPRE_Operator__cast( A_b );
      ierr += bHYPRE_PCG_SetOperator( solver_PCG, A_O );
      b_V = bHYPRE_Vector__cast( b_SV );
      x_V = bHYPRE_Vector__cast( x_SV );
      /* obsolete solver_PCG = bHYPRE_PCG_Constructor( comm );*/

      bHYPRE_PCG_SetIntParameter( solver_PCG, "max_iter", 50 );
      bHYPRE_PCG_SetDoubleParameter( solver_PCG, "tol", 1.0e-06);
      bHYPRE_PCG_SetIntParameter( solver_PCG, "2-norm", 1);
      bHYPRE_PCG_SetIntParameter( solver_PCG, "relative change test", 0);
      bHYPRE_PCG_SetIntParameter( solver_PCG, "log", 1);

      if (solver_id == 10)
      {
         /* use symmetric SMG as preconditioner */
         solver_SMG = bHYPRE_StructSMG__create();
         ierr += bHYPRE_StructSMG_SetCommunicator( solver_SMG, (void *) comm );
         ierr += bHYPRE_StructSMG_SetOperator( solver_SMG, A_O );

         bHYPRE_StructSMG_SetIntParameter( solver_SMG, "memory use", 0 );
         bHYPRE_StructSMG_SetIntParameter( solver_SMG, "max iter", 1 );
         bHYPRE_StructSMG_SetDoubleParameter( solver_SMG, "tol", 0.0 );
         bHYPRE_StructSMG_SetIntParameter( solver_SMG, "zero guess", 1 );
         bHYPRE_StructSMG_SetIntParameter( solver_SMG, "rel change", 0 );
         bHYPRE_StructSMG_SetIntParameter( solver_SMG, "num prerelax", n_pre );
         bHYPRE_StructSMG_SetIntParameter( solver_SMG, "num postrelax", n_post );
         bHYPRE_StructSMG_SetIntParameter( solver_SMG, "logging", 0 );

         ierr += bHYPRE_StructSMG_Setup( solver_SMG, b_V, x_V );
         /* obsolete solver_SMG = bHYPRE_StructSMG_Constructor( comm );*/

         precond = (bHYPRE_Solver) bHYPRE_StructSMG__cast2
            ( solver_SMG, "bHYPRE.Solver" ); 
      }
#if 0
      else if (solver_id == 17)
      {
         /* use two-step Jacobi as preconditioner */
         solver_SJ = bHYPRE_StructJacobi_Constructor( comm );
         precond = (bHYPRE_Solver) bHYPRE_StructJacobi_castTo
            ( solver_SJ, "bHYPRE.Solver" ); 
         bHYPRE_StructJacobi_SetDoubleParameter( solver_SJ, "tol", 0.0 );
         bHYPRE_StructJacobi_SetIntParameter( solver_SJ, "max_iter", 2 );
         bHYPRE_StructJacobi_SetIntParameter( solver_SJ, "zero guess", 0 );
      
         bHYPRE_StructJacobi_Setup( solver_SJ, A_LO, b_V, x_V );
      }
#endif
      else {
         printf( "Preconditioner not supported! Solver_id=%i\n", solver_id );
      }
      
      bHYPRE_PCG_SetPreconditioner( solver_PCG, precond );

      ierr += bHYPRE_PCG_Setup( solver_PCG, b_V, x_V );
      /* obsolete bHYPRE_PCG_Setup( solver_PCG, A_LO, b_V, x_V );*/

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      bHYPRE_PCG_Apply( solver_PCG, b_V, &x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_PCG_GetNumIterations( solver_PCG, &num_iterations );
      ierr += bHYPRE_PCG_GetRelResidualNorm( solver_PCG, &final_res_norm );
      /* obsolete bHYPRE_PCG_GetConvergenceInfo( solver_PCG, "number of iterations",
         &doubtemp );
         num_iterations = floor(doubtemp*1.001);  i.e. round(doubtemp)
      bHYPRE_PCG_GetConvergenceInfo( solver_PCG, "residual norm",
      &final_res_norm);*/

/* not available yet      bHYPRE_PCG_PrintLogging( solver_PCG ); */

      bHYPRE_PCG_deleteRef( solver_PCG );
   /* obsolete bHYPRE_PCG_destructor( solver_PCG );*/

      if (solver_id == 10)
      {
         /* obsolete bHYPRE_StructSMG_destructor( solver_SMG );*/
         bHYPRE_StructSMG_deleteRef( solver_SMG );
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
      else if (solver_id == 17)
      {
         bHYPRE_StructJacobi_destructor( solver_SJ );
      }
*/

   }

   /*-----------------------------------------------------------
    * Solve the system using SMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      time_index = hypre_InitializeTiming("SMG Setup");
      hypre_BeginTiming(time_index);

      solver_SMG = bHYPRE_StructSMG__create();
      ierr += bHYPRE_StructSMG_SetCommunicator( solver_SMG, (void *) comm );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "memory use", 0 );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "max iter", 50 );
      bHYPRE_StructSMG_SetDoubleParameter( solver_SMG, "tol", 1.0e-6 );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "rel change", 0 );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "num prerelax", n_pre );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "num postrelax", n_post );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "logging", 1 );

      A_O = bHYPRE_Operator__cast( A_b );
      ierr += bHYPRE_StructSMG_SetOperator( solver_SMG, A_O );
      b_V = bHYPRE_Vector__cast( b_SV );
      x_V = bHYPRE_Vector__cast( x_SV );
      ierr += bHYPRE_StructSMG_Setup( solver_SMG, b_V, x_V );
      /* obsolete solver_SMG = bHYPRE_StructSMG_Constructor( comm );*/

      /* obsolete bHYPRE_StructSMG_Setup( solver_SMG, A_LO, b_V, x_V );*/
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

      bHYPRE_StructSMG_Apply( solver_SMG, b_V, &x_V );
/*      
      HYPRE_StructSMGSolve(solver, A, b, x);
*/
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      ierr += bHYPRE_StructSMG_GetNumIterations( solver_SMG, &num_iterations );
      ierr += bHYPRE_StructSMG_GetRelResidualNorm( solver_SMG, &final_res_norm );
      /* obsolete bHYPRE_StructSMG_GetConvergenceInfo(
         solver_SMG, "num iterations", &doubtemp );
         num_iterations = floor( 1.001*doubtemp );
         bHYPRE_StructSMG_GetConvergenceInfo(
         solver_SMG, "final relative residual norm", &final_res_norm );*/
/*
      HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
*/
      /* obsolete bHYPRE_StructSMG_destructor( solver_SMG );*/
      bHYPRE_StructSMG_deleteRef( solver_SMG );
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
#if 0
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
#endif
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
#if 0
      else if (solver_id == 17)
      {
         HYPRE_StructJacobiDestroy(precond);
      }
#endif
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
   bHYPRE_StructVector_Print( x );
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

   bHYPRE_StructStencil_deleteRef( stencil );
   bHYPRE_StructGrid_deleteRef( grid );
   bHYPRE_StructMatrix_deleteRef( A_b );
   bHYPRE_StructVector_deleteRef( b_SV );
   bHYPRE_StructVector_deleteRef( x_SV );
   /* obsolete bHYPRE_MPI_Com_destructor(comm);
   bHYPRE_StructStencil_destructor(stencil);
   bHYPRE_StructGrid_destructor(grid);
   for (ib = 0; ib < nblocks; ib++)
      bHYPRE_Box_destructor(box[ib]);
   bHYPRE_StructMatrix_destructor(A_b);
   bHYPRE_StructVector_destructor(b_SV);
   bHYPRE_StructVector_destructor(x_SV);
   */
            
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
   /* obsolete hypre_TFree(box);*/

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

/*********************************************************************************
 * this function sets to zero the stencil entries that are on the boundary
 * Grid, matrix and the period are needed. 
 *********************************************************************************/ 

int SetStencilBndry
( bHYPRE_StructMatrix A_b, bHYPRE_StructGrid grid, int* period )
{
  int ierr=0;
  hypre_BoxArray    *gridboxes;
  int                size,i,j,d,ib;
  int              **ilower;
  int              **iupper;
  int               *vol;
  int               *istart, *iend;
  hypre_Box         *box;
  hypre_Box         *dummybox;
  hypre_Box         *boundingbox;
  double            *values;
  int                volume, dim;
  int               *stencil_indices;
  int                constant_coefficient;
  int zero = 0;
  int one = 1;
  struct sidl_int__array* sidl_upper;
  struct sidl_int__array* sidl_lower;
  struct sidl_int__array* sidl_stencil_indices;
  struct sidl_double__array* sidl_values;

  struct bHYPRE_StructGrid__data * grid_data;
  HYPRE_StructGrid Hgrid;
  HYPRE_StructGrid gridmatrix;
  grid_data = bHYPRE_StructGrid__get_data( grid );
  Hgrid = grid_data -> grid;
  gridmatrix = Hgrid;

  gridboxes       = hypre_StructGridBoxes(gridmatrix);
  boundingbox     = hypre_StructGridBoundingBox(gridmatrix);
  istart          = hypre_BoxIMin(boundingbox);
  iend            = hypre_BoxIMax(boundingbox);
  size            = hypre_StructGridNumBoxes(gridmatrix);
  dim             = hypre_StructGridDim(gridmatrix);
  stencil_indices = hypre_CTAlloc(int, 1);

  /*  constant_coefficient = hypre_StructMatrixConstantCoefficient(A);*/
  constant_coefficient = 0; /* Babel interface doesn't support c.c. yet */
  if ( constant_coefficient>0 ) return 1;
  /*...no space dependence if constant_coefficient==1,
    and space dependence only for diagonal if constant_coefficient==2 --
    and this function only touches off-diagonal entries */

  vol    = hypre_CTAlloc(int, size);
  ilower = hypre_CTAlloc(int*, size);
  iupper = hypre_CTAlloc(int*, size);
  for (i = 0; i < size; i++)
  {
     ilower[i] = hypre_CTAlloc(int, dim);
     iupper[i] = hypre_CTAlloc(int, dim);
  }

  i = 0;
  ib = 0;
  hypre_ForBoxI(i, gridboxes)
     {
        dummybox = hypre_BoxCreate( );
        box      = hypre_BoxArrayBox(gridboxes, i);
        volume   =  hypre_BoxVolume(box);
        vol[i]   = volume;
        hypre_CopyBox(box,dummybox);
        for (d = 0; d < dim; d++)
        {
	   ilower[ib][d] = hypre_BoxIMinD(dummybox,d);
	   iupper[ib][d] = hypre_BoxIMaxD(dummybox,d);
        }
	ib++ ;
        hypre_BoxDestroy(dummybox);
     }

  if ( constant_coefficient==0 )
  {
     sidl_lower= sidl_int__array_create1d( dim );
     sidl_upper= sidl_int__array_create1d( dim );
     sidl_stencil_indices= sidl_int__array_create1d( 1 );
     for (d = 0; d < dim; d++)
     {
        for (ib = 0; ib < size; ib++)
        {
           values = hypre_CTAlloc(double, vol[ib]);
        
           for (i = 0; i < vol[ib]; i++)
           {
              values[i] = 0.0;
           }
           sidl_values= sidl_double__array_borrow( values, 1, &zero, &(vol[ib]), &one );

           if( ilower[ib][d] == istart[d] && period[d] == 0 )
           {
              j = iupper[ib][d];
              iupper[ib][d] = istart[d];
              stencil_indices[0] = d;

              for ( i=0; i < dim; i++ )
              {
                 sidl_int__array_set1( sidl_lower, i, ilower[ib][i] );
                 sidl_int__array_set1( sidl_upper, i, iupper[ib][i] );
              }
              sidl_int__array_set1( sidl_stencil_indices, 0, stencil_indices[0] );
              bHYPRE_StructMatrix_SetBoxValues
                 ( A_b, sidl_lower, sidl_upper, 1, sidl_stencil_indices, sidl_values );
              /* HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                 1, stencil_indices, values);*/
              iupper[ib][d] = j;
           }

           if( iupper[ib][d] == iend[d] && period[d] == 0 )
           {
              j = ilower[ib][d];
              ilower[ib][d] = iend[d];
              stencil_indices[0] = dim + 1 + d;
              for ( i=0; i < dim; i++ )
              {
                 sidl_int__array_set1( sidl_lower, i, ilower[ib][i] );
                 sidl_int__array_set1( sidl_upper, i, iupper[ib][i] );
              }
              sidl_int__array_set1( sidl_stencil_indices, 0, stencil_indices[0] );
              bHYPRE_StructMatrix_SetBoxValues
                 ( A_b, sidl_lower, sidl_upper, 1, sidl_stencil_indices, sidl_values );
              /* HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                 1, stencil_indices, values);*/
              ilower[ib][d] = j;
           }
           sidl_double__array_deleteRef( sidl_values );
           hypre_TFree(values);
        }
     }
     sidl_int__array_deleteRef( sidl_lower );
     sidl_int__array_deleteRef( sidl_upper );
     sidl_int__array_deleteRef( sidl_stencil_indices );
  }
  
  hypre_TFree(vol);
  hypre_TFree(stencil_indices);
  for (ib =0 ; ib < size ; ib++)
  {
     hypre_TFree(ilower[ib]);
     hypre_TFree(iupper[ib]);
  }
  hypre_TFree(ilower);
  hypre_TFree(iupper);

  

  return ierr;
}

