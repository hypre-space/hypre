#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE_struct_ls.h"
#include "struct_mv.h"
 
#include "bHYPRE_StructMatrix.h"
#include "bHYPRE_StructVector.h"
#include "bHYPRE_Operator.h"
#include "bHYPRE_Solver.h"
#include "bHYPRE_StructSMG.h"
#include "bHYPRE_StructPFMG.h"
#include "bHYPRE_IdentitySolver.h"
#include "bHYPRE_StructDiagScale.h"
#include "bHYPRE_PCG.h"
#include "bHYPRE_Hybrid.h"
#include "bHYPRE_StructGrid.h"
#include "bHYPRE_StructStencil.h"
#include "bHYPRE_StructGrid_Impl.h"

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

int SetStencilBndry
( bHYPRE_StructMatrix A_b, bHYPRE_StructGrid grid, int* periodic );

int
AddValuesMatrix( bHYPRE_StructMatrix A_b,
                 int dim, int nblocks, int ** ilower, int ** iupper,
                 double cx, double cy, double cz,
                 double conx, double cony, double conz,
                 int symmetric, int constant_coefficient );

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
   double              conx, cony, conz;
   int                 solver_id;
   int                 relax, rap;

   int                 A_num_ghost[6] = {0, 0, 0, 0, 0, 0};
                     
/* not currently used   bHYPRE_Operator lo_test;*/
   bHYPRE_MPICommunicator bmpicomm;
   bHYPRE_StructMatrix  A_b;
/* not currently used   bHYPRE_Operator A_LO;*/
   bHYPRE_Operator A_O;
   bHYPRE_StructVector  b_SV;
   bHYPRE_Vector  b_V;
   bHYPRE_StructVector  x_SV;
   bHYPRE_Vector  x_V;

/* not currently used   bHYPRE_Solver  solver;*/
   bHYPRE_Solver  precond;
   bHYPRE_PreconditionedSolver  krylov_solver;
/*   bHYPRE_StructJacobi  solver_SJ;*/
   bHYPRE_StructSMG solver_SMG;
   bHYPRE_StructPFMG solver_PFMG;
   bHYPRE_IdentitySolver solver_Id;
   bHYPRE_PCG  solver_PCG;
   bHYPRE_PCG  solver_PCG_1;
   bHYPRE_StructDiagScale  solver_DS;
   bHYPRE_Hybrid solver_Hybrid;

   int constant_coefficient = 0;
   int symmetric = 1;
   MPI_Comm mpi_comm = MPI_COMM_WORLD;

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
   int               *constant_stencil_points = NULL;

   bHYPRE_StructGrid grid;
   bHYPRE_StructStencil stencil;

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
   bmpicomm = bHYPRE_MPICommunicator_CreateC( (void *)(&mpi_comm) );

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
   rap = 0;
   relax = 1;

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
   conx = 0.0;
   cony = 0.0;
   conz = 0.0;

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
      else if ( strcmp(argv[arg_index], "-convect") == 0 )
      {
         arg_index++;
         conx = atof(argv[arg_index++]);
         cony = atof(argv[arg_index++]);
         conz = atof(argv[arg_index++]);
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
      printf("                         2 *- SparseMSG\n");
      printf("                         3  - PFMG constant coefficients\n");
      printf("                         4  - PFMG constant coefficients variable diagonal\n");
      printf("                         10 - CG with SMG precond\n");
      printf("                         11 - CG with PFMG precond\n");
      printf("                         12*- CG with SparseMSG precond\n");
      printf("                         13 - CG with PFMG precond, constant coefficients\n");
      printf("                         14 - CG with PFMG precond, const.coeff.,variable diagonal\n");
      printf("                         17*- CG with 2-step Jacobi\n");
      printf("                         18 - CG with diagonal scaling\n");
      printf("                         19 - CG\n");
      printf("                         20 - Hybrid with SMG precond\n");
      printf("                         21*- Hybrid with PFMG precond\n");
      printf("                         22*- Hybrid with SparseMSG precond\n");
      printf("Solvers marked with '*' have not yet been implemented.\n");
      /* >>> TO DO SOON: solvers 18,19 <<< */
      printf("\n");

      bHYPRE_MPICommunicator_deleteRef( bmpicomm );
      MPI_Finalize();
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

   if ((conx != 0.0 || cony !=0 || conz != 0) && symmetric == 1 )
   {
      printf("\n*** Warning: convection produces non-symetric matrix ***\n\n");
      symmetric = 0;
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
      printf("  (conx,cony,conz)= (%f, %f, %f)\n", conx, cony, conz);
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
    * Set up dxyz for PFMG solver  >>> NOT IMPLEMENTED <<<
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
    * Set up the grid structure and some of the stencil
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
         if ( solver_id == 3 || solver_id == 13)
            {
               constant_stencil_points = hypre_CTAlloc(int, 2);
               constant_stencil_points[0] = 0;
               constant_stencil_points[1] = 1;
            }
         if ( solver_id == 4 || solver_id == 14)
            {
               constant_stencil_points = hypre_CTAlloc(int, 2);
               constant_stencil_points[0] = 0;
               constant_stencil_points[1] = 0;
            }
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
         if ( solver_id == 3 || solver_id == 13)
            {
               constant_stencil_points = hypre_CTAlloc(int, 3);
               constant_stencil_points[0] = 0;
               constant_stencil_points[1] = 1;
               constant_stencil_points[2] = 2;
            }
         if ( solver_id == 4 || solver_id == 14)
            {
               constant_stencil_points = hypre_CTAlloc(int, 3);
               constant_stencil_points[0] = 0;
               constant_stencil_points[1] = 1;
               constant_stencil_points[2] = 1;
            }
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
         if ( solver_id == 3 || solver_id == 13)
            {
               constant_stencil_points = hypre_CTAlloc(int, 4);
               constant_stencil_points[0] = 0;
               constant_stencil_points[1] = 1;
               constant_stencil_points[2] = 2;
               constant_stencil_points[3] = 3;
            }
         if ( solver_id == 4 || solver_id == 14)
            {
               constant_stencil_points = hypre_CTAlloc(int, 4);
               constant_stencil_points[0] = 0;
               constant_stencil_points[1] = 1;
               constant_stencil_points[2] = 2;
               constant_stencil_points[3] = 2;
            }
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

   grid = bHYPRE_StructGrid_Create( bmpicomm, dim );
   for (ib = 0; ib < nblocks; ib++)
   {
      bHYPRE_StructGrid_SetExtents( grid, ilower[ib], iupper[ib], dim );
   }

   bHYPRE_StructGrid_SetPeriodic( grid, periodic, dim );

   bHYPRE_StructGrid_Assemble( grid );

   /*-----------------------------------------------------------
    * Set up the stencil structure
    *-----------------------------------------------------------*/
 
   stencil = bHYPRE_StructStencil_Create( dim, dim+1 );

   for (s = 0; s < dim + 1; s++)
   {
      bHYPRE_StructStencil_SetElement( stencil, s, offsets[s], dim );
   };

   /*-----------------------------------------------------------
    * Set up the matrix structure
    *-----------------------------------------------------------*/
 
   if ( solver_id == 3 || solver_id == 13 ) constant_coefficient = 1;
   else if ( solver_id == 4 || solver_id == 14 ) constant_coefficient = 2;

   /* This test code, and probably the present Babel interface, has only
      been tested, and probably only works, with symmetric matrix storage.
      It may not be a big deal to test & support nonsymmetric storage. */
   hypre_assert( symmetric== 1 );

   A_b = bHYPRE_StructMatrix_Create( bmpicomm, grid, stencil );

   ierr += bHYPRE_StructMatrix_SetSymmetric( A_b, symmetric );
   ierr += bHYPRE_StructMatrix_SetNumGhost( A_b, A_num_ghost, 2*dim );

   if ( solver_id == 3 || solver_id == 4 || solver_id == 13 || solver_id == 14 )
   {
      bHYPRE_StructMatrix_SetConstantEntries( A_b, dim+1, constant_stencil_points );
   }

   ierr += bHYPRE_StructMatrix_Initialize( A_b );

   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/

   ierr += AddValuesMatrix( A_b, dim, nblocks, ilower, iupper,
                            cx, cy, cz, conx, cony, conz,
                            symmetric, constant_coefficient );

   /* Zero out stencils reaching to real boundary */
   if ( constant_coefficient == 0 ) ierr += SetStencilBndry( A_b, grid, periodic); 

   ierr += bHYPRE_StructMatrix_Assemble( A_b );

#if 0
   bHYPRE_StructMatrix_print( A_b );
/*   HYPRE_StructMatrixPrint("driver.out.A", A, 0); */
#endif

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(double, volume);

   b_SV = bHYPRE_StructVector_Create( bmpicomm, grid );

   ierr += bHYPRE_StructVector_Initialize( b_SV );

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

   for (ib = 0; ib < nblocks; ib++)
   {
      ierr += bHYPRE_StructVector_SetBoxValues( b_SV, ilower[ib], iupper[ib],
                                                dim, values, volume );
   }

   bHYPRE_StructVector_Assemble( b_SV );

#if 0
   bHYPRE_StructVector_Print( b_SV );
#endif

   x_SV = bHYPRE_StructVector_Create( bmpicomm, grid );

   ierr += bHYPRE_StructVector_Initialize( x_SV );

   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   for (ib = 0; ib < nblocks; ib++)
   {
      ierr += bHYPRE_StructVector_SetBoxValues( x_SV, ilower[ib], iupper[ib],
                                                dim, values, volume );
   }

   bHYPRE_StructVector_Assemble( b_SV );

#if 0
   bHYPRE_StructVector_Print( x_SV );
#endif
 
   hypre_TFree(values);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Struct Interface", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();


/* >>> The following commented-out section is ANCIENT; revisit still later... >>> */
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

   /*-----------------------------------------------------------
    * Solve the system using SMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      time_index = hypre_InitializeTiming("SMG Setup");
      hypre_BeginTiming(time_index);

      solver_SMG = bHYPRE_StructSMG_Create( bmpicomm, A_b );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MemoryUse", 0 );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MaxIter", 50 );
      bHYPRE_StructSMG_SetDoubleParameter( solver_SMG, "Tol", 1.0e-6 );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "RelChange", 0 );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "NumPrerelax", n_pre );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "NumPostrelax", n_post );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "Logging", 1 );

      b_V = bHYPRE_Vector__cast( b_SV );
      x_V = bHYPRE_Vector__cast( x_SV );
      ierr += bHYPRE_StructSMG_Setup( solver_SMG, b_V, x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SMG Solve");
      hypre_BeginTiming(time_index);

      bHYPRE_StructSMG_Apply( solver_SMG, b_V, &x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      ierr += bHYPRE_StructSMG_GetIntValue( solver_SMG, "NumIterations", &num_iterations );
      ierr += bHYPRE_StructSMG_GetDoubleValue( solver_SMG, "RelResidualNorm", &final_res_norm );

      bHYPRE_StructSMG_deleteRef( solver_SMG );
   }


   /*-----------------------------------------------------------
    * Solve the system using PFMG
    *-----------------------------------------------------------*/

   else if ( solver_id == 1 || solver_id == 3 || solver_id == 4 )
   {
      time_index = hypre_InitializeTiming("PFMG Setup");
      hypre_BeginTiming(time_index);


      solver_PFMG = bHYPRE_StructPFMG_Create( bmpicomm, A_b );

      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "MaxIter", 50 );
      bHYPRE_StructPFMG_SetDoubleParameter( solver_PFMG, "Tol", 1.0e-6 );
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "RelChange", 0 );
      /* weighted Jacobi = 1; red-black GS = 2 */
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "RelaxType", relax );
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "NumPrerelax", n_pre );
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "NumPostrelax", n_post );
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "SkipRelax", skip );
      /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "Logging", 1 );
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "PrintLevel", 1 );

      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "rap type", rap );

      b_V = bHYPRE_Vector__cast( b_SV );
      x_V = bHYPRE_Vector__cast( x_SV );
      ierr += bHYPRE_StructPFMG_Setup( solver_PFMG, b_V, x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PFMG Solve");
      hypre_BeginTiming(time_index);

      bHYPRE_StructPFMG_Apply( solver_PFMG, b_V, &x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      ierr += bHYPRE_StructPFMG_GetIntValue( solver_PFMG, "NumIterations", &num_iterations );
      ierr += bHYPRE_StructPFMG_GetDoubleValue( solver_PFMG, "RelResidualNorm", &final_res_norm );

      bHYPRE_StructPFMG_deleteRef( solver_PFMG );
   }

   /*-----------------------------------------------------------
    * Solve the system using SparseMSG
    *-----------------------------------------------------------*/

   else if (solver_id == 2)
   {
      hypre_assert( "solver 2 not implemented"==0 );
#if 0
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
#endif
   }

   /*-----------------------------------------------------------
    * Solve the system using CG
    *-----------------------------------------------------------*/

   /* Conjugate Gradient */
   if ((solver_id > 9) && (solver_id < 20))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);


      A_O = bHYPRE_Operator__cast( A_b );
      solver_PCG = bHYPRE_PCG_Create( bmpicomm, A_O );
      b_V = bHYPRE_Vector__cast( b_SV );
      x_V = bHYPRE_Vector__cast( x_SV );

      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "MaxIter", 50 );
      ierr += bHYPRE_PCG_SetDoubleParameter( solver_PCG, "Tol", 1.0e-06);
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "2-norm", 1);
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "relative change test", 0);
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "Logging", 1);

      if (solver_id == 10)
      {
         /* use symmetric SMG as preconditioner */
         solver_SMG = bHYPRE_StructSMG_Create( bmpicomm, A_b );

         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MemoryUse", 0 );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MaxIter", 1 );
         ierr += bHYPRE_StructSMG_SetDoubleParameter( solver_SMG, "Tol", 0.0 );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "ZeroGuess", 1 );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "RelChange", 0 );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "NumPreRelax", n_pre );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "NumPostRelax", n_post );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "Logging", 0 );

         ierr += bHYPRE_StructSMG_Setup( solver_SMG, b_V, x_V );
         hypre_assert( ierr==0 );

         precond = (bHYPRE_Solver) bHYPRE_StructSMG__cast2
            ( solver_SMG, "bHYPRE.Solver" ); 
      }
      else if ( solver_id == 11 || solver_id == 13 || solver_id == 14 )
      {
         /* use symmetric PFMG as preconditioner */
         solver_PFMG = bHYPRE_StructPFMG_Create( bmpicomm, A_b );

         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "MaxIterations", 1 );
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "Tolerance", 0.0 );
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "ZeroGuess", 1 );
         /* weighted Jacobi = 1; red-black GS = 2 */
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "RelaxType", 1 );
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "NumPreRelax", n_pre );
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "NumPostRelax", n_post );
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "SkipRelax", skip );
         /*bHYPRE_StructPFMG_SetDxyz( solver_PFMG, dxyz);*/
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "Logging", 0 );

         ierr += bHYPRE_StructPFMG_Setup( solver_PFMG, b_V, x_V );
         precond = (bHYPRE_Solver) bHYPRE_StructPFMG__cast2
            ( solver_PFMG, "bHYPRE.Solver" ); 

      }
/* not implemented yet (JfP jan2000) ... */
      else if (solver_id == 12)
      {
         hypre_assert( "solver 12 not implemented"==0 );
         /* use symmetric SparseMSG as preconditioner */
#if 0
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
#endif
      }
/* not implemented yet (JfP jan2000) ... */
      else if (solver_id == 17)
      {
         hypre_assert( "solver not implemented"==0 );
#if 0
         /* use two-step Jacobi as preconditioner */
         HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructJacobiSetMaxIter(precond, 2);
         HYPRE_StructJacobiSetTol(precond, 0.0);
         HYPRE_StructJacobiSetZeroGuess(precond);
         HYPRE_StructPCGSetPrecond(solver,
                                   HYPRE_StructJacobiSolve,
                                   HYPRE_StructJacobiSetup,
                                   precond);
#endif
      }
      else if (solver_id == 17)
      {
         hypre_assert( "solver 17 not implemented"==0 );
#if 0
         /* use two-step Jacobi as preconditioner */
         solver_SJ = bHYPRE_StructJacobi_Constructor( comm );
         precond = (bHYPRE_Solver) bHYPRE_StructJacobi_castTo
            ( solver_SJ, "bHYPRE.Solver" ); 
         bHYPRE_StructJacobi_SetDoubleParameter( solver_SJ, "Tol", 0.0 );
         bHYPRE_StructJacobi_SetIntParameter( solver_SJ, "MaxIter", 2 );
         bHYPRE_StructJacobi_SetIntParameter( solver_SJ, "ZeroGuess", 0 );
      
         bHYPRE_StructJacobi_Setup( solver_SJ, A_LO, b_V, x_V );
#endif
      }
      else if ( solver_id == 18 )
      {
         /* use diagonal scaling as preconditioner */
         solver_DS = bHYPRE_StructDiagScale_Create( bmpicomm, A_b );
         ierr += bHYPRE_StructDiagScale_Setup( solver_DS, b_V, x_V );
         hypre_assert( ierr==0 );

         precond = (bHYPRE_Solver) bHYPRE_StructDiagScale__cast2
            ( solver_DS, "bHYPRE.Solver" ); 
      }
      else if ( solver_id == 19 )
      {
         /* no preconditioner; with PCG we use the "identity preconditioner" */
         solver_Id = bHYPRE_IdentitySolver_Create( bmpicomm );
         ierr += bHYPRE_IdentitySolver_Setup( solver_Id, b_V, x_V );
         hypre_assert( ierr==0 );

         precond = (bHYPRE_Solver) bHYPRE_IdentitySolver__cast2
            ( solver_Id, "bHYPRE.Solver" ); 

      }
      else {
         printf( "Preconditioner not supported! Solver_id=%i\n", solver_id );
      }
      
      bHYPRE_PCG_SetPreconditioner( solver_PCG, precond );

      ierr += bHYPRE_PCG_Setup( solver_PCG, b_V, x_V );

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

      ierr += bHYPRE_PCG_GetIntValue( solver_PCG, "NumIterations", &num_iterations );
      ierr += bHYPRE_PCG_GetDoubleValue( solver_PCG, "RelResidualNorm", &final_res_norm );

      bHYPRE_PCG_deleteRef( solver_PCG );

      if (solver_id == 10)
      {
         bHYPRE_StructSMG_deleteRef( solver_SMG );
      }
      else if ( solver_id == 11 || solver_id == 13 || solver_id == 14 )
      {
         bHYPRE_StructPFMG_deleteRef( solver_PFMG );
      }
      else if (solver_id == 12)
      {
      hypre_assert( "solver not implemented"==0 );
         /*HYPRE_StructSparseMSGDestroy(precond);*/
      }
      else if (solver_id == 17)
      {
         hypre_assert( "solver not implemented"==0 );
         /*bHYPRE_StructJacobi_destructor( solver_SJ );*/
      }
      else if ( solver_id == 18 )
      {
         bHYPRE_StructDiagScale_deleteRef( solver_DS );
      }
      else if ( solver_id == 19 )
      {
         bHYPRE_IdentitySolver_deleteRef( solver_Id );
      }

   }



   /*-----------------------------------------------------------
    * Solve the system using Hybrid
    *-----------------------------------------------------------*/
   if ((solver_id > 19) && (solver_id < 30))
   {
      time_index = hypre_InitializeTiming("Hybrid Setup");
      hypre_BeginTiming(time_index);

      /* The Hybrid scheme is built on top of a PCG solver; so
         make the PCG solver first */

      A_O = bHYPRE_Operator__cast( A_b );
      solver_PCG = bHYPRE_PCG_Create( bmpicomm, A_O );
      b_V = bHYPRE_Vector__cast( b_SV );
      x_V = bHYPRE_Vector__cast( x_SV );

      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "MaxIter", 50 );
      ierr += bHYPRE_PCG_SetDoubleParameter( solver_PCG, "Tol", 1.0e-06);
      ierr += bHYPRE_PCG_SetDoubleParameter( solver_PCG, "ConvergenceFactorTol", 0.90 );
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "2-norm", 1);
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "relative change test", 0);
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "Logging", 1);

      if (solver_id == 20)
      {
         /* use symmetric SMG as preconditioner */
         solver_SMG = bHYPRE_StructSMG_Create( bmpicomm, A_b );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MemoryUse", 0 );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MaxIter", 1 );
         ierr += bHYPRE_StructSMG_SetDoubleParameter(
            solver_SMG, "Tolerance", 0.0 );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "ZeroGuess", 1 );
         ierr += bHYPRE_StructSMG_SetIntParameter(
            solver_SMG, "NumPreRelax", n_pre );
         ierr += bHYPRE_StructSMG_SetIntParameter(
            solver_SMG, "NumPostRelax", n_post );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "Loggin", 0 );

         precond = (bHYPRE_Solver) bHYPRE_StructSMG__cast2(
            solver_SMG, "bHYPRE.Solver" );
         ierr += bHYPRE_PCG_SetPreconditioner( solver_PCG, precond );
      }

      else if (solver_id == 21)
      {
         hypre_assert( "solver 21 not implemented"==0 );
#if 0
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
#endif
      }

      else if (solver_id == 22)
      {
         hypre_assert( "solver 22 not implemented"==0 );
#if 0
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
#endif
      }
      else
         hypre_assert( "solver not implemented"==0 );

      /* Now make the Hybrid solver, and adjust the first
         (diagonal-scaling-preconditioned) solver */

      krylov_solver = (bHYPRE_PreconditionedSolver) bHYPRE_PCG__cast2(
         solver_PCG, "bHYPRE.PreconditionedSolver" );
      solver_Hybrid = bHYPRE_Hybrid_Create( bmpicomm, krylov_solver, A_O );

      /* This Setup call does Setup on the PCG solvers as well. */
      ierr += bHYPRE_Hybrid_Setup( solver_Hybrid, b_V, x_V );

      ierr += bHYPRE_Hybrid_GetFirstSolver( solver_Hybrid, &krylov_solver );
      bHYPRE_PreconditionedSolver_addRef( krylov_solver );
      solver_PCG_1 = (bHYPRE_PCG) bHYPRE_PCG__cast( krylov_solver );
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG_1, "MaxIter", 100 );

      hypre_assert( ierr==0 );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_Hybrid_Apply( solver_Hybrid, b_V, &x_V );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_Hybrid_GetNumIterations( solver_Hybrid, &num_iterations);
      ierr += bHYPRE_Hybrid_GetRelResidualNorm( solver_Hybrid, &final_res_norm);

      bHYPRE_Hybrid_deleteRef( solver_Hybrid );
      if (solver_id == 20)
      {
         bHYPRE_PCG_deleteRef( solver_PCG );
         bHYPRE_PCG_deleteRef( solver_PCG_1 );
      }
#if 0
      else if (solver_id == 21)
      {
         HYPRE_StructPFMGDestroy(precond);
      }
      else if (solver_id == 22)
      {
         HYPRE_StructSparseMSGDestroy(precond);
      }
#endif
   }


   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

#if 0
   bHYPRE_StructVector_Print( x );
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

   for (i = 0; i < nblocks; i++)
   {
      hypre_TFree(iupper[i]);
      hypre_TFree(ilower[i]);
   }
   hypre_TFree(ilower);
   hypre_TFree(iupper);
   hypre_TFree(stencil_indices);

   for ( i = 0; i < (dim + 1); i++)
      hypre_TFree(offsets[i]);
   hypre_TFree(offsets);
   if ( constant_stencil_points != NULL) hypre_TFree(constant_stencil_points);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   bHYPRE_MPICommunicator_deleteRef( bmpicomm );
   MPI_Finalize();

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

  bHYPRE_StructMatrix_GetIntValue( A_b, "ConstantCoefficient",
                                   &constant_coefficient );
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
     for (d = 0; d < dim; d++)
     {
        for (ib = 0; ib < size; ib++)
        {
           values = hypre_CTAlloc(double, vol[ib]);
        
           for (i = 0; i < vol[ib]; i++)
           {
              values[i] = 0.0;
           }

           if( ilower[ib][d] == istart[d] && period[d] == 0 )
           {
              j = iupper[ib][d];
              iupper[ib][d] = istart[d];
              stencil_indices[0] = d;

              bHYPRE_StructMatrix_SetBoxValues
                 ( A_b, ilower[ib], iupper[ib], dim, 1, stencil_indices,
                   values, vol[ib] );
              /* HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                 1, stencil_indices, values);*/
              iupper[ib][d] = j;
           }

           if( iupper[ib][d] == iend[d] && period[d] == 0 )
           {
              j = ilower[ib][d];
              ilower[ib][d] = iend[d];
              stencil_indices[0] = dim + 1 + d;
              bHYPRE_StructMatrix_SetBoxValues
                 ( A_b, ilower[ib], iupper[ib], dim, 1, stencil_indices,
                   values, vol[ib] );
              /* HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                 1, stencil_indices, values);*/
              ilower[ib][d] = j;
           }
           hypre_TFree(values);
        }
     }
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

/******************************************************************************
* Adds values to matrix based on a 7 point (3d) 
* symmetric stencil for a convection-diffusion problem.
* It need an initialized matrix, an assembled grid, and the constants
* that determine the 7 point (3d) convection-diffusion.
******************************************************************************/
int
AddValuesMatrix( bHYPRE_StructMatrix A_b,
                 int dim, int nblocks, int ** ilower, int ** iupper,
                 double cx, double cy, double cz,
                 double conx, double cony, double conz,
                 int symmetric, int constant_coefficient )
{

  int ierr=0;
  int                 i, s, bi;
  double             *values;
  double              east,west;
  double              north,south;
  double              top,bottom;
  double              center;
  int                 volume ;
  int                *stencil_indices;
  int                 stencil_size, size;

  bi=0;

  east = -cx;
  west = -cx;
  north = -cy;
  south = -cy;
  top = -cz;
  bottom = -cz;
  center = 2.0*cx;
  if (dim > 1) center += 2.0*cy;
  if (dim > 2) center += 2.0*cz;

  stencil_size = 1 + (2 - symmetric) * dim;
  stencil_indices = hypre_CTAlloc(int, stencil_size);
  for (s = 0; s < stencil_size; s++)
  {
     stencil_indices[s] = s;
  }

  if(symmetric)
  {
     if ( constant_coefficient==0 )
     {
        for ( bi=0; bi<nblocks; ++bi )
           {
              volume = 1;
              for ( i=0; i < dim; i++ )
              {
                 volume *= ( iupper[bi][i] - ilower[bi][i] + 1 );
              }
              values   = hypre_CTAlloc(double, stencil_size*volume);

              for (i = 0; i < stencil_size*volume; i += stencil_size)
              {
                 switch (dim)
                 {
                 case 1:
                    values[i  ] = west;
                    values[i+1] = center;
                    break;
                 case 2:
                    values[i  ] = west;
                    values[i+1] = south;
                    values[i+2] = center;
                    break;
                 case 3:
                    values[i  ] = west;
                    values[i+1] = south;
                    values[i+2] = bottom;
                    values[i+3] = center;
                    break;
                 }
              }
              size = stencil_size*volume;

              bHYPRE_StructMatrix_SetBoxValues(
                 A_b, ilower[bi], iupper[bi], dim, stencil_size,
                 stencil_indices, values, size );

              hypre_TFree(values);
           }
     }
     else if ( constant_coefficient==1 )
     {
        values   = hypre_CTAlloc(double, stencil_size);
        switch (dim)
        {
        case 1:
           values[0] = west;
           values[1] = center;
           break;
        case 2:
           values[0] = west;
           values[1] = south;
           values[2] = center;
           break;
        case 3:
           values[0] = west;
           values[1] = south;
           values[2] = bottom;
           values[3] = center;
           break;
        }

        bHYPRE_StructMatrix_SetConstantValues(
           A_b, stencil_size,
           stencil_indices, values );

        hypre_TFree(values);
     }
     else
     {
        hypre_assert( constant_coefficient==2 );

        /* stencil index for the center equals dim, so it's easy to leave out */
        values   = hypre_CTAlloc(double, stencil_size-1);
        switch (dim)
        {
        case 1:
           values[0] = west;
           break;
        case 2:
           values[0] = west;
           values[1] = south;
           break;
        case 3:
           values[0] = west;
           values[1] = south;
           values[2] = bottom;
           break;
        }

        bHYPRE_StructMatrix_SetConstantValues(
           A_b, stencil_size-1,
           stencil_indices, values );

        hypre_TFree(values);

        for ( bi=0; bi<nblocks; ++bi )
           {
              volume = 1;
              for ( i=0; i < dim; i++ )
              {
                 volume *= ( iupper[bi][i] - ilower[bi][i] + 1 );
              }
              values   = hypre_CTAlloc(double, volume);

              for ( i=0; i < volume; ++i )
              {
                 values[i] = center;
              }

              bHYPRE_StructMatrix_SetBoxValues(
                 A_b, ilower[bi], iupper[bi], dim, 1,
                 &(stencil_indices[dim]), values, volume );

              hypre_TFree(values);
           }
     }
  }
  else
  {
     if (conx > 0.0)
     {
        west   -= conx;
        center += conx;
     }
     else if (conx < 0.0) 
     {
        east   += conx;
        center -= conx;
     }
     if (cony > 0.0)
     {
        south  -= cony;
        center += cony;
     }
     else if (cony < 0.0) 
     {
        north  += cony;
        center -= cony;
     }
     if (conz > 0.0)
     {
        bottom -= conz;
        center += conz;
     }
     else if (cony < 0.0) 
     {
        top    += conz;
        center -= conz;
     }

     if ( constant_coefficient==0 )
     {
        for ( bi=0; bi<nblocks; ++bi )
           {
              volume = 1;
              for ( i=0; i < dim; i++ )
              {
                 volume *= ( iupper[bi][i] - ilower[bi][i] + 1 );
              }
              values   = hypre_CTAlloc(double, stencil_size*volume);

              for (i = 0; i < stencil_size*volume; i += stencil_size)
              {
                 switch (dim)
                 {
                 case 1:
                    values[i  ] = west;
                    values[i+1] = center;
                    values[i+2] = east;
                    break;
                 case 2:
                    values[i  ] = west;
                    values[i+1] = south;
                    values[i+2] = center;
                    values[i+3] = east;
                    values[i+4] = north;
                    break;
                 case 3:
                    values[i  ] = west;
                    values[i+1] = south;
                    values[i+2] = bottom;
                    values[i+3] = center;
                    values[i+4] = east;
                    values[i+5] = north;
                    values[i+6] = top;
                    break;
                 }
              }
              size = stencil_size*volume;

              bHYPRE_StructMatrix_SetBoxValues(
                 A_b, ilower[bi], iupper[bi], dim, stencil_size,
                 stencil_indices, values, size );

              hypre_TFree(values);
           }
     }
     else if ( constant_coefficient==1 )
     {
        values = hypre_CTAlloc( double, stencil_size );

        switch (dim)
        {
        case 1:
           values[0] = west;
           values[1] = center;
           values[2] = east;
           break;
        case 2:
           values[0] = west;
           values[1] = south;
           values[2] = center;
           values[3] = east;
           values[4] = north;
           break;
        case 3:
           values[0] = west;
           values[1] = south;
           values[2] = bottom;
           values[3] = center;
           values[4] = east;
           values[5] = north;
           values[6] = top;
           break;
        }

        bHYPRE_StructMatrix_SetConstantValues(
           A_b, stencil_size,
           stencil_indices, values );

        hypre_TFree(values);
     }
     else
     {
        hypre_assert( constant_coefficient==2 );
        values = hypre_CTAlloc( double, stencil_size-1 );
        switch (dim)
        {  /* no center in stencil_indices and values */
        case 1:
           stencil_indices[0] = 0;
           stencil_indices[1] = 2;
           values[0] = west;
           values[1] = east;
           break;
        case 2:
           stencil_indices[0] = 0;
           stencil_indices[1] = 1;
           stencil_indices[2] = 3;
           stencil_indices[3] = 4;
           values[0] = west;
           values[1] = south;
           values[2] = east;
           values[3] = north;
           break;
        case 3:
           stencil_indices[0] = 0;
           stencil_indices[1] = 1;
           stencil_indices[2] = 2;
           stencil_indices[3] = 4;
           stencil_indices[4] = 5;
           stencil_indices[5] = 6;
           values[0] = west;
           values[1] = south;
           values[2] = bottom;
           values[3] = east;
           values[4] = north;
           values[5] = top;
           break;
        }

        bHYPRE_StructMatrix_SetConstantValues(
           A_b, stencil_size-1,
           stencil_indices, values );

        hypre_TFree(values);


        /* center is variable */
        stencil_indices[0] = dim; /* refers to center */
        for ( bi=0; bi<nblocks; ++bi )
           {
              volume = 1;
              for ( i=0; i < dim; i++ )
              {
                 volume *= ( iupper[bi][i] - ilower[bi][i] + 1 );
              }
              values   = hypre_CTAlloc(double, volume);

              for ( i=0; i < volume; ++i )
              {
                 values[i] = center;
              }
              size = volume;

              bHYPRE_StructMatrix_SetBoxValues(
                 A_b, ilower[bi], iupper[bi], dim, 1,
                 &(stencil_indices[dim]), values, size );

              hypre_TFree(values);
           }
     }
  }

  hypre_TFree(stencil_indices);

  return ierr;
}

