/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.31 $
 ***********************************************************************EHEADER*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"
#include "_hypre_struct_mv.h"
 
#include "bHYPRE_StructMatrix.h"
#include "bHYPRE_StructVector.h"
#include "bHYPRE_Operator.h"
#include "bHYPRE_Solver.h"
#include "bHYPRE_StructSMG.h"
#include "bHYPRE_StructPFMG.h"
#include "bHYPRE_IdentitySolver.h"
#include "bHYPRE_StructDiagScale.h"
#include "bHYPRE_StructJacobi.h"
#include "bHYPRE_PCG.h"
#include "bHYPRE_Hybrid.h"
#include "bHYPRE_StructGrid.h"
#include "bHYPRE_StructStencil.h"
#include "bHYPRE_StructGrid_Impl.h"
#include "sidl_Exception.h"

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

HYPRE_Int SetStencilBndry
( bHYPRE_StructMatrix A_b, bHYPRE_StructGrid grid, HYPRE_Int* periodic );

HYPRE_Int
AddValuesMatrix( bHYPRE_StructMatrix A_b,
                 HYPRE_Int dim, HYPRE_Int nblocks, HYPRE_Int ** ilower, HYPRE_Int ** iupper,
                 double cx, double cy, double cz,
                 double conx, double cony, double conz,
                 HYPRE_Int symmetric, HYPRE_Int constant_coefficient );

HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{
   HYPRE_Int                 ierr;
   HYPRE_Int                 arg_index;
   HYPRE_Int                 print_usage;
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   HYPRE_Int                 bx, by, bz;
   HYPRE_Int                 px, py, pz;
   double              cx, cy, cz;
   double              conx, cony, conz;
   HYPRE_Int                 solver_id;
   HYPRE_Int                 relax, rap;

   HYPRE_Int                 A_num_ghost[6] = {0, 0, 0, 0, 0, 0};
                     
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
   bHYPRE_StructJacobi  solver_SJ;
   bHYPRE_StructSMG solver_SMG;
   bHYPRE_StructPFMG solver_PFMG;
   bHYPRE_IdentitySolver solver_Id;
   bHYPRE_PCG  solver_PCG;
   bHYPRE_PCG  solver_PCG_1;
   bHYPRE_StructDiagScale  solver_DS;
   bHYPRE_Hybrid solver_Hybrid;

   HYPRE_Int constant_coefficient = 0;
   HYPRE_Int symmetric = 1;
   MPI_Comm mpi_comm = hypre_MPI_COMM_WORLD;

   HYPRE_Int                 num_iterations;
   HYPRE_Int                 time_index;
   double              final_res_norm;

   HYPRE_Int                 num_procs, myid;

   HYPRE_Int                 p, q, r;
   HYPRE_Int                 dim;
   HYPRE_Int                 n_pre, n_post;
   HYPRE_Int                 nblocks, volume;
   HYPRE_Int                 skip;
   HYPRE_Int                 jump;

   HYPRE_Int               **iupper;
   HYPRE_Int               **ilower;

   HYPRE_Int                 istart[3];
   HYPRE_Int                 periodic[3];

   HYPRE_Int               **offsets;
   HYPRE_Int               *constant_stencil_points = NULL;

   bHYPRE_StructGrid grid;
   bHYPRE_StructStencil stencil;

   HYPRE_Int                *stencil_indices;
   double             *values;

   HYPRE_Int                 i, s;
/* not currently used   HYPRE_Int                 isave, d;*/
   HYPRE_Int                 ix, iy, iz, ib;

   HYPRE_Int                 periodic_error = 0;
   sidl_BaseInterface _ex = NULL;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
 
   ierr = 0;

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   bmpicomm = bHYPRE_MPICommunicator_CreateC( (void *)(&mpi_comm), &_ex );

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
      hypre_printf("\n");
      hypre_printf("Usage: %s [<options>]\n", argv[0]);
      hypre_printf("\n");
      hypre_printf("  -n <nx> <ny> <nz>    : problem size per block\n");
      hypre_printf("  -P <Px> <Py> <Pz>    : processor topology\n");
      hypre_printf("  -b <bx> <by> <bz>    : blocking per processor\n");
      hypre_printf("  -p <px> <py> <pz>    : periodicity in each dimension\n");
      hypre_printf("  -c <cx> <cy> <cz>    : diffusion coefficients\n");
      hypre_printf("  -v <n_pre> <n_post>  : number of pre and post relaxations\n");
      hypre_printf("  -d <dim>             : problem dimension (2 or 3)\n");
      hypre_printf("  -skip <s>            : skip some relaxation in PFMG (0 or 1)\n");
      hypre_printf("  -jump <num>          : num levels to jump in SparseMSG\n");
      hypre_printf("  -solver <ID>         : solver ID (default = 0)\n");
      hypre_printf("                         0  - SMG\n");
      hypre_printf("                         1  - PFMG\n");
      hypre_printf("                         2 *- SparseMSG\n");
      hypre_printf("                         3  - PFMG constant coefficients\n");
      hypre_printf("                         4  - PFMG constant coefficients variable diagonal\n");
      hypre_printf("                         10 - CG with SMG precond\n");
      hypre_printf("                         11 - CG with PFMG precond\n");
      hypre_printf("                         12*- CG with SparseMSG precond\n");
      hypre_printf("                         13 - CG with PFMG precond, constant coefficients\n");
      hypre_printf("                         14 - CG with PFMG precond, const.coeff.,variable diagonal\n");
      hypre_printf("                         17 - CG with 2-step Jacobi\n");
      hypre_printf("                         18 - CG with diagonal scaling\n");
      hypre_printf("                         19 - CG\n");
      hypre_printf("                         20 - Hybrid with SMG precond\n");
      hypre_printf("                         21*- Hybrid with PFMG precond\n");
      hypre_printf("                         22*- Hybrid with SparseMSG precond\n");
      hypre_printf("Solvers marked with '*' have not yet been implemented.\n");
      hypre_printf("\n");

      bHYPRE_MPICommunicator_deleteRef( bmpicomm, &_ex );
      hypre_MPI_Finalize();
      exit(1);
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   if ((px+py+pz) != 0 && solver_id != 0 )
   {
      hypre_printf("Error: Periodic implemented only for solver 0, SMG \n");
      periodic_error++;
   }
   if (periodic_error != 0)
   {
      exit(1);
   }

   if ((conx != 0.0 || cony !=0 || conz != 0) && symmetric == 1 )
   {
      hypre_printf("\n*** Warning: convection produces non-symetric matrix ***\n\n");
      symmetric = 0;
   }


   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("Running with these driver parameters:\n");
      hypre_printf("  (nx, ny, nz)    = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("  (istart[0],istart[1],istart[2]) = (%d, %d, %d)\n", \
                 istart[0],istart[1],istart[2]);
      hypre_printf("  (Px, Py, Pz)    = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("  (bx, by, bz)    = (%d, %d, %d)\n", bx, by, bz);
      hypre_printf("  (px, py, pz)    = (%d, %d, %d)\n", px, py, pz);
      hypre_printf("  (cx, cy, cz)    = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("  (conx,cony,conz)= (%f, %f, %f)\n", conx, cony, conz);
      hypre_printf("  (n_pre, n_post) = (%d, %d)\n", n_pre, n_post);
      hypre_printf("  dim             = %d\n", dim);
      hypre_printf("  skip            = %d\n", skip);
      hypre_printf("  jump            = %d\n", jump);
      hypre_printf("  solver ID       = %d\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up periodic flags
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("Struct Interface");
   hypre_BeginTiming(time_index);

   periodic[0] = px;
   periodic[1] = py;
   periodic[2] = pz;

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
         stencil_indices = hypre_CTAlloc(HYPRE_Int, 2);
         offsets = hypre_CTAlloc(HYPRE_Int*, 2);
         offsets[0] = hypre_CTAlloc(HYPRE_Int, 1);
         offsets[0][0] = -1; 
         offsets[1] = hypre_CTAlloc(HYPRE_Int, 1);
         offsets[1][0] = 0; 
         if ( solver_id == 3 || solver_id == 13)
            {
               constant_stencil_points = hypre_CTAlloc(HYPRE_Int, 2);
               constant_stencil_points[0] = 0;
               constant_stencil_points[1] = 1;
            }
         if ( solver_id == 4 || solver_id == 14)
            {
               constant_stencil_points = hypre_CTAlloc(HYPRE_Int, 2);
               constant_stencil_points[0] = 0;
               constant_stencil_points[1] = 0;
            }
         /* compute p from P and myid */
         p = myid % P;
         break;
      case 2:
         volume  = nx*ny;
         nblocks = bx*by;
         stencil_indices = hypre_CTAlloc(HYPRE_Int, 3);
         offsets = hypre_CTAlloc(HYPRE_Int*, 3);
         offsets[0] = hypre_CTAlloc(HYPRE_Int, 2);
         offsets[0][0] = -1; 
         offsets[0][1] = 0; 
         offsets[1] = hypre_CTAlloc(HYPRE_Int, 2);
         offsets[1][0] = 0; 
         offsets[1][1] = -1; 
         offsets[2] = hypre_CTAlloc(HYPRE_Int, 2);
         offsets[2][0] = 0; 
         offsets[2][1] = 0; 
         if ( solver_id == 3 || solver_id == 13)
            {
               constant_stencil_points = hypre_CTAlloc(HYPRE_Int, 3);
               constant_stencil_points[0] = 0;
               constant_stencil_points[1] = 1;
               constant_stencil_points[2] = 2;
            }
         if ( solver_id == 4 || solver_id == 14)
            {
               constant_stencil_points = hypre_CTAlloc(HYPRE_Int, 3);
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
         stencil_indices = hypre_CTAlloc(HYPRE_Int, 4);
         offsets = hypre_CTAlloc(HYPRE_Int*, 4);
         offsets[0] = hypre_CTAlloc(HYPRE_Int, 3);
         offsets[0][0] = -1; 
         offsets[0][1] = 0; 
         offsets[0][2] = 0; 
         offsets[1] = hypre_CTAlloc(HYPRE_Int, 3);
         offsets[1][0] = 0; 
         offsets[1][1] = -1; 
         offsets[1][2] = 0; 
         offsets[2] = hypre_CTAlloc(HYPRE_Int, 3);
         offsets[2][0] = 0; 
         offsets[2][1] = 0; 
         offsets[2][2] = -1; 
         offsets[3] = hypre_CTAlloc(HYPRE_Int, 3);
         offsets[3][0] = 0; 
         offsets[3][1] = 0; 
         offsets[3][2] = 0; 
         if ( solver_id == 3 || solver_id == 13)
            {
               constant_stencil_points = hypre_CTAlloc(HYPRE_Int, 4);
               constant_stencil_points[0] = 0;
               constant_stencil_points[1] = 1;
               constant_stencil_points[2] = 2;
               constant_stencil_points[3] = 3;
            }
         if ( solver_id == 4 || solver_id == 14)
            {
               constant_stencil_points = hypre_CTAlloc(HYPRE_Int, 4);
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

   ilower = hypre_CTAlloc(HYPRE_Int*, nblocks);
   iupper = hypre_CTAlloc(HYPRE_Int*, nblocks);
   for (i = 0; i < nblocks; i++)
   {
      ilower[i] = hypre_CTAlloc(HYPRE_Int, dim);
      iupper[i] = hypre_CTAlloc(HYPRE_Int, dim);
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

   grid = bHYPRE_StructGrid_Create( bmpicomm, dim, &_ex );
   for (ib = 0; ib < nblocks; ib++)
   {
      bHYPRE_StructGrid_SetExtents( grid, ilower[ib], iupper[ib], dim, &_ex );
   }

   bHYPRE_StructGrid_SetPeriodic( grid, periodic, dim, &_ex );

   bHYPRE_StructGrid_Assemble( grid, &_ex );

   /*-----------------------------------------------------------
    * Set up the stencil structure
    *-----------------------------------------------------------*/
 
   stencil = bHYPRE_StructStencil_Create( dim, dim+1, &_ex );

   for (s = 0; s < dim + 1; s++)
   {
      bHYPRE_StructStencil_SetElement( stencil, s, offsets[s], dim, &_ex );
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

   A_b = bHYPRE_StructMatrix_Create( bmpicomm, grid, stencil, &_ex );

   ierr += bHYPRE_StructMatrix_SetSymmetric( A_b, symmetric, &_ex );
   ierr += bHYPRE_StructMatrix_SetNumGhost( A_b, A_num_ghost, 2*dim, &_ex );

   if ( solver_id == 3 || solver_id == 4 || solver_id == 13 || solver_id == 14 )
   {
      bHYPRE_StructMatrix_SetConstantEntries( A_b, dim+1, constant_stencil_points, &_ex );
   }

   ierr += bHYPRE_StructMatrix_Initialize( A_b, &_ex );

   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/

   ierr += AddValuesMatrix( A_b, dim, nblocks, ilower, iupper,
                            cx, cy, cz, conx, cony, conz,
                            symmetric, constant_coefficient );

   /* Zero out stencils reaching to real boundary */
   if ( constant_coefficient == 0 ) ierr += SetStencilBndry( A_b, grid, periodic); 

   ierr += bHYPRE_StructMatrix_Assemble( A_b, &_ex );

#if 0
   bHYPRE_StructMatrix_print( A_b, &_ex );
/*   HYPRE_StructMatrixPrint("driver.out.A", A, 0); */
#endif

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(double, volume);

   b_SV = bHYPRE_StructVector_Create( bmpicomm, grid, &_ex );

   ierr += bHYPRE_StructVector_Initialize( b_SV, &_ex );

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
                                                dim, values, volume, &_ex );
   }

   bHYPRE_StructVector_Assemble( b_SV, &_ex );

#if 0
   bHYPRE_StructVector_Print( b_SV, &_ex );
#endif

   x_SV = bHYPRE_StructVector_Create( bmpicomm, grid, &_ex );

   ierr += bHYPRE_StructVector_Initialize( x_SV, &_ex );

   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   for (ib = 0; ib < nblocks; ib++)
   {
      ierr += bHYPRE_StructVector_SetBoxValues( x_SV, ilower[ib], iupper[ib],
                                                dim, values, volume, &_ex );
   }

   bHYPRE_StructVector_Assemble( b_SV, &_ex );

#if 0
   bHYPRE_StructVector_Print( x_SV, &_ex );
#endif
 
   hypre_TFree(values);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Struct Interface", hypre_MPI_COMM_WORLD);
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

      solver_SJ = bHYPRE_StructJacobi_Constructor( comm, &_ex );

      bHYPRE_StructJacobi_SetDoubleParameter( solver_SJ, "tol", 1.0e-4, &_ex );
      bHYPRE_StructJacobi_SetParameterInt( solver_SJ, "max_iter", 500, &_ex );

      bHYPRE_StructJacobi_Setup( solver_SJ, A_LO, b_V, x_V, &_ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Jacobi Solve");
      hypre_BeginTiming(time_index);

      bHYPRE_StructJacobi_Apply( solver_SJ, b_V, &x_V, &_ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
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
                                                         "bHYPRE.Solver", &_ex ); 
      bHYPRE_Solver_GetSystemOperator( solver, &lo_test, &_ex );

      bHYPRE_StructJacobi_destructor( solver_SJ, &_ex );
   }
#endif

   /*-----------------------------------------------------------
    * Solve the system using SMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      time_index = hypre_InitializeTiming("SMG Setup");
      hypre_BeginTiming(time_index);

      solver_SMG = bHYPRE_StructSMG_Create( bmpicomm, A_b, &_ex );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MemoryUse", 0, &_ex );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MaxIter", 50, &_ex );
      bHYPRE_StructSMG_SetDoubleParameter( solver_SMG, "Tol", 1.0e-6, &_ex );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "RelChange", 0, &_ex );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "NumPrerelax", n_pre, &_ex );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "NumPostrelax", n_post, &_ex );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "PrintLevel", 1, &_ex );
      bHYPRE_StructSMG_SetIntParameter( solver_SMG, "Logging", 1, &_ex );

      b_V = bHYPRE_Vector__cast( b_SV, &_ex );
      x_V = bHYPRE_Vector__cast( x_SV, &_ex );
      ierr += bHYPRE_StructSMG_Setup( solver_SMG, b_V, x_V, &_ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SMG Solve");
      hypre_BeginTiming(time_index);

      bHYPRE_StructSMG_Apply( solver_SMG, b_V, &x_V, &_ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      ierr += bHYPRE_StructSMG_GetIntValue( solver_SMG, "NumIterations", &num_iterations, &_ex );
      ierr += bHYPRE_StructSMG_GetDoubleValue( solver_SMG, "RelResidualNorm", &final_res_norm, &_ex );

      bHYPRE_Vector_deleteRef( b_V, &_ex );
      bHYPRE_Vector_deleteRef( x_V, &_ex );
      bHYPRE_StructSMG_deleteRef( solver_SMG, &_ex );
   }


   /*-----------------------------------------------------------
    * Solve the system using PFMG
    *-----------------------------------------------------------*/

   else if ( solver_id == 1 || solver_id == 3 || solver_id == 4 )
   {
      time_index = hypre_InitializeTiming("PFMG Setup");
      hypre_BeginTiming(time_index);


      solver_PFMG = bHYPRE_StructPFMG_Create( bmpicomm, A_b, &_ex );

      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "MaxIter", 50, &_ex );
      bHYPRE_StructPFMG_SetDoubleParameter( solver_PFMG, "Tol", 1.0e-6, &_ex );
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "RelChange", 0, &_ex );
      /* weighted Jacobi = 1; red-black GS = 2 */
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "RelaxType", relax, &_ex );
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "NumPrerelax", n_pre, &_ex );
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "NumPostrelax", n_post, &_ex );
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "SkipRelax", skip, &_ex );
      /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "Logging", 1, &_ex );
      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "PrintLevel", 1, &_ex );

      bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "rap type", rap, &_ex );

      b_V = bHYPRE_Vector__cast( b_SV, &_ex );
      x_V = bHYPRE_Vector__cast( x_SV, &_ex );
      ierr += bHYPRE_StructPFMG_Setup( solver_PFMG, b_V, x_V, &_ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PFMG Solve");
      hypre_BeginTiming(time_index);

      bHYPRE_StructPFMG_Apply( solver_PFMG, b_V, &x_V, &_ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      ierr += bHYPRE_StructPFMG_GetIntValue( solver_PFMG, "NumIterations", &num_iterations, &_ex );
      ierr += bHYPRE_StructPFMG_GetDoubleValue( solver_PFMG, "RelResidualNorm", &final_res_norm, &_ex );

      bHYPRE_Vector_deleteRef( b_V, &_ex );
      bHYPRE_Vector_deleteRef( x_V, &_ex );
      bHYPRE_StructPFMG_deleteRef( solver_PFMG, &_ex );
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

      HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &solver);
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
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SparseMSG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructSparseMSGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
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


      A_O = bHYPRE_Operator__cast( A_b, &_ex );
      solver_PCG = bHYPRE_PCG_Create( bmpicomm, A_O, &_ex );
      bHYPRE_Operator_deleteRef( A_O, &_ex );
      b_V = bHYPRE_Vector__cast( b_SV, &_ex );
      x_V = bHYPRE_Vector__cast( x_SV, &_ex );

      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "MaxIter", 50, &_ex );
      ierr += bHYPRE_PCG_SetDoubleParameter( solver_PCG, "Tol", 1.0e-06, &_ex);
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "2-norm", 1, &_ex );
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "relative change test", 0, &_ex );
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "Logging", 1, &_ex );

      if (solver_id == 10)
      {
         /* use symmetric SMG as preconditioner */
         solver_SMG = bHYPRE_StructSMG_Create( bmpicomm, A_b, &_ex );

         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MemoryUse", 0, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MaxIter", 1, &_ex );
         ierr += bHYPRE_StructSMG_SetDoubleParameter( solver_SMG, "Tol", 0.0, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "ZeroGuess", 1, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "RelChange", 0, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "NumPreRelax", n_pre, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "NumPostRelax", n_post, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "Logging", 0, &_ex );

         ierr += bHYPRE_StructSMG_Setup( solver_SMG, b_V, x_V, &_ex );
         hypre_assert( ierr==0 );

         precond = (bHYPRE_Solver) bHYPRE_StructSMG__cast2
            ( solver_SMG, "bHYPRE.Solver", &_ex ); 
      }
      else if ( solver_id == 11 || solver_id == 13 || solver_id == 14 )
      {
         /* use symmetric PFMG as preconditioner */
         solver_PFMG = bHYPRE_StructPFMG_Create( bmpicomm, A_b, &_ex );

         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "MaxIterations", 1, &_ex );
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "Tolerance", 0.0, &_ex );
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "ZeroGuess", 1, &_ex );
         /* weighted Jacobi = 1; red-black GS = 2 */
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "RelaxType", 1, &_ex );
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "NumPreRelax", n_pre, &_ex );
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "NumPostRelax", n_post, &_ex );
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "SkipRelax", skip, &_ex );
         /*bHYPRE_StructPFMG_SetDxyz( solver_PFMG, dxyz, &_ex );*/
         bHYPRE_StructPFMG_SetIntParameter( solver_PFMG, "Logging", 0, &_ex );

         ierr += bHYPRE_StructPFMG_Setup( solver_PFMG, b_V, x_V, &_ex );
         precond = (bHYPRE_Solver) bHYPRE_StructPFMG__cast2
            ( solver_PFMG, "bHYPRE.Solver", &_ex ); 

      }
/* not implemented yet (JfP jan2000) ... */
      else if (solver_id == 12)
      {
         hypre_assert( "solver 12 not implemented"==0 );
         /* use symmetric SparseMSG as preconditioner */
#if 0
         HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &precond);
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
      else if (solver_id == 17)
      {
         /* use two-step Jacobi as preconditioner */
         solver_SJ = bHYPRE_StructJacobi_Create( bmpicomm, A_b, &_ex );
         ierr += bHYPRE_StructJacobi_SetIntParameter( solver_SJ, "MaxIter", 2, &_ex );
         ierr += bHYPRE_StructJacobi_SetDoubleParameter( solver_SJ, "Tol", 0.0, &_ex );
         ierr += bHYPRE_StructJacobi_SetIntParameter( solver_SJ, "ZeroGuess", 1, &_ex );
         hypre_assert( ierr==0 );
         precond = (bHYPRE_Solver) bHYPRE_StructJacobi__cast2
            ( solver_SJ, "bHYPRE.Solver", &_ex );
      }
      else if ( solver_id == 18 )
      {
         /* use diagonal scaling as preconditioner */
         solver_DS = bHYPRE_StructDiagScale_Create( bmpicomm, A_b, &_ex );
         ierr += bHYPRE_StructDiagScale_Setup( solver_DS, b_V, x_V, &_ex );
         hypre_assert( ierr==0 );

         precond = (bHYPRE_Solver) bHYPRE_StructDiagScale__cast2
            ( solver_DS, "bHYPRE.Solver", &_ex ); 
      }
      else if ( solver_id == 19 )
      {
         /* no preconditioner; with PCG we use the "identity preconditioner" */
         solver_Id = bHYPRE_IdentitySolver_Create( bmpicomm, &_ex );
         ierr += bHYPRE_IdentitySolver_Setup( solver_Id, b_V, x_V, &_ex );
         hypre_assert( ierr==0 );

         precond = (bHYPRE_Solver) bHYPRE_IdentitySolver__cast2
            ( solver_Id, "bHYPRE.Solver", &_ex );

      }
      else {
         hypre_printf( "Preconditioner not supported! Solver_id=%i\n", solver_id );
      }
      
      bHYPRE_PCG_SetPreconditioner( solver_PCG, precond, &_ex );

      ierr += bHYPRE_PCG_Setup( solver_PCG, b_V, x_V, &_ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      bHYPRE_PCG_Apply( solver_PCG, b_V, &x_V, &_ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_PCG_GetIntValue( solver_PCG, "NumIterations", &num_iterations, &_ex );
      ierr += bHYPRE_PCG_GetDoubleValue( solver_PCG, "RelResidualNorm", &final_res_norm, &_ex );

      bHYPRE_Vector_deleteRef( b_V, &_ex );
      bHYPRE_Vector_deleteRef( x_V, &_ex );
      bHYPRE_PCG_deleteRef( solver_PCG, &_ex );
      bHYPRE_Solver_deleteRef( precond, &_ex );
      if (solver_id == 10)
      {
         bHYPRE_StructSMG_deleteRef( solver_SMG, &_ex );
      }
      else if ( solver_id == 11 || solver_id == 13 || solver_id == 14 )
      {
         bHYPRE_StructPFMG_deleteRef( solver_PFMG, &_ex );
      }
      else if (solver_id == 12)
      {
      hypre_assert( "solver not implemented"==0 );
         /*HYPRE_StructSparseMSGDestroy(precond);*/
      }
      else if (solver_id == 17)
      {
         bHYPRE_StructJacobi_deleteRef( solver_SJ, &_ex );
      }
      else if ( solver_id == 18 )
      {
         bHYPRE_StructDiagScale_deleteRef( solver_DS, &_ex );
      }
      else if ( solver_id == 19 )
      {
         bHYPRE_IdentitySolver_deleteRef( solver_Id, &_ex );
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

      A_O = bHYPRE_Operator__cast( A_b, &_ex );
      solver_PCG = bHYPRE_PCG_Create( bmpicomm, A_O, &_ex );
      b_V = bHYPRE_Vector__cast( b_SV, &_ex );
      x_V = bHYPRE_Vector__cast( x_SV, &_ex );

      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "MaxIter", 50, &_ex );
      ierr += bHYPRE_PCG_SetDoubleParameter( solver_PCG, "Tol", 1.0e-06, &_ex );
      ierr += bHYPRE_PCG_SetDoubleParameter( solver_PCG, "ConvergenceFactorTol", 0.90, &_ex );
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "2-norm", 1, &_ex );
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "relative change test", 0, &_ex );
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG, "Logging", 1, &_ex );

      if (solver_id == 20)
      {
         /* use symmetric SMG as preconditioner */
         solver_SMG = bHYPRE_StructSMG_Create( bmpicomm, A_b, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MemoryUse", 0, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "MaxIter", 1, &_ex );
         ierr += bHYPRE_StructSMG_SetDoubleParameter(
            solver_SMG, "Tolerance", 0.0, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "ZeroGuess", 1, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter(
            solver_SMG, "NumPreRelax", n_pre, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter(
            solver_SMG, "NumPostRelax", n_post, &_ex );
         ierr += bHYPRE_StructSMG_SetIntParameter( solver_SMG, "Logging", 0, &_ex );

         precond = (bHYPRE_Solver) bHYPRE_StructSMG__cast2(
            solver_SMG, "bHYPRE.Solver", &_ex );
         ierr += bHYPRE_PCG_SetPreconditioner( solver_PCG, precond, &_ex );
      }

      else if (solver_id == 21)
      {
         hypre_assert( "solver 21 not implemented"==0 );
#if 0
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
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
         HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &precond);
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
         solver_PCG, "bHYPRE.PreconditionedSolver", &_ex );
      solver_Hybrid = bHYPRE_Hybrid_Create( bmpicomm, krylov_solver, A_O, &_ex );
      bHYPRE_PreconditionedSolver_deleteRef( krylov_solver, &_ex );

      /* This Setup call does Setup on the PCG solvers as well. */
      ierr += bHYPRE_Hybrid_Setup( solver_Hybrid, b_V, x_V, &_ex );

      ierr += bHYPRE_Hybrid_GetFirstSolver( solver_Hybrid, &krylov_solver, &_ex );
      solver_PCG_1 = (bHYPRE_PCG) bHYPRE_PCG__cast( krylov_solver, &_ex );
      ierr += bHYPRE_PCG_SetIntParameter( solver_PCG_1, "MaxIter", 100, &_ex );
      bHYPRE_PCG_deleteRef( solver_PCG_1, &_ex );

      hypre_assert( ierr==0 );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_Hybrid_Apply( solver_Hybrid, b_V, &x_V, &_ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_Hybrid_GetNumIterations( solver_Hybrid, &num_iterations, &_ex );
      ierr += bHYPRE_Hybrid_GetRelResidualNorm( solver_Hybrid, &final_res_norm, &_ex );

      bHYPRE_Operator_deleteRef( A_O, &_ex );
      bHYPRE_Vector_deleteRef( b_V, &_ex );
      bHYPRE_Vector_deleteRef( x_V, &_ex );
      bHYPRE_Hybrid_deleteRef( solver_Hybrid, &_ex );
      bHYPRE_Solver_deleteRef( precond, &_ex );
      bHYPRE_PCG_deleteRef( solver_PCG, &_ex );
      if (solver_id == 20)
      {
         bHYPRE_StructSMG_deleteRef( solver_SMG, &_ex );
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
   bHYPRE_StructVector_Print( x, &_ex );
#endif

   if (myid == 0)
   {
      hypre_printf("\n");
      hypre_printf("Iterations = %d\n", num_iterations);
      hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
      hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   bHYPRE_StructStencil_deleteRef( stencil, &_ex );
   bHYPRE_StructGrid_deleteRef( grid, &_ex );
   bHYPRE_StructMatrix_deleteRef( A_b, &_ex );
   bHYPRE_StructVector_deleteRef( b_SV, &_ex );
   bHYPRE_StructVector_deleteRef( x_SV, &_ex );

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
   bHYPRE_MPICommunicator_deleteRef( bmpicomm, &_ex );
   hypre_MPI_Finalize();

   return (0);
}

/*********************************************************************************
 * this function sets to zero the stencil entries that are on the boundary
 * Grid, matrix and the period are needed. 
 *********************************************************************************/ 

HYPRE_Int SetStencilBndry
( bHYPRE_StructMatrix A_b, bHYPRE_StructGrid grid, HYPRE_Int* period )
{
  HYPRE_Int ierr=0;
  hypre_BoxArray    *gridboxes;
  HYPRE_Int                size,i,j,d,ib;
  HYPRE_Int              **ilower;
  HYPRE_Int              **iupper;
  HYPRE_Int               *vol;
  HYPRE_Int               *istart, *iend;
  hypre_Box         *box;
  hypre_Box         *dummybox;
  hypre_Box         *boundingbox;
  double            *values;
  HYPRE_Int                volume, dim;
  HYPRE_Int               *stencil_indices;
  HYPRE_Int                constant_coefficient;
  sidl_BaseInterface _ex = NULL;

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
  stencil_indices = hypre_CTAlloc(HYPRE_Int, 1);

  bHYPRE_StructMatrix_GetIntValue( A_b, "ConstantCoefficient",
                                   &constant_coefficient, &_ex );
  if ( constant_coefficient>0 ) return 1;
  /*...no space dependence if constant_coefficient==1,
    and space dependence only for diagonal if constant_coefficient==2 --
    and this function only touches off-diagonal entries */

  vol    = hypre_CTAlloc(HYPRE_Int, size);
  ilower = hypre_CTAlloc(HYPRE_Int*, size);
  iupper = hypre_CTAlloc(HYPRE_Int*, size);
  for (i = 0; i < size; i++)
  {
     ilower[i] = hypre_CTAlloc(HYPRE_Int, dim);
     iupper[i] = hypre_CTAlloc(HYPRE_Int, dim);
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
                   values, vol[ib], &_ex );
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
                   values, vol[ib], &_ex );
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
HYPRE_Int
AddValuesMatrix( bHYPRE_StructMatrix A_b,
                 HYPRE_Int dim, HYPRE_Int nblocks, HYPRE_Int ** ilower, HYPRE_Int ** iupper,
                 double cx, double cy, double cz,
                 double conx, double cony, double conz,
                 HYPRE_Int symmetric, HYPRE_Int constant_coefficient )
{

  HYPRE_Int ierr=0;
  HYPRE_Int                 i, s, bi;
  double             *values;
  double              east,west;
  double              north,south;
  double              top,bottom;
  double              center;
  HYPRE_Int                 volume ;
  HYPRE_Int                *stencil_indices;
  HYPRE_Int                 stencil_size, size;
  sidl_BaseInterface _ex = NULL;

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
  stencil_indices = hypre_CTAlloc(HYPRE_Int, stencil_size);
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
                 stencil_indices, values, size, &_ex );

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
           stencil_indices, values, &_ex );

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
           stencil_indices, values, &_ex );

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
                 &(stencil_indices[dim]), values, volume, &_ex );

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
                 stencil_indices, values, size, &_ex );

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
           stencil_indices, values, &_ex );

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
           stencil_indices, values, &_ex );

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
                 &(stencil_indices[dim]), values, size, &_ex );

              hypre_TFree(values);
           }
     }
  }

  hypre_TFree(stencil_indices);

  return ierr;
}

