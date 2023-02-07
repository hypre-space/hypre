/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_krylov.h"

#if defined( KOKKOS_HAVE_MPI )
#include <mpi.h>
#endif

#define HYPRE_MFLOPS 0
#if HYPRE_MFLOPS
#include "_hypre_struct_mv.h"
#endif

/* RDF: Why is this include here? */
#include "_hypre_struct_mv.h"

#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

/* begin lobpcg */

#define NO_SOLVER -9198

#include <time.h>

#include "HYPRE_lobpcg.h"

/* end lobpcg */

HYPRE_Int  SetStencilBndry(HYPRE_StructMatrix A, HYPRE_StructGrid gridmatrix, HYPRE_Int* period);

HYPRE_Int  AddValuesMatrix(HYPRE_StructMatrix A, HYPRE_StructGrid gridmatrix,
                           HYPRE_Real        cx,
                           HYPRE_Real        cy,
                           HYPRE_Real        cz,
                           HYPRE_Real        conx,
                           HYPRE_Real        cony,
                           HYPRE_Real        conz) ;

HYPRE_Int AddValuesVector( hypre_StructGrid  *gridvector,
                           hypre_StructVector *zvector,
                           HYPRE_Int          *period,
                           HYPRE_Real         value  )  ;

/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 *--------------------------------------------------------------------------*/

/*----------------------------------------------------------------------
 * Standard 7-point laplacian in 3D with grid and anisotropy determined
 * as command line arguments.  Do `driver -help' for usage info.
 *----------------------------------------------------------------------*/

hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int           arg_index;
   HYPRE_Int           print_usage;
   HYPRE_Int           nx, ny, nz;
   HYPRE_Int           P, Q, R;
   HYPRE_Int           bx, by, bz;
   HYPRE_Int           px, py, pz;
   HYPRE_Real          cx, cy, cz;
   HYPRE_Real          conx, cony, conz;
   HYPRE_Int           solver_id;

   /*HYPRE_Real          dxyz[3];*/

   HYPRE_Int           A_num_ghost[6] = {0, 0, 0, 0, 0, 0};


   HYPRE_StructMatrix  A;
   HYPRE_StructVector  b;
   HYPRE_StructVector  x;

   HYPRE_StructSolver  solver;

   HYPRE_Int           num_iterations;
   HYPRE_Int           time_index;
   HYPRE_Real          final_res_norm;

   HYPRE_Int           num_procs, myid;

   HYPRE_Int           p, q, r;
   HYPRE_Int           dim;
   HYPRE_Int           n_pre, n_post;
   HYPRE_Int           nblocks ;
   HYPRE_Int           skip;
   HYPRE_Int           sym;
   HYPRE_Int           rap;
   HYPRE_Int           relax;
   HYPRE_Int           jump;
   HYPRE_Int           rep, reps;

   HYPRE_Int         **iupper;
   HYPRE_Int         **ilower;

   HYPRE_Int           istart[3];
   HYPRE_Int           periodic[3];
   HYPRE_Int         **offsets;
   HYPRE_Int           constant_coefficient = 0;
   HYPRE_Int          *stencil_entries;
   HYPRE_Int           stencil_size;
   HYPRE_Int           diag_rank;
   hypre_Index         diag_index;

   HYPRE_StructGrid    grid;
   HYPRE_StructStencil stencil;

   HYPRE_Int           i, s;
   HYPRE_Int           ix, iy, iz, ib;

   HYPRE_Int           periodx0[3] = {0, 0, 0};
   HYPRE_Int           sum;
   HYPRE_Real          inner;

   HYPRE_Int           print_system = 0;

   /* begin lobpcg */

   HYPRE_Int lobpcgFlag = 0;






   /* end lobpcg */

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

#if defined(HYPRE_USING_KOKKOS)
   Kokkos::InitArguments args;
   args.num_threads = 12;
   Kokkos::initialize (args);
#endif

#if defined(HYPRE_USING_CUDA)
   printf("initCudaReductionMemBlock\n");
   initCudaReductionMemBlock();
   printf("Finish initCudaReductionMemBlock\n");
#endif
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );


#ifdef HYPRE_DEBUG
   cegdb(&argc, &argv, myid);
#endif

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   dim = 3;

   skip  = 0;
   sym  = 1;
   rap = 0;
   relax = 1;
   jump  = 0;
   reps = 1;

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

   /* setting defaults for the reading parameters    */
   sum = 0;

   /* ghosts for the building of matrix: default  */
   for (i = 0; i < dim; i++)
   {
      A_num_ghost[2 * i] = 1;
      A_num_ghost[2 * i + 1] = 1;
   }

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
      else if ( strcmp(argv[arg_index], "-istart") == 0 )
      {
         arg_index++;
         istart[0] = atoi(argv[arg_index++]);
         istart[1] = atoi(argv[arg_index++]);
         istart[2] = atoi(argv[arg_index++]);
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
         cx = (HYPRE_Real)atof(argv[arg_index++]);
         cy = (HYPRE_Real)atof(argv[arg_index++]);
         cz = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-convect") == 0 )
      {
         arg_index++;
         conx = (HYPRE_Real)atof(argv[arg_index++]);
         cony = (HYPRE_Real)atof(argv[arg_index++]);
         conz = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-d") == 0 )
      {
         arg_index++;
         dim = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-repeats") == 0 )
      {
         arg_index++;
         reps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;

         /* begin lobpcg */
         if ( strcmp(argv[arg_index], "none") == 0 )
         {
            solver_id = NO_SOLVER;
            arg_index++;
         }
         else /* end lobpcg */
         {
            solver_id = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-v") == 0 )
      {
         arg_index++;
         n_pre = atoi(argv[arg_index++]);
         n_post = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rap") == 0 )
      {
         arg_index++;
         rap = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-relax") == 0 )
      {
         arg_index++;
         relax = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sym") == 0 )
      {
         arg_index++;
         sym = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
         break;
      }
      /* begin lobpcg */
      else if ( strcmp(argv[arg_index], "-lobpcg") == 0 )
      {
         /* use lobpcg */
         arg_index++;
         lobpcgFlag = 1;
      }


      /* end lobpcg */
      else
      {
         arg_index++;
      }
   }

   /* begin lobpcg */

   if ( solver_id == 0 && lobpcgFlag )
   {
      solver_id = 10;
   }

   /*end lobpcg */

   sum = 0;

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/

   if ( (print_usage) && (myid == 0) )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [<options>]\n", argv[0]);
      hypre_printf("\n");
      hypre_printf("  -n <nx> <ny> <nz>   : problem size per block\n");
      hypre_printf("  -istart <istart[0]> <istart[1]> <istart[2]> : start of box\n");
      hypre_printf("  -P <Px> <Py> <Pz>   : processor topology\n");
      hypre_printf("  -b <bx> <by> <bz>   : blocking per processor\n");
      hypre_printf("  -p <px> <py> <pz>   : periodicity in each dimension\n");
      hypre_printf("  -c <cx> <cy> <cz>   : diffusion coefficients\n");
      hypre_printf("  -convect <x> <y> <z>: convection coefficients\n");
      hypre_printf("  -d <dim>            : problem dimension (2 or 3)\n");
      hypre_printf("  -fromfile <name>    : prefix name for matrixfiles\n");
      hypre_printf("  -rhsfromfile <name> : prefix name for rhsfiles\n");
      hypre_printf("  -x0fromfile <name>  : prefix name for firstguessfiles\n");
      hypre_printf("  -repeats <reps>     : number of times to repeat the run, default 1.\n");
      hypre_printf("  -solver <ID>        : solver ID\n");
      hypre_printf("                        0  - axpy\n");
      hypre_printf("                        1  - spMV\n");
      hypre_printf("                        2  - inner product\n");
      hypre_printf("                        3  - spMV with constant coeffs\n");
      hypre_printf("                        4  - spMV with constant coeffs var diag\n");
      hypre_printf("                        8  - Jacobi\n");
      hypre_printf("  -sym <s>            : symmetric storage (1) or not (0)\n");
      hypre_printf("  -jump <num>         : num levels to jump in SparseMSG\n");
      hypre_printf("\n");
   }

   if ( print_usage )
   {
      exit(1);
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) > num_procs)
   {
      if (myid == 0)
      {
         hypre_printf("Error: PxQxR is more than the number of processors\n");
      }
      exit(1);
   }
   else if ((P * Q * R) < num_procs)
   {
      if (myid == 0)
      {
         hypre_printf("Warning: PxQxR is less than the number of processors\n");
      }
   }

   if ((conx != 0.0 || cony != 0 || conz != 0) && sym == 1 )
   {
      if (myid == 0)
      {
         hypre_printf("Warning: Convection produces non-symmetric matrix\n");
      }
      sym = 0;
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0 && sum == 0)
   {
#ifdef HYPRE_USE_DEFAULT
      hypre_printf("Running with openMP macro\n");
#endif
#ifdef HYPRE_USING_KOKKOS
      hypre_printf("Running with Kokkos macro\n");
#endif
#ifdef HYPRE_USING_CUDA
      hypre_printf("Running with CUDA macro\n");
#endif
#ifdef HYPRE_USING_RAJA
      hypre_printf("Running with RAJA macro\n");
#endif

      hypre_printf("Running with these driver parameters:\n");
      hypre_printf("  (nx, ny, nz)    = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("  (istart[0],istart[1],istart[2]) = (%d, %d, %d)\n", \
                   istart[0], istart[1], istart[2]);
      hypre_printf("  (Px, Py, Pz)    = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("  (bx, by, bz)    = (%d, %d, %d)\n", bx, by, bz);
      hypre_printf("  (px, py, pz)    = (%d, %d, %d)\n", px, py, pz);
      hypre_printf("  (cx, cy, cz)    = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("  (conx,cony,conz)= (%f, %f, %f)\n", conx, cony, conz);
      hypre_printf("  (n_pre, n_post) = (%d, %d)\n", n_pre, n_post);
      hypre_printf("  dim             = %d\n", dim);
      hypre_printf("  skip            = %d\n", skip);
      hypre_printf("  sym             = %d\n", sym);
      hypre_printf("  rap             = %d\n", rap);
      hypre_printf("  relax           = %d\n", relax);
      hypre_printf("  jump            = %d\n", jump);
      hypre_printf("  solver ID       = %d\n", solver_id);
   }

   if (myid == 0 && sum > 0)
   {
      hypre_printf("Running with these driver parameters:\n");
      hypre_printf("  (cx, cy, cz)    = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("  (conx,cony,conz)= (%f, %f, %f)\n", conx, cony, conz);
      hypre_printf("  (n_pre, n_post) = (%d, %d)\n", n_pre, n_post);
      hypre_printf("  dim             = %d\n", dim);
      hypre_printf("  skip            = %d\n", skip);
      hypre_printf("  sym             = %d\n", sym);
      hypre_printf("  rap             = %d\n", rap);
      hypre_printf("  relax           = %d\n", relax);
      hypre_printf("  jump            = %d\n", jump);
      hypre_printf("  solver ID       = %d\n", solver_id);
      hypre_printf("  the grid is read from  file \n");

   }

   /*-----------------------------------------------------------
    * Set up the stencil structure (7 points) when matrix is NOT read from file
    * Set up the grid structure used when NO files are read
    *-----------------------------------------------------------*/

   switch (dim)
   {
      case 1:
         nblocks = bx;
         if (sym)
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  2, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
         }
         else
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  3, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
            offsets[2] = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
            offsets[2][0] = 1;
         }
         /* compute p from P and myid */
         p = myid % P;
         break;

      case 2:
         nblocks = bx * by;
         if (sym)
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  3, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[0][1] = 0;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
            offsets[1][1] = -1;
            offsets[2] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[2][0] = 0;
            offsets[2][1] = 0;
         }
         else
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  5, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[0][1] = 0;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
            offsets[1][1] = -1;
            offsets[2] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[2][0] = 0;
            offsets[2][1] = 0;
            offsets[3] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[3][0] = 1;
            offsets[3][1] = 0;
            offsets[4] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[4][0] = 0;
            offsets[4][1] = 1;
         }
         /* compute p,q from P,Q and myid */
         p = myid % P;
         q = (( myid - p) / P) % Q;
         break;

      case 3:
         nblocks = bx * by * bz;
         if (sym)
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  4, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[0][1] = 0;
            offsets[0][2] = 0;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
            offsets[1][1] = -1;
            offsets[1][2] = 0;
            offsets[2] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[2][0] = 0;
            offsets[2][1] = 0;
            offsets[2][2] = -1;
            offsets[3] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[3][0] = 0;
            offsets[3][1] = 0;
            offsets[3][2] = 0;
         }
         else
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  7, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[0][1] = 0;
            offsets[0][2] = 0;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
            offsets[1][1] = -1;
            offsets[1][2] = 0;
            offsets[2] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[2][0] = 0;
            offsets[2][1] = 0;
            offsets[2][2] = -1;
            offsets[3] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[3][0] = 0;
            offsets[3][1] = 0;
            offsets[3][2] = 0;
            offsets[4] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[4][0] = 1;
            offsets[4][1] = 0;
            offsets[4][2] = 0;
            offsets[5] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[5][0] = 0;
            offsets[5][1] = 1;
            offsets[5][2] = 0;
            offsets[6] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[6][0] = 0;
            offsets[6][1] = 0;
            offsets[6][2] = 1;
         }
         /* compute p,q,r from P,Q,R and myid */
         p = myid % P;
         q = (( myid - p) / P) % Q;
         r = ( myid - p - P * q) / ( P * Q );
         break;
   }

   if (myid >= (P * Q * R))
   {
      /* My processor has no data on it */
      nblocks = bx = by = bz = 0;
   }

   /*-----------------------------------------------------------
    * Set up the stencil structure needed for matrix creation
    * which is always the case for read_fromfile_param == 0
    *-----------------------------------------------------------*/

   HYPRE_StructStencilCreate(dim, (2 - sym)*dim + 1, &stencil);
   for (s = 0; s < (2 - sym)*dim + 1; s++)
   {
      HYPRE_StructStencilSetElement(stencil, s, offsets[s]);
   }

   /*-----------------------------------------------------------
    * Set up periodic
    *-----------------------------------------------------------*/

   periodic[0] = px;
   periodic[1] = py;
   periodic[2] = pz;

   /*-----------------------------------------------------------
    * Set up dxyz for PFMG solver
    *-----------------------------------------------------------*/

   /* beginning of sum == 0  */
   if (sum == 0)    /* no read from any file */
   {
      /*-----------------------------------------------------------
       * prepare space for the extents
       *-----------------------------------------------------------*/

      ilower = hypre_CTAlloc(HYPRE_Int*,  nblocks, HYPRE_MEMORY_HOST);
      iupper = hypre_CTAlloc(HYPRE_Int*,  nblocks, HYPRE_MEMORY_HOST);
      for (i = 0; i < nblocks; i++)
      {
         ilower[i] = hypre_CTAlloc(HYPRE_Int,  dim, HYPRE_MEMORY_HOST);
         iupper[i] = hypre_CTAlloc(HYPRE_Int,  dim, HYPRE_MEMORY_HOST);
      }

      /* compute ilower and iupper from (p,q,r), (bx,by,bz), and (nx,ny,nz) */
      ib = 0;
      switch (dim)
      {
         case 1:
            for (ix = 0; ix < bx; ix++)
            {
               ilower[ib][0] = istart[0] + nx * (bx * p + ix);
               iupper[ib][0] = istart[0] + nx * (bx * p + ix + 1) - 1;
               ib++;
            }
            break;
         case 2:
            for (iy = 0; iy < by; iy++)
               for (ix = 0; ix < bx; ix++)
               {
                  ilower[ib][0] = istart[0] + nx * (bx * p + ix);
                  iupper[ib][0] = istart[0] + nx * (bx * p + ix + 1) - 1;
                  ilower[ib][1] = istart[1] + ny * (by * q + iy);
                  iupper[ib][1] = istart[1] + ny * (by * q + iy + 1) - 1;
                  ib++;
               }
            break;
         case 3:
            for (iz = 0; iz < bz; iz++)
               for (iy = 0; iy < by; iy++)
                  for (ix = 0; ix < bx; ix++)
                  {
                     ilower[ib][0] = istart[0] + nx * (bx * p + ix);
                     iupper[ib][0] = istart[0] + nx * (bx * p + ix + 1) - 1;
                     ilower[ib][1] = istart[1] + ny * (by * q + iy);
                     iupper[ib][1] = istart[1] + ny * (by * q + iy + 1) - 1;
                     ilower[ib][2] = istart[2] + nz * (bz * r + iz);
                     iupper[ib][2] = istart[2] + nz * (bz * r + iz + 1) - 1;
                     ib++;
                  }
            break;
      }

      HYPRE_StructGridCreate(hypre_MPI_COMM_WORLD, dim, &grid);
      for (ib = 0; ib < nblocks; ib++)
      {
         /* Add to the grid a new box defined by ilower[ib], iupper[ib]...*/
         HYPRE_StructGridSetExtents(grid, ilower[ib], iupper[ib]);
      }
      HYPRE_StructGridSetPeriodic(grid, periodic);
      HYPRE_StructGridAssemble(grid);

      /*-----------------------------------------------------------
       * Set up the matrix structure
       *-----------------------------------------------------------*/

      for (i = 0; i < dim; i++)
      {
         A_num_ghost[2 * i] = 1;
         A_num_ghost[2 * i + 1] = 1;
      }

      HYPRE_StructMatrixCreate(hypre_MPI_COMM_WORLD, grid, stencil, &A);
      if ( solver_id == 3 || solver_id == 4 ||
           solver_id == 13 || solver_id == 14 )
      {
         stencil_size  = hypre_StructStencilSize(stencil);
         stencil_entries = hypre_CTAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
         if ( solver_id == 3 || solver_id == 13)
         {
            for ( i = 0; i < stencil_size; ++i ) { stencil_entries[i] = i; }
            hypre_StructMatrixSetConstantEntries(
               A, stencil_size, stencil_entries );
            /* ... note: SetConstantEntries is where the constant_coefficient
               flag is set in A */
            hypre_TFree( stencil_entries, HYPRE_MEMORY_HOST);
            constant_coefficient = 1;
         }
         if ( solver_id == 4 || solver_id == 14)
         {
            hypre_SetIndex3(diag_index, 0, 0, 0);
            diag_rank = hypre_StructStencilElementRank( stencil, diag_index );
            hypre_assert( stencil_size >= 1 );
            if ( diag_rank == 0 ) { stencil_entries[diag_rank] = 1; }
            else { stencil_entries[diag_rank] = 0; }
            for ( i = 0; i < stencil_size; ++i )
            {
               if ( i != diag_rank ) { stencil_entries[i] = i; }
            }
            hypre_StructMatrixSetConstantEntries(
               A, stencil_size, stencil_entries );
            hypre_TFree( stencil_entries, HYPRE_MEMORY_HOST);
            constant_coefficient = 2;
         }
      }
      HYPRE_StructMatrixSetSymmetric(A, sym);
      HYPRE_StructMatrixSetNumGhost(A, A_num_ghost);
      //if (nx*ny*nz < 1000)
      //{
      HYPRE_StructGridSetDataLocation(grid, HYPRE_MEMORY_HOST);
      hypre_SetDeviceOff();
      //}
      //else
      //{
      //HYPRE_StructGridSetDataLocation(grid, HYPRE_MEMORY_DEVICE);
      //hypre_SetDeviceOn();
      //}

      HYPRE_StructMatrixInitialize(A);

      /*-----------------------------------------------------------
       * Fill in the matrix elements
       *-----------------------------------------------------------*/

      AddValuesMatrix(A, grid, cx, cy, cz, conx, cony, conz);

      /* Zero out stencils reaching to real boundary */
      /* But in constant coefficient case, no special stencils! */

      if ( constant_coefficient == 0 ) { SetStencilBndry(A, grid, periodic); }
      HYPRE_StructMatrixAssemble(A);

      /*-----------------------------------------------------------
       * Set up the linear system
       *-----------------------------------------------------------*/

      HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, grid, &b);
      HYPRE_StructVectorInitialize(b);

      /*-----------------------------------------------------------
       * For periodic b.c. in all directions, need rhs to satisfy
       * compatibility condition. Achieved by setting a source and
       *  sink of equal strength.  All other problems have rhs = 1.
       *-----------------------------------------------------------*/

      AddValuesVector(grid, b, periodic, 1.0);
      HYPRE_StructVectorAssemble(b);

      HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, grid, &x);
      HYPRE_StructVectorInitialize(x);

      AddValuesVector(grid, x, periodx0, 1.0);
      HYPRE_StructVectorAssemble(x);

      HYPRE_StructGridDestroy(grid);

      for (i = 0; i < nblocks; i++)
      {
         hypre_TFree(iupper[i], HYPRE_MEMORY_HOST);
         hypre_TFree(ilower[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(ilower, HYPRE_MEMORY_HOST);
      hypre_TFree(iupper, HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_StructMatrixPrint("struct.out.A", A, 0);
      HYPRE_StructVectorPrint("struct.out.b", b, 0);
      HYPRE_StructVectorPrint("struct.out.x0", x, 0);
   }

   /*-----------------------------------------------------------
    * axpy
    *-----------------------------------------------------------*/

#if !HYPRE_MFLOPS

   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

   if (solver_id == 0)
   {
      //timeval tstart,tstop;
      time_index = hypre_InitializeTiming("axpy");
      hypre_BeginTiming(time_index);
      //gettimeofday(&tstart,NULL);

      for ( rep = 0; rep < reps; ++rep )
      {
         hypre_StructAxpy(2.0, b, x);
      }
      //gettimeofday(&tstop,NULL);
      hypre_EndTiming(time_index);

      //HYPRE_Real telapsed = (tstop.tv_sec - tstart.tv_sec) + (tstop.tv_usec - tstart.tv_usec)/1e6 ;
      //hypre_printf("axpy, \t %d, \t %f\n",nx*ny*nz,telapsed/reps);
      HYPRE_Real innerb = hypre_StructInnerProd(b, b);
      HYPRE_Real innerx = hypre_StructInnerProd(x, x);
      hypre_printf("inner, \t %f, \t %f\n", innerb, innerx);

      hypre_PrintTiming("Time for axpy", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   }

   /*-----------------------------------------------------------
    * sparse matrix vector multiplication
    *-----------------------------------------------------------*/

   else if ( solver_id == 1 || solver_id == 3 || solver_id == 4 )
   {
      //timeval tstart,tstop;
      void *matvec_data;

      matvec_data = hypre_StructMatvecCreate();
      hypre_StructMatvecSetup(matvec_data, A, x);

      time_index = hypre_InitializeTiming("Mat-Vec");
      hypre_BeginTiming(time_index);
      //gettimeofday(&tstart,NULL);

      for ( rep = 0; rep < reps; ++rep )
      {
         hypre_StructMatvecCompute(matvec_data, 1.0, A, x, 1.0, b);
      }
      //gettimeofday(&tstop,NULL);
      hypre_EndTiming(time_index);
      HYPRE_Real innerb = hypre_StructInnerProd(b, b);
      HYPRE_Real innerx = hypre_StructInnerProd(x, x);
      hypre_printf("inner, \t %f, \t %f\n", innerb, innerx);
      //HYPRE_Real telapsed = (tstop.tv_sec - tstart.tv_sec) + (tstop.tv_usec - tstart.tv_usec)/1e6 ;
      //hypre_printf("Mat-Vec, \t %d, \t %f\n",nx*ny*nz,telapsed/reps);

      hypre_PrintTiming("Time for Mat-Vec", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   }

   /*-----------------------------------------------------------
    * inner product
    *-----------------------------------------------------------*/

   else if (solver_id == 2)
   {
      //timeval tstart,tstop;
      time_index = hypre_InitializeTiming("inner");
      hypre_BeginTiming(time_index);
      //gettimeofday(&tstart,NULL);
      for ( rep = 0; rep < reps; ++rep )
      {
         inner = hypre_StructInnerProd(x, b);
      }
      //gettimeofday(&tstop,NULL);
      hypre_EndTiming(time_index);
      //HYPRE_Real telapsed = (tstop.tv_sec - tstart.tv_sec) + (tstop.tv_usec - tstart.tv_usec)/1e6 ;
      //hypre_printf("inner, \t %d, \t %f\n",nx*ny*nz,telapsed/reps);

      hypre_printf("inner = %f\n", inner);

      hypre_PrintTiming("Time for inner product", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   }

   /*-----------------------------------------------------------
    * Solve the system using Jacobi
    *-----------------------------------------------------------*/

   else if ( solver_id == 8 )
   {
      time_index = hypre_InitializeTiming("Jacobi Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &solver);
      HYPRE_StructJacobiSetMaxIter(solver, 100);
      HYPRE_StructJacobiSetTol(solver, 1.0e-06);
      HYPRE_StructJacobiSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Jacobi Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructJacobiSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructJacobiGetNumIterations(solver, &num_iterations);
      HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructJacobiDestroy(solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using CG
    *-----------------------------------------------------------*/

   if ((solver_id > 9) && (solver_id < 20))
   {
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_StructVectorPrint("struct.out.x", x, 0);
   }
#endif
   /*-----------------------------------------------------------
    * Compute MFLOPs for Matvec
    *-----------------------------------------------------------*/

#if HYPRE_MFLOPS
   {
      void *matvec_data;
      HYPRE_Int   i, imax, N;

      /* compute imax */
      N = (P * nx) * (Q * ny) * (R * nz);
      imax = (5 * 1000000) / N;

      matvec_data = hypre_StructMatvecCreate();
      hypre_StructMatvecSetup(matvec_data, A, x);

      time_index = hypre_InitializeTiming("Matvec");
      hypre_BeginTiming(time_index);

      for (i = 0; i < imax; i++)
      {
         hypre_StructMatvecCompute(matvec_data, 1.0, A, x, 1.0, b);
      }
      /* this counts mult-adds */
      hypre_IncFLOPCount(7 * N * imax);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Matvec time", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      hypre_StructMatvecDestroy(matvec_data);
   }
#endif

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_StructStencilDestroy(stencil);
   HYPRE_StructMatrixDestroy(A);
   HYPRE_StructVectorDestroy(b);
   HYPRE_StructVectorDestroy(x);

   for ( i = 0; i < (dim + 1); i++)
   {
      hypre_TFree(offsets[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(offsets, HYPRE_MEMORY_HOST);

   /* Finalize MPI */
   hypre_MPI_Finalize();
#if defined(HYPRE_USING_KOKKOS)
   Kokkos::finalize ();
#endif
   return (0);
}

/*-------------------------------------------------------------------------
 * add constant values to a vector. Need to pass the initialized vector, grid,
 * period of grid and the constant value.
 *-------------------------------------------------------------------------*/

HYPRE_Int
AddValuesVector( hypre_StructGrid  *gridvector,
                 hypre_StructVector *zvector,
                 HYPRE_Int          *period,
                 HYPRE_Real         value  )
{
   /* #include  "_hypre_struct_mv.h" */
   HYPRE_Int ierr = 0;
   hypre_BoxArray     *gridboxes;
   HYPRE_Int          ib;
   hypre_IndexRef     ilower;
   hypre_IndexRef     iupper;
   hypre_Box          *box;
   HYPRE_Real         *values;
   HYPRE_Int          volume, dim;
#if 0 //defined(HYPRE_USING_CUDA)
   HYPRE_Int          data_location = hypre_StructGridDataLocation(hypre_StructVectorGrid(zvector));
#endif

   gridboxes =  hypre_StructGridBoxes(gridvector);
   dim       =  hypre_StructGridNDim(gridvector);

   ib = 0;
   hypre_ForBoxI(ib, gridboxes)
   {
      box      = hypre_BoxArrayBox(gridboxes, ib);
      volume   =  hypre_BoxVolume(box);
#if 0 //defined(HYPRE_USING_CUDA)
      if (data_location != HYPRE_MEMORY_HOST)
      {
         values   = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_DEVICE);
      }
      else
      {
         values   = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_HOST);
      }
#else
      values   = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_DEVICE);
#endif
      /*-----------------------------------------------------------
       * For periodic b.c. in all directions, need rhs to satisfy
       * compatibility condition. Achieved by setting a source and
       *  sink of equal strength.  All other problems have rhs = 1.
       *-----------------------------------------------------------*/

      if ((dim == 2 && period[0] != 0 && period[1] != 0) ||
          (dim == 3 && period[0] != 0 && period[1] != 0 && period[2] != 0))
      {
         hypre_LoopBegin (volume, i);
         {
            values[i] = 0.0;
            if (i == 0)
            {
               values[0] = value;
               values[volume - 1] = -value;
            }
         }
         hypre_LoopEnd();
      }
      else
      {
         hypre_LoopBegin (volume, i);
         {
            values[i] = value;
         }
         hypre_LoopEnd();
      }

      ilower = hypre_BoxIMin(box);
      iupper = hypre_BoxIMax(box);
      HYPRE_StructVectorSetBoxValues(zvector, ilower, iupper, values);
#if 0 //defined(HYPRE_USING_CUDA)
      if (data_location < 1)
      {
         hypre_TFree(values, HYPRE_MEMORY_DEVICE);
      }
      else
      {
         hypre_TFree(values, HYPRE_MEMORY_HOST);
      }
#else
      hypre_TFree(values, HYPRE_MEMORY_DEVICE);
#endif

   }

   return ierr;
}

/******************************************************************************
 * Adds values to matrix based on a 7 point (3d)
 * symmetric stencil for a convection-diffusion problem.
 * It need an initialized matrix, an assembled grid, and the constants
 * that determine the 7 point (3d) convection-diffusion.
 ******************************************************************************/

HYPRE_Int
AddValuesMatrix(HYPRE_StructMatrix A, HYPRE_StructGrid gridmatrix,
                HYPRE_Real        cx,
                HYPRE_Real        cy,
                HYPRE_Real        cz,
                HYPRE_Real        conx,
                HYPRE_Real        cony,
                HYPRE_Real        conz)
{

   HYPRE_Int ierr = 0;
   hypre_BoxArray     *gridboxes;
   HYPRE_Int           s, bi;
   hypre_IndexRef      ilower;
   hypre_IndexRef      iupper;
   hypre_Box          *box;
   HYPRE_Real         *values;
   HYPRE_Real          east, west;
   HYPRE_Real          north, south;
   HYPRE_Real          top, bottom;
   HYPRE_Real          center;
   HYPRE_Int           volume, dim, sym;
   HYPRE_Int          *stencil_indices;
   HYPRE_Int           stencil_size;
   HYPRE_Int           constant_coefficient;
#if 0 //defined(HYPRE_USING_CUDA)
   HYPRE_Int           data_location = hypre_StructGridDataLocation(hypre_StructMatrixGrid(A));
#endif

   gridboxes =  hypre_StructGridBoxes(gridmatrix);
   dim       =  hypre_StructGridNDim(gridmatrix);
   sym       =  hypre_StructMatrixSymmetric(A);
   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

   bi = 0;

   east = -cx;
   west = -cx;
   north = -cy;
   south = -cy;
   top = -cz;
   bottom = -cz;
   center = 2.0 * cx;
   if (dim > 1) { center += 2.0 * cy; }
   if (dim > 2) { center += 2.0 * cz; }

   stencil_size = 1 + (2 - sym) * dim;
   stencil_indices = hypre_CTAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
   for (s = 0; s < stencil_size; s++)
   {
      stencil_indices[s] = s;
   }

   if (sym)
   {
      if ( constant_coefficient == 0 )
      {
         hypre_ForBoxI(bi, gridboxes)
         {
            box      = hypre_BoxArrayBox(gridboxes, bi);
            volume   =  hypre_BoxVolume(box);
#if 0 //defined(HYPRE_USING_CUDA)
            if (data_location != HYPRE_MEMORY_HOST)
            {
               values     = hypre_CTAlloc(HYPRE_Real, stencil_size * volume, HYPRE_MEMORY_DEVICE);
            }
            else
            {
               values     = hypre_CTAlloc(HYPRE_Real, stencil_size * volume, HYPRE_MEMORY_HOST);
            }
#else
            values     = hypre_CTAlloc(HYPRE_Real, stencil_size * volume, HYPRE_MEMORY_DEVICE);
#endif
            hypre_LoopBegin(volume, d)
            {
               HYPRE_Int i = stencil_size * d;
               switch (dim)
               {
                  case 1:
                     values[i  ] = west;
                     values[i + 1] = center;
                     break;
                  case 2:
                     values[i  ] = west;
                     values[i + 1] = south;
                     values[i + 2] = center;
                     break;
                  case 3:
                     values[i  ] = west;
                     values[i + 1] = south;
                     values[i + 2] = bottom;
                     values[i + 3] = center;
                     break;
               }
            }
            hypre_LoopEnd()

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, stencil_size,
                                           stencil_indices, values);
#if 0 //defined(HYPRE_USING_CUDA)
            if (data_location != HYPRE_MEMORY_HOST)
            {
               hypre_TFree(values, HYPRE_MEMORY_DEVICE);
            }
            else
            {
               hypre_TFree(values, HYPRE_MEMORY_HOST);
            }
#else
            hypre_TFree(values, HYPRE_MEMORY_DEVICE);
#endif
         }
      }
      else if ( constant_coefficient == 1 )
      {
         values   = hypre_CTAlloc(HYPRE_Real,  stencil_size, HYPRE_MEMORY_HOST);
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
         if (hypre_BoxArraySize(gridboxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size,
                                                stencil_indices, values);
         }
         hypre_TFree(values, HYPRE_MEMORY_HOST);
      }
      else
      {
         hypre_assert( constant_coefficient == 2 );

         /* stencil index for the center equals dim, so it's easy to leave out */
         values   = hypre_CTAlloc(HYPRE_Real,  stencil_size - 1, HYPRE_MEMORY_HOST);
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
         if (hypre_BoxArraySize(gridboxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size - 1,
                                                stencil_indices, values);
         }
         hypre_TFree(values, HYPRE_MEMORY_HOST);

         hypre_ForBoxI(bi, gridboxes)
         {
            box      = hypre_BoxArrayBox(gridboxes, bi);
            volume   =  hypre_BoxVolume(box);
#if 0 //defined(HYPRE_USING_CUDA)
            if (data_location != HYPRE_MEMORY_HOST)
            {
               values   = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_DEVICE);
            }
            else
            {
               values   = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_HOST);
            }
#else
            values   = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_DEVICE);
#endif
            hypre_LoopBegin(volume, i)
            {
               values[i] = center;
            }
            hypre_LoopEnd()

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices + dim, values);
#if 0 //defined(HYPRE_USING_CUDA)
            if (data_location == HYPRE_MEMORY_DEVICE)
            {
               hypre_TFree(values, HYPRE_MEMORY_DEVICE);
            }
            else
            {
               hypre_TFree(values, HYPRE_MEMORY_HOST);
            }
#else
            hypre_TFree(values, HYPRE_MEMORY_DEVICE);
#endif
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

      if ( constant_coefficient == 0 )
      {
         hypre_ForBoxI(bi, gridboxes)
         {
            box      = hypre_BoxArrayBox(gridboxes, bi);
            volume   =  hypre_BoxVolume(box);
#if 0 //defined(HYPRE_USING_CUDA)
            if (data_location < 1)
            {
               values   = hypre_CTAlloc(HYPRE_Real, stencil_size * volume, HYPRE_MEMORY_DEVICE);
            }
            else
            {
               values   = hypre_CTAlloc(HYPRE_Real, stencil_size * volume, HYPRE_MEMORY_HOST);
            }
#else
            values   = hypre_CTAlloc(HYPRE_Real, stencil_size * volume, HYPRE_MEMORY_DEVICE);
#endif
            hypre_LoopBegin(volume, d)
            {
               HYPRE_Int i = stencil_size * d;
               switch (dim)
               {
                  case 1:
                     values[i  ] = west;
                     values[i + 1] = center;
                     values[i + 2] = east;
                     break;
                  case 2:
                     values[i  ] = west;
                     values[i + 1] = south;
                     values[i + 2] = center;
                     values[i + 3] = east;
                     values[i + 4] = north;
                     break;
                  case 3:
                     values[i  ] = west;
                     values[i + 1] = south;
                     values[i + 2] = bottom;
                     values[i + 3] = center;
                     values[i + 4] = east;
                     values[i + 5] = north;
                     values[i + 6] = top;
                     break;
               }
            }
            hypre_LoopEnd()

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, stencil_size,
                                           stencil_indices, values);

#if 0 //defined(HYPRE_USING_CUDA)
            if (data_location < 1)
            {
               hypre_TFree(values, HYPRE_MEMORY_DEVICE);
            }
            else
            {
               hypre_TFree(values, HYPRE_MEMORY_HOST);
            }
#else
            hypre_TFree(values, HYPRE_MEMORY_DEVICE);
#endif
         }
      }
      else if ( constant_coefficient == 1 )
      {
         values = hypre_CTAlloc( HYPRE_Real,  stencil_size, HYPRE_MEMORY_HOST);

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

         if (hypre_BoxArraySize(gridboxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size,
                                                stencil_indices, values);
         }

         hypre_TFree(values, HYPRE_MEMORY_HOST);
      }
      else
      {
         hypre_assert( constant_coefficient == 2 );
         values = hypre_CTAlloc( HYPRE_Real,  stencil_size - 1, HYPRE_MEMORY_HOST);
         switch (dim)
         {
            /* no center in stencil_indices and values */
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

         if (hypre_BoxArraySize(gridboxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size,
                                                stencil_indices, values);
         }
         hypre_TFree(values, HYPRE_MEMORY_HOST);


         /* center is variable */
         stencil_indices[0] = dim; /* refers to center */
         hypre_ForBoxI(bi, gridboxes)
         {
            box      = hypre_BoxArrayBox(gridboxes, bi);
            volume   =  hypre_BoxVolume(box);
#if 0 //defined(HYPRE_USING_CUDA)
            if (data_location < 1)
            {
               values   = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_DEVICE);
            }
            else
            {
               values   = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_HOST);
            }
#else
            values   = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_DEVICE);
#endif
            hypre_LoopBegin(volume, i)
            {
               values[i] = center;
            }
            hypre_LoopEnd()

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#if 0 //defined(HYPRE_USING_CUDA)
            if (data_location < 1)
            {
               hypre_TFree(values, HYPRE_MEMORY_DEVICE);
            }
            else
            {
               hypre_TFree(values, HYPRE_MEMORY_HOST);
            }
#else
            hypre_TFree(values, HYPRE_MEMORY_DEVICE);
#endif
         }
      }
   }

   hypre_TFree(stencil_indices, HYPRE_MEMORY_HOST);

   return ierr;
}

/*********************************************************************************
 * this function sets to zero the stencil entries that are on the boundary
 * Grid, matrix and the period are needed.
 *********************************************************************************/

HYPRE_Int
SetStencilBndry(HYPRE_StructMatrix A, HYPRE_StructGrid gridmatrix, HYPRE_Int* period)
{

   HYPRE_Int ierr = 0;
   hypre_BoxArray    *gridboxes;
   HYPRE_Int          size, i, j, d, ib;
   HYPRE_Int        **ilower;
   HYPRE_Int        **iupper;
   HYPRE_Int         *vol;
   HYPRE_Int         *istart, *iend;
   hypre_Box         *box;
   hypre_Box         *dummybox;
   hypre_Box         *boundingbox;
   HYPRE_Real        *values;
   HYPRE_Int          volume, dim;
   HYPRE_Int         *stencil_indices;
   HYPRE_Int          constant_coefficient;
#if 0 //defined(HYPRE_USING_CUDA)
   HYPRE_Int          data_location = hypre_StructGridDataLocation(hypre_StructMatrixGrid(A));
#endif
   gridboxes       = hypre_StructGridBoxes(gridmatrix);
   boundingbox     = hypre_StructGridBoundingBox(gridmatrix);
   istart          = hypre_BoxIMin(boundingbox);
   iend            = hypre_BoxIMax(boundingbox);
   size            = hypre_StructGridNumBoxes(gridmatrix);
   dim             = hypre_StructGridNDim(gridmatrix);
   stencil_indices = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if ( constant_coefficient > 0 ) { return 1; }
   /*...no space dependence if constant_coefficient==1,
     and space dependence only for diagonal if constant_coefficient==2 --
     and this function only touches off-diagonal entries */

   vol    = hypre_CTAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST);
   ilower = hypre_CTAlloc(HYPRE_Int*,  size, HYPRE_MEMORY_HOST);
   iupper = hypre_CTAlloc(HYPRE_Int*,  size, HYPRE_MEMORY_HOST);
   for (i = 0; i < size; i++)
   {
      ilower[i] = hypre_CTAlloc(HYPRE_Int,  dim, HYPRE_MEMORY_HOST);
      iupper[i] = hypre_CTAlloc(HYPRE_Int,  dim, HYPRE_MEMORY_HOST);
   }

   i = 0;
   ib = 0;
   hypre_ForBoxI(i, gridboxes)
   {
      dummybox = hypre_BoxCreate(dim);
      box      = hypre_BoxArrayBox(gridboxes, i);
      volume   =  hypre_BoxVolume(box);
      vol[i]   = volume;
      hypre_CopyBox(box, dummybox);
      for (d = 0; d < dim; d++)
      {
         ilower[ib][d] = hypre_BoxIMinD(dummybox, d);
         iupper[ib][d] = hypre_BoxIMaxD(dummybox, d);
      }
      ib++ ;
      hypre_BoxDestroy(dummybox);
   }

   if ( constant_coefficient == 0 )
   {
      for (d = 0; d < dim; d++)
      {
         for (ib = 0; ib < size; ib++)
         {
#if 0 //defined(HYPRE_USING_CUDA)
            if (data_location < 1)
            {
               values = hypre_CTAlloc(HYPRE_Real, vol[ib], HYPRE_MEMORY_DEVICE);
            }
            else
            {
               values = hypre_CTAlloc(HYPRE_Real, vol[ib], HYPRE_MEMORY_HOST);
            }
#else
            values = hypre_CTAlloc(HYPRE_Real, vol[ib], HYPRE_MEMORY_DEVICE);
#endif
            hypre_LoopBegin(vol[ib], i)
            {
               values[i] = 0.0;
            }
            hypre_LoopEnd()

            if ( ilower[ib][d] == istart[d] && period[d] == 0 )
            {
               j = iupper[ib][d];
               iupper[ib][d] = istart[d];
               stencil_indices[0] = d;
               HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                                              1, stencil_indices, values);
               iupper[ib][d] = j;
            }

            if ( iupper[ib][d] == iend[d] && period[d] == 0 )
            {
               j = ilower[ib][d];
               ilower[ib][d] = iend[d];
               stencil_indices[0] = dim + 1 + d;
               HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                                              1, stencil_indices, values);
               ilower[ib][d] = j;
            }
#if 0 //defined(HYPRE_USING_CUDA)
            if (data_location < 1)
            {
               hypre_TFree(values, HYPRE_MEMORY_DEVICE);
            }
            else
            {
               hypre_TFree(values, HYPRE_MEMORY_HOST);
            }
#else
            hypre_TFree(values, HYPRE_MEMORY_DEVICE);
#endif
         }
      }
   }

   hypre_TFree(vol, HYPRE_MEMORY_HOST);
   hypre_TFree(stencil_indices, HYPRE_MEMORY_HOST);
   for (ib = 0 ; ib < size ; ib++)
   {
      hypre_TFree(ilower[ib], HYPRE_MEMORY_HOST);
      hypre_TFree(iupper[ib], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(ilower, HYPRE_MEMORY_HOST);
   hypre_TFree(iupper, HYPRE_MEMORY_HOST);

   return ierr;
}
