/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_struct_mv.h"
#include "HYPRE_struct_mv_mp.h"
#include "HYPRE_struct_ls.h"
#include "_hypre_struct_ls.h"
#include "HYPRE_krylov.h"

#define HYPRE_MFLOPS 0
#if HYPRE_MFLOPS
#include "_hypre_struct_mv.h"
#endif

/* RDF: Why is this include here? */
//#include "_hypre_struct_mv.h"

#ifdef HYPRE_DEBUG
/*#include <cegdb.h>*/
#endif

HYPRE_Int  SetStencilBndry_mp(HYPRE_StructMatrix A_dbl, HYPRE_StructMatrix A_flt,
                              HYPRE_StructGrid gridmatrix, HYPRE_Int* period);

HYPRE_Int  AddValuesMatrix_flt(HYPRE_StructMatrix A, HYPRE_StructGrid gridmatrix,
                               float             cx,
                               float             cy,
                               float             cz,
                               float             conx,
                               float             cony,
                               float             conz) ;

HYPRE_Int  AddValuesMatrix_dbl(HYPRE_StructMatrix A, HYPRE_StructGrid gridmatrix,
                               double            cx,
                               double            cy,
                               double            cz,
                               double            conx,
                               double            cony,
                               double            conz) ;

HYPRE_Int AddValuesVector_flt( hypre_StructGrid  *gridvector,
                               hypre_StructVector *zvector,
                               HYPRE_Int          *period,
                               float             value  )  ;

HYPRE_Int AddValuesVector_dbl( hypre_StructGrid  *gridvector,
                               hypre_StructVector *zvector,
                               HYPRE_Int          *period,
                               double             value  )  ;


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
   HYPRE_Int           print_level = 0;
   HYPRE_Int           nx, ny, nz;
   HYPRE_Int           P, Q, R;
   HYPRE_Int           bx, by, bz;
   HYPRE_Int           px, py, pz;
   double              cx, cy, cz;
   double              conx, cony, conz;
   HYPRE_Int           solver_id;
   HYPRE_Int           solver_type;
   HYPRE_Int           recompute_res;
   HYPRE_Int           flex = 0;

   /*HYPRE_Real          dxyz[3];*/

   HYPRE_Int           num_ghost[6]   = {0, 0, 0, 0, 0, 0};
   HYPRE_Int           A_num_ghost[6] = {0, 0, 0, 0, 0, 0};
   HYPRE_Int           v_num_ghost[6] = {0, 0, 0, 0, 0, 0};

   HYPRE_StructMatrix  A_dbl;
   HYPRE_StructVector  b_dbl;
   HYPRE_StructVector  x_dbl;
   HYPRE_StructMatrix  A_flt;
   HYPRE_StructVector  b_flt;
   HYPRE_StructVector  x_flt;

   HYPRE_StructSolver  solver;
   HYPRE_StructSolver  precond;
   HYPRE_Int           num_iterations;
   HYPRE_Int           time_index;
   double              final_res_norm;
   double              tol;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           device_id = -1;
   HYPRE_Int           lazy_device_init = 0;

   HYPRE_Int           p, q, r;
   HYPRE_Int           dim;
   HYPRE_Int           n_pre, n_post;
   HYPRE_Int           nblocks = 0;
   HYPRE_Int           skip;
   HYPRE_Int           sym;
   HYPRE_Int           rap;
   HYPRE_Int           relax;
   double              jacobi_weight;
   HYPRE_Int           usr_jacobi_weight;
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
   HYPRE_StructGrid    readgrid;
   HYPRE_StructStencil stencil;

   HYPRE_Int           i, s;
   HYPRE_Int           ix, iy, iz, ib;

   HYPRE_Int           read_fromfile_param;
   HYPRE_Int           read_fromfile_index;
   HYPRE_Int           read_rhsfromfile_param;
   HYPRE_Int           read_rhsfromfile_index;
   HYPRE_Int           read_x0fromfile_param;
   HYPRE_Int           read_x0fromfile_index;
   HYPRE_Int           periodx0[3] = {0, 0, 0};
   HYPRE_Int          *readperiodic;
   HYPRE_Int           sum;

   HYPRE_Int           print_system = 0;
#if defined(HYPRE_USING_MEMORY_TRACKER)
   HYPRE_Int           print_mem_tracker = 0;
   char                mem_tracker_name[HYPRE_MAX_FILE_NAME_LEN] = {0};
#endif

   /* default execution policy and memory space */
#if defined(HYPRE_TEST_USING_HOST)
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_HOST;
#else
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_DEVICE;
#endif

   //HYPRE_Int device_level = -2;
   int precision = 0; /* 0=dbl, 1=flt, 2=ldbl*/
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   dim = 3;

   skip  = 0;
   sym  = 1;
   rap = 0;
   relax = 1;
   jacobi_weight = 1.0;
   usr_jacobi_weight = 0;
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
   solver_type = 1;
   recompute_res = 1;   /* What should be the default here? */

   istart[0] = -3;
   istart[1] = -3;
   istart[2] = -3;

   px = 0;
   py = 0;
   pz = 0;

   tol = 1.e-8;

   /* setting defaults for the reading parameters    */
   read_fromfile_param = 0;
   read_fromfile_index = argc;
   read_rhsfromfile_param = 0;
   read_rhsfromfile_index = argc;
   read_x0fromfile_param = 0;
   read_x0fromfile_index = argc;
   sum = 0;

   /* ghost defaults */
   for (i = 0; i < 2 * dim; i++)
   {
      num_ghost[i]   = 1;
      A_num_ghost[i] = num_ghost[i];
      v_num_ghost[i] = num_ghost[i];
   }

   //device_level = nx*ny*nz;
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
         //device_level = nx*ny*nz;
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
         cx = (double)atof(argv[arg_index++]);
         cy = (double)atof(argv[arg_index++]);
         cz = (double)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-convect") == 0 )
      {
         arg_index++;
         conx = (double)atof(argv[arg_index++]);
         cony = (double)atof(argv[arg_index++]);
         conz = (double)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-d") == 0 )
      {
         arg_index++;
         dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         read_fromfile_param = 1;
         read_fromfile_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         read_rhsfromfile_param = 1;
         read_rhsfromfile_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-x0fromfile") == 0 )
      {
         arg_index++;
         read_x0fromfile_param = 1;
         read_x0fromfile_index = arg_index;
      }
      else if (strcmp(argv[arg_index], "-repeats") == 0 )
      {
         arg_index++;
         reps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;

         solver_id = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         jacobi_weight = (double)atof(argv[arg_index++]);
         usr_jacobi_weight = 1; /* flag user weight */
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
      else if ( strcmp(argv[arg_index], "-solver_type") == 0 )
      {
         arg_index++;
         solver_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-recompute") == 0 )
      {
         arg_index++;
         recompute_res = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-flex") == 0 )
      {
         arg_index++;
         flex = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol = (double)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-print_lvl") == 0 )
      {
         arg_index++;
         print_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
         break;
      }
      else if ( strcmp(argv[arg_index], "-memory_host") == 0 )
      {
         arg_index++;
         memory_location = HYPRE_MEMORY_HOST;
      }
      else if ( strcmp(argv[arg_index], "-memory_device") == 0 )
      {
         arg_index++;
         memory_location = HYPRE_MEMORY_DEVICE;
      }
      else if ( strcmp(argv[arg_index], "-exec_host") == 0 )
      {
         arg_index++;
         default_exec_policy = HYPRE_EXEC_HOST;
      }
      else if ( strcmp(argv[arg_index], "-exec_device") == 0 )
      {
         arg_index++;
         default_exec_policy = HYPRE_EXEC_DEVICE;
      }
      else if ( strcmp(argv[arg_index], "-double") == 0 )
      {
         arg_index++;
         precision = 0;
      }
      else if ( strcmp(argv[arg_index], "-single") == 0 )
      {
         arg_index++;
         precision = 1;
      }
      else
      {
         arg_index++;
      }
   }

   /* default memory location */
   //HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   //HYPRE_SetExecutionPolicy(default_exec_policy);

   sum = read_x0fromfile_param + read_rhsfromfile_param + read_fromfile_param;

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/

   if ( (print_usage) && (myid == 0) )
   {
      hypre_printf_dbl("\n");
      hypre_printf_dbl("Usage: %s [<options>]\n", argv[0]);
      hypre_printf_dbl("\n");
      hypre_printf_dbl("  -n <nx> <ny> <nz>   : problem size per block\n");
      hypre_printf_dbl("  -istart <istart[0]> <istart[1]> <istart[2]> : start of box\n");
      hypre_printf_dbl("  -P <Px> <Py> <Pz>   : processor topology\n");
      hypre_printf_dbl("  -b <bx> <by> <bz>   : blocking per processor\n");
      hypre_printf_dbl("  -p <px> <py> <pz>   : periodicity in each dimension\n");
      hypre_printf_dbl("  -c <cx> <cy> <cz>   : diffusion coefficients\n");
      hypre_printf_dbl("  -convect <x> <y> <z>: convection coefficients\n");
      hypre_printf_dbl("  -d <dim>            : problem dimension (2 or 3)\n");
      hypre_printf_dbl("  -fromfile <name>    : prefix name for matrixfiles\n");
      hypre_printf_dbl("  -rhsfromfile <name> : prefix name for rhsfiles\n");
      hypre_printf_dbl("  -x0fromfile <name>  : prefix name for firstguessfiles\n");
      hypre_printf_dbl("  -repeats <reps>     : number of times to repeat the run, default 1.  For solver 0,1,3\n");
      hypre_printf_dbl("  -solver <ID>        : solver ID\n");
      hypre_printf_dbl("                        0  - SMG (default)\n");
      hypre_printf_dbl("                        1  - PFMG\n");
      hypre_printf_dbl("                        10 - CG with SMG precond\n");
      hypre_printf_dbl("                        11 - CG with PFMG precond\n");
      hypre_printf_dbl("                        15 - CG with single-prec SMG precond\n");
      hypre_printf_dbl("                        16 - CG with single-prec PFMG precond\n");
      hypre_printf_dbl("                        18 - CG with diagonal scaling\n");
      hypre_printf_dbl("                        19 - CG\n");
      hypre_printf_dbl("                        30 - GMRES with SMG precond\n");
      hypre_printf_dbl("                        31 - GMRES with PFMG precond\n");
      hypre_printf_dbl("                        35 - GMRES with single-prec SMG precond\n");
      hypre_printf_dbl("                        36 - GMRES with single-prec PFMG precond\n");
      hypre_printf_dbl("                        38 - GMRES with diagonal scaling\n");
      hypre_printf_dbl("                        39 - GMRES\n");
      hypre_printf_dbl("                        40 - BiCGSTAB with SMG precond\n");
      hypre_printf_dbl("                        41 - BiCGSTAB with PFMG precond\n");
      hypre_printf_dbl("                        45 - BiCGSTAB with single-prec SMG precond\n");
      hypre_printf_dbl("                        46 - BiCGSTAB with single-prec PFMG precond\n");
      hypre_printf_dbl("                        48 - BiCGSTAB with diagonal scaling\n");
      hypre_printf_dbl("                        49 - BiCGSTAB\n");
      hypre_printf_dbl("  -v <n_pre> <n_post> : number of pre and post relaxations\n");
      hypre_printf_dbl("  -rap <r>            : coarse grid operator type\n");
      hypre_printf_dbl("                        0 - Galerkin (default)\n");
      hypre_printf_dbl("                        1 - non-Galerkin ParFlow operators\n");
      hypre_printf_dbl("                        2 - Galerkin, general operators\n");
      hypre_printf_dbl("  -relax <r>          : relaxation type\n");
      hypre_printf_dbl("                        0 - Jacobi\n");
      hypre_printf_dbl("                        1 - Weighted Jacobi (default)\n");
      hypre_printf_dbl("                        2 - R/B Gauss-Seidel\n");
      hypre_printf_dbl("                        3 - R/B Gauss-Seidel (nonsymmetric)\n");
      hypre_printf_dbl("  -w <jacobi weight>  : jacobi weight\n");
      hypre_printf_dbl("  -skip <s>           : skip levels in PFMG (0 or 1)\n");
      hypre_printf_dbl("  -sym <s>            : symmetric storage (1) or not (0)\n");
      hypre_printf_dbl("  -solver_type <ID>   : solver type for Hybrid\n");
      hypre_printf_dbl("                        1 - PCG (default)\n");
      hypre_printf_dbl("                        2 - GMRES\n");
      hypre_printf_dbl("  -recompute <bool>   : Recompute residual in PCG?\n");
      hypre_printf_dbl("  -cf <cf>            : convergence factor for Hybrid\n");
      hypre_printf_dbl("\n");
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
         hypre_printf_dbl("Error: PxQxR is more than the number of processors\n");
      }
      exit(1);
   }
   else if ((P * Q * R) < num_procs)
   {
      if (myid == 0)
      {
         hypre_printf_dbl("Warning: PxQxR is less than the number of processors\n");
      }
   }

   if ((conx != 0.0 || cony != 0 || conz != 0) && sym == 1 )
   {
      if (myid == 0)
      {
         hypre_printf_dbl("Warning: Convection produces non-symmetric matrix\n");
      }
      sym = 0;
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0 && sum == 0)
   {
#if defined(HYPRE_DEVELOP_STRING) && defined(HYPRE_DEVELOP_BRANCH)
      hypre_printf_dbl("\nUsing HYPRE_DEVELOP_STRING: %s (branch %s; the develop branch)\n\n",
                       HYPRE_DEVELOP_STRING, HYPRE_DEVELOP_BRANCH);

#elif defined(HYPRE_DEVELOP_STRING) && !defined(HYPRE_DEVELOP_BRANCH)
      hypre_printf_dbl("\nUsing HYPRE_DEVELOP_STRING: %s (branch %s; not the develop branch)\n\n",
                       HYPRE_DEVELOP_STRING, HYPRE_BRANCH_NAME);

#elif defined(HYPRE_RELEASE_VERSION)
      hypre_printf_dbl("\nUsing HYPRE_RELEASE_VERSION: %s\n\n",
                       HYPRE_RELEASE_VERSION);
#endif

      hypre_printf_dbl("Running with these driver parameters:\n");
      hypre_printf_dbl("  (nx, ny, nz)    = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf_dbl("  (istart[0],istart[1],istart[2]) = (%d, %d, %d)\n", \
                       istart[0], istart[1], istart[2]);
      hypre_printf_dbl("  (Px, Py, Pz)    = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf_dbl("  (bx, by, bz)    = (%d, %d, %d)\n", bx, by, bz);
      hypre_printf_dbl("  (px, py, pz)    = (%d, %d, %d)\n", px, py, pz);
      hypre_printf_dbl("  (cx, cy, cz)    = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf_dbl("  (conx,cony,conz)= (%f, %f, %f)\n", conx, cony, conz);
      hypre_printf_dbl("  (n_pre, n_post) = (%d, %d)\n", n_pre, n_post);
      hypre_printf_dbl("  dim             = %d\n", dim);
      hypre_printf_dbl("  skip            = %d\n", skip);
      hypre_printf_dbl("  sym             = %d\n", sym);
      hypre_printf_dbl("  rap             = %d\n", rap);
      hypre_printf_dbl("  relax           = %d\n", relax);
      hypre_printf_dbl("  solver ID       = %d\n", solver_id);
      /* hypre_printf_dbl("  Device level    = %d\n", device_level); */
   }

   if (myid == 0 && sum > 0)
   {
      hypre_printf_dbl("Running with these driver parameters:\n");
      hypre_printf_dbl("  (cx, cy, cz)    = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf_dbl("  (conx,cony,conz)= (%f, %f, %f)\n", conx, cony, conz);
      hypre_printf_dbl("  (n_pre, n_post) = (%d, %d)\n", n_pre, n_post);
      hypre_printf_dbl("  dim             = %d\n", dim);
      hypre_printf_dbl("  skip            = %d\n", skip);
      hypre_printf_dbl("  sym             = %d\n", sym);
      hypre_printf_dbl("  rap             = %d\n", rap);
      hypre_printf_dbl("  relax           = %d\n", relax);
      hypre_printf_dbl("  solver ID       = %d\n", solver_id);
      hypre_printf_dbl("  the grid is read from  file \n");

   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   MPI_Barrier(MPI_COMM_WORLD);

   for ( rep = 0; rep < reps; ++rep )
   {
      time_index = hypre_InitializeTiming_dbl("Struct Interface");
      hypre_BeginTiming_dbl(time_index);

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
               offsets = (HYPRE_Int **) hypre_CAlloc_dbl(2, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc_dbl(1, (size_t)sizeof(HYPRE_Int ), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc_dbl(1, (size_t)sizeof(HYPRE_Int ), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
            }
            else
            {
               offsets = (HYPRE_Int **) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc_dbl(1, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc_dbl(1, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
               offsets[2] = (HYPRE_Int *) hypre_CAlloc_dbl(1, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[2][0] = 1;
            }
            /* compute p from P and myid */
            p = myid % P;
            break;

         case 2:
            nblocks = bx * by;
            if (sym)
            {
               offsets = (HYPRE_Int **) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc_dbl(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[0][1] = 0;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc_dbl(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
               offsets[1][1] = -1;
               offsets[2] = (HYPRE_Int *) hypre_CAlloc_dbl(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[2][0] = 0;
               offsets[2][1] = 0;
            }
            else
            {
               offsets = (HYPRE_Int **) hypre_CAlloc_dbl(5, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = hypre_CAlloc_dbl(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc_dbl(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[0][1] = 0;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc_dbl(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
               offsets[1][1] = -1;
               offsets[2] = (HYPRE_Int *) hypre_CAlloc_dbl(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[2][0] = 0;
               offsets[2][1] = 0;
               offsets[3] = (HYPRE_Int *) hypre_CAlloc_dbl(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[3][0] = 1;
               offsets[3][1] = 0;
               offsets[4] = (HYPRE_Int *) hypre_CAlloc_dbl(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
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
               offsets = (HYPRE_Int **) hypre_CAlloc_dbl(4, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[0][1] = 0;
               offsets[0][2] = 0;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
               offsets[1][1] = -1;
               offsets[1][2] = 0;
               offsets[2] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[2][0] = 0;
               offsets[2][1] = 0;
               offsets[2][2] = -1;
               offsets[3] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[3][0] = 0;
               offsets[3][1] = 0;
               offsets[3][2] = 0;
            }
            else
            {
               offsets = (HYPRE_Int **) hypre_CAlloc_dbl(7, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[3] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[0][1] = 0;
               offsets[0][2] = 0;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
               offsets[1][1] = -1;
               offsets[1][2] = 0;
               offsets[2] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[2][0] = 0;
               offsets[2][1] = 0;
               offsets[2][2] = -1;
               offsets[3] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[3][0] = 0;
               offsets[3][1] = 0;
               offsets[3][2] = 0;
               offsets[4] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[4][0] = 1;
               offsets[4][1] = 0;
               offsets[4][2] = 0;
               offsets[5] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[5][0] = 0;
               offsets[5][1] = 1;
               offsets[5][2] = 0;
               offsets[6] = (HYPRE_Int *) hypre_CAlloc_dbl(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
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

      HYPRE_StructStencilCreate_dbl(dim, (2 - sym)*dim + 1, &stencil);
      for (s = 0; s < (2 - sym)*dim + 1; s++)
      {
         HYPRE_StructStencilSetElement_dbl(stencil, s, offsets[s]);
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

#if 0
      dxyz[0] = 1.0e+123;
      dxyz[1] = 1.0e+123;
      dxyz[2] = 1.0e+123;
      if (cx > 0)
      {
         dxyz[0] = hypre_sqrt(1.0 / cx);
      }
      if (cy > 0)
      {
         dxyz[1] = hypre_sqrt(1.0 / cy);
      }
      if (cz > 0)
      {
         dxyz[2] = hypre_sqrt(1.0 / cz);
      }
#endif

      /* We do the extreme cases first reading everything from files => sum = 3
       * building things from scratch (grid,stencils,extents) sum = 0 */

      /*if ( (read_fromfile_param == 1) &&
           (read_x0fromfile_param == 1) &&
           (read_rhsfromfile_param == 1)
         )
      {*/
      /* ghost selection for reading the matrix and vectors */
      /* for (i = 0; i < dim; i++)
       {
          A_num_ghost[2 * i] = 1;
          A_num_ghost[2 * i + 1] = 1;
          v_num_ghost[2 * i] = 1;
          v_num_ghost[2 * i + 1] = 1;
       }

       HYPRE_StructMatrixRead(hypre_MPI_COMM_WORLD,
                              argv[read_fromfile_index],
                              A_num_ghost, &A);

       HYPRE_StructVectorRead(hypre_MPI_COMM_WORLD,
                              argv[read_rhsfromfile_index],
                              v_num_ghost, &b);

       HYPRE_StructVectorRead(hypre_MPI_COMM_WORLD,
                              argv[read_x0fromfile_index],
                              v_num_ghost, &x);
      }*/

      /* beginning of sum == 0  */
      if (sum == 0)    /* no read from any file */
      {
         /*-----------------------------------------------------------
          * prepare space for the extents
          *-----------------------------------------------------------*/

         ilower = (HYPRE_Int **) hypre_CAlloc_dbl((size_t)(nblocks), (size_t)sizeof(HYPRE_Int*),
                                                  HYPRE_MEMORY_HOST);
         iupper = (HYPRE_Int **) hypre_CAlloc_dbl((size_t)(nblocks), (size_t)sizeof(HYPRE_Int*),
                                                  HYPRE_MEMORY_HOST);
         for (i = 0; i < nblocks; i++)
         {
            ilower[i] = (HYPRE_Int *) hypre_CAlloc_dbl((size_t)(dim), (size_t)sizeof(HYPRE_Int),
                                                       HYPRE_MEMORY_HOST);
            iupper[i] = (HYPRE_Int *) hypre_CAlloc_dbl((size_t)(dim), (size_t)sizeof(HYPRE_Int),
                                                       HYPRE_MEMORY_HOST);
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

         HYPRE_StructGridCreate_dbl(hypre_MPI_COMM_WORLD, dim, &grid);
         for (ib = 0; ib < nblocks; ib++)
         {
            /* Add to the grid a new box defined by ilower[ib], iupper[ib]...*/
            HYPRE_StructGridSetExtents_dbl(grid, ilower[ib], iupper[ib]);
         }
         HYPRE_StructGridSetPeriodic_dbl(grid, periodic);
         HYPRE_StructGridSetNumGhost_dbl(grid, num_ghost);
         HYPRE_StructGridAssemble_dbl(grid);

         /*-----------------------------------------------------------
          * Set up the matrix structure
          *-----------------------------------------------------------*/

         HYPRE_StructMatrixCreate_dbl(hypre_MPI_COMM_WORLD, grid, stencil, &A_dbl);
         HYPRE_StructMatrixCreate_flt(hypre_MPI_COMM_WORLD, grid, stencil, &A_flt);

         if ( solver_id == 3 || solver_id == 4 ||
              solver_id == 13 || solver_id == 14 )
         {
            stencil_size  = hypre_StructStencilSize(stencil);
            stencil_entries = (HYPRE_Int *) hypre_CAlloc_dbl ((size_t)(stencil_size), (size_t)sizeof(HYPRE_Int),
                                                              HYPRE_MEMORY_HOST);
            if ( solver_id == 3 || solver_id == 13)
            {
               for ( i = 0; i < stencil_size; ++i )
               {
                  stencil_entries[i] = i;
               }
               hypre_StructMatrixSetConstantEntries_dbl( A_dbl, stencil_size, stencil_entries );
               hypre_StructMatrixSetConstantEntries_flt( A_flt, stencil_size, stencil_entries );
               /* ... note: SetConstantEntries is where the constant_coefficient
                  flag is set in A */
               hypre_Free_dbl( stencil_entries, HYPRE_MEMORY_HOST);
               constant_coefficient = 1;
            }
            if ( solver_id == 4 || solver_id == 14)
            {
               hypre_SetIndex3(diag_index, 0, 0, 0);
               diag_rank = hypre_StructStencilElementRank_flt( stencil, diag_index );
               //hypre_assert( stencil_size >= 1 );
               if ( diag_rank == 0 )
               {
                  stencil_entries[diag_rank] = 1;
               }
               else
               {
                  stencil_entries[diag_rank] = 0;
               }
               for ( i = 0; i < stencil_size; ++i )
               {
                  if ( i != diag_rank )
                  {
                     stencil_entries[i] = i;
                  }
               }
               hypre_StructMatrixSetConstantEntries_dbl( A_dbl, stencil_size, stencil_entries );
               hypre_StructMatrixSetConstantEntries_flt( A_flt, stencil_size, stencil_entries );
               hypre_Free_dbl( stencil_entries, HYPRE_MEMORY_HOST);
               constant_coefficient = 2;
            }
         }

         HYPRE_StructMatrixSetSymmetric_dbl(A_dbl, sym);
         HYPRE_StructMatrixInitialize_dbl(A_dbl);
         HYPRE_StructMatrixSetSymmetric_flt(A_flt, sym);
         HYPRE_StructMatrixInitialize_flt(A_flt);

         /*-----------------------------------------------------------
          * Fill in the matrix elements
          *-----------------------------------------------------------*/

         AddValuesMatrix_dbl(A_dbl, grid, cx, cy, cz, conx, cony, conz);
         AddValuesMatrix_flt(A_flt, grid, (float)cx, (float)cy, (float)cz, (float)conx, (float)cony,
                             (float)conz);

         /* Zero out stencils reaching to real boundary */
         /* But in constant coefficient case, no special stencils! */

         if ( constant_coefficient == 0 )
         {
            SetStencilBndry_mp(A_dbl, A_flt, grid, periodic);
         }
         HYPRE_StructMatrixAssemble_dbl(A_dbl);
         HYPRE_StructMatrixAssemble_flt(A_flt);
         /*-----------------------------------------------------------
          * Set up the linear system
          *-----------------------------------------------------------*/

         HYPRE_StructVectorCreate_dbl(hypre_MPI_COMM_WORLD, grid, &b_dbl);
         HYPRE_StructVectorInitialize_dbl(b_dbl);
         HYPRE_StructVectorCreate_flt(hypre_MPI_COMM_WORLD, grid, &b_flt);
         HYPRE_StructVectorInitialize_flt(b_flt);

         /*-----------------------------------------------------------
          * For periodic b.c. in all directions, need rhs to satisfy
          * compatibility condition. Achieved by setting a source and
          *  sink of equal strength.  All other problems have rhs = 1.
          *-----------------------------------------------------------*/

         AddValuesVector_dbl(grid, b_dbl, periodic, 1.0);
         AddValuesVector_flt(grid, b_flt, periodic, 1.0);
         HYPRE_StructVectorAssemble_dbl(b_dbl);
         HYPRE_StructVectorAssemble_flt(b_flt);

         HYPRE_StructVectorCreate_dbl(hypre_MPI_COMM_WORLD, grid, &x_dbl);
         HYPRE_StructVectorCreate_flt(hypre_MPI_COMM_WORLD, grid, &x_flt);
         HYPRE_StructVectorInitialize_dbl(x_dbl);
         HYPRE_StructVectorInitialize_flt(x_flt);

         AddValuesVector_dbl(grid, x_dbl, periodx0, 0.0);
         AddValuesVector_flt(grid, x_flt, periodx0, 0.0);
         HYPRE_StructVectorAssemble_dbl(x_dbl);
         HYPRE_StructVectorAssemble_flt(x_flt);

         HYPRE_StructGridDestroy_dbl(grid);

         for (i = 0; i < nblocks; i++)
         {
            hypre_Free_dbl(iupper[i], HYPRE_MEMORY_HOST);
            hypre_Free_dbl(ilower[i], HYPRE_MEMORY_HOST);
         }
         hypre_Free_dbl(ilower, HYPRE_MEMORY_HOST);
         hypre_Free_dbl(iupper, HYPRE_MEMORY_HOST);
      }

      /* linear system complete  */

      hypre_EndTiming_dbl(time_index);
      if ( reps == 1 || (solver_id != 0 && solver_id != 1 && solver_id != 3 && solver_id != 4) )
      {
         hypre_PrintTiming_dbl("Struct Interface", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming_dbl(time_index);
         hypre_ClearTiming_dbl();
      }
      else if ( rep == reps - 1 )
      {
         hypre_FinalizeTiming_dbl(time_index);
      }

      /*-----------------------------------------------------------
       * Print out the system and initial guess
       *-----------------------------------------------------------*/

      /*if (print_system)
      {
         HYPRE_StructMatrixPrint("struct.out.A", A, 0);
         HYPRE_StructVectorPrint("struct.out.b", b, 0);
         HYPRE_StructVectorPrint("struct.out.x0", x, 0);
      }*/

      /*-----------------------------------------------------------
       * Solve the system using SMG
       *-----------------------------------------------------------*/

#if !HYPRE_MFLOPS

      if (solver_id == 0)
      {
         time_index = hypre_InitializeTiming_dbl("SMG Setup");
         hypre_BeginTiming_dbl(time_index);

         HYPRE_StructSMGCreate_dbl(MPI_COMM_WORLD, &solver);
         //HYPRE_StructSMGSetMemoryUse_dbl(solver, 0);
         HYPRE_StructSMGSetMaxIter_dbl(solver, 50);
         HYPRE_StructSMGSetTol_dbl(solver, tol);
         HYPRE_StructSMGSetRelChange_dbl(solver, 0);
         HYPRE_StructSMGSetNumPreRelax_dbl(solver, n_pre);
         HYPRE_StructSMGSetNumPostRelax_dbl(solver, n_post);
         HYPRE_StructSMGSetPrintLevel_dbl(solver, 1);
         HYPRE_StructSMGSetLogging_dbl(solver, 1);
#if 0//defined(HYPRE_USING_CUDA)
         HYPRE_StructSMGSetDeviceLevel(solver, device_level);
#endif

#if 0//defined(HYPRE_USING_CUDA)
         hypre_box_print = 0;
#endif
         HYPRE_StructSMGSetup_dbl(solver, A_dbl, b_dbl, x_dbl);

#if 0//defined(HYPRE_USING_CUDA)
         hypre_box_print = 0;
#endif

         hypre_EndTiming_dbl(time_index);
         if ( reps == 1 )
         {
            hypre_PrintTiming_dbl("Setup phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming_dbl(time_index);
            hypre_ClearTiming_dbl();
         }
         else if ( rep == reps - 1 )
         {
            hypre_FinalizeTiming_dbl(time_index);
         }

         time_index = hypre_InitializeTiming_dbl("SMG Solve");
         hypre_BeginTiming_dbl(time_index);

         HYPRE_StructSMGSolve_dbl(solver, A_dbl, b_dbl, x_dbl);

         hypre_EndTiming_dbl(time_index);
         if ( reps == 1 )
         {
            hypre_PrintTiming_dbl("Solve phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming_dbl(time_index);
            hypre_ClearTiming_dbl();
         }
         else if ( rep == reps - 1 )
         {
            hypre_PrintTiming_dbl("Interface, Setup, and Solve times:",
                                  hypre_MPI_COMM_WORLD );
            hypre_FinalizeTiming_dbl(time_index);
            hypre_ClearTiming_dbl();
         }

         HYPRE_StructSMGGetNumIterations_dbl(solver, &num_iterations);
         HYPRE_StructSMGGetFinalRelativeResidualNorm_dbl(solver, &final_res_norm);
         HYPRE_StructSMGDestroy_dbl(solver);
      }

      /*-----------------------------------------------------------
       * Solve the system using PFMG
       *-----------------------------------------------------------*/

      else if ( solver_id == 1 || solver_id == 3 || solver_id == 4 )
      {
         time_index = hypre_InitializeTiming_dbl("PFMG Setup");
         hypre_BeginTiming_dbl(time_index);

         HYPRE_StructPFMGCreate_dbl(hypre_MPI_COMM_WORLD, &solver);
         /*HYPRE_StructPFMGSetMaxLevels_dbl( solver, 9 );*/
         HYPRE_StructPFMGSetMaxIter_dbl(solver, 200);
         HYPRE_StructPFMGSetTol_dbl(solver, tol);
         HYPRE_StructPFMGSetRelChange_dbl(solver, 0);
         HYPRE_StructPFMGSetRAPType_dbl(solver, rap);
         HYPRE_StructPFMGSetRelaxType_dbl(solver, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight_dbl(solver, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax_dbl(solver, n_pre);
         HYPRE_StructPFMGSetNumPostRelax_dbl(solver, n_post);
         HYPRE_StructPFMGSetSkipRelax_dbl(solver, skip);
         /*HYPRE_StructPFMGSetDxyz_dbl(solver, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel_dbl(solver, 1);
         HYPRE_StructPFMGSetLogging_dbl(solver, 1);

#if 0//defined(HYPRE_USING_CUDA)
         HYPRE_StructPFMGSetDeviceLevel(solver, device_level);
#endif

         HYPRE_StructPFMGSetup_dbl(solver, A_dbl, b_dbl, x_dbl);

         hypre_EndTiming_dbl(time_index);
         if ( reps == 1 )
         {
            hypre_PrintTiming_dbl("Setup phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming_dbl(time_index);
            hypre_ClearTiming_dbl();
         }
         else if ( rep == reps - 1 )
         {
            hypre_FinalizeTiming_dbl(time_index);
         }

         time_index = hypre_InitializeTiming_dbl("PFMG Solve");
         hypre_BeginTiming_dbl(time_index);


         HYPRE_StructPFMGSolve_dbl(solver, A_dbl, b_dbl, x_dbl);

         hypre_EndTiming_dbl(time_index);
         if ( reps == 1 )
         {
            hypre_PrintTiming_dbl("Solve phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming_dbl(time_index);
            hypre_ClearTiming_dbl();
         }
         else if ( rep == reps - 1 )
         {
            hypre_PrintTiming_dbl("Interface, Setup, and Solve times",
                                  hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming_dbl(time_index);
            hypre_ClearTiming_dbl();
         }

         HYPRE_StructPFMGGetNumIterations_dbl(solver, &num_iterations);
         HYPRE_StructPFMGGetFinalRelativeResidualNorm_dbl(solver, &final_res_norm);
         HYPRE_StructPFMGDestroy_dbl(solver);
      }

      /*-----------------------------------------------------------
       * Solve the system using CG
       *-----------------------------------------------------------*/

      if ((solver_id > 9) && (solver_id < 20))
      {
         time_index = hypre_InitializeTiming_dbl("PCG Setup");
         hypre_BeginTiming_dbl(time_index);

         HYPRE_StructPCGCreate_dbl(MPI_COMM_WORLD, &solver);
         HYPRE_PCGSetMaxIter_dbl( (HYPRE_Solver)solver, 100 );
         HYPRE_PCGSetTol_dbl( (HYPRE_Solver)solver, tol );
         HYPRE_PCGSetTwoNorm_dbl( (HYPRE_Solver)solver, 1 );
         HYPRE_PCGSetRelChange_dbl( (HYPRE_Solver)solver, 0 );
         HYPRE_PCGSetPrintLevel_dbl( (HYPRE_Solver)solver, print_level );
         HYPRE_PCGSetRecomputeResidual_dbl( (HYPRE_Solver)solver, 1 );
         HYPRE_PCGSetFlex_dbl( (HYPRE_Solver)solver, flex );

         if (solver_id == 10)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate_dbl(hypre_MPI_COMM_WORLD, &precond);
            //HYPRE_StructSMGSetMemoryUse_dbl(precond, 0);
            HYPRE_StructSMGSetMaxIter_dbl(precond, 1);
            HYPRE_StructSMGSetTol_dbl(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_dbl(precond);
            HYPRE_StructSMGSetNumPreRelax_dbl(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_dbl(precond, n_post);
            HYPRE_StructSMGSetPrintLevel_dbl(precond, 0);
            HYPRE_StructSMGSetLogging_dbl(precond, 0);

            HYPRE_PCGSetPrecond_dbl( (HYPRE_Solver) solver,
                                     (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_dbl,
                                     (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_dbl,
                                     (HYPRE_Solver) precond);
         }
         else if (solver_id == 15)
         {
            HYPRE_StructSMGCreate_flt(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMaxIter_flt(precond, 1);
            HYPRE_StructSMGSetTol_flt(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_flt(precond);
            HYPRE_StructSMGSetNumPreRelax_flt(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_flt(precond, n_post);
            HYPRE_StructSMGSetPrintLevel_flt(precond, 0);
            HYPRE_StructSMGSetLogging_flt(precond, 0);
            HYPRE_PCGSetPrecondMatrix_dbl((HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_PCGSetPrecond_dbl( (HYPRE_Solver) solver,
                                     (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_mp,
                                     (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_mp,
                                     (HYPRE_Solver) precond);
         }

         else if (solver_id == 11 || solver_id == 13 || solver_id == 14)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate_dbl(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_dbl(precond, 1);
            HYPRE_StructPFMGSetTol_dbl(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_dbl(precond);
            HYPRE_StructPFMGSetRAPType_dbl(precond, rap);
            HYPRE_StructPFMGSetRelaxType_dbl(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_dbl(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_dbl(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_dbl(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_dbl(precond, skip);
            /*HYPRE_StructPFMGSetDxyz_dbl(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel_dbl(precond, 0);
            HYPRE_StructPFMGSetLogging_dbl(precond, 0);
#if 0//defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_PCGSetPrecond_dbl((HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_dbl,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_dbl,
                                    (HYPRE_Solver) precond);
         }
         else if (solver_id == 16)
         {
            HYPRE_StructPFMGCreate_flt(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_flt(precond, 1);
            HYPRE_StructPFMGSetTol_flt(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_flt(precond);
            HYPRE_StructPFMGSetRAPType_flt(precond, rap);
            HYPRE_StructPFMGSetRelaxType_flt(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_flt(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_flt(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_flt(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_flt(precond, skip);
            HYPRE_StructPFMGSetPrintLevel_flt(precond, 0);
            HYPRE_StructPFMGSetLogging_flt(precond, 0);
            HYPRE_PCGSetPrecondMatrix_dbl((HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_PCGSetPrecond_dbl((HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_mp,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_mp,
                                    (HYPRE_Solver) precond);
         }

         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_PCGSetPrecond_dbl((HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale_dbl,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup_dbl,
                                    (HYPRE_Solver) precond);
         }

         HYPRE_PCGSetup_dbl( (HYPRE_Solver)solver,
                             (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl );

         hypre_EndTiming_dbl(time_index);
         hypre_PrintTiming_dbl("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming_dbl(time_index);
         hypre_ClearTiming_dbl();

         time_index = hypre_InitializeTiming_dbl("PCG Solve");
         hypre_BeginTiming_dbl(time_index);

         HYPRE_PCGSolve_dbl( (HYPRE_Solver) solver,
                             (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

         hypre_EndTiming_dbl(time_index);
         hypre_PrintTiming_dbl("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming_dbl(time_index);
         hypre_ClearTiming_dbl();

         HYPRE_PCGGetNumIterations_dbl( (HYPRE_Solver)solver, &num_iterations );
         HYPRE_PCGGetFinalRelativeResidualNorm_dbl( (HYPRE_Solver)solver,
                                                    &final_res_norm );
         HYPRE_StructPCGDestroy_dbl(solver);

         if (solver_id == 10)
         {
            HYPRE_StructSMGDestroy_dbl(precond);
         }
         else if (solver_id == 15)
         {
            HYPRE_StructSMGDestroy_flt(precond);
         }
         else if (solver_id == 11 || solver_id == 13 || solver_id == 14)
         {
            HYPRE_StructPFMGDestroy_dbl(precond);
         }
         else if (solver_id == 16)
         {
            HYPRE_StructPFMGDestroy_flt(precond);
         }
      }

      /*-----------------------------------------------------------
       * Solve the system using GMRES
       *-----------------------------------------------------------*/

      if ((solver_id > 29) && (solver_id < 40))
      {
         time_index = hypre_InitializeTiming_dbl("GMRES Setup");
         hypre_BeginTiming_dbl(time_index);

         HYPRE_StructGMRESCreate_dbl(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_GMRESSetKDim_dbl( (HYPRE_Solver) solver, 5 );
         HYPRE_GMRESSetMaxIter_dbl( (HYPRE_Solver)solver, 100 );
         HYPRE_GMRESSetTol_dbl( (HYPRE_Solver)solver, tol );
         HYPRE_GMRESSetRelChange_dbl( (HYPRE_Solver)solver, 0 );
         HYPRE_GMRESSetPrintLevel_dbl( (HYPRE_Solver)solver, print_level );
         HYPRE_GMRESSetLogging_dbl( (HYPRE_Solver)solver, 1 );

         if (solver_id == 30)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate_dbl(hypre_MPI_COMM_WORLD, &precond);
            //HYPRE_StructSMGSetMemoryUse_dbl(precond, 0);
            HYPRE_StructSMGSetMaxIter_dbl(precond, 1);
            HYPRE_StructSMGSetTol_dbl(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_dbl(precond);
            HYPRE_StructSMGSetNumPreRelax_dbl(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_dbl(precond, n_post);
            HYPRE_StructSMGSetPrintLevel_dbl(precond, 0);
            HYPRE_StructSMGSetLogging_dbl(precond, 0);
            HYPRE_GMRESSetPrecond_dbl((HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_dbl,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_dbl,
                                      (HYPRE_Solver)precond);
         }
         else if (solver_id == 35)
         {
            HYPRE_StructSMGCreate_flt(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMaxIter_flt(precond, 1);
            HYPRE_StructSMGSetTol_flt(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_flt(precond);
            HYPRE_StructSMGSetNumPreRelax_flt(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_flt(precond, n_post);
            HYPRE_StructSMGSetPrintLevel_flt(precond, 0);
            HYPRE_StructSMGSetLogging_flt(precond, 0);
            HYPRE_GMRESSetPrecondMatrix_dbl((HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_GMRESSetPrecond_dbl((HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_mp,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_mp,
                                      (HYPRE_Solver)precond);
         }

         else if (solver_id == 31)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate_dbl(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_dbl(precond, 1);
            HYPRE_StructPFMGSetTol_dbl(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_dbl(precond);
            HYPRE_StructPFMGSetRAPType_dbl(precond, rap);
            HYPRE_StructPFMGSetRelaxType_dbl(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_dbl(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_dbl(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_dbl(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_dbl(precond, skip);
            /*HYPRE_StructPFMGSetDxyz_dbl(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel_dbl(precond, 0);
            HYPRE_StructPFMGSetLogging_dbl(precond, 0);
            HYPRE_GMRESSetPrecond_dbl((HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_dbl,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_dbl,
                                      (HYPRE_Solver)precond);
         }
         else if (solver_id == 36)
         {
            HYPRE_StructPFMGCreate_flt(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_flt(precond, 1);
            HYPRE_StructPFMGSetTol_flt(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_flt(precond);
            HYPRE_StructPFMGSetRAPType_flt(precond, rap);
            HYPRE_StructPFMGSetRelaxType_flt(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_flt(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_flt(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_flt(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_flt(precond, skip);
            HYPRE_StructPFMGSetPrintLevel_flt(precond, 0);
            HYPRE_StructPFMGSetLogging_flt(precond, 0);
            HYPRE_GMRESSetPrecondMatrix_dbl((HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_GMRESSetPrecond_dbl((HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_mp,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_mp,
                                      (HYPRE_Solver)precond);
         }

         else if (solver_id == 38)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_GMRESSetPrecond_dbl((HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale_dbl,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup_dbl,
                                      (HYPRE_Solver)precond);
         }

         HYPRE_GMRESSetup_dbl ( (HYPRE_Solver)solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl,
                                (HYPRE_Vector)x_dbl );

         hypre_EndTiming_dbl(time_index);
         hypre_PrintTiming_dbl("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming_dbl(time_index);
         hypre_ClearTiming_dbl();

         time_index = hypre_InitializeTiming_dbl("GMRES Solve");
         hypre_BeginTiming_dbl(time_index);

         HYPRE_GMRESSolve_dbl
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

         hypre_EndTiming_dbl(time_index);
         hypre_PrintTiming_dbl("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming_dbl(time_index);
         hypre_ClearTiming_dbl();

         HYPRE_GMRESGetNumIterations_dbl( (HYPRE_Solver)solver, &num_iterations);
         HYPRE_GMRESGetFinalRelativeResidualNorm_dbl( (HYPRE_Solver)solver, &final_res_norm);
         HYPRE_StructGMRESDestroy_dbl(solver);

         if (solver_id == 30)
         {
            HYPRE_StructSMGDestroy_dbl(precond);
         }
         else if (solver_id == 35)
         {
            HYPRE_StructSMGDestroy_flt(precond);
         }
         else if (solver_id == 31)
         {
            HYPRE_StructPFMGDestroy_dbl(precond);
         }
         else if (solver_id == 36)
         {
            HYPRE_StructPFMGDestroy_flt(precond);
         }
      }

      /*-----------------------------------------------------------
       * Solve the system using BiCGTAB
       *-----------------------------------------------------------*/

      if ((solver_id > 39) && (solver_id < 50))
      {
         time_index = hypre_InitializeTiming_dbl("BiCGSTAB Setup");
         hypre_BeginTiming_dbl(time_index);

         HYPRE_StructBiCGSTABCreate_dbl(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_BiCGSTABSetMaxIter_dbl( (HYPRE_Solver)solver, 100 );
         HYPRE_BiCGSTABSetTol_dbl( (HYPRE_Solver)solver, tol );
         HYPRE_BiCGSTABSetPrintLevel_dbl( (HYPRE_Solver)solver, print_level );
         HYPRE_BiCGSTABSetLogging_dbl( (HYPRE_Solver)solver, 1 );

         if (solver_id == 40)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate_dbl(hypre_MPI_COMM_WORLD, &precond);
            //HYPRE_StructSMGSetMemoryUse_dbl(precond, 0);
            HYPRE_StructSMGSetMaxIter_dbl(precond, 1);
            HYPRE_StructSMGSetTol_dbl(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_dbl(precond);
            HYPRE_StructSMGSetNumPreRelax_dbl(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_dbl(precond, n_post);
            HYPRE_StructSMGSetPrintLevel_dbl(precond, 0);
            HYPRE_StructSMGSetLogging_dbl(precond, 0);
            HYPRE_BiCGSTABSetPrecond_dbl((HYPRE_Solver)solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_dbl,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_dbl,
                                         (HYPRE_Solver)precond);
         }
         else if (solver_id == 45)
         {
            HYPRE_StructSMGCreate_flt(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMaxIter_flt(precond, 1);
            HYPRE_StructSMGSetTol_flt(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_flt(precond);
            HYPRE_StructSMGSetNumPreRelax_flt(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_flt(precond, n_post);
            HYPRE_StructSMGSetPrintLevel_flt(precond, 0);
            HYPRE_StructSMGSetLogging_flt(precond, 0);
            HYPRE_BiCGSTABSetPrecondMatrix_dbl((HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_BiCGSTABSetPrecond_dbl((HYPRE_Solver)solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_mp,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_mp,
                                         (HYPRE_Solver)precond);
         }

         else if (solver_id == 41)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate_dbl(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_dbl(precond, 1);
            HYPRE_StructPFMGSetTol_dbl(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_dbl(precond);
            HYPRE_StructPFMGSetRAPType_dbl(precond, rap);
            HYPRE_StructPFMGSetRelaxType_dbl(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_dbl(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_dbl(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_dbl(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_dbl(precond, skip);
            /*HYPRE_StructPFMGSetDxyz_dbl(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel_dbl(precond, 0);
            HYPRE_StructPFMGSetLogging_dbl(precond, 0);
            HYPRE_BiCGSTABSetPrecond_dbl((HYPRE_Solver)solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_dbl,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_dbl,
                                         (HYPRE_Solver)precond);
         }
         else if (solver_id == 46)
         {
            HYPRE_StructPFMGCreate_flt(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_flt(precond, 1);
            HYPRE_StructPFMGSetTol_flt(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_dbl(precond);
            HYPRE_StructPFMGSetRAPType_flt(precond, rap);
            HYPRE_StructPFMGSetRelaxType_flt(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_flt(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_flt(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_flt(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_flt(precond, skip);
            HYPRE_StructPFMGSetPrintLevel_flt(precond, 0);
            HYPRE_StructPFMGSetLogging_flt(precond, 0);
            HYPRE_BiCGSTABSetPrecondMatrix_dbl((HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_BiCGSTABSetPrecond_dbl((HYPRE_Solver)solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_mp,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_mp,
                                         (HYPRE_Solver)precond);
         }

         else if (solver_id == 48)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_BiCGSTABSetPrecond_dbl((HYPRE_Solver)solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale_dbl,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup_dbl,
                                         (HYPRE_Solver)precond);
         }

         HYPRE_BiCGSTABSetup_dbl
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl );

         hypre_EndTiming_dbl(time_index);
         hypre_PrintTiming_dbl("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming_dbl(time_index);
         hypre_ClearTiming_dbl();

         time_index = hypre_InitializeTiming_dbl("BiCGSTAB Solve");
         hypre_BeginTiming_dbl(time_index);

         HYPRE_BiCGSTABSolve_dbl
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

         hypre_EndTiming_dbl(time_index);
         hypre_PrintTiming_dbl("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming_dbl(time_index);
         hypre_ClearTiming_dbl();

         HYPRE_BiCGSTABGetNumIterations_dbl( (HYPRE_Solver)solver, &num_iterations);
         HYPRE_BiCGSTABGetFinalRelativeResidualNorm_dbl( (HYPRE_Solver)solver, &final_res_norm);
         HYPRE_StructBiCGSTABDestroy_dbl(solver);

         if (solver_id == 40)
         {
            HYPRE_StructSMGDestroy_dbl(precond);
         }
         else if (solver_id == 45)
         {
            HYPRE_StructSMGDestroy_flt(precond);
         }
         else if (solver_id == 41)
         {
            HYPRE_StructPFMGDestroy_dbl(precond);
         }
         else if (solver_id == 46)
         {
            HYPRE_StructPFMGDestroy_flt(precond);
         }
      }

      /*-----------------------------------------------------------
       * Print the solution and other info
       *-----------------------------------------------------------*/

      if (print_system)
      {
         HYPRE_StructVectorPrint_dbl("struct.out.x", x_dbl, 0);
      }

      if (myid == 0 && rep == reps - 1 )
      {
         hypre_printf_dbl("\n");
         hypre_printf_dbl("Iterations = %d\n", num_iterations);
         hypre_printf_dbl("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf_dbl("\n");
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

         matvec_data = hypre_StructMatvecCreate_dbl();
         hypre_StructMatvecSetup_dbl(matvec_data, A, x);

         time_index = hypre_InitializeTiming_dbl("Matvec");
         hypre_BeginTiming_dbl(time_index);

         for (i = 0; i < imax; i++)
         {
            hypre_StructMatvecCompute_dbl(matvec_data, 1.0, A, x, 1.0, b);
         }
         /* this counts mult-adds */
         hypre_IncFLOPCount_dbl(7 * N * imax);

         hypre_EndTiming_dbl(time_index);
         hypre_PrintTiming_dbl("Matvec time", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming_dbl(time_index);
         hypre_ClearTiming_dbl();

         hypre_StructMatvecDestroy_dbl(matvec_data);
      }
#endif

      /*-----------------------------------------------------------
       * Finalize things
       *-----------------------------------------------------------*/

      HYPRE_StructStencilDestroy_dbl(stencil);
      HYPRE_StructMatrixDestroy_dbl(A_dbl);
      HYPRE_StructVectorDestroy_dbl(b_dbl);
      HYPRE_StructVectorDestroy_dbl(x_dbl);

      for ( i = 0; i < (dim + 1); i++)
      {
         hypre_Free_dbl(offsets[i], HYPRE_MEMORY_HOST);
      }
      hypre_Free_dbl(offsets, HYPRE_MEMORY_HOST);
   }

   /* Finalize Hypre */
   //HYPRE_Finalize();

   /* Finalize MPI */
   MPI_Finalize();

#if defined(HYPRE_USING_MEMORY_TRACKER)
   if (memory_location == HYPRE_MEMORY_HOST)
   {
      if (hypre_total_bytes[hypre_MEMORY_DEVICE] || hypre_total_bytes[hypre_MEMORY_UNIFIED])
      {
         hypre_printf_dbl("Error: nonzero GPU memory allocated with the HOST mode\n");
         //hypre_assert(0);
      }
   }
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
   /* use this for the stats of the offloading counts */
   //HYPRE_OMPOffloadStatPrint();
#endif

   return (0);
}


/*-------------------------------------------------------------------------
 * add constant values to a vector. Need to pass the initialized vector, grid,
 * period of grid and the constant value.
 *-------------------------------------------------------------------------*/

HYPRE_Int
AddValuesVector_flt( hypre_StructGrid   *gridvector,
                     hypre_StructVector *zvector,
                     HYPRE_Int          *period,
                     float               value  )
{
   /* #include  "_hypre_struct_mv.h" */
   HYPRE_Int            i, ierr = 0;
   hypre_BoxArray      *gridboxes;
   HYPRE_Int            ib;
   hypre_IndexRef       ilower;
   hypre_IndexRef       iupper;
   hypre_Box           *box;
   float               *values;
   HYPRE_Int            volume, dim;

   gridboxes =  hypre_StructGridBoxes(gridvector);
   dim       =  hypre_StructGridNDim(gridvector);

   ib = 0;
   hypre_ForBoxI(ib, gridboxes)
   {
      box      = hypre_BoxArrayBox(gridboxes, ib);
      volume   = hypre_BoxVolume_flt(box);
      values = hypre_CAlloc_flt((size_t) volume, (size_t)sizeof(float), HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------
       * For periodic b.c. in all directions, need rhs to satisfy
       * compatibility condition. Achieved by setting a source and
       *  sink of equal strength.  All other problems have rhs = 1.
       *-----------------------------------------------------------*/

      if ((dim == 2 && period[0] != 0 && period[1] != 0) ||
          (dim == 3 && period[0] != 0 && period[1] != 0 && period[2] != 0))
      {
         values[0] = value;
         values[volume - 1] = -value;
      }
      else
      {
         for (i = 0; i < volume; i++)
         {
            values[i] = value;
         }
      }

      ilower = hypre_BoxIMin(box);
      iupper = hypre_BoxIMax(box);

      HYPRE_StructVectorSetBoxValues_flt(zvector, ilower, iupper, values);

      hypre_Free_flt(values, HYPRE_MEMORY_HOST);
   }

   return ierr;
}

/*-------------------------------------------------------------------------
 * add constant values to a vector. Need to pass the initialized vector, grid,
 * period of grid and the constant value.
 *-------------------------------------------------------------------------*/

HYPRE_Int
AddValuesVector_dbl( hypre_StructGrid   *gridvector,
                     hypre_StructVector *zvector,
                     HYPRE_Int          *period,
                     double              value  )
{
   /* #include  "_hypre_struct_mv.h" */
   HYPRE_Int            i, ierr = 0;
   hypre_BoxArray      *gridboxes;
   HYPRE_Int            ib;
   hypre_IndexRef       ilower;
   hypre_IndexRef       iupper;
   hypre_Box           *box;
   double              *values;
   HYPRE_Int            volume, dim;

   gridboxes =  hypre_StructGridBoxes(gridvector);
   dim       =  hypre_StructGridNDim(gridvector);

   ib = 0;
   hypre_ForBoxI(ib, gridboxes)
   {
      box      = hypre_BoxArrayBox(gridboxes, ib);
      volume   = hypre_BoxVolume_dbl(box);
      values   = hypre_CAlloc_dbl((size_t) volume, (size_t)sizeof(double), HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------
       * For periodic b.c. in all directions, need rhs to satisfy
       * compatibility condition. Achieved by setting a source and
       *  sink of equal strength.  All other problems have rhs = 1.
       *-----------------------------------------------------------*/

      if ((dim == 2 && period[0] != 0 && period[1] != 0) ||
          (dim == 3 && period[0] != 0 && period[1] != 0 && period[2] != 0))
      {
         values[0] = value;
         values[volume - 1] = -value;
      }
      else
      {
         for (i = 0; i < volume; i++)
         {
            values[i] = value;
         }
      }

      ilower = hypre_BoxIMin(box);
      iupper = hypre_BoxIMax(box);

      HYPRE_StructVectorSetBoxValues_dbl(zvector, ilower, iupper, values);

      hypre_Free_dbl(values, HYPRE_MEMORY_HOST);
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
AddValuesMatrix_flt(HYPRE_StructMatrix A,
                    HYPRE_StructGrid   gridmatrix,
                    float              cx,
                    float              cy,
                    float              cz,
                    float              conx,
                    float              cony,
                    float              conz)
{

   HYPRE_Int            d, ierr = 0;
   hypre_BoxArray      *gridboxes;
   HYPRE_Int            s, bi;
   hypre_IndexRef       ilower;
   hypre_IndexRef       iupper;
   hypre_Box           *box;
   float               *values;
   float                east, west;
   float                north, south;
   float                top, bottom;
   float                center;
   HYPRE_Int            volume, dim, sym;
   HYPRE_Int           *stencil_indices;
   HYPRE_Int            stencil_size;
   HYPRE_Int            constant_coefficient;

   gridboxes = hypre_StructGridBoxes(gridmatrix);
   dim       = hypre_StructGridNDim(gridmatrix);
   sym       = hypre_StructMatrixSymmetric(A);
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
   stencil_indices = hypre_CAlloc_flt((size_t) stencil_size, (size_t)sizeof(HYPRE_Int),
                                      HYPRE_MEMORY_HOST);
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
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume_flt(box);
            values = hypre_CAlloc_flt((size_t)(stencil_size * volume), (size_t)sizeof(float ),
                                      HYPRE_MEMORY_HOST);

            if (dim == 1)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  values[i] = west;
                  values[i + 1] = center;
               }
            }
            else if (dim == 2)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  values[i] = west;
                  values[i + 1] = south;
                  values[i + 2] = center;
               }
            }
            else if (dim == 3)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  values[i] = west;
                  values[i + 1] = south;
                  values[i + 2] = bottom;
                  values[i + 3] = center;
               }
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);

            HYPRE_StructMatrixSetBoxValues_flt(A, ilower, iupper, stencil_size,
                                               stencil_indices, values);

            hypre_Free_flt(values, HYPRE_MEMORY_HOST);
         }
      }
      else if ( constant_coefficient == 1 )
      {
         values = hypre_CAlloc_flt((size_t)stencil_size, (size_t)sizeof(float ), HYPRE_MEMORY_HOST);
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
            HYPRE_StructMatrixSetConstantValues_flt(A, stencil_size,
                                                    stencil_indices, values);
         }
         hypre_Free_flt(values, HYPRE_MEMORY_HOST);
      }
      else
      {
         //hypre_assert( constant_coefficient == 2 );

         /* stencil index for the center equals dim, so it's easy to leave out */
         values = hypre_CAlloc_flt((size_t)(stencil_size - 1), (size_t)sizeof(float ), HYPRE_MEMORY_HOST);
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
            HYPRE_StructMatrixSetConstantValues_flt(A, stencil_size - 1,
                                                    stencil_indices, values);
         }
         hypre_Free_flt(values, HYPRE_MEMORY_HOST);

         hypre_ForBoxI(bi, gridboxes)
         {
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume_flt(box);
            values = hypre_CAlloc_flt((size_t) volume, (size_t)sizeof(float ), HYPRE_MEMORY_HOST);
            HYPRE_Int i;

            for (i = 0; i < volume; i++)
            {
               values[i] = center;
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues_flt(A, ilower, iupper, 1,
                                               stencil_indices + dim, values);
            hypre_Free_flt(values, HYPRE_MEMORY_HOST);
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
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume_flt(box);
            values = hypre_CAlloc_flt((size_t)(stencil_size * volume), (size_t)sizeof(float ),
                                      HYPRE_MEMORY_HOST);

            for (d = 0; d < volume; d++)
            {
               HYPRE_Int i = stencil_size * d;
               switch (dim)
               {
                  case 1:
                     values[i] = west;
                     values[i + 1] = center;
                     values[i + 2] = east;
                     break;
                  case 2:
                     values[i] = west;
                     values[i + 1] = south;
                     values[i + 2] = center;
                     values[i + 3] = east;
                     values[i + 4] = north;
                     break;
                  case 3:
                     values[i] = west;
                     values[i + 1] = south;
                     values[i + 2] = bottom;
                     values[i + 3] = center;
                     values[i + 4] = east;
                     values[i + 5] = north;
                     values[i + 6] = top;
                     break;
               }
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues_flt(A, ilower, iupper, stencil_size,
                                               stencil_indices, values);

            hypre_Free_flt(values, HYPRE_MEMORY_HOST);
         }
      }
      else if ( constant_coefficient == 1 )
      {
         values = hypre_CAlloc_flt( (size_t)stencil_size, (size_t)sizeof(float), HYPRE_MEMORY_HOST);

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
            HYPRE_StructMatrixSetConstantValues_flt(A, stencil_size,
                                                    stencil_indices, values);
         }

         hypre_Free_flt(values, HYPRE_MEMORY_HOST);
      }
      else
      {
         //hypre_assert( constant_coefficient == 2 );
         values =  hypre_CAlloc_flt( (size_t) (stencil_size - 1), (size_t)sizeof(float ), HYPRE_MEMORY_HOST);
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
            HYPRE_StructMatrixSetConstantValues_flt(A, stencil_size,
                                                    stencil_indices, values);
         }
         hypre_Free_flt(values, HYPRE_MEMORY_HOST);


         /* center is variable */
         stencil_indices[0] = dim; /* refers to center */
         hypre_ForBoxI(bi, gridboxes)
         {
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume_flt(box);
            values = hypre_CAlloc_flt((size_t) volume, (size_t)sizeof(float ), HYPRE_MEMORY_HOST);
            HYPRE_Int i;

            for (i = 0; i < volume; i++)
            {
               values[i] = center;
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues_flt(A, ilower, iupper, 1,
                                               stencil_indices, values);
            hypre_Free_flt(values, HYPRE_MEMORY_HOST);
         }
      }
   }

   hypre_Free_flt(stencil_indices, HYPRE_MEMORY_HOST);

   return ierr;
}

/******************************************************************************
 * Adds values to matrix based on a 7 point (3d)
 * symmetric stencil for a convection-diffusion problem.
 * It need an initialized matrix, an assembled grid, and the constants
 * that determine the 7 point (3d) convection-diffusion.
 ******************************************************************************/

HYPRE_Int
AddValuesMatrix_dbl(HYPRE_StructMatrix A,
                    HYPRE_StructGrid   gridmatrix,
                    double             cx,
                    double             cy,
                    double             cz,
                    double             conx,
                    double             cony,
                    double             conz)
{

   HYPRE_Int            d, ierr = 0;
   hypre_BoxArray      *gridboxes;
   HYPRE_Int            s, bi;
   hypre_IndexRef       ilower;
   hypre_IndexRef       iupper;
   hypre_Box           *box;
   double              *values;
   double               east, west;
   double               north, south;
   double               top, bottom;
   double               center;
   HYPRE_Int            volume, dim, sym;
   HYPRE_Int           *stencil_indices;
   HYPRE_Int            stencil_size;
   HYPRE_Int            constant_coefficient;

   gridboxes = hypre_StructGridBoxes(gridmatrix);
   dim       = hypre_StructGridNDim(gridmatrix);
   sym       = hypre_StructMatrixSymmetric(A);
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
   stencil_indices = (HYPRE_Int *) hypre_CAlloc_dbl((size_t)stencil_size, (size_t)sizeof(HYPRE_Int),
                                                    HYPRE_MEMORY_HOST);
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
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume_dbl(box);
            values = hypre_CAlloc_dbl((size_t)(stencil_size * volume), (size_t)sizeof(double),
                                      HYPRE_MEMORY_HOST);

            if (dim == 1)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  values[i] = west;
                  values[i + 1] = center;
               }
            }
            else if (dim == 2)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  values[i] = west;
                  values[i + 1] = south;
                  values[i + 2] = center;
               }
            }
            else if (dim == 3)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  values[i] = west;
                  values[i + 1] = south;
                  values[i + 2] = bottom;
                  values[i + 3] = center;
               }
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);

            HYPRE_StructMatrixSetBoxValues_dbl(A, ilower, iupper, stencil_size,
                                               stencil_indices, values);

            hypre_Free_dbl(values, HYPRE_MEMORY_HOST);
         }
      }
      else if ( constant_coefficient == 1 )
      {
         values = hypre_CAlloc_dbl((size_t)stencil_size, (size_t)sizeof(double), HYPRE_MEMORY_HOST);
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
            HYPRE_StructMatrixSetConstantValues_dbl(A, stencil_size,
                                                    stencil_indices, values);
         }
         hypre_Free_dbl(values, HYPRE_MEMORY_HOST);
      }
      else
      {
         //hypre_assert( constant_coefficient == 2 );

         /* stencil index for the center equals dim, so it's easy to leave out */
         values = hypre_CAlloc_dbl((size_t)(stencil_size - 1), (size_t)sizeof(double), HYPRE_MEMORY_HOST);
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
            HYPRE_StructMatrixSetConstantValues_dbl(A, stencil_size - 1,
                                                    stencil_indices, values);
         }
         hypre_Free_dbl(values, HYPRE_MEMORY_HOST);

         hypre_ForBoxI(bi, gridboxes)
         {
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume_dbl(box);
            values = hypre_CAlloc_dbl((size_t) volume, (size_t)sizeof(double), HYPRE_MEMORY_HOST);
            HYPRE_Int i;

            for (i = 0; i < volume; i++)
            {
               values[i] = center;
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues_dbl(A, ilower, iupper, 1,
                                               stencil_indices + dim, values);
            hypre_Free_dbl(values, HYPRE_MEMORY_HOST);
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
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume_dbl(box);
            values = hypre_CAlloc_dbl((size_t)(stencil_size * volume), (size_t)sizeof(double),
                                      HYPRE_MEMORY_HOST);

            for (d = 0; d < volume; d++)
            {
               HYPRE_Int i = stencil_size * d;
               switch (dim)
               {
                  case 1:
                     values[i] = west;
                     values[i + 1] = center;
                     values[i + 2] = east;
                     break;
                  case 2:
                     values[i] = west;
                     values[i + 1] = south;
                     values[i + 2] = center;
                     values[i + 3] = east;
                     values[i + 4] = north;
                     break;
                  case 3:
                     values[i] = west;
                     values[i + 1] = south;
                     values[i + 2] = bottom;
                     values[i + 3] = center;
                     values[i + 4] = east;
                     values[i + 5] = north;
                     values[i + 6] = top;
                     break;
               }
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues_dbl(A, ilower, iupper, stencil_size,
                                               stencil_indices, values);

            hypre_Free_flt(values, HYPRE_MEMORY_HOST);
         }
      }
      else if ( constant_coefficient == 1 )
      {
         values = hypre_CAlloc_dbl( (size_t)stencil_size, (size_t)sizeof(double), HYPRE_MEMORY_HOST);

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
            HYPRE_StructMatrixSetConstantValues_dbl(A, stencil_size,
                                                    stencil_indices, values);
         }

         hypre_Free_dbl(values, HYPRE_MEMORY_HOST);
      }
      else
      {
         //hypre_assert( constant_coefficient == 2 );
         values =  hypre_CAlloc_dbl( (size_t) (stencil_size - 1), (size_t)sizeof(double), HYPRE_MEMORY_HOST);
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
            HYPRE_StructMatrixSetConstantValues_dbl(A, stencil_size,
                                                    stencil_indices, values);
         }
         hypre_Free_dbl(values, HYPRE_MEMORY_HOST);


         /* center is variable */
         stencil_indices[0] = dim; /* refers to center */
         hypre_ForBoxI(bi, gridboxes)
         {
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume_dbl(box);
            values = hypre_CAlloc_dbl((size_t) volume, (size_t)sizeof(double), HYPRE_MEMORY_HOST);
            HYPRE_Int i;

            for (i = 0; i < volume; i++)
            {
               values[i] = center;
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues_dbl(A, ilower, iupper, 1,
                                               stencil_indices, values);
            hypre_Free_dbl(values, HYPRE_MEMORY_HOST);
         }
      }
   }

   hypre_Free_dbl(stencil_indices, HYPRE_MEMORY_HOST);

   return ierr;
}

/*********************************************************************************
 * this function sets to zero the stencil entries that are on the boundary
 * Grid, matrix and the period are needed.
 *********************************************************************************/

HYPRE_Int
SetStencilBndry_mp(HYPRE_StructMatrix A_dbl,
                   HYPRE_StructMatrix A_flt,
                   HYPRE_StructGrid   gridmatrix,
                   HYPRE_Int         *period)
{
   HYPRE_Int            ierr = 0;
   hypre_BoxArray      *gridboxes;
   HYPRE_Int            size, i, j, d, ib;
   HYPRE_Int          **ilower;
   HYPRE_Int          **iupper;
   HYPRE_Int           *vol;
   HYPRE_Int           *istart, *iend;
   hypre_Box           *box;
   hypre_Box           *dummybox;
   hypre_Box           *boundingbox;
   double              *values_dbl;
   float               *values_flt;
   HYPRE_Int            volume, dim;
   HYPRE_Int           *stencil_indices;
   HYPRE_Int            constant_coefficient;

   gridboxes       = hypre_StructGridBoxes(gridmatrix);
   boundingbox     = hypre_StructGridBoundingBox(gridmatrix);
   istart          = hypre_BoxIMin(boundingbox);
   iend            = hypre_BoxIMax(boundingbox);
   size            = hypre_StructGridNumBoxes(gridmatrix);
   dim             = hypre_StructGridNDim(gridmatrix);
   stencil_indices = (HYPRE_Int *)hypre_CAlloc_dbl(1, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A_dbl);
   if ( constant_coefficient > 0 ) { return 1; }
   /*...no space dependence if constant_coefficient==1,
     and space dependence only for diagonal if constant_coefficient==2 --
     and this function only touches off-diagonal entries */

   vol    = hypre_CAlloc_dbl((size_t) size, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
   ilower = hypre_CAlloc_dbl((size_t) size, (size_t)sizeof(HYPRE_Int*), HYPRE_MEMORY_HOST);
   iupper = hypre_CAlloc_dbl((size_t) size, (size_t)sizeof(HYPRE_Int*), HYPRE_MEMORY_HOST);
   for (i = 0; i < size; i++)
   {
      ilower[i] = hypre_CAlloc_dbl((size_t) dim, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
      iupper[i] = hypre_CAlloc_dbl((size_t) dim, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
   }

   i = 0;
   ib = 0;
   hypre_ForBoxI(i, gridboxes)
   {
      dummybox = hypre_BoxCreate_dbl(dim);
      box      = hypre_BoxArrayBox(gridboxes, i);
      volume   =  hypre_BoxVolume_dbl(box);
      vol[i]   = volume;
      hypre_CopyBox_dbl(box, dummybox);
      for (d = 0; d < dim; d++)
      {
         ilower[ib][d] = hypre_BoxIMinD(dummybox, d);
         iupper[ib][d] = hypre_BoxIMaxD(dummybox, d);
      }
      ib++ ;
      hypre_BoxDestroy_dbl(dummybox);
   }

   if ( constant_coefficient == 0 )
   {
      for (d = 0; d < dim; d++)
      {
         for (ib = 0; ib < size; ib++)
         {
            values_dbl = hypre_CAlloc_dbl((size_t)(vol[ib]), (size_t)sizeof(double), HYPRE_MEMORY_HOST);
            values_flt = hypre_CAlloc_flt((size_t)(vol[ib]), (size_t)sizeof(float), HYPRE_MEMORY_HOST);

            if ( ilower[ib][d] == istart[d] && period[d] == 0 )
            {
               j = iupper[ib][d];
               iupper[ib][d] = istart[d];
               stencil_indices[0] = d;
               HYPRE_StructMatrixSetBoxValues_dbl(A_dbl, ilower[ib], iupper[ib],
                                                  1, stencil_indices, values_dbl);
               HYPRE_StructMatrixSetBoxValues_flt(A_flt, ilower[ib], iupper[ib],
                                                  1, stencil_indices, values_flt);
               iupper[ib][d] = j;
            }

            if ( iupper[ib][d] == iend[d] && period[d] == 0 )
            {
               j = ilower[ib][d];
               ilower[ib][d] = iend[d];
               stencil_indices[0] = dim + 1 + d;
               HYPRE_StructMatrixSetBoxValues_dbl(A_dbl, ilower[ib], iupper[ib],
                                                  1, stencil_indices, values_dbl);
               HYPRE_StructMatrixSetBoxValues_flt(A_flt, ilower[ib], iupper[ib],
                                                  1, stencil_indices, values_flt);
               ilower[ib][d] = j;
            }

            hypre_Free_dbl(values_dbl, HYPRE_MEMORY_HOST);
            hypre_Free_flt(values_flt, HYPRE_MEMORY_HOST);
         }
      }
   }

   hypre_Free_dbl(vol, HYPRE_MEMORY_HOST);
   hypre_Free_dbl(stencil_indices, HYPRE_MEMORY_HOST);
   for (ib = 0 ; ib < size ; ib++)
   {
      hypre_Free_dbl(ilower[ib], HYPRE_MEMORY_HOST);
      hypre_Free_dbl(iupper[ib], HYPRE_MEMORY_HOST);
   }
   hypre_Free_dbl(ilower, HYPRE_MEMORY_HOST);
   hypre_Free_dbl(iupper, HYPRE_MEMORY_HOST);

   return ierr;
}
