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

#include "HYPRE.h"
#include "HYPRE_struct_mv.h"
#include "HYPRE_struct_ls.h"
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

HYPRE_Int  SetStencilBndry_mp(HYPRE_StructMatrix A, HYPRE_StructGrid gridmatrix, HYPRE_Int* period);

HYPRE_Int AddValuesVector_mp(hypre_StructGrid  *gridvector,
                               hypre_StructVector *zvector,
                               HYPRE_Int          *period,
                               void               *value,
                               size_t             size)  ;

HYPRE_Int AddValuesMatrix_mp(HYPRE_StructMatrix A,
                    HYPRE_StructGrid   gridmatrix,
                    void              *cx,
                    void              *cy,
                    void              *cz,
                    void              *conx,
                    void              *cony,
                    void              *conz,
                    size_t            size);

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
   HYPRE_Int           stencil_diag_entry;

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

   /* Default precision values */
   HYPRE_Int precision_id; /* 0=flt, 1=dbl, 2=ldbl */
   HYPRE_Precision solver_precision = HYPRE_REAL_DOUBLE;
   HYPRE_Precision precond_precision = HYPRE_REAL_SINGLE;
   hypre_long_double one_ldbl = 1.0;
   hypre_double      one_dbl = 1.0;
   hypre_float       one_flt = 1.0f;            
   hypre_long_double zero_ldbl = 0.0;
   hypre_double      zero_dbl = 0.0;
   hypre_float       zero_flt = 0.0f; 
   void        *one_slvr = &one_dbl;
   void        *one_pc = &one_flt;
   void        *zero_slvr = &zero_dbl;
   void        *zero_pc = &zero_flt;
   /* convection variables */
   hypre_long_double   params_ldbl[6]; /*cx, cy, cz, conx, cony, conz*/
   hypre_double        params_dbl[6];
   hypre_float         params_flt[6];
   
   void        *conx_slvr, *cony_slvr, *conz_slvr;
   void        *cx_slvr, *cy_slvr, *cz_slvr;   
   void        *conx_pc, *cony_pc, *conz_pc;
   void        *cx_pc, *cy_pc, *cz_pc;
   size_t      slvr_size = sizeof(hypre_double);
   size_t      pc_size = sizeof(hypre_float);
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
   
   params_ldbl[0] = (long double)cx;
   params_ldbl[1] = (long double)cy;
   params_ldbl[2] = (long double)cz;
   params_dbl[0] = (double)cx;
   params_dbl[1] = (double)cy;
   params_dbl[2] = (double)cz;
   params_flt[0] = (float)cx;
   params_flt[1] = (float)cy;
   params_flt[2] = (float)cz; 

   params_ldbl[3] = (long double)conx;
   params_ldbl[4] = (long double)cony;
   params_ldbl[5] = (long double)conz;
   params_dbl[3] = (double)conx;
   params_dbl[4] = (double)cony;
   params_dbl[5] = (double)conz;
   params_flt[3] = (float)conx;
   params_flt[4] = (float)cony;
   params_flt[5] = (float)conz;  

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
         params_ldbl[0] = (long double)cx;
         params_ldbl[1] = (long double)cy;
         params_ldbl[2] = (long double)cz;
         params_dbl[0] = (double)cx;
         params_dbl[1] = (double)cy;
         params_dbl[2] = (double)cz;
         params_flt[0] = (float)cx;
         params_flt[1] = (float)cy;
         params_flt[2] = (float)cz;         
      }
      else if ( strcmp(argv[arg_index], "-convect") == 0 )
      {
         arg_index++;
         conx = (double)atof(argv[arg_index++]);
         cony = (double)atof(argv[arg_index++]);
         conz = (double)atof(argv[arg_index++]);
         params_ldbl[3] = (long double)conx;
         params_ldbl[4] = (long double)cony;
         params_ldbl[5] = (long double)conz;
         params_dbl[3] = (double)conx;
         params_dbl[4] = (double)cony;
         params_dbl[5] = (double)conz;
         params_flt[3] = (float)conx;
         params_flt[4] = (float)cony;
         params_flt[5] = (float)conz;
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
      else if ( strcmp(argv[arg_index], "-solver_precision") == 0 )
      {
         arg_index++;
         precision_id = atoi(argv[arg_index++]);
         
         switch (precision_id)
         {
            case 0:
               solver_precision = HYPRE_REAL_SINGLE;
               one_slvr = &one_flt;
               zero_slvr = &zero_flt;
               break;
            case 1:
               solver_precision = HYPRE_REAL_DOUBLE;
               one_slvr = &one_dbl;
               zero_slvr = &zero_dbl;
               break;
            case 2:
               solver_precision = HYPRE_REAL_LONGDOUBLE;
               one_slvr = &one_ldbl;
               zero_slvr = &zero_ldbl;
               break;
         }
      }
      else if ( strcmp(argv[arg_index], "-pc_precision") == 0 )
      {
         arg_index++;
         precision_id = atoi(argv[arg_index++]);
         
         switch (precision_id)
         {
            case 0:
               precond_precision = HYPRE_REAL_SINGLE;
               one_pc = &one_flt;
               zero_pc = &zero_flt;
               break;
            case 1:
               precond_precision = HYPRE_REAL_DOUBLE;
               one_pc = &one_dbl;
               zero_pc = &zero_dbl;
               break;
            case 2:
               precond_precision = HYPRE_REAL_LONGDOUBLE;
               one_pc = &one_ldbl;
               zero_pc = &zero_ldbl;
               break;
         }      
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
      hypre_printf("  -repeats <reps>     : number of times to repeat the run, default 1.  For solver 0,1,3\n");
      hypre_printf("  -solver <ID>        : solver ID\n");
      hypre_printf("                        0  - SMG (default)\n");
      hypre_printf("                        1  - PFMG\n");
      hypre_printf("                        3  - PFMG constant coeffs\n");
      hypre_printf("                        4  - PFMG constant coeffs var diag\n");
      hypre_printf("                        10 -- 19 - CG\n");
      hypre_printf("                        10 - CG with SMG precond\n");
      hypre_printf("                        11 - CG with PFMG precond\n");
      hypre_printf("                        13 - CG with PFMG-3 precond\n");
      hypre_printf("                        14 - CG with PFMG-4 precond\n");
      hypre_printf("                        15 - CG with single-prec SMG precond\n");
      hypre_printf("                        16 - CG with single-prec PFMG precond\n");
      hypre_printf("                        18 - CG with diagonal scaling\n");
      hypre_printf("                        20 -- 29 - GMRES\n");
      hypre_printf("                        20 - GMRES with SMG precond\n");
      hypre_printf("                        21 - GMRES with PFMG precond\n");
      hypre_printf("                        25 - GMRES with single-prec SMG precond\n");
      hypre_printf("                        26 - GMRES with single-prec PFMG precond\n");
      hypre_printf("                        28 - GMRES with diagonal scaling\n");
      hypre_printf("                        30 -- 39 - FlexGMRES\n");
      hypre_printf("                        30 - FlexGMRES with SMG precond\n");
      hypre_printf("                        31 - FlexGMRES with PFMG precond\n");
      hypre_printf("                        35 - FlexGMRES with single-prec SMG precond\n");
      hypre_printf("                        36 - FlexGMRES with single-prec PFMG precond\n");
      hypre_printf("                        38 - FlexGMRES with diagonal scaling\n");
      hypre_printf("                        39 - FlexGMRES\n");
      hypre_printf("                        40 -- 49 - BiCGSTAB\n");
      hypre_printf("                        40 - BiCGSTAB with SMG precond\n");
      hypre_printf("                        41 - BiCGSTAB with PFMG precond\n");
      hypre_printf("                        45 - BiCGSTAB with single-prec SMG precond\n");
      hypre_printf("                        46 - BiCGSTAB with single-prec PFMG precond\n");
      hypre_printf("                        48 - BiCGSTAB with diagonal scaling\n");
      hypre_printf("  -v <n_pre> <n_post> : number of pre and post relaxations\n");
      hypre_printf("  -rap <r>            : coarse grid operator type\n");
      hypre_printf("                        0 - Galerkin (default)\n");
      hypre_printf("                        1 - non-Galerkin ParFlow operators\n");
      hypre_printf("                        2 - Galerkin, general operators\n");
      hypre_printf("  -relax <r>          : relaxation type\n");
      hypre_printf("                        0 - Jacobi\n");
      hypre_printf("                        1 - Weighted Jacobi (default)\n");
      hypre_printf("                        2 - R/B Gauss-Seidel\n");
      hypre_printf("                        3 - R/B Gauss-Seidel (nonsymmetric)\n");
      hypre_printf("  -w <jacobi weight>  : jacobi weight\n");
      hypre_printf("  -skip <s>           : skip levels in PFMG (0 or 1)\n");
      hypre_printf("  -sym <s>            : symmetric storage (1) or not (0)\n");
      hypre_printf("  -solver_type <ID>   : solver type for Hybrid\n");
      hypre_printf("                        1 - PCG (default)\n");
      hypre_printf("                        2 - GMRES\n");
      hypre_printf("  -recompute <bool>   : Recompute residual in PCG?\n");
      hypre_printf("  -cf <cf>            : convergence factor for Hybrid\n");

      hypre_printf("\n");
      hypre_printf("  -solver_precision <p>     : solver precision\n");
      hypre_printf("                       Precision for outer solver (Krylov solver) \n");
      hypre_printf("                        0 = flt, 1 = dbl, 2 = long_dbl\n");
      hypre_printf("\n");
      hypre_printf("  -precond_precision <p>     : precond precision\n");
      hypre_printf("                       Precision for inner solver (preconditioner) \n");
      hypre_printf("                        0 = flt, 1 = dbl, 2 = long_dbl\n");
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
#if defined(HYPRE_DEVELOP_STRING) && defined(HYPRE_DEVELOP_BRANCH)
      hypre_printf("\nUsing HYPRE_DEVELOP_STRING: %s (branch %s; the develop branch)\n\n",
                       HYPRE_DEVELOP_STRING, HYPRE_DEVELOP_BRANCH);

#elif defined(HYPRE_DEVELOP_STRING) && !defined(HYPRE_DEVELOP_BRANCH)
      hypre_printf("\nUsing HYPRE_DEVELOP_STRING: %s (branch %s; not the develop branch)\n\n",
                       HYPRE_DEVELOP_STRING, HYPRE_BRANCH_NAME);

#elif defined(HYPRE_RELEASE_VERSION)
      hypre_printf("\nUsing HYPRE_RELEASE_VERSION: %s\n\n",
                       HYPRE_RELEASE_VERSION);
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
      hypre_printf("  solver ID       = %d\n", solver_id);
      /* hypre_printf("  Device level    = %d\n", device_level); */
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
      hypre_printf("  solver ID       = %d\n", solver_id);
      hypre_printf("  the grid is read from  file \n");

   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   MPI_Barrier(MPI_COMM_WORLD);

   for ( rep = 0; rep < reps; ++rep )
   {
      time_index = hypre_InitializeTiming("Struct Interface");
      hypre_BeginTiming(time_index);

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
               offsets = (HYPRE_Int **) hypre_CAlloc(2, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc(1, (size_t)sizeof(HYPRE_Int ), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc(1, (size_t)sizeof(HYPRE_Int ), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
            }
            else
            {
               offsets = (HYPRE_Int **) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc(1, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc(1, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
               offsets[2] = (HYPRE_Int *) hypre_CAlloc(1, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[2][0] = 1;
            }
            /* compute p from P and myid */
            p = myid % P;
            break;

         case 2:
            nblocks = bx * by;
            if (sym)
            {
               offsets = (HYPRE_Int **) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[0][1] = 0;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
               offsets[1][1] = -1;
               offsets[2] = (HYPRE_Int *) hypre_CAlloc(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[2][0] = 0;
               offsets[2][1] = 0;
            }
            else
            {
               offsets = (HYPRE_Int **) hypre_CAlloc(5, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[0][1] = 0;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
               offsets[1][1] = -1;
               offsets[2] = (HYPRE_Int *) hypre_CAlloc(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[2][0] = 0;
               offsets[2][1] = 0;
               offsets[3] = (HYPRE_Int *) hypre_CAlloc(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[3][0] = 1;
               offsets[3][1] = 0;
               offsets[4] = (HYPRE_Int *) hypre_CAlloc(2, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
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
               offsets = (HYPRE_Int **) hypre_CAlloc(4, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[0][1] = 0;
               offsets[0][2] = 0;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
               offsets[1][1] = -1;
               offsets[1][2] = 0;
               offsets[2] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[2][0] = 0;
               offsets[2][1] = 0;
               offsets[2][2] = -1;
               offsets[3] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[3][0] = 0;
               offsets[3][1] = 0;
               offsets[3][2] = 0;
            }
            else
            {
               offsets = (HYPRE_Int **) hypre_CAlloc(7, (size_t)sizeof(HYPRE_Int *), HYPRE_MEMORY_HOST);
               offsets[0] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[0][0] = -1;
               offsets[0][1] = 0;
               offsets[0][2] = 0;
               offsets[1] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[1][0] = 0;
               offsets[1][1] = -1;
               offsets[1][2] = 0;
               offsets[2] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[2][0] = 0;
               offsets[2][1] = 0;
               offsets[2][2] = -1;
               offsets[3] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[3][0] = 0;
               offsets[3][1] = 0;
               offsets[3][2] = 0;
               offsets[4] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[4][0] = 1;
               offsets[4][1] = 0;
               offsets[4][2] = 0;
               offsets[5] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
               offsets[5][0] = 0;
               offsets[5][1] = 1;
               offsets[5][2] = 0;
               offsets[6] = (HYPRE_Int *) hypre_CAlloc(3, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
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
       * 
       * The following struct stencil functions are integer operations and 
       * can be called in any precision. We default to the solver precision here.
       *-----------------------------------------------------------*/

      HYPRE_StructStencilCreate_pre( solver_precision,dim, (2 - sym)*dim + 1, &stencil);
      for (s = 0; s < (2 - sym)*dim + 1; s++)
      {
         HYPRE_StructStencilSetElement_pre( solver_precision,stencil, s, offsets[s]);
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

      /* beginning of sum == 0  */
      if (sum == 0)    /* no read from any file */
      {
         /*-----------------------------------------------------------
          * prepare space for the extents
          *-----------------------------------------------------------*/

         ilower = (HYPRE_Int **) hypre_CAlloc((size_t)(nblocks), (size_t)sizeof(HYPRE_Int*),
                                                  HYPRE_MEMORY_HOST);
         iupper = (HYPRE_Int **) hypre_CAlloc((size_t)(nblocks), (size_t)sizeof(HYPRE_Int*),
                                                  HYPRE_MEMORY_HOST);
         for (i = 0; i < nblocks; i++)
         {
            ilower[i] = (HYPRE_Int *) hypre_CAlloc((size_t)(dim), (size_t)sizeof(HYPRE_Int),
                                                       HYPRE_MEMORY_HOST);
            iupper[i] = (HYPRE_Int *) hypre_CAlloc((size_t)(dim), (size_t)sizeof(HYPRE_Int),
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
         
         /* The following struct grid functions are integer operations and 
          * can be called in any precision. We default to the solver precision here.
         */
         HYPRE_StructGridCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, dim, &grid);
         for (ib = 0; ib < nblocks; ib++)
         {
            /* Add to the grid a new box defined by ilower[ib], iupper[ib]...*/
            HYPRE_StructGridSetExtents_pre( solver_precision,grid, ilower[ib], iupper[ib]);
         }
         HYPRE_StructGridSetPeriodic_pre( solver_precision,grid, periodic);
         HYPRE_StructGridSetNumGhost_pre( solver_precision,grid, num_ghost);
         HYPRE_StructGridAssemble_pre( solver_precision,grid);

         /*-----------------------------------------------------------
          * Set up the matrix structure
          *-----------------------------------------------------------*/

         HYPRE_StructMatrixCreate_pre(solver_precision, hypre_MPI_COMM_WORLD, grid, stencil, &A_dbl);
         HYPRE_StructMatrixCreate_pre(precond_precision, hypre_MPI_COMM_WORLD, grid, stencil, &A_flt);

         if ( solver_id == 3 || solver_id == 4 ||
              solver_id == 13 || solver_id == 14 )
         {
            stencil_size  = hypre_StructStencilSize(stencil);
            /* integer array allocation */
            stencil_entries = (HYPRE_Int *) hypre_CAlloc ((size_t)(stencil_size), (size_t)sizeof(HYPRE_Int),
                                                              HYPRE_MEMORY_HOST);
            if ( solver_id == 3 || solver_id == 13)
            {
               for ( i = 0; i < stencil_size; ++i )
               {
                  stencil_entries[i] = i;
               }
               HYPRE_StructMatrixSetConstantEntries_pre( solver_precision, A_dbl, stencil_size, stencil_entries );
               HYPRE_StructMatrixSetConstantEntries_pre( precond_precision, A_flt, stencil_size, stencil_entries );
               /* ... note: SetConstantEntries is where the constant_coefficient
                  flag is set in A */

               /* Free integer array */
               hypre_Free(stencil_entries, HYPRE_MEMORY_HOST);
               constant_coefficient = 1;
            }
            if ( solver_id == 4 || solver_id == 14)
            {
               stencil_diag_entry = hypre_StructStencilDiagEntry(stencil);
               hypre_assert( stencil_size >= 1 );
               if ( stencil_diag_entry == 0 )
               {
                  stencil_entries[stencil_diag_entry] = 1;
               }
               else
               {
                  stencil_entries[stencil_diag_entry] = 0;
               }
               for ( i = 0; i < stencil_size; ++i )
               {
                  if ( i != stencil_diag_entry )
                  {
                     stencil_entries[i] = i;
                  }
               }
               HYPRE_StructMatrixSetConstantEntries_pre( solver_precision, A_dbl, stencil_size, stencil_entries );
               HYPRE_StructMatrixSetConstantEntries_pre( precond_precision, A_flt, stencil_size, stencil_entries );
               hypre_Free( stencil_entries, HYPRE_MEMORY_HOST);
               constant_coefficient = 2;
            }
         }

         HYPRE_StructMatrixSetSymmetric_pre( solver_precision,A_dbl, sym);
         HYPRE_StructMatrixInitialize_pre( solver_precision,A_dbl);
         HYPRE_StructMatrixSetSymmetric_pre( precond_precision,A_flt, sym);
         HYPRE_StructMatrixInitialize_pre( precond_precision,A_flt);

         /*-----------------------------------------------------------
          * Fill in the matrix elements
          *-----------------------------------------------------------*/
         switch (solver_precision)
         {
            case HYPRE_REAL_SINGLE:
               cx_slvr = &params_flt[0];
               cy_slvr = &params_flt[1];
               cz_slvr = &params_flt[2];
               conx_slvr = &params_flt[3];
               cony_slvr = &params_flt[4];
               conz_slvr = &params_flt[5];
               break;
            case HYPRE_REAL_DOUBLE:
               cx_slvr = &params_dbl[0];
               cy_slvr = &params_dbl[1];
               cz_slvr = &params_dbl[2];
               conx_slvr = &params_dbl[3];
               cony_slvr = &params_dbl[4];
               conz_slvr = &params_dbl[5];
               break;               
            case HYPRE_REAL_LONGDOUBLE:
               cx_slvr = &params_ldbl[0];
               cy_slvr = &params_ldbl[1];
               cz_slvr = &params_ldbl[2];
               conx_slvr = &params_ldbl[3];
               cony_slvr = &params_ldbl[4];
               conz_slvr = &params_ldbl[5];
               break;
         }

         switch (precond_precision)
         {
            case HYPRE_REAL_SINGLE:
               cx_pc = &params_flt[0];
               cy_pc = &params_flt[1];
               cz_pc = &params_flt[2];
               conx_pc = &params_flt[3];
               cony_pc = &params_flt[4];
               conz_pc = &params_flt[5];
               break;
            case HYPRE_REAL_DOUBLE:
               cx_pc = &params_dbl[0];
               cy_pc = &params_dbl[1];
               cz_pc = &params_dbl[2];
               conx_pc = &params_dbl[3];
               cony_pc = &params_dbl[4];
               conz_pc = &params_dbl[5];
               break;               
            case HYPRE_REAL_LONGDOUBLE:
               cx_pc = &params_ldbl[0];
               cy_pc = &params_ldbl[1];
               cz_pc = &params_ldbl[2];
               conx_pc = &params_ldbl[3];
               cony_pc = &params_ldbl[4];
               conz_pc = &params_ldbl[5];
               break;
         }         
         AddValuesMatrix_mp(A_dbl, grid, cx_slvr, cy_slvr, cz_slvr, conx_slvr, cony_slvr, conz_slvr, sizeof(hypre_double));
         AddValuesMatrix_mp(A_flt, grid, cx_pc, cy_pc, cz_pc, conx_pc, cony_pc, conz_pc, sizeof(hypre_float));

         /* Zero out stencils reaching to real boundary */
         /* But in constant coefficient case, no special stencils! */

         if ( constant_coefficient == 0 )
         {
            SetStencilBndry_mp(A_dbl, grid, periodic);
            SetStencilBndry_mp(A_flt, grid, periodic);
         }
         HYPRE_StructMatrixAssemble_pre( solver_precision,A_dbl);
         HYPRE_StructMatrixAssemble_pre( precond_precision,A_flt);
         /*-----------------------------------------------------------
          * Set up the linear system
          *-----------------------------------------------------------*/

         HYPRE_StructVectorCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, grid, &b_dbl);
         HYPRE_StructVectorInitialize_pre( solver_precision,b_dbl);
         HYPRE_StructVectorCreate_pre( precond_precision,hypre_MPI_COMM_WORLD, grid, &b_flt);
         HYPRE_StructVectorInitialize_pre( precond_precision,b_flt);

         /*-----------------------------------------------------------
          * For periodic b.c. in all directions, need rhs to satisfy
          * compatibility condition. Achieved by setting a source and
          *  sink of equal strength.  All other problems have rhs = 1.
          *-----------------------------------------------------------*/

         AddValuesVector_mp(grid, b_dbl, periodic, one_slvr, sizeof(double));
         AddValuesVector_mp(grid, b_flt, periodic, one_pc, sizeof(float));
         HYPRE_StructVectorAssemble_pre( solver_precision,b_dbl);
         HYPRE_StructVectorAssemble_pre( precond_precision,b_flt);

         HYPRE_StructVectorCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, grid, &x_dbl);
         HYPRE_StructVectorCreate_pre( precond_precision,hypre_MPI_COMM_WORLD, grid, &x_flt);
         HYPRE_StructVectorInitialize_pre( solver_precision,x_dbl);
         HYPRE_StructVectorInitialize_pre( precond_precision,x_flt);
       
         AddValuesVector_mp(grid, x_dbl, periodx0, zero_slvr, sizeof(double));
         AddValuesVector_mp(grid, x_flt, periodx0, zero_pc, sizeof(float));
         HYPRE_StructVectorAssemble_pre(solver_precision,x_dbl);
         HYPRE_StructVectorAssemble_pre(precond_precision,x_flt);

         HYPRE_StructGridDestroy_pre( solver_precision,grid);

         for (i = 0; i < nblocks; i++)
         {
            hypre_Free(iupper[i], HYPRE_MEMORY_HOST);
            hypre_Free(ilower[i], HYPRE_MEMORY_HOST);
         }
         hypre_Free(ilower, HYPRE_MEMORY_HOST);
         hypre_Free(iupper, HYPRE_MEMORY_HOST);
      }

      /* linear system complete  */

      hypre_EndTiming(time_index);
      if ( reps == 1 || (solver_id != 0 && solver_id != 1 && solver_id != 3 && solver_id != 4) )
      {
         hypre_PrintTiming("Struct Interface", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }
      else if ( rep == reps - 1 )
      {
         hypre_FinalizeTiming(time_index);
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
         time_index = hypre_InitializeTiming("SMG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructSMGCreate_pre( solver_precision,MPI_COMM_WORLD, &solver);
         HYPRE_StructSMGSetMaxIter_pre( solver_precision,solver, 50);
         HYPRE_StructSMGSetTol_pre( solver_precision,solver, tol);
         HYPRE_StructSMGSetRelChange_pre( solver_precision,solver, 0);
         HYPRE_StructSMGSetNumPreRelax_pre( solver_precision,solver, n_pre);
         HYPRE_StructSMGSetNumPostRelax_pre( solver_precision,solver, n_post);
         HYPRE_StructSMGSetPrintLevel_pre( solver_precision,solver, 1);
         HYPRE_StructSMGSetLogging_pre( solver_precision,solver, 1);
#if 0//defined(HYPRE_USING_CUDA)
         HYPRE_StructSMGSetDeviceLevel(solver, device_level);
#endif

#if 0//defined(HYPRE_USING_CUDA)
         hypre_box_print = 0;
#endif
         HYPRE_StructSMGSetup_pre( solver_precision,solver, A_dbl, b_dbl, x_dbl);

#if 0//defined(HYPRE_USING_CUDA)
         hypre_box_print = 0;
#endif

         hypre_EndTiming(time_index);
         if ( reps == 1 )
         {
            hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }
         else if ( rep == reps - 1 )
         {
            hypre_FinalizeTiming(time_index);
         }

         time_index = hypre_InitializeTiming("SMG Solve");
         hypre_BeginTiming(time_index);

         HYPRE_StructSMGSolve_pre( solver_precision,solver, A_dbl, b_dbl, x_dbl);

         hypre_EndTiming(time_index);
         if ( reps == 1 )
         {
            hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }
         else if ( rep == reps - 1 )
         {
            hypre_PrintTiming("Interface, Setup, and Solve times:",
                                  hypre_MPI_COMM_WORLD );
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }

         HYPRE_StructSMGGetNumIterations_pre( solver_precision,solver, &num_iterations);
         HYPRE_StructSMGGetFinalRelativeResidualNorm_pre( solver_precision,solver, &final_res_norm);
         HYPRE_StructSMGDestroy_pre( solver_precision,solver);
      }

      /*-----------------------------------------------------------
       * Solve the system using PFMG
       *-----------------------------------------------------------*/

      else if ( solver_id == 1 || solver_id == 3 || solver_id == 4 )
      {
         time_index = hypre_InitializeTiming("PFMG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructPFMGCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, &solver);
         HYPRE_StructPFMGSetMaxIter_pre( solver_precision,solver, 200);
         HYPRE_StructPFMGSetTol_pre( solver_precision,solver, tol);
         HYPRE_StructPFMGSetRelChange_pre( solver_precision,solver, 0);
         HYPRE_StructPFMGSetRAPType_pre( solver_precision,solver, rap);
         HYPRE_StructPFMGSetRelaxType_pre( solver_precision,solver, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight_pre( solver_precision,solver, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax_pre( solver_precision,solver, n_pre);
         HYPRE_StructPFMGSetNumPostRelax_pre( solver_precision,solver, n_post);
         HYPRE_StructPFMGSetSkipRelax_pre( solver_precision,solver, skip);
         HYPRE_StructPFMGSetPrintLevel_pre( solver_precision,solver, 1);
         HYPRE_StructPFMGSetLogging_pre( solver_precision,solver, 1);

#if 0//defined(HYPRE_USING_CUDA)
         HYPRE_StructPFMGSetDeviceLevel(solver, device_level);
#endif

         HYPRE_StructPFMGSetup_pre( solver_precision,solver, A_dbl, b_dbl, x_dbl);

         hypre_EndTiming(time_index);
         if ( reps == 1 )
         {
            hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }
         else if ( rep == reps - 1 )
         {
            hypre_FinalizeTiming(time_index);
         }

         time_index = hypre_InitializeTiming("PFMG Solve");
         hypre_BeginTiming(time_index);


         HYPRE_StructPFMGSolve_pre( solver_precision,solver, A_dbl, b_dbl, x_dbl);

         hypre_EndTiming(time_index);
         if ( reps == 1 )
         {
            hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }
         else if ( rep == reps - 1 )
         {
            hypre_PrintTiming("Interface, Setup, and Solve times",
                                  hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }

         HYPRE_StructPFMGGetNumIterations_pre( solver_precision,solver, &num_iterations);
         HYPRE_StructPFMGGetFinalRelativeResidualNorm_pre( solver_precision,solver, &final_res_norm);
         HYPRE_StructPFMGDestroy_pre( solver_precision,solver);
      }

      /*-----------------------------------------------------------
       * Solve the system using CG
       *-----------------------------------------------------------*/

      if ((solver_id > 9) && (solver_id < 20))
      {
         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructPCGCreate_pre( solver_precision,MPI_COMM_WORLD, &solver);
         HYPRE_PCGSetMaxIter_pre( solver_precision, (HYPRE_Solver)solver, 100 );
         HYPRE_PCGSetTol_pre( solver_precision, (HYPRE_Solver)solver, tol );
         HYPRE_PCGSetTwoNorm_pre( solver_precision, (HYPRE_Solver)solver, 1 );
         HYPRE_PCGSetRelChange_pre( solver_precision, (HYPRE_Solver)solver, 0 );
         HYPRE_PCGSetPrintLevel_pre( solver_precision, (HYPRE_Solver)solver, print_level );
         HYPRE_PCGSetRecomputeResidual_pre( solver_precision, (HYPRE_Solver)solver, 1 );
         HYPRE_PCGSetFlex_pre( solver_precision, (HYPRE_Solver)solver, flex );

         if (solver_id == 10)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, &precond);
            //HYPRE_StructSMGSetMemoryUse_pre( solver_precision,precond, 0);
            HYPRE_StructSMGSetMaxIter_pre( solver_precision,precond, 1);
            HYPRE_StructSMGSetTol_pre( solver_precision,precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_pre( solver_precision,precond);
            HYPRE_StructSMGSetNumPreRelax_pre( solver_precision,precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_pre( solver_precision,precond, n_post);
            HYPRE_StructSMGSetPrintLevel_pre( solver_precision,precond, 0);
            HYPRE_StructSMGSetLogging_pre( solver_precision,precond, 0);

            HYPRE_PCGSetPrecond_pre( solver_precision, (HYPRE_Solver) solver,
                                     (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_dbl,
                                     (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_dbl,
                                     (HYPRE_Solver) precond);
         }
         else if (solver_id == 15)
         {
            HYPRE_StructSMGCreate_pre( precond_precision,hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMaxIter_pre( precond_precision,precond, 1);
            HYPRE_StructSMGSetTol_pre( precond_precision,precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_pre( precond_precision,precond);
            HYPRE_StructSMGSetNumPreRelax_pre( precond_precision,precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_pre( precond_precision,precond, n_post);
            HYPRE_StructSMGSetPrintLevel_pre( precond_precision,precond, 0);
            HYPRE_StructSMGSetLogging_pre( precond_precision,precond, 0);
            HYPRE_PCGSetPrecondMatrix_pre( solver_precision,(HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_PCGSetPrecond_pre( solver_precision, (HYPRE_Solver) solver,
                                     (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_mp,
                                     (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_mp,
                                     (HYPRE_Solver) precond);
         }

         else if (solver_id == 11 || solver_id == 13 || solver_id == 14)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_pre( solver_precision,precond, 1);
            HYPRE_StructPFMGSetTol_pre( solver_precision,precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_pre( solver_precision,precond);
            HYPRE_StructPFMGSetRAPType_pre( solver_precision,precond, rap);
            HYPRE_StructPFMGSetRelaxType_pre( solver_precision,precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_pre( solver_precision,precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_pre( solver_precision,precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_pre( solver_precision,precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_pre( solver_precision,precond, skip);
            /*HYPRE_StructPFMGSetDxyz_pre( solver_precision,precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel_pre( solver_precision,precond, 0);
            HYPRE_StructPFMGSetLogging_pre( solver_precision,precond, 0);
#if 0//defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_PCGSetPrecond_pre( solver_precision,(HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_dbl,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_dbl,
                                    (HYPRE_Solver) precond);
         }
         else if (solver_id == 16)
         {
            HYPRE_StructPFMGCreate_pre( precond_precision,hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_pre( precond_precision,precond, 1);
            HYPRE_StructPFMGSetTol_pre( precond_precision,precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_pre( precond_precision,precond);
            HYPRE_StructPFMGSetRAPType_pre( precond_precision,precond, rap);
            HYPRE_StructPFMGSetRelaxType_pre( precond_precision,precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_pre( precond_precision,precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_pre( precond_precision,precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_pre( precond_precision,precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_pre( precond_precision,precond, skip);
            HYPRE_StructPFMGSetPrintLevel_pre( precond_precision,precond, 0);
            HYPRE_StructPFMGSetLogging_pre( precond_precision,precond, 0);
            HYPRE_PCGSetPrecondMatrix_pre( solver_precision,(HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_PCGSetPrecond_pre( solver_precision,(HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_mp,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_mp,
                                    (HYPRE_Solver) precond);
         }

         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_PCGSetPrecond_pre( solver_precision,(HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale_dbl,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup_dbl,
                                    (HYPRE_Solver) precond);
         }

         HYPRE_PCGSetup_pre( solver_precision, (HYPRE_Solver)solver,
                             (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("PCG Solve");
         hypre_BeginTiming(time_index);

         HYPRE_PCGSolve_pre( solver_precision, (HYPRE_Solver) solver,
                             (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_PCGGetNumIterations_pre( solver_precision, (HYPRE_Solver)solver, &num_iterations );
         HYPRE_PCGGetFinalRelativeResidualNorm_pre( solver_precision, (HYPRE_Solver)solver,
                                                    &final_res_norm );
         HYPRE_StructPCGDestroy_pre( solver_precision,solver);

         if (solver_id == 10)
         {
            HYPRE_StructSMGDestroy_pre( solver_precision,precond);
         }
         else if (solver_id == 15)
         {
            HYPRE_StructSMGDestroy_pre( precond_precision,precond);
         }
         else if (solver_id == 11 || solver_id == 13 || solver_id == 14)
         {
            HYPRE_StructPFMGDestroy_pre( solver_precision,precond);
         }
         else if (solver_id == 16)
         {
            HYPRE_StructPFMGDestroy_pre( precond_precision,precond);
         }
      }

      /*-----------------------------------------------------------
       * Solve the system using GMRES
       *-----------------------------------------------------------*/

      if ((solver_id > 29) && (solver_id < 40))
      {
         time_index = hypre_InitializeTiming("GMRES Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructGMRESCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, &solver);
         HYPRE_GMRESSetKDim_pre( solver_precision, (HYPRE_Solver) solver, 5 );
         HYPRE_GMRESSetMaxIter_pre( solver_precision, (HYPRE_Solver)solver, 100 );
         HYPRE_GMRESSetTol_pre( solver_precision, (HYPRE_Solver)solver, tol );
         HYPRE_GMRESSetRelChange_pre( solver_precision, (HYPRE_Solver)solver, 0 );
         HYPRE_GMRESSetPrintLevel_pre( solver_precision, (HYPRE_Solver)solver, print_level );
         HYPRE_GMRESSetLogging_pre( solver_precision, (HYPRE_Solver)solver, 1 );

         if (solver_id == 30)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, &precond);
            //HYPRE_StructSMGSetMemoryUse_pre( solver_precision,precond, 0);
            HYPRE_StructSMGSetMaxIter_pre( solver_precision,precond, 1);
            HYPRE_StructSMGSetTol_pre( solver_precision,precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_pre( solver_precision,precond);
            HYPRE_StructSMGSetNumPreRelax_pre( solver_precision,precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_pre( solver_precision,precond, n_post);
            HYPRE_StructSMGSetPrintLevel_pre( solver_precision,precond, 0);
            HYPRE_StructSMGSetLogging_pre( solver_precision,precond, 0);
            HYPRE_GMRESSetPrecond_pre( solver_precision,(HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_dbl,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_dbl,
                                      (HYPRE_Solver)precond);
         }
         else if (solver_id == 35)
         {
            HYPRE_StructSMGCreate_pre( precond_precision,hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMaxIter_pre( precond_precision,precond, 1);
            HYPRE_StructSMGSetTol_pre( precond_precision,precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_pre( precond_precision,precond);
            HYPRE_StructSMGSetNumPreRelax_pre( precond_precision,precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_pre( precond_precision,precond, n_post);
            HYPRE_StructSMGSetPrintLevel_pre( precond_precision,precond, 0);
            HYPRE_StructSMGSetLogging_pre( precond_precision,precond, 0);
            HYPRE_GMRESSetPrecondMatrix_pre( solver_precision,(HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_GMRESSetPrecond_pre( solver_precision,(HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_mp,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_mp,
                                      (HYPRE_Solver)precond);
         }

         else if (solver_id == 31)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_pre( solver_precision,precond, 1);
            HYPRE_StructPFMGSetTol_pre( solver_precision,precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_pre( solver_precision,precond);
            HYPRE_StructPFMGSetRAPType_pre( solver_precision,precond, rap);
            HYPRE_StructPFMGSetRelaxType_pre( solver_precision,precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_pre( solver_precision,precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_pre( solver_precision,precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_pre( solver_precision,precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_pre( solver_precision,precond, skip);
            /*HYPRE_StructPFMGSetDxyz_pre( solver_precision,precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel_pre( solver_precision,precond, 0);
            HYPRE_StructPFMGSetLogging_pre( solver_precision,precond, 0);
            HYPRE_GMRESSetPrecond_pre( solver_precision,(HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_dbl,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_dbl,
                                      (HYPRE_Solver)precond);
         }
         else if (solver_id == 36)
         {
            HYPRE_StructPFMGCreate_pre( precond_precision,hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_pre( precond_precision,precond, 1);
            HYPRE_StructPFMGSetTol_pre( precond_precision,precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_pre( precond_precision,precond);
            HYPRE_StructPFMGSetRAPType_pre( precond_precision,precond, rap);
            HYPRE_StructPFMGSetRelaxType_pre( precond_precision,precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_pre( precond_precision,precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_pre( precond_precision,precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_pre( precond_precision,precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_pre( precond_precision,precond, skip);
            HYPRE_StructPFMGSetPrintLevel_pre( precond_precision,precond, 0);
            HYPRE_StructPFMGSetLogging_pre( precond_precision,precond, 0);
            HYPRE_GMRESSetPrecondMatrix_pre( solver_precision,(HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_GMRESSetPrecond_pre( solver_precision,(HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_mp,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_mp,
                                      (HYPRE_Solver)precond);
         }

         else if (solver_id == 38)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_GMRESSetPrecond_pre( solver_precision,(HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale_dbl,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup_dbl,
                                      (HYPRE_Solver)precond);
         }

         HYPRE_GMRESSetup_dbl ( (HYPRE_Solver)solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl,
                                (HYPRE_Vector)x_dbl );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("GMRES Solve");
         hypre_BeginTiming(time_index);

         HYPRE_GMRESSolve_dbl
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_GMRESGetNumIterations_pre( solver_precision, (HYPRE_Solver)solver, &num_iterations);
         HYPRE_GMRESGetFinalRelativeResidualNorm_pre( solver_precision, (HYPRE_Solver)solver, &final_res_norm);
         HYPRE_StructGMRESDestroy_pre( solver_precision,solver);

         if (solver_id == 30)
         {
            HYPRE_StructSMGDestroy_pre( solver_precision,precond);
         }
         else if (solver_id == 35)
         {
            HYPRE_StructSMGDestroy_pre( precond_precision,precond);
         }
         else if (solver_id == 31)
         {
            HYPRE_StructPFMGDestroy_pre( solver_precision,precond);
         }
         else if (solver_id == 36)
         {
            HYPRE_StructPFMGDestroy_pre( precond_precision,precond);
         }
      }

      /*-----------------------------------------------------------
       * Solve the system using BiCGTAB
       *-----------------------------------------------------------*/

      if ((solver_id > 39) && (solver_id < 50))
      {
         time_index = hypre_InitializeTiming("BiCGSTAB Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructBiCGSTABCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, &solver);
         HYPRE_BiCGSTABSetMaxIter_pre( solver_precision, (HYPRE_Solver)solver, 100 );
         HYPRE_BiCGSTABSetTol_pre( solver_precision, (HYPRE_Solver)solver, tol );
         HYPRE_BiCGSTABSetPrintLevel_pre( solver_precision, (HYPRE_Solver)solver, print_level );
         HYPRE_BiCGSTABSetLogging_pre( solver_precision, (HYPRE_Solver)solver, 1 );

         if (solver_id == 40)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, &precond);
            //HYPRE_StructSMGSetMemoryUse_pre( solver_precision,precond, 0);
            HYPRE_StructSMGSetMaxIter_pre( solver_precision,precond, 1);
            HYPRE_StructSMGSetTol_pre( solver_precision,precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_pre( solver_precision,precond);
            HYPRE_StructSMGSetNumPreRelax_pre( solver_precision,precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_pre( solver_precision,precond, n_post);
            HYPRE_StructSMGSetPrintLevel_pre( solver_precision,precond, 0);
            HYPRE_StructSMGSetLogging_pre( solver_precision,precond, 0);
            HYPRE_BiCGSTABSetPrecond_pre( solver_precision,(HYPRE_Solver)solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_dbl,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_dbl,
                                         (HYPRE_Solver)precond);
         }
         else if (solver_id == 45)
         {
            HYPRE_StructSMGCreate_pre( precond_precision,hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMaxIter_pre( precond_precision,precond, 1);
            HYPRE_StructSMGSetTol_pre( precond_precision,precond, 0.0);
            HYPRE_StructSMGSetZeroGuess_pre( precond_precision,precond);
            HYPRE_StructSMGSetNumPreRelax_pre( precond_precision,precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax_pre( precond_precision,precond, n_post);
            HYPRE_StructSMGSetPrintLevel_pre( precond_precision,precond, 0);
            HYPRE_StructSMGSetLogging_pre( precond_precision,precond, 0);
            HYPRE_BiCGSTABSetPrecondMatrix_pre( solver_precision,(HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_BiCGSTABSetPrecond_pre( solver_precision,(HYPRE_Solver)solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve_mp,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup_mp,
                                         (HYPRE_Solver)precond);
         }

         else if (solver_id == 41)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate_pre( solver_precision,hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_pre( solver_precision,precond, 1);
            HYPRE_StructPFMGSetTol_pre( solver_precision,precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_pre( solver_precision,precond);
            HYPRE_StructPFMGSetRAPType_pre( solver_precision,precond, rap);
            HYPRE_StructPFMGSetRelaxType_pre( solver_precision,precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_pre( solver_precision,precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_pre( solver_precision,precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_pre( solver_precision,precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_pre( solver_precision,precond, skip);
            /*HYPRE_StructPFMGSetDxyz_pre( solver_precision,precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel_pre( solver_precision,precond, 0);
            HYPRE_StructPFMGSetLogging_pre( solver_precision,precond, 0);
            HYPRE_BiCGSTABSetPrecond_pre( solver_precision,(HYPRE_Solver)solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_dbl,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_dbl,
                                         (HYPRE_Solver)precond);
         }
         else if (solver_id == 46)
         {
            HYPRE_StructPFMGCreate_pre( precond_precision,hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter_pre( precond_precision,precond, 1);
            HYPRE_StructPFMGSetTol_pre( precond_precision,precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess_pre( solver_precision,precond);
            HYPRE_StructPFMGSetRAPType_pre( precond_precision,precond, rap);
            HYPRE_StructPFMGSetRelaxType_pre( precond_precision,precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight_pre( precond_precision,precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax_pre( precond_precision,precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax_pre( precond_precision,precond, n_post);
            HYPRE_StructPFMGSetSkipRelax_pre( precond_precision,precond, skip);
            HYPRE_StructPFMGSetPrintLevel_pre( precond_precision,precond, 0);
            HYPRE_StructPFMGSetLogging_pre( precond_precision,precond, 0);
            HYPRE_BiCGSTABSetPrecondMatrix_pre( solver_precision,(HYPRE_Solver)solver, (HYPRE_Matrix)A_flt);
            HYPRE_BiCGSTABSetPrecond_pre( solver_precision,(HYPRE_Solver)solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve_mp,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup_mp,
                                         (HYPRE_Solver)precond);
         }

         else if (solver_id == 48)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_BiCGSTABSetPrecond_pre( solver_precision,(HYPRE_Solver)solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale_dbl,
                                         (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup_dbl,
                                         (HYPRE_Solver)precond);
         }

         HYPRE_BiCGSTABSetup_dbl
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("BiCGSTAB Solve");
         hypre_BeginTiming(time_index);

         HYPRE_BiCGSTABSolve_dbl
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A_dbl, (HYPRE_Vector)b_dbl, (HYPRE_Vector)x_dbl);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_BiCGSTABGetNumIterations_pre( solver_precision, (HYPRE_Solver)solver, &num_iterations);
         HYPRE_BiCGSTABGetFinalRelativeResidualNorm_pre( solver_precision, (HYPRE_Solver)solver, &final_res_norm);
         HYPRE_StructBiCGSTABDestroy_pre( solver_precision,solver);

         if (solver_id == 40)
         {
            HYPRE_StructSMGDestroy_pre( solver_precision,precond);
         }
         else if (solver_id == 45)
         {
            HYPRE_StructSMGDestroy_pre( precond_precision,precond);
         }
         else if (solver_id == 41)
         {
            HYPRE_StructPFMGDestroy_pre( solver_precision,precond);
         }
         else if (solver_id == 46)
         {
            HYPRE_StructPFMGDestroy_pre( precond_precision,precond);
         }
      }

      /*-----------------------------------------------------------
       * Print the solution and other info
       *-----------------------------------------------------------*/

      if (print_system)
      {
         HYPRE_StructVectorPrint_pre( solver_precision,"struct.out.x", x_dbl, 0);
      }

      if (myid == 0 && rep == reps - 1 )
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
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

         matvec_data = hypre_StructMatvecCreate_pre( solver_precision,);
         hypre_StructMatvecSetup_pre( solver_precision,matvec_data, A, x);

         time_index = hypre_InitializeTiming("Matvec");
         hypre_BeginTiming(time_index);

         for (i = 0; i < imax; i++)
         {
            hypre_StructMatvecCompute_pre( solver_precision,matvec_data, 1.0, A, x, 1.0, b);
         }
         /* this counts mult-adds */
         hypre_IncFLOPCount_pre( solver_precision,7 * N * imax);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Matvec time", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         hypre_StructMatvecDestroy_pre( solver_precision,matvec_data);
      }
#endif

      /*-----------------------------------------------------------
       * Finalize things
       *-----------------------------------------------------------*/

      HYPRE_StructStencilDestroy_pre( solver_precision,stencil);
      HYPRE_StructMatrixDestroy_pre( solver_precision,A_dbl);
      HYPRE_StructVectorDestroy_pre( solver_precision,b_dbl);
      HYPRE_StructVectorDestroy_pre( solver_precision,x_dbl);

      for ( i = 0; i < (dim + 1); i++)
      {
         hypre_Free(offsets[i], HYPRE_MEMORY_HOST);
      }
      hypre_Free(offsets, HYPRE_MEMORY_HOST);
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
         hypre_printf("Error: nonzero GPU memory allocated with the HOST mode\n");
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
AddValuesVector_mp(hypre_StructGrid   *gridvector,
                     hypre_StructVector *zvector,
                     HYPRE_Int          *period,
                     void               *value,
                     size_t             size  )
{
   /* #include  "_hypre_struct_mv.h" */
   HYPRE_Int            i, ierr = 0;
   hypre_BoxArray      *gridboxes;
   HYPRE_Int            ib;
   hypre_IndexRef       ilower;
   hypre_IndexRef       iupper;
   hypre_Box           *box;
   char               *values;
   HYPRE_Int            volume, dim;

   gridboxes =  hypre_StructGridBoxes(gridvector);
   dim       =  hypre_StructGridNDim(gridvector);
   
   ib = 0;
   hypre_ForBoxI(ib, gridboxes)
   {
      box      = hypre_BoxArrayBox(gridboxes, ib);
      volume   = hypre_BoxVolume(box);
      values = hypre_CAlloc((size_t) volume, size, HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------
       * For periodic b.c. in all directions, need rhs to satisfy
       * compatibility condition. Achieved by setting a source and
       *  sink of equal strength.  All other problems have rhs = 1.
       *-----------------------------------------------------------*/

      if ((dim == 2 && period[0] != 0 && period[1] != 0) ||
          (dim == 3 && period[0] != 0 && period[1] != 0 && period[2] != 0))
      {
         memcpy(&values[0],&value,size);
         switch (hypre_StructVectorPrecision(zvector)){
            case HYPRE_REAL_SINGLE:
               ((hypre_float*)values)[volume - 1] = -(*(hypre_float *)value);
               break;
            case HYPRE_REAL_DOUBLE:
               ((hypre_double*)values)[volume - 1] = -(*(hypre_double *)value);
               break;
            case HYPRE_REAL_LONGDOUBLE:
               ((hypre_long_double*)values)[volume - 1] = -(*(hypre_long_double *)value);
               break;
         }
      }
      else
      {
         for (i = 0; i < volume; i++)
         {
            memcpy((values + i*size),value,size);
         }
      }

      ilower = hypre_BoxIMin(box);
      iupper = hypre_BoxIMax(box);

      HYPRE_StructVectorSetBoxValues_pre(hypre_StructVectorPrecision(zvector), zvector, ilower, iupper, values);

      hypre_Free(values, HYPRE_MEMORY_HOST);
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
AddValuesMatrix_mp(HYPRE_StructMatrix A,
                    HYPRE_StructGrid   gridmatrix,
                    void              *cx,
                    void              *cy,
                    void              *cz,
                    void              *conx,
                    void              *cony,
                    void              *conz,
                    size_t            size)
{

   HYPRE_Int            d, ierr = 0;
   hypre_BoxArray      *gridboxes;
   HYPRE_Int            s, bi;
   hypre_IndexRef       ilower;
   hypre_IndexRef       iupper;
   hypre_Box           *box;
   char                *values;
   void                *east, *west;
   void                *north, *south;
   void                *top, *bottom;
   void                *center;
   char                *stcoeffs; /* Stencil Coeffs: center, east, west, north, south, top, bottom */
   HYPRE_Int            volume, dim, sym;
   HYPRE_Int           *stencil_indices;
   HYPRE_Int            stencil_size;
   HYPRE_Int            constant_coefficient;

   /* get precision of struct matrix */
   HYPRE_Precision precision_A = hypre_StructMatrixPrecision(A);

   gridboxes = hypre_StructGridBoxes(gridmatrix);
   dim       = hypre_StructGridNDim(gridmatrix);
   sym       = hypre_StructMatrixSymmetric(A);
   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

   bi = 0;
   
   /* Alocate and set pointers to stencil coeffs */
   stcoeffs = hypre_CAlloc((size_t) 7, size, HYPRE_MEMORY_HOST);
   center = stcoeffs;
   east   = stcoeffs + size;
   west   = stcoeffs + 2*size;
   north  = stcoeffs + 3*size;
   south  = stcoeffs + 4*size;
   top    = stcoeffs + 5*size;
   bottom = stcoeffs + 6*size;
   /* set stencil values */
   switch (precision_A) {
      case HYPRE_REAL_SINGLE:
         *(hypre_float*)east = -(*(hypre_float *)cx);
         *(hypre_float*)west = -(*(hypre_float *)cx);
         *(hypre_float*)north = -(*(hypre_float *)cy);
         *(hypre_float*)south = -(*(hypre_float *)cy);
         *(hypre_float*)top = -(*(hypre_float *)cz);
         *(hypre_float*)bottom = -(*(hypre_float *)cz);
         *(hypre_float*)center = 2 * (*(hypre_float *)cx);
         if (dim > 1) { *(hypre_float*)center += 2.0 * (*(hypre_float *)cy); }
         if (dim > 2) { *(hypre_float*)center += 2.0 * (*(hypre_float *)cz); }
         break;
      case HYPRE_REAL_DOUBLE:
         *(hypre_double*)east = -(*(hypre_double *)cx);
         *(hypre_double*)west = -(*(hypre_double *)cx);
         *(hypre_double*)north = -(*(hypre_double *)cy);
         *(hypre_double*)south = -(*(hypre_double *)cy);
         *(hypre_double*)top = -(*(hypre_double *)cz);
         *(hypre_double*)bottom = -(*(hypre_double *)cz);
         *(hypre_double*)center = 2 * (*(hypre_double *)cx);
         if (dim > 1) { *(hypre_double*)center += 2.0 * (*(hypre_double *)cy); }
         if (dim > 2) { *(hypre_double*)center += 2.0 * (*(hypre_double *)cz); }
         break;      
      case HYPRE_REAL_LONGDOUBLE:
         *(hypre_long_double*)east = -(*(hypre_long_double *)cx);
         *(hypre_long_double*)west = -(*(hypre_long_double *)cx);
         *(hypre_long_double*)north = -(*(hypre_long_double *)cy);
         *(hypre_long_double*)south = -(*(hypre_long_double *)cy);
         *(hypre_long_double*)top = -(*(hypre_long_double *)cz);
         *(hypre_long_double*)bottom = -(*(hypre_long_double *)cz);
         *(hypre_long_double*)center = 2 * (*(hypre_long_double *)cx);
         if (dim > 1) { *(hypre_long_double*)center += 2.0 * (*(hypre_long_double *)cy); }
         if (dim > 2) { *(hypre_long_double*)center += 2.0 * (*(hypre_long_double *)cz); }
         break;
   }
   stencil_size = 1 + (2 - sym) * dim;
   stencil_indices = hypre_CAlloc((size_t) stencil_size, (size_t)sizeof(HYPRE_Int),
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
            values = hypre_CAlloc((size_t)(stencil_size * volume), size,  HYPRE_MEMORY_HOST);

            if (dim == 1)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  memcpy((values + i*size),west,size);
                  memcpy((values + (i+1)*size),center,size);
               }
            }
            else if (dim == 2)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  memcpy((values + i*size),west,size);
                  memcpy((values + (i+1)*size),south,size);                  
                  memcpy((values + (i+2)*size),center,size);    
               }
            }
            else if (dim == 3)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  memcpy((values + i*size),west,size);
                  memcpy((values + (i+1)*size),south,size);                  
                  memcpy((values + (i+2)*size),bottom,size);   
                  memcpy((values + (i+3)*size),center,size);
               }
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues_pre(precision_A, A, ilower, iupper, stencil_size,
                                               stencil_indices, values);
            hypre_Free(values, HYPRE_MEMORY_HOST);
         }
      }
      else if ( constant_coefficient == 1 )
      {
         values = hypre_CAlloc((size_t)stencil_size, size, HYPRE_MEMORY_HOST);
         switch (dim)
         {
            case 1:
               memcpy((values),west,size);
               memcpy((values + size),center,size);
               break;
            case 2:
               memcpy((values),west,size);
               memcpy((values + size),south,size);
               memcpy((values + 2*size),center,size);
               break;
            case 3:
               memcpy((values),west,size);
               memcpy((values + size),south,size);
               memcpy((values + 2*size),bottom,size);
               memcpy((values + 3*size),center,size);
               break;
         }
         if (hypre_BoxArraySize(gridboxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues_pre(precision_A, A, stencil_size,
                                                    stencil_indices, values);
         }
         hypre_Free(values, HYPRE_MEMORY_HOST);
      }
      else
      {
         /* stencil index for the center equals dim, so it's easy to leave out */
         values = hypre_CAlloc_flt((size_t)(stencil_size - 1), (size_t)sizeof(float ), HYPRE_MEMORY_HOST);
         switch (dim)
         {
            case 1:
               memcpy((values),west,size);
               break;
            case 2:
               memcpy((values),west,size);
               memcpy((values + size),south,size);
               break;
            case 3:
               memcpy((values),west,size);
               memcpy((values + size),south,size);
               memcpy((values + 2*size),bottom,size);
               break;
         }
         if (hypre_BoxArraySize(gridboxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues_pre(precision_A, A, stencil_size - 1,
                                                    stencil_indices, values);
         }
         hypre_Free(values, HYPRE_MEMORY_HOST);

         hypre_ForBoxI(bi, gridboxes)
         {
            box    = hypre_BoxArrayBox(gridboxes, bi);
            /* hypre_BoxVolume is an integer option so we can use default function call */
            volume = hypre_BoxVolume(box);
            values = hypre_CAlloc((size_t) volume, size, HYPRE_MEMORY_HOST);
            HYPRE_Int i;

            for (i = 0; i < volume; i++)
            {
               memcpy((values +i*size), center, size);
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues_pre(precision_A, A, ilower, iupper, 1,
                                               stencil_indices + dim, values);
            hypre_Free(values, HYPRE_MEMORY_HOST);
         }
      }
   }
   else
   {
      switch (precision_A){
         case HYPRE_REAL_SINGLE:
            if (*(hypre_float *)conx > 0.0f)
            {
               *(hypre_float*)west -= *(hypre_float*)conx;
               *(hypre_float*)center += *(hypre_float*)conx;
            }
            else if (*(hypre_float *)conx < 0.0f)
            {
               *(hypre_float*)east += *(hypre_float*)conx;
               *(hypre_float*)center -= *(hypre_float*)conx;
            }
            if (*(hypre_float *)cony > 0.0)
            {
               *(hypre_float*)south -= *(hypre_float*)cony;
               *(hypre_float*)center += *(hypre_float*)cony;
            }
            else if (*(hypre_float *)cony < 0.0)
            {
               *(hypre_float*)north += *(hypre_float*)cony;
               *(hypre_float*)center -= *(hypre_float*)cony;
            }
            if (*(hypre_float *)conz > 0.0)
            {
               *(hypre_float*)bottom -= *(hypre_float*)conz;
               *(hypre_float*)center += *(hypre_float*)conz;
            }
            else if (*(hypre_float *)conz < 0.0)
            {
               *(hypre_float*)top += *(hypre_float*)conz;
               *(hypre_float*)center -= *(hypre_float*)conz;
            }
            break;
         case HYPRE_REAL_DOUBLE:
            if (*(hypre_double *)conx > 0.0f)
            {
               *(hypre_double*)west -= *(hypre_double*)conx;
               *(hypre_double*)center += *(hypre_double*)conx;
            }
            else if (*(hypre_double *)conx < 0.0f)
            {
               *(hypre_double*)east += *(hypre_double*)conx;
               *(hypre_double*)center -= *(hypre_double*)conx;
            }
            if (*(hypre_double *)cony > 0.0)
            {
               *(hypre_double*)south -= *(hypre_double*)cony;
               *(hypre_double*)center += *(hypre_double*)cony;
            }
            else if (*(hypre_double *)cony < 0.0)
            {
               *(hypre_double*)north += *(hypre_double*)cony;
               *(hypre_double*)center -= *(hypre_double*)cony;
            }
            if (*(hypre_double *)conz > 0.0)
            {
               *(hypre_double*)bottom -= *(hypre_double*)conz;
               *(hypre_double*)center += *(hypre_double*)conz;
            }
            else if (*(hypre_double *)conz < 0.0)
            {
               *(hypre_double*)top += *(hypre_double*)conz;
               *(hypre_double*)center -= *(hypre_double*)conz;
            }
            break;
         case HYPRE_REAL_LONGDOUBLE:
            if (*(hypre_long_double *)conx > 0.0f)
            {
               *(hypre_long_double*)west -= *(hypre_long_double*)conx;
               *(hypre_long_double*)center += *(hypre_long_double*)conx;
            }
            else if (*(hypre_long_double *)conx < 0.0f)
            {
               *(hypre_long_double*)east += *(hypre_long_double*)conx;
               *(hypre_long_double*)center -= *(hypre_long_double*)conx;
            }
            if (*(hypre_long_double *)cony > 0.0)
            {
               *(hypre_long_double*)south -= *(hypre_long_double*)cony;
               *(hypre_long_double*)center += *(hypre_long_double*)cony;
            }
            else if (*(hypre_long_double *)cony < 0.0)
            {
               *(hypre_long_double*)north += *(hypre_long_double*)cony;
               *(hypre_long_double*)center -= *(hypre_long_double*)cony;
            }
            if (*(hypre_long_double *)conz > 0.0)
            {
               *(hypre_long_double*)bottom -= *(hypre_long_double*)conz;
               *(hypre_long_double*)center += *(hypre_long_double*)conz;
            }
            else if (*(hypre_long_double *)conz < 0.0)
            {
               *(hypre_long_double*)top += *(hypre_long_double*)conz;
               *(hypre_long_double*)center -= *(hypre_long_double*)conz;
            }
            break;
      }
      if ( constant_coefficient == 0 )
      {
         hypre_ForBoxI(bi, gridboxes)
         {
            box    = hypre_BoxArrayBox(gridboxes, bi);
            /* hypre_BoxVolume is an integer option so we can use default function call */
            volume = hypre_BoxVolume(box);
            values = hypre_CAlloc((size_t)(stencil_size * volume), size, HYPRE_MEMORY_HOST);

            for (d = 0; d < volume; d++)
            {
               HYPRE_Int i = stencil_size * d;
               switch (dim)
               {
                  case 1:
                     memcpy(values + i*size, west, size);
                     memcpy(values + (i+1)*size, center, size);
                     memcpy(values + (i+2)*size, east, size);
                     break;
                  case 2:
                     memcpy(values + i*size, west, size);
                     memcpy(values + (i+1)*size, south, size);
                     memcpy(values + (i+2)*size, center, size);
                     memcpy(values + (i+3)*size, east, size);
                     memcpy(values + (i+4)*size, north, size);
                     break;
                  case 3:
                     memcpy(values + i*size, west, size);
                     memcpy(values + (i+1)*size, south, size);
                     memcpy(values + (i+2)*size, bottom, size);
                     memcpy(values + (i+3)*size, center, size);
                     memcpy(values + (i+4)*size, east, size);
                     memcpy(values + (i+5)*size, north, size);
                     memcpy(values + (i+6)*size, top, size);
                     break;
               }
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues_pre(precision_A, A, ilower, iupper, stencil_size,
                                               stencil_indices, values);

            hypre_Free(values, HYPRE_MEMORY_HOST);
         }
      }
      else if ( constant_coefficient == 1 )
      {
         values = hypre_CAlloc( (size_t)stencil_size, size, HYPRE_MEMORY_HOST);

         switch (dim)
         {
            case 1:
               memcpy((values),west,size);
               memcpy((values + size),center,size);
               memcpy((values + 2*size),east,size);
               break;
            case 2:
               memcpy((values),west,size);
               memcpy((values + size),south,size);
               memcpy((values + 2*size),center,size);            
               memcpy((values + 3*size),east,size);     
               memcpy((values + 4*size),north,size);   
               break;
            case 3:
               memcpy((values),west,size);
               memcpy((values + size),south,size);
               memcpy((values + 2*size),bottom,size);            
               memcpy((values + 3*size),center,size);     
               memcpy((values + 4*size),east,size);             
               memcpy((values + 5*size),north,size);             
               memcpy((values + 6*size),top,size);
               break;
         }

         if (hypre_BoxArraySize(gridboxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues_pre(precision_A, A, stencil_size,
                                                    stencil_indices, values);
         }
         hypre_Free(values, HYPRE_MEMORY_HOST);
      }
      else
      {
         values =  hypre_CAlloc( (size_t) (stencil_size - 1), size, HYPRE_MEMORY_HOST);
         switch (dim)
         {
            /* no center in stencil_indices and values */
            case 1:
               stencil_indices[0] = 0;
               stencil_indices[1] = 2;
               memcpy((values), west, size);
               memcpy((values + size), east, size);
               break;
            case 2:
               stencil_indices[0] = 0;
               stencil_indices[1] = 1;
               stencil_indices[2] = 3;
               stencil_indices[3] = 4;
               memcpy((values), west, size);
               memcpy((values + size), south, size);
               memcpy((values + 2*size), east, size);
               memcpy((values + 3*size), north, size);
               break;
            case 3:
               stencil_indices[0] = 0;
               stencil_indices[1] = 1;
               stencil_indices[2] = 2;
               stencil_indices[3] = 4;
               stencil_indices[4] = 5;
               stencil_indices[5] = 6;
               memcpy((values), west, size);
               memcpy((values + size), south, size);
               memcpy((values + 2*size), bottom, size);
               memcpy((values + 3*size), east, size);
               memcpy((values + 4*size), north, size);
               memcpy((values + 5*size), top, size);
               break;
         }
         if (hypre_BoxArraySize(gridboxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues_pre(precision_A, A, stencil_size,
                                                    stencil_indices, values);
         }
         hypre_Free(values, HYPRE_MEMORY_HOST);
         /* center is variable */
         stencil_indices[0] = dim; /* refers to center */
         hypre_ForBoxI(bi, gridboxes)
         {
            box    = hypre_BoxArrayBox(gridboxes, bi);
            /* hypre_BoxVolume is an integer option so we can use default function call */
            volume = hypre_BoxVolume(box);
            values = hypre_CAlloc((size_t) volume, size, HYPRE_MEMORY_HOST);
            HYPRE_Int i;

            for (i = 0; i < volume; i++)
            {
               memcpy(values + i*size, &center, size);
            }

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues_pre(precision_A, A, ilower, iupper, 1,
                                               stencil_indices, values);
            hypre_Free(values, HYPRE_MEMORY_HOST);
         }
      }
   }

   hypre_Free(stencil_indices, HYPRE_MEMORY_HOST);
   hypre_Free(stcoeffs, HYPRE_MEMORY_HOST);

   return ierr;
}

/*********************************************************************************
 * this function sets to zero the stencil entries that are on the boundary
 * Grid, matrix and the period are needed.
 *********************************************************************************/

HYPRE_Int
SetStencilBndry_mp(HYPRE_StructMatrix A,
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
   HYPRE_Int            volume, dim;
   HYPRE_Int           *stencil_indices;
   HYPRE_Int            constant_coefficient;

   gridboxes       = hypre_StructGridBoxes(gridmatrix);
   boundingbox     = hypre_StructGridBoundingBox(gridmatrix);
   istart          = hypre_BoxIMin(boundingbox);
   iend            = hypre_BoxIMax(boundingbox);
   size            = hypre_StructGridNumBoxes(gridmatrix);
   dim             = hypre_StructGridNDim(gridmatrix);

   /* get precision of struct matrix */
   HYPRE_Precision precision_A = hypre_StructMatrixPrecision(A);

   /* Declare values as a (void *) to allow for generic data types */
   void              *values;
   
   stencil_indices = (HYPRE_Int *)hypre_CAlloc(1, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if ( constant_coefficient > 0 ) { return 1; }
   /*...no space dependence if constant_coefficient==1,
     and space dependence only for diagonal if constant_coefficient==2 --
     and this function only touches off-diagonal entries */

   vol    = hypre_CAlloc((size_t) size, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
   ilower = hypre_CAlloc((size_t) size, (size_t)sizeof(HYPRE_Int*), HYPRE_MEMORY_HOST);
   iupper = hypre_CAlloc((size_t) size, (size_t)sizeof(HYPRE_Int*), HYPRE_MEMORY_HOST);
   for (i = 0; i < size; i++)
   {
      ilower[i] = hypre_CAlloc((size_t) dim, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
      iupper[i] = hypre_CAlloc((size_t) dim, (size_t)sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);
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
            /* Allocate memory for values data. 
             * Use long_double size to ensure enough memory is allocated for all supported types
             */
            values = hypre_CAlloc((size_t)(vol[ib]), (size_t)sizeof(hypre_long_double), HYPRE_MEMORY_HOST);

            if ( ilower[ib][d] == istart[d] && period[d] == 0 )
            {
               j = iupper[ib][d];
               iupper[ib][d] = istart[d];
               stencil_indices[0] = d;
               
               /* Precision should match values datatype */
               switch (precision_A)
               {
                  case HYPRE_REAL_SINGLE:
                     HYPRE_StructMatrixSetBoxValues_pre( HYPRE_REAL_SINGLE, A, ilower[ib], iupper[ib],
                                                  1, stencil_indices, (hypre_float *)values);
                     break;
                  case HYPRE_REAL_DOUBLE:
                     HYPRE_StructMatrixSetBoxValues_pre( HYPRE_REAL_DOUBLE, A, ilower[ib], iupper[ib],
                                                  1, stencil_indices, (hypre_double *)values);                  
                     break;
                  case HYPRE_REAL_LONGDOUBLE:
                     HYPRE_StructMatrixSetBoxValues_pre( HYPRE_REAL_LONGDOUBLE, A, ilower[ib], iupper[ib],
                                                  1, stencil_indices, (hypre_long_double *)values);                  
                     break;
               }
                                                
               iupper[ib][d] = j;
            }

            if ( iupper[ib][d] == iend[d] && period[d] == 0 )
            {
               j = ilower[ib][d];
               ilower[ib][d] = iend[d];
               stencil_indices[0] = dim + 1 + d;
               
               /* Precision should match values datatype */
               switch (precision_A)
               {
                  case HYPRE_REAL_SINGLE:
                     HYPRE_StructMatrixSetBoxValues_pre( HYPRE_REAL_SINGLE, A, ilower[ib], iupper[ib],
                                                  1, stencil_indices, (hypre_float *)values);
                     break;
                  case HYPRE_REAL_DOUBLE:
                     HYPRE_StructMatrixSetBoxValues_pre( HYPRE_REAL_DOUBLE, A, ilower[ib], iupper[ib],
                                                  1, stencil_indices, (hypre_double *)values);                  
                     break;
                  case HYPRE_REAL_LONGDOUBLE:
                     HYPRE_StructMatrixSetBoxValues_pre( HYPRE_REAL_LONGDOUBLE, A, ilower[ib], iupper[ib],
                                                  1, stencil_indices, (hypre_long_double *)values);                  
                     break;
               }
               ilower[ib][d] = j;
            }

            /* Free values pointer */
            hypre_Free(values, HYPRE_MEMORY_HOST);
         }
      }
   }

   hypre_Free(vol, HYPRE_MEMORY_HOST);
   hypre_Free(stencil_indices, HYPRE_MEMORY_HOST);
   for (ib = 0 ; ib < size ; ib++)
   {
      hypre_Free(ilower[ib], HYPRE_MEMORY_HOST);
      hypre_Free(iupper[ib], HYPRE_MEMORY_HOST);
   }
   hypre_Free(ilower, HYPRE_MEMORY_HOST);
   hypre_Free(iupper, HYPRE_MEMORY_HOST);

   return ierr;
}
