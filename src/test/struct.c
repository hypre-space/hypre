/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_krylov.h"

#define HYPRE_MFLOPS 0
#if HYPRE_MFLOPS
#include "_hypre_struct_mv.h"
#endif

/* RDF: Why is this include here? */
#include "_hypre_struct_mv.h"

#ifdef HYPRE_DEBUG
/*#include <cegdb.h>*/
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
   HYPRE_Int           solver_type;
   HYPRE_Int           recompute_res;

   /*HYPRE_Real          dxyz[3];*/

   HYPRE_Int           num_ghost[6]   = {0, 0, 0, 0, 0, 0};
   HYPRE_Int           A_num_ghost[6] = {0, 0, 0, 0, 0, 0};
   HYPRE_Int           v_num_ghost[6] = {0, 0, 0, 0, 0, 0};

   HYPRE_StructMatrix  A;
   HYPRE_StructVector  b;
   HYPRE_StructVector  x;

   HYPRE_StructSolver  solver;
   HYPRE_StructSolver  precond;
   HYPRE_Int           num_iterations;
   HYPRE_Int           time_index;
   HYPRE_Real          final_res_norm;
   HYPRE_Real          cf_tol;

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
   HYPRE_Real          jacobi_weight;
   HYPRE_Int           usr_jacobi_weight;
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

   /* begin lobpcg */

   HYPRE_Int lobpcgFlag = 0;
   HYPRE_Int lobpcgSeed = 0;
   HYPRE_Int blockSize = 1;
   HYPRE_Int verbosity = 1;
   HYPRE_Int iterations;
   HYPRE_Int maxIterations = 100;
   HYPRE_Int checkOrtho = 0;
   HYPRE_Int printLevel = 0;
   HYPRE_Int pcgIterations = 0;
   HYPRE_Int pcgMode = 0;
   HYPRE_Real tol = 1e-6;
   HYPRE_Real pcgTol = 1e-2;
   HYPRE_Real nonOrthF;

   FILE* filePtr;

   mv_MultiVectorPtr eigenvectors = NULL;
   mv_MultiVectorPtr constrains = NULL;
   HYPRE_Real* eigenvalues = NULL;

   HYPRE_Real* residuals;
   utilities_FortranMatrix* residualNorms;
   utilities_FortranMatrix* residualNormsHistory;
   utilities_FortranMatrix* eigenvaluesHistory;
   utilities_FortranMatrix* printBuffer;
   utilities_FortranMatrix* gramXX;
   utilities_FortranMatrix* identity;

   HYPRE_StructSolver        lobpcg_solver;

   mv_InterfaceInterpreter* interpreter;
   HYPRE_MatvecFunctions matvec_fn;
   /* end lobpcg */

   /* default execution policy and memory space */
#if defined(HYPRE_TEST_USING_HOST)
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_HOST;
#else
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_DEVICE;
#endif
   HYPRE_Int gpu_aware_mpi = 0;

   //HYPRE_Int device_level = -2;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before HYPRE_Initialize() and should not be changed after
    *-----------------------------------------------------------------*/
   for (arg_index = 1; arg_index < argc; arg_index ++)
   {
      if (strcmp(argv[arg_index], "-lazy_device_init") == 0)
      {
         lazy_device_init = atoi(argv[++arg_index]);
      }
      else if (strcmp(argv[arg_index], "-device_id") == 0)
      {
         device_id = atoi(argv[++arg_index]);
      }
   }

   hypre_bind_device_id(device_id, myid, num_procs, hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   HYPRE_Initialize();

   if (!lazy_device_init)
   {
      HYPRE_DeviceInitialize();
   }

#if defined(HYPRE_USING_KOKKOS)
   Kokkos::initialize (argc, argv);
#endif

#ifdef HYPRE_DEBUG
   /*cegdb(&argc, &argv, myid);*/
#endif

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
   solver_type = 1;
   recompute_res = 0;   /* What should be the default here? */

   istart[0] = -3;
   istart[1] = -3;
   istart[2] = -3;

   px = 0;
   py = 0;
   pz = 0;

   cf_tol = 0.90;

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
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         jacobi_weight = (HYPRE_Real)atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-jump") == 0 )
      {
         arg_index++;
         jump = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol = (HYPRE_Real)atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-orthchk") == 0 )
      {
         /* lobpcg: check orthonormality */
         arg_index++;
         checkOrtho = 1;
      }
      else if ( strcmp(argv[arg_index], "-verb") == 0 )
      {
         /* lobpcg: verbosity level */
         arg_index++;
         verbosity = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vrand") == 0 )
      {
         /* lobpcg: block size */
         arg_index++;
         blockSize = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seed") == 0 )
      {
         /* lobpcg: seed for srand */
         arg_index++;
         lobpcgSeed = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-itr") == 0 )
      {
         /* lobpcg: max # of iterations */
         arg_index++;
         maxIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         /* lobpcg: tolerance */
         arg_index++;
         tol = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgitr") == 0 )
      {
         /* lobpcg: max inner pcg iterations */
         arg_index++;
         pcgIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgtol") == 0 )
      {
         /* lobpcg: inner pcg iterations tolerance */
         arg_index++;
         pcgTol = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgmode") == 0 )
      {
         /* lobpcg: initial guess for inner pcg */
         arg_index++;      /* 0: zero, otherwise rhs */
         pcgMode = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vout") == 0 )
      {
         /* lobpcg: print level */
         arg_index++;
         printLevel = atoi(argv[arg_index++]);
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
#if defined(HYPRE_USING_MEMORY_TRACKER)
      else if ( strcmp(argv[arg_index], "-print_mem_tracker") == 0 )
      {
         arg_index++;
         print_mem_tracker = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mem_tracker_filename") == 0 )
      {
         arg_index++;
         snprintf(mem_tracker_name, HYPRE_MAX_FILE_NAME_LEN, "%s", argv[arg_index++]);
      }
#endif
      else if ( strcmp(argv[arg_index], "-gpu_mpi") == 0 )
      {
         arg_index++;
         gpu_aware_mpi = atoi(argv[arg_index++]);
      }
      /* end lobpcg */
      else
      {
         arg_index++;
      }
   }

#if defined(HYPRE_USING_MEMORY_TRACKER)
   hypre_MemoryTrackerSetPrint(print_mem_tracker);
   if (mem_tracker_name[0]) { hypre_MemoryTrackerSetFileName(mem_tracker_name); }
#endif

   /* default memory location */
   HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   HYPRE_SetExecutionPolicy(default_exec_policy);

   HYPRE_SetGpuAwareMPI(gpu_aware_mpi);

   /* begin lobpcg */

   if ( solver_id == 0 && lobpcgFlag )
   {
      solver_id = 10;
   }

   /*end lobpcg */

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
      hypre_printf("                        2  - SparseMSG\n");
      hypre_printf("                        3  - PFMG constant coeffs\n");
      hypre_printf("                        4  - PFMG constant coeffs var diag\n");
      hypre_printf("                        8  - Jacobi\n");
      hypre_printf("                        10 - CG with SMG precond\n");
      hypre_printf("                        11 - CG with PFMG precond\n");
      hypre_printf("                        12 - CG with SparseMSG precond\n");
      hypre_printf("                        13 - CG with PFMG-3 precond\n");
      hypre_printf("                        14 - CG with PFMG-4 precond\n");
      hypre_printf("                        17 - CG with 2-step Jacobi\n");
      hypre_printf("                        18 - CG with diagonal scaling\n");
      hypre_printf("                        19 - CG\n");
      hypre_printf("                        20 - Hybrid with SMG precond\n");
      hypre_printf("                        21 - Hybrid with PFMG precond\n");
      hypre_printf("                        22 - Hybrid with SparseMSG precond\n");
      hypre_printf("                        30 - GMRES with SMG precond\n");
      hypre_printf("                        31 - GMRES with PFMG precond\n");
      hypre_printf("                        32 - GMRES with SparseMSG precond\n");
      hypre_printf("                        37 - GMRES with 2-step Jacobi\n");
      hypre_printf("                        38 - GMRES with diagonal scaling\n");
      hypre_printf("                        39 - GMRES\n");
      hypre_printf("                        40 - BiCGSTAB with SMG precond\n");
      hypre_printf("                        41 - BiCGSTAB with PFMG precond\n");
      hypre_printf("                        42 - BiCGSTAB with SparseMSG precond\n");
      hypre_printf("                        47 - BiCGSTAB with 2-step Jacobi\n");
      hypre_printf("                        48 - BiCGSTAB with diagonal scaling\n");
      hypre_printf("                        49 - BiCGSTAB\n");
      hypre_printf("                        50 - LGMRES with SMG precond\n");
      hypre_printf("                        51 - LGMRES with PFMG precond\n");
      hypre_printf("                        59 - LGMRES\n");
      hypre_printf("                        60 - FlexGMRES with SMG precond\n");
      hypre_printf("                        61 - FlexGMRES with PFMG precond\n");
      hypre_printf("                        69 - FlexGMRES\n");
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
      hypre_printf("  -jump <num>         : num levels to jump in SparseMSG\n");
      hypre_printf("  -solver_type <ID>   : solver type for Hybrid\n");
      hypre_printf("                        1 - PCG (default)\n");
      hypre_printf("                        2 - GMRES\n");
      hypre_printf("  -recompute <bool>   : Recompute residual in PCG?\n");
      hypre_printf("  -cf <cf>            : convergence factor for Hybrid\n");
      hypre_printf("\n");

      /* begin lobpcg */

      hypre_printf("LOBPCG options:\n");
      hypre_printf("\n");
      hypre_printf("  -lobpcg             : run LOBPCG instead of PCG\n");
      hypre_printf("\n");
      hypre_printf("  -solver none        : no HYPRE preconditioner is used\n");
      hypre_printf("\n");
      hypre_printf("  -itr <val>          : maximal number of LOBPCG iterations (default 100);\n");
      hypre_printf("\n");
      hypre_printf("  -tol <val>          : residual tolerance (default 1e-6)\n");
      hypre_printf("\n");
      hypre_printf("  -vrand <val>        : compute <val> eigenpairs using random initial vectors (default 1)\n");
      hypre_printf("\n");
      hypre_printf("  -seed <val>         : use <val> as the seed for the pseudo-random number generator\n");
      hypre_printf("                        (default seed is based on the time of the run)\n");
      hypre_printf("\n");
      hypre_printf("  -orthchk            : check eigenvectors for orthonormality\n");
      hypre_printf("\n");
      hypre_printf("  -verb <val>         : verbosity level\n");
      hypre_printf("  -verb 0             : no print\n");
      hypre_printf("  -verb 1             : print initial eigenvalues and residuals, iteration number, number of\n");
      hypre_printf("                        non-convergent eigenpairs and final eigenvalues and residuals (default)\n");
      hypre_printf("  -verb 2             : print eigenvalues and residuals on each iteration\n");
      hypre_printf("\n");
      hypre_printf("  -pcgitr <val>       : maximal number of inner PCG iterations for preconditioning (default 1);\n");
      hypre_printf("                        if <val> = 0 then the preconditioner is applied directly\n");
      hypre_printf("\n");
      hypre_printf("  -pcgtol <val>       : residual tolerance for inner iterations (default 0.01)\n");
      hypre_printf("\n");
      hypre_printf("  -vout <val>         : file output level\n");
      hypre_printf("  -vout 0             : no files created (default)\n");
      hypre_printf("  -vout 1             : write eigenvalues to values.txt and residuals to residuals.txt\n");
      hypre_printf("  -vout 2             : in addition to the above, write the eigenvalues history (the matrix whose\n");
      hypre_printf("                        i-th column contains eigenvalues at (i+1)-th iteration) to val_hist.txt and\n");
      hypre_printf("                        residuals history to res_hist.txt\n");
      hypre_printf("\nNOTE: in this test driver LOBPCG only works with solvers 10, 11, 12, 17 and 18\n");
      hypre_printf("\ndefault solver is 10\n");
      hypre_printf("\n");

      /* end lobpcg */
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
      hypre_printf("  jump            = %d\n", jump);
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
      hypre_printf("  jump            = %d\n", jump);
      hypre_printf("  solver ID       = %d\n", solver_id);
      hypre_printf("  the grid is read from  file \n");

   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

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

      if ( (read_fromfile_param == 1) &&
           (read_x0fromfile_param == 1) &&
           (read_rhsfromfile_param == 1)
         )
      {
         /* ghost selection for reading the matrix and vectors */
         for (i = 0; i < dim; i++)
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
      }

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
         HYPRE_StructGridSetNumGhost(grid, num_ghost);
         HYPRE_StructGridAssemble(grid);

         /*-----------------------------------------------------------
          * Set up the matrix structure
          *-----------------------------------------------------------*/

         HYPRE_StructMatrixCreate(hypre_MPI_COMM_WORLD, grid, stencil, &A);

         if ( solver_id == 3 || solver_id == 4 ||
              solver_id == 13 || solver_id == 14 )
         {
            stencil_size  = hypre_StructStencilSize(stencil);
            stencil_entries = hypre_CTAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
            if ( solver_id == 3 || solver_id == 13)
            {
               for ( i = 0; i < stencil_size; ++i )
               {
                  stencil_entries[i] = i;
               }
               hypre_StructMatrixSetConstantEntries( A, stencil_size, stencil_entries );
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
               hypre_StructMatrixSetConstantEntries( A, stencil_size, stencil_entries );
               hypre_TFree( stencil_entries, HYPRE_MEMORY_HOST);
               constant_coefficient = 2;
            }
         }

         HYPRE_StructMatrixSetSymmetric(A, sym);
         HYPRE_StructMatrixInitialize(A);

         /*-----------------------------------------------------------
          * Fill in the matrix elements
          *-----------------------------------------------------------*/

         AddValuesMatrix(A, grid, cx, cy, cz, conx, cony, conz);

         /* Zero out stencils reaching to real boundary */
         /* But in constant coefficient case, no special stencils! */

         if ( constant_coefficient == 0 )
         {
            SetStencilBndry(A, grid, periodic);
         }
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

         AddValuesVector(grid, x, periodx0, 0.0);
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

      /* the grid will be read from file.  */
      if ( (sum > 0 ) && (sum < 3))
      {
         /* the grid will come from rhs or from x0 */
         if (read_fromfile_param == 0)
         {

            if ((read_rhsfromfile_param > 0) && (read_x0fromfile_param == 0))
            {
               /* read right hand side, extract grid, construct matrix,
                  construct x0 */

               hypre_printf("\ninitial rhs from file prefix :%s\n",
                            argv[read_rhsfromfile_index]);

               HYPRE_StructVectorRead(hypre_MPI_COMM_WORLD,
                                      argv[read_rhsfromfile_index],
                                      v_num_ghost, &b);

               readgrid = hypre_StructVectorGrid(b) ;
               readperiodic = hypre_StructGridPeriodic(readgrid);

               HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, readgrid, &x);
               HYPRE_StructVectorInitialize(x);

               AddValuesVector(readgrid, x, periodx0, 0.0);
               HYPRE_StructVectorAssemble(x);

               HYPRE_StructMatrixCreate(hypre_MPI_COMM_WORLD,
                                        readgrid, stencil, &A);
               HYPRE_StructMatrixSetSymmetric(A, 1);
               HYPRE_StructMatrixInitialize(A);

               /*-----------------------------------------------------------
                * Fill in the matrix elements
                *-----------------------------------------------------------*/

               AddValuesMatrix(A, readgrid, cx, cy, cz, conx, cony, conz);

               /* Zero out stencils reaching to real boundary */

               if ( constant_coefficient == 0 )
               {
                  SetStencilBndry(A, readgrid, readperiodic);
               }
               HYPRE_StructMatrixAssemble(A);
            }
            /* done with one case rhs=1 x0 = 0 */

            /* case when rhs=0 and read x0=1 */
            if ((read_rhsfromfile_param == 0) && (read_x0fromfile_param > 0))
            {
               /* read right hand side, extract grid, construct matrix,
                  construct x0 */

               hypre_printf("\ninitial x0 from file prefix :%s\n",
                            argv[read_x0fromfile_index]);

               HYPRE_StructVectorRead(hypre_MPI_COMM_WORLD,
                                      argv[read_x0fromfile_index],
                                      v_num_ghost, &x);

               readgrid = hypre_StructVectorGrid(x) ;
               readperiodic = hypre_StructGridPeriodic(readgrid);

               HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, readgrid, &b);
               HYPRE_StructVectorInitialize(b);
               AddValuesVector(readgrid, b, readperiodic, 1.0);

               HYPRE_StructVectorAssemble(b);

               HYPRE_StructMatrixCreate(hypre_MPI_COMM_WORLD,
                                        readgrid, stencil, &A);
               HYPRE_StructMatrixSetSymmetric(A, 1);
               HYPRE_StructMatrixInitialize(A);

               /*-----------------------------------------------------------
                * Fill in the matrix elements
                *-----------------------------------------------------------*/

               AddValuesMatrix(A, readgrid, cx, cy, cz, conx, cony, conz);

               /* Zero out stencils reaching to real boundary */

               if ( constant_coefficient == 0 )
               {
                  SetStencilBndry(A, readgrid, readperiodic);
               }
               HYPRE_StructMatrixAssemble(A);
            }
            /* done with one case rhs=0 x0 = 1  */

            /* the other case when read rhs > 0 and read x0 > 0  */
            if ((read_rhsfromfile_param > 0) && (read_x0fromfile_param > 0))
            {
               /* read right hand side, extract grid, construct matrix,
                  construct x0 */

               hypre_printf("\ninitial rhs  from file prefix :%s\n",
                            argv[read_rhsfromfile_index]);
               hypre_printf("\ninitial x0  from file prefix :%s\n",
                            argv[read_x0fromfile_index]);

               HYPRE_StructVectorRead(hypre_MPI_COMM_WORLD,
                                      argv[read_rhsfromfile_index],
                                      v_num_ghost, &b);

               HYPRE_StructVectorRead(hypre_MPI_COMM_WORLD,
                                      argv[read_x0fromfile_index],
                                      v_num_ghost, &x);

               readgrid = hypre_StructVectorGrid(b) ;
               readperiodic = hypre_StructGridPeriodic(readgrid);

               HYPRE_StructMatrixCreate(hypre_MPI_COMM_WORLD,
                                        readgrid, stencil, &A);
               HYPRE_StructMatrixSetSymmetric(A, 1);
               HYPRE_StructMatrixInitialize(A);

               /*-----------------------------------------------------------
                * Fill in the matrix elements
                *-----------------------------------------------------------*/

               AddValuesMatrix(A, readgrid, cx, cy, cz, conx, cony, conz);

               /* Zero out stencils reaching to real boundary */

               if ( constant_coefficient == 0 )
               {
                  SetStencilBndry(A, readgrid, readperiodic);
               }
               HYPRE_StructMatrixAssemble(A);
            }
            /* done with one case rhs=1 x0 = 1  */
         }
         /* done with the case where you no read matrix  */

         if (read_fromfile_param == 1)  /* still sum > 0  */
         {
            hypre_printf("\nreading matrix from file:%s\n",
                         argv[read_fromfile_index]);

            HYPRE_StructMatrixRead(hypre_MPI_COMM_WORLD,
                                   argv[read_fromfile_index],
                                   A_num_ghost, &A);

            readgrid = hypre_StructMatrixGrid(A);
            readperiodic  =  hypre_StructGridPeriodic(readgrid);

            if ((read_rhsfromfile_param > 0) && (read_x0fromfile_param == 0))
            {
               /* read right hand side ,construct x0 */
               hypre_printf("\ninitial rhs from file prefix :%s\n",
                            argv[read_rhsfromfile_index]);

               HYPRE_StructVectorRead(hypre_MPI_COMM_WORLD,
                                      argv[read_rhsfromfile_index],
                                      v_num_ghost, &b);

               HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, readgrid, &x);
               HYPRE_StructVectorInitialize(x);
               AddValuesVector(readgrid, x, periodx0, 0.0);
               HYPRE_StructVectorAssemble(x);
            }

            if ((read_rhsfromfile_param == 0) && (read_x0fromfile_param > 0))
            {
               /* read x0, construct rhs*/
               hypre_printf("\ninitial x0 from file prefix :%s\n",
                            argv[read_x0fromfile_index]);

               HYPRE_StructVectorRead(hypre_MPI_COMM_WORLD,
                                      argv[read_x0fromfile_index],
                                      v_num_ghost, &x);

               HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, readgrid, &b);
               HYPRE_StructVectorInitialize(b);
               AddValuesVector(readgrid, b, readperiodic, 1.0);
               HYPRE_StructVectorAssemble(b);
            }

            if ((read_rhsfromfile_param == 0) && (read_x0fromfile_param == 0))
            {
               /* construct x0 , construct b*/
               HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, readgrid, &b);
               HYPRE_StructVectorInitialize(b);
               AddValuesVector(readgrid, b, readperiodic, 1.0);
               HYPRE_StructVectorAssemble(b);


               HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, readgrid, &x);
               HYPRE_StructVectorInitialize(x);
               AddValuesVector(readgrid, x, periodx0, 0.0);
               HYPRE_StructVectorAssemble(x);
            }
         }
         /* finish the read of matrix  */
      }
      /* finish the sum > 0 case   */

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

      if (print_system)
      {
         HYPRE_StructMatrixPrint("struct.out.A", A, 0);
         HYPRE_StructVectorPrint("struct.out.b", b, 0);
         HYPRE_StructVectorPrint("struct.out.x0", x, 0);
      }

      /*-----------------------------------------------------------
       * Solve the system using SMG
       *-----------------------------------------------------------*/

#if !HYPRE_MFLOPS

      if (solver_id == 0)
      {
         time_index = hypre_InitializeTiming("SMG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_StructSMGSetMemoryUse(solver, 0);
         HYPRE_StructSMGSetMaxIter(solver, 50);
         HYPRE_StructSMGSetTol(solver, tol);
         HYPRE_StructSMGSetRelChange(solver, 0);
         HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
         HYPRE_StructSMGSetNumPostRelax(solver, n_post);
         HYPRE_StructSMGSetPrintLevel(solver, 1);
         HYPRE_StructSMGSetLogging(solver, 1);
#if 0//defined(HYPRE_USING_CUDA)
         HYPRE_StructSMGSetDeviceLevel(solver, device_level);
#endif

#if 0//defined(HYPRE_USING_CUDA)
         hypre_box_print = 0;
#endif
         HYPRE_StructSMGSetup(solver, A, b, x);

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

         HYPRE_StructSMGSolve(solver, A, b, x);

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

         HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
         HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_StructSMGDestroy(solver);
      }

      /*-----------------------------------------------------------
       * Solve the system using PFMG
       *-----------------------------------------------------------*/

      else if ( solver_id == 1 || solver_id == 3 || solver_id == 4 )
      {
         time_index = hypre_InitializeTiming("PFMG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &solver);
         /*HYPRE_StructPFMGSetMaxLevels( solver, 9 );*/
         HYPRE_StructPFMGSetMaxIter(solver, 200);
         HYPRE_StructPFMGSetTol(solver, tol);
         HYPRE_StructPFMGSetRelChange(solver, 0);
         HYPRE_StructPFMGSetRAPType(solver, rap);
         HYPRE_StructPFMGSetRelaxType(solver, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight(solver, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax(solver, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(solver, n_post);
         HYPRE_StructPFMGSetSkipRelax(solver, skip);
         /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel(solver, 1);
         HYPRE_StructPFMGSetLogging(solver, 1);

#if 0//defined(HYPRE_USING_CUDA)
         HYPRE_StructPFMGSetDeviceLevel(solver, device_level);
#endif

         HYPRE_StructPFMGSetup(solver, A, b, x);

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


         HYPRE_StructPFMGSolve(solver, A, b, x);

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

         HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_StructSparseMSGSetMaxIter(solver, 50);
         HYPRE_StructSparseMSGSetJump(solver, jump);
         HYPRE_StructSparseMSGSetTol(solver, tol);
         HYPRE_StructSparseMSGSetRelChange(solver, 0);
         HYPRE_StructSparseMSGSetRelaxType(solver, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructSparseMSGSetJacobiWeight(solver, jacobi_weight);
         }
         HYPRE_StructSparseMSGSetNumPreRelax(solver, n_pre);
         HYPRE_StructSparseMSGSetNumPostRelax(solver, n_post);
         HYPRE_StructSparseMSGSetPrintLevel(solver, 1);
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
         HYPRE_StructJacobiSetTol(solver, tol);
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
         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructPCGCreate(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_PCGSetMaxIter( (HYPRE_Solver)solver, 100 );
         HYPRE_PCGSetTol( (HYPRE_Solver)solver, tol );
         HYPRE_PCGSetTwoNorm( (HYPRE_Solver)solver, 1 );
         HYPRE_PCGSetRelChange( (HYPRE_Solver)solver, 0 );
         HYPRE_PCGSetPrintLevel( (HYPRE_Solver)solver, 1 );

         if (solver_id == 10)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, 0);
            HYPRE_StructSMGSetLogging(precond, 0);

#if 0//defined(HYPRE_USING_CUDA)
            HYPRE_StructSMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                 (HYPRE_Solver) precond);
         }

         else if (solver_id == 11 || solver_id == 13 || solver_id == 14)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, 0);
            HYPRE_StructPFMGSetLogging(precond, 0);
#if 0//defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                 (HYPRE_Solver) precond);
         }

         else if (solver_id == 12)
         {
            /* use symmetric SparseMSG as preconditioner */
            HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSparseMSGSetMaxIter(precond, 1);
            HYPRE_StructSparseMSGSetJump(precond, jump);
            HYPRE_StructSparseMSGSetTol(precond, 0.0);
            HYPRE_StructSparseMSGSetZeroGuess(precond);
            HYPRE_StructSparseMSGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructSparseMSGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
            HYPRE_StructSparseMSGSetPrintLevel(precond, 0);
            HYPRE_StructSparseMSGSetLogging(precond, 0);
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                 (HYPRE_Solver) precond);
         }

         else if (solver_id == 17)
         {
            /* use two-step Jacobi as preconditioner */
            HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructJacobiSetMaxIter(precond, 2);
            HYPRE_StructJacobiSetTol(precond, 0.0);
            HYPRE_StructJacobiSetZeroGuess(precond);
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                 (HYPRE_Solver) precond);
         }

         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                 (HYPRE_Solver) precond);
         }

         HYPRE_PCGSetup( (HYPRE_Solver)solver,
                         (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("PCG Solve");
         hypre_BeginTiming(time_index);

         HYPRE_PCGSolve( (HYPRE_Solver) solver,
                         (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_PCGGetNumIterations( (HYPRE_Solver)solver, &num_iterations );
         HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver)solver,
                                                &final_res_norm );
         HYPRE_StructPCGDestroy(solver);

         if (solver_id == 10)
         {
            HYPRE_StructSMGDestroy(precond);
         }
         else if (solver_id == 11 || solver_id == 13 || solver_id == 14)
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

      /* begin lobpcg */

      /*-----------------------------------------------------------
       * Solve the system using LOBPCG
       *-----------------------------------------------------------*/

      if ( lobpcgFlag )
      {

         interpreter = hypre_CTAlloc(mv_InterfaceInterpreter, 1, HYPRE_MEMORY_HOST);

         HYPRE_StructSetupInterpreter( interpreter );
         HYPRE_StructSetupMatvec(&matvec_fn);

         if (myid != 0)
         {
            verbosity = 0;
         }

         if ( pcgIterations > 0 )
         {

            time_index = hypre_InitializeTiming("PCG Setup");
            hypre_BeginTiming(time_index);

            HYPRE_StructPCGCreate(hypre_MPI_COMM_WORLD, &solver);
            HYPRE_PCGSetMaxIter( (HYPRE_Solver)solver, pcgIterations );
            HYPRE_PCGSetTol( (HYPRE_Solver)solver, pcgTol );
            HYPRE_PCGSetTwoNorm( (HYPRE_Solver)solver, 1 );
            HYPRE_PCGSetRelChange( (HYPRE_Solver)solver, 0 );
            HYPRE_PCGSetPrintLevel( (HYPRE_Solver)solver, 0 );

            if (solver_id == 10)
            {
               /* use symmetric SMG as preconditioner */
               HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
               HYPRE_StructSMGSetMemoryUse(precond, 0);
               HYPRE_StructSMGSetMaxIter(precond, 1);
               HYPRE_StructSMGSetTol(precond, 0.0);
               HYPRE_StructSMGSetZeroGuess(precond);
               HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
               HYPRE_StructSMGSetNumPostRelax(precond, n_post);
               HYPRE_StructSMGSetPrintLevel(precond, 0);
               HYPRE_StructSMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
               HYPRE_StructSMGSetDeviceLevel(precond, device_level);
#endif
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                    (HYPRE_Solver) precond);
            }

            else if (solver_id == 11)
            {
               /* use symmetric PFMG as preconditioner */
               HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
               HYPRE_StructPFMGSetMaxIter(precond, 1);
               HYPRE_StructPFMGSetTol(precond, 0.0);
               HYPRE_StructPFMGSetZeroGuess(precond);
               HYPRE_StructPFMGSetRAPType(precond, rap);
               HYPRE_StructPFMGSetRelaxType(precond, relax);
               if (usr_jacobi_weight)
               {
                  HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
               }
               HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
               HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
               HYPRE_StructPFMGSetSkipRelax(precond, skip);
               /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
               HYPRE_StructPFMGSetPrintLevel(precond, 0);
               HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
               HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                    (HYPRE_Solver) precond);
            }

            else if (solver_id == 12)
            {
               /* use symmetric SparseMSG as preconditioner */
               HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &precond);
               HYPRE_StructSparseMSGSetMaxIter(precond, 1);
               HYPRE_StructSparseMSGSetJump(precond, jump);
               HYPRE_StructSparseMSGSetTol(precond, 0.0);
               HYPRE_StructSparseMSGSetZeroGuess(precond);
               HYPRE_StructSparseMSGSetRelaxType(precond, relax);
               if (usr_jacobi_weight)
               {
                  HYPRE_StructSparseMSGSetJacobiWeight(precond, jacobi_weight);
               }
               HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
               HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
               HYPRE_StructSparseMSGSetPrintLevel(precond, 0);
               HYPRE_StructSparseMSGSetLogging(precond, 0);
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                    (HYPRE_Solver) precond);
            }

            else if (solver_id == 17)
            {
               /* use two-step Jacobi as preconditioner */
               HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &precond);
               HYPRE_StructJacobiSetMaxIter(precond, 2);
               HYPRE_StructJacobiSetTol(precond, 0.0);
               HYPRE_StructJacobiSetZeroGuess(precond);
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                    (HYPRE_Solver) precond);
            }

            else if (solver_id == 18)
            {
               /* use diagonal scaling as preconditioner */
               precond = NULL;
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                    (HYPRE_Solver) precond);
            }
            else if (solver_id != NO_SOLVER )
            {
               if ( verbosity )
               {
                  hypre_printf("Solver ID not recognized - running inner PCG iterations without preconditioner\n\n");
               }
            }

            hypre_EndTiming(time_index);
            hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();

            HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&lobpcg_solver);
            HYPRE_LOBPCGSetMaxIter((HYPRE_Solver)lobpcg_solver, maxIterations);
            HYPRE_LOBPCGSetPrecondUsageMode((HYPRE_Solver)lobpcg_solver, pcgMode);
            HYPRE_LOBPCGSetTol((HYPRE_Solver)lobpcg_solver, tol);
            HYPRE_LOBPCGSetPrintLevel((HYPRE_Solver)lobpcg_solver, verbosity);

            HYPRE_LOBPCGSetPrecond((HYPRE_Solver)lobpcg_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_PCGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_PCGSetup,
                                   (HYPRE_Solver)solver);

            HYPRE_LOBPCGSetup((HYPRE_Solver)lobpcg_solver, (HYPRE_Matrix)A,
                              (HYPRE_Vector)b, (HYPRE_Vector)x);

            eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                                 blockSize,
                                                                 x );
            eigenvalues = hypre_CTAlloc(HYPRE_Real,  blockSize, HYPRE_MEMORY_HOST);

            if ( lobpcgSeed )
            {
               mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
            }
            else
            {
               mv_MultiVectorSetRandom( eigenvectors, (HYPRE_Int)time(0) );
            }

            time_index = hypre_InitializeTiming("LOBPCG Solve");
            hypre_BeginTiming(time_index);

            HYPRE_LOBPCGSolve((HYPRE_Solver)lobpcg_solver, constrains,
                              eigenvectors, eigenvalues );

            hypre_EndTiming(time_index);
            hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();

            if ( checkOrtho )
            {

               gramXX = utilities_FortranMatrixCreate();
               identity = utilities_FortranMatrixCreate();

               utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
               utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

               lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
               utilities_FortranMatrixSetToIdentity( identity );
               utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
               nonOrthF = utilities_FortranMatrixFNorm( gramXX );
               if ( myid == 0 )
               {
                  hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
               }

               utilities_FortranMatrixDestroy( gramXX );
               utilities_FortranMatrixDestroy( identity );

            }

            if ( printLevel )
            {

               if ( myid == 0 )
               {
                  if ( (filePtr = fopen("values.txt", "w")) )
                  {
                     hypre_fprintf(filePtr, "%d\n", blockSize);
                     for ( i = 0; i < blockSize; i++ )
                     {
                        hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                     }
                     fclose(filePtr);
                  }

                  if ( (filePtr = fopen("residuals.txt", "w")) )
                  {
                     residualNorms = HYPRE_LOBPCGResidualNorms( (HYPRE_Solver)lobpcg_solver );
                     residuals = utilities_FortranMatrixValues( residualNorms );
                     hypre_fprintf(filePtr, "%d\n", blockSize);
                     for ( i = 0; i < blockSize; i++ )
                     {
                        hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                     }
                     fclose(filePtr);
                  }

                  if ( printLevel > 1 )
                  {

                     printBuffer = utilities_FortranMatrixCreate();

                     iterations = HYPRE_LOBPCGIterations( (HYPRE_Solver)lobpcg_solver );

                     eigenvaluesHistory = HYPRE_LOBPCGEigenvaluesHistory( (HYPRE_Solver)lobpcg_solver );
                     utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                         1, blockSize, 1, iterations + 1, printBuffer );
                     utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

                     residualNormsHistory = HYPRE_LOBPCGResidualNormsHistory( (HYPRE_Solver)lobpcg_solver );
                     utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                        1, blockSize, 1, iterations + 1, printBuffer );
                     utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );

                     utilities_FortranMatrixDestroy( printBuffer );
                  }
               }
            }

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

            HYPRE_LOBPCGDestroy((HYPRE_Solver)lobpcg_solver);
            mv_MultiVectorDestroy( eigenvectors );
            hypre_TFree(eigenvalues, HYPRE_MEMORY_HOST);
         }
         else
         {
            time_index = hypre_InitializeTiming("LOBPCG Setup");
            hypre_BeginTiming(time_index);

            HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&solver);
            HYPRE_LOBPCGSetMaxIter( (HYPRE_Solver)solver, maxIterations );
            HYPRE_LOBPCGSetTol( (HYPRE_Solver)solver, tol );
            HYPRE_LOBPCGSetPrintLevel( (HYPRE_Solver)solver, verbosity );

            if (solver_id == 10)
            {
               /* use symmetric SMG as preconditioner */
               HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
               HYPRE_StructSMGSetMemoryUse(precond, 0);
               HYPRE_StructSMGSetMaxIter(precond, 1);
               HYPRE_StructSMGSetTol(precond, 0.0);
               HYPRE_StructSMGSetZeroGuess(precond);
               HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
               HYPRE_StructSMGSetNumPostRelax(precond, n_post);
               HYPRE_StructSMGSetPrintLevel(precond, 0);
               HYPRE_StructSMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
               HYPRE_StructSMGSetDeviceLevel(precond, device_level);
#endif
               HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                       (HYPRE_Solver) precond);
            }

            else if (solver_id == 11)
            {
               /* use symmetric PFMG as preconditioner */
               HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
               HYPRE_StructPFMGSetMaxIter(precond, 1);
               HYPRE_StructPFMGSetTol(precond, 0.0);
               HYPRE_StructPFMGSetZeroGuess(precond);
               HYPRE_StructPFMGSetRAPType(precond, rap);
               HYPRE_StructPFMGSetRelaxType(precond, relax);
               if (usr_jacobi_weight)
               {
                  HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
               }
               HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
               HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
               HYPRE_StructPFMGSetSkipRelax(precond, skip);
               /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
               HYPRE_StructPFMGSetPrintLevel(precond, 0);
               HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
               HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
               HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                       (HYPRE_Solver) precond);
            }

            else if (solver_id == 12)
            {
               /* use symmetric SparseMSG as preconditioner */
               HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &precond);
               HYPRE_StructSparseMSGSetMaxIter(precond, 1);
               HYPRE_StructSparseMSGSetJump(precond, jump);
               HYPRE_StructSparseMSGSetTol(precond, 0.0);
               HYPRE_StructSparseMSGSetZeroGuess(precond);
               HYPRE_StructSparseMSGSetRelaxType(precond, relax);
               if (usr_jacobi_weight)
               {
                  HYPRE_StructSparseMSGSetJacobiWeight(precond, jacobi_weight);
               }
               HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
               HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
               HYPRE_StructSparseMSGSetPrintLevel(precond, 0);
               HYPRE_StructSparseMSGSetLogging(precond, 0);
               HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                       (HYPRE_Solver) precond);
            }

            else if (solver_id == 17)
            {
               /* use two-step Jacobi as preconditioner */
               HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &precond);
               HYPRE_StructJacobiSetMaxIter(precond, 2);
               HYPRE_StructJacobiSetTol(precond, 0.0);
               HYPRE_StructJacobiSetZeroGuess(precond);
               HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                       (HYPRE_Solver) precond);
            }

            else if (solver_id == 18)
            {
               /* use diagonal scaling as preconditioner */
               precond = NULL;
               HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                       (HYPRE_Solver) precond);
            }
            else if (solver_id != NO_SOLVER )
            {
               if ( verbosity )
               {
                  hypre_printf("Solver ID not recognized - running LOBPCG without preconditioner\n\n");
               }
            }

            HYPRE_LOBPCGSetup
            ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x );

            hypre_EndTiming(time_index);
            hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();

            eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                                 blockSize,
                                                                 x );
            eigenvalues = hypre_CTAlloc(HYPRE_Real,  blockSize, HYPRE_MEMORY_HOST);

            if ( lobpcgSeed )
            {
               mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
            }
            else
            {
               mv_MultiVectorSetRandom( eigenvectors, (HYPRE_Int)time(0) );
            }

            time_index = hypre_InitializeTiming("PCG Solve");
            hypre_BeginTiming(time_index);

            HYPRE_LOBPCGSolve
            ( (HYPRE_Solver)solver, constrains, eigenvectors, eigenvalues );

            hypre_EndTiming(time_index);
            hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();

            if ( checkOrtho )
            {

               gramXX = utilities_FortranMatrixCreate();
               identity = utilities_FortranMatrixCreate();

               utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
               utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

               lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
               utilities_FortranMatrixSetToIdentity( identity );
               utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
               nonOrthF = utilities_FortranMatrixFNorm( gramXX );
               if ( myid == 0 )
               {
                  hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
               }

               utilities_FortranMatrixDestroy( gramXX );
               utilities_FortranMatrixDestroy( identity );

            }

            if ( printLevel )
            {

               if ( myid == 0 )
               {
                  if ( (filePtr = fopen("values.txt", "w")) )
                  {
                     hypre_fprintf(filePtr, "%d\n", blockSize);
                     for ( i = 0; i < blockSize; i++ )
                     {
                        hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                     }
                     fclose(filePtr);
                  }

                  if ( (filePtr = fopen("residuals.txt", "w")) )
                  {
                     residualNorms = HYPRE_LOBPCGResidualNorms( (HYPRE_Solver)solver );
                     residuals = utilities_FortranMatrixValues( residualNorms );
                     hypre_fprintf(filePtr, "%d\n", blockSize);
                     for ( i = 0; i < blockSize; i++ )
                     {
                        hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                     }
                     fclose(filePtr);
                  }

                  if ( printLevel > 1 )
                  {

                     printBuffer = utilities_FortranMatrixCreate();

                     iterations = HYPRE_LOBPCGIterations( (HYPRE_Solver)solver );

                     eigenvaluesHistory = HYPRE_LOBPCGEigenvaluesHistory( (HYPRE_Solver)solver );
                     utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                         1, blockSize, 1, iterations + 1, printBuffer );
                     utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

                     residualNormsHistory = HYPRE_LOBPCGResidualNormsHistory( (HYPRE_Solver)solver );
                     utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                        1, blockSize, 1, iterations + 1, printBuffer );
                     utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );

                     utilities_FortranMatrixDestroy( printBuffer );
                  }
               }
            }

            HYPRE_LOBPCGDestroy((HYPRE_Solver)solver);

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

            mv_MultiVectorDestroy( eigenvectors );
            hypre_TFree(eigenvalues, HYPRE_MEMORY_HOST);
         }

         hypre_TFree( interpreter, HYPRE_MEMORY_HOST);

      }

      /* end lobpcg */

      /*-----------------------------------------------------------
       * Solve the system using Hybrid
       *-----------------------------------------------------------*/

      if ((solver_id > 19) && (solver_id < 30))
      {
         time_index = hypre_InitializeTiming("Hybrid Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructHybridCreate(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_StructHybridSetDSCGMaxIter(solver, 100);
         HYPRE_StructHybridSetPCGMaxIter(solver, 100);
         HYPRE_StructHybridSetTol(solver, tol);
         /*HYPRE_StructHybridSetPCGAbsoluteTolFactor(solver, 1.0e-200);*/
         HYPRE_StructHybridSetConvergenceTol(solver, cf_tol);
         HYPRE_StructHybridSetTwoNorm(solver, 1);
         HYPRE_StructHybridSetRelChange(solver, 0);
         if (solver_type == 2) /* for use with GMRES */
         {
            HYPRE_StructHybridSetStopCrit(solver, 0);
            HYPRE_StructHybridSetKDim(solver, 10);
         }
         HYPRE_StructHybridSetPrintLevel(solver, 1);
         HYPRE_StructHybridSetLogging(solver, 1);
         HYPRE_StructHybridSetSolverType(solver, solver_type);
         HYPRE_StructHybridSetRecomputeResidual(solver, recompute_res);

         if (solver_id == 20)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, 0);
            HYPRE_StructSMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructSMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_StructHybridSetPrecond(solver,
                                         HYPRE_StructSMGSolve,
                                         HYPRE_StructSMGSetup,
                                         precond);
         }

         else if (solver_id == 21)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, 0);
            HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_StructHybridSetPrecond(solver,
                                         HYPRE_StructPFMGSolve,
                                         HYPRE_StructPFMGSetup,
                                         precond);
         }

         else if (solver_id == 22)
         {
            /* use symmetric SparseMSG as preconditioner */
            HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSparseMSGSetJump(precond, jump);
            HYPRE_StructSparseMSGSetMaxIter(precond, 1);
            HYPRE_StructSparseMSGSetTol(precond, 0.0);
            HYPRE_StructSparseMSGSetZeroGuess(precond);
            HYPRE_StructSparseMSGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructSparseMSGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
            HYPRE_StructSparseMSGSetPrintLevel(precond, 0);
            HYPRE_StructSparseMSGSetLogging(precond, 0);
            HYPRE_StructHybridSetPrecond(solver,
                                         HYPRE_StructSparseMSGSolve,
                                         HYPRE_StructSparseMSGSetup,
                                         precond);
         }

         HYPRE_StructHybridSetup(solver, A, b, x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("Hybrid Solve");
         hypre_BeginTiming(time_index);

         HYPRE_StructHybridSolve(solver, A, b, x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
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

      /*-----------------------------------------------------------
       * Solve the system using GMRES
       *-----------------------------------------------------------*/

      if ((solver_id > 29) && (solver_id < 40))
      {
         time_index = hypre_InitializeTiming("GMRES Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructGMRESCreate(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_GMRESSetKDim( (HYPRE_Solver) solver, 5 );
         HYPRE_GMRESSetMaxIter( (HYPRE_Solver)solver, 100 );
         HYPRE_GMRESSetTol( (HYPRE_Solver)solver, tol );
         HYPRE_GMRESSetRelChange( (HYPRE_Solver)solver, 0 );
         HYPRE_GMRESSetPrintLevel( (HYPRE_Solver)solver, 1 );
         HYPRE_GMRESSetLogging( (HYPRE_Solver)solver, 1 );

         if (solver_id == 30)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, 0);
            HYPRE_StructSMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructSMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                   (HYPRE_Solver)precond);
         }

         else if (solver_id == 31)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, 0);
            HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                   (HYPRE_Solver)precond);
         }

         else if (solver_id == 32)
         {
            /* use symmetric SparseMSG as preconditioner */
            HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSparseMSGSetMaxIter(precond, 1);
            HYPRE_StructSparseMSGSetJump(precond, jump);
            HYPRE_StructSparseMSGSetTol(precond, 0.0);
            HYPRE_StructSparseMSGSetZeroGuess(precond);
            HYPRE_StructSparseMSGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructSparseMSGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
            HYPRE_StructSparseMSGSetPrintLevel(precond, 0);
            HYPRE_StructSparseMSGSetLogging(precond, 0);
            HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                   (HYPRE_Solver)precond);
         }

         else if (solver_id == 37)
         {
            /* use two-step Jacobi as preconditioner */
            HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructJacobiSetMaxIter(precond, 2);
            HYPRE_StructJacobiSetTol(precond, 0.0);
            HYPRE_StructJacobiSetZeroGuess(precond);
            HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                   (HYPRE_Solver)precond);
         }

         else if (solver_id == 38)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                   (HYPRE_Solver)precond);
         }

         HYPRE_GMRESSetup
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("GMRES Solve");
         hypre_BeginTiming(time_index);

         HYPRE_GMRESSolve
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_GMRESGetNumIterations( (HYPRE_Solver)solver, &num_iterations);
         HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm);
         HYPRE_StructGMRESDestroy(solver);

         if (solver_id == 30)
         {
            HYPRE_StructSMGDestroy(precond);
         }
         else if (solver_id == 31)
         {
            HYPRE_StructPFMGDestroy(precond);
         }
         else if (solver_id == 32)
         {
            HYPRE_StructSparseMSGDestroy(precond);
         }
         else if (solver_id == 37)
         {
            HYPRE_StructJacobiDestroy(precond);
         }
      }

      /*-----------------------------------------------------------
       * Solve the system using BiCGTAB
       *-----------------------------------------------------------*/

      if ((solver_id > 39) && (solver_id < 50))
      {
         time_index = hypre_InitializeTiming("BiCGSTAB Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructBiCGSTABCreate(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver)solver, 100 );
         HYPRE_BiCGSTABSetTol( (HYPRE_Solver)solver, tol );
         HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver)solver, 1 );
         HYPRE_BiCGSTABSetLogging( (HYPRE_Solver)solver, 1 );

         if (solver_id == 40)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, 0);
            HYPRE_StructSMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructSMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                      (HYPRE_Solver)precond);
         }

         else if (solver_id == 41)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, 0);
            HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                      (HYPRE_Solver)precond);
         }

         else if (solver_id == 42)
         {
            /* use symmetric SparseMSG as preconditioner */
            HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSparseMSGSetMaxIter(precond, 1);
            HYPRE_StructSparseMSGSetJump(precond, jump);
            HYPRE_StructSparseMSGSetTol(precond, 0.0);
            HYPRE_StructSparseMSGSetZeroGuess(precond);
            HYPRE_StructSparseMSGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructSparseMSGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
            HYPRE_StructSparseMSGSetPrintLevel(precond, 0);
            HYPRE_StructSparseMSGSetLogging(precond, 0);
            HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                      (HYPRE_Solver)precond);
         }

         else if (solver_id == 47)
         {
            /* use two-step Jacobi as preconditioner */
            HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructJacobiSetMaxIter(precond, 2);
            HYPRE_StructJacobiSetTol(precond, 0.0);
            HYPRE_StructJacobiSetZeroGuess(precond);
            HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                      (HYPRE_Solver)precond);
         }

         else if (solver_id == 48)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                      (HYPRE_Solver)precond);
         }

         HYPRE_BiCGSTABSetup
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("BiCGSTAB Solve");
         hypre_BeginTiming(time_index);

         HYPRE_BiCGSTABSolve
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_BiCGSTABGetNumIterations( (HYPRE_Solver)solver, &num_iterations);
         HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm);
         HYPRE_StructBiCGSTABDestroy(solver);

         if (solver_id == 40)
         {
            HYPRE_StructSMGDestroy(precond);
         }
         else if (solver_id == 41)
         {
            HYPRE_StructPFMGDestroy(precond);
         }
         else if (solver_id == 42)
         {
            HYPRE_StructSparseMSGDestroy(precond);
         }
         else if (solver_id == 47)
         {
            HYPRE_StructJacobiDestroy(precond);
         }
      }
      /*-----------------------------------------------------------
       * Solve the system using LGMRES
       *-----------------------------------------------------------*/

      if ((solver_id > 49) && (solver_id < 60))
      {
         time_index = hypre_InitializeTiming("LGMRES Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructLGMRESCreate(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_LGMRESSetKDim( (HYPRE_Solver) solver, 5 );
         HYPRE_LGMRESSetMaxIter( (HYPRE_Solver)solver, 100 );
         HYPRE_LGMRESSetTol( (HYPRE_Solver)solver, tol );
         HYPRE_LGMRESSetPrintLevel( (HYPRE_Solver)solver, 1 );
         HYPRE_LGMRESSetLogging( (HYPRE_Solver)solver, 1 );

         if (solver_id == 50)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, 0);
            HYPRE_StructSMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructSMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_LGMRESSetPrecond( (HYPRE_Solver)solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                    (HYPRE_Solver)precond);
         }

         else if (solver_id == 51)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, 0);
            HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_LGMRESSetPrecond( (HYPRE_Solver)solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                    (HYPRE_Solver)precond);
         }

         HYPRE_LGMRESSetup
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("LGMRES Solve");
         hypre_BeginTiming(time_index);

         HYPRE_LGMRESSolve
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_LGMRESGetNumIterations( (HYPRE_Solver)solver, &num_iterations);
         HYPRE_LGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm);
         HYPRE_StructLGMRESDestroy(solver);

         if (solver_id == 50)
         {
            HYPRE_StructSMGDestroy(precond);
         }
         else if (solver_id == 51)
         {
            HYPRE_StructPFMGDestroy(precond);
         }

      }
      /*-----------------------------------------------------------
       * Solve the system using FlexGMRES
       *-----------------------------------------------------------*/

      if ((solver_id > 59) && (solver_id < 70))
      {
         time_index = hypre_InitializeTiming("FlexGMRES Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructFlexGMRESCreate(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_FlexGMRESSetKDim( (HYPRE_Solver) solver, 5 );
         HYPRE_FlexGMRESSetMaxIter( (HYPRE_Solver)solver, 100 );
         HYPRE_FlexGMRESSetTol( (HYPRE_Solver)solver, tol );
         HYPRE_FlexGMRESSetPrintLevel( (HYPRE_Solver)solver, 1 );
         HYPRE_FlexGMRESSetLogging( (HYPRE_Solver)solver, 1 );

         if (solver_id == 60)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, 0);
            HYPRE_StructSMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructSMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver)solver,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                       (HYPRE_Solver)precond);
         }

         else if (solver_id == 61)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, 0);
            HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver)solver,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                       (HYPRE_Solver)precond);
         }

         HYPRE_FlexGMRESSetup
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("FlexGMRES Solve");
         hypre_BeginTiming(time_index);

         HYPRE_FlexGMRESSolve
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_FlexGMRESGetNumIterations( (HYPRE_Solver)solver, &num_iterations);
         HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm);
         HYPRE_StructFlexGMRESDestroy(solver);

         if (solver_id == 60)
         {
            HYPRE_StructSMGDestroy(precond);
         }
         else if (solver_id == 61)
         {
            HYPRE_StructPFMGDestroy(precond);
         }
      }

      /*-----------------------------------------------------------
       * Print the solution and other info
       *-----------------------------------------------------------*/

      if (print_system)
      {
         HYPRE_StructVectorPrint("struct.out.x", x, 0);
      }

      if (myid == 0 && rep == reps - 1 /* begin lobpcg */ && !lobpcgFlag /* end lobpcg */)
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
   }

#if defined(HYPRE_USING_KOKKOS)
   Kokkos::finalize ();
#endif

   /* Finalize Hypre */
   HYPRE_Finalize();

   /* Finalize MPI */
   hypre_MPI_Finalize();

#if defined(HYPRE_USING_MEMORY_TRACKER)
   if (memory_location == HYPRE_MEMORY_HOST)
   {
      if (hypre_total_bytes[hypre_MEMORY_DEVICE] || hypre_total_bytes[hypre_MEMORY_UNIFIED])
      {
         hypre_printf("Error: nonzero GPU memory allocated with the HOST mode\n");
         hypre_assert(0);
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
AddValuesVector( hypre_StructGrid   *gridvector,
                 hypre_StructVector *zvector,
                 HYPRE_Int          *period,
                 HYPRE_Real          value  )
{
   /* #include  "_hypre_struct_mv.h" */
   HYPRE_Int            i, ierr = 0;
   hypre_BoxArray      *gridboxes;
   HYPRE_Int            ib;
   hypre_IndexRef       ilower;
   hypre_IndexRef       iupper;
   hypre_Box           *box;
   HYPRE_Real          *values;
   HYPRE_Real          *values_h;
   HYPRE_Int            volume, dim;
   HYPRE_MemoryLocation memory_location = hypre_StructVectorMemoryLocation(zvector);

   gridboxes =  hypre_StructGridBoxes(gridvector);
   dim       =  hypre_StructGridNDim(gridvector);

   ib = 0;
   hypre_ForBoxI(ib, gridboxes)
   {
      box      = hypre_BoxArrayBox(gridboxes, ib);
      volume   = hypre_BoxVolume(box);
      values   = hypre_CTAlloc(HYPRE_Real, volume, memory_location);
      values_h = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------
       * For periodic b.c. in all directions, need rhs to satisfy
       * compatibility condition. Achieved by setting a source and
       *  sink of equal strength.  All other problems have rhs = 1.
       *-----------------------------------------------------------*/

      if ((dim == 2 && period[0] != 0 && period[1] != 0) ||
          (dim == 3 && period[0] != 0 && period[1] != 0 && period[2] != 0))
      {
         values_h[0] = value;
         values_h[volume - 1] = -value;
      }
      else
      {
         for (i = 0; i < volume; i++)
         {
            values_h[i] = value;
         }
      }

      hypre_TMemcpy(values, values_h, HYPRE_Real, volume, memory_location, HYPRE_MEMORY_HOST);

      ilower = hypre_BoxIMin(box);
      iupper = hypre_BoxIMax(box);

      HYPRE_StructVectorSetBoxValues(zvector, ilower, iupper, values);

      hypre_TFree(values, memory_location);
      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
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
AddValuesMatrix(HYPRE_StructMatrix A,
                HYPRE_StructGrid   gridmatrix,
                HYPRE_Real         cx,
                HYPRE_Real         cy,
                HYPRE_Real         cz,
                HYPRE_Real         conx,
                HYPRE_Real         cony,
                HYPRE_Real         conz)
{

   HYPRE_Int            d, ierr = 0;
   hypre_BoxArray      *gridboxes;
   HYPRE_Int            s, bi;
   hypre_IndexRef       ilower;
   hypre_IndexRef       iupper;
   hypre_Box           *box;
   HYPRE_Real          *values;
   HYPRE_Real          *values_h;
   HYPRE_Real           east, west;
   HYPRE_Real           north, south;
   HYPRE_Real           top, bottom;
   HYPRE_Real           center;
   HYPRE_Int            volume, dim, sym;
   HYPRE_Int           *stencil_indices;
   HYPRE_Int            stencil_size;
   HYPRE_Int            constant_coefficient;
   HYPRE_MemoryLocation memory_location = hypre_StructMatrixMemoryLocation(A);

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
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume(box);
            values = hypre_CTAlloc(HYPRE_Real, stencil_size * volume, memory_location);
            values_h = hypre_CTAlloc(HYPRE_Real, stencil_size * volume, HYPRE_MEMORY_HOST);

            if (dim == 1)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  values_h[i] = west;
                  values_h[i + 1] = center;
               }
            }
            else if (dim == 2)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  values_h[i] = west;
                  values_h[i + 1] = south;
                  values_h[i + 2] = center;
               }
            }
            else if (dim == 3)
            {
               for (d = 0; d < volume; d++)
               {
                  HYPRE_Int i = stencil_size * d;
                  values_h[i] = west;
                  values_h[i + 1] = south;
                  values_h[i + 2] = bottom;
                  values_h[i + 3] = center;
               }
            }

            hypre_TMemcpy(values, values_h, HYPRE_Real, stencil_size * volume, memory_location,
                          HYPRE_MEMORY_HOST);

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, stencil_size,
                                           stencil_indices, values);

            hypre_TFree(values, memory_location);
            hypre_TFree(values_h, HYPRE_MEMORY_HOST);
         }
      }
      else if ( constant_coefficient == 1 )
      {
         values = hypre_CTAlloc(HYPRE_Real, stencil_size, HYPRE_MEMORY_HOST);
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
         values = hypre_CTAlloc(HYPRE_Real, stencil_size - 1, HYPRE_MEMORY_HOST);
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
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume(box);
            values = hypre_CTAlloc(HYPRE_Real, volume, memory_location);
            values_h = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_HOST);
            HYPRE_Int i;

            for (i = 0; i < volume; i++)
            {
               values_h[i] = center;
            }

            hypre_TMemcpy(values, values_h, HYPRE_Real, volume, memory_location, HYPRE_MEMORY_HOST);

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices + dim, values);
            hypre_TFree(values, memory_location);
            hypre_TFree(values_h, HYPRE_MEMORY_HOST);
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
            volume = hypre_BoxVolume(box);
            values = hypre_CTAlloc(HYPRE_Real, stencil_size * volume, memory_location);
            values_h = hypre_CTAlloc(HYPRE_Real, stencil_size * volume, HYPRE_MEMORY_HOST);

            for (d = 0; d < volume; d++)
            {
               HYPRE_Int i = stencil_size * d;
               switch (dim)
               {
                  case 1:
                     values_h[i] = west;
                     values_h[i + 1] = center;
                     values_h[i + 2] = east;
                     break;
                  case 2:
                     values_h[i] = west;
                     values_h[i + 1] = south;
                     values_h[i + 2] = center;
                     values_h[i + 3] = east;
                     values_h[i + 4] = north;
                     break;
                  case 3:
                     values_h[i] = west;
                     values_h[i + 1] = south;
                     values_h[i + 2] = bottom;
                     values_h[i + 3] = center;
                     values_h[i + 4] = east;
                     values_h[i + 5] = north;
                     values_h[i + 6] = top;
                     break;
               }
            }

            hypre_TMemcpy(values, values_h, HYPRE_Real, stencil_size * volume, memory_location,
                          HYPRE_MEMORY_HOST);

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, stencil_size,
                                           stencil_indices, values);

            hypre_TFree(values, memory_location);
            hypre_TFree(values_h, HYPRE_MEMORY_HOST);
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
         values =  hypre_CTAlloc( HYPRE_Real,  stencil_size - 1, HYPRE_MEMORY_HOST);
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
            box    = hypre_BoxArrayBox(gridboxes, bi);
            volume = hypre_BoxVolume(box);
            values = hypre_CTAlloc(HYPRE_Real, volume, memory_location);
            values_h = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_HOST);
            HYPRE_Int i;

            for (i = 0; i < volume; i++)
            {
               values_h[i] = center;
            }

            hypre_TMemcpy(values, values_h, HYPRE_Real, volume, memory_location, HYPRE_MEMORY_HOST);

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
            hypre_TFree(values, memory_location);
            hypre_TFree(values_h, HYPRE_MEMORY_HOST);
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
SetStencilBndry(HYPRE_StructMatrix A,
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
   HYPRE_Real          *values;
   HYPRE_Int            volume, dim;
   HYPRE_Int           *stencil_indices;
   HYPRE_Int            constant_coefficient;
   HYPRE_MemoryLocation memory_location = hypre_StructMatrixMemoryLocation(A);

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
            values = hypre_CTAlloc(HYPRE_Real, vol[ib], memory_location);

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

            hypre_TFree(values, memory_location);
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
